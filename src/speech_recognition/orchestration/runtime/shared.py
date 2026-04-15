"""Executable train/eval runtime used by isolated child commands."""

import contextlib
import logging
import math
import queue
import random
import threading
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as nn_functional
from tqdm.auto import tqdm

from ...config import ExperimentConfig
from ...dataset.unknown import UnknownSampleGenerationMixin
from ...features.extractors import WaveformLoader
from ...training import TrainingEngine

COMMAND_LABELS: tuple[str, ...] = (
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
    "on",
    "off",
    "stop",
    "go",
)
"""10 command labels used in the 12-class formulation."""
NON_COMMAND_LABELS: tuple[str, str] = ("__unknown__", "__silence__")
"""Special non-command labels for unknown and silence classes."""
ALL_LABELS: tuple[str, ...] = (*COMMAND_LABELS, *NON_COMMAND_LABELS)
"""All 12 labels (10 commands + unknown + silence)."""
GATE_COMMAND_LABEL: str = "__command__"
"""Virtual label representing the union of all command classes
for two-stage detection strategies."""
GATE_NON_COMMAND_LABEL: str = "__non_command__"
"""Virtual label representing the union of all non-command classes
for two-stage detection strategies."""


class _RuntimeUnknownBlender(UnknownSampleGenerationMixin):
    """Runtime adapter for reusing dataset unknown-sample blending logic."""

    UNKNOWN_LABEL = "__unknown__"
    """Label used for blended unknown samples."""

    def __init__(self, dataset_root: Path, seed: int, target_unknown_count: int) -> None:
        self.dataset_root = dataset_root
        self.seed = seed
        self.unknown_label_samples_size = target_unknown_count
        self._unknown_source_profile_cache: dict[str, tuple[str, float]] = {}


def _set_reproducibility(seed: int, deterministic: bool) -> None:
    """Set Python, NumPy, and PyTorch random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass(frozen=True, slots=True)
class AudioRecord:
    """Audio file descriptor with label."""

    path: Path
    """Path to the audio file."""
    label: str
    """Label/class for the audio sample."""


def _phase_dir_name(config: ExperimentConfig) -> str:
    """Return the phase directory name from experiment config."""
    return config.phase.phase


def _workspace_root() -> Path:
    """Return the absolute path to the workspace root directory."""
    return Path(__file__).resolve().parents[4]


def _resolve_dataset_root(config: ExperimentConfig) -> Path:
    """Resolve the dataset root directory from config, searching subdirectories if needed."""
    root = _workspace_root() / config.dataset.root_dir
    if (root / "train" / "audio").exists():
        return root
    candidates = list(root.rglob("train/audio"))
    if candidates:
        return candidates[0].parent.parent
    raise FileNotFoundError(f"Dataset root not found under '{root}'.")


def _split_filename(split_name: str) -> str:
    """Map split name to its corresponding list filename in the dataset."""
    if split_name == "train_small":
        return "small_training_list.txt"
    if split_name == "valid_small":
        return "small_validation_list.txt"
    if split_name == "test_small":
        return "small_testing_list.txt"
    if split_name == "train_extended":
        return "extended_training_list.txt"
    if split_name == "valid_extended":
        return "extended_validation_list.txt"
    if split_name == "test_extended":
        return "extended_testing_list.txt"
    raise ValueError(f"Unsupported split '{split_name}'.")


def _normalize_label(label: str) -> str:
    if label == "_background_noise_":
        return "__silence__"
    if label in COMMAND_LABELS:
        return label
    return "__unknown__"


def _load_split_records(dataset_root: Path, split_name: str) -> list[AudioRecord]:
    """Load audio records for a given split from the dataset,
    normalizing labels and filtering missing files."""
    split_file = dataset_root / "train" / "split_lists" / _split_filename(split_name)
    if not split_file.exists():
        return []
    audio_root = dataset_root / "train" / "audio"
    records: list[AudioRecord] = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        rel_path = line.strip()
        if not rel_path:
            continue
        sample_path = audio_root / rel_path
        label = _normalize_label(Path(rel_path).parts[0])
        if label not in ALL_LABELS:
            continue
        if sample_path.exists():
            records.append(AudioRecord(path=sample_path, label=label))
    return records


class FeatureBatchLoader:
    """Deterministic batch loader with async background prefetching for GPU/CPU overlap.

    Features:
    - Computes features on-the-fly in a background thread
    - Maintains a prefetch queue to overlap GPU training with CPU feature extraction
    - Deterministic iteration order based on seed
    - Automatic cleanup on exit
    """

    def __init__(
        self,
        records: list[AudioRecord],
        label_to_idx: dict[str, int],
        batch_size: int,
        seed: int,
        waveform_loader: WaveformLoader,
        feature_extractor: nn.Module,
        shuffle: bool,
        sample_weights: list[float] | None = None,
        sampled_count: int | None = None,
    ) -> None:
        self.records = records
        self.label_to_idx = label_to_idx
        self.batch_size = max(1, batch_size)
        self.seed = seed
        self.waveform_loader = waveform_loader
        self.feature_extractor = feature_extractor
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self._sample_weights_tensor = (
            torch.tensor(sample_weights, dtype=torch.float32)
            if sample_weights is not None
            else None
        )
        self.sampled_count = sampled_count
        self._epoch = 0
        # Prefetch queue: stores 2-3 batches ahead to avoid GPU stalls
        self._prefetch_queue: queue.Queue[tuple[Tensor, Tensor] | None] = queue.Queue(maxsize=3)
        self._worker_thread: threading.Thread | None = None
        self._worker_stop_event: threading.Event | None = None

    def __len__(self) -> int:
        count = self.sampled_count if self.sampled_count is not None else len(self.records)
        if count <= 0:
            return 0
        return math.ceil(count / self.batch_size)

    def _prepare_batch(self, batch_records: list[AudioRecord]) -> tuple[Tensor, Tensor]:
        """Load waveforms and extract features for a batch."""
        paths = [record.path for record in batch_records]
        targets = torch.tensor(
            [self.label_to_idx[record.label] for record in batch_records],
            dtype=torch.long,
        )
        with torch.no_grad():
            waveforms = self.waveform_loader(paths)
            features = self.feature_extractor(waveforms)
        return features, targets

    def _prefetch_worker(self, indices: list[int]) -> None:
        """Background thread worker that prepares batches and puts them in the prefetch queue."""
        try:
            for start in range(0, len(indices), self.batch_size):
                if self._worker_stop_event.is_set():
                    break
                batch_indices = indices[start : start + self.batch_size]
                batch_records = [self.records[i] for i in batch_indices]
                batch = self._prepare_batch(batch_records)
                while True:
                    if self._worker_stop_event.is_set():
                        break
                    try:
                        # Use short retries so shutdown can interrupt a blocked producer.
                        self._prefetch_queue.put(batch, timeout=1)
                    except queue.Full:
                        continue
                    break
        except Exception as exc:
            logger = logging.getLogger(__name__)
            logger.exception("Prefetch worker error: %s", exc)
        finally:
            # Signal completion
            with contextlib.suppress(queue.Full):
                self._prefetch_queue.put_nowait(None)

    def __iter__(self):
        """Iterate over batches with background prefetching."""
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        if self.sample_weights is not None and self.records:
            num_samples = self.sampled_count or len(self.records)
            indices = torch.multinomial(
                self._sample_weights_tensor,
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            indices = list(range(len(self.records)))
            if self.shuffle and indices:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]

        # Start background prefetch worker
        self._worker_stop_event = threading.Event()
        self._prefetch_queue = queue.Queue(maxsize=3)
        self._worker_thread = threading.Thread(
            target=self._prefetch_worker,
            args=(indices,),
            daemon=False,
        )
        self._worker_thread.start()

        try:
            # Consume prefetched batches from queue
            while True:
                try:
                    batch = self._prefetch_queue.get(timeout=60)
                except queue.Empty:
                    # Slow feature extraction can exceed the queue timeout, especially
                    # during first-batch warm-up on CPU/MPS fallback paths.
                    if self._worker_thread and self._worker_thread.is_alive():
                        continue
                    break
                if batch is None:
                    break
                yield batch
        finally:
            # Ensure worker thread is stopped
            if self._worker_stop_event:
                self._worker_stop_event.set()
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=10)


def _macro_f1(targets: list[int], preds: list[int], labels: list[int]) -> float:
    """Compute macro-averaged F1 score for the given targets and predictions,
    considering only specified labels."""
    if not labels:
        return 0.0

    label_set = set(labels)
    tp: dict[int, int] = dict.fromkeys(labels, 0)
    fp: dict[int, int] = dict.fromkeys(labels, 0)
    fn: dict[int, int] = dict.fromkeys(labels, 0)
    for target, pred in zip(targets, preds, strict=True):
        if target == pred:
            if target in label_set:
                tp[target] += 1
            continue
        if pred in label_set:
            fp[pred] += 1
        if target in label_set:
            fn[target] += 1

    scores: list[float] = []
    for label in labels:
        label_tp = tp[label]
        label_fp = fp[label]
        label_fn = fn[label]
        precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) else 0.0
        recall = label_tp / (label_tp + label_fn) if (label_tp + label_fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores.append(f1)
    return float(sum(scores) / len(scores))


def _per_class_metrics(
    targets: list[int],
    preds: list[int],
    label_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute precision, recall, and F1 score for each class based
    on the given targets and predictions."""
    label_count = len(label_names)
    tp = [0] * label_count
    fp = [0] * label_count
    fn = [0] * label_count
    for target, pred in zip(targets, preds, strict=True):
        if 0 <= target < label_count and 0 <= pred < label_count:
            if target == pred:
                tp[target] += 1
            else:
                fp[pred] += 1
                fn[target] += 1

    rows: dict[str, dict[str, float]] = {}
    for index, name in enumerate(label_names):
        precision = tp[index] / (tp[index] + fp[index]) if (tp[index] + fp[index]) else 0.0
        recall = tp[index] / (tp[index] + fn[index]) if (tp[index] + fn[index]) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return rows


def _priors(config: ExperimentConfig) -> dict[str, float]:
    """Build class priors dictionary from experiment config
    for reweighting and sampling strategies."""
    priors = dict.fromkeys(COMMAND_LABELS, config.evaluation.command_prior)
    priors["__unknown__"] = config.evaluation.unknown_prior
    priors["__silence__"] = config.evaluation.silence_prior
    return priors


def _sampling_weights(records: list[AudioRecord], config: ExperimentConfig) -> list[float]:
    """Build per-sample weights for weighted sampling strategies
    based on class priors and sample counts."""
    priors = _priors(config)
    counts: dict[str, int] = dict.fromkeys(ALL_LABELS, 0)
    for record in records:
        counts[record.label] += 1
    weights: list[float] = []
    for record in records:
        class_count = max(1, counts[record.label])
        weights.append(priors[record.label] / class_count)
    return weights


def _phase_four_silence_boost(counts: dict[str, int]) -> float:
    """Compute silence boost so silence approaches mean command support."""
    command_counts = [counts[label] for label in COMMAND_LABELS]
    if not command_counts:
        return 1.0
    target_average = sum(command_counts) / len(command_counts)
    silence_count = max(1, counts["__silence__"])
    return max(1.0, target_average / silence_count)


def _phase_four_weighted_sampling(records: list[AudioRecord]) -> list[float]:
    """Build per-sample train weights with explicit silence up-weighting."""
    counts: dict[str, int] = dict.fromkeys(ALL_LABELS, 0)
    for record in records:
        counts[record.label] += 1

    silence_boost = _phase_four_silence_boost(counts)
    weights: list[float] = []
    for record in records:
        base_weight = 1.0 / max(1, counts[record.label])
        if record.label == "__silence__":
            base_weight *= silence_boost
        weights.append(base_weight)
    return weights


def _augment_unknown_records_for_phase_four(
    dataset_root: Path,
    records: list[AudioRecord],
    seed: int,
) -> list[AudioRecord]:
    """Augment unknown samples using blending when unknown support is too low."""
    counts: dict[str, int] = dict.fromkeys(ALL_LABELS, 0)
    for record in records:
        counts[record.label] += 1

    command_average = round(sum(counts[label] for label in COMMAND_LABELS) / len(COMMAND_LABELS))
    unknown_count = counts["__unknown__"]
    if unknown_count >= command_average:
        return records

    blender = _RuntimeUnknownBlender(
        dataset_root=dataset_root,
        seed=seed,
        target_unknown_count=command_average,
    )
    blender._create_unknown_label_samples()

    existing_paths = {record.path.resolve() for record in records}
    unknown_dir = dataset_root / "train" / "audio" / "__unknown__"
    appended_records = records[:]
    for path in sorted(unknown_dir.glob("*.wav")):
        resolved = path.resolve()
        if resolved in existing_paths:
            continue
        appended_records.append(AudioRecord(path=resolved, label="__unknown__"))
        existing_paths.add(resolved)
        unknown_count += 1
        if unknown_count >= command_average:
            break

    return appended_records


def _loss_weights(
    records: list[AudioRecord],
    config: ExperimentConfig,
    label_to_idx: dict[str, int],
) -> Tensor:
    """Build class priors dictionary from experiment config for
    reweighting and sampling strategies."""
    priors = _priors(config)
    counts: dict[str, int] = dict.fromkeys(ALL_LABELS, 0)
    for record in records:
        counts[record.label] += 1
    weights = torch.ones(len(label_to_idx), dtype=torch.float32)
    for label, index in label_to_idx.items():
        weights[index] = priors[label] / max(1, counts[label])
    return weights / weights.mean().clamp_min(1e-8)


def _build_scheduler(config: ExperimentConfig, optimizer: torch.optim.Optimizer):
    """Build a learning rate scheduler based on experiment config,
    supporting warmup and plateau strategies."""
    if config.scheduler.name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=config.scheduler.plateau_factor,
            patience=config.scheduler.plateau_patience,
            min_lr=config.scheduler.min_learning_rate,
        )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.scheduler.total_epochs - config.scheduler.warmup_epochs),
        eta_min=config.scheduler.min_learning_rate,
    )
    if config.scheduler.warmup_epochs <= 0:
        return cosine_scheduler

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        # LinearLR requires start_factor in (0, 1]. Use a tiny positive value
        # to approximate a near-zero warmup start without triggering validation.
        start_factor=1e-3,
        end_factor=1.0,
        total_iters=config.scheduler.warmup_epochs,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config.scheduler.warmup_epochs],
    )


def _prediction_from_probs(strategy: str, probs: np.ndarray) -> np.ndarray:
    """Convert model output probabilities to predicted class indices
    based on the specified evaluation strategy."""
    cmd_idx = np.arange(len(COMMAND_LABELS))
    unknown_idx = len(COMMAND_LABELS)
    silence_idx = len(COMMAND_LABELS) + 1
    if strategy in {"flat_multiclass", "sampling_control", "loss_reweighting"}:
        return probs.argmax(axis=1)
    if strategy == "two_stage_detector":
        nc_score = probs[:, unknown_idx] + probs[:, silence_idx]
        cmd_score = probs[:, cmd_idx].sum(axis=1)
        preds = probs[:, cmd_idx].argmax(axis=1)
        preds = cmd_idx[preds]
        nc_choice = np.where(
            probs[:, unknown_idx] >= probs[:, silence_idx],
            unknown_idx,
            silence_idx,
        )
        return np.where(nc_score > cmd_score, nc_choice, preds)
    if strategy == "shared_two_head":
        command_head = probs[:, cmd_idx].max(axis=1)
        unknown_head = probs[:, unknown_idx]
        silence_head = probs[:, silence_idx]
        nc_max = np.maximum(unknown_head, silence_head)
        cmd_pred = cmd_idx[probs[:, cmd_idx].argmax(axis=1)]
        nc_pred = np.where(unknown_head >= silence_head, unknown_idx, silence_idx)
        return np.where(nc_max > command_head, nc_pred, cmd_pred)
    raise ValueError(f"Unsupported strategy '{strategy}'.")


def _evaluate_predictions(
    targets: list[int],
    probs: list[list[float]],
    strategy: str,
) -> dict[str, Any]:
    """ "Compute evaluation metrics based on the given targets,
    predicted probabilities, and evaluation strategy."""
    probs_np = np.array(probs, dtype=np.float32)
    preds = _prediction_from_probs(strategy, probs_np)
    targets_list = [int(value) for value in targets]
    preds_list = [int(value) for value in preds.tolist()]

    all_labels = list(range(len(ALL_LABELS)))
    command_labels = list(range(len(COMMAND_LABELS)))
    unknown_idx = len(COMMAND_LABELS)
    silence_idx = len(COMMAND_LABELS) + 1
    command_label_set = set(command_labels)

    unknown_total = 0
    unknown_to_command = 0
    silence_total = 0
    silence_to_command = 0
    for target, pred in zip(targets_list, preds_list, strict=True):
        if target == unknown_idx:
            unknown_total += 1
            if pred in command_label_set:
                unknown_to_command += 1
        elif target == silence_idx:
            silence_total += 1
            if pred in command_label_set:
                silence_to_command += 1

    per_class = _per_class_metrics(targets_list, preds_list, ALL_LABELS)
    return {
        "macro_f1": _macro_f1(targets_list, preds_list, all_labels),
        "command_macro_f1": _macro_f1(targets_list, preds_list, command_labels),
        "core_command_macro_f1": _macro_f1(targets_list, preds_list, command_labels),
        "unknown_f1": per_class["__unknown__"]["f1"],
        "silence_f1": per_class["__silence__"]["f1"],
        "macro_f1_nc": (per_class["__unknown__"]["f1"] + per_class["__silence__"]["f1"]) / 2.0,
        "per_class": per_class,
        "unknown_to_command_leakage": float(unknown_to_command / max(1, unknown_total)),
        "silence_false_trigger_rate": float(silence_to_command / max(1, silence_total)),
    }


def _predict(
    model: nn.Module,
    loader: FeatureBatchLoader,
    strategy: str,
    device: torch.device,
    warmup_iterations: int = 0,
) -> tuple[list[int], list[list[float]], float]:
    """ "Run model inference on the given data loader
    and return targets, predicted probabilities, and average latency."""
    model.eval()
    targets: list[int] = []
    probs: list[list[float]] = []
    timings_ms: list[float] = []
    iteration_index = 0
    with torch.no_grad():
        for batch_features, batch_targets in tqdm(
            loader,
            total=len(loader),
            desc="inference batches",
            leave=False,
        ):
            batch_features = batch_features.to(device)
            started_at = perf_counter()
            logits = model(batch_features)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                with contextlib.suppress(Exception):
                    torch.mps.synchronize()  # pyright: ignore[reportAttributeAccessIssue]

            elapsed_ms = (perf_counter() - started_at) * 1000.0
            if iteration_index >= warmup_iterations:
                timings_ms.append(float(elapsed_ms))
            iteration_index += 1
            probabilities = torch.softmax(logits, dim=-1)
            targets.extend(batch_targets.detach().cpu().tolist())
            probs.extend(probabilities.detach().cpu().tolist())
    _ = strategy
    latency = float(sum(timings_ms) / len(timings_ms)) if timings_ms else 0.0
    return targets, probs, latency


def _build_run_dir(config: ExperimentConfig, output_dir: Path, run_name: str) -> Path:
    """Build the directory path for the current run based on
    the experiment config and run name, creating it if needed."""
    phase_dir = output_dir / _phase_dir_name(config) / "runs" / run_name
    if config.mlflow.enabled:
        return phase_dir

    phase_dir.mkdir(parents=True, exist_ok=True)
    for index_root in (output_dir, output_dir / _phase_dir_name(config), phase_dir):
        with contextlib.suppress(OSError):
            (index_root / ".metadata_never_index").touch(exist_ok=True)

    return phase_dir


class SharedTwoHeadLoss(nn.Module):
    """Loss for shared-backbone two-head training (10 command + 2 non-command)."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
        """Compute combined loss for shared two-head strategy
        by separating command and non-command samples."""
        command_mask = targets < len(COMMAND_LABELS)
        non_command_mask = ~command_mask
        losses: list[Tensor] = []

        if command_mask.any():
            command_logits = logits[command_mask, : len(COMMAND_LABELS)]
            command_targets = targets[command_mask]
            losses.append(nn_functional.cross_entropy(command_logits, command_targets))

        if non_command_mask.any():
            non_command_logits = logits[non_command_mask, len(COMMAND_LABELS) :]
            non_command_targets = targets[non_command_mask] - len(COMMAND_LABELS)
            losses.append(nn_functional.cross_entropy(non_command_logits, non_command_targets))

        if not losses:
            return logits.sum() * 0.0
        return sum(losses) / len(losses)


def _fit_model(
    config: ExperimentConfig,
    model: nn.Module,
    train_loader: FeatureBatchLoader,
    val_loader: FeatureBatchLoader,
    checkpoint_dir: Path,
    tracker: Any | None = None,
    loss_fn: nn.Module | None = None,
) -> tuple[TrainingEngine, dict[str, Any]]:
    """Train one model component and return engine plus fit payload."""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.learning_rate,
        betas=config.optimizer.betas,
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )
    scheduler = _build_scheduler(config, optimizer)
    engine = TrainingEngine(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=config.training,
        checkpoint_dir=checkpoint_dir,
        keep_last_n=config.checkpointing.keep_last_n,
        loss_fn=loss_fn or nn.CrossEntropyLoss(),
    )

    t0 = perf_counter()
    fit_payload = engine.fit(train_loader, val_loader, tracker=tracker)
    fit_time_ms = (perf_counter() - t0) * 1000.0
    if tracker is not None:
        tracker.log_training_metrics(
            epoch=int(fit_payload.get("epoch", config.training.epochs)),
            step=int(fit_payload.get("step", 0)),
            extra_metrics={
                "training_elapsed_ms": float(fit_time_ms),
            },
        )
    return engine, fit_payload


def _start_run_tracker(config: ExperimentConfig, run_name: str) -> Any | None:
    """Create and start an MLflow tracker when enabled in config."""
    if getattr(config, "mlflow", None) is None or not getattr(config.mlflow, "enabled", False):
        return None
    try:
        from ..tracking import build_mlflow_tracker

        tracker = build_mlflow_tracker(config, run_name=run_name)
        tracker.start()
        return tracker
    except Exception:
        logging.getLogger(__name__).exception(
            "Failed to initialize MLflow tracker for run '%s'; "
            "training metrics will not be logged to MLflow.",
            run_name,
        )
        return None
