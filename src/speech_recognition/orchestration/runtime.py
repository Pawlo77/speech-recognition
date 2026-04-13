"""Executable train/eval runtime used by isolated child commands."""

import contextlib
import json
import math
import random
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as nn_functional

from ..config import ExperimentConfig
from ..dataset.unknown import UnknownSampleGenerationMixin
from ..features.extractors import WaveformLoader, build_feature_extractor
from ..models.registry import build_model_adapter
from ..training import TrainingEngine

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
GATE_NON_COMMAND_LABEL: str = "__non_command__"


class _RuntimeUnknownBlender(UnknownSampleGenerationMixin):
    """Runtime adapter for reusing dataset unknown-sample blending logic."""

    UNKNOWN_LABEL = "__unknown__"

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
    return Path(__file__).resolve().parents[3]


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
    if split_name == "train_extended":
        return "extended_training_list.txt"
    if split_name == "valid_extended":
        return "extended_validation_list.txt"
    if split_name == "test":
        return "testing_list.txt"
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
    """Simple deterministic batch loader that computes features on the fly."""

    def __init__(
        self,
        records: list[AudioRecord],
        label_to_idx: dict[str, int],
        *,
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
        self.sampled_count = sampled_count
        self._epoch = 0

    def __len__(self) -> int:
        count = self.sampled_count if self.sampled_count is not None else len(self.records)
        if count <= 0:
            return 0
        return math.ceil(count / self.batch_size)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self._epoch)
        self._epoch += 1

        if self.sample_weights is not None and self.records:
            num_samples = self.sampled_count or len(self.records)
            weights = torch.tensor(self.sample_weights, dtype=torch.float32)
            indices = torch.multinomial(
                weights,
                num_samples=num_samples,
                replacement=True,
                generator=generator,
            ).tolist()
        else:
            indices = list(range(len(self.records)))
            if self.shuffle and indices:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in order]

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            batch_records = [self.records[i] for i in batch_indices]
            paths = [record.path for record in batch_records]
            targets = torch.tensor(
                [self.label_to_idx[record.label] for record in batch_records],
                dtype=torch.long,
            )
            with torch.no_grad():
                waveforms = self.waveform_loader(paths)
                features = self.feature_extractor(waveforms)
            yield features, targets


def _macro_f1(targets: list[int], preds: list[int], labels: list[int]) -> float:
    if not labels:
        return 0.0
    scores: list[float] = []
    for label in labels:
        tp = sum(1 for t, p in zip(targets, preds, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(targets, preds, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(targets, preds, strict=True) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores.append(f1)
    return float(sum(scores) / len(scores))


def _per_class_metrics(
    targets: list[int],
    preds: list[int],
    label_names: tuple[str, ...],
) -> dict[str, Any]:
    rows: dict[str, dict[str, float]] = {}
    for index, name in enumerate(label_names):
        tp = sum(1 for t, p in zip(targets, preds, strict=True) if t == index and p == index)
        fp = sum(1 for t, p in zip(targets, preds, strict=True) if t != index and p == index)
        fn = sum(1 for t, p in zip(targets, preds, strict=True) if t == index and p != index)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return rows


def _priors(config: ExperimentConfig) -> dict[str, float]:
    priors = dict.fromkeys(COMMAND_LABELS, config.evaluation.command_prior)
    priors["__unknown__"] = config.evaluation.unknown_prior
    priors["__silence__"] = config.evaluation.silence_prior
    return priors


def _sampling_weights(records: list[AudioRecord], config: ExperimentConfig) -> list[float]:
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
    priors = _priors(config)
    counts: dict[str, int] = dict.fromkeys(ALL_LABELS, 0)
    for record in records:
        counts[record.label] += 1
    weights = torch.ones(len(label_to_idx), dtype=torch.float32)
    for label, index in label_to_idx.items():
        weights[index] = priors[label] / max(1, counts[label])
    return weights / weights.mean().clamp_min(1e-8)


def _build_scheduler(config: ExperimentConfig, optimizer: torch.optim.Optimizer):
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
    probs_np = np.array(probs, dtype=np.float32)
    preds = _prediction_from_probs(strategy, probs_np)
    targets_list = [int(value) for value in targets]
    preds_list = [int(value) for value in preds.tolist()]

    all_labels = list(range(len(ALL_LABELS)))
    command_labels = list(range(len(COMMAND_LABELS)))
    unknown_idx = len(COMMAND_LABELS)
    silence_idx = len(COMMAND_LABELS) + 1

    per_class = _per_class_metrics(targets_list, preds_list, ALL_LABELS)
    return {
        "macro_f1": _macro_f1(targets_list, preds_list, all_labels),
        "command_macro_f1": _macro_f1(targets_list, preds_list, command_labels),
        "core_command_macro_f1": _macro_f1(targets_list, preds_list, command_labels),
        "unknown_f1": per_class["__unknown__"]["f1"],
        "silence_f1": per_class["__silence__"]["f1"],
        "macro_f1_nc": (per_class["__unknown__"]["f1"] + per_class["__silence__"]["f1"]) / 2.0,
        "per_class": per_class,
        "unknown_to_command_leakage": float(
            sum(
                1
                for t, p in zip(targets_list, preds_list, strict=True)
                if t == unknown_idx and p in command_labels
            )
            / max(1, sum(1 for t in targets_list if t == unknown_idx))
        ),
        "silence_false_trigger_rate": float(
            sum(
                1
                for t, p in zip(targets_list, preds_list, strict=True)
                if t == silence_idx and p in command_labels
            )
            / max(1, sum(1 for t in targets_list if t == silence_idx))
        ),
    }


def _predict(
    model: nn.Module,
    loader: FeatureBatchLoader,
    strategy: str,
    device: torch.device,
    warmup_iterations: int = 0,
) -> tuple[list[int], list[list[float]], float]:
    model.eval()
    targets: list[int] = []
    probs: list[list[float]] = []
    timings_ms: list[float] = []
    iteration_index = 0
    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
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
    phase_dir = output_dir / _phase_dir_name(config) / "runs" / run_name
    phase_dir.mkdir(parents=True, exist_ok=True)
    return phase_dir


class SharedTwoHeadLoss(nn.Module):
    """Loss for shared-backbone two-head training (10 command + 2 non-command)."""

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:  # type: ignore[override]
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
    *,
    config: ExperimentConfig,
    model: nn.Module,
    train_loader: FeatureBatchLoader,
    val_loader: FeatureBatchLoader,
    checkpoint_dir: Path,
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
        loss_fn=loss_fn or nn.CrossEntropyLoss(),
    )
    fit_payload = engine.fit(train_loader, val_loader)
    return engine, fit_payload


def _compose_two_stage_probs(
    gate_probs: list[list[float]],
    command_probs: list[list[float]],
    non_command_probs: list[list[float]],
) -> list[list[float]]:
    """Compose 12-class probabilities from gate/command/non-command heads."""

    gate = np.array(gate_probs, dtype=np.float32)
    command = np.array(command_probs, dtype=np.float32)
    non_command = np.array(non_command_probs, dtype=np.float32)
    combined = np.zeros((gate.shape[0], len(ALL_LABELS)), dtype=np.float32)
    combined[:, : len(COMMAND_LABELS)] = gate[:, 0:1] * command
    combined[:, len(COMMAND_LABELS) :] = gate[:, 1:2] * non_command
    return combined.tolist()


def _predict_logits(
    model: nn.Module,
    loader: FeatureBatchLoader,
    device: torch.device,
    warmup_iterations: int = 0,
) -> tuple[list[int], list[list[float]], float]:
    """Predict raw logits and latency for a loader."""

    model.eval()
    targets: list[int] = []
    logits_payload: list[list[float]] = []
    timings_ms: list[float] = []
    iteration_index = 0

    with torch.no_grad():
        for batch_features, batch_targets in loader:
            batch_features = batch_features.to(device)
            batch_targets = batch_targets.to(device)
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

            targets.extend(batch_targets.detach().cpu().tolist())
            logits_payload.extend(logits.detach().cpu().tolist())

    latency = float(sum(timings_ms) / len(timings_ms)) if timings_ms else 0.0
    return targets, logits_payload, latency


def _execute_two_stage(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
    evaluation_split: str,
    *,
    warmup_iterations: int,
) -> dict[str, Any]:
    """Train and evaluate a true two-stage detector pipeline."""

    _set_reproducibility(config.seed, config.training.deterministic)
    dataset_root = _resolve_dataset_root(config)
    train_records = _load_split_records(dataset_root, config.dataset.train_split)
    val_records = _load_split_records(dataset_root, config.dataset.valid_split)
    eval_records = _load_split_records(dataset_root, evaluation_split)
    if not train_records or not val_records or not eval_records:
        raise RuntimeError("Two-stage strategy requires non-empty train/valid/eval splits.")

    if config.phase.phase == "phase_4":
        train_records = _augment_unknown_records_for_phase_four(
            dataset_root,
            train_records,
            config.seed,
        )

    command_train_records = [record for record in train_records if record.label in COMMAND_LABELS]
    command_val_records = [record for record in val_records if record.label in COMMAND_LABELS]
    non_command_train_records = [
        record for record in train_records if record.label in NON_COMMAND_LABELS
    ]
    non_command_val_records = [
        record for record in val_records if record.label in NON_COMMAND_LABELS
    ]
    if not command_train_records or not non_command_train_records:
        raise RuntimeError(
            "Two-stage strategy requires both command and non-command training data."
        )
    if not command_val_records or not non_command_val_records:
        raise RuntimeError(
            "Two-stage strategy requires both command and non-command validation data."
        )

    run_dir = _build_run_dir(config, output_dir, run_name)
    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)

    gate_label_to_idx = dict.fromkeys(COMMAND_LABELS, 0)
    gate_label_to_idx.update(dict.fromkeys(NON_COMMAND_LABELS, 1))
    command_label_to_idx = {label: index for index, label in enumerate(COMMAND_LABELS)}
    non_command_label_to_idx = {
        NON_COMMAND_LABELS[0]: 0,
        NON_COMMAND_LABELS[1]: 1,
    }

    gate_train_loader = FeatureBatchLoader(
        train_records,
        gate_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
        sample_weights=(
            _phase_four_weighted_sampling(train_records)
            if config.phase.phase == "phase_4"
            else None
        ),
        sampled_count=len(train_records),
    )
    gate_val_loader = FeatureBatchLoader(
        val_records,
        gate_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )
    gate_eval_loader = FeatureBatchLoader(
        eval_records,
        gate_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )

    command_train_loader = FeatureBatchLoader(
        command_train_records,
        command_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=True,
    )
    command_val_loader = FeatureBatchLoader(
        command_val_records,
        command_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )
    command_eval_loader = FeatureBatchLoader(
        eval_records,
        {label: command_label_to_idx.get(label, 0) for label in ALL_LABELS},
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )

    non_command_train_loader = FeatureBatchLoader(
        non_command_train_records,
        non_command_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
        sample_weights=(
            _phase_four_weighted_sampling(non_command_train_records)
            if config.phase.phase == "phase_4"
            else None
        ),
        sampled_count=len(non_command_train_records),
    )
    non_command_val_loader = FeatureBatchLoader(
        non_command_val_records,
        non_command_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )
    non_command_eval_loader = FeatureBatchLoader(
        eval_records,
        {label: non_command_label_to_idx.get(label, 0) for label in ALL_LABELS},
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )

    gate_model = build_model_adapter(
        family=config.model.family,
        num_classes=2,
        pretrained=config.model.pretrained,
        model_config=replace(config.model, num_classes=2),
    )
    gate_engine, gate_fit = _fit_model(
        config=config,
        model=gate_model,
        train_loader=gate_train_loader,
        val_loader=gate_val_loader,
        checkpoint_dir=run_dir / "checkpoints" / "gate",
    )

    command_model = build_model_adapter(
        family=config.model.family,
        num_classes=len(COMMAND_LABELS),
        pretrained=config.model.pretrained,
        model_config=replace(config.model, num_classes=len(COMMAND_LABELS)),
    )
    command_engine, _ = _fit_model(
        config=config,
        model=command_model,
        train_loader=command_train_loader,
        val_loader=command_val_loader,
        checkpoint_dir=run_dir / "checkpoints" / "command",
    )

    non_command_model = build_model_adapter(
        family=config.model.family,
        num_classes=2,
        pretrained=config.model.pretrained,
        model_config=replace(config.model, num_classes=2),
    )
    non_command_engine, _ = _fit_model(
        config=config,
        model=non_command_model,
        train_loader=non_command_train_loader,
        val_loader=non_command_val_loader,
        checkpoint_dir=run_dir / "checkpoints" / "non_command",
    )

    def _combined_probs(
        records: list[AudioRecord],
        gate_loader: FeatureBatchLoader,
        cmd_loader: FeatureBatchLoader,
        nc_loader: FeatureBatchLoader,
        *,
        warmup: int,
    ) -> tuple[list[int], list[list[float]], float]:
        _, gate_probs, latency_ms = _predict(
            gate_model,
            gate_loader,
            GATE_COMMAND_LABEL,
            gate_engine.device,
            warmup_iterations=warmup,
        )
        _, command_probs, _ = _predict(
            command_model,
            cmd_loader,
            GATE_COMMAND_LABEL,
            command_engine.device,
        )
        _, non_command_probs, _ = _predict(
            non_command_model,
            nc_loader,
            GATE_NON_COMMAND_LABEL,
            non_command_engine.device,
        )
        targets = [ALL_LABELS.index(record.label) for record in records]
        probs = _compose_two_stage_probs(gate_probs, command_probs, non_command_probs)
        return targets, probs, latency_ms

    val_targets, val_probs, _ = _combined_probs(
        val_records,
        gate_val_loader,
        FeatureBatchLoader(
            val_records,
            {label: command_label_to_idx.get(label, 0) for label in ALL_LABELS},
            batch_size=config.training.batch_size,
            seed=config.seed,
            waveform_loader=waveform_loader,
            feature_extractor=feature_extractor,
            shuffle=False,
        ),
        FeatureBatchLoader(
            val_records,
            {label: non_command_label_to_idx.get(label, 0) for label in ALL_LABELS},
            batch_size=config.training.batch_size,
            seed=config.seed,
            waveform_loader=waveform_loader,
            feature_extractor=feature_extractor,
            shuffle=False,
        ),
        warmup=0,
    )
    val_metrics = _evaluate_predictions(val_targets, val_probs, "flat_multiclass")

    eval_targets, eval_probs, eval_latency_ms = _combined_probs(
        eval_records,
        gate_eval_loader,
        command_eval_loader,
        non_command_eval_loader,
        warmup=warmup_iterations,
    )
    eval_metrics = _evaluate_predictions(eval_targets, eval_probs, "flat_multiclass")
    eval_metrics["inference_latency_ms_mean"] = eval_latency_ms

    prediction_filename = (
        "test_predictions.json"
        if evaluation_split == config.dataset.test_split
        else "validation_predictions.json"
    )
    predictions_path = run_dir / prediction_filename
    predictions_path.write_text(
        json.dumps(
            {
                "targets": eval_targets,
                "probs": eval_probs,
                "labels": list(ALL_LABELS),
                "strategy": config.evaluation.strategy,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "epoch": int(gate_fit.get("epoch", config.training.epochs)),
        "step": int(gate_fit.get("step", 0)),
        "checkpoint_path": gate_fit.get("checkpoint_path"),
        "validation_macro_f1": float(val_metrics["macro_f1"]),
        "metrics": eval_metrics,
        "prediction_artifact": str(predictions_path),
    }


def _execute_shared_two_head(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
    evaluation_split: str,
    *,
    warmup_iterations: int,
) -> dict[str, Any]:
    """Train and evaluate a shared-backbone two-head pipeline."""

    _set_reproducibility(config.seed, config.training.deterministic)
    dataset_root = _resolve_dataset_root(config)
    train_records = _load_split_records(dataset_root, config.dataset.train_split)
    val_records = _load_split_records(dataset_root, config.dataset.valid_split)
    eval_records = _load_split_records(dataset_root, evaluation_split)
    if not train_records or not val_records or not eval_records:
        raise RuntimeError("Shared-two-head strategy requires non-empty train/valid/eval splits.")

    if config.phase.phase == "phase_4":
        train_records = _augment_unknown_records_for_phase_four(
            dataset_root,
            train_records,
            config.seed,
        )

    label_to_idx = {label: index for index, label in enumerate(ALL_LABELS)}
    run_dir = _build_run_dir(config, output_dir, run_name)
    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)

    train_loader = FeatureBatchLoader(
        train_records,
        label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
        sample_weights=(
            _phase_four_weighted_sampling(train_records)
            if config.phase.phase == "phase_4"
            else None
        ),
        sampled_count=len(train_records),
    )
    val_loader = FeatureBatchLoader(
        val_records,
        label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )
    eval_loader = FeatureBatchLoader(
        eval_records,
        label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )

    model = build_model_adapter(
        family=config.model.family,
        num_classes=len(ALL_LABELS),
        pretrained=config.model.pretrained,
        model_config=config.model,
    )
    engine, fit_payload = _fit_model(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=run_dir / "checkpoints" / "shared_two_head",
        loss_fn=SharedTwoHeadLoss(),
    )

    def _combined_head_probs(
        loader: FeatureBatchLoader,
        *,
        warmup: int,
    ) -> tuple[list[int], list[list[float]], float]:
        targets, logits_payload, latency_ms = _predict_logits(
            model,
            loader,
            engine.device,
            warmup_iterations=warmup,
        )
        logits = np.array(logits_payload, dtype=np.float32)
        cmd_logits = logits[:, : len(COMMAND_LABELS)]
        nc_logits = logits[:, len(COMMAND_LABELS) :]
        cmd_probs = np.exp(cmd_logits - cmd_logits.max(axis=1, keepdims=True))
        cmd_probs /= cmd_probs.sum(axis=1, keepdims=True)
        nc_probs = np.exp(nc_logits - nc_logits.max(axis=1, keepdims=True))
        nc_probs /= nc_probs.sum(axis=1, keepdims=True)

        cmd_conf = cmd_probs.max(axis=1, keepdims=True)
        nc_conf = nc_probs.max(axis=1, keepdims=True)
        denom = np.clip(cmd_conf + nc_conf, a_min=1e-8, a_max=None)
        cmd_mass = cmd_conf / denom
        nc_mass = nc_conf / denom

        combined = np.zeros((logits.shape[0], len(ALL_LABELS)), dtype=np.float32)
        combined[:, : len(COMMAND_LABELS)] = cmd_mass * cmd_probs
        combined[:, len(COMMAND_LABELS) :] = nc_mass * nc_probs
        return targets, combined.tolist(), latency_ms

    val_targets, val_probs, _ = _combined_head_probs(val_loader, warmup=0)
    val_metrics = _evaluate_predictions(val_targets, val_probs, "flat_multiclass")

    eval_targets, eval_probs, eval_latency_ms = _combined_head_probs(
        eval_loader,
        warmup=warmup_iterations,
    )
    eval_metrics = _evaluate_predictions(eval_targets, eval_probs, "flat_multiclass")
    eval_metrics["inference_latency_ms_mean"] = eval_latency_ms

    prediction_filename = (
        "test_predictions.json"
        if evaluation_split == config.dataset.test_split
        else "validation_predictions.json"
    )
    predictions_path = run_dir / prediction_filename
    predictions_path.write_text(
        json.dumps(
            {
                "targets": eval_targets,
                "probs": eval_probs,
                "labels": list(ALL_LABELS),
                "strategy": config.evaluation.strategy,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "epoch": int(fit_payload.get("epoch", config.training.epochs)),
        "step": int(fit_payload.get("step", 0)),
        "checkpoint_path": fit_payload.get("checkpoint_path"),
        "validation_macro_f1": float(val_metrics["macro_f1"]),
        "metrics": eval_metrics,
        "prediction_artifact": str(predictions_path),
    }


def execute_single_train(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    if config.evaluation.strategy == "two_stage_detector":
        payload = _execute_two_stage(
            config,
            output_dir,
            run_name,
            config.dataset.valid_split,
            warmup_iterations=0,
        )
        return {
            **payload,
            "phase": config.phase.phase,
        }

    if config.evaluation.strategy == "shared_two_head":
        payload = _execute_shared_two_head(
            config,
            output_dir,
            run_name,
            config.dataset.valid_split,
            warmup_iterations=0,
        )
        return {
            **payload,
            "phase": config.phase.phase,
        }

    _set_reproducibility(config.seed, config.training.deterministic)
    dataset_root = _resolve_dataset_root(config)
    train_records = _load_split_records(dataset_root, config.dataset.train_split)
    val_records = _load_split_records(dataset_root, config.dataset.valid_split)
    if not train_records or not val_records:
        raise RuntimeError("Training and validation splits must be non-empty.")

    if config.phase.phase == "phase_4":
        train_records = _augment_unknown_records_for_phase_four(
            dataset_root,
            train_records,
            config.seed,
        )

    label_to_idx = {label: index for index, label in enumerate(ALL_LABELS)}
    run_dir = _build_run_dir(config, output_dir, run_name)

    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)
    train_weights = None
    loss_fn: nn.Module = nn.CrossEntropyLoss()
    phase_four_weights = (
        _phase_four_weighted_sampling(train_records) if config.phase.phase == "phase_4" else None
    )
    if config.evaluation.strategy == "sampling_control":
        train_weights = _sampling_weights(train_records, config)
        if phase_four_weights is not None:
            train_weights = [
                base * silence
                for base, silence in zip(train_weights, phase_four_weights, strict=True)
            ]
    elif phase_four_weights is not None:
        train_weights = phase_four_weights
    if config.evaluation.strategy == "loss_reweighting":
        class_weights = _loss_weights(train_records, config, label_to_idx)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = FeatureBatchLoader(
        train_records,
        label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=train_weights is None,
        sample_weights=train_weights,
        sampled_count=len(train_records),
    )
    val_loader = FeatureBatchLoader(
        val_records,
        label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )

    adapter = build_model_adapter(
        family=config.model.family,
        num_classes=len(ALL_LABELS),
        pretrained=config.model.pretrained,
        model_config=config.model,
    )
    model = adapter
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
        checkpoint_dir=run_dir / "checkpoints",
        loss_fn=loss_fn,
    )

    fit_payload = engine.fit(train_loader, val_loader)

    val_targets, val_probs, _ = _predict(
        model,
        val_loader,
        config.evaluation.strategy,
        engine.device,
    )
    val_metrics = _evaluate_predictions(val_targets, val_probs, config.evaluation.strategy)

    predictions_path = run_dir / "validation_predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "targets": val_targets,
                "probs": val_probs,
                "labels": list(ALL_LABELS),
                "strategy": config.evaluation.strategy,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    return {
        "epoch": fit_payload.get("epoch", config.training.epochs),
        "step": fit_payload.get("step", 0),
        "checkpoint_path": fit_payload.get("checkpoint_path"),
        "validation_macro_f1": float(val_metrics["macro_f1"]),
        "metrics": val_metrics,
        "prediction_artifact": str(predictions_path),
        "phase": config.phase.phase,
    }


def execute_single_eval(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    if config.evaluation.strategy == "two_stage_detector":
        payload = _execute_two_stage(
            config,
            output_dir,
            run_name,
            config.dataset.test_split,
            warmup_iterations=config.evaluation.warmup_iterations,
        )
        metrics = dict(payload["metrics"])
        metrics["validation_macro_f1"] = float(payload["validation_macro_f1"])
        return {
            **payload,
            "metrics": metrics,
            "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
            "unknown_f1": float(metrics["unknown_f1"]),
            "silence_f1": float(metrics["silence_f1"]),
            "macro_f1_nc": float(metrics["macro_f1_nc"]),
            "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
            "phase": config.phase.phase,
        }

    if config.evaluation.strategy == "shared_two_head":
        payload = _execute_shared_two_head(
            config,
            output_dir,
            run_name,
            config.dataset.test_split,
            warmup_iterations=config.evaluation.warmup_iterations,
        )
        metrics = dict(payload["metrics"])
        metrics["validation_macro_f1"] = float(payload["validation_macro_f1"])
        return {
            **payload,
            "metrics": metrics,
            "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
            "unknown_f1": float(metrics["unknown_f1"]),
            "silence_f1": float(metrics["silence_f1"]),
            "macro_f1_nc": float(metrics["macro_f1_nc"]),
            "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
            "phase": config.phase.phase,
        }

    train_payload = execute_single_train(config, output_dir, run_name)
    dataset_root = _resolve_dataset_root(config)
    test_records = _load_split_records(dataset_root, config.dataset.test_split)
    if not test_records:
        raise RuntimeError("Test split must be non-empty for evaluation.")

    label_to_idx = {label: index for index, label in enumerate(ALL_LABELS)}
    run_dir = _build_run_dir(config, output_dir, run_name)
    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)

    adapter = build_model_adapter(
        family=config.model.family,
        num_classes=len(ALL_LABELS),
        pretrained=config.model.pretrained,
        model_config=config.model,
    )
    model = adapter

    checkpoint_path = train_payload.get("checkpoint_path")
    if checkpoint_path:
        state_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_payload["model_state_dict"])

    loader = FeatureBatchLoader(
        test_records,
        label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    targets, probs, latency_ms = _predict(
        model,
        loader,
        config.evaluation.strategy,
        device,
        warmup_iterations=config.evaluation.warmup_iterations,
    )
    test_metrics = _evaluate_predictions(targets, probs, config.evaluation.strategy)
    test_metrics["inference_latency_ms_mean"] = latency_ms

    predictions_path = run_dir / "test_predictions.json"
    predictions_path.write_text(
        json.dumps(
            {
                "targets": targets,
                "probs": probs,
                "labels": list(ALL_LABELS),
                "strategy": config.evaluation.strategy,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    metrics = dict(test_metrics)
    metrics["validation_macro_f1"] = float(train_payload["validation_macro_f1"])

    return {
        "epoch": train_payload.get("epoch", config.training.epochs),
        "step": train_payload.get("step", 0),
        "checkpoint_path": train_payload.get("checkpoint_path"),
        "validation_macro_f1": float(train_payload["validation_macro_f1"]),
        "metrics": metrics,
        "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
        "unknown_f1": float(metrics["unknown_f1"]),
        "silence_f1": float(metrics["silence_f1"]),
        "macro_f1_nc": float(metrics["macro_f1_nc"]),
        "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
        "prediction_artifact": str(predictions_path),
        "phase": config.phase.phase,
    }
