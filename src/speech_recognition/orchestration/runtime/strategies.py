"""Training strategy execution helpers for runtime orchestration."""

import contextlib
import importlib
import os
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ...config import ExperimentConfig
from ...features.extractors import WaveformLoader, build_feature_extractor
from ...models.registry import build_model_adapter
from ..tracking import _TRACKING_LOGGER, _resolve_tracking_uri
from .shared import (
    ALL_LABELS,
    COMMAND_LABELS,
    GATE_COMMAND_LABEL,
    GATE_NON_COMMAND_LABEL,
    NON_COMMAND_LABELS,
    AudioRecord,
    FeatureBatchLoader,
    SharedTwoHeadLoss,
    _augment_unknown_records_for_phase_four,
    _build_run_dir,
    _evaluate_predictions,
    _fit_model,
    _load_split_records,
    _loss_weights,
    _phase_four_weighted_sampling,
    _predict,
    _resolve_dataset_root,
    _sampling_weights,
    _set_reproducibility,
    _shared_two_head_probs_from_logits,
    _start_run_tracker,
)

_ACTIVE_MLFLOW_RUN_ID_ENV = "SPEECH_MLFLOW_ACTIVE_RUN_ID"
"""Environment variable name for an active MLflow run ID to exclude from checkpoint restoration."""


def _latest_local_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Return the latest numbered checkpoint in a local directory."""
    candidates = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def _checkpoint_step_from_artifact_path(path: str) -> int:
    """Extract checkpoint step from an MLflow artifact path."""
    name = Path(path).name
    if not name.startswith("checkpoint_step_") or not name.endswith(".pt"):
        return -1
    try:
        return int(Path(name).stem.removeprefix("checkpoint_step_"))
    except ValueError:
        return -1


def _collect_artifact_file_paths(client: Any, run_id: str, root: str) -> list[str]:
    """Collect leaf artifact paths recursively for one root path."""
    files: list[str] = []
    stack = [root]
    while stack:
        path = stack.pop()
        try:
            artifacts = client.list_artifacts(run_id, path)
        except Exception as exc:
            _TRACKING_LOGGER.debug(
                "Failed to list artifacts for run_id=%s path=%s: %s",
                run_id,
                path,
                exc,
            )
            continue
        for artifact in artifacts:
            if artifact.is_dir:
                stack.append(artifact.path)
            else:
                files.append(artifact.path)
    return files


def _latest_checkpoint_artifact_for_run(
    client: Any,
    run: Any,
    artifact_root: str,
) -> tuple[str, int] | None:
    """Return the latest checkpoint artifact path and step for one MLflow run."""
    run_id = run.info.run_id
    candidates: list[tuple[str, int]] = []

    tags = getattr(run.data, "tags", {}) or {}
    tagged_artifact = tags.get("latest_checkpoint_artifact")
    tagged_step_raw = tags.get("latest_checkpoint_step")
    tagged_step = None
    if tagged_step_raw is not None:
        try:
            tagged_step = int(tagged_step_raw)
        except (TypeError, ValueError):
            tagged_step = None

    if isinstance(tagged_artifact, str):
        if tagged_step is not None and tagged_step >= 0:
            return tagged_artifact, tagged_step
        parsed_tagged_step = _checkpoint_step_from_artifact_path(tagged_artifact)
        if parsed_tagged_step >= 0:
            return tagged_artifact, parsed_tagged_step

    for artifact_path in _collect_artifact_file_paths(client, run_id, artifact_root):
        parsed_step = _checkpoint_step_from_artifact_path(artifact_path)
        if parsed_step >= 0:
            candidates.append((artifact_path, parsed_step))

    if not candidates:
        return None
    return max(candidates, key=lambda item: item[1])


def _restore_checkpoint_from_mlflow_if_needed(
    config: ExperimentConfig,
    run_name: str,
    checkpoint_dir: Path,
    artifact_root: str,
) -> Path | None:
    """Restore the latest checkpoint artifact when no local checkpoint exists."""
    local_checkpoint = _latest_local_checkpoint(checkpoint_dir)
    if local_checkpoint is not None or not config.mlflow.enabled:
        return local_checkpoint

    try:
        mlflow = importlib.import_module("mlflow")
    except Exception:
        return None

    try:
        mlflow.set_tracking_uri(_resolve_tracking_uri(config.mlflow.tracking_uri))
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
        if experiment is None:
            return None

        safe_run_name = run_name.replace("'", "\\'")
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.pipeline.run_name = '{safe_run_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=50,
        )
        if not runs:
            return None

        active_run_id = os.environ.get(_ACTIVE_MLFLOW_RUN_ID_ENV)
        for run in runs:
            run_id = run.info.run_id
            if active_run_id and run_id == active_run_id:
                continue

            latest_artifact = _latest_checkpoint_artifact_for_run(client, run, artifact_root)
            if latest_artifact is None:
                continue
            artifact_path, step = latest_artifact

            if step < 0:
                continue

            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            destination = checkpoint_dir / f"checkpoint_step_{step:010d}.pt"
            downloaded = Path(client.download_artifacts(run_id, artifact_path, str(checkpoint_dir)))
            if downloaded != destination:
                destination.unlink(missing_ok=True)
                downloaded.replace(destination)

            _TRACKING_LOGGER.info(
                "[train] restored checkpoint from MLflow run_id=%s artifact=%s",
                run_id,
                artifact_path,
            )
            os.environ[_ACTIVE_MLFLOW_RUN_ID_ENV] = run_id
            return destination
    except Exception as exc:
        _TRACKING_LOGGER.warning("Failed to restore checkpoint from MLflow: %s", exc)

    return None


def _should_persist_predictions(config: ExperimentConfig, evaluation_split: str) -> bool:
    """Return whether prediction artifacts should be persisted for this run."""
    return config.phase.phase == "phase_4" or evaluation_split == config.dataset.test_split


def _log_prediction_artifact(
    tracker: Any | None,
    prediction_filename: str,
    payload: dict[str, Any],
) -> str | None:
    """Log prediction payload as MLflow artifact when tracker supports it."""
    if tracker is None or not hasattr(tracker, "log_named_json_artifact"):
        return None
    artifact_name = f"predictions/{prediction_filename}"
    tracker.log_named_json_artifact(artifact_name, payload)
    return artifact_name


def _prediction_artifact_filename(config: ExperimentConfig, evaluation_split: str) -> str:
    """Return the prediction artifact filename for the requested split."""
    return (
        "test_predictions.json"
        if evaluation_split == config.dataset.test_split
        else "validation_predictions.json"
    )


def _make_feature_loader(
    records: list[AudioRecord],
    label_to_idx: dict[str, int],
    config: ExperimentConfig,
    waveform_loader: WaveformLoader,
    feature_extractor: torch.nn.Module,
    *,
    shuffle: bool,
    sample_weights: list[float] | None = None,
    sampled_count: int | None = None,
) -> FeatureBatchLoader:
    """Build a FeatureBatchLoader with the common runtime defaults."""
    return FeatureBatchLoader(
        records,
        label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=shuffle,
        sample_weights=sample_weights,
        sampled_count=sampled_count,
    )


def _build_training_result(
    *,
    epoch: int,
    step: int,
    checkpoint_path: str | None,
    validation_macro_f1: float,
    metrics: dict[str, Any],
    prediction_artifact: str | None,
    phase: str,
    metric_phase: str,
) -> dict[str, Any]:
    """Assemble the standard runtime training result payload."""
    return {
        "epoch": int(epoch),
        "step": int(step),
        "checkpoint_path": checkpoint_path,
        "validation_macro_f1": float(validation_macro_f1),
        "metrics": metrics,
        "metric_phase": metric_phase,
        "prediction_artifact": prediction_artifact,
        "phase": phase,
    }


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
        for batch_features, batch_targets in tqdm(
            loader,
            total=len(loader),
            desc="logit batches",
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

            targets.extend(batch_targets.detach().cpu().tolist())
            logits_payload.extend(logits.detach().cpu().tolist())

    latency = float(sum(timings_ms) / len(timings_ms)) if timings_ms else 0.0
    return targets, logits_payload, latency


def _execute_two_stage(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
    evaluation_split: str,
    warmup_iterations: int,
) -> dict[str, Any]:
    """Train and evaluate a true two-stage detector pipeline."""
    tracker: Any | None = None
    try:
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

        command_train_records = [
            record for record in train_records if record.label in COMMAND_LABELS
        ]
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

        gate_train_loader = _make_feature_loader(
            train_records,
            gate_label_to_idx,
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
            sample_weights=(
                _phase_four_weighted_sampling(train_records)
                if config.phase.phase == "phase_4"
                else None
            ),
            sampled_count=len(train_records),
        )
        gate_val_loader = _make_feature_loader(
            val_records,
            gate_label_to_idx,
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
        )
        gate_eval_loader = _make_feature_loader(
            eval_records,
            gate_label_to_idx,
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
        )

        command_train_loader = _make_feature_loader(
            command_train_records,
            command_label_to_idx,
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
            sample_weights=(
                _phase_four_weighted_sampling(command_train_records)
                if config.phase.phase == "phase_4"
                else None
            ),
            sampled_count=len(command_train_records),
        )
        command_val_loader = _make_feature_loader(
            command_val_records,
            command_label_to_idx,
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
        )
        command_eval_loader = _make_feature_loader(
            eval_records,
            {label: command_label_to_idx.get(label, 0) for label in ALL_LABELS},
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
        )

        non_command_train_loader = _make_feature_loader(
            non_command_train_records,
            non_command_label_to_idx,
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
            sample_weights=(
                _phase_four_weighted_sampling(non_command_train_records)
                if config.phase.phase == "phase_4"
                else None
            ),
            sampled_count=len(non_command_train_records),
        )
        non_command_val_loader = _make_feature_loader(
            non_command_val_records,
            non_command_label_to_idx,
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
        )
        non_command_eval_loader = _make_feature_loader(
            eval_records,
            {label: non_command_label_to_idx.get(label, 0) for label in ALL_LABELS},
            config,
            waveform_loader,
            feature_extractor,
            shuffle=False,
        )

        gate_model = build_model_adapter(
            family=config.model.family,
            num_classes=2,
            pretrained=config.model.pretrained,
            model_config=replace(config.model, num_classes=2),
        )
        _restore_checkpoint_from_mlflow_if_needed(
            config=config,
            run_name=run_name,
            checkpoint_dir=run_dir / "checkpoints" / "gate",
            artifact_root="checkpoints/gate",
        )

        command_model = build_model_adapter(
            family=config.model.family,
            num_classes=len(COMMAND_LABELS),
            pretrained=config.model.pretrained,
            model_config=replace(config.model, num_classes=len(COMMAND_LABELS)),
        )
        _restore_checkpoint_from_mlflow_if_needed(
            config=config,
            run_name=run_name,
            checkpoint_dir=run_dir / "checkpoints" / "command",
            artifact_root="checkpoints/command",
        )

        non_command_model = build_model_adapter(
            family=config.model.family,
            num_classes=2,
            pretrained=config.model.pretrained,
            model_config=replace(config.model, num_classes=2),
        )
        _restore_checkpoint_from_mlflow_if_needed(
            config=config,
            run_name=run_name,
            checkpoint_dir=run_dir / "checkpoints" / "non_command",
            artifact_root="checkpoints/non_command",
        )

        tracker = _start_run_tracker(config, run_name)
        gate_engine, gate_fit = _fit_model(
            config=config,
            model=gate_model,
            train_loader=gate_train_loader,
            val_loader=gate_val_loader,
            checkpoint_dir=run_dir / "checkpoints" / "gate",
            tracker=tracker,
        )
        command_engine, _ = _fit_model(
            config=config,
            model=command_model,
            train_loader=command_train_loader,
            val_loader=command_val_loader,
            checkpoint_dir=run_dir / "checkpoints" / "command",
            tracker=tracker,
        )
        non_command_engine, _ = _fit_model(
            config=config,
            model=non_command_model,
            train_loader=non_command_train_loader,
            val_loader=non_command_val_loader,
            checkpoint_dir=run_dir / "checkpoints" / "non_command",
            tracker=tracker,
        )

        def _combined_probs(
            records: list[AudioRecord],
            gate_loader: FeatureBatchLoader,
            cmd_loader: FeatureBatchLoader,
            nc_loader: FeatureBatchLoader,
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

        prediction_artifact: str | None = None
        if _should_persist_predictions(config, evaluation_split):
            prediction_filename = (
                "test_predictions.json"
                if evaluation_split == config.dataset.test_split
                else "validation_predictions.json"
            )
            prediction_artifact = _log_prediction_artifact(
                tracker,
                prediction_filename,
                {
                    "targets": eval_targets,
                    "probs": eval_probs,
                    "labels": list(ALL_LABELS),
                    "strategy": config.evaluation.strategy,
                },
            )

        result = {
            "epoch": int(gate_fit.get("epoch", config.training.epochs)),
            "step": int(gate_fit.get("step", 0)),
            "checkpoint_path": gate_fit.get("checkpoint_path"),
            "validation_macro_f1": float(val_metrics["macro_f1"]),
            "metrics": eval_metrics,
            "metric_phase": ("test" if evaluation_split == config.dataset.test_split else "val"),
            "prediction_artifact": prediction_artifact,
        }
        if tracker is not None:
            tracker.log_payload(result)
        return result
    finally:
        if tracker is not None:
            tracker.close()


def _execute_shared_two_head(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
    evaluation_split: str,
    warmup_iterations: int,
) -> dict[str, Any]:
    """Train and evaluate a shared-backbone two-head pipeline."""
    tracker: Any | None = None
    try:
        _set_reproducibility(config.seed, config.training.deterministic)
        dataset_root = _resolve_dataset_root(config)
        train_records = _load_split_records(dataset_root, config.dataset.train_split)
        val_records = _load_split_records(dataset_root, config.dataset.valid_split)
        eval_records = _load_split_records(dataset_root, evaluation_split)
        if not train_records or not val_records or not eval_records:
            raise RuntimeError(
                "Shared-two-head strategy requires non-empty train/valid/eval splits."
            )

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
        _restore_checkpoint_from_mlflow_if_needed(
            config=config,
            run_name=run_name,
            checkpoint_dir=run_dir / "checkpoints" / "shared_two_head",
            artifact_root="checkpoints/shared_two_head",
        )

        tracker = _start_run_tracker(config, run_name)
        engine, fit_payload = _fit_model(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=run_dir / "checkpoints" / "shared_two_head",
            tracker=tracker,
            loss_fn=SharedTwoHeadLoss(),
        )

        def _combined_head_probs(
            loader: FeatureBatchLoader,
            warmup: int,
        ) -> tuple[list[int], list[list[float]], float]:
            """Predict combined head probabilities and latency for a loader."""
            targets, logits_payload, latency_ms = _predict_logits(
                model,
                loader,
                engine.device,
                warmup_iterations=warmup,
            )
            logits = np.array(logits_payload, dtype=np.float32)
            return targets, _shared_two_head_probs_from_logits(logits), latency_ms

        val_targets, val_probs, _ = _combined_head_probs(val_loader, warmup=0)
        val_metrics = _evaluate_predictions(val_targets, val_probs, "flat_multiclass")

        eval_targets, eval_probs, eval_latency_ms = _combined_head_probs(
            eval_loader,
            warmup=warmup_iterations,
        )
        eval_metrics = _evaluate_predictions(eval_targets, eval_probs, "flat_multiclass")
        eval_metrics["inference_latency_ms_mean"] = eval_latency_ms

        prediction_artifact: str | None = None
        if _should_persist_predictions(config, evaluation_split):
            prediction_filename = (
                "test_predictions.json"
                if evaluation_split == config.dataset.test_split
                else "validation_predictions.json"
            )
            prediction_artifact = _log_prediction_artifact(
                tracker,
                prediction_filename,
                {
                    "targets": eval_targets,
                    "probs": eval_probs,
                    "labels": list(ALL_LABELS),
                    "strategy": config.evaluation.strategy,
                },
            )

        result = {
            "epoch": int(fit_payload.get("epoch", config.training.epochs)),
            "step": int(fit_payload.get("step", 0)),
            "checkpoint_path": fit_payload.get("checkpoint_path"),
            "validation_macro_f1": float(val_metrics["macro_f1"]),
            "metrics": eval_metrics,
            "metric_phase": ("test" if evaluation_split == config.dataset.test_split else "val"),
            "prediction_artifact": prediction_artifact,
        }
        if tracker is not None:
            tracker.log_payload(result)
        return result
    finally:
        if tracker is not None:
            tracker.close()


def execute_single_train(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """Execute the single-model training strategy
    for the given config and return evaluation results."""
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

    tracker: Any | None = None
    try:
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
            _phase_four_weighted_sampling(train_records)
            if config.phase.phase == "phase_4"
            else None
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

        model = build_model_adapter(
            family=config.model.family,
            num_classes=len(ALL_LABELS),
            pretrained=config.model.pretrained,
            model_config=config.model,
        )
        _restore_checkpoint_from_mlflow_if_needed(
            config=config,
            run_name=run_name,
            checkpoint_dir=run_dir / "checkpoints",
            artifact_root="checkpoints",
        )

        tracker = _start_run_tracker(config, run_name)
        engine, fit_payload = _fit_model(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=run_dir / "checkpoints",
            tracker=tracker,
            loss_fn=loss_fn,
        )

        val_targets, val_probs, _ = _predict(
            model,
            val_loader,
            config.evaluation.strategy,
            engine.device,
        )
        val_metrics = _evaluate_predictions(val_targets, val_probs, config.evaluation.strategy)

        prediction_artifact: str | None = None
        if _should_persist_predictions(config, config.dataset.valid_split):
            prediction_artifact = _log_prediction_artifact(
                tracker,
                "validation_predictions.json",
                {
                    "targets": val_targets,
                    "probs": val_probs,
                    "labels": list(ALL_LABELS),
                    "strategy": config.evaluation.strategy,
                },
            )

        result = {
            "epoch": fit_payload.get("epoch", config.training.epochs),
            "step": fit_payload.get("step", 0),
            "checkpoint_path": fit_payload.get("checkpoint_path"),
            "validation_macro_f1": float(val_metrics["macro_f1"]),
            "metrics": val_metrics,
            "metric_phase": "val",
            "prediction_artifact": prediction_artifact,
            "phase": config.phase.phase,
        }
        if tracker is not None:
            tracker.log_payload(result)
        return result
    finally:
        if tracker is not None:
            tracker.close()
