"""Evaluation strategy execution helpers for runtime orchestration."""

import importlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

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
    FeatureBatchLoader,
    _build_run_dir,
    _evaluate_predictions,
    _load_split_records,
    _predict,
    _resolve_dataset_root,
)
from .strategies import (
    _compose_two_stage_probs,
    _execute_shared_two_head,
    _execute_two_stage,
    _predict_logits,
    execute_single_train,
)


def _latest_checkpoint_path(checkpoint_dir: Path) -> Path | None:
    """Return the latest numbered checkpoint in a directory."""
    candidates = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
    if not candidates:
        return None
    return candidates[-1]


def _mlflow_client_and_run(
    config: ExperimentConfig,
    run_name: str,
) -> tuple[Any, str] | None:
    """Return MLflow client and latest run id for a pipeline run name."""
    if not config.mlflow.enabled:
        return None

    try:
        mlflow = importlib.import_module("mlflow")
    except Exception:
        return None

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
        max_results=1,
    )
    if not runs:
        return None
    return client, runs[0].info.run_id


def _collect_artifact_file_paths(client: Any, run_id: str, root: str) -> list[str]:
    """Collect leaf artifact file paths recursively under a root path."""
    files: list[str] = []
    stack = [root]
    while stack:
        path = stack.pop()
        try:
            artifacts = client.list_artifacts(run_id, path)
        except Exception as e:
            _TRACKING_LOGGER.warning(f"Failed to list artifacts under {path} for run {run_id}: {e}")
            continue
        for artifact in artifacts:
            if artifact.is_dir:
                stack.append(artifact.path)
            else:
                files.append(artifact.path)
    return files


def _checkpoint_step(path: str) -> int:
    """Extract numeric step from checkpoint artifact filename."""
    name = Path(path).stem
    if not name.startswith("checkpoint_step_"):
        return -1
    try:
        return int(name.removeprefix("checkpoint_step_"))
    except ValueError:
        return -1


def _latest_mlflow_checkpoint_artifact(client: Any, run_id: str, root: str) -> str | None:
    """Return latest checkpoint artifact path under a root directory."""
    candidates = [
        path
        for path in _collect_artifact_file_paths(client, run_id, root)
        if Path(path).name.startswith("checkpoint_step_") and Path(path).suffix == ".pt"
    ]
    if not candidates:
        return None
    return max(candidates, key=_checkpoint_step)


def _load_checkpoint_payload(
    local_checkpoint: Path | None,
    artifact_root: str,
    run_context: tuple[Any, str] | None,
) -> tuple[dict[str, Any], str] | None:
    """Load checkpoint payload from local disk or MLflow artifact fallback."""
    if local_checkpoint is not None and local_checkpoint.exists():
        payload = torch.load(local_checkpoint, map_location="cpu", weights_only=False)
        return dict(payload), str(local_checkpoint)

    if run_context is None:
        return None

    client, run_id = run_context
    artifact_path = _latest_mlflow_checkpoint_artifact(client, run_id, artifact_root)
    if artifact_path is None:
        return None

    with tempfile.TemporaryDirectory() as temporary_dir:
        local_path = client.download_artifacts(run_id, artifact_path, temporary_dir)
        payload = torch.load(local_path, map_location="cpu", weights_only=False)
    return dict(payload), artifact_path


def _load_prediction_payload_from_mlflow(
    config: ExperimentConfig,
    run_name: str,
    artifact_path: str,
) -> dict[str, Any] | None:
    """Load a JSON prediction artifact payload from MLflow when available."""
    run_context = _mlflow_client_and_run(config, run_name)
    if run_context is None:
        return None
    client, run_id = run_context

    with tempfile.TemporaryDirectory() as temporary_dir:
        try:
            local_path = client.download_artifacts(run_id, artifact_path, temporary_dir)
        except Exception as e:
            _TRACKING_LOGGER.warning(
                f"Failed to download artifact {artifact_path} for run {run_id}: {e}"
            )
            return None

        try:
            payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if isinstance(payload, dict):
            return payload
    return None


def _validation_macro_f1_from_artifact(
    run_dir: Path,
    strategy: str,
    config: ExperimentConfig,
    run_name: str,
) -> float:
    """Read validation macro-F1 from validation prediction artifact when available."""
    payload: dict[str, Any] | None = None
    local_path = run_dir / "validation_predictions.json"
    if local_path.exists():
        try:
            candidate = json.loads(local_path.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                payload = candidate
        except (json.JSONDecodeError, OSError):
            payload = None

    if payload is None:
        payload = _load_prediction_payload_from_mlflow(
            config,
            run_name,
            "predictions/validation_predictions.json",
        )

    if payload is None:
        return 0.0

    try:
        targets = payload.get("targets")
        probs = payload.get("probs")
        if not isinstance(targets, list) or not isinstance(probs, list):
            return 0.0
        effective_strategy = (
            "flat_multiclass" if strategy in {"two_stage_detector", "shared_two_head"} else strategy
        )
        metrics = _evaluate_predictions(targets, probs, effective_strategy)
        return float(metrics.get("macro_f1", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _execute_two_stage_eval_only(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any] | None:
    """Evaluate two-stage detector on held-out split using existing checkpoints only."""
    run_dir = _build_run_dir(config, output_dir, run_name)
    run_context = _mlflow_client_and_run(config, run_name)

    gate_loaded = _load_checkpoint_payload(
        _latest_checkpoint_path(run_dir / "checkpoints" / "gate"),
        "checkpoints/gate",
        run_context,
    )
    command_loaded = _load_checkpoint_payload(
        _latest_checkpoint_path(run_dir / "checkpoints" / "command"),
        "checkpoints/command",
        run_context,
    )
    non_command_loaded = _load_checkpoint_payload(
        _latest_checkpoint_path(run_dir / "checkpoints" / "non_command"),
        "checkpoints/non_command",
        run_context,
    )
    if gate_loaded is None or command_loaded is None or non_command_loaded is None:
        return None

    gate_state, gate_reference = gate_loaded
    command_state, _ = command_loaded
    non_command_state, _ = non_command_loaded

    dataset_root = _resolve_dataset_root(config)
    test_records = _load_split_records(dataset_root, config.dataset.test_split)
    if not test_records:
        raise RuntimeError("Test split must be non-empty for evaluation.")

    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)
    gate_label_to_idx = dict.fromkeys(COMMAND_LABELS, 0)
    gate_label_to_idx.update(dict.fromkeys(NON_COMMAND_LABELS, 1))
    command_label_to_idx = {label: index for index, label in enumerate(COMMAND_LABELS)}
    non_command_label_to_idx = {
        NON_COMMAND_LABELS[0]: 0,
        NON_COMMAND_LABELS[1]: 1,
    }

    gate_loader = FeatureBatchLoader(
        test_records,
        gate_label_to_idx,
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )
    command_loader = FeatureBatchLoader(
        test_records,
        {label: command_label_to_idx.get(label, 0) for label in ALL_LABELS},
        batch_size=config.training.batch_size,
        seed=config.seed,
        waveform_loader=waveform_loader,
        feature_extractor=feature_extractor,
        shuffle=False,
    )
    non_command_loader = FeatureBatchLoader(
        test_records,
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
    command_model = build_model_adapter(
        family=config.model.family,
        num_classes=len(COMMAND_LABELS),
        pretrained=config.model.pretrained,
        model_config=replace(config.model, num_classes=len(COMMAND_LABELS)),
    )
    non_command_model = build_model_adapter(
        family=config.model.family,
        num_classes=2,
        pretrained=config.model.pretrained,
        model_config=replace(config.model, num_classes=2),
    )

    gate_model.load_state_dict(gate_state["model_state_dict"])
    command_model.load_state_dict(command_state["model_state_dict"])
    non_command_model.load_state_dict(non_command_state["model_state_dict"])

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    gate_model.to(device)
    command_model.to(device)
    non_command_model.to(device)

    _, gate_probs, latency_ms = _predict(
        gate_model,
        gate_loader,
        GATE_COMMAND_LABEL,
        device,
        warmup_iterations=config.evaluation.warmup_iterations,
    )
    _, command_probs, _ = _predict(
        command_model,
        command_loader,
        GATE_COMMAND_LABEL,
        device,
    )
    _, non_command_probs, _ = _predict(
        non_command_model,
        non_command_loader,
        GATE_NON_COMMAND_LABEL,
        device,
    )

    targets = [ALL_LABELS.index(record.label) for record in test_records]
    probs = _compose_two_stage_probs(gate_probs, command_probs, non_command_probs)
    test_metrics = _evaluate_predictions(targets, probs, "flat_multiclass")
    test_metrics["inference_latency_ms_mean"] = latency_ms

    validation_macro_f1 = _validation_macro_f1_from_artifact(
        run_dir,
        config.evaluation.strategy,
        config,
        run_name,
    )
    metrics = dict(test_metrics)
    metrics["validation_macro_f1"] = validation_macro_f1

    return {
        "epoch": config.training.epochs,
        "step": 0,
        "checkpoint_path": gate_reference,
        "validation_macro_f1": validation_macro_f1,
        "metrics": metrics,
        "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
        "unknown_f1": float(metrics["unknown_f1"]),
        "silence_f1": float(metrics["silence_f1"]),
        "macro_f1_nc": float(metrics["macro_f1_nc"]),
        "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
        "prediction_artifact": "predictions/test_predictions.json",
        "phase": config.phase.phase,
    }


def _execute_shared_two_head_eval_only(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any] | None:
    """Evaluate shared-two-head model on held-out split using existing checkpoints only."""
    run_dir = _build_run_dir(config, output_dir, run_name)
    run_context = _mlflow_client_and_run(config, run_name)
    loaded = _load_checkpoint_payload(
        _latest_checkpoint_path(run_dir / "checkpoints" / "shared_two_head"),
        "checkpoints/shared_two_head",
        run_context,
    )
    if loaded is None:
        return None

    checkpoint_state, checkpoint_reference = loaded

    dataset_root = _resolve_dataset_root(config)
    test_records = _load_split_records(dataset_root, config.dataset.test_split)
    if not test_records:
        raise RuntimeError("Test split must be non-empty for evaluation.")

    label_to_idx = {label: index for index, label in enumerate(ALL_LABELS)}
    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)
    loader = FeatureBatchLoader(
        test_records,
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
    model.load_state_dict(checkpoint_state["model_state_dict"])
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    targets, logits_payload, latency_ms = _predict_logits(
        model,
        loader,
        device,
        warmup_iterations=config.evaluation.warmup_iterations,
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
    probs = combined.tolist()

    test_metrics = _evaluate_predictions(targets, probs, "flat_multiclass")
    test_metrics["inference_latency_ms_mean"] = latency_ms

    validation_macro_f1 = _validation_macro_f1_from_artifact(
        run_dir,
        config.evaluation.strategy,
        config,
        run_name,
    )
    metrics = dict(test_metrics)
    metrics["validation_macro_f1"] = validation_macro_f1

    return {
        "epoch": config.training.epochs,
        "step": 0,
        "checkpoint_path": checkpoint_reference,
        "validation_macro_f1": validation_macro_f1,
        "metrics": metrics,
        "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
        "unknown_f1": float(metrics["unknown_f1"]),
        "silence_f1": float(metrics["silence_f1"]),
        "macro_f1_nc": float(metrics["macro_f1_nc"]),
        "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
        "prediction_artifact": "predictions/test_predictions.json",
        "phase": config.phase.phase,
    }


def execute_single_eval(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
    """Execute single-stage evaluation strategy,
    running training if needed and returning test metrics."""
    if config.evaluation.strategy == "two_stage_detector":
        cached_payload = _execute_two_stage_eval_only(config, output_dir, run_name)
        if cached_payload is not None:
            return cached_payload
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
        cached_payload = _execute_shared_two_head_eval_only(config, output_dir, run_name)
        if cached_payload is not None:
            return cached_payload
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

    run_dir = _build_run_dir(config, output_dir, run_name)
    run_context = _mlflow_client_and_run(config, run_name)
    loaded = _load_checkpoint_payload(
        _latest_checkpoint_path(run_dir / "checkpoints"),
        "checkpoints",
        run_context,
    )
    train_payload: dict[str, Any] | None = None

    if loaded is None:
        train_payload = execute_single_train(config, output_dir, run_name)
        run_context = _mlflow_client_and_run(config, run_name)
        loaded = _load_checkpoint_payload(
            None,
            "checkpoints",
            run_context,
        )
        if loaded is None:
            raise RuntimeError("No checkpoint available for evaluation after training.")

    checkpoint_state, checkpoint_reference = loaded
    validation_macro_f1 = (
        float(train_payload.get("validation_macro_f1", 0.0))
        if train_payload is not None
        else _validation_macro_f1_from_artifact(
            run_dir,
            config.evaluation.strategy,
            config,
            run_name,
        )
    )

    dataset_root = _resolve_dataset_root(config)
    test_records = _load_split_records(dataset_root, config.dataset.test_split)
    if not test_records:
        raise RuntimeError("Test split must be non-empty for evaluation.")

    label_to_idx = {label: index for index, label in enumerate(ALL_LABELS)}
    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)

    model = build_model_adapter(
        family=config.model.family,
        num_classes=len(ALL_LABELS),
        pretrained=config.model.pretrained,
        model_config=config.model,
    )
    model.load_state_dict(checkpoint_state["model_state_dict"])

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

    metrics = dict(test_metrics)
    metrics["validation_macro_f1"] = validation_macro_f1

    return {
        "epoch": train_payload.get("epoch", config.training.epochs)
        if train_payload is not None
        else config.training.epochs,
        "step": train_payload.get("step", 0) if train_payload is not None else 0,
        "checkpoint_path": checkpoint_reference,
        "validation_macro_f1": validation_macro_f1,
        "metrics": metrics,
        "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
        "unknown_f1": float(metrics["unknown_f1"]),
        "silence_f1": float(metrics["silence_f1"]),
        "macro_f1_nc": float(metrics["macro_f1_nc"]),
        "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
        "prediction_artifact": "predictions/test_predictions.json",
        "phase": config.phase.phase,
    }
