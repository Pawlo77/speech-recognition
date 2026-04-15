"""Evaluation strategy execution helpers for runtime orchestration."""

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...config import ExperimentConfig
from ...features.extractors import WaveformLoader, build_feature_extractor
from ...models.registry import build_model_adapter
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


def _validation_macro_f1_from_artifact(run_dir: Path, strategy: str) -> float:
    """Read validation macro-F1 from saved validation prediction artifact when available."""

    artifact_path = run_dir / "validation_predictions.json"
    if not artifact_path.exists():
        return 0.0
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        targets = payload.get("targets")
        probs = payload.get("probs")
        if not isinstance(targets, list) or not isinstance(probs, list):
            return 0.0
        effective_strategy = (
            "flat_multiclass" if strategy in {"two_stage_detector", "shared_two_head"} else strategy
        )
        metrics = _evaluate_predictions(targets, probs, effective_strategy)
        return float(metrics.get("macro_f1", 0.0))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return 0.0


def _execute_two_stage_eval_only(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any] | None:
    """Evaluate two-stage detector on held-out split using existing checkpoints only."""

    run_dir = _build_run_dir(config, output_dir, run_name)
    gate_checkpoint = _latest_checkpoint_path(run_dir / "checkpoints" / "gate")
    command_checkpoint = _latest_checkpoint_path(run_dir / "checkpoints" / "command")
    non_command_checkpoint = _latest_checkpoint_path(run_dir / "checkpoints" / "non_command")
    if not gate_checkpoint or not command_checkpoint or not non_command_checkpoint:
        return None

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

    gate_state = torch.load(gate_checkpoint, map_location="cpu", weights_only=False)
    command_state = torch.load(command_checkpoint, map_location="cpu", weights_only=False)
    non_command_state = torch.load(non_command_checkpoint, map_location="cpu", weights_only=False)
    gate_model.load_state_dict(gate_state["model_state_dict"])
    command_model.load_state_dict(command_state["model_state_dict"])
    non_command_model.load_state_dict(non_command_state["model_state_dict"])

    gate_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    command_device = gate_device
    non_command_device = gate_device
    gate_model.to(gate_device)
    command_model.to(command_device)
    non_command_model.to(non_command_device)

    _, gate_probs, latency_ms = _predict(
        gate_model,
        gate_loader,
        GATE_COMMAND_LABEL,
        gate_device,
        warmup_iterations=config.evaluation.warmup_iterations,
    )
    _, command_probs, _ = _predict(
        command_model,
        command_loader,
        GATE_COMMAND_LABEL,
        command_device,
    )
    _, non_command_probs, _ = _predict(
        non_command_model,
        non_command_loader,
        GATE_NON_COMMAND_LABEL,
        non_command_device,
    )
    targets = [ALL_LABELS.index(record.label) for record in test_records]
    probs = _compose_two_stage_probs(gate_probs, command_probs, non_command_probs)
    test_metrics = _evaluate_predictions(targets, probs, "flat_multiclass")
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

    validation_macro_f1 = _validation_macro_f1_from_artifact(run_dir, config.evaluation.strategy)
    metrics = dict(test_metrics)
    metrics["validation_macro_f1"] = validation_macro_f1
    return {
        "epoch": config.training.epochs,
        "step": 0,
        "checkpoint_path": str(gate_checkpoint),
        "validation_macro_f1": validation_macro_f1,
        "metrics": metrics,
        "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
        "unknown_f1": float(metrics["unknown_f1"]),
        "silence_f1": float(metrics["silence_f1"]),
        "macro_f1_nc": float(metrics["macro_f1_nc"]),
        "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
        "prediction_artifact": str(predictions_path),
        "phase": config.phase.phase,
    }


def _execute_shared_two_head_eval_only(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any] | None:
    """Evaluate shared-two-head model on held-out split using existing checkpoints only."""

    run_dir = _build_run_dir(config, output_dir, run_name)
    checkpoint_path = _latest_checkpoint_path(run_dir / "checkpoints" / "shared_two_head")
    if checkpoint_path is None:
        return None

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
    state_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_payload["model_state_dict"])
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

    validation_macro_f1 = _validation_macro_f1_from_artifact(run_dir, config.evaluation.strategy)
    metrics = dict(test_metrics)
    metrics["validation_macro_f1"] = validation_macro_f1
    return {
        "epoch": config.training.epochs,
        "step": 0,
        "checkpoint_path": str(checkpoint_path),
        "validation_macro_f1": validation_macro_f1,
        "metrics": metrics,
        "core_command_macro_f1": float(metrics["core_command_macro_f1"]),
        "unknown_f1": float(metrics["unknown_f1"]),
        "silence_f1": float(metrics["silence_f1"]),
        "macro_f1_nc": float(metrics["macro_f1_nc"]),
        "inference_latency_ms_mean": float(metrics["inference_latency_ms_mean"]),
        "prediction_artifact": str(predictions_path),
        "phase": config.phase.phase,
    }


def execute_single_eval(
    config: ExperimentConfig,
    output_dir: Path,
    run_name: str,
) -> dict[str, Any]:
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
    checkpoint_path = _latest_checkpoint_path(run_dir / "checkpoints")
    train_payload: dict[str, Any]
    if checkpoint_path is None:
        train_payload = execute_single_train(config, output_dir, run_name)
        checkpoint_path = Path(str(train_payload.get("checkpoint_path", "")))
    else:
        train_payload = {
            "epoch": config.training.epochs,
            "step": 0,
            "checkpoint_path": str(checkpoint_path),
            "validation_macro_f1": _validation_macro_f1_from_artifact(
                run_dir,
                config.evaluation.strategy,
            ),
        }
    dataset_root = _resolve_dataset_root(config)
    test_records = _load_split_records(dataset_root, config.dataset.test_split)
    if not test_records:
        raise RuntimeError("Test split must be non-empty for evaluation.")

    label_to_idx = {label: index for index, label in enumerate(ALL_LABELS)}
    waveform_loader = WaveformLoader()
    feature_extractor = build_feature_extractor(config.features)

    adapter = build_model_adapter(
        family=config.model.family,
        num_classes=len(ALL_LABELS),
        pretrained=config.model.pretrained,
        model_config=config.model,
    )
    model = adapter

    if checkpoint_path:
        state_payload = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
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
