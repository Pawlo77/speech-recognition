"""Helper utilities for phase-4 orchestration."""

from collections.abc import Mapping
from typing import Any

from ...config import ModelConfig
from ..phase_three import PhaseThreeTrialRecord
from ..sweep_utils import iter_nested_payloads


def _phase_four_score(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract phase-4 metrics from a child payload."""
    if not isinstance(payload, Mapping):
        return {
            "core_command_macro_f1": 0.0,
            "unknown_f1": 0.0,
            "silence_f1": 0.0,
            "macro_f1_nc": 0.0,
            "inference_latency_ms_mean": 0.0,
            "unknown_to_command_leakage": 0.0,
            "silence_false_trigger_rate": 0.0,
            "per_class": {},
        }

    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    core_command_macro_f1 = payload.get(
        "core_command_macro_f1",
        metrics.get("core_command_macro_f1", metrics.get("macro_f1", 0.0)),
    )
    unknown_f1 = payload.get("unknown_f1", metrics.get("unknown_f1", 0.0))
    silence_f1 = payload.get("silence_f1", metrics.get("silence_f1", 0.0))
    macro_f1_nc = payload.get(
        "macro_f1_nc",
        metrics.get("macro_f1_nc", (float(unknown_f1) + float(silence_f1)) / 2.0),
    )
    inference_latency_ms_mean = payload.get(
        "inference_latency_ms_mean",
        metrics.get("inference_latency_ms_mean", 0.0),
    )
    unknown_to_command_leakage = payload.get(
        "unknown_to_command_leakage",
        metrics.get("unknown_to_command_leakage", 0.0),
    )
    silence_false_trigger_rate = payload.get(
        "silence_false_trigger_rate",
        metrics.get("silence_false_trigger_rate", 0.0),
    )
    per_class = payload.get(
        "per_class",
        metrics.get("per_class", {}),
    )

    return {
        "core_command_macro_f1": float(core_command_macro_f1),
        "unknown_f1": float(unknown_f1),
        "silence_f1": float(silence_f1),
        "macro_f1_nc": float(macro_f1_nc),
        "inference_latency_ms_mean": float(inference_latency_ms_mean),
        "unknown_to_command_leakage": float(unknown_to_command_leakage),
        "silence_false_trigger_rate": float(silence_false_trigger_rate),
        "per_class": per_class,
    }


def _phase_four_score_recursive(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract phase-4 metrics from nested child payload structures."""
    for nested_payload in iter_nested_payloads(payload):
        metrics = _phase_four_score(nested_payload)
        if metrics["core_command_macro_f1"] > 0.0 or metrics["macro_f1_nc"] > 0.0:
            return metrics
    return _phase_four_score(payload)


def _phase_four_prediction_artifact_recursive(payload: Mapping[str, Any]) -> str | None:
    """Extract prediction artifact path from nested child payload structures."""
    for nested_payload in iter_nested_payloads(payload):
        direct_path = nested_payload.get("prediction_artifact")
        if isinstance(direct_path, str) and direct_path:
            return direct_path
    return None


def _build_backbone_model_config(backbone_trial: PhaseThreeTrialRecord) -> ModelConfig:
    """Build a model config representative of one frozen phase-3 backbone."""
    family = backbone_trial.family
    params = backbone_trial.architecture_params
    if family == "ast":
        return ModelConfig(
            family="ast",
            pretrained=False,
            dropout=float(params["dropout"]),
            ast_head=str(params.get("head", "linear")),
            ast_positional_embedding=str(params.get("positional_embedding", "interp")),
            ast_hidden_size=int(params.get("hidden_size", 512)),
            ast_num_hidden_layers=int(params.get("num_layers", 10)),
            ast_num_attention_heads=int(params.get("num_heads", 8)),
            ast_intermediate_size=int(params.get("intermediate_size", 2048)),
        )
    if family == "convnext":
        return ModelConfig(
            family="convnext",
            pretrained=False,
            stochastic_depth=float(params["stochastic_depth"]),
        )
    if family == "ssamba":
        return ModelConfig(
            family="ssamba",
            pretrained=False,
            ssamba_pooling=str(params.get("pooling", "mean")),
            ssamba_use_cls=bool(params.get("use_cls", True)),
            ssamba_stride_ms=int(params.get("stride_ms", 10)),
            ssamba_d_model=int(params.get("d_model", 768)),
            ssamba_d_state=int(params.get("d_state", 64)),
            ssamba_expand=int(params.get("expand", 2)),
            ssamba_num_layers=int(params.get("num_layers", 8)),
        )
    if family == "xlstm":
        return ModelConfig(
            family="xlstm",
            pretrained=False,
            xlstm_dim=int(params.get("dimension", 768)),
            xlstm_num_blocks=int(params.get("num_blocks", 10)),
            xlstm_state_reset=bool(params.get("state_reset", True)),
            xlstm_output_mode=str(params.get("output_mode", "final")),
        )
    if family == "mlp_mixer":
        return ModelConfig(
            family="mlp_mixer",
            pretrained=False,
            dropout=float(params.get("dropout", 0.0)),
            mlp_head_l2_norm=bool(params.get("head_l2_norm", True)),
        )
    raise ValueError(f"Unsupported phase-3 backbone family '{family}'.")


def _phase_four_optimizer_config(optim_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the phase-2 winning optimizer payload for phase-4 reuse."""
    trial = optim_artifact.get("trial")
    if not isinstance(trial, Mapping):
        raise ValueError("Phase-2 best optimization artifact is missing the trial payload.")

    scheduler_name = trial.get("scheduler_name")
    if not isinstance(scheduler_name, str):
        raise ValueError("Phase-2 best optimization artifact is missing scheduler_name.")

    weight_decay = trial.get("weight_decay")
    if not isinstance(weight_decay, int | float):
        raise ValueError("Phase-2 best optimization artifact is missing weight_decay.")

    return {
        "scheduler_name": scheduler_name,
        "weight_decay": float(weight_decay),
        "trial": dict(trial),
    }
