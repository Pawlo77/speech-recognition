"""Trial models and helpers for phase-4 held-out evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...config import ExperimentConfig
from ..phase_three import PhaseThreeTrialRecord
from ..phase_two import _scheduler_config_for_trial
from ..state import _serialize
from .constants import (
    PHASE_FOUR_METHODS,
    PHASE_FOUR_STRICT_DROP_LIMIT,
    PHASE_FOUR_WARMUP_ITERATIONS,
)
from .utils import _build_backbone_model_config


@dataclass(frozen=True, slots=True)
class PhaseFourTrialSpec:
    """Describe one held-out evaluation trial."""

    trial_id: str
    """Unique trial identifier."""
    method: str
    """Evaluation strategy method name."""
    backbone_ids: tuple[str, ...]
    """Backbone model trial IDs to ensemble."""
    seed: int
    """Random seed for this trial."""

    def to_config(
        self,
        base_config: ExperimentConfig,
        backbone_trials: Mapping[str, PhaseThreeTrialRecord],
        optimizer_payload: Mapping[str, Any],
    ) -> ExperimentConfig:
        """Return the concrete config for this trial."""
        frozen_backbones = tuple(backbone_trials)
        representative_backbone = backbone_trials[self.backbone_ids[0]]
        evaluation = replace(
            base_config.evaluation,
            strategy=self.method,
            backbone_ids=frozen_backbones,
            ensemble_members=self.backbone_ids,
            warmup_iterations=PHASE_FOUR_WARMUP_ITERATIONS,
            max_core_command_f1_drop=PHASE_FOUR_STRICT_DROP_LIMIT,
        )
        dataset = replace(
            base_config.dataset,
            train_split="train_extended",
            valid_split="valid_extended",
            test_split="test_extended",
        )
        phase_config = replace(base_config.phase, phase="phase_4")
        optimizer = replace(
            base_config.optimizer,
            weight_decay=float(optimizer_payload["weight_decay"]),
        )
        scheduler = _scheduler_config_for_trial(
            str(optimizer_payload["scheduler_name"]),
            total_epochs=base_config.training.epochs,
        )
        return replace(
            base_config,
            dataset=dataset,
            model=_build_backbone_model_config(representative_backbone),
            optimizer=optimizer,
            scheduler=scheduler,
            evaluation=evaluation,
            phase=phase_config,
            seed=self.seed,
        )


def _build_method_trials(
    backbone_ids: tuple[str, ...], method: str, seeds: tuple[int, ...]
) -> tuple[PhaseFourTrialSpec, ...]:
    """Return the trial specifications for one evaluation method given backbone IDs and seeds."""
    trials: list[PhaseFourTrialSpec] = []
    trial_index = 0
    for backbone_id in backbone_ids:
        for seed in seeds:
            trial_index += 1
            trials.append(
                PhaseFourTrialSpec(
                    trial_id=f"trial_{trial_index:02d}_{method}_{backbone_id}_seed_{seed}",
                    method=method,
                    backbone_ids=(backbone_id,),
                    seed=seed,
                )
            )
    return tuple(trials)


def build_phase_four_trials(
    backbone_ids: tuple[str, ...], seeds: tuple[int, ...] = (0, 42, 2003)
) -> tuple[PhaseFourTrialSpec, ...]:
    """Return the 45 evaluation trial specifications for phase 4."""
    return tuple(
        trial
        for method in PHASE_FOUR_METHODS
        for trial in _build_method_trials(backbone_ids, method, seeds)
    )


@dataclass(frozen=True, slots=True)
class PhaseFourTrialRecord:
    """Persisted record for one completed phase-4 evaluation."""

    trial_id: str
    """Unique trial identifier."""
    method: str
    """Evaluation strategy method used."""
    backbone_ids: tuple[str, ...]
    """Backbone trial IDs evaluated."""
    seed: int
    """Random seed used."""
    baseline_backbone_id: str
    """Baseline backbone ID for comparison."""
    baseline_core_command_macro_f1: float
    """Baseline core-command macro-F1 score."""
    core_command_macro_f1: float
    """Core-command macro-F1 score achieved."""
    unknown_f1: float
    """Unknown class F1 score."""
    silence_f1: float
    """Silence class F1 score."""
    macro_f1_nc: float
    """Macro-F1 score for non-command classes."""
    inference_latency_ms_mean: float
    """Mean inference latency in milliseconds."""
    accepted: bool
    """Whether trial met acceptance criteria."""
    config_path: str
    """Path to the trial's config file."""
    child_state_path: str
    """Path to the child run's state."""
    completed_at: str
    """ISO-8601 timestamp when completed."""
    unknown_to_command_leakage: float = 0.0
    """Rate of unknown samples misrouted into command classes."""
    silence_false_trigger_rate: float = 0.0
    """Rate of silence samples misrouted into command classes."""
    prediction_artifact: str | None = None
    """Path to prediction artifact if available."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseFourTrialRecord":
        """Build a record from JSON data."""
        payload = dict(data)
        payload["backbone_ids"] = tuple(payload["backbone_ids"])
        payload["seed"] = int(payload.get("seed", 0))
        payload["baseline_core_command_macro_f1"] = float(payload["baseline_core_command_macro_f1"])
        payload["core_command_macro_f1"] = float(payload["core_command_macro_f1"])
        payload["unknown_f1"] = float(payload["unknown_f1"])
        payload["silence_f1"] = float(payload["silence_f1"])
        payload["macro_f1_nc"] = float(payload["macro_f1_nc"])
        payload["inference_latency_ms_mean"] = float(payload["inference_latency_ms_mean"])
        payload["unknown_to_command_leakage"] = float(
            payload.get("unknown_to_command_leakage", 0.0)
        )
        payload["silence_false_trigger_rate"] = float(
            payload.get("silence_false_trigger_rate", 0.0)
        )
        payload["accepted"] = bool(payload["accepted"])
        payload["prediction_artifact"] = payload.get("prediction_artifact")
        return cls(**payload)
