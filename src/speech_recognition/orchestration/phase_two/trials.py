"""Trial models and helpers for phase-2 hyperparameter sweep."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...config import ExperimentConfig, SchedulerConfig
from ..state import _serialize
from ..sweep_utils import sweep_seeds
from .constants import (
    PHASE_TWO_PROXY_MODELS,
    PHASE_TWO_SCHEDULERS,
    PHASE_TWO_SEEDS,
    PHASE_TWO_WEIGHT_DECAYS,
)


def _sweep_seeds() -> tuple[int, ...]:
    """Return default seeds or a single-seed override for smoke runs."""
    return sweep_seeds(PHASE_TWO_SEEDS)


def _scheduler_config_for_trial(scheduler_name: str, total_epochs: int) -> SchedulerConfig:
    """Build the scheduler config for one trial."""
    if scheduler_name == "cosine_annealing_warmup":
        return SchedulerConfig(name="cosine_annealing_warmup", total_epochs=total_epochs)
    if scheduler_name == "reduce_on_plateau":
        return SchedulerConfig(name="reduce_on_plateau", total_epochs=total_epochs)
    raise ValueError(f"Unsupported phase-2 scheduler '{scheduler_name}'.")


def _model_family_for_trial(proxy_model: str) -> str:
    """Validate the proxy model used by a phase-2 trial."""
    if proxy_model not in PHASE_TWO_PROXY_MODELS:
        raise ValueError(f"Unsupported phase-2 proxy model '{proxy_model}'.")
    return proxy_model


@dataclass(frozen=True, slots=True)
class PhaseTwoTrialSpec:
    """Describe one trial in the phase-2 optimization sweep."""

    trial_id: str
    feature_name: str
    proxy_model: str
    weight_decay: float
    scheduler_name: str
    seed: int

    def to_config(self, base_config: ExperimentConfig) -> ExperimentConfig:
        """Return the concrete config for this trial."""
        dataset = replace(
            base_config.dataset,
            train_split="train_small",
            valid_split="valid_small",
            test_split="test_small",
        )
        model = replace(
            base_config.model, family=_model_family_for_trial(self.proxy_model), pretrained=False
        )
        optimizer = replace(base_config.optimizer, weight_decay=self.weight_decay)
        scheduler = _scheduler_config_for_trial(
            self.scheduler_name, total_epochs=base_config.training.epochs
        )
        phase_config = replace(base_config.phase, phase="phase_2")
        return replace(
            base_config,
            dataset=dataset,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            phase=phase_config,
            seed=self.seed,
        )


def build_phase_two_trials(feature_name: str) -> tuple[PhaseTwoTrialSpec, ...]:
    """Return the 24 trial specifications for phase 2."""
    trials: list[PhaseTwoTrialSpec] = []
    trial_index = 0
    for weight_decay in PHASE_TWO_WEIGHT_DECAYS:
        for scheduler_name in PHASE_TWO_SCHEDULERS:
            for proxy_model in PHASE_TWO_PROXY_MODELS:
                for seed in _sweep_seeds():
                    trial_index += 1
                    trials.append(
                        PhaseTwoTrialSpec(
                            trial_id=(
                                f"trial_{trial_index:02d}_{feature_name}_{proxy_model}_{scheduler_name}_"
                                f"wd_{weight_decay}_seed_{seed}"
                            ),
                            feature_name=feature_name,
                            proxy_model=proxy_model,
                            weight_decay=weight_decay,
                            scheduler_name=scheduler_name,
                            seed=seed,
                        )
                    )
    return tuple(trials)


@dataclass(frozen=True, slots=True)
class PhaseTwoTrialRecord:
    """Persisted record for one completed phase-2 trial."""

    trial_id: str
    feature_name: str
    proxy_model: str
    weight_decay: float
    scheduler_name: str
    seed: int
    run_name: str
    config_path: str
    child_state_path: str
    validation_macro_f1: float
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseTwoTrialRecord":
        """Build a record from JSON data."""
        payload = dict(data)
        payload["weight_decay"] = float(payload["weight_decay"])
        payload["seed"] = int(payload["seed"])
        payload["validation_macro_f1"] = float(payload["validation_macro_f1"])
        return cls(**payload)
