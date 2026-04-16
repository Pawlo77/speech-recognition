"""Trial models and helpers for phase-1 feature ablation sweep."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...config import ExperimentConfig, FeaturePipelineConfig
from ..state import _serialize
from ..sweep_utils import sweep_seeds
from .constants import PHASE_ONE_FEATURES, PHASE_ONE_PROXY_MODELS, PHASE_ONE_SEEDS


def _sweep_seeds() -> tuple[int, ...]:
    """Return default seeds or a single-seed override for smoke runs."""
    return sweep_seeds(PHASE_ONE_SEEDS)


def _feature_pipeline_for_trial(feature_name: str) -> FeaturePipelineConfig:
    """Build the feature configuration for one trial."""
    if feature_name == "mel_spectrogram":
        return FeaturePipelineConfig(name="mel_spectrogram")
    if feature_name == "high_temporal_mel":
        return FeaturePipelineConfig(name="high_temporal_mel", n_fft=512, hop_length=80)
    if feature_name == "mfcc":
        return FeaturePipelineConfig(name="mfcc", n_mfcc=40)
    if feature_name == "pcen":
        return FeaturePipelineConfig(name="pcen", pcen_smoothing=0.1)
    if feature_name == "mel_specaugment":
        return FeaturePipelineConfig(name="mel_specaugment", specaugment=True)
    raise ValueError(f"Unsupported phase-1 feature '{feature_name}'.")


@dataclass(frozen=True, slots=True)
class PhaseOneTrialSpec:
    """Describe one trial in the phase-1 ablation grid."""

    trial_id: str
    """Unique trial identifier."""
    feature_name: str
    """Feature pipeline name for this trial."""
    proxy_model: str
    """Proxy model family for this trial."""
    seed: int
    """Random seed for this trial."""

    def to_config(self, base_config: ExperimentConfig) -> ExperimentConfig:
        """Return the concrete config for this trial."""
        dataset = replace(
            base_config.dataset,
            train_split="train_small",
            valid_split="valid_small",
            test_split="test_small",
        )
        features = _feature_pipeline_for_trial(self.feature_name)
        model = replace(base_config.model, family=self.proxy_model, pretrained=False)
        phase_config = replace(base_config.phase, phase="phase_1")
        return replace(
            base_config,
            dataset=dataset,
            features=features,
            model=model,
            phase=phase_config,
            seed=self.seed,
        )


def build_phase_one_trials() -> tuple[PhaseOneTrialSpec, ...]:
    """Return the 30 trial specifications for phase 1."""
    trials: list[PhaseOneTrialSpec] = []
    trial_index = 0
    for feature_name in PHASE_ONE_FEATURES:
        for proxy_model in PHASE_ONE_PROXY_MODELS:
            for seed in _sweep_seeds():
                trial_index += 1
                trials.append(
                    PhaseOneTrialSpec(
                        trial_id=f"trial_{trial_index:02d}_{feature_name}_{proxy_model}_seed_{seed}",
                        feature_name=feature_name,
                        proxy_model=proxy_model,
                        seed=seed,
                    )
                )
    return tuple(trials)


@dataclass(frozen=True, slots=True)
class PhaseOneTrialRecord:
    """Persisted record for one completed phase-1 trial."""

    trial_id: str
    """Unique trial identifier."""
    feature_name: str
    """Feature pipeline name for this trial."""
    proxy_model: str
    """Proxy model family used in this trial."""
    seed: int
    """Random seed used in this trial."""
    run_name: str
    """Child pipeline run name."""
    config_path: str
    """Path to the trial's experiment config file."""
    child_state_path: str
    """Path to the child run's state file."""
    validation_macro_f1: float
    """Validation macro-F1 score achieved."""
    completed_at: str
    """ISO-8601 timestamp when trial completed."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseOneTrialRecord":
        """Build a record from JSON data."""
        payload = dict(data)
        payload["seed"] = int(payload["seed"])
        payload["validation_macro_f1"] = float(payload["validation_macro_f1"])
        return cls(**payload)
