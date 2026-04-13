"""Phase-1 feature ablation sweep orchestration."""

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, FeaturePipelineConfig
from .services import build_isolated_subprocess_env

PHASE_ONE_FEATURES: tuple[str, ...] = (
    "mel_spectrogram",
    "high_temporal_mel",
    "mfcc",
    "pcen",
    "mel_specaugment",
)
"""Feature families included in the phase-1 sweep."""

PHASE_ONE_PROXY_MODELS: tuple[str, ...] = ("convnext", "xlstm")
"""Proxy model families included in the phase-1 sweep."""

PHASE_ONE_SEEDS: tuple[int, int, int] = (0, 42, 2003)
"""Fixed seeds used for the phase-1 sweep grid."""

PHASE_ONE_STATE_SCHEMA_VERSION = 1
"""Schema version for the phase-1 sweep state file."""


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(UTC).isoformat()


def _serialize(value: Any) -> Any:
    """Convert nested dataclasses and tuples into JSON-friendly values."""

    if hasattr(value, "__dataclass_fields__"):
        return {
            field.name: _serialize(getattr(value, field.name))
            for field in value.__dataclass_fields__.values()
        }
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically via a temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Load JSON from disk."""

    return json.loads(path.read_text(encoding="utf-8"))


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


def _trial_score(payload: Mapping[str, Any]) -> float:
    """Extract the validation macro-F1 score from a child payload."""

    if not isinstance(payload, Mapping):
        return 0.0

    score = payload.get("validation_macro_f1")
    if isinstance(score, int | float):
        return float(score)

    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        metric_value = metrics.get("validation_macro_f1", metrics.get("macro_f1"))
        if isinstance(metric_value, int | float):
            return float(metric_value)

    phase_artifacts = payload.get("phase_artifacts")
    if isinstance(phase_artifacts, Mapping):
        for artifact in phase_artifacts.values():
            if isinstance(artifact, Mapping):
                output_data = artifact.get("output_data")
                if isinstance(output_data, Mapping):
                    score = _trial_score(output_data)
                    if score > 0.0:
                        return score
    return 0.0


def _load_child_summary(child_state_path: Path) -> dict[str, Any]:
    """Load the child run summary used to score a trial."""

    if not child_state_path.exists():
        return {}
    try:
        return _read_json(child_state_path)
    except (json.JSONDecodeError, OSError):
        return {}


def build_phase_one_command(config_path: Path, run_name: str) -> list[str]:
    """Build the isolated subprocess command for one phase-1 trial."""

    return [
        sys.executable,
        "-m",
        "speech_recognition.cli",
        "run-single-train",
        "--config",
        str(config_path),
        "--run-name",
        run_name,
    ]


@dataclass(frozen=True, slots=True)
class PhaseOneTrialSpec:
    """Describe one trial in the phase-1 ablation grid."""

    trial_id: str
    feature_name: str
    proxy_model: str
    seed: int

    def to_config(self, base_config: ExperimentConfig) -> ExperimentConfig:
        """Return the concrete config for this trial."""

        dataset = replace(base_config.dataset, train_split="train_small", valid_split="valid_small")
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
            for seed in PHASE_ONE_SEEDS:
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
    feature_name: str
    proxy_model: str
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
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseOneTrialRecord":
        """Build a record from JSON data."""

        payload = dict(data)
        payload["seed"] = int(payload["seed"])
        payload["validation_macro_f1"] = float(payload["validation_macro_f1"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PhaseOneSweepState:
    """Persistent state for the phase-1 sweep."""

    schema_version: int = PHASE_ONE_STATE_SCHEMA_VERSION
    output_dir: str = ""
    completed_trials: dict[str, PhaseOneTrialRecord] = field(default_factory=dict)
    best_trial_id: str | None = None
    best_validation_macro_f1: float | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema_version != PHASE_ONE_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported phase-1 state schema version.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the sweep state."""

        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseOneSweepState":
        """Build phase-1 state from JSON."""

        payload = dict(data)
        payload["completed_trials"] = {
            key: PhaseOneTrialRecord.from_dict(value)
            for key, value in payload.get("completed_trials", {}).items()
        }
        if payload.get("best_validation_macro_f1") is not None:
            payload["best_validation_macro_f1"] = float(payload["best_validation_macro_f1"])
        return cls(**payload)

    @classmethod
    def fresh(cls, output_dir: Path) -> "PhaseOneSweepState":
        """Create a new empty state for an output directory."""

        return cls(output_dir=str(output_dir))


class PhaseOneSweepRunner:
    """Run the phase-1 feature ablation sweep using child processes."""

    def __init__(self, output_dir: Path, base_config: ExperimentConfig | None = None) -> None:
        self.output_dir = output_dir
        self.phase_dir = self.output_dir / "phase_1"
        self.state_path = self.phase_dir / "state.json"
        self.best_feature_path = self.phase_dir / "best_feature.json"
        self.base_config = base_config or ExperimentConfig()

    def load_state(self) -> PhaseOneSweepState:
        """Load the persisted sweep state or create a new one."""

        if not self.state_path.exists():
            return PhaseOneSweepState.fresh(self.output_dir)
        return PhaseOneSweepState.from_dict(_read_json(self.state_path))

    def _save_state(self, state: PhaseOneSweepState) -> None:
        """Persist the sweep state and best-feature summary."""

        _atomic_write_json(self.state_path, state.to_dict())
        if state.best_trial_id is not None:
            best_trial = state.completed_trials[state.best_trial_id]
            _atomic_write_json(
                self.best_feature_path,
                {
                    "schema_version": PHASE_ONE_STATE_SCHEMA_VERSION,
                    "trial": best_trial.to_dict(),
                },
            )

    def _trial_run_name(self, trial: PhaseOneTrialSpec) -> str:
        """Return the child run name for one trial."""

        return trial.trial_id

    def _trial_output_paths(self, trial: PhaseOneTrialSpec) -> tuple[Path, Path, Path]:
        """Return config, child state, and trial directory paths for one trial."""

        trial_dir = self.phase_dir / "runs" / trial.trial_id
        config_path = trial_dir / "temp_config.json"
        child_state_path = self.output_dir / "phase_1" / "runs" / trial.trial_id / "state.json"
        return trial_dir, config_path, child_state_path

    def _build_trial_config(self, trial: PhaseOneTrialSpec) -> ExperimentConfig:
        """Build the concrete experiment config for a trial."""

        return trial.to_config(self.base_config)

    def _run_trial(self, trial: PhaseOneTrialSpec) -> PhaseOneTrialRecord:
        """Execute one trial in an isolated child process and return its record."""

        trial_dir, config_path, child_state_path = self._trial_output_paths(trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_trial_config(trial)
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        command = build_phase_one_command(config_path, self._trial_run_name(trial))
        subprocess.run(  # noqa: S603
            command,
            check=True,
            env=build_isolated_subprocess_env(),
            text=True,
            capture_output=True,
        )

        summary = _load_child_summary(child_state_path)
        validation_macro_f1 = _trial_score(summary)
        return PhaseOneTrialRecord(
            trial_id=trial.trial_id,
            feature_name=trial.feature_name,
            proxy_model=trial.proxy_model,
            seed=trial.seed,
            run_name=self._trial_run_name(trial),
            config_path=str(config_path),
            child_state_path=str(child_state_path),
            validation_macro_f1=validation_macro_f1,
            completed_at=_utc_now(),
        )

    def execute(self) -> dict[str, Any]:
        """Run the full phase-1 sweep, skipping completed trials."""

        state = self.load_state()
        best_trial_id = state.best_trial_id
        best_score = (
            state.best_validation_macro_f1 if state.best_validation_macro_f1 is not None else -1.0
        )

        for trial in build_phase_one_trials():
            if trial.trial_id in state.completed_trials:
                existing = state.completed_trials[trial.trial_id]
                if existing.validation_macro_f1 >= best_score:
                    best_trial_id = trial.trial_id
                    best_score = existing.validation_macro_f1
                continue

            trial_record = self._run_trial(trial)
            state = PhaseOneSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                completed_trials={**state.completed_trials, trial.trial_id: trial_record},
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score if best_score >= 0.0 else None,
                created_at=state.created_at,
                updated_at=_utc_now(),
            )

            if trial_record.validation_macro_f1 >= best_score:
                best_trial_id = trial.trial_id
                best_score = trial_record.validation_macro_f1

            state = PhaseOneSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score if best_score >= 0.0 else None,
                created_at=state.created_at,
                updated_at=trial_record.completed_at,
            )
            self._save_state(state)

        if best_trial_id is None and state.completed_trials:
            best_trial = max(
                state.completed_trials.values(), key=lambda record: record.validation_macro_f1
            )
            best_trial_id = best_trial.trial_id
            best_score = best_trial.validation_macro_f1
            state = PhaseOneSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score,
                created_at=state.created_at,
                updated_at=_utc_now(),
            )
            self._save_state(state)

        return {
            "phase": "phase-1",
            "total_trials": len(build_phase_one_trials()),
            "completed_trials": len(state.completed_trials),
            "best_trial": state.completed_trials[best_trial_id].to_dict()
            if best_trial_id
            else None,
            "best_validation_macro_f1": best_score if best_score >= 0.0 else None,
            "state_path": str(self.state_path),
            "best_feature_path": str(self.best_feature_path),
        }
