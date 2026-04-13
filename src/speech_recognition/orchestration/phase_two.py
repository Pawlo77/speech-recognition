"""Phase-2 global hyperparameter sweep orchestration."""

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, SchedulerConfig
from .phase_one import _atomic_write_json, _feature_pipeline_for_trial, _read_json, _serialize
from .services import build_isolated_subprocess_env

PHASE_TWO_WEIGHT_DECAYS: tuple[float, float] = (0.01, 0.1)
"""Weight decay values included in the phase-2 sweep."""

PHASE_TWO_SCHEDULERS: tuple[str, ...] = ("cosine_annealing_warmup", "reduce_on_plateau")
"""Scheduler families included in the phase-2 sweep."""

PHASE_TWO_PROXY_MODELS: tuple[str, ...] = ("convnext", "xlstm")
"""Proxy model families included in the phase-2 sweep."""

PHASE_TWO_SEEDS: tuple[int, int, int] = (0, 42, 2003)
"""Fixed seeds used for the phase-2 sweep grid."""

PHASE_TWO_STATE_SCHEMA_VERSION = 1
"""Schema version for the phase-2 sweep state file."""


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(UTC).isoformat()


def _phase_two_score(payload: Mapping[str, Any]) -> float:
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
                    score = _phase_two_score(output_data)
                    if score > 0.0:
                        return score
    return 0.0


def _feature_artifact_trial_payload(feature_artifact_path: Path) -> dict[str, Any]:
    """Load the phase-1 best-feature artifact used as phase-2 input."""

    if not feature_artifact_path.exists():
        raise FileNotFoundError(
            f"Phase-1 best feature artifact not found at '{feature_artifact_path}'."
        )
    payload = _read_json(feature_artifact_path)
    if not isinstance(payload, dict):
        raise ValueError("Phase-1 best feature artifact must be a JSON object.")
    return payload


def _phase_two_feature_config(feature_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the winning feature config payload from phase 1."""

    trial = feature_artifact.get("trial")
    if not isinstance(trial, Mapping):
        raise ValueError("Phase-1 best feature artifact is missing the trial payload.")

    feature_pipeline = trial.get("feature_name")
    if not isinstance(feature_pipeline, str):
        raise ValueError("Phase-1 best feature artifact is missing feature_name.")
    return {
        "feature_name": feature_pipeline,
        "feature_config": _feature_pipeline_for_trial(feature_pipeline).to_dict(),
        "feature_trial": dict(trial),
    }


def build_phase_two_command(config_path: Path, run_name: str) -> list[str]:
    """Build the isolated subprocess command for one phase-2 trial."""

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


def _scheduler_config_for_trial(scheduler_name: str, *, total_epochs: int) -> SchedulerConfig:
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

        dataset = replace(base_config.dataset, train_split="train_small", valid_split="valid_small")
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
                for seed in PHASE_TWO_SEEDS:
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


@dataclass(frozen=True, slots=True)
class PhaseTwoSweepState:
    """Persistent state for the phase-2 sweep."""

    schema_version: int = PHASE_TWO_STATE_SCHEMA_VERSION
    output_dir: str = ""
    phase_one_best_feature_path: str = ""
    completed_trials: dict[str, PhaseTwoTrialRecord] = field(default_factory=dict)
    best_trial_id: str | None = None
    best_validation_macro_f1: float | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema_version != PHASE_TWO_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported phase-2 state schema version.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the sweep state."""

        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseTwoSweepState":
        """Build phase-2 state from JSON."""

        payload = dict(data)
        payload["completed_trials"] = {
            key: PhaseTwoTrialRecord.from_dict(value)
            for key, value in payload.get("completed_trials", {}).items()
        }
        if payload.get("best_validation_macro_f1") is not None:
            payload["best_validation_macro_f1"] = float(payload["best_validation_macro_f1"])
        return cls(**payload)

    @classmethod
    def fresh(cls, output_dir: Path, phase_one_best_feature_path: Path) -> "PhaseTwoSweepState":
        """Create a new empty state for an output directory."""

        return cls(
            output_dir=str(output_dir), phase_one_best_feature_path=str(phase_one_best_feature_path)
        )


class PhaseTwoSweepRunner:
    """Run the phase-2 hyperparameter sweep using child processes."""

    def __init__(self, output_dir: Path, base_config: ExperimentConfig | None = None) -> None:
        self.output_dir = output_dir
        self.phase_dir = self.output_dir / "phase_2"
        self.state_path = self.phase_dir / "state.json"
        self.best_optim_path = self.phase_dir / "best_optim.json"
        self.base_config = base_config or ExperimentConfig()
        self.phase_one_best_feature_path = self.output_dir / "phase_1" / "best_feature.json"

    def load_state(self) -> PhaseTwoSweepState:
        """Load the persisted sweep state or create a new one."""

        if not self.state_path.exists():
            return PhaseTwoSweepState.fresh(self.output_dir, self.phase_one_best_feature_path)
        return PhaseTwoSweepState.from_dict(_read_json(self.state_path))

    def _save_state(self, state: PhaseTwoSweepState) -> None:
        """Persist the sweep state and best-optimization summary."""

        _atomic_write_json(self.state_path, state.to_dict())
        if state.best_trial_id is not None:
            best_trial = state.completed_trials[state.best_trial_id]
            _atomic_write_json(
                self.best_optim_path,
                {
                    "schema_version": PHASE_TWO_STATE_SCHEMA_VERSION,
                    "trial": best_trial.to_dict(),
                },
            )

    def _trial_run_name(self, trial: PhaseTwoTrialSpec) -> str:
        """Return the child run name for one trial."""

        return trial.trial_id

    def _trial_output_paths(self, trial: PhaseTwoTrialSpec) -> tuple[Path, Path, Path]:
        """Return config, child state, and trial directory paths for one trial."""

        trial_dir = self.phase_dir / "runs" / trial.trial_id
        config_path = trial_dir / "temp_config.json"
        child_state_path = self.output_dir / "phase_2" / "runs" / trial.trial_id / "state.json"
        return trial_dir, config_path, child_state_path

    def _build_trial_config(
        self, trial: PhaseTwoTrialSpec, feature_payload: Mapping[str, Any]
    ) -> ExperimentConfig:
        """Build the concrete experiment config for a trial."""

        config = trial.to_config(self.base_config)
        feature_config_payload = feature_payload.get("feature_config")
        if isinstance(feature_config_payload, Mapping):
            config = replace(
                config, features=_feature_pipeline_for_trial(feature_config_payload["name"])
            )
        return config

    def _run_trial(
        self, trial: PhaseTwoTrialSpec, feature_payload: Mapping[str, Any]
    ) -> PhaseTwoTrialRecord:
        """Execute one trial in an isolated child process and return its record."""

        trial_dir, config_path, child_state_path = self._trial_output_paths(trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_trial_config(trial, feature_payload)
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        command = build_phase_two_command(config_path, self._trial_run_name(trial))
        subprocess.run(  # noqa: S603
            command,
            check=True,
            env=build_isolated_subprocess_env(),
            text=True,
            capture_output=True,
        )

        summary = _read_json(child_state_path) if child_state_path.exists() else {}
        validation_macro_f1 = _phase_two_score(summary)
        return PhaseTwoTrialRecord(
            trial_id=trial.trial_id,
            feature_name=trial.feature_name,
            proxy_model=trial.proxy_model,
            weight_decay=trial.weight_decay,
            scheduler_name=trial.scheduler_name,
            seed=trial.seed,
            run_name=self._trial_run_name(trial),
            config_path=str(config_path),
            child_state_path=str(child_state_path),
            validation_macro_f1=validation_macro_f1,
            completed_at=_utc_now(),
        )

    def execute(self) -> dict[str, Any]:
        """Run the full phase-2 sweep, skipping completed trials."""

        feature_artifact = _feature_artifact_trial_payload(self.phase_one_best_feature_path)
        feature_summary = _phase_two_feature_config(feature_artifact)
        feature_name = feature_summary["feature_name"]

        state = self.load_state()
        best_trial_id = state.best_trial_id
        best_score = (
            state.best_validation_macro_f1 if state.best_validation_macro_f1 is not None else -1.0
        )

        for trial in build_phase_two_trials(feature_name):
            if trial.trial_id in state.completed_trials:
                existing = state.completed_trials[trial.trial_id]
                if existing.validation_macro_f1 >= best_score:
                    best_trial_id = trial.trial_id
                    best_score = existing.validation_macro_f1
                continue

            trial_record = self._run_trial(trial, feature_artifact)
            completed_trials = {**state.completed_trials, trial.trial_id: trial_record}
            if trial_record.validation_macro_f1 >= best_score:
                best_trial_id = trial.trial_id
                best_score = trial_record.validation_macro_f1

            state = PhaseTwoSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_one_best_feature_path=state.phase_one_best_feature_path,
                completed_trials=completed_trials,
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
            state = PhaseTwoSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_one_best_feature_path=state.phase_one_best_feature_path,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score,
                created_at=state.created_at,
                updated_at=_utc_now(),
            )
            self._save_state(state)

        return {
            "phase": "phase-2",
            "feature_source": feature_summary,
            "total_trials": len(build_phase_two_trials(feature_name)),
            "completed_trials": len(state.completed_trials),
            "best_trial": state.completed_trials[best_trial_id].to_dict()
            if best_trial_id
            else None,
            "best_validation_macro_f1": best_score if best_score >= 0.0 else None,
            "state_path": str(self.state_path),
            "best_optim_path": str(self.best_optim_path),
        }
