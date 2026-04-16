"""Phase-2 global hyperparameter sweep orchestration."""

import json
import logging
import sys
from collections.abc import Mapping
from dataclasses import replace
from importlib import import_module
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ...config import ExperimentConfig
from ..phase_one import _atomic_write_json, _feature_pipeline_for_trial, _read_json
from ..services import build_isolated_subprocess_env
from ..state import (
    PipelineStateStore,
    load_json_artifact,
    load_json_artifact_or_raise,
    save_json_artifact,
)
from ..sweep_utils import (
    apply_trial_cap,
    build_trial_output_paths,
    iter_nested_payloads,
    run_subprocess_with_live_output,
    summary_from_completed_process,
    utc_now,
)
from .constants import PHASE_TWO_STATE_SCHEMA_VERSION
from .state import PhaseTwoSweepState
from .trials import PhaseTwoTrialRecord, PhaseTwoTrialSpec, build_phase_two_trials


def _phase_two_package() -> Any:
    """Return the imported phase-two package containing trial-building and subprocess utilities."""
    return import_module("speech_recognition.orchestration.phase_two")


def _build_phase_two_trials(feature_name: str) -> tuple[Any, ...]:
    """Return the full trial list for phase 2 based on the winning feature."""
    package = _phase_two_package()
    builder = getattr(package, "build_phase_two_trials", build_phase_two_trials)
    return builder(feature_name)


def _run_subprocess(command: list[str], env: Mapping[str, str], check: bool) -> Any:
    """Run a subprocess with live output and return the completed process."""
    package = _phase_two_package()
    runner = getattr(package, "run_subprocess_with_live_output", run_subprocess_with_live_output)
    return runner(command, env=env, check=check)


def _phase_two_group_key(record: PhaseTwoTrialRecord) -> tuple[str, str, float, str]:
    """Return the grouping key that identifies one phase-2 configuration."""
    return (
        record.feature_name,
        record.proxy_model,
        record.weight_decay,
        record.scheduler_name,
    )


def _select_phase_two_winner(
    records: Mapping[str, PhaseTwoTrialRecord],
) -> tuple[str | None, float | None, dict[str, Any] | None]:
    """Select the phase-2 winner using mean macro-F1 over fixed seeds."""
    if not records:
        return None, None, None

    grouped: dict[tuple[str, str, float, str], list[PhaseTwoTrialRecord]] = {}
    for record in records.values():
        grouped.setdefault(_phase_two_group_key(record), []).append(record)

    best_key: tuple[str, str, float, str] | None = None
    best_mean = -1.0
    best_records: list[PhaseTwoTrialRecord] = []
    for key, grouped_records in grouped.items():
        mean_score = sum(item.validation_macro_f1 for item in grouped_records) / len(
            grouped_records
        )
        if mean_score > best_mean:
            best_key = key
            best_mean = mean_score
            best_records = grouped_records

    if best_key is None or not best_records:
        return None, None, None

    representative = max(best_records, key=lambda item: item.validation_macro_f1)
    aggregate = {
        "feature_name": best_key[0],
        "proxy_model": best_key[1],
        "weight_decay": best_key[2],
        "scheduler_name": best_key[3],
        "seed_count": len(best_records),
        "mean_validation_macro_f1": best_mean,
        "std_validation_macro_f1": (
            (
                sum((item.validation_macro_f1 - best_mean) ** 2 for item in best_records)
                / len(best_records)
            )
            ** 0.5
        ),
    }
    return representative.trial_id, best_mean, aggregate


def _phase_two_score(payload: Mapping[str, Any]) -> float:
    """Extract the validation macro-F1 score from a child payload."""
    for nested_payload in iter_nested_payloads(payload):
        score = nested_payload.get("validation_macro_f1")
        if isinstance(score, int | float):
            return float(score)

        metrics = nested_payload.get("metrics")
        if isinstance(metrics, Mapping):
            metric_value = metrics.get("validation_macro_f1", metrics.get("macro_f1"))
            if isinstance(metric_value, int | float):
                return float(metric_value)
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


def build_phase_two_command(config_path: Path, run_name: str, output_dir: Path) -> list[str]:
    """Build the isolated subprocess command for one phase-2 trial."""
    return [
        sys.executable,
        "-m",
        "speech_recognition.cli",
        "run-single-train",
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]


class PhaseTwoSweepRunner:
    """Run the phase-2 hyperparameter sweep using child processes."""

    def __init__(
        self,
        output_dir: Path,
        base_config: ExperimentConfig | None = None,
        run_name: str = "default",
        use_mlflow_persistence: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.phase_dir = self.output_dir / "phase_2"
        self.state_path = self.phase_dir / "state.json"
        self.best_optim_path = self.phase_dir / "best_optim.json"
        self.base_config = base_config or ExperimentConfig()
        self.phase_one_best_feature_path = self.output_dir / "phase_1" / "best_feature.json"
        self.run_name = run_name
        self.use_mlflow_persistence = use_mlflow_persistence
        self.state_store = PipelineStateStore(
            output_dir,
            use_mlflow=use_mlflow_persistence,
            tracking_uri=self.base_config.mlflow.tracking_uri,
            experiment_name=self.base_config.mlflow.experiment_name,
        )

    def load_state(self) -> PhaseTwoSweepState:
        """Load the persisted sweep state or create a new one."""
        if self.use_mlflow_persistence:
            payload = load_json_artifact(
                self.state_store,
                self.state_path,
                self.run_name,
                "pipeline_state/phase_2/state.json",
            )
            if payload is None:
                return PhaseTwoSweepState.fresh(self.output_dir, self.phase_one_best_feature_path)
            return PhaseTwoSweepState.from_dict(payload)

        if not self.state_path.exists():
            return PhaseTwoSweepState.fresh(self.output_dir, self.phase_one_best_feature_path)
        return PhaseTwoSweepState.from_dict(_read_json(self.state_path))

    def _save_state(self, state: PhaseTwoSweepState) -> None:
        """Persist the sweep state and best-optimization summary."""
        payload = state.to_dict()
        if self.use_mlflow_persistence:
            save_json_artifact(
                self.state_store,
                self.state_path,
                self.run_name,
                "pipeline_state/phase_2/state.json",
                payload,
            )
        else:
            _atomic_write_json(self.state_path, payload)
        if state.best_trial_id is not None:
            best_trial = state.completed_trials[state.best_trial_id]
            _, _, aggregate = _select_phase_two_winner(state.completed_trials)
            best_payload = {
                "schema_version": PHASE_TWO_STATE_SCHEMA_VERSION,
                "trial": best_trial.to_dict(),
                "aggregate": aggregate,
            }
            if self.use_mlflow_persistence:
                save_json_artifact(
                    self.state_store,
                    self.best_optim_path,
                    self.run_name,
                    "pipeline_state/phase_2/best_optim.json",
                    best_payload,
                )
            else:
                _atomic_write_json(self.best_optim_path, best_payload)

    def _trial_run_name(self, trial: PhaseTwoTrialSpec) -> str:
        """Return the child run name for one trial."""
        return trial.trial_id

    def _trial_output_paths(self, trial: PhaseTwoTrialSpec) -> tuple[Path, Path, Path]:
        """Return config, child state, and trial directory paths for one trial."""
        return build_trial_output_paths(self.phase_dir, trial.trial_id)

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

        command = build_phase_two_command(config_path, self._trial_run_name(trial), self.output_dir)
        completed_process = _run_subprocess(
            command,
            env=build_isolated_subprocess_env(),
            check=True,
        )

        summary = summary_from_completed_process(
            child_state_path,
            completed_process,
            read_json=_read_json,
        )
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
            completed_at=utc_now(),
        )

    def execute(self) -> dict[str, Any]:
        """Run the full phase-2 sweep, skipping completed trials."""
        if self.use_mlflow_persistence:
            feature_artifact = load_json_artifact_or_raise(
                self.state_store,
                self.phase_one_best_feature_path,
                self.run_name,
                "pipeline_state/phase_1/best_feature.json",
                f"Phase-1 best feature artifact not found for run '{self.run_name}'.",
            )
        else:
            feature_artifact = _feature_artifact_trial_payload(self.phase_one_best_feature_path)
        feature_summary = _phase_two_feature_config(feature_artifact)
        feature_name = feature_summary["feature_name"]
        trials = apply_trial_cap(_build_phase_two_trials(feature_name))
        total_trials = len(trials)
        logger = logging.getLogger(__name__)

        state = self.load_state()
        best_trial_id, best_score, aggregate = _select_phase_two_winner(state.completed_trials)

        for trial_index, trial in enumerate(
            tqdm(trials, total=total_trials, desc="phase-2 trials", leave=False), start=1
        ):
            if trial.trial_id in state.completed_trials:
                logger.info(
                    "[phase-2] [%d/%d] skipping completed %s",
                    trial_index,
                    total_trials,
                    trial.trial_id,
                )
                continue

            logger.info(
                "[phase-2] [%d/%d] running %s",
                trial_index,
                total_trials,
                trial.trial_id,
            )

            trial_record = self._run_trial(trial, feature_artifact)
            logger.info(
                "[phase-2] finished %s validation_macro_f1=%.6f",
                trial.trial_id,
                trial_record.validation_macro_f1,
            )
            completed_trials = {**state.completed_trials, trial.trial_id: trial_record}
            best_trial_id, best_score, aggregate = _select_phase_two_winner(completed_trials)

            state = PhaseTwoSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_one_best_feature_path=state.phase_one_best_feature_path,
                completed_trials=completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score,
                created_at=state.created_at,
                updated_at=trial_record.completed_at,
            )
            self._save_state(state)

        if state.completed_trials:
            best_trial_id, best_score, aggregate = _select_phase_two_winner(state.completed_trials)
            state = PhaseTwoSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_one_best_feature_path=state.phase_one_best_feature_path,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score,
                created_at=state.created_at,
                updated_at=utc_now(),
            )
            self._save_state(state)

        return {
            "phase": "phase-2",
            "feature_source": feature_summary,
            "total_trials": len(trials),
            "completed_trials": len(state.completed_trials),
            "best_trial": state.completed_trials[best_trial_id].to_dict()
            if best_trial_id
            else None,
            "best_validation_macro_f1": best_score,
            "best_aggregate": aggregate,
            "state_path": str(self.state_path),
            "best_optim_path": str(self.best_optim_path),
        }
