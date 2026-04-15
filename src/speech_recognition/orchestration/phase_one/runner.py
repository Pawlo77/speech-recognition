"""Phase-1 feature ablation sweep orchestration."""

import json
import logging
import subprocess
import sys
from collections.abc import Mapping
from importlib import import_module
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ...config import ExperimentConfig
from ..services import build_isolated_subprocess_env
from ..sweep_utils import (
    apply_trial_cap,
    iter_nested_payloads,
    run_subprocess_with_live_output,
    summary_from_completed_process,
    utc_now,
)
from .constants import PHASE_ONE_STATE_SCHEMA_VERSION
from .state import PhaseOneSweepState
from .trials import PhaseOneTrialRecord, PhaseOneTrialSpec, build_phase_one_trials


def _phase_one_package() -> Any:
    """Return the imported phase-one package for accessing trial builders and subprocess runners."""
    return import_module("speech_recognition.orchestration.phase_one")


def _build_phase_one_trials() -> tuple[Any, ...]:
    """Return the trial specifications for phase 1."""
    package = _phase_one_package()
    builder = getattr(package, "build_phase_one_trials", build_phase_one_trials)
    return builder()


def _run_subprocess(command: list[str], env: Mapping[str, str], check: bool) -> Any:
    """Run a subprocess command with live output and return the completed process."""
    package = _phase_one_package()
    runner = getattr(package, "run_subprocess_with_live_output", run_subprocess_with_live_output)
    return runner(command, env=env, check=check)


def _phase_one_group_key(record: PhaseOneTrialRecord) -> tuple[str, str]:
    """Return the grouping key that identifies one phase-1 configuration."""
    return record.feature_name, record.proxy_model


def _select_phase_one_winner(
    records: Mapping[str, PhaseOneTrialRecord],
) -> tuple[str | None, float | None, dict[str, Any] | None]:
    """Select the phase-1 winner using mean macro-F1 over fixed seeds."""
    if not records:
        return None, None, None

    grouped: dict[tuple[str, str], list[PhaseOneTrialRecord]] = {}
    for record in records.values():
        grouped.setdefault(_phase_one_group_key(record), []).append(record)

    best_key: tuple[str, str] | None = None
    best_mean = -1.0
    best_records: list[PhaseOneTrialRecord] = []
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


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Load JSON from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _trial_score(payload: Mapping[str, Any]) -> float:
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


def _summary_from_completed_process(
    child_state_path: Path, completed_process: subprocess.CompletedProcess[str]
) -> dict[str, Any]:
    return summary_from_completed_process(
        child_state_path,
        completed_process,
        read_json=_read_json,
    )


def build_phase_one_command(config_path: Path, run_name: str, output_dir: Path) -> list[str]:
    """Build the isolated subprocess command for one phase-1 trial."""
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


class PhaseOneSweepRunner:
    """Run the phase-1 feature ablation sweep using child processes."""

    def __init__(self, output_dir: Path, base_config: ExperimentConfig | None = None) -> None:
        self.output_dir: Path = output_dir
        """Root output directory for all sweep artifacts."""
        self.phase_dir: Path = self.output_dir / "phase_1"
        """Phase 1 subdirectory."""
        self.state_path: Path = self.phase_dir / "state.json"
        """Path to the persisted sweep state file."""
        self.best_feature_path: Path = self.phase_dir / "best_feature.json"
        """Path to the best feature summary file."""
        self.base_config: ExperimentConfig = base_config or ExperimentConfig()
        """Base experiment config used for all trials."""

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
            _, _, aggregate = _select_phase_one_winner(state.completed_trials)
            _atomic_write_json(
                self.best_feature_path,
                {
                    "schema_version": PHASE_ONE_STATE_SCHEMA_VERSION,
                    "trial": best_trial.to_dict(),
                    "aggregate": aggregate,
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

        command = build_phase_one_command(config_path, self._trial_run_name(trial), self.output_dir)
        completed_process = _run_subprocess(
            command,
            env=build_isolated_subprocess_env(),
            check=True,
        )

        summary = _summary_from_completed_process(child_state_path, completed_process)
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
            completed_at=utc_now(),
        )

    def execute(self) -> dict[str, Any]:
        """Run the full phase-1 sweep, skipping completed trials."""
        trials = apply_trial_cap(_build_phase_one_trials())
        total_trials = len(trials)
        logger = logging.getLogger(__name__)
        state = self.load_state()
        best_trial_id, best_score, aggregate = _select_phase_one_winner(state.completed_trials)

        for trial_index, trial in enumerate(
            tqdm(trials, total=total_trials, desc="phase-1 trials", leave=False), start=1
        ):
            if trial.trial_id in state.completed_trials:
                logger.info(
                    "[phase-1] [%d/%d] skipping completed %s",
                    trial_index,
                    total_trials,
                    trial.trial_id,
                )
                continue

            logger.info(
                "[phase-1] [%d/%d] running %s",
                trial_index,
                total_trials,
                trial.trial_id,
            )

            trial_record = self._run_trial(trial)
            logger.info(
                "[phase-1] finished %s validation_macro_f1=%.6f",
                trial.trial_id,
                trial_record.validation_macro_f1,
            )
            completed_trials = {**state.completed_trials, trial.trial_id: trial_record}
            best_trial_id, best_score, aggregate = _select_phase_one_winner(completed_trials)
            state = PhaseOneSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                completed_trials=completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score,
                created_at=state.created_at,
                updated_at=utc_now(),
            )

            self._save_state(state)

        if state.completed_trials:
            best_trial_id, best_score, aggregate = _select_phase_one_winner(state.completed_trials)
            state = PhaseOneSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score,
                created_at=state.created_at,
                updated_at=utc_now(),
            )
            self._save_state(state)

        return {
            "phase": "phase-1",
            "total_trials": len(trials),
            "completed_trials": len(state.completed_trials),
            "best_trial": state.completed_trials[best_trial_id].to_dict()
            if best_trial_id
            else None,
            "best_validation_macro_f1": best_score,
            "best_aggregate": aggregate,
            "state_path": str(self.state_path),
            "best_feature_path": str(self.best_feature_path),
        }
