"""Phase-3 architecture comparison sweep orchestration."""

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
from ..phase_one import _atomic_write_json, _read_json
from ..phase_two import _phase_two_feature_config
from ..services import build_isolated_subprocess_env
from ..state import (
    PipelineStateStore,
    _serialize,
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
from .constants import PHASE_THREE_STATE_SCHEMA_VERSION
from .state import PhaseThreeSweepState
from .trials import PhaseThreeTrialRecord, PhaseThreeTrialSpec, build_phase_three_trials


def _phase_three_package() -> Any:
    """Import and return the phase-three-specific implementation package."""
    return import_module("speech_recognition.orchestration.phase_three")


def _build_phase_three_trials() -> tuple[Any, ...]:
    """Return the list of trial specifications for the phase-3 sweep.

    SSamba trials are slow on the macOS Mamba backend (no fused parallel
    scan), so they are scheduled last. This keeps faster families churning
    while the long-running ones happen at the end. Trial identities and
    parameters are unchanged; only execution order is affected.
    """
    package = _phase_three_package()
    builder = getattr(package, "build_phase_three_trials", build_phase_three_trials)
    trials = builder()
    return tuple(
        sorted(
            trials,
            key=lambda trial: (1 if getattr(trial, "family", "") == "ssamba" else 0,),
        )
    )


def _run_subprocess(command: list[str], env: Mapping[str, str], check: bool) -> Any:
    """Run a subprocess with live output and return the completed process."""
    package = _phase_three_package()
    runner = getattr(package, "run_subprocess_with_live_output", run_subprocess_with_live_output)
    return runner(command, env=env, check=check)


def _phase_three_group_key(record: PhaseThreeTrialRecord) -> tuple[str, str]:
    """Return the grouping key that identifies one phase-3 architecture config."""
    return record.family, json.dumps(record.architecture_params, sort_keys=True)


def _phase_three_group_stats(
    records: Mapping[str, PhaseThreeTrialRecord],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Compute per-configuration aggregate statistics over seed runs."""
    grouped: dict[tuple[str, str], list[PhaseThreeTrialRecord]] = {}
    for record in records.values():
        grouped.setdefault(_phase_three_group_key(record), []).append(record)

    stats: dict[tuple[str, str], dict[str, Any]] = {}
    for key, grouped_records in grouped.items():
        mean_score = sum(item.validation_macro_f1 for item in grouped_records) / len(
            grouped_records
        )
        representative = max(grouped_records, key=lambda item: item.validation_macro_f1)
        stats[key] = {
            "family": key[0],
            "architecture_params": representative.architecture_params,
            "seed_count": len(grouped_records),
            "mean_validation_macro_f1": mean_score,
            "std_validation_macro_f1": (
                (
                    sum((item.validation_macro_f1 - mean_score) ** 2 for item in grouped_records)
                    / len(grouped_records)
                )
                ** 0.5
            ),
            "representative_trial_id": representative.trial_id,
            "representative_score": representative.validation_macro_f1,
        }
    return stats


def _phase_three_score(payload: Mapping[str, Any]) -> float:
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


def _phase_three_core_command_score(payload: Mapping[str, Any]) -> float | None:
    """Extract core-command macro-F1 from a child payload when available."""
    for nested_payload in iter_nested_payloads(payload):
        direct = nested_payload.get("core_command_macro_f1")
        if isinstance(direct, int | float):
            return float(direct)

        metrics = nested_payload.get("metrics")
        if isinstance(metrics, Mapping):
            metric_value = metrics.get("core_command_macro_f1")
            if isinstance(metric_value, int | float):
                return float(metric_value)
    return None


def _phase_three_trial_metadata(trial: Mapping[str, Any]) -> dict[str, Any]:
    """Return a stable JSON-serializable trial metadata payload."""
    return {key: _serialize(value) for key, value in trial.items()}


def _phase_three_feature_config(feature_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the phase-1 winning feature config payload."""
    return _phase_two_feature_config(feature_artifact)


def _phase_three_optimizer_config(optim_artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the phase-2 winning optimization payload."""
    trial = optim_artifact.get("trial")
    if not isinstance(trial, Mapping):
        raise ValueError("Phase-2 best optimization artifact is missing the trial payload.")

    scheduler_name = trial.get("scheduler_name")
    if not isinstance(scheduler_name, str):
        raise ValueError("Phase-2 best optimization artifact is missing scheduler_name.")

    weight_decay = trial.get("weight_decay")
    if not isinstance(weight_decay, int | float):
        raise ValueError("Phase-2 best optimization artifact is missing weight_decay.")

    proxy_model = trial.get("proxy_model")
    if not isinstance(proxy_model, str):
        raise ValueError("Phase-2 best optimization artifact is missing proxy_model.")

    return {
        "scheduler_name": scheduler_name,
        "weight_decay": float(weight_decay),
        "proxy_model": proxy_model,
        "trial": dict(trial),
    }


def build_phase_three_command(config_path: Path, run_name: str, output_dir: Path) -> list[str]:
    """Build the isolated subprocess command for one phase-3 trial."""
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


def _read_artifact(path: Path, description: str) -> dict[str, Any]:
    """Read and validate a phase artifact JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at '{path}'.")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return payload


class PhaseThreeSweepRunner:
    """Run the phase-3 architecture comparison sweep using child processes."""

    def __init__(
        self,
        output_dir: Path,
        base_config: ExperimentConfig | None = None,
        run_name: str = "default",
        use_mlflow_persistence: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.phase_dir = self.output_dir / "phase_3"
        self.state_path = self.phase_dir / "state.json"
        self.best_backbones_path = self.phase_dir / "best_backbones.json"
        self.family_winners_path = self.phase_dir / "family_winners.json"
        self.base_config = base_config or ExperimentConfig()
        self.run_name = run_name
        self.use_mlflow_persistence = use_mlflow_persistence
        self.state_store = PipelineStateStore(
            output_dir,
            use_mlflow=use_mlflow_persistence,
            tracking_uri=self.base_config.mlflow.tracking_uri,
            experiment_name=self.base_config.mlflow.experiment_name,
        )
        self.phase_one_best_feature_path = self.output_dir / "phase_1" / "best_feature.json"
        self.phase_two_best_optim_path = self.output_dir / "phase_2" / "best_optim.json"

    def load_state(self) -> PhaseThreeSweepState:
        """Load the persisted sweep state or create a new one."""
        if self.use_mlflow_persistence:
            payload = load_json_artifact(
                self.state_store,
                self.state_path,
                self.run_name,
                "pipeline_state/phase_3/state.json",
            )
            if payload is None:
                return PhaseThreeSweepState.fresh(
                    self.output_dir,
                    self.phase_one_best_feature_path,
                    self.phase_two_best_optim_path,
                )
            return PhaseThreeSweepState.from_dict(payload)

        if not self.state_path.exists():
            return PhaseThreeSweepState.fresh(
                self.output_dir,
                self.phase_one_best_feature_path,
                self.phase_two_best_optim_path,
            )
        return PhaseThreeSweepState.from_dict(_read_json(self.state_path))

    def _save_state(self, state: PhaseThreeSweepState) -> None:
        """Persist the sweep state and selection artifacts."""
        payload = state.to_dict()
        if self.use_mlflow_persistence:
            save_json_artifact(
                self.state_store,
                self.state_path,
                self.run_name,
                "pipeline_state/phase_3/state.json",
                payload,
            )
        else:
            _atomic_write_json(self.state_path, payload)
        if state.best_trial_id is not None:
            best_trial = state.completed_trials[state.best_trial_id]
            grouped_stats = _phase_three_group_stats(state.completed_trials)
            best_payload = {
                "schema_version": PHASE_THREE_STATE_SCHEMA_VERSION,
                "best_trial": best_trial.to_dict(),
                "best_aggregate": next(
                    (
                        payload
                        for payload in grouped_stats.values()
                        if payload["representative_trial_id"] == state.best_trial_id
                    ),
                    None,
                ),
                "top_three_trials": [
                    state.completed_trials[trial_id].to_dict()
                    for trial_id in state.top_three_trial_ids
                ],
            }
            if self.use_mlflow_persistence:
                save_json_artifact(
                    self.state_store,
                    self.best_backbones_path,
                    self.run_name,
                    "pipeline_state/phase_3/best_backbones.json",
                    best_payload,
                )
            else:
                _atomic_write_json(self.best_backbones_path, best_payload)
        if state.family_winner_ids:
            family_payload = {
                "schema_version": PHASE_THREE_STATE_SCHEMA_VERSION,
                "family_winners": {
                    family: state.completed_trials[trial_id].to_dict()
                    for family, trial_id in state.family_winner_ids.items()
                },
            }
            if self.use_mlflow_persistence:
                save_json_artifact(
                    self.state_store,
                    self.family_winners_path,
                    self.run_name,
                    "pipeline_state/phase_3/family_winners.json",
                    family_payload,
                )
            else:
                _atomic_write_json(self.family_winners_path, family_payload)

    def _trial_run_name(self, trial: PhaseThreeTrialSpec) -> str:
        """Return the run name to use for a child trial process."""
        return trial.trial_id

    def _trial_output_paths(self, trial: PhaseThreeTrialSpec) -> tuple[Path, Path, Path]:
        """Return the trial directory, config path, and child state path for a trial."""
        return build_trial_output_paths(self.phase_dir, trial.trial_id)

    def _build_trial_config(
        self,
        trial: PhaseThreeTrialSpec,
        feature_payload: Mapping[str, Any],
        optimizer_payload: Mapping[str, Any],
    ) -> ExperimentConfig:
        """Build the ExperimentConfig to use for a child trial process."""
        feature_name = str(feature_payload["feature_name"])
        config = trial.to_config(self.base_config, feature_name, optimizer_payload)
        model_config = trial._build_model_config(config.model)
        return replace(config, model=model_config)

    def _run_trial(
        self,
        trial: PhaseThreeTrialSpec,
        feature_payload: Mapping[str, Any],
        optimizer_payload: Mapping[str, Any],
    ) -> PhaseThreeTrialRecord:
        """Run one trial as a child subprocess and return the completed trial record."""
        trial_dir, config_path, child_state_path = self._trial_output_paths(trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_trial_config(trial, feature_payload, optimizer_payload)
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        command = build_phase_three_command(
            config_path,
            self._trial_run_name(trial),
            self.output_dir,
        )
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
        validation_macro_f1 = _phase_three_score(summary)
        core_command_macro_f1 = _phase_three_core_command_score(summary)
        return PhaseThreeTrialRecord(
            trial_id=trial.trial_id,
            family=trial.family,
            seed=trial.seed,
            architecture_params=dict(trial.architecture_params),
            run_name=self._trial_run_name(trial),
            config_path=str(config_path),
            child_state_path=str(child_state_path),
            validation_macro_f1=validation_macro_f1,
            core_command_macro_f1=core_command_macro_f1,
            completed_at=utc_now(),
        )

    def _select_winners(
        self, records: Mapping[str, PhaseThreeTrialRecord]
    ) -> tuple[str | None, tuple[str, ...], dict[str, str]]:
        """Select the best trial, top three trials, and family
        winners from the completed trial records."""
        if not records:
            return None, (), {}

        grouped_stats = _phase_three_group_stats(records)
        sorted_groups = sorted(
            grouped_stats.values(),
            key=lambda payload: payload["mean_validation_macro_f1"],
            reverse=True,
        )
        best_trial_id = str(sorted_groups[0]["representative_trial_id"])
        top_three_trial_ids = tuple(
            str(payload["representative_trial_id"]) for payload in sorted_groups[:3]
        )

        family_winner_ids: dict[str, str] = {}
        family_groups: dict[str, list[dict[str, Any]]] = {}
        for payload in sorted_groups:
            family_groups.setdefault(str(payload["family"]), []).append(payload)

        for family, family_payloads in family_groups.items():
            winner = max(family_payloads, key=lambda payload: payload["mean_validation_macro_f1"])
            family_winner_ids[family] = str(winner["representative_trial_id"])

        return best_trial_id, top_three_trial_ids, family_winner_ids

    def execute(self) -> dict[str, Any]:
        """Run the full phase-3 sweep, skipping completed trials."""
        if self.use_mlflow_persistence:
            feature_artifact = load_json_artifact_or_raise(
                self.state_store,
                self.phase_one_best_feature_path,
                self.run_name,
                "pipeline_state/phase_1/best_feature.json",
                f"Phase-1 best feature artifact not found for run '{self.run_name}'.",
            )
        else:
            feature_artifact = _read_artifact(
                self.phase_one_best_feature_path, "Phase-1 best feature artifact"
            )
        feature_summary = _phase_three_feature_config(feature_artifact)
        if self.use_mlflow_persistence:
            optimizer_artifact = load_json_artifact_or_raise(
                self.state_store,
                self.phase_two_best_optim_path,
                self.run_name,
                "pipeline_state/phase_2/best_optim.json",
                f"Phase-2 best optimization artifact not found for run '{self.run_name}'.",
            )
        else:
            optimizer_artifact = _read_artifact(
                self.phase_two_best_optim_path, "Phase-2 best optimization artifact"
            )
        optimizer_summary = _phase_three_optimizer_config(optimizer_artifact)

        state = self.load_state()
        trials = apply_trial_cap(_build_phase_three_trials())
        total_trials = len(trials)
        logger = logging.getLogger(__name__)
        best_trial_id, top_three_trial_ids, family_winner_ids = self._select_winners(
            state.completed_trials
        )
        best_score = (
            state.completed_trials[best_trial_id].validation_macro_f1
            if best_trial_id is not None
            else None
        )

        for trial_index, trial in enumerate(
            tqdm(trials, total=total_trials, desc="phase-3 trials", leave=False), start=1
        ):
            if trial.trial_id in state.completed_trials:
                logger.info(
                    "[phase-3] [%d/%d] skipping completed %s",
                    trial_index,
                    total_trials,
                    trial.trial_id,
                )
                continue

            logger.info(
                "[phase-3] [%d/%d] running %s",
                trial_index,
                total_trials,
                trial.trial_id,
            )

            trial_record = self._run_trial(trial, feature_summary, optimizer_summary)
            logger.info(
                "[phase-3] finished %s validation_macro_f1=%.6f",
                trial.trial_id,
                trial_record.validation_macro_f1,
            )
            completed_trials = {**state.completed_trials, trial.trial_id: trial_record}

            best_trial_id, top_three_trial_ids, family_winner_ids = self._select_winners(
                completed_trials
            )
            best_score = (
                completed_trials[best_trial_id].validation_macro_f1
                if best_trial_id is not None
                else None
            )
            state = PhaseThreeSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_one_best_feature_path=state.phase_one_best_feature_path,
                phase_two_best_optim_path=state.phase_two_best_optim_path,
                completed_trials=completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score,
                top_three_trial_ids=top_three_trial_ids,
                family_winner_ids=family_winner_ids,
                created_at=state.created_at,
                updated_at=trial_record.completed_at,
            )
            self._save_state(state)

        if state.completed_trials:
            best_trial_id, top_three_trial_ids, family_winner_ids = self._select_winners(
                state.completed_trials
            )
            if best_trial_id is not None:
                best_score = state.completed_trials[best_trial_id].validation_macro_f1
            state = PhaseThreeSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_one_best_feature_path=state.phase_one_best_feature_path,
                phase_two_best_optim_path=state.phase_two_best_optim_path,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score if best_score >= 0.0 else None,
                top_three_trial_ids=top_three_trial_ids,
                family_winner_ids=family_winner_ids,
                created_at=state.created_at,
                updated_at=utc_now(),
            )
            self._save_state(state)

        return {
            "phase": "phase-3",
            "feature_source": feature_summary,
            "optimizer_source": optimizer_summary,
            "total_trials": len(trials),
            "completed_trials": len(state.completed_trials),
            "best_validation_macro_f1": state.best_validation_macro_f1,
            "best_trial": state.completed_trials[best_trial_id].to_dict()
            if best_trial_id
            else None,
            "top_three": [
                state.completed_trials[trial_id].to_dict() for trial_id in state.top_three_trial_ids
            ],
            "family_winners": {
                family: state.completed_trials[trial_id].to_dict()
                for family, trial_id in state.family_winner_ids.items()
            },
            "state_path": str(self.state_path),
            "best_backbones_path": str(self.best_backbones_path),
            "family_winners_path": str(self.family_winners_path),
        }
