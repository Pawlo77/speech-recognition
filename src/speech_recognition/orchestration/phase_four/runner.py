"""Phase-4 held-out evaluation orchestration."""

import json
import logging
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from importlib import import_module
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ...config import ExperimentConfig
from ..phase_one import _atomic_write_json, _read_json
from ..phase_three import PhaseThreeTrialRecord
from ..services import build_isolated_subprocess_env
from ..sweep_utils import (
    apply_trial_cap,
    run_subprocess_with_live_output,
    summary_from_completed_process,
    sweep_seeds,
    utc_now,
)
from ..tracking import _resolve_tracking_uri
from .constants import (
    PHASE_FOUR_METHODS,
    PHASE_FOUR_STATE_SCHEMA_VERSION,
    PHASE_FOUR_STRICT_DROP_LIMIT,
)
from .state import PhaseFourSweepState
from .trials import (
    PhaseFourTrialRecord,
    PhaseFourTrialSpec,
    build_phase_four_trials,
)
from .utils import (
    _phase_four_optimizer_config,
    _phase_four_prediction_artifact_recursive,
    _phase_four_score_recursive,
)


def _phase_four_package() -> Any:
    """Return the imported phase-four package
    for accessing trial builders and subprocess runners."""
    return import_module("speech_recognition.orchestration.phase_four")


def _build_phase_four_trials(
    backbone_ids: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[Any, ...]:
    """Return the trial specifications for phase 4 given backbone IDs and seeds."""
    package = _phase_four_package()
    builder = getattr(package, "build_phase_four_trials", build_phase_four_trials)
    return builder(backbone_ids, seeds)


def _run_subprocess(command: list[str], env: Mapping[str, str], check: bool) -> Any:
    """Run a subprocess command with live output and return the completed process."""
    package = _phase_four_package()
    runner = getattr(package, "run_subprocess_with_live_output", run_subprocess_with_live_output)
    return runner(command, env=env, check=check)


def build_phase_four_command(config_path: Path, run_name: str) -> list[str]:
    """Build the isolated subprocess command for one phase-4 validation trial."""
    return [
        sys.executable,
        "-m",
        "speech_recognition.cli",
        "run-single-train",
        "--config",
        str(config_path),
        "--output-dir",
        str(config_path.parent.parent.parent.parent),
        "--run-name",
        run_name,
    ]


def build_phase_four_test_command(config_path: Path, run_name: str) -> list[str]:
    """Build the isolated subprocess command for one phase-4 held-out test run."""
    return [
        sys.executable,
        "-m",
        "speech_recognition.cli",
        "run-single-eval",
        "--config",
        str(config_path),
        "--output-dir",
        str(config_path.parent.parent.parent.parent),
        "--run-name",
        run_name,
    ]


def _summary_from_completed_process(
    child_state_path: Path, completed_process: subprocess.CompletedProcess[str]
) -> dict[str, Any]:
    """Extract the summary payload from a completed subprocess,
    preferring the child state file over stdout."""
    return summary_from_completed_process(
        child_state_path,
        completed_process,
        read_json=_read_json,
    )


def _read_artifact(path: Path, description: str) -> dict[str, Any]:
    """Read a JSON artifact from the given path with error handling and validation."""
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at '{path}'.")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return payload


def _mlflow_client_and_run(
    config: ExperimentConfig,
    run_name: str,
) -> tuple[Any, str] | None:
    """Return MLflow client and latest run id for a pipeline run name."""
    if not config.mlflow.enabled:
        return None

    try:
        mlflow = import_module("mlflow")
    except Exception:
        return None

    mlflow.set_tracking_uri(_resolve_tracking_uri(config.mlflow.tracking_uri))
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        return None

    safe_run_name = run_name.replace("'", "\\'")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string=f"tags.pipeline.run_name = '{safe_run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        return None
    return client, runs[0].info.run_id


def _load_prediction_payload_from_mlflow(
    config: ExperimentConfig,
    run_name: str,
    artifact_path: str,
) -> dict[str, Any] | None:
    """Load a JSON prediction artifact payload from MLflow when available."""
    run_context = _mlflow_client_and_run(config, run_name)
    if run_context is None:
        return None
    client, run_id = run_context

    with tempfile.TemporaryDirectory() as temporary_dir:
        try:
            local_path = client.download_artifacts(run_id, artifact_path, temporary_dir)
        except Exception:
            return None

        try:
            payload = json.loads(Path(local_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if isinstance(payload, dict):
            return payload
    return None


class PhaseFourSweepRunner:
    """Run the phase-4 held-out evaluation sweep using child processes."""

    def __init__(self, output_dir: Path, base_config: ExperimentConfig | None = None) -> None:
        self.output_dir = output_dir
        self.phase_dir = self.output_dir / "phase_4"
        self.state_path = self.phase_dir / "state.json"
        self.best_eval_path = self.phase_dir / "best_eval.json"
        self.method_winners_path = self.phase_dir / "method_winners.json"
        self.selected_test_eval_path = self.phase_dir / "selected_test_eval.json"
        self.base_config = base_config or ExperimentConfig()
        self.phase_three_best_backbones_path = self.output_dir / "phase_3" / "best_backbones.json"
        self.phase_two_best_optim_path = self.output_dir / "phase_2" / "best_optim.json"

    def _load_phase_three_backbones(self) -> dict[str, PhaseThreeTrialRecord]:
        artifact = _read_artifact(
            self.phase_three_best_backbones_path, "Phase-3 best backbones artifact"
        )
        top_three_trials = artifact.get("top_three_trials")
        if not isinstance(top_three_trials, list) or not top_three_trials:
            raise ValueError("Phase-3 best backbones artifact is missing top_three_trials.")

        backbone_trials = {
            trial_data["trial_id"]: PhaseThreeTrialRecord.from_dict(trial_data)
            for trial_data in top_three_trials
        }
        if len(backbone_trials) != len(top_three_trials):
            raise ValueError("Phase-3 best backbones artifact contains duplicate trial ids.")
        return backbone_trials

    def load_state(self) -> PhaseFourSweepState:
        """Load the persisted sweep state or create a new one."""
        backbone_trials = self._load_phase_three_backbones()
        if not self.state_path.exists():
            return PhaseFourSweepState.fresh(
                self.output_dir,
                self.phase_three_best_backbones_path,
                tuple(backbone_trials),
            )
        return PhaseFourSweepState.from_dict(_read_json(self.state_path))

    def _save_state(self, state: PhaseFourSweepState) -> None:
        """Persist the sweep state and selection artifacts."""
        _atomic_write_json(self.state_path, state.to_dict())
        if state.best_trial_id is not None:
            best_trial = state.completed_trials[state.best_trial_id]
            _atomic_write_json(
                self.best_eval_path,
                {
                    "schema_version": PHASE_FOUR_STATE_SCHEMA_VERSION,
                    "best_trial": best_trial.to_dict(),
                },
            )
        if state.method_winner_ids:
            _atomic_write_json(
                self.method_winners_path,
                {
                    "schema_version": PHASE_FOUR_STATE_SCHEMA_VERSION,
                    "method_winners": {
                        method: state.completed_trials[trial_id].to_dict()
                        for method, trial_id in state.method_winner_ids.items()
                    },
                },
            )

    def _trial_run_name(self, trial: PhaseFourTrialSpec) -> str:
        """Return the MLflow run name for a given trial specification."""
        return trial.trial_id

    def _trial_output_paths(self, trial: PhaseFourTrialSpec) -> tuple[Path, Path, Path]:
        """Return the trial directory, config path, and child
        state path for a given trial specification."""
        trial_dir = self.phase_dir / "runs" / trial.trial_id
        config_path = trial_dir / "temp_config.json"
        child_state_path = self.output_dir / "phase_4" / "runs" / trial.trial_id / "state.json"
        return trial_dir, config_path, child_state_path

    def _build_trial_config(
        self,
        trial: PhaseFourTrialSpec,
        backbone_trials: Mapping[str, PhaseThreeTrialRecord],
        optimizer_payload: Mapping[str, Any],
    ) -> ExperimentConfig:
        """Return the concrete config for a given trial specification."""
        return trial.to_config(self.base_config, backbone_trials, optimizer_payload)

    def _run_trial(
        self,
        trial: PhaseFourTrialSpec,
        backbone_trials: Mapping[str, PhaseThreeTrialRecord],
        optimizer_payload: Mapping[str, Any],
    ) -> PhaseFourTrialRecord:
        """Run one trial specification as a child subprocess
        and return the completed trial record."""
        trial_dir, config_path, child_state_path = self._trial_output_paths(trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_trial_config(trial, backbone_trials, optimizer_payload)
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        command = build_phase_four_command(config_path, self._trial_run_name(trial))
        completed_process = _run_subprocess(
            command,
            env=build_isolated_subprocess_env(),
            check=True,
        )

        summary = _summary_from_completed_process(child_state_path, completed_process)
        metrics = _phase_four_score_recursive(summary)
        prediction_artifact = _phase_four_prediction_artifact_recursive(summary)
        baseline_trial_id = trial.backbone_ids[0]
        baseline_record = backbone_trials[baseline_trial_id]
        baseline_score = (
            float(baseline_record.core_command_macro_f1)
            if baseline_record.core_command_macro_f1 is not None
            else float(baseline_record.validation_macro_f1)
        )
        accepted = metrics["core_command_macro_f1"] >= baseline_score - PHASE_FOUR_STRICT_DROP_LIMIT
        return PhaseFourTrialRecord(
            trial_id=trial.trial_id,
            method=trial.method,
            backbone_ids=trial.backbone_ids,
            seed=trial.seed,
            baseline_backbone_id=baseline_trial_id,
            baseline_core_command_macro_f1=baseline_score,
            core_command_macro_f1=metrics["core_command_macro_f1"],
            unknown_f1=metrics["unknown_f1"],
            silence_f1=metrics["silence_f1"],
            macro_f1_nc=metrics["macro_f1_nc"],
            inference_latency_ms_mean=metrics["inference_latency_ms_mean"],
            unknown_to_command_leakage=metrics["unknown_to_command_leakage"],
            silence_false_trigger_rate=metrics["silence_false_trigger_rate"],
            accepted=accepted,
            prediction_artifact=prediction_artifact,
            config_path=str(config_path),
            child_state_path=str(child_state_path),
            completed_at=utc_now(),
        )

    def _ensemble_results(
        self,
        records: Mapping[str, PhaseFourTrialRecord],
        phase_three_trial_ids: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Compute within-method ensemble metrics from stored prediction artifacts."""
        grouped: dict[tuple[str, int], list[PhaseFourTrialRecord]] = {}
        for record in records.values():
            if not record.prediction_artifact:
                continue
            grouped.setdefault((record.method, record.seed), []).append(record)

        results: list[dict[str, Any]] = []
        for (method, seed), method_records in grouped.items():
            artifact_by_backbone = {
                record.baseline_backbone_id: record.prediction_artifact
                for record in method_records
                if record.prediction_artifact
            }
            run_name_by_backbone = {
                record.baseline_backbone_id: record.trial_id for record in method_records
            }
            ordered_ids = [
                backbone_id
                for backbone_id in phase_three_trial_ids
                if backbone_id in artifact_by_backbone
            ]
            payload_by_backbone = self._load_prediction_payloads(
                artifact_by_backbone,
                run_name_by_backbone,
                ordered_ids,
            )
            results.extend(
                self._build_ensemble_rows(
                    method,
                    seed,
                    ordered_ids,
                    payload_by_backbone,
                )
            )
        return results

    def _load_prediction_payloads(
        self,
        artifact_by_backbone: Mapping[str, str | None],
        run_name_by_backbone: Mapping[str, str],
        ordered_ids: list[str],
    ) -> dict[str, dict[str, Any]]:
        """Load prediction payloads for available backbone artifacts."""
        payload_by_backbone: dict[str, dict[str, Any]] = {}
        for backbone_id in ordered_ids:
            prediction_path = artifact_by_backbone.get(backbone_id)
            if not prediction_path:
                continue

            prediction_file = Path(prediction_path)
            payload: dict[str, Any] | None = None
            if prediction_file.exists():
                try:
                    candidate = json.loads(prediction_file.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    candidate = None
                if isinstance(candidate, dict):
                    payload = candidate

            if payload is None:
                run_name = run_name_by_backbone.get(backbone_id)
                if run_name is not None:
                    payload = _load_prediction_payload_from_mlflow(
                        self.base_config,
                        run_name,
                        prediction_path,
                    )

            if payload is not None:
                payload_by_backbone[backbone_id] = payload
        return payload_by_backbone

    def _f1(self, tp: int, fp: int, fn: int) -> float:
        """Compute the F1 score given true positives, false positives, and false negatives."""
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def _build_ensemble_rows(
        self,
        method: str,
        seed: int,
        ordered_ids: list[str],
        payload_by_backbone: Mapping[str, Mapping[str, Any]],
        eval_split: str | None = None,
    ) -> list[dict[str, Any]]:
        """Compute ensemble rows for all backbone subsets for one method/seed pair."""
        rows: list[dict[str, Any]] = []
        for subset_size in range(1, len(ordered_ids) + 1):
            for subset in combinations(ordered_ids, subset_size):
                if any(backbone_id not in payload_by_backbone for backbone_id in subset):
                    continue

                payloads = [payload_by_backbone[backbone_id] for backbone_id in subset]
                targets = payloads[0].get("targets")
                labels = payloads[0].get("labels", [])
                if not isinstance(targets, list) or not isinstance(labels, list):
                    continue

                unknown_idx = labels.index("__unknown__") if "__unknown__" in labels else None
                silence_idx = labels.index("__silence__") if "__silence__" in labels else None
                if unknown_idx is None or silence_idx is None:
                    continue

                probabilities = [np.array(payload["probs"], dtype=float) for payload in payloads]
                averaged = sum(probabilities) / float(len(probabilities))
                predictions = averaged.argmax(axis=1).tolist()

                unknown_tp = sum(
                    1
                    for target, pred in zip(targets, predictions, strict=True)
                    if target == unknown_idx and pred == unknown_idx
                )
                unknown_fp = sum(
                    1
                    for target, pred in zip(targets, predictions, strict=True)
                    if target != unknown_idx and pred == unknown_idx
                )
                unknown_fn = sum(
                    1
                    for target, pred in zip(targets, predictions, strict=True)
                    if target == unknown_idx and pred != unknown_idx
                )
                silence_tp = sum(
                    1
                    for target, pred in zip(targets, predictions, strict=True)
                    if target == silence_idx and pred == silence_idx
                )
                silence_fp = sum(
                    1
                    for target, pred in zip(targets, predictions, strict=True)
                    if target != silence_idx and pred == silence_idx
                )
                silence_fn = sum(
                    1
                    for target, pred in zip(targets, predictions, strict=True)
                    if target == silence_idx and pred != silence_idx
                )

                unknown_f1 = self._f1(unknown_tp, unknown_fp, unknown_fn)
                silence_f1 = self._f1(silence_tp, silence_fp, silence_fn)
                row = {
                    "method": method,
                    "seed": seed,
                    "subset": list(subset),
                    "subset_size": len(subset),
                    "macro_f1_nc": (unknown_f1 + silence_f1) / 2.0,
                    "unknown_f1": unknown_f1,
                    "silence_f1": silence_f1,
                }
                if eval_split:
                    row["eval_split"] = eval_split
                rows.append(row)
        return rows

    def _ensemble_results_test(
        self,
        test_prediction_artifacts: dict[str, str],
        records: Mapping[str, PhaseFourTrialRecord],
        phase_three_trial_ids: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Compute within-method ensemble metrics from test prediction artifacts."""
        grouped: dict[tuple[str, int], list[tuple[str, PhaseFourTrialRecord]]] = {}
        for trial_id, prediction_path in test_prediction_artifacts.items():
            record = records.get(trial_id)
            if record is None or not prediction_path:
                continue
            key = (record.method, record.seed)
            grouped.setdefault(key, []).append((trial_id, record))

        results: list[dict[str, Any]] = []
        for (method, seed), trial_records in grouped.items():
            by_backbone = {
                record.baseline_backbone_id: trial_id for trial_id, record in trial_records
            }
            ordered_ids = [
                backbone_id for backbone_id in phase_three_trial_ids if backbone_id in by_backbone
            ]

            artifact_by_backbone = {
                backbone_id: test_prediction_artifacts[phase_four_trial_id]
                for backbone_id, phase_four_trial_id in by_backbone.items()
                if phase_four_trial_id in test_prediction_artifacts
            }
            payload_by_backbone = self._load_prediction_payloads(
                artifact_by_backbone,
                by_backbone,
                ordered_ids,
            )
            results.extend(
                self._build_ensemble_rows(
                    method,
                    seed,
                    ordered_ids,
                    payload_by_backbone,
                    eval_split="test",
                )
            )
        return results

    def _select_winners(
        self,
        records: Mapping[str, PhaseFourTrialRecord],
    ) -> tuple[str | None, dict[str, str], float | None]:
        """Select phase-4 winners by seed-aggregated metrics per method/backbone config."""
        grouped: dict[tuple[str, tuple[str, ...]], list[PhaseFourTrialRecord]] = {}
        for record in records.values():
            grouped.setdefault((record.method, record.backbone_ids), []).append(record)

        if not grouped:
            return None, {}, None

        aggregate_rows: list[dict[str, Any]] = []
        for (method, backbone_ids), grouped_records in grouped.items():
            seed_count = len(grouped_records)
            accepted = all(item.accepted for item in grouped_records)
            mean_macro_f1_nc = float(
                sum(item.macro_f1_nc for item in grouped_records) / max(1, seed_count)
            )
            mean_silence_false_trigger_rate = float(
                sum(item.silence_false_trigger_rate for item in grouped_records)
                / max(1, seed_count)
            )
            mean_unknown_to_command_leakage = float(
                sum(item.unknown_to_command_leakage for item in grouped_records)
                / max(1, seed_count)
            )
            representative = max(grouped_records, key=lambda item: item.macro_f1_nc)
            aggregate_rows.append(
                {
                    "method": method,
                    "backbone_ids": backbone_ids,
                    "seed_count": seed_count,
                    "accepted": accepted,
                    "mean_macro_f1_nc": mean_macro_f1_nc,
                    "mean_silence_false_trigger_rate": mean_silence_false_trigger_rate,
                    "mean_unknown_to_command_leakage": mean_unknown_to_command_leakage,
                    "representative_trial_id": representative.trial_id,
                }
            )

        accepted_rows = [row for row in aggregate_rows if row["accepted"]]
        if not accepted_rows:
            return None, {}, None

        def _winner_key(row: Mapping[str, Any]) -> tuple[float, float, float]:
            return (
                float(row["mean_macro_f1_nc"]),
                -float(row["mean_silence_false_trigger_rate"]),
                -float(row["mean_unknown_to_command_leakage"]),
            )

        best_row = max(accepted_rows, key=_winner_key)
        best_trial_id = str(best_row["representative_trial_id"])
        best_macro_f1_nc = float(best_row["mean_macro_f1_nc"])

        method_winner_ids: dict[str, str] = {}
        for method in PHASE_FOUR_METHODS:
            method_rows = [row for row in accepted_rows if row["method"] == method]
            if method_rows:
                winner_row = max(method_rows, key=_winner_key)
                method_winner_ids[method] = str(winner_row["representative_trial_id"])
        return best_trial_id, method_winner_ids, best_macro_f1_nc

    def _evaluate_completed_on_test(
        self, state: PhaseFourSweepState
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        """Evaluate all completed phase-4 systems once on the held-out test split.

        Returns:
            Tuple of (test_results, test_prediction_artifacts) where test_prediction_artifacts
            maps trial_id to the path of test predictions for ensemble computation.
        """
        completed_ids = sorted(state.completed_trials.keys())
        if not completed_ids:
            return [], {}

        if self.selected_test_eval_path.exists():
            try:
                cached_payload = _read_json(self.selected_test_eval_path)
            except (json.JSONDecodeError, OSError):
                cached_payload = {}

            if isinstance(cached_payload, dict):
                cached_ids_raw = cached_payload.get("completed_trial_ids")
                cached_results = cached_payload.get("results")
                cached_artifacts_raw = cached_payload.get("test_prediction_artifacts")
                cached_ids = (
                    sorted(str(item) for item in cached_ids_raw)
                    if isinstance(cached_ids_raw, list)
                    else None
                )
                if cached_ids == completed_ids and isinstance(cached_results, list):
                    cached_artifacts: dict[str, str] = {}
                    if isinstance(cached_artifacts_raw, Mapping):
                        for key, value in cached_artifacts_raw.items():
                            if isinstance(key, str) and isinstance(value, str) and value:
                                cached_artifacts[key] = value
                    if not cached_artifacts:
                        for row in cached_results:
                            if not isinstance(row, Mapping):
                                continue
                            trial_id = row.get("trial_id")
                            artifact_path = row.get("test_prediction_artifact")
                            if isinstance(trial_id, str) and isinstance(artifact_path, str):
                                cached_artifacts[trial_id] = artifact_path
                    logging.getLogger(__name__).info(
                        "[phase-4] reusing cached heldout test evaluation for %d completed trials",
                        len(completed_ids),
                    )
                    cached_rows = [dict(row) for row in cached_results if isinstance(row, Mapping)]
                    return cached_rows, cached_artifacts

        results: list[dict[str, Any]] = []
        test_prediction_artifacts: dict[str, str] = {}

        for trial_id in tqdm(
            completed_ids,
            total=len(completed_ids),
            desc="phase-4 heldout eval",
            leave=False,
        ):
            record = state.completed_trials.get(trial_id)
            if record is None:
                continue

            config_path = Path(record.config_path)
            if not config_path.exists():
                continue

            run_name = record.trial_id
            command = build_phase_four_test_command(config_path, run_name)
            completed_process = _run_subprocess(
                command,
                env=build_isolated_subprocess_env(),
                check=True,
            )

            child_state_path = self.output_dir / "phase_4" / "runs" / run_name / "state.json"
            summary = _summary_from_completed_process(child_state_path, completed_process)
            metrics = _phase_four_score_recursive(summary)

            # Extract test prediction artifact path from the test eval output
            test_prediction_artifact = _phase_four_prediction_artifact_recursive(summary)
            if test_prediction_artifact:
                test_prediction_artifacts[trial_id] = test_prediction_artifact

            result_dict = {
                "trial_id": record.trial_id,
                "method": record.method,
                "seed": record.seed,
                "baseline_backbone_id": record.baseline_backbone_id,
                "core_command_macro_f1": metrics["core_command_macro_f1"],
                "unknown_f1": metrics["unknown_f1"],
                "silence_f1": metrics["silence_f1"],
                "macro_f1_nc": metrics["macro_f1_nc"],
                "inference_latency_ms_mean": metrics["inference_latency_ms_mean"],
                "unknown_to_command_leakage": metrics["unknown_to_command_leakage"],
                "silence_false_trigger_rate": metrics["silence_false_trigger_rate"],
                "per_class": metrics.get("per_class", {}),
                "heldout_run_name": run_name,
                "heldout_state_path": str(child_state_path),
                "test_prediction_artifact": test_prediction_artifact,
            }
            results.append(result_dict)

        _atomic_write_json(
            self.selected_test_eval_path,
            {
                "schema_version": PHASE_FOUR_STATE_SCHEMA_VERSION,
                "completed_trial_ids": completed_ids,
                "test_prediction_artifacts": test_prediction_artifacts,
                "results": results,
            },
        )
        return results, test_prediction_artifacts

    def execute(self) -> dict[str, Any]:
        """Run the full phase-4 sweep, skipping completed trials."""
        backbone_trials = self._load_phase_three_backbones()
        optim_artifact = _read_artifact(
            self.phase_two_best_optim_path,
            "Phase-2 best optimization artifact",
        )
        optimizer_payload = _phase_four_optimizer_config(optim_artifact)
        seeds = sweep_seeds(self.base_config.seeds)
        trials = apply_trial_cap(_build_phase_four_trials(tuple(backbone_trials), seeds))
        total_trials = len(trials)
        logger = logging.getLogger(__name__)
        state = self.load_state()

        for trial_index, trial in enumerate(
            tqdm(trials, total=total_trials, desc="phase-4 trials", leave=False), start=1
        ):
            if trial.trial_id in state.completed_trials:
                logger.info(
                    "[phase-4] [%d/%d] skipping completed %s",
                    trial_index,
                    total_trials,
                    trial.trial_id,
                )
                continue

            logger.info(
                "[phase-4] [%d/%d] running %s",
                trial_index,
                total_trials,
                trial.trial_id,
            )

            trial_record = self._run_trial(trial, backbone_trials, optimizer_payload)
            logger.info(
                "[phase-4] finished %s macro_f1_nc=%.6f accepted=%s",
                trial.trial_id,
                trial_record.macro_f1_nc,
                trial_record.accepted,
            )
            completed_trials = {**state.completed_trials, trial.trial_id: trial_record}
            best_trial_id, method_winner_ids, best_score = self._select_winners(completed_trials)
            state = PhaseFourSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_three_best_backbones_path=state.phase_three_best_backbones_path,
                phase_three_trial_ids=state.phase_three_trial_ids,
                completed_trials=completed_trials,
                best_trial_id=best_trial_id,
                best_macro_f1_nc=best_score,
                method_winner_ids=method_winner_ids,
                created_at=state.created_at,
                updated_at=trial_record.completed_at,
            )
            self._save_state(state)

        if state.completed_trials:
            best_trial_id, method_winner_ids, best_score = self._select_winners(
                state.completed_trials
            )
            state = PhaseFourSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_three_best_backbones_path=state.phase_three_best_backbones_path,
                phase_three_trial_ids=state.phase_three_trial_ids,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_macro_f1_nc=best_score,
                method_winner_ids=method_winner_ids,
                created_at=state.created_at,
                updated_at=utc_now(),
            )
            self._save_state(state)

        selected_test_results, test_prediction_artifacts = self._evaluate_completed_on_test(state)

        ensemble_test_results = self._ensemble_results_test(
            test_prediction_artifacts,
            state.completed_trials,
            state.phase_three_trial_ids,
        )

        return {
            "phase": "phase-4",
            "frozen_backbones": [
                backbone_trials[trial_id].to_dict() for trial_id in state.phase_three_trial_ids
            ],
            "total_trials": len(trials),
            "completed_trials": len(state.completed_trials),
            "best_macro_f1_nc": state.best_macro_f1_nc,
            "best_trial": state.completed_trials[best_trial_id].to_dict()
            if best_trial_id
            else None,
            "method_winners": {
                method: state.completed_trials[trial_id].to_dict()
                for method, trial_id in state.method_winner_ids.items()
            },
            "ensemble_results": self._ensemble_results(
                state.completed_trials,
                state.phase_three_trial_ids,
            ),
            "ensemble_test_results": ensemble_test_results,
            "selected_test_results": selected_test_results,
            "state_path": str(self.state_path),
            "best_eval_path": str(self.best_eval_path),
            "method_winners_path": str(self.method_winners_path),
            "selected_test_eval_path": str(self.selected_test_eval_path),
        }
