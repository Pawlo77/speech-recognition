"""Phase-4 held-out evaluation orchestration."""

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from ..config import ExperimentConfig, ModelConfig
from .phase_one import _atomic_write_json, _read_json, _serialize
from .phase_three import PhaseThreeTrialRecord
from .services import build_isolated_subprocess_env

PHASE_FOUR_STATE_SCHEMA_VERSION: int = 1
"""Schema version for the phase-4 sweep state file."""

PHASE_FOUR_METHODS: tuple[str, ...] = (
    "flat_multiclass",
    "sampling_control",
    "loss_reweighting",
    "two_stage_detector",
    "shared_two_head",
)
"""Supported final non-command handling strategies."""

PHASE_FOUR_STRICT_DROP_LIMIT: float = 0.01
"""Maximum tolerated core-command macro-F1 drop relative to the Phase 3 baseline."""

PHASE_FOUR_WARMUP_ITERATIONS: int = 50
"""Warmup iterations excluded from latency measurement."""


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(UTC).isoformat()


def _phase_four_score(payload: Mapping[str, Any]) -> dict[str, float]:
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

    return {
        "core_command_macro_f1": float(core_command_macro_f1),
        "unknown_f1": float(unknown_f1),
        "silence_f1": float(silence_f1),
        "macro_f1_nc": float(macro_f1_nc),
        "inference_latency_ms_mean": float(inference_latency_ms_mean),
        "unknown_to_command_leakage": float(unknown_to_command_leakage),
        "silence_false_trigger_rate": float(silence_false_trigger_rate),
    }


def _phase_four_score_recursive(payload: Mapping[str, Any]) -> dict[str, float]:
    """Extract phase-4 metrics from nested child payload structures."""

    metrics = _phase_four_score(payload)
    if metrics["core_command_macro_f1"] > 0.0 or metrics["macro_f1_nc"] > 0.0:
        return metrics

    phase_artifacts = payload.get("phase_artifacts")
    if isinstance(phase_artifacts, Mapping):
        for artifact in phase_artifacts.values():
            if isinstance(artifact, Mapping):
                output_data = artifact.get("output_data")
                if isinstance(output_data, Mapping):
                    nested_metrics = _phase_four_score_recursive(output_data)
                    if (
                        nested_metrics["core_command_macro_f1"] > 0.0
                        or nested_metrics["macro_f1_nc"] > 0.0
                    ):
                        return nested_metrics
    return metrics


def _phase_four_prediction_artifact_recursive(payload: Mapping[str, Any]) -> str | None:
    """Extract prediction artifact path from nested child payload structures."""

    direct_path = payload.get("prediction_artifact")
    if isinstance(direct_path, str) and direct_path:
        return direct_path

    phase_artifacts = payload.get("phase_artifacts")
    if isinstance(phase_artifacts, Mapping):
        for artifact in phase_artifacts.values():
            if isinstance(artifact, Mapping):
                output_data = artifact.get("output_data")
                if isinstance(output_data, Mapping):
                    nested = _phase_four_prediction_artifact_recursive(output_data)
                    if nested:
                        return nested
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
        )
    if family == "xlstm":
        return ModelConfig(
            family="xlstm",
            pretrained=False,
            xlstm_dim=int(params.get("dimension", 32)),
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
    """Load child summary from state file first, then subprocess JSON stdout."""

    if child_state_path.exists():
        try:
            payload = _read_json(child_state_path)
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, OSError):
            pass
    try:
        payload = json.loads(completed_process.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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
        dataset = replace(base_config.dataset, train_split="train_full", valid_split="valid_full")
        phase_config = replace(base_config.phase, phase="phase_4")
        return replace(
            base_config,
            dataset=dataset,
            model=_build_backbone_model_config(representative_backbone),
            evaluation=evaluation,
            phase=phase_config,
            seed=self.seed,
        )


def _build_method_trials(
    backbone_ids: tuple[str, ...], method: str, seeds: tuple[int, ...]
) -> tuple[PhaseFourTrialSpec, ...]:
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


@dataclass(frozen=True, slots=True)
class PhaseFourSweepState:
    """Persistent state for the phase-4 sweep."""

    schema_version: int = PHASE_FOUR_STATE_SCHEMA_VERSION
    """State file schema version."""
    output_dir: str = ""
    """Root output directory for the sweep."""
    phase_three_best_backbones_path: str = ""
    """Path to phase-3 best backbones summary file."""
    phase_three_trial_ids: tuple[str, ...] = ()
    """Phase-3 trial IDs available for evaluation."""
    completed_trials: dict[str, PhaseFourTrialRecord] = field(default_factory=dict)
    """Mapping of trial ID to completed trial records."""
    best_trial_id: str | None = None
    """Trial ID of the best-performing trial."""
    best_macro_f1_nc: float | None = None
    """Best macro-F1 score for non-command classes."""
    method_winner_ids: dict[str, str] = field(default_factory=dict)
    """Best trial ID per evaluation method."""
    created_at: str = field(default_factory=_utc_now)
    """ISO-8601 timestamp when sweep state was created."""
    updated_at: str = field(default_factory=_utc_now)
    """ISO-8601 timestamp when sweep state was last updated."""

    def __post_init__(self) -> None:
        if self.schema_version != PHASE_FOUR_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported phase-4 state schema version.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the sweep state."""

        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseFourSweepState":
        """Build phase-4 state from JSON."""

        payload = dict(data)
        payload["phase_three_trial_ids"] = tuple(payload.get("phase_three_trial_ids", ()))
        payload["completed_trials"] = {
            key: PhaseFourTrialRecord.from_dict(value)
            for key, value in payload.get("completed_trials", {}).items()
        }
        if payload.get("best_macro_f1_nc") is not None:
            payload["best_macro_f1_nc"] = float(payload["best_macro_f1_nc"])
        return cls(**payload)

    @classmethod
    def fresh(
        cls,
        output_dir: Path,
        phase_three_best_backbones_path: Path,
        phase_three_trial_ids: tuple[str, ...],
    ) -> "PhaseFourSweepState":
        """Create a new empty state for an output directory."""

        return cls(
            output_dir=str(output_dir),
            phase_three_best_backbones_path=str(phase_three_best_backbones_path),
            phase_three_trial_ids=phase_three_trial_ids,
        )


def _read_artifact(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at '{path}'.")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return payload


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
        return trial.trial_id

    def _trial_output_paths(self, trial: PhaseFourTrialSpec) -> tuple[Path, Path, Path]:
        trial_dir = self.phase_dir / "runs" / trial.trial_id
        config_path = trial_dir / "temp_config.json"
        child_state_path = self.output_dir / "phase_4" / "runs" / trial.trial_id / "state.json"
        return trial_dir, config_path, child_state_path

    def _build_trial_config(
        self,
        trial: PhaseFourTrialSpec,
        backbone_trials: Mapping[str, PhaseThreeTrialRecord],
    ) -> ExperimentConfig:
        return trial.to_config(self.base_config, backbone_trials)

    def _run_trial(
        self,
        trial: PhaseFourTrialSpec,
        backbone_trials: Mapping[str, PhaseThreeTrialRecord],
    ) -> PhaseFourTrialRecord:
        trial_dir, config_path, child_state_path = self._trial_output_paths(trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_trial_config(trial, backbone_trials)
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        command = build_phase_four_command(config_path, self._trial_run_name(trial))
        completed_process = subprocess.run(  # noqa: S603
            command,
            check=True,
            env=build_isolated_subprocess_env(),
            text=True,
            capture_output=True,
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
            completed_at=_utc_now(),
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
            by_backbone = {record.baseline_backbone_id: record for record in method_records}
            ordered_ids = [
                trial_id for trial_id in phase_three_trial_ids if trial_id in by_backbone
            ]
            for subset_size in range(1, len(ordered_ids) + 1):
                for subset in combinations(ordered_ids, subset_size):
                    payloads = []
                    for trial_id in subset:
                        prediction_path = by_backbone[trial_id].prediction_artifact
                        if not prediction_path:
                            payloads = []
                            break
                        prediction_file = Path(prediction_path)
                        if not prediction_file.exists():
                            payloads = []
                            break
                        payloads.append(json.loads(prediction_file.read_text(encoding="utf-8")))
                    if not payloads:
                        continue
                    targets = payloads[0]["targets"]
                    probabilities = [
                        np.array(payload["probs"], dtype=float) for payload in payloads
                    ]
                    averaged = sum(probabilities) / float(len(probabilities))
                    predictions = averaged.argmax(axis=1).tolist()
                    unknown_idx = 30
                    silence_idx = 31
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
                    unknown_precision = (
                        unknown_tp / (unknown_tp + unknown_fp) if (unknown_tp + unknown_fp) else 0.0
                    )
                    unknown_recall = (
                        unknown_tp / (unknown_tp + unknown_fn) if (unknown_tp + unknown_fn) else 0.0
                    )
                    silence_precision = (
                        silence_tp / (silence_tp + silence_fp) if (silence_tp + silence_fp) else 0.0
                    )
                    silence_recall = (
                        silence_tp / (silence_tp + silence_fn) if (silence_tp + silence_fn) else 0.0
                    )
                    unknown_f1 = (
                        2.0
                        * unknown_precision
                        * unknown_recall
                        / (unknown_precision + unknown_recall)
                        if (unknown_precision + unknown_recall)
                        else 0.0
                    )
                    silence_f1 = (
                        2.0
                        * silence_precision
                        * silence_recall
                        / (silence_precision + silence_recall)
                        if (silence_precision + silence_recall)
                        else 0.0
                    )
                    results.append(
                        {
                            "method": method,
                            "seed": seed,
                            "subset": list(subset),
                            "subset_size": len(subset),
                            "macro_f1_nc": (unknown_f1 + silence_f1) / 2.0,
                            "unknown_f1": unknown_f1,
                            "silence_f1": silence_f1,
                        }
                    )
        return results

    def _select_winners(
        self, records: Mapping[str, PhaseFourTrialRecord]
    ) -> tuple[str | None, dict[str, str]]:
        accepted_records = [record for record in records.values() if record.accepted]
        if not accepted_records:
            return None, {}

        def _winner_key(record: PhaseFourTrialRecord) -> tuple[float, float, float, float]:
            return (
                float(record.macro_f1_nc),
                -float(record.silence_false_trigger_rate),
                -float(record.unknown_to_command_leakage),
                float(record.core_command_macro_f1),
            )

        best_trial_id = max(accepted_records, key=_winner_key).trial_id
        method_winner_ids: dict[str, str] = {}
        for method in PHASE_FOUR_METHODS:
            method_records = [record for record in accepted_records if record.method == method]
            if method_records:
                method_winner_ids[method] = max(method_records, key=_winner_key).trial_id
        return best_trial_id, method_winner_ids

    def _evaluate_selected_on_test(self, state: PhaseFourSweepState) -> list[dict[str, Any]]:
        """Evaluate selected phase-4 systems once on the held-out test split."""

        selected_ids: set[str] = set(state.method_winner_ids.values())
        if state.best_trial_id is not None:
            selected_ids.add(state.best_trial_id)
        if not selected_ids:
            return []

        results: list[dict[str, Any]] = []
        for trial_id in sorted(selected_ids):
            record = state.completed_trials.get(trial_id)
            if record is None:
                continue

            config_path = Path(record.config_path)
            if not config_path.exists():
                continue

            run_name = f"{record.trial_id}_heldout_test"
            command = build_phase_four_test_command(config_path, run_name)
            completed_process = subprocess.run(  # noqa: S603
                command,
                check=True,
                env=build_isolated_subprocess_env(),
                text=True,
                capture_output=True,
            )

            child_state_path = self.output_dir / "phase_4" / "runs" / run_name / "state.json"
            summary = _summary_from_completed_process(child_state_path, completed_process)
            metrics = _phase_four_score_recursive(summary)
            results.append(
                {
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
                    "heldout_run_name": run_name,
                    "heldout_state_path": str(child_state_path),
                }
            )

        _atomic_write_json(
            self.selected_test_eval_path,
            {
                "schema_version": PHASE_FOUR_STATE_SCHEMA_VERSION,
                "results": results,
            },
        )
        return results

    def execute(self) -> dict[str, Any]:
        """Run the full phase-4 sweep, skipping completed trials."""

        backbone_trials = self._load_phase_three_backbones()
        state = self.load_state()
        best_trial_id = state.best_trial_id
        best_score = state.best_macro_f1_nc if state.best_macro_f1_nc is not None else -1.0

        for trial in build_phase_four_trials(tuple(backbone_trials), self.base_config.seeds):
            if trial.trial_id in state.completed_trials:
                existing = state.completed_trials[trial.trial_id]
                if existing.accepted and existing.macro_f1_nc >= best_score:
                    best_trial_id = trial.trial_id
                    best_score = existing.macro_f1_nc
                continue

            trial_record = self._run_trial(trial, backbone_trials)
            completed_trials = {**state.completed_trials, trial.trial_id: trial_record}
            best_trial_id, method_winner_ids = self._select_winners(completed_trials)
            if trial_record.accepted and trial_record.macro_f1_nc >= best_score:
                best_score = trial_record.macro_f1_nc
            state = PhaseFourSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_three_best_backbones_path=state.phase_three_best_backbones_path,
                phase_three_trial_ids=state.phase_three_trial_ids,
                completed_trials=completed_trials,
                best_trial_id=best_trial_id,
                best_macro_f1_nc=best_score if best_score >= 0.0 else None,
                method_winner_ids=method_winner_ids,
                created_at=state.created_at,
                updated_at=trial_record.completed_at,
            )
            self._save_state(state)

        if state.completed_trials:
            best_trial_id, method_winner_ids = self._select_winners(state.completed_trials)
            if best_trial_id is not None:
                best_score = state.completed_trials[best_trial_id].macro_f1_nc
            state = PhaseFourSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_three_best_backbones_path=state.phase_three_best_backbones_path,
                phase_three_trial_ids=state.phase_three_trial_ids,
                completed_trials=state.completed_trials,
                best_trial_id=best_trial_id,
                best_macro_f1_nc=best_score if best_score >= 0.0 else None,
                method_winner_ids=method_winner_ids,
                created_at=state.created_at,
                updated_at=_utc_now(),
            )
            self._save_state(state)

        selected_test_results = self._evaluate_selected_on_test(state)

        return {
            "phase": "phase-4",
            "frozen_backbones": [
                backbone_trials[trial_id].to_dict() for trial_id in state.phase_three_trial_ids
            ],
            "total_trials": len(
                build_phase_four_trials(tuple(backbone_trials), self.base_config.seeds)
            ),
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
            "selected_test_results": selected_test_results,
            "state_path": str(self.state_path),
            "best_eval_path": str(self.best_eval_path),
            "method_winners_path": str(self.method_winners_path),
            "selected_test_eval_path": str(self.selected_test_eval_path),
        }
