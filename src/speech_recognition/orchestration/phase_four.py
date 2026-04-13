"""Phase-4 held-out evaluation orchestration."""

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig, ModelConfig
from .phase_one import _atomic_write_json, _read_json, _serialize
from .phase_three import PhaseThreeTrialRecord
from .services import build_isolated_subprocess_env

PHASE_FOUR_STATE_SCHEMA_VERSION = 1
"""Schema version for the phase-4 sweep state file."""

PHASE_FOUR_METHODS: tuple[str, ...] = (
    "flat_multiclass",
    "sampling_control",
    "loss_reweighting",
    "two_stage_detector",
    "shared_two_head",
)
"""Supported final non-command handling strategies."""

PHASE_FOUR_STRICT_DROP_LIMIT = 0.01
"""Maximum tolerated core-command macro-F1 drop relative to the Phase 3 baseline."""

PHASE_FOUR_WARMUP_ITERATIONS = 50
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

    return {
        "core_command_macro_f1": float(core_command_macro_f1),
        "unknown_f1": float(unknown_f1),
        "silence_f1": float(silence_f1),
        "macro_f1_nc": float(macro_f1_nc),
        "inference_latency_ms_mean": float(inference_latency_ms_mean),
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


def _build_backbone_model_config(backbone_trial: PhaseThreeTrialRecord) -> ModelConfig:
    """Build a model config representative of one frozen phase-3 backbone."""

    family = backbone_trial.family
    params = backbone_trial.architecture_params
    if family == "ast":
        return ModelConfig(
            family="ast",
            pretrained=False,
            dropout=float(params["dropout"]),
        )
    if family == "convnext":
        return ModelConfig(
            family="convnext",
            pretrained=False,
            stochastic_depth=float(params["stochastic_depth"]),
        )
    if family == "ssamba":
        return ModelConfig(family="ssamba", pretrained=False)
    if family == "xlstm":
        return ModelConfig(family="xlstm", pretrained=False)
    if family == "mlp_mixer":
        return ModelConfig(family="mlp_mixer", pretrained=False)
    raise ValueError(f"Unsupported phase-3 backbone family '{family}'.")


def build_phase_four_command(config_path: Path, run_name: str) -> list[str]:
    """Build the isolated subprocess command for one phase-4 trial."""

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


@dataclass(frozen=True, slots=True)
class PhaseFourTrialSpec:
    """Describe one held-out evaluation trial."""

    trial_id: str
    method: str
    backbone_ids: tuple[str, ...]
    seed: int

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
        phase_config = replace(base_config.phase, phase="phase_4")
        return replace(
            base_config,
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
    method: str
    backbone_ids: tuple[str, ...]
    seed: int
    baseline_backbone_id: str
    baseline_core_command_macro_f1: float
    core_command_macro_f1: float
    unknown_f1: float
    silence_f1: float
    macro_f1_nc: float
    inference_latency_ms_mean: float
    accepted: bool
    config_path: str
    child_state_path: str
    completed_at: str

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
        payload["accepted"] = bool(payload["accepted"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PhaseFourSweepState:
    """Persistent state for the phase-4 sweep."""

    schema_version: int = PHASE_FOUR_STATE_SCHEMA_VERSION
    output_dir: str = ""
    phase_three_best_backbones_path: str = ""
    phase_three_trial_ids: tuple[str, ...] = ()
    completed_trials: dict[str, PhaseFourTrialRecord] = field(default_factory=dict)
    best_trial_id: str | None = None
    best_macro_f1_nc: float | None = None
    method_winner_ids: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

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
        subprocess.run(  # noqa: S603
            command,
            check=True,
            env=build_isolated_subprocess_env(),
            text=True,
            capture_output=True,
        )

        summary = _read_json(child_state_path) if child_state_path.exists() else {}
        metrics = _phase_four_score_recursive(summary)
        baseline_trial_id = trial.backbone_ids[0]
        baseline_score = float(backbone_trials[baseline_trial_id].validation_macro_f1)
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
            accepted=accepted,
            config_path=str(config_path),
            child_state_path=str(child_state_path),
            completed_at=_utc_now(),
        )

    def _select_winners(
        self, records: Mapping[str, PhaseFourTrialRecord]
    ) -> tuple[str | None, dict[str, str]]:
        accepted_records = [record for record in records.values() if record.accepted]
        if not accepted_records:
            return None, {}

        best_trial_id = max(accepted_records, key=lambda record: record.macro_f1_nc).trial_id
        method_winner_ids: dict[str, str] = {}
        for method in PHASE_FOUR_METHODS:
            method_records = [record for record in accepted_records if record.method == method]
            if method_records:
                method_winner_ids[method] = max(
                    method_records, key=lambda record: record.macro_f1_nc
                ).trial_id
        return best_trial_id, method_winner_ids

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
            "state_path": str(self.state_path),
            "best_eval_path": str(self.best_eval_path),
            "method_winners_path": str(self.method_winners_path),
        }
