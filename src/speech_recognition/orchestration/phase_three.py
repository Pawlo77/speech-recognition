"""Phase-3 architecture comparison sweep orchestration."""

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..config import DEFAULT_SEEDS, ExperimentConfig, ModelConfig
from .phase_one import _atomic_write_json, _feature_pipeline_for_trial, _read_json, _serialize
from .phase_two import _phase_two_feature_config, _scheduler_config_for_trial
from .services import build_isolated_subprocess_env

PHASE_THREE_STATE_SCHEMA_VERSION = 1
"""Schema version for the phase-3 sweep state file."""

PHASE_THREE_SEEDS: tuple[int, int, int] = DEFAULT_SEEDS
"""Fixed seeds used for the phase-3 sweep grid."""

PHASE_THREE_AST_DROPOUTS: tuple[float, float] = (0.1, 0.5)
PHASE_THREE_AST_HEADS: tuple[str, str] = ("linear", "mlp_256")
PHASE_THREE_AST_POSITIONAL_EMBEDDINGS: tuple[str, str] = ("interp", "learned")

PHASE_THREE_CONVNEXT_STOCH_DEPTHS: tuple[float, float] = (0.0, 0.2)
PHASE_THREE_CONVNEXT_KERNEL_SIZES: tuple[int, ...] = (7,)

PHASE_THREE_SSAMBA_POOLINGS: tuple[str, str] = ("mean", "max")
PHASE_THREE_SSAMBA_CLS: tuple[bool, bool] = (True, False)
PHASE_THREE_SSAMBA_STRIDES_MS: tuple[int, int] = (10, 5)

PHASE_THREE_XLSTM_DIMS: tuple[int, int] = (32, 64)
PHASE_THREE_XLSTM_STATE_RESETS: tuple[bool, bool] = (True, False)
PHASE_THREE_XLSTM_OUTPUTS: tuple[str, str] = ("final", "mean")

PHASE_THREE_MLP_MIXER_DROPOUTS: tuple[float, float] = (0.0, 0.2)
PHASE_THREE_MLP_MIXER_HEAD_L2_NORM: tuple[bool, bool] = (True, False)


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(UTC).isoformat()


def _phase_three_score(payload: Mapping[str, Any]) -> float:
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
                    score = _phase_three_score(output_data)
                    if score > 0.0:
                        return score
    return 0.0


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


def build_phase_three_command(config_path: Path, run_name: str) -> list[str]:
    """Build the isolated subprocess command for one phase-3 trial."""

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
class PhaseThreeTrialSpec:
    """Describe one trial in the phase-3 architecture sweep."""

    trial_id: str
    family: str
    seed: int
    architecture_params: dict[str, Any]

    def to_config(
        self, base_config: ExperimentConfig, feature_name: str, optimizer_payload: Mapping[str, Any]
    ) -> ExperimentConfig:
        """Return the concrete config for this trial."""

        dataset = replace(base_config.dataset, train_split="train_small", valid_split="valid_small")
        features = _feature_pipeline_for_trial(feature_name)
        optimizer = replace(
            base_config.optimizer, weight_decay=float(optimizer_payload["weight_decay"])
        )
        scheduler = _scheduler_config_for_trial(
            str(optimizer_payload["scheduler_name"]), total_epochs=base_config.training.epochs
        )
        model = self._build_model_config(base_config.model)
        phase_config = replace(base_config.phase, phase="phase_3")
        return replace(
            base_config,
            dataset=dataset,
            features=features,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            phase=phase_config,
        )

    def _build_model_config(self, base_model: ModelConfig) -> ModelConfig:
        """Build the model configuration for one trial."""

        params = self.architecture_params
        family = self.family
        if family == "ast":
            return replace(
                base_model,
                family=family,
                pretrained=False,
                dropout=float(params["dropout"]),
            )
        if family == "convnext":
            return replace(
                base_model,
                family=family,
                pretrained=False,
                stochastic_depth=float(params["stochastic_depth"]),
            )
        if family == "ssamba":
            return replace(base_model, family=family, pretrained=False)
        if family == "xlstm":
            return replace(base_model, family=family, pretrained=False)
        if family == "mlp_mixer":
            return replace(base_model, family=family, pretrained=False)
        raise ValueError(f"Unsupported phase-3 model family '{family}'.")


def _build_ast_trials() -> tuple[PhaseThreeTrialSpec, ...]:
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in PHASE_THREE_SEEDS:
        for dropout in PHASE_THREE_AST_DROPOUTS:
            for head in PHASE_THREE_AST_HEADS:
                for positional_embedding in PHASE_THREE_AST_POSITIONAL_EMBEDDINGS:
                    trial_index += 1
                    trials.append(
                        PhaseThreeTrialSpec(
                            trial_id=(
                                f"trial_{trial_index:02d}_ast_{head}_{positional_embedding}_"
                                f"dropout_{dropout}_seed_{seed}"
                            ),
                            family="ast",
                            seed=seed,
                            architecture_params={
                                "dropout": dropout,
                                "head": head,
                                "positional_embedding": positional_embedding,
                            },
                        )
                    )
    return tuple(trials)


def _build_convnext_trials() -> tuple[PhaseThreeTrialSpec, ...]:
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in PHASE_THREE_SEEDS:
        for stochastic_depth in PHASE_THREE_CONVNEXT_STOCH_DEPTHS:
            for kernel_size in PHASE_THREE_CONVNEXT_KERNEL_SIZES:
                trial_index += 1
                trials.append(
                    PhaseThreeTrialSpec(
                        trial_id=(
                            f"trial_{trial_index:02d}_convnext_sd_{stochastic_depth}_"
                            f"kernel_{kernel_size}_seed_{seed}"
                        ),
                        family="convnext",
                        seed=seed,
                        architecture_params={
                            "stochastic_depth": stochastic_depth,
                            "kernel_size": kernel_size,
                        },
                    )
                )
    return tuple(trials)


def _build_ssamba_trials() -> tuple[PhaseThreeTrialSpec, ...]:
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in PHASE_THREE_SEEDS:
        for pooling in PHASE_THREE_SSAMBA_POOLINGS:
            for use_cls in PHASE_THREE_SSAMBA_CLS:
                for stride_ms in PHASE_THREE_SSAMBA_STRIDES_MS:
                    trial_index += 1
                    trials.append(
                        PhaseThreeTrialSpec(
                            trial_id=(
                                f"trial_{trial_index:02d}_ssamba_{pooling}_cls_{use_cls}_"
                                f"stride_{stride_ms}ms_seed_{seed}"
                            ),
                            family="ssamba",
                            seed=seed,
                            architecture_params={
                                "pooling": pooling,
                                "use_cls": use_cls,
                                "stride_ms": stride_ms,
                            },
                        )
                    )
    return tuple(trials)


def _build_xlstm_trials() -> tuple[PhaseThreeTrialSpec, ...]:
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in PHASE_THREE_SEEDS:
        for dimension in PHASE_THREE_XLSTM_DIMS:
            for state_reset in PHASE_THREE_XLSTM_STATE_RESETS:
                for output_mode in PHASE_THREE_XLSTM_OUTPUTS:
                    trial_index += 1
                    trials.append(
                        PhaseThreeTrialSpec(
                            trial_id=(
                                f"trial_{trial_index:02d}_xlstm_d_{dimension}_reset_{state_reset}_"
                                f"output_{output_mode}_seed_{seed}"
                            ),
                            family="xlstm",
                            seed=seed,
                            architecture_params={
                                "dimension": dimension,
                                "state_reset": state_reset,
                                "output_mode": output_mode,
                            },
                        )
                    )
    return tuple(trials)


def _build_mlp_mixer_trials() -> tuple[PhaseThreeTrialSpec, ...]:
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in PHASE_THREE_SEEDS:
        for dropout in PHASE_THREE_MLP_MIXER_DROPOUTS:
            for head_l2_norm in PHASE_THREE_MLP_MIXER_HEAD_L2_NORM:
                trial_index += 1
                trials.append(
                    PhaseThreeTrialSpec(
                        trial_id=(
                            f"trial_{trial_index:02d}_mlp_mixer_dropout_{dropout}_"
                            f"head_l2_{head_l2_norm}_seed_{seed}"
                        ),
                        family="mlp_mixer",
                        seed=seed,
                        architecture_params={
                            "dropout": dropout,
                            "head_l2_norm": head_l2_norm,
                        },
                    )
                )
    return tuple(trials)


def build_phase_three_trials() -> tuple[PhaseThreeTrialSpec, ...]:
    """Return the 90 trial specifications for phase 3."""

    return (
        *_build_ast_trials(),
        *_build_convnext_trials(),
        *_build_ssamba_trials(),
        *_build_xlstm_trials(),
        *_build_mlp_mixer_trials(),
    )


@dataclass(frozen=True, slots=True)
class PhaseThreeTrialRecord:
    """Persisted record for one completed phase-3 trial."""

    trial_id: str
    family: str
    seed: int
    architecture_params: dict[str, Any]
    run_name: str
    config_path: str
    child_state_path: str
    validation_macro_f1: float
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record."""

        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseThreeTrialRecord":
        """Build a record from JSON data."""

        payload = dict(data)
        payload["seed"] = int(payload["seed"])
        payload["validation_macro_f1"] = float(payload["validation_macro_f1"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PhaseThreeSweepState:
    """Persistent state for the phase-3 sweep."""

    schema_version: int = PHASE_THREE_STATE_SCHEMA_VERSION
    output_dir: str = ""
    phase_one_best_feature_path: str = ""
    phase_two_best_optim_path: str = ""
    completed_trials: dict[str, PhaseThreeTrialRecord] = field(default_factory=dict)
    best_trial_id: str | None = None
    best_validation_macro_f1: float | None = None
    top_three_trial_ids: tuple[str, ...] = ()
    family_winner_ids: dict[str, str] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema_version != PHASE_THREE_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported phase-3 state schema version.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the sweep state."""

        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseThreeSweepState":
        """Build phase-3 state from JSON."""

        payload = dict(data)
        payload["completed_trials"] = {
            key: PhaseThreeTrialRecord.from_dict(value)
            for key, value in payload.get("completed_trials", {}).items()
        }
        if payload.get("best_validation_macro_f1") is not None:
            payload["best_validation_macro_f1"] = float(payload["best_validation_macro_f1"])
        if "top_three_trial_ids" in payload:
            payload["top_three_trial_ids"] = tuple(payload["top_three_trial_ids"])
        return cls(**payload)

    @classmethod
    def fresh(
        cls, output_dir: Path, phase_one_best_feature_path: Path, phase_two_best_optim_path: Path
    ) -> "PhaseThreeSweepState":
        """Create a new empty state for an output directory."""

        return cls(
            output_dir=str(output_dir),
            phase_one_best_feature_path=str(phase_one_best_feature_path),
            phase_two_best_optim_path=str(phase_two_best_optim_path),
        )


def _read_artifact(path: Path, description: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at '{path}'.")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{description} must be a JSON object.")
    return payload


class PhaseThreeSweepRunner:
    """Run the phase-3 architecture comparison sweep using child processes."""

    def __init__(self, output_dir: Path, base_config: ExperimentConfig | None = None) -> None:
        self.output_dir = output_dir
        self.phase_dir = self.output_dir / "phase_3"
        self.state_path = self.phase_dir / "state.json"
        self.best_backbones_path = self.phase_dir / "best_backbones.json"
        self.family_winners_path = self.phase_dir / "family_winners.json"
        self.base_config = base_config or ExperimentConfig()
        self.phase_one_best_feature_path = self.output_dir / "phase_1" / "best_feature.json"
        self.phase_two_best_optim_path = self.output_dir / "phase_2" / "best_optim.json"

    def load_state(self) -> PhaseThreeSweepState:
        """Load the persisted sweep state or create a new one."""

        if not self.state_path.exists():
            return PhaseThreeSweepState.fresh(
                self.output_dir,
                self.phase_one_best_feature_path,
                self.phase_two_best_optim_path,
            )
        return PhaseThreeSweepState.from_dict(_read_json(self.state_path))

    def _save_state(self, state: PhaseThreeSweepState) -> None:
        """Persist the sweep state and selection artifacts."""

        _atomic_write_json(self.state_path, state.to_dict())
        if state.best_trial_id is not None:
            best_trial = state.completed_trials[state.best_trial_id]
            _atomic_write_json(
                self.best_backbones_path,
                {
                    "schema_version": PHASE_THREE_STATE_SCHEMA_VERSION,
                    "best_trial": best_trial.to_dict(),
                    "top_three_trials": [
                        state.completed_trials[trial_id].to_dict()
                        for trial_id in state.top_three_trial_ids
                    ],
                },
            )
        if state.family_winner_ids:
            _atomic_write_json(
                self.family_winners_path,
                {
                    "schema_version": PHASE_THREE_STATE_SCHEMA_VERSION,
                    "family_winners": {
                        family: state.completed_trials[trial_id].to_dict()
                        for family, trial_id in state.family_winner_ids.items()
                    },
                },
            )

    def _trial_run_name(self, trial: PhaseThreeTrialSpec) -> str:
        return trial.trial_id

    def _trial_output_paths(self, trial: PhaseThreeTrialSpec) -> tuple[Path, Path, Path]:
        trial_dir = self.phase_dir / "runs" / trial.trial_id
        config_path = trial_dir / "temp_config.json"
        child_state_path = self.output_dir / "phase_3" / "runs" / trial.trial_id / "state.json"
        return trial_dir, config_path, child_state_path

    def _build_trial_config(
        self,
        trial: PhaseThreeTrialSpec,
        feature_payload: Mapping[str, Any],
        optimizer_payload: Mapping[str, Any],
    ) -> ExperimentConfig:
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
        trial_dir, config_path, child_state_path = self._trial_output_paths(trial)
        trial_dir.mkdir(parents=True, exist_ok=True)

        config = self._build_trial_config(trial, feature_payload, optimizer_payload)
        config_path.write_text(
            json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

        command = build_phase_three_command(config_path, self._trial_run_name(trial))
        subprocess.run(  # noqa: S603
            command,
            check=True,
            env=build_isolated_subprocess_env(),
            text=True,
            capture_output=True,
        )

        summary = _read_json(child_state_path) if child_state_path.exists() else {}
        validation_macro_f1 = _phase_three_score(summary)
        return PhaseThreeTrialRecord(
            trial_id=trial.trial_id,
            family=trial.family,
            seed=trial.seed,
            architecture_params=dict(trial.architecture_params),
            run_name=self._trial_run_name(trial),
            config_path=str(config_path),
            child_state_path=str(child_state_path),
            validation_macro_f1=validation_macro_f1,
            completed_at=_utc_now(),
        )

    def _select_winners(
        self, records: Mapping[str, PhaseThreeTrialRecord]
    ) -> tuple[str | None, tuple[str, ...], dict[str, str]]:
        if not records:
            return None, (), {}

        sorted_records = sorted(
            records.values(), key=lambda record: record.validation_macro_f1, reverse=True
        )
        best_trial_id = sorted_records[0].trial_id
        top_three_trial_ids = tuple(record.trial_id for record in sorted_records[:3])

        family_winner_ids: dict[str, str] = {}
        family_groups: dict[str, list[PhaseThreeTrialRecord]] = {}
        for record in records.values():
            family_groups.setdefault(record.family, []).append(record)

        for family, family_records in family_groups.items():
            winner = max(family_records, key=lambda record: record.validation_macro_f1)
            family_winner_ids[family] = winner.trial_id

        return best_trial_id, top_three_trial_ids, family_winner_ids

    def execute(self) -> dict[str, Any]:
        """Run the full phase-3 sweep, skipping completed trials."""

        feature_artifact = _read_artifact(
            self.phase_one_best_feature_path, "Phase-1 best feature artifact"
        )
        feature_summary = _phase_three_feature_config(feature_artifact)
        optimizer_artifact = _read_artifact(
            self.phase_two_best_optim_path, "Phase-2 best optimization artifact"
        )
        optimizer_summary = _phase_three_optimizer_config(optimizer_artifact)

        state = self.load_state()
        best_trial_id = state.best_trial_id
        best_score = (
            state.best_validation_macro_f1 if state.best_validation_macro_f1 is not None else -1.0
        )

        for trial in build_phase_three_trials():
            if trial.trial_id in state.completed_trials:
                existing = state.completed_trials[trial.trial_id]
                if existing.validation_macro_f1 >= best_score:
                    best_trial_id = trial.trial_id
                    best_score = existing.validation_macro_f1
                continue

            trial_record = self._run_trial(trial, feature_summary, optimizer_summary)
            completed_trials = {**state.completed_trials, trial.trial_id: trial_record}
            if trial_record.validation_macro_f1 >= best_score:
                best_trial_id = trial.trial_id
                best_score = trial_record.validation_macro_f1

            best_trial_id, top_three_trial_ids, family_winner_ids = self._select_winners(
                completed_trials
            )
            state = PhaseThreeSweepState(
                schema_version=state.schema_version,
                output_dir=state.output_dir,
                phase_one_best_feature_path=state.phase_one_best_feature_path,
                phase_two_best_optim_path=state.phase_two_best_optim_path,
                completed_trials=completed_trials,
                best_trial_id=best_trial_id,
                best_validation_macro_f1=best_score if best_score >= 0.0 else None,
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
                updated_at=_utc_now(),
            )
            self._save_state(state)

        return {
            "phase": "phase-3",
            "feature_source": feature_summary,
            "optimizer_source": optimizer_summary,
            "total_trials": len(build_phase_three_trials()),
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
