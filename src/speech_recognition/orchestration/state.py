"""State persistence for the pipeline runner."""

import importlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Self

from ..config import ExperimentConfig
from .sweep_utils import utc_now

PHASE_ORDER: tuple[str, ...] = ("phase-1", "phase-2", "phase-3", "phase-4")
"""Canonical execution order for the pipeline."""
STATE_SCHEMA_VERSION: int = 1
"""Current schema version for persisted pipeline state."""
_ACTIVE_MLFLOW_RUN_ID_ENV = "SPEECH_MLFLOW_ACTIVE_RUN_ID"


def _serialize(value: Any) -> Any:
    """Convert nested dataclasses into JSON-friendly values."""
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a JSON document atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary_path.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON document from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _phase_directory_name(phase: str) -> str:
    """Convert a phase name to a stable output directory name."""
    if phase not in PHASE_ORDER:
        raise ValueError(f"Unknown phase '{phase}'.")
    return phase.replace("-", "_")


@dataclass(frozen=True, slots=True)
class PhaseArtifact:
    """Serialized output for one completed phase."""

    phase: str
    """Phase identifier (e.g., phase-1)."""
    artifact_path: str
    """Path where this artifact was persisted."""
    input_data: dict[str, Any]
    """Input data provided to the phase."""
    output_data: dict[str, Any]
    """Output payload produced by the phase."""
    depends_on: tuple[str, ...] = ()
    """Upstream phase dependencies."""

    def __post_init__(self) -> None:
        if self.phase not in PHASE_ORDER:
            raise ValueError(f"Unknown phase '{self.phase}'.")
        if not isinstance(self.depends_on, tuple):
            raise ValueError("depends_on must be a tuple.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the phase artifact."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a phase artifact from a mapping."""
        payload = dict(data)
        if "depends_on" in payload:
            payload["depends_on"] = tuple(payload["depends_on"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PipelineState:
    """Persistent run state for the full pipeline."""

    schema_version: int = STATE_SCHEMA_VERSION
    """State file schema version."""
    run_name: str = "default"
    """Name identifier for this pipeline run."""
    run_root: str = ""
    """Root directory where run artifacts are stored."""
    config: dict[str, Any] = field(default_factory=dict)
    """Serialized experiment configuration."""
    completed_phases: tuple[str, ...] = ()
    """Tuple of completed phase names in execution order."""
    phase_artifacts: dict[str, PhaseArtifact] = field(default_factory=dict)
    """Completed phase artifacts keyed by phase name."""
    best_params: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Best hyperparameters per phase."""
    metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    """Best metrics per phase."""
    selected_phase_outputs: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Selected outputs from each phase."""
    checkpoint_pointers: dict[str, str] = field(default_factory=dict)
    """Checkpoint file paths per phase."""
    latest_checkpoint: str | None = None
    """Path to the most recent checkpoint."""
    created_at: str = field(default_factory=utc_now)
    """ISO-8601 timestamp when state was created."""
    updated_at: str = field(default_factory=utc_now)
    """ISO-8601 timestamp when state was last updated."""

    def __post_init__(self) -> None:
        if self.schema_version != STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported state schema version.")
        if not self.run_name:
            raise ValueError("run_name must not be empty.")
        if not isinstance(self.completed_phases, tuple):
            raise ValueError("completed_phases must be a tuple.")

    def with_artifact(self, artifact: PhaseArtifact) -> Self:
        """Return a new state with a completed phase recorded."""
        phase_artifacts = dict(self.phase_artifacts)
        phase_artifacts[artifact.phase] = artifact
        completed_phases = tuple(
            phase for phase in self.completed_phases if phase != artifact.phase
        )
        completed_phases = (*completed_phases, artifact.phase)
        return replace(
            self,
            completed_phases=completed_phases,
            phase_artifacts=phase_artifacts,
            selected_phase_outputs={
                **self.selected_phase_outputs,
                artifact.phase: artifact.output_data,
            },
            best_params={
                **self.best_params,
                artifact.phase: dict(artifact.output_data.get("best_params", {})),
            },
            metrics={
                **self.metrics,
                artifact.phase: {
                    "macro_f1": float(artifact.output_data.get("metrics", {}).get("macro_f1", 0.0))
                },
            },
            checkpoint_pointers={
                **self.checkpoint_pointers,
                **(
                    {artifact.phase: artifact.output_data["checkpoint_pointer"]}
                    if "checkpoint_pointer" in artifact.output_data
                    else {}
                ),
            },
            latest_checkpoint=artifact.output_data.get(
                "checkpoint_pointer", self.latest_checkpoint
            ),
            updated_at=utc_now(),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the state."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build state from a mapping."""
        payload = dict(data)
        if "completed_phases" in payload:
            payload["completed_phases"] = tuple(payload["completed_phases"])
        if "phase_artifacts" in payload:
            payload["phase_artifacts"] = {
                key: PhaseArtifact.from_dict(value)
                for key, value in payload["phase_artifacts"].items()
            }
        return cls(**payload)

    @classmethod
    def from_config(cls, run_name: str, run_root: Path, config: ExperimentConfig) -> Self:
        """Create a fresh state for a new pipeline run."""
        return cls(run_name=run_name, run_root=str(run_root), config=config.to_dict())


@dataclass(frozen=True, slots=True)
class PipelineStateStore:
    """Manage persistent run state and phase artifacts on disk."""

    base_dir: Path
    """Root directory for all run state and artifacts."""
    use_mlflow: bool = False
    """Persist state and artifacts in MLflow instead of local files."""
    tracking_uri: str | None = None
    """MLflow tracking URI used when MLflow-backed persistence is enabled."""
    experiment_name: str | None = None
    """MLflow experiment name used to locate run artifacts for hot-start."""

    def phase_dir(self, phase: str) -> Path:
        """Return the top-level directory for one phase."""
        return self.base_dir / _phase_directory_name(phase)

    def phase_runs_dir(self, phase: str) -> Path:
        """Return the run directory root for one phase."""
        return self.phase_dir(phase) / "runs"

    def run_dir(self, run_name: str, phase: str) -> Path:
        """Return the directory that stores one run for one phase."""
        return self.phase_runs_dir(phase) / run_name

    def checkpoints_dir(self) -> Path:
        """Return the directory for checkpoint metadata and pointers."""
        return self.base_dir / "checkpoints"

    def checkpoint_pointer_path(self, run_name: str) -> Path:
        """Return checkpoint pointer metadata path for one run."""
        return self.checkpoints_dir() / f"{run_name}.json"

    def state_path(self, run_name: str, phase: str) -> Path:
        """Return the path of the state file for a run and phase."""
        return self.run_dir(run_name, phase) / "state.json"

    def phase_artifact_path(self, run_name: str, phase: str) -> Path:
        """Return the artifact path for one run and phase."""
        if self.use_mlflow:
            return Path("mlflow") / self._mlflow_phase_artifact_path(phase)
        return self.run_dir(run_name, phase) / "artifact.json"

    def _resolve_tracking_uri(self) -> str:
        """Resolve local tracking URIs to absolute paths when needed."""
        raw = (self.tracking_uri or "sqlite:///mlruns.db").strip()
        if raw.startswith("sqlite:"):
            sqlite_target = raw.removeprefix("sqlite:")
            if sqlite_target.lstrip("/") == ":memory:":
                return "sqlite:///:memory:"
            if sqlite_target.startswith("///"):
                db_target = sqlite_target[3:]
            elif sqlite_target.startswith("//"):
                db_target = sqlite_target[2:]
            elif sqlite_target.startswith("/"):
                db_target = sqlite_target[1:]
            else:
                db_target = sqlite_target
            if not db_target:
                db_target = "mlruns.db"
            resolved_db_path = Path(db_target).expanduser().resolve()
            resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
            return f"sqlite:///{resolved_db_path.as_posix()}"

        if "://" in raw and not raw.startswith("file:"):
            return raw

        resolved_path = Path(raw).expanduser().resolve()
        resolved_path.mkdir(parents=True, exist_ok=True)
        return str(resolved_path)

    def _mlflow_phase_state_path(self, phase: str) -> str:
        return f"pipeline_state/{_phase_directory_name(phase)}/state.json"

    def _mlflow_phase_artifact_path(self, phase: str) -> str:
        return f"pipeline_state/{_phase_directory_name(phase)}/artifact.json"

    def _mlflow_checkpoint_pointer_path(self, run_name: str) -> str:
        return f"pipeline_state/checkpoints/{run_name}.json"

    def _mlflow_modules(self) -> tuple[Any, Any]:
        """Return configured MLflow module and client."""
        mlflow = importlib.import_module("mlflow")
        mlflow.set_tracking_uri(self._resolve_tracking_uri())
        client = mlflow.tracking.MlflowClient()
        return mlflow, client

    def _active_mlflow_run_id(self, run_name: str | None = None) -> str | None:
        """Return active MLflow run id from environment when it can be validated."""
        run_id = os.environ.get(_ACTIVE_MLFLOW_RUN_ID_ENV)
        if run_id is None:
            return None
        stripped = run_id.strip()
        if not stripped:
            return None
        if run_name is None:
            return stripped

        try:
            _, client = self._mlflow_modules()
            get_run = getattr(client, "get_run", None)
            if not callable(get_run):
                return None
            run = get_run(stripped)
            tags = getattr(getattr(run, "data", None), "tags", {}) or {}
            if tags.get("pipeline.run_name") != run_name:
                return None
            return stripped
        except Exception:
            return None

    def _find_mlflow_run_id(self, run_name: str) -> str:
        """Find the latest MLflow run id matching a pipeline run name."""
        _, client = self._mlflow_modules()
        experiment_name = self.experiment_name or "speech-recognition"
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            raise FileNotFoundError(
                f"MLflow experiment '{experiment_name}' does not exist for run '{run_name}'."
            )

        safe_run_name = run_name.replace("'", "\\'")
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"tags.pipeline.run_name = '{safe_run_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            raise FileNotFoundError(
                f"No MLflow run with tag pipeline.run_name='{run_name}' "
                f"found in experiment '{experiment_name}'."
            )
        return runs[0].info.run_id

    def _find_or_create_mlflow_run_id(self, run_name: str) -> str:
        """Find or create an MLflow run used for pipeline state persistence."""
        _, client = self._mlflow_modules()
        experiment_name = self.experiment_name or "speech-recognition"
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            try:
                experiment_id = client.create_experiment(experiment_name)
            except Exception:
                experiment = client.get_experiment_by_name(experiment_name)
                if experiment is None:
                    raise
                experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id

        safe_run_name = run_name.replace("'", "\\'")
        runs = client.search_runs(
            [experiment_id],
            filter_string=f"tags.pipeline.run_name = '{safe_run_name}'",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id

        created_run = client.create_run(
            experiment_id=experiment_id,
            tags={
                "pipeline.run_name": run_name,
                "mlflow.runName": run_name,
                "pipeline.run_role": "pipeline-state",
            },
        )
        return created_run.info.run_id

    def _mlflow_log_json(
        self, artifact_path: str, payload: Mapping[str, Any], run_name: str
    ) -> None:
        """Log a JSON payload into MLflow artifacts."""
        mlflow, client = self._mlflow_modules()
        active_run_id = self._active_mlflow_run_id(run_name=run_name)

        if active_run_id is not None:
            log_dict = getattr(mlflow, "log_dict", None)
            if callable(log_dict):
                log_dict(dict(payload), artifact_path)
                return
            log_text = getattr(mlflow, "log_text", None)
            if callable(log_text):
                log_text(json.dumps(payload, indent=2, sort_keys=True), artifact_path)
                return

            with tempfile.TemporaryDirectory() as temporary_dir:
                local_name = Path(artifact_path).name
                local_path = Path(temporary_dir) / local_name
                local_path.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                client.log_artifact(
                    active_run_id,
                    str(local_path),
                    artifact_path=str(Path(artifact_path).parent),
                )
            return

        run_id = self._find_or_create_mlflow_run_id(run_name)
        with tempfile.TemporaryDirectory() as temporary_dir:
            local_name = Path(artifact_path).name
            local_path = Path(temporary_dir) / local_name
            local_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            client.log_artifact(
                run_id,
                str(local_path),
                artifact_path=str(Path(artifact_path).parent),
            )

    def _mlflow_load_json(self, run_id: str, artifact_path: str) -> dict[str, Any] | None:
        """Load a JSON artifact payload from MLflow, returning None when absent."""
        _, client = self._mlflow_modules()
        with tempfile.TemporaryDirectory() as temporary_dir:
            try:
                local_path = client.download_artifacts(run_id, artifact_path, temporary_dir)
            except Exception:
                return None
            try:
                return json.loads(Path(local_path).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                return None

    def _state_candidates(self, run_name: str) -> list[Path]:
        """Return state file candidates from newest phase to oldest."""
        return [self.state_path(run_name, phase) for phase in reversed(PHASE_ORDER)]

    def _load_first_valid_state(self, paths: list[Path]) -> PipelineState | None:
        """Load the first valid state from a list of candidate paths."""
        for path in paths:
            if not path.exists():
                continue
            try:
                payload = _read_json(path)
                return PipelineState.from_dict(payload)
            except (json.JSONDecodeError, OSError, ValueError):
                continue
        return None

    def load(self, run_name: str) -> PipelineState:
        """Load state from disk with corruption-tolerant fallback."""
        if self.use_mlflow:
            run_id = self._active_mlflow_run_id(run_name=run_name) or self._find_mlflow_run_id(
                run_name
            )
            for phase in reversed(PHASE_ORDER):
                payload = self._mlflow_load_json(run_id, self._mlflow_phase_state_path(phase))
                if payload is None:
                    continue
                try:
                    return PipelineState.from_dict(payload)
                except ValueError:
                    continue

            pointer_payload = self._mlflow_load_json(
                run_id,
                self._mlflow_checkpoint_pointer_path(run_name),
            )
            if isinstance(pointer_payload, Mapping):
                phase = pointer_payload.get("latest_phase")
                if isinstance(phase, str):
                    fallback_payload = self._mlflow_load_json(
                        run_id,
                        self._mlflow_phase_state_path(phase),
                    )
                    if fallback_payload is not None:
                        try:
                            return PipelineState.from_dict(fallback_payload)
                        except ValueError:
                            pass

            raise FileNotFoundError(
                f"No valid MLflow-backed state found for run '{run_name}' "
                f"in experiment '{self.experiment_name or 'speech-recognition'}'."
            )

        state = self._load_first_valid_state(self._state_candidates(run_name))
        if state is not None:
            return state

        pointer_path = self.checkpoint_pointer_path(run_name)
        if pointer_path.exists():
            try:
                pointer_payload = _read_json(pointer_path)
                phase = pointer_payload["latest_phase"]
                return PipelineState.from_dict(_read_json(self.state_path(run_name, phase)))
            except (json.JSONDecodeError, OSError, KeyError, ValueError):
                pass

        raise FileNotFoundError(
            f"No valid state found for run '{run_name}' under '{self.base_dir}'."
        )

    def load_or_create(
        self,
        run_name: str,
        run_root: Path,
        config: ExperimentConfig | None,
    ) -> PipelineState:
        """Load an existing run or create a fresh one."""
        try:
            state = self.load(run_name)
        except FileNotFoundError:
            return PipelineState.from_config(run_name, run_root, config or ExperimentConfig())

        if config is not None and state.config != config.to_dict():
            raise ValueError("Existing run state does not match the requested config.")
        return state

    def save(self, state: PipelineState) -> None:
        """Persist state snapshots and checkpoint pointer metadata."""
        latest_phase = state.completed_phases[-1] if state.completed_phases else "phase-1"

        if self.use_mlflow:
            self._mlflow_log_json(
                self._mlflow_phase_state_path(latest_phase),
                state.to_dict(),
                state.run_name,
            )
            self._mlflow_log_json(
                self._mlflow_checkpoint_pointer_path(state.run_name),
                {
                    "schema_version": STATE_SCHEMA_VERSION,
                    "run_name": state.run_name,
                    "latest_phase": latest_phase,
                    "latest_checkpoint": state.latest_checkpoint,
                    "checkpoint_pointers": state.checkpoint_pointers,
                    "updated_at": state.updated_at,
                },
                state.run_name,
            )
            return

        _atomic_write_json(self.state_path(state.run_name, latest_phase), state.to_dict())

        _atomic_write_json(
            self.checkpoint_pointer_path(state.run_name),
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "run_name": state.run_name,
                "latest_phase": latest_phase,
                "latest_checkpoint": state.latest_checkpoint,
                "checkpoint_pointers": state.checkpoint_pointers,
                "updated_at": state.updated_at,
            },
        )

    def save_artifact(self, state: PipelineState, artifact: PhaseArtifact) -> None:
        """Persist a phase artifact to disk."""
        if self.use_mlflow:
            self._mlflow_log_json(
                self._mlflow_phase_artifact_path(artifact.phase),
                artifact.to_dict(),
                state.run_name,
            )
            return

        _atomic_write_json(
            self.phase_artifact_path(state.run_name, artifact.phase), artifact.to_dict()
        )


def save_json_artifact(
    store: PipelineStateStore,
    local_path: Path,
    run_name: str,
    mlflow_artifact_path: str,
    payload: Mapping[str, Any],
) -> None:
    """Persist a JSON payload locally or to MLflow depending on store configuration."""
    if store.use_mlflow:
        store._mlflow_log_json(mlflow_artifact_path, payload, run_name)
        return
    _atomic_write_json(local_path, payload)


def load_json_artifact(
    store: PipelineStateStore,
    local_path: Path,
    run_name: str,
    mlflow_artifact_path: str,
) -> dict[str, Any] | None:
    """Load a JSON payload locally or from MLflow depending on store configuration."""
    if store.use_mlflow:
        try:
            run_id = store._active_mlflow_run_id() or store._find_mlflow_run_id(run_name)
        except FileNotFoundError:
            return None
        return store._mlflow_load_json(run_id, mlflow_artifact_path)

    if not local_path.exists():
        return None
    return _read_json(local_path)


def load_json_artifact_or_raise(
    store: PipelineStateStore,
    local_path: Path,
    run_name: str,
    mlflow_artifact_path: str,
    description: str,
) -> dict[str, Any]:
    """Load a JSON payload or raise FileNotFoundError when it cannot be found."""
    payload = load_json_artifact(store, local_path, run_name, mlflow_artifact_path)
    if payload is None:
        raise FileNotFoundError(description)
    return payload
