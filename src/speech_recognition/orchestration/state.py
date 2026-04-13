"""State persistence for the pipeline runner."""

import json
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

from ..config import ExperimentConfig

PHASE_ORDER: tuple[str, ...] = ("phase-1", "phase-2", "phase-3", "phase-4")
"""Canonical execution order for the pipeline."""

STATE_SCHEMA_VERSION: int = 1
"""Current schema version for persisted pipeline state."""


def _utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""

    return datetime.now(UTC).isoformat()


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
    created_at: str = field(default_factory=_utc_now)
    """ISO-8601 timestamp when state was created."""
    updated_at: str = field(default_factory=_utc_now)
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
            updated_at=_utc_now(),
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

        return self.run_dir(run_name, phase) / "artifact.json"

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

        _atomic_write_json(
            self.phase_artifact_path(state.run_name, artifact.phase), artifact.to_dict()
        )
