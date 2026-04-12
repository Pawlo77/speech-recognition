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


@dataclass(frozen=True, slots=True)
class PhaseArtifact:
    """Serialized output for one completed phase."""

    phase: str
    artifact_path: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    depends_on: tuple[str, ...] = ()

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

    schema_version: int = 1
    run_name: str = "default"
    run_root: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    completed_phases: tuple[str, ...] = ()
    phase_artifacts: dict[str, PhaseArtifact] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        if self.schema_version != 1:
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

    def run_dir(self, run_name: str) -> Path:
        """Return the directory that stores one run."""

        return self.base_dir / run_name

    def state_path(self, run_name: str) -> Path:
        """Return the path of the state file for a run."""

        return self.run_dir(run_name) / "state.json"

    def phase_artifact_path(self, run_name: str, phase: str) -> Path:
        """Return the artifact path for one phase."""

        return self.run_dir(run_name) / f"{phase}.json"

    def load(self, run_name: str) -> PipelineState:
        """Load state from disk."""

        state_path = self.state_path(run_name)
        if not state_path.exists():
            raise FileNotFoundError(state_path)
        return PipelineState.from_dict(_read_json(state_path))

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
        """Persist state to disk."""

        _atomic_write_json(self.state_path(state.run_name), state.to_dict())

    def save_artifact(self, state: PipelineState, artifact: PhaseArtifact) -> None:
        """Persist a phase artifact to disk."""

        _atomic_write_json(
            self.phase_artifact_path(state.run_name, artifact.phase), artifact.to_dict()
        )
