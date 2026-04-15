"""Persistent sweep state model for phase-1 feature ablation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..sweep_utils import utc_now
from .constants import PHASE_ONE_STATE_SCHEMA_VERSION
from .trials import PhaseOneTrialRecord, _serialize


@dataclass(frozen=True, slots=True)
class PhaseOneSweepState:
    """Persistent state for the phase-1 sweep."""

    schema_version: int = PHASE_ONE_STATE_SCHEMA_VERSION
    """State file schema version."""
    output_dir: str = ""
    """Root output directory for the sweep."""
    completed_trials: dict[str, PhaseOneTrialRecord] = field(default_factory=dict)
    """Mapping of trial ID to completed trial records."""
    best_trial_id: str | None = None
    """Trial ID of the best-performing trial."""
    best_validation_macro_f1: float | None = None
    """Best validation macro-F1 score achieved."""
    created_at: str = field(default_factory=utc_now)
    """ISO-8601 timestamp when sweep state was created."""
    updated_at: str = field(default_factory=utc_now)
    """ISO-8601 timestamp when sweep state was last updated."""

    def __post_init__(self) -> None:
        if self.schema_version != PHASE_ONE_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported phase-1 state schema version.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the sweep state."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseOneSweepState":
        """Build phase-1 state from JSON."""
        payload = dict(data)
        payload["completed_trials"] = {
            key: PhaseOneTrialRecord.from_dict(value)
            for key, value in payload.get("completed_trials", {}).items()
        }
        if payload.get("best_validation_macro_f1") is not None:
            payload["best_validation_macro_f1"] = float(payload["best_validation_macro_f1"])
        return cls(**payload)

    @classmethod
    def fresh(cls, output_dir: Path) -> "PhaseOneSweepState":
        """Create a new empty state for an output directory."""
        return cls(output_dir=str(output_dir))
