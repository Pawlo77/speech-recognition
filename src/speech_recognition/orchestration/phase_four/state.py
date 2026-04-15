"""Persistent sweep state model for phase-4 held-out evaluation."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..phase_one import _serialize
from ..sweep_utils import utc_now
from .constants import PHASE_FOUR_STATE_SCHEMA_VERSION
from .trials import PhaseFourTrialRecord


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
    created_at: str = field(default_factory=utc_now)
    """ISO-8601 timestamp when sweep state was created."""
    updated_at: str = field(default_factory=utc_now)
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
