"""Persistent sweep state model for phase-2 hyperparameter sweep."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..state import _serialize
from ..sweep_utils import utc_now
from .constants import PHASE_TWO_STATE_SCHEMA_VERSION
from .trials import PhaseTwoTrialRecord


@dataclass(frozen=True, slots=True)
class PhaseTwoSweepState:
    """Persistent state for the phase-2 sweep."""

    schema_version: int = PHASE_TWO_STATE_SCHEMA_VERSION
    output_dir: str = ""
    phase_one_best_feature_path: str = ""
    completed_trials: dict[str, PhaseTwoTrialRecord] = field(default_factory=dict)
    best_trial_id: str | None = None
    best_validation_macro_f1: float | None = None
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        if self.schema_version != PHASE_TWO_STATE_SCHEMA_VERSION:
            raise ValueError("Unsupported phase-2 state schema version.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the sweep state."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseTwoSweepState":
        """Build phase-2 state from JSON."""
        payload = dict(data)
        payload["completed_trials"] = {
            key: PhaseTwoTrialRecord.from_dict(value)
            for key, value in payload.get("completed_trials", {}).items()
        }
        if payload.get("best_validation_macro_f1") is not None:
            payload["best_validation_macro_f1"] = float(payload["best_validation_macro_f1"])
        return cls(**payload)

    @classmethod
    def fresh(cls, output_dir: Path, phase_one_best_feature_path: Path) -> "PhaseTwoSweepState":
        """Create a new empty state for an output directory."""
        return cls(
            output_dir=str(output_dir), phase_one_best_feature_path=str(phase_one_best_feature_path)
        )
