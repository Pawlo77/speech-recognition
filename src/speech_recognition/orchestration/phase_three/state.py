"""Persistent sweep state model for phase-3 architecture sweep."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..phase_one import _serialize
from ..sweep_utils import utc_now
from .constants import PHASE_THREE_STATE_SCHEMA_VERSION
from .trials import PhaseThreeTrialRecord


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
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

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
