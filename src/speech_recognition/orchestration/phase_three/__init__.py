"""Phase-3 architecture comparison sweep orchestration."""

from ..sweep_utils import run_subprocess_with_live_output
from .constants import PHASE_THREE_SEEDS, PHASE_THREE_STATE_SCHEMA_VERSION
from .runner import PhaseThreeSweepRunner, build_phase_three_command
from .state import PhaseThreeSweepState
from .trials import PhaseThreeTrialRecord, PhaseThreeTrialSpec, build_phase_three_trials

__all__ = [
    "PHASE_THREE_SEEDS",
    "PHASE_THREE_STATE_SCHEMA_VERSION",
    "PhaseThreeSweepRunner",
    "PhaseThreeSweepState",
    "PhaseThreeTrialRecord",
    "PhaseThreeTrialSpec",
    "build_phase_three_command",
    "build_phase_three_trials",
    "run_subprocess_with_live_output",
]
