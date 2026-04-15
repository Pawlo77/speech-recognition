"""Phase-4 held-out evaluation orchestration."""

from ..sweep_utils import run_subprocess_with_live_output
from .constants import (
    PHASE_FOUR_METHODS,
    PHASE_FOUR_STATE_SCHEMA_VERSION,
    PHASE_FOUR_STRICT_DROP_LIMIT,
    PHASE_FOUR_WARMUP_ITERATIONS,
)
from .runner import (
    PhaseFourSweepRunner,
    build_phase_four_command,
    build_phase_four_test_command,
)
from .state import PhaseFourSweepState
from .trials import (
    PhaseFourTrialRecord,
    PhaseFourTrialSpec,
    build_phase_four_trials,
)

__all__ = [
    "PHASE_FOUR_METHODS",
    "PHASE_FOUR_STATE_SCHEMA_VERSION",
    "PHASE_FOUR_STRICT_DROP_LIMIT",
    "PHASE_FOUR_WARMUP_ITERATIONS",
    "PhaseFourSweepRunner",
    "PhaseFourSweepState",
    "PhaseFourTrialRecord",
    "PhaseFourTrialSpec",
    "build_phase_four_command",
    "build_phase_four_test_command",
    "build_phase_four_trials",
    "run_subprocess_with_live_output",
]
