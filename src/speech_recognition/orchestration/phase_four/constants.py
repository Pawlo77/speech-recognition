"""Shared constants for phase-4 held-out evaluation."""

PHASE_FOUR_STATE_SCHEMA_VERSION: int = 1
"""Schema version for the phase-4 sweep state file."""
PHASE_FOUR_METHODS: tuple[str, ...] = (
    "flat_multiclass",
    "sampling_control",
    "loss_reweighting",
    "two_stage_detector",
    "shared_two_head",
)
"""Supported final non-command handling strategies."""
PHASE_FOUR_STRICT_DROP_LIMIT: float = 0.01
"""Maximum tolerated core-command macro-F1 drop relative to the Phase 3 baseline."""
PHASE_FOUR_WARMUP_ITERATIONS: int = 50
"""Warmup iterations excluded from latency measurement."""
