"""Shared constants for phase-2 hyperparameter sweep."""

PHASE_TWO_WEIGHT_DECAYS: tuple[float, float] = (0.01, 0.1)
"""Weight decay values included in the phase-2 sweep."""
PHASE_TWO_SCHEDULERS: tuple[str, ...] = ("cosine_annealing_warmup", "reduce_on_plateau")
"""Scheduler families included in the phase-2 sweep."""
PHASE_TWO_PROXY_MODELS: tuple[str, ...] = ("convnext", "xlstm")
"""Proxy model families included in the phase-2 sweep."""
PHASE_TWO_SEEDS: tuple[int, int, int] = (0, 42, 2003)
"""Fixed seeds used for the phase-2 sweep grid."""
PHASE_TWO_STATE_SCHEMA_VERSION: int = 1
"""Schema version for the phase-2 sweep state file."""
