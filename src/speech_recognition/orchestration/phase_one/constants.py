"""Shared constants for phase-1 feature ablation sweep."""

PHASE_ONE_FEATURES: tuple[str, ...] = (
    "mel_spectrogram",
    "high_temporal_mel",
    "mfcc",
    "pcen",
    "mel_specaugment",
)
"""Feature families included in the phase-1 sweep."""
PHASE_ONE_PROXY_MODELS: tuple[str, ...] = ("convnext", "xlstm")
"""Proxy model families included in the phase-1 sweep."""
PHASE_ONE_SEEDS: tuple[int, int, int] = (0, 42, 2003)
"""Fixed seeds used for the phase-1 sweep grid."""
PHASE_ONE_STATE_SCHEMA_VERSION: int = 1
"""Schema version for the phase-1 sweep state file."""
