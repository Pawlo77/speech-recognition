"""Shared constants for phase-3 architecture sweep."""

from ...config import DEFAULT_SEEDS

PHASE_THREE_STATE_SCHEMA_VERSION: int = 1
"""Schema version for the phase-3 sweep state file."""
PHASE_THREE_SEEDS: tuple[int, int, int] = DEFAULT_SEEDS
"""Fixed seeds used for the phase-3 sweep grid."""
PHASE_THREE_AST_DROPOUTS: tuple[float, float] = (0.1, 0.5)
"""AST dropout probability values to sweep."""
PHASE_THREE_AST_HEADS: tuple[str, str] = ("linear", "mlp_256")
"""AST classification head types to sweep."""
PHASE_THREE_AST_POSITIONAL_EMBEDDINGS: tuple[str, str] = ("interp", "learned")
"""AST positional embedding modes to sweep."""
PHASE_THREE_AST_HIDDEN_SIZE: int = 512
"""AST hidden size fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_AST_NUM_LAYERS: int = 8
"""AST depth fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_AST_NUM_HEADS: int = 8
"""AST attention heads fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_AST_INTERMEDIATE_SIZE: int = 2048
"""AST FFN width fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_CONVNEXT_STOCH_DEPTHS: tuple[float, float] = (0.0, 0.2)
"""ConvNeXt stochastic depth values to sweep."""
PHASE_THREE_CONVNEXT_KERNEL_SIZES: tuple[int, ...] = (7,)
"""ConvNeXt kernel sizes to sweep."""
PHASE_THREE_SSAMBA_POOLINGS: tuple[str, str] = ("mean", "max")
"""SSAMBA pooling modes to sweep."""
PHASE_THREE_SSAMBA_CLS: tuple[bool, bool] = (True, False)
"""SSAMBA CLS token configurations to sweep."""
PHASE_THREE_SSAMBA_STRIDES_MS: tuple[int, int] = (10, 5)
"""SSAMBA temporal stride values (ms) to sweep."""
PHASE_THREE_SSAMBA_D_MODEL: int = 768
"""SSAMBA hidden size fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_SSAMBA_D_STATE: int = 64
"""SSAMBA state size fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_SSAMBA_EXPAND: int = 2
"""SSAMBA expansion factor fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_SSAMBA_NUM_LAYERS: int = 6
"""SSAMBA depth fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_XLSTM_DIMS: tuple[int, int] = (704, 768)
"""xLSTM hidden/memory dimensions to sweep."""
PHASE_THREE_XLSTM_NUM_BLOCKS: int = 8
"""xLSTM stack depth fixed for ~25M-target phase-3 comparisons."""
PHASE_THREE_XLSTM_STATE_RESETS: tuple[bool, bool] = (True, False)
"""xLSTM state reset configurations to sweep."""
PHASE_THREE_XLSTM_OUTPUTS: tuple[str, str] = ("final", "mean")
"""xLSTM output reduction modes to sweep."""
PHASE_THREE_MLP_MIXER_DROPOUTS: tuple[float, float] = (0.0, 0.2)
"""MLP-Mixer dropout probability values to sweep."""
PHASE_THREE_MLP_MIXER_HEAD_L2_NORM: tuple[bool, bool] = (True, False)
"""MLP-Mixer L2-normalized head configurations to sweep."""
