"""Model registry and adapters for keyword-spotting backbones."""

import importlib
from collections.abc import Callable
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import init as nn_init

from ..config import ModelConfig

NUM_KAGGLE_CLASSES: Final[int] = 12
"""Number of classes in the 12-class KWS label space."""
DEFAULT_TARGET_FRAMES: Final[int] = 101
"""Temporal size for a 1-second baseline STFT-like feature tensor."""
DEFAULT_INPUT_BINS: Final[int] = 128
"""Default feature-bin count used by the feature extraction stack."""
SOURCE_LIBRARY_BY_FAMILY: Final[dict[str, tuple[str, ...]]] = {
    "ast": ("transformers",),
    "convnext": ("torchvision", "timm"),
    "ssamba": ("mamba-ssm",),
    "xlstm": ("xlstm",),
    "mlp_mixer": ("timm",),
}
"""Official source libraries for each registry family."""
MLP_MIXER_MODEL_NAME: Final[str] = "gmixer_24_224"
"""Official timm MLP-family model used for the mlp_mixer track."""


def _apply_kaiming_initialization(module: nn.Module) -> None:
    """Apply Kaiming normal initialization to trainable affine and convolution layers."""
    for child_module in module.modules():
        if isinstance(child_module, nn.Conv1d | nn.Conv2d | nn.Linear):
            nn_init.kaiming_normal_(child_module.weight, nonlinearity="relu")
            if child_module.bias is not None:
                nn_init.zeros_(child_module.bias)


class ASTBackboneWrapper(nn.Module):
    """Wrapper that normalizes AST logits extraction for transformers models."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.squeeze(1).transpose(1, 2)
        outputs = self.model(input_values=values)
        return cast(Tensor, outputs.logits)


class LogitNormalizationWrapper(nn.Module):
    """Apply optional L2 normalization to classifier logits."""

    def __init__(self, model: nn.Module, normalize_logits: bool) -> None:
        super().__init__()
        self.model = model
        self.normalize_logits = normalize_logits

    def forward(self, inputs: Tensor) -> Tensor:
        """Run a forward pass and optionally L2-normalize the output logits."""
        logits = self.model(inputs)
        if self.normalize_logits:
            logits = nn.functional.normalize(logits, p=2.0, dim=-1)
        return logits


class KWSModelAdapter(nn.Module):
    """Shared adapter interface for KWS backbones in the model registry."""

    def __init__(
        self,
        family: str,
        source_library: str,
        backbone: nn.Module,
        num_classes: int = NUM_KAGGLE_CLASSES,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        ast_uses_three_channels: bool = False,
    ) -> None:
        super().__init__()
        self.family = family
        self.source_library = source_library
        self.backbone = backbone
        self.num_classes = num_classes
        self.target_frames = target_frames
        self.ast_uses_three_channels = ast_uses_three_channels

    def validate_input_shape(self, input_tensor: Tensor) -> Tensor:
        """Normalize shape to [B, 1, bins, frames] and clamp temporal outliers."""
        if input_tensor.dim() == 2:
            normalized = input_tensor.unsqueeze(0).unsqueeze(0)
        elif input_tensor.dim() == 3:
            normalized = input_tensor.unsqueeze(1)
        elif input_tensor.dim() == 4:
            normalized = input_tensor
        else:
            raise ValueError("input_tensor must have shape [B, F, T] or [B, C, F, T].")

        if normalized.size(1) != 1:
            normalized = normalized.mean(dim=1, keepdim=True)

        frame_count = normalized.size(-1)
        if frame_count < self.target_frames:
            pad_width = self.target_frames - frame_count
            normalized = nn.functional.pad(normalized, (0, pad_width))
        elif frame_count > self.target_frames:
            normalized = normalized[..., : self.target_frames]

        return normalized.to(torch.float32)

    def forward_pass(self, input_tensor: Tensor) -> Tensor:
        """Run a forward pass and return logits with shape [B, num_classes]."""
        normalized = self.validate_input_shape(input_tensor)
        if self.family == "ast" and self.ast_uses_three_channels:
            normalized = normalized.repeat(1, 3, 1, 1)
        if self.family == "mlp_mixer":
            normalized = nn.functional.interpolate(
                normalized,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        logits = self.backbone(normalized)
        if logits.dim() != 2 or logits.size(-1) != self.num_classes:
            raise ValueError(
                f"{self.family} adapter must return [B, {self.num_classes}] logits, "
                f"got {tuple(logits.shape)}."
            )
        return logits

    def profile_efficiency(self, input_tensor: Tensor) -> dict[str, int]:
        """Profile MACs and parameter count with fvcore on normalized input."""
        from fvcore.nn import FlopCountAnalysis, parameter_count

        normalized = self.validate_input_shape(input_tensor)
        if self.family == "ast" and self.ast_uses_three_channels:
            normalized = normalized.repeat(1, 3, 1, 1)
        if self.family == "mlp_mixer":
            normalized = nn.functional.interpolate(
                normalized,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

        self.backbone.eval()
        flops = int(FlopCountAnalysis(self.backbone, normalized).total())
        params = int(sum(parameter_count(self.backbone).values()))
        return {
            "model_parameters_total": params,
            "model_macs_1sec": flops,
        }

    def forward(self, input_tensor: Tensor) -> Tensor:
        """Delegate ``nn.Module`` forward to ``forward_pass``."""
        return self.forward_pass(input_tensor)


ModelBuilder = Callable[[ModelConfig], tuple[nn.Module, str, bool]]


class ModelRegistry:
    """Composable model registry that builds per-family KWS adapters."""

    def __init__(self) -> None:
        self._builders: dict[str, ModelBuilder] = {
            "ast": _build_ast_backbone,
            "convnext": _build_convnext_backbone,
            "ssamba": _build_ssamba_backbone,
            "xlstm": _build_xlstm_backbone,
            "mlp_mixer": _build_mlp_mixer_backbone,
        }

    def supported_families(self) -> tuple[str, ...]:
        """Return registry family names in deterministic order."""
        return tuple(sorted(self._builders))

    def create(
        self,
        family: str | None = None,
        num_classes: int = NUM_KAGGLE_CLASSES,
        pretrained: bool = False,
        target_frames: int = DEFAULT_TARGET_FRAMES,
        model_config: ModelConfig | None = None,
    ) -> KWSModelAdapter:
        """Build a family adapter with the shared KWS interface."""
        if model_config is None:
            if family is None:
                raise ValueError("Either model_config or family must be provided.")
            model_config = ModelConfig(
                family=family,
                num_classes=num_classes,
                pretrained=pretrained,
            )

        family = model_config.family
        if family not in self._builders:
            raise ValueError(f"Unsupported model family '{family}'.")

        backbone, source_library, ast_three_channel = self._builders[family](model_config)
        return KWSModelAdapter(
            family=family,
            source_library=source_library,
            backbone=backbone,
            num_classes=model_config.num_classes,
            target_frames=target_frames,
            ast_uses_three_channels=ast_three_channel,
        )


def build_model_adapter(
    family: str,
    num_classes: int = NUM_KAGGLE_CLASSES,
    pretrained: bool = False,
    target_frames: int = DEFAULT_TARGET_FRAMES,
    model_config: ModelConfig | None = None,
) -> KWSModelAdapter:
    """Factory helper around ``ModelRegistry`` for direct adapter construction."""
    effective_config = model_config or ModelConfig(
        family=family,
        num_classes=num_classes,
        pretrained=pretrained,
    )
    return ModelRegistry().create(model_config=effective_config, target_frames=target_frames)


def _dependency_error(family: str, dependency: str, err: Exception) -> RuntimeError:
    """Helper to build a consistent error message when a
    family-specific dependency fails to import or use.
    """
    return RuntimeError(
        f"Unable to build '{family}' from official backend '{dependency}'. "
        f"Install/repair dependency '{dependency}'. Original error: {err!r}"
    )


def _build_ast_backbone(model_config: ModelConfig) -> tuple[nn.Module, str, bool]:
    """Build AST from Hugging Face Transformers only."""
    try:
        transformers_module = importlib.import_module("transformers")
        ast_config_cls = transformers_module.ASTConfig
        ast_for_audio_cls = transformers_module.ASTForAudioClassification
    except Exception as err:
        raise _dependency_error("ast", "transformers", err) from err

    config = ast_config_cls(
        num_labels=model_config.num_classes,
        hidden_dropout_prob=float(model_config.dropout),
        attention_probs_dropout_prob=float(model_config.dropout),
        hidden_size=int(model_config.ast_hidden_size),
        num_hidden_layers=int(model_config.ast_num_hidden_layers),
        num_attention_heads=int(model_config.ast_num_attention_heads),
        intermediate_size=int(model_config.ast_intermediate_size),
        max_length=DEFAULT_TARGET_FRAMES,
        num_mel_bins=DEFAULT_INPUT_BINS,
    )
    ast_model = ast_for_audio_cls(config)
    if model_config.ast_head == "mlp_256" and hasattr(ast_model, "classifier"):
        ast_model.classifier = nn.Sequential(
            nn.Linear(int(model_config.ast_hidden_size), 256),
            nn.GELU(),
            nn.Linear(256, model_config.num_classes),
        )
    model = ASTBackboneWrapper(ast_model)
    _apply_kaiming_initialization(model)
    return model, "transformers", False


def _build_convnext_backbone(model_config: ModelConfig) -> tuple[nn.Module, str, bool]:
    """Build ConvNeXt from torchvision first, then timm as official fallback."""
    try:
        torchvision_models = importlib.import_module("torchvision.models")
        convnext_tiny_fn = torchvision_models.convnext_tiny

        model = convnext_tiny_fn(weights=None, num_classes=model_config.num_classes)
        original = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        _apply_kaiming_initialization(model)
        return model, "torchvision", False
    except Exception as torchvision_err:
        try:
            timm = importlib.import_module("timm")

            model = timm.create_model(
                "convnext_tiny",
                pretrained=model_config.pretrained,
                in_chans=1,
                num_classes=model_config.num_classes,
                drop_path_rate=float(model_config.stochastic_depth),
            )
            _apply_kaiming_initialization(model)
            return model, "timm", False
        except Exception as timm_err:
            raise RuntimeError(
                "Unable to build 'convnext' from official backends ('torchvision', 'timm'). "
                f"torchvision error: {torchvision_err!r}; timm error: {timm_err!r}"
            ) from timm_err


def _build_ssamba_backbone(model_config: ModelConfig) -> tuple[nn.Module, str, bool]:
    """Build SSAMBA from mamba-ssm only."""
    try:
        mamba_module = importlib.import_module("mamba_ssm")
        mamba_cls = mamba_module.Mamba
    except Exception as err:
        raise _dependency_error("ssamba", "mamba-ssm", err) from err

    class MambaHead(nn.Module):
        """Thin wrapper around ``mamba_ssm.Mamba`` for logits output."""

        def __init__(self) -> None:
            super().__init__()
            d_model = int(model_config.ssamba_d_model)
            d_state = int(model_config.ssamba_d_state)
            d_conv = int(model_config.ssamba_d_conv)
            expand = int(model_config.ssamba_expand)
            num_layers = int(model_config.ssamba_num_layers)

            self.input_proj = nn.Linear(1, d_model)
            self.layers = nn.ModuleList(
                [
                    mamba_cls(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.pooling = model_config.ssamba_pooling
            self.use_cls = model_config.ssamba_use_cls
            self.stride_frames = 1 if model_config.ssamba_stride_ms == 10 else 2
            self.head = nn.Linear(d_model, model_config.num_classes)

        def forward(self, inputs: Tensor) -> Tensor:
            """Run a forward pass through the mamba backbone and return logits."""
            sequence = inputs.mean(dim=2).transpose(1, 2)
            hidden = self.input_proj(sequence)
            if self.stride_frames > 1:
                hidden = hidden[:, :: self.stride_frames, :]
            for layer in self.layers:
                hidden = layer(hidden)
            if self.use_cls:
                pooled = hidden[:, 0, :]
            elif self.pooling == "max":
                pooled = hidden.max(dim=1).values
            else:
                pooled = hidden.mean(dim=1)
            return self.head(pooled)

    model = MambaHead()
    _apply_kaiming_initialization(model)
    return model, "mamba-ssm", False


def _build_xlstm_backbone(model_config: ModelConfig) -> tuple[nn.Module, str, bool]:
    """Build xLSTM from the official ``xlstm`` package."""
    try:
        xlstm_module = importlib.import_module("xlstm")
        feed_forward_config_cls = xlstm_module.FeedForwardConfig
        mlstm_block_config_cls = xlstm_module.mLSTMBlockConfig
        mlstm_layer_config_cls = xlstm_module.mLSTMLayerConfig
        slstm_block_config_cls = xlstm_module.sLSTMBlockConfig
        slstm_layer_config_cls = xlstm_module.sLSTMLayerConfig
        xlstm_block_stack_cls = xlstm_module.xLSTMBlockStack
        xlstm_block_stack_config_cls = xlstm_module.xLSTMBlockStackConfig
    except Exception as err:
        raise _dependency_error("xlstm", "xlstm", err) from err

    class XLSTMHead(nn.Module):
        """Adapter that maps spectrogram sequences into xLSTM logits."""

        def __init__(self) -> None:
            super().__init__()
            dim = int(model_config.xlstm_dim)
            num_blocks = int(model_config.xlstm_num_blocks)
            slstm_at = [index for index in range(num_blocks) if index % 2 == 1]
            config = xlstm_block_stack_config_cls(
                mlstm_block=mlstm_block_config_cls(
                    mlstm=mlstm_layer_config_cls(
                        conv1d_kernel_size=4,
                        qkv_proj_blocksize=4,
                    )
                ),
                slstm_block=slstm_block_config_cls(
                    slstm=slstm_layer_config_cls(backend="vanilla"),
                    feedforward=feed_forward_config_cls(proj_factor=1.3, act_fn="gelu"),
                ),
                context_length=DEFAULT_TARGET_FRAMES,
                num_blocks=num_blocks,
                embedding_dim=dim,
                slstm_at=slstm_at,
            )
            self.input_proj = nn.Linear(1, dim)
            self.xlstm = xlstm_block_stack_cls(config)
            self.output_mode = model_config.xlstm_output_mode
            self.head = nn.Linear(dim, model_config.num_classes)

        def forward(self, inputs: Tensor) -> Tensor:
            """Run a forward pass through the xLSTM backbone and return logits."""
            sequence = inputs.mean(dim=2).transpose(1, 2)
            hidden = self.input_proj(sequence)
            hidden = self.xlstm(hidden)
            pooled = hidden[:, -1, :] if self.output_mode == "final" else hidden.mean(dim=1)
            return self.head(pooled)

    model = XLSTMHead()
    _apply_kaiming_initialization(model)
    return model, "xlstm", False


def _build_mlp_mixer_backbone(model_config: ModelConfig) -> tuple[nn.Module, str, bool]:
    """Build MLP-Mixer from timm only."""
    try:
        timm = importlib.import_module("timm")
    except Exception as err:
        raise _dependency_error("mlp_mixer", "timm", err) from err

    model = timm.create_model(
        MLP_MIXER_MODEL_NAME,
        pretrained=model_config.pretrained,
        in_chans=1,
        num_classes=model_config.num_classes,
        drop_rate=float(model_config.dropout),
    )
    model = LogitNormalizationWrapper(model, normalize_logits=model_config.mlp_head_l2_norm)
    _apply_kaiming_initialization(model)
    return model, "timm", False
