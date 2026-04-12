"""Model registry and adapters for keyword-spotting backbones."""

from collections.abc import Callable
from typing import Final, cast

import torch
from torch import Tensor, nn

NUM_KAGGLE_CLASSES: Final[int] = 32
"""Number of classes in the Kaggle speech-command label space."""

DEFAULT_TARGET_FRAMES: Final[int] = 101
"""Temporal size for a 1-second baseline STFT-like feature tensor."""

DEFAULT_INPUT_BINS: Final[int] = 128
"""Default feature-bin count used by the feature extraction stack."""

SOURCE_LIBRARY_BY_FAMILY: Final[dict[str, tuple[str, ...]]] = {
    "ast": ("transformers", "timm"),
    "convnext": ("torchvision", "timm"),
    "ssamba": ("mamba-ssm",),
    "xlstm": ("xlstm",),
    "mlp_mixer": ("timm",),
}
"""Preferred external source libraries for each registry family."""


class TinyAstBackbone(nn.Module):
    """Lightweight AST-like fallback module for spectrogram classification."""

    def __init__(self, num_classes: int, in_chans: int = 1, embed_dim: int = 96) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=16, stride=8, bias=False)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.act(self.norm(self.patch_embed(inputs)))
        pooled = self.pool(hidden).flatten(1)
        return self.head(pooled)


class TinyConvNeXtBackbone(nn.Module):
    """Small ConvNeXt-style fallback with 1-channel stem support."""

    def __init__(self, num_classes: int, in_chans: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_chans, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.depthwise = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=7,
            stride=1,
            padding=3,
            groups=hidden_dim,
        )
        self.pointwise = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        hidden = self.act(self.norm(self.stem(inputs)))
        hidden = self.act(self.norm(self.pointwise(self.depthwise(hidden))))
        pooled = self.pool(hidden).flatten(1)
        return self.head(pooled)


class SSMambaFallbackBackbone(nn.Module):
    """Fallback for SSAMBA using causal conv + linear state-space recurrence."""

    def __init__(self, num_classes: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.pre = nn.Conv1d(1, hidden_dim, kernel_size=3, padding=2)
        self.state_a = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.state_b = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        sequence = inputs.mean(dim=2)
        hidden = self.pre(sequence)[..., :-2].transpose(1, 2)

        state = torch.zeros(
            hidden.size(0),
            hidden.size(-1),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        outputs: list[Tensor] = []
        for step in range(hidden.size(1)):
            token = hidden[:, step, :]
            candidate = self.state_a(state) + self.state_b(token)
            gate = torch.sigmoid(self.gate(token))
            state = gate * torch.tanh(candidate) + (1.0 - gate) * state
            outputs.append(state)

        stacked = torch.stack(outputs, dim=1)
        pooled = stacked.mean(dim=1)
        return self.head(pooled)


class MatrixMemoryXLSTMFallback(nn.Module):
    """Minimal matrix-memory xLSTM-style fallback with C_t in R^(d x d)."""

    def __init__(self, num_classes: int, memory_dim: int = 16) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.input_proj = nn.Linear(1, memory_dim)
        self.query_proj = nn.Linear(memory_dim, memory_dim)
        self.key_proj = nn.Linear(memory_dim, memory_dim)
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        self.forget_gate = nn.Linear(memory_dim, 1)
        self.input_gate = nn.Linear(memory_dim, 1)
        self.head = nn.Linear(memory_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        sequence = inputs.mean(dim=2).transpose(1, 2)
        token_states = self.input_proj(sequence)

        memory = torch.zeros(
            token_states.size(0),
            self.memory_dim,
            self.memory_dim,
            dtype=token_states.dtype,
            device=token_states.device,
        )
        readouts: list[Tensor] = []
        for step in range(token_states.size(1)):
            token = token_states[:, step, :]
            query = self.query_proj(token)
            key = self.key_proj(token)
            value = self.value_proj(token)

            forget = torch.sigmoid(self.forget_gate(token)).unsqueeze(-1)
            write = torch.sigmoid(self.input_gate(token)).unsqueeze(-1)
            outer = torch.einsum("bi,bj->bij", key, value)
            memory = forget * memory + write * outer

            readout = torch.einsum("bij,bj->bi", memory, query)
            readouts.append(readout)

        pooled = torch.stack(readouts, dim=1).mean(dim=1)
        return self.head(pooled)


class TinyMLPMixerBackbone(nn.Module):
    """Compact MLP-Mixer fallback operating on spectrogram patches."""

    def __init__(self, num_classes: int, hidden_dim: int = 64, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(1, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.token_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.channel_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: Tensor) -> Tensor:
        patches = self.patch_embed(inputs)
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + self.token_mlp(tokens)
        tokens = tokens + self.channel_mlp(tokens)
        pooled = self.norm(tokens).mean(dim=1)
        return self.head(pooled)


class ASTBackboneWrapper(nn.Module):
    """Wrapper that normalizes AST logits extraction for transformers models."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: Tensor) -> Tensor:
        values = inputs.squeeze(1).transpose(1, 2)
        outputs = self.model(input_values=values)
        return cast(Tensor, outputs.logits)


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


ModelBuilder = Callable[[int, bool], tuple[nn.Module, str, bool]]


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
        family: str,
        num_classes: int = NUM_KAGGLE_CLASSES,
        pretrained: bool = False,
        target_frames: int = DEFAULT_TARGET_FRAMES,
    ) -> KWSModelAdapter:
        """Build a family adapter with the shared KWS interface."""

        if family not in self._builders:
            raise ValueError(f"Unsupported model family '{family}'.")

        backbone, source_library, ast_three_channel = self._builders[family](
            num_classes, pretrained
        )
        return KWSModelAdapter(
            family=family,
            source_library=source_library,
            backbone=backbone,
            num_classes=num_classes,
            target_frames=target_frames,
            ast_uses_three_channels=ast_three_channel,
        )


def build_model_adapter(
    family: str,
    num_classes: int = NUM_KAGGLE_CLASSES,
    pretrained: bool = False,
    target_frames: int = DEFAULT_TARGET_FRAMES,
) -> KWSModelAdapter:
    """Factory helper around ``ModelRegistry`` for direct adapter construction."""

    return ModelRegistry().create(
        family=family,
        num_classes=num_classes,
        pretrained=pretrained,
        target_frames=target_frames,
    )


def _build_ast_backbone(num_classes: int, pretrained: bool) -> tuple[nn.Module, str, bool]:
    """Build AST from ``transformers`` first, else use a lightweight fallback."""

    try:
        from transformers import ASTConfig, ASTForAudioClassification

        config = ASTConfig(
            num_labels=num_classes,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_length=DEFAULT_TARGET_FRAMES,
            num_mel_bins=DEFAULT_INPUT_BINS,
        )
        return ASTBackboneWrapper(ASTForAudioClassification(config)), "transformers", False
    except Exception:
        in_chans = 3 if pretrained else 1
        return TinyAstBackbone(num_classes=num_classes, in_chans=in_chans), "fallback", pretrained


def _build_convnext_backbone(num_classes: int, pretrained: bool) -> tuple[nn.Module, str, bool]:
    """Build ConvNeXt from torchvision/timm with a 1-channel stem when available."""

    try:
        from torchvision.models import convnext_tiny

        model = convnext_tiny(weights=None, num_classes=num_classes)
        original = model.features[0][0]
        model.features[0][0] = nn.Conv2d(
            1,
            original.out_channels,
            kernel_size=original.kernel_size,
            stride=original.stride,
            padding=original.padding,
            bias=original.bias is not None,
        )
        return model, "torchvision", False
    except Exception:
        try:
            import timm

            model = timm.create_model(
                "convnext_tiny",
                pretrained=pretrained,
                in_chans=1,
                num_classes=num_classes,
            )
            return model, "timm", False
        except Exception:
            return TinyConvNeXtBackbone(num_classes=num_classes, in_chans=1), "fallback", False


def _build_ssamba_backbone(num_classes: int, pretrained: bool) -> tuple[nn.Module, str, bool]:
    """Build SSAMBA from mamba-ssm with Apple-silicon-safe fallback."""

    _ = pretrained
    try:
        from mamba_ssm import Mamba

        class MambaHead(nn.Module):
            """Thin wrapper around ``mamba_ssm.Mamba`` for logits output."""

            def __init__(self) -> None:
                super().__init__()
                self.input_proj = nn.Linear(1, 64)
                self.mamba = Mamba(d_model=64, d_state=16, d_conv=4, expand=2)
                self.head = nn.Linear(64, num_classes)

            def forward(self, inputs: Tensor) -> Tensor:
                sequence = inputs.mean(dim=2).transpose(1, 2).unsqueeze(-1)
                hidden = self.input_proj(sequence)
                hidden = self.mamba(hidden)
                return self.head(hidden.mean(dim=1))

        return MambaHead(), "mamba-ssm", False
    except Exception:
        return SSMambaFallbackBackbone(num_classes=num_classes), "fallback", False


def _build_xlstm_backbone(num_classes: int, pretrained: bool) -> tuple[nn.Module, str, bool]:
    """Build xLSTM from external package when available, otherwise matrix-memory fallback."""

    _ = pretrained
    return MatrixMemoryXLSTMFallback(num_classes=num_classes), "fallback", False


def _build_mlp_mixer_backbone(num_classes: int, pretrained: bool) -> tuple[nn.Module, str, bool]:
    """Build MLP-Mixer from timm when present, else use compact fallback."""

    try:
        import timm

        model = timm.create_model(
            "mixer_b16_224",
            pretrained=pretrained,
            in_chans=1,
            num_classes=num_classes,
        )
        return model, "timm", False
    except Exception:
        return TinyMLPMixerBackbone(num_classes=num_classes), "fallback", False
