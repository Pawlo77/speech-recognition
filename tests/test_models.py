from collections.abc import Iterable

import pytest

pytest.importorskip("fvcore")
torch = pytest.importorskip("torch")

pytestmark = pytest.mark.filterwarnings(
    "ignore:`torch.jit.script` is deprecated.*:DeprecationWarning"
)

from speech_recognition.models import (  # noqa: E402
    DEFAULT_INPUT_BINS,
    DEFAULT_TARGET_FRAMES,
    NUM_KAGGLE_CLASSES,
    SOURCE_LIBRARY_BY_FAMILY,
    ModelRegistry,
    build_model_adapter,
)


def _families() -> Iterable[str]:
    return ("ast", "convnext", "ssamba", "xlstm", "mlp_mixer")


def _dummy_input(batch: int = 2, frames: int | None = None) -> torch.Tensor:
    if frames is None:
        frames = DEFAULT_TARGET_FRAMES
    return torch.randn(batch, 1, DEFAULT_INPUT_BINS, frames, dtype=torch.float32)


@pytest.mark.parametrize("family", _families())
def test_model_registry_forward_output_shape_is_batch_by_class_count(family: str) -> None:
    registry = ModelRegistry()
    adapter = registry.create(family=family, num_classes=NUM_KAGGLE_CLASSES, pretrained=False)

    logits = adapter.forward_pass(_dummy_input(batch=3, frames=73))

    assert logits.shape == (3, NUM_KAGGLE_CLASSES)
    assert logits.dtype == torch.float32


def test_model_registry_exposes_explicit_source_libraries() -> None:
    assert SOURCE_LIBRARY_BY_FAMILY["ast"] == ("transformers",)
    assert SOURCE_LIBRARY_BY_FAMILY["convnext"] == ("torchvision", "timm")
    assert SOURCE_LIBRARY_BY_FAMILY["ssamba"] == ("mamba-ssm",)
    assert SOURCE_LIBRARY_BY_FAMILY["xlstm"] == ("xlstm",)
    assert SOURCE_LIBRARY_BY_FAMILY["mlp_mixer"] == ("timm",)


def test_build_model_adapter_supports_ast_pretrained_channel_duplication() -> None:
    adapter = build_model_adapter(
        family="ast",
        num_classes=NUM_KAGGLE_CLASSES,
        pretrained=True,
    )

    logits = adapter.forward_pass(_dummy_input(batch=1, frames=88))

    assert logits.shape == (1, NUM_KAGGLE_CLASSES)


@pytest.mark.parametrize("family", _families())
def test_validate_input_shape_handles_short_and_long_outliers(family: str) -> None:
    registry = ModelRegistry()
    adapter = registry.create(family=family, num_classes=NUM_KAGGLE_CLASSES, pretrained=False)

    short = _dummy_input(batch=1, frames=37)
    long = _dummy_input(batch=1, frames=9518)

    short_validated = adapter.validate_input_shape(short)
    long_validated = adapter.validate_input_shape(long)

    assert short_validated.shape == (1, 1, DEFAULT_INPUT_BINS, DEFAULT_TARGET_FRAMES)
    assert long_validated.shape == (1, 1, DEFAULT_INPUT_BINS, DEFAULT_TARGET_FRAMES)


@pytest.mark.parametrize("family", _families())
def test_profile_efficiency_returns_valid_integers(family: str) -> None:
    registry = ModelRegistry()
    adapter = registry.create(family=family, num_classes=NUM_KAGGLE_CLASSES, pretrained=False)

    profile = adapter.profile_efficiency(_dummy_input(batch=1, frames=DEFAULT_TARGET_FRAMES))

    assert isinstance(profile["model_parameters_total"], int)
    assert isinstance(profile["model_macs_1sec"], int)
    assert profile["model_parameters_total"] > 0
    assert profile["model_macs_1sec"] > 0
