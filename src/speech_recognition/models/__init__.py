"""Model registry and adapters for speech-command backbones."""

from .registry import (
    DEFAULT_INPUT_BINS,
    DEFAULT_TARGET_FRAMES,
    NUM_KAGGLE_CLASSES,
    SOURCE_LIBRARY_BY_FAMILY,
    KWSModelAdapter,
    ModelRegistry,
    build_model_adapter,
)

__all__ = [
    "DEFAULT_INPUT_BINS",
    "DEFAULT_TARGET_FRAMES",
    "NUM_KAGGLE_CLASSES",
    "SOURCE_LIBRARY_BY_FAMILY",
    "KWSModelAdapter",
    "ModelRegistry",
    "build_model_adapter",
]
