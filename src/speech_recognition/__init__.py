"""Speech recognition package exports."""

import logging

from .config import (
    DEFAULT_SEEDS,
    CheckpointConfig,
    ConfigValidationError,
    DatasetConfig,
    ExperimentConfig,
    FeaturePipelineConfig,
    MLflowTrackingConfig,
    ModelConfig,
    OptimizerConfig,
    PhaseSelectionConfig,
    SchedulerConfig,
    TrainingControlConfig,
)
from .dataset import Sample, SpeechCommandsDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


__all__ = [
    "DEFAULT_SEEDS",
    "CheckpointConfig",
    "ConfigValidationError",
    "DatasetConfig",
    "ExperimentConfig",
    "FeaturePipelineConfig",
    "MLflowTrackingConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PhaseSelectionConfig",
    "Sample",
    "SchedulerConfig",
    "SpeechCommandsDataset",
    "TrainingControlConfig",
]
