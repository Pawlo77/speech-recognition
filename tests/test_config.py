import json

import pytest

from speech_recognition.config import (
    CheckpointConfig,
    ConfigValidationError,
    DatasetConfig,
    EvaluationConfig,
    ExperimentConfig,
    FeaturePipelineConfig,
    MLflowTrackingConfig,
    ModelConfig,
    OptimizerConfig,
    PhaseSelectionConfig,
    SchedulerConfig,
    TrainingControlConfig,
)


def test_experiment_config_defaults_are_explicit() -> None:
    config = ExperimentConfig()

    assert config.dataset == DatasetConfig()
    assert config.features == FeaturePipelineConfig()
    assert config.model == ModelConfig()
    assert config.optimizer == OptimizerConfig()
    assert config.scheduler == SchedulerConfig()
    assert config.training == TrainingControlConfig()
    assert config.checkpointing == CheckpointConfig()
    assert config.mlflow == MLflowTrackingConfig()
    assert config.evaluation == EvaluationConfig()
    assert config.phase == PhaseSelectionConfig()
    assert config.seed == 0
    assert config.seeds == (0, 42, 2003)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: DatasetConfig(train_split="invalid"), "train_split"),
        (lambda: FeaturePipelineConfig(name="mfcc", n_mfcc=0), "n_mfcc"),
        (lambda: ModelConfig(family="transformer"), "family"),
        (lambda: ModelConfig(num_classes=35), "num_classes"),
        (lambda: OptimizerConfig(name="sgd"), "adamw"),
        (lambda: SchedulerConfig(warmup_epochs=60, total_epochs=60), "warmup_epochs"),
        (lambda: CheckpointConfig(save_best=False, save_last=False), "checkpoint"),
        (lambda: EvaluationConfig(strategy="bogus"), "strategy"),
        (lambda: ExperimentConfig(seed=7), "seed"),
        (lambda: PhaseSelectionConfig(phase="phase_5"), "phase"),
        (lambda: ExperimentConfig(seeds=(0, 42, 42)), "seeds"),
    ],
)
def test_invalid_values_raise_clear_errors(factory, message: str) -> None:
    with pytest.raises(ConfigValidationError, match=message):
        factory()


def test_experiment_config_round_trip_serializes_to_json() -> None:
    config = ExperimentConfig(
        features=FeaturePipelineConfig(name="mfcc", n_fft=1024, hop_length=160, n_mfcc=40),
        model=ModelConfig(family="xlstm", dropout=0.2, pretrained=False),
        evaluation=EvaluationConfig(
            strategy="sampling_control",
            backbone_ids=("backbone_a", "backbone_b", "backbone_c"),
            ensemble_members=("backbone_a", "backbone_b"),
        ),
        training=TrainingControlConfig(epochs=60, use_mixed_precision=True),
        phase=PhaseSelectionConfig(phase="phase_3"),
        checkpointing=CheckpointConfig(keep_last_n=3),
    )

    payload = config.to_dict()
    encoded = json.dumps(payload)
    restored = ExperimentConfig.from_dict(json.loads(encoded))

    assert restored == config
    assert payload["seeds"] == [0, 42, 2003]
    assert payload["phase"]["available_phases"] == ["phase_1", "phase_2", "phase_3", "phase_4"]
