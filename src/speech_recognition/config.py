"""Typed experiment configuration for the training pipeline."""

from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Self


class ConfigValidationError(ValueError):
    """Raised when a configuration value is invalid."""


DEFAULT_SEEDS: tuple[int, int, int] = (0, 42, 2003)
"""Default random seeds for reproducibility across all experiment phases."""
ALLOWED_TRAIN_SPLITS: set[str] = {"train_small", "train_extended"}
"""Allowed dataset splits for training."""
ALLOWED_VALID_SPLITS: set[str] = {"valid_small", "valid_extended"}
"""Allowed dataset splits for validation."""
ALLOWED_TEST_SPLITS: set[str] = {"test_small", "test_extended"}
"""Allowed dataset splits for testing."""
ALLOWED_FEATURE_PIPELINES: set[str] = {
    "mel_spectrogram",
    "high_temporal_mel",
    "mfcc",
    "pcen",
    "mel_specaugment",
}
"""Allowed feature pipeline identifiers used by the experiment stack."""

ALLOWED_MODEL_FAMILIES: set[str] = {"ast", "convnext", "ssamba", "xlstm", "mlp_mixer"}
"""Allowed model family identifiers used by the experiment stack."""
ALLOWED_SCHEDULERS: set[str] = {"cosine_annealing_warmup", "reduce_on_plateau"}
"""Allowed learning-rate scheduler identifiers."""
ALLOWED_PHASES: set[str] = {"phase_1", "phase_2", "phase_3", "phase_4"}
"""Allowed phase identifiers for the experiment funnel."""
ALLOWED_EVALUATION_STRATEGIES: set[str] = {
    "flat_multiclass",
    "sampling_control",
    "loss_reweighting",
    "two_stage_detector",
    "shared_two_head",
}
"""Allowed final held-out evaluation strategies."""


def _require(condition: bool, message: str) -> None:
    """Raise a validation error when a condition is false."""
    if not condition:
        raise ConfigValidationError(message)


def _require_str(name: str, value: str) -> None:
    """Validate that a value is a non-empty string."""
    _require(isinstance(value, str), f"{name} must be a string.")
    _require(bool(value.strip()), f"{name} must not be empty.")


def _require_int(name: str, value: int, minimum: int = 1) -> None:
    """Validate that a value is an integer and meets a minimum bound."""
    _require(isinstance(value, int), f"{name} must be an integer.")
    _require(value >= minimum, f"{name} must be greater than or equal to {minimum}.")


def _require_float(name: str, value: float, minimum: float | None = None) -> None:
    """Validate that a value is numeric and optionally above a minimum."""
    _require(isinstance(value, int | float), f"{name} must be a number.")
    if minimum is not None:
        _require(float(value) >= minimum, f"{name} must be greater than or equal to {minimum}.")


def _require_fraction(name: str, value: float) -> None:
    """Validate that a value lies in the closed interval [0, 1]."""
    _require_float(name, value)
    _require(0.0 <= float(value) <= 1.0, f"{name} must be between 0 and 1.")


def _serialize(value: Any) -> Any:
    """Convert nested dataclasses and tuples into JSON-friendly values."""
    if is_dataclass(value):
        return {field.name: _serialize(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    return value


def _extract_mapping(data: Mapping[str, Any] | Any, name: str) -> Mapping[str, Any]:
    """Validate and return a mapping for deserialization."""
    _require(isinstance(data, Mapping), f"{name} must be a mapping.")
    return data


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """Dataset split configuration for the speech commands corpus."""

    train_split: str = "train_small"
    """Training split name."""
    valid_split: str = "valid_small"
    """Validation split name."""
    test_split: str = "test_small"
    """Test split name."""
    root_dir: str = "data/kaggle_speech_commands"
    """Root directory for the speech commands dataset."""
    allow_missing_official_splits: bool = False
    """Allow missing official split files when loading the dataset."""

    def __post_init__(self) -> None:
        _require(
            self.train_split in ALLOWED_TRAIN_SPLITS,
            "train_split must be one of 'train_small' or 'train_extended'.",
        )
        _require(
            self.valid_split in ALLOWED_VALID_SPLITS,
            "valid_split must be one of 'valid_small' or 'valid_extended'.",
        )
        _require(
            self.test_split in ALLOWED_TEST_SPLITS,
            "test_split must be one of 'test_small' or 'test_extended'.",
        )
        _require_str("root_dir", self.root_dir)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the dataset config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a dataset config from a mapping."""
        mapping = _extract_mapping(data, name="DatasetConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class FeaturePipelineConfig:
    """Feature extraction pipeline used before model training."""

    name: str = "mel_spectrogram"
    """Feature pipeline identifier."""
    n_fft: int = 1024
    """FFT size used by the feature transform."""
    hop_length: int = 160
    """Hop length in samples."""
    n_mels: int = 128
    """Number of mel bins."""
    n_mfcc: int = 40
    """Number of MFCC coefficients."""
    pcen_smoothing: float = 0.1
    """PCEN smoothing constant."""
    specaugment: bool = False
    """Enable SpecAugment for the mel pipeline."""

    def __post_init__(self) -> None:
        _require(
            self.name in ALLOWED_FEATURE_PIPELINES,
            f"name must be one of {sorted(ALLOWED_FEATURE_PIPELINES)}.",
        )
        _require_int("n_fft", self.n_fft)
        _require_int("hop_length", self.hop_length)
        _require_int("n_mels", self.n_mels)
        _require_int("n_mfcc", self.n_mfcc)
        _require_fraction("pcen_smoothing", self.pcen_smoothing)

        if self.name == "high_temporal_mel":
            _require(self.n_fft == 512, "high_temporal_mel requires n_fft=512.")
            _require(self.hop_length == 80, "high_temporal_mel requires hop_length=80.")
        else:
            _require(self.n_fft == 1024, f"{self.name} requires n_fft=1024.")
            _require(self.hop_length == 160, f"{self.name} requires hop_length=160.")

        if self.name == "mfcc":
            _require(self.n_mfcc > 0, "mfcc requires a positive n_mfcc value.")
            _require(self.n_mfcc <= self.n_mels, "n_mfcc must not exceed n_mels.")
        else:
            _require(self.n_mfcc == 40, f"{self.name} requires n_mfcc=40.")

        if self.name == "pcen":
            _require(self.pcen_smoothing > 0.0, "pcen requires a positive smoothing constant.")
        else:
            _require(self.pcen_smoothing == 0.1, f"{self.name} requires pcen_smoothing=0.1.")

        if self.name == "mel_specaugment":
            _require(self.specaugment, "mel_specaugment requires specaugment=True.")
        else:
            _require(not self.specaugment, f"{self.name} does not use specaugment.")

        _require(self.n_mels == 128, "n_mels must be 128 for the current experiment stack.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the feature config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a feature pipeline config from a mapping."""
        mapping = _extract_mapping(data, name="FeaturePipelineConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model family and model-wide regularization options."""

    family: str = "convnext"
    """Model family identifier."""
    dropout: float = 0.1
    """Dropout probability."""
    pretrained: bool = False
    """Whether the model starts from pretrained weights."""
    stochastic_depth: float = 0.0
    """Stochastic depth probability."""
    ast_head: str = "linear"
    """AST classification head type (linear or mlp_256)."""
    ast_positional_embedding: str = "interp"
    """AST positional embedding mode (interp or learned)."""
    ast_hidden_size: int = 512
    """AST transformer hidden size."""
    ast_num_hidden_layers: int = 8
    """AST transformer depth."""
    ast_num_attention_heads: int = 8
    """AST attention head count."""
    ast_intermediate_size: int = 2048
    """AST feed-forward hidden size."""
    ssamba_pooling: str = "mean"
    """SSAMBA pooling mode (mean or max)."""
    ssamba_use_cls: bool = True
    """Whether SSAMBA uses a CLS token path."""
    ssamba_stride_ms: int = 10
    """SSAMBA temporal stride in milliseconds (10 or 5)."""
    ssamba_d_model: int = 768
    """SSAMBA hidden dimension."""
    ssamba_d_state: int = 64
    """SSAMBA state dimension."""
    ssamba_d_conv: int = 4
    """SSAMBA convolution kernel size."""
    ssamba_expand: int = 2
    """SSAMBA expansion factor."""
    ssamba_num_layers: int = 6
    """Number of stacked SSAMBA blocks."""
    xlstm_dim: int = 704
    """xLSTM hidden/memory dimension."""
    xlstm_num_blocks: int = 8
    """Number of xLSTM blocks in the stack."""
    xlstm_state_reset: bool = True
    """Whether xLSTM state resets between utterances."""
    xlstm_output_mode: str = "final"
    """xLSTM output reduction mode (final or mean)."""
    mlp_head_l2_norm: bool = True
    """Enable L2-normalized MLP-Mixer head logits."""
    num_classes: int = 12
    """Number of output classes."""

    def __post_init__(self) -> None:
        _require(
            self.family in ALLOWED_MODEL_FAMILIES,
            f"family must be one of {sorted(ALLOWED_MODEL_FAMILIES)}.",
        )
        _require_fraction("dropout", self.dropout)
        _require_fraction("stochastic_depth", self.stochastic_depth)
        _require(self.ast_head in {"linear", "mlp_256"}, "ast_head must be 'linear' or 'mlp_256'.")
        _require(
            self.ast_positional_embedding in {"interp", "learned"},
            "ast_positional_embedding must be 'interp' or 'learned'.",
        )
        _require(
            self.ast_hidden_size in {384, 512, 576, 640, 768},
            "ast_hidden_size must be one of {384, 512, 576, 640, 768}.",
        )
        _require(
            self.ast_num_hidden_layers in {8, 10, 12},
            "ast_num_hidden_layers must be 8, 10, or 12.",
        )
        _require(
            self.ast_num_attention_heads in {6, 8, 9, 10, 12},
            "ast_num_attention_heads must be one of {6, 8, 9, 10, 12}.",
        )
        _require(
            self.ast_intermediate_size in {1536, 2048, 2304, 2560, 3072},
            "ast_intermediate_size must be one of {1536, 2048, 2304, 2560, 3072}.",
        )
        _require(
            self.ast_hidden_size % self.ast_num_attention_heads == 0,
            "ast_hidden_size must be divisible by ast_num_attention_heads.",
        )
        _require(self.ssamba_pooling in {"mean", "max"}, "ssamba_pooling must be 'mean' or 'max'.")
        _require(self.ssamba_stride_ms in {5, 10}, "ssamba_stride_ms must be 5 or 10.")
        _require(
            self.ssamba_d_model in {64, 128, 256, 384, 512, 640, 768},
            "ssamba_d_model must be one of {64, 128, 256, 384, 512, 640, 768}.",
        )
        _require(
            self.ssamba_d_state in {16, 32, 64, 128},
            "ssamba_d_state must be 16, 32, 64, or 128.",
        )
        _require(self.ssamba_d_conv in {2, 3, 4, 5, 7}, "ssamba_d_conv must be 2, 3, 4, 5, or 7.")
        _require(self.ssamba_expand in {1, 2, 4}, "ssamba_expand must be 1, 2, or 4.")
        _require(
            self.ssamba_num_layers in {1, 2, 4, 6, 8, 10, 12},
            "ssamba_num_layers must be 1, 2, 4, 6, 8, 10, or 12.",
        )
        _require(
            self.xlstm_dim in {32, 64, 128, 256, 384, 512, 704, 768, 896, 1024},
            "xlstm_dim must be one of {32, 64, 128, 256, 384, 512, 704, 768, 896, 1024}.",
        )
        _require(
            self.xlstm_num_blocks in {1, 2, 4, 6, 8, 10, 12},
            "xlstm_num_blocks must be 1, 2, 4, 6, 8, 10, or 12.",
        )
        _require(
            self.xlstm_output_mode in {"final", "mean"},
            "xlstm_output_mode must be 'final' or 'mean'.",
        )
        _require_int("num_classes", self.num_classes)
        _require(2 <= self.num_classes <= 12, "num_classes must be between 2 and 12.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the model config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a model config from a mapping."""
        mapping = _extract_mapping(data, name="ModelConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """Optimizer configuration."""

    name: str = "adamw"
    """Optimizer name."""
    learning_rate: float = 3e-4
    """Initial learning rate."""
    weight_decay: float = 1e-4
    """Weight decay coefficient."""
    betas: tuple[float, float] = (0.9, 0.999)
    """AdamW beta coefficients."""
    eps: float = 1e-8
    """Numerical stability epsilon."""

    def __post_init__(self) -> None:
        _require(self.name == "adamw", "optimizer name must be 'adamw'.")
        _require_float("learning_rate", self.learning_rate, minimum=0.0)
        _require(float(self.learning_rate) > 0.0, "learning_rate must be greater than 0.")
        _require_float("weight_decay", self.weight_decay, minimum=0.0)
        _require(
            isinstance(self.betas, tuple) and len(self.betas) == 2,
            "betas must contain two values.",
        )
        _require_fraction("betas[0]", self.betas[0])
        _require_fraction("betas[1]", self.betas[1])
        _require(0.0 < self.betas[0] < 1.0, "betas[0] must be between 0 and 1.")
        _require(0.0 < self.betas[1] < 1.0, "betas[1] must be between 0 and 1.")
        _require_float("eps", self.eps, minimum=0.0)
        _require(float(self.eps) > 0.0, "eps must be greater than 0.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the optimizer config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build an optimizer config from a mapping."""
        mapping = _extract_mapping(data, name="OptimizerConfig")
        kwargs = dict(mapping)
        if "betas" in kwargs:
            kwargs["betas"] = tuple(kwargs["betas"])
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Learning-rate scheduler configuration."""

    name: str = "cosine_annealing_warmup"
    """Scheduler name."""
    warmup_epochs: int = 5
    """Number of warmup epochs."""
    total_epochs: int = 200
    """Total number of training epochs."""
    min_learning_rate: float = 1e-6
    """Lower bound for the learning rate."""
    plateau_factor: float = 0.5
    """Reduction factor for plateau scheduling."""
    plateau_patience: int = 5
    """Number of epochs to wait before plateau reduction."""

    def __post_init__(self) -> None:
        _require(
            self.name in ALLOWED_SCHEDULERS,
            f"name must be one of {sorted(ALLOWED_SCHEDULERS)}.",
        )
        _require_int("warmup_epochs", self.warmup_epochs, minimum=0)
        _require_int("total_epochs", self.total_epochs)
        _require(
            float(self.warmup_epochs) < float(self.total_epochs),
            "warmup_epochs must be smaller than total_epochs.",
        )
        _require_float("min_learning_rate", self.min_learning_rate, minimum=0.0)
        _require_fraction("plateau_factor", self.plateau_factor)
        _require(0.0 < self.plateau_factor < 1.0, "plateau_factor must be between 0 and 1.")
        _require_int("plateau_patience", self.plateau_patience)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the scheduler config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a scheduler config from a mapping."""
        mapping = _extract_mapping(data, name="SchedulerConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class TrainingControlConfig:
    """Training loop controls that are independent from the optimizer."""

    epochs: int = 200
    """Number of training epochs."""
    batch_size: int = 32
    """Mini-batch size."""
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients."""
    num_workers: int = 4
    """DataLoader worker count."""
    log_every_n_steps: int = 100
    """Logging interval in steps."""
    validate_every_n_epochs: int = 1
    """Validation interval in epochs."""
    early_stopping_patience: int = 10
    """Patience for early stopping."""
    max_grad_norm: float | None = 1.0
    """Optional gradient clipping threshold."""
    deterministic: bool = True
    """Enable deterministic execution where possible."""
    use_mixed_precision: bool = True
    """Enable mixed precision when the backend supports it (GPU throughput optimization)."""

    def __post_init__(self) -> None:
        _require_int("epochs", self.epochs)
        _require_int("batch_size", self.batch_size)
        _require_int("gradient_accumulation_steps", self.gradient_accumulation_steps)
        _require_int("num_workers", self.num_workers, minimum=0)
        _require_int("log_every_n_steps", self.log_every_n_steps)
        _require_int("validate_every_n_epochs", self.validate_every_n_epochs)
        _require_int("early_stopping_patience", self.early_stopping_patience, minimum=0)
        if self.max_grad_norm is not None:
            _require_float("max_grad_norm", self.max_grad_norm, minimum=0.0)
        _require(isinstance(self.use_mixed_precision, bool), "use_mixed_precision must be a bool.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the training config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a training control config from a mapping."""
        mapping = _extract_mapping(data, name="TrainingControlConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class CheckpointConfig:
    """Checkpoint storage and selection policy."""

    directory: str = "outputs/checkpoints"
    """Directory used for checkpoint files."""
    save_best: bool = True
    """Persist the best checkpoint."""
    save_last: bool = True
    """Persist the latest checkpoint."""
    monitor: str = "macro_f1"
    """Metric used for checkpoint selection."""
    mode: str = "max"
    """Optimization direction for checkpoint selection."""
    keep_last_n: int = 2
    """Number of recent checkpoints to retain."""

    def __post_init__(self) -> None:
        _require_str("directory", self.directory)
        _require_str("monitor", self.monitor)
        _require(self.mode in {"max", "min"}, "mode must be either 'max' or 'min'.")
        _require_int("keep_last_n", self.keep_last_n)
        _require(
            self.save_best or self.save_last,
            "at least one checkpoint policy must be enabled.",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the checkpoint config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a checkpoint config from a mapping."""
        mapping = _extract_mapping(data, name="CheckpointConfig")
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class MLflowTrackingConfig:
    """Local MLflow tracking configuration."""

    enabled: bool = True
    """Enable MLflow tracking."""
    tracking_uri: str = "sqlite:///mlruns.db"
    """MLflow tracking URI (SQLite backend for SQL-based trace metrics support)."""
    experiment_name: str = "speech-recognition"
    """MLflow experiment name."""
    run_name: str | None = None
    """Optional MLflow run name."""
    log_params: bool = True
    """Log hyperparameters to MLflow."""
    log_metrics: bool = True
    """Log metrics to MLflow."""
    log_artifacts: bool = True
    """Log artifacts to MLflow."""
    retain_local_checkpoints: bool = False
    """Keep local checkpoint files after they are uploaded to MLflow artifacts."""

    def __post_init__(self) -> None:
        if self.enabled:
            _require_str("tracking_uri", self.tracking_uri)
        _require_str("experiment_name", self.experiment_name)
        if self.run_name is not None:
            _require_str("run_name", self.run_name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the MLflow config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build an MLflow tracking config from a mapping."""
        mapping = _extract_mapping(data, name="MLflowTrackingConfig")
        if mapping.get("tracking_uri") == "mlruns":
            mapping = {**mapping, "tracking_uri": "sqlite:///mlruns.db"}
        return cls(**dict(mapping))


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    """Final held-out evaluation strategy configuration."""

    strategy: str = "flat_multiclass"
    """Selected non-command handling strategy."""
    backbone_ids: tuple[str, ...] = ()
    """Frozen phase-3 backbone identifiers used for evaluation."""
    ensemble_members: tuple[str, ...] = ()
    """Backbone identifiers used in the current single-model or ensemble trial."""
    unknown_prior: float = 0.25
    """Target prior for the unknown class under sampling control."""
    silence_prior: float = 0.15
    """Target prior for the silence class under sampling control."""
    command_prior: float = 0.06
    """Target prior for each command class under sampling control."""
    warmup_iterations: int = 50
    """Warmup iterations excluded from inference-latency timing."""
    max_core_command_f1_drop: float = 0.01
    """Maximum tolerated core-command macro-F1 drop relative to the Phase 3 baseline."""

    def __post_init__(self) -> None:
        _require(
            self.strategy in ALLOWED_EVALUATION_STRATEGIES,
            f"strategy must be one of {sorted(ALLOWED_EVALUATION_STRATEGIES)}.",
        )
        _require(isinstance(self.backbone_ids, tuple), "backbone_ids must be a tuple.")
        _require(isinstance(self.ensemble_members, tuple), "ensemble_members must be a tuple.")
        _require_float("unknown_prior", self.unknown_prior, minimum=0.0)
        _require_float("silence_prior", self.silence_prior, minimum=0.0)
        _require_float("command_prior", self.command_prior, minimum=0.0)
        _require_int("warmup_iterations", self.warmup_iterations, minimum=0)
        _require_float("max_core_command_f1_drop", self.max_core_command_f1_drop, minimum=0.0)
        if self.backbone_ids:
            _require(
                len(set(self.backbone_ids)) == len(self.backbone_ids),
                "backbone_ids must not contain duplicates.",
            )
        if self.ensemble_members:
            known_backbone_ids = self.backbone_ids or self.ensemble_members
            _require(
                set(self.ensemble_members).issubset(set(known_backbone_ids)),
                "ensemble_members must reference known backbone ids.",
            )
        _require(
            0.0 <= self.unknown_prior <= 1.0,
            "unknown_prior must be between 0 and 1.",
        )
        _require(
            0.0 <= self.silence_prior <= 1.0,
            "silence_prior must be between 0 and 1.",
        )
        _require(
            0.0 <= self.command_prior <= 1.0,
            "command_prior must be between 0 and 1.",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the evaluation config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build an evaluation config from a mapping."""
        mapping = _extract_mapping(data, name="EvaluationConfig")
        kwargs = dict(mapping)
        if "backbone_ids" in kwargs:
            kwargs["backbone_ids"] = tuple(kwargs["backbone_ids"])
        if "ensemble_members" in kwargs:
            kwargs["ensemble_members"] = tuple(kwargs["ensemble_members"])
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class PhaseSelectionConfig:
    """Select which experimental phase to execute."""

    phase: str = "phase_1"
    """Selected experiment phase."""
    available_phases: tuple[str, ...] = ("phase_1", "phase_2", "phase_3", "phase_4")
    """Phases that the runner can execute."""

    def __post_init__(self) -> None:
        _require(
            self.phase in ALLOWED_PHASES,
            f"phase must be one of {sorted(ALLOWED_PHASES)}.",
        )
        _require(isinstance(self.available_phases, tuple), "available_phases must be a tuple.")
        _require(bool(self.available_phases), "available_phases must not be empty.")
        _require(
            len(set(self.available_phases)) == len(self.available_phases),
            "available_phases must not contain duplicates.",
        )
        _require(self.phase in self.available_phases, "phase must be included in available_phases.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the phase config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a phase selection config from a mapping."""
        mapping = _extract_mapping(data, name="PhaseSelectionConfig")
        kwargs = dict(mapping)
        if "available_phases" in kwargs:
            kwargs["available_phases"] = tuple(kwargs["available_phases"])
        return cls(**kwargs)


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Bundle the full experiment stack into one frozen object."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    """Dataset configuration."""
    features: FeaturePipelineConfig = field(default_factory=FeaturePipelineConfig)
    """Feature pipeline configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    """Model configuration."""
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """Optimizer configuration."""
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """Scheduler configuration."""
    training: TrainingControlConfig = field(default_factory=TrainingControlConfig)
    """Training control configuration."""
    checkpointing: CheckpointConfig = field(default_factory=CheckpointConfig)
    """Checkpoint configuration."""
    mlflow: MLflowTrackingConfig = field(default_factory=MLflowTrackingConfig)
    """MLflow tracking configuration."""
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    """Final held-out evaluation configuration."""
    phase: PhaseSelectionConfig = field(default_factory=PhaseSelectionConfig)
    """Phase selection configuration."""
    seed: int = DEFAULT_SEEDS[0]
    """Active reproducibility seed for the current run."""
    seeds: tuple[int, int, int] = DEFAULT_SEEDS
    """Fixed reproducibility seeds."""

    def __post_init__(self) -> None:
        _require(self.seed in DEFAULT_SEEDS, f"seed must be one of {DEFAULT_SEEDS}.")
        _require(self.seeds == DEFAULT_SEEDS, "seeds must remain fixed at (0, 42, 2003).")
        _require(
            self.scheduler.total_epochs == self.training.epochs,
            "scheduler.total_epochs must match training.epochs.",
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the full experiment config."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Build a full experiment config from a mapping."""
        mapping = _extract_mapping(data, name="ExperimentConfig")
        kwargs = dict(mapping)
        kwargs["dataset"] = DatasetConfig.from_dict(kwargs["dataset"])
        kwargs["features"] = FeaturePipelineConfig.from_dict(kwargs["features"])
        kwargs["model"] = ModelConfig.from_dict(kwargs["model"])
        kwargs["optimizer"] = OptimizerConfig.from_dict(kwargs["optimizer"])
        kwargs["scheduler"] = SchedulerConfig.from_dict(kwargs["scheduler"])
        kwargs["training"] = TrainingControlConfig.from_dict(kwargs["training"])
        kwargs["checkpointing"] = CheckpointConfig.from_dict(kwargs["checkpointing"])
        kwargs["mlflow"] = MLflowTrackingConfig.from_dict(kwargs["mlflow"])
        kwargs["evaluation"] = EvaluationConfig.from_dict(kwargs["evaluation"])
        kwargs["phase"] = PhaseSelectionConfig.from_dict(kwargs["phase"])
        if "seeds" in kwargs:
            kwargs["seeds"] = tuple(kwargs["seeds"])
        return cls(**kwargs)
