"""Trial models and builders for phase-3 architecture sweep."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

from ...config import ExperimentConfig, ModelConfig
from ..phase_one import _feature_pipeline_for_trial
from ..phase_two import _scheduler_config_for_trial
from ..state import _serialize
from ..sweep_utils import sweep_seeds
from .constants import (
    PHASE_THREE_AST_DROPOUTS,
    PHASE_THREE_AST_HEADS,
    PHASE_THREE_AST_HIDDEN_SIZE,
    PHASE_THREE_AST_INTERMEDIATE_SIZE,
    PHASE_THREE_AST_NUM_HEADS,
    PHASE_THREE_AST_NUM_LAYERS,
    PHASE_THREE_AST_POSITIONAL_EMBEDDINGS,
    PHASE_THREE_CONVNEXT_KERNEL_SIZES,
    PHASE_THREE_CONVNEXT_STOCH_DEPTHS,
    PHASE_THREE_MLP_MIXER_DROPOUTS,
    PHASE_THREE_MLP_MIXER_HEAD_L2_NORM,
    PHASE_THREE_SEEDS,
    PHASE_THREE_SSAMBA_CLS,
    PHASE_THREE_SSAMBA_D_MODEL,
    PHASE_THREE_SSAMBA_D_STATE,
    PHASE_THREE_SSAMBA_EXPAND,
    PHASE_THREE_SSAMBA_NUM_LAYERS,
    PHASE_THREE_SSAMBA_POOLINGS,
    PHASE_THREE_SSAMBA_STRIDES_MS,
    PHASE_THREE_XLSTM_DIMS,
    PHASE_THREE_XLSTM_NUM_BLOCKS,
    PHASE_THREE_XLSTM_OUTPUTS,
    PHASE_THREE_XLSTM_STATE_RESETS,
)


def _sweep_seeds() -> tuple[int, ...]:
    """Return default seeds or a single-seed override for smoke runs."""
    return sweep_seeds(PHASE_THREE_SEEDS)


@dataclass(frozen=True, slots=True)
class PhaseThreeTrialSpec:
    """Describe one trial in the phase-3 architecture sweep."""

    trial_id: str
    family: str
    seed: int
    architecture_params: dict[str, Any]

    def to_config(
        self, base_config: ExperimentConfig, feature_name: str, optimizer_payload: Mapping[str, Any]
    ) -> ExperimentConfig:
        """Return the concrete config for this trial."""
        dataset = replace(
            base_config.dataset,
            train_split="train_small",
            valid_split="valid_small",
            test_split="test_small",
        )
        features = _feature_pipeline_for_trial(feature_name)
        optimizer = replace(
            base_config.optimizer, weight_decay=float(optimizer_payload["weight_decay"])
        )
        scheduler = _scheduler_config_for_trial(
            str(optimizer_payload["scheduler_name"]), total_epochs=base_config.training.epochs
        )
        model = self._build_model_config(base_config.model)
        phase_config = replace(base_config.phase, phase="phase_3")
        return replace(
            base_config,
            dataset=dataset,
            features=features,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            phase=phase_config,
            seed=self.seed,
        )

    def _build_model_config(self, base_model: ModelConfig) -> ModelConfig:
        """Build the model configuration for one trial."""
        params = self.architecture_params
        family = self.family
        if family == "ast":
            return replace(
                base_model,
                family=family,
                pretrained=False,
                dropout=float(params["dropout"]),
                ast_head=str(params["head"]),
                ast_positional_embedding=str(params["positional_embedding"]),
                ast_hidden_size=int(params["hidden_size"]),
                ast_num_hidden_layers=int(params["num_layers"]),
                ast_num_attention_heads=int(params["num_heads"]),
                ast_intermediate_size=int(params["intermediate_size"]),
            )
        if family == "convnext":
            return replace(
                base_model,
                family=family,
                pretrained=False,
                stochastic_depth=float(params["stochastic_depth"]),
            )
        if family == "ssamba":
            return replace(
                base_model,
                family=family,
                pretrained=False,
                ssamba_pooling=str(params["pooling"]),
                ssamba_use_cls=bool(params["use_cls"]),
                ssamba_stride_ms=int(params["stride_ms"]),
                ssamba_d_model=int(params["d_model"]),
                ssamba_d_state=int(params["d_state"]),
                ssamba_expand=int(params["expand"]),
                ssamba_num_layers=int(params["num_layers"]),
            )
        if family == "xlstm":
            return replace(
                base_model,
                family=family,
                pretrained=False,
                xlstm_dim=int(params["dimension"]),
                xlstm_num_blocks=int(params["num_blocks"]),
                xlstm_state_reset=bool(params["state_reset"]),
                xlstm_output_mode=str(params["output_mode"]),
            )
        if family == "mlp_mixer":
            return replace(
                base_model,
                family=family,
                pretrained=False,
                dropout=float(params["dropout"]),
                mlp_head_l2_norm=bool(params["head_l2_norm"]),
            )
        raise ValueError(f"Unsupported phase-3 model family '{family}'.")


def _build_ast_trials(seeds: tuple[int, ...]) -> tuple[PhaseThreeTrialSpec, ...]:
    """Return the 48 AST trial specifications for phase 3."""
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in seeds:
        for dropout in PHASE_THREE_AST_DROPOUTS:
            for head in PHASE_THREE_AST_HEADS:
                for positional_embedding in PHASE_THREE_AST_POSITIONAL_EMBEDDINGS:
                    trial_index += 1
                    trials.append(
                        PhaseThreeTrialSpec(
                            trial_id=(
                                f"trial_{trial_index:02d}_ast_{head}_{positional_embedding}_"
                                f"dropout_{dropout}_seed_{seed}"
                            ),
                            family="ast",
                            seed=seed,
                            architecture_params={
                                "dropout": dropout,
                                "head": head,
                                "positional_embedding": positional_embedding,
                                "hidden_size": PHASE_THREE_AST_HIDDEN_SIZE,
                                "num_layers": PHASE_THREE_AST_NUM_LAYERS,
                                "num_heads": PHASE_THREE_AST_NUM_HEADS,
                                "intermediate_size": PHASE_THREE_AST_INTERMEDIATE_SIZE,
                            },
                        )
                    )
    return tuple(trials)


def _build_convnext_trials(seeds: tuple[int, ...]) -> tuple[PhaseThreeTrialSpec, ...]:
    """Return the 12 ConvNeXT trial specifications for phase 3."""
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in seeds:
        for stochastic_depth in PHASE_THREE_CONVNEXT_STOCH_DEPTHS:
            for kernel_size in PHASE_THREE_CONVNEXT_KERNEL_SIZES:
                trial_index += 1
                trials.append(
                    PhaseThreeTrialSpec(
                        trial_id=(
                            f"trial_{trial_index:02d}_convnext_sd_{stochastic_depth}_"
                            f"kernel_{kernel_size}_seed_{seed}"
                        ),
                        family="convnext",
                        seed=seed,
                        architecture_params={
                            "stochastic_depth": stochastic_depth,
                            "kernel_size": kernel_size,
                        },
                    )
                )
    return tuple(trials)


def _build_ssamba_trials(seeds: tuple[int, ...]) -> tuple[PhaseThreeTrialSpec, ...]:
    """Return the 30 SSAMBA trial specifications for phase 3."""
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in seeds:
        for pooling in PHASE_THREE_SSAMBA_POOLINGS:
            for use_cls in PHASE_THREE_SSAMBA_CLS:
                for stride_ms in PHASE_THREE_SSAMBA_STRIDES_MS:
                    trial_index += 1
                    trials.append(
                        PhaseThreeTrialSpec(
                            trial_id=(
                                f"trial_{trial_index:02d}_ssamba_{pooling}_cls_{use_cls}_"
                                f"stride_{stride_ms}ms_seed_{seed}"
                            ),
                            family="ssamba",
                            seed=seed,
                            architecture_params={
                                "pooling": pooling,
                                "use_cls": use_cls,
                                "stride_ms": stride_ms,
                                "d_model": PHASE_THREE_SSAMBA_D_MODEL,
                                "d_state": PHASE_THREE_SSAMBA_D_STATE,
                                "expand": PHASE_THREE_SSAMBA_EXPAND,
                                "num_layers": PHASE_THREE_SSAMBA_NUM_LAYERS,
                            },
                        )
                    )
    return tuple(trials)


def _build_xlstm_trials(seeds: tuple[int, ...]) -> tuple[PhaseThreeTrialSpec, ...]:
    """Return the 12 XLSTM trial specifications for phase 3."""
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in seeds:
        for dimension in PHASE_THREE_XLSTM_DIMS:
            for state_reset in PHASE_THREE_XLSTM_STATE_RESETS:
                for output_mode in PHASE_THREE_XLSTM_OUTPUTS:
                    trial_index += 1
                    trials.append(
                        PhaseThreeTrialSpec(
                            trial_id=(
                                f"trial_{trial_index:02d}_xlstm_d_{dimension}_reset_{state_reset}_"
                                f"output_{output_mode}_seed_{seed}"
                            ),
                            family="xlstm",
                            seed=seed,
                            architecture_params={
                                "dimension": dimension,
                                "num_blocks": PHASE_THREE_XLSTM_NUM_BLOCKS,
                                "state_reset": state_reset,
                                "output_mode": output_mode,
                            },
                        )
                    )
    return tuple(trials)


def _build_mlp_mixer_trials(seeds: tuple[int, ...]) -> tuple[PhaseThreeTrialSpec, ...]:
    """Return the 12 MLP-Mixer trial specifications for phase 3."""
    trials: list[PhaseThreeTrialSpec] = []
    trial_index = 0
    for seed in seeds:
        for dropout in PHASE_THREE_MLP_MIXER_DROPOUTS:
            for head_l2_norm in PHASE_THREE_MLP_MIXER_HEAD_L2_NORM:
                trial_index += 1
                trials.append(
                    PhaseThreeTrialSpec(
                        trial_id=(
                            f"trial_{trial_index:02d}_mlp_mixer_dropout_{dropout}_"
                            f"head_l2_{head_l2_norm}_seed_{seed}"
                        ),
                        family="mlp_mixer",
                        seed=seed,
                        architecture_params={
                            "dropout": dropout,
                            "head_l2_norm": head_l2_norm,
                        },
                    )
                )
    return tuple(trials)


def build_phase_three_trials() -> tuple[PhaseThreeTrialSpec, ...]:
    """Return the 90 trial specifications for phase 3."""
    seeds = _sweep_seeds()
    return (
        *_build_ast_trials(seeds),
        *_build_convnext_trials(seeds),
        *_build_ssamba_trials(seeds),
        *_build_xlstm_trials(seeds),
        *_build_mlp_mixer_trials(seeds),
    )


@dataclass(frozen=True, slots=True)
class PhaseThreeTrialRecord:
    """Persisted record for one completed phase-3 trial."""

    trial_id: str
    """Unique trial identifier."""
    family: str
    """Model family name."""
    seed: int
    """Random seed used."""
    architecture_params: dict[str, Any]
    """Architecture hyperparameters used."""
    run_name: str
    """Child pipeline run name."""
    config_path: str
    """Path to the trial's config file."""
    child_state_path: str
    """Path to the child run's state."""
    validation_macro_f1: float
    """Validation macro-F1 score achieved."""
    completed_at: str
    """ISO-8601 timestamp when completed."""
    core_command_macro_f1: float | None = None
    """Core-command macro-F1 score when present in child metrics."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record."""
        return _serialize(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PhaseThreeTrialRecord":
        """Build a record from JSON data."""
        payload = dict(data)
        payload["seed"] = int(payload["seed"])
        payload["validation_macro_f1"] = float(payload["validation_macro_f1"])
        if payload.get("core_command_macro_f1") is not None:
            payload["core_command_macro_f1"] = float(payload["core_command_macro_f1"])
        return cls(**payload)
