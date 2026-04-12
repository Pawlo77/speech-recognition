"""Deterministic phase services used by the pipeline runner."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from .state import PHASE_ORDER, PhaseArtifact, PipelineState


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Provide config, state, and upstream artifacts to a phase service."""

    config: ExperimentConfig
    run_name: str
    run_root: Path
    state: PipelineState

    @property
    def upstream_artifacts(self) -> Mapping[str, PhaseArtifact]:
        """Return completed phase artifacts keyed by phase name."""

        return self.state.phase_artifacts


@dataclass(frozen=True, slots=True)
class PhaseService:
    """Base class for deterministic phase services."""

    phase: str
    dependencies: tuple[str, ...]

    def execute(self, context: PipelineContext) -> dict[str, Any]:
        """Execute the phase and return a JSON-serializable payload."""

        raise NotImplementedError

    def _upstream_payload(self, context: PipelineContext) -> dict[str, Any]:
        """Return upstream phase outputs for downstream consumption."""

        return {
            name: artifact.output_data
            for name, artifact in context.upstream_artifacts.items()
            if name in self.dependencies or name in PHASE_ORDER
        }


@dataclass(frozen=True, slots=True)
class PhaseOneService(PhaseService):
    """Summarize the feature-strategy configuration."""

    phase: str = "phase-1"
    dependencies: tuple[str, ...] = ()

    def execute(self, context: PipelineContext) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "dataset": context.config.dataset.to_dict(),
            "feature_pipeline": context.config.features.to_dict(),
            "selected_proxy_models": ["convnext", "xlstm"],
            "upstream": self._upstream_payload(context),
        }


@dataclass(frozen=True, slots=True)
class PhaseTwoService(PhaseService):
    """Summarize the global optimization configuration."""

    phase: str = "phase-2"
    dependencies: tuple[str, ...] = ("phase-1",)

    def execute(self, context: PipelineContext) -> dict[str, Any]:
        upstream = self._upstream_payload(context)
        return {
            "phase": self.phase,
            "feature_pipeline": upstream["phase-1"]["feature_pipeline"],
            "optimizer": context.config.optimizer.to_dict(),
            "scheduler": context.config.scheduler.to_dict(),
            "upstream": upstream,
        }


@dataclass(frozen=True, slots=True)
class PhaseThreeService(PhaseService):
    """Summarize the architecture comparison configuration."""

    phase: str = "phase-3"
    dependencies: tuple[str, ...] = ("phase-1", "phase-2")

    def execute(self, context: PipelineContext) -> dict[str, Any]:
        upstream = self._upstream_payload(context)
        return {
            "phase": self.phase,
            "feature_pipeline": upstream["phase-1"]["feature_pipeline"],
            "optimizer": upstream["phase-2"]["optimizer"],
            "scheduler": upstream["phase-2"]["scheduler"],
            "model": context.config.model.to_dict(),
            "upstream": upstream,
        }


@dataclass(frozen=True, slots=True)
class PhaseFourService(PhaseService):
    """Summarize the final evaluation configuration."""

    phase: str = "phase-4"
    dependencies: tuple[str, ...] = ("phase-1", "phase-2", "phase-3")

    def execute(self, context: PipelineContext) -> dict[str, Any]:
        upstream = self._upstream_payload(context)
        return {
            "phase": self.phase,
            "model": upstream["phase-3"]["model"],
            "checkpointing": context.config.checkpointing.to_dict(),
            "mlflow": context.config.mlflow.to_dict(),
            "upstream": upstream,
            "final_status": "ready-for-evaluation",
        }


PHASE_SERVICES: dict[str, PhaseService] = {
    "phase-1": PhaseOneService(),
    "phase-2": PhaseTwoService(),
    "phase-3": PhaseThreeService(),
    "phase-4": PhaseFourService(),
}
"""Registry of built-in phase services."""
