"""Deterministic phase services used by the pipeline runner."""

import os
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from .state import PHASE_ORDER, PhaseArtifact, PipelineState

ISOLATED_CHILD_ENV: dict[str, str] = {
    "PYTHONPATH": ".",
    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
    "OMP_NUM_THREADS": "1",
}
"""Environment keys enforced for isolated child-process orchestration."""


def build_isolated_subprocess_command(
    command: str,
    config_path: Path,
    output_dir: Path,
    run_name: str,
) -> list[str]:
    """Build a child-process CLI command for isolated phase execution."""
    if command not in {"run-single-train", "run-single-eval"}:
        raise ValueError(f"Unsupported isolated command '{command}'.")

    return [
        sys.executable,
        "-m",
        "speech_recognition.cli",
        command,
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        run_name,
    ]


def build_isolated_subprocess_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build environment variables for isolated child-process execution."""
    env = dict(base_env or os.environ)
    env.update(ISOLATED_CHILD_ENV)
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def run_isolated_subprocess(
    command: str,
    config_path: Path,
    output_dir: Path,
    run_name: str,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a child process using the isolated CLI invocation contract."""
    invocation = build_isolated_subprocess_command(
        command=command,
        config_path=config_path,
        output_dir=output_dir,
        run_name=run_name,
    )
    env = build_isolated_subprocess_env()
    return subprocess.run(  # noqa: S603
        invocation,
        check=check,
        env=env,
        text=True,
        capture_output=True,
    )


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Provide config, state, and upstream artifacts to a phase service."""

    config: ExperimentConfig
    """Experiment configuration for this pipeline run."""
    run_name: str
    """Name identifier for this pipeline run."""
    run_root: Path
    """Root directory where run artifacts are stored."""
    state: PipelineState
    """Current persistence state of the pipeline."""

    @property
    def upstream_artifacts(self) -> Mapping[str, PhaseArtifact]:
        """Return completed phase artifacts keyed by phase name."""
        return self.state.phase_artifacts


@dataclass(frozen=True, slots=True)
class PhaseService:
    """Base class for deterministic phase services."""

    phase: str
    """Canonical phase identifier (e.g., phase-1)."""
    dependencies: tuple[str, ...]
    """Tuple of upstream phase names this service depends on."""

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
            "evaluation": context.config.evaluation.to_dict(),
            "frozen_backbones": context.config.evaluation.backbone_ids,
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
