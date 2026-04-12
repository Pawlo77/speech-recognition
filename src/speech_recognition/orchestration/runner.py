"""Pipeline runner that persists state after every completed phase."""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import ExperimentConfig
from .performance import execute_with_profile
from .services import PHASE_SERVICES, PipelineContext
from .state import PHASE_ORDER, PhaseArtifact, PipelineState, PipelineStateStore


def _phase_index(phase: str) -> int:
    """Return the zero-based index for a supported phase name."""

    if phase not in PHASE_ORDER:
        raise ValueError(f"Unknown phase '{phase}'.")
    return PHASE_ORDER.index(phase)


@dataclass(frozen=True, slots=True)
class PipelineRunner:
    """Execute a phased pipeline with resumable state."""

    store: PipelineStateStore
    config: ExperimentConfig | None
    run_name: str = "default"

    def _run_root(self) -> Path:
        """Return the directory used for the active run."""

        return self.store.base_dir / self.run_name

    def load_state(self) -> PipelineState:
        """Load the persisted state or create a fresh one."""

        return self.store.load_or_create(self.run_name, self._run_root(), self.config)

    def _effective_config(self, state: PipelineState) -> ExperimentConfig:
        """Return the config used for execution."""

        if self.config is not None:
            return self.config
        return ExperimentConfig.from_dict(state.config)

    def _build_context(self, state: PipelineState) -> PipelineContext:
        """Build the execution context for a phase."""

        return PipelineContext(
            config=self._effective_config(state),
            run_name=self.run_name,
            run_root=self._run_root(),
            state=state,
        )

    def _execute_phase(self, state: PipelineState, phase: str) -> PipelineState:
        """Execute one phase and persist its artifact."""

        service = PHASE_SERVICES[phase]
        context = self._build_context(state)
        config = self._effective_config(state)
        input_data: dict[str, Any] = {
            "config": config.to_dict(),
            "completed_phases": list(state.completed_phases),
            "upstream": {
                name: artifact.output_data for name, artifact in state.phase_artifacts.items()
            },
        }
        phase_output, performance = execute_with_profile(lambda: service.execute(context))
        if not isinstance(phase_output, Mapping):
            raise ValueError(f"Phase '{phase}' must return a mapping payload.")
        output_data = {**dict(phase_output), "performance": performance}
        artifact = PhaseArtifact(
            phase=phase,
            artifact_path=str(self.store.phase_artifact_path(self.run_name, phase)),
            input_data=input_data,
            output_data=output_data,
            depends_on=service.dependencies,
        )
        next_state = state.with_artifact(artifact)
        self.store.save_artifact(next_state, artifact)
        self.store.save(next_state)
        return next_state

    def execute_until(self, phase: str) -> PipelineState:
        """Execute all phases up to and including the requested phase."""

        target_index = _phase_index(phase)
        state = self.load_state()
        for current_phase in PHASE_ORDER[: target_index + 1]:
            if current_phase in state.completed_phases:
                continue
            state = self._execute_phase(state, current_phase)
        return state

    def execute_training(self) -> PipelineState:
        """Execute the training phases."""

        return self.execute_until("phase-3")

    def execute_evaluation(self) -> PipelineState:
        """Execute the final evaluation phase."""

        return self.execute_until("phase-4")

    def execute_all(self) -> PipelineState:
        """Execute the full pipeline from phase 1 through phase 4."""

        return self.execute_until("phase-4")


def state_to_json(state: PipelineState) -> str:
    """Render a pipeline state as pretty-printed JSON."""

    return json.dumps(state.to_dict(), indent=2, sort_keys=True)
