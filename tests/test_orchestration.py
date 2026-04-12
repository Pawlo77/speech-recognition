import sys
from pathlib import Path

import pytest

from speech_recognition.config import (
    ExperimentConfig,
    FeaturePipelineConfig,
    SchedulerConfig,
    TrainingControlConfig,
)
from speech_recognition.orchestration.runner import PipelineRunner
from speech_recognition.orchestration.services import (
    ISOLATED_CHILD_ENV,
    PhaseThreeService,
    build_isolated_subprocess_command,
    build_isolated_subprocess_env,
)
from speech_recognition.orchestration.state import PHASE_ORDER, PipelineStateStore


def _configured_experiment() -> ExperimentConfig:
    """Build a config with consistent training and scheduler settings."""

    training = TrainingControlConfig(epochs=12)
    scheduler = SchedulerConfig(total_epochs=12)
    features = FeaturePipelineConfig(name="mfcc")
    return ExperimentConfig(features=features, training=training, scheduler=scheduler)


def test_pipeline_resume_keeps_completed_phases_and_data_handoff(
    tmp_path: Path, monkeypatch
) -> None:
    store = PipelineStateStore(tmp_path / "runs")
    runner = PipelineRunner(store=store, config=_configured_experiment(), run_name="resume-demo")

    original_execute = PhaseThreeService.execute
    call_count = {"value": 0}

    def flaky_execute(self, context):
        call_count["value"] += 1
        if call_count["value"] == 1:
            raise KeyboardInterrupt
        return original_execute(self, context)

    monkeypatch.setattr(PhaseThreeService, "execute", flaky_execute)

    with pytest.raises(KeyboardInterrupt):
        runner.execute_training()

    interrupted_state = store.load("resume-demo")
    assert interrupted_state.completed_phases == ("phase-1", "phase-2")

    resumed_state = runner.execute_all()
    assert resumed_state.completed_phases == PHASE_ORDER

    phase_two_artifact = resumed_state.phase_artifacts["phase-2"]
    assert (
        phase_two_artifact.input_data["upstream"]["phase-1"]["feature_pipeline"]["name"] == "mfcc"
    )
    assert (
        resumed_state.phase_artifacts["phase-4"].output_data["final_status"]
        == "ready-for-evaluation"
    )
    assert resumed_state.phase_artifacts["phase-1"].output_data["performance"]["elapsed_ms"] >= 0


def test_build_isolated_subprocess_command_uses_cli_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "trial.json"
    command = build_isolated_subprocess_command(
        command="run-single-train",
        config_path=config_path,
        output_dir=tmp_path / "outputs",
        run_name="demo",
    )

    assert command[:3] == [sys.executable, "-m", "speech_recognition.cli"]
    assert command[3] == "run-single-train"
    assert "--config" in command
    assert "--output-dir" in command
    assert "--run-name" in command


def test_build_isolated_subprocess_env_injects_required_keys() -> None:
    env = build_isolated_subprocess_env({"PATH": "x"})

    for key, value in ISOLATED_CHILD_ENV.items():
        assert env[key] == value
