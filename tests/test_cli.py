import json
from pathlib import Path

import pytest

from speech_recognition.cli import build_parser, load_experiment_config, main
from speech_recognition.config import (
    ExperimentConfig,
    FeaturePipelineConfig,
    MLflowTrackingConfig,
    SchedulerConfig,
    TrainingControlConfig,
)
from speech_recognition.orchestration.state import PHASE_ORDER


def _configured_experiment(tracking_uri: str | None = None) -> ExperimentConfig:
    """Build a config with matching scheduler and training epochs for CLI tests."""

    training = TrainingControlConfig(epochs=12)
    scheduler = SchedulerConfig(total_epochs=12)
    features = FeaturePipelineConfig(name="mfcc")
    mlflow = MLflowTrackingConfig(tracking_uri=tracking_uri or "mlruns", run_name="demo")
    return ExperimentConfig(
        features=features,
        training=training,
        scheduler=scheduler,
        mlflow=mlflow,
    )


def test_cli_help_lists_expected_commands(capsys) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])

    assert excinfo.value.code == 0
    stdout = capsys.readouterr().out
    for command in (
        "phase-1",
        "phase-2",
        "phase-3",
        "phase-4",
        "train",
        "eval",
        "resume",
        "status",
        "mlflow-ui",
        "run",
    ):
        assert command in stdout


def test_cli_accepts_isolated_child_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["run-single-train"]).command == "run-single-train"
    assert parser.parse_args(["run-single-eval"]).command == "run-single-eval"


def test_phase_three_command_routes_to_sweep_runner(tmp_path: Path, monkeypatch) -> None:
    class FakePhaseThreeRunner:
        def __init__(self, output_dir, base_config=None) -> None:
            self.output_dir = output_dir
            self.base_config = base_config

        def execute(self):
            return {"phase": "phase-3", "output_dir": str(self.output_dir)}

    monkeypatch.setattr("speech_recognition.cli.PhaseThreeSweepRunner", FakePhaseThreeRunner)

    exit_code = main(["phase-3", "--output-dir", str(tmp_path / "runs")])

    assert exit_code == 0


def test_cli_merges_config_file_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_configured_experiment().to_dict()), encoding="utf-8")

    loaded = load_experiment_config(
        config_path,
        ["model.dropout=0.25", "optimizer.betas=[0.8, 0.99]"],
    )

    assert loaded.features.name == "mfcc"
    assert loaded.training.epochs == 12
    assert loaded.scheduler.total_epochs == 12
    assert loaded.model.dropout == 0.25
    assert loaded.optimizer.betas == (0.8, 0.99)


def test_cli_reports_invalid_override(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["run", "--set", "broken-override"])

    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    assert "Expected KEY=VALUE" in stderr


def test_run_command_executes_full_pipeline_and_writes_state(tmp_path: Path, capsys) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(_configured_experiment(tracking_uri=str(tmp_path / "mlruns")).to_dict()),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "demo",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["completed_phases"] == list(PHASE_ORDER)

    phase_one_state = tmp_path / "runs" / "phase_1" / "runs" / "demo" / "state.json"
    phase_two_artifact = tmp_path / "runs" / "phase_2" / "runs" / "demo" / "artifact.json"
    checkpoint_pointer = tmp_path / "runs" / "checkpoints" / "demo.json"

    assert phase_one_state.exists()
    assert phase_two_artifact.exists()
    assert checkpoint_pointer.exists()

    phase_two_payload = json.loads(phase_two_artifact.read_text(encoding="utf-8"))
    assert (
        phase_two_payload["input_data"]["upstream"]["phase-1"]["feature_pipeline"]["name"] == "mfcc"
    )

    resume_exit_code = main(
        [
            "resume",
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "demo",
        ]
    )
    assert resume_exit_code == 0
    resumed_payload = json.loads(capsys.readouterr().out)
    assert resumed_payload["completed_phases"] == list(PHASE_ORDER)

    status_exit_code = main(
        [
            "status",
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "demo",
        ]
    )
    assert status_exit_code == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["completed_phases"] == list(PHASE_ORDER)
