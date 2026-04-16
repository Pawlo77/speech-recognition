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


def _configured_experiment(tracking_uri: str | None = None) -> ExperimentConfig:
    """Build a config with matching scheduler and training epochs for CLI tests."""
    training = TrainingControlConfig(epochs=12)
    scheduler = SchedulerConfig(total_epochs=12)
    features = FeaturePipelineConfig(name="mfcc")
    mlflow = MLflowTrackingConfig(
        tracking_uri=tracking_uri or "sqlite:///mlruns.db",
        run_name="demo",
    )
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


def test_phase_four_command_routes_to_sweep_runner(tmp_path: Path, monkeypatch) -> None:
    class FakePhaseFourRunner:
        def __init__(self, output_dir, base_config=None) -> None:
            self.output_dir = output_dir
            self.base_config = base_config

        def execute(self):
            return {"phase": "phase-4", "output_dir": str(self.output_dir)}

    monkeypatch.setattr("speech_recognition.cli.PhaseFourSweepRunner", FakePhaseFourRunner)

    exit_code = main(["phase-4", "--output-dir", str(tmp_path / "runs")])

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


def test_run_resume_and_status_route_through_sweep_pipeline(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    called = {"run": [], "status": []}

    def fake_run_sweep_pipeline(output_dir, config, run_name, include_phase_four):
        called["run"].append(
            {
                "output_dir": output_dir,
                "config": config,
                "run_name": run_name,
                "include_phase_four": include_phase_four,
            }
        )
        return {
            "command": "run",
            "output_dir": str(output_dir),
            "run_name": run_name,
            "completed_phases": ["phase-1", "phase-2", "phase-3", "phase-4"],
            "phases": {
                "phase-1": {"completed_trials": 1},
                "phase-2": {"completed_trials": 1},
                "phase-3": {"completed_trials": 1},
                "phase-4": {"completed_trials": 1},
            },
        }

    def fake_build_status_payload(output_dir, config, run_name):
        called["status"].append(
            {
                "output_dir": output_dir,
                "config": config,
                "run_name": run_name,
            }
        )
        return {
            "command": "status",
            "output_dir": str(output_dir),
            "run_name": run_name,
            "completed_phases": ["phase-1"],
            "phases": {"phase-1": {"completed_trials": 1}},
        }

    monkeypatch.setattr("speech_recognition.cli._run_sweep_pipeline", fake_run_sweep_pipeline)
    monkeypatch.setattr(
        "speech_recognition.cli._build_sweep_status_payload",
        fake_build_status_payload,
    )

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_configured_experiment().to_dict()), encoding="utf-8")

    run_exit_code = main(
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
    assert run_exit_code == 0
    run_payload = json.loads(capsys.readouterr().out)
    assert run_payload["completed_phases"] == ["phase-1", "phase-2", "phase-3", "phase-4"]
    assert called["run"][0]["include_phase_four"] is True

    resume_exit_code = main(
        [
            "resume",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "demo",
        ]
    )
    assert resume_exit_code == 0
    resume_payload = json.loads(capsys.readouterr().out)
    assert resume_payload["completed_phases"] == ["phase-1", "phase-2", "phase-3", "phase-4"]
    assert called["run"][1]["include_phase_four"] is True

    status_exit_code = main(
        [
            "status",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "runs"),
            "--run-name",
            "demo",
        ]
    )
    assert status_exit_code == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["completed_phases"] == ["phase-1"]
    assert len(called["status"]) == 1


def test_run_mlflow_only_uses_temporary_output_dir(tmp_path: Path, monkeypatch, capsys) -> None:
    seen: dict[str, Path] = {}

    def fake_run_sweep_pipeline(output_dir, config, run_name, include_phase_four):
        assert config is not None
        assert config.mlflow.enabled
        assert include_phase_four is True
        seen["output_dir"] = output_dir
        return {
            "command": "run",
            "output_dir": str(output_dir),
            "run_name": run_name,
            "completed_phases": [],
            "phases": {},
        }

    monkeypatch.setattr("speech_recognition.cli._run_sweep_pipeline", fake_run_sweep_pipeline)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(_configured_experiment().to_dict()), encoding="utf-8")

    exit_code = main(
        [
            "run",
            "--config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "outputs"),
            "--run-name",
            "demo",
            "--mlflow-only",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    used_output_dir = seen["output_dir"]
    assert used_output_dir != (tmp_path / "outputs")
    assert "speech-recognition-mlflow-" in str(used_output_dir)
    assert payload["output_dir"] == str(used_output_dir)


def test_run_binds_default_experiment_name_to_run_name(monkeypatch, capsys) -> None:
    seen: dict[str, str] = {}

    def fake_run_sweep_pipeline(output_dir, config, run_name, include_phase_four):  # noqa: ARG001
        assert config is not None
        seen["experiment_name"] = config.mlflow.experiment_name
        return {
            "command": "run",
            "output_dir": str(output_dir),
            "run_name": run_name,
            "completed_phases": [],
            "phases": {},
        }

    monkeypatch.setattr("speech_recognition.cli._run_sweep_pipeline", fake_run_sweep_pipeline)

    exit_code = main(["run", "--run-name", "default"])

    assert exit_code == 0
    assert seen["experiment_name"] == "default"
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_name"] == "default"


def test_run_keeps_explicit_experiment_name_from_config(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    seen: dict[str, str] = {}

    def fake_run_sweep_pipeline(output_dir, config, run_name, include_phase_four):  # noqa: ARG001
        assert config is not None
        seen["experiment_name"] = config.mlflow.experiment_name
        return {
            "command": "run",
            "output_dir": str(output_dir),
            "run_name": run_name,
            "completed_phases": [],
            "phases": {},
        }

    monkeypatch.setattr("speech_recognition.cli._run_sweep_pipeline", fake_run_sweep_pipeline)

    config = _configured_experiment()
    config = ExperimentConfig(
        dataset=config.dataset,
        features=config.features,
        model=config.model,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        training=config.training,
        checkpointing=config.checkpointing,
        mlflow=MLflowTrackingConfig(
            enabled=config.mlflow.enabled,
            tracking_uri=config.mlflow.tracking_uri,
            experiment_name="speech-recognition-explicit",
            run_name=config.mlflow.run_name,
            log_params=config.mlflow.log_params,
            log_metrics=config.mlflow.log_metrics,
            log_artifacts=config.mlflow.log_artifacts,
            retain_local_checkpoints=config.mlflow.retain_local_checkpoints,
        ),
        evaluation=config.evaluation,
        phase=config.phase,
        seed=config.seed,
        seeds=config.seeds,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")

    exit_code = main(["run", "--config", str(config_path), "--run-name", "default"])

    assert exit_code == 0
    assert seen["experiment_name"] == "speech-recognition-explicit"
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_name"] == "default"
