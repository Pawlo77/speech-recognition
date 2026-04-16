import json
import os
from pathlib import Path

import pytest

from speech_recognition.config import (
    CheckpointConfig,
    ExperimentConfig,
    MLflowTrackingConfig,
    ModelConfig,
)
from speech_recognition.orchestration.tracking import _resolve_tracking_uri, build_mlflow_tracker


def test_mlflow_tracker_logs_training_and_efficiency_metrics(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow_client = mlflow.tracking.MlflowClient

    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns.db').resolve().as_posix()}"
    config = ExperimentConfig(
        model=ModelConfig(family="mlp_mixer"),
        mlflow=MLflowTrackingConfig(
            tracking_uri=tracking_uri,
            experiment_name="tracking-test",
            run_name="demo",
        ),
    )
    tracker = build_mlflow_tracker(config, run_name="demo")

    tracker.start()

    checkpoint_path = tmp_path / "checkpoint.pt"
    checkpoint_path.write_text("checkpoint", encoding="utf-8")

    tracker.log_training_metrics(
        loss=0.125,
        validation_macro_f1=0.875,
        checkpoint_path=checkpoint_path,
        epoch=3,
        step=42,
    )
    tracker.close()

    client = mlflow_client(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("tracking-test")
    assert experiment is not None

    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert run.data.params["model.family"] == "mlp_mixer"
    assert float(run.data.metrics["training_loss"]) == pytest.approx(0.125)
    assert float(run.data.metrics["val_macro_f1"]) == pytest.approx(0.875)
    assert float(run.data.tags["model_parameters_total"]) > 0
    assert float(run.data.tags["model_macs_1sec"]) > 0
    assert float(run.data.metrics["epoch"]) == pytest.approx(3.0)
    assert float(run.data.metrics["step"]) == pytest.approx(42.0)

    artifact_names = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert "reproducibility_report.json" in artifact_names
    checkpoint_artifacts = {
        artifact.path for artifact in client.list_artifacts(run.info.run_id, path="checkpoints")
    }
    assert "checkpoints/checkpoint.pt" in checkpoint_artifacts
    assert not checkpoint_path.exists()

    report_path = Path(client.download_artifacts(run.info.run_id, "reproducibility_report.json"))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert int(report_payload["total_ram_bytes"]) >= 0
    assert int(report_payload["total_disk_bytes"]) > 0


def test_resolve_tracking_uri_normalizes_sqlite_single_slash(tmp_path: Path) -> None:
    malformed_uri = "sqlite:/mlruns.db"

    previous_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        resolved = _resolve_tracking_uri(malformed_uri)
    finally:
        os.chdir(previous_cwd)

    assert resolved.startswith("sqlite:///")
    resolved_db_path = Path(resolved.removeprefix("sqlite:///"))
    assert resolved_db_path.is_absolute()
    assert resolved_db_path.name == "mlruns.db"
    assert resolved_db_path.parent == tmp_path


def test_mlflow_tracker_registers_best_checkpoint_model(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow_client = mlflow.tracking.MlflowClient

    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns.db').resolve().as_posix()}"
    config = ExperimentConfig(
        model=ModelConfig(family="mlp_mixer"),
        mlflow=MLflowTrackingConfig(
            tracking_uri=tracking_uri,
            experiment_name="tracking-model-registry-test",
            run_name="demo-best",
        ),
    )
    tracker = build_mlflow_tracker(config, run_name="demo-best")
    tracker.start()

    first_checkpoint = tmp_path / "checkpoint_step_1.pt"
    second_checkpoint = tmp_path / "checkpoint_step_2.pt"
    best_checkpoint = tmp_path / "checkpoint_best.pt"
    first_checkpoint.write_text("first", encoding="utf-8")
    second_checkpoint.write_text("second", encoding="utf-8")
    best_checkpoint.write_text("best", encoding="utf-8")

    tracker.log_training_metrics(
        validation_macro_f1=0.40,
        checkpoint_path=first_checkpoint,
        epoch=1,
        step=1,
    )
    tracker.log_training_metrics(
        validation_macro_f1=0.85,
        checkpoint_path=second_checkpoint,
        epoch=2,
        step=2,
    )
    tracker.log_training_metrics(
        validation_macro_f1=0.85,
        checkpoint_path=best_checkpoint,
        epoch=2,
        step=2,
    )
    tracker.close()

    client = mlflow_client(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("tracking-model-registry-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert float(run.data.tags["best_validation_macro_f1"]) == pytest.approx(0.85)
    assert run.data.tags["best_checkpoint_path"].endswith("checkpoint_best.pt")
    assert run.data.tags["best_checkpoint_artifact"] == "checkpoints/checkpoint_best.pt"
    assert run.data.tags["best_model_uri"].endswith("checkpoints/checkpoint_best.pt")
    assert run.data.tags["latest_checkpoint_step"] == "2"
    assert run.data.tags["best_registered_model_name"] == "tracking-model-registry-test-best"

    model_name = run.data.tags["best_registered_model_name"]
    model_versions = list(client.search_model_versions(f"name='{model_name}'"))
    assert model_versions
    assert any(
        version.source.endswith("checkpoints/checkpoint_best.pt") for version in model_versions
    )


def test_mlflow_tracker_rolls_step_checkpoint_artifacts(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow_client = mlflow.tracking.MlflowClient

    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns.db').resolve().as_posix()}"
    config = ExperimentConfig(
        model=ModelConfig(family="mlp_mixer"),
        checkpointing=CheckpointConfig(keep_last_n=3),
        mlflow=MLflowTrackingConfig(
            tracking_uri=tracking_uri,
            experiment_name="tracking-rolling-checkpoints-test",
            run_name="demo-rolling",
        ),
    )
    tracker = build_mlflow_tracker(config, run_name="demo-rolling")
    tracker.start()

    for step in range(1, 8):
        checkpoint = tmp_path / f"checkpoint_step_{step}.pt"
        checkpoint.write_text(f"checkpoint-{step}", encoding="utf-8")
        tracker.log_training_metrics(checkpoint_path=checkpoint, epoch=1, step=step)

    best_checkpoint = tmp_path / "checkpoint_best.pt"
    best_checkpoint.write_text("best", encoding="utf-8")
    tracker.log_training_metrics(
        checkpoint_path=best_checkpoint,
        validation_macro_f1=0.99,
        epoch=1,
        step=7,
    )
    tracker.close()

    client = mlflow_client(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("tracking-rolling-checkpoints-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    checkpoint_artifacts = {
        artifact.path for artifact in client.list_artifacts(run.info.run_id, path="checkpoints")
    }
    assert "checkpoints/rolling" in checkpoint_artifacts
    assert "checkpoints/checkpoint_best.pt" in checkpoint_artifacts

    rolling_artifacts = {
        artifact.path
        for artifact in client.list_artifacts(run.info.run_id, path="checkpoints/rolling")
    }
    assert rolling_artifacts == {
        "checkpoints/rolling/checkpoint_slot_0.pt",
        "checkpoints/rolling/checkpoint_slot_1.pt",
        "checkpoints/rolling/checkpoint_slot_2.pt",
    }


def test_mlflow_tracker_resumes_existing_run_id_from_env(tmp_path: Path, monkeypatch) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow_client = mlflow.tracking.MlflowClient

    tracking_uri = f"sqlite:///{(tmp_path / 'mlruns.db').resolve().as_posix()}"
    config = ExperimentConfig(
        model=ModelConfig(family="mlp_mixer"),
        mlflow=MLflowTrackingConfig(
            tracking_uri=tracking_uri,
            experiment_name="tracking-resume-run-id-test",
            run_name="demo-resume",
        ),
    )

    first_tracker = build_mlflow_tracker(config, run_name="demo-resume")
    first_tracker.start()
    first_run_id = first_tracker._active_run_id
    assert first_run_id is not None
    first_tracker.close()

    monkeypatch.setenv("SPEECH_MLFLOW_ACTIVE_RUN_ID", first_run_id)
    resumed_tracker = build_mlflow_tracker(config, run_name="demo-resume")
    resumed_tracker.start()
    assert resumed_tracker._active_run_id == first_run_id
    resumed_tracker.close()

    client = mlflow_client(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name("tracking-resume-run-id-test")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
