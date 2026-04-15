import json
import os
from pathlib import Path

import pytest

from speech_recognition.config import ExperimentConfig, MLflowTrackingConfig, ModelConfig
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
    assert float(run.data.metrics["loss"]) == pytest.approx(0.125)
    assert float(run.data.metrics["validation_macro_f1"]) == pytest.approx(0.875)
    assert float(run.data.metrics["model_parameters_total"]) > 0
    assert float(run.data.metrics["model_macs_1sec"]) > 0
    assert float(run.data.metrics["hardware_total_ram_bytes"]) >= 0
    assert float(run.data.metrics["epoch"]) == pytest.approx(3.0)
    assert float(run.data.metrics["step"]) == pytest.approx(42.0)
    assert int(run.data.tags["hardware.total_ram_bytes"]) >= 0

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
