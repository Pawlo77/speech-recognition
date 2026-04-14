import json
from pathlib import Path

import pytest

from speech_recognition.config import ExperimentConfig, MLflowTrackingConfig, ModelConfig
from speech_recognition.orchestration.tracking import build_mlflow_tracker


def test_mlflow_tracker_logs_training_and_efficiency_metrics(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow")
    mlflow_client = mlflow.tracking.MlflowClient

    tracking_uri = tmp_path / "mlruns"
    config = ExperimentConfig(
        model=ModelConfig(family="mlp_mixer"),
        mlflow=MLflowTrackingConfig(
            tracking_uri=str(tracking_uri),
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

    client = mlflow_client(tracking_uri=str(tracking_uri))
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
    assert float(run.data.metrics["hardware_total_disk_bytes"]) > 0
    assert float(run.data.metrics["epoch"]) == pytest.approx(3.0)
    assert float(run.data.metrics["step"]) == pytest.approx(42.0)
    assert int(run.data.tags["hardware.total_ram_bytes"]) >= 0
    assert int(run.data.tags["hardware.total_disk_bytes"]) > 0

    artifact_names = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert "reproducibility_report.json" in artifact_names

    report_path = Path(client.download_artifacts(run.info.run_id, "reproducibility_report.json"))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert int(report_payload["total_ram_bytes"]) >= 0
    assert int(report_payload["total_disk_bytes"]) > 0
