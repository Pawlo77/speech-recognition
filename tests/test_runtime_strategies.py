import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from speech_recognition.config import ExperimentConfig, MLflowTrackingConfig
from speech_recognition.orchestration.runtime import strategies


def _mlflow_enabled_config(tmp_path: Path) -> ExperimentConfig:
    config = ExperimentConfig()
    return replace(
        config,
        mlflow=MLflowTrackingConfig(
            enabled=True,
            tracking_uri=f"sqlite:///{(tmp_path / 'mlruns.db').as_posix()}",
            experiment_name="resume-test",
            run_name=config.mlflow.run_name,
            log_params=config.mlflow.log_params,
            log_metrics=config.mlflow.log_metrics,
            log_artifacts=config.mlflow.log_artifacts,
            retain_local_checkpoints=config.mlflow.retain_local_checkpoints,
        ),
    )


def test_restore_checkpoint_from_mlflow_if_needed_noops_when_local_checkpoint_exists(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    existing_checkpoint = checkpoint_dir / "checkpoint_step_0000000012.pt"
    existing_checkpoint.write_bytes(b"local")

    def _unexpected_import(name: str):
        raise AssertionError(f"Unexpected import: {name}")

    monkeypatch.setattr(strategies.importlib, "import_module", _unexpected_import)

    restored = strategies._restore_checkpoint_from_mlflow_if_needed(
        config=_mlflow_enabled_config(tmp_path),
        run_name="trial_01",
        checkpoint_dir=checkpoint_dir,
        artifact_root="checkpoints",
    )

    assert restored == existing_checkpoint


def test_restore_checkpoint_from_mlflow_if_needed_skips_active_run_and_downloads_previous(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    class _FakeMlflowClient:
        def get_experiment_by_name(self, name: str):
            assert name == "resume-test"
            return SimpleNamespace(experiment_id="exp-1")

        def search_runs(self, experiment_ids, filter_string, order_by, max_results):
            assert experiment_ids == ["exp-1"]
            assert "tags.pipeline.run_name" in filter_string
            assert order_by == ["attributes.start_time DESC"]
            assert max_results == 50
            return [
                SimpleNamespace(
                    info=SimpleNamespace(run_id="active-run"),
                    data=SimpleNamespace(
                        tags={
                            "latest_checkpoint_artifact": "checkpoints/checkpoint_step_0000000999.pt"  # noqa: E501
                        }
                    ),
                ),
                SimpleNamespace(
                    info=SimpleNamespace(run_id="previous-run"),
                    data=SimpleNamespace(
                        tags={
                            "latest_checkpoint_artifact": "checkpoints/checkpoint_step_0000000100.pt"  # noqa: E501
                        }
                    ),
                ),
            ]

        def list_artifacts(self, run_id, root):
            _ = root
            if run_id == "active-run":
                return [
                    SimpleNamespace(path="checkpoints/checkpoint_step_0000000999.pt", is_dir=False)
                ]
            return [SimpleNamespace(path="checkpoints/checkpoint_step_0000000100.pt", is_dir=False)]

        def download_artifacts(self, run_id, artifact_path, destination):
            assert run_id == "previous-run"
            local_file = Path(destination) / Path(artifact_path).name
            local_file.write_bytes(b"from-mlflow")
            return str(local_file)

    class _FakeMlflowModule:
        def __init__(self) -> None:
            self.tracking = SimpleNamespace(MlflowClient=_FakeMlflowClient)

        def set_tracking_uri(self, uri: str) -> None:
            assert uri.startswith("sqlite:///")

    monkeypatch.setenv(strategies._ACTIVE_MLFLOW_RUN_ID_ENV, "active-run")
    monkeypatch.setattr(strategies.importlib, "import_module", lambda name: _FakeMlflowModule())  # noqa: ARG005

    restored = strategies._restore_checkpoint_from_mlflow_if_needed(
        config=_mlflow_enabled_config(tmp_path),
        run_name="trial_01",
        checkpoint_dir=checkpoint_dir,
        artifact_root="checkpoints",
    )

    assert restored is not None
    assert restored.name == "checkpoint_step_0000000100.pt"
    assert restored.exists()
    assert restored.read_bytes() == b"from-mlflow"
    assert os.environ[strategies._ACTIVE_MLFLOW_RUN_ID_ENV] == "previous-run"


def test_restore_checkpoint_from_mlflow_if_needed_uses_tagged_step_with_rolling_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    class _FakeMlflowClient:
        def get_experiment_by_name(self, name: str):
            assert name == "resume-test"
            return SimpleNamespace(experiment_id="exp-1")

        def search_runs(self, experiment_ids, filter_string, order_by, max_results):
            assert experiment_ids == ["exp-1"]
            assert "tags.pipeline.run_name" in filter_string
            assert order_by == ["attributes.start_time DESC"]
            assert max_results == 50
            return [
                SimpleNamespace(
                    info=SimpleNamespace(run_id="rolling-run"),
                    data=SimpleNamespace(
                        tags={
                            "latest_checkpoint_artifact": "checkpoints/rolling/checkpoint_slot_2.pt",  # noqa: E501
                            "latest_checkpoint_step": "123",
                        }
                    ),
                )
            ]

        def list_artifacts(self, run_id, root):
            _ = run_id, root
            return []

        def download_artifacts(self, run_id, artifact_path, destination):
            assert run_id == "rolling-run"
            assert artifact_path == "checkpoints/rolling/checkpoint_slot_2.pt"
            local_file = Path(destination) / Path(artifact_path).name
            local_file.write_bytes(b"rolling")
            return str(local_file)

    class _FakeMlflowModule:
        def __init__(self) -> None:
            self.tracking = SimpleNamespace(MlflowClient=_FakeMlflowClient)

        def set_tracking_uri(self, uri: str) -> None:
            assert uri.startswith("sqlite:///")

    monkeypatch.setattr(strategies.importlib, "import_module", lambda name: _FakeMlflowModule())  # noqa: ARG005

    restored = strategies._restore_checkpoint_from_mlflow_if_needed(
        config=_mlflow_enabled_config(tmp_path),
        run_name="trial_01",
        checkpoint_dir=checkpoint_dir,
        artifact_root="checkpoints",
    )

    assert restored is not None
    assert restored.name == "checkpoint_step_0000000123.pt"
    assert restored.exists()
    assert restored.read_bytes() == b"rolling"
    assert os.environ[strategies._ACTIVE_MLFLOW_RUN_ID_ENV] == "rolling-run"
