from pathlib import Path

from speech_recognition.orchestration.runtime import eval as runtime_eval


def test_preferred_checkpoint_path_prefers_best_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "checkpoint_step_0000000001.pt").write_bytes(b"step-1")
    (checkpoint_dir / "checkpoint_step_0000000007.pt").write_bytes(b"step-7")
    best = checkpoint_dir / "checkpoint_best.pt"
    best.write_bytes(b"best")

    selected = runtime_eval._preferred_checkpoint_path(checkpoint_dir)

    assert selected == best


def test_preferred_checkpoint_path_falls_back_to_latest_step(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    oldest = checkpoint_dir / "checkpoint_step_0000000003.pt"
    newest = checkpoint_dir / "checkpoint_step_0000000011.pt"
    oldest.write_bytes(b"step-3")
    newest.write_bytes(b"step-11")

    selected = runtime_eval._preferred_checkpoint_path(checkpoint_dir)

    assert selected == newest


def test_preferred_mlflow_checkpoint_artifact_prefers_best_then_latest_step(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime_eval,
        "_collect_artifact_file_paths",
        lambda _client, _run_id, _root: [
            "checkpoints/checkpoint_step_0000000002.pt",
            "checkpoints/nested/checkpoint_best.pt",
            "checkpoints/checkpoint_step_0000000009.pt",
            "checkpoints/checkpoint_best.pt",
        ],
    )

    selected_best = runtime_eval._preferred_mlflow_checkpoint_artifact(
        client=object(),
        run_id="run-1",
        root="checkpoints",
    )

    assert selected_best == "checkpoints/checkpoint_best.pt"

    monkeypatch.setattr(
        runtime_eval,
        "_collect_artifact_file_paths",
        lambda _client, _run_id, _root: [
            "checkpoints/checkpoint_step_0000000002.pt",
            "checkpoints/checkpoint_step_0000000015.pt",
        ],
    )

    selected_latest = runtime_eval._preferred_mlflow_checkpoint_artifact(
        client=object(),
        run_id="run-2",
        root="checkpoints",
    )

    assert selected_latest == "checkpoints/checkpoint_step_0000000015.pt"
