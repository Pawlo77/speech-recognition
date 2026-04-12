from pathlib import Path

import pytest

from speech_recognition.config import ExperimentConfig
from speech_recognition.orchestration.state import (
    PhaseArtifact,
    PipelineState,
    PipelineStateStore,
)


def _make_state(store: PipelineStateStore, run_name: str = "state-demo") -> PipelineState:
    return PipelineState.from_config(
        run_name=run_name,
        run_root=store.base_dir / run_name,
        config=ExperimentConfig(),
    )


def _make_artifact(phase: str, macro_f1: float, checkpoint: str | None = None) -> PhaseArtifact:
    output_data = {
        "metrics": {"macro_f1": macro_f1},
        "best_params": {"optimizer": "adamw"},
        "selected": {"phase": phase},
    }
    if checkpoint is not None:
        output_data["checkpoint_pointer"] = checkpoint

    return PhaseArtifact(
        phase=phase,
        artifact_path=f"{phase}.json",
        input_data={"phase": phase},
        output_data=output_data,
    )


def test_state_round_trip_persists_metrics_and_checkpoint_pointers(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "outputs")
    state = _make_state(store)

    artifact_one = _make_artifact(
        "phase-1",
        macro_f1=0.71,
        checkpoint="outputs/checkpoints/ckpt_a.pt",
    )
    state = state.with_artifact(artifact_one)
    store.save_artifact(state, artifact_one)
    store.save(state)

    artifact_two = _make_artifact(
        "phase-2",
        macro_f1=0.79,
        checkpoint="outputs/checkpoints/ckpt_b.pt",
    )
    state = state.with_artifact(artifact_two)
    store.save_artifact(state, artifact_two)
    store.save(state)

    loaded = store.load("state-demo")

    assert loaded.completed_phases == ("phase-1", "phase-2")
    assert loaded.metrics["phase-1"]["macro_f1"] == pytest.approx(0.71)
    assert loaded.metrics["phase-2"]["macro_f1"] == pytest.approx(0.79)
    assert loaded.best_params["phase-2"]["optimizer"] == "adamw"
    assert loaded.latest_checkpoint == "outputs/checkpoints/ckpt_b.pt"
    assert loaded.checkpoint_pointers["phase-2"] == "outputs/checkpoints/ckpt_b.pt"


def test_load_falls_back_when_newest_state_json_is_corrupted(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "outputs")
    state = _make_state(store)

    artifact_one = _make_artifact("phase-1", macro_f1=0.62, checkpoint="outputs/checkpoints/a.pt")
    state = state.with_artifact(artifact_one)
    store.save_artifact(state, artifact_one)
    store.save(state)

    artifact_two = _make_artifact("phase-2", macro_f1=0.75, checkpoint="outputs/checkpoints/b.pt")
    state = state.with_artifact(artifact_two)
    store.save_artifact(state, artifact_two)
    store.save(state)

    newest_state_path = store.state_path("state-demo", "phase-2")
    newest_state_path.write_text('{"schema_version": 1, "broken": ', encoding="utf-8")

    recovered = store.load("state-demo")

    assert recovered.completed_phases == ("phase-1",)
    assert recovered.latest_checkpoint == "outputs/checkpoints/a.pt"


def test_load_uses_checkpoint_pointer_when_state_candidates_are_bad(tmp_path: Path) -> None:
    store = PipelineStateStore(tmp_path / "outputs")
    state = _make_state(store)

    artifact_one = _make_artifact("phase-1", macro_f1=0.63, checkpoint="outputs/checkpoints/a.pt")
    state = state.with_artifact(artifact_one)
    store.save_artifact(state, artifact_one)
    store.save(state)

    artifact_two = _make_artifact("phase-2", macro_f1=0.77, checkpoint="outputs/checkpoints/b.pt")
    state = state.with_artifact(artifact_two)
    store.save_artifact(state, artifact_two)
    store.save(state)

    phase_two_state_path = store.state_path("state-demo", "phase-2")
    phase_two_state_path.write_text('{"schema_version": 1, "broken": ', encoding="utf-8")

    phase_one_state_path = store.state_path("state-demo", "phase-1")
    valid_phase_one_payload = phase_one_state_path.read_text(encoding="utf-8")
    phase_one_state_path.write_text('{"schema_version": 1, "broken": ', encoding="utf-8")

    phase_two_state_path.write_text(valid_phase_one_payload, encoding="utf-8")
    store.checkpoint_pointer_path("state-demo").write_text(
        '{"latest_phase": "phase-2", "run_name": "state-demo", "schema_version": 1}',
        encoding="utf-8",
    )

    recovered = store.load("state-demo")

    assert recovered.completed_phases == ("phase-1",)
    assert recovered.metrics["phase-1"]["macro_f1"] == pytest.approx(0.63)
