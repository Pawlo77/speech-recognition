import json
import sys
from pathlib import Path

import pytest

from speech_recognition.config import ExperimentConfig
from speech_recognition.orchestration.phase_one import (
    PhaseOneSweepRunner,
    PhaseOneSweepState,
    PhaseOneTrialRecord,
    PhaseOneTrialSpec,
    build_phase_one_command,
)


def test_build_phase_one_command_uses_isolated_child_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "temp_config.json"
    output_dir = tmp_path / "outputs"

    command = build_phase_one_command(
        config_path,
        "trial_01_mel_spectrogram_convnext_seed_0",
        output_dir,
    )

    assert command[:4] == [sys.executable, "-m", "speech_recognition.cli", "run-single-train"]
    assert command[4:] == [
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
        "--run-name",
        "trial_01_mel_spectrogram_convnext_seed_0",
    ]


def test_phase_one_sweep_skips_completed_trials_and_persists_best_feature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    runner = PhaseOneSweepRunner(output_dir=output_dir, base_config=ExperimentConfig())

    completed_trial = PhaseOneTrialSpec(
        trial_id="trial_01_mel_spectrogram_convnext_seed_0",
        feature_name="mel_spectrogram",
        proxy_model="convnext",
        seed=0,
    )
    pending_trial = PhaseOneTrialSpec(
        trial_id="trial_02_high_temporal_mel_xlstm_seed_42",
        feature_name="high_temporal_mel",
        proxy_model="xlstm",
        seed=42,
    )

    def fake_trials() -> tuple[PhaseOneTrialSpec, ...]:
        return (completed_trial, pending_trial)

    monkeypatch.setattr(
        "speech_recognition.orchestration.phase_one.build_phase_one_trials",
        fake_trials,
    )

    completed_record = PhaseOneTrialRecord(
        trial_id=completed_trial.trial_id,
        feature_name=completed_trial.feature_name,
        proxy_model=completed_trial.proxy_model,
        seed=completed_trial.seed,
        run_name=completed_trial.trial_id,
        config_path=str(
            output_dir / "phase_1" / "runs" / completed_trial.trial_id / "temp_config.json"
        ),
        child_state_path=str(
            output_dir / "phase_1" / "runs" / completed_trial.trial_id / "state.json"
        ),
        validation_macro_f1=0.61,
        completed_at="2026-04-13T00:00:00+00:00",
    )
    initial_state = PhaseOneSweepState(
        output_dir=str(output_dir),
        completed_trials={completed_record.trial_id: completed_record},
        best_trial_id=completed_record.trial_id,
        best_validation_macro_f1=completed_record.validation_macro_f1,
        created_at="2026-04-13T00:00:00+00:00",
        updated_at="2026-04-13T00:00:00+00:00",
    )
    runner.state_path.parent.mkdir(parents=True, exist_ok=True)
    runner.state_path.write_text(
        json.dumps(initial_state.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    calls: list[list[str]] = []

    def fake_run(command, *, check, env, text, capture_output):
        _ = check, env, text, capture_output
        calls.append(list(command))
        run_name = command[command.index("--run-name") + 1]
        child_state_path = output_dir / "phase_1" / "runs" / run_name / "state.json"
        child_state_path.parent.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "phase_1" / "runs" / run_name / "temp_config.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert config_payload["seed"] == pending_trial.seed
        child_state_path.write_text(
            json.dumps(
                {
                    "phase_artifacts": {
                        "phase-1": {
                            "output_data": {"metrics": {"macro_f1": 0.88}},
                        }
                    }
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

        class _CompletedProcess:
            returncode = 0

        return _CompletedProcess()

    monkeypatch.setattr("speech_recognition.orchestration.phase_one.subprocess.run", fake_run)

    payload = runner.execute()

    assert len(calls) == 1
    assert calls[0][:4] == [sys.executable, "-m", "speech_recognition.cli", "run-single-train"]
    assert payload["completed_trials"] == 2
    assert payload["best_validation_macro_f1"] == pytest.approx(0.88)
    assert payload["best_trial"]["trial_id"] == pending_trial.trial_id
    assert runner.state_path.exists()
    assert runner.best_feature_path.exists()
