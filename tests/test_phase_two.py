import json
import sys
from pathlib import Path

import pytest

from speech_recognition.config import ExperimentConfig
from speech_recognition.orchestration.phase_two import (
    PhaseTwoSweepRunner,
    PhaseTwoSweepState,
    PhaseTwoTrialRecord,
    PhaseTwoTrialSpec,
    build_phase_two_command,
)


def test_build_phase_two_command_uses_isolated_child_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "temp_config.json"

    command = build_phase_two_command(
        config_path, "trial_01_mfcc_convnext_cosine_annealing_warmup_wd_0.01_seed_0"
    )

    assert command[:4] == [sys.executable, "-m", "speech_recognition.cli", "run-single-train"]
    assert command[4:] == [
        "--config",
        str(config_path),
        "--run-name",
        "trial_01_mfcc_convnext_cosine_annealing_warmup_wd_0.01_seed_0",
    ]


def test_phase_two_sweep_skips_completed_trials_and_uses_phase_one_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    runner = PhaseTwoSweepRunner(output_dir=output_dir, base_config=ExperimentConfig())

    phase_one_best_feature_path = output_dir / "phase_1" / "best_feature.json"
    phase_one_best_feature_path.parent.mkdir(parents=True, exist_ok=True)
    phase_one_best_feature_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial": {
                    "trial_id": "trial_11_mfcc_convnext_seed_42",
                    "feature_name": "mfcc",
                    "proxy_model": "convnext",
                    "seed": 42,
                    "run_name": "trial_11_mfcc_convnext_seed_42",
                    "config_path": "phase_1/runs/trial_11_mfcc_convnext_seed_42/temp_config.json",
                    "child_state_path": "phase_1/runs/trial_11_mfcc_convnext_seed_42/state.json",
                    "validation_macro_f1": 0.73,
                    "completed_at": "2026-04-13T00:00:00+00:00",
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    completed_trial = PhaseTwoTrialSpec(
        trial_id="trial_01_mfcc_convnext_cosine_annealing_warmup_wd_0.01_seed_0",
        feature_name="mfcc",
        proxy_model="convnext",
        weight_decay=0.01,
        scheduler_name="cosine_annealing_warmup",
        seed=0,
    )
    pending_trial = PhaseTwoTrialSpec(
        trial_id="trial_02_mfcc_xlstm_reduce_on_plateau_wd_0.1_seed_42",
        feature_name="mfcc",
        proxy_model="xlstm",
        weight_decay=0.1,
        scheduler_name="reduce_on_plateau",
        seed=42,
    )

    def fake_trials(feature_name: str) -> tuple[PhaseTwoTrialSpec, ...]:
        assert feature_name == "mfcc"
        return (completed_trial, pending_trial)

    monkeypatch.setattr(
        "speech_recognition.orchestration.phase_two.build_phase_two_trials", fake_trials
    )

    completed_record = PhaseTwoTrialRecord(
        trial_id=completed_trial.trial_id,
        feature_name=completed_trial.feature_name,
        proxy_model=completed_trial.proxy_model,
        weight_decay=completed_trial.weight_decay,
        scheduler_name=completed_trial.scheduler_name,
        seed=completed_trial.seed,
        run_name=completed_trial.trial_id,
        config_path=str(
            output_dir / "phase_2" / "runs" / completed_trial.trial_id / "temp_config.json"
        ),
        child_state_path=str(
            output_dir / "phase_2" / "runs" / completed_trial.trial_id / "state.json"
        ),
        validation_macro_f1=0.68,
        completed_at="2026-04-13T00:00:00+00:00",
    )
    initial_state = PhaseTwoSweepState(
        output_dir=str(output_dir),
        phase_one_best_feature_path=str(phase_one_best_feature_path),
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
        config_path = output_dir / "phase_2" / "runs" / run_name / "temp_config.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert config_payload["seed"] == pending_trial.seed
        child_state_path = output_dir / "phase_2" / "runs" / run_name / "state.json"
        child_state_path.parent.mkdir(parents=True, exist_ok=True)
        child_state_path.write_text(
            json.dumps(
                {
                    "phase_artifacts": {
                        "phase-2": {
                            "output_data": {"metrics": {"macro_f1": 0.91}},
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

    monkeypatch.setattr("speech_recognition.orchestration.phase_two.subprocess.run", fake_run)

    payload = runner.execute()

    assert len(calls) == 1
    assert calls[0][:4] == [sys.executable, "-m", "speech_recognition.cli", "run-single-train"]
    assert payload["completed_trials"] == 2
    assert payload["best_validation_macro_f1"] == pytest.approx(0.91)
    assert payload["best_trial"]["trial_id"] == pending_trial.trial_id
    assert runner.state_path.exists()
    assert runner.best_optim_path.exists()
