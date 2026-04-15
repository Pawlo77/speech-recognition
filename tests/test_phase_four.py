import json
import sys
from pathlib import Path

import pytest

from speech_recognition.config import ExperimentConfig
from speech_recognition.orchestration.phase_four import (
    PhaseFourSweepRunner,
    PhaseFourSweepState,
    PhaseFourTrialRecord,
    PhaseFourTrialSpec,
    build_phase_four_command,
)


def _phase_three_backbone(trial_id: str, family: str, score: float, **architecture_params):
    return {
        "trial_id": trial_id,
        "family": family,
        "seed": 0,
        "architecture_params": architecture_params,
        "run_name": trial_id,
        "config_path": f"phase_3/runs/{trial_id}/temp_config.json",
        "child_state_path": f"phase_3/runs/{trial_id}/state.json",
        "validation_macro_f1": score,
        "completed_at": "2026-04-13T00:00:00+00:00",
    }


def test_build_phase_four_command_uses_isolated_child_contract(tmp_path: Path) -> None:
    config_path = tmp_path / "outputs" / "phase_4" / "runs" / "trial_01" / "temp_config.json"

    command = build_phase_four_command(config_path, "trial_01_flat_multiclass_trial_a_seed_0")

    assert command[:4] == [sys.executable, "-m", "speech_recognition.cli", "run-single-train"]
    assert command[4:] == [
        "--config",
        str(config_path),
        "--output-dir",
        str(tmp_path / "outputs"),
        "--run-name",
        "trial_01_flat_multiclass_trial_a_seed_0",
    ]


def test_phase_four_sweep_skips_completed_trials_and_applies_strict_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "outputs"
    runner = PhaseFourSweepRunner(output_dir=output_dir, base_config=ExperimentConfig())

    phase_three_best_backbones_path = output_dir / "phase_3" / "best_backbones.json"
    phase_three_best_backbones_path.parent.mkdir(parents=True, exist_ok=True)
    phase_three_best_backbones_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "best_trial": _phase_three_backbone(
                    "trial_01_ast_linear_interp_dropout_0.1_seed_0",
                    "ast",
                    0.88,
                    dropout=0.1,
                    head="linear",
                    positional_embedding="interp",
                ),
                "top_three_trials": [
                    _phase_three_backbone(
                        "trial_01_ast_linear_interp_dropout_0.1_seed_0",
                        "ast",
                        0.88,
                        dropout=0.1,
                        head="linear",
                        positional_embedding="interp",
                    ),
                    _phase_three_backbone(
                        "trial_02_convnext_sd_0.2_kernel_7_seed_42",
                        "convnext",
                        0.86,
                        stochastic_depth=0.2,
                        kernel_size=7,
                    ),
                    _phase_three_backbone(
                        "trial_03_xlstm_d_64_reset_False_output_mean_seed_2003",
                        "xlstm",
                        0.84,
                        dimension=64,
                        state_reset=False,
                        output_mode="mean",
                    ),
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    phase_two_best_optim_path = output_dir / "phase_2" / "best_optim.json"
    phase_two_best_optim_path.parent.mkdir(parents=True, exist_ok=True)
    phase_two_best_optim_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "trial": {
                    "trial_id": (
                        "trial_01_mel_spectrogram_convnext_cosine_annealing_warmup_wd_0.1_seed_0"
                    ),
                    "feature_name": "mel_spectrogram",
                    "proxy_model": "convnext",
                    "weight_decay": 0.1,
                    "scheduler_name": "reduce_on_plateau",
                    "seed": 0,
                },
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    completed_trial = PhaseFourTrialSpec(
        trial_id="trial_01_flat_multiclass_trial_01_ast_linear_interp_dropout_0.1_seed_0",
        method="flat_multiclass",
        backbone_ids=("trial_01_ast_linear_interp_dropout_0.1_seed_0",),
        seed=0,
    )
    pending_trial = PhaseFourTrialSpec(
        trial_id="trial_02_sampling_control_trial_02_convnext_sd_0.2_kernel_7_seed_42",
        method="sampling_control",
        backbone_ids=("trial_02_convnext_sd_0.2_kernel_7_seed_42",),
        seed=42,
    )

    def fake_trials(
        backbone_ids: tuple[str, ...], seeds: tuple[int, ...]
    ) -> tuple[PhaseFourTrialSpec, ...]:
        assert backbone_ids == (
            "trial_01_ast_linear_interp_dropout_0.1_seed_0",
            "trial_02_convnext_sd_0.2_kernel_7_seed_42",
            "trial_03_xlstm_d_64_reset_False_output_mean_seed_2003",
        )
        assert seeds == (0, 42, 2003)
        return (completed_trial, pending_trial)

    monkeypatch.setattr(
        "speech_recognition.orchestration.phase_four.build_phase_four_trials", fake_trials
    )

    completed_record = PhaseFourTrialRecord(
        trial_id=completed_trial.trial_id,
        method=completed_trial.method,
        backbone_ids=completed_trial.backbone_ids,
        seed=completed_trial.seed,
        baseline_backbone_id=completed_trial.backbone_ids[0],
        baseline_core_command_macro_f1=0.88,
        core_command_macro_f1=0.875,
        unknown_f1=0.91,
        silence_f1=0.89,
        macro_f1_nc=0.90,
        inference_latency_ms_mean=4.2,
        accepted=True,
        config_path=str(
            output_dir / "phase_4" / "runs" / completed_trial.trial_id / "temp_config.json"
        ),
        child_state_path=str(
            output_dir / "phase_4" / "runs" / completed_trial.trial_id / "state.json"
        ),
        completed_at="2026-04-13T00:00:00+00:00",
    )
    initial_state = PhaseFourSweepState(
        output_dir=str(output_dir),
        phase_three_best_backbones_path=str(phase_three_best_backbones_path),
        phase_three_trial_ids=(
            "trial_01_ast_linear_interp_dropout_0.1_seed_0",
            "trial_02_convnext_sd_0.2_kernel_7_seed_42",
            "trial_03_xlstm_d_64_reset_False_output_mean_seed_2003",
        ),
        completed_trials={completed_record.trial_id: completed_record},
        best_trial_id=completed_record.trial_id,
        best_macro_f1_nc=completed_record.macro_f1_nc,
        created_at="2026-04-13T00:00:00+00:00",
        updated_at="2026-04-13T00:00:00+00:00",
    )
    runner.state_path.parent.mkdir(parents=True, exist_ok=True)
    runner.state_path.write_text(
        json.dumps(initial_state.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )

    calls: list[list[str]] = []

    def fake_run(command, env, check):
        _ = env, check
        calls.append(list(command))
        run_name = command[command.index("--run-name") + 1]
        trial_name = run_name.removesuffix("_heldout_test")
        config_path = output_dir / "phase_4" / "runs" / trial_name / "temp_config.json"
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))

        if not run_name.endswith("_heldout_test"):
            assert config_payload["evaluation"]["strategy"] == "sampling_control"
            assert config_payload["evaluation"]["ensemble_members"] == [
                "trial_02_convnext_sd_0.2_kernel_7_seed_42",
            ]
            assert config_payload["evaluation"]["warmup_iterations"] == 50
            assert config_payload["evaluation"]["max_core_command_f1_drop"] == pytest.approx(0.01)
            assert config_payload["model"]["family"] == "convnext"
            assert config_payload["optimizer"]["weight_decay"] == pytest.approx(0.1)
            assert config_payload["scheduler"]["name"] == "reduce_on_plateau"
            assert config_payload["seed"] == 42

        child_state_path = output_dir / "phase_4" / "runs" / run_name / "state.json"
        child_state_path.parent.mkdir(parents=True, exist_ok=True)
        child_state_path.write_text(
            json.dumps(
                {
                    "phase_artifacts": {
                        "phase-4": {
                            "output_data": {
                                "metrics": {
                                    "core_command_macro_f1": 0.93,
                                    "unknown_f1": 0.92,
                                    "silence_f1": 0.90,
                                    "inference_latency_ms_mean": 7.5,
                                }
                            }
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
            stdout = ""

        return _CompletedProcess()

    monkeypatch.setattr(
        "speech_recognition.orchestration.phase_four.run_subprocess_with_live_output",
        fake_run,
    )

    payload = runner.execute()

    assert len(calls) >= 1
    assert calls[0][:4] == [sys.executable, "-m", "speech_recognition.cli", "run-single-train"]
    assert payload["completed_trials"] == 2
    assert payload["best_macro_f1_nc"] == pytest.approx(0.91)
    assert payload["best_trial"]["trial_id"] == pending_trial.trial_id
    assert runner.state_path.exists()
    assert runner.best_eval_path.exists()
    assert runner.method_winners_path.exists()
