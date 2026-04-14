"""Historical ETA estimator for the orchestration pipeline.

This utility computes per-phase trial-duration quantiles from persisted phase state
files and projects full-pipeline runtime using fixed full-sweep trial counts.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

FULL_SWEEP_TRIALS: dict[str, int] = {
    "phase_1": 30,
    "phase_2": 24,
    "phase_3": 90,
    "phase_4": 45,
}


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * q)
    return float(ordered[index])


def _trial_durations_seconds(state_path: Path) -> list[float]:
    if not state_path.exists():
        return []

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    created_at = payload.get("created_at")
    completed_trials = payload.get("completed_trials")
    if not isinstance(created_at, str) or not isinstance(completed_trials, dict):
        return []

    completed_at_values = [
        record.get("completed_at")
        for record in completed_trials.values()
        if isinstance(record, dict) and isinstance(record.get("completed_at"), str)
    ]
    if not completed_at_values:
        return []

    timestamps = sorted(_parse_ts(value) for value in completed_at_values)
    durations: list[float] = []
    previous = _parse_ts(created_at)
    for current in timestamps:
        delta = (current - previous).total_seconds()
        durations.append(max(0.0, delta))
        previous = current
    return durations


def _hms(seconds: float) -> str:
    total = round(seconds)
    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60
    return f"{hours}h {minutes}m {secs}s"


def build_historical_eta_lines(run_root: Path) -> list[str]:
    phase_samples: dict[str, list[float]] = {
        phase: _trial_durations_seconds(run_root / phase / "state.json")
        for phase in FULL_SWEEP_TRIALS
    }

    lines: list[str] = []
    eta_p50_seconds = 0.0
    eta_p90_seconds = 0.0
    for phase, full_trials in FULL_SWEEP_TRIALS.items():
        samples = phase_samples[phase]
        p50 = _quantile(samples, 0.5)
        p90 = _quantile(samples, 0.9)
        lines.append(f"historical_{phase}_samples={len(samples)}")
        lines.append(f"historical_{phase}_p50_sec={p50:.3f}")
        lines.append(f"historical_{phase}_p90_sec={p90:.3f}")
        eta_p50_seconds += p50 * full_trials
        eta_p90_seconds += p90 * full_trials

    lines.append(f"historical_estimated_full_seconds_p50={round(eta_p50_seconds)}")
    lines.append(f"historical_estimated_full_hms_p50={_hms(eta_p50_seconds)}")
    lines.append(f"historical_estimated_full_seconds_p90={round(eta_p90_seconds)}")
    lines.append(f"historical_estimated_full_hms_p90={_hms(eta_p90_seconds)}")
    lines.append(
        "historical_estimate_note=Derived from completed-trial timestamp deltas in "
        "phase state files; "
        "implicitly includes early stopping behavior observed in those runs."
    )
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute historical P50/P90 ETA for full pipeline."
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("outputs/full-pipeline-check"),
        help="Run root containing phase_1..phase_4 state.json files.",
    )
    args = parser.parse_args()

    for line in build_historical_eta_lines(args.run_root):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
