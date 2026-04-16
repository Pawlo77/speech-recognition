"""Shared helpers for sweep-style orchestration phases."""

import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SWEEP_MAX_TRIALS_ENV = "SPEECH_SWEEP_MAX_TRIALS"
"""Environment variable name for overriding the maximum number
of trials to run in a sweep phase. If set to a positive integer,
the trial list will be truncated to that length for smoke testing purposes."""
SWEEP_SEED_ENV = "SPEECH_SWEEP_SEED"
"""Environment variable name for overriding the random seed(s)
used in a sweep phase."""


def apply_trial_cap(
    trials: tuple[Any, ...],
    env_var: str = SWEEP_MAX_TRIALS_ENV,
) -> tuple[Any, ...]:
    """Return a capped trial tuple when a smoke-limit env override is set."""
    raw_limit = os.environ.get(env_var)
    if raw_limit in {None, ""}:
        return trials
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer.") from exc
    if limit <= 0:
        raise ValueError(f"{env_var} must be greater than zero.")
    return trials[:limit]


def sweep_seeds(
    default_seeds: Iterable[int],
    env_var: str = SWEEP_SEED_ENV,
) -> tuple[int, ...]:
    """Return default seeds or a single-seed override for smoke runs."""
    seeds = tuple(int(seed) for seed in default_seeds)
    raw_seed = os.environ.get(env_var)
    if raw_seed in {None, ""}:
        return seeds
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be an integer.") from exc
    if seed not in seeds:
        raise ValueError(f"{env_var} must be one of {seeds}.")
    return (seed,)


def utc_now() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def build_trial_output_paths(phase_dir: Path, trial_id: str) -> tuple[Path, Path, Path]:
    """Return the standard trial directory, config path, and child state path."""
    trial_dir = phase_dir / "runs" / trial_id
    config_path = trial_dir / "temp_config.json"
    child_state_path = phase_dir / "runs" / trial_id / "state.json"
    return trial_dir, config_path, child_state_path


def summary_from_completed_process(
    child_state_path: Path,
    completed_process: subprocess.CompletedProcess[str],
    read_json: Callable[[Path], dict[str, Any]],
) -> dict[str, Any]:
    """Load child summary from state file first, then subprocess JSON stdout."""
    if child_state_path.exists():
        try:
            payload = read_json(child_state_path)
            if isinstance(payload, dict):
                return payload
        except (json.JSONDecodeError, OSError):
            pass
    try:
        payload = json.loads(completed_process.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def run_subprocess_with_live_output(
    command: list[str],
    env: Mapping[str, str],
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess while streaming merged stdout/stderr to the parent console."""
    process = subprocess.Popen(  # noqa: S603
        command,
        env=dict(env),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )

    streamed_lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            streamed_lines.append(line)

    return_code = process.wait()
    combined_stdout = "".join(streamed_lines)
    if check and return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=combined_stdout)
    return subprocess.CompletedProcess(command, return_code, combined_stdout, "")


def iter_nested_payloads(payload: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    """Yield payload and nested phase-artifact output payloads depth-first."""
    if not isinstance(payload, Mapping):
        return

    stack: list[Mapping[str, Any]] = [payload]
    while stack:
        current = stack.pop()
        yield current

        phase_artifacts = current.get("phase_artifacts")
        if not isinstance(phase_artifacts, Mapping):
            continue

        for artifact in phase_artifacts.values():
            if not isinstance(artifact, Mapping):
                continue
            output_data = artifact.get("output_data")
            if isinstance(output_data, Mapping):
                stack.append(output_data)
