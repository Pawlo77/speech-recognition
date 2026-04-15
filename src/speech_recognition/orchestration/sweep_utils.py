"""Shared helpers for sweep-style orchestration phases."""

import json
import os
import subprocess
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SWEEP_MAX_TRIALS_ENV = "SPEECH_SWEEP_MAX_TRIALS"
SWEEP_SEED_ENV = "SPEECH_SWEEP_SEED"


def apply_trial_cap(
    trials: tuple[Any, ...],
    *,
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
    *,
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


def summary_from_completed_process(
    child_state_path: Path,
    completed_process: subprocess.CompletedProcess[str],
    *,
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
