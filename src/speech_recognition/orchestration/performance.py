"""Low-overhead runtime profiling helpers for orchestration phases."""

import resource
import sys
from collections.abc import Callable
from time import perf_counter
from typing import Any

from .sweep_utils import utc_now


def _utc_now() -> str:
    return utc_now()


def _rss_mb() -> float:
    """Return process resident set size in megabytes.

    On macOS, ``ru_maxrss`` is measured in bytes.
    On Linux, it is measured in kilobytes.
    """

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(usage) / (1024.0 * 1024.0)
    return float(usage) / 1024.0


def execute_with_profile(
    func: Callable[[], dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Execute a callable and return its output plus runtime performance metadata."""

    started_at = _utc_now()
    rss_before = _rss_mb()
    start = perf_counter()
    payload = func()
    elapsed_ms = round((perf_counter() - start) * 1000.0)
    rss_after = _rss_mb()
    finished_at = _utc_now()

    performance = {
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_ms": elapsed_ms,
        "rss_mb_before": round(rss_before, 3),
        "rss_mb_after": round(rss_after, 3),
        "rss_mb_delta": round(rss_after - rss_before, 3),
    }
    return payload, performance
