#!/usr/bin/env python3
"""Estimate RAM (params + activations) for registered models.

This script builds each family adapter available in the registry and runs
a single forward pass for provided batch sizes, summing parameter memory and
module output activations (as an approximation of peak activation memory).

Usage: python scripts/estimate_ram.py --batch-sizes 1 8 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch

# Ensure repo root is on sys.path so `src` package is importable when running
# the script from the repository root.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.speech_recognition.models.registry import (  # noqa: E402
    DEFAULT_INPUT_BINS,
    DEFAULT_TARGET_FRAMES,
    ModelRegistry,
)


def sizeof_tensor(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())


def estimate_model_ram(adapter: torch.nn.Module, input_tensor: torch.Tensor) -> dict[str, int]:
    activations: list[int] = []
    hooks = []

    def hook(_module: torch.nn.Module, _inputs: Any, outputs: Any) -> None:
        def add(obj: Any) -> None:
            if isinstance(obj, torch.Tensor):
                activations.append(sizeof_tensor(obj))
            elif isinstance(obj, list | tuple):
                for item in obj:
                    add(item)

        add(outputs)

    for m in adapter.modules():
        hooks.append(m.register_forward_hook(hook))

    adapter.eval()
    with torch.no_grad():
        outputs = adapter(input_tensor)

    for h in hooks:
        h.remove()

    params_mem = int(sum(p.numel() * p.element_size() for p in adapter.parameters()))
    activations_mem = int(sum(activations))
    output_mem = sizeof_tensor(outputs) if isinstance(outputs, torch.Tensor) else 0

    return {
        "params_bytes": params_mem,
        "activations_bytes": activations_mem,
        "output_bytes": output_mem,
        "total_bytes": params_mem + activations_mem,
    }


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.2f}{unit}"
        n /= 1024.0
    return f"{n:.2f}TB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate RAM per model for batch sizes")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=int,
        default=[1, 8, 32],
        help="Batch sizes to evaluate (space separated)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the forward passes on (cpu or cuda)",
    )

    args = parser.parse_args()

    registry = ModelRegistry()
    families = registry.supported_families()

    print("Estimating RAM per model (params + activations approximation)\n")

    for family in families:
        print(f"Family: {family}")
        try:
            adapter = registry.create(family=family)
        except Exception as exc:
            print(f"  Skipped (build error): {exc}")
            continue

        adapter.to(args.device)

        for B in args.batch_sizes:
            # Input shape: [B, 1, bins, frames]
            inp = torch.randn(B, 1, DEFAULT_INPUT_BINS, DEFAULT_TARGET_FRAMES, device=args.device)
            try:
                stats = estimate_model_ram(adapter, inp)
            except Exception as exc:
                print(f"  Batch {B}: failed during forward: {exc}")
                continue

            print(
                f"  Batch {B}: params={human(stats['params_bytes'])}, "
                f"activations={human(stats['activations_bytes'])}, "
                f"total≈{human(stats['total_bytes'])}"
            )

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
