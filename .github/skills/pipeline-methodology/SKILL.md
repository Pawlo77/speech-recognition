---
name: pipeline-methodology
description: "Use when validating implementation progress against prompts, running final test/lint gates, checking phase handoff/resume behavior, and enforcing repo coding rules such as avoiding from __future__ import annotations unless required."
---

# Pipeline Methodology Checklist

## Purpose
Use this checklist before reporting completion for any implementation task in this repository.

## Always Validate Against Prompt Requirements
- Re-read the active prompt and map each requirement to a concrete code path or test.
- Confirm command surface matches exactly when CLI requirements are specified.
- Confirm phase handoff data is persisted and consumed by downstream phases.
- Confirm interruption and resume behavior is tested, not only implemented.

## Validation Gates
- Run `make test`.
- Run `make pre-commit-all`.
- If either fails, fix all reported issues before reporting completion.

## Resume and State Guarantees
- Ensure state writes are atomic.
- Ensure per-phase artifacts are persisted after each successful phase.
- Ensure `resume` and `status` work from persisted state without requiring the original config file.
- Ensure mismatched explicit config and persisted state fails clearly.

## CLI Stability Rules
- Keep CLI argument names stable.
- Keep orchestration logic out of CLI command handlers.
- Keep output machine-readable (JSON payloads).
- Support both config files and CLI overrides (`--set key=value`).

## Repository Coding Rules
- Avoid `from __future__ import annotations` unless genuinely needed.
- Keep dataclass and config validation errors explicit and user-facing.
- Add or update tests whenever behavior changes.
- Do not use dataclass auto-init for wrappers that own `nn.Module` submodules; initialize `nn.Module` first in an explicit `__init__`.

## Optional Dependency Policy
- Declare source libraries explicitly in code-level constants for each family/component.
- Treat heavyweight libraries (`transformers`, `torchvision`, `timm`, `mamba-ssm`, `xlstm`) as optional at runtime unless guaranteed by project dependencies.
- Provide deterministic fallback implementations when optional libraries are unavailable, especially on Apple Silicon/MPS.
- Ensure fallback and primary paths expose the same interface and output shape contract.

## Model Adapter Contract Checks
- Enforce a shared adapter interface with `forward_pass`, `validate_input_shape`, and `profile_efficiency`.
- Validate input tensors as `[B, 1, F, T]` and normalize temporal length via padding/truncation before forward.
- Keep Kaggle class count explicit (`32`) and verify logits shape `(batch, 32)` in tests.
- Run profiling with `fvcore` (`FlopCountAnalysis`, `parameter_count`) and verify integer outputs in tests.

## Completion Review Template
Before final response, summarize:
1. Requirement coverage status.
2. Any deviations and why.
3. Exact validation commands run.
4. Final pass/fail status.
