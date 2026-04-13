"""Command-line interface for the resumable experiment pipeline."""

import argparse
import importlib.util
import json
from collections.abc import Sequence
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

from .config import ConfigValidationError, ExperimentConfig, MLflowTrackingConfig
from .orchestration import PipelineRunner, PipelineStateStore, build_mlflow_tracker
from .orchestration.phase_one import PhaseOneSweepRunner
from .orchestration.phase_three import PhaseThreeSweepRunner
from .orchestration.phase_two import PhaseTwoSweepRunner

DEFAULT_OUTPUT_DIR = Path("outputs")
"""Default directory for pipeline runs."""


def _parse_override_value(text: str) -> Any:
    """Parse a CLI override value as JSON when possible."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _coerce_value(current_value: Any, new_value: Any) -> Any:
    """Coerce CLI overrides to the current field shape when needed."""

    if isinstance(current_value, tuple) and isinstance(new_value, list):
        return tuple(new_value)
    return new_value


def _apply_override(config: Any, dotted_path: str, value: Any) -> Any:
    """Apply a dotted-path override to a frozen dataclass tree."""

    if not is_dataclass(config):
        raise ConfigValidationError(
            f"Override path '{dotted_path}' does not resolve to a dataclass field."
        )

    head, dot, tail = dotted_path.partition(".")
    if not head:
        raise ConfigValidationError("Override path must not be empty.")
    if dot:
        try:
            nested_value = getattr(config, head)
        except AttributeError as exc:
            raise ConfigValidationError(
                f"Override path '{dotted_path}' does not match a known config field."
            ) from exc
        updated_value = _apply_override(nested_value, tail, value)
        return replace(config, **{head: updated_value})

    try:
        current_value = getattr(config, head)
    except AttributeError as exc:
        raise ConfigValidationError(
            f"Override path '{dotted_path}' does not match a known config field."
        ) from exc
    updated_value = _coerce_value(current_value, value)
    return replace(config, **{head: updated_value})


def load_experiment_config(config_path: Path | None, overrides: Sequence[str]) -> ExperimentConfig:
    """Load an experiment config from disk and apply CLI overrides."""

    config = ExperimentConfig()
    if config_path is not None:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(payload)

    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ConfigValidationError(f"Invalid override '{override}'. Expected KEY=VALUE.")
        config = _apply_override(config, key, _parse_override_value(raw_value))

    return config


def _shared_parent_parser() -> argparse.ArgumentParser:
    """Create the shared argument parser used by all commands."""

    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON config file produced by ExperimentConfig.to_dict().",
    )
    parent.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that stores run state and phase artifacts.",
    )
    parent.add_argument(
        "--run-name",
        default="default",
        help="Name of the pipeline run to load or create.",
    )
    parent.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a nested config value such as training.epochs=80.",
    )
    return parent


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""

    parent = _shared_parent_parser()
    parser = argparse.ArgumentParser(
        prog="speech-recognition",
        description="Resumable speech-recognition pipeline CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        parents=[parent],
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_phase_command(name: str, help_text: str, target: str) -> None:
        command_parser = subparsers.add_parser(
            name,
            help=help_text,
            parents=[parent],
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        command_parser.set_defaults(target_phase=target)

    add_phase_command("phase-1", "Run the feature-strategy phase.", "phase-1")
    add_phase_command("phase-2", "Run the optimization phase.", "phase-2")
    add_phase_command("phase-3", "Run the architecture-comparison phase.", "phase-3")
    add_phase_command("phase-4", "Run the final evaluation phase.", "phase-4")
    add_phase_command("train", "Run the training pipeline through phase 3.", "phase-3")
    add_phase_command("eval", "Run the final evaluation phase.", "phase-4")
    add_phase_command("resume", "Resume the pipeline from the latest saved phase.", "phase-4")
    add_phase_command("status", "Show the current run state.", "status")
    add_phase_command("mlflow-ui", "Show or launch the MLflow UI.", "mlflow-ui")
    add_phase_command("run", "Run the full pipeline end to end.", "phase-4")
    add_phase_command("run-single-train", argparse.SUPPRESS, "phase-3")
    add_phase_command("run-single-eval", argparse.SUPPRESS, "phase-4")
    return parser


def _run_pipeline(command: str, runner: PipelineRunner) -> dict[str, Any]:
    """Dispatch a pipeline command to the runner."""

    if command == "phase-1":
        return runner.execute_until("phase-1").to_dict()
    if command == "phase-2":
        return runner.execute_until("phase-2").to_dict()
    if command == "phase-3":
        return runner.execute_until("phase-3").to_dict()
    if command in {"phase-4", "run", "resume"}:
        return runner.execute_all().to_dict()
    if command == "train":
        return runner.execute_training().to_dict()
    if command == "eval":
        return runner.execute_evaluation().to_dict()
    if command == "run-single-train":
        return runner.execute_training().to_dict()
    if command == "run-single-eval":
        return runner.execute_evaluation().to_dict()
    if command == "status":
        return runner.load_state().to_dict()
    if command == "mlflow-ui":
        state = runner.load_state()
        tracking = runner._effective_config(state).mlflow
        return _launch_or_report_mlflow_ui(tracking)
    raise ConfigValidationError(f"Unknown command '{command}'.")


def _build_tracking_state(
    command: str,
    runner: PipelineRunner,
) -> tuple[ExperimentConfig | None, bool]:
    """Return the effective config and whether MLflow should track the command."""

    if command in {"status", "mlflow-ui"}:
        return None, False

    state = runner.load_state()
    effective_config = runner._effective_config(state)
    return effective_config, effective_config.mlflow.enabled


def _launch_or_report_mlflow_ui(tracking: MLflowTrackingConfig) -> dict[str, Any]:
    """Return the MLflow UI command payload."""

    if not tracking.enabled:
        raise ConfigValidationError("MLflow tracking is disabled in the current config.")

    tracking_uri = tracking.tracking_uri
    if importlib.util.find_spec("mlflow") is None:
        return {
            "command": "mlflow-ui",
            "tracking_uri": tracking_uri,
            "experiment_name": tracking.experiment_name,
            "launched": False,
            "launch_command": ["python", "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri],
        }

    return {
        "command": "mlflow-ui",
        "tracking_uri": tracking_uri,
        "experiment_name": tracking.experiment_name,
        "launched": False,
        "launch_command": ["python", "-m", "mlflow", "ui", "--backend-store-uri", tracking_uri],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = None
        if args.config is not None or args.overrides:
            config = load_experiment_config(args.config, args.overrides)
        if args.command == "phase-1":
            sweep_runner = PhaseOneSweepRunner(args.output_dir, base_config=config)
            payload = sweep_runner.execute()
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "phase-2":
            sweep_runner = PhaseTwoSweepRunner(args.output_dir, base_config=config)
            payload = sweep_runner.execute()
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        if args.command == "phase-3":
            sweep_runner = PhaseThreeSweepRunner(args.output_dir, base_config=config)
            payload = sweep_runner.execute()
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0
        store = PipelineStateStore(args.output_dir)
        runner = PipelineRunner(store=store, config=config, run_name=args.run_name)
        tracker = None
        effective_config, should_track = _build_tracking_state(args.command, runner)
        if should_track and effective_config is not None:
            tracker = build_mlflow_tracker(effective_config, run_name=args.run_name)
            tracker.start()
        try:
            payload = _run_pipeline(args.command, runner)
            if tracker is not None:
                tracker.log_payload(payload)
        finally:
            if tracker is not None:
                tracker.close()
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
