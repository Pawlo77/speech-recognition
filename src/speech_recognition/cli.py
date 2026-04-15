"""Command-line interface for the resumable experiment pipeline."""

import argparse
import importlib.util
import json
import tempfile
from collections.abc import Sequence
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any

from .config import (
    ConfigValidationError,
    ExperimentConfig,
    MLflowTrackingConfig,
    SchedulerConfig,
    TrainingControlConfig,
)
from .orchestration import PipelineRunner, PipelineStateStore
from .orchestration.phase_four import PhaseFourSweepRunner
from .orchestration.phase_one import PhaseOneSweepRunner
from .orchestration.phase_three import PhaseThreeSweepRunner
from .orchestration.phase_two import PhaseTwoSweepRunner
from .orchestration.runtime import execute_single_eval, execute_single_train

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
    config = _default_experiment_config()
    if config_path is not None:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        config = ExperimentConfig.from_dict(payload)

    for override in overrides:
        key, separator, raw_value = override.partition("=")
        if not separator:
            raise ConfigValidationError(f"Invalid override '{override}'. Expected KEY=VALUE.")
        config = _apply_override(config, key, _parse_override_value(raw_value))

    return config


def _default_experiment_config() -> ExperimentConfig:
    """Build a default config with scheduler/training epoch consistency."""
    training = TrainingControlConfig()
    scheduler = SchedulerConfig(total_epochs=training.epochs)
    return ExperimentConfig(training=training, scheduler=scheduler)


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
    parent.add_argument(
        "--mlflow-only",
        action="store_true",
        help=(
            "Use a temporary local run directory and persist artifacts/state only via MLflow. "
            "Requires mlflow.enabled=true."
        ),
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


def _phase_status_summary(state: Any, score_field: str) -> dict[str, Any]:
    """Build a compact status payload from one phase state object."""
    completed_trials = getattr(state, "completed_trials", {})
    return {
        "completed_trials": len(completed_trials),
        "best_trial_id": getattr(state, "best_trial_id", None),
        score_field: getattr(state, score_field, None),
        "updated_at": getattr(state, "updated_at", None),
    }


def _run_sweep_pipeline(
    output_dir: Path,
    config: ExperimentConfig | None,
    run_name: str,
    include_phase_four: bool,
) -> dict[str, Any]:
    """Execute real phase sweeps in order, reusing sweep-resume semantics."""
    phase_one = PhaseOneSweepRunner(output_dir, base_config=config)
    phase_two = PhaseTwoSweepRunner(output_dir, base_config=config)
    phase_three = PhaseThreeSweepRunner(output_dir, base_config=config)

    phase_payloads: dict[str, dict[str, Any]] = {
        "phase-1": phase_one.execute(),
        "phase-2": phase_two.execute(),
        "phase-3": phase_three.execute(),
    }
    if include_phase_four:
        phase_four = PhaseFourSweepRunner(output_dir, base_config=config)
        phase_payloads["phase-4"] = phase_four.execute()

    completed_phases = [
        phase
        for phase, payload in phase_payloads.items()
        if int(payload.get("completed_trials", 0)) > 0
    ]
    return {
        "command": "run" if include_phase_four else "train",
        "run_name": run_name,
        "output_dir": str(output_dir),
        "completed_phases": completed_phases,
        "phases": phase_payloads,
    }


def _build_sweep_status_payload(
    output_dir: Path,
    config: ExperimentConfig | None,
    run_name: str,
) -> dict[str, Any]:
    """Build pipeline status from persisted sweep state files."""
    phase_one = PhaseOneSweepRunner(output_dir, base_config=config)
    phase_two = PhaseTwoSweepRunner(output_dir, base_config=config)
    phase_three = PhaseThreeSweepRunner(output_dir, base_config=config)

    phase_payloads: dict[str, dict[str, Any]] = {
        "phase-1": _phase_status_summary(
            phase_one.load_state(),
            score_field="best_validation_macro_f1",
        ),
        "phase-2": _phase_status_summary(
            phase_two.load_state(),
            score_field="best_validation_macro_f1",
        ),
        "phase-3": _phase_status_summary(
            phase_three.load_state(),
            score_field="best_validation_macro_f1",
        ),
    }

    try:
        phase_four = PhaseFourSweepRunner(output_dir, base_config=config)
        phase_payloads["phase-4"] = _phase_status_summary(
            phase_four.load_state(),
            score_field="best_macro_f1_nc",
        )
    except (FileNotFoundError, ValueError) as exc:
        phase_payloads["phase-4"] = {
            "completed_trials": 0,
            "best_trial_id": None,
            "best_macro_f1_nc": None,
            "updated_at": None,
            "blocked": str(exc),
        }

    completed_phases = [
        phase
        for phase, payload in phase_payloads.items()
        if int(payload.get("completed_trials", 0)) > 0
    ]
    return {
        "command": "status",
        "run_name": run_name,
        "output_dir": str(output_dir),
        "completed_phases": completed_phases,
        "phases": phase_payloads,
    }


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
        effective_config = config or _default_experiment_config()
        output_dir = args.output_dir
        temporary_output_dir: tempfile.TemporaryDirectory[str] | None = None

        if args.mlflow_only:
            if not effective_config.mlflow.enabled:
                raise ConfigValidationError(
                    "--mlflow-only requires mlflow.enabled=true in the active config."
                )
            temporary_output_dir = tempfile.TemporaryDirectory(prefix="speech-recognition-mlflow-")
            output_dir = Path(temporary_output_dir.name)

        if args.command == "run-single-train":
            payload = execute_single_train(effective_config, output_dir, args.run_name)
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command == "run-single-eval":
            payload = execute_single_eval(effective_config, output_dir, args.run_name)
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command == "phase-1":
            sweep_runner = PhaseOneSweepRunner(output_dir, base_config=effective_config)
            payload = sweep_runner.execute()
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command == "phase-2":
            sweep_runner = PhaseTwoSweepRunner(output_dir, base_config=effective_config)
            payload = sweep_runner.execute()
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command == "phase-3":
            sweep_runner = PhaseThreeSweepRunner(output_dir, base_config=effective_config)
            payload = sweep_runner.execute()
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command in {"phase-4", "eval"}:
            sweep_runner = PhaseFourSweepRunner(output_dir, base_config=effective_config)
            payload = sweep_runner.execute()
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command in {"run", "resume"}:
            payload = _run_sweep_pipeline(
                output_dir=output_dir,
                config=effective_config,
                run_name=args.run_name,
                include_phase_four=True,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command == "train":
            payload = _run_sweep_pipeline(
                output_dir=output_dir,
                config=effective_config,
                run_name=args.run_name,
                include_phase_four=False,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command == "status":
            payload = _build_sweep_status_payload(
                output_dir=output_dir,
                config=effective_config,
                run_name=args.run_name,
            )
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0
        if args.command == "mlflow-ui":
            store_config = effective_config
            store = PipelineStateStore(
                output_dir,
                use_mlflow=store_config.mlflow.enabled,
                tracking_uri=store_config.mlflow.tracking_uri,
                experiment_name=store_config.mlflow.experiment_name,
            )
            runner = PipelineRunner(store=store, config=config, run_name=args.run_name)
            state = runner.load_state()
            tracking = runner._effective_config(state).mlflow
            payload = _launch_or_report_mlflow_ui(tracking)
            print(json.dumps(payload, indent=2, sort_keys=True))
            if temporary_output_dir is not None:
                temporary_output_dir.cleanup()
            return 0

        raise ConfigValidationError(f"Unknown command '{args.command}'.")
    except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
