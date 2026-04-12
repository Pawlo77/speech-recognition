"""Local MLflow tracking helpers for resumable runs."""

import importlib
import json
import logging
import platform
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig, MLflowTrackingConfig
from ..models import DEFAULT_INPUT_BINS, DEFAULT_TARGET_FRAMES, build_model_adapter


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    """Minimal environment snapshot stored alongside each MLflow run."""

    processor: str
    torch_version: str
    git_commit_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-serializable representation of the report."""

        return {
            "processor": self.processor,
            "torch_version": self.torch_version,
            "git_commit_hash": self.git_commit_hash,
            "created_at": self.created_at,
        }


def collect_reproducibility_report() -> ReproducibilityReport:
    """Capture the processor, PyTorch version, and current git commit hash."""

    git_commit_hash = _read_git_commit_hash(Path(__file__).resolve().parents[3])

    return ReproducibilityReport(
        processor=platform.processor() or "unknown",
        torch_version=torch.__version__,
        git_commit_hash=git_commit_hash,
    )


def _read_git_commit_hash(repo_root: Path) -> str:
    """Read the current commit hash from the local git metadata."""

    git_dir = repo_root / ".git"
    head_path = git_dir / "HEAD"
    if git_dir.is_file():
        git_dir_text = git_dir.read_text(encoding="utf-8").strip()
        if git_dir_text.startswith("gitdir:"):
            git_dir = (repo_root / git_dir_text.split("gitdir:", maxsplit=1)[1].strip()).resolve()
            head_path = git_dir / "HEAD"

    try:
        head_contents = head_path.read_text(encoding="utf-8").strip()
    except OSError:
        return "unknown"

    if head_contents.startswith("ref: "):
        ref_name = head_contents.removeprefix("ref: ").strip()
        ref_path = git_dir / ref_name
        try:
            return ref_path.read_text(encoding="utf-8").strip() or "unknown"
        except OSError:
            return "unknown"

    return head_contents or "unknown"


def _resolve_tracking_uri(tracking_uri: str) -> str:
    """Resolve local tracking URIs to an absolute path when needed."""

    if "://" in tracking_uri and not tracking_uri.startswith("file:"):
        return tracking_uri
    return str(Path(tracking_uri).expanduser().resolve())


def _flatten_params(prefix: str, value: Any) -> dict[str, str]:
    """Flatten nested config values into MLflow-compatible parameter strings."""

    flattened: dict[str, str] = {}
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_flatten_params(next_prefix, nested_value))
        return flattened

    if isinstance(value, tuple):
        flattened[prefix] = json.dumps(list(value))
    elif isinstance(value, list):
        flattened[prefix] = json.dumps(value)
    elif isinstance(value, bool):
        flattened[prefix] = "true" if value else "false"
    elif value is None:
        flattened[prefix] = "null"
    else:
        flattened[prefix] = str(value)
    return flattened


@dataclass(slots=True)
class MlflowRunTracker:
    """Track a single run with MLflow while tolerating offline/local-only setups."""

    tracking: MLflowTrackingConfig
    experiment_config: ExperimentConfig
    model_adapter: Any
    run_name: str
    _mlflow: Any = field(default=None, init=False, repr=False)
    _run_active: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        """Start a run and log static hyperparameters and efficiency metrics."""

        if not self.tracking.enabled:
            return

        try:
            mlflow = importlib.import_module("mlflow")
            tracking_uri = _resolve_tracking_uri(self.tracking.tracking_uri)
            Path(tracking_uri).mkdir(parents=True, exist_ok=True)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The filesystem tracking backend .*",
                    category=FutureWarning,
                )
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.tracking.experiment_name)
                mlflow.start_run(run_name=self.run_name or self.tracking.run_name)
            self._mlflow = mlflow
            self._run_active = True

            if self.tracking.log_params:
                mlflow.log_params(_flatten_params("", self.experiment_config.to_dict()))

            fvcore_logger = logging.getLogger("fvcore.nn.jit_analysis")
            previous_level = fvcore_logger.level
            fvcore_logger.setLevel(logging.ERROR)
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="`torch.jit.script` is deprecated.*",
                        category=DeprecationWarning,
                    )
                    efficiency = self.model_adapter.profile_efficiency(
                        torch.randn(
                            1,
                            1,
                            DEFAULT_INPUT_BINS,
                            DEFAULT_TARGET_FRAMES,
                            dtype=torch.float32,
                        )
                    )
            finally:
                fvcore_logger.setLevel(previous_level)

            if self.tracking.log_metrics:
                mlflow.log_metrics(
                    {
                        "model_parameters_total": float(efficiency["model_parameters_total"]),
                        "model_macs_1sec": float(efficiency["model_macs_1sec"]),
                    },
                    step=0,
                )

            self._log_reproducibility_report(mlflow)
        except Exception:
            self._run_active = False
            self._mlflow = None

    def _log_reproducibility_report(self, mlflow: Any) -> None:
        """Persist the reproducibility snapshot as an MLflow artifact."""

        report = collect_reproducibility_report()
        with tempfile.TemporaryDirectory() as temporary_dir:
            report_path = Path(temporary_dir) / "reproducibility_report.json"
            report_path.write_text(
                json.dumps(report.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(report_path))

    def log_training_metrics(
        self,
        *,
        loss: float | None = None,
        validation_macro_f1: float | None = None,
        checkpoint_path: str | Path | None = None,
        epoch: int | None = None,
        step: int | None = None,
        extra_metrics: Mapping[str, float] | None = None,
    ) -> None:
        """Log dynamic training metrics and checkpoint metadata."""

        if not self._run_active or self._mlflow is None:
            return

        metrics: dict[str, float] = {}
        if loss is not None:
            metrics["loss"] = float(loss)
        if validation_macro_f1 is not None:
            metrics["validation_macro_f1"] = float(validation_macro_f1)
        if extra_metrics is not None:
            metrics.update({key: float(value) for key, value in extra_metrics.items()})

        if self.tracking.log_metrics and metrics:
            self._mlflow.log_metrics(metrics, step=step)

        if epoch is not None:
            self._mlflow.log_metric("epoch", float(epoch), step=step)
        if step is not None:
            self._mlflow.log_metric("step", float(step), step=step)

        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path)
            self._mlflow.log_param("checkpoint_path", str(checkpoint))
            if self.tracking.log_artifacts and checkpoint.exists():
                self._mlflow.log_artifact(str(checkpoint))

    def log_payload(self, payload: Mapping[str, Any]) -> None:
        """Recursively log any training-style fields found in a payload."""

        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        checkpoint_path = payload.get("checkpoint_path")
        epoch = payload.get("epoch")
        step = payload.get("step")
        loss = payload.get("loss")
        validation_macro_f1 = payload.get("validation_macro_f1")

        flattened_metrics = _flatten_numeric_metrics(metrics) if metrics else {}

        self.log_training_metrics(
            loss=float(loss) if loss is not None else None,
            validation_macro_f1=(
                float(validation_macro_f1) if validation_macro_f1 is not None else None
            ),
            checkpoint_path=checkpoint_path,
            epoch=int(epoch) if epoch is not None else None,
            step=int(step) if step is not None else None,
            extra_metrics=flattened_metrics if flattened_metrics else None,
        )

    def close(self) -> None:
        """End the active MLflow run if one was started."""

        if self._run_active and self._mlflow is not None:
            try:
                self._mlflow.end_run()
            finally:
                self._run_active = False
                self._mlflow = None


def build_mlflow_tracker(experiment_config: ExperimentConfig, run_name: str) -> MlflowRunTracker:
    """Construct a tracker using the current experiment configuration."""

    model_adapter = build_model_adapter(
        family=experiment_config.model.family,
        num_classes=experiment_config.model.num_classes,
        pretrained=experiment_config.model.pretrained,
    )
    return MlflowRunTracker(
        tracking=experiment_config.mlflow,
        experiment_config=experiment_config,
        model_adapter=model_adapter,
        run_name=run_name,
    )


def _flatten_numeric_metrics(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    """Flatten nested metric mappings into numeric leaf metrics."""

    flattened: dict[str, float] = {}
    for key, item in value.items():
        metric_name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_numeric_metrics(item, prefix=metric_name))
        elif isinstance(item, int | float) and not isinstance(item, bool):
            flattened[metric_name] = float(item)
    return flattened
