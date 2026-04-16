"""Local MLflow tracking helpers for resumable runs."""

import contextlib
import importlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..config import ExperimentConfig, MLflowTrackingConfig
from ..models import DEFAULT_INPUT_BINS, DEFAULT_TARGET_FRAMES, build_model_adapter

os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
os.environ["MLFLOW_RUN_CONTEXT_PROVIDER"] = "sysmetrics"

_EFFICIENCY_CACHE: dict[str, dict[str, float]] = {}
_TRACKING_LOGGER = logging.getLogger(__name__)
_ACTIVE_MLFLOW_RUN_ID_ENV = "SPEECH_MLFLOW_ACTIVE_RUN_ID"
_CHECKPOINT_STEP_PATTERN = re.compile(r"^checkpoint_step_(\d+)\.pt$")


@dataclass(frozen=True, slots=True)
class ReproducibilityReport:
    """Minimal environment snapshot stored alongside each MLflow run."""

    processor: str
    """Processor architecture string."""
    system: str
    """Operating system name."""
    release: str
    """Operating system release string."""
    machine: str
    """Machine architecture string."""
    python_version: str
    """Python interpreter version."""
    python_implementation: str
    """Python implementation name."""
    platform: str
    """Full platform string."""
    torch_version: str
    """PyTorch version string."""
    total_ram_bytes: int
    """Total physical RAM in bytes when detectable."""
    total_disk_bytes: int
    """Total filesystem capacity in bytes for the repository volume."""
    installed_packages: dict[str, str]
    """Installed package version mapping."""
    git_commit_hash: str
    """Current git commit hash."""
    git_branch: str
    """Current git branch or ref name."""
    git_dirty: bool
    """Whether the working tree has uncommitted changes."""
    git_status_porcelain: str
    """Raw porcelain status for the working tree."""
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    """ISO-8601 timestamp when report was created."""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""
        return {
            "processor": self.processor,
            "system": self.system,
            "release": self.release,
            "machine": self.machine,
            "python_version": self.python_version,
            "python_implementation": self.python_implementation,
            "platform": self.platform,
            "torch_version": self.torch_version,
            "total_ram_bytes": self.total_ram_bytes,
            "total_disk_bytes": self.total_disk_bytes,
            "installed_packages": self.installed_packages,
            "git_commit_hash": self.git_commit_hash,
            "git_branch": self.git_branch,
            "git_dirty": self.git_dirty,
            "git_status_porcelain": self.git_status_porcelain,
            "created_at": self.created_at,
        }


def collect_reproducibility_report() -> ReproducibilityReport:
    """Capture the processor, PyTorch version, and current git commit hash."""
    repo_root = Path(__file__).resolve().parents[3]
    git_commit_hash = _read_git_commit_hash(repo_root)
    git_branch = _read_git_branch(repo_root)
    git_status_porcelain = _read_git_status_porcelain(repo_root)
    total_ram_bytes = _read_total_ram_bytes()
    total_disk_bytes = _read_total_disk_bytes(repo_root)

    return ReproducibilityReport(
        processor=platform.processor() or "unknown",
        system=platform.system(),
        release=platform.release(),
        machine=platform.machine(),
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        platform=platform.platform(),
        torch_version=torch.__version__,
        total_ram_bytes=total_ram_bytes,
        total_disk_bytes=total_disk_bytes,
        installed_packages=_cached_installed_packages(),
        git_commit_hash=git_commit_hash,
        git_branch=git_branch,
        git_dirty=bool(git_status_porcelain.strip()),
        git_status_porcelain=git_status_porcelain,
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


def _read_git_branch(repo_root: Path) -> str:
    """Read the current git branch or detached HEAD description."""
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
        return head_contents.removeprefix("ref: ").strip() or "unknown"
    return "detached-head"


def _read_git_status_porcelain(repo_root: Path) -> str:
    """Read the working tree status in porcelain format."""
    try:
        git_executable = shutil.which("git")
        if git_executable is None:
            return ""
        completed = subprocess.run(  # noqa: S603
            [git_executable, "status", "--porcelain=v1", "-uall"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return ""
    return completed.stdout.strip()


def _read_total_ram_bytes() -> int:
    """Read total physical RAM in bytes with platform-aware fallbacks."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if isinstance(pages, int) and isinstance(page_size, int) and pages > 0 and page_size > 0:
            return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        pass

    if platform.system() == "Darwin":
        try:
            sysctl_executable = shutil.which("sysctl")
            if sysctl_executable is None:
                return 0
            completed = subprocess.run(  # noqa: S603
                [sysctl_executable, "-n", "hw.memsize"],
                check=True,
                capture_output=True,
                text=True,
            )
            value = int(completed.stdout.strip())
            if value > 0:
                return value
        except (OSError, ValueError, subprocess.CalledProcessError):
            pass

    return 0


def _read_total_disk_bytes(repo_root: Path) -> int:
    """Read total filesystem capacity for the repository volume in bytes."""
    try:
        return int(shutil.disk_usage(repo_root).total)
    except OSError:
        return 0


def _collect_installed_packages() -> dict[str, str]:
    """Collect installed package versions for the active Python environment."""
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not name or not version:
            continue
        packages[name] = version
    return dict(sorted(packages.items(), key=lambda item: item[0].lower()))


@lru_cache(maxsize=1)
def _cached_installed_packages() -> dict[str, str]:
    """Return a cached package-version snapshot for the active environment."""
    return _collect_installed_packages()


def _resolve_tracking_uri(tracking_uri: str) -> str:
    """Resolve local tracking URIs to an absolute path when needed."""
    if tracking_uri.startswith("sqlite:"):
        sqlite_target = tracking_uri.removeprefix("sqlite:")
        if sqlite_target.lstrip("/") == ":memory:":
            return "sqlite:///:memory:"

        if sqlite_target.startswith("///"):
            db_target = sqlite_target[3:]
        elif sqlite_target.startswith("//"):
            db_target = sqlite_target[2:]
        elif sqlite_target.startswith("/"):
            db_target = sqlite_target[1:]
        else:
            db_target = sqlite_target

        if not db_target:
            db_target = "mlruns.db"

        resolved_db_path = Path(db_target).expanduser().resolve()
        resolved_db_path.parent.mkdir(parents=True, exist_ok=True)
        return f"sqlite:///{resolved_db_path.as_posix()}"

    if "://" in tracking_uri and not tracking_uri.startswith("file:"):
        return tracking_uri

    resolved_path = Path(tracking_uri).expanduser().resolve()
    resolved_path.mkdir(parents=True, exist_ok=True)
    return str(resolved_path)


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
    """MLflow tracking configuration."""
    experiment_config: ExperimentConfig
    """Experiment configuration being tracked."""
    model_adapter: Any
    """Model adapter for efficiency profiling."""
    run_name: str
    """Name identifier for this MLflow run."""
    _mlflow: Any = field(default=None, init=False, repr=False)
    """MLflow module reference (lazy-loaded)."""
    _run_active: bool = field(default=False, init=False, repr=False)
    """Whether MLflow run is currently active."""
    _logged_checkpoint_paths: set[str] = field(default_factory=set, init=False, repr=False)
    """Checkpoint artifact paths already logged for this run."""
    _logged_checkpoint_artifacts: set[str] = field(default_factory=set, init=False, repr=False)
    """Checkpoint artifact events already logged for this run."""
    _latest_checkpoint_tag_value: str | None = field(default=None, init=False, repr=False)
    """Last checkpoint path tag value sent to MLflow."""
    _logged_checkpoint_artifact_value: str | None = field(default=None, init=False, repr=False)
    """Last checkpoint artifact pointer tag value sent to MLflow."""
    _active_run_id: str | None = field(default=None, init=False, repr=False)
    """Active MLflow run id when a run is open."""
    _best_validation_macro_f1: float | None = field(default=None, init=False, repr=False)
    """Best validation macro-F1 observed for this run."""
    _best_checkpoint_path: str | None = field(default=None, init=False, repr=False)
    """Local path for the best checkpoint observed in this run."""
    _best_checkpoint_artifact: str | None = field(default=None, init=False, repr=False)
    """MLflow artifact path for the best checkpoint observed in this run."""

    def start(self) -> None:
        """Start a run and log static hyperparameters and efficiency metrics."""
        if not self.tracking.enabled:
            return

        try:
            mlflow = importlib.import_module("mlflow")
            tracking_uri = _resolve_tracking_uri(self.tracking.tracking_uri)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="The filesystem tracking backend .*",
                    category=FutureWarning,
                )
                mlflow.set_tracking_uri(tracking_uri)
                mlflow.set_experiment(self.tracking.experiment_name)
                resume_run_id = os.environ.get(_ACTIVE_MLFLOW_RUN_ID_ENV)
                if resume_run_id:
                    try:
                        mlflow.start_run(run_id=resume_run_id)
                    except Exception:
                        _TRACKING_LOGGER.warning(
                            "Failed to resume MLflow run_id=%s; starting a new run.",
                            resume_run_id,
                        )
                        mlflow.start_run(run_name=self.run_name or self.tracking.run_name)
                else:
                    mlflow.start_run(run_name=self.run_name or self.tracking.run_name)
            self._mlflow = mlflow
            self._run_active = True
            active_run = mlflow.active_run()
            if active_run is not None:
                self._active_run_id = active_run.info.run_id
                os.environ[_ACTIVE_MLFLOW_RUN_ID_ENV] = active_run.info.run_id
            mlflow.set_tag("pipeline.run_name", self.run_name)

            if self.tracking.log_params:
                mlflow.log_params(_flatten_params("", self.experiment_config.to_dict()))

            efficiency_key = json.dumps(self.experiment_config.model.to_dict(), sort_keys=True)
            efficiency = _EFFICIENCY_CACHE.get(efficiency_key)
            if efficiency is None:
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
                _EFFICIENCY_CACHE[efficiency_key] = {
                    "model_parameters_total": float(efficiency["model_parameters_total"]),
                    "model_macs_1sec": float(efficiency["model_macs_1sec"]),
                }

            mlflow.set_tag("model_parameters_total", float(efficiency["model_parameters_total"]))
            mlflow.set_tag("model_macs_1sec", float(efficiency["model_macs_1sec"]))

            self._log_reproducibility_report()
        except Exception:
            self._run_active = False
            self._mlflow = None

    def _log_reproducibility_report(self) -> None:
        """Persist the reproducibility snapshot as an MLflow artifact."""
        report = collect_reproducibility_report()
        self._log_json_artifact("reproducibility_report.json", report.to_dict(), force=True)

    def _log_json_artifact(
        self,
        artifact_name: str,
        payload: Mapping[str, Any],
        *,
        force: bool = False,
    ) -> None:
        """Persist a JSON payload via MLflow, or print it to console."""
        if not force and not self.tracking.log_artifacts:
            return

        if self._run_active and self._mlflow is not None:
            log_dict = getattr(self._mlflow, "log_dict", None)
            if callable(log_dict):
                log_dict(dict(payload), artifact_name)
                return
            self._mlflow.log_text(
                json.dumps(payload, indent=2, sort_keys=True),
                artifact_name,
            )
            return

        _TRACKING_LOGGER.info(
            "MLflow inactive; %s payload follows:\n%s",
            artifact_name,
            json.dumps(payload, indent=2, sort_keys=True),
        )

    def log_training_metrics(
        self,
        loss: float | None = None,
        validation_macro_f1: float | None = None,
        checkpoint_path: str | Path | None = None,
        epoch: int | None = None,
        step: int | None = None,
        extra_metrics: Mapping[str, float] | None = None,
        metric_phase: str | None = None,
    ) -> None:
        """Log dynamic training metrics and checkpoint metadata."""
        if (
            loss is None
            and validation_macro_f1 is None
            and checkpoint_path is None
            and epoch is None
            and step is None
            and not extra_metrics
        ):
            return

        metrics: dict[str, float] = {}
        if loss is not None:
            metrics["loss"] = float(loss)
        if validation_macro_f1 is not None:
            metrics["validation_macro_f1"] = float(validation_macro_f1)
        if extra_metrics is not None:
            metrics.update({key: float(value) for key, value in extra_metrics.items()})

        normalized_metric_phase = _normalize_metric_phase(metric_phase)
        metrics = _normalize_metric_names(metrics, metric_phase=normalized_metric_phase)

        if not self._run_active or self._mlflow is None:
            if metrics:
                _TRACKING_LOGGER.info(
                    "MLflow inactive; metrics step=%s: %s",
                    step,
                    json.dumps(metrics, sort_keys=True),
                )
            if epoch is not None:
                _TRACKING_LOGGER.info("MLflow inactive; epoch=%s step=%s", epoch, step)
            if checkpoint_path is not None:
                _TRACKING_LOGGER.info(
                    "MLflow inactive; latest checkpoint: %s",
                    str(checkpoint_path),
                )
            return

        if self.tracking.log_metrics and metrics:
            self._mlflow.log_metrics(metrics, step=step)

        if self.tracking.log_metrics and epoch is not None:
            self._mlflow.log_metric("epoch", float(epoch), step=step)
        if self.tracking.log_metrics and step is not None:
            self._mlflow.log_metric("step", float(step), step=step)

        if checkpoint_path is not None:
            checkpoint = Path(checkpoint_path)
            checkpoint_step = _checkpoint_step_from_path(checkpoint)
            checkpoint_artifact = _checkpoint_artifact_path(
                checkpoint,
                keep_last_n=self.experiment_config.checkpointing.keep_last_n,
                step=checkpoint_step,
            )
            # Use a mutable tag for "latest" pointer; params are immutable in MLflow.
            checkpoint_key = str(checkpoint)
            if checkpoint_key != self._latest_checkpoint_tag_value:
                self._mlflow.set_tag("latest_checkpoint_path", checkpoint_key)
                self._latest_checkpoint_tag_value = checkpoint_key
            if checkpoint_step is not None:
                self._mlflow.set_tag("latest_checkpoint_step", int(checkpoint_step))
            checkpoint_key = str(checkpoint)
            artifact_log_key = (
                f"{checkpoint_key}::{checkpoint_artifact}::"
                f"{checkpoint_step if checkpoint_step is not None else 'na'}"
            )
            if (
                self.tracking.log_artifacts
                and checkpoint.exists()
                and artifact_log_key not in self._logged_checkpoint_artifacts
            ):
                artifact_parent = str(Path(checkpoint_artifact).parent)
                target_artifact_name = Path(checkpoint_artifact).name
                if checkpoint.name == target_artifact_name:
                    self._mlflow.log_artifact(
                        str(checkpoint),
                        artifact_path=artifact_parent,
                    )
                else:
                    with tempfile.TemporaryDirectory(dir=str(checkpoint.parent)) as temporary_dir:
                        temporary_checkpoint = Path(temporary_dir) / target_artifact_name
                        try:
                            os.link(checkpoint, temporary_checkpoint)
                        except OSError:
                            shutil.copy2(checkpoint, temporary_checkpoint)
                        self._mlflow.log_artifact(
                            str(temporary_checkpoint),
                            artifact_path=artifact_parent,
                        )
                if checkpoint_artifact != self._logged_checkpoint_artifact_value:
                    self._mlflow.set_tag("latest_checkpoint_artifact", checkpoint_artifact)
                    self._logged_checkpoint_artifact_value = checkpoint_artifact
                self._logged_checkpoint_artifacts.add(artifact_log_key)
                self._logged_checkpoint_paths.add(checkpoint_key)

            if validation_macro_f1 is not None:
                self._update_best_checkpoint(
                    validation_macro_f1=float(validation_macro_f1),
                    checkpoint_path=checkpoint_key,
                    checkpoint_artifact=(
                        checkpoint_artifact
                        if self.tracking.log_artifacts and checkpoint.exists()
                        else None
                    ),
                )

    def _update_best_checkpoint(
        self,
        validation_macro_f1: float,
        checkpoint_path: str,
        checkpoint_artifact: str | None,
    ) -> None:
        """Update best-checkpoint tags when a better validation score is observed."""
        current_best = self._best_validation_macro_f1
        if current_best is not None:
            if validation_macro_f1 < current_best:
                return
            # Promote dedicated best-checkpoint snapshots when score ties current best.
            if (
                validation_macro_f1 == current_best
                and Path(checkpoint_path).name != "checkpoint_best.pt"
            ):
                return

        self._best_validation_macro_f1 = validation_macro_f1
        self._best_checkpoint_path = checkpoint_path
        self._best_checkpoint_artifact = checkpoint_artifact

        self._mlflow.set_tag("best_validation_macro_f1", float(validation_macro_f1))
        self._mlflow.set_tag("best_checkpoint_path", checkpoint_path)
        if checkpoint_artifact is not None:
            self._mlflow.set_tag("best_checkpoint_artifact", checkpoint_artifact)

    def _register_best_model(self) -> None:
        """Register the best checkpoint artifact as an MLflow model version."""
        if (
            not self._run_active
            or self._mlflow is None
            or self._active_run_id is None
            or self._best_checkpoint_artifact is None
        ):
            return

        model_name = _model_registry_name(self.tracking.experiment_name)
        model_source = self._mlflow.get_artifact_uri(self._best_checkpoint_artifact)
        client = self._mlflow.tracking.MlflowClient()
        try:
            with contextlib.suppress(Exception):
                client.create_registered_model(model_name)
            model_version = client.create_model_version(
                name=model_name,
                source=model_source,
                run_id=self._active_run_id,
            )
        except Exception as exc:
            _TRACKING_LOGGER.warning("Failed to register best model '%s': %s", model_name, exc)
            return

        self._mlflow.set_tag("best_model_uri", model_source)
        self._mlflow.set_tag("best_registered_model_name", model_name)
        self._mlflow.set_tag("best_registered_model_version", str(model_version.version))

    def _cleanup_logged_checkpoints(self) -> None:
        """Delete uploaded local checkpoint files when retention is disabled."""
        if self.tracking.retain_local_checkpoints:
            return
        for checkpoint_path in self._logged_checkpoint_paths:
            checkpoint = Path(checkpoint_path)
            with contextlib.suppress(OSError):
                checkpoint.unlink()

            # Best-effort pruning of empty checkpoint directories left behind.
            for parent in checkpoint.parents:
                with contextlib.suppress(OSError):
                    parent.rmdir()

    def log_named_json_artifact(self, artifact_name: str, payload: Mapping[str, Any]) -> None:
        """Public helper for logging a JSON artifact through the active tracker."""
        self._log_json_artifact(artifact_name, payload)

    def log_payload(self, payload: Mapping[str, Any]) -> None:
        """Recursively log any training-style fields found in a payload."""
        metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
        checkpoint_path = payload.get("checkpoint_path")
        epoch = payload.get("epoch")
        step = payload.get("step")
        loss = payload.get("loss")
        validation_macro_f1 = payload.get("validation_macro_f1")
        metric_phase = _infer_metric_phase(payload)

        flattened_metrics = _flatten_numeric_metrics(metrics) if metrics else {}
        flattened_metrics.update(_extract_phase_performance_metrics(payload))

        self.log_training_metrics(
            loss=float(loss) if loss is not None else None,
            validation_macro_f1=(
                float(validation_macro_f1) if validation_macro_f1 is not None else None
            ),
            checkpoint_path=checkpoint_path,
            epoch=int(epoch) if epoch is not None else None,
            step=int(step) if step is not None else None,
            extra_metrics=flattened_metrics if flattened_metrics else None,
            metric_phase=metric_phase,
        )

        decisions_payload = _extract_decisions(payload)
        if decisions_payload:
            self._log_json_artifact("decisions.json", decisions_payload)
        if self.tracking.log_artifacts:
            self._log_json_artifact("run_payload.json", dict(payload))

    def close(self) -> None:
        """End the active MLflow run if one was started."""
        if self._run_active and self._mlflow is not None:
            try:
                self._register_best_model()
                self._mlflow.end_run()
            finally:
                self._cleanup_logged_checkpoints()
                self._run_active = False
                self._mlflow = None
                os.environ.pop(_ACTIVE_MLFLOW_RUN_ID_ENV, None)
                self._logged_checkpoint_paths.clear()
                self._logged_checkpoint_artifacts.clear()
                self._latest_checkpoint_tag_value = None
                self._logged_checkpoint_artifact_value = None
                self._active_run_id = None
                self._best_validation_macro_f1 = None
                self._best_checkpoint_path = None
                self._best_checkpoint_artifact = None


def build_mlflow_tracker(experiment_config: ExperimentConfig, run_name: str) -> MlflowRunTracker:
    """Construct a tracker using the current experiment configuration."""
    model_adapter = build_model_adapter(
        family=experiment_config.model.family,
        num_classes=experiment_config.model.num_classes,
        pretrained=experiment_config.model.pretrained,
        model_config=experiment_config.model,
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


def _normalize_metric_phase(value: str | None) -> str | None:
    """Normalize metric phase aliases to stable prefixes used in metric names."""
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"train", "training"}:
        return "training"
    if normalized in {"val", "valid", "validation"}:
        return "val"
    if normalized == "test":
        return "test"
    return None


def _infer_metric_phase(payload: Mapping[str, Any]) -> str | None:
    """Infer metric phase from payload metadata or prediction artifact naming."""
    declared = payload.get("metric_phase")
    if isinstance(declared, str):
        normalized = _normalize_metric_phase(declared)
        if normalized is not None:
            return normalized

    prediction_artifact = payload.get("prediction_artifact")
    if isinstance(prediction_artifact, str):
        if "test_predictions" in prediction_artifact:
            return "test"
        if "validation_predictions" in prediction_artifact:
            return "val"

    return None


def _normalize_metric_name(metric_name: str, metric_phase: str | None) -> str:
    """Convert metric names to phase-aware naming (training/val/test)."""
    if metric_name == "loss":
        if metric_phase in {"val", "test"}:
            return f"{metric_phase}_loss"
        return "training_loss"
    if metric_name.startswith("validation_"):
        metric_name = f"val_{metric_name.removeprefix('validation_')}"
    elif metric_name.startswith("train_"):
        metric_name = f"training_{metric_name.removeprefix('train_')}"

    if (
        metric_phase is not None
        and "." not in metric_name
        and not metric_name.startswith(("training_", "val_", "test_"))
        and metric_name not in {"epoch", "step"}
    ):
        return f"{metric_phase}_{metric_name}"

    return metric_name


def _normalize_metric_names(
    metrics: Mapping[str, float],
    metric_phase: str | None,
) -> dict[str, float]:
    """Normalize all metric names to include explicit phase semantics."""
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        normalized[_normalize_metric_name(key, metric_phase)] = float(value)
    return normalized


def _extract_phase_performance_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    """Extract runtime/resource metrics from top-level and phase outputs."""
    metrics: dict[str, float] = {}

    direct_performance = payload.get("performance")
    if isinstance(direct_performance, Mapping):
        metrics.update(_flatten_numeric_metrics(direct_performance, prefix="runtime"))

    training_performance = payload.get("training_performance")
    if isinstance(training_performance, Mapping):
        metrics.update(_flatten_numeric_metrics(training_performance, prefix="training"))

    phase_artifacts = payload.get("phase_artifacts")
    if isinstance(phase_artifacts, Mapping):
        for phase_name, artifact in phase_artifacts.items():
            if not isinstance(artifact, Mapping):
                continue
            output_data = artifact.get("output_data")
            if not isinstance(output_data, Mapping):
                continue
            perf = output_data.get("performance")
            if isinstance(perf, Mapping):
                metrics.update(_flatten_numeric_metrics(perf, prefix=f"{phase_name}.performance"))
            phase_metrics = output_data.get("metrics")
            if isinstance(phase_metrics, Mapping):
                metrics.update(
                    _flatten_numeric_metrics(phase_metrics, prefix=f"{phase_name}.metrics")
                )

    return metrics


def _extract_decisions(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract high-level model selection decisions from run payload."""
    keys = (
        "phase",
        "best_trial",
        "best_trial_id",
        "best_validation_macro_f1",
        "best_macro_f1_nc",
        "best_aggregate",
        "feature_source",
        "optimizer_source",
        "top_three",
        "family_winners",
        "method_winners",
        "selected_test_results",
        "ensemble_results",
    )
    return {key: payload[key] for key in keys if key in payload}


def _checkpoint_step_from_path(checkpoint: Path) -> int | None:
    """Extract a training step from a checkpoint filename when available."""
    match = _CHECKPOINT_STEP_PATTERN.match(checkpoint.name)
    if match is None:
        return None
    return int(match.group(1))


def _checkpoint_artifact_path(
    checkpoint: Path,
    keep_last_n: int,
    step: int | None,
) -> str:
    """Return a stable MLflow artifact path for a checkpoint file."""
    component = checkpoint.parent.name
    if component in {"gate", "command", "non_command", "shared_two_head"}:
        prefix = f"checkpoints/{component}"
    else:
        prefix = "checkpoints"

    if checkpoint.name == "checkpoint_best.pt":
        return f"{prefix}/{checkpoint.name}"

    if step is None:
        return f"{prefix}/{checkpoint.name}"

    slot = step % max(1, keep_last_n)
    return f"{prefix}/rolling/checkpoint_slot_{slot}.pt"


def _model_registry_name(experiment_name: str) -> str:
    """Return a registry-safe model name for best-checkpoint registration."""
    sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "-", experiment_name).strip("-._")
    if not sanitized:
        sanitized = "speech-recognition"
    return f"{sanitized}-best"
