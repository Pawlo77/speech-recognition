"""Explicit epoch/step training loop with resumable checkpoints."""

import contextlib
import logging
import os
import pickle
import random
import re
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm

from ..config import TrainingControlConfig

_CHECKPOINT_PATTERN = re.compile(r"^checkpoint_step_(\d+)$")
_TRAIN_MAX_STEPS_ENV = "SPEECH_TRAIN_MAX_STEPS"


def _max_train_steps_override() -> int | None:
    """Return an optional max-step override used for smoke checks."""

    raw_limit = os.environ.get(_TRAIN_MAX_STEPS_ENV)
    if raw_limit in {None, ""}:
        return None
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise ValueError(f"{_TRAIN_MAX_STEPS_ENV} must be an integer.") from exc
    if limit <= 0:
        raise ValueError(f"{_TRAIN_MAX_STEPS_ENV} must be greater than zero.")
    return limit


@dataclass(frozen=True, slots=True)
class TrainingCheckpoint:
    """Serializable checkpoint payload for exact training resumption."""

    model_state_dict: dict[str, Any]
    """Model parameter state dictionary."""
    optimizer_state_dict: dict[str, Any]
    """Optimizer state dictionary."""
    scheduler_state_dict: dict[str, Any] | None
    """Scheduler state dictionary or None if no scheduler."""
    rng_state: dict[str, Any]
    """Random number generator states for Python, NumPy, and PyTorch."""
    epoch: int
    """Epoch number when checkpoint was saved."""
    step: int
    """Total training step number when checkpoint was saved."""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dictionary suitable for torch.save."""

        return {
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "rng_state": self.rng_state,
            "epoch": self.epoch,
            "step": self.step,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainingCheckpoint":
        """Build a checkpoint from a loaded payload."""

        return cls(
            model_state_dict=dict(payload["model_state_dict"]),
            optimizer_state_dict=dict(payload["optimizer_state_dict"]),
            scheduler_state_dict=payload.get("scheduler_state_dict"),
            rng_state=dict(payload["rng_state"]),
            epoch=int(payload["epoch"]),
            step=int(payload["step"]),
        )


@dataclass(slots=True)
class TrainingEngine:
    """Train and resume a model on MPS when available, otherwise on CPU."""

    model: nn.Module
    """PyTorch model to train."""
    optimizer: torch.optim.Optimizer
    """Optimizer for updating model parameters."""
    scheduler: Any
    """Learning rate scheduler or None."""
    training_config: TrainingControlConfig
    """Training hyperparameter configuration."""
    checkpoint_dir: Path
    """Directory for storing training checkpoints."""
    keep_last_n: int = 2
    """Number of recent checkpoints to retain on disk."""
    device: torch.device = field(default_factory=lambda: select_training_device())
    """Device for training (MPS, CUDA, or CPU)."""
    use_mixed_precision: bool | None = None
    """Whether to use mixed precision training (None = use config value)."""
    loss_fn: nn.Module = field(default_factory=nn.CrossEntropyLoss)
    """Loss function for training."""

    def __post_init__(self) -> None:
        if self.keep_last_n < 1:
            raise ValueError("keep_last_n must be at least 1.")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.to(self.device)
        # Keep loss buffers (e.g. class weights) on the same device as logits/targets.
        self.loss_fn.to(self.device)
        if self.use_mixed_precision is None:
            self.use_mixed_precision = self.training_config.use_mixed_precision

    def _autocast_context(self) -> contextlib.AbstractContextManager[Any]:
        """Return an autocast context only when mixed precision is enabled."""

        if not self.use_mixed_precision:
            return nullcontext()

        if self.device.type == "mps":
            return torch.autocast(device_type="mps", dtype=torch.float16)
        if self.device.type == "cpu":
            return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
        return nullcontext()

    def _unpack_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Normalize common batch layouts into input and target tensors."""

        if isinstance(batch, Mapping):
            inputs = batch["inputs"]
            targets = batch["targets"]
            return inputs, targets
        if isinstance(batch, Sequence) and len(batch) == 2:
            inputs, targets = batch
            return inputs, targets
        raise ValueError("Batches must be mappings with inputs/targets or 2-tuples.")

    def _move_batch(self, batch: Any) -> tuple[Tensor, Tensor]:
        """Move a batch to the active training device."""

        inputs, targets = self._unpack_batch(batch)
        return inputs.to(self.device), targets.to(self.device)

    def _capture_rng_state(self) -> dict[str, Any]:
        """Capture Python, NumPy, and PyTorch RNG states."""

        return {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state().cpu(),
        }

    def _restore_rng_state(self, rng_state: Mapping[str, Any]) -> None:
        """Restore Python, NumPy, and PyTorch RNG states."""

        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"].cpu())

    def _checkpoint_path(self, step: int) -> Path:
        """Return the path for a numbered checkpoint file."""

        return self.checkpoint_dir / f"checkpoint_step_{step:010d}.pt"

    def _sorted_checkpoint_paths(self) -> list[Path]:
        """Return valid checkpoint paths sorted by step number ascending."""

        def _step(path: Path) -> int:
            match = _CHECKPOINT_PATTERN.match(path.stem)
            return int(match.group(1)) if match else -1

        candidates = [
            path
            for path in self.checkpoint_dir.glob("checkpoint_step_*.pt")
            if _CHECKPOINT_PATTERN.match(path.stem)
        ]
        return sorted(candidates, key=_step)

    def _prune_old_checkpoints(self) -> None:
        """Delete old checkpoint files, keeping only the most recent N files."""

        checkpoints = self._sorted_checkpoint_paths()
        excess = len(checkpoints) - self.keep_last_n
        if excess <= 0:
            return
        for path in checkpoints[:excess]:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()

    def _atomic_torch_save(self, payload: Mapping[str, Any], path: Path) -> None:
        """Persist a checkpoint atomically via a temporary file."""

        temporary_path = path.with_suffix(f"{path.suffix}.tmp")
        torch.save(payload, temporary_path)
        temporary_path.replace(path)

    def save_checkpoint(self, *, epoch: int, step: int) -> Path:
        """Save the current training state and return the file path."""

        checkpoint = TrainingCheckpoint(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=(
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            rng_state=self._capture_rng_state(),
            epoch=epoch,
            step=step,
        )
        checkpoint_path = self._checkpoint_path(step)
        self._atomic_torch_save(checkpoint.to_dict(), checkpoint_path)
        self._prune_old_checkpoints()
        return checkpoint_path

    def _load_checkpoint_payload(self, path: Path) -> TrainingCheckpoint:
        """Load and validate a checkpoint payload from disk."""

        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("Checkpoint payload must be a mapping.")
        return TrainingCheckpoint.from_dict(payload)

    def load_latest_checkpoint(self) -> tuple[TrainingCheckpoint, Path] | None:
        """Return the newest valid checkpoint, skipping corrupt files."""

        candidates = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda candidate: int(_CHECKPOINT_PATTERN.match(candidate.stem).group(1))
            if _CHECKPOINT_PATTERN.match(candidate.stem)
            else -1,
            reverse=True,
        )
        for candidate in candidates:
            try:
                checkpoint = self._load_checkpoint_payload(candidate)
            except (OSError, RuntimeError, ValueError, KeyError, EOFError, pickle.UnpicklingError):
                continue
            return checkpoint, candidate
        return None

    def _restore_checkpoint(self, checkpoint: TrainingCheckpoint) -> None:
        """Restore model, optimizer, scheduler, and RNG state from a checkpoint."""

        self.model.load_state_dict(checkpoint.model_state_dict)
        self.optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if self.scheduler is not None and checkpoint.scheduler_state_dict is not None:
            self.scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        self._restore_rng_state(checkpoint.rng_state)

    def _train_batch(self, batch: Any) -> float:
        """Run a single optimization step and return the batch loss."""

        inputs, targets = self._move_batch(batch)
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        with self._autocast_context():
            logits = self.model(inputs)
            loss = self.loss_fn(logits, targets)
        loss.backward()
        if self.training_config.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.max_grad_norm)
        self.optimizer.step()
        return float(loss.detach().cpu())

    def evaluate(self, data_loader: Any) -> dict[str, float]:
        """Evaluate loss and macro-F1 over a validation loader."""

        self.model.eval()
        total_loss = 0.0
        total_examples = 0
        predictions: list[int] = []
        targets: list[int] = []

        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                total=len(data_loader),
                desc="validation batches",
                leave=False,
            ):
                inputs, batch_targets = self._move_batch(batch)
                with self._autocast_context():
                    logits = self.model(inputs)
                    loss = self.loss_fn(logits, batch_targets)
                batch_size = int(batch_targets.shape[0])
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
                predictions.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
                targets.extend(batch_targets.detach().cpu().tolist())

        macro_f1 = _macro_f1_score(targets, predictions)
        average_loss = total_loss / total_examples if total_examples else 0.0
        return {"validation_loss": average_loss, "validation_macro_f1": macro_f1}

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        tracker: Any | None = None,
    ) -> dict[str, Any]:
        """Train the model with explicit epoch and step control.

        If ``tracker`` is provided and implements ``log_training_metrics``,
        periodic training/validation metrics and checkpoint artifacts will be
        forwarded to the tracker (e.g., an MLflow tracker).
        """

        max_train_steps = _max_train_steps_override()
        if val_loader is not None:
            _ = len(val_loader)

        logger = logging.getLogger(__name__)
        checkpoint_interval = max(1, self.training_config.log_every_n_steps)

        steps_per_epoch = len(train_loader)
        if steps_per_epoch <= 0:
            raise ValueError("train_loader must contain at least one batch.")

        latest_checkpoint = self.load_latest_checkpoint()
        start_epoch = 0
        global_step = 0
        batch_offset = 0
        if latest_checkpoint is not None:
            checkpoint, _ = latest_checkpoint
            self._restore_checkpoint(checkpoint)
            start_epoch = checkpoint.epoch
            global_step = checkpoint.step
            batch_offset = global_step % steps_per_epoch if global_step % steps_per_epoch else 0

        last_validation: dict[str, float] = {}
        last_checkpoint_path: Path | None = None
        stop_training = False
        best_validation_macro_f1 = float("-inf")
        epochs_without_improvement = 0
        completed_epoch = start_epoch

        epoch_indices = range(start_epoch, self.training_config.epochs)
        for epoch in tqdm(
            epoch_indices,
            total=max(0, self.training_config.epochs - start_epoch),
            desc="epochs",
        ):
            try:
                steps_this_epoch = len(train_loader)
                for batch_index, batch in tqdm(
                    enumerate(train_loader),
                    total=steps_this_epoch,
                    desc=f"epoch {epoch + 1}/{self.training_config.epochs} batches",
                    leave=False,
                ):
                    if epoch == start_epoch and batch_index < batch_offset:
                        continue
                    loss_value = self._train_batch(batch)
                    global_step += 1
                    should_log_step = (
                        self.training_config.log_every_n_steps > 0
                        and global_step % self.training_config.log_every_n_steps == 0
                    )
                    if (
                        should_log_step
                        and tracker is not None
                        and hasattr(tracker, "log_training_metrics")
                    ):
                        try:
                            learning_rate = (
                                float(self.optimizer.param_groups[0].get("lr", 0.0))
                                if self.optimizer.param_groups
                                else 0.0
                            )
                            tracker.log_training_metrics(
                                loss=loss_value,
                                epoch=epoch + 1,
                                step=global_step,
                                extra_metrics={"learning_rate": learning_rate},
                            )
                        except Exception:
                            logger.exception("tracker.log_training_metrics failed")
                    if should_log_step:
                        logger.info(
                            "[train] epoch=%d/%d step=%d loss=%.6f",
                            epoch + 1,
                            self.training_config.epochs,
                            global_step,
                            loss_value,
                        )

                    if global_step % checkpoint_interval == 0:
                        last_checkpoint_path = self.save_checkpoint(epoch=epoch, step=global_step)
                    if max_train_steps is not None and global_step >= max_train_steps:
                        logger.info(
                            (
                                "[train] reached SPEECH_TRAIN_MAX_STEPS=%d; "
                                "stopping early for smoke run"
                            ),
                            max_train_steps,
                        )
                        stop_training = True
                        break
                batch_offset = 0
                if val_loader is not None:
                    last_validation = self.evaluate(val_loader)
                    logger.info(
                        "[val] epoch=%d/%d validation_loss=%.6f validation_macro_f1=%.6f",
                        epoch + 1,
                        self.training_config.epochs,
                        last_validation.get("validation_loss", 0.0),
                        last_validation.get("validation_macro_f1", 0.0),
                    )
                    if tracker is not None and hasattr(tracker, "log_training_metrics"):
                        try:
                            tracker.log_training_metrics(
                                validation_macro_f1=last_validation.get(
                                    "validation_macro_f1", None
                                ),
                                checkpoint_path=(
                                    str(last_checkpoint_path) if last_checkpoint_path else None
                                ),
                                epoch=epoch + 1,
                                step=global_step,
                                extra_metrics={
                                    "validation_loss": last_validation.get("validation_loss", 0.0)
                                },
                            )
                        except Exception:
                            logger.exception("tracker.log_training_metrics failed")
                    current_validation_macro_f1 = last_validation.get("validation_macro_f1", 0.0)
                    if current_validation_macro_f1 > best_validation_macro_f1:
                        best_validation_macro_f1 = current_validation_macro_f1
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if (
                            self.training_config.early_stopping_patience >= 0
                            and epochs_without_improvement
                            >= self.training_config.early_stopping_patience
                        ):
                            logger.info(
                                (
                                    "[train] early stopping triggered after %d epochs without "
                                    "validation_macro_f1 improvement"
                                ),
                                epochs_without_improvement,
                            )
                            stop_training = True
                if self.scheduler is not None:
                    scheduler_step = getattr(self.scheduler, "step", None)
                    if callable(scheduler_step):
                        if last_validation and hasattr(self.scheduler, "optimizer"):
                            try:
                                scheduler_step(last_validation.get("validation_loss", 0.0))
                            except TypeError:
                                scheduler_step()
                        else:
                            scheduler_step()
                last_checkpoint_path = self.save_checkpoint(epoch=epoch + 1, step=global_step)
                logger.info(
                    "[train] completed epoch %d/%d global_step=%d",
                    epoch + 1,
                    self.training_config.epochs,
                    global_step,
                )
                if tracker is not None and hasattr(tracker, "log_training_metrics"):
                    try:
                        tracker.log_training_metrics(
                            checkpoint_path=(
                                str(last_checkpoint_path) if last_checkpoint_path else None
                            ),
                            epoch=epoch + 1,
                            step=global_step,
                        )
                    except Exception:
                        logger.exception("tracker.log_training_metrics failed")
                completed_epoch = epoch + 1
                if stop_training:
                    break
            except KeyboardInterrupt:
                last_checkpoint_path = self.save_checkpoint(epoch=epoch, step=global_step)
                raise

        return {
            "epoch": completed_epoch,
            "step": global_step,
            "checkpoint_path": str(last_checkpoint_path) if last_checkpoint_path else None,
            **last_validation,
        }


def select_training_device() -> torch.device:
    """Prefer MPS when available, otherwise fall back to CPU."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _macro_f1_score(targets: Sequence[int], predictions: Sequence[int]) -> float:
    """Compute a macro-F1 score without external dependencies."""

    labels = sorted(set(targets) | set(predictions))
    if not labels:
        return 0.0

    scores: list[float] = []
    for label in labels:
        true_positive = sum(
            1
            for target, pred in zip(targets, predictions, strict=True)
            if target == label and pred == label
        )
        false_positive = sum(
            1
            for target, pred in zip(targets, predictions, strict=True)
            if target != label and pred == label
        )
        false_negative = sum(
            1
            for target, pred in zip(targets, predictions, strict=True)
            if target == label and pred != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive)
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative)
            else 0.0
        )
        if precision + recall == 0.0:
            scores.append(0.0)
        else:
            scores.append(2.0 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)
