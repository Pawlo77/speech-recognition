from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from speech_recognition.config import TrainingControlConfig
from speech_recognition.training import TrainingEngine, select_training_device


def _build_loader() -> DataLoader:
    inputs = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=2, shuffle=False)


def _build_engine(checkpoint_dir: Path) -> TrainingEngine:
    model = nn.Sequential(nn.Flatten(), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    config = TrainingControlConfig(
        epochs=1,
        batch_size=2,
        num_workers=0,
        log_every_n_steps=1,
        checkpoint_every_n_steps=1,
        validate_every_n_epochs=1,
        use_mixed_precision=False,
    )
    return TrainingEngine(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=config,
        checkpoint_dir=checkpoint_dir,
        keep_last_n=3,
        device=select_training_device(),
    )


def test_training_engine_resumes_from_latest_valid_checkpoint_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_loader = _build_loader()
    checkpoint_dir = tmp_path / "checkpoints"

    interrupting_engine = _build_engine(checkpoint_dir)
    resumed_engine = _build_engine(checkpoint_dir)
    original_train_batch = TrainingEngine._train_batch
    state = {
        "interrupt_engine": interrupting_engine,
        "resume_engine": resumed_engine,
        "interrupt_calls": 0,
        "resume_calls": 0,
        "interrupt_enabled": True,
    }

    def patched_train_batch(self, batch):
        if self is state["interrupt_engine"]:
            state["interrupt_calls"] += 1
            if state["interrupt_enabled"] and state["interrupt_calls"] == 3:
                raise KeyboardInterrupt
        if self is state["resume_engine"]:
            state["resume_calls"] += 1
        return original_train_batch(self, batch)

    monkeypatch.setattr(TrainingEngine, "_train_batch", patched_train_batch)

    with pytest.raises(KeyboardInterrupt):
        interrupting_engine.fit(train_loader)

    checkpoint, path = interrupting_engine.load_latest_checkpoint()
    assert checkpoint is not None
    assert checkpoint.step == 2
    assert checkpoint.epoch == 0
    assert path.name == "checkpoint_step_0000000002.pt"

    state["interrupt_enabled"] = False
    result = resumed_engine.fit(train_loader)

    assert state["resume_calls"] == 1
    assert result["step"] == 3
    assert result["epoch"] == 1

    final_checkpoint, _ = resumed_engine.load_latest_checkpoint()
    assert final_checkpoint is not None
    assert final_checkpoint.step == 3
    assert final_checkpoint.epoch == 1


def test_training_engine_logs_epoch_timing_metrics_to_tracker(tmp_path: Path) -> None:
    class _Tracker:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def log_training_metrics(self, **kwargs) -> None:
            self.calls.append(dict(kwargs))

    train_loader = _build_loader()
    val_loader = _build_loader()
    engine = _build_engine(tmp_path / "checkpoints")
    tracker = _Tracker()

    _ = engine.fit(train_loader, val_loader, tracker=tracker)

    timing_calls = [
        call
        for call in tracker.calls
        if isinstance(call.get("extra_metrics"), dict)
        and "epoch_elapsed_ms" in call["extra_metrics"]
    ]
    assert timing_calls
    latest = timing_calls[-1]
    extra_metrics = latest["extra_metrics"]
    assert isinstance(extra_metrics, dict)
    assert float(extra_metrics["epoch_elapsed_ms"]) >= 0.0
    assert float(extra_metrics["validation_elapsed_ms"]) >= 0.0


def test_training_engine_logs_checkpoint_path_when_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Tracker:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def log_training_metrics(self, **kwargs) -> None:
            self.calls.append(dict(kwargs))

    train_loader = _build_loader()
    engine = _build_engine(tmp_path / "checkpoints")
    tracker = _Tracker()

    original_train_batch = TrainingEngine._train_batch
    call_count = {"value": 0}

    def interrupting_train_batch(self, batch):
        _ = batch
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise KeyboardInterrupt
        return original_train_batch(self, batch)

    monkeypatch.setattr(TrainingEngine, "_train_batch", interrupting_train_batch)

    with pytest.raises(KeyboardInterrupt):
        engine.fit(train_loader, tracker=tracker)

    checkpoint_calls = [
        call
        for call in tracker.calls
        if isinstance(call.get("checkpoint_path"), str) and call["checkpoint_path"]
    ]
    assert checkpoint_calls
    assert any("checkpoint_step_" in str(call["checkpoint_path"]) for call in checkpoint_calls)


def test_training_engine_keeps_last_three_step_checkpoints_and_best(tmp_path: Path) -> None:
    train_loader = _build_loader()
    val_loader = _build_loader()
    engine = _build_engine(tmp_path / "checkpoints")

    _ = engine.fit(train_loader, val_loader)

    step_checkpoints = sorted(engine.checkpoint_dir.glob("checkpoint_step_*.pt"))
    assert len(step_checkpoints) == 3
    assert [path.name for path in step_checkpoints] == [
        "checkpoint_step_0000000001.pt",
        "checkpoint_step_0000000002.pt",
        "checkpoint_step_0000000003.pt",
    ]

    best_checkpoint = engine.checkpoint_dir / "checkpoint_best.pt"
    assert best_checkpoint.exists()
