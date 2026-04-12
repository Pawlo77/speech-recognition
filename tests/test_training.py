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
        validate_every_n_epochs=1,
        use_mixed_precision=False,
    )
    return TrainingEngine(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        training_config=config,
        checkpoint_dir=checkpoint_dir,
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
