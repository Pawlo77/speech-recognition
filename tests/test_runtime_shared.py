import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from speech_recognition.config import (
    DatasetConfig,
    ExperimentConfig,
    FeaturePipelineConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingControlConfig,
)
from speech_recognition.orchestration.runtime import shared as runtime_shared
from speech_recognition.orchestration.runtime.shared import AudioRecord, FeatureBatchLoader


class _QueueThatRaisesFullOnce:
    def __init__(self) -> None:
        self.calls = 0
        self.items: list[object | None] = []

    def put(self, item, timeout=None):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1:
            raise queue.Full
        self.items.append(item)


def test_signal_completion_retries_when_queue_is_full() -> None:
    records = [AudioRecord(path=Path("sample.wav"), label="yes")]
    loader = FeatureBatchLoader(
        records=records,
        label_to_idx={"yes": 0},
        batch_size=1,
        seed=0,
        waveform_loader=lambda paths: torch.zeros((len(paths), 1, 4)),
        feature_extractor=nn.Identity(),
        shuffle=False,
    )
    loader._worker_stop_event = threading.Event()
    fake_queue = _QueueThatRaisesFullOnce()
    loader._prefetch_queue = fake_queue  # type: ignore[assignment]

    loader._signal_completion()

    assert fake_queue.calls == 2
    assert fake_queue.items == [None]


def test_fit_model_forwards_validation_metrics_to_tracker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _DummyEngine:
        def fit(self, train_loader, val_loader, tracker=None):
            _ = train_loader, val_loader, tracker
            return {
                "epoch": 2,
                "step": 7,
                "validation_macro_f1": 0.73,
                "validation_loss": 0.42,
            }

    class _Tracker:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def log_training_metrics(self, **kwargs):
            self.calls.append(dict(kwargs))

    def _fake_engine_factory(*args, **kwargs):
        _ = args, kwargs
        return _DummyEngine()

    def _fake_scheduler(*args, **kwargs):
        _ = args, kwargs
        return

    monkeypatch.setattr(runtime_shared, "TrainingEngine", _fake_engine_factory)
    monkeypatch.setattr(runtime_shared, "_build_scheduler", _fake_scheduler)

    config = ExperimentConfig(
        dataset=DatasetConfig(),
        features=FeaturePipelineConfig(),
        model=ModelConfig(),
        optimizer=OptimizerConfig(),
        scheduler=SchedulerConfig(total_epochs=2, warmup_epochs=1),
        training=TrainingControlConfig(epochs=2),
    )

    model = nn.Linear(4, 2)
    train_loader = SimpleNamespace()
    val_loader = SimpleNamespace()
    tracker = _Tracker()

    runtime_shared._fit_model(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=tmp_path / "checkpoints",
        tracker=tracker,
    )

    assert len(tracker.calls) == 1
    call = tracker.calls[0]
    assert call["epoch"] == 2
    assert call["step"] == 7
    assert float(call["validation_macro_f1"]) == pytest.approx(0.73)
    assert isinstance(call["extra_metrics"], dict)
    assert float(call["extra_metrics"]["validation_loss"]) == pytest.approx(0.42)
