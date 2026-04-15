import queue
import threading
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

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
