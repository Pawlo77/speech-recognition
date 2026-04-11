"""Speech recognition package exports."""

import logging

from .dataset import Sample, SpeechCommandsDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


__all__ = ["Sample", "SpeechCommandsDataset"]
