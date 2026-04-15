"""Shared dataset manager type definitions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

type Split = Literal["train", "val", "test"]


@dataclass(frozen=True, slots=True)
class Sample:
    """Single audio sample descriptor."""

    path: Path
    label: str
    filename: str
