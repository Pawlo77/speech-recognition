"""Runtime execution facade for training and evaluation entrypoints."""

from .eval import execute_single_eval
from .strategies import execute_single_train

__all__ = ["execute_single_eval", "execute_single_train"]
