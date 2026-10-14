"""Seed every RNG a Run touches."""

from __future__ import annotations

import random


def seed_all(seed: int) -> None:
    import numpy
    import torch

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
