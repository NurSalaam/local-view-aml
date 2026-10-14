"""The torch device: the CPU, on which every Run of the study trained, unless
``M_DEVICE`` names another (``mps``, ``cuda``)."""

from __future__ import annotations

import os


def device() -> "torch.device":  # noqa: F821
    import torch

    return torch.device(os.environ.get("M_DEVICE", "cpu"))
