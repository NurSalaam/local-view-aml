"""Device policy.

The GNN family trains on MPS wherever the operation set allows, with CPU
fallback for unsupported kernels; the fallback flag must be set before torch
initialises, hence at module import. ``M_DEVICE`` overrides the choice
(``cpu``, ``mps``, ``cuda``) for cross-plane runs and debugging.
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def device() -> "torch.device":  # noqa: F821
    import torch

    override = os.environ.get("M_DEVICE")
    if override:
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
