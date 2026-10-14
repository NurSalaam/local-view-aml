"""Per-Run environment fingerprint."""

from __future__ import annotations

import platform
import subprocess
import sys
from typing import Any

from m.runlog import REPO_ROOT


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True)
    return out.stdout.strip()


def fingerprint(seed: int | None = None, config_hash: str | None = None) -> dict[str, Any]:
    import torch

    try:
        import torch_geometric
        pyg_version = torch_geometric.__version__
    except ImportError:
        pyg_version = None
    try:
        import pyg_lib  # noqa: F401
        with_pyg_lib = True
    except ImportError:
        with_pyg_lib = False

    from importlib.metadata import version

    return {
        "platform": f"{platform.system()}-{platform.machine()}",
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "lightgbm": version("lightgbm"),
        "numpy": version("numpy"),
        "scikit_learn": version("scikit-learn"),
        "pyarrow": version("pyarrow"),
        "cuda": torch.version.cuda,
        "mps_available": torch.backends.mps.is_available(),
        "torch_geometric": pyg_version,
        "with_pyg_lib": with_pyg_lib,
        "git_sha": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "config_hash": config_hash,
        "seed": seed,
    }
