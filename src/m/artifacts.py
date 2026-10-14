"""Run artefacts.

Large derived objects — per-payment score vectors, model weights, binary
datasets — live under ``run_dir/artifacts`` and stay out of git; the Run's
``metrics.json`` carries each artefact's sha256 and size, so a projection can
verify what it reads and a reader can tell a stale copy from the recorded
one. Every writer here is atomic (temporary file, then rename) so a killed
Run never leaves a half-written artefact under its final name.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_of_array(array: np.ndarray) -> str:
    """The digest :func:`write_array` would record, without writing the file,
    so an array held in memory can be compared with one another Run persisted."""
    import io

    buffer = io.BytesIO()
    np.lib.format.write_array(buffer, array, allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {"file": path.name, "sha256": sha256_of(path),
            "bytes": path.stat().st_size}


def write_array(run_dir: Path, name: str, array: np.ndarray) -> dict[str, Any]:
    path = run_dir / "artifacts" / f"{name}.npy"
    tmp = path.with_suffix(".npy.tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array)
    tmp.rename(path)
    return _record(path)


def write_state(run_dir: Path, name: str, state: dict) -> dict[str, Any]:
    import torch

    path = run_dir / "artifacts" / f"{name}.pt"
    tmp = path.with_suffix(".pt.tmp")
    with tmp.open("wb") as handle:
        torch.save(state, handle)
    tmp.rename(path)
    return _record(path)


def write_json(run_dir: Path, name: str, payload: Any) -> dict[str, Any]:
    path = run_dir / "artifacts" / f"{name}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.rename(path)
    return _record(path)


def write_text(run_dir: Path, name: str, text: str,
               suffix: str = ".txt") -> dict[str, Any]:
    path = run_dir / "artifacts" / f"{name}{suffix}"
    tmp = path.with_suffix(suffix + ".tmp")
    tmp.write_text(text)
    tmp.rename(path)
    return _record(path)


def write_csv(run_dir: Path, name: str, frame) -> dict[str, Any]:
    path = run_dir / "artifacts" / f"{name}.csv"
    tmp = path.with_suffix(".csv.tmp")
    frame.to_csv(tmp)
    tmp.rename(path)
    return _record(path)


def verify(run_dir: Path, record: dict[str, Any]) -> Path:
    """The artefact's path, after its recorded sha256 has been checked."""
    path = run_dir / "artifacts" / record["file"]
    if sha256_of(path) != record["sha256"]:
        raise ValueError(f"{path} does not match its recorded sha256")
    return path


def load_array(run_dir: Path, name: str, expected_sha256: str | None = None) -> np.ndarray:
    path = run_dir / "artifacts" / f"{name}.npy"
    if expected_sha256 is not None and sha256_of(path) != expected_sha256:
        raise ValueError(f"{path} does not match its recorded sha256")
    return np.load(path)
