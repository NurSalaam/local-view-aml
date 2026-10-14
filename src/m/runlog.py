"""The append-only Run log.

A Run is one directory under ``experiments/<experiment>/runs/<run_id>/``:
``config.json``, ``fingerprint.json``, ``metrics.json``, ``status.json``,
``artifacts/``, ``logs/``. A Run is never mutated once complete; a crashed
Run keeps status ``running`` forever and stands as the crash record, while
resume simply appends a fresh Run for the same config. Skip-if-done is a pure
function over the log: a unit is done when a Run with its ``config_hash``
reads status ``complete``.

Single-writer by design: two processes sweeping the same units may both
compute one; the log stays consistent (append-only) and projections dedup by
``config_hash``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]


def experiments_root() -> Path:
    return Path(os.environ.get("M_EXPERIMENTS_DIR", REPO_ROOT / "experiments"))


def canonical(config: Any) -> Any:
    if dataclasses.is_dataclass(config) and not isinstance(config, type):
        return canonical(dataclasses.asdict(config))
    if isinstance(config, dict):
        return {str(k): canonical(v) for k, v in sorted(config.items())}
    if isinstance(config, (list, tuple)):
        return [canonical(v) for v in config]
    if isinstance(config, Path):
        return str(config)
    return config


def canonical_json(config: Any) -> str:
    return json.dumps(canonical(config), sort_keys=True, separators=(",", ":"))


def config_hash(config: Any) -> str:
    return hashlib.sha256(canonical_json(config).encode()).hexdigest()[:12]


def _write_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.rename(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def completed(experiment: str) -> dict[str, Path]:
    """Scan the Run log once; map each completed config_hash to its run dir."""
    runs = experiments_root() / experiment / "runs"
    done: dict[str, Path] = {}
    if not runs.exists():
        return done
    for status_path in runs.glob("*/status.json"):
        status = json.loads(status_path.read_text())
        if status["status"] == "complete":
            done[status["config_hash"]] = status_path.parent
    return done


def index(experiment: str) -> dict[str, tuple[Path, dict]]:
    """Every completed Run of one experiment, by config hash, with its config."""
    return {h: (p, json.loads((p / "config.json").read_text()))
            for h, p in completed(experiment).items()}


def select(runs: dict[str, tuple[Path, dict]], **want) -> list[Path]:
    """The Runs whose config holds every ``want`` field at the given value."""
    return [p for p, c in runs.values()
            if all(c.get(k) == v for k, v in want.items())]


def find(runs: dict[str, tuple[Path, dict]], **want) -> Path:
    """The one Run whose config holds every ``want`` field at the given value."""
    found = select(runs, **want)
    if len(found) != 1:
        raise SystemExit(f"{want}: {len(found)} completed Runs, expected one")
    return found[0]


def run(
    experiment: str,
    config: Any,
    fn: Callable[[Any, Path], dict],
    *,
    seed: int | None = None,
    done: dict[str, Path] | None = None,
) -> dict:
    """Execute ``fn(config, run_dir)`` once per config, ever.

    If the log already holds a completed Run for this config, return its
    recorded metrics without executing. ``done`` is the index from
    :func:`completed`; pass it when sweeping so the scan happens once, not
    per unit. ``fn`` returns the metrics dict and may write files under
    ``run_dir / "artifacts"`` and ``run_dir / "logs"``.
    """
    from m.fingerprint import fingerprint

    chash = config_hash(config)
    index = done if done is not None else completed(experiment)
    if chash in index:
        return json.loads((index[chash] / "metrics.json").read_text())

    environment = fingerprint(seed=seed, config_hash=chash)
    run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
    run_dir = experiments_root() / experiment / "runs" / run_id
    (run_dir / "artifacts").mkdir(parents=True)
    (run_dir / "logs").mkdir()

    started = _now()
    _write_atomic(run_dir / "config.json", canonical(config))
    _write_atomic(run_dir / "fingerprint.json", environment)
    _write_atomic(run_dir / "status.json",
                  {"status": "running", "config_hash": chash, "started": started})
    try:
        metrics = fn(config, run_dir)
    except BaseException as exc:
        _write_atomic(run_dir / "status.json",
                      {"status": "failed", "config_hash": chash, "started": started,
                       "failed": _now(),
                       "error": "".join(traceback.format_exception_only(exc)).strip()})
        raise
    _write_atomic(run_dir / "metrics.json", metrics)
    _write_atomic(run_dir / "status.json",
                  {"status": "complete", "config_hash": chash, "started": started,
                   "finished": _now()})
    if done is not None:
        done[chash] = run_dir
    return metrics
