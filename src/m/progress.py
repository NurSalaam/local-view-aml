"""Nested progress reporting.

Every level of a long-running loop gets a bar with rate and ETA. On a
terminal the bars nest by ``level``; without one (Databricks jobs, logs)
tqdm degrades to one printed line per refresh, throttled to
``M_PROGRESS_INTERVAL`` seconds (default 30) so logs stay readable.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Iterator, TypeVar

from tqdm.auto import tqdm

T = TypeVar("T")


def track(iterable: Iterable[T], desc: str, *, level: int = 0,
          total: int | None = None) -> Iterator[T]:
    tty = sys.stderr.isatty()
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        position=level,
        leave=level == 0,
        dynamic_ncols=True,
        mininterval=0.1 if tty else float(os.environ.get("M_PROGRESS_INTERVAL", 30)),
    )
