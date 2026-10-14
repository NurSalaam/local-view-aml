#!/usr/bin/env python
"""Global-sight feature tables for the detector checks.

One canonical extractor frame per (world, dataset) — AMLworld and TransXion
already share the canonical payments schema; AMLgentex real transfers are
adapted onto it (step days become epoch seconds, the single implicit currency
maps every amount to itself under the USD rule). The pooled global-sight
table (one row per payment; the row rule's global copies are identical, so
the pooled table carries the same information without double-weighting
cross-bank payments) is cached as parquet under ``data/interim/features``.

Usage:  uv run python scripts/extract.py [dataset ...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from m.features import extract, graph_feature_names, usd_rate_table, OWN_FIELDS
from m.runlog import REPO_ROOT
from m.worlds import amlgentex, amlworld, transxion

CACHE = REPO_ROOT / "data" / "interim" / "features"
AMLGENTEX_DATASET = "sweden_100K_easy"

DATASETS = {
    "HI-Small": ("amlworld", "HI-Small"),
    "LI-Small": ("amlworld", "LI-Small"),
    "HI-Medium": ("amlworld", "HI-Medium"),
    "LI-Medium": ("amlworld", "LI-Medium"),
    "transxion": ("transxion", "-"),
    AMLGENTEX_DATASET: ("amlgentex", AMLGENTEX_DATASET),
}

FEATURE_COLUMNS = OWN_FIELDS + graph_feature_names()


def amlgentex_frame(name: str) -> pd.DataFrame:
    real = amlgentex.drop_pseudo(amlgentex.load(name))
    single = pd.CategoricalDtype(["-"])
    df = pd.DataFrame({
        "ts": real["step"].to_numpy(np.int64) * 86_400,
        "from_bank": real["bank_orig"],
        "from_acct": real["name_orig"].astype(str).astype("category"),
        "to_bank": real["bank_dest"],
        "to_acct": real["name_dest"].astype(str).astype("category"),
        "amount_received": real["amount"],
        "recv_ccy": pd.Categorical(["-"] * len(real), dtype=single),
        "amount_paid": real["amount"],
        "pay_ccy": pd.Categorical(["-"] * len(real), dtype=single),
        "fmt": real["type"].astype("category"),
        "is_laundering": real["is_sar"],
        "split": amlgentex.temporal_split(real),
        "same_account": (real["name_orig"] == real["name_dest"]).to_numpy(),
    })
    return df


def frame(key: str) -> pd.DataFrame:
    world, dataset = DATASETS[key]
    if world == "amlworld":
        return amlworld.load(dataset)
    if world == "transxion":
        return transxion.load()
    return amlgentex_frame(dataset)


def table_path(key: str) -> Path:
    return CACHE / f"{key}_global.parquet"


def build(key: str) -> tuple[Path, float, int]:
    df = frame(key)
    rates = usd_rate_table(df)
    t0 = time.time()
    columns = extract(df, rates)
    seconds = time.time() - t0
    columns["is_laundering"] = df["is_laundering"].to_numpy(np.int8)
    columns["split"] = df["split"].to_numpy(np.int8)
    path = table_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), path, compression="zstd",
                   row_group_size=1_000_000)
    return path, seconds, len(df)


def load_column(key: str, column: str) -> np.ndarray:
    return pq.read_table(table_path(key), columns=[column])[column].to_numpy()


def main() -> int:
    args = sys.argv[1:]
    for key in args or list(DATASETS):
        if table_path(key).exists():
            print(f"{key}: cached")
            continue
        path, seconds, n_rows = build(key)
        print(f"{key}: {n_rows:,} rows in {seconds:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
