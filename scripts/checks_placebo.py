#!/usr/bin/env python
"""The placebo on HI-Small (Chapter 3, Checks on the measurement).

The blind tree reads the payment's own fields only. Its training tables under
ledger sight and network sight are assembled under the row rule through the
same code paths the real detectors use — the network-sight table indexes the
pooled fields by copy, the ledger-sight table is scattered bank by bank from
each bank's own ledger — and its value of sight must be exactly zero: the two
tables must be identical to the byte, and the scores at every one of the
eight configurations must match to the last digit. A nonzero reading points
first to row handling, then to leakage.

Usage:  uv run python scripts/checks_placebo.py
"""
from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from m import completed, run, track
from m.features import (CATEGORICAL_FIELDS, OWN_FIELDS, VERSION, own_fields,
                        row_copies, usd_rate_table)
from m.grid import EARLY_STOP, FIXED, GRID, NUM_ROUND
from m.stats import average_precision, minority_f1
from m.worlds import amlworld

DATASET = "HI-Small"


def blind_matrix(fields: dict[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    return np.column_stack([fields[c][rows] for c in OWN_FIELDS]).astype(np.float32)


def placebo(config: dict, run_dir) -> dict:
    df = amlworld.load(DATASET)
    rates = usd_rate_table(df)
    from_bank = df["from_bank"].cat.codes.to_numpy(np.int64)
    to_bank = df["to_bank"].cat.codes.to_numpy(np.int64)
    payment, group = row_copies(from_bank, to_bank)

    pooled = own_fields(df, rates)
    global_table = blind_matrix(pooled, payment)

    local_table = np.full_like(global_table, np.nan)
    for g in track(np.unique(group), desc="bank ledgers"):
        ledger = np.flatnonzero((from_bank == g) | (to_bank == g))
        ledger_fields = own_fields(df.iloc[ledger], rates)
        copy_rows = np.flatnonzero(group == g)
        position = np.empty(len(df), np.int64)
        position[ledger] = np.arange(len(ledger))
        local_table[copy_rows] = blind_matrix(
            ledger_fields, position[payment[copy_rows]])

    tables_identical = bool(np.array_equal(global_table, local_table))
    digest = {
        "global": hashlib.sha256(global_table.tobytes()).hexdigest()[:16],
        "local": hashlib.sha256(local_table.tobytes()).hexdigest()[:16],
    }

    y = df["is_laundering"].to_numpy(np.int8)[payment]
    split = df["split"].to_numpy(np.int8)[payment]
    cat_idx = [OWN_FIELDS.index(c) for c in CATEGORICAL_FIELDS]
    tr, va, te = split == 0, split == 1, split == 2

    t0 = time.time()
    by_sight = {}
    for sight, table in (("global", global_table), ("local", local_table)):
        dtr = lgb.Dataset(table[tr], label=y[tr], categorical_feature=cat_idx,
                          params={"max_bin": FIXED["max_bin"]})
        dva = lgb.Dataset(table[va], label=y[va], categorical_feature=cat_idx,
                          params={"max_bin": FIXED["max_bin"]}, reference=dtr)
        runs = []
        for cfg in track(GRID, desc=f"{sight} grid"):
            bst = lgb.train({**FIXED, **cfg}, dtr, num_boost_round=NUM_ROUND,
                            valid_sets=[dva],
                            callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)])
            pte = bst.predict(table[te], num_iteration=bst.best_iteration)
            runs.append({"ap": average_precision(y[te], pte),
                         "f1": minority_f1(y[te], pte),
                         "predictions": pte})
        by_sight[sight] = runs

    settings = []
    for i, cfg in enumerate(GRID):
        network, ledger = by_sight["global"][i], by_sight["local"][i]
        identical = bool(np.array_equal(network["predictions"], ledger["predictions"]))
        settings.append({
            "params": cfg,
            "predictions_identical": identical,
            "value_of_sight_ap": network["ap"] - ledger["ap"],
            "value_of_sight_f1": network["f1"] - ledger["f1"],
            "test_ap_global": network["ap"],
            "test_f1_global": network["f1"],
        })
        print(f"  {cfg} identical={identical} "
              f"vos_ap={settings[-1]['value_of_sight_ap']:+.2e}", flush=True)

    all_zero = all(s["predictions_identical"]
                   and s["value_of_sight_ap"] == 0.0
                   and s["value_of_sight_f1"] == 0.0 for s in settings)
    return {
        "dataset": DATASET,
        "n_payments": int(len(df)),
        "n_copies": int(len(payment)),
        "cross_bank_copy_share": float((len(payment) - len(df)) / len(df)),
        "tables_identical": tables_identical,
        "table_sha256_16": digest,
        "settings": settings,
        "fit_seconds": time.time() - t0,
        "pass": bool(tables_identical and all_zero),
    }


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    metrics = run("checks", {"experiment": "checks", "check": "placebo",
                         "world": "amlworld", "dataset": DATASET,
                         "family": "blind_tree", "grid": GRID, "fixed": FIXED,
                         "num_round": NUM_ROUND, "early_stop": EARLY_STOP,
                         "features_version": VERSION},
                  placebo, done=completed("checks"))
    print(f"placebo: pass={metrics['pass']} "
          f"tables_identical={metrics['tables_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
