#!/usr/bin/env python
"""Census check: the AMLworld censuses (Chapter 3, Census).

Per dataset, one Run under ``experiments/checks/runs/`` records the numbers
of payments, laundering payments and bank identifiers (zero-padded strings,
so "011" and "0011" are two banks); the qualifying banks, those with at least
thirty laundering payments in their test set, the payment rail excluded; the
range of the qualifying banks' ledger sizes over all three windows; and, with
the rail's payments and payments from an account to itself left out, the
share of laundering and of normal payments that join two accounts at the
same bank beside the share expected if accounts were assigned to banks at
random. The per-bank table lands in the Run's ``artifacts/banks.csv``.

Usage:  uv run python scripts/checks_census_amlworld.py [HI-Small ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from m import completed, run
from m.census import either_endpoint_counts, within_bank
from m.progress import track
from m.worlds import amlworld

MIN_TEST_POS = 30


def census(config: dict, run_dir) -> dict:
    df = amlworld.load(config["dataset"])
    from_bank = df["from_bank"].cat.codes.to_numpy(np.int64)
    to_bank = df["to_bank"].cat.codes.to_numpy(np.int64)
    names = [str(c) for c in df["from_bank"].cat.categories]
    y = df["is_laundering"].to_numpy(np.int8)
    split = df["split"].to_numpy(np.int8)

    banks = pd.DataFrame({
        "ledger_payments": either_endpoint_counts(
            from_bank, to_bank, np.ones(len(df), bool), len(names)),
        "test_pos": either_endpoint_counts(
            from_bank, to_bank, (split == 2) & (y == 1), len(names)),
    }, index=pd.Index(names, name="bank"))
    banks = banks[banks["ledger_payments"] > 0]
    banks.sort_values("test_pos", ascending=False).to_csv(run_dir / "artifacts" / "banks.csv")
    qualifying = banks[(banks["test_pos"] >= MIN_TEST_POS)
                       & (banks.index != amlworld.RAIL_BANK)]

    kept = (~df["same_account"].to_numpy()
            & (df["from_bank"] != amlworld.RAIL_BANK).to_numpy()
            & (df["to_bank"] != amlworld.RAIL_BANK).to_numpy())
    return {
        "dataset": config["dataset"],
        "n_rows": int(len(df)),
        "n_laundering": int(y.sum()),
        "n_banks_padded": int(len(np.unique(np.concatenate([from_bank, to_bank])))),
        "qualifying_banks": int(len(qualifying)),
        "ledger_payments_range_qualifying": [int(qualifying["ledger_payments"].min()),
                                             int(qualifying["ledger_payments"].max())],
        "within_bank_structure_excl_070": within_bank(from_bank[kept], to_bank[kept],
                                                      y[kept]),
    }


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    datasets = sys.argv[1:] or amlworld.SETS
    done = completed("checks")
    for dataset in track(datasets, "amlworld census"):
        metrics = run("checks", {"experiment": "checks", "check": "amlworld_census",
                                 "dataset": dataset, "min_test_pos": MIN_TEST_POS},
                      census, done=done)
        within = metrics["within_bank_structure_excl_070"]
        print(f"\n{dataset}: {metrics['qualifying_banks']} qualifying banks of "
              f"{metrics['n_banks_padded']} identifiers; same-bank share over chance "
              f"{within['laundering']['over_chance']:.2f} (laundering), "
              f"{within['licit']['over_chance']:.2f} (normal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
