#!/usr/bin/env python
"""Census check: the TransXion census (Chapter 3, Census).

One Run under ``experiments/checks/runs/`` records the numbers of payments,
laundering payments and banks; the qualifying banks, those with at least
thirty laundering payments in their test set; and, with payments from an
account to itself left out, the share of laundering and of normal payments
that join two accounts at the same bank beside the share expected if
accounts were assigned to banks at random. The per-bank table lands in the
Run's ``artifacts/banks.csv``.

Usage:  uv run python scripts/checks_census_transxion.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from m import completed, run
from m.census import either_endpoint_counts, within_bank
from m.worlds import transxion

MIN_TEST_POS = 30


def census(config: dict, run_dir) -> dict:
    df = transxion.load()
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
    banks.sort_values("test_pos", ascending=False).to_csv(run_dir / "artifacts" / "banks.csv")

    kept = ~df["same_account"].to_numpy()
    return {
        "n_rows": int(len(df)),
        "n_laundering": int(y.sum()),
        "n_banks": int(len(np.unique(np.concatenate([from_bank, to_bank])))),
        "qualifying_banks": int((banks["test_pos"] >= MIN_TEST_POS).sum()),
        "within_bank_structure": within_bank(from_bank[kept], to_bank[kept], y[kept]),
    }


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    metrics = run("checks", {"experiment": "checks", "check": "transxion_census",
                             "min_test_pos": MIN_TEST_POS},
                  census, done=completed("checks"))
    within = metrics["within_bank_structure"]
    print(f"transxion: {metrics['qualifying_banks']} qualifying banks of "
          f"{metrics['n_banks']}; same-bank share over chance "
          f"{within['laundering']['over_chance']:.2f} (laundering), "
          f"{within['licit']['over_chance']:.2f} (normal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
