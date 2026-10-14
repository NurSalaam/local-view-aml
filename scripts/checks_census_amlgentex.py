#!/usr/bin/env python
"""Census check: the AMLgentex census (Chapter 3, Census).

One Run under ``experiments/checks/runs/`` records, on the AMLgentex dataset,
the share of raw rows that touch the source or sink account; then, on the
transfers that remain, the numbers of payments, laundering payments, banks and
daily steps; each bank's laundering payments in its test set, with their
minimum and maximum and the qualifying banks, those with at least thirty; and
the share of laundering and of normal payments that join two accounts at the
same bank beside the share expected if accounts were assigned to banks at
random. The per-bank table lands in the Run's ``artifacts/banks.csv``.

Usage:  uv run python scripts/checks_census_amlgentex.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from extract import AMLGENTEX_DATASET  # noqa: E402
from m import completed, run  # noqa: E402
from m.census import either_endpoint_counts, within_bank  # noqa: E402
from m.worlds import amlgentex  # noqa: E402

MIN_TEST_POS = 30


def census(config: dict, run_dir) -> dict:
    df = amlgentex.load(config["dataset"])
    pseudo = amlgentex.pseudo_mask(df)
    raw = {"n_rows_raw": int(len(df)),
           "raw_pseudo_row_share": float(pseudo.mean())}

    real = amlgentex.drop_pseudo(df)
    del df, pseudo
    split = amlgentex.temporal_split(real)
    is_sar = real["is_sar"].to_numpy() == 1

    cats = real["bank_orig"].cat.categories
    bank_o = real["bank_orig"].cat.codes.to_numpy().astype(np.int64)
    bank_d = real["bank_dest"].cat.codes.to_numpy().astype(np.int64)
    used = np.union1d(np.unique(bank_o), np.unique(bank_d))
    remap = np.full(len(cats), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    bank_o, bank_d = remap[bank_o], remap[bank_d]
    banks = [str(c) for c in cats[used]]
    n_banks = len(banks)

    test_sar = either_endpoint_counts(bank_o, bank_d, (split == 2) & is_sar, n_banks)
    pd.DataFrame({"bank": banks, "test_sar": test_sar}) \
        .to_csv(run_dir / "artifacts" / "banks.csv", index=False)

    return {"dataset": config["dataset"], **raw,
            "n_real_transfers": int(len(real)),
            "n_sar_transfers": int(is_sar.sum()),
            "n_real_banks": n_banks,
            "steps": int(real["step"].nunique()),
            "qualifying_banks": int((test_sar >= MIN_TEST_POS).sum()),
            "test_sar_min": int(test_sar.min()),
            "test_sar_max": int(test_sar.max()),
            "within_bank": within_bank(bank_o, bank_d, is_sar.astype(np.int8))}


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    m = run("checks", {"experiment": "checks", "check": "amlgentex_census",
                       "dataset": AMLGENTEX_DATASET, "min_test_pos": MIN_TEST_POS},
            census, done=completed("checks"))
    print(f"{AMLGENTEX_DATASET}: {m['qualifying_banks']} qualifying banks of "
          f"{m['n_real_banks']} over {m['steps']} daily steps; laundering payments "
          f"per bank in the test set "
          f"{m['test_sar_min']} to {m['test_sar_max']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
