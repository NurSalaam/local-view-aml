#!/usr/bin/env python
"""Census check: the AMLgentex censuses (Chapter 3, Census).

Per released dataset, one Run under ``experiments/checks/runs/`` records the
share of raw rows that touch the source or sink account, and the share of
laundering rows that do; then, on the transfers that remain, the numbers of
payments, laundering payments and banks; each bank's laundering payments in
its test set, with their minimum and maximum and the qualifying banks, those
with at least thirty; the share of laundering and of normal payments that
join two accounts at the same bank beside the share expected if accounts were
assigned to banks at random; and the mean number of banks an injected
typology instance spans beside the mean expected if its accounts had been
assigned to banks at random. The per-bank and per-instance tables land in
the Run's ``artifacts/``.

Usage:  uv run python scripts/checks_census_amlgentex.py [sweden_100K_easy ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from m import completed, run
from m.census import either_endpoint_counts, within_bank
from m.progress import track
from m.worlds import amlgentex

MIN_TEST_POS = 30


def span_stats(pid: np.ndarray, bank_o: np.ndarray, bank_d: np.ndarray,
               acct_o: np.ndarray, acct_d: np.ndarray, p_acc: np.ndarray,
               run_dir) -> dict:
    """The mean number of banks a typology instance spans, beside the mean
    expected if each of its accounts lay at bank b with probability p_b, the
    bank's share of accounts: an instance of a accounts spans, in expectation,
    sum_b (1 - (1 - p_b)^a) banks."""
    n_banks = len(p_acc)
    bank_keys = np.unique(np.concatenate([pid * n_banks + bank_o,
                                          pid * n_banks + bank_d]))
    pids, spans = np.unique(bank_keys // n_banks, return_counts=True)
    stride = int(max(acct_o.max(), acct_d.max())) + 1
    acct_keys = np.unique(np.concatenate([pid * stride + acct_o,
                                          pid * stride + acct_d]))
    _, accounts = np.unique(acct_keys // stride, return_counts=True)
    _, transfers = np.unique(pid, return_counts=True)
    expected = (1.0 - np.power.outer(1.0 - p_acc, accounts)).sum(axis=0)

    pd.DataFrame({"pattern_id": pids, "n_transfers": transfers,
                  "n_accounts": accounts, "span": spans,
                  "expected_span": expected}) \
        .to_csv(run_dir / "artifacts" / "patterns.csv", index=False)
    return {"n_patterns": int(len(pids)), "mean": float(spans.mean()),
            "chance_mean": float(expected.mean())}


def census(config: dict, run_dir) -> dict:
    df = amlgentex.load(config["dataset"])
    pseudo = amlgentex.pseudo_mask(df)
    sar_raw = df["is_sar"].to_numpy() == 1
    raw = {"n_rows_raw": int(len(df)),
           "raw_pseudo_row_share": float(pseudo.mean()),
           "sar_rows_pseudo_share": float(pseudo[sar_raw].mean())}

    real = amlgentex.drop_pseudo(df)
    del df, pseudo, sar_raw
    split = amlgentex.temporal_split(real)
    is_sar = real["is_sar"].to_numpy() == 1
    pat = real["pattern_id"].to_numpy()

    cats = real["bank_orig"].cat.categories
    bank_o = real["bank_orig"].cat.codes.to_numpy().astype(np.int64)
    bank_d = real["bank_dest"].cat.codes.to_numpy().astype(np.int64)
    used = np.union1d(np.unique(bank_o), np.unique(bank_d))
    remap = np.full(len(cats), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    bank_o, bank_d = remap[bank_o], remap[bank_d]
    banks = [str(c) for c in cats[used]]
    n_banks = len(banks)

    acct_o = real["name_orig"].to_numpy(np.int64)
    acct_d = real["name_dest"].to_numpy(np.int64)
    node_keys = np.unique(np.concatenate([acct_o * n_banks + bank_o,
                                          acct_d * n_banks + bank_d]))
    p_acc = np.bincount(node_keys % n_banks, minlength=n_banks) / len(node_keys)

    # alertID labels only laundering rows, so every pattern here is a
    # laundering typology instance.
    laundering_pattern = (pat >= 0) & is_sar
    span = span_stats(pat[laundering_pattern].astype(np.int64),
                      bank_o[laundering_pattern], bank_d[laundering_pattern],
                      acct_o[laundering_pattern], acct_d[laundering_pattern],
                      p_acc, run_dir)

    test_sar = either_endpoint_counts(bank_o, bank_d, (split == 2) & is_sar, n_banks)
    pd.DataFrame({"bank": banks, "test_sar": test_sar}) \
        .to_csv(run_dir / "artifacts" / "banks.csv", index=False)

    return {"dataset": config["dataset"], **raw,
            "n_real_transfers": int(len(real)),
            "n_sar_transfers": int(is_sar.sum()),
            "n_real_banks": n_banks,
            "qualifying_banks": int((test_sar >= MIN_TEST_POS).sum()),
            "test_sar_min": int(test_sar.min()),
            "test_sar_max": int(test_sar.max()),
            "within_bank": within_bank(bank_o, bank_d, is_sar.astype(np.int8)),
            "pattern_span": span}


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    released = amlgentex.released_names()
    targets = sys.argv[1:] or released
    unknown = [n for n in targets if n not in released]
    if unknown:
        raise SystemExit(f"unknown or incomplete dataset: {unknown}")

    done = completed("checks")
    for name in track(targets, "amlgentex census"):
        m = run("checks", {"experiment": "checks", "check": "amlgentex_census",
                           "dataset": name, "min_test_pos": MIN_TEST_POS},
                census, done=done)
        print(f"\n{name}: laundering payments per bank in the test set "
              f"{m['test_sar_min']} to {m['test_sar_max']}; typology span "
              f"{m['pattern_span']['mean']:.3f} against "
              f"{m['pattern_span']['chance_mean']:.3f} at random")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
