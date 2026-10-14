#!/usr/bin/env python
"""The never-seen-counterparty cue (Chapter 3, Census).

For every qualifying bank of a dataset, the probe takes the payments in the
bank's test set between an account at the bank and an account at another
bank, the counterparty, and counts the counterparty's earlier appearances in
the bank's ledger (``m.frame``), capping the count at five. Pooled over the
qualifying banks, one Run per dataset records the share of laundering
payments among those with a never-seen counterparty and among those whose
counterparty had appeared five times or more. The counts and laundering
payments at every history, per bank and pooled, land in the Run's
``artifacts/``. Nothing is trained.

LightGBM is never imported here; nothing in the probe needs it, and its OpenMP
runtime clashes with torch's on macOS when it loads first (see `scripts/exp1.py`).
Torch enters only through the per-Run fingerprint, after everything else.

  uv run python scripts/checks_frame_probe.py                 # every dataset
  uv run python scripts/checks_frame_probe.py HI-Medium       # one dataset
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1_data as data  # noqa: E402
from m import completed, run, track  # noqa: E402
from m.artifacts import write_csv  # noqa: E402
from m.features import VERSION as FEATURES_VERSION  # noqa: E402
from m.features import account_codes  # noqa: E402
from m.frame import endpoint_history, external_history  # noqa: E402

EXPERIMENT = "checks"
CHECK = "frame_probe"
CAP = 5
KEYS = ["transxion", data.AMLGENTEX_DATASET,
        "HI-Medium", "HI-Small", "LI-Small", "LI-Medium"]


def probe_config(key: str) -> dict:
    return {"experiment": EXPERIMENT, "check": CHECK,
            "world": data.DATASETS[key][0], "dataset": data.DATASETS[key][1],
            "dataset_key": key,
            "roster": {"min_test_pos": data.MIN_TEST_POS,
                       "exclude": data.excluded_banks(key)},
            "payments": "the bank's test-set payments with an account at "
                        "another bank, the counterparty",
            "history": "the counterparty's strictly earlier appearances in "
                       "the bank's ledger, on either side",
            "history_cap": CAP, "features_version": FEATURES_VERSION}


def probe_metrics(key: str, run_dir: Path) -> dict:
    t0 = time.time()
    df = data.frame(key)
    roster = data.roster(df, key)
    from_bank, to_bank, _ = data.bank_codes(df)
    payer, payee = account_codes(df)
    ts = df["ts"].to_numpy(np.int64)
    y = df["is_laundering"].to_numpy(np.int8)
    split = df["split"].to_numpy(np.int8)
    del df

    rows = []
    for bank in track(list(roster.index), f"frame probe {key}"):
        code = int(roster.loc[bank, "code"])
        idx = np.flatnonzero((from_bank == code) | (to_bank == code))
        test = split[idx] == 2
        test_set = idx[test]
        crosses = from_bank[test_set] != to_bank[test_set]
        ledger_payer, ledger_payee = endpoint_history(payer[idx], payee[idx], ts[idx])
        history = np.minimum(external_history(ledger_payer[test], ledger_payee[test],
                                              from_bank[test_set] == code), CAP)
        labels = y[test_set]
        for count in range(CAP + 1):
            mask = crosses & (history == count)
            rows.append({"bank": bank, "history": count, "payments": int(mask.sum()),
                         "laundering": int(labels[mask].sum())})

    table = pd.DataFrame(rows)
    pooled = table.groupby("history")[["payments", "laundering"]].sum()
    rate = pooled["laundering"] / pooled["payments"]
    return {
        "dataset_key": key, "world": data.DATASETS[key][0], "banks": int(len(roster)),
        "laundering_rate_never_seen": float(rate.loc[0]),
        "laundering_rate_seen_five_or_more": float(rate.loc[CAP]),
        "seconds": time.time() - t0,
        "artifacts": {"roster": write_csv(run_dir, "roster", roster),
                      "history": write_csv(run_dir, "history", table.set_index("bank")),
                      "history_pooled": write_csv(run_dir, "history_pooled",
                                                  pooled.assign(laundering_rate=rate))},
    }


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    keys = sys.argv[1:] or KEYS
    unknown = [key for key in keys if key not in data.DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset keys: {unknown}")
    done = completed(EXPERIMENT)
    for key in keys:
        metrics = run(EXPERIMENT, probe_config(key),
                      lambda config, run_dir: probe_metrics(key, run_dir), done=done)
        print(f"frame_probe {key}: {metrics['banks']} banks; laundering share "
              f"{metrics['laundering_rate_never_seen']:.3%} with a never-seen "
              f"counterparty, {metrics['laundering_rate_seen_five_or_more']:.3%} "
              f"with one seen {CAP} times or more", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
