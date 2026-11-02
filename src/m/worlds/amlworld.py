"""AMLworld (Altman et al. 2024): canonical payments table per dataset.

Conventions this module owns:

* Bank identifiers are variable-width zero-padded decimal strings; "011" and
  "0011" are different banks. An int cast merges 61 pairs on HI-Medium.
* An account is keyed (bank, account); 24 account strings recur across banks.
* The canonical row order is a stable sort by timestamp, file order breaking
  ties; the 60/20/20 split compares timestamps at the row-count quantile
  boundaries, so a tied minute lands wholly on the later side.
* Same-account rows (reinvestment bookkeeping) are flagged, not dropped:
  bank-structure statistics exclude them, evaluation sets keep them.
* Bank "070" is the payment rail the bank roster excludes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv

from m.runlog import REPO_ROOT

RAW = REPO_ROOT / "data" / "raw" / "ibm-aml"
CACHE = REPO_ROOT / "data" / "interim" / "ibm-aml" / "canonical"

SETS = ["HI-Small", "LI-Small", "HI-Medium", "LI-Medium"]
RAIL_BANK = "070"
TRAIN_FRAC, VAL_FRAC = 0.6, 0.8

_STR = ["from_bank", "from_acct", "to_bank", "to_acct"]


def load(dataset: str) -> pd.DataFrame:
    """The canonical table, cached: ts, banks/accounts (categorical strings),
    amounts, currencies, format, is_laundering, split, same_account."""
    path = CACHE / f"{dataset}.parquet"
    if not path.exists():
        _build(dataset, path)
    return pd.read_parquet(path)


def _build(dataset: str, path: Path) -> None:
    table = pyarrow.csv.read_csv(
        RAW / f"{dataset}_Trans.csv",
        convert_options=pyarrow.csv.ConvertOptions(
            column_types={"From Bank": "string", "Account": "string",
                          "To Bank": "string", "Account.1": "string"}),
    )
    df = table.to_pandas()
    df.columns = ["ts", "from_bank", "from_acct", "to_bank", "to_acct",
                  "amount_received", "recv_ccy", "amount_paid", "pay_ccy",
                  "fmt", "is_laundering"]
    df["ts"] = ((pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M")
                 - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).astype("int64")

    order = np.argsort(df["ts"].values, kind="stable")
    df = df.iloc[order].reset_index(drop=True)

    ts = df["ts"].values
    n = len(df)
    t1, t2 = ts[int(TRAIN_FRAC * n)], ts[int(VAL_FRAC * n)]
    df["split"] = np.where(ts < t1, 0, np.where(ts < t2, 1, 2)).astype(np.int8)

    df["same_account"] = ((df["from_bank"] == df["to_bank"])
                          & (df["from_acct"] == df["to_acct"]))
    df["is_laundering"] = df["is_laundering"].astype(np.int8)

    bank_cat = pd.Categorical(pd.concat([df["from_bank"], df["to_bank"]],
                                        ignore_index=True))
    df["from_bank"] = pd.Categorical.from_codes(bank_cat.codes[:n], bank_cat.categories)
    df["to_bank"] = pd.Categorical.from_codes(bank_cat.codes[n:], bank_cat.categories)
    for col in ("from_acct", "to_acct", "recv_ccy", "pay_ccy", "fmt"):
        df[col] = df[col].astype("category")

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    meta = {
        "dataset": dataset, "n_rows": int(n),
        "n_laundering": int(df["is_laundering"].sum()),
        "t1_epoch_s": int(t1), "t2_epoch_s": int(t2),
        "split_sizes": {int(k): int(v)
                        for k, v in df["split"].value_counts().sort_index().items()},
    }
    (path.parent / f"{dataset}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
