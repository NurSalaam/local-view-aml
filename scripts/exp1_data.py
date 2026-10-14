"""Experiment 1's data plane: dataset keys, rosters, per-bank ledgers, and the pooled
network-sight tables.

Datasets are the six keys of extract.py, read through its canonical frames.
``PRIMARY`` holds HI-Medium, HI-Small and the two controls; ``ANCHOR_ONLY``
holds LI-Small and LI-Medium, which take only the global Runs that the
comparison with published scores reads. A roster is a dataset's qualifying
banks: those with at least ``MIN_TEST_POS`` laundering payments in their test
set, the payment rail excluded on AMLworld. A bank's ledger (the payments touching it, all splits,
canonical order, with the canonical payment index) is cached once per dataset
under ``data/interim/exp1/ledgers`` so per-bank workers never load a whole
world (a derived cache; delete it and it rebuilds). The two account dictionaries are compacted to the ledger's own
accounts (the bank roster is shared and stays whole); node codes and features
are unchanged by it, since both are computed from the union of the two columns.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import AMLGENTEX_DATASET, DATASETS, frame, table_path  # noqa: E402
from m.features import extract, usd_rate_table  # noqa: E402
from m.runlog import REPO_ROOT  # noqa: E402
from m.worlds import amlworld  # noqa: E402

MIN_TEST_POS = 30
PRIMARY = ["HI-Medium", "HI-Small", "transxion", AMLGENTEX_DATASET]
ANCHOR_ONLY = ["LI-Small", "LI-Medium"]
CONTROLS = ["transxion", AMLGENTEX_DATASET]
LEDGERS = REPO_ROOT / "data" / "interim" / "exp1" / "ledgers"


def world_of(key: str) -> str:
    return DATASETS[key][0]


def excluded_banks(key: str) -> list[str]:
    return [amlworld.RAIL_BANK] if world_of(key) == "amlworld" else []


def global_table(key: str) -> Path:
    """The pooled network-sight feature table, built on first use."""
    path = table_path(key)
    if not path.exists():
        df = frame(key)
        columns = extract(df, usd_rate_table(df))
        columns["is_laundering"] = df["is_laundering"].to_numpy(np.int8)
        columns["split"] = df["split"].to_numpy(np.int8)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.parquet.tmp")
        pq.write_table(pa.table(columns), tmp, compression="zstd",
                       row_group_size=1_000_000)
        tmp.rename(path)
    return path


def bank_codes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return (df["from_bank"].cat.codes.to_numpy(np.int64),
            df["to_bank"].cat.codes.to_numpy(np.int64),
            [str(c) for c in df["from_bank"].cat.categories])


def either_endpoint_counts(fb: np.ndarray, tb: np.ndarray, mask: np.ndarray,
                           n_banks: int) -> np.ndarray:
    same = mask & (fb == tb)
    return (np.bincount(fb[mask], minlength=n_banks)
            + np.bincount(tb[mask], minlength=n_banks)
            - np.bincount(fb[same], minlength=n_banks))


def roster(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Usable banks with per-split rows and positives; index = bank label."""
    fb, tb, cats = bank_codes(df)
    y = df["is_laundering"].to_numpy(np.int8) == 1
    split = df["split"].to_numpy(np.int8)
    n = len(cats)
    table = pd.DataFrame({
        "bank": cats, "code": np.arange(n),
        "touching": either_endpoint_counts(fb, tb, np.ones(len(df), bool), n),
        "train_rows": either_endpoint_counts(fb, tb, split == 0, n),
        "val_rows": either_endpoint_counts(fb, tb, split == 1, n),
        "test_rows": either_endpoint_counts(fb, tb, split == 2, n),
        "train_pos": either_endpoint_counts(fb, tb, (split == 0) & y, n),
        "val_pos": either_endpoint_counts(fb, tb, (split == 1) & y, n),
        "test_pos": either_endpoint_counts(fb, tb, (split == 2) & y, n),
    })
    table["test_base_rate"] = table["test_pos"] / table["test_rows"].replace(0, np.nan)
    keep = (table["test_pos"] >= MIN_TEST_POS) & ~table["bank"].isin(excluded_banks(key))
    return table[keep].sort_values("touching", ascending=False).set_index("bank")


def ledger_dir(key: str) -> Path:
    return LEDGERS / key


def ledger_path(key: str, bank: str) -> Path:
    return ledger_dir(key) / f"{bank}.parquet"


def ledgers_ready(key: str) -> bool:
    return (ledger_dir(key) / "roster.csv").exists() and (ledger_dir(key) / "copies.npy").exists()


def write_ledgers(key: str) -> pd.DataFrame:
    """Cache every roster bank's ledger, the roster, the dataset-level USD
    table, and the row rule's copy count per payment; returns the roster."""
    out = ledger_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    df = frame(key)
    table = roster(df, key)
    fb, tb, _ = bank_codes(df)
    for bank, row in table.iterrows():
        code = int(row["code"])
        idx = np.flatnonzero((fb == code) | (tb == code))
        ledger = df.iloc[idx].reset_index(drop=True)
        for column in ("from_acct", "to_acct"):
            ledger[column] = ledger[column].cat.remove_unused_categories()
        ledger["payment"] = idx.astype(np.int64)
        target = out / f"{bank}.parquet"
        tmp = target.with_suffix(f".{os.getpid()}.parquet.tmp")
        ledger.to_parquet(tmp, index=False, compression="zstd")
        tmp.rename(target)
    (out / "rates.json").write_text(json.dumps(usd_rate_table(df), indent=2) + "\n")
    np.save(out / "copies.npy", np.where(fb != tb, 2, 1).astype(np.uint8))
    tmp = out / "roster.csv.tmp"
    table.to_csv(tmp)
    tmp.rename(out / "roster.csv")
    return table


def read_roster(key: str) -> pd.DataFrame:
    path = ledger_dir(key) / "roster.csv"
    if not path.exists():
        raise SystemExit(f"{key}: Experiment 1 has not cached its ledgers yet; "
                         "run scripts/exp1.py launch first")
    return pd.read_csv(path, dtype={"bank": str}).set_index("bank")


def read_rates(key: str) -> dict[str, float]:
    return json.loads((ledger_dir(key) / "rates.json").read_text())


def read_ledger(key: str, bank: str) -> pd.DataFrame:
    return pd.read_parquet(ledger_path(key, bank))
