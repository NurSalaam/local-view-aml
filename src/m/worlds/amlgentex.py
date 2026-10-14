"""AMLgentex (Östman et al. 2025): canonical transaction log per dataset.

Conventions this module owns:

* Released datasets live under
  ``data/raw/amlgentex/<name>/tx_log.csv``; only the columns the study reads
  are loaded: the step, the payment type and amount, the two accounts and
  their banks, and the label.
* ``source`` and ``sink`` are pseudo-banks, and ``is_sar`` also flags a
  laundering account's rows against them, so every census statistic runs
  after :func:`drop_pseudo`. The filter is exposed, not hard-wired, so the raw
  share of their rows stays measurable.
* The canonical row order is a stable sort by ``step``, file order breaking
  ties; :func:`temporal_split` cuts 60/20/20 at row-count quantiles of the
  frame it is given, so a tied step lands wholly on the later side.
* ``bank_orig`` and ``bank_dest`` share one categorical roster, so their
  codes align. Dropping pseudo rows leaves the roster untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.csv

from m.runlog import REPO_ROOT

RELEASED = REPO_ROOT / "data" / "raw" / "amlgentex"
CACHE = REPO_ROOT / "data" / "interim" / "amlgentex" / "canonical"

PSEUDO_BANKS = ("source", "sink")
TRAIN_FRAC, VAL_FRAC = 0.6, 0.8

_CSV_TYPES = {
    "step": "int32", "amount": "float64",
    "nameOrig": "int32", "nameDest": "int32", "isSAR": "int8",
}

_RENAME = {
    "nameOrig": "name_orig", "bankOrig": "bank_orig",
    "nameDest": "name_dest", "bankDest": "bank_dest",
    "isSAR": "is_sar",
}

_COLUMNS = [
    "step", "type", "amount",
    "name_orig", "bank_orig", "name_dest", "bank_dest", "is_sar",
]

_DTYPES = {
    "step": "int32", "amount": "float64",
    "name_orig": "int32", "name_dest": "int32", "is_sar": "int8",
}


def load(name: str) -> pd.DataFrame:
    """The canonical table, cached: step, type, amount, endpoint accounts and
    banks (shared categorical roster), is_sar."""
    path = CACHE / f"{name}.parquet"
    if not path.exists():
        _build(name, path)
    return pd.read_parquet(path)


def pseudo_mask(df: pd.DataFrame) -> np.ndarray:
    """True for rows touching the source or sink pseudo-bank."""
    return (df["bank_orig"].isin(PSEUDO_BANKS)
            | df["bank_dest"].isin(PSEUDO_BANKS)).to_numpy()


def drop_pseudo(df: pd.DataFrame) -> pd.DataFrame:
    """Real transfers only; row filter, categories unchanged."""
    return df[~pseudo_mask(df)].reset_index(drop=True)


def split_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """The 60/20/20 step boundaries of a frame in canonical order."""
    step = df["step"].to_numpy()
    n = len(df)
    return int(step[int(TRAIN_FRAC * n)]), int(step[int(VAL_FRAC * n)])


def temporal_split(df: pd.DataFrame) -> np.ndarray:
    """Split codes 0/1/2 for a frame in canonical order, tied steps later."""
    t1, t2 = split_bounds(df)
    step = df["step"].to_numpy()
    return np.where(step < t1, 0, np.where(step < t2, 1, 2)).astype(np.int8)


def _build(name: str, path: Path) -> None:
    df = _canonicalise(_read_released(name))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="zstd")
    meta = {
        "dataset": name, "n_rows": int(len(df)),
        "n_sar_rows": int((df["is_sar"] == 1).sum()),
        "step_range": [int(df["step"].iloc[0]), int(df["step"].iloc[-1])],
    }
    (path.parent / f"{name}_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _read_released(name: str) -> pd.DataFrame:
    table = pyarrow.csv.read_csv(
        RELEASED / name / "tx_log.csv",
        convert_options=pyarrow.csv.ConvertOptions(
            column_types=_CSV_TYPES, auto_dict_encode=True,
            include_columns=["step", "type", "amount", *_RENAME]))
    return table.to_pandas(self_destruct=True, split_blocks=True)


def _canonicalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_RENAME)[_COLUMNS]
    step = df["step"].to_numpy()
    if np.any(np.diff(step) < 0):
        df = df.iloc[np.argsort(step, kind="stable")]
    df = df.reset_index(drop=True)
    for col, dtype in _DTYPES.items():
        if df[col].dtype != dtype:
            df[col] = df[col].astype(dtype)
    roster = df["bank_orig"].cat.categories.union(df["bank_dest"].cat.categories)
    df["bank_orig"] = df["bank_orig"].cat.set_categories(roster)
    df["bank_dest"] = df["bank_dest"].cat.set_categories(roster)
    return df
