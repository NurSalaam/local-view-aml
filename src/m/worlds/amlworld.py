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
from decimal import ROUND_HALF_EVEN, Decimal
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


PATTERN_FAMILIES = ("BIPARTITE", "CYCLE", "FAN-IN", "FAN-OUT", "GATHER-SCATTER",
                    "RANDOM", "SCATTER-GATHER", "STACK")
CHAIN_FAMILIES = ("STACK", "CYCLE", "RANDOM")


def pattern_lines(dataset: str) -> pd.DataFrame:
    """Every injected-pattern member line with the family its block names.

    A block header reads "BEGIN LAUNDERING ATTEMPT - FAMILY: qualifier"; only
    the family before the colon is kept, so the sixteen fan-out degrees and the
    thirteen cycle lengths collapse to the eight shapes the generator injects.
    A member line repeats the payment in the released file's own column order —
    timestamp, both endpoints, the received amount and currency, the paid amount
    and currency, the format, the label — and both amounts are carried as exact
    cents parsed from the decimal text, never through a float, so the join below
    is an integer comparison on every field.
    """
    def cents(text: str) -> int:
        """The amount in exact cents, rounded the way ``numpy.rint`` rounds the
        canonical column it is compared with: half to even, and signed. Naive
        truncation of the decimal text disagrees with it on the pattern files'
        six-decimal Bitcoin amounts — one line on LI-Medium — and loses the
        sign on a negative amount."""
        return int(Decimal(text).scaleb(2).to_integral_value(ROUND_HALF_EVEN))

    rows, family = [], None
    with open(RAW / f"{dataset}_Patterns.txt") as handle:
        for line in handle:
            if line.startswith("BEGIN"):
                family = line.split("-", 1)[1].split(":")[0].strip()
            elif line.startswith("END"):
                family = None
            elif family and line.strip():
                fields = line.rstrip("\n").split(",")
                rows.append((fields[0], fields[1], fields[2], fields[3], fields[4],
                             cents(fields[5]), fields[6], cents(fields[7]),
                             fields[8], fields[9], family))
    frame = pd.DataFrame(rows, columns=["ts", "from_bank", "from_acct", "to_bank",
                                        "to_acct", "received_cents", "recv_ccy",
                                        "paid_cents", "pay_ccy", "fmt", "family"])
    frame["ts"] = ((pd.to_datetime(frame["ts"], format="%Y/%m/%d %H:%M")
                    - pd.Timestamp("1970-01-01")) // pd.Timedelta("1s")).astype("int64")
    return frame


JOIN_FIELDS = ["ts", "from_bank", "from_acct", "to_bank", "to_acct",
               "received_cents", "recv_ccy", "paid_cents", "pay_ccy", "fmt"]


def pattern_family_codes(dataset: str,
                         df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Per canonical row, the index of its injected-pattern family in
    :data:`PATTERN_FAMILIES` — or -1 where the pattern file names none — and
    the join's coverage beside it.

    The join is on the whole payment, since the released file repeats it rather
    than carrying an identifier: timestamp, both endpoints, both amounts in
    exact cents, both currencies and the format. The paid amount is the
    selective field, so the canonical table is first narrowed to the rows whose
    cents appear in the pattern file at all — an exact filter, because a true
    match agrees on that amount — and the ten-field join runs on what remains. A
    member line matching no canonical row, or more than one, is left unassigned
    and counted, so coverage is reported rather than assumed.
    """
    lines = pattern_lines(dataset).assign(line=lambda f: np.arange(len(f)))
    paid = np.rint(df["amount_paid"].to_numpy() * 100).astype(np.int64)
    received = np.rint(df["amount_received"].to_numpy() * 100).astype(np.int64)
    candidate = np.flatnonzero(np.isin(paid, lines["paid_cents"].to_numpy()))

    narrowed = pd.DataFrame({
        "ts": df["ts"].to_numpy()[candidate],
        "from_bank": df["from_bank"].astype(str).to_numpy()[candidate],
        "from_acct": df["from_acct"].astype(str).to_numpy()[candidate],
        "to_bank": df["to_bank"].astype(str).to_numpy()[candidate],
        "to_acct": df["to_acct"].astype(str).to_numpy()[candidate],
        "received_cents": received[candidate],
        "recv_ccy": df["recv_ccy"].astype(str).to_numpy()[candidate],
        "paid_cents": paid[candidate],
        "pay_ccy": df["pay_ccy"].astype(str).to_numpy()[candidate],
        "fmt": df["fmt"].astype(str).to_numpy()[candidate],
        "row": candidate})

    joined = lines.merge(narrowed, on=JOIN_FIELDS, how="left")
    hits = joined.groupby("line")["row"].count()
    exact = joined[joined["line"].map(hits).eq(1) & joined["row"].notna()]

    family_index = {name: i for i, name in enumerate(PATTERN_FAMILIES)}
    codes = np.full(len(df), -1, np.int64)
    rows = exact["row"].to_numpy().astype(np.int64)
    codes[rows] = [family_index[name] for name in exact["family"]]

    coverage = {"member_lines": int(len(lines)),
                "matched_lines": int(len(exact)),
                "ambiguous_lines": int((hits > 1).sum()),
                "unmatched_lines": int((hits == 0).sum()),
                "assigned_rows": int((codes >= 0).sum()),
                "collisions": int(len(exact) - len(np.unique(rows))),
                "families_seen": sorted(set(lines["family"]))}
    return codes, coverage
