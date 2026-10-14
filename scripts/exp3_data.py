"""The data plane of Experiment 3 and its descriptor selection.

Two kinds of thing are read here and nothing else is. **Ledgers** — Experiment 1's
cached per-bank parquet files — supply the descriptor values, and they are
read through one conversion so that the descriptor selection and Experiment 3 cannot disagree about what
a bank's training window is. **Targets** — Experiment 1's `shortfall` Runs — supply each
bank's measured Δ_B, its standard error s_B, and the 1,000 joint-bootstrap
replicates behind them, always by artefact digest, so a stale copy cannot be
read for the recorded one.

The conversion is where identity is settled, and it is settled the same way
three times: accounts are keyed by the (bank, account) pair on one dictionary
shared by both endpoint columns, and the banks and the currencies each get one
dictionary shared by their two columns. A per-column dictionary would give one
account, one bank or one currency two identities. Two of the three are
load-bearing on the data as it stands — twenty-four account strings occur at
two banks on HI-Medium and twenty on `sweden_100K_easy`, and the two endpoint
account columns are categorised independently by the loader — while the bank
and currency unions are inert today and guard against a loader change rather
than against the data.

Neither LightGBM nor torch is imported: nothing here trains, and their bundled
OpenMP runtimes clash on macOS when LightGBM loads first (`scripts/exp1.py`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1_data as data  # noqa: E402
from m import completed  # noqa: E402
from m.artifacts import verify  # noqa: E402
from m.descriptors import Ledger  # noqa: E402
from m.features import account_codes  # noqa: E402

FIT_KEY = "HI-Medium"
HOLDOUT_KEY = "HI-Small"
KEYS = (FIT_KEY, HOLDOUT_KEY)
FAMILIES = ("tree", "gnn", "blind")
PRIMARY_FAMILIES = ("tree", "gnn")

TRAINING_SPLIT = 0
WINDOW_RULE = ("split == 0 (the training window), same-account rows excluded, "
               "no label read")


# ------------------------------------------------------------- ledgers --

def training_rows(ledger: pd.DataFrame) -> pd.DataFrame:
    """The rows a descriptor may count: the training window, bookkeeping
    excluded."""
    keep = (ledger["split"].to_numpy() == TRAINING_SPLIT) & ~ledger["same_account"].to_numpy()
    return ledger.loc[keep]


def ledger_arrays(rows: pd.DataFrame, bank: str) -> Ledger:
    """One bank's training rows as the arrays :mod:`m.descriptors` reads."""
    payer, payee = account_codes(rows)
    payer_is_own = (rows["from_bank"].to_numpy().astype(str) == bank)
    payee_is_own = (rows["to_bank"].to_numpy().astype(str) == bank)

    def shared(left: str, right: str):
        """Both columns' codes over one dictionary. The loader happens to give
        the bank columns a shared one and the currency columns a shared one
        today; nothing downstream may depend on that, because a per-column
        dictionary would give one bank, or one currency, two identities."""
        vocabulary = rows[left].cat.categories.union(rows[right].cat.categories)

        def codes(column: str) -> np.ndarray:
            source = rows[column]
            lookup = vocabulary.get_indexer(source.cat.categories).astype(np.int64)
            return lookup[source.cat.codes.to_numpy()]

        return codes(left), codes(right)

    from_bank, to_bank = shared("from_bank", "to_bank")
    pay_ccy, recv_ccy = shared("pay_ccy", "recv_ccy")
    counterparty = np.where(payer_is_own, to_bank, from_bank)
    counterparty[payer_is_own & payee_is_own] = -1

    return Ledger(payer=payer, payee=payee,
                  payer_is_own=payer_is_own, payee_is_own=payee_is_own,
                  counterparty_bank=counterparty,
                  fmt=rows["fmt"].cat.codes.to_numpy(np.int64),
                  pay_ccy=pay_ccy, recv_ccy=recv_ccy)


def bank_ledger(key: str, bank: str) -> Ledger:
    return ledger_arrays(training_rows(data.read_ledger(key, bank)), bank)


def roster(key: str) -> pd.DataFrame:
    return data.read_roster(key)


# ------------------------------------------------------------- targets --

def _exp1_runs() -> dict[tuple[str, str], Path]:
    """Experiment 1's completed `shortfall` Runs, keyed by (dataset key, family)."""
    out: dict[tuple[str, str], Path] = {}
    for path in sorted(completed("exp1").values()):
        config = json.loads((path / "config.json").read_text())
        if config.get("check") == "shortfall":
            out[(config["dataset_key"], config["family"])] = path
    return out


_RUNS: dict[tuple[str, str], Path] | None = None


def shortfall_run(key: str, family: str) -> Path:
    global _RUNS
    if _RUNS is None:
        _RUNS = _exp1_runs()
    return _RUNS[(key, family)]


def shortfall_hash(key: str, family: str) -> str:
    path = shortfall_run(key, family)
    return json.loads((path / "status.json").read_text())["config_hash"]


def targets(key: str, family: str) -> pd.DataFrame:
    """Per bank: the measured Δ_B, s_B, the bootstrap-only error, the interval,
    the exam paper's size and its base rate — the recorded artefact, verified
    against its digest."""
    path = shortfall_run(key, family)
    metrics = json.loads((path / "metrics.json").read_text())
    table = pd.read_csv(verify(path, metrics["artifacts"]["banks"]),
                        dtype={"bank": str}).set_index("bank")
    return table[["delta", "se", "se_boot", "ci_lo", "ci_hi",
                  "test_rows", "test_pos", "test_base_rate"]]


def target_replicates(key: str, family: str) -> np.ndarray:
    """Experiment 1's joint-bootstrap replicates of Δ_B, (replicates, banks), the banks in
    the order :func:`targets` returns."""
    path = shortfall_run(key, family)
    metrics = json.loads((path / "metrics.json").read_text())
    return np.load(verify(path, metrics["artifacts"]["replicates"]))


def weights(target: pd.DataFrame) -> np.ndarray:
    """The fitting weights: 1/s_B², the weights Experiment 1's heterogeneity statistic
    carries."""
    return 1.0 / target["se"].to_numpy(np.float64) ** 2


def aligned(descriptors: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Descriptor values in the target's bank order, after asserting the two
    cover exactly the same banks."""
    assert set(descriptors.index) == set(target.index), (
        f"the descriptor table covers {len(descriptors)} banks and the target "
        f"table {len(target)}; the symmetric difference is "
        f"{set(descriptors.index) ^ set(target.index)}")
    return descriptors.loc[target.index]
