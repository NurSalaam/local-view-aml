#!/usr/bin/env python
"""Experiment 2's reading of confirmed cases across banks (Chapter 4). Analysis
only: no model trains. One ``examples`` Run per family (tree, GNN) correlates
the bank's number of confirmed cases (its training-window laundering payments)
with its Experiment 1 shortfall, raw and with size held fixed, and assigns
each bank its quarter by its number of confirmed cases. The config cites the
shortfall Run's hash, so a re-done input re-keys the reading; skip-if-done is
by config hash.

``counterparty_view`` splits each bank's test set into two parts by whether
the counterparty appears in the bank's ledger before the test window
(``NEVER_SEEN`` defines the first part); ``facts.py exp2`` reads the parts.

  uv run python scripts/exp2_examples.py examples <family>
  uv run python scripts/exp2_examples.py all
  uv run python scripts/exp2_examples.py status
"""
from __future__ import annotations

import json
import os
import sys
from functools import cache
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402, F401  (before lightgbm: OpenMP runtimes, see exp1.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1  # noqa: E402
import exp1_analysis  # noqa: E402
import exp1_data as data  # noqa: E402
from m import completed, run  # noqa: E402
from m.artifacts import verify, write_csv  # noqa: E402
from m.rank import conditional_permutation_p, spearman  # noqa: E402
from m.runlog import config_hash, experiments_root  # noqa: E402

EXPERIMENT = "exp2"
KEY = "HI-Medium"
FAMILIES = ("tree", "gnn")
PERMUTATIONS = 10_000
BLOCK_WIDTH = 5
SEED = 0

NEVER_SEEN = ("a cross-bank test payment whose counterparty account -- the "
              "endpoint at the other bank, keyed by its (bank, account) pair "
              "-- appears on none of the bank's training- or "
              "validation-window rows; a payment between two of the bank's "
              "own accounts has no counterparty and is never never-seen")
CORRELATIONS = ("Spearman rho; the partial by rank residualisation with the "
                f"conditional within-block permutation (width {BLOCK_WIDTH}, "
                f"{PERMUTATIONS} draws at seed {SEED}), holding fixed the "
                "log of the bank's training-window rows")


def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def banks_table(path: Path, metrics: dict) -> pd.DataFrame:
    return pd.read_csv(verify(path, metrics["artifacts"]["banks"]),
                       dtype={"bank": str}).set_index("bank")


def check_execution_settings() -> None:
    """The examples reading trains nothing, so Experiment 1's thread settings
    do not bind; the committed-tree rule does, except on a scratch Run log."""
    scratch = experiments_root().resolve() != (exp1.REPO_ROOT / "experiments").resolve()
    if not scratch and exp1.committed_tree():
        raise SystemExit("Runs launch only from a committed tree; uncommitted: "
                         + ", ".join(exp1.committed_tree()[:12]))


@cache
def counterparty_view(bank: str) -> dict:
    """One bank's ledger rows: the test-window mask, the ``NEVER_SEEN`` mask
    and the payment index. A pure function of the cached ledger."""
    ledger = data.read_ledger(KEY, bank)
    fb = ledger["from_bank"].astype(str)
    tb = ledger["to_bank"].astype(str)
    from_key = fb + ":" + ledger["from_acct"].astype(str)
    to_key = tb + ":" + ledger["to_acct"].astype(str)
    own_from = (fb == bank).to_numpy()
    cross = (fb != tb).to_numpy()
    cp_key = np.where(own_from, to_key.to_numpy(), from_key.to_numpy())
    test = ledger["split"].to_numpy() == 2
    seen = pd.Index(pd.unique(pd.concat([from_key[~test], to_key[~test]],
                                        ignore_index=True)))
    return {"test": test, "never": cross & (seen.get_indexer(cp_key) < 0),
            "payment": ledger["payment"].to_numpy()}


def correlation_block(x: np.ndarray, delta: np.ndarray,
                      log_rows: np.ndarray) -> dict:
    partial = conditional_permutation_p(x, delta, log_rows, width=BLOCK_WIDTH,
                                        draws=PERMUTATIONS, seed=SEED)
    return {"k": int(len(x)), "rho": spearman(x, delta),
            "partial_rho": partial["rho"],
            "partial_p_conditional": partial["p_conditional"]}


# ------------------------------------------------------------- examples --

def examples_config(family: str, shortfall: tuple[Path, dict]) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": EXPERIMENT, "check": "examples", "world": world,
            "dataset": dataset, "dataset_key": KEY, "family": family,
            "predictor": "the bank's number of confirmed cases, its "
                         "training-window laundering payments (roster "
                         "train_pos, either-endpoint attribution)",
            "target": "the bank's Experiment 1 shortfall (banks.csv delta)",
            "statistics": CORRELATIONS,
            "quartiles": "pandas qcut over the count, four bins, as the "
                         "examples.csv quartile column (0 holds the fewest)",
            "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(KEY)},
            "inputs": {"shortfall": _hash_of(shortfall[0])}}


def examples_metrics(family: str, shortfall: tuple[Path, dict],
                     run_dir: Path) -> dict:
    banks = banks_table(*shortfall)
    roster = data.read_roster(KEY)
    table = banks[["delta", "se", "test_rows", "test_pos"]].copy()
    table["train_pos"] = roster.loc[table.index, "train_pos"]
    table["train_rows"] = roster.loc[table.index, "train_rows"]
    train_pos = table["train_pos"].to_numpy(np.float64)
    delta = table["delta"].to_numpy()
    log_rows = np.log(table["train_rows"].to_numpy(np.float64))
    correlation = correlation_block(train_pos, delta, log_rows)
    table["quartile"] = pd.qcut(table["train_pos"], 4, labels=False)
    return {"dataset_key": KEY, "family": family, "k": len(table),
            "correlation_with_shortfall": correlation,
            "artifacts": {"examples": write_csv(run_dir, "examples", table)}}


def cmd_examples(family: str, done: dict | None = None) -> dict | None:
    done = done if done is not None else completed(EXPERIMENT)
    shortfall = exp1_analysis.shortfall_run(completed(exp1.EXPERIMENT), KEY, family)
    if shortfall is None:
        print(f"examples {family}: Experiment 1 shortfall Run not found", flush=True)
        return None
    metrics = run(EXPERIMENT, examples_config(family, shortfall),
                  lambda config, run_dir: examples_metrics(family, shortfall,
                                                           run_dir),
                  done=done)
    c = metrics["correlation_with_shortfall"]
    print(f"examples {family}: rho {c['rho']:+.3f} partial "
          f"{c['partial_rho']:+.3f} (p {c['partial_p_conditional']:.4f})", flush=True)
    return metrics


# ----------------------------------------------------------------- main --

def cmd_all() -> int:
    done = completed(EXPERIMENT)
    for family in FAMILIES:
        cmd_examples(family, done)
    return 0


def cmd_status() -> int:
    done = completed(EXPERIMENT)
    done_exp1 = completed(exp1.EXPERIMENT)
    for family in FAMILIES:
        shortfall = exp1_analysis.shortfall_run(done_exp1, KEY, family)
        state = ("inputs incomplete" if shortfall is None else
                 "analysed" if config_hash(examples_config(family, shortfall))
                 in done else "ready")
        print(f"examples {family:4s} {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    check_execution_settings()
    exp1.install_signal_handlers()
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "examples":
        return 0 if cmd_examples(args[0]) is not None else 1
    if cmd == "all":
        return cmd_all()
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
