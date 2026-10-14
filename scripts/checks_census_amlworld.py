#!/usr/bin/env python
"""Census check: the AMLworld censuses.

Per dataset, one Run under ``experiments/checks/runs/`` records: bank counts under
padded and int-cast identifiers; evaluable banks at >= 10/30/50 test-window
laundering payments (either-endpoint attribution); the usable roster after the
bank-070 exclusion; per-bank test base rates; within-bank shares against their
size-weighted chance expectations, with same-account rows excluded and a
shuffled-partition z; the same-account row share; the injected-pattern
composition; and active-account counts beside payment counts. The per-bank
table lands in the Run's ``artifacts/banks.csv``.

Regression targets: evaluable at >= 30 must read
HI-Small 12, LI-Small 4, HI-Medium 108, LI-Medium 3 (11 and 107 usable after
the 070 exclusion); composition 62.0 / 28.7 / 64.6 / 24.4 %.

Usage:  uv run python scripts/checks_census_amlworld.py [HI-Small ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from m import completed, run
from m.progress import track
from m.worlds import amlworld

SHUFFLES = 20
SHUFFLE_SEED = 0
THRESHOLDS = (10, 30, 50)


def within_share_stats(df: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Observed within-bank share vs chance, per class, with shuffle z.

    Operates on real transfers only (same-account rows excluded). The chance
    expectation is size-weighted: sum of squared endpoint shares. The z score
    permutes bank labels over (bank, account) nodes, holding each node's
    payment pattern fixed.
    """
    node = pd.Categorical(
        pd.concat([df["from_bank"].astype(str) + "/" + df["from_acct"].astype(str),
                   df["to_bank"].astype(str) + "/" + df["to_acct"].astype(str)],
                  ignore_index=True))
    n = len(df)
    from_node, to_node = node.codes[:n], node.codes[n:]
    node_bank = pd.Categorical(
        pd.Series(node.categories).str.split("/", n=1).str[0]).codes

    out = {}
    for label, mask in [("licit", df["is_laundering"].values == 0),
                        ("laundering", df["is_laundering"].values == 1)]:
        f, t = from_node[mask], to_node[mask]
        observed = float((node_bank[f] == node_bank[t]).mean())
        endpoints = np.bincount(node_bank[f], minlength=node_bank.max() + 1) \
            + np.bincount(node_bank[t], minlength=node_bank.max() + 1)
        p = endpoints / endpoints.sum()
        chance = float((p ** 2).sum())
        null = np.empty(SHUFFLES)
        for s in range(SHUFFLES):
            perm = rng.permutation(node_bank)
            null[s] = (perm[f] == perm[t]).mean()
        z = (observed - null.mean()) / null.std() if null.std() > 0 else np.inf
        out[label] = {"within_share": observed, "chance": chance,
                      "over_chance": observed / chance, "shuffle_z": float(z)}
    return out


def census(config: dict, run_dir) -> dict:
    dataset = config["dataset"]
    df = amlworld.load(dataset)

    real = df[~df["same_account"]]
    test = df[df["split"] == 2]

    pos = test[test["is_laundering"] == 1]
    either = pd.concat([
        pos.groupby("from_bank", observed=True).size(),
        pos.groupby("to_bank", observed=True).size()], axis=1).fillna(0)
    test_pos = either[0].add(either[1], fill_value=0)
    same_bank_pos = pos[pos["from_bank"] == pos["to_bank"]] \
        .groupby("from_bank", observed=True).size()
    test_pos = test_pos.sub(same_bank_pos, fill_value=0).astype(int)
    test_pos = test_pos[test_pos > 0]

    test_rows = pd.concat([
        test.groupby("from_bank", observed=True).size(),
        test.groupby("to_bank", observed=True).size()], axis=1).fillna(0)
    touching_test = test_rows[0].add(test_rows[1], fill_value=0)
    same_bank_rows = test[test["from_bank"] == test["to_bank"]] \
        .groupby("from_bank", observed=True).size()
    touching_test = touching_test.sub(same_bank_rows, fill_value=0).astype(int)

    touching_all = pd.concat([
        df.groupby("from_bank", observed=True).size(),
        df.groupby("to_bank", observed=True).size()], axis=1).fillna(0)
    touching_all = touching_all[0].add(touching_all[1], fill_value=0)
    same_all = df[df["from_bank"] == df["to_bank"]] \
        .groupby("from_bank", observed=True).size()
    touching_all = touching_all.sub(same_all, fill_value=0).astype(int)

    endpoints = pd.concat([
        df[["from_bank", "from_acct"]].rename(columns={"from_bank": "bank", "from_acct": "acct"}),
        df[["to_bank", "to_acct"]].rename(columns={"to_bank": "bank", "to_acct": "acct"}),
    ], ignore_index=True)
    active = endpoints.groupby("bank", observed=True)["acct"].nunique().astype(int)
    listed = amlworld.accounts_per_bank(dataset)

    banks = pd.DataFrame({
        "touching_payments": touching_all,
        "touching_test_rows": touching_test,
        "test_pos": test_pos,
        "active_accounts": active,
    }).fillna(0).astype(int)
    # The accounts file ships unpadded identifiers (already int-cast), so the
    # listed-account count cannot distinguish zero-padded bank pairs; the join
    # goes through int() and the resolution is disclosed in the metrics.
    banks["listed_accounts_intcast"] = (
        banks.index.map(lambda b: listed.get(str(int(b)), 0)).astype(int))
    banks["test_base_rate"] = banks["test_pos"] / banks["touching_test_rows"].replace(0, np.nan)
    banks.index.name = "bank"
    banks.sort_values("test_pos", ascending=False).to_csv(run_dir / "artifacts" / "banks.csv")

    rng = np.random.default_rng(SHUFFLE_SEED)
    structure = within_share_stats(real, rng)
    no_rail = real[(real["from_bank"] != amlworld.RAIL_BANK)
                   & (real["to_bank"] != amlworld.RAIL_BANK)]
    structure_no_rail = within_share_stats(no_rail, np.random.default_rng(SHUFFLE_SEED))

    evaluable = {str(t): int((test_pos >= t).sum()) for t in THRESHOLDS}
    usable = {str(t): int(((test_pos >= t) & (test_pos.index != amlworld.RAIL_BANK)).sum())
              for t in THRESHOLDS}

    eligible = banks[banks["test_pos"] >= 30]
    base = eligible["test_base_rate"].dropna()

    all_banks = pd.Categorical(pd.concat([df["from_bank"].astype(str),
                                          df["to_bank"].astype(str)])).categories
    return {
        "dataset": dataset,
        "n_rows": int(len(df)),
        "n_laundering": int(df["is_laundering"].sum()),
        "n_banks_padded": int(len(all_banks)),
        "n_banks_int_cast": int(len({int(b) for b in all_banks})),
        "same_account_row_share": float(df["same_account"].mean()),
        "cross_bank_share_real": {
            "licit": float((real.loc[real["is_laundering"] == 0, "from_bank"].astype(str).values
                            != real.loc[real["is_laundering"] == 0, "to_bank"].astype(str).values).mean()),
            "laundering": float((real.loc[real["is_laundering"] == 1, "from_bank"].astype(str).values
                                 != real.loc[real["is_laundering"] == 1, "to_bank"].astype(str).values).mean()),
        },
        "evaluable_banks": evaluable,
        "usable_banks_excl_070": usable,
        "rail_bank_test_pos": int(test_pos.get(amlworld.RAIL_BANK, 0)),
        "test_pos_total": int(test["is_laundering"].sum()),
        "test_base_rate_range": [float(base.min()), float(base.max())] if len(base) else None,
        "test_base_rate_fold": float(base.max() / base.min()) if len(base) else None,
        "touching_payments_range_eligible_excl_070":
            [int(eligible.loc[eligible.index != amlworld.RAIL_BANK, "touching_payments"].min()),
             int(eligible.loc[eligible.index != amlworld.RAIL_BANK, "touching_payments"].max())]
            if len(eligible) else None,
        "n_banks_accounts_file_intcast": int(len(listed)),
        "banks_with_1000_accounts_intcast": int((listed >= 1000).sum()),
        "pattern_composition": amlworld.pattern_member_rows(dataset)
            / int(df["is_laundering"].sum()),
        "within_bank_structure": structure,
        "within_bank_structure_excl_070": structure_no_rail,
    }


def main() -> int:
    datasets = sys.argv[1:] or amlworld.SETS
    done = completed("checks")
    for dataset in track(datasets, "amlworld census"):
        metrics = run("checks", {"experiment": "checks", "check": "amlworld_census",
                             "dataset": dataset, "shuffles": SHUFFLES,
                             "shuffle_seed": SHUFFLE_SEED},
                      census, done=done)
        print(f"\n{dataset}: evaluable {metrics['evaluable_banks']} "
              f"usable {metrics['usable_banks_excl_070']} "
              f"composition {metrics['pattern_composition']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
