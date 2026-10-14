#!/usr/bin/env python
"""Census check: the TransXion census.

One Run under ``experiments/checks/runs/`` records: the account-to-bank
assignment (a clean function the transaction, person, and merchant tables
agree on); the attribute reconciliation of 47,730 person + merchant rows
against the transacting accounts; within-bank shares against size-weighted
chance for licit and laundering traffic separately; the planted-region
reconstruction (region = int(account[1:]) mod 20, with a mod-19 control
parse), its within-region shares, shuffle z, and region-bank independence;
per-bank and per-region test-window laundering positives under
either-endpoint attribution; per-bank test base rates; and the mean share of
payments a single bank's local view removes. Per-bank and per-region tables
land in the Run's ``artifacts/banks.csv`` and ``artifacts/regions.csv``.

Regression targets: 30 banks at 100 % assignment
agreement; 47,730 attribute rows vs 47,526 transacting accounts; within-bank
licit 3.36 % vs 3.35 % chance and laundering 3.66 % vs 3.45 % (the
laundering figure counts same-account rows); within-region licit 88.4 % vs
5.1 % chance, laundering 24.4 %, shuffle z ~ +43; per-bank test positives
36-111, per-region 41-158; local-view edge drop ~ 93 %.

Usage:  uv run python scripts/checks_census_transxion.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from m import completed, run
from m.worlds import transxion

SHUFFLES = 20
SHUFFLE_SEED = 0
REGION_MOD = 20
CONTROL_MOD = 19
BANK_POS_BAND = (36, 111)
REGION_POS_BAND = (41, 158)


def within_share_stats(from_node: np.ndarray, to_node: np.ndarray,
                       node_group: np.ndarray, is_laundering: np.ndarray,
                       rng: np.random.Generator) -> dict:
    """Observed within-group share vs chance, per class, with shuffle z.

    Operates on real transfers only (same-account rows excluded). The chance
    expectation is size-weighted: sum of squared endpoint shares. The z score
    permutes group labels over account nodes, holding each node's payment
    pattern fixed.
    """
    n_groups = int(node_group.max()) + 1
    out = {}
    for label, mask in [("licit", is_laundering == 0),
                        ("laundering", is_laundering == 1)]:
        f, t = from_node[mask], to_node[mask]
        observed = float((node_group[f] == node_group[t]).mean())
        endpoints = np.bincount(node_group[f], minlength=n_groups) \
            + np.bincount(node_group[t], minlength=n_groups)
        p = endpoints / endpoints.sum()
        chance = float((p ** 2).sum())
        null = np.empty(SHUFFLES)
        for s in range(SHUFFLES):
            perm = rng.permutation(node_group)
            null[s] = (perm[f] == perm[t]).mean()
        z = (observed - null.mean()) / null.std() if null.std() > 0 else np.inf
        out[label] = {"within_share": observed, "chance": chance,
                      "over_chance": observed / chance, "shuffle_z": float(z)}
    return out


def either_endpoint_counts(from_group: pd.Series, to_group: pd.Series) -> pd.Series:
    """Rows attributed to each group once: cross-group rows count for both
    endpoints, within-group rows count once."""
    total = from_group.value_counts().add(to_group.value_counts(), fill_value=0)
    same = from_group[from_group.values == to_group.values].value_counts()
    return total.sub(same, fill_value=0).astype(int)


def census(config: dict, run_dir) -> dict:
    df = transxion.load()
    person = transxion.persons()
    merchant = transxion.merchants()

    real = df[~df["same_account"]]
    test = df[df["split"] == 2]
    pos = test[test["is_laundering"] == 1]

    endpoints = pd.concat([
        df[["from_bank", "from_acct"]].rename(columns={"from_bank": "bank", "from_acct": "acct"}),
        df[["to_bank", "to_acct"]].rename(columns={"to_bank": "bank", "to_acct": "acct"}),
    ], ignore_index=True).astype({"bank": str, "acct": str})
    pairs = endpoints.drop_duplicates()
    accounts_multi_bank = int((pairs.groupby("acct")["bank"].nunique() > 1).sum())
    tx_bank = pairs.set_index("acct")["bank"]

    agreement = {}
    for name, table in [("person", person), ("merchant", merchant)]:
        listed = table.set_index("bank_account_number")["bank"].astype(str)
        compared = listed.index.intersection(tx_bank.index)
        agreement[name] = {
            "compared": int(len(compared)),
            "agree_share": float((listed.loc[compared] == tx_bank.loc[compared]).mean()),
        }

    attributed = pd.concat([person["bank_account_number"], merchant["bank_account_number"]]).astype(str)
    silent = set(attributed) - set(tx_bank.index)
    silent_persons = person[person["bank_account_number"].astype(str).isin(silent)]
    silent_merchants = merchant[merchant["bank_account_number"].astype(str).isin(silent)]
    representatives = set(merchant["legal_representative_id"].astype(str))

    n = len(real)
    acct = pd.Categorical(pd.concat([real["from_acct"].astype(str),
                                     real["to_acct"].astype(str)], ignore_index=True))
    from_node, to_node = acct.codes[:n], acct.codes[n:]
    node_digits = pd.Series(acct.categories).str[1:].astype(int)
    node_region = (node_digits % REGION_MOD).to_numpy()
    node_control = (node_digits % CONTROL_MOD).to_numpy()
    node_bank = pd.Categorical(tx_bank.reindex(acct.categories).values).codes
    lab = real["is_laundering"].values

    bank_structure = within_share_stats(from_node, to_node, node_bank, lab,
                                        np.random.default_rng(SHUFFLE_SEED))
    region_structure = within_share_stats(from_node, to_node, node_region, lab,
                                          np.random.default_rng(SHUFFLE_SEED))

    control = {}
    for label, mask in [("licit", lab == 0), ("laundering", lab == 1)]:
        f, t = from_node[mask], to_node[mask]
        e = np.bincount(node_control[f], minlength=CONTROL_MOD) \
            + np.bincount(node_control[t], minlength=CONTROL_MOD)
        p = e / e.sum()
        control[label] = {"within_share": float((node_control[f] == node_control[t]).mean()),
                          "chance": float((p ** 2).sum())}

    # The laundering regression targets (3.66 % within bank, 24.4 % within
    # region) were computed over all rows; same-account rows are within-bank
    # and within-region by construction, so both variants are recorded.
    incl = {}
    for scope, cols in [("bank", ("from_bank", "to_bank")), ("region", None)]:
        incl[scope] = {}
        for label, sub in [("licit", df[df["is_laundering"] == 0]),
                           ("laundering", df[df["is_laundering"] == 1])]:
            if cols:
                match = sub[cols[0]].astype(str).values == sub[cols[1]].astype(str).values
            else:
                match = (sub["from_acct"].astype(str).str[1:].astype(int).values % REGION_MOD
                         == sub["to_acct"].astype(str).str[1:].astype(int).values % REGION_MOD)
            incl[scope][label] = float(match.mean())

    bank_pos = either_endpoint_counts(pos["from_bank"].astype(str), pos["to_bank"].astype(str))
    touching_test = either_endpoint_counts(test["from_bank"].astype(str), test["to_bank"].astype(str))
    touching_all = either_endpoint_counts(df["from_bank"].astype(str), df["to_bank"].astype(str))
    active = endpoints.groupby("bank")["acct"].nunique().astype(int)

    banks = pd.DataFrame({
        "touching_payments": touching_all,
        "touching_test_rows": touching_test,
        "test_pos": bank_pos,
        "active_accounts": active,
        "persons_listed": person.groupby("bank").size(),
        "merchants_listed": merchant.groupby("bank").size(),
    }).fillna(0).astype(int)
    banks["test_base_rate"] = banks["test_pos"] / banks["touching_test_rows"].replace(0, np.nan)
    banks.index.name = "bank"
    banks.sort_values("test_pos", ascending=False).to_csv(run_dir / "artifacts" / "banks.csv")

    region_pos = either_endpoint_counts(
        pos["from_acct"].astype(str).str[1:].astype(int) % REGION_MOD,
        pos["to_acct"].astype(str).str[1:].astype(int) % REGION_MOD)
    regions = pd.DataFrame({
        "test_pos": region_pos,
        "active_accounts": pd.Series(node_region).value_counts(),
    }).fillna(0).astype(int)
    regions.index.name = "region"
    regions.sort_index().to_csv(run_dir / "artifacts" / "regions.csv")

    contingency = pd.crosstab(node_region, node_bank).values
    n_accounts = contingency.sum()
    expected = np.outer(contingency.sum(1), contingency.sum(0)) / n_accounts
    chi2 = float(((contingency - expected) ** 2 / expected).sum())
    cramers_v = float(np.sqrt(chi2 / (n_accounts * (min(contingency.shape) - 1))))

    drop = 1 - touching_all / len(df)
    base = banks["test_base_rate"].dropna()

    return {
        "n_rows": int(len(df)),
        "n_laundering": int(df["is_laundering"].sum()),
        "test_pos_total": int(test["is_laundering"].sum()),
        "same_account": {"rows": int(df["same_account"].sum()),
                         "laundering": int(df.loc[df["same_account"], "is_laundering"].sum()),
                         "row_share": float(df["same_account"].mean())},
        "bank_assignment": {
            "n_banks_tx": int(tx_bank.nunique()),
            "n_banks_person": int(person["bank"].nunique()),
            "n_banks_merchant": int(merchant["bank"].nunique()),
            "bank_sets_identical": bool(set(tx_bank) == set(person["bank"].astype(str))
                                        and set(tx_bank) == set(merchant["bank"].astype(str))),
            "accounts_multi_bank_tx": accounts_multi_bank,
            "agreement": agreement,
        },
        "attributes": {
            "person_rows": int(len(person)),
            "merchant_rows": int(len(merchant)),
            "attributed_accounts": int(attributed.nunique()),
            "transacting_accounts": int(len(tx_bank)),
            "attributed_not_transacting": {
                "total": int(len(silent)),
                "persons": int(len(silent_persons)),
                "merchants": int(len(silent_merchants)),
                "silent_persons_serving_as_legal_representative":
                    int(silent_persons["person_id"].astype(str).isin(representatives).sum()),
            },
            "transacting_not_attributed": int(len(set(tx_bank.index) - set(attributed))),
        },
        "within_bank_structure": bank_structure,
        "within_bank_share_incl_same_account": incl["bank"],
        "region": {
            "formula": "int(account[1:]) % 20",
            "n_regions": int(len(regions)),
            "within_region_structure": region_structure,
            "within_region_share_incl_same_account": incl["region"],
            "control_formula": f"int(account[1:]) % {CONTROL_MOD}",
            "control_within_share": control,
            "region_bank_chi2": chi2,
            "region_bank_dof": int((contingency.shape[0] - 1) * (contingency.shape[1] - 1)),
            "region_bank_cramers_v": cramers_v,
            "region_accounts_range": [int(contingency.sum(1).min()), int(contingency.sum(1).max())],
            "test_pos_range": [int(region_pos.min()), int(region_pos.max())],
            "all_regions_in_band": bool(region_pos.between(*REGION_POS_BAND).all()
                                        and len(region_pos) == REGION_MOD),
        },
        "per_bank": {
            "test_pos_range": [int(bank_pos.min()), int(bank_pos.max())],
            "all_banks_in_band": bool(bank_pos.between(*BANK_POS_BAND).all()
                                      and len(bank_pos) == 30),
            "test_base_rate_range": [float(base.min()), float(base.max())],
            "test_base_rate_fold": float(base.max() / base.min()),
        },
        "local_view_edge_drop": {"mean": float(drop.mean()),
                                 "min": float(drop.min()),
                                 "max": float(drop.max())},
    }


def main() -> int:
    metrics = run("checks", {"experiment": "checks", "check": "transxion_census"},
                  census, done=completed("checks"))
    print(f"transxion: banks {metrics['bank_assignment']['n_banks_tx']} "
          f"multi-bank accts {metrics['bank_assignment']['accounts_multi_bank_tx']} "
          f"attr {metrics['attributes']['attributed_accounts']}-"
          f"{metrics['attributes']['transacting_accounts']}="
          f"{metrics['attributes']['attributed_not_transacting']['total']} "
          f"pos/bank {metrics['per_bank']['test_pos_range']} "
          f"pos/region {metrics['region']['test_pos_range']} "
          f"drop {metrics['local_view_edge_drop']['mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
