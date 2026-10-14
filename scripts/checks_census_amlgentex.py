#!/usr/bin/env python
"""Census check: the AMLgentex exchangeability census.

Per dataset, one Run under ``experiments/checks/runs/`` records: the raw
pseudo-bank row share and the share of SAR rows touching source/sink (label
semantics are measured, not assumed); after the pseudo-bank filter, the
real-transfer, SAR-transfer, and distinct-pattern counts; the
laundering-pattern-to-bank span (patterns on SAR rows; alertID labels only
SAR rows) against its chance expectation — the analytic form
sum_b (1 - (1 - p_b)^a) over each pattern's a distinct accounts, with a
simulated multinomial null supplying the z; within-bank shares against
size-weighted chance for licit and laundering traffic separately, with a
shuffled-partition z; per-bank test-window SAR counts under either-endpoint
attribution; SAR front-loading (step quartiles and the test-window share);
and the control-selection verdict fields ``random_bank`` and
``meets_count_threshold``. The temporal split is cut on the filtered frame.
Per-bank and per-pattern tables land in the Run's ``artifacts/``.

Regression targets: on sweden_100K_dist_change_difficult,
97.8 % of SAR rows touch a pseudo-bank, and 2,474 SAR real transfers span
180 distinct patterns at mean 6.861 against a 6.819 chance prediction; on
sweden_100K_easy the span pair reads 6.690 against 6.687.

Usage:  uv run python scripts/checks_census_amlgentex.py [sweden_100K_easy ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from m import completed, run
from m.progress import track
from m.worlds import amlgentex

SHUFFLES = 20
SPAN_SIMS = 200
SEED = 0
COUNT_THRESHOLD = 30
CHANCE_Z = 3.0


def class_within_stats(f: np.ndarray, t: np.ndarray, node_bank: np.ndarray,
                       n_banks: int, rng: np.random.Generator) -> dict:
    """Observed within-bank share vs size-weighted chance, with shuffle z.

    The chance expectation is the sum of squared endpoint shares. The z
    permutes bank labels over (bank, account) nodes, holding each node's
    payment pattern fixed.
    """
    observed = float((node_bank[f] == node_bank[t]).mean())
    endpoints = np.bincount(node_bank[f], minlength=n_banks) \
        + np.bincount(node_bank[t], minlength=n_banks)
    p = endpoints / endpoints.sum()
    chance = float((p ** 2).sum())
    null = np.empty(SHUFFLES)
    for s in range(SHUFFLES):
        perm = rng.permutation(node_bank)
        null[s] = (perm[f] == perm[t]).mean()
    z = float((observed - null.mean()) / null.std()) if null.std() > 0 else float("inf")
    return {"within_share": observed, "chance": chance,
            "over_chance": observed / chance if chance > 0 else None,
            "shuffle_z": z}


def span_stats(pid: np.ndarray, bank_o: np.ndarray, bank_d: np.ndarray,
               acct_o: np.ndarray, acct_d: np.ndarray, p_acc: np.ndarray,
               rng: np.random.Generator, run_dir) -> dict:
    """Mean pattern-to-bank span vs chance under random account-to-bank
    assignment: analytic expectation per pattern, simulated null for the z."""
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
    sim = np.empty((SPAN_SIMS, len(pids)))
    for i, a in enumerate(accounts):
        sim[:, i] = (rng.multinomial(int(a), p_acc, size=SPAN_SIMS) > 0).sum(axis=1)
    sim_means = sim.mean(axis=1)

    pd.DataFrame({"pattern_id": pids, "n_transfers": transfers,
                  "n_accounts": accounts, "span": spans,
                  "expected_span": expected}) \
        .to_csv(run_dir / "artifacts" / "patterns.csv", index=False)

    mean_span = float(spans.mean())
    null_std = float(sim_means.std())
    return {"mean": mean_span,
            "chance_mean": float(expected.mean()),
            "chance_method": "analytic",
            "sim_null_mean": float(sim_means.mean()),
            "sim_null_std": null_std,
            "sim_z": (mean_span - float(sim_means.mean())) / null_std
                     if null_std > 0 else float("inf")}


def census(config: dict, run_dir) -> dict:
    df = amlgentex.load(config["dataset"])
    pseudo = amlgentex.pseudo_mask(df)
    sar_raw = df["is_sar"].to_numpy() == 1
    n_rows_raw = int(len(df))
    raw_pseudo_row_share = float(pseudo.mean())
    n_sar_rows_raw = int(sar_raw.sum())
    sar_rows_pseudo_share = float(pseudo[sar_raw].mean()) if n_sar_rows_raw else None

    real = amlgentex.drop_pseudo(df)
    del df, pseudo, sar_raw
    n = len(real)
    split = amlgentex.temporal_split(real)
    t1, t2 = amlgentex.split_bounds(real)
    step = real["step"].to_numpy()
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
    node_keys, node_codes = np.unique(
        np.concatenate([acct_o * n_banks + bank_o, acct_d * n_banks + bank_d]),
        return_inverse=True)
    from_node, to_node = node_codes[:n], node_codes[n:]
    node_bank = node_keys % n_banks

    rng = np.random.default_rng(SEED)
    within = {}
    for label, mask in (("licit", ~is_sar), ("laundering", is_sar)):
        within[label] = class_within_stats(from_node[mask], to_node[mask],
                                           node_bank, n_banks, rng) \
            if mask.any() else None

    # The laundering patterns are those on SAR rows, which on released data
    # is every pattern (alertID appears only on SAR rows, measured).
    laundering_pattern = (pat >= 0) & is_sar
    if laundering_pattern.any():
        p_acc = np.bincount(node_bank, minlength=n_banks) / len(node_bank)
        span = span_stats(pat[laundering_pattern].astype(np.int64),
                          bank_o[laundering_pattern], bank_d[laundering_pattern],
                          acct_o[laundering_pattern], acct_d[laundering_pattern],
                          p_acc, rng, run_dir)
        n_patterns = int(np.unique(pat[laundering_pattern]).size)
    else:
        span, n_patterns = None, 0
    n_patterns_incl_licit = int(np.unique(pat[pat >= 0]).size)

    def either_counts(mask: np.ndarray) -> np.ndarray:
        same = mask & (bank_o == bank_d)
        return (np.bincount(bank_o[mask], minlength=n_banks)
                + np.bincount(bank_d[mask], minlength=n_banks)
                - np.bincount(bank_o[same], minlength=n_banks))

    test_mask = split == 2
    test_sar = either_counts(test_mask & is_sar)
    all_sar = either_counts(is_sar)
    test_rows = either_counts(test_mask)

    pd.DataFrame({"bank": banks,
                  "accounts": np.bincount(node_bank, minlength=n_banks),
                  "test_sar_either": test_sar,
                  "sar_either": all_sar,
                  "test_rows_either": test_rows,
                  "test_base_rate": test_sar / np.where(test_rows > 0, test_rows, np.nan)}) \
        .to_csv(run_dir / "artifacts" / "banks.csv", index=False)

    sar_steps = step[is_sar]
    if len(sar_steps):
        q25, q50, q75 = np.quantile(sar_steps, [0.25, 0.5, 0.75])
        front_loading = {
            "step_min": int(sar_steps.min()), "step_q25": float(q25),
            "step_q50": float(q50), "step_q75": float(q75),
            "step_max": int(sar_steps.max()),
            "test_window_share": float((split[is_sar] == 2).mean()),
        }
    else:
        front_loading = None

    laundering_z = within["laundering"]["shuffle_z"] if within["laundering"] else None
    span_z = span["sim_z"] if span else None
    random_bank = bool(span_z is not None and laundering_z is not None
                       and abs(span_z) < CHANCE_Z and abs(laundering_z) < CHANCE_Z)
    meets_count_threshold = bool(n_banks > 0
                                 and int(test_sar.min()) >= COUNT_THRESHOLD)

    return {
        "dataset": config["dataset"],
        "n_rows_raw": n_rows_raw,
        "raw_pseudo_row_share": raw_pseudo_row_share,
        "n_sar_rows_raw": n_sar_rows_raw,
        "sar_rows_pseudo_share": sar_rows_pseudo_share,
        "n_real_transfers": int(n),
        "n_sar_transfers": int(is_sar.sum()),
        "n_patterns": n_patterns,
        "n_patterns_incl_licit": n_patterns_incl_licit,
        "n_real_banks": n_banks,
        "pattern_span": span,
        "within_bank": within,
        "per_bank_test_sar": {b: int(c) for b, c in zip(banks, test_sar)},
        "test_sar_min": int(test_sar.min()) if n_banks else None,
        "test_sar_max": int(test_sar.max()) if n_banks else None,
        "sar_front_loading": front_loading,
        "split_bounds_step": [t1, t2],
        "step_range": [int(step[0]), int(step[-1])],
        "verdict": {"random_bank": random_bank,
                    "meets_count_threshold": meets_count_threshold},
        "null_model": {"shuffles": SHUFFLES, "span_sims": SPAN_SIMS,
                       "seed": SEED, "chance_z": CHANCE_Z,
                       "count_threshold": COUNT_THRESHOLD},
    }


def summary(m: dict) -> str:
    span = m["pattern_span"]
    span_txt = f"{span['mean']:.3f}/{span['chance_mean']:.3f}" if span else "n/a"
    return (f"{m['dataset']}: "
            f"sar_pseudo {m['sar_rows_pseudo_share']:.3f} "
            f"span {span_txt} test_sar min {m['test_sar_min']} "
            f"random_bank={m['verdict']['random_bank']} "
            f"counts_ok={m['verdict']['meets_count_threshold']}")


def main() -> int:
    released = amlgentex.released_names()
    targets = sys.argv[1:] or released
    unknown = [n for n in targets if n not in released]
    if unknown:
        raise SystemExit(f"unknown or incomplete dataset: {unknown}")

    done = completed("checks")
    for name in track(targets, "amlgentex census"):
        metrics = run("checks", {"experiment": "checks", "check": "amlgentex_census",
                             "dataset": name},
                      census, done=done)
        print(f"\n{summary(metrics)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
