#!/usr/bin/env python
"""Experiment 1's readings beside the verdict (Chapter 4; Appendix C).

Three analysis-only Runs read Experiment 1's persisted shortfall artefacts.
``residual_after_size`` fits the weighted line of the shortfall on log size
and reports tau about it (tau, the estimated standard deviation of the true
shortfalls, is the value at which the expected statistic Q equals the
observed one); ``own_best`` sets the paired shortfall beside each scenario
scored at its own best configuration; ``spread_readings`` holds the readings
of the spread that Chapter 4 and Appendix C quote beyond the shortfall Runs:
leaving each bank out in turn, the null check of each robustness check, the
factor on the error bars, the two terms of tau and tau in the world with no
real spread, the banks with at least 100 confirmed cases, the agreement
between the two families, size counted in either window, the own-best
reading of the tree family, and the ranges of the controls. Every config
cites its inputs' Run hashes, so a re-done input re-keys the analysis.

  uv run python scripts/exp1_readings.py residual
  uv run python scripts/exp1_readings.py paired
  uv run python scripts/exp1_readings.py spread
  uv run python scripts/exp1_readings.py all
  uv run python scripts/exp1_readings.py status
"""
from __future__ import annotations

import json
import os
import sys
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
from m.rank import partial_spearman  # noqa: E402
from m.rank import spearman as rank_spearman  # noqa: E402
from m.runlog import config_hash, find, index  # noqa: E402
from m.stats import (heterogeneity, leave_each_out, null_q_check,  # noqa: E402
                     null_q_replicates, spearman, tau_reading)

KEY = exp1.PRIMARY_KEY
FAMILIES = ("tree", "gnn")
PAIRED_FAMILIES = ("tree", "blind")
SIZE_REGRESSOR = "natural log of the bank's test-window payments (test_rows)"
SMALL = "HI-Small"
FIFTY = 50
MATCHED_TRAIN_POS = 100
WORKED_PAIR = ("0272142", "008904")
NULL_LEVEL = 0.95
SPREAD_KEYS = (KEY, SMALL, "transxion", data.AMLGENTEX_DATASET, *data.RELEASES)


def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def estimate_tau(deltas: np.ndarray, ses: np.ndarray) -> float:
    """tau, the estimated standard deviation of the true shortfalls: the
    spread that would make the observed Q equal its expectation under that
    spread."""
    usable = ses > 0
    d, s = deltas[usable], ses[usable]
    k = len(d)
    w = 1.0 / s ** 2
    mean = (w * d).sum() / w.sum()
    q = (w * (d - mean) ** 2).sum()
    denom = w.sum() - (w ** 2).sum() / w.sum()
    if denom <= 0:
        return 0.0
    return float(np.sqrt(max(0.0, (q - (k - 1)) / denom)))


def size_gradient(deltas: np.ndarray, test_rows: np.ndarray) -> float:
    return spearman(deltas, np.log(test_rows.astype(np.float64)))


def wls_residual(deltas: np.ndarray, ses: np.ndarray,
                 log_size: np.ndarray) -> dict:
    """Weighted least squares of the shortfall on log size; the residual Q,
    I-squared (k - 2 degrees of freedom) and tau about the line alongside the
    fit and the residuals."""
    usable = ses > 0
    d, s, x = deltas[usable], ses[usable], log_size[usable]
    k = len(d)
    w = 1.0 / s ** 2
    X = np.column_stack([np.ones(k), x])
    XtWX = X.T @ (w[:, None] * X)
    beta = np.linalg.solve(XtWX, X.T @ (w * d))
    fitted = X @ beta
    resid = d - fitted
    q = float((w * resid ** 2).sum())
    p = 2
    i2 = max(0.0, (q - (k - p)) / q) if q > 0 else 0.0
    trace = w.sum() - np.trace(np.linalg.solve(XtWX, X.T @ ((w ** 2)[:, None] * X)))
    tau2 = max(0.0, (q - (k - p)) / trace) if trace > 0 else 0.0
    return {"intercept": float(beta[0]), "slope": float(beta[1]),
            "Q": q, "I2": float(i2), "tau": float(np.sqrt(tau2)), "k": k,
            "fitted": fitted, "residuals": resid, "usable": usable}


def shortfall_inputs(done: dict, families: tuple[str, ...]) -> dict[str, tuple[Path, dict]]:
    """Experiment 1's HI-Medium shortfall Runs, one per family."""
    out = {}
    for family in families:
        found = exp1_analysis.shortfall_run(done, KEY, family)
        if found is None:
            raise SystemExit(f"Experiment 1 shortfall Run for {KEY}/{family} not found")
        out[family] = found
    return out


def banks_table(path: Path, metrics: dict) -> pd.DataFrame:
    return pd.read_csv(verify(path, metrics["artifacts"]["banks"]),
                       dtype={"bank": str}).set_index("bank")


# --------------------------------------------------- residual after size --

def residual_config(inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": exp1.EXPERIMENT, "check": "residual_after_size",
            "world": world, "dataset": dataset, "dataset_key": KEY,
            "families": list(FAMILIES),
            "reading": "residual after size",
            "regressor": SIZE_REGRESSOR,
            "weights": "1/s_B^2 from each family's Experiment 1 shortfall banks.csv",
            "estimator": "weighted least squares; residual I2 on k-2 degrees "
                         "of freedom; tau about the line, the value at which "
                         "the expected Q about the weighted line equals the "
                         "observed one",
            "inputs": {family: _hash_of(path)
                       for family, (path, _) in inputs.items()}}


def residual_metrics(inputs: dict, run_dir: Path) -> dict:
    rows = []
    readings = {}
    for family, (path, metrics) in inputs.items():
        table = banks_table(path, metrics)
        deltas = table["delta"].to_numpy()
        ses = table["se"].to_numpy()
        log_size = np.log(table["test_rows"].to_numpy(np.float64))
        raw = heterogeneity(deltas, ses)
        fit = wls_residual(deltas, ses, log_size)
        usable = fit.pop("usable")
        fitted, residuals = fit.pop("fitted"), fit.pop("residuals")
        readings[family] = {
            "k": raw["k"],
            "raw": {"Q": raw["Q"], "I2": raw["I2"],
                    "tau": estimate_tau(deltas, ses),
                    "weighted_mean": raw["weighted_mean"]},
            "fit": {"intercept": fit["intercept"], "slope": fit["slope"]},
            "residual": {"Q": fit["Q"], "I2": fit["I2"], "tau": fit["tau"]}}
        banks = table.index.to_numpy()[usable]
        for bank, d, s, x, f, r in zip(banks, deltas[usable], ses[usable],
                                       log_size[usable], fitted, residuals):
            rows.append({"bank": bank, "family": family, "delta": d, "se": s,
                         "log_test_rows": x, "fitted": f, "residual": r})
    frame = pd.DataFrame(rows).set_index("bank")
    return {"dataset_key": KEY, "readings": readings,
            "artifacts": {"residuals": write_csv(run_dir, "residuals", frame)}}


def cmd_residual(done: dict | None = None) -> dict:
    done = done if done is not None else completed(exp1.EXPERIMENT)
    inputs = shortfall_inputs(done, FAMILIES)
    metrics = run(exp1.EXPERIMENT, residual_config(inputs),
                  lambda config, run_dir: residual_metrics(inputs, run_dir),
                  done=done)
    for family, r in metrics["readings"].items():
        print(f"residual {family}: I2 {r['raw']['I2']:.3f} -> "
              f"{r['residual']['I2']:.3f}, tau {r['raw']['tau']:.4f} -> "
              f"{r['residual']['tau']:.4f}, slope {r['fit']['slope']:+.4f}",
              flush=True)
    return metrics


# -------------------------------------------- each scenario's own best --

def paired_config(inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": exp1.EXPERIMENT, "check": "own_best",
            "world": world, "dataset": dataset, "dataset_key": KEY,
            "families": list(PAIRED_FAMILIES),
            "reading": "pairing beside each scenario's own best configuration",
            "own_best": "each scenario scored at its own configuration of "
                        "highest validation average precision, from the "
                        "own_best_delta column Experiment 1 recorded",
            "note": "I2 at each scenario's own best configuration uses the "
                    "paired shortfall's error bar s_B; the GNN family is "
                    "fitted at one configuration, so it has none to choose",
            "weights": "1/s_B^2 from each family's Experiment 1 shortfall banks.csv",
            "inputs": {family: _hash_of(inputs[family][0])
                       for family in PAIRED_FAMILIES}}


def paired_metrics(inputs: dict, run_dir: Path) -> dict:
    readings = {}
    for family in PAIRED_FAMILIES:
        path, metrics = inputs[family]
        table = banks_table(path, metrics)
        paired = table["delta"].to_numpy()
        own_best = table["own_best_delta"].to_numpy()
        ses = table["se"].to_numpy()
        het_paired = heterogeneity(paired, ses)
        het_own_best = heterogeneity(own_best, ses)
        readings[family] = {
            "k": het_paired["k"],
            "weighted_mean_paired": het_paired["weighted_mean"],
            "weighted_mean_own_best": het_own_best["weighted_mean"],
            "I2_paired": het_paired["I2"], "I2_own_best": het_own_best["I2"],
            "sign_flips": int((np.sign(paired) != np.sign(own_best)).sum()),
            "rank_agreement_spearman": spearman(paired, own_best)}
    return {"dataset_key": KEY, "readings": readings}


def cmd_paired(done: dict | None = None) -> dict:
    done = done if done is not None else completed(exp1.EXPERIMENT)
    inputs = shortfall_inputs(done, PAIRED_FAMILIES)
    metrics = run(exp1.EXPERIMENT, paired_config(inputs),
                  lambda config, run_dir: paired_metrics(inputs, run_dir),
                  done=done)
    for family, r in metrics["readings"].items():
        print(f"paired {family}: mean {r['weighted_mean_paired']:+.4f} -> "
              f"own-best {r['weighted_mean_own_best']:+.4f}, flips "
              f"{r['sign_flips']}/{r['k']}, I2 {r['I2_paired']:.3f} -> "
              f"{r['I2_own_best']:.3f}, rho {r['rank_agreement_spearman']:.3f}",
              flush=True)
    return metrics


# ----------------------------------------------------- spread readings --

def cell(labels, deltas, ses, reps=None, seed_var=None) -> dict:
    """One set of shortfalls to read: the arrays every reading needs."""
    return {"labels": list(labels), "deltas": np.asarray(deltas, np.float64),
            "ses": np.asarray(ses, np.float64), "reps": reps, "seed_var": seed_var}


def shortfall_cell(banks: pd.DataFrame, reps: np.ndarray, mask=None) -> dict:
    keep = np.ones(len(banks), bool) if mask is None else mask
    seed_var = banks["seed_var"].to_numpy()[keep] if "seed_var" in banks else None
    return cell(banks.index[keep], banks["delta"].to_numpy()[keep],
                banks["se"].to_numpy()[keep], reps[:, keep], seed_var)


def seed_cell(banks: pd.DataFrame, reps_by_seed: np.ndarray, seed: int) -> dict:
    """One GNN seed on its own replicates, s_B their standard deviation."""
    reps = reps_by_seed[:, :, seed]
    return cell(banks.index, banks[f"delta_s{seed}"],
                np.nanstd(reps, axis=0, ddof=1), reps)


def spread_cells(tables: dict, reps: dict, reps_by_seed: np.ndarray) -> dict[str, dict]:
    """The sets of shortfalls read: on HI-Medium as measured, at the common
    laundering rate, at the threshold of fifty, at each GNN seed and, in the
    tree family, at each scenario's own best configuration, whose null check
    moves the paired replicates onto the own-best values and keeps the paired
    error bars, as its I2 does; and every other dataset as measured."""
    out = {}
    for family in FAMILIES:
        hm = tables[(KEY, family)]
        out[f"{KEY}/{family}/primary"] = shortfall_cell(hm, reps[(KEY, family)])
        out[f"{KEY}/{family}/common_rate"] = cell(
            hm.index, hm["ap_common_delta"], hm["ap_common_se"])
        out[f"{KEY}/{family}/fifty"] = shortfall_cell(
            hm, reps[(KEY, family)], (hm["test_pos"] >= FIFTY).to_numpy())
    for seed in range(reps_by_seed.shape[2]):
        out[f"{KEY}/gnn/seed{seed}"] = seed_cell(tables[(KEY, "gnn")], reps_by_seed, seed)
    tree = tables[(KEY, "tree")]
    shift = (tree["own_best_delta"] - tree["delta"]).to_numpy()
    out[f"{KEY}/tree/own_best"] = cell(tree.index, tree["own_best_delta"], tree["se"],
                                       reps[(KEY, "tree")] + shift[None, :])
    for key in SPREAD_KEYS[1:]:
        for family in FAMILIES:
            out[f"{key}/{family}/primary"] = shortfall_cell(tables[(key, family)],
                                                            reps[(key, family)])
    return out


def null_of(c: dict) -> dict:
    null = null_q_check(c["reps"], c["deltas"], c["ses"], c["seed_var"])
    return {"Q": null["observed_Q"], "null_q95": null["null_q95"],
            "exceedance": null["exceedance"], "passes": null["passes"],
            "Q_over_q95": null["observed_Q"] / null["null_q95"]}


def leave_each_summary(c: dict, out: dict) -> dict:
    rows = pd.DataFrame(out["removals"]).set_index("bank")
    lowest = rows.loc[out["min_bank"]]
    largest_term = rows["q_share"].idxmax()
    summary = {
        "k": len(c["deltas"]), "weighted_mean": out["full"]["weighted_mean"],
        "Q": out["full"]["Q"], "I2": out["full"]["I2"], "tau": out["full"]["tau"],
        "min_I2": out["min_I2"], "min_bank": out["min_bank"],
        "min_bank_weight_share": lowest["weight_share"],
        "min_bank_q_share": lowest["q_share"],
        "without_min_bank": {"Q": lowest["Q"], "tau": lowest["tau"]},
        "largest_q_term": {"bank": largest_term,
                           "weight_share": rows.loc[largest_term, "weight_share"],
                           "q_share": rows.loc[largest_term, "q_share"],
                           "I2_without": rows.loc[largest_term, "I2"]}}
    if "passes" in rows:
        ratio = rows["Q"] / rows["null_q95"]
        summary |= {
            "null_passes": int(rows["passes"].sum()),
            "removals": len(rows),
            "smallest_Q_over_q95": ratio.min(),
            "smallest_Q_over_q95_bank": ratio.idxmin(),
            "failing_removals": sorted(rows.index[~rows["passes"].astype(bool)])}
        summary["without_min_bank"] |= {"null_q95": lowest["null_q95"],
                                        "exceedance": lowest["exceedance"],
                                        "passes": bool(lowest["passes"])}
    return summary


def leave_each_readings(cells: dict) -> tuple[dict, pd.DataFrame]:
    summaries, frames = {}, []
    for name, c in cells.items():
        out = leave_each_out(c["labels"], c["deltas"], c["ses"], c["reps"], c["seed_var"])
        summaries[name] = leave_each_summary(c, out)
        frames.append(pd.DataFrame(out["removals"]).assign(cell=name))
    return summaries, pd.concat(frames).set_index(["cell", "bank"])


def widening(c: dict, summary: dict) -> dict:
    """The largest factor c by which every s_B can grow before the null check
    fails, Q falling to Q / c^2 while the null's 95th percentile holds still:
    on every bank, and the least over the removals of one bank."""
    null = null_of(c)
    return {"c_null": np.sqrt(null["Q_over_q95"]),
            "c_null_leave_each_out": np.sqrt(summary["smallest_Q_over_q95"]),
            "Q": null["Q"], "null_q95": null["null_q95"], "k": len(c["deltas"])}


def null_tau(c: dict) -> dict:
    """tau on each centred null replicate: the spread the world with no real
    spread reads."""
    q_null = null_q_replicates(c["reps"], c["deltas"], c["ses"], c["seed_var"])
    terms = tau_reading(c["deltas"], c["ses"])
    tau = np.sqrt(np.maximum(0.0, (q_null - (len(c["deltas"]) - 1)) / terms["weight_term"]))
    return {"replicates": len(tau), "median": np.median(tau),
            "q95": np.quantile(tau, NULL_LEVEL), "max": tau.max(),
            "observed": terms["tau"], "share_at_or_above_observed": (tau >= terms["tau"]).mean()}


def matched_range(c: dict, train_pos: np.ndarray) -> dict:
    """The banks holding at least MATCHED_TRAIN_POS confirmed cases."""
    keep = train_pos >= MATCHED_TRAIN_POS
    sub = cell(np.asarray(c["labels"])[keep], c["deltas"][keep], c["ses"][keep],
               c["reps"][:, keep], None if c["seed_var"] is None else c["seed_var"][keep])
    het = heterogeneity(sub["deltas"], sub["ses"])
    return {"min_train_pos": MATCHED_TRAIN_POS, "k": int(keep.sum()),
            "banks": sub["labels"], "weighted_mean": het["weighted_mean"],
            "I2": het["I2"], "tau": tau_reading(sub["deltas"], sub["ses"])["tau"],
            **null_of(sub)}


def agreement(tree: pd.DataFrame, gnn: pd.DataFrame, roster: pd.DataFrame,
              reps: dict) -> dict:
    """rho between the families' shortfalls, with size or confirmed cases
    held fixed, and over the paired centred replicates of the two families."""
    t, g = tree["delta"].to_numpy(), gnn["delta"].to_numpy()
    partial = {name: partial_spearman(t, g, roster[name].to_numpy(np.float64))["rho"]
               for name in ("train_pos", "test_rows", "train_rows")}
    centred_t = reps[(KEY, "tree")] - t[None, :]
    centred_g = reps[(KEY, "gnn")] - g[None, :]
    null = np.array([rank_spearman(a, b) for a, b in zip(centred_t, centred_g)])
    return {"rho": rank_spearman(t, g), "partial_rho_given": partial,
            "null_rho": {"replicates": len(null), "mean": null.mean(),
                         "q95": np.quantile(null, NULL_LEVEL), "max": null.max()}}


def size_windows(tables: dict, roster: pd.DataFrame) -> dict:
    """Size counted in the training window beside the test window."""
    ratio = roster["train_rows"] / roster["test_rows"]
    out = {"rho_train_rows_test_rows": rank_spearman(roster["train_rows"].to_numpy(),
                                                     roster["test_rows"].to_numpy()),
           "train_over_test": {"min": ratio.min(), "median": ratio.median(),
                               "max": ratio.max()},
           "worked_pair": {b: {"test_rows": roster.loc[b, "test_rows"],
                               "train_rows": roster.loc[b, "train_rows"],
                               "train_pos": roster.loc[b, "train_pos"]}
                           for b in WORKED_PAIR}}
    for family in FAMILIES:
        b = tables[(KEY, family)]
        out[family] = {}
        for window in ("test_rows", "train_rows"):
            log_size = np.log(roster[window].to_numpy(np.float64))
            line = wls_residual(b["delta"].to_numpy(), b["se"].to_numpy(), log_size)
            out[family][window] = {"rho": rank_spearman(b["delta"].to_numpy(), log_size),
                                   "slope": line["slope"], "I2_after_line": line["I2"],
                                   "tau_after_line": line["tau"]}
    return out


def own_best_reading(banks: pd.DataFrame, metrics: dict) -> dict:
    """The tree family's own-best minus paired shortfall by weighted means
    (w = 1/s_B^2): the global side's gain from choosing its configuration,
    less the solo side's, plus the gap between the median of differences and
    the difference of medians; and the banks that change sign."""
    n = banks.columns.str.fullmatch(r"global_ap_s\d+").sum()
    glob = banks[[f"global_ap_s{s}" for s in range(n)]].to_numpy()
    solo = banks[[f"solo_ap_s{s}" for s in range(n)]].to_numpy()
    per_setting = banks[[f"delta_s{s}" for s in range(n)]].to_numpy()
    selected = metrics["global_level"]["selected"]
    own_setting = banks["solo_own_setting"].to_numpy(int)
    paired, own = banks["delta"].to_numpy(), banks["own_best_delta"].to_numpy()
    s = banks["se"].to_numpy()
    w = 1.0 / s ** 2

    def wmean(x):
        return float((w * x).sum() / w.sum())

    global_gain = glob[:, selected] - np.median(glob, axis=1)
    own_gain = solo[np.arange(len(own_setting)), own_setting] - np.median(solo, axis=1)
    residual = (np.median(glob, axis=1) - np.median(solo, axis=1)) - paired
    flips = np.sign(paired) != np.sign(own)
    covers = (banks["ci_lo"] <= 0).to_numpy() & (banks["ci_hi"] >= 0).to_numpy()
    below = (banks["ci_hi"] < 0).to_numpy()
    return {"weighted_mean": {"paired": wmean(paired), "own_best": wmean(own),
                              "difference": wmean(own - paired)},
            "decomposition": {"global_gain": wmean(global_gain), "own_gain": wmean(own_gain),
                              "median_gap": wmean(residual)},
            "global_selected_configuration": selected + 1,
            "paired_at_selected_configuration": wmean(per_setting[:, selected]),
            "sign_changes": int(flips.sum()),
            "negative_to_positive": int(((paired < 0) & (own > 0)).sum()),
            "positive_to_negative": int(((paired > 0) & (own < 0)).sum()),
            "changes_interval_covers_zero": int((flips & covers).sum()),
            "changes_interval_wholly_below": int((flips & below).sum()),
            "negative_own_best": sorted(own[own < 0].tolist()),
            "I2": heterogeneity(own, s)["I2"], "tau": tau_reading(own, s)["tau"]}


def control_ranges(rosters: dict) -> dict:
    """Confirmed cases and the laundering rate of the test set, as ranges."""
    out = {}
    for key, roster in rosters.items():
        rate = roster["test_pos"] / roster["test_rows"]
        out[key] = {"train_pos_range": [roster["train_pos"].min(), roster["train_pos"].max()],
                    "test_rate_one_in": [1.0 / rate.max(), 1.0 / rate.min()]}
    return out


def spread_inputs() -> dict:
    runs = index(exp1.EXPERIMENT)
    return {"shortfall": {(key, family): find(runs, check="shortfall",
                                              dataset_key=key, family=family)
                          for key in SPREAD_KEYS for family in FAMILIES},
            "own_best": find(runs, check="own_best")}


def spread_config(inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": exp1.EXPERIMENT, "check": "spread_readings",
            "world": world, "dataset": dataset, "dataset_key": KEY,
            "families": list(FAMILIES), "datasets": list(SPREAD_KEYS),
            "leave_each_out": "m.stats.leave_each_out: each bank removed in turn; Q, "
                              "I2 and tau on the rest; the null check on the kept "
                              "replicate columns where replicates are recorded",
            "null_check": "m.stats.null_q_check, centred replicates, 95th "
                          "percentile; GNN seed noise of variance seed_var/3 from "
                          "default_rng(0); a GNN seed on its own replicates, s_B "
                          "their standard deviation, no seed noise",
            "common_rate_null": "the shortfall Run's common_rate.null_check",
            "own_best_null": "paired replicates moved by own_best_delta - delta, "
                             "paired s_B",
            "fifty": f"test_pos >= {FIFTY}",
            "matched_range": f"train_pos >= {MATCHED_TRAIN_POS}",
            "widening": "sqrt(Q / q95), and its least over the removals of one bank",
            "spearman": "m.rank.spearman; partial by m.rank.partial_spearman",
            "inputs": {"shortfall": {f"{k}/{f}": _hash_of(p)
                                     for (k, f), p in inputs["shortfall"].items()},
                       "own_best": _hash_of(inputs["own_best"])}}


def spread_metrics(inputs: dict, run_dir: Path) -> dict:
    shortfall = inputs["shortfall"]
    recorded = {c: json.loads((p / "metrics.json").read_text()) for c, p in shortfall.items()}
    tables = {c: banks_table(p, recorded[c]) for c, p in shortfall.items()}
    roster = banks_table_named(shortfall[(KEY, "tree")], recorded[(KEY, "tree")], "roster")
    for family in FAMILIES:
        tables[(KEY, family)] = tables[(KEY, family)].loc[roster.index]
    reps = {c: np.load(verify(p, recorded[c]["artifacts"]["replicates"])).astype(np.float64)
            for c, p in shortfall.items()}
    reps_by_seed = np.load(verify(shortfall[(KEY, "gnn")],
                                  recorded[(KEY, "gnn")]["artifacts"]["replicates_by_seed"])
                           ).astype(np.float64)

    cells = spread_cells(tables, reps, reps_by_seed)
    loo, long = leave_each_readings(cells)

    rows = {}
    for family in FAMILIES:
        for row in ("primary", "fifty"):
            rows[f"{KEY}/{family}/{row}"] = null_of(cells[f"{KEY}/{family}/{row}"])
        common = recorded[(KEY, family)]["common_rate"]["null_check"]
        rows[f"{KEY}/{family}/common_rate"] = {
            "Q": common["observed_Q"], "null_q95": common["null_q95"],
            "exceedance": common["exceedance"], "passes": common["passes"],
            "Q_over_q95": common["observed_Q"] / common["null_q95"]}
        rows[f"{SMALL}/{family}/primary"] = null_of(cells[f"{SMALL}/{family}/primary"])
    for seed in range(reps_by_seed.shape[2]):
        name = f"{KEY}/gnn/seed{seed}"
        rows[name] = null_of(cells[name]) | {
            "I2": heterogeneity(cells[name]["deltas"], cells[name]["ses"])["I2"]}
    rows[f"{KEY}/tree/own_best"] = null_of(cells[f"{KEY}/tree/own_best"])

    widen = {f"{key}/{family}": widening(cells[f"{key}/{family}/primary"],
                                         loo[f"{key}/{family}/primary"])
             for key in (KEY, SMALL) for family in FAMILIES}
    tau_null = {name: null_tau(cells[name]) for name in
                (f"{k}/{f}/primary" for k in (KEY, "transxion", data.AMLGENTEX_DATASET)
                 for f in FAMILIES)}
    terms = {family: tau_reading(cells[f"{KEY}/{family}/primary"]["deltas"],
                                 cells[f"{KEY}/{family}/primary"]["ses"])
             for family in FAMILIES}
    for family in FAMILIES:
        terms[family]["tau_squared"] = terms[family]["excess"] / terms[family]["weight_term"]
    matched = {family: matched_range(cells[f"{KEY}/{family}/primary"],
                                     roster["train_pos"].to_numpy())
               for family in FAMILIES}
    own = own_best_reading(tables[(KEY, "tree")], recorded[(KEY, "tree")])
    own_best_run = json.loads((inputs["own_best"] / "metrics.json").read_text())["readings"]
    checks = {
        "primary_Q_vs_shortfall_runs": max(
            abs(rows[f"{KEY}/{f}/primary"]["Q"] - recorded[(KEY, f)]["primary"]["Q"])
            for f in FAMILIES),
        "primary_q95_vs_shortfall_runs": max(
            abs(rows[f"{k}/{f}/primary"]["null_q95"]
                - recorded[(k, f)]["primary"]["null_check"]["null_q95"])
            for k in (KEY, SMALL) for f in FAMILIES),
        "per_seed_I2_vs_gnn_run": max(
            abs(rows[f"{KEY}/gnn/seed{i}"]["I2"] - s["I2"])
            for i, s in enumerate(recorded[(KEY, "gnn")]["primary"]["per_seed"])),
        "per_seed_null_passes_agree": all(
            rows[f"{KEY}/gnn/seed{i}"]["passes"] == s["null_passes"]
            for i, s in enumerate(recorded[(KEY, "gnn")]["primary"]["per_seed"])),
        "own_best_I2_vs_own_best_run": abs(own["I2"] - own_best_run["tree"]["I2_own_best"])}

    rosters = {key: banks_table_named(shortfall[(key, "tree")], recorded[(key, "tree")],
                                      "roster")
               for key in (KEY, "transxion", data.AMLGENTEX_DATASET)}
    artifacts = {"leave_each_out": write_csv(run_dir, "leave_each_out", long),
                 "robustness_null": write_csv(run_dir, "robustness_null",
                                              pd.DataFrame(rows).T.rename_axis("row"))}
    return json.loads(json.dumps({
        "dataset_key": KEY, "leave_each_out": loo, "robustness_null": rows,
        "widening": widen, "tau_terms": terms, "tau_null": tau_null,
        "matched_range": matched,
        "agreement": agreement(tables[(KEY, "tree")], tables[(KEY, "gnn")], roster, reps),
        "size_windows": size_windows(tables, roster), "own_best": own,
        "control_ranges": control_ranges(rosters), "checks": checks,
        "artifacts": artifacts}, default=lambda v: v.tolist()))


def banks_table_named(path: Path, metrics: dict, name: str) -> pd.DataFrame:
    return pd.read_csv(verify(path, metrics["artifacts"][name]),
                       dtype={"bank": str}).set_index("bank")


def cmd_spread() -> dict:
    inputs = spread_inputs()
    metrics = run(exp1.EXPERIMENT, spread_config(inputs),
                  lambda config, run_dir: spread_metrics(inputs, run_dir))
    for family in FAMILIES:
        r = metrics["leave_each_out"][f"{KEY}/{family}/primary"]
        print(f"spread {family}: lowest I2 {r['min_I2']:.3f} without {r['min_bank']}; "
              f"null passes at {r['null_passes']} of {r['removals']} removals", flush=True)
    print(f"spread checks: {metrics['checks']}", flush=True)
    return metrics


# ----------------------------------------------------------------- main --

def cmd_all() -> int:
    done = completed(exp1.EXPERIMENT)
    cmd_residual(done)
    cmd_paired(done)
    cmd_spread()
    return 0


def cmd_status() -> int:
    done = completed(exp1.EXPERIMENT)
    for check, config_fn, families in (
            ("residual_after_size", residual_config, FAMILIES),
            ("own_best", paired_config, PAIRED_FAMILIES)):
        try:
            inputs = shortfall_inputs(done, families)
        except SystemExit as missing:
            print(f"{check}: inputs incomplete ({missing})")
            continue
        state = "analysed" if config_hash(config_fn(inputs)) in done else "ready"
        print(f"{check}: {state}")
    try:
        inputs = spread_inputs()
    except SystemExit as missing:
        print(f"spread_readings: inputs incomplete ({missing})")
        return 0
    state = "analysed" if config_hash(spread_config(inputs)) in done else "ready"
    print(f"spread_readings: {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    if sys.argv[1] == "status":
        return cmd_status()
    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    cmd = sys.argv[1]
    if cmd == "residual":
        return 0 if cmd_residual() else 1
    if cmd == "paired":
        return 0 if cmd_paired() else 1
    if cmd == "spread":
        return 0 if cmd_spread() else 1
    if cmd == "all":
        return cmd_all()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
