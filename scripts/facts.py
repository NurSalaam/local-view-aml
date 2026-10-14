#!/usr/bin/env python
"""The thesis's derived numbers as Runs.

Analysis only: no model trains. Every number here is computed from recorded
artefacts; one ``facts`` Run per block holds it, keyed by short names grouped
by the table, figure or paragraph that quotes it. There is one block per
experiment the results draw on (Experiments 1, 2 and 4), plus a ``thesis``
block for the numbers other chapters quote from the censuses and the cached
data. Per-bank tables, bootstrap replicates and verdict blocks are read from
their Runs. The qualifying-pool shortfall is re-read from the cells' recorded
test scores through Experiment 2's own seeded four-cell draws, since the
components Runs record each component's median over configurations and a
median of sums is not a sum of medians. Every config cites its inputs' Run
hashes, so a re-done input re-keys the Run; skip-if-done is by config hash.
All blocks write under experiments/facts.

  uv run python scripts/facts.py exp1
  uv run python scripts/facts.py exp2
  uv run python scripts/facts.py exp4
  uv run python scripts/facts.py thesis
  uv run python scripts/facts.py all
  uv run python scripts/facts.py status
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402, F401  (before lightgbm: OpenMP runtimes, see exp1.py)
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1  # noqa: E402
import exp1_analysis  # noqa: E402
import exp1_data as data  # noqa: E402
import exp2  # noqa: E402
import exp2_analysis  # noqa: E402
import exp2_examples  # noqa: E402
from exp1_readings import (FAMILIES, FIFTY, PAIRED_FAMILIES, size_gradient,  # noqa: E402
                           tau_dl, wls_residual)
from m import completed, run  # noqa: E402
from m.artifacts import load_array, verify, write_array, write_csv  # noqa: E402
from m.features import graph_feature_names  # noqa: E402
from m.rank import spearman  # noqa: E402
from m.runlog import config_hash, find, index  # noqa: E402
from m.stats import heterogeneity, weighted_average_precision  # noqa: E402
from m.worlds import amlgentex, amlworld  # noqa: E402

EXPERIMENT = "facts"
CHECK = "facts"
KEY = "HI-Medium"
DETECTORS = FAMILIES + ("blind",)
SEED = 0
DIAL_LEVELS = (5, 10, 30, 100)
LEVEL_100_GROUPS = {"103_to_122": (103, 122), "132_to_265": (132, 265)}
NAMED_BANKS = ("0143079", "020")
VARIANTS = {"tree": "tree", "gnn": "gnn", "blind": "blind",
            "gnn_copy_weighted": "gnn"}

TAU = ("DerSimonian-Laird moment estimator, tau^2 = max(0, (Q - (k - 1)) / "
       "(sum w - sum w^2 / sum w)) with w = 1/s_B^2 (scripts/exp1_readings.py "
       "tau_dl); after the line on log size, its generalisation to weighted "
       "least squares (wls_residual)")
SPEARMAN = "Spearman rho over banks, average ranks; a log changes nothing"
THREE_TESTS = ("I2 >= 0.25, the assumption-free null check (centred joint "
               "replicates, plus normal seed noise of variance seed_var/3 for "
               "the GNN family, default_rng(0), 95th percentile) and "
               "leave-one-out I2 >= 0.25, by scripts/exp1_analysis.py "
               "heterogeneity_reading")
ROSTER = ("per dataset, over its qualifying banks: the payments in a bank's "
          "ledger over all three windows (touching) and its confirmed cases "
          "(train_pos), each as a range; HI-Medium, HI-Small and the two "
          "controls from the roster artefact of the tree family's shortfall "
          "Run, LI-Small and LI-Medium, which take no shortfall Run, from "
          "Experiment 1's cached roster")
QUALIFYING = ("M(network, qualifying) - M(ledger, own), C-global minus solo: "
              "Experiment 2's four-cell joint bootstrap (seed 0, 1,000 replicates, "
              "chunks of 100) re-read from the recorded test scores of the "
              "cells the components (or sightcheck) Run cites, its recorded "
              "components reproduced alongside; the tree and blind families "
              "take the median over the eight configurations of the "
              "per-configuration differences, the GNN family the mean over "
              "seeds; the weighted mean uses Experiment 1's weights 1/s_B^2 with Experiment 2's "
              "interval (GNN seed noise from a fresh default_rng(0)); the "
              "three tests and tau use the difference's own s_B, as Experiment 2's "
              "descriptive heterogeneity does")
QUARTILES = ("the quarters of the qualifying banks by confirmed cases, the "
             "quartile column of Experiment 2's examples Run: pandas qcut over "
             "the training-window laundering count, four bins; unweighted mean")
CHANCE = ("the datasets whose banks can differ only by chance: the two "
          "controls and the eleven other AMLgentex datasets, tree and GNN "
          "families, each read by its shortfall Run")
SPLIT = ("each bank's test set split by Experiment 2's never-seen mask (scripts/exp2_examples.py "
         "counterparty_view); on each part the four cells' point APs per "
         "setting (seed), the components as per-setting differences paired "
         "as in Experiment 2 (tree and blind: median over the eight "
         "configurations; GNN: mean over seeds); a part holding no "
         "laundering payment has no AP and the bank leaves that part")
DROP_VS_SIGHT = ("at each bank of the case reduction, the fall from the full "
                 "count to five confirmed cases beside the bank's value of "
                 "sight on either path (second-path Run banks.csv): its point "
                 "value and the upper end of its interval")
LEVEL_100 = ("at the banks holding a level of 100: the fall is the "
             "full-count score minus the mean level-100 score; the headroom "
             "is the full-count score minus the band floor; the GNN's "
             "full-count score is its seed-0 solo Run, the case reduction's "
             "seed")
QUARTILE_COMPONENTS = ("the path-1 value of training examples and value of "
                       "sight, and their difference, in each quarter of "
                       "QUARTILES: Experiment 1-weighted means of the components Run's "
                       "per-bank deltas, intervals from its recorded "
                       "replicates (GNN: seed noise of variance seed_var/3 "
                       "from the per-seed columns, fresh default_rng(0) per "
                       "quantity)")
RECOVERY = ("over the quarter of the qualifying banks holding the fewest "
            "confirmed cases, the weighted mean gain as a share of the "
            "weighted mean of two yardsticks: the Experiment 1 shortfall "
            "(global minus solo) and the value of training examples with "
            "sight held at the ledger (C-local minus solo); weights 1/s_B^2 "
            "from Experiment 1, as the policy Run's weight_se_exp1 column "
            "holds them")


# --------------------------------------------------------------- inputs --

def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def _metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


def _config(path: Path) -> dict:
    return json.loads((path / "config.json").read_text())


def frame(path: Path, name: str = "banks") -> pd.DataFrame:
    return pd.read_csv(verify(path, _metrics(path)["artifacts"][name]),
                       dtype={"bank": str}).set_index("bank")


def array(path: Path, name: str) -> np.ndarray:
    return np.load(verify(path, _metrics(path)["artifacts"][name])).astype(np.float64)


def scores(path: Path, name: str) -> np.ndarray:
    return load_array(path, name, _metrics(path)["artifacts"][name]["sha256"])


def settings(banks: pd.DataFrame, prefix: str) -> int:
    """How many per-setting (per-seed) columns ``<prefix>_s<i>`` a table holds."""
    return banks.columns.str.fullmatch(rf"{prefix}_s\d+").sum()


def plain(value):
    """numpy scalars and arrays as JSON's own types."""
    return json.loads(json.dumps(value, default=lambda v: v.tolist()))


# ------------------------------------------------------------ summaries --

def span(values) -> list:
    return [np.min(values), np.max(values)]


def extremes(series: pd.Series) -> dict:
    return {"lowest": {"bank": series.idxmin(), "value": series.min()},
            "highest": {"bank": series.idxmax(), "value": series.max()}}


def weight_shares(ses: np.ndarray) -> dict:
    w = np.sort(1.0 / ses ** 2)[::-1]
    return {"top1": w[0] / w.sum(), "top3": w[:3].sum() / w.sum()}


def dl_terms(deltas: np.ndarray, ses: np.ndarray) -> dict:
    """The two terms of the DerSimonian-Laird ratio, for the worked example."""
    w = 1.0 / ses ** 2
    return {"excess": heterogeneity(deltas, ses)["Q"] - (len(deltas) - 1),
            "weight_term": w.sum() - (w ** 2).sum() / w.sum(),
            "tau": tau_dl(deltas, ses)}


def three_tests(reading: dict, deltas: np.ndarray, ses: np.ndarray) -> dict:
    null, loo = reading["null_check"], reading["leave_one_out"]
    return {"k": reading["k"], "weighted_mean": reading["weighted_mean"],
            "Q": reading["Q"], "I2": reading["I2"],
            "null_q95": null["null_q95"],
            "null_exceed": round(null["exceedance"] * null["null_replicates"]),
            "null_passes": null["passes"],
            "loo_I2": loo["I2"], "loo_dropped": loo["dropped"],
            "loo_weight_share": loo["dropped_weight_share"],
            "passes_three_tests": bool(reading["clears_low_band"]
                                       and null["passes"]
                                       and reading["loo_clears_low_band"]),
            "tau": tau_dl(deltas, ses)}


def interval_counts(deltas: pd.Series, lo: pd.Series, hi: pd.Series) -> dict:
    return {"positive": (deltas > 0).sum(), "negative": (deltas < 0).sum(),
            "wholly_above": (lo > 0).sum(), "wholly_below": (hi < 0).sum(),
            "median": deltas.median(), "range": span(deltas)}


def max_diff(pairs) -> float:
    return max(float(np.max(np.abs(np.asarray(a, np.float64)
                                   - np.asarray(b, np.float64))))
               for a, b in pairs)


# -------------------------------------------------------------- exp1 --

def exp1_inputs() -> dict:
    runs = index("exp1")
    return {"shortfall": {(key, family): find(runs, check="shortfall",
                                              dataset_key=key, family=family)
                          for key in data.PRIMARY + data.SENSITIVITY
                          for family in exp1.shortfall_families(key)},
            "residual": find(runs, check="residual_after_size"),
            "fifty": find(runs, check="roster_fifty"),
            "paired": find(runs, check="paired_vs_naive"),
            "verdict": find(runs, check="verdict")}


def exp1_config(inputs: dict) -> dict:
    return {"experiment": "exp1", "check": CHECK, "dataset_key": KEY,
            "tau": TAU, "spearman": SPEARMAN,
            "three_tests": THREE_TESTS, "fifty": FIFTY,
            "chance": CHANCE, "roster": ROSTER,
            "inputs": {
                "shortfall": {f"{k}/{f}": _hash_of(p)
                              for (k, f), p in inputs["shortfall"].items()},
                "residual_after_size": _hash_of(inputs["residual"]),
                "roster_fifty": _hash_of(inputs["fifty"]),
                "paired_vs_naive": _hash_of(inputs["paired"]),
                "verdict": _hash_of(inputs["verdict"]),
                "ledgers": {k: f"data/interim/exp1/ledgers/{k}"
                            for k in data.ANCHOR_ONLY}}}


def ledger_facts(roster: pd.DataFrame) -> dict:
    """Table meth-census, the last two columns: the payments in a qualifying
    bank's ledger and its confirmed cases."""
    return {"banks": len(roster),
            "ledger_payments_range": span(roster["touching"]),
            "train_pos_range": span(roster["train_pos"])}


def roster_facts(banks: pd.DataFrame) -> dict:
    rate = banks["test_base_rate"]
    return {"test_pos_range": span(banks["test_pos"]),
            "test_pos_median": banks["test_pos"].median(),
            "test_rows_range": span(banks["test_rows"]),
            "test_rows_median": banks["test_rows"].median(),
            "test_rows_fold": banks["test_rows"].max() / banks["test_rows"].min(),
            "rate_range": span(rate), "rate_fold": rate.max() / rate.min(),
            "rate_median": rate.median(),
            "pooled_rate": banks["test_pos"].sum() / banks["test_rows"].sum()}


def spread_facts(banks: pd.DataFrame, metrics: dict) -> dict:
    """Table sf-spread and Figure sf-shortfalls, one family."""
    d, s = banks["delta"], banks["se"]
    lo, hi = banks["ci_lo"], banks["ci_hi"]
    primary = metrics["primary"]
    null = primary["null_check"]
    return {"median": d.median(), "sd": d.std(ddof=1),
            "below_zero": (d < 0).sum(), "above_half": (d > 0.5).sum(),
            "wholly_above": (lo > 0).sum(), "wholly_below": (hi < 0).sum(),
            "covers_zero": ((lo <= 0) & (hi >= 0)).sum(),
            "covers_zero_positive": ((lo <= 0) & (hi >= 0) & (d > 0)).sum(),
            "covers_zero_negative": ((lo <= 0) & (hi >= 0) & (d < 0)).sum(),
            "interval_holds_point": ((lo <= d) & (d <= hi)).sum(),
            "interval_width_median": (hi - lo).median(),
            "interval_span": [lo.min(), hi.max()],
            "se_median": s.median(), "se_range": span(s),
            "shortfall": extremes(d),
            "weight_share": weight_shares(s.to_numpy()),
            "q_over_df": primary["Q"] / (primary["k"] - 1),
            "q_over_null_q95": primary["Q"] / null["null_q95"],
            "null_exceed": round(null["exceedance"] * null["null_replicates"]),
            "loo_I2_change": primary["I2"] - primary["leave_one_out"]["I2"],
            "dl": dl_terms(d.to_numpy(), s.to_numpy()),
            "solo_ap": {"median": banks["solo_ap"].median(),
                        "range": span(banks["solo_ap"])},
            "global_ap": {"median": banks["global_ap"].median(),
                          "range": span(banks["global_ap"])},
            "median_over_median_global": d.median() / banks["global_ap"].median(),
            "setting_spread_median": banks["setting_spread"].median()}


def per_seed_facts(banks: pd.DataFrame, reps_by_seed: np.ndarray) -> list:
    """The GNN family's three tests at each seed, as Experiment 1 reads them."""
    labels = list(banks.index)
    out = []
    for s in range(reps_by_seed.shape[2]):
        d = banks[f"delta_s{s}"].to_numpy()
        reps = reps_by_seed[:, :, s]
        se = np.nanstd(reps, axis=0, ddof=1)
        reading = exp1_analysis.heterogeneity_reading(labels, d, se, reps)
        out.append({"seed": exp1.SEEDS[s], "below_zero": (d < 0).sum(),
                    **three_tests(reading, d, se)})
    return out


def fifty_facts(banks: pd.DataFrame, reps: np.ndarray) -> dict:
    keep = (banks["test_pos"] >= FIFTY).to_numpy()
    d, s = banks["delta"].to_numpy()[keep], banks["se"].to_numpy()[keep]
    seed_var = banks["seed_var"].to_numpy()[keep] if "seed_var" in banks else None
    reading = exp1_analysis.heterogeneity_reading(list(banks.index[keep]), d, s,
                                                reps[:, keep], seed_var)
    return {"smallest_test_pos": banks["test_pos"][keep].min(),
            **three_tests(reading, d, s)}


def selection_facts(banks: pd.DataFrame, metrics: dict) -> dict:
    """Each scenario at its own validation-selected configuration."""
    n = settings(banks, "global_ap")
    glob = banks[[f"global_ap_s{s}" for s in range(n)]].to_numpy()
    solo = banks[[f"solo_ap_s{s}" for s in range(n)]].to_numpy()
    selected = metrics["global_level"]["selected"]
    own = banks["solo_own_setting"].to_numpy(int)
    global_gain = glob[:, selected] - np.median(glob, axis=1)
    own_gain = solo[np.arange(len(own)), own] - np.median(solo, axis=1)
    return {"global_selected": selected,
            "global_gain_median": np.median(global_gain),
            "global_gain_positive": (global_gain > 0).sum(),
            "global_gain_negative": (global_gain < 0).sum(),
            "own_gain_median": np.median(own_gain),
            "own_gain_positive": (own_gain > 0).sum(),
            "own_gain_zero": (own_gain == 0).sum(),
            "tau_own_best": tau_dl(banks["own_best_delta"].to_numpy(),
                                   banks["se"].to_numpy())}


def size_facts(banks: pd.DataFrame) -> dict:
    rows = banks["test_rows"].to_numpy()
    return {name: size_gradient(banks[column].to_numpy(), rows)
            for name, column in (("shortfall", "delta"),
                                 ("common_rate", "ap_common_delta"),
                                 ("solo_ap", "solo_ap"),
                                 ("global_ap", "global_ap"))}


def chance_facts(runs: dict, tables: dict) -> dict:
    """The chance-only benchmark: the three tests and tau at every dataset
    whose banks can differ only by chance, tree and GNN families."""
    cells = {f"{key}/{f}": three_tests(runs[(key, f)]["primary"],
                                       tables[(key, f)]["delta"].to_numpy(),
                                       tables[(key, f)]["se"].to_numpy())
             for key in data.CONTROLS + data.SENSITIVITY for f in FAMILIES}
    widest = max(cells, key=lambda c: cells[c]["tau"])
    return {"cells": cells, "combinations": len(cells),
            "passing": sorted(c for c, r in cells.items() if r["passes_three_tests"]),
            "passing_count": sum(r["passes_three_tests"] for r in cells.values()),
            "tau_max": {"cell": widest, "value": cells[widest]["tau"]}}


def exp1_metrics(inputs: dict, run_dir: Path) -> dict:
    t0 = time.time()
    runs = {cell: _metrics(p) for cell, p in inputs["shortfall"].items()}
    tables = {cell: frame(p) for cell, p in inputs["shortfall"].items()}
    hm = {f: tables[(KEY, f)] for f in exp1.shortfall_families(KEY)}
    roster = frame(inputs["shortfall"][(KEY, "tree")], "roster")
    reps = {f: array(inputs["shortfall"][(KEY, f)], "replicates") for f in FAMILIES}
    residual = _metrics(inputs["residual"])
    fifty_run = _metrics(inputs["fifty"])
    paired = _metrics(inputs["paired"])
    verdict = _metrics(inputs["verdict"])

    rosters = {key: ledger_facts(frame(inputs["shortfall"][(key, "tree")], "roster"))
               | roster_facts(tables[(key, "tree")])
               for key in data.PRIMARY}
    rosters |= {key: ledger_facts(data.read_roster(key)) for key in data.ANCHOR_ONLY}
    rosters[KEY] |= {"val_pos_range": span(roster["val_pos"]),
                     "rho_train_rows_test_rows": spearman(
                         roster["train_rows"].to_numpy(), roster["test_rows"].to_numpy()),
                     "rho_train_pos_test_rows": spearman(
                         roster["train_pos"].to_numpy(), roster["test_rows"].to_numpy())}
    rosters["HI-Small"]["shared_with_hi_medium"] = sorted(
        set(tables[("HI-Small", "tree")].index) & set(hm["tree"].index))

    spread = {f: spread_facts(hm[f], runs[(KEY, f)]) for f in FAMILIES}
    gnn = hm["gnn"]
    spread["gnn"]["seed_noise_share_median"] = (
        (gnn["seed_var"] / len(exp1.SEEDS)) / gnn["se"] ** 2).median()
    per_seed = per_seed_facts(gnn, array(inputs["shortfall"][(KEY, "gnn")],
                                         "replicates_by_seed"))
    fifty = {f: fifty_facts(hm[f], reps[f]) for f in FAMILIES}
    selection = {f: selection_facts(hm[f], runs[(KEY, f)]) for f in PAIRED_FAMILIES}

    def tau_of(key, prefix=""):
        return {f: tau_dl(tables[(key, f)][f"{prefix}delta"].to_numpy(),
                          tables[(key, f)][f"{prefix}se"].to_numpy())
                for f in FAMILIES}

    after_line = {f: wls_residual(hm[f]["delta"].to_numpy(), hm[f]["se"].to_numpy(),
                                  np.log(hm[f]["test_rows"].to_numpy(np.float64)))
                  for f in FAMILIES}
    controls_tau = {key: tau_of(key) for key in data.CONTROLS}
    tau = {"as_measured": tau_of(KEY), "HI-Small": tau_of("HI-Small"),
           **controls_tau,
           "controls_max": max(t for c in controls_tau.values() for t in c.values()),
           "common_rate": tau_of(KEY, "ap_common_"), "roc_auc": tau_of(KEY, "auc_"),
           "fifty": {f: fifty[f]["tau"] for f in FAMILIES},
           "after_size_line": {f: after_line[f]["tau"] for f in FAMILIES},
           "own_best": {f: selection[f]["tau_own_best"] for f in selection},
           "eleven": {key: tau_of(key) for key in data.SENSITIVITY}}

    size = {f: size_facts(hm[f]) for f in FAMILIES}
    size["size_vs_rate"] = spearman(np.log(hm["tree"]["test_rows"].to_numpy(np.float64)),
                                    hm["tree"]["test_base_rate"].to_numpy())
    size["HI-Small"] = {f: size_gradient(tables[("HI-Small", f)]["delta"].to_numpy(),
                                         tables[("HI-Small", f)]["test_rows"].to_numpy())
                        for f in FAMILIES}

    chance = chance_facts(runs, tables)
    verdict_fires = ({f"{k}/{f}": r["fires_triple"]
                      for k, fams in verdict["sensitivity_eleven"].items()
                      for f, r in fams.items()}
                     | {f"{k}/{f}": not verdict["criteria"]["3"][k][f]["silent"]
                        for k in data.CONTROLS for f in FAMILIES})

    checks = {
        "weighted_mean_vs_shortfall_runs": max_diff(
            (heterogeneity(tables[c]["delta"].to_numpy(),
                           tables[c]["se"].to_numpy())["weighted_mean"],
             runs[c]["primary"]["weighted_mean"]) for c in runs),
        "tau_as_measured_vs_residual_run": max_diff(
            (tau["as_measured"][f], residual["readings"][f]["raw"]["tau"])
            for f in FAMILIES),
        "tau_after_size_line_vs_residual_run": max_diff(
            (tau["after_size_line"][f], residual["readings"][f]["residual"]["tau"])
            for f in FAMILIES),
        "tau_fifty_vs_fifty_run": max_diff(
            (fifty[f]["tau"], fifty_run["readings"][f]["at_fifty"]["tau"])
            for f in FAMILIES),
        "null_q95_fifty_vs_fifty_run": max_diff(
            (fifty[f]["null_q95"], fifty_run["readings"][f]["at_fifty"]["null_q95"])
            for f in FAMILIES),
        "I2_fifty_vs_fifty_run": max_diff(
            (fifty[f]["I2"], fifty_run["readings"][f]["at_fifty"]["I2"])
            for f in FAMILIES),
        "per_seed_I2_vs_gnn_run": max_diff(
            (p["I2"], r["I2"]) for p, r in
            zip(per_seed, runs[(KEY, "gnn")]["primary"]["per_seed"])),
        "per_seed_Q_vs_gnn_run": max_diff(
            (p["Q"], r["Q"]) for p, r in
            zip(per_seed, runs[(KEY, "gnn")]["primary"]["per_seed"])),
        "I2_own_best_vs_paired_run": max_diff(
            (heterogeneity(hm[f]["own_best_delta"].to_numpy(),
                           hm[f]["se"].to_numpy())["I2"],
             paired["readings"][f]["I2_own_best"]) for f in selection),
        "seed_noise_share_vs_verdict_run": abs(
            spread["gnn"]["seed_noise_share_median"]
            - verdict["robustness"][f"{KEY}/gnn"]["seed_noise_share_median"]),
        "chance_passes_disagreeing_with_verdict_run": sum(
            chance["cells"][c]["passes_three_tests"] != fires
            for c, fires in verdict_fires.items()),
        "chance_cells_read_by_verdict_run": len(verdict_fires)}

    return plain({
        "dataset_key": KEY, "roster": rosters, "spread": spread,
        "per_seed": per_seed, "tau": tau,
        "readings": {"fifty": fifty, "own_best": selection},
        "size": size, "chance": chance, "checks": checks,
        "seconds": time.time() - t0})


# -------------------------------------------------------------- exp2 --

def cell_dirs(recorded: Path, runs1: dict, runs2: dict) -> dict:
    """The four cells a components Run read, from the hashes it cites, in
    its own bank order (the order its seed-noise draws followed)."""
    inputs = _config(recorded)["inputs"]
    roster = frame(recorded, "roster")
    return {"solo": {b: [runs1[h][0] for h in inputs["solo"][b]] for b in roster.index},
            "global": [runs1[h][0] for h in inputs["global"]],
            "clocal": [runs2[h][0] for h in inputs["clocal"]],
            "cglobal": [runs2[h][0] for h in inputs["cglobal"]],
            "roster": roster}


def exp2_inputs() -> dict:
    runs1, runs2 = index("exp1"), index("exp2")
    components = {f: find(runs2, check="components", family=f) for f in DETECTORS}
    sightcheck = find(runs2, check="sightcheck")
    recorded = components | {"gnn_copy_weighted": sightcheck}
    tree = _config(components["tree"])["inputs"]
    return {"recorded": recorded,
            "cells": {v: cell_dirs(recorded[v], runs1, runs2) for v in VARIANTS},
            "dial": {f: find(runs2, check="dial", family=f) for f in FAMILIES},
            "secondpath": {f: find(runs2, check="secondpath", family=f)
                           for f in FAMILIES},
            "shortfall": {f: runs1[_config(components[f])["weights_run"]][0]
                          for f in DETECTORS},
            "network_all_tree": runs1[tree["global"][0]][0],
            "ledger_qualifying_tree": runs2[tree["clocal"][0]][0],
            "census": find(index("checks"), check="amlworld_census", dataset=KEY),
            "examples": find(runs2, check="examples", family="tree")}


def exp2_config(inputs: dict) -> dict:
    return {"experiment": "exp2", "check": CHECK, "dataset_key": KEY,
            "qualifying": QUALIFYING, "quartiles": QUARTILES,
            "three_tests": THREE_TESTS, "tau": TAU, "spearman": SPEARMAN,
            "replicates": exp2_analysis.REPLICATES, "seed": SEED,
            "dial_levels": list(DIAL_LEVELS),
            "level_100_groups": {k: list(v) for k, v in LEVEL_100_GROUPS.items()},
            "dial_counts": ("within = band floor <= level mean <= band ceiling, "
                            "below and above outside it; the band is the range "
                            "of the bank's score at the full count (tree: eight "
                            "configurations; GNN: three seeds)"),
            "level_100": LEVEL_100, "drop_vs_sight": DROP_VS_SIGHT, "split": SPLIT,
            "never_seen": exp2_examples.NEVER_SEEN,
            "quartile_components": QUARTILE_COMPONENTS,
            "inputs": {
                "components": {f: _hash_of(inputs["recorded"][f]) for f in DETECTORS},
                "sightcheck": _hash_of(inputs["recorded"]["gnn_copy_weighted"]),
                "dial": {f: _hash_of(p) for f, p in inputs["dial"].items()},
                "secondpath": {f: _hash_of(p) for f, p in inputs["secondpath"].items()},
                "ledgers": f"data/interim/exp1/ledgers/{KEY}",
                "shortfall": {f: _hash_of(p) for f, p in inputs["shortfall"].items()},
                "network_all_tree": _hash_of(inputs["network_all_tree"]),
                "ledger_qualifying_tree": _hash_of(inputs["ledger_qualifying_tree"]),
                "amlworld_census": _hash_of(inputs["census"]),
                "examples": _hash_of(inputs["examples"])}}


def qualifying_view(boot: dict) -> dict:
    """The four-cell draws with (network, qualifying) standing in for
    (network, all), so that Experiment 2's ``total`` component reads C-global minus
    solo under Experiment 2's own pairing, standard errors and intervals."""
    def swap(cells: dict) -> dict:
        return {g: sides | {"global": sides["cglobal"]} for g, sides in cells.items()}
    return boot | {"point": swap(boot["point"]), "reps": swap(boot["reps"])}


def component_arrays(est: dict, labels: list, name: str) -> dict:
    return {"delta": np.array([est[g][name]["delta"] for g in labels]),
            "se": np.array([est[g][name]["se"] for g in labels]),
            "ci_lo": np.array([est[g][name]["ci95"][0] for g in labels]),
            "ci_hi": np.array([est[g][name]["ci95"][1] for g in labels]),
            "reps": np.stack([est[g][name]["delta_reps"] for g in labels], axis=1),
            "seed_var": (np.array([est[g][name]["seed_var"] for g in labels])
                         if "seed_var" in est[labels[0]][name] else None)}


def qualifying_reading(family: str, cell: dict, recorded: Path,
                       train_pos: pd.Series, quartile: pd.Series) -> dict:
    """The shortfall against the qualifying pool, beside the recorded
    components re-read from the same draws: their agreement with the Run
    shows the draws are the recorded ones."""
    banks, metrics = frame(recorded), _metrics(recorded)
    boot = exp2_analysis.four_cell_bootstrap(cell["n_test"], cell["groups"])
    labels = boot["labels"]
    assert labels == list(banks.index)
    weights = 1.0 / banks["weight_se_exp1"].to_numpy() ** 2

    components = exp2_analysis.component_estimates(boot, family)
    rng = np.random.default_rng(SEED)
    readings = {c: exp2_analysis.weighted_mean_reading(labels, components, weights,
                                                     c, family, rng)
                for c in exp2_analysis.COMPONENT_ORDER}
    recorded_reps = array(recorded, "replicates")
    arrays = {c: component_arrays(components, labels, c)
              for c in exp2_analysis.COMPONENT_ORDER}
    checks = {
        "per_bank_delta": max_diff((arrays[c]["delta"], banks[f"{c}_delta"])
                                   for c in arrays),
        "replicates": max_diff((arrays[c]["reps"], recorded_reps[:, :, i])
                               for i, c in enumerate(exp2_analysis.COMPONENT_ORDER)),
        "weighted_mean": max_diff((readings[c]["weighted_mean"],
                                   metrics["readings"][c]["weighted_mean"])
                                  for c in readings),
        "ci95": max_diff((readings[c]["ci95"], metrics["readings"][c]["ci95"])
                         for c in readings)}

    est = exp2_analysis.component_estimates(qualifying_view(boot), family)
    reading = exp2_analysis.weighted_mean_reading(labels, est, weights, "total",
                                                family, np.random.default_rng(SEED))
    q = component_arrays(est, labels, "total")
    spread = exp1_analysis.heterogeneity_reading(labels, q["delta"], q["se"],
                                               q["reps"], q["seed_var"])
    d = pd.Series(q["delta"], index=labels)
    tp = train_pos.loc[labels]
    quartiles = [{"quartile": int(k),
                  "count_range": span(tp[group.index]),
                  "banks": len(group), "mean": group.mean(),
                  "below_zero": (group < 0).sum()}
                 for k, group in d.groupby(quartile.loc[labels])]
    summary = {"weighted_mean": reading["weighted_mean"], "ci95": reading["ci95"],
               "se": reading["se"],
               "sum_of_recorded_components": (metrics["readings"]["examples"]["weighted_mean"]
                                              + metrics["readings"]["sight"]["weighted_mean"]),
               **interval_counts(d, pd.Series(q["ci_lo"], index=labels),
                                 pd.Series(q["ci_hi"], index=labels)),
               "rho_size": spearman(q["delta"], banks["test_rows"].to_numpy()),
               "rho_train_pos": spearman(q["delta"], tp.to_numpy()),
               "spread": three_tests(spread, q["delta"], q["se"]),
               "quartiles": quartiles, "checks": checks}
    table = pd.DataFrame({"delta": q["delta"], "se": q["se"], "ci_lo": q["ci_lo"],
                          "ci_hi": q["ci_hi"]}, index=pd.Index(labels, name="bank"))
    return {"summary": summary, "table": table, "reps": q["reps"]}


def split_components(family: str, cell: dict, solo: dict) -> pd.DataFrame:
    """Per bank, the four components on the whole test set and on each of
    its two parts, from the cells' point scores."""
    reduce = np.median if exp2_analysis.PAIRING[family] == "median_over_grid" else np.mean
    columns = [f"{part}_{c}" for part in ("all", "never_seen", "rest")
               for c in exp2_analysis.COMPONENT_ORDER]
    rows = {}
    for bank, g in cell["groups"].items():
        view = exp2_examples.counterparty_view(bank)
        assert np.array_equal(view["payment"][view["test"]],
                              exp2_analysis._scores(solo[bank][0], "test_payment"))
        never = view["never"][view["test"]]
        rec = {}
        for part, mask in (("all", np.ones_like(never)), ("never_seen", never),
                           ("rest", ~never)):
            y = g["y"][mask]
            if y.sum() == 0:
                continue
            ones = np.ones((1, len(y)))
            ap = {side: np.array([weighted_average_precision(y, s[mask], ones)[0]
                                  for s in g[side]])
                  for side in exp2_analysis.SIDES}
            for c, (hi, lo) in exp2_analysis.COMPONENTS.items():
                rec[f"{part}_{c}"] = reduce(ap[hi] - ap[lo])
        rows[bank] = rec
    table = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=columns)
    table.index.name = "bank"
    return table


def split_facts(table: pd.DataFrame, banks: pd.DataFrame) -> dict:
    """The components on each of the test set's two parts, one family; the
    whole-set column reproduces the components Run's per-bank table."""
    out = {part: {"banks": table[f"{part}_total"].notna().sum(),
                  **{c: {"median": table[f"{part}_{c}"].median(),
                         "above_zero": (table[f"{part}_{c}"] > 0).sum()}
                     for c in exp2_analysis.COMPONENT_ORDER}}
           for part in ("never_seen", "rest")}
    out["all_vs_components_run"] = max_diff(
        (table.loc[banks.index, f"all_{c}"], banks[f"{c}_delta"])
        for c in exp2_analysis.COMPONENT_ORDER)
    return out


def secondpath_facts(banks: pd.DataFrame, metrics: dict) -> dict:
    """The value of sight at the own pool, per bank, from the second-path
    Run's table; its recorded per-bank block beside it."""
    counts = interval_counts(banks["sight_own_delta"], banks["sight_own_ci_lo"],
                             banks["sight_own_ci_hi"])
    recorded = metrics["readings"]["sight_own"]["per_bank"]
    return {"sight_own": counts,
            "check_vs_run": max_diff([
                (counts["wholly_above"], recorded["interval_above_zero"]),
                (counts["wholly_below"], recorded["interval_below_zero"]),
                (counts["positive"], recorded["above_zero"]),
                (counts["median"], recorded["median"])])}


def component_facts(banks: pd.DataFrame, metrics: dict, reps: np.ndarray,
                    roster: pd.DataFrame, exp1_mean: float) -> dict:
    """Table sf-components and Figure sf-components, one family."""
    readings = metrics["readings"]
    w = 1.0 / banks["weight_se_exp1"] ** 2
    out = {}
    for i, c in enumerate(exp2_analysis.COMPONENT_ORDER):
        d = banks[f"{c}_delta"]
        wm = (reps[:, :, i] * w.to_numpy()).sum(axis=1) / w.sum()
        out[c] = {**interval_counts(d, banks[f"{c}_ci_lo"], banks[f"{c}_ci_hi"]),
                  "zero": (d == 0).sum(),
                  "replicate_mean": wm.mean(),
                  "replicate_interval": np.percentile(wm, [2.5, 97.5])}
    steps = sum(readings[c]["weighted_mean"] for c in ("examples", "sight", "stub"))
    gap = (banks["total_delta"] - banks["examples_delta"] - banks["sight_delta"]
           - banks["stub_delta"]).abs()
    train_pos = roster.loc[banks.index, "train_pos"].to_numpy()
    out |= {"steps_sum": steps,
            "steps_gap": readings["total"]["weighted_mean"] - steps,
            "per_bank_gap_max": gap.max(), "per_bank_gap_median": gap.median(),
            "sight_exceeds_examples": (banks["sight_delta"] > banks["examples_delta"]).sum(),
            "top_weight": {"bank": w.idxmax(), "share": w.max() / w.sum()},
            "total_minus_exp1": readings["total"]["weighted_mean"] - exp1_mean,
            "rho": {"examples_train_pos": spearman(banks["examples_delta"].to_numpy(), train_pos),
                    "total_examples": spearman(banks["total_delta"].to_numpy(),
                                               banks["examples_delta"].to_numpy()),
                    "sight_train_pos": spearman(banks["sight_delta"].to_numpy(), train_pos),
                    "sight_total": spearman(banks["sight_delta"].to_numpy(),
                                            banks["total_delta"].to_numpy()),
                    "stub_train_pos": spearman(banks["stub_delta"].to_numpy(), train_pos)}}
    return out


def quartile_components(banks: pd.DataFrame, reps: np.ndarray, family: str,
                        quartile: pd.Series, train_pos: pd.Series) -> list:
    """The value of training examples and the value of sight (path 1) and
    their difference, by quartile of confirmed cases, one family."""
    order = list(exp2_analysis.COMPONENT_ORDER)
    labels = list(banks.index)
    per_bank = {"examples": (banks["examples_delta"], reps[:, :, order.index("examples")]),
                "sight": (banks["sight_delta"], reps[:, :, order.index("sight")])}
    per_bank["difference"] = (per_bank["examples"][0] - per_bank["sight"][0],
                              per_bank["examples"][1] - per_bank["sight"][1])
    seed_var = {}
    if family == "gnn":
        seeds = [c[-2:] for c in banks.columns if c.startswith("examples_s")]
        ex = banks[[f"examples_{s}" for s in seeds]].to_numpy()
        si = banks[[f"sight_{s}" for s in seeds]].to_numpy()
        seed_var = {"examples": np.var(ex, axis=1, ddof=1), "sight": np.var(si, axis=1, ddof=1),
                    "difference": np.var(ex - si, axis=1, ddof=1)}
    q = quartile.loc[labels].to_numpy()
    w_all = 1.0 / banks["weight_se_exp1"].to_numpy() ** 2
    out = []
    for k in sorted(set(q)):
        m = q == k
        w = w_all[m]
        row = {"quartile": int(k), "banks": int(m.sum()),
               "count_range": span(train_pos.loc[labels][m]),
               "sight_exceeds_examples": int((banks["sight_delta"][m]
                                              > banks["examples_delta"][m]).sum())}
        for name, (delta, rep) in per_bank.items():
            r = rep[:, m]
            if seed_var:
                r = r + (np.random.default_rng(SEED).standard_normal(r.shape)
                         * np.sqrt(seed_var[name][m] / 3.0)[None, :])
            finite = np.all(np.isfinite(r), axis=1)
            wm = (r[finite] * w).sum(axis=1) / w.sum()
            row[name] = {"weighted_mean": float((w * delta.to_numpy()[m]).sum() / w.sum()),
                         "ci95": np.percentile(wm, [2.5, 97.5]).tolist(),
                         "replicates_valid": int(finite.sum())}
        out.append(row)
    return out


def bank_facts(banks: pd.DataFrame, bank: str) -> dict:
    row = banks.loc[bank]
    return {"steps_sum": row["examples_delta"] + row["sight_delta"] + row["stub_delta"],
            "level_differences": {
                "examples": row["clocal_ap"] - row["solo_ap"],
                "sight": row["cglobal_ap"] - row["clocal_ap"],
                "stub": row["global_ap"] - row["cglobal_ap"],
                "total": row["global_ap"] - row["solo_ap"]}}


def placebo_facts(clocal: Path, cglobal: Path) -> dict:
    """The blind tree's two qualifying cells, payment by payment."""
    cl, cg = scores(clocal, "test_scores"), scores(cglobal, "test_scores")
    cl_payment, cg_payment = scores(clocal, "test_payment"), scores(cglobal, "test_payment")
    idx = np.searchsorted(cg_payment, cl_payment)
    matched = cg[:, idx]
    return {"payments_matched": np.array_equal(cg_payment[idx], cl_payment),
            "scores_compared": cl.size,
            "max_abs_diff": np.abs(cl.astype(np.float64) - matched).max(),
            "bit_identical": np.array_equal(cl, matched)}


def sightcheck_facts(original: pd.DataFrame, copy_weighted: pd.DataFrame,
                     metrics: dict) -> dict:
    diff = copy_weighted["sight_delta"] - original["sight_delta"]
    w = 1.0 / original["weight_se_exp1"] ** 2
    seeds = settings(original, "sight")
    return {"difference_positive": (diff > 0).sum(),
            "difference_median": diff.median(),
            "difference_per_seed": [
                (w * (copy_weighted[f"sight_s{s}"] - original[f"sight_s{s}"])).sum()
                / w.sum() for s in range(seeds)],
            "share_of_original": (metrics["copy_weighted_minus_original"]["weighted_mean"]
                                  / abs(metrics["original_sight"]["weighted_mean"])),
            "copy_weighted_sight": interval_counts(copy_weighted["sight_delta"],
                                                   copy_weighted["sight_ci_lo"],
                                                   copy_weighted["sight_ci_hi"])}


def level_facts(cells: dict, level: int) -> dict:
    mean = np.array([s["by_level"][str(level)]["mean"] for s in cells.values()])
    lo = np.array([s["band_all"][0] for s in cells.values()])
    hi = np.array([s["band_all"][1] for s in cells.values()])
    fall = np.array([s["ap_all"] for s in cells.values()]) - mean
    return {"banks": len(cells),
            "within": ((mean >= lo) & (mean <= hi)).sum(),
            "below": (mean < lo).sum(), "above": (mean > hi).sum(),
            "median_fall": np.median(fall), "fall_range": span(fall),
            "median_score": np.median(mean), "score_range": span(mean)}


def level_100_facts(cells: dict, seed_lowest: pd.Series | None) -> dict:
    """The level-100 reading beside the band: how many banks fall below the
    full count, by less than the band's width, and the headroom under it."""
    mean = np.array([s["by_level"]["100"]["mean"] for s in cells.values()])
    full = np.array([s["ap_all"] for s in cells.values()])
    lo = np.array([s["band_all"][0] for s in cells.values()])
    hi = np.array([s["band_all"][1] for s in cells.values()])
    return {"below_full_count": (mean < full).sum(),
            "fall_under_band_width": (full - mean < hi - lo).sum(),
            "within_as_recorded": sum(s["by_level"]["100"]["within_band"]
                                      for s in cells.values()),
            "median_headroom": np.median(full - lo),
            **({} if seed_lowest is None else
               {"dial_seed_lowest": seed_lowest.loc[list(cells)].sum()})}


def reduction_facts(dial: dict, shortfall: pd.DataFrame, family: str,
                    paths: pd.DataFrame) -> dict:
    """Figure sf-reduction: the case reduction at every level, and the
    five-case fall beside each bank's value of sight on either path."""
    banks = dial["banks"]
    levels = {}
    for level in DIAL_LEVELS:
        cells = {b: s for b, s in banks.items() if str(level) in s["by_level"]}
        levels[str(level)] = level_facts(cells, level)
    solo = shortfall.loc[list(banks),
                         [f"solo_ap_s{s}" for s in range(settings(shortfall, "solo_ap"))]]
    seed = exp1.SEEDS.index(exp2.DIAL_GNN_SEED)
    seed_lowest = (None if family == "tree" else
                   pd.Series(np.argmin(solo.to_numpy(), axis=1) == seed, index=solo.index))
    top = {b: s for b, s in banks.items() if "100" in s["by_level"]}
    levels["100"] |= level_100_facts(top, seed_lowest)
    levels["100"]["groups"] = {
        name: level_facts(group, 100) | level_100_facts(group, seed_lowest)
        for name, (lo, hi) in LEVEL_100_GROUPS.items()
        for group in [{b: s for b, s in top.items() if lo <= s["train_pos"] <= hi}]}
    full_count = (solo.median(axis=1) if family == "tree"
                  else solo.iloc[:, seed])
    band_check = max_diff([
        ([banks[b]["ap_all"] for b in solo.index], full_count),
        ([banks[b]["band_all"][0] for b in solo.index], solo.min(axis=1)),
        ([banks[b]["band_all"][1] for b in solo.index], solo.max(axis=1))])
    draw_ranges = np.array([np.ptp(s["by_level"][lv]["draws"])
                            for s in banks.values() for lv in s["by_level"]])
    full = np.array([s["ap_all"] for s in banks.values()])
    delta = shortfall.loc[list(banks), "delta"]
    curves = {b: [s["by_level"][str(lv)]["mean"] for lv in DIAL_LEVELS
                  if str(lv) in s["by_level"]] + [s["ap_all"]]
              for b, s in banks.items()}
    return {"banks": len(banks),
            "train_pos_range": span([s["train_pos"] for s in banks.values()]),
            "below_100_range": span([s["train_pos"] for s in banks.values()
                                     if "100" not in s["by_level"]]),
            "levels": levels,
            "full_count": {"median_score": np.median(full), "score_range": span(full),
                           "at_band_floor": sum(s["ap_all"] == s["band_all"][0]
                                                for s in banks.values()),
                           "at_band_ceiling": sum(s["ap_all"] == s["band_all"][1]
                                                  for s in banks.values())},
            "highest_level5_minus_floor": max(s["by_level"]["5"]["mean"] - s["band_all"][0]
                                              for s in banks.values()),
            "draw_range_median": np.median(draw_ranges),
            "draw_cells": len(draw_ranges),
            "draw_cells_over_0.2": (draw_ranges > 0.2).sum(),
            "not_monotone": sorted(b for b, c in curves.items()
                                   if np.any(np.diff(c) <= 0)),
            "shortfall": {"range": span(delta), "median": delta.median(),
                          "below_zero": (delta < 0).sum()},
            "drop5_exceeds_shortfall": sum(
                s["drop_all_to_lowest"] > abs(delta[b]) for b, s in banks.items()),
            "drop5_exceeds_sight": {
                path: {"value": sum(s["drop_all_to_lowest"] > paths.loc[b, f"{path}_delta"]
                                    for b, s in banks.items()),
                       "interval_upper_end": sum(
                           s["drop_all_to_lowest"] > paths.loc[b, f"{path}_ci_hi"]
                           for b, s in banks.items())}
                for path in ("sight", "sight_own")},
            "full_count_vs_shortfall_table": band_check}


def pool_facts(roster: pd.DataFrame, network: dict, ledger: dict,
               census: dict) -> dict:
    fewest = roster["train_pos"].min()
    return {"network_train_rows": network["rows"]["train_copies"],
            "ledger_train_rows": ledger["rows"]["train_copies"],
            "train_rows_ratio": (network["rows"]["train_copies"]
                                 / ledger["rows"]["train_copies"]),
            "identifiers": census["n_banks_padded"],
            "identifiers_outside_pool": census["n_banks_padded"] - len(roster),
            "train_pos_range": span(roster["train_pos"]),
            "train_pos_fewest": sorted(roster.index[roster["train_pos"] == fewest]),
            "train_pos_at_least_30": (roster["train_pos"] >= 30).sum()}


def copy_facts(clocal: dict, cglobal: dict) -> dict:
    """How often a payment enters the qualifying pool twice, from the GNN
    family's C-local copies and C-global unique payments."""
    out = {}
    for window in ("train", "val"):
        pos_c, pos_u = clocal["positives"][window], cglobal["positives"][window]
        neg_c = clocal["rows"][window] - pos_c
        neg_u = cglobal["rows"][window] - pos_u
        out[window] = {"second_laundering": pos_c - pos_u,
                       "laundering_share": (pos_c - pos_u) / pos_u,
                       "second_normal": neg_c - neg_u,
                       "normal_share": (neg_c - neg_u) / neg_u,
                       "factor": (pos_c / pos_u) / (neg_c / neg_u)}
    return out


def exp2_metrics(inputs: dict, run_dir: Path) -> dict:
    t0 = time.time()
    recorded = inputs["recorded"]
    banks = {v: frame(p) for v, p in recorded.items()}
    runs = {v: _metrics(p) for v, p in recorded.items()}
    roster = frame(recorded["tree"], "roster")
    shortfall = {f: _metrics(p) for f, p in inputs["shortfall"].items()}
    quartile = frame(inputs["examples"], "examples")["quartile"] + 1

    components = {f: component_facts(banks[f], runs[f], array(recorded[f], "replicates"),
                                     roster, shortfall[f]["primary"]["weighted_mean"])
                  for f in DETECTORS}
    for f in DETECTORS:
        components[f]["quartiles"] = quartile_components(
            banks[f], array(recorded[f], "replicates"), f, quartile, roster["train_pos"])
        components[f]["per_bank_total_minus_exp1"] = max_diff(
            [(banks[f]["total_delta"],
              frame(inputs["shortfall"][f]).loc[banks[f].index, "delta"])])
    blind, gnn = inputs["cells"]["blind"], inputs["cells"]["gnn"]
    recorded_facts = {
        "pool": pool_facts(roster, _metrics(inputs["network_all_tree"]),
                           _metrics(inputs["ledger_qualifying_tree"]),
                           _metrics(inputs["census"])),
        "copies": copy_facts(_metrics(gnn["clocal"][0]), _metrics(gnn["cglobal"][0])),
        "components": components,
        "banks": {b: {f: bank_facts(banks[f], b) for f in DETECTORS} for b in NAMED_BANKS},
        "placebo": placebo_facts(blind["clocal"][0], blind["cglobal"][0]),
        "sightcheck": sightcheck_facts(banks["gnn"], banks["gnn_copy_weighted"],
                                       runs["gnn_copy_weighted"]),
        "reduction": {f: reduction_facts(_metrics(p), frame(inputs["shortfall"][f]), f,
                                         frame(inputs["secondpath"][f]))
                      for f, p in inputs["dial"].items()},
        "secondpath": {f: secondpath_facts(frame(p), _metrics(p))
                       for f, p in inputs["secondpath"].items()}}

    qualifying, tables, replicates, split, split_tables = {}, [], [], {}, []
    for variant, family in VARIANTS.items():
        dirs = inputs["cells"][variant]
        cell = exp2_analysis.load_four_cells(family, dirs)
        if variant in DETECTORS:
            table = split_components(family, cell, dirs["solo"])
            split[variant] = split_facts(table, banks[variant])
            split_tables.append(table.add_prefix(f"{variant}_"))
        result = qualifying_reading(family, cell, recorded[variant],
                                    roster["train_pos"], quartile)
        qualifying[variant] = result["summary"]
        tables.append(result["table"].add_prefix(f"{variant}_"))
        replicates.append(result["reps"])
        print(f"qualifying {variant}: {result['summary']['weighted_mean']:+.4f} "
              f"{result['summary']['ci95']} checks {result['summary']['checks']}",
              flush=True)

    return plain({
        "dataset_key": KEY, **recorded_facts, "qualifying": qualifying,
        "never_seen_split": split, "seconds": time.time() - t0,
        "artifacts": {
            "qualifying": write_csv(run_dir, "qualifying", pd.concat(tables, axis=1)),
            "never_seen_split": write_csv(run_dir, "never_seen_split",
                                          pd.concat(split_tables, axis=1)),
            "qualifying_replicates": write_array(
                run_dir, "qualifying_replicates",
                np.stack(replicates, axis=2).astype(np.float32))}})


# -------------------------------------------------------------- exp4 --

def exp4_inputs() -> dict:
    runs4, runs2 = index("exp4"), index("exp2")
    verdict = find(runs4, check="verdict")
    policy = {f: runs4[_config(verdict)["inputs"][KEY][f]][0] for f in ("tree", "gnn")}
    return {"verdict": verdict, "policy": policy,
            "components": {f: runs2[_config(p)["inputs"]["components"]][0]
                           for f, p in policy.items()}}


def exp4_config(inputs: dict) -> dict:
    return {"experiment": "exp4", "check": CHECK, "dataset_key": KEY,
            "recovery": RECOVERY,
            "inputs": {"verdict": _hash_of(inputs["verdict"]),
                       "policy": {f: _hash_of(p) for f, p in inputs["policy"].items()},
                       "components": {f: _hash_of(p)
                                      for f, p in inputs["components"].items()}}}


def recovery_facts(policy: pd.DataFrame, components: pd.DataFrame,
                   recorded: dict) -> dict:
    """Section bd-gain: the weighted mean gain over the quarter of banks with
    the fewest confirmed cases, as a share of their weighted mean shortfall
    and of their weighted mean value of training examples with sight held at
    the ledger; checked against the verdict Run's gain and share."""
    q1 = policy.index[policy["quartile"] == 1]
    w = 1.0 / policy.loc[q1, "weight_se_exp1"] ** 2
    gain_weighted = (w * policy.loc[q1, "policy_gain"]).sum() / w.sum()
    out = {"banks": len(q1), "count_range": span(policy.loc[q1, "train_pos"]),
           "gain": {"weighted": gain_weighted}}
    for name, y in (("shortfall", policy["delta_exp1"]),
                    ("examples", components["examples_delta"])):
        weighted = (w * y.loc[q1]).sum() / w.sum()
        out[name] = {"weighted_mean": weighted,
                     "share_weighted": gain_weighted / weighted}
    out["checks"] = {
        "gain_vs_verdict_run": gain_weighted - recorded["weighted_mean"],
        "share_vs_verdict_run": (out["shortfall"]["share_weighted"]
                                 - recorded["share_of_shortfall_recovered"])}
    return out


def exp4_metrics(inputs: dict, run_dir: Path) -> dict:
    verdict = _metrics(inputs["verdict"])
    return plain({
        "dataset_key": KEY,
        "recovery": {f: recovery_facts(frame(inputs["policy"][f]),
                                       frame(inputs["components"][f]),
                                       verdict["families"][f])
                     for f in inputs["policy"]}})


# --------------------------------------------------------------- thesis --

WEIGHTED = ("Experiment 1's weights 1/s_B^2 from the Experiment 1 shortfall Run the policy Run "
            "cites (banks.csv weight_se_exp1); quarters are the policy Run's "
            "quartile column")
SINGLE_LENDER = ("each lender in turn as the only detector offered at each "
                 "bank of the quarter holding the fewest confirmed cases, "
                 "under the abstain rule: "
                 "borrowed when its median (tree) or mean (GNN) validation "
                 "AP over settings exceeds the bank's own, as Experiment 4's policy "
                 "decides; the gain is the per-setting difference of the "
                 "lender's test AP and the bank's Experiment 1 solo test AP, paired as "
                 "Experiment 4 pairs, and zero where the bank abstains")
WITHOUT_ABSTAIN = ("the chosen detector borrowed at every bank: the policy "
                   "Run's gain_if_borrowed")
LINE = ("25 % of the weighted mean Experiment 1 shortfall of the quarter of "
        "banks holding the fewest confirmed cases, as each policy Run records it")
CENSUS = ("Table meth-census's laundering rate as one laundering payment in so "
          "many payments, from each census Run's payments and laundering "
          "payments (AMLgentex: the transfers left once the source and sink "
          "rows are dropped); and the range of laundering payments per bank in "
          "the test set over the six released AMLgentex datasets labelled "
          "difficult, from their census Runs' test_sar_min and test_sar_max")
LARGEST = 30
SIZES = ("touching", "train_rows", "test_rows")
TYPOLOGY = ("the laundering_count-weighted mean of pattern_match_share over "
            "the qualifying banks")
RAIL = ("the rail bank's distinct accounts on either side, its share of all "
        "payments (either endpoint), and the payments between two of its "
        "accounts")
SHARED = "account strings that occur at two or more banks, either side"
ALWAYS_ZERO = "graph-derived columns of the global table that are zero on every payment"
NEIGHBOURHOOD = {"dataset_key": "HI-Small", "bank": "023",
                 "rule": ("the laundering payments among the qualifying "
                          "banks split into connected components; the "
                          "smallest spanning exactly four qualifying banks, "
                          "with every qualifying-bank account that pays or "
                          "is paid by one of its accounts")}


def thesis_inputs() -> dict:
    runs0, runs1, runs2, runs4 = index("checks"), index("exp1"), index("exp2"), index("exp4")
    cells = exp4_inputs()
    cited = {f: _config(p)["inputs"] for f, p in cells["policy"].items()}
    facts2 = completed(EXPERIMENT).get(config_hash(exp2_config(exp2_inputs())))
    if facts2 is None:
        raise SystemExit("exp2: no facts Run; run that block first")
    facts2_inputs = _config(facts2)["inputs"]
    return {"verdict": cells["verdict"], "policy": cells["policy"],
            "transfer": {f: {b: runs4[h][0] for b, h in c["transfer"].items()}
                         for f, c in cited.items()},
            "solo": {f: {b: [runs1[h][0] for h in hs] for b, hs in c["solo"].items()}
                     for f, c in cited.items()},
            "roster": runs1[cited["tree"]["shortfall"]][0],
            "facts_exp2": facts2,
            "components": runs2[facts2_inputs["components"]["tree"]][0],
            "secondpath": runs2[facts2_inputs["secondpath"]["tree"]][0],
            "census": {k: find(runs0, check="amlworld_census", dataset=k)
                       for k in amlworld.SETS},
            "transxion_census": find(runs0, check="transxion_census"),
            "amlgentex_census": {name: find(runs0, check="amlgentex_census", dataset=name)
                                 for name in amlgentex.released_names()},
            "ceiling": find(index("exp3"), check="ceiling_descriptors"),
            "descriptors": find(index("exp3"), check="descriptors",
                                dataset_key=KEY)}


def thesis_config(inputs: dict) -> dict:
    return {"experiment": "thesis", "check": CHECK, "dataset_key": KEY,
            "weighted": WEIGHTED,
            "single_lender": SINGLE_LENDER, "without_abstain": WITHOUT_ABSTAIN,
            "line": LINE, "largest": {"banks": LARGEST, "sizes": list(SIZES)},
            "census": CENSUS, "typology": TYPOLOGY, "rail": RAIL, "shared": SHARED,
            "always_zero": ALWAYS_ZERO, "neighbourhood": NEIGHBOURHOOD,
            "inputs": {
                "verdict": _hash_of(inputs["verdict"]),
                "policy": {f: _hash_of(p) for f, p in inputs["policy"].items()},
                "facts_exp2": _hash_of(inputs["facts_exp2"]),
                "components_tree": _hash_of(inputs["components"]),
                "secondpath_tree": _hash_of(inputs["secondpath"]),
                "amlworld_census": {k: _hash_of(p)
                                    for k, p in inputs["census"].items()},
                "transxion_census": _hash_of(inputs["transxion_census"]),
                "amlgentex_census": {k: _hash_of(p)
                                     for k, p in inputs["amlgentex_census"].items()},
                "ceiling_descriptors": _hash_of(inputs["ceiling"]),
                "descriptors": _hash_of(inputs["descriptors"]),
                "canonical": "data/interim/ibm-aml/canonical",
                "global_table": f"data/interim/features/{data.AMLGENTEX_DATASET}_global.parquet",
                "ledgers": f"data/interim/exp1/ledgers/{NEIGHBOURHOOD['dataset_key']}"}}


def policy_frame(path: Path) -> pd.DataFrame:
    """A policy Run's banks.csv, the picked lender read as the identifier it is."""
    return pd.read_csv(verify(path, _metrics(path)["artifacts"]["banks"]),
                       dtype={"bank": str, "pick": str}).set_index("bank")


def weighted(values: pd.Series, se: pd.Series) -> float:
    w = 1.0 / se.loc[values.index] ** 2
    return (w * values).sum() / w.sum()


def solo_test_ap(dirs: list[Path], family: str) -> np.ndarray:
    if family == "gnn":
        return np.array([_metrics(d)["test_ap"] for d in dirs])
    return np.array([s["test_ap"] for s in _metrics(dirs[0])["settings"]])


def lender_gains(banks: pd.DataFrame, transfer: dict, solo: dict,
                 family: str) -> tuple[pd.DataFrame, pd.Series]:
    """Per bank, each candidate's gain if borrowed and whether the abstain
    rule lets the bank borrow it."""
    reduce = np.median if exp2_analysis.PAIRING[family] == "median_over_grid" else np.mean
    gains, borrows = {}, {}
    for bank in banks.index:
        tm = _metrics(transfer[bank])
        own_val = reduce(list(tm["own_val_ap"].values()) if family == "gnn"
                         else tm["own_val_ap"])
        own_test = solo_test_ap(solo[bank], family)
        gains[bank] = {c: reduce(np.subtract(v["test_ap"], own_test))
                       for c, v in tm["per_candidate"].items()}
        borrows[bank] = {c: reduce(v["val_ap"]) > own_val
                         for c, v in tm["per_candidate"].items()}
    return (pd.DataFrame.from_dict(gains, orient="index"),
            pd.DataFrame.from_dict(borrows, orient="index").fillna(False).astype(bool))


def single_lender_facts(banks: pd.DataFrame, gains: pd.DataFrame,
                        borrows: pd.DataFrame) -> dict:
    """Section bd-rule: one lender's detector in place of the choice."""
    q1 = banks.index[banks["quartile"] == 1]
    se = banks["weight_se_exp1"]
    lenders = sorted(banks.index[banks["is_lender"]])
    single = gains.loc[q1, lenders].where(borrows.loc[q1, lenders], 0.0)
    means = pd.Series({c: weighted(single[c], se) for c in lenders})
    picked = pd.Series({b: gains.loc[b, banks.loc[b, "pick"]] for b in banks.index})
    return {"weighted_mean": means.to_dict(), "range": span(means),
            "best_over_worst": means.max() / means.min(),
            "below_zero": (single < 0).sum().to_dict(),
            "below_zero_most": (single < 0).sum().max(),
            "borrows": borrows.loc[q1, lenders].sum().to_dict(),
            "check_pick_vs_gain_if_borrowed": (picked - banks["gain_if_borrowed"]).abs().max()}


def without_abstain_facts(banks: pd.DataFrame, metrics: dict) -> dict:
    """Section bd-rule: the abstain rule's effect, and the banks it shields."""
    se = banks["weight_se_exp1"]
    q1 = banks.index[banks["quartile"] == 1]
    free, policy = banks["gain_if_borrowed"], banks["policy_gain"]
    harmed = free < 0
    out = {}
    for scope, rows, recorded in (("quartile_1", q1, metrics["primary"]),
                                  ("all", banks.index, metrics["all_banks"])):
        without, with_rule = weighted(free.loc[rows], se), weighted(policy.loc[rows], se)
        out[scope] = {"without": without, "with": with_rule,
                      "rule_adds": with_rule - without,
                      "with_vs_recorded": with_rule - recorded["weighted_mean"]}
    return out | {"harmed": harmed.sum(),
                  "harmed_kept_own": (harmed & ~banks["borrow"]).sum(),
                  "harmed_borrowed": (harmed & banks["borrow"]).sum(),
                  "largest_fall": -policy.min(),
                  "q1_intervals": {
                      "wholly_above": (banks.loc[q1, "policy_ci_lo"] > 0).sum(),
                      "wholly_below": (banks.loc[q1, "policy_ci_hi"] < 0).sum()}}


def quarter_facts(banks: pd.DataFrame, qualifying: pd.Series | None) -> list:
    """Tables sf-quartiles and bd-quartiles, weighted."""
    se = banks["weight_se_exp1"]
    out = []
    for q, sub in banks.groupby("quartile"):
        row = {"quartile": q, "banks": len(sub), "count_range": span(sub["train_pos"]),
               "shortfall": weighted(sub["delta_exp1"], se),
               "shortfall_below_zero": (sub["delta_exp1"] < 0).sum(),
               "gain": weighted(sub["policy_gain"], se),
               "gain_below_zero": (sub["policy_gain"] < 0).sum(),
               "borrows": sub["borrow"].sum()}
        if qualifying is not None:
            row["qualifying"] = weighted(qualifying.loc[sub.index], se)
            row["qualifying_below_zero"] = (qualifying.loc[sub.index] < 0).sum()
        out.append(row)
    return out


def largest_facts(roster: pd.DataFrame, q1: pd.Index) -> dict:
    """The banks a design scoring only the largest would include."""
    out = {}
    for size in SIZES:
        top = roster.sort_values(size, ascending=False, kind="stable").head(LARGEST)
        out[size] = {"fewest_cases": top["train_pos"].min(),
                     "in_quartile_1": top.index.isin(q1).sum()}
    return out


def rail_facts(df: pd.DataFrame) -> dict:
    rail = amlworld.RAIL_BANK
    on_from = (df["from_bank"] == rail).to_numpy()
    on_to = (df["to_bank"] == rail).to_numpy()
    accounts = (set(df["from_acct"][on_from].astype(str))
                | set(df["to_acct"][on_to].astype(str)))
    return {"accounts": len(accounts), "payments": len(df),
            "share_of_payments": (on_from | on_to).mean(),
            "between_rail_accounts": (on_from & on_to).sum()}


def shared_account_facts(df: pd.DataFrame) -> dict:
    keys = pd.concat([
        pd.DataFrame({"bank": df["from_bank"].astype(str), "acct": df["from_acct"].astype(str)}),
        pd.DataFrame({"bank": df["to_bank"].astype(str), "acct": df["to_acct"].astype(str)})
    ]).drop_duplicates()
    banks_per = keys.groupby("acct")["bank"].nunique()
    return {"shared": (banks_per > 1).sum(), "most_banks": banks_per.max()}


def always_zero_facts(key: str) -> dict:
    names = graph_feature_names()
    table = pd.read_parquet(data.global_table(key), columns=names)
    zero = [n for n in names if not table[n].any()]
    return {"columns": len(names), "always_zero": len(zero),
            "windows": sorted({n.rsplit("_", 1)[1] for n in zero})}


def neighbourhood_facts(key: str, focal: str) -> dict:
    """Figure meth-scenarios: the counts its caption and text quote."""
    roster = list(data.read_roster(key).index)
    net = pd.concat([data.read_ledger(key, b)[["from_bank", "from_acct", "to_bank",
                                                "to_acct", "is_laundering", "payment"]]
                     for b in roster]).drop_duplicates("payment")
    payer = net["from_bank"].astype(str) + "/" + net["from_acct"].astype(str)
    payee = net["to_bank"].astype(str) + "/" + net["to_acct"].astype(str)
    net = net.assign(payer=payer, payee=payee)[payer != payee]

    def bank_of(node):
        return node.split("/")[0]

    laundering = net[net["is_laundering"] == 1]
    components = nx.connected_components(
        nx.from_pandas_edgelist(laundering, "payer", "payee"))
    chain = min((c for c in components
                 if len({bank_of(n) for n in c}) == 4
                 and {bank_of(n) for n in c} <= set(roster)),
                key=lambda c: (len(c), sorted(c)[0]))
    hood = set(chain)
    for node in chain:
        touching = net[(net["payer"] == node) | (net["payee"] == node)]
        hood |= {n for n in set(touching["payer"]) | set(touching["payee"])
                 if bank_of(n) in roster}
    drawn = net[net["payer"].isin(hood) & net["payee"].isin(hood)]
    pairs = {tuple(sorted(p)) for p in zip(drawn["payer"], drawn["payee"])}
    seen = (drawn["payer"].map(bank_of) == focal) | (drawn["payee"].map(bank_of) == focal)
    return {"accounts": len(hood), "banks": len({bank_of(n) for n in hood}),
            "focal_in_chain": focal in {bank_of(n) for n in chain},
            "payments": len(drawn), "pairs": len(pairs),
            "laundering": int(drawn["is_laundering"].sum()),
            "in_focal_sight": int(seen.sum())}


def census_facts(amlworld_census: dict, transxion: dict, amlgentex_census: dict) -> dict:
    """Table meth-census's laundering rates; and, for Section meth-roles, the
    range of a bank's laundering payments in the test window over the
    AMLgentex datasets labelled difficult."""
    counts = {k: (m["n_rows"], m["n_laundering"]) for k, m in amlworld_census.items()}
    counts["transxion"] = (transxion["n_rows"], transxion["n_laundering"])
    released = amlgentex_census[data.AMLGENTEX_DATASET]
    counts[data.AMLGENTEX_DATASET] = (released["n_real_transfers"],
                                      released["n_sar_transfers"])
    difficult = [m for name, m in amlgentex_census.items() if name.endswith("_difficult")]
    return {"laundering_one_in": {k: rows / laundering
                                  for k, (rows, laundering) in counts.items()},
            "difficult_test_sar_range": [min(m["test_sar_min"] for m in difficult),
                                         max(m["test_sar_max"] for m in difficult)]}


def thesis_metrics(inputs: dict, run_dir: Path) -> dict:
    t0 = time.time()
    policy = {f: policy_frame(p) for f, p in inputs["policy"].items()}
    qualifying = frame(inputs["facts_exp2"], "qualifying")["tree_delta"]
    roster = frame(inputs["roster"], "roster")
    readings = {f: _metrics(p) for f, p in inputs["policy"].items()}
    lender = {}
    for f, banks in policy.items():
        gains, borrows = lender_gains(banks, inputs["transfer"][f], inputs["solo"][f], f)
        lender[f] = single_lender_facts(banks, gains, borrows)
    components = _metrics(inputs["components"])["readings"]
    secondpath = _metrics(inputs["secondpath"])["readings"]
    ceiling = frame(inputs["ceiling"], "ceiling_descriptors")
    descriptors = frame(inputs["descriptors"], "descriptors")
    over_chance = [_metrics(p)["within_bank_structure_excl_070"][c]["over_chance"]
                   for p in inputs["census"].values() for c in ("laundering", "licit")]
    rail, shared = {}, None
    for k in amlworld.SETS:
        df = amlworld.load(k)
        rail[k] = rail_facts(df)
        if k == KEY:
            shared = shared_account_facts(df)
        del df
    q1 = policy["tree"].index[policy["tree"]["quartile"] == 1]
    lenders = policy["tree"].loc[policy["tree"]["is_lender"], "train_pos"]
    return plain({
        "dataset_key": KEY,
        "single_lender": lender,
        "without_abstain": {f: without_abstain_facts(b, readings[f])
                            for f, b in policy.items()},
        "quarters": {f: quarter_facts(b, qualifying if f == "tree" else None)
                     for f, b in policy.items()},
        "line": {f: readings[f]["threshold"]["line_quarter_of_weighted_shortfall"]
                 for f in policy},
        "largest": largest_facts(roster, q1),
        "lenders": {"banks": sorted(lenders.index), "count_range": span(lenders)},
        "pool_train_rows_ratio": _metrics(inputs["facts_exp2"])["pool"]["train_rows_ratio"],
        "interaction_tree": {
            "sight_difference": (components["sight"]["weighted_mean"]
                                 - secondpath["sight_own"]["weighted_mean"]),
            "examples_difference": (secondpath["examples_network"]["weighted_mean"]
                                    - components["examples"]["weighted_mean"]),
            "per_configuration": secondpath["interaction"]["weighted_mean"]},
        "over_chance_range": span(over_chance),
        "census": census_facts({k: _metrics(p) for k, p in inputs["census"].items()},
                               _metrics(inputs["transxion_census"]),
                               {k: _metrics(p)
                                for k, p in inputs["amlgentex_census"].items()}),
        "typology": {"coverage": ((ceiling["pattern_match_share"]
                                   * ceiling["laundering_count"]).sum()
                                  / ceiling["laundering_count"].sum()),
                     "per_bank_range": span(ceiling["pattern_match_share"])},
        "rho_payments_active_accounts": spearman(
            descriptors["payment_count"].to_numpy(),
            descriptors["active_accounts"].to_numpy()),
        "rail": rail,
        "shared_accounts": shared,
        "always_zero": always_zero_facts(data.AMLGENTEX_DATASET),
        "neighbourhood": neighbourhood_facts(NEIGHBOURHOOD["dataset_key"],
                                             NEIGHBOURHOOD["bank"]),
        "seconds": time.time() - t0})


# ----------------------------------------------------------------- main --

BLOCKS = {"exp1": (exp1_inputs, exp1_config, exp1_metrics),
          "exp2": (exp2_inputs, exp2_config, exp2_metrics),
          "exp4": (exp4_inputs, exp4_config, exp4_metrics),
          "thesis": (thesis_inputs, thesis_config, thesis_metrics)}


def cmd_run(name: str) -> dict:
    inputs_fn, config_fn, metrics_fn = BLOCKS[name]
    inputs = inputs_fn()
    metrics = run(EXPERIMENT, config_fn(inputs),
                  lambda config, run_dir: metrics_fn(inputs, run_dir))
    print(f"{name}: {len(metrics)} groups recorded", flush=True)
    return metrics


def cmd_status() -> int:
    for name, (inputs_fn, config_fn, _) in BLOCKS.items():
        try:
            inputs = inputs_fn()
        except SystemExit as missing:
            print(f"{name}: inputs incomplete ({missing})")
            continue
        state = "analysed" if config_hash(config_fn(inputs)) in completed(EXPERIMENT) else "ready"
        print(f"{name}: {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    exp2_examples.check_execution_settings()
    exp1.install_signal_handlers()
    cmd = sys.argv[1]
    if cmd in BLOCKS:
        cmd_run(cmd)
        return 0
    if cmd == "all":
        for name in BLOCKS:
            cmd_run(name)
        return 0
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
