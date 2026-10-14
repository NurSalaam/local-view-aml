#!/usr/bin/env python
"""Experiment 1's analysis Runs: one ``shortfall`` Run per (dataset, family) over the unit Runs,
and one ``verdict`` Run on hypothesis 1 over the shortfall Runs. Both are pure
functions over the Run log; an analysis config lists its inputs' config
hashes, so a re-done input re-keys the analysis. ``scores`` prints the
measured scores beside the published ones (Chapter 3) from the global Runs.

  uv run python scripts/exp1_analysis.py shortfall <key> <family>
  uv run python scripts/exp1_analysis.py all          # every complete cell
  uv run python scripts/exp1_analysis.py verdict
  uv run python scripts/exp1_analysis.py scores
  uv run python scripts/exp1_analysis.py status
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402, F401  (before lightgbm: OpenMP runtimes, see exp1.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1  # noqa: E402
import exp1_data as data  # noqa: E402
from extract import load_column  # noqa: E402
from m import completed, run  # noqa: E402
from m.artifacts import load_array, verify, write_array, write_csv  # noqa: E402
from m.runlog import config_hash  # noqa: E402
from m.stats import (METRICS, aggregate_shortfalls, heterogeneity,  # noqa: E402
                     joint_bootstrap, leave_one_out, null_q_check,
                     spearman_bank_bootstrap)

REPLICATES = 1000
CHUNK = 100
SEED = 0
NULL_LEVEL = 0.95
LOW_BAND, SUBSTANTIAL_BAND = 0.25, 0.50
SPEARMAN = {"draws": 1000, "permutations": 10_000, "seed": 0}
PAIRING = {"tree": "median_over_grid", "blind": "median_over_grid",
           "gnn": "mean_over_seeds"}
AGGREGATE = {"median_over_grid": "median", "mean_over_seeds": "mean"}
VERDICT_FAMILIES = ("tree", "gnn")
SCORE_KEYS = ("HI-Small", "HI-Medium", "LI-Small", "LI-Medium", "transxion",
              data.AMLGENTEX_DATASET)

STABILITY_DRAWS = 200
CONTROL_FIRES_WHEN = ("I2 >= 0.25, the assumption-free null check passes, and "
                      "leave-one-out I2 >= 0.25 — criterion 1's three tests")


# --------------------------------------------------------------- inputs --

def unit_dirs(done: dict, key: str, family: str) -> dict | None:
    """Run directories of one cell's inputs, or None while incomplete."""
    roster = exp1.roster_for(key)
    if family == "gnn":
        glob = [exp1.recorded_dir(done, exp1.Unit("gnn_global", key, seed=s)) for s in exp1.SEEDS]
        solo = {b: [exp1.recorded_dir(done, exp1.Unit("gnn_solo", key, b, s)) for s in exp1.SEEDS]
                for b in roster.index}
    else:
        glob = [exp1.recorded_dir(done, exp1.Unit(f"{family}_global", key))]
        solo = {b: [exp1.recorded_dir(done, exp1.Unit(f"{family}_solo", key, b))]
                for b in roster.index}
    if any(d is None for d in glob) or any(d is None for ds in solo.values() for d in ds):
        return None
    return {"global": glob, "solo": solo, "roster": roster}


def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def shortfall_config(key: str, family: str, dirs: dict) -> dict:
    world, dataset = data.DATASETS[key]
    config = {"experiment": exp1.EXPERIMENT, "check": "shortfall", "world": world,
              "dataset": dataset, "dataset_key": key, "family": family,
              "setups": ["solo", "global"], "pairing": PAIRING[family],
              "replicates": REPLICATES, "seed": SEED, "chunk": CHUNK,
              "metrics": list(METRICS),
              "common_base_rate": "roster minimum test base rate; negatives "
                                  "reweighted, positives unit weight",
              "null_check": {"level": NULL_LEVEL, "form": "centred joint "
                             "replicates" + (" plus normal seed noise of "
                                             "variance seed_var/3" if family == "gnn" else "")},
              "bands": {"low": LOW_BAND, "substantial": SUBSTANTIAL_BAND},
              "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(key)},
              "inputs": {"global": [_hash_of(d) for d in dirs["global"]],
                         "solo": {b: [_hash_of(d) for d in ds]
                                  for b, ds in dirs["solo"].items()}},
              "features_version": exp1.FEATURES_VERSION}
    if family == "gnn":
        config["gnn_version"] = exp1.GNN_VERSION
        config["seeds"] = exp1.SEEDS
    return config


def _scores(path: Path, family: str, name: str) -> np.ndarray:
    metrics = json.loads((path / "metrics.json").read_text())
    return load_array(path, name, metrics["artifacts"][name]["sha256"])


def load_cell(key: str, family: str, dirs: dict) -> dict:
    split = load_column(key, "split")
    y_all = load_column(key, "is_laundering")
    test_rows = np.flatnonzero(split == 2)
    n_test = len(test_rows)
    position = np.full(len(split), -1, np.int64)
    position[test_rows] = np.arange(n_test)
    if family == "gnn":
        glob = np.stack([_scores(d, family, "test_probs") for d in dirs["global"]])
    else:
        glob = _scores(dirs["global"][0], family, "test_scores")
    assert glob.shape[1] == n_test
    groups, meta = {}, {}
    for bank, ds in dirs["solo"].items():
        payments = [_scores(d, family, "test_payment") for d in ds]
        for p in payments[1:]:
            assert np.array_equal(p, payments[0]), f"{key} {bank}: seed Runs score different papers"
        payment = payments[0]
        idx = position[payment]
        assert np.all(idx >= 0), f"{key} {bank}: a scored payment lies outside the test window"
        roster_row = dirs["roster"].loc[bank]
        assert len(payment) == int(roster_row["test_rows"]) and len(np.unique(payment)) == len(payment), \
            f"{key} {bank}: the scored paper is not the bank's whole test window"
        if family == "gnn":
            solo = np.stack([_scores(d, family, "test_probs") for d in ds])
        else:
            solo = _scores(ds[0], family, "test_scores")
        y = y_all[payment].astype(np.int8)
        assert int(y.sum()) == int(roster_row["test_pos"]), f"{key} {bank}: test positives differ from the roster"
        groups[bank] = {"idx": idx, "y": y, "solo": solo, "global": glob[:, idx]}
        meta[bank] = {"test_rows": int(len(y)), "test_pos": int(y.sum()),
                      "test_payment_sha256": json.loads((ds[0] / "metrics.json").read_text())
                      ["artifacts"]["test_payment"]["sha256"]}
    rates = np.array([meta[b]["test_pos"] / meta[b]["test_rows"] for b in groups])
    common = float(rates.min())
    for bank, spec in groups.items():
        pi = meta[bank]["test_pos"] / meta[bank]["test_rows"]
        spec["neg_weight"] = float((pi / (1 - pi)) / (common / (1 - common)))
        weighted_rate = meta[bank]["test_pos"] / (
            meta[bank]["test_pos"] + spec["neg_weight"] * (meta[bank]["test_rows"] - meta[bank]["test_pos"]))
        assert abs(weighted_rate - common) < 1e-9 and spec["neg_weight"] >= 1.0
        meta[bank]["test_base_rate"] = pi
    return {"groups": groups, "meta": meta, "n_test": n_test,
            "common_base_rate": common, "global": glob}


def own_best(dirs: dict, family: str, groups: dict) -> dict:
    """Descriptive: each side at its own validation-selected setting."""
    from m.stats import weighted_average_precision as wap

    gm = json.loads((dirs["global"][0] / "metrics.json").read_text())
    g_best = int(np.argmax([s["val_ap"] for s in gm["settings"]]))
    out = {}
    for bank, ds in dirs["solo"].items():
        sm = json.loads((ds[0] / "metrics.json").read_text())
        s_best = int(np.argmax([s["val_ap"] for s in sm["settings"]]))
        spec = groups[bank]
        ones = np.ones((1, len(spec["y"])))
        out[bank] = {"solo_setting": s_best, "global_setting": g_best,
                     "delta": float(wap(spec["y"], spec["global"][g_best], ones)[0]
                                    - wap(spec["y"], spec["solo"][s_best], ones)[0])}
    return out


def global_levels(dirs: dict, family: str) -> dict:
    if family == "gnn":
        seeds = [json.loads((d / "metrics.json").read_text()) for d in dirs["global"]]
        return {"seeds": [{"seed": m["seed"], "test_ap": m["test_ap"],
                           "test_f1_x100": m["test_f1_x100"], "val_ap": m["val_ap"],
                           "epochs": m["epochs"]} for m in seeds],
                "test_ap_mean": float(np.mean([m["test_ap"] for m in seeds])),
                "test_f1_x100_mean": float(np.mean([m["test_f1_x100"] for m in seeds]))}
    m = json.loads((dirs["global"][0] / "metrics.json").read_text())
    return {"settings": [{"params": s["params"], "val_ap": s["val_ap"],
                          "test_ap": s["test_ap"], "test_f1_x100": s["test_f1"] * 100,
                          "best_iteration": s["best_iteration"]} for s in m["settings"]],
            "selected": m["selected"], "test_ap_selected": m["test_ap"],
            "test_f1_x100_selected": m["test_f1_x100"],
            "test_ap_median": float(np.median([s["test_ap"] for s in m["settings"]]))}


# ------------------------------------------------------------ analysis --

def band(i2: float) -> str:
    return "low" if i2 < LOW_BAND else ("moderate" if i2 < SUBSTANTIAL_BAND else "substantial")


def heterogeneity_reading(labels, deltas, ses, delta_reps, seed_var=None) -> dict:
    het = heterogeneity(deltas, ses)
    null = null_q_check(delta_reps, deltas, ses, seed_var, seed=SEED, level=NULL_LEVEL)
    loo = leave_one_out(labels, deltas, ses)
    return {**het, "band": band(het["I2"]), "null_check": null, "leave_one_out": loo,
            "clears_low_band": bool(het["I2"] >= LOW_BAND),
            "loo_clears_low_band": bool(loo["I2"] >= LOW_BAND)}


def shortfall_metrics(key: str, family: str, dirs: dict, run_dir: Path) -> dict:
    t0 = time.time()
    cell = load_cell(key, family, dirs)
    groups, meta = cell["groups"], cell["meta"]
    labels = list(groups)
    aggregate = AGGREGATE[PAIRING[family]]
    boot = joint_bootstrap(cell["n_test"], groups, replicates=REPLICATES,
                           seed=SEED, chunk=CHUNK, metrics=METRICS)
    t_boot = time.time() - t0

    readings, per_setting_ses = {}, {}
    table = pd.DataFrame.from_dict(meta, orient="index").loc[labels]
    table.index.name = "bank"
    primary_reps = None
    for metric in METRICS:
        agg = aggregate_shortfalls(boot, metric, aggregate)
        deltas = np.array([agg[b]["delta"] for b in labels])
        se_boot = np.array([agg[b]["se_boot"] for b in labels])
        reps = np.stack([agg[b]["delta_reps"] for b in labels], axis=1)
        if family == "gnn":
            seed_var = np.array([np.var(agg[b]["per_setting"], ddof=1) for b in labels])
            ses = np.sqrt(se_boot ** 2 + seed_var / len(exp1.SEEDS))
        else:
            seed_var = None
            ses = se_boot
        reading = heterogeneity_reading(labels, deltas, ses, reps, seed_var)
        reading["delta_range"] = [float(deltas.min()), float(deltas.max())]
        readings[metric] = reading
        prefix = "" if metric == "ap" else f"{metric}_"
        table[f"{prefix}delta"] = deltas
        table[f"{prefix}se_boot"] = se_boot
        table[f"{prefix}se"] = ses
        table[f"{prefix}ci_lo"] = [agg[b]["ci95"][0] for b in labels]
        table[f"{prefix}ci_hi"] = [agg[b]["ci95"][1] for b in labels]
        if metric == "ap":
            primary_reps = reps
            per_setting = np.stack([agg[b]["per_setting"] for b in labels])
            solo_by = np.stack([agg[b]["solo_by_setting"] for b in labels])
            glob_by = np.stack([agg[b]["global_by_setting"] for b in labels])
            for s in range(per_setting.shape[1]):
                table[f"delta_s{s}"] = per_setting[:, s]
                table[f"solo_ap_s{s}"] = solo_by[:, s]
                table[f"global_ap_s{s}"] = glob_by[:, s]
            table["setting_spread"] = per_setting.max(axis=1) - per_setting.min(axis=1)
            table["solo_ap"] = (np.median if aggregate == "median" else np.mean)(solo_by, axis=1)
            table["global_ap"] = (np.median if aggregate == "median" else np.mean)(glob_by, axis=1)
            if family == "gnn":
                table["seed_var"] = seed_var
                per_seed = []
                for s in range(per_setting.shape[1]):
                    d_s = per_setting[:, s]
                    reps_s = np.stack([boot["reps"]["ap"][b][:, s] for b in labels], axis=1)
                    se_s = np.nanstd(reps_s, axis=0, ddof=1)
                    r = heterogeneity_reading(labels, d_s, se_s, reps_s)
                    per_setting_ses[s] = se_s
                    per_seed.append({"seed": exp1.SEEDS[s], "I2": r["I2"], "Q": r["Q"],
                                     "band": r["band"], "clears_low_band": r["clears_low_band"],
                                     "null_passes": r["null_check"]["passes"],
                                     "loo_clears_low_band": r["loo_clears_low_band"],
                                     "weighted_mean": r["weighted_mean"]})
                pooled = readings["ap"]
                verdict = lambda c, n, o: (bool(c), bool(n), bool(o))  # noqa: E731
                pooled_verdict = verdict(pooled["clears_low_band"], pooled["null_check"]["passes"],
                                         pooled["loo_clears_low_band"])
                for p in per_seed:
                    p["verdict"] = list(verdict(p["clears_low_band"], p["null_passes"],
                                                p["loo_clears_low_band"]))
                readings["ap"]["per_seed"] = per_seed
                readings["ap"]["pooled_verdict"] = list(pooled_verdict)
                readings["ap"]["per_seed_agree"] = all(
                    tuple(p["verdict"]) == pooled_verdict for p in per_seed)
    if family != "gnn":
        ob = own_best(dirs, family, groups)
        table["own_best_delta"] = [ob[b]["delta"] for b in labels]
        table["solo_own_setting"] = [ob[b]["solo_setting"] for b in labels]
    artifacts = {"banks": write_csv(run_dir, "banks", table),
                 "roster": write_csv(run_dir, "roster", dirs["roster"]),
                 "replicates": write_array(run_dir, "replicates", primary_reps.astype(np.float32))}
    if family == "gnn":
        by_seed = np.stack([np.stack([boot["reps"]["ap"][b][:, s] for b in labels], axis=1)
                            for s in range(len(exp1.SEEDS))], axis=2)
        artifacts["replicates_by_seed"] = write_array(run_dir, "replicates_by_seed",
                                                      by_seed.astype(np.float32))
    return {"dataset_key": key, "family": family, "k": len(labels),
            "n_test": int(cell["n_test"]), "common_base_rate": cell["common_base_rate"],
            "pairing": PAIRING[family], "replicates": REPLICATES,
            "primary": readings["ap"],
            "sensitivity": {m: readings[m] for m in METRICS if m != "ap"},
            "global_level": global_levels(dirs, family),
            "solo_level": {"ap_median_over_banks": float(np.median(table["solo_ap"])),
                           "ap_range": [float(table["solo_ap"].min()),
                                        float(table["solo_ap"].max())]},
            "seconds": {"bootstrap": t_boot, "total": time.time() - t0},
            "artifacts": artifacts}


def cmd_shortfall(key: str, family: str, done: dict | None = None) -> dict | None:
    done = done if done is not None else completed(exp1.EXPERIMENT)
    dirs = unit_dirs(done, key, family)
    if dirs is None:
        print(f"shortfall {key} {family}: inputs incomplete", flush=True)
        return None
    metrics = run(exp1.EXPERIMENT, shortfall_config(key, family, dirs),
                  lambda config, run_dir: shortfall_metrics(key, family, dirs, run_dir),
                  done=done)
    p = metrics["primary"]
    print(f"shortfall {key} {family}: k {metrics['k']} I2 {p['I2']:.3f} Q {p['Q']:.1f} "
          f"null {'pass' if p['null_check']['passes'] else 'fail'} "
          f"loo I2 {p['leave_one_out']['I2']:.3f} mean {p['weighted_mean']:+.4f}", flush=True)
    return metrics


def shortfall_run(done: dict, key: str, family: str) -> tuple[Path, dict] | None:
    dirs = unit_dirs(done, key, family)
    if dirs is None:
        return None
    path = done.get(config_hash(shortfall_config(key, family, dirs)))
    if path is None:
        return None
    return path, json.loads((path / "metrics.json").read_text())


# ------------------------------------------------------------- verdict --

def verdict_config(cells: dict) -> dict:
    return {"experiment": exp1.EXPERIMENT, "check": "verdict",
            "primary_dataset": exp1.PRIMARY_KEY, "controls": data.CONTROLS,
            "families": list(VERDICT_FAMILIES),
            "criteria": {
                "1": f"{exp1.PRIMARY_KEY}, both families: I2 >= 0.25, null check "
                     "passes, leave-one-out I2 >= 0.25; the GNN family's reading "
                     "also holds on each of its seeds in turn",
                "2": f"{exp1.PRIMARY_KEY} tree-vs-GNN Spearman rho: bank-bootstrap "
                     "5th percentile > 0",
                "3": "no control fires for either family on TransXion or "
                     f"sweden_100K_easy, where a control fires when {CONTROL_FIRES_WHEN}"},
            "spearman": SPEARMAN, "bands": {"low": LOW_BAND},
            "outcomes": {"supported": "all three criteria hold",
                         "refuted": "any criterion fails"},
            "robustness": "per reading, the common misstatement factor c of every "
                          "s_B at which the reading first fails a line of the triple: "
                          "c_null = sqrt(Q / null_q95), c_band = sqrt(0.75 Q / (k-1)), "
                          "c_loo the same on the leave-one-out reading; binding = min "
                          "over the lines that are defined. Exact where s_B is the "
                          "bootstrap standard error alone; a lower bound for the GNN "
                          "family, whose s_B also carries between-seed variance",
            "null_stability": {"draws": STABILITY_DRAWS, "seeds": "0 to "
                               f"{STABILITY_DRAWS - 1}",
                               "note": "the null's 95th percentile redrawn; "
                                       "deterministic for the tree family"},
            "inputs": {k: {f: h for f, h in fams.items()} for k, fams in cells.items()}}


def robustness(reading: dict, seed_noise_share: float | None = None) -> dict:
    """The factor by which every s_B would have to be misstated for this
    heterogeneity reading to fall to each line of the triple (the robustness
    table). Scaling every s_B by c divides Q by c-squared while the
    assumption-free null holds still, its replicates widening by c as its
    weights fall by c-squared. A factor below 1 says the reading already fails
    that line, and how far past it the reading sits.

    ``seed_noise_share`` is the median share of s_B-squared that the
    between-seed term contributes (the GNN family). Where it is given, s_B is
    not the bootstrap standard error alone, so the factor is a lower bound on
    what the resampling conjecture would require of the bootstrap component by
    itself; where it is absent the factor prices that conjecture exactly. A
    leave-one-out line needs at least three banks to mean anything, and is
    reported as absent below that."""
    loo, q95 = reading["leave_one_out"], reading["null_check"]["null_q95"]
    threshold = 1.0 - LOW_BAND
    factors = {
        "c_null": math.sqrt(reading["Q"] / q95) if q95 > 0 else None,
        "c_band": (math.sqrt(threshold * reading["Q"] / (reading["k"] - 1))
                   if reading["k"] > 1 else None),
        "c_loo": (math.sqrt(threshold * loo["Q"] / (loo["k"] - 1))
                  if loo["k"] > 1 else None)}
    defined = [value for value in factors.values() if value is not None]
    return {**factors, "binding": min(defined) if defined else None,
            "Q": reading["Q"], "k": reading["k"], "null_q95": q95,
            "prices_the_conjecture_exactly": seed_noise_share is None,
            "seed_noise_share_median": seed_noise_share}


def null_stability(reading: dict, reps: np.ndarray, deltas: np.ndarray,
                   ses: np.ndarray, seed_var: np.ndarray | None) -> dict:
    """How much the null's 95th percentile moves when its draw is redrawn.

    For the tree family the null is a deterministic function of
    the recorded replicates, so the line does not move at all. For the GNN
    family each null replicate takes a fresh normal seed-noise draw, so the
    line is a Monte-Carlo estimate: redrawing it ``STABILITY_DRAWS`` times
    measures how often the null check would have read the other way. The
    recomputation reads the persisted float32 replicates, and its seed-0
    reading is reported against the recorded one."""
    q95 = np.array([null_q_check(reps, deltas, ses, seed_var, seed=seed,
                                 level=NULL_LEVEL)["null_q95"]
                    for seed in range(STABILITY_DRAWS if seed_var is not None else 1)])
    observed = reading["Q"]
    return {"draws": int(len(q95)), "deterministic": bool(seed_var is None),
            "null_q95_min": float(q95.min()), "null_q95_median": float(np.median(q95)),
            "null_q95_max": float(q95.max()),
            "passes_share": float((observed > q95).mean()),
            "recorded_null_q95": reading["null_check"]["null_q95"],
            "recorded_passes": reading["null_check"]["passes"],
            "seed0_difference": float(abs(q95[0] - reading["null_check"]["null_q95"]))}


def verdict_metrics(runs: dict, run_dir: Path) -> dict:
    P = exp1.PRIMARY_KEY

    def reading(key, fam):
        return runs[key][fam][1]["primary"]

    def table(key, fam):
        path, metrics = runs[key][fam]
        return pd.read_csv(verify(path, metrics["artifacts"]["banks"]),
                           dtype={"bank": str}).set_index("bank")

    def seed_noise_share(key, fam):
        """Median share of s_B-squared carried by the between-seed term."""
        if fam != "gnn":
            return None
        rows = table(key, fam)
        return float(((rows["seed_var"] / len(exp1.SEEDS)) / rows["se"] ** 2).median())

    def stability(key, fam):
        path, metrics = runs[key][fam]
        rows = table(key, fam)
        reps = np.load(verify(path, metrics["artifacts"]["replicates"])).astype(np.float64)
        seed_var = rows["seed_var"].to_numpy() if fam == "gnn" else None
        return null_stability(reading(key, fam), reps, rows["delta"].to_numpy(),
                              rows["se"].to_numpy(), seed_var)

    identity = {}
    for key, fams in runs.items():
        shas = {fam: table(key, fam)["test_payment_sha256"] for fam in fams}
        first = next(iter(shas.values()))
        identity[key] = all(s.index.equals(first.index) and (s == first).all()
                            for s in shas.values())
    assert all(identity.values()), \
        f"families scored different test sets on {[k for k, v in identity.items() if not v]}"

    c1 = {}
    for fam in ("tree", "gnn"):
        r = reading(P, fam)
        c1[fam] = {"I2": r["I2"], "Q": r["Q"], "k": r["k"], "band": r["band"],
                   "null_passes": r["null_check"]["passes"],
                   "loo_I2": r["leave_one_out"]["I2"],
                   "pass": bool(r["clears_low_band"] and r["null_check"]["passes"]
                                and r["loo_clears_low_band"])}
    g = reading(P, "gnn")
    c1["per_seed"] = {"readings": g.get("per_seed"), "agree": g.get("per_seed_agree")}
    c1["pass"] = (all(c1[f]["pass"] for f in ("tree", "gnn"))
                  and bool(g.get("per_seed_agree")))

    tt, tg = table(P, "tree"), table(P, "gnn")
    banks = tt.index.intersection(tg.index)
    test_set_identity = bool((tt.loc[banks, "test_payment_sha256"]
                              == tg.loc[banks, "test_payment_sha256"]).all())
    assert test_set_identity and len(banks) == len(tt) == len(tg), \
        "the tree and GNN families scored different test sets"
    rho = spearman_bank_bootstrap(tt.loc[banks, "delta"].to_numpy(),
                                  tg.loc[banks, "delta"].to_numpy(),
                                  draws=SPEARMAN["draws"],
                                  permutations=SPEARMAN["permutations"],
                                  seed=SPEARMAN["seed"])
    c2 = {**rho, "pass": rho["positive"], "test_set_identity": test_set_identity}

    c3 = {}
    for key in data.CONTROLS:
        c3[key] = {}
        for fam in ("tree", "gnn"):
            if fam in runs.get(key, {}):
                r = reading(key, fam)
                fires = bool(r["clears_low_band"] and r["null_check"]["passes"]
                             and r["loo_clears_low_band"])
                c3[key][fam] = {"I2": r["I2"], "Q": r["Q"], "k": r["k"],
                                "null_passes": r["null_check"]["passes"],
                                "loo_I2": r["leave_one_out"]["I2"],
                                "silent": not fires,
                                "fails_triple_on": [
                                    line for line, held in
                                    (("band", r["clears_low_band"]),
                                     ("null", r["null_check"]["passes"]),
                                     ("leave_one_out", r["loo_clears_low_band"]))
                                    if not held]}
    c3["complete"] = all(fam in c3.get(key, {}) for key in data.CONTROLS
                         for fam in ("tree", "gnn"))
    gated = [(key, fam) for key in data.CONTROLS for fam in ("tree", "gnn")
             if fam in c3.get(key, {})]

    firing = [[k, f] for k, f in gated if not c3[k][f]["silent"]]
    c3["firing"] = firing
    c3["pass"] = c3["complete"] and not firing

    criteria = {"1": c1, "2": c2, "3": c3}
    failing = sorted(c for c, held in criteria.items() if not held["pass"])
    outcome = "refuted" if failing else "supported"

    def three_tests(r: dict) -> dict:
        return {"I2": r["I2"], "Q": r["Q"], "k": r["k"], "band": r["band"],
                "null_passes": r["null_check"]["passes"],
                "loo_I2": r["leave_one_out"]["I2"],
                "clears_band": r["clears_low_band"],
                "loo_clears_band": r["loo_clears_low_band"],
                "fires_triple": bool(r["clears_low_band"] and r["null_check"]["passes"]
                                     and r["loo_clears_low_band"]),
                "weighted_mean": r["weighted_mean"]}

    same_world = {key: {fam: three_tests(reading(key, fam)) for fam in runs[key]}
                  for key in ("HI-Small",) if key in runs}
    sensitivity = {key: {fam: three_tests(reading(key, fam)) for fam in runs[key]}
                   for key in data.SENSITIVITY if key in runs}
    return {"criteria": criteria, "failing": failing, "hypothesis_1": outcome,
            "test_set_identity": identity,
            "robustness": {f"{key}/{fam}":
                           robustness(reading(key, fam), seed_noise_share(key, fam))
                           for key, fams in runs.items() for fam in fams},
            "null_stability": {f"{key}/{fam}": stability(key, fam)
                               for key in [P] + data.CONTROLS + ["HI-Small"]
                               for fam in runs.get(key, {})},
            "same_world_sensitivity": same_world,
            "sensitivity_eleven": sensitivity}


def cmd_verdict() -> int:
    done = completed(exp1.EXPERIMENT)
    runs: dict[str, dict[str, tuple[Path, dict]]] = {}
    for key in data.PRIMARY + data.SENSITIVITY:
        for fam in VERDICT_FAMILIES:
            found = shortfall_run(done, key, fam)
            if found is not None:
                runs.setdefault(key, {})[fam] = found
    needed = [(exp1.PRIMARY_KEY, "tree"), (exp1.PRIMARY_KEY, "gnn")] + \
             [(k, f) for k in data.CONTROLS for f in ("tree", "gnn")]
    missing = [f"{k}/{f}" for k, f in needed if f not in runs.get(k, {})]
    if missing:
        print(f"verdict: shortfall Runs missing for {missing}", flush=True)
        return 1
    cells = {k: {f: _hash_of(p) for f, (p, _) in fams.items()} for k, fams in runs.items()}
    metrics = run(exp1.EXPERIMENT, verdict_config(cells),
                  lambda config, run_dir: verdict_metrics(runs, run_dir), done=done)
    print(json.dumps({"hypothesis_1": metrics["hypothesis_1"], "failing": metrics["failing"],
                      "c1": {f: metrics["criteria"]["1"][f]["pass"] for f in ("tree", "gnn")},
                      "rho": metrics["criteria"]["2"]["rho"],
                      "rho_q05": metrics["criteria"]["2"]["boot_q05"],
                      "controls": metrics["criteria"]["3"]["pass"],
                      "per_seed": metrics["criteria"]["1"]["per_seed"]["agree"]},
                     indent=2))
    return 0


def cmd_scores() -> int:
    """F1 on the laundering class (times 100) and average precision on each
    dataset's test window at (network, all): the tree family and the
    blind tree at their configuration of highest validation average
    precision, the GNN family as the mean over its three seeds."""
    done = completed(exp1.EXPERIMENT)
    for key in SCORE_KEYS:
        cells = []
        for check in ("tree_global", "gnn_global", "blind_global"):
            if check not in exp1.dataset_checks(key):
                continue
            seeds = exp1.SEEDS if check == "gnn_global" else [-1]
            metrics = [exp1.recorded(done, exp1.Unit(check, key, seed=s)) for s in seeds]
            name = check.split("_")[0]
            if any(m is None for m in metrics):
                cells.append(f"{name} not run")
                continue
            f1 = np.mean([m["test_f1_x100"] for m in metrics])
            ap = np.mean([m["test_ap"] for m in metrics])
            cells.append(f"{name} F1 {f1:.1f} AP {ap:.2f}")
        print(f"{key:18s} " + "  ".join(cells))
    return 0


def cmd_all() -> int:
    done = completed(exp1.EXPERIMENT)
    for key in data.PRIMARY + data.SENSITIVITY:
        if not data.ledgers_ready(key):
            continue
        for fam in exp1.shortfall_families(key):
            cmd_shortfall(key, fam, done)
    return 0


def cmd_status() -> int:
    done = completed(exp1.EXPERIMENT)
    for key in data.PRIMARY + data.SENSITIVITY:
        if not data.ledgers_ready(key):
            print(f"{key}: ledgers not built")
            continue
        for fam in exp1.shortfall_families(key):
            dirs = unit_dirs(done, key, fam)
            found = shortfall_run(done, key, fam) if dirs else None
            state = "analysed" if found else ("inputs complete" if dirs else "inputs incomplete")
            print(f"{key:36s} {fam:6s} {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "shortfall":
        return 0 if cmd_shortfall(args[0], args[1]) is not None else 1
    if cmd == "all":
        return cmd_all()
    if cmd == "verdict":
        return cmd_verdict()
    if cmd == "scores":
        return cmd_scores()
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
