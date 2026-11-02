#!/usr/bin/env python
"""Experiment 1's analysis Runs: one ``shortfall`` Run per (dataset, family) over the unit Runs,
and one ``verdict`` Run on hypothesis 1 over the shortfall Runs. Both are pure
functions over the Run log; an analysis config lists its inputs' config
hashes, so a re-done input re-keys the analysis. Hypothesis 1 is judged in
each family on HI-Medium on the null check alone; the verdict Run also holds
the readings beside it: the null check at each GNN seed, the agreement
between the families, the controls, the factor on the error bars and the
null check with the draws for the variation between seeds made afresh.

  uv run python scripts/exp1_analysis.py shortfall <key> <family>
  uv run python scripts/exp1_analysis.py all          # every complete cell
  uv run python scripts/exp1_analysis.py verdict
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
I2_THRESHOLD = 0.25
SPEARMAN = {"draws": 1000, "seed": 0}
PAIRING = {"tree": "median_over_grid", "blind": "median_over_grid",
           "gnn": "mean_over_seeds"}
AGGREGATE = {"median_over_grid": "median", "mean_over_seeds": "mean"}
VERDICT_FAMILIES = ("tree", "gnn")

STABILITY_DRAWS = 200
STABILITY_CELLS = ((exp1.PRIMARY_KEY, "gnn"), ("transxion", "gnn"),
                   (data.AMLGENTEX_DATASET, "gnn"))
ROBUSTNESS_KEYS = (exp1.PRIMARY_KEY, "HI-Small")
CRITERION = ("hypothesis 1 is judged in each family on HI-Medium on the null "
             "check alone: supported when the observed Q exceeds the 95th "
             "percentile of Q over the 1,000 centred replicates (GNN: plus "
             "normal seed noise of variance seed_var/3), not supported otherwise")


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
              "i2_test": I2_THRESHOLD,
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

def heterogeneity_reading(labels, deltas, ses, delta_reps, seed_var=None) -> dict:
    het = heterogeneity(deltas, ses)
    null = null_q_check(delta_reps, deltas, ses, seed_var, seed=SEED, level=NULL_LEVEL)
    loo = leave_one_out(labels, deltas, ses)
    return {**het, "null_check": null, "leave_one_out": loo,
            "passes_i2_test": bool(het["I2"] >= I2_THRESHOLD),
            "passes_leave_one_out": bool(loo["I2"] >= I2_THRESHOLD)}


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
                                     "passes_i2_test": r["passes_i2_test"],
                                     "null_passes": r["null_check"]["passes"],
                                     "passes_leave_one_out": r["passes_leave_one_out"],
                                     "weighted_mean": r["weighted_mean"]})
                pooled = readings["ap"]
                verdict = lambda c, n, o: (bool(c), bool(n), bool(o))  # noqa: E731
                pooled_verdict = verdict(pooled["passes_i2_test"], pooled["null_check"]["passes"],
                                         pooled["passes_leave_one_out"])
                for p in per_seed:
                    p["verdict"] = list(verdict(p["passes_i2_test"], p["null_passes"],
                                                p["passes_leave_one_out"]))
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
            "common_rate": readings["ap_common"],
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
            "hypothesis": 1,
            "primary_dataset": exp1.PRIMARY_KEY, "controls": data.CONTROLS,
            "families": list(VERDICT_FAMILIES),
            "criterion": CRITERION,
            "outcomes": {"supported": "the null check passes in the family",
                         "not supported": "it fails"},
            "readings": {
                "per_seed": "the GNN family's null check with the shortfalls of "
                            "each seed in turn",
                "agreement": f"{exp1.PRIMARY_KEY} tree-vs-GNN Spearman rho with "
                             "its bank-bootstrap 5th percentile and interval",
                "controls": "the null check, I2 and leave-one-out on TransXion and "
                            "the AMLgentex dataset"},
            "spearman": SPEARMAN,
            "robustness": {"datasets": list(ROBUSTNESS_KEYS),
                           "factor": "the largest common factor c by which every "
                                     "error bar can widen before the null check "
                                     "fails: c = sqrt(Q / null_q95)"},
            "null_stability": {"cells": [f"{k}/{f}" for k, f in STABILITY_CELLS],
                               "draws": STABILITY_DRAWS, "seeds": "0 to "
                               f"{STABILITY_DRAWS - 1}",
                               "note": "the null check's draws for the variation "
                                       "between seeds made afresh"},
            "inputs": {k: {f: h for f, h in fams.items()} for k, fams in cells.items()}}


def robustness(reading: dict, seed_noise_share: float | None = None) -> dict:
    """The largest common factor c by which every error bar can widen before
    the null check fails. Widening every s_B by c divides Q by c-squared while
    the null check's 95th percentile holds still, its resampled shortfalls
    widening by c as their weights fall by c-squared.

    ``seed_noise_share`` is the median share of s_B-squared that comes from
    the variation between seeds (the GNN family), whose part of s_B does not
    come from the bootstrap."""
    q95 = reading["null_check"]["null_q95"]
    return {"c_null": math.sqrt(reading["Q"] / q95),
            "Q": reading["Q"], "k": reading["k"], "null_q95": q95,
            "seed_noise_share_median": seed_noise_share}


def null_stability(reading: dict, reps: np.ndarray, deltas: np.ndarray,
                   ses: np.ndarray, seed_var: np.ndarray) -> dict:
    """How much the GNN family's null-check 95th percentile moves when the
    draws for the variation between seeds are made afresh ``STABILITY_DRAWS``
    times, and how often the check passes. The recomputation reads the
    persisted float32 replicates, and its seed-0 reading is reported against
    the recorded one."""
    q95 = np.array([null_q_check(reps, deltas, ses, seed_var, seed=seed,
                                 level=NULL_LEVEL)["null_q95"]
                    for seed in range(STABILITY_DRAWS)])
    observed = reading["Q"]
    return {"draws": int(len(q95)),
            "null_q95_min": float(q95.min()), "null_q95_median": float(np.median(q95)),
            "null_q95_max": float(q95.max()),
            "passes": int((observed > q95).sum()),
            "passes_share": float((observed > q95).mean()),
            "recorded_null_q95": reading["null_check"]["null_q95"],
            "recorded_passes": reading["null_check"]["passes"],
            "seed0_difference": float(abs(q95[0] - reading["null_check"]["null_q95"]))}


def null_reading(r: dict) -> dict:
    return {"I2": r["I2"], "Q": r["Q"], "k": r["k"],
            "null_q95": r["null_check"]["null_q95"],
            "null_passes": r["null_check"]["passes"],
            "loo_I2": r["leave_one_out"]["I2"],
            "weighted_mean": r["weighted_mean"]}


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
        return null_stability(reading(key, fam), reps, rows["delta"].to_numpy(),
                              rows["se"].to_numpy(), rows["seed_var"].to_numpy())

    identity = {}
    for key, fams in runs.items():
        shas = {fam: table(key, fam)["test_payment_sha256"] for fam in fams}
        first = next(iter(shas.values()))
        identity[key] = all(s.index.equals(first.index) and (s == first).all()
                            for s in shas.values())
    assert all(identity.values()), \
        f"families scored different test sets on {[k for k, v in identity.items() if not v]}"

    families = {fam: null_reading(reading(P, fam)) for fam in VERDICT_FAMILIES}
    outcome = {fam: "supported" if r["null_passes"] else "not supported"
               for fam, r in families.items()}
    g = reading(P, "gnn")
    per_seed = [{"seed": p["seed"], "I2": p["I2"], "Q": p["Q"],
                 "null_passes": p["null_passes"]} for p in g["per_seed"]]

    tt, tg = table(P, "tree"), table(P, "gnn")
    banks = tt.index.intersection(tg.index)
    assert len(banks) == len(tt) == len(tg), \
        "the tree and GNN families scored different banks"
    agreement = spearman_bank_bootstrap(tt.loc[banks, "delta"].to_numpy(),
                                        tg.loc[banks, "delta"].to_numpy(),
                                        draws=SPEARMAN["draws"], seed=SPEARMAN["seed"])

    controls = {key: {fam: null_reading(reading(key, fam)) for fam in VERDICT_FAMILIES}
                for key in data.CONTROLS}
    same_world = {key: {fam: null_reading(reading(key, fam)) for fam in runs[key]}
                  for key in ("HI-Small",) if key in runs}
    return {"families": families, "hypothesis_1": outcome,
            "per_seed": per_seed, "agreement": agreement, "controls": controls,
            "test_set_identity": identity,
            "robustness": {f"{key}/{fam}":
                           robustness(reading(key, fam), seed_noise_share(key, fam))
                           for key in ROBUSTNESS_KEYS for fam in runs.get(key, {})},
            "null_stability": {f"{key}/{fam}": stability(key, fam)
                               for key, fam in STABILITY_CELLS},
            "same_world_sensitivity": same_world}


def cmd_verdict() -> int:
    done = completed(exp1.EXPERIMENT)
    runs: dict[str, dict[str, tuple[Path, dict]]] = {}
    for key in data.PRIMARY:
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
    print(json.dumps({"hypothesis_1": metrics["hypothesis_1"],
                      "null_passes": {f: r["null_passes"]
                                      for f, r in metrics["families"].items()},
                      "rho": metrics["agreement"]["rho"],
                      "rho_q05": metrics["agreement"]["boot_q05"]},
                     indent=2))
    return 0


def cmd_all() -> int:
    done = completed(exp1.EXPERIMENT)
    missing = 0
    for key in data.SHORTFALL_KEYS:
        if not data.ledgers_ready(key):
            print(f"{key}: ledgers not built", flush=True)
            missing += 1
            continue
        for fam in exp1.shortfall_families(key):
            missing += cmd_shortfall(key, fam, done) is None
    return 1 if missing else 0


def cmd_status() -> int:
    done = completed(exp1.EXPERIMENT)
    for key in data.SHORTFALL_KEYS:
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
    if sys.argv[1] == "status":
        return cmd_status()
    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "shortfall":
        return 0 if cmd_shortfall(args[0], args[1]) is not None else 1
    if cmd == "all":
        return cmd_all()
    if cmd == "verdict":
        return cmd_verdict()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
