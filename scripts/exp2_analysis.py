#!/usr/bin/env python
"""Experiment 2's analysis Runs (Chapter 4; Appendix C).

One ``components`` Run per detector (tree family, GNN family, blind tree)
reads the decomposition of the shortfall along path 1 from four scenarios --
(ledger, own) and (network, all) from Experiment 1, (ledger, qualifying) and
(network, qualifying) from Experiment 2 -- through one seeded set of
joint-bootstrap draws of the test window, so every difference is paired. One
``secondpath`` Run per detector reads path 2 through the grid, by way of
(network, own). The ``sightcheck`` Run reads the GNN family's value of sight
on path 1 with the count held equal (``gnn_cglobaldup`` in place of
(network, qualifying)). One ``dial`` Run per family reads the case reduction
against Experiment 1's solo Runs at the full count. ``all`` runs every one of
them.

  uv run python scripts/exp2_analysis.py components <family>
  uv run python scripts/exp2_analysis.py secondpath <family>
  uv run python scripts/exp2_analysis.py sightcheck
  uv run python scripts/exp2_analysis.py dial <family>
  uv run python scripts/exp2_analysis.py all
  uv run python scripts/exp2_analysis.py status
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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1  # noqa: E402
import exp1_analysis  # noqa: E402
import exp1_data as data  # noqa: E402
import exp2  # noqa: E402
from extract import load_column  # noqa: E402
from m import completed, run  # noqa: E402
from m.artifacts import load_array, verify, write_array, write_csv  # noqa: E402
from m.runlog import config_hash  # noqa: E402
from m.stats import weighted_average_precision  # noqa: E402

KEY = exp2.KEY
REPLICATES = 1000
CHUNK = 100
SEED = 0
SIDES = ("solo", "clocal", "cglobal", "global")
COMPONENTS = {"examples": ("clocal", "solo"), "sight": ("cglobal", "clocal"),
              "pool_extension": ("global", "cglobal"), "total": ("global", "solo")}
COMPONENT_ORDER = ["examples", "sight", "pool_extension", "total"]
PAIRING = {"tree": "median_over_grid", "blind": "median_over_grid",
           "gnn": "mean_over_seeds"}

GNN_INTERVAL_NOTE = ("between-seed variance folded into the weighted-mean "
                     "interval: each replicate's per-bank component is "
                     "perturbed by an independent normal draw of variance "
                     "seed_var/3 (default_rng(0), components in fixed "
                     "order), as Experiment 1's assumption-free null does")

DIAL_RANGE = {
    "tree": "band_all is the lowest and highest per-setting test AP over the "
            "eight settings at the full count; ap_all is their median",
    "gnn": "band_all is the lowest and highest test AP of Experiment 1's "
           "three solo seeds at the full count; ap_all is the seed-0 Run's, "
           "the case reduction's fixed seed",
}


# --------------------------------------------------------------- inputs --

def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def _metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


def _scores(path: Path, name: str) -> np.ndarray:
    metrics = _metrics(path)
    return load_array(path, name, metrics["artifacts"][name]["sha256"])


def exp2_dir(done2: dict, unit: exp2.Unit, roster: pd.DataFrame) -> Path | None:
    return done2.get(config_hash(exp2.unit_config(unit, roster)))


def cell_dirs(done1: dict, done2: dict, family: str) -> dict | None:
    """Run directories of one family's four cells, or None while incomplete."""
    dirs = exp1_analysis.unit_dirs(done1, KEY, family)
    if dirs is None:
        return None
    roster = dirs["roster"]
    if family == "gnn":
        clocal = [exp2_dir(done2, exp2.Unit("gnn_clocal", KEY, seed=s), roster)
                  for s in exp2.SEEDS]
        cglobal = [exp2_dir(done2, exp2.Unit("gnn_cglobal", KEY, seed=s), roster)
                   for s in exp2.SEEDS]
    else:
        clocal = [exp2_dir(done2, exp2.Unit(f"{family}_clocal", KEY), roster)]
        cglobal = [exp2_dir(done2, exp2.Unit(f"{family}_cglobal", KEY), roster)]
    if any(d is None for d in clocal + cglobal):
        return None
    return dirs | {"clocal": clocal, "cglobal": cglobal}


def weights_run(done1: dict, family: str) -> tuple[Path, dict]:
    found = exp1_analysis.shortfall_run(done1, KEY, family)
    if found is None:
        raise SystemExit(f"Experiment 1 shortfall Run for {KEY}/{family} not found; "
                         f"the weights come from it")
    return found


def load_four_cells(family: str, dirs: dict) -> dict:
    """Per bank: test-window positions, labels, and (settings, n_B) scores
    for each of the four cells, every identity asserted."""
    split = load_column(KEY, "split")
    y_all = load_column(KEY, "is_laundering")
    test_rows = np.flatnonzero(split == 2)
    n_test = len(test_rows)
    position = np.full(len(split), -1, np.int64)
    position[test_rows] = np.arange(n_test)

    if family == "gnn":
        glob = np.stack([_scores(d, "test_probs") for d in dirs["global"]])
        cl_scores = np.stack([_scores(d, "test_probs") for d in dirs["clocal"]])
        cg_scores = np.stack([_scores(d, "test_probs") for d in dirs["cglobal"]])
        cl_payment = _scores(dirs["clocal"][0], "test_payment")
        cl_bank = _scores(dirs["clocal"][0], "test_bank_code")
        cg_payment = _scores(dirs["cglobal"][0], "test_payment")
        for d in dirs["clocal"][1:]:
            assert np.array_equal(_scores(d, "test_payment"), cl_payment)
        for d in dirs["cglobal"][1:]:
            assert np.array_equal(_scores(d, "test_payment"), cg_payment)
    else:
        glob = _scores(dirs["global"][0], "test_scores")
        cl_scores = _scores(dirs["clocal"][0], "test_scores")
        cg_scores = _scores(dirs["cglobal"][0], "test_scores")
        cl_payment = _scores(dirs["clocal"][0], "test_payment")
        cl_bank = _scores(dirs["clocal"][0], "test_bank_code")
        cg_payment = _scores(dirs["cglobal"][0], "test_payment")
    assert glob.shape[1] == n_test
    assert np.all(np.diff(cg_payment) > 0)

    roster = dirs["roster"]
    groups, meta = {}, {}
    for bank, ds in dirs["solo"].items():
        payments = [_scores(d, "test_payment") for d in ds]
        for p in payments[1:]:
            assert np.array_equal(p, payments[0])
        payment = payments[0]
        code = int(roster.loc[bank, "code"])
        mask = cl_bank == code
        assert np.array_equal(cl_payment[mask], payment), \
            f"{bank}: (ledger, qualifying) scored a different paper than (ledger, own)"
        cg_idx = np.searchsorted(cg_payment, payment)
        assert np.array_equal(cg_payment[cg_idx], payment), \
            f"{bank}: a payment is missing from (network, qualifying)'s paper"
        idx = position[payment]
        assert np.all(idx >= 0)
        if family == "gnn":
            solo = np.stack([_scores(d, "test_probs") for d in ds])
        else:
            solo = _scores(ds[0], "test_scores")
        y = y_all[payment].astype(np.int8)
        assert len(payment) == int(roster.loc[bank, "test_rows"])
        assert int(y.sum()) == int(roster.loc[bank, "test_pos"])
        groups[bank] = {"idx": idx, "y": y, "solo": solo,
                        "clocal": cl_scores[:, mask],
                        "cglobal": cg_scores[:, cg_idx],
                        "global": glob[:, idx]}
        meta[bank] = {"test_rows": int(len(y)), "test_pos": int(y.sum())}
    return {"groups": groups, "meta": meta, "n_test": n_test}


# ------------------------------------------------------------ bootstrap --

def four_cell_bootstrap(n_test: int, groups: dict,
                        sides: tuple[str, ...] = SIDES) -> dict:
    """Point APs and joint-bootstrap AP replicates for every cell in
    ``sides`` from one seeded set of multiplicity draws (the draws Experiment 1's
    analysis used: seed 0, 1,000 replicates, chunks of 100); the draws do
    not depend on which cells are read."""
    labels = list(groups)
    n_settings = next(iter(groups.values()))["solo"].shape[0]
    point = {g: {side: np.array([
        weighted_average_precision(groups[g]["y"], groups[g][side][s],
                                   np.ones((1, len(groups[g]["y"]))))[0]
        for s in range(n_settings)]) for side in sides} for g in labels}
    reps = {g: {side: np.empty((REPLICATES, n_settings)) for side in sides}
            for g in labels}
    rng = np.random.default_rng(SEED)
    done = 0
    while done < REPLICATES:
        r = min(CHUNK, REPLICATES - done)
        multiplicity = np.zeros((r, n_test), np.float32)
        for j in range(r):
            draw = rng.integers(0, n_test, n_test)
            multiplicity[j] = np.bincount(draw, minlength=n_test)
        for g in labels:
            spec = groups[g]
            w = multiplicity[:, spec["idx"]]
            for side in sides:
                for s in range(n_settings):
                    reps[g][side][done:done + r, s] = weighted_average_precision(
                        spec["y"], spec[side][s], w)
        done += r
    return {"point": point, "reps": reps, "labels": labels,
            "n_settings": n_settings}


def component_estimates(boot: dict, family: str) -> dict:
    """Per bank and component: the point estimate (aggregated over settings
    or seeds), its replicate distribution, the bootstrap standard error, and
    for the GNN family the between-seed variance of the per-seed points."""
    reduce = np.nanmedian if PAIRING[family] == "median_over_grid" else np.nanmean
    out = {}
    for g in boot["labels"]:
        rec = {}
        for name, (hi, lo) in COMPONENTS.items():
            per_setting = boot["point"][g][hi] - boot["point"][g][lo]
            delta_reps = reduce(boot["reps"][g][hi] - boot["reps"][g][lo], axis=1)
            valid = np.isfinite(delta_reps)
            entry = {"delta": float(reduce(per_setting)),
                     "per_setting": per_setting.tolist(),
                     "delta_reps": delta_reps,
                     "se_boot": float(np.std(delta_reps[valid], ddof=1)),
                     "ci95": [float(v) for v in
                              np.percentile(delta_reps[valid], [2.5, 97.5])]}
            if family == "gnn":
                entry["seed_var"] = float(np.var(per_setting, ddof=1))
                entry["se"] = float(np.sqrt(entry["se_boot"] ** 2
                                            + entry["seed_var"] / len(exp2.SEEDS)))
            else:
                entry["se"] = entry["se_boot"]
            rec[name] = entry
        rec["levels"] = {side: [float(v) for v in boot["point"][g][side]]
                         for side in SIDES}
        out[g] = rec
    return out


def weighted_mean_reading(labels: list[str], est: dict, weights: np.ndarray,
                          component: str, family: str,
                          rng: np.random.Generator) -> dict:
    """The roster-weighted mean of one component under Experiment 1's weights,
    with its interval from the shared draws (GNN: seed noise folded in)."""
    deltas = np.array([est[g][component]["delta"] for g in labels])
    reps = np.stack([est[g][component]["delta_reps"] for g in labels], axis=1)
    if family == "gnn":
        seed_var = np.array([est[g][component]["seed_var"] for g in labels])
        reps = reps + rng.standard_normal(reps.shape) * np.sqrt(seed_var / 3.0)[None, :]
    wsum = weights.sum()
    point = float((weights * deltas).sum() / wsum)
    finite = np.all(np.isfinite(reps), axis=1)
    wmean_reps = (reps[finite] * weights).sum(axis=1) / wsum
    lo, hi = np.percentile(wmean_reps, [2.5, 97.5])
    return {"weighted_mean": point, "ci95": [float(lo), float(hi)],
            "se": float(np.std(wmean_reps, ddof=1)),
            "replicates_valid": int(finite.sum())}


# ----------------------------------------------------------- components --

def components_config(family: str, dirs: dict, wpath: Path) -> dict:
    world, dataset = data.DATASETS[KEY]
    config = {"experiment": exp2.EXPERIMENT, "check": "components", "world": world,
              "dataset": dataset, "dataset_key": KEY, "family": family,
              "cells": list(SIDES),
              "components": {k: f"{hi} minus {lo}"
                             for k, (hi, lo) in COMPONENTS.items()},
              "pairing": PAIRING[family], "replicates": REPLICATES,
              "seed": SEED, "chunk": CHUNK,
              "weights": "Experiment 1's per-bank inverse-variance weights, 1/s_B^2 "
                         "from the Experiment 1 shortfall Run's banks.csv",
              "weights_run": _hash_of(wpath),
              "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(KEY)},
              "inputs": {
                  "solo": {b: [_hash_of(d) for d in ds]
                           for b, ds in dirs["solo"].items()},
                  "global": [_hash_of(d) for d in dirs["global"]],
                  "clocal": [_hash_of(d) for d in dirs["clocal"]],
                  "cglobal": [_hash_of(d) for d in dirs["cglobal"]]},
              "features_version": exp1.FEATURES_VERSION}
    if family == "gnn":
        config["gnn_version"] = exp1.GNN_VERSION
        config["seeds"] = exp2.SEEDS
        config["interval"] = GNN_INTERVAL_NOTE
    if family == "blind":
        config["placebo"] = ("the blind tree reads only a payment's own fields, "
                             "so its scores at (ledger, qualifying) and "
                             "(network, qualifying) must be bit-identical and "
                             "its value of sight exactly zero")
    return config


def components_metrics(family: str, dirs: dict, wpath: Path, wmetrics: dict,
                       run_dir: Path) -> dict:
    t0 = time.time()
    cell = load_four_cells(family, dirs)
    groups = cell["groups"]
    labels = list(groups)

    banks_csv = pd.read_csv(verify(wpath, wmetrics["artifacts"]["banks"]),
                            dtype={"bank": str}).set_index("bank")
    assert set(labels) == set(banks_csv.index)
    se_exp1 = banks_csv.loc[labels, "se"].to_numpy()
    weights = 1.0 / se_exp1 ** 2

    placebo = None
    if family == "blind":
        diffs = [float(np.max(np.abs(groups[g]["cglobal"] - groups[g]["clocal"])))
                 for g in labels]
        placebo = {"bit_identical": bool(all(d == 0.0 for d in diffs)),
                   "max_abs_diff": max(diffs)}
        assert placebo["bit_identical"], \
            ("the placebo failed: the blind tree's scores at (ledger, qualifying) "
             "and (network, qualifying) differ")

    boot = four_cell_bootstrap(cell["n_test"], groups)
    est = component_estimates(boot, family)
    t_boot = time.time() - t0

    rng = np.random.default_rng(SEED)
    readings = {name: weighted_mean_reading(labels, est, weights, name, family, rng)
                for name in COMPONENT_ORDER}

    table = pd.DataFrame.from_dict(cell["meta"], orient="index").loc[labels]
    table.index.name = "bank"
    table["weight_se_exp1"] = se_exp1
    for side in SIDES:
        agg = np.median if PAIRING[family] == "median_over_grid" else np.mean
        table[f"{side}_ap"] = [agg(est[g]["levels"][side]) for g in labels]
    for name in COMPONENT_ORDER:
        table[f"{name}_delta"] = [est[g][name]["delta"] for g in labels]
        table[f"{name}_se"] = [est[g][name]["se"] for g in labels]
        table[f"{name}_ci_lo"] = [est[g][name]["ci95"][0] for g in labels]
        table[f"{name}_ci_hi"] = [est[g][name]["ci95"][1] for g in labels]
        per = np.stack([est[g][name]["per_setting"] for g in labels])
        for s in range(per.shape[1]):
            table[f"{name}_s{s}"] = per[:, s]
    replicates = np.stack([np.stack([est[g][name]["delta_reps"]
                                     for g in labels], axis=1)
                           for name in COMPONENT_ORDER], axis=2)
    artifacts = {"banks": write_csv(run_dir, "banks", table),
                 "roster": write_csv(run_dir, "roster", dirs["roster"]),
                 "replicates": write_array(run_dir, "replicates",
                                           replicates.astype(np.float32))}
    out = {"dataset_key": KEY, "family": family, "k": len(labels),
           "n_test": int(cell["n_test"]), "pairing": PAIRING[family],
           "replicates": REPLICATES, "component_order": COMPONENT_ORDER,
           "readings": readings,
           "seconds": {"bootstrap": t_boot, "total": time.time() - t0},
           "artifacts": artifacts}
    if placebo is not None:
        out["placebo"] = placebo
    return out


def cmd_components(family: str, done1: dict | None = None,
                   done2: dict | None = None) -> dict | None:
    done1 = done1 if done1 is not None else completed(exp1.EXPERIMENT)
    done2 = done2 if done2 is not None else completed(exp2.EXPERIMENT)
    dirs = cell_dirs(done1, done2, family)
    if dirs is None:
        print(f"components {family}: inputs incomplete", flush=True)
        return None
    wpath, wmetrics = weights_run(done1, family)
    metrics = run(exp2.EXPERIMENT, components_config(family, dirs, wpath),
                  lambda config, run_dir: components_metrics(
                      family, dirs, wpath, wmetrics, run_dir),
                  done=done2)
    for name in COMPONENT_ORDER:
        r = metrics["readings"][name]
        print(f"components {family} {name}: mean {r['weighted_mean']:+.4f} "
              f"ci [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]", flush=True)
    return metrics


def components_run(done1: dict, done2: dict, family: str) -> tuple[Path, dict] | None:
    dirs = cell_dirs(done1, done2, family)
    if dirs is None:
        return None
    wpath, _ = weights_run(done1, family)
    path = done2.get(config_hash(components_config(family, dirs, wpath)))
    if path is None:
        return None
    return path, _metrics(path)


# ------------------------------------------------------------ sightcheck --

def sightcheck_dirs(done1: dict, done2: dict) -> dict | None:
    """The GNN family's four cells with (network, qualifying), each payment
    fitted as often as under ledger sight, in place of (network, qualifying)."""
    dirs = cell_dirs(done1, done2, "gnn")
    if dirs is None:
        return None
    dup = [exp2_dir(done2, exp2.Unit("gnn_cglobaldup", KEY, seed=s), dirs["roster"])
           for s in exp2.SEEDS]
    if any(d is None for d in dup):
        return None
    return dirs | {"cglobal": dup, "cglobal_original": dirs["cglobal"]}


def sightcheck_config(dirs: dict, wpath: Path, original: Path) -> dict:
    config = components_config("gnn", dirs, wpath)
    config["check"] = "sightcheck"
    config["cglobal_cell"] = ("gnn_cglobaldup: (network, qualifying) with each "
                              "payment fitted once per qualifying bank it "
                              "touches, as under ledger sight")
    config["original_components_run"] = _hash_of(original)
    config["paired_difference"] = ("copy-weighted sight minus original sight, "
                                   "per replicate on the shared draws, Experiment 1's "
                                   "weights, no seed-noise draw")
    return config


def sightcheck_metrics(dirs: dict, wpath: Path, wmetrics: dict, original: Path,
                       run_dir: Path) -> dict:
    out = components_metrics("gnn", dirs, wpath, wmetrics, run_dir)
    order = COMPONENT_ORDER.index("sight")
    ometrics = _metrics(original)
    orep = load_array(original, "replicates",
                      ometrics["artifacts"]["replicates"]["sha256"])[:, :, order]
    nrep = load_array(run_dir, "replicates",
                      out["artifacts"]["replicates"]["sha256"])[:, :, order]
    obanks = pd.read_csv(verify(original, ometrics["artifacts"]["banks"]),
                         dtype={"bank": str}).set_index("bank")
    nbanks = pd.read_csv(verify(run_dir, out["artifacts"]["banks"]),
                         dtype={"bank": str}).set_index("bank")
    assert list(obanks.index) == list(nbanks.index)
    w = 1.0 / obanks["weight_se_exp1"].to_numpy() ** 2
    diff_point = float((w * (nbanks["sight_delta"] - obanks["sight_delta"])).sum()
                       / w.sum())
    finite = np.all(np.isfinite(orep) & np.isfinite(nrep), axis=1)
    diff_reps = ((nrep[finite] - orep[finite]) * w).sum(axis=1) / w.sum()
    lo, hi = np.percentile(diff_reps, [2.5, 97.5])
    per_bank = (nbanks["sight_delta"] - obanks["sight_delta"]).to_numpy()
    out["original_sight"] = ometrics["readings"]["sight"]
    out["copy_weighted_minus_original"] = {
        "weighted_mean": diff_point, "ci95": [float(lo), float(hi)],
        "replicates_valid": int(finite.sum()),
        "per_bank_median": float(np.median(per_bank)),
        "per_bank_range": [float(per_bank.min()), float(per_bank.max())]}
    return out


def cmd_sightcheck() -> dict | None:
    done1 = completed(exp1.EXPERIMENT)
    done2 = completed(exp2.EXPERIMENT)
    dirs = sightcheck_dirs(done1, done2)
    found = components_run(done1, done2, "gnn")
    if dirs is None or found is None:
        print("sightcheck: inputs incomplete", flush=True)
        return None
    original, _ = found
    wpath, wmetrics = weights_run(done1, "gnn")
    metrics = run(exp2.EXPERIMENT, sightcheck_config(dirs, wpath, original),
                  lambda config, run_dir: sightcheck_metrics(
                      dirs, wpath, wmetrics, original, run_dir),
                  done=done2)
    r, d = metrics["readings"]["sight"], metrics["copy_weighted_minus_original"]
    o = metrics["original_sight"]
    print(f"sightcheck: copy-weighted sight {r['weighted_mean']:+.4f} "
          f"[{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]; original "
          f"{o['weighted_mean']:+.4f} [{o['ci95'][0]:+.4f}, {o['ci95'][1]:+.4f}]; "
          f"difference {d['weighted_mean']:+.4f} [{d['ci95'][0]:+.4f}, "
          f"{d['ci95'][1]:+.4f}]", flush=True)
    return metrics


# ------------------------------------------------------------ secondpath --

# Each quantity is a signed sum of cells, read per setting (tree) or per seed
# (GNN) and then reduced as the four components are.
SECOND_PATH = {
    "sight_own": (("nown", 1), ("solo", -1)),
    "examples_network": (("cglobal", 1), ("nown", -1)),
    "interaction": (("cglobal", 1), ("clocal", -1), ("nown", -1), ("solo", 1)),
    "examples_over_sight_path1": (("clocal", 2), ("solo", -1), ("cglobal", -1)),
    "examples_over_sight_path2": (("cglobal", 1), ("nown", -2), ("solo", 1)),
}
EXAMPLES_OVER_SIGHT = {"path1": "examples_over_sight_path1",
                       "path2": "examples_over_sight_path2"}
COPY_WEIGHTED = "_copyweighted"


def second_path_combinations(family: str) -> dict:
    """The GNN family also reads every quantity holding (network,
    qualifying) with each payment fitted as often as under ledger sight:
    every statement of its value of sight states both."""
    if family != "gnn":
        return SECOND_PATH
    swapped = {name + COPY_WEIGHTED: tuple(("cglobaldup" if c == "cglobal" else c, k)
                                           for c, k in terms)
               for name, terms in SECOND_PATH.items()
               if any(c == "cglobal" for c, _ in terms)}
    return SECOND_PATH | swapped


def second_path_sides(family: str) -> tuple[str, ...]:
    return SIDES + (("nown", "cglobaldup") if family == "gnn" else ("nown",))


def second_path_dirs(done1: dict, done2: dict, family: str) -> dict | None:
    """The four cells plus the own-pool network-sight Runs per
    bank (and, for the GNN family, (network, qualifying) with each payment
    fitted as often as under ledger sight)."""
    dirs = cell_dirs(done1, done2, family)
    if dirs is None:
        return None
    roster = dirs["roster"]
    seeds = exp2.SEEDS if family == "gnn" else [-1]
    nown = {b: [exp2_dir(done2, exp2.Unit(f"{family}_nown", KEY, b, seed=s), roster)
                for s in seeds] for b in dirs["solo"]}
    if any(d is None for ds in nown.values() for d in ds):
        return None
    out = dirs | {"nown": nown}
    if family == "gnn":
        dup = [exp2_dir(done2, exp2.Unit("gnn_cglobaldup", KEY, seed=s), roster)
               for s in exp2.SEEDS]
        if any(d is None for d in dup):
            return None
        out["cglobaldup"] = dup
    return out


def add_second_path_cells(family: str, dirs: dict, cell: dict) -> dict:
    """The own-pool network-sight scores per bank, each Run's test payments
    asserted equal to (ledger, own)'s, in its order; for the GNN family the
    scores of (network, qualifying), each payment fitted as often as under
    ledger sight, on the same payments."""
    score = "test_probs" if family == "gnn" else "test_scores"
    if family == "gnn":
        dup_scores = np.stack([_scores(d, "test_probs") for d in dirs["cglobaldup"]])
        dup_payment = _scores(dirs["cglobaldup"][0], "test_payment")
        for d in dirs["cglobaldup"][1:]:
            assert np.array_equal(_scores(d, "test_payment"), dup_payment)
    groups = {}
    for bank, ds in dirs["nown"].items():
        payment = _scores(dirs["solo"][bank][0], "test_payment")
        for d in ds:
            assert np.array_equal(_scores(d, "test_payment"), payment), \
                f"{bank}: the own-pool network-sight cell scored a different paper"
        scores = np.stack([_scores(d, score) for d in ds]) if family == "gnn" \
            else _scores(ds[0], score)
        extra = {"nown": scores}
        if family == "gnn":
            idx = np.searchsorted(dup_payment, payment)
            assert np.array_equal(dup_payment[idx], payment)
            extra["cglobaldup"] = dup_scores[:, idx]
        groups[bank] = cell["groups"][bank] | extra
    return cell | {"groups": groups}


def combination_estimates(boot: dict, family: str, combos: dict) -> dict:
    """``component_estimates`` for signed sums of cells."""
    reduce = np.nanmedian if PAIRING[family] == "median_over_grid" else np.nanmean
    out = {}
    for g in boot["labels"]:
        rec = {}
        for name, terms in combos.items():
            per_setting = sum(k * boot["point"][g][c] for c, k in terms)
            delta_reps = reduce(sum(k * boot["reps"][g][c] for c, k in terms), axis=1)
            valid = np.isfinite(delta_reps)
            entry = {"delta": float(reduce(per_setting)),
                     "per_setting": per_setting.tolist(),
                     "delta_reps": delta_reps,
                     "se_boot": float(np.std(delta_reps[valid], ddof=1)),
                     "ci95": [float(v) for v in
                              np.percentile(delta_reps[valid], [2.5, 97.5])]}
            if family == "gnn":
                entry["seed_var"] = float(np.var(per_setting, ddof=1))
                entry["se"] = float(np.sqrt(entry["se_boot"] ** 2
                                            + entry["seed_var"] / len(exp2.SEEDS)))
            else:
                entry["se"] = entry["se_boot"]
            rec[name] = entry
        out[g] = rec
    return out


def second_path_config(family: str, dirs: dict, wpath: Path,
                       components: Path) -> dict:
    config = components_config(family, dirs, wpath)
    combos = second_path_combinations(family)
    config |= {
        "check": "secondpath", "cells": list(second_path_sides(family)),
        "quantities": {name: " ".join(f"{k:+d} {c}" for c, k in terms)
                       for name, terms in combos.items()},
        "examples_exceed_sight": ("training examples outweigh sight on a path "
                                  "when the value of training examples less "
                                  "the value of sight has its 95 % interval "
                                  "wholly above zero"),
        "reading_order": ("the four components first, on the "
                          "components Run's draws and seed-noise stream, "
                          "then the second-path quantities in the order "
                          "listed"),
        "components_run": _hash_of(components)}
    config["inputs"]["nown"] = {b: [_hash_of(d) for d in ds]
                                for b, ds in dirs["nown"].items()}
    if family == "gnn":
        config["inputs"]["cglobaldup"] = [_hash_of(d) for d in dirs["cglobaldup"]]
    if family == "blind":
        config["placebo"] = ("own fields ignore the network, so the own-pool "
                             "network-sight scores must equal solo's bit for "
                             "bit and the value of sight at the own pool be "
                             "exactly zero")
    return config


def second_path_metrics(family: str, dirs: dict, wpath: Path, wmetrics: dict,
                        components: Path, run_dir: Path) -> dict:
    t0 = time.time()
    cell = add_second_path_cells(family, dirs, load_four_cells(family, dirs))
    groups = cell["groups"]
    labels = list(groups)
    sides = second_path_sides(family)
    combos = second_path_combinations(family)

    banks_csv = pd.read_csv(verify(wpath, wmetrics["artifacts"]["banks"]),
                            dtype={"bank": str}).set_index("bank")
    se_exp1 = banks_csv.loc[labels, "se"].to_numpy()
    weights = 1.0 / se_exp1 ** 2

    placebo = None
    if family == "blind":
        diffs = [float(np.max(np.abs(groups[g]["nown"] - groups[g]["solo"])))
                 for g in labels]
        placebo = {"bit_identical": bool(all(d == 0.0 for d in diffs)),
                   "max_abs_diff": max(diffs)}
        assert placebo["bit_identical"], \
            ("the placebo failed: the blind tree's scores at (network, own) and "
             "(ledger, own) differ")

    boot = four_cell_bootstrap(cell["n_test"], groups, sides)
    path1_est = component_estimates(boot, family)
    second = combination_estimates(boot, family, combos)
    est = {g: path1_est[g] | second[g] for g in labels}
    t_boot = time.time() - t0

    rng = np.random.default_rng(SEED)
    path1 = {name: weighted_mean_reading(labels, est, weights, name, family, rng)
             for name in COMPONENT_ORDER}
    original = _metrics(components)["readings"]
    for name in COMPONENT_ORDER:
        assert (path1[name]["weighted_mean"] == original[name]["weighted_mean"]
                and path1[name]["ci95"] == original[name]["ci95"]), \
            f"{name}: the shared draws did not reproduce the components Run"
    readings = {}
    for name in combos:
        reading = weighted_mean_reading(labels, est, weights, name, family, rng)
        deltas = np.array([est[g][name]["delta"] for g in labels])
        reading["per_bank"] = {"median": float(np.median(deltas)),
                               "range": [float(deltas.min()), float(deltas.max())],
                               "above_zero": int((deltas > 0).sum()),
                               "interval_above_zero": int(sum(
                                   est[g][name]["ci95"][0] > 0 for g in labels)),
                               "interval_below_zero": int(sum(
                                   est[g][name]["ci95"][1] < 0 for g in labels))}
        readings[name] = reading

    suffixes = ("", COPY_WEIGHTED) if family == "gnn" else ("",)
    examples_exceed_sight = {}
    for suffix in suffixes:
        holds = {path: readings[name + suffix]["ci95"][0] > 0.0
                 for path, name in EXAMPLES_OVER_SIGHT.items()}
        examples_exceed_sight["original" if not suffix else suffix.lstrip("_")] = \
            holds | {"both_paths": all(holds.values())}

    agg = np.median if PAIRING[family] == "median_over_grid" else np.mean
    table = pd.DataFrame.from_dict(cell["meta"], orient="index").loc[labels]
    table.index.name = "bank"
    table["weight_se_exp1"] = se_exp1
    for side in sides:
        table[f"{side}_ap"] = [agg(boot["point"][g][side]) for g in labels]
    for name in list(COMPONENT_ORDER) + list(combos):
        table[f"{name}_delta"] = [est[g][name]["delta"] for g in labels]
        table[f"{name}_se"] = [est[g][name]["se"] for g in labels]
        table[f"{name}_ci_lo"] = [est[g][name]["ci95"][0] for g in labels]
        table[f"{name}_ci_hi"] = [est[g][name]["ci95"][1] for g in labels]
    order = list(combos)
    replicates = np.stack([np.stack([est[g][name]["delta_reps"] for g in labels],
                                    axis=1) for name in order], axis=2)
    out = {"dataset_key": KEY, "family": family, "k": len(labels),
           "n_test": int(cell["n_test"]), "pairing": PAIRING[family],
           "replicates": REPLICATES, "quantity_order": order,
           "path1_readings": path1, "path1_reproduced": True,
           "readings": readings, "examples_exceed_sight": examples_exceed_sight,
           "seconds": {"bootstrap": t_boot, "total": time.time() - t0},
           "artifacts": {"banks": write_csv(run_dir, "banks", table),
                         "replicates": write_array(run_dir, "replicates",
                                                   replicates.astype(np.float32))}}
    if placebo is not None:
        out["placebo"] = placebo
    return out


def cmd_secondpath(family: str) -> dict | None:
    done1 = completed(exp1.EXPERIMENT)
    done2 = completed(exp2.EXPERIMENT)
    dirs = second_path_dirs(done1, done2, family)
    found = components_run(done1, done2, family)
    if dirs is None or found is None:
        print(f"secondpath {family}: inputs incomplete", flush=True)
        return None
    components, _ = found
    wpath, wmetrics = weights_run(done1, family)
    metrics = run(exp2.EXPERIMENT, second_path_config(family, dirs, wpath, components),
                  lambda config, run_dir: second_path_metrics(
                      family, dirs, wpath, wmetrics, components, run_dir),
                  done=done2)
    for name, r in metrics["readings"].items():
        print(f"secondpath {family} {name}: mean {r['weighted_mean']:+.4f} "
              f"ci [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]", flush=True)
    print(f"secondpath {family} training examples outweigh sight: "
          f"{json.dumps(metrics['examples_exceed_sight'])}", flush=True)
    if "placebo" in metrics:
        print(f"secondpath {family} placebo: {metrics['placebo']}", flush=True)
    return metrics


# ------------------------------------------------------------------ dial --

def dial_inputs(done1: dict, done2: dict, family: str) -> dict | None:
    """Case-reduction Run directories per (bank, level, draw), plus Experiment
    1's adopted solo Runs at the full count; None while incomplete."""
    roster = exp1.roster_for(KEY)
    table = exp2.dial_roster(KEY)
    check = f"dial_{family}"
    dirs, alls = {}, {}
    for bank, row in table.iterrows():
        for level in exp2.dial_levels_for(int(row["train_pos"])):
            for draw in range(exp2.DIAL_DRAWS):
                unit = exp2.Unit(check, KEY, bank, level, draw)
                d = exp2_dir(done2, unit, roster)
                if d is None:
                    return None
                dirs[(bank, level, draw)] = d
        if family == "tree":
            d = exp1.recorded_dir(done1, exp1.Unit("tree_solo", KEY, bank))
        else:
            d = exp1.recorded_dir(done1, exp1.Unit("gnn_solo", KEY, bank,
                                               exp2.DIAL_GNN_SEED))
            alls[f"{bank}_seeds"] = [exp1.recorded_dir(
                done1, exp1.Unit("gnn_solo", KEY, bank, s)) for s in exp1.SEEDS]
            if any(x is None for x in alls[f"{bank}_seeds"]):
                return None
        if d is None:
            return None
        alls[bank] = d
    return {"dial": dirs, "all": alls, "table": table}


def dial_config(family: str, inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    table = inputs["table"]
    config = {"experiment": exp2.EXPERIMENT, "check": "dial", "world": world,
              "dataset": dataset, "dataset_key": KEY, "family": family,
              "banks": list(table.index), "levels": exp2.DIAL_LEVELS,
              "draws": exp2.DIAL_DRAWS, "salt": exp2.DIAL_SALT,
              "bank_rule": "the largest roster banks by training-window "
                           "laundering payments",
              "all_level": "adopted from Experiment 1's solo Runs by config identity",
              "range_at_full_count": DIAL_RANGE[family],
              "level_at_or_above_count": "collapses to the all level and is "
                                         "not fitted",
              "inputs": {"dial": {f"{b}:{lv}:{dr}": _hash_of(d)
                                  for (b, lv, dr), d in inputs["dial"].items()},
                         "all": {b: _hash_of(d) for b, d in inputs["all"].items()
                                 if not str(b).endswith("_seeds")},
                         "all_seeds": {b: [_hash_of(d) for d in ds]
                                       for b, ds in inputs["all"].items()
                                       if str(b).endswith("_seeds")}},
              "features_version": exp1.FEATURES_VERSION}
    if family == "gnn":
        config["seed"] = exp2.DIAL_GNN_SEED
        config["gnn_version"] = exp1.GNN_VERSION
    return config


def dial_metrics(family: str, inputs: dict, run_dir: Path) -> dict:
    t0 = time.time()
    table = inputs["table"]
    lowest = min(exp2.DIAL_LEVELS)
    rows, summary = [], {}
    kept_check = True
    for bank, roster_row in table.iterrows():
        n_pos = int(roster_row["train_pos"])
        levels = exp2.dial_levels_for(n_pos)
        all_metrics = _metrics(inputs["all"][bank])
        if family == "tree":
            ap_all = np.array([s["test_ap"] for s in all_metrics["settings"]])
            band = [float(ap_all.min()), float(ap_all.max())]
            centre_all = float(np.median(ap_all))
        else:
            # The curve is seed-0 throughout (the case reduction's fixed
            # seed); the range prices seed noise from Experiment 1's three
            # solo seeds.
            seed_aps = [_metrics(d)["test_ap"]
                        for d in inputs["all"][f"{bank}_seeds"]]
            band = [float(min(seed_aps)), float(max(seed_aps))]
            centre_all = float(all_metrics["test_ap"])
        by_level = {}
        for level in levels:
            draws = []
            for draw in range(exp2.DIAL_DRAWS):
                m = _metrics(inputs["dial"][(bank, level, draw)])
                if family == "tree":
                    aps = np.array([s["test_ap"] for s in m["settings"]])
                    centre = float(np.median(aps))
                    for s, ap in enumerate(aps):
                        rows.append({"bank": bank, "level": level, "draw": draw,
                                     "setting": s, "test_ap": float(ap)})
                else:
                    centre = float(m["test_ap"])
                    rows.append({"bank": bank, "level": level, "draw": draw,
                                 "setting": -1, "test_ap": centre})
                draws.append(centre)
            by_level[level] = {"mean": float(np.mean(draws)),
                               "draws": draws,
                               "within_band": bool(np.mean(draws) >= band[0])}
        # Nesting: within a draw, each level's kept set contains the smaller
        # levels' sets; checked across the recorded payment lists.
        for draw in range(exp2.DIAL_DRAWS):
            kept = [set(_metrics(inputs["dial"][(bank, lv, draw)])["kept_payments"])
                    for lv in levels]
            for small, big in zip(kept, kept[1:]):
                kept_check &= small <= big
        drop = centre_all - by_level[lowest]["mean"]
        summary[bank] = {
            "train_pos": n_pos, "levels": levels, "band_all": band,
            "ap_all": centre_all, "by_level": by_level,
            "drop_all_to_lowest": float(drop)}
    curve = pd.DataFrame(rows)
    return {"dataset_key": KEY, "family": family, "lowest_level": lowest,
            "banks": summary, "nested_draws_hold": bool(kept_check),
            "n_banks": len(summary),
            "artifacts": {"curve": write_csv(run_dir, "curve",
                                             curve.set_index("bank"))},
            "seconds": time.time() - t0}


def cmd_dial(family: str, done1: dict | None = None,
             done2: dict | None = None) -> dict | None:
    done1 = done1 if done1 is not None else completed(exp1.EXPERIMENT)
    done2 = done2 if done2 is not None else completed(exp2.EXPERIMENT)
    inputs = dial_inputs(done1, done2, family)
    if inputs is None:
        print(f"dial {family}: inputs incomplete", flush=True)
        return None
    metrics = run(exp2.EXPERIMENT, dial_config(family, inputs),
                  lambda config, run_dir: dial_metrics(family, inputs, run_dir),
                  done=done2)
    print(f"dial {family}: {metrics['n_banks']} banks; nesting "
          f"{'holds' if metrics['nested_draws_hold'] else 'BROKEN'}", flush=True)
    return metrics


def cmd_all() -> int:
    done1 = completed(exp1.EXPERIMENT)
    done2 = completed(exp2.EXPERIMENT)
    for family in ("tree", "gnn", "blind"):
        cmd_components(family, done1, done2)
    done2 = completed(exp2.EXPERIMENT)
    for family in ("tree", "gnn"):
        cmd_dial(family, done1, done2)
    cmd_sightcheck()
    for family in ("tree", "gnn", "blind"):
        cmd_secondpath(family)
    return 0


def _state(done2: dict, config: dict | None) -> str:
    if config is None:
        return "inputs incomplete"
    return "analysed" if config_hash(config) in done2 else "inputs complete"


def cmd_status() -> int:
    done1 = completed(exp1.EXPERIMENT)
    done2 = completed(exp2.EXPERIMENT)
    for family in ("tree", "gnn", "blind"):
        dirs = cell_dirs(done1, done2, family)
        config = (components_config(family, dirs, weights_run(done1, family)[0])
                  if dirs else None)
        print(f"components {family:6s} {_state(done2, config)}")
    for family in ("tree", "gnn"):
        inputs = dial_inputs(done1, done2, family)
        print(f"dial       {family:6s} "
              f"{_state(done2, dial_config(family, inputs) if inputs else None)}")
    dirs, found = sightcheck_dirs(done1, done2), components_run(done1, done2, "gnn")
    config = (sightcheck_config(dirs, weights_run(done1, "gnn")[0], found[0])
              if dirs and found else None)
    print(f"sightcheck {'gnn':6s} {_state(done2, config)}")
    for family in ("tree", "gnn", "blind"):
        dirs = second_path_dirs(done1, done2, family)
        found = components_run(done1, done2, family)
        config = (second_path_config(family, dirs, weights_run(done1, family)[0],
                                     found[0]) if dirs and found else None)
        print(f"secondpath {family:6s} {_state(done2, config)}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "components":
        return 0 if cmd_components(args[0]) is not None else 1
    if cmd == "dial":
        return 0 if cmd_dial(args[0]) is not None else 1
    if cmd == "all":
        return cmd_all()
    if cmd == "sightcheck":
        return 0 if cmd_sightcheck() is not None else 1
    if cmd == "secondpath":
        return 0 if cmd_secondpath(args[0]) is not None else 1
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
