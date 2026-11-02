#!/usr/bin/env python
"""Experiment 3 (Chapter 5, Ranking the banks by their shortfall): the
stages after the label-free properties and the sixteen regression models.

Five stages, in order. **Inputs** checks every shortfall table read here
against its recorded digest and its bootstrap replicates. **Screen** gives
each label-free property's rank correlation with the shortfall on HI-Medium's
107 qualifying banks, and its partial rank correlation with both properties of
scale held fixed. **Holdout** fits the scale model on those 107 banks and
applies it unchanged to HI-Small's eleven. **Ceiling** computes the two
quantities computed from labels and scores the two models that use labels out
of fold, exactly as the selection scores the models on label-free properties.
**Bank facing** fits the model on scale and confirmed cases on the 107 banks
and applies it unchanged to HI-Small's eleven. The tree family, the GNN family
and the payment-only tree are read on HI-Medium; HI-Small is read in the tree
and GNN families.

No model trains. Every number is a function of the cached AMLworld tables, the
Runs of `exp3_descriptors.py` and Experiment 1's recorded shortfalls, each Run
cited by config hash and verified by digest.

  uv run python scripts/exp3.py              # every stage, in order
  uv run python scripts/exp3.py screen       # one stage

LightGBM is never imported; torch enters only through the per-Run fingerprint.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp3_data as dd  # noqa: E402
import exp3_descriptors  # noqa: E402
from m import completed, config_hash, run, track  # noqa: E402
from m import ceiling as lab  # noqa: E402
from m import descriptors as desc  # noqa: E402
from m import rank, rule  # noqa: E402
from m.artifacts import verify, write_csv  # noqa: E402
from m.worlds import amlworld  # noqa: E402

EXPERIMENT = "exp3"
SEED = 0
BOOTSTRAP_DRAWS = 1000

SCALE = desc.GROUPS[desc.BASELINE]

CEILING_CANDIDATES = (rule.Candidate(desc.BASELINE, SCALE),) + tuple(
    rule.Candidate(f"{desc.BASELINE}+{name}", SCALE + (name,)) for name in lab.NAMES)
CEILING_ORDER = [candidate.name for candidate in CEILING_CANDIDATES]

BANK_FACING = ("payment_count", "active_accounts", "laundering_count")
# HI-Medium's range of training-window payments per bank (Appendix D); the
# assertion catches a fit on other rows than the ceiling Run's.
FITTED_RANGE = (5198.0, 125014.0)


def _selection_run(family: str, done: dict) -> Path:
    return done[config_hash(exp3_descriptors.selection_config(family))]


def _design(table: pd.DataFrame, descriptors: tuple[str, ...]) -> np.ndarray:
    return table[[f"log1p_{name}" for name in descriptors]].to_numpy(np.float64)


def _coefficients(descriptors: tuple[str, ...], intercept: float,
                  coefficients: np.ndarray) -> pd.DataFrame:
    rows = [{"term": "intercept", "value": intercept}] + [
        {"term": f"log1p_{name}", "value": float(value)}
        for name, value in zip(descriptors, coefficients)]
    return pd.DataFrame(rows).set_index("term")


def _inputs(families: tuple[str, ...], keys: tuple[str, ...], done: dict) -> dict:
    out = {"descriptors": {key: config_hash(exp3_descriptors.descriptor_config(key))
                           for key in keys},
           "shortfall": {f"{key}/{family}": dd.shortfall_hash(key, family)
                         for key in keys for family in families}}
    for family in families:
        path = _selection_run(family, done)
        out.setdefault("selection", {})[family] = json.loads(
            (path / "status.json").read_text())["config_hash"]
    return out


# ----------------------------------------------------------------- gate --

def inputs_config(done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "inputs_gate",
            "rule": "every target table Experiment 3 reads is verified against its recorded "
                    "digest, stands in its roster's order, and is reproduced by its own "
                    "persisted bootstrap replicates; every table of label-free properties "
                    "covers exactly the same banks",
            "tolerance": {"bootstrap_standard_error": 1e-6},
            "cells": [f"{key}/{family}" for key in dd.KEYS
                      for family in dd.families_of(key)],
            "inputs": {**_inputs(dd.FAMILIES, (dd.FIT_KEY,), done),
                       "holdout": _inputs(dd.HOLDOUT_FAMILIES, (dd.HOLDOUT_KEY,),
                                          done)["shortfall"]}}


def inputs_metrics(run_dir: Path, done: dict) -> dict:
    rows = []
    for key in dd.KEYS:
        values = exp3_descriptors.descriptor_table(key, done)
        for family in dd.families_of(key):
            target = dd.targets(key, family)
            path = dd.shortfall_run(key, family)
            roster = pd.read_csv(path / "artifacts" / "roster.csv",
                                 dtype={"bank": str}).set_index("bank")
            assert list(target.index) == list(roster.index), (
                f"{key}/{family}: the target table and its roster disagree on the "
                f"banks' order, so every array Experiment 3 builds from them is misaligned")
            replicates = dd.target_replicates(key, family).astype(np.float64)
            assert replicates.shape == (1000, len(target)), (
                f"{key}/{family}: the replicate array is {replicates.shape} against "
                f"{len(target)} banks")
            reproduced = float(np.abs(np.nanstd(replicates, axis=0, ddof=1)
                                      - target["se_boot"].to_numpy()).max())
            assert reproduced < 1e-6, (
                f"{key}/{family}: the persisted replicates reproduce the recorded "
                f"bootstrap standard error only to {reproduced:g}")
            aligned = dd.aligned(values, target)
            rows.append({"dataset_key": key, "family": family, "k": int(len(target)),
                         "shortfall": json.loads(
                             (path / "status.json").read_text())["config_hash"],
                         "roster_order_matches": True,
                         "standard_error_reproduced": reproduced,
                         "property_banks": int(len(aligned)),
                         "delta_inside_interval": float(
                             ((target["delta"] >= target["ci_lo"])
                              & (target["delta"] <= target["ci_hi"])).mean())})

    table = pd.DataFrame(rows).set_index(["dataset_key", "family"])
    worst = float(table["standard_error_reproduced"].max())
    return {"cells": int(len(table)), "worst_standard_error_difference": worst,
            "tolerance": 1e-6, "pass": bool(worst < 1e-6),
            "artifacts": {"inputs": write_csv(run_dir, "inputs", table)}}


# --------------------------------------------------------------- screen --

def screen_config(done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "screen", "dataset_key": dd.FIT_KEY,
            "families": list(dd.FAMILIES),
            "descriptors": list(desc.NAMES),
            "rho": "Spearman's rank correlation between the label-free property and "
                   "the shortfall",
            "rho_partial_scale": "the partial rank correlation with both properties "
                                 "of scale held fixed: the ranks of the property and "
                                 "of the shortfall are each fitted by least squares "
                                 "on the ranks of both properties of scale, and the "
                                 "two sets of leftovers are correlated",
            "scale": list(SCALE),
            "inputs": _inputs(dd.FAMILIES, (dd.FIT_KEY,), done),
            "descriptors_version": desc.VERSION, "rank_version": rank.VERSION}


def screen_metrics(run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    values = exp3_descriptors.descriptor_table(dd.FIT_KEY, done)
    rows = []
    for family in dd.FAMILIES:
        target_table = dd.targets(dd.FIT_KEY, family)
        table = dd.aligned(values, target_table)
        target = target_table["delta"].to_numpy(np.float64)
        scale = _design(table, SCALE)
        for name in desc.NAMES:
            x = table[f"log1p_{name}"].to_numpy(np.float64)
            rows.append({
                "family": family, "property": name,
                "group": next(g for g, members in desc.GROUPS.items() if name in members),
                "rho": rank.spearman(x, target),
                "rho_partial_scale": (np.nan if name in SCALE
                                      else rank.partial_spearman(x, target, scale)["rho"])})

    table = pd.DataFrame(rows).set_index(["family", "property"])
    return {"dataset_key": dd.FIT_KEY, "k": int(len(values)),
            "seconds": time.time() - t0,
            "artifacts": {"screen": write_csv(run_dir, "screen", table)}}


# -------------------------------------------------------------- held-out --

def holdout_config(family: str, done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "holdout", "family": family,
            "fit_on": dd.FIT_KEY, "test_on": dd.HOLDOUT_KEY,
            "model": "the scale model: one weighted least-squares fit (weights 1 over "
                    "s_B squared) on every qualifying bank of the fitting dataset, "
                    "written to coefficients.csv and applied unchanged to the "
                    "held-out banks",
            "descriptors": list(SCALE),
            "inputs": _inputs((family,), (dd.FIT_KEY, dd.HOLDOUT_KEY), done),
            "rule_version": rule.VERSION, "rank_version": rank.VERSION}


def holdout_metrics(family: str, run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    fitting = dd.targets(dd.FIT_KEY, family)
    design = _design(dd.aligned(exp3_descriptors.descriptor_table(dd.FIT_KEY, done),
                                fitting), SCALE)
    intercept, coefficients = rule.fit(design, fitting["delta"].to_numpy(np.float64),
                                       dd.weights(fitting),
                                       ~rule.constant_columns(design))

    target_table = dd.targets(dd.HOLDOUT_KEY, family)
    table = dd.aligned(exp3_descriptors.descriptor_table(dd.HOLDOUT_KEY, done),
                       target_table)
    prediction = rule.predict(_design(table, SCALE), intercept, coefficients)
    target = target_table["delta"].to_numpy(np.float64)
    banks = pd.DataFrame({"prediction": prediction, "delta": target,
                          "se": target_table["se"].to_numpy()},
                         index=target_table.index)

    return {"family": family, "fit_on": dd.FIT_KEY, "test_on": dd.HOLDOUT_KEY,
            "k": int(len(target)), "model": desc.BASELINE, "properties": list(SCALE),
            "rho": rank.spearman(prediction, target),
            "seconds": time.time() - t0,
            "artifacts": {"holdout": write_csv(run_dir, "holdout", banks),
                          "coefficients": write_csv(
                              run_dir, "coefficients",
                              _coefficients(SCALE, intercept, coefficients))}}


# -------------------------------------------------------------- ceiling --

def ceiling_descriptor_config() -> dict:
    return {"experiment": EXPERIMENT, "check": "ceiling_descriptors",
            "dataset_key": dd.FIT_KEY,
            "window": "split == 0 (the training window), same-account rows excluded, "
                      "labels read: the bank's own confirmed cases",
            "descriptors": {name: lab.DESCRIPTIONS[name] for name in lab.NAMES},
            "roster_from": {
                "note": "the qualifying banks are read from Experiment 1's shortfall "
                        "Run for the tree family; the roster artefact is "
                        "byte-identical across the families",
                "shortfall": dd.shortfall_hash(dd.FIT_KEY, dd.FAMILIES[0])},
            "ceiling_version": lab.VERSION, "descriptors_version": desc.VERSION}


def ceiling_descriptor_metrics(run_dir: Path) -> dict:
    t0 = time.time()
    df = amlworld.load(dd.data.DATASETS[dd.FIT_KEY][1])
    training = (df["split"].to_numpy() == dd.TRAINING_SPLIT) & ~df["same_account"].to_numpy()
    frame = df.iloc[np.flatnonzero(training)]
    laundering = frame["is_laundering"].to_numpy(np.int8) == 1
    from_bank = frame["from_bank"].astype(str).to_numpy()
    to_bank = frame["to_bank"].astype(str).to_numpy()
    del df, frame

    banks = list(dd.targets(dd.FIT_KEY, dd.FAMILIES[0]).index)
    table = pd.DataFrame({bank: lab.quantities(from_bank, to_bank, laundering, bank)
                          for bank in track(banks, "quantities computed from labels")}).T
    table.index.name = "bank"
    for name in lab.NAMES:
        table[f"log1p_{name}"] = np.log1p(table[name].to_numpy(np.float64))

    summary = {name: {"defined": int(table[name].notna().sum()),
                      "distinct": int(table[name].nunique()),
                      "zero": int((table[name] == 0).sum()),
                      "min": float(table[name].min()),
                      "median": float(table[name].median()),
                      "max": float(table[name].max())} for name in lab.NAMES}
    return {"dataset_key": dd.FIT_KEY, "k": int(len(table)),
            "quantity": summary, "seconds": time.time() - t0,
            "artifacts": {"ceiling_descriptors": write_csv(
                run_dir, "ceiling_descriptors", table)}}


def ceiling_config(family: str, done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "ceiling", "family": family,
            "dataset_key": dd.FIT_KEY,
            "candidates": {candidate.name: list(candidate.descriptors)
                           for candidate in CEILING_CANDIDATES},
            "folds": {"n_folds": rule.N_FOLDS, "repeats": rule.REPEATS},
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED},
            "candidate_counts": {"label_free": len(exp3_descriptors.CANDIDATES),
                                 "label_using": len(CEILING_CANDIDATES) - 1},
            "protocol": "the folds, fitting, ranking metric and comparison with "
                        "the scale model of the models on label-free properties, "
                        "unchanged",
            "inputs": {**_inputs((family,), (dd.FIT_KEY,), done),
                       "ceiling_descriptors": config_hash(ceiling_descriptor_config())},
            "ceiling_version": lab.VERSION, "rule_version": rule.VERSION,
            "rank_version": rank.VERSION, "descriptors_version": desc.VERSION}


def ceiling_metrics(family: str, run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    ceiling_path = done[config_hash(ceiling_descriptor_config())]
    label_using = pd.read_csv(ceiling_path / "artifacts" / "ceiling_descriptors.csv",
                              dtype={"bank": str}).set_index("bank")
    target_table = dd.targets(dd.FIT_KEY, family)
    values = dd.aligned(exp3_descriptors.descriptor_table(dd.FIT_KEY, done), target_table)
    table = values.join(dd.aligned(label_using, target_table), how="inner")
    target = target_table["delta"].to_numpy(np.float64)
    weights = dd.weights(target_table)

    results = {}
    for candidate in CEILING_CANDIDATES:
        design = _design(table, candidate.descriptors)
        predictions = rule.out_of_fold(design, target, weights,
                                       repeats=rule.REPEATS, n_folds=rule.N_FOLDS)
        per_repeat = np.array([rank.spearman(row, target) for row in predictions])
        results[candidate.name] = {"score": float(per_repeat.mean()),
                                   "predictions": predictions,
                                   "prediction": predictions.mean(axis=0)}
    score = {name: results[name]["score"] for name in CEILING_ORDER}
    best = max(CEILING_ORDER, key=score.get)
    baseline = results[desc.BASELINE]["predictions"]
    against_baseline = {name: rank.bootstrap_repeat_mean_difference(
        results[name]["predictions"], baseline, target,
        draws=BOOTSTRAP_DRAWS, seed=SEED) for name in CEILING_ORDER}
    beaters = [name for name in CEILING_ORDER
               if name != desc.BASELINE and against_baseline[name]["q05"] > 0]

    rows = [{"candidate": name, "terms": " ".join(
        {c.name: c.descriptors for c in CEILING_CANDIDATES}[name]),
             "score": score[name],
             "delta_vs_baseline": against_baseline[name]["delta"],
             "q05_vs_baseline": against_baseline[name]["q05"],
             "outperforms": name in beaters,
             "ceiling": name == best} for name in CEILING_ORDER]
    oof = pd.DataFrame({name: results[name]["prediction"] for name in CEILING_ORDER},
                       index=target_table.index)
    oof.insert(0, "delta", target)

    return {"family": family, "dataset_key": dd.FIT_KEY, "k": int(len(target)),
            "best_model": best, "best_score": score[best],
            "outperform_the_scale_model": beaters,
            "candidate_counts": {"label_free": len(exp3_descriptors.CANDIDATES),
                                 "label_using": len(CEILING_CANDIDATES) - 1},
            "seconds": time.time() - t0,
            "artifacts": {"candidates": write_csv(
                run_dir, "candidates", pd.DataFrame(rows).set_index("candidate")),
                "oof": write_csv(run_dir, "oof", oof)}}


# ------------------------------- the model on scale and confirmed cases --

def bank_facing_counts_config() -> dict:
    return {"experiment": EXPERIMENT, "check": "bank_facing_counts",
            "dataset_key": dd.HOLDOUT_KEY,
            "window": "split == 0 (the training window), same-account rows excluded, "
                      "labels read: the count is the bank's own confirmed cases",
            "descriptor": {"laundering_count": lab.DESCRIPTIONS["laundering_count"]},
            "definition": "the ceiling's laundering_count, computed on the held-out "
                          "dataset so that the model on scale and confirmed cases can "
                          "be applied there",
            "roster_and_targets_from": {
                "shortfall": dd.shortfall_hash(dd.HOLDOUT_KEY, dd.FAMILIES[0])},
            "ceiling_version": lab.VERSION}


def bank_facing_counts_metrics(run_dir: Path) -> dict:
    t0 = time.time()
    df = amlworld.load(dd.data.DATASETS[dd.HOLDOUT_KEY][1])
    training = (df["split"].to_numpy() == dd.TRAINING_SPLIT) \
        & ~df["same_account"].to_numpy()
    frame = df.iloc[np.flatnonzero(training)]
    y = frame["is_laundering"].to_numpy(np.int8)
    from_bank = frame["from_bank"].astype(str).to_numpy()
    to_bank = frame["to_bank"].astype(str).to_numpy()
    del df

    target_table = dd.targets(dd.HOLDOUT_KEY, dd.FAMILIES[0])
    records = [{"bank": bank,
                "laundering_count": int((((from_bank == bank) | (to_bank == bank))
                                         & (y == 1)).sum())}
               for bank in target_table.index]
    table = pd.DataFrame(records).set_index("bank")
    table["log1p_laundering_count"] = np.log1p(
        table["laundering_count"].to_numpy(np.float64))
    return {"dataset_key": dd.HOLDOUT_KEY, "k": int(len(table)),
            "training_rows": int(training.sum()),
            "training_laundering": int((y == 1).sum()),
            "counts": {"min": int(table["laundering_count"].min()),
                       "median": float(table["laundering_count"].median()),
                       "max": int(table["laundering_count"].max())},
            "seconds": time.time() - t0,
            "artifacts": {"counts": write_csv(run_dir, "counts", table)}}


def bank_facing_config(family: str, done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "bank_facing", "family": family,
            "fit_on": dd.FIT_KEY, "test_on": dd.HOLDOUT_KEY,
            "model": "the model on scale and confirmed cases: one weighted least-squares "
                    "fit (weights 1 over s_B squared, as the selection used) on the "
                    "ceiling Run's 107 rows, inputs log(1+payment count), "
                    "log(1+active accounts), log(1+laundering count), written to "
                    "coefficients.csv and applied unchanged to the held-out banks",
            "descriptors": list(BANK_FACING),
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED},
            "beside_it": "the scale model's held-out reading, from the holdout Run, and "
                         "the ceiling Run's out-of-fold score for the same model",
            "inputs": {**_inputs((family,), (dd.FIT_KEY, dd.HOLDOUT_KEY), done),
                       "ceiling_descriptors": config_hash(ceiling_descriptor_config()),
                       "bank_facing_counts": config_hash(bank_facing_counts_config()),
                       "holdout": config_hash(holdout_config(family, done)),
                       "ceiling": config_hash(ceiling_config(family, done))},
            "rule_version": rule.VERSION, "rank_version": rank.VERSION,
            "ceiling_version": lab.VERSION}


def bank_facing_metrics(family: str, run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    target_table = dd.targets(dd.FIT_KEY, family)
    values = dd.aligned(exp3_descriptors.descriptor_table(dd.FIT_KEY, done), target_table)
    ceiling_path = done[config_hash(ceiling_descriptor_config())]
    ceiling_desc = json.loads((ceiling_path / "metrics.json").read_text())
    label_using = pd.read_csv(
        verify(ceiling_path, ceiling_desc["artifacts"]["ceiling_descriptors"]),
        dtype={"bank": str}).set_index("bank")
    table = values.join(dd.aligned(label_using, target_table), how="inner")
    design = _design(table, BANK_FACING)
    target = target_table["delta"].to_numpy(np.float64)
    keep = ~rule.constant_columns(design)
    intercept, coefficients = rule.fit(design, target, dd.weights(target_table), keep)

    payments = table["payment_count"].to_numpy(np.float64)
    fitted_range = (float(payments.min()), float(payments.max()))
    assert fitted_range == FITTED_RANGE, (
        f"the fitting banks' payment-count range is {fitted_range}, not "
        f"{FITTED_RANGE}; the rows are not the ceiling Run's")

    holdout_targets = dd.targets(dd.HOLDOUT_KEY, family)
    holdout_values = dd.aligned(exp3_descriptors.descriptor_table(dd.HOLDOUT_KEY, done),
                                holdout_targets)
    counts_path = done[config_hash(bank_facing_counts_config())]
    counts_metrics_ = json.loads((counts_path / "metrics.json").read_text())
    counts = pd.read_csv(verify(counts_path, counts_metrics_["artifacts"]["counts"]),
                         dtype={"bank": str}).set_index("bank")
    holdout_table = holdout_values.join(dd.aligned(counts, holdout_targets),
                                        how="inner")
    prediction = rule.predict(_design(holdout_table, BANK_FACING),
                              intercept, coefficients)
    held = holdout_targets["delta"].to_numpy(np.float64)
    interval = rank.bootstrap_rho(prediction, held, draws=BOOTSTRAP_DRAWS, seed=SEED)

    holdout_run = done[config_hash(holdout_config(family, done))]
    label_free = json.loads((holdout_run / "metrics.json").read_text())
    ceiling_run = done[config_hash(ceiling_config(family, done))]
    ceiling_metrics_ = json.loads((ceiling_run / "metrics.json").read_text())
    candidates = pd.read_csv(
        verify(ceiling_run, ceiling_metrics_["artifacts"]["candidates"])) \
        .set_index("candidate")

    holdout_csv = pd.read_csv(verify(holdout_run, label_free["artifacts"]["holdout"]),
                              dtype={"bank": str}).set_index("bank")
    banks = pd.DataFrame({"laundering_count": holdout_table["laundering_count"].to_numpy(),
                          "prediction": prediction,
                          "label_free_prediction": holdout_csv.loc[
                              holdout_targets.index, "prediction"].to_numpy(),
                          "delta": held, "se": holdout_targets["se"].to_numpy()},
                         index=holdout_targets.index)
    coefficient_table = _coefficients(BANK_FACING, intercept, coefficients)

    return {"family": family, "fit_on": dd.FIT_KEY, "test_on": dd.HOLDOUT_KEY,
            "k": int(len(held)),
            "terms": list(BANK_FACING),
            "coefficients": coefficient_table["value"].to_dict(),
            "fitted_range_payment_count": list(fitted_range),
            "ceiling_oof_score": float(candidates.loc["scale+laundering_count",
                                                      "score"]),
            "rho": rank.spearman(prediction, held),
            "bootstrap_ci95": interval["ci95"],
            "label_free": {"model": label_free["model"], "rho": label_free["rho"]},
            "seconds": time.time() - t0,
            "artifacts": {"holdout": write_csv(run_dir, "holdout", banks),
                          "coefficients": write_csv(run_dir, "coefficients",
                                                    coefficient_table)}}


# ----------------------------------------------------------------- main --

def cmd(stage: str, done: dict) -> None:
    if stage == "inputs":
        metrics = run(EXPERIMENT, inputs_config(done),
                      lambda config, run_dir: inputs_metrics(run_dir, done), done=done)
        print(f"inputs_gate: {metrics['cells']} cells verified, worst standard-error "
              f"difference {metrics['worst_standard_error_difference']:.2e} — "
              f"{'pass' if metrics['pass'] else 'FAIL'}", flush=True)
        if not metrics["pass"]:
            raise SystemExit("the inputs gate failed; localise the difference before "
                             "reading any Experiment 3 result")
    elif stage == "screen":
        metrics = run(EXPERIMENT, screen_config(done),
                      lambda config, run_dir: screen_metrics(run_dir, done), done=done)
        print(f"screen: {len(desc.NAMES)} label-free properties against "
              f"{len(dd.FAMILIES)} families' shortfalls on {metrics['k']} banks",
              flush=True)
    elif stage == "holdout":
        for family in dd.HOLDOUT_FAMILIES:
            metrics = run(EXPERIMENT, holdout_config(family, done),
                          lambda config, run_dir, f=family:
                          holdout_metrics(f, run_dir, done), done=done)
            print(f"holdout {family}: the scale model ranks {dd.HOLDOUT_KEY}'s "
                  f"{metrics['k']} banks at rho {metrics['rho']:+.4f}", flush=True)
    elif stage == "ceiling":
        metrics = run(EXPERIMENT, ceiling_descriptor_config(),
                      lambda config, run_dir: ceiling_descriptor_metrics(run_dir),
                      done=done)
        print(f"ceiling_descriptors: {metrics['k']} banks, confirmed cases "
              f"{metrics['quantity']['laundering_count']['min']:.0f} to "
              f"{metrics['quantity']['laundering_count']['max']:.0f}", flush=True)
        for family in dd.FAMILIES:
            out = run(EXPERIMENT, ceiling_config(family, done),
                      lambda config, run_dir, f=family:
                      ceiling_metrics(f, run_dir, done), done=done)
            print(f"ceiling {family}: {out['best_model']} at rho "
                  f"{out['best_score']:+.4f}; outperform the scale model: "
                  f"{out['outperform_the_scale_model'] or 'none'}", flush=True)
    elif stage == "bank_facing":
        metrics = run(EXPERIMENT, bank_facing_counts_config(),
                      lambda config, run_dir: bank_facing_counts_metrics(run_dir),
                      done=done)
        print(f"bank_facing_counts: {metrics['k']} held-out banks, laundering "
              f"counts {metrics['counts']['min']} to {metrics['counts']['max']} "
              f"about a median of {metrics['counts']['median']:.0f}", flush=True)
        done.update(completed(EXPERIMENT))
        for family in dd.HOLDOUT_FAMILIES:
            out = run(EXPERIMENT, bank_facing_config(family, done),
                      lambda config, run_dir, f=family:
                      bank_facing_metrics(f, run_dir, done), done=done)
            low, high = out["bootstrap_ci95"]
            print(f"bank_facing {family}: held-out rho {out['rho']:+.4f} "
                  f"(95 % interval {low:+.2f} to {high:+.2f}); scale model "
                  f"{out['label_free']['rho']:+.4f}; out-of-fold "
                  f"{out['ceiling_oof_score']:+.4f}", flush=True)
    else:
        raise SystemExit(f"unknown stage {stage!r}")


STAGES = ("inputs", "screen", "holdout", "ceiling", "bank_facing")


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    stages = sys.argv[1:] or list(STAGES)
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}")
    done = completed(EXPERIMENT)
    for stage in stages:
        cmd(stage, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
