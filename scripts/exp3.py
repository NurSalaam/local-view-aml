#!/usr/bin/env python
"""Experiment 3 — predicting the shortfall from the bank's own ledger.

Five steps, in a fixed order. **Screen** correlates every
descriptor with every target on HI-Medium's 107 banks, under one
false-discovery family and two partials. **Rank** asks whether the rule the descriptor selection
froze beats the scale-only baseline out of fold. **Held-out** applies the frozen
coefficients to HI-Small's eleven banks, which nothing in the fit ever saw.
**Ceiling** prices what label-freeness costs by
running the identical machinery over label-using quantities. **Bank facing**
freezes the rule a bank may
compute once its own labels are admitted: scale plus the bank's own training-window
laundering-payment count, one weighted least-squares fit on the ceiling Run's
107 rows, held-out transfer to HI-Small's eleven banks beside the label-free
rule's.

No model trains. Every number is a function of the descriptor selection's descriptor Runs
and Experiment 1's recorded shortfalls, each cited by config hash and verified by digest.

  uv run python scripts/exp3.py              # every stage, in order
  uv run python scripts/exp3.py screen       # one stage

LightGBM is never imported; torch enters only through the per-Run fingerprint.
"""
from __future__ import annotations

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
PERMUTATIONS = 10_000
BOOTSTRAP_DRAWS = 1000
FDR_Q = 0.10
PARTIAL_LEVEL = 0.10
ONE_SIDED_LEVEL = 0.05
CALIBRATION_REPLICATES = 500
CALIBRATION_PERMUTATIONS = 999
BLOCK_WIDTH = 5

SCALE = desc.GROUPS[desc.BASELINE]

CEILING_GROUPS = {desc.BASELINE: SCALE, **lab.GROUPS}
CEILING_CANDIDATES = rule.candidates(CEILING_GROUPS, desc.BASELINE)
CEILING_ORDER = [candidate.name for candidate in CEILING_CANDIDATES]

BANK_FACING = ("payment_count", "active_accounts", "laundering_count")
# The fitting roster's training-window payment counts, the range the frozen
# ordering is valid within; asserted where the fit is made.
FITTED_RANGE = (5198.0, 125014.0)


def _selection_run(family: str, done: dict) -> Path:
    return done[config_hash(exp3_descriptors.selection_config(family))]


def _frozen(family: str, done: dict) -> tuple[str, tuple[str, ...]]:
    import json
    metrics = json.loads((_selection_run(family, done) / "metrics.json").read_text())
    return metrics["freeze"]["winner"], tuple(metrics["winner_descriptors"])


def _digest(path: Path) -> str:
    from m.artifacts import sha256_of

    return sha256_of(path)


def _design(table: pd.DataFrame, descriptors: tuple[str, ...]) -> np.ndarray:
    return table[[f"log1p_{name}" for name in descriptors]].to_numpy(np.float64)


def _inputs(families: tuple[str, ...], keys: tuple[str, ...], done: dict) -> dict:
    out = {"descriptors": {key: config_hash(exp3_descriptors.descriptor_config(key))
                           for key in keys},
           "shortfall": {f"{key}/{family}": dd.shortfall_hash(key, family)
                         for key in keys for family in families}}
    for family in families:
        import json
        path = _selection_run(family, done)
        out.setdefault("selection", {})[family] = json.loads(
            (path / "status.json").read_text())["config_hash"]
    return out


# ----------------------------------------------------------------- gate --

def inputs_config(done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "inputs_gate",
            "rule": "every target table Experiment 3 reads is verified against its recorded "
                    "digest, stands in its roster's order, and is reproduced by its own "
                    "persisted bootstrap replicates; every descriptor table covers "
                    "exactly the same banks",
            "tolerance": {"bootstrap_standard_error": 1e-6},
            "cells": [f"{key}/{family}" for key in dd.KEYS for family in dd.FAMILIES],
            "inputs": _inputs(dd.FAMILIES, dd.KEYS, done)}


def inputs_metrics(run_dir: Path, done: dict) -> dict:
    import json

    rows = []
    for key in dd.KEYS:
        values = exp3_descriptors.descriptor_table(key, done)
        for family in dd.FAMILIES:
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
                         "descriptor_banks": int(len(aligned)),
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
            "targets": {family: "the per-bank shortfall Experiment 1 measured"
                        for family in dd.FAMILIES},
            "primary_targets": list(dd.PRIMARY_FAMILIES),
            "descriptors": list(desc.NAMES),
            "family_size": len(desc.NAMES) * len(dd.FAMILIES),
            "fdr": {"procedure": "Benjamini-Hochberg step-up", "q": FDR_Q},
            "raw_p": {"null": "bank permutation", "permutations": PERMUTATIONS,
                      "seed": SEED, "alternative": "two-sided"},
            "partials": {"base_rate": ["test_base_rate"], "scale": list(SCALE)},
            "partial_p": {"null": "permutation of the target within blocks of five "
                                  "banks consecutive in the controls' rank sum",
                          "permutations": PERMUTATIONS, "seed": SEED,
                          "block_width": BLOCK_WIDTH,
                          "beside_it": "the t-approximation on k - 2 - q degrees "
                                       "of freedom"},
            "declaration": f"declared under Benjamini-Hochberg at {FDR_Q} and both "
                           f"partials keeping the raw sign at a conditional p below "
                           f"{PARTIAL_LEVEL}; the "
                           f"scale-partial-only reading is recorded beside it",
            "calibration": {"replicates": CALIBRATION_REPLICATES,
                            "permutations": CALIBRATION_PERMUTATIONS,
                            "block_widths": [3, 5, 10],
                            "beside_it": "the t-approximation's own rate on the same "
                                         "null",
                            "null": "the descriptor and the target independent given "
                                    "the controls, both driven by exp(3z) with z the "
                                    "standardised rank sum"},
            "inputs": _inputs(dd.FAMILIES, (dd.FIT_KEY,), done),
            "descriptors_version": desc.VERSION, "rank_version": rank.VERSION}


def association_rows(table: pd.DataFrame, target: np.ndarray, base_rate: np.ndarray,
                     scale: np.ndarray, label: dict, progress: str) -> list[dict]:
    """One target's row per descriptor: the raw correlation, both
    partials, and their permutation and asymptotic p-values."""
    rows = []
    for name in track(list(desc.NAMES), progress):
        x = table[f"log1p_{name}"].to_numpy(np.float64)
        raw = rank.permutation_p(x, target, alternative="two-sided",
                                 draws=PERMUTATIONS, seed=SEED)
        by_rate = rank.conditional_permutation_p(x, target, base_rate,
                                                 width=BLOCK_WIDTH,
                                                 draws=PERMUTATIONS, seed=SEED)
        row = {**label, "descriptor": name,
               "group": next(g for g, members in desc.GROUPS.items()
                             if name in members),
               "constant_here": bool(np.ptp(x) == 0.0),
               "rho": raw["rho"], "p_permutation": raw["p"],
               "p_asymptotic": rank.asymptotic_p(raw["rho"], len(target)),
               "rho_partial_base_rate": by_rate["rho"],
               "p_partial_base_rate": by_rate["p_conditional"],
               "p_partial_base_rate_asymptotic": by_rate["p_asymptotic"],
               "tau_partial_base_rate": rank.partial_kendall(x, target, base_rate)}
        if name in SCALE:
            row |= {"rho_partial_scale": np.nan, "p_partial_scale": np.nan,
                    "p_partial_scale_asymptotic": np.nan,
                    "tau_partial_scale": np.nan}
        else:
            by_scale = rank.conditional_permutation_p(x, target, scale,
                                                      width=BLOCK_WIDTH,
                                                      draws=PERMUTATIONS, seed=SEED)
            row |= {"rho_partial_scale": by_scale["rho"],
                    "p_partial_scale": by_scale["p_conditional"],
                    "p_partial_scale_asymptotic": by_scale["p_asymptotic"],
                    "tau_partial_scale": rank.partial_kendall(x, target, scale)}
        rows.append(row)
    return rows


def declare(table: pd.DataFrame) -> dict:
    """One Benjamini-Hochberg pass over the whole table plus the
    declaration columns, in place; returns the step-up's reading."""
    fdr = rank.benjamini_hochberg(table["p_permutation"].to_numpy(), FDR_Q)
    table["bh_position"] = fdr["position"]
    table["bh_bar"] = fdr["bar"]
    table["bh_declared"] = fdr["declared"]
    survives = (np.sign(table["rho_partial_scale"]) == np.sign(table["rho"])) \
        & (table["p_partial_scale"] < PARTIAL_LEVEL)
    table["survives_scale_partial"] = survives.fillna(False)
    survives_rate = (np.sign(table["rho_partial_base_rate"]) == np.sign(table["rho"])) \
        & (table["p_partial_base_rate"] < PARTIAL_LEVEL)
    table["survives_base_rate_partial"] = survives_rate.fillna(False)
    table["declared_on_the_scale_partial"] = (table["bh_declared"]
                                              & table["survives_scale_partial"])
    table["declared"] = (table["declared_on_the_scale_partial"]
                         & table["survives_base_rate_partial"])
    table["tracks_arithmetic"] = (table["declared_on_the_scale_partial"]
                                  & ~table["survives_base_rate_partial"])
    return fdr


def screen_metrics(run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    rows = []
    values = exp3_descriptors.descriptor_table(dd.FIT_KEY, done)
    for family in dd.FAMILIES:
        target_table = dd.targets(dd.FIT_KEY, family)
        table = dd.aligned(values, target_table)
        rows += association_rows(
            table, target_table["delta"].to_numpy(np.float64),
            target_table["test_base_rate"].to_numpy(np.float64),
            _design(table, SCALE), {"family": family}, f"screen {family}")

    table = pd.DataFrame(rows)
    fdr = declare(table)
    table = table.set_index(["family", "descriptor"])

    reference = dd.targets(dd.FIT_KEY, dd.FAMILIES[0])
    calibration = {
        "base_rate": rank.partial_size(reference["test_base_rate"].to_numpy(np.float64),
                                       replicates=CALIBRATION_REPLICATES,
                                       draws=CALIBRATION_PERMUTATIONS,
                                       width=BLOCK_WIDTH, seed=SEED),
        "scale": rank.partial_size(_design(dd.aligned(values, reference), SCALE),
                                   replicates=CALIBRATION_REPLICATES,
                                   draws=CALIBRATION_PERMUTATIONS,
                                   width=BLOCK_WIDTH, seed=SEED)}

    declared = {family: sorted(table.loc[family].index[table.loc[family, "declared"]])
                for family in dd.FAMILIES}
    on_scale_only = {
        family: sorted(table.loc[family].index[
            table.loc[family, "declared_on_the_scale_partial"]])
        for family in dd.FAMILIES}
    groups = {family: sorted({group for group, members in desc.GROUPS.items()
                              if group != desc.BASELINE
                              and set(members) & set(names)})
              for family, names in declared.items()}
    return {"dataset_key": dd.FIT_KEY, "k": int(len(reference)),
            "family_size": int(len(table)),
            "bh_declared_count": int(fdr["declared_count"]),
            "bh_threshold": fdr["threshold"],
            "declared": declared,
            "declared_non_scale": {family: [n for n in names if n not in SCALE]
                                   for family, names in declared.items()},
            "declared_groups": groups,
            "declared_on_the_scale_partial_only": {
                family: [n for n in names if n not in SCALE]
                for family, names in on_scale_only.items()},
            "tracks_arithmetic": {
                family: sorted(table.loc[family].index[table.loc[family, "tracks_arithmetic"]])
                for family in dd.FAMILIES},
            "partial_calibration": calibration,
            "seconds": time.time() - t0,
            "artifacts": {"screen": write_csv(run_dir, "screen", table)}}


# ----------------------------------------------------------------- rank --

def rank_config(family: str, done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "rank", "family": family,
            "dataset_key": dd.FIT_KEY, "baseline": desc.BASELINE,
            "statistic": "the paired bank-bootstrap of rho(frozen rule) minus "
                         "rho(scale baseline) on the same out-of-fold predictions",
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED},
            "permutation": {"draws": PERMUTATIONS, "seed": SEED,
                            "alternative": "greater"},
            "criterion": "the bootstrap 5th percentile of the difference exceeds zero",
            "target_resampling": "the same correlations recomputed on each of Experiment 1's "
                                 "1,000 joint-bootstrap replicates of the target",
            "inputs": _inputs((family,), (dd.FIT_KEY,), done),
            "rule_version": rule.VERSION, "rank_version": rank.VERSION}


def rank_metrics(family: str, run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    winner, descriptors = _frozen(family, done)
    selection = _selection_run(family, done)
    oof = pd.read_csv(selection / "artifacts" / "oof.csv",
                      dtype={"bank": str}).set_index("bank")
    candidates = pd.read_csv(selection / "artifacts" / "candidates.csv") \
        .set_index("candidate")
    target_table = dd.targets(dd.FIT_KEY, family)
    oof = oof.loc[target_table.index]
    target = target_table["delta"].to_numpy(np.float64)

    import json as _json
    order = _json.loads((selection / "metrics.json").read_text())["candidate_order"]
    repeats = np.load(selection / "artifacts" / "oof_repeats.npy")
    by_candidate = {name: repeats[i] for i, name in enumerate(order)}

    frozen_prediction = by_candidate[winner]
    baseline_prediction = by_candidate[desc.BASELINE]
    paired = rank.bootstrap_repeat_mean_difference(
        frozen_prediction, baseline_prediction, target,
        draws=BOOTSTRAP_DRAWS, seed=SEED)
    permutation = rank.permutation_p_difference(
        oof[winner].to_numpy(np.float64), oof[desc.BASELINE].to_numpy(np.float64),
        target, draws=PERMUTATIONS, seed=SEED)

    others = [name for name in candidates.index if name != desc.BASELINE]
    best_other = max(others, key=lambda name: float(candidates.loc[name, "score"]))
    other_paired = rank.bootstrap_repeat_mean_difference(
        by_candidate[best_other], baseline_prediction, target,
        draws=BOOTSTRAP_DRAWS, seed=SEED)

    replicates = dd.target_replicates(dd.FIT_KEY, family)
    rows = [{"arm": name,
             "rho": rank.repeat_mean_rho(by_candidate[name], target),
             **{f"target_{key}": value for key, value in
                rank.target_resampling_rho(oof[name].to_numpy(np.float64),
                                           replicates).items()
                if not isinstance(value, list)}}
            for name in dict.fromkeys((winner, desc.BASELINE, best_other))]

    return {"family": family, "dataset_key": dd.FIT_KEY, "k": int(len(target)),
            "frozen_rule": winner, "frozen_descriptors": list(descriptors),
            "estimand": "the mean over repeats of the out-of-fold rank correlation",
            "baseline_rho": rank.repeat_mean_rho(baseline_prediction, target),
            "frozen_rho": rank.repeat_mean_rho(frozen_prediction, target),
            "delta_rho": paired["delta"], "delta_rho_q05": paired["q05"],
            "delta_rho_ci95": paired["ci95"], "delta_rho_sd": paired["sd"],
            "permutation_p": permutation,
            "beats_baseline": bool(paired["q05"] > 0),
            "best_non_baseline": {
                "candidate": best_other, "delta_rho": other_paired["delta"],
                "q05": other_paired["q05"], "ci95": other_paired["ci95"]},
            "frozen_is_baseline": bool(winner == desc.BASELINE),
            "seconds": time.time() - t0,
            "artifacts": {"arms": write_csv(run_dir, "arms",
                                            pd.DataFrame(rows).set_index("arm"))}}


# -------------------------------------------------------------- held-out --

def grade(rho: float, p: float) -> str:
    if not np.isfinite(rho):
        return "not evaluable"
    if rho > 0 and p < ONE_SIDED_LEVEL:
        return "transfers"
    return "directionally consistent, underpowered" if rho > 0 else "killed"


def fitted_rule(family: str, done: dict, key: str,
                descriptors: tuple[str, ...] | None = None) -> tuple[tuple[str, ...],
                                                                     float, np.ndarray]:
    """A candidate's coefficients from the weighted fit on every bank of ``key``;
    the descriptor selection's winner unless another candidate is named."""
    if descriptors is None:
        descriptors = _frozen(family, done)[1]
    target_table = dd.targets(key, family)
    table = dd.aligned(exp3_descriptors.descriptor_table(key, done), target_table)
    design = _design(table, descriptors)
    keep = ~rule.constant_columns(design)
    intercept, coefficients = rule.fit(design, target_table["delta"].to_numpy(np.float64),
                                       dd.weights(target_table), keep)
    return descriptors, intercept, coefficients


def apply_rule(key: str, done: dict, family: str, descriptors: tuple[str, ...],
               intercept: float, coefficients: np.ndarray) -> tuple[pd.DataFrame,
                                                                    np.ndarray]:
    target_table = dd.targets(key, family)
    table = dd.aligned(exp3_descriptors.descriptor_table(key, done), target_table)
    prediction = rule.predict(_design(table, descriptors), intercept, coefficients)
    return target_table, prediction


def holdout_config(family: str, done: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "holdout", "family": family,
            "fit_on": dd.FIT_KEY, "test_on": dd.HOLDOUT_KEY,
            "rule": "the descriptor selection's winner, coefficients frozen from the weighted fit on "
                    "every fitting bank; nothing re-fitted within the held-out banks",
            "beside_it": "the scale-only baseline, fitted and applied the same way, so "
                         "the frozen rule's transfer and the baseline's are read on one "
                         "page",
            "permutation": {"draws": PERMUTATIONS, "seed": SEED,
                            "alternative": "greater"},
            "grading": {"transfers": f"rho > 0 and one-sided p < {ONE_SIDED_LEVEL}",
                        "directionally consistent, underpowered":
                            f"rho > 0, p >= {ONE_SIDED_LEVEL}",
                        "killed": "rho <= 0"},
            "level": ONE_SIDED_LEVEL,
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED},
            "target_resampling": "the same correlation recomputed on each of Experiment 1's "
                                 "joint-bootstrap replicates of the held-out target",
            "power": "44.5 % at a true rho of 0.5, n = 11, one-sided",
            "reverse": "fit on the held-out banks and test on the fitting banks; "
                       "a sanity check that gates nothing",
            "inputs": _inputs((family,), (dd.FIT_KEY, dd.HOLDOUT_KEY), done),
            "rule_version": rule.VERSION, "rank_version": rank.VERSION}


def holdout_metrics(family: str, run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    descriptors, intercept, coefficients = fitted_rule(family, done, dd.FIT_KEY)
    target_table, prediction = apply_rule(dd.HOLDOUT_KEY, done, family,
                                          descriptors, intercept, coefficients)
    target = target_table["delta"].to_numpy(np.float64)
    forward = rank.permutation_p(prediction, target, alternative="greater",
                                 draws=PERMUTATIONS, seed=SEED)
    interval = rank.bootstrap_rho(prediction, target, draws=BOOTSTRAP_DRAWS, seed=SEED)
    resampled = rank.target_resampling_rho(
        prediction, dd.target_replicates(dd.HOLDOUT_KEY, family))

    baseline_descriptors = desc.GROUPS[desc.BASELINE]
    _, baseline_intercept, baseline_coefficients = fitted_rule(
        family, done, dd.FIT_KEY, baseline_descriptors)
    _, baseline_prediction = apply_rule(dd.HOLDOUT_KEY, done, family,
                                        baseline_descriptors, baseline_intercept,
                                        baseline_coefficients)
    baseline = rank.permutation_p(baseline_prediction, target, alternative="greater",
                                  draws=PERMUTATIONS, seed=SEED)

    reverse_descriptors, reverse_intercept, reverse_coefficients = fitted_rule(
        family, done, dd.HOLDOUT_KEY)
    reverse_table, reverse_prediction = apply_rule(dd.FIT_KEY, done, family,
                                                   reverse_descriptors,
                                                   reverse_intercept,
                                                   reverse_coefficients)
    reverse = rank.permutation_p(reverse_prediction,
                                 reverse_table["delta"].to_numpy(np.float64),
                                 alternative="greater", draws=PERMUTATIONS, seed=SEED)

    banks = pd.DataFrame({"prediction": prediction,
                          "baseline_prediction": baseline_prediction, "delta": target,
                          "se": target_table["se"].to_numpy(),
                          "test_base_rate": target_table["test_base_rate"].to_numpy()},
                         index=target_table.index)
    coefficient_rows = [{"term": "intercept", "value": intercept}] + [
        {"term": f"log1p_{name}", "value": float(value)}
        for name, value in zip(descriptors, coefficients)]

    return {"family": family, "fit_on": dd.FIT_KEY, "test_on": dd.HOLDOUT_KEY,
            "k": int(len(target)),
            "frozen_rule": _frozen(family, done)[0],
            "frozen_descriptors": list(descriptors),
            "rho": forward["rho"], "p_one_sided": forward["p"],
            "bootstrap_ci95": interval["ci95"], "bootstrap_q05": interval["q05"],
            "target_resampling": resampled,
            "grade": grade(forward["rho"], forward["p"]),
            "baseline": {"rho": baseline["rho"], "p_one_sided": baseline["p"],
                         "grade": grade(baseline["rho"], baseline["p"]),
                         "descriptors": list(baseline_descriptors)},
            "reverse": {"rho": reverse["rho"], "p_one_sided": reverse["p"],
                        "grade": grade(reverse["rho"], reverse["p"])},
            "seconds": time.time() - t0,
            "artifacts": {"holdout": write_csv(run_dir, "holdout", banks),
                          "coefficients": write_csv(
                              run_dir, "coefficients",
                              pd.DataFrame(coefficient_rows).set_index("term"))}}


# -------------------------------------------------------------- ceiling --

def ceiling_descriptor_config() -> dict:
    return {"experiment": EXPERIMENT, "check": "ceiling_descriptors",
            "dataset_key": dd.FIT_KEY,
            "window": "split == 0 (the training window), same-account rows excluded, "
                      "labels read: these quantities are the yardstick, not a bank's tool",
            "descriptors": {name: lab.DESCRIPTIONS[name] for name in lab.NAMES},
            "groups": {name: list(members) for name, members in lab.GROUPS.items()},
            "adjacency": "two payments are adjacent when they share an account, "
                         "keyed by the (bank, account) pair",
            "homophily": {"primary": "the pooled ratio over the bank's focal "
                                     "payments",
                          "beside_it": "the mean over focal payments of each one's own "
                                       "ratio, which weights a payment's neighbourhood "
                                       "by its size not at all"},
            "rail": {"bank": amlworld.RAIL_BANK,
                     "rule": "the payment rail's accounts are excluded from the "
                             "neighbourhood counts; the rail-included reading is "
                             "computed beside it"},
            "patterns": {"families": list(amlworld.PATTERN_FAMILIES),
                         "chain": list(amlworld.CHAIN_FAMILIES),
                         "join": amlworld.JOIN_FIELDS},
            "patterns_file_sha256": _digest(
                amlworld.RAW / f"{dd.data.DATASETS[dd.FIT_KEY][1]}_Patterns.txt"),
            "roster_and_base_rate_from": {
                "note": "the roster and its test base rates are read from Experiment 1's "
                        "shortfall Run for the tree family; the roster artefact is "
                        "byte-identical across the three families",
                "shortfall": dd.shortfall_hash(dd.FIT_KEY, dd.FAMILIES[0])},
            "ceiling_version": lab.VERSION, "descriptors_version": desc.VERSION,
            "rank_version": rank.VERSION}


def ceiling_descriptor_metrics(run_dir: Path) -> dict:
    from m.features import account_codes

    t0 = time.time()
    df = amlworld.load(dd.data.DATASETS[dd.FIT_KEY][1])
    codes, coverage = amlworld.pattern_family_codes(dd.data.DATASETS[dd.FIT_KEY][1], df)

    training = (df["split"].to_numpy() == dd.TRAINING_SPLIT) & ~df["same_account"].to_numpy()
    rows = np.flatnonzero(training)
    frame = df.iloc[rows]
    payer, payee = account_codes(frame)
    y = frame["is_laundering"].to_numpy(np.int8).astype(np.float64)
    from_bank = frame["from_bank"].astype(str).to_numpy()
    to_bank = frame["to_bank"].astype(str).to_numpy()
    family = codes[rows]
    del df

    rail = (from_bank == amlworld.RAIL_BANK) | (to_bank == amlworld.RAIL_BANK)
    total, laundering_total = lab.neighbourhood(payer, payee, y)
    keep = np.flatnonzero(~rail)
    total_norail, laundering_norail = lab.neighbourhood(payer[keep], payee[keep], y[keep])

    target_table = dd.targets(dd.FIT_KEY, dd.FAMILIES[0])
    n_families = len(amlworld.PATTERN_FAMILIES)
    chain = np.array([amlworld.PATTERN_FAMILIES.index(name)
                      for name in amlworld.CHAIN_FAMILIES])

    records = []
    for bank in track(list(target_table.index), "ceiling descriptors"):
        touches = (from_bank == bank) | (to_bank == bank)
        focal = touches & (y == 1)
        matched = family[focal] >= 0
        focal_norail = focal[keep] & (y[keep] == 1)
        records.append({
            "bank": bank,
            "laundering_count": float(focal.sum()),
            "test_base_rate": float(target_table.loc[bank, "test_base_rate"]),
            "laundering_internal_share": float(
                ((from_bank == bank) & (to_bank == bank) & (y == 1)).sum() / focal.sum())
            if focal.sum() else np.nan,
            "chain_pattern_share": float(np.isin(family[focal][matched], chain).mean())
            if matched.any() else np.nan,
            "class_homophily": lab.homophily(total_norail, laundering_norail,
                                             focal_norail),
            "class_homophily_with_rail": lab.homophily(total, laundering_total, focal),
            "class_homophily_mean_of_ratios": lab.mean_of_ratios(
                total_norail, laundering_norail, focal_norail),
            "pattern_entropy": lab.entropy(family[focal][matched], n_families),
            "pattern_match_share": float(matched.mean()) if focal.sum() else np.nan,
            "mean_neighbourhood": float(total[focal].mean()) if focal.sum() else np.nan,
        })

    table = pd.DataFrame(records).set_index("bank")
    for name in lab.NAMES:
        table[f"log1p_{name}"] = np.log1p(table[name].to_numpy(np.float64))
    table["log1p_class_homophily_with_rail"] = np.log1p(
        table["class_homophily_with_rail"].to_numpy(np.float64))

    summary = {name: {"defined": int(table[name].notna().sum()),
                      "distinct": int(table[name].nunique()),
                      "zero": int((table[name] == 0).sum()),
                      "min": float(table[name].min()),
                      "q25": float(table[name].quantile(0.25)),
                      "median": float(table[name].median()),
                      "q75": float(table[name].quantile(0.75)),
                      "max": float(table[name].max())} for name in lab.NAMES}
    return {"dataset_key": dd.FIT_KEY, "k": int(len(table)),
            "pattern_coverage": coverage,
            "training_laundering_with_a_family": float((family >= 0)[y == 1].mean()),
            "pattern_match_share": {"min": float(table["pattern_match_share"].min()),
                                    "median": float(table["pattern_match_share"].median()),
                                    "max": float(table["pattern_match_share"].max())},
            "rail_payments_dropped": int(rail.sum()),
            "rail_laundering_dropped": int((rail & (y == 1)).sum()),
            "homophily_rank_agreement_with_neighbourhood_size": rank.spearman(
                table["class_homophily_with_rail"].to_numpy(),
                table["mean_neighbourhood"].to_numpy()),
            "descriptor": summary, "seconds": time.time() - t0,
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
                                 "label_using": len(CEILING_CANDIDATES)},
            "protocol": "the descriptor selection's folds, fitting rule, ranking metric and freeze rule, "
                        "unchanged",
            "price": "the best-scoring label-using candidate's mean out-of-fold "
                     "correlation minus the frozen label-free rule's; a ceiling is a "
                     "maximum, so it is read off the best score and not off the "
                     "parsimony-favouring freeze rule, whose own winner is recorded "
                     "beside it",
            "beside_it": "the same reading over the label-using candidates that exclude "
                         "the test base rate, since average precision's chance level "
                         "is that base rate, so the descriptor shares the target's "
                         "arithmetic",
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
                                   "repeat_sd": float(per_repeat.std(ddof=1)),
                                   "predictions": predictions,
                                   "prediction": predictions.mean(axis=0)}
    score = {name: results[name]["score"] for name in CEILING_ORDER}
    best = rule.best_by_score(CEILING_ORDER, score)
    leader = results[best]["predictions"]
    baseline = results[desc.BASELINE]["predictions"]
    paired = {name: rank.bootstrap_repeat_mean_difference(
        results[name]["predictions"], leader, target,
        draws=BOOTSTRAP_DRAWS, seed=SEED)["sd"] for name in CEILING_ORDER}
    against_baseline = {name: rank.bootstrap_repeat_mean_difference(
        results[name]["predictions"], baseline, target,
        draws=BOOTSTRAP_DRAWS, seed=SEED) for name in CEILING_ORDER}
    size = {candidate.name: len(candidate.descriptors) for candidate in CEILING_CANDIDATES}
    frozen = rule.freeze(CEILING_ORDER, score, paired, size, desc.BASELINE,
                         {n: bool(against_baseline[n]["q05"] > 0) for n in CEILING_ORDER})

    without_base_rate = [name for name in CEILING_ORDER
                         if "test_base_rate" not in
                         {c.name: c.descriptors for c in CEILING_CANDIDATES}[name]]
    best_without_base_rate = rule.best_by_score(without_base_rate, score)

    import json
    selection = json.loads((_selection_run(family, done) / "metrics.json").read_text())
    label_free_score = selection["winner_score"]
    rows = [{"candidate": name, "descriptors": " ".join(
        {c.name: c.descriptors for c in CEILING_CANDIDATES}[name]),
             "n_descriptors": size[name], "score": score[name],
             "repeat_sd": results[name]["repeat_sd"], "paired_sd_vs_leader": paired[name],
             "delta_vs_baseline": against_baseline[name]["delta"],
             "q05_vs_baseline": against_baseline[name]["q05"],
             "ceiling": name == best,
             "freeze_winner": name == frozen["winner"]} for name in CEILING_ORDER]
    oof = pd.DataFrame({name: results[name]["prediction"] for name in CEILING_ORDER},
                       index=target_table.index)
    oof.insert(0, "delta", target)

    return {"family": family, "dataset_key": dd.FIT_KEY, "k": int(len(target)),
            "ceiling_rule": best, "ceiling_score": score[best],
            "ceiling_reading": "the best-scoring label-using candidate, since a ceiling "
                               "is a maximum; the freeze rule's own winner is recorded "
                               "beside it and is not the ceiling",
            "ceiling_rule_without_base_rate": best_without_base_rate,
            "ceiling_score_without_base_rate": score[best_without_base_rate],
            "label_free_rule": selection["freeze"]["winner"],
            "label_free_score": label_free_score,
            "price_of_label_freeness": float(score[best] - label_free_score),
            "price_without_base_rate": float(
                score[best_without_base_rate] - label_free_score),
            "candidate_counts": {"label_free": len(exp3_descriptors.CANDIDATES),
                                 "label_using": len(CEILING_CANDIDATES)},
            "freeze": frozen, "seconds": time.time() - t0,
            "artifacts": {"candidates": write_csv(
                run_dir, "candidates", pd.DataFrame(rows).set_index("candidate")),
                "oof": write_csv(run_dir, "oof", oof)}}


# ---------------------------------------------------- the bank-facing rule --

def bank_facing_counts_config() -> dict:
    return {"experiment": EXPERIMENT, "check": "bank_facing_counts",
            "dataset_key": dd.HOLDOUT_KEY,
            "window": "split == 0 (the training window), same-account rows excluded, "
                      "labels read: the count is the bank's own confirmed cases",
            "descriptor": {"laundering_count": lab.DESCRIPTIONS["laundering_count"]},
            "definition": "the ceiling's laundering_count, computed once on "
                          "the held-out dataset so the frozen rule can transfer",
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
            "rule": "scale plus the bank's own training-window laundering-payment "
                    "count: one weighted least-squares fit (weights 1 over s_B "
                    "squared, as the descriptor selection used) on the ceiling Run's 107 rows, inputs "
                    "log(1+payment count), log(1+active accounts), "
                    "log(1+laundering count); the coefficients are frozen to the "
                    "coefficients.csv artefact and applied unchanged to the held-out "
                    "banks",
            "admissibility": "own ledger only holds: every input is the "
                             "bank's own. No labels does not: the count is "
                             "the bank's own confirmed cases, so the label-free rule "
                             "remains the fallback where case counts are "
                             "withheld",
            "presentation": "the rule's output is an ordering, valid within the "
                            "fitted range of 5,198 to 125,014 training-window "
                            "payments per bank; the level is not quoted",
            "descriptors": list(BANK_FACING),
            "permutation": {"draws": PERMUTATIONS, "seed": SEED,
                            "alternative": "greater"},
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED},
            "grading": {"transfers": f"rho > 0 and one-sided p < {ONE_SIDED_LEVEL}",
                        "directionally consistent, underpowered":
                            f"rho > 0, p >= {ONE_SIDED_LEVEL}",
                        "killed": "rho <= 0"},
            "level": ONE_SIDED_LEVEL,
            "target_resampling": "the same correlation recomputed on each of Experiment 1's "
                                 "joint-bootstrap replicates of the held-out target",
            "beside_it": "the frozen label-free rule's held-out reading, from the "
                         "holdout Run, and the ceiling Run's out-of-fold score for "
                         "the same candidate",
            "inputs": {**_inputs((family,), (dd.FIT_KEY, dd.HOLDOUT_KEY), done),
                       "ceiling_descriptors": config_hash(ceiling_descriptor_config()),
                       "bank_facing_counts": config_hash(bank_facing_counts_config()),
                       "holdout": config_hash(holdout_config(family, done)),
                       "ceiling": config_hash(ceiling_config(family, done))},
            "rule_version": rule.VERSION, "rank_version": rank.VERSION,
            "ceiling_version": lab.VERSION}


def bank_facing_metrics(family: str, run_dir: Path, done: dict) -> dict:
    import json

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
    fitted = rule.predict(design, intercept, coefficients)

    payments = table["payment_count"].to_numpy(np.float64)
    fitted_range = (float(payments.min()), float(payments.max()))
    assert fitted_range == FITTED_RANGE, (
        f"the fitting roster's payment-count range is {fitted_range}, not the "
        f"fixed {FITTED_RANGE}; the rows are not the ceiling Run's")

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
    forward = rank.permutation_p(prediction, held, alternative="greater",
                                 draws=PERMUTATIONS, seed=SEED)
    interval = rank.bootstrap_rho(prediction, held, draws=BOOTSTRAP_DRAWS, seed=SEED)
    resampled = rank.target_resampling_rho(
        prediction, dd.target_replicates(dd.HOLDOUT_KEY, family))

    label_free = json.loads(
        (done[config_hash(holdout_config(family, done))] / "metrics.json").read_text())
    ceiling_run = done[config_hash(ceiling_config(family, done))]
    ceiling_metrics_ = json.loads((ceiling_run / "metrics.json").read_text())
    candidates = pd.read_csv(
        verify(ceiling_run, ceiling_metrics_["artifacts"]["candidates"])) \
        .set_index("candidate")

    banks = pd.DataFrame({"laundering_count": holdout_table["laundering_count"].to_numpy(),
                          "prediction": prediction,
                          "label_free_prediction": np.nan, "delta": held,
                          "se": holdout_targets["se"].to_numpy()},
                         index=holdout_targets.index)
    holdout_csv = pd.read_csv(
        verify(done[config_hash(holdout_config(family, done))],
               label_free["artifacts"]["holdout"]),
        dtype={"bank": str}).set_index("bank")
    banks["label_free_prediction"] = holdout_csv.loc[banks.index, "prediction"]
    coefficient_rows = [{"term": "intercept", "value": intercept}] + [
        {"term": f"log1p_{name}", "value": float(value)}
        for name, value in zip(BANK_FACING, coefficients)]

    return {"family": family, "fit_on": dd.FIT_KEY, "test_on": dd.HOLDOUT_KEY,
            "k": int(len(held)),
            "descriptors": list(BANK_FACING),
            "coefficients": {row["term"]: row["value"] for row in coefficient_rows},
            "fitted_range_payment_count": list(fitted_range),
            "fit_rho_in_sample": rank.spearman(fitted, target),
            "fit_rho_in_sample_reading": "the weighted fit read back on its own 107 "
                                         "fitting banks, optimistic by construction; "
                                         "the out-of-fold score below is the honest "
                                         "level",
            "ceiling_oof_score": float(candidates.loc["scale+laundering_count",
                                                      "score"]),
            "rho": forward["rho"], "p_one_sided": forward["p"],
            "bootstrap_ci95": interval["ci95"], "bootstrap_q05": interval["q05"],
            "target_resampling": resampled,
            "grade": grade(forward["rho"], forward["p"]),
            "label_free": {"rule": label_free["frozen_rule"],
                           "rho": label_free["rho"],
                           "p_one_sided": label_free["p_one_sided"],
                           "grade": label_free["grade"]},
            "seconds": time.time() - t0,
            "artifacts": {"holdout": write_csv(run_dir, "holdout", banks),
                          "coefficients": write_csv(
                              run_dir, "coefficients",
                              pd.DataFrame(coefficient_rows).set_index("term"))}}


# ----------------------------------------------------------------- main --

def cmd(stage: str, done: dict) -> None:
    if stage == "inputs":
        metrics = run(EXPERIMENT, inputs_config(done),
                      lambda config, run_dir: inputs_metrics(run_dir, done), done=done)
        print(f"inputs_gate: {metrics['cells']} cells verified, worst standard-error "
              f"difference {metrics['worst_standard_error_difference']:.2e} — "
              f"{'pass' if metrics['pass'] else 'FAIL'}", flush=True)
        if not metrics["pass"]:
            raise SystemExit("the inputs gate failed; no Experiment 3 reading is believed until "
                             "the difference is localised")
    elif stage == "screen":
        metrics = run(EXPERIMENT, screen_config(done),
                      lambda config, run_dir: screen_metrics(run_dir, done), done=done)
        print(f"screen: {metrics['bh_declared_count']} of {metrics['family_size']} "
              f"associations declared under Benjamini-Hochberg; declared after the "
              f"scale partial: "
              f"{ {f: len(v) for f, v in metrics['declared_non_scale'].items()} }",
              flush=True)
    elif stage == "rank":
        for family in dd.FAMILIES:
            metrics = run(EXPERIMENT, rank_config(family, done),
                          lambda config, run_dir, f=family:
                          rank_metrics(f, run_dir, done), done=done)
            print(f"rank {family}: frozen {metrics['frozen_rule']} at rho "
                  f"{metrics['frozen_rho']:+.4f}, baseline {metrics['baseline_rho']:+.4f}, "
                  f"delta {metrics['delta_rho']:+.4f} (q05 {metrics['delta_rho_q05']:+.4f}) "
                  f"— beats baseline: {metrics['beats_baseline']}", flush=True)
    elif stage == "holdout":
        for family in dd.FAMILIES:
            metrics = run(EXPERIMENT, holdout_config(family, done),
                          lambda config, run_dir, f=family:
                          holdout_metrics(f, run_dir, done), done=done)
            print(f"holdout {family}: rho {metrics['rho']:+.4f}, one-sided p "
                  f"{metrics['p_one_sided']:.4f} — {metrics['grade']}; baseline rho "
                  f"{metrics['baseline']['rho']:+.4f}; reverse rho "
                  f"{metrics['reverse']['rho']:+.4f}", flush=True)
    elif stage == "ceiling":
        metrics = run(EXPERIMENT, ceiling_descriptor_config(),
                      lambda config, run_dir: ceiling_descriptor_metrics(run_dir),
                      done=done)
        print(f"ceiling_descriptors: {metrics['k']} banks, pattern coverage "
              f"{metrics['training_laundering_with_a_family']:.3f} of the training "
              f"window's laundering rows, per-bank match share median "
              f"{metrics['pattern_match_share']['median']:.3f}", flush=True)
        for family in dd.FAMILIES:
            out = run(EXPERIMENT, ceiling_config(family, done),
                      lambda config, run_dir, f=family:
                      ceiling_metrics(f, run_dir, done), done=done)
            print(f"ceiling {family}: {out['ceiling_rule']} at rho "
                  f"{out['ceiling_score']:+.4f} against the label-free "
                  f"{out['label_free_score']:+.4f} — price "
                  f"{out['price_of_label_freeness']:+.4f}; without the base rate "
                  f"{out['ceiling_rule_without_base_rate']} at "
                  f"{out['ceiling_score_without_base_rate']:+.4f} "
                  f"(price {out['price_without_base_rate']:+.4f})", flush=True)
    elif stage == "bank_facing":
        metrics = run(EXPERIMENT, bank_facing_counts_config(),
                      lambda config, run_dir: bank_facing_counts_metrics(run_dir),
                      done=done)
        print(f"bank_facing_counts: {metrics['k']} held-out banks, laundering "
              f"counts {metrics['counts']['min']} to {metrics['counts']['max']} "
              f"about a median of {metrics['counts']['median']:.0f}", flush=True)
        done.update(completed(EXPERIMENT))
        for family in dd.FAMILIES:
            out = run(EXPERIMENT, bank_facing_config(family, done),
                      lambda config, run_dir, f=family:
                      bank_facing_metrics(f, run_dir, done), done=done)
            print(f"bank_facing {family}: held-out rho {out['rho']:+.4f}, "
                  f"one-sided p {out['p_one_sided']:.4f} — {out['grade']}; "
                  f"label-free rho {out['label_free']['rho']:+.4f}; "
                  f"out-of-fold {out['ceiling_oof_score']:+.4f}", flush=True)
    else:
        raise SystemExit(f"unknown stage {stage!r}")


STAGES = ("inputs", "screen", "rank", "holdout", "ceiling", "bank_facing")


def main() -> int:
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
