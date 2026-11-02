#!/usr/bin/env python
"""Experiment 3's readings beside the selection (Chapter 5; Appendix D).

Analysis only: no detector trains. The only fits are the weighted
least-squares regression models of Experiment 3 on the 107 recorded
shortfalls of HI-Medium, so every number is a function of the
``descriptors`` and ``selection`` Runs of ``exp3_descriptors.py``, the
``ceiling_descriptors`` and ``ceiling`` Runs of ``exp3.py``, Experiment 1's
shortfall Runs (each verified by digest) and, for the recurrence threshold
alone, Experiment 1's cached per-bank ledgers. Two Runs, both under
experiments/exp3, for the tree family, the GNN family and the payment-only
tree:

  properties   the models on one count of scale, the coefficients of the
               scale model, the share of recurring counterparties at other
               thresholds, the rank-correlation matrix of the twelve
               properties and how far each varies across the banks
  uncertainty  every model's difference from the scale model under four
               redraws: the banks with the recorded predictions, Experiment
               1's replicates of the shortfalls, the two jointly, and the
               banks with every model fitted again on each redraw; the
               agreement between replicates and the overlap of the banks'
               intervals by distance in rank

  uv run python scripts/exp3_readings.py properties
  uv run python scripts/exp3_readings.py uncertainty
  uv run python scripts/exp3_readings.py all
  uv run python scripts/exp3_readings.py status

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
from m import completed, config_hash, run  # noqa: E402
from m import descriptors as desc  # noqa: E402
from m import rank, rule  # noqa: E402
from m.artifacts import verify, write_csv  # noqa: E402
from m.runlog import find, index  # noqa: E402

EXPERIMENT = "exp3"
KEY = dd.FIT_KEY
FAMILIES = dd.FAMILIES
SEED = 0
DRAWS = 1000
THRESHOLDS = (2, 3, 5, 10, 20)
RANK_DISTANCE_BINS = ((1, 10), (11, 30), (31, 60), (61, 106))

PROPS = desc.NAMES
SCALE = desc.GROUPS[desc.BASELINE]
OTHERS = tuple(name for name in PROPS if name not in SCALE)
CONFIRMED = "scale+laundering_count"
CASES_INTERNAL = "scale+laundering_internal_share"
EXTRA_MODELS = {CONFIRMED: SCALE + ("laundering_count",),
                CASES_INTERNAL: SCALE + ("laundering_internal_share",),
                "all_twelve": PROPS,
                "scale_squares_and_product": SCALE + ("sq_payments", "sq_accounts",
                                                      "prod_scale")}
PROPERTY_MODELS = tuple(f"scale+{name}" for name in OTHERS)

SPEARMAN = "Spearman rho over banks, average ranks (src/m/rank.py spearman)"
MODEL = ("weighted least squares at 1/s_B^2 on log(1 + v) plus an intercept, "
         "ten folds repeated ten times (src/m/rule.py out_of_fold); the score "
         "is the mean over repeats of the out-of-fold Spearman rho")
BANK_REDRAW = ("1,000 paired redraws of the 107 banks at default_rng(0), one "
               "draw shared by every model, each model's out-of-fold "
               "predictions held as recorded (src/m/rank.py "
               "bootstrap_repeat_mean_difference)")
SHORTFALL_REDRAW = ("each difference recomputed on each of Experiment 1's 1,000 "
                    "joint-bootstrap replicates of the shortfalls, the "
                    "predictions held fixed")
JOINT_REDRAW = ("draw i resamples the banks at a fresh default_rng(0) and "
                "reads Experiment 1's replicate i on them, the predictions held fixed")
REFIT_REDRAW = ("1,000 redraws of the banks at a fresh default_rng(0), sorted; "
                "on each the ten-fold, ten-repeat cross-validation is run "
                "again with the folds cut over the distinct banks of the draw, "
                "so that copies of a bank share a fold")
REPLICATE_AGREEMENT = ("Spearman rho between replicates 2j and 2j + 1 "
                       "(500 pairs), and between each replicate and the "
                       "measured shortfalls")
OVERLAP = ("a pair of banks overlaps when the two 95 % intervals of Experiment "
           "1's shortfall Run share a point; distance in rank is the difference "
           "of the two banks' ordinal ranks of the measured shortfall")
RECURRENCE = ("share of distinct external counterparty accounts appearing "
              "on k or more training-window rows, same-account and internal "
              "rows excluded (src/m/descriptors.py compute, threshold k)")
SCALE_PAIR = ("full-fit weighted least squares of the shortfall on log(1 + "
              "payments) and log(1 + active accounts); payments per account "
              "is the ratio of the two raw counts")
SUMMARY = ("median, 2.5th, 5th, 95th and 97.5th percentiles, share above "
           "zero and standard deviation of the finite values")


# --------------------------------------------------------------- inputs --

def _status_hash(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def _metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


def _artifact(path: Path, name: str) -> Path:
    return verify(path, _metrics(path)["artifacts"][name])


def inputs() -> dict:
    done, runs = completed(EXPERIMENT), index(EXPERIMENT)
    found = {"descriptors": done.get(config_hash(exp3_descriptors.descriptor_config(KEY))),
             "selection": {f: done.get(config_hash(exp3_descriptors.selection_config(f)))
                           for f in FAMILIES}}
    if found["descriptors"] is None or None in found["selection"].values():
        raise SystemExit("exp3: no descriptors or selection Run; run "
                         "scripts/exp3_descriptors.py first")
    return found | {
        "ceiling_descriptors": find(runs, check="ceiling_descriptors"),
        "ceiling": {f: find(runs, check="ceiling", family=f) for f in FAMILIES},
        "shortfall": {f: dd.shortfall_run(KEY, f) for f in FAMILIES}}


def _input_hashes(found: dict) -> dict:
    return {"descriptors": _status_hash(found["descriptors"]),
            "selection": {f: _status_hash(p) for f, p in found["selection"].items()},
            "ceiling_descriptors": _status_hash(found["ceiling_descriptors"]),
            "ceiling": {f: _status_hash(p) for f, p in found["ceiling"].items()},
            "shortfall": {f"{KEY}/{f}": _status_hash(p)
                          for f, p in found["shortfall"].items()}}


def descriptor_table(found: dict) -> pd.DataFrame:
    """Raw and log1p property values, the two quantities computed from labels
    beside them, and the curved-scale columns."""
    d = pd.read_csv(_artifact(found["descriptors"], "descriptors"),
                    dtype={"bank": str}).set_index("bank")
    c = pd.read_csv(_artifact(found["ceiling_descriptors"], "ceiling_descriptors"),
                    dtype={"bank": str}).set_index("bank")
    table = d.join(c[["log1p_laundering_count", "log1p_laundering_internal_share"]])
    lp, la = table["log1p_payment_count"], table["log1p_active_accounts"]
    return table.assign(sq_payments=lp ** 2, sq_accounts=la ** 2, prod_scale=lp * la)


def design(table: pd.DataFrame, names: tuple[str, ...]) -> np.ndarray:
    columns = [n if n.startswith(("sq_", "prod_")) else f"log1p_{n}" for n in names]
    return table[columns].to_numpy(np.float64)


def family_target(family: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    target = dd.targets(KEY, family)
    return target, target["delta"].to_numpy(np.float64), dd.weights(target)


def aligned(frame: pd.DataFrame, family: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """``frame``'s rows in the family's shortfall order, with its target and weights."""
    target, y, w = family_target(family)
    return frame.loc[target.index], y, w


# ------------------------------------------------------------ summaries --

def summary(values: np.ndarray) -> dict:
    v = values[np.isfinite(values)]
    return {"median": float(np.median(v)), "q025": float(np.percentile(v, 2.5)),
            "q05": float(np.percentile(v, 5)), "q95": float(np.percentile(v, 95)),
            "q975": float(np.percentile(v, 97.5)), "share_gt0": float((v > 0).mean()),
            "sd": float(v.std(ddof=1))}


def bank_redraw(preds: dict[str, np.ndarray], base: str, y: np.ndarray) -> dict:
    """Every model's difference from ``base`` on the same 1,000 bank redraws
    at default_rng(SEED), as bootstrap_repeat_mean_difference draws them."""
    rng = np.random.default_rng(SEED)
    names = [n for n in preds if n != base]
    values = {n: np.empty(DRAWS) for n in names}
    for i in range(DRAWS):
        idx = rng.integers(0, len(y), len(y))
        b = rank.repeat_mean_rho(preds[base][:, idx], y[idx])
        for n in names:
            values[n][i] = rank.repeat_mean_rho(preds[n][:, idx], y[idx]) - b
    base_score = rank.repeat_mean_rho(preds[base], y)
    return {n: {"score": rank.repeat_mean_rho(preds[n], y),
                "delta": rank.repeat_mean_rho(preds[n], y) - base_score,
                **summary(values[n]),
                "positive": bool(np.percentile(values[n], 5) > 0)}
            for n in names}


# ----------------------------------------------------------- properties --

def properties_config(found: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "properties",
            "dataset_key": KEY, "seed": SEED, "draws": DRAWS,
            "thresholds": list(THRESHOLDS), "ledgers": "data/interim/exp1/ledgers/HI-Medium",
            "spearman": SPEARMAN, "model": MODEL, "bank_redraw": BANK_REDRAW,
            "recurrence": RECURRENCE, "scale_pair": SCALE_PAIR,
            "one_count": ("payments alone and accounts alone against scale; "
                          "and payments plus each other property against "
                          "payments alone, on the bank redraw"),
            "inputs": _input_hashes(found),
            "versions": {"descriptors": desc.VERSION, "rule": rule.VERSION,
                         "rank": rank.VERSION}}


def scale_pair_facts(table: pd.DataFrame) -> dict:
    per_account = (table["payment_count"] / table["active_accounts"]).to_numpy()
    out = {"rho_payments_accounts": rank.spearman(table["payment_count"].to_numpy(),
                                                  table["active_accounts"].to_numpy()),
           "payments_per_account": {"min": float(per_account.min()),
                                    "median": float(np.median(per_account)),
                                    "max": float(per_account.max())}}
    for family in FAMILIES:
        T, y, w = aligned(table, family)
        ratio = (T["payment_count"] / T["active_accounts"]).to_numpy()
        intercept, coefficients = rule.fit(design(T, SCALE), y, w, np.ones(2, bool))
        out[family] = {"intercept": intercept, "log_payments": float(coefficients[0]),
                       "log_accounts": float(coefficients[1]),
                       "rho_per_account_shortfall": rank.spearman(ratio, y),
                       "partial_per_account_given_payments": rank.partial_spearman(
                           ratio, y, design(T, ("payment_count",)))["rho"]}
    return out


def one_count_facts(table: pd.DataFrame) -> dict:
    out = {}
    for family in FAMILIES:
        T, y, w = aligned(table, family)
        fit = {name: rule.out_of_fold(design(T, names), y, w)
               for name, names in {"scale": SCALE, "payments": ("payment_count",),
                                   "accounts": ("active_accounts",)}.items()}
        against_payments = {"payments": fit["payments"],
                            **{f"payments+{n}": rule.out_of_fold(
                                design(T, ("payment_count", n)), y, w)
                               for n in PROPS[1:]}}
        out[family] = {"against_scale": bank_redraw(fit, "scale", y),
                       "scale_score": rank.repeat_mean_rho(fit["scale"], y),
                       "against_payments": bank_redraw(against_payments, "payments", y),
                       "payments_score": rank.repeat_mean_rho(fit["payments"], y)}
    return out


def recurrence_shares(banks: pd.Index) -> pd.DataFrame:
    def shares(bank: str) -> list[float]:
        ledger = dd.bank_ledger(KEY, bank)
        internal = ledger.payer_is_own & ledger.payee_is_own
        external = np.where(ledger.payer_is_own, ledger.payee, ledger.payer)[~internal]
        counts = np.unique(external, return_counts=True)[1]
        return [float((counts >= k).mean()) for k in THRESHOLDS]

    return pd.DataFrame([shares(b) for b in banks], index=banks,
                        columns=[f"k{k}" for k in THRESHOLDS])


def recurrence_facts(table: pd.DataFrame, shares: pd.DataFrame) -> dict:
    between = {f"k{a}": {f"k{b}": rank.spearman(shares[f"k{a}"].to_numpy(),
                                                shares[f"k{b}"].to_numpy())
                         for b in THRESHOLDS} for a in THRESHOLDS}
    off_diagonal = [between[f"k{a}"][f"k{b}"] for i, a in enumerate(THRESHOLDS)
                    for b in THRESHOLDS[i + 1:]]
    out = {"reproduces_recorded_k5": float(np.abs(
               shares["k5"].to_numpy() - table["external_recurrence"].to_numpy()).max()),
           "median_share": {f"k{k}": float(shares[f"k{k}"].median()) for k in THRESHOLDS},
           "rho_between_thresholds": between,
           "rho_between_thresholds_range": [min(off_diagonal), max(off_diagonal)]}
    for family in FAMILIES:
        T, y, w = aligned(table, family)
        S, _, _ = aligned(shares, family)
        scale = design(T, SCALE)
        preds = {"scale": rule.out_of_fold(scale, y, w)}
        for k in THRESHOLDS:
            x = desc.transform(S[f"k{k}"].to_numpy())
            preds[f"k{k}"] = rule.out_of_fold(np.column_stack([scale, x]), y, w)
        diff = bank_redraw(preds, "scale", y)
        rows = {f"k{k}": {"raw_rho": rank.spearman(S[f"k{k}"].to_numpy(), y),
                          "partial_rho_given_scale": rank.partial_spearman(
                              S[f"k{k}"].to_numpy(), y, scale)["rho"],
                          **diff[f"k{k}"]}
                for k in THRESHOLDS}
        partials = [r["partial_rho_given_scale"] for r in rows.values()]
        out[family] = {**rows,
                       "partial_range": [min(partials), max(partials)],
                       "every_q05_below_zero": all(r["q05"] < 0 for r in rows.values())}
    return out


def property_matrix(table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([[rank.spearman(table[a].to_numpy(), table[b].to_numpy())
                          for b in PROPS] for a in PROPS], index=PROPS, columns=PROPS)


def variation_facts(table: pd.DataFrame) -> dict:
    def one(v: np.ndarray) -> dict:
        values, counts = np.unique(v, return_counts=True)
        return {"distinct": int(len(values)), "modal_value": float(values[counts.argmax()]),
                "modal_count": int(counts.max()), "min": float(v.min()),
                "q1": float(np.percentile(v, 25)), "median": float(np.median(v)),
                "q3": float(np.percentile(v, 75)), "max": float(v.max())}

    out = {n: one(table[n].to_numpy()) for n in PROPS}
    out["boundary_share_exactly_one"] = int((table["boundary_share"] == 1.0).sum())
    out["external_encounters_values"] = sorted(float(v) for v in
                                               table["external_encounters"].unique())
    return out


def properties_metrics(found: dict, run_dir: Path) -> dict:
    t0 = time.time()
    banks = dd.targets(KEY, FAMILIES[0]).index
    table = descriptor_table(found).loc[banks]
    shares = recurrence_shares(banks)
    matrix = property_matrix(table)
    return {"dataset_key": KEY, "k": len(banks),
            "scale_pair": scale_pair_facts(table),
            "one_count": one_count_facts(table),
            "recurrence_thresholds": recurrence_facts(table, shares),
            "property_rho": matrix.to_dict(),
            "variation": variation_facts(table),
            "artifacts": {"property_rho": write_csv(run_dir, "property_rho", matrix),
                          "recurrence_shares": write_csv(run_dir, "recurrence_shares",
                                                         shares)},
            "seconds": time.time() - t0}


# ---------------------------------------------------------- uncertainty --

def uncertainty_config(found: dict) -> dict:
    return {"experiment": EXPERIMENT, "check": "uncertainty",
            "dataset_key": KEY, "seed": SEED, "draws": DRAWS,
            "models": {"recorded": "the sixteen models of the selection Run, read "
                                   "from oof_repeats.npy and reproduced",
                       **{n: list(v) for n, v in EXTRA_MODELS.items()}},
            "curved_scale": "sq_payments, sq_accounts: squares of the two log "
                            "counts; prod_scale: their product",
            "spearman": SPEARMAN, "model": MODEL, "bank_redraw": BANK_REDRAW,
            "shortfall_redraw": SHORTFALL_REDRAW, "joint_redraw": JOINT_REDRAW,
            "refit_redraw": REFIT_REDRAW, "replicate_agreement": REPLICATE_AGREEMENT,
            "overlap": OVERLAP, "rank_distance_bins": [list(b) for b in RANK_DISTANCE_BINS],
            "summary": SUMMARY, "inputs": _input_hashes(found),
            "versions": {"descriptors": desc.VERSION, "rule": rule.VERSION,
                         "rank": rank.VERSION}}


def model_designs(table: pd.DataFrame, order: list[str]) -> dict[str, np.ndarray]:
    candidates = {c.name: c.descriptors for c in exp3_descriptors.CANDIDATES}
    return {n: design(table, names)
            for n, names in {**{n: candidates[n] for n in order}, **EXTRA_MODELS}.items()}


def shortfall_redraw(preds: dict[str, np.ndarray], reps: np.ndarray) -> dict:
    base = np.array([rank.repeat_mean_rho(preds["scale"], r) for r in reps])
    return {"scale_rho": summary(base),
            **{n: summary(np.array([rank.repeat_mean_rho(p, r) for r in reps]) - base)
               for n, p in preds.items() if n != "scale"}}


def joint_redraw(preds: dict[str, np.ndarray], reps: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    names = [n for n in preds if n != "scale"]
    values = {n: np.empty(DRAWS) for n in names}
    k = reps.shape[1]
    for i in range(DRAWS):
        idx = rng.integers(0, k, k)
        b = rank.repeat_mean_rho(preds["scale"][:, idx], reps[i][idx])
        for n in names:
            values[n][i] = rank.repeat_mean_rho(preds[n][:, idx], reps[i][idx]) - b
    return {n: summary(v) for n, v in values.items()}


def grouped_out_of_fold(X: np.ndarray, y: np.ndarray, w: np.ndarray,
                        group: np.ndarray, repeat: int) -> np.ndarray:
    uniq = np.unique(group)
    ids = rule.fold_ids(len(uniq), repeat)[np.searchsorted(uniq, group)]
    keep = ~rule.constant_columns(X)
    pred = np.empty(len(y))
    for fold in range(rule.N_FOLDS):
        held = ids == fold
        intercept, coefficients = rule.fit(X[~held], y[~held], w[~held], keep)
        pred[held] = rule.predict(X[held], intercept, coefficients)
    return pred


def refit_redraw(designs: dict[str, np.ndarray], y: np.ndarray, w: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    values = {n: np.empty(DRAWS) for n in designs}
    for i in range(DRAWS):
        idx = np.sort(rng.integers(0, len(y), len(y)))
        for n, X in designs.items():
            values[n][i] = np.mean([rank.spearman(
                grouped_out_of_fold(X[idx], y[idx], w[idx], idx, r), y[idx])
                for r in range(rule.REPEATS)])
    return {"scale_rho": summary(values["scale"]),
            **{n: summary(v - values["scale"]) for n, v in values.items() if n != "scale"}}


def replicate_agreement(reps: np.ndarray, y: np.ndarray) -> dict:
    return {"replicate_vs_replicate": summary(np.array(
                [rank.spearman(reps[i], reps[i + 1]) for i in range(0, len(reps), 2)])),
            "replicate_vs_measured": summary(np.array(
                [rank.spearman(r, y) for r in reps]))}


def overlap_facts(target: pd.DataFrame) -> dict:
    y = target["delta"].to_numpy()
    lo, hi = target["ci_lo"].to_numpy(), target["ci_hi"].to_numpy()
    ranks = np.argsort(np.argsort(y))
    upper = np.triu_indices(len(y), 1)
    overlaps = ((lo[:, None] <= hi[None, :]) & (lo[None, :] <= hi[:, None]))[upper]
    distance = np.abs(ranks[:, None] - ranks[None, :])[upper]
    return {"pairs": int(len(overlaps)), "share": float(overlaps.mean()),
            "by_rank_distance": {
                f"{a}_to_{b}": {"pairs": int(((distance >= a) & (distance <= b)).sum()),
                                "share": float(overlaps[(distance >= a)
                                                        & (distance <= b)].mean())}
                for a, b in RANK_DISTANCE_BINS}}


def property_bounds(reading: dict) -> dict:
    """Over the ten scale+property models: the largest difference, the
    largest 95th percentile, and which (if any) clear zero at the 5th."""
    largest = max(PROPERTY_MODELS, key=lambda n: reading[n].get("delta", reading[n]["median"]))
    widest = max(PROPERTY_MODELS, key=lambda n: reading[n]["q95"])
    return {"largest_difference": {"model": largest, **reading[largest]},
            "largest_q95": {"model": widest, "q95": reading[widest]["q95"]},
            "q05_above_zero": [n for n in PROPERTY_MODELS if reading[n]["q05"] > 0],
            "any_model_q05_above_zero": [n for n in reading
                                         if n != "scale_rho" and reading[n]["q05"] > 0]}


def recorded_scores(found: dict, family: str) -> dict:
    ceiling = pd.read_csv(_artifact(found["ceiling"][family], "candidates")
                          ).set_index("candidate")["score"]
    return {n: float(ceiling[n]) for n in (CONFIRMED, CASES_INTERNAL)}


def uncertainty_family(found: dict, table: pd.DataFrame, family: str) -> tuple[dict, pd.DataFrame]:
    target, y, w = family_target(family)
    reps = dd.target_replicates(KEY, family).astype(np.float64)
    T, _, _ = aligned(table, family)
    selection = found["selection"][family]
    order = _metrics(selection)["candidate_order"]
    recorded = np.load(_artifact(selection, "oof_repeats"))
    designs = model_designs(T, order)
    preds = {n: rule.out_of_fold(X, y, w) for n, X in designs.items()}
    recorded_ceiling = recorded_scores(found, family)

    banks = bank_redraw(preds, "scale", y)
    shortfalls = shortfall_redraw(preds, reps)
    joint = joint_redraw(preds, reps)
    refit = refit_redraw(designs, y, w)
    rows = pd.DataFrame({n: {"score": banks[n]["score"], "delta": banks[n]["delta"],
                             **{f"banks_{s}": banks[n][s] for s in ("q025", "q05", "q95", "q975")},
                             **{f"shortfalls_{s}": shortfalls[n][s] for s in ("median", "q05", "q95")},
                             **{f"joint_{s}": joint[n][s] for s in ("q05", "q95", "share_gt0")},
                             **{f"refit_{s}": refit[n][s] for s in ("median", "q05", "q95")}}
                         for n in banks}).T
    rows.index.name = "model"
    return {"k": len(y), "scale_score": rank.repeat_mean_rho(preds["scale"], y),
            "reproduces_recorded_oof": float(max(np.abs(preds[n] - recorded[i]).max()
                                                 for i, n in enumerate(order))),
            "reproduces_recorded_ceiling_score": {
                n: rank.repeat_mean_rho(preds[n], y) - s for n, s in recorded_ceiling.items()},
            "banks": banks, "banks_bounds": property_bounds(banks),
            "shortfalls": shortfalls, "shortfalls_bounds": property_bounds(shortfalls),
            "joint": joint, "joint_bounds": property_bounds(joint),
            "refit": refit, "refit_bounds": property_bounds(refit),
            "replicate_agreement": replicate_agreement(reps, y),
            "overlap": overlap_facts(target)}, rows.assign(family=family)


def uncertainty_metrics(found: dict, run_dir: Path) -> dict:
    t0 = time.time()
    table = descriptor_table(found)
    readings = {f: uncertainty_family(found, table, f) for f in FAMILIES}
    differences = pd.concat([rows for _, rows in readings.values()])
    return {"dataset_key": KEY, **{f: r for f, (r, _) in readings.items()},
            "artifacts": {"differences": write_csv(run_dir, "differences", differences)},
            "seconds": time.time() - t0}


# ----------------------------------------------------------------- main --

READINGS = {"properties": (properties_config, properties_metrics),
            "uncertainty": (uncertainty_config, uncertainty_metrics)}


def cmd_run(name: str) -> None:
    config_fn, metrics_fn = READINGS[name]
    found = inputs()
    run(EXPERIMENT, config_fn(found), lambda config, run_dir: metrics_fn(found, run_dir))
    print(f"{name}: recorded", flush=True)


def cmd_status() -> int:
    try:
        found = inputs()
    except SystemExit as missing:
        print(f"inputs incomplete ({missing})")
        return 0
    done = completed(EXPERIMENT)
    for name, (config_fn, _) in READINGS.items():
        state = "analysed" if config_hash(config_fn(found)) in done else "ready"
        print(f"{name}: {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    cmd = sys.argv[1]
    if cmd in READINGS:
        cmd_run(cmd)
        return 0
    if cmd == "all":
        for name in READINGS:
            cmd_run(name)
        return 0
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
