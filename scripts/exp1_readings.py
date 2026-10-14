#!/usr/bin/env python
"""Experiment 1's robustness readings beside the verdict, on HI-Medium
(Chapter 4; Appendix C).

Three analysis-only Runs read Experiment 1's persisted shortfall artefacts.
``residual_after_size`` fits the weighted line of the shortfall on log size
and reports tau about it (tau is the between-bank standard deviation of the
true shortfall the error bars imply, by the DerSimonian-Laird moment
estimator); ``paired_vs_naive`` sets the paired shortfall beside each
scenario read at its own best configuration; ``roster_fifty`` repeats the
three tests on the banks with at least fifty laundering payments in their test
set. Every config cites its inputs' Run hashes, so a re-done input re-keys the
analysis.

  uv run python scripts/exp1_readings.py residual
  uv run python scripts/exp1_readings.py paired
  uv run python scripts/exp1_readings.py fifty
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
from m.runlog import config_hash  # noqa: E402
from m.stats import heterogeneity, spearman  # noqa: E402

KEY = exp1.PRIMARY_KEY
FAMILIES = ("tree", "gnn")
PAIRED_FAMILIES = ("tree", "blind")
SIZE_REGRESSOR = "natural log of the bank's test-window payments (test_rows)"
FIFTY = 50


def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def tau_dl(deltas: np.ndarray, ses: np.ndarray) -> float:
    """DerSimonian-Laird moment estimator of the between-bank standard
    deviation: the spread of the true shortfalls that would make the
    observed Q equal its expectation under that spread."""
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
    I-squared (k - 2 degrees of freedom) and tau (the moment estimator
    generalised to the regression) alongside the fit and the residuals."""
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
                         "of freedom; residual tau by the DerSimonian-Laird "
                         "moment estimator generalised to the regression",
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
                    "tau": tau_dl(deltas, ses),
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


# ------------------------------------------------------ paired vs naive --

def paired_config(inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": exp1.EXPERIMENT, "check": "paired_vs_naive",
            "world": world, "dataset": dataset, "dataset_key": KEY,
            "families": list(PAIRED_FAMILIES),
            "reading": "paired against own best",
            "own_best": "each arm read at its own validation-selected grid "
                        "setting (the prior work's design), from the "
                        "own_best_delta column Experiment 1 recorded",
            "note": "the own-best spread is read with the paired delta's "
                    "s_B, the only recorded wobble; the GNN family has no "
                    "grid and is out of scope",
            "weights": "1/s_B^2 from each family's Experiment 1 shortfall banks.csv",
            "inputs": {family: _hash_of(inputs[family][0])
                       for family in PAIRED_FAMILIES}}


def paired_metrics(inputs: dict, run_dir: Path) -> dict:
    readings = {}
    for family in PAIRED_FAMILIES:
        path, metrics = inputs[family]
        table = banks_table(path, metrics)
        paired = table["delta"].to_numpy()
        naive = table["own_best_delta"].to_numpy()
        ses = table["se"].to_numpy()
        het_paired = heterogeneity(paired, ses)
        het_naive = heterogeneity(naive, ses)
        readings[family] = {
            "k": het_paired["k"],
            "weighted_mean_paired": het_paired["weighted_mean"],
            "weighted_mean_own_best": het_naive["weighted_mean"],
            "I2_paired": het_paired["I2"], "I2_own_best": het_naive["I2"],
            "sign_flips": int((np.sign(paired) != np.sign(naive)).sum()),
            "rank_agreement_spearman": spearman(paired, naive)}
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


# ------------------------------------------------------- roster at fifty --

def fifty_config(inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": exp1.EXPERIMENT, "check": "roster_fifty",
            "world": world, "dataset": dataset, "dataset_key": KEY,
            "families": list(FAMILIES), "threshold": FIFTY,
            "roster_threshold": data.MIN_TEST_POS,
            "reading": "at least fifty laundering payments",
            "statistics": "I2, Q, the assumption-free null, leave-one-out "
                          "and tau, recomputed on the banks holding at least "
                          "fifty test-window laundering payments, from the "
                          "persisted per-bank deltas and replicates",
            "inputs": {family: _hash_of(path)
                       for family, (path, _) in inputs.items()}}


def fifty_metrics(inputs: dict, run_dir: Path) -> dict:
    readings = {}
    for family, (path, metrics) in inputs.items():
        table = banks_table(path, metrics)
        reps = np.load(verify(path, metrics["artifacts"]["replicates"])
                       ).astype(np.float64)
        keep = (table["test_pos"] >= FIFTY).to_numpy()
        labels = list(table.index[keep])
        deltas = table["delta"].to_numpy()[keep]
        ses = table["se"].to_numpy()[keep]
        seed_var = (table["seed_var"].to_numpy()[keep]
                    if "seed_var" in table else None)
        reading = exp1_analysis.heterogeneity_reading(
            labels, deltas, ses, reps[:, keep], seed_var)
        at30 = metrics["primary"]
        readings[family] = {
            "n_banks": int(keep.sum()), "n_banks_at_thirty": int(len(table)),
            "at_fifty": {"Q": reading["Q"], "I2": reading["I2"],
                         "band": reading["band"],
                         "tau": tau_dl(deltas, ses),
                         "weighted_mean": reading["weighted_mean"],
                         "null_passes": reading["null_check"]["passes"],
                         "null_q95": reading["null_check"]["null_q95"],
                         "loo_I2": reading["leave_one_out"]["I2"]},
            "at_thirty": {"Q": at30["Q"], "I2": at30["I2"],
                          "tau": tau_dl(table["delta"].to_numpy(),
                                        table["se"].to_numpy()),
                          "weighted_mean": at30["weighted_mean"],
                          "null_passes": at30["null_check"]["passes"],
                          "loo_I2": at30["leave_one_out"]["I2"]}}
    return {"dataset_key": KEY, "threshold": FIFTY, "readings": readings}


def cmd_fifty(done: dict | None = None) -> dict:
    done = done if done is not None else completed(exp1.EXPERIMENT)
    inputs = shortfall_inputs(done, FAMILIES)
    metrics = run(exp1.EXPERIMENT, fifty_config(inputs),
                  lambda config, run_dir: fifty_metrics(inputs, run_dir),
                  done=done)
    for family, r in metrics["readings"].items():
        f, t = r["at_fifty"], r["at_thirty"]
        print(f"fifty {family}: {r['n_banks']} of {r['n_banks_at_thirty']} "
              f"banks; I2 {t['I2']:.3f} -> {f['I2']:.3f}, tau {t['tau']:.4f} "
              f"-> {f['tau']:.4f}, null {'pass' if f['null_passes'] else 'fail'}, "
              f"loo I2 {f['loo_I2']:.3f}", flush=True)
    return metrics


# ----------------------------------------------------------------- main --

def cmd_all() -> int:
    done = completed(exp1.EXPERIMENT)
    cmd_residual(done)
    cmd_paired(done)
    cmd_fifty(done)
    return 0


def cmd_status() -> int:
    done = completed(exp1.EXPERIMENT)
    for check, config_fn, families in (
            ("residual_after_size", residual_config, FAMILIES),
            ("paired_vs_naive", paired_config, PAIRED_FAMILIES),
            ("roster_fifty", fifty_config, FAMILIES)):
        try:
            inputs = shortfall_inputs(done, families)
        except SystemExit as missing:
            print(f"{check}: inputs incomplete ({missing})")
            continue
        state = "analysed" if config_hash(config_fn(inputs)) in done else "ready"
        print(f"{check}: {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 2
    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    cmd = sys.argv[1]
    if cmd == "residual":
        return 0 if cmd_residual() else 1
    if cmd == "paired":
        return 0 if cmd_paired() else 1
    if cmd == "fifty":
        return 0 if cmd_fifty() else 1
    if cmd == "all":
        return cmd_all()
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
