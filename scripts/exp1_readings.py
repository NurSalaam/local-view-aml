#!/usr/bin/env python
"""Experiment 1's readings beside the verdict.

One ``shortfall_shuffled`` Run per family reads the shuffled-bank-label
control: the shuffled solo Runs against Experiment 1's unchanged global Runs, through
the same joint bootstrap, reporting I-squared, tau (the between-bank
standard deviation of the true shortfall the error bars imply, by the
DerSimonian-Laird moment estimator) and the size gradient beside the
observed readings. Three analysis-only Runs read Experiment 1's persisted shortfall
artefacts: ``residual_after_size`` (declared post hoc in its config, the
disclosure permanent), ``paired_vs_naive`` and ``roster_fifty``. Every
config cites its inputs' Run hashes, so a re-done input re-keys the
analysis.

  uv run python scripts/exp1_readings.py shuffled <family>
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
import exp1_shuffle  # noqa: E402
from m import completed, run  # noqa: E402
from m.artifacts import verify, write_array, write_csv  # noqa: E402
from m.runlog import config_hash  # noqa: E402
from m.stats import (aggregate_shortfalls, heterogeneity, joint_bootstrap,  # noqa: E402
                     spearman)

KEY = exp1_shuffle.KEY
SMOKE = exp1.SMOKE
FAMILIES = ("tree", "blind", "gnn")
REPLICATES = exp1_analysis.REPLICATES
CHUNK = exp1_analysis.CHUNK
SEED = exp1_analysis.SEED
PAIRING = exp1_analysis.PAIRING
AGGREGATE = exp1_analysis.AGGREGATE
SIZE_REGRESSOR = "natural log of the bank's test-window payments (test_rows)"
FIFTY = 50


def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def _metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


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


def shortfall_inputs(done: dict) -> dict[str, tuple[Path, dict]]:
    """Experiment 1's observed HI-Medium shortfall Runs, one per family."""
    out = {}
    for family in FAMILIES:
        found = exp1_analysis.shortfall_run(done, KEY, family)
        if found is None:
            raise SystemExit(f"Experiment 1 shortfall Run for {KEY}/{family} not found")
        out[family] = found
    return out


def banks_table(path: Path, metrics: dict) -> pd.DataFrame:
    return pd.read_csv(verify(path, metrics["artifacts"]["banks"]),
                       dtype={"bank": str}).set_index("bank")


# ---------------------------------------------------- shuffled shortfall --

def shuffled_unit_dirs(done: dict, family: str) -> dict | None:
    roster = exp1_shuffle.roster_for(KEY)
    if family == "gnn":
        glob = [exp1.recorded_dir(done, exp1.Unit("gnn_global", KEY, seed=s))
                for s in exp1.SEEDS]
        solo = {b: [done.get(config_hash(exp1_shuffle.unit_config(
                    exp1.Unit("gnn_solo", KEY, b, s)))) for s in exp1.SEEDS]
                for b in roster.index}
    else:
        glob = [exp1.recorded_dir(done, exp1.Unit(f"{family}_global", KEY))]
        solo = {b: [done.get(config_hash(exp1_shuffle.unit_config(
                    exp1.Unit(f"{family}_solo", KEY, b))))]
                for b in roster.index}
    if any(d is None for d in glob) or any(d is None for ds in solo.values()
                                           for d in ds):
        return None
    return {"global": glob, "solo": solo, "roster": roster}


def shuffled_config(family: str, dirs: dict, observed: tuple[Path, dict]) -> dict:
    world, dataset = data.DATASETS[KEY]
    config = {"experiment": exp1.EXPERIMENT, "check": "shortfall_shuffled",
              "world": world, "dataset": dataset, "dataset_key": KEY,
              "family": family, "control": "shuffled_bank_labels",
              "permutation": exp1_shuffle.PERMUTATION,
              "reading": exp1_shuffle.READING,
              "setups": ["solo", "global"], "pairing": PAIRING[family],
              "replicates": REPLICATES, "seed": SEED, "chunk": CHUNK,
              "global_side": "Experiment 1's global Runs and their persisted test "
                             "scores, reused unchanged: the network itself "
                             "is unchanged by the permutation",
              "statistics": "I2 and Q with the assumption-free null and "
                            "leave-one-out as in Experiment 1; tau by the "
                            "DerSimonian-Laird moment estimator; the size "
                            f"gradient as Spearman rho of delta against the "
                            f"{SIZE_REGRESSOR}",
              "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(KEY)},
              "observed_run": _hash_of(observed[0]),
              "inputs": {"global": [_hash_of(d) for d in dirs["global"]],
                         "solo": {b: [_hash_of(d) for d in ds]
                                  for b, ds in dirs["solo"].items()}},
              "features_version": exp1.FEATURES_VERSION}
    if family == "gnn":
        config["gnn_version"] = exp1.GNN_VERSION
        config["seeds"] = exp1.SEEDS
    if SMOKE:
        config["smoke"] = {"banks": exp1.SMOKE_BANKS}
    return config


def shuffled_metrics(family: str, dirs: dict, observed: tuple[Path, dict],
                     run_dir: Path) -> dict:
    t0 = time.time()
    cell = exp1_analysis.load_cell(KEY, family, dirs)
    groups, meta = cell["groups"], cell["meta"]
    labels = list(groups)
    roster = dirs["roster"]
    boot = joint_bootstrap(cell["n_test"], groups, replicates=REPLICATES,
                           seed=SEED, chunk=CHUNK, metrics=("ap",))
    agg = aggregate_shortfalls(boot, "ap", AGGREGATE[PAIRING[family]])
    deltas = np.array([agg[b]["delta"] for b in labels])
    se_boot = np.array([agg[b]["se_boot"] for b in labels])
    reps = np.stack([agg[b]["delta_reps"] for b in labels], axis=1)
    if family == "gnn":
        seed_var = np.array([np.var(agg[b]["per_setting"], ddof=1) for b in labels])
        ses = np.sqrt(se_boot ** 2 + seed_var / len(exp1.SEEDS))
    else:
        seed_var = None
        ses = se_boot
    reading = exp1_analysis.heterogeneity_reading(labels, deltas, ses, reps, seed_var)
    reading["delta_range"] = [float(deltas.min()), float(deltas.max())]
    test_rows = np.array([meta[b]["test_rows"] for b in labels], np.float64)

    obs_path, obs_metrics = observed
    obs_banks = banks_table(obs_path, obs_metrics)
    obs_primary = obs_metrics["primary"]
    observed_block = {
        "run": _hash_of(obs_path), "k": obs_primary["k"],
        "I2": obs_primary["I2"], "Q": obs_primary["Q"],
        "weighted_mean": obs_primary["weighted_mean"],
        "tau": tau_dl(obs_banks["delta"].to_numpy(), obs_banks["se"].to_numpy()),
        "size_gradient": size_gradient(obs_banks["delta"].to_numpy(),
                                       obs_banks["test_rows"].to_numpy())}

    aggregate = np.median if AGGREGATE[PAIRING[family]] == "median" else np.mean
    table = pd.DataFrame.from_dict(meta, orient="index").loc[labels]
    table.index.name = "bank"
    for column in ("train_pos", "val_pos"):
        table[column] = roster.loc[labels, column].to_numpy()
    table["delta"] = deltas
    table["se_boot"] = se_boot
    table["se"] = ses
    table["ci_lo"] = [agg[b]["ci95"][0] for b in labels]
    table["ci_hi"] = [agg[b]["ci95"][1] for b in labels]
    per_setting = np.stack([agg[b]["per_setting"] for b in labels])
    for s in range(per_setting.shape[1]):
        table[f"delta_s{s}"] = per_setting[:, s]
    table["solo_ap"] = [float(aggregate(agg[b]["solo_by_setting"])) for b in labels]
    table["global_ap"] = [float(aggregate(agg[b]["global_by_setting"])) for b in labels]
    if family == "gnn":
        table["seed_var"] = seed_var

    shuffled_tau = tau_dl(deltas, ses)
    shuffled_gradient = size_gradient(deltas, test_rows)
    artifacts = {"banks": write_csv(run_dir, "banks", table),
                 "roster": write_csv(run_dir, "roster", roster),
                 "replicates": write_array(run_dir, "replicates",
                                           reps.astype(np.float32))}
    return {"dataset_key": KEY, "family": family,
            "control": "shuffled_bank_labels",
            "permutation_seed": exp1_shuffle.SHUFFLE_SEED,
            "k": len(labels), "roster_size": int(len(roster)),
            "roster_overlap_with_observed": int(len(set(labels)
                                                    & set(obs_banks.index))),
            "degenerate_validation_banks": sorted(
                table.index[table["val_pos"] == 0]),
            "n_test": int(cell["n_test"]), "pairing": PAIRING[family],
            "replicates": REPLICATES, "primary": reading,
            "tau": shuffled_tau,
            "size_gradient": {"spearman_rho": shuffled_gradient,
                              "regressor": SIZE_REGRESSOR},
            "observed": observed_block,
            "comparison": {
                "I2_at_or_above_observed": bool(reading["I2"] >= observed_block["I2"]),
                "tau_at_or_above_observed": bool(shuffled_tau >= observed_block["tau"])},
            "seconds": time.time() - t0, "artifacts": artifacts}


def cmd_shuffled(family: str, done: dict | None = None) -> dict | None:
    done = done if done is not None else completed(exp1.EXPERIMENT)
    dirs = shuffled_unit_dirs(done, family)
    if dirs is None:
        print(f"shuffled {family}: inputs incomplete", flush=True)
        return None
    observed = exp1_analysis.shortfall_run(done, KEY, family)
    if observed is None:
        raise SystemExit(f"Experiment 1 shortfall Run for {KEY}/{family} not found; "
                         f"the observed reading comes from it")
    metrics = run(exp1.EXPERIMENT, shuffled_config(family, dirs, observed),
                  lambda config, run_dir: shuffled_metrics(family, dirs,
                                                           observed, run_dir),
                  done=done)
    p, o = metrics["primary"], metrics["observed"]
    print(f"shuffled {family}: k {metrics['k']} I2 {p['I2']:.3f} "
          f"(observed {o['I2']:.3f}) tau {metrics['tau']:.4f} "
          f"(observed {o['tau']:.4f}) gradient "
          f"{metrics['size_gradient']['spearman_rho']:+.3f} "
          f"(observed {o['size_gradient']:+.3f}) "
          f"null {'pass' if p['null_check']['passes'] else 'fail'} "
          f"loo I2 {p['leave_one_out']['I2']:.3f}", flush=True)
    return metrics


def shuffled_run(done: dict, family: str) -> tuple[Path, dict] | None:
    dirs = shuffled_unit_dirs(done, family)
    if dirs is None:
        return None
    observed = exp1_analysis.shortfall_run(done, KEY, family)
    if observed is None:
        return None
    path = done.get(config_hash(shuffled_config(family, dirs, observed)))
    if path is None:
        return None
    return path, _metrics(path)


# --------------------------------------------------- residual after size --

def residual_config(inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": exp1.EXPERIMENT, "check": "residual_after_size",
            "world": world, "dataset": dataset, "dataset_key": KEY,
            "families": list(FAMILIES),
            "reading": "residual after size",
            "post_hoc": "declared post hoc: this statistic was chosen after "
                        "Experiment 1's result was known; the "
                        "disclosure is permanent",
            "regressor": SIZE_REGRESSOR,
            "weights": "1/s_B^2 from each family's Experiment 1 shortfall banks.csv",
            "estimator": "weighted least squares; residual I2 on k-2 degrees "
                         "of freedom; residual tau by the DerSimonian-Laird "
                         "moment estimator generalised to the regression",
            "inputs": {family: _hash_of(path)
                       for family, (path, _) in inputs.items()},
            "smoke": SMOKE or None}


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
    inputs = shortfall_inputs(done)
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
            "families": ["tree", "blind"],
            "reading": "paired against own best",
            "own_best": "each arm read at its own validation-selected grid "
                        "setting (the prior work's design), from the "
                        "own_best_delta column Experiment 1 recorded",
            "note": "the own-best spread is read with the paired delta's "
                    "s_B, the only recorded wobble; the GNN family has no "
                    "grid and is out of scope",
            "weights": "1/s_B^2 from each family's Experiment 1 shortfall banks.csv",
            "inputs": {family: _hash_of(inputs[family][0])
                       for family in ("tree", "blind")},
            "smoke": SMOKE or None}


def paired_metrics(inputs: dict, run_dir: Path) -> dict:
    readings = {}
    for family in ("tree", "blind"):
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
    inputs = shortfall_inputs(done)
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
            "sensitivity": "hypothesis 1's verdict stands at the "
                           "roster threshold of thirty either way; the "
                           "count of qualifying banks is computed here, not "
                           "assumed",
            "statistics": "I2, Q, the assumption-free null, leave-one-out "
                          "and tau, recomputed on the banks holding at least "
                          "fifty test-window laundering payments, from the "
                          "persisted per-bank deltas and replicates",
            "inputs": {family: _hash_of(path)
                       for family, (path, _) in inputs.items()},
            "smoke": SMOKE or None}


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
    inputs = shortfall_inputs(done)
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
    for family in FAMILIES:
        cmd_shuffled(family, done)
    cmd_residual(done)
    cmd_paired(done)
    cmd_fifty(done)
    return 0


def cmd_status() -> int:
    done = completed(exp1.EXPERIMENT)
    for family in FAMILIES:
        found = shuffled_run(done, family)
        dirs = shuffled_unit_dirs(done, family)
        state = ("analysed" if found else
                 "inputs complete" if dirs else "inputs incomplete")
        print(f"shuffled {family:6s} {state}")
    for check, config_fn in (("residual_after_size", residual_config),
                             ("paired_vs_naive", paired_config),
                             ("roster_fifty", fifty_config)):
        try:
            inputs = shortfall_inputs(done)
        except SystemExit:
            print(f"{check}: inputs incomplete")
            continue
        state = "analysed" if config_hash(config_fn(inputs)) in done else "ready"
        print(f"{check}: {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "shuffled":
        return 0 if cmd_shuffled(args[0]) is not None else 1
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
