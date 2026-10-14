#!/usr/bin/env python
"""Experiment 4's analysis Runs. One ``policy`` Run per (dataset, family) reads every borrower's
transfer Run beside Experiment 1's solo Runs through one seeded set of joint-bootstrap
draws; one ``verdict`` Run applies hypothesis 3's gate, fixed before the result.

  uv run python scripts/exp4_analysis.py policy <key> <family>
  uv run python scripts/exp4_analysis.py verdict
  uv run python scripts/exp4_analysis.py all
  uv run python scripts/exp4_analysis.py status

The bootstrap resamples the whole test window (1,000 replicates, seed 0,
chunks of 100) and re-reads every bank from the same replicate. The pick and
the abstain decision are point decisions read once from the validation
window: the interval prices test-window noise, not selection noise, and this
is disclosed in the Run's metrics.
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
import exp1_data as data  # noqa: E402
import exp4  # noqa: E402
from extract import load_column  # noqa: E402
from m import completed, run  # noqa: E402
from m.artifacts import load_array, verify, write_array, write_csv  # noqa: E402
from m.runlog import config_hash, experiments_root  # noqa: E402
from m.stats import weighted_average_precision  # noqa: E402

REPLICATES = 1000
CHUNK = 100
SEED = 0
FAMILIES = ("tree", "gnn", "blind")
GATED_FAMILIES = ("tree", "gnn")
SIDES = ("solo", "policy", "ensemble")
PAIRING = exp4.PROTOCOL["pairing"]
GNN_INTERVAL_NOTE = ("between-seed variance folded into the weighted-mean "
                     "interval: each replicate's per-bank gain is perturbed "
                     "by an independent normal draw of variance seed_var/3 "
                     "(default_rng(0)), as Experiment 1's assumption-free null and "
                     "Experiment 2's component intervals do")
CRITERION = ("the example-poor quartile's roster-weighted mean "
             "settings-paired policy gain has a joint-bootstrap interval "
             "that excludes zero and a lower end above one quarter of that "
             "quartile's Experiment 1 mean shortfall, per family against its own line")


def check_execution_settings() -> None:
    """The policy Runs train nothing, so Experiment 1's thread settings do not
    bind; the committed-tree rule does, except on a scratch Run log."""
    scratch = experiments_root().resolve() != (exp1.REPO_ROOT / "experiments").resolve()
    if not scratch and exp1.committed_tree():
        raise SystemExit("Runs launch only from a committed tree; uncommitted: "
                         + ", ".join(exp1.committed_tree()[:12]))


# --------------------------------------------------------------- inputs --

_hash_of = exp4._hash_of


def _metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


def _scores(path: Path, name: str) -> np.ndarray:
    return load_array(path, name, _metrics(path)["artifacts"][name]["sha256"])


def scan_run(experiment: str, want: dict) -> tuple[Path, dict] | None:
    """The unique completed Run whose config carries every wanted item."""
    found = []
    for chash, path in completed(experiment).items():
        config = json.loads((path / "config.json").read_text())
        if all(config.get(k) == v for k, v in want.items()):
            found.append(path)
    if not found:
        return None
    assert len(found) == 1, f"{experiment} {want}: more than one Run"
    return found[0], _metrics(found[0])


def policy_inputs(done: dict, done1: dict, key: str, family: str) -> dict | None:
    """Every input Run of one (dataset, family) policy cell, or None while
    incomplete."""
    roster = data.read_roster(key)
    check = "transfer_gnn" if family == "gnn" else f"transfer_{family}"
    transfer, solo = {}, {}
    for bank in roster.index:
        unit = exp4.Unit(check, key, bank)
        try:
            path = done.get(config_hash(exp4.unit_config(unit, done, done1)))
        except (SystemExit, AssertionError):
            return None
        if path is None:
            return None
        transfer[bank] = path
        solo[bank] = exp4.solo_dirs(done1, key, family, bank)
    shortfall = scan_run("exp1", {"check": "shortfall", "dataset_key": key,
                                "family": family})
    if shortfall is None:
        return None
    components = scan_run("exp2", {"check": "components", "dataset_key": key,
                                 "family": family})
    examples = scan_run("exp2", {"check": "examples", "dataset_key": key,
                               "family": family})
    out = {"roster": roster, "transfer": transfer, "solo": solo,
           "shortfall": shortfall, "components": components,
           "examples": examples}
    if family == "gnn":
        gate = exp4.gate_run(done, done1, key)
        if gate is None or not gate[1]["pass"]:
            return None
        out["gate"] = gate
    return out


def policy_config(key: str, family: str, inputs: dict) -> dict:
    world, dataset = data.DATASETS[key]
    config = {"experiment": exp4.EXPERIMENT, "check": "policy", "world": world,
              "dataset": dataset, "dataset_key": key, "family": family,
              "protocol": exp4.PROTOCOL, "lenders": exp4.lenders_for(key),
              "pairing": PAIRING[family],
              "sides": list(SIDES),
              "ensemble": "the per-setting (per-seed) mean of the available "
                          "candidates' predicted probabilities; descriptive, "
                          "gates nothing",
              "quartiles": "pandas qcut over the roster's training-window "
                           "laundering count, four bins (Experiment 2's rule); the "
                           "example-poor quartile is the lowest",
              "threshold": "one quarter of the example-poor quartile's "
                           "unweighted mean Experiment 1 shortfall (Experiment 2's quartile "
                           "discipline), fixed before this Run",
              "weights": "Experiment 1's per-bank inverse-variance weights, 1/s_B^2 "
                         "from the Experiment 1 shortfall Run's banks.csv",
              "replicates": REPLICATES, "seed": SEED, "chunk": CHUNK,
              "selection_noise": "the pick and the abstain decision are "
                                 "point decisions from the validation "
                                 "window; the bootstrap resamples the test "
                                 "window only, so the interval prices "
                                 "test-window noise, not selection noise",
              "no_random_bank_control": "stated in advance: the claim is a "
                                        "regime-level mean, and on worlds "
                                        "whose banks are random labels "
                                        "lender identity is meaningless by "
                                        "construction",
              "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(key)},
              "inputs": {
                  "transfer": {b: _hash_of(p)
                               for b, p in inputs["transfer"].items()},
                  "solo": {b: [_hash_of(d) for d in ds]
                           for b, ds in inputs["solo"].items()},
                  "shortfall": _hash_of(inputs["shortfall"][0]),
                  "components": (_hash_of(inputs["components"][0])
                                 if inputs["components"] else None),
                  "examples": (_hash_of(inputs["examples"][0])
                               if inputs["examples"] else None)},
              "features_version": exp1.FEATURES_VERSION}
    if family == "gnn":
        config["gnn_version"] = exp4.GNN_VERSION
        config["seeds"] = exp4.SEEDS
        config["interval"] = GNN_INTERVAL_NOTE
        config["inputs"]["gate"] = _hash_of(inputs["gate"][0])
    if exp4.SMOKE:
        config["smoke"] = True
    return config


# ------------------------------------------------------------ bootstrap --

def load_groups(key: str, family: str, inputs: dict) -> dict:
    """Per bank: test-window positions, labels, and (settings, n_B) scores
    for the three sides, plus the point decisions of the policy."""
    split = load_column(key, "split")
    y_all = load_column(key, "is_laundering")
    test_rows = np.flatnonzero(split == 2)
    n_test = len(test_rows)
    position = np.full(len(split), -1, np.int64)
    position[test_rows] = np.arange(n_test)

    reduce = np.median if PAIRING[family] == "median_over_grid" else np.mean
    groups, decisions = {}, {}
    for bank, tpath in inputs["transfer"].items():
        tm = _metrics(tpath)
        candidates = tm["candidates"]
        payment = _scores(tpath, "test_payment")
        idx = position[payment]
        assert np.all(idx >= 0), f"{bank}: a scored payment is outside the test window"
        if family == "gnn":
            solo = np.stack([_scores(d, "test_probs")
                             for d in inputs["solo"][bank]])
            transfer = _scores(tpath, "test_probs")
            own_val = float(np.mean(list(tm["own_val_ap"].values())))
        else:
            solo = _scores(inputs["solo"][bank][0], "test_scores")
            transfer = _scores(tpath, "test_scores")
            own_val = float(np.median(tm["own_val_ap"]))
        assert transfer.shape == (len(candidates), solo.shape[0], len(idx))
        cand_val = {c: float(reduce(tm["per_candidate"][c]["val_ap"]))
                    for c in candidates}
        pick = max(candidates, key=lambda c: cand_val[c])
        borrow = cand_val[pick] > own_val
        borrowed = transfer[candidates.index(pick)]
        y = y_all[payment].astype(np.int8)
        groups[bank] = {"idx": idx, "y": y, "solo": solo,
                        "policy": borrowed if borrow else solo,
                        "borrowed": borrowed,
                        "ensemble": transfer.mean(axis=0)}
        decisions[bank] = {"pick": pick, "pick_val_ap": cand_val[pick],
                           "own_val_ap": own_val, "borrow": bool(borrow),
                           "candidate_val_ap": cand_val,
                           "never_seen_ap": tm["never_seen_test_ap"],
                           "val_pos": tm["positives"]["val"],
                           "test_rows": tm["rows"]["test"],
                           "test_pos": tm["positives"]["test"]}
    return {"groups": groups, "decisions": decisions, "n_test": n_test}


def sides_bootstrap(n_test: int, groups: dict) -> dict:
    """Point APs and joint-bootstrap AP replicates for the three sides from
    one seeded set of multiplicity draws (Experiment 1's discipline: seed 0, 1,000
    replicates, chunks of 100). A side that shares its score array with an
    already-computed side reuses that side's numbers (an abstaining bank's
    policy side is its solo side)."""
    labels = list(groups)
    n_settings = next(iter(groups.values()))["solo"].shape[0]

    def side_arrays(spec):
        seen: dict[int, str] = {}
        for side in SIDES:
            alias = seen.get(id(spec[side]))
            yield side, alias
            seen.setdefault(id(spec[side]), side)

    point = {g: {} for g in labels}
    for g in labels:
        spec = groups[g]
        ones = np.ones((1, len(spec["y"])))
        for side, alias in side_arrays(spec):
            point[g][side] = (point[g][alias] if alias else np.array(
                [weighted_average_precision(spec["y"], spec[side][s], ones)[0]
                 for s in range(n_settings)]))

    reps = {g: {} for g in labels}
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
            for side, alias in side_arrays(spec):
                if side not in reps[g]:
                    reps[g][side] = np.empty((REPLICATES, n_settings))
                block = reps[g][side]
                if alias:
                    block[done:done + r] = reps[g][alias][done:done + r]
                    continue
                for s in range(n_settings):
                    block[done:done + r, s] = weighted_average_precision(
                        spec["y"], spec[side][s], w)
        done += r
    return {"point": point, "reps": reps, "labels": labels,
            "n_settings": n_settings}


def gain_estimates(boot: dict, groups: dict, family: str) -> dict:
    """Per bank: the settings-paired point gain, replicate gains, bootstrap
    standard error and interval, for the policy and the ensemble; the GNN
    family adds the between-seed variance of the per-seed point gains."""
    reduce = np.nanmedian if PAIRING[family] == "median_over_grid" else np.nanmean
    out = {}
    for g in boot["labels"]:
        rec = {}
        for name, side in (("policy", "policy"), ("ensemble", "ensemble"),
                           ("borrowed", None)):
            if side is None:
                borrowed_point = np.array(
                    [weighted_average_precision(
                        groups[g]["y"], groups[g]["borrowed"][s],
                        np.ones((1, len(groups[g]["y"]))))[0]
                     for s in range(boot["n_settings"])])
                per_setting = borrowed_point - boot["point"][g]["solo"]
                rec[name] = {"delta": float(reduce(per_setting)),
                             "per_setting": per_setting.tolist(),
                             "ap": float(reduce(borrowed_point))}
                continue
            per_setting = boot["point"][g][side] - boot["point"][g]["solo"]
            delta_reps = reduce(boot["reps"][g][side]
                                - boot["reps"][g]["solo"], axis=1)
            valid = np.isfinite(delta_reps)
            entry = {"delta": float(reduce(per_setting)),
                     "per_setting": per_setting.tolist(),
                     "delta_reps": delta_reps,
                     "se_boot": float(np.std(delta_reps[valid], ddof=1)),
                     "ci95": [float(v) for v in
                              np.percentile(delta_reps[valid], [2.5, 97.5])]}
            if family == "gnn":
                entry["seed_var"] = float(np.var(per_setting, ddof=1))
            rec[name] = entry
        rec["levels"] = {side: [float(v) for v in boot["point"][g][side]]
                         for side in SIDES}
        out[g] = rec
    return out


def weighted_mean_reading(labels: list[str], est: dict, weights: np.ndarray,
                          name: str, family: str,
                          rng: np.random.Generator) -> dict:
    deltas = np.array([est[g][name]["delta"] for g in labels])
    reps = np.stack([est[g][name]["delta_reps"] for g in labels], axis=1)
    if family == "gnn":
        seed_var = np.array([est[g][name]["seed_var"] for g in labels])
        reps = reps + rng.standard_normal(reps.shape) \
            * np.sqrt(seed_var / 3.0)[None, :]
    wsum = weights.sum()
    point = float((weights * deltas).sum() / wsum)
    finite = np.all(np.isfinite(reps), axis=1)
    wmean_reps = (reps[finite] * weights).sum(axis=1) / wsum
    lo, hi = np.percentile(wmean_reps, [2.5, 97.5])
    return {"weighted_mean": point, "ci95": [float(lo), float(hi)],
            "se": float(np.std(wmean_reps, ddof=1)),
            "replicates_valid": int(finite.sum()), "k": len(labels)}


# ---------------------------------------------------------------- policy --

def policy_metrics(key: str, family: str, inputs: dict, run_dir: Path) -> dict:
    t0 = time.time()
    cell = load_groups(key, family, inputs)
    groups, decisions = cell["groups"], cell["decisions"]
    labels = list(groups)
    roster = inputs["roster"]
    lenders = exp4.lenders_for(key)

    spath, smetrics = inputs["shortfall"]
    banks_exp1 = pd.read_csv(verify(spath, smetrics["artifacts"]["banks"]),
                           dtype={"bank": str}).set_index("bank")
    assert set(labels) == set(banks_exp1.index)
    se_exp1 = banks_exp1.loc[labels, "se"].to_numpy()
    weights = 1.0 / se_exp1 ** 2

    table = pd.DataFrame(index=pd.Index(labels, name="bank"))
    table["train_pos"] = roster.loc[labels, "train_pos"].to_numpy()
    table["test_rows"] = [decisions[b]["test_rows"] for b in labels]
    table["test_pos"] = [decisions[b]["test_pos"] for b in labels]
    table["quartile"] = pd.qcut(table["train_pos"], 4, labels=False) + 1
    table["is_lender"] = [b in lenders for b in labels]
    table["weight_se_exp1"] = se_exp1
    table["delta_exp1"] = banks_exp1.loc[labels, "delta"].to_numpy()

    q1 = [b for b in labels if table.loc[b, "quartile"] == 1]
    d_q1 = float(table.loc[q1, "delta_exp1"].mean())
    threshold = d_q1 / 4.0
    threshold_block = {"quartile_banks": len(q1),
                       "count_range": [int(table.loc[q1, "train_pos"].min()),
                                       int(table.loc[q1, "train_pos"].max())],
                       "exp1_mean_shortfall": d_q1, "line_D_over_4": threshold}
    if inputs["examples"] is not None:
        examples_q1 = inputs["examples"][1]["quartile_table"][0]
        assert abs(examples_q1["mean_shortfall"] - d_q1) < 1e-9 \
            and examples_q1["banks"] == len(q1), \
            "the quartile line disagrees with Experiment 2's examples Run"
        threshold_block["matches_exp2_examples_run"] = True

    boot = sides_bootstrap(cell["n_test"], groups)
    est = gain_estimates(boot, groups, family)
    t_boot = time.time() - t0

    # The reading order (primary, ensemble_q1, all_banks) is load-bearing:
    # the GNN seed-noise draws come from one shared generator in sequence.
    rng = np.random.default_rng(SEED)
    primary = weighted_mean_reading(q1, est, weights[[labels.index(b) for b in q1]],
                                    "policy", family, rng)
    primary["excludes_zero"] = bool(primary["ci95"][0] > 0.0
                                    or primary["ci95"][1] < 0.0)
    primary["lower_end_above_line"] = bool(primary["ci95"][0] > threshold)
    ensemble_q1 = weighted_mean_reading(q1, est,
                                        weights[[labels.index(b) for b in q1]],
                                        "ensemble", family, rng)
    all_banks = weighted_mean_reading(labels, est, weights, "policy", family, rng)

    reduce = np.median if PAIRING[family] == "median_over_grid" else np.mean
    table["val_pos"] = [decisions[b]["val_pos"] for b in labels]
    table["own_val_ap"] = [decisions[b]["own_val_ap"] for b in labels]
    table["pick"] = [decisions[b]["pick"] for b in labels]
    table["pick_val_ap"] = [decisions[b]["pick_val_ap"] for b in labels]
    table["borrow"] = [decisions[b]["borrow"] for b in labels]
    table["solo_ap"] = [float(reduce(est[b]["levels"]["solo"])) for b in labels]
    table["policy_ap"] = [float(reduce(est[b]["levels"]["policy"])) for b in labels]
    table["ensemble_ap"] = [float(reduce(est[b]["levels"]["ensemble"]))
                            for b in labels]
    table["borrowed_ap"] = [est[b]["borrowed"]["ap"] for b in labels]
    table["global_ap"] = banks_exp1.loc[labels, "global_ap"].to_numpy()
    if inputs["components"] is not None:
        cpath, cmetrics = inputs["components"]
        banks_exp2 = pd.read_csv(verify(cpath, cmetrics["artifacts"]["banks"]),
                               dtype={"bank": str}).set_index("bank")
        table["clocal_ap"] = banks_exp2.loc[labels, "clocal_ap"].to_numpy()
    else:
        table["clocal_ap"] = np.nan
    table["never_seen_ap"] = [decisions[b]["never_seen_ap"] for b in labels]
    table["gain_if_borrowed"] = [est[b]["borrowed"]["delta"] for b in labels]
    table["policy_gain"] = [est[b]["policy"]["delta"] for b in labels]
    table["ensemble_gain"] = [est[b]["ensemble"]["delta"] for b in labels]
    table["policy_se_boot"] = [est[b]["policy"]["se_boot"] for b in labels]
    table["policy_ci_lo"] = [est[b]["policy"]["ci95"][0] for b in labels]
    table["policy_ci_hi"] = [est[b]["policy"]["ci95"][1] for b in labels]
    if family == "gnn":
        table["seed_var"] = [est[b]["policy"]["seed_var"] for b in labels]
    per = np.stack([est[b]["policy"]["per_setting"] for b in labels])
    for s in range(per.shape[1]):
        table[f"policy_gain_s{s}"] = per[:, s]

    solo_by_setting = {b: np.array(est[b]["levels"]["solo"]) for b in labels}
    band_floor = np.array([solo_by_setting[b].min() for b in labels])
    below_band = table["policy_ap"].to_numpy() < band_floor
    safety = {
        "band": ("the per-setting spread of the bank's own solo test APs"
                 if family != "gnn" else "the bank's own solo seed band"),
        "banks_below_own_band": int(below_band.sum()),
        "below_own_band": sorted(np.array(labels)[below_band].tolist()),
        "abstain_rate": float((~table["borrow"]).mean()),
        "borrows": int(table["borrow"].sum()),
        "error_table": {
            "harmful_borrows_not_avoided": int(
                (table["borrow"] & (table["gain_if_borrowed"] < 0)).sum()),
            "beneficial_borrows_forgone": int(
                (~table["borrow"] & (table["gain_if_borrowed"] > 0)).sum()),
            "beneficial_borrows_forgone_over_0.05": int(
                (~table["borrow"] & (table["gain_if_borrowed"] > 0.05)).sum()),
            "banks_borrowing_harm": sorted(
                table.index[table["borrow"]
                            & (table["gain_if_borrowed"] < 0)].tolist())}}

    # The selection signal is thin exactly where the gate reads: a bank with
    # few validation-window laundering payments compares APs measured on a
    # handful of positives, so the pick and the abstain decision discriminate
    # weakly there. Disclosed per quartile; the gain itself is measured on
    # test windows of at least thirty positives.
    selection_signal = {
        "median_val_pos_by_quartile": {
            int(q): float(sub["val_pos"].median())
            for q, sub in table.groupby("quartile")},
        "banks_with_val_pos_at_most_10": int((table["val_pos"] <= 10).sum()),
        "q1_banks_with_val_pos_at_most_10": int(
            (table.loc[table["quartile"] == 1, "val_pos"] <= 10).sum()),
        "min_val_pos": int(table["val_pos"].min())}

    gap = table["global_ap"] - table["solo_ap"]
    ok = gap > 0.01
    quartiles = []
    for q in (1, 2, 3, 4):
        sub = table[table["quartile"] == q]
        quartiles.append({
            "quartile": q, "banks": int(len(sub)),
            "count_range": [int(sub["train_pos"].min()),
                            int(sub["train_pos"].max())],
            "median_test_base_rate": float(
                (sub["test_pos"] / sub["test_rows"]).median()),
            "median_solo_ap": float(sub["solo_ap"].median()),
            "median_policy_ap": float(sub["policy_ap"].median()),
            "median_ensemble_ap": float(sub["ensemble_ap"].median()),
            "median_global_ap": float(sub["global_ap"].median()),
            "median_clocal_ap": (float(sub["clocal_ap"].median())
                                 if sub["clocal_ap"].notna().any() else None),
            "median_never_seen_ap": float(sub["never_seen_ap"].median()),
            "median_policy_gain": float(sub["policy_gain"].median()),
            "mean_policy_gain": float(sub["policy_gain"].mean()),
            "borrows": int(sub["borrow"].sum()),
            "policy_above_never_seen": int(
                (sub["policy_ap"] > sub["never_seen_ap"]).sum()),
            "policy_negative": int((sub["policy_gain"] < 0).sum())})
    yardsticks = {
        "quartile_table": quartiles,
        "gap_closed_policy": {
            "banks": int(ok.sum()),
            "median": float(((table.loc[ok, "policy_ap"]
                              - table.loc[ok, "solo_ap"]) / gap[ok]).median())},
        "gap_closed_ensemble": {
            "banks": int(ok.sum()),
            "median": float(((table.loc[ok, "ensemble_ap"]
                              - table.loc[ok, "solo_ap"]) / gap[ok]).median())},
        "policy_above_never_seen": int(
            (table["policy_ap"] > table["never_seen_ap"]).sum())}

    reps_policy = np.stack([est[b]["policy"]["delta_reps"] for b in labels],
                           axis=1)
    reps_ensemble = np.stack([est[b]["ensemble"]["delta_reps"] for b in labels],
                             axis=1)
    artifacts = {"banks": write_csv(run_dir, "banks", table),
                 "roster": write_csv(run_dir, "roster", roster),
                 "replicates": write_array(run_dir, "replicates",
                                           reps_policy.astype(np.float32)),
                 "replicates_ensemble": write_array(
                     run_dir, "replicates_ensemble",
                     reps_ensemble.astype(np.float32))}
    return {"dataset_key": key, "family": family, "k": len(labels),
            "n_test": int(cell["n_test"]), "pairing": PAIRING[family],
            "replicates": REPLICATES, "lenders": lenders,
            "primary": primary, "threshold": threshold_block,
            "ensemble_q1": ensemble_q1, "all_banks": all_banks,
            "safety": safety, "yardsticks": yardsticks,
            "selection_signal": selection_signal,
            "selection_noise": "the interval prices test-window noise only; "
                               "the pick and abstain decisions are point "
                               "decisions from the validation window",
            "seconds": {"bootstrap": t_boot, "total": time.time() - t0},
            "artifacts": artifacts}


def cmd_policy(key: str, family: str, done: dict | None = None,
               done1: dict | None = None) -> dict | None:
    done = done if done is not None else completed(exp4.EXPERIMENT)
    done1 = done1 if done1 is not None else completed("exp1")
    inputs = policy_inputs(done, done1, key, family)
    if inputs is None:
        print(f"policy {key} {family}: inputs incomplete", flush=True)
        return None
    metrics = run(exp4.EXPERIMENT, policy_config(key, family, inputs),
                  lambda config, run_dir: policy_metrics(key, family, inputs,
                                                         run_dir),
                  done=done)
    p, t = metrics["primary"], metrics["threshold"]
    print(f"policy {key} {family}: Q1 weighted mean {p['weighted_mean']:+.4f} "
          f"ci [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}] line "
          f"{t['line_D_over_4']:+.4f} borrows {metrics['safety']['borrows']} "
          f"of {metrics['k']}", flush=True)
    return metrics


def policy_run(done: dict, done1: dict, key: str,
               family: str) -> tuple[Path, dict] | None:
    inputs = policy_inputs(done, done1, key, family)
    if inputs is None:
        return None
    path = done.get(config_hash(policy_config(key, family, inputs)))
    if path is None:
        return None
    return path, _metrics(path)


# -------------------------------------------------------------- verdict --

def verdict_config(cells: dict) -> dict:
    return {"experiment": exp4.EXPERIMENT, "check": "verdict",
            "hypothesis": 3,
            "primary_dataset": exp4.PRIMARY_KEY,
            "gated_families": list(GATED_FAMILIES),
            "criterion": CRITERION,
            "outcomes": {
                "confirmed": "the criterion holds for both families",
                "killed": "it fails for both",
                "supported_one_family": "it holds for exactly one, reported "
                                        "as exactly that weaker finding"},
            "blind": "descriptive floor, gates nothing",
            "sensitivity": "HI-Small under the identical protocol, gating "
                           "nothing",
            "world_note": "AMLworld plants the same pattern families at "
                          "every bank, which plausibly inflates cross-bank "
                          "transfer; stated wherever a transfer number is "
                          "quoted",
            "inputs": cells, "smoke": exp4.SMOKE or None}


def verdict_metrics(runs: dict, run_dir: Path) -> dict:
    P = exp4.PRIMARY_KEY
    verdicts = {}
    for family in GATED_FAMILIES:
        m = runs[P][family][1]
        p, t = m["primary"], m["threshold"]
        verdicts[family] = {
            "weighted_mean": p["weighted_mean"], "ci95": p["ci95"],
            "line_D_over_4": t["line_D_over_4"],
            "exp1_mean_shortfall_q1": t["exp1_mean_shortfall"],
            "excludes_zero": p["excludes_zero"],
            "lower_end_above_line": p["lower_end_above_line"],
            "share_of_shortfall_recovered": (
                p["weighted_mean"] / t["exp1_mean_shortfall"]
                if t["exp1_mean_shortfall"] else None),
            "pass": bool(p["excludes_zero"] and p["lower_end_above_line"])}
    passing = [f for f in GATED_FAMILIES if verdicts[f]["pass"]]
    if len(passing) == 2:
        outcome = "confirmed"
    elif not passing:
        outcome = "killed"
    else:
        outcome = f"supported for one family only ({passing[0]})"
    descriptive = {}
    for key, fams in runs.items():
        for family, (path, m) in fams.items():
            if key == P and family in GATED_FAMILIES:
                continue
            descriptive[f"{key}/{family}"] = {
                "weighted_mean_q1": m["primary"]["weighted_mean"],
                "ci95": m["primary"]["ci95"],
                "line_D_over_4": m["threshold"]["line_D_over_4"],
                "borrows": m["safety"]["borrows"], "k": m["k"]}
    safety = {f: runs[P][f][1]["safety"] for f in runs.get(P, {})}
    return {"criterion": CRITERION, "families": verdicts,
            "hypothesis_3": outcome, "passing": passing,
            "descriptive": descriptive, "safety_disclosure": safety,
            "reaction": (None if passing else
                         "the examples channel needs the bank's own traffic "
                         "distribution, and the sharing ladder's "
                         "zero-data-movement rung is measured absent")}


def cmd_verdict() -> int:
    done = completed(exp4.EXPERIMENT)
    done1 = completed("exp1")
    runs: dict[str, dict] = {}
    for key in exp4.KEYS:
        for family in FAMILIES:
            found = policy_run(done, done1, key, family)
            if found is not None:
                runs.setdefault(key, {})[family] = found
    missing = [f"{exp4.PRIMARY_KEY}/{f}" for f in GATED_FAMILIES
               if f not in runs.get(exp4.PRIMARY_KEY, {})]
    if missing:
        print(f"verdict: policy Runs missing for {missing}", flush=True)
        return 1
    cells = {k: {f: _hash_of(p) for f, (p, _) in fams.items()}
             for k, fams in runs.items()}
    metrics = run(exp4.EXPERIMENT, verdict_config(cells),
                  lambda config, run_dir: verdict_metrics(runs, run_dir), done=done)
    print(json.dumps({"hypothesis_3": metrics["hypothesis_3"],
                      "families": {f: {"pass": v["pass"], "ci95": v["ci95"],
                                       "line": v["line_D_over_4"]}
                                   for f, v in metrics["families"].items()}},
                     indent=2))
    return 0


def cmd_all() -> int:
    done = completed(exp4.EXPERIMENT)
    done1 = completed("exp1")
    for key in exp4.KEYS:
        for family in FAMILIES:
            cmd_policy(key, family, done, done1)
    return cmd_verdict()


def cmd_status() -> int:
    done = completed(exp4.EXPERIMENT)
    done1 = completed("exp1")
    for key in exp4.KEYS:
        for family in FAMILIES:
            found = policy_run(done, done1, key, family)
            inputs = policy_inputs(done, done1, key, family)
            state = ("analysed" if found else
                     "inputs complete" if inputs else "inputs incomplete")
            print(f"policy {key:20s} {family:6s} {state}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    check_execution_settings()
    exp1.install_signal_handlers()
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "policy":
        return 0 if cmd_policy(args[0], args[1]) is not None else 1
    if cmd == "verdict":
        return cmd_verdict()
    if cmd == "all":
        return cmd_all()
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
