#!/usr/bin/env python
"""Experiment 3: Rules on the ledger (Chapter 5). The label-free properties
and the sixteen rules on them.

Computes the twelve label-free properties of every qualifying bank of HI-Medium
and HI-Small, then scores the sixteen rules on HI-Medium's 107 qualifying banks
for each family: each rule's out-of-fold rho, and its paired bank-bootstrap
difference from the scale rule, which gives the verdict on hypothesis 2. Nothing trains; every number is a function of
the cached ledgers and of Experiment 1's recorded shortfalls.

A check precedes every reading. It recomputes ten banks' properties by a
deliberately naive second implementation — pandas group-bys over account
labels, selecting its own rows — and requires agreement to 1e-12. It then
**deletes**, in turn, the rows outside the training window and the same-account
rows, and requires every property to be bit-identical after each; and it
rewrites every label first to 0 and then to 1, requiring the same. Deletion
rather than permutation, because every property is a symmetric function of
the rows it counts: shuffling a block of rows among themselves preserves the
multiset and could not move a property that reads them, so a permutation
check would pass whatever the properties did.

  uv run python scripts/exp3_descriptors.py                 # check, properties, rules
  uv run python scripts/exp3_descriptors.py HI-Medium       # one dataset's properties

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
from m import completed, config_hash, run, track  # noqa: E402
from m import descriptors as desc  # noqa: E402
from m import rank, rule  # noqa: E402
from m.artifacts import write_array, write_csv  # noqa: E402

EXPERIMENT = "exp3"
CHECK = "descriptors"
GATE = "descriptors_gate"
SELECTION = "selection"
GATE_KEY = dd.FIT_KEY
GATE_BANKS = 10
GATE_TOLERANCE = 1e-12
BOOTSTRAP_DRAWS = 1000
SEED = 0

CANDIDATES = rule.candidates(desc.GROUPS, desc.BASELINE)
CANDIDATE_ORDER = [candidate.name for candidate in CANDIDATES]


# ------------------------------------------------------- descriptor Runs --

def descriptor_config(key: str) -> dict:
    return {"experiment": EXPERIMENT, "check": CHECK,
            "world": dd.data.DATASETS[key][0], "dataset": dd.data.DATASETS[key][1],
            "dataset_key": key,
            "roster": {"min_test_pos": dd.data.MIN_TEST_POS,
                       "exclude": dd.data.excluded_banks(key)},
            "window": dd.WINDOW_RULE,
            "descriptors": {name: desc.DESCRIPTIONS[name] for name in desc.NAMES},
            "groups": {name: list(members) for name, members in desc.GROUPS.items()},
            "transform": "log(1 + v)",
            "account_identity": "(bank, account) on one dictionary shared by both "
                                "endpoint columns; currencies likewise",
            "descriptors_version": desc.VERSION}


def descriptor_metrics(key: str, run_dir: Path) -> dict:
    t0 = time.time()
    roster = dd.roster(key)
    values = {bank: desc.compute(dd.bank_ledger(key, bank))
              for bank in track(list(roster.index), f"descriptors {key}")}
    table = pd.DataFrame(values).T[list(desc.NAMES)].astype(float)
    table.index.name = "bank"
    for name in desc.NAMES:
        table[f"log1p_{name}"] = desc.transform(table[name].to_numpy())

    summary = {}
    for name in desc.NAMES:
        column = table[name].to_numpy()
        finite = column[np.isfinite(column)]
        summary[name] = {
            "defined": int(len(finite)), "undefined": int(len(column) - len(finite)),
            "zero": int((finite == 0).sum()), "distinct": int(len(np.unique(finite))),
            "min": float(finite.min()), "q25": float(np.percentile(finite, 25)),
            "median": float(np.median(finite)),
            "q75": float(np.percentile(finite, 75)), "max": float(finite.max()),
            "sd": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "constant": bool(len(np.unique(finite)) <= 1)}
    return {"dataset_key": key, "world": dd.data.DATASETS[key][0],
            "k": int(len(table)),
            "property": summary,
            "constant_properties": sorted(n for n in desc.NAMES if summary[n]["constant"]),
            "banks_with_any_undefined": int(
                (~np.isfinite(table[list(desc.NAMES)].to_numpy())).any(axis=1).sum()),
            "seconds": time.time() - t0,
            "artifacts": {"descriptors": write_csv(run_dir, "descriptors", table),
                          "roster": write_csv(run_dir, "roster", roster)}}


def cmd_descriptors(key: str, done: dict) -> dict:
    metrics = run(EXPERIMENT, descriptor_config(key),
                  lambda config, run_dir: descriptor_metrics(key, run_dir), done=done)
    constant = metrics["constant_properties"]
    print(f"{CHECK} {key}: k {metrics['k']}, "
          f"{metrics['banks_with_any_undefined']} banks with an undefined property, "
          f"constant here: {constant or 'none'}", flush=True)
    return metrics


def descriptor_table(key: str, done: dict) -> pd.DataFrame:
    path = done[config_hash(descriptor_config(key))]
    return pd.read_csv(path / "artifacts" / "descriptors.csv",
                       dtype={"bank": str}).set_index("bank")


# ------------------------------------------------------------------ gate --

def naive_descriptors(ledger: pd.DataFrame, bank: str) -> dict[str, float]:
    """A second implementation. It selects its own rows, keys its accounts by
    label rather than by code, and reaches every count through a pandas group-by
    or a plain loop. What it shares with the first is the recurrence
    threshold, which is a constant rather than a computation."""
    rows = ledger[(ledger["split"] == 0) & ~ledger["same_account"]]
    payer = rows["from_bank"].astype(str) + "\x1f" + rows["from_acct"].astype(str)
    payee = rows["to_bank"].astype(str) + "\x1f" + rows["to_acct"].astype(str)
    payer_bank = rows["from_bank"].astype(str)
    payee_bank = rows["to_bank"].astype(str)
    frame = pd.DataFrame({"payer": payer.to_numpy(), "payee": payee.to_numpy(),
                          "payer_bank": payer_bank.to_numpy(),
                          "payee_bank": payee_bank.to_numpy(),
                          "fmt": rows["fmt"].astype(str).to_numpy(),
                          "pay": rows["pay_ccy"].astype(str).to_numpy(),
                          "recv": rows["recv_ccy"].astype(str).to_numpy()})

    internal = (frame.payer_bank == bank) & (frame.payee_bank == bank)
    own = pd.concat([frame.payer[frame.payer_bank == bank],
                     frame.payee[frame.payee_bank == bank]])
    own_counts = own.value_counts()
    external = pd.concat([frame.payee[(frame.payer_bank == bank) & ~internal],
                          frame.payer[(frame.payee_bank == bank) & ~internal]])
    external_counts = external.value_counts()
    boundary = pd.concat([frame.payer[(frame.payer_bank == bank) & ~internal],
                          frame.payee[(frame.payee_bank == bank) & ~internal]]).nunique()
    external_banks = pd.concat([frame.payee_bank[(frame.payer_bank == bank) & ~internal],
                                frame.payer_bank[(frame.payee_bank == bank) & ~internal]])
    bank_shares = external_banks.value_counts(normalize=True)

    pairs = set(zip(frame.payer, frame.payee))
    mutual = sum(1 for a, b in pairs if (b, a) in pairs)

    def shannon(series: pd.Series) -> float:
        share = series.value_counts(normalize=True).to_numpy()
        return float(-(share * np.log(share)).sum())

    def gini_of(counts: np.ndarray) -> float:
        x = np.sort(counts.astype(float))
        m = len(x)
        if m <= 1:
            return 0.0
        cumulative = 0.0
        for i, value in enumerate(x, start=1):
            cumulative += i * value
        return float(2.0 * cumulative / (m * x.sum()) - (m + 1.0) / m)

    return {
        "payment_count": float(len(frame)),
        "active_accounts": float(len(own_counts)),
        "internal_share": float(internal.mean()),
        "boundary_share": float(boundary / len(own_counts)),
        "external_recurrence": float((external_counts >= desc.RECURRENCE_THRESHOLD).mean()),
        "external_encounters": float(external_counts.median()),
        "external_banks": float(len(bank_shares)),
        "external_hhi": float((bank_shares.to_numpy() ** 2).sum()),
        "activity_gini": gini_of(own_counts.to_numpy()),
        "reciprocity": float(mutual / len(pairs)),
        "format_entropy": shannon(frame.fmt),
        "currency_entropy": shannon(pd.concat([frame.pay, frame.recv])),
    }


def _without(ledger: pd.DataFrame, drop: np.ndarray) -> pd.DataFrame:
    """The ledger with the selected rows physically removed. Deletion, not
    permutation: every label-free property is a symmetric function of the rows
    it counts, so permuting a block of rows among themselves preserves the
    multiset and cannot move a property that reads them. Only removing them
    can."""
    return ledger.loc[~drop].reset_index(drop=True)


def _relabelled(ledger: pd.DataFrame, value: int) -> pd.DataFrame:
    out = ledger.copy()
    out["is_laundering"] = np.full(len(out), value, np.int8)
    return out


def gate_banks(key: str) -> list[str]:
    roster = dd.roster(key)
    chosen = np.random.default_rng(SEED).choice(len(roster), GATE_BANKS, replace=False)
    return [str(roster.index[i]) for i in sorted(chosen)]


def gate_config() -> dict:
    return {"experiment": EXPERIMENT, "check": GATE,
            "dataset_key": GATE_KEY, "banks": gate_banks(GATE_KEY),
            "tolerance": GATE_TOLERANCE,
            "checks": {
                "independent_recomputation": "a second, naive implementation, which "
                                             "selects its own rows and keys its own "
                                             "accounts by label, reproduces every "
                                             "label-free property",
                "window_restriction": "deleting the rows outside the training window "
                                      "leaves every property bit-identical",
                "housekeeping": "deleting the same-account rows leaves every property "
                                "bit-identical",
                "label_freeness": "setting every label to 0, and again to 1, leaves "
                                  "every property bit-identical"},
            "why_deletion": "a property is a symmetric function of the rows it "
                            "counts, so permuting a block of rows among themselves "
                            "cannot move it; the check has to remove them",
            "seed": SEED, "descriptors_version": desc.VERSION}


def gate_metrics(run_dir: Path) -> dict:
    banks = gate_banks(GATE_KEY)
    rows = []
    for bank in track(banks, f"{GATE} {GATE_KEY}"):
        ledger = dd.data.read_ledger(GATE_KEY, bank)
        reference = desc.compute(dd.ledger_arrays(dd.training_rows(ledger), bank))
        naive = naive_descriptors(ledger, bank)
        outside = ledger["split"].to_numpy() != dd.TRAINING_SPLIT
        housekeeping = ledger["same_account"].to_numpy()

        variants = {
            "window_restriction": [_without(ledger, outside)],
            "housekeeping": [_without(ledger, housekeeping)],
            "label_freeness": [_relabelled(ledger, 0), _relabelled(ledger, 1)],
        }
        row = {"bank": bank, "rows": int(len(ledger)),
               "rows_outside_the_window": int(outside.sum()),
               "same_account_rows": int(housekeeping.sum()),
               "independent_recomputation": max(
                   abs(reference[name] - naive[name]) for name in desc.NAMES)}
        for label, ledgers in variants.items():
            row[label] = max(
                abs(reference[name]
                    - desc.compute(dd.ledger_arrays(dd.training_rows(variant), bank))[name])
                for variant in ledgers for name in desc.NAMES)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("bank")
    checks = ["independent_recomputation", "window_restriction", "housekeeping",
              "label_freeness"]
    worst = {name: float(table[name].max()) for name in checks}
    return {"banks": int(len(table)), "worst": worst, "tolerance": GATE_TOLERANCE,
            "rows_removed": {
                "outside_the_window": int(table["rows_outside_the_window"].sum()),
                "same_account": int(table["same_account_rows"].sum())},
            "pass": bool(worst["independent_recomputation"] <= GATE_TOLERANCE
                         and all(worst[name] == 0.0 for name in checks[1:])),
            "artifacts": {"gate": write_csv(run_dir, "gate", table)}}


def cmd_gate(done: dict) -> bool:
    result = run(EXPERIMENT, gate_config(),
                 lambda config, run_dir: gate_metrics(run_dir), done=done)
    worst = result["worst"]
    print(f"{GATE}: recomputation {worst['independent_recomputation']:.3e}, "
          f"labels {worst['label_freeness']:.3e}, window {worst['window_restriction']:.3e}, "
          f"housekeeping {worst['housekeeping']:.3e} over {result['banks']} banks — "
          f"{'pass' if result['pass'] else 'FAIL'}", flush=True)
    return bool(result["pass"])


# ------------------------------------------------------------ selection --

def selection_config(family: str) -> dict:
    return {"experiment": EXPERIMENT, "check": SELECTION, "family": family,
            "dataset_key": dd.FIT_KEY,
            "candidates": {candidate.name: list(candidate.descriptors)
                           for candidate in CANDIDATES},
            "model": "linear on log(1 + v); weighted least squares at 1/s_B^2, "
                     "the unweighted fit beside it",
            "folds": {"n_folds": rule.N_FOLDS, "repeats": rule.REPEATS,
                      "assignment": "the roster's canonical order permuted at the "
                                    "repeat index, cut into contiguous blocks"},
            "metric": "mean over repeats of the out-of-fold Spearman correlation",
            "outperforms": "a rule outperforms the scale rule when the 5th "
                           "percentile of the paired bank-bootstrap difference "
                           "between the two exceeds zero",
            "hypothesis_2": "refuted for the family when none of the fifteen "
                            "other rules outperforms the scale rule",
            "estimand": "the mean over repeats of the out-of-fold rank correlation; "
                        "every score, standard deviation and difference is that "
                        "same quantity",
            "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": SEED},
            "inputs": {"descriptors": config_hash(descriptor_config(dd.FIT_KEY)),
                       "shortfall": dd.shortfall_hash(dd.FIT_KEY, family)},
            "descriptors_version": desc.VERSION, "rule_version": rule.VERSION}


def evaluate(design_all: pd.DataFrame, target: np.ndarray, weights: np.ndarray) -> dict:
    """Every candidate's out-of-fold predictions, its out-of-fold rho and its
    paired bank-bootstrap difference from the scale rule, all of them the
    *same* estimand: the mean over repeats of the out-of-fold rank
    correlation."""
    out = {}
    for candidate in CANDIDATES:
        columns = [f"log1p_{name}" for name in candidate.descriptors]
        design = design_all[columns].to_numpy(np.float64)
        predictions = rule.out_of_fold(design, target, weights,
                                       repeats=rule.REPEATS, n_folds=rule.N_FOLDS)
        per_repeat = np.array([rank.spearman(row, target) for row in predictions])
        out[candidate.name] = {
            "per_repeat": per_repeat, "score": float(per_repeat.mean()),
            "predictions": predictions, "prediction": predictions.mean(axis=0)}

    baseline = out[desc.BASELINE]["predictions"]
    for name in CANDIDATE_ORDER:
        out[name]["paired_baseline"] = rank.bootstrap_repeat_mean_difference(
            out[name]["predictions"], baseline, target, draws=BOOTSTRAP_DRAWS, seed=SEED)
    return out


def outperform(readings: dict) -> list[str]:
    """The rules that outperform the scale rule, in candidate order."""
    return [name for name in CANDIDATE_ORDER
            if name != desc.BASELINE and readings[name]["paired_baseline"]["q05"] > 0]


def selection_metrics(family: str, run_dir: Path, done: dict) -> dict:
    t0 = time.time()
    target_table = dd.targets(dd.FIT_KEY, family)
    design_all = dd.aligned(descriptor_table(dd.FIT_KEY, done), target_table)
    target = target_table["delta"].to_numpy(np.float64)
    weights = dd.weights(target_table)

    weighted = evaluate(design_all, target, weights)
    unweighted = evaluate(design_all, target, np.ones_like(weights))

    score = {name: weighted[name]["score"] for name in CANDIDATE_ORDER}
    beaters = outperform(weighted)

    candidate_rows, coefficient_rows = [], []
    for candidate in CANDIDATES:
        entry = weighted[candidate.name]
        columns = [f"log1p_{name}" for name in candidate.descriptors]
        design = design_all[columns].to_numpy(np.float64)
        keep = ~rule.constant_columns(design)
        intercept, coefficients = rule.fit(design, target, weights, keep)
        marginal = rank.bootstrap_repeat_mean(entry["predictions"], target,
                                              draws=BOOTSTRAP_DRAWS, seed=SEED)
        candidate_rows.append({
            "candidate": candidate.name,
            "properties": " ".join(candidate.descriptors),
            "score": entry["score"],
            **{f"score_r{i}": float(v) for i, v in enumerate(entry["per_repeat"])},
            "marginal_sd": marginal["sd"],
            "delta_rho_vs_baseline": entry["paired_baseline"]["delta"],
            "paired_sd_vs_baseline": entry["paired_baseline"]["sd"],
            "paired_q05_vs_baseline": entry["paired_baseline"]["q05"],
            "outperforms": candidate.name in beaters})
        coefficient_rows.append({
            "candidate": candidate.name, "term": "intercept", "value": intercept,
            "dropped": False})
        for name, value, kept in zip(candidate.descriptors, coefficients, keep):
            coefficient_rows.append({"candidate": candidate.name,
                                     "term": f"log1p_{name}", "value": float(value),
                                     "dropped": bool(not kept)})

    candidates_table = pd.DataFrame(candidate_rows).set_index("candidate")
    coefficients_table = pd.DataFrame(coefficient_rows).set_index("candidate")
    oof = pd.DataFrame({"delta": target, "se": target_table["se"].to_numpy(),
                        **{name: weighted[name]["prediction"] for name in CANDIDATE_ORDER}},
                       index=target_table.index)
    per_repeat = np.stack([weighted[name]["predictions"] for name in CANDIDATE_ORDER])

    return {"family": family, "dataset_key": dd.FIT_KEY, "k": int(len(target)),
            "candidates": len(CANDIDATES),
            "baseline": desc.BASELINE,
            "baseline_score": score[desc.BASELINE],
            "score": score,
            "outperform_the_scale_rule": beaters,
            "outperform_the_scale_rule_unweighted": outperform(unweighted),
            "hypothesis_2": "supported" if beaters else "refuted",
            "seconds": time.time() - t0,
            "candidate_order": CANDIDATE_ORDER,
            "artifacts": {"candidates": write_csv(run_dir, "candidates", candidates_table),
                          "coefficients": write_csv(run_dir, "coefficients",
                                                    coefficients_table),
                          "oof": write_csv(run_dir, "oof", oof),
                          "oof_repeats": write_array(run_dir, "oof_repeats",
                                                     per_repeat.astype(np.float64))}}


def cmd_selection(family: str, done: dict) -> dict:
    metrics = run(EXPERIMENT, selection_config(family),
                  lambda config, run_dir: selection_metrics(family, run_dir, done),
                  done=done)
    print(f"{SELECTION} {family}: the scale rule at rho "
          f"{metrics['baseline_score']:+.4f}; outperformed by "
          f"{metrics['outperform_the_scale_rule'] or 'no other rule'}; hypothesis 2 "
          f"{metrics['hypothesis_2']}", flush=True)
    return metrics


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    keys = sys.argv[1:] or list(dd.KEYS)
    unknown = [key for key in keys if key not in dd.data.DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset keys: {unknown}")
    done = completed(EXPERIMENT)
    if not cmd_gate(done):
        raise SystemExit("the check on the label-free properties failed; localise "
                         "the difference before reading any property")
    for key in keys:
        cmd_descriptors(key, done)
    if set(keys) >= {dd.FIT_KEY}:
        for family in dd.FAMILIES:
            cmd_selection(family, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
