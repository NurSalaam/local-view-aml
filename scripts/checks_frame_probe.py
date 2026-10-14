#!/usr/bin/env python
"""The frame probe.

The census checks ask whether a world's banks are interchangeable. None of them
asks whether the local frame itself creates predictive signal, as it does on
the released AMLgentex data: inside one bank's
ledger the trivial score "minus the earlier appearances of the counterparty"
carries a detector by itself, while the same score on the global frame reads
chance. The probe measures that property for any dataset, before any model
trains on it.

For every roster bank the probe scores that bank's exam paper by each
trivial rule (`m.frame`) twice — once with the rule's history
counted in the bank's own ledger, once with it counted on the global frame —
and reports the two average precisions and their gap. Three of the rules
cannot depend on the frame, and their two readings must be identical; the
probe asserts that, which tests the history arithmetic against itself across
the two frames rather than the identity of the exam paper. The paper is
pinned separately: each bank's row carries the sha256 of the payment indices
scored, in the form Experiment 1's solo Runs persist them, so the two logs can be
compared without trusting either. Beside the rules the probe tabulates the
laundering rate by the external endpoint's ledger history, split by whether
the payment crosses banks. Nothing is trained, and every number is a function
of the canonical frame alone.

LightGBM is never imported here; nothing in the probe needs it, and its OpenMP
runtime clashes with torch's on macOS when it loads first (see `scripts/exp1.py`).
Torch enters only through the per-Run fingerprint, after everything else.

  uv run python scripts/checks_frame_probe.py                 # every dataset
  uv run python scripts/checks_frame_probe.py HI-Medium       # one dataset
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

import exp1_data as data  # noqa: E402
from m import completed, run, track  # noqa: E402
from m import frame as rules  # noqa: E402
from m.artifacts import sha256_of_array, write_csv  # noqa: E402
from m.features import VERSION as FEATURES_VERSION  # noqa: E402
from m.features import account_codes, own_fields, usd_rate_table  # noqa: E402
from m.stats import average_precision  # noqa: E402

EXPERIMENT = "checks"
CHECK = "frame_probe"
BUCKETS = 5
KEYS = ["transxion", data.AMLGENTEX_DATASET,
        "HI-Medium", "HI-Small", "LI-Small", "LI-Medium"]


def probe_config(key: str) -> dict:
    return {"experiment": EXPERIMENT, "check": CHECK,
            "world": data.DATASETS[key][0], "dataset": data.DATASETS[key][1],
            "dataset_key": key,
            "roster": {"min_test_pos": data.MIN_TEST_POS,
                       "exclude": data.excluded_banks(key)},
            "frames": ["ledger", "global"],
            "rules": {name: rules.RULE_DESCRIPTIONS[name] for name in rules.RULES},
            "frame_invariant_rules": list(rules.FRAME_INVARIANT),
            "history_buckets": BUCKETS,
            "amount_rule": "the dataset-level USD table computed from the global "
                           "training window (m.features.usd_rate_table)",
            "probe_version": rules.VERSION, "features_version": FEATURES_VERSION}


def probe_metrics(key: str, run_dir: Path) -> dict:
    t0 = time.time()
    df = data.frame(key)
    roster = data.roster(df, key)
    from_bank, to_bank, _ = data.bank_codes(df)
    payer, payee = account_codes(df)
    ts = df["ts"].to_numpy(np.int64)
    y = df["is_laundering"].to_numpy(np.int8)
    split = df["split"].to_numpy(np.int8)
    amount_usd = own_fields(df, usd_rate_table(df))["amount_usd"]
    n_rows = len(df)
    del df

    payer_hist, payee_hist = rules.endpoint_history(payer, payee, ts)
    pair_hist = rules.pair_history(payer, payee, ts)

    rows, buckets, invariance = [], [], 0.0
    for bank in track(list(roster.index), f"frame probe {key}"):
        code = int(roster.loc[bank, "code"])
        idx = np.flatnonzero((from_bank == code) | (to_bank == code))
        test = split[idx] == 2
        paper = idx[test]
        own_side = from_bank[idx] == code
        ledger_payer, ledger_payee = rules.endpoint_history(payer[idx], payee[idx], ts[idx])
        ledger_pair = rules.pair_history(payer[idx], payee[idx], ts[idx])
        ledger = rules.rule_scores(ledger_payer[test], ledger_payee[test],
                                   ledger_pair[test], own_side[test],
                                   amount_usd[idx][test])
        globally = rules.rule_scores(payer_hist[paper], payee_hist[paper],
                                     pair_hist[paper], from_bank[paper] == code,
                                     amount_usd[paper])
        labels = y[paper]

        row = {"bank": bank, "ledger_rows": int(len(idx)), "test_rows": int(test.sum()),
               "test_pos": int(labels.sum()), "base_rate": float(labels.mean()),
               "cross_bank_share_test": float((from_bank[paper] != to_bank[paper]).mean()),
               "test_payment_sha256": sha256_of_array(paper)}
        for name in rules.RULES:
            local_ap = average_precision(labels, ledger[name])
            global_ap = average_precision(labels, globally[name])
            row[f"ledger_ap_{name}"] = local_ap
            row[f"global_ap_{name}"] = global_ap
            row[f"gap_{name}"] = local_ap - global_ap
            if name in rules.FRAME_INVARIANT:
                difference = float(np.abs(ledger[name] - globally[name]).max())
                assert difference == 0.0, (
                    f"bank {bank}: the frame-invariant rule {name!r} read "
                    f"differently on the two frames (max |difference| "
                    f"{difference:g}), so the ledger history and the global "
                    f"history disagree where they cannot")
                invariance = max(invariance, difference)

        history = np.minimum(rules.external_history(ledger_payer[test], ledger_payee[test],
                                                    own_side[test]), BUCKETS)
        novel, familiar = history == 0, history == BUCKETS
        row["novel_external_share"] = float(novel.mean())
        row["novel_external_sar_rate"] = float(labels[novel].mean()) if novel.any() else np.nan
        row["familiar_external_sar_rate"] = (float(labels[familiar].mean())
                                             if familiar.any() else np.nan)
        rows.append(row)

        cross = (from_bank[paper] != to_bank[paper])
        for crosses in (False, True):
            for bucket in range(BUCKETS + 1):
                mask = (cross == crosses) & (history == bucket)
                if mask.any():
                    buckets.append({"bank": bank, "cross_bank": crosses,
                                    "external_history": bucket,
                                    "rows": int(mask.sum()),
                                    "sar_rate": float(labels[mask].mean())})

    table = pd.DataFrame(rows).set_index("bank")
    bucket_table = pd.DataFrame(buckets)
    pooled = (bucket_table.groupby(["cross_bank", "external_history"])
              .apply(lambda g: pd.Series({"rows": int(g["rows"].sum()),
                                          "sar_rate": float((g["rows"] * g["sar_rate"]).sum()
                                                            / g["rows"].sum())}),
                     include_groups=False).reset_index())
    rule_ap = {name: {"ledger_median": float(table[f"ledger_ap_{name}"].median()),
                      "global_median": float(table[f"global_ap_{name}"].median()),
                      "gap_median": float(table[f"gap_{name}"].median()),
                      "gap_range": [float(table[f"gap_{name}"].min()),
                                    float(table[f"gap_{name}"].max())]}
               for name in rules.RULES}
    rule_table = pd.DataFrame([
        {"rule": name, "frame_invariant": name in rules.FRAME_INVARIANT,
         "ledger_ap_median": entry["ledger_median"],
         "global_ap_median": entry["global_median"],
         "gap_median": entry["gap_median"],
         "gap_min": entry["gap_range"][0], "gap_max": entry["gap_range"][1]}
        for name, entry in rule_ap.items()]).set_index("rule")

    def pooled_rate(crosses: bool, bucket: int) -> float | None:
        hit = pooled[(pooled.cross_bank == crosses) & (pooled.external_history == bucket)]
        return float(hit["sar_rate"].iloc[0]) if len(hit) else None

    base_rate_median = float(table["base_rate"].median())
    return {
        "dataset_key": key, "world": data.DATASETS[key][0], "k": int(len(table)),
        "n_rows": n_rows, "n_test_rows": int(table["test_rows"].sum()),
        "base_rate_median": base_rate_median,
        "rule_ap": rule_ap,
        "cue_strength": {"rule": "external", **rule_ap["external"],
                         "lift_over_base_rate":
                             rule_ap["external"]["ledger_median"] / base_rate_median},
        "frame_invariance": {"rules": list(rules.FRAME_INVARIANT),
                             "max_abs_score_difference": invariance, "holds": True},
        "novel_external": {
            "share_of_test_rows_median": float(table["novel_external_share"].median()),
            "sar_rate_median": float(table["novel_external_sar_rate"].median()),
            "familiar_sar_rate_median": float(table["familiar_external_sar_rate"].median())},
        "external_history_zero_cross_bank_sar_rate": pooled_rate(True, 0),
        "external_history_cap_cross_bank_sar_rate": pooled_rate(True, BUCKETS),
        "seconds": time.time() - t0,
        "artifacts": {"banks": write_csv(run_dir, "banks", table),
                      "rules": write_csv(run_dir, "rules", rule_table),
                      "roster": write_csv(run_dir, "roster", roster),
                      "history_buckets": write_csv(run_dir, "history_buckets",
                                                   bucket_table.set_index("bank")),
                      "history_buckets_pooled": write_csv(
                          run_dir, "history_buckets_pooled",
                          pooled.set_index(["cross_bank", "external_history"]))},
    }


def cmd_probe(key: str, done: dict) -> dict:
    metrics = run(EXPERIMENT, probe_config(key),
                  lambda config, run_dir: probe_metrics(key, run_dir), done=done)
    cue = metrics["cue_strength"]
    print(f"frame_probe {key}: k {metrics['k']} novelty AP {cue['ledger_median']:.4f} "
          f"(ledger) against {cue['global_median']:.4f} (global), gap "
          f"{cue['gap_median']:+.4f}, base rate {metrics['base_rate_median']:.4f}",
          flush=True)
    return metrics


def main() -> int:
    keys = sys.argv[1:] or KEYS
    unknown = [key for key in keys if key not in data.DATASETS]
    if unknown:
        raise SystemExit(f"unknown dataset keys: {unknown}")
    done = completed(EXPERIMENT)
    for key in keys:
        cmd_probe(key, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
