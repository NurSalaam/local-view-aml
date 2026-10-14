#!/usr/bin/env python
"""Experiment 2's four readings of the examples. Analysis
only: no model trains. Inputs are Experiment 1's persisted artefacts -- the per-payment
solo and global test scores, the shortfall Runs' banks.csv -- and the cached
per-bank ledgers (data/interim/exp1/ledgers/HI-Medium).

One ``never_seen`` Run reads, per roster bank, the share of its laundering
test payments arriving from a never-seen counterparty beside the same share
for all its test payments. One ``advantage`` Run per family
splits each bank's test payments by counterparty novelty, reads the pooled
model's AP advantage on each part, and correlates both with the bank's
measured shortfall. One ``examples`` Run per family correlates
the bank's training-window laundering payment count with its shortfall and
prints the quartile table. One ``partner_census`` Run
reads each bank's counterparty-bank concentration and how much of its
never-seen laundering counterparties' history the top partners hold.
Every config cites its inputs' Run hashes, so a re-done input
re-keys the analysis; skip-if-done is by config hash.

  uv run python scripts/exp2_examples.py never_seen
  uv run python scripts/exp2_examples.py advantage <family>
  uv run python scripts/exp2_examples.py examples <family>
  uv run python scripts/exp2_examples.py census
  uv run python scripts/exp2_examples.py all
  uv run python scripts/exp2_examples.py status
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
from extract import load_column  # noqa: E402
from m import completed, run  # noqa: E402
from m.artifacts import verify, write_csv  # noqa: E402
from m.rank import conditional_permutation_p, permutation_p  # noqa: E402
from m.runlog import config_hash, experiments_root  # noqa: E402
from m.stats import average_precision  # noqa: E402

EXPERIMENT = "exp2"
KEY = "HI-Medium"
FAMILIES = ("tree", "gnn", "blind")
RAIL = data.excluded_banks(KEY)[0]
PERMUTATIONS = 10_000
BLOCK_WIDTH = 5
SEED = 0
TOPS = (1, 5, 10, 25)

NEVER_SEEN = ("a cross-bank test payment whose counterparty account -- the "
              "endpoint at the other bank, keyed by its (bank, account) pair "
              "-- appears on none of the bank's training- or "
              "validation-window rows; a payment between two of the bank's "
              "own accounts has no counterparty and is never never-seen")
FRAME_PROBE = ("'never seen in the ledger' is also the cue the frame probe "
               "measured: on "
               "HI-Medium the bare novelty rule reads AP 0.159 on the ledger "
               "frame against 0.017 on the global frame, so the advantage "
               "split is net of a cue that favours the local view")
CORRELATIONS = ("Spearman rho with a two-sided permutation p over "
                f"{PERMUTATIONS} bank permutations at seed {SEED}; the "
                "partial by rank residualisation with the conditional "
                f"within-block permutation (width {BLOCK_WIDTH}, "
                f"{PERMUTATIONS} draws at seed {SEED}), holding the log of "
                "the bank's training-window rows fixed -- Experiment 3's "
                "method and scale variable (payment_count), unchanged")


def _hash_of(path: Path) -> str:
    return json.loads((path / "status.json").read_text())["config_hash"]


def banks_table(path: Path, metrics: dict) -> pd.DataFrame:
    return pd.read_csv(verify(path, metrics["artifacts"]["banks"]),
                       dtype={"bank": str}).set_index("bank")


def check_execution_settings() -> None:
    """The examples readings train nothing, so Experiment 1's thread settings do not bind; the
    committed-tree rule does, except on a scratch Run log."""
    scratch = experiments_root().resolve() != (exp1.REPO_ROOT / "experiments").resolve()
    if not scratch and exp1.committed_tree():
        raise SystemExit("Runs launch only from a committed tree; uncommitted: "
                         + ", ".join(exp1.committed_tree()[:12]))


# ---------------------------------------------------- counterparty views --

_VIEWS: dict[str, dict] = {}


def counterparty_view(bank: str) -> dict:
    """One bank's ledger read for novelty: the never-seen mask, the pre-test
    partner-bank counts, and the never-seen laundering counterparty keys.
    Cached per bank; a pure function of the cached ledger. The per-row string
    keys are not kept -- only the masks and the small key set survive."""
    if bank in _VIEWS:
        return _VIEWS[bank]
    ledger = data.read_ledger(KEY, bank)
    fb = ledger["from_bank"].astype(str)
    tb = ledger["to_bank"].astype(str)
    from_key = fb + ":" + ledger["from_acct"].astype(str)
    to_key = tb + ":" + ledger["to_acct"].astype(str)
    own_from = (fb == bank).to_numpy()
    cross = (fb != tb).to_numpy()
    cp_bank = np.where(own_from, tb.to_numpy(), fb.to_numpy())
    cp_key = np.where(own_from, to_key.to_numpy(), from_key.to_numpy())
    test = ledger["split"].to_numpy() == 2
    seen = pd.Index(pd.unique(pd.concat([from_key[~test], to_key[~test]],
                                        ignore_index=True)))
    never = cross & (seen.get_indexer(cp_key) < 0)
    y = ledger["is_laundering"].to_numpy() == 1
    partners = pd.Series(cp_bank[~test & cross])
    view = {"test": test, "cross": cross, "never": never, "y": y,
            "payment": ledger["payment"].to_numpy(),
            "never_laundering_keys": sorted(set(cp_key[test & y & never])),
            "partner_counts": partners[partners != RAIL].value_counts(),
            "rail_share_pretest_cross": float((partners == RAIL).mean())}
    _VIEWS[bank] = view
    return view


# ----------------------------------------------------------- never_seen --

def never_seen_config() -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": EXPERIMENT, "check": "never_seen", "world": world,
            "dataset": dataset, "dataset_key": KEY,
            "never_seen": NEVER_SEEN,
            "window": "prior appearance means any row with split < 2, the "
                      "training and validation windows together",
            "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(KEY)},
            "inputs": {"ledgers": f"data/interim/exp1/ledgers/{KEY}"}}


def never_seen_metrics(run_dir: Path) -> dict:
    t0 = time.time()
    roster = data.read_roster(KEY)
    rows = []
    for bank in roster.index:
        view = counterparty_view(bank)
        test, y, never = view["test"], view["y"], view["never"]
        cross = view["cross"]
        laundering = test & y
        rows.append({
            "bank": bank,
            "test_rows": int(test.sum()), "test_pos": int(laundering.sum()),
            "never_seen_share_laundering": float(never[laundering].mean()),
            "never_seen_share_all": float(never[test].mean()),
            "cross_share_laundering": float(cross[laundering].mean()),
            "cross_share_all": float(cross[test].mean()),
            "never_seen_laundering": int(never[laundering].sum()),
            "never_seen_all": int(never[test].sum())})
    table = pd.DataFrame(rows).set_index("bank")
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    summary = {c: [float(v) for v in table[c].quantile(quantiles)]
               for c in ("never_seen_share_laundering", "never_seen_share_all",
                         "cross_share_laundering", "cross_share_all")}
    pooled = {
        "laundering": float(table["never_seen_laundering"].sum()
                            / table["test_pos"].sum()),
        "all": float(table["never_seen_all"].sum() / table["test_rows"].sum())}
    return {"dataset_key": KEY, "k": len(table),
            "pooled_never_seen_share": pooled,
            "quantiles_min_q25_median_q75_max": summary,
            "banks_where_laundering_share_exceeds_all": int(
                (table["never_seen_share_laundering"]
                 > table["never_seen_share_all"]).sum()),
            "seconds": time.time() - t0,
            "artifacts": {"never_seen": write_csv(run_dir, "never_seen", table)}}


def cmd_never_seen(done: dict | None = None) -> dict:
    done = done if done is not None else completed(EXPERIMENT)
    metrics = run(EXPERIMENT, never_seen_config(),
                  lambda config, run_dir: never_seen_metrics(run_dir),
                  done=done)
    p = metrics["pooled_never_seen_share"]
    q = metrics["quantiles_min_q25_median_q75_max"]
    print(f"never_seen: k {metrics['k']} pooled laundering {p['laundering']:.3f} "
          f"all {p['all']:.3f} median laundering "
          f"{q['never_seen_share_laundering'][2]:.3f} "
          f"all {q['never_seen_share_all'][2]:.3f}", flush=True)
    return metrics


# ------------------------------------------------------------ advantage --

def advantage_inputs(done_exp1: dict, family: str) -> dict | None:
    dirs = exp1_analysis.unit_dirs(done_exp1, KEY, family)
    if dirs is None:
        return None
    shortfall = exp1_analysis.shortfall_run(done_exp1, KEY, family)
    if shortfall is None:
        return None
    return {"dirs": dirs, "shortfall": shortfall}


def advantage_config(family: str, inputs: dict) -> dict:
    world, dataset = data.DATASETS[KEY]
    dirs = inputs["dirs"]
    return {"experiment": EXPERIMENT, "check": "advantage", "world": world,
            "dataset": dataset, "dataset_key": KEY, "family": family,
            "never_seen": NEVER_SEEN, "frame_probe": FRAME_PROBE,
            "parts": "the never-seen part, and the remainder (a counterparty "
                     "seen before the test window, or a payment between the "
                     "bank's own accounts)",
            "pairing": exp1_analysis.PAIRING[family],
            "advantage": "per part, AP(global) - AP(solo) at the same "
                         "setting, aggregated as the family's Experiment 1 pairing "
                         "aggregates; a part with no laundering payment "
                         "reads NaN and leaves the correlation",
            "statistics": CORRELATIONS,
            "target": "the bank's Experiment 1 shortfall (banks.csv delta)",
            "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(KEY)},
            "inputs": {"shortfall": _hash_of(inputs["shortfall"][0]),
                       "global": [_hash_of(d) for d in dirs["global"]],
                       "solo": {b: [_hash_of(d) for d in ds]
                                for b, ds in dirs["solo"].items()}}}


def correlation_block(x: np.ndarray, delta: np.ndarray,
                      log_rows: np.ndarray) -> dict:
    usable = np.isfinite(x)
    raw = permutation_p(x[usable], delta[usable], alternative="two-sided",
                        draws=PERMUTATIONS, seed=SEED)
    partial = conditional_permutation_p(x[usable], delta[usable],
                                        log_rows[usable], width=BLOCK_WIDTH,
                                        draws=PERMUTATIONS, seed=SEED)
    return {"k": int(usable.sum()), "excluded_nan": int((~usable).sum()),
            "rho": raw["rho"], "p": raw["p"],
            "partial_rho": partial["rho"],
            "partial_p_conditional": partial["p_conditional"],
            "partial_p_asymptotic": partial["p_asymptotic"]}


def advantage_metrics(family: str, inputs: dict, run_dir: Path) -> dict:
    t0 = time.time()
    dirs = inputs["dirs"]
    cell = exp1_analysis.load_cell(KEY, family, dirs)
    split = load_column(KEY, "split")
    test_positions = np.flatnonzero(split == 2)
    banks = banks_table(*inputs["shortfall"])
    aggregate = {"median": np.median, "mean": np.mean}[
        exp1_analysis.AGGREGATE[exp1_analysis.PAIRING[family]]]
    rows = []
    for bank, spec in cell["groups"].items():
        view = counterparty_view(bank)
        test = view["test"]
        assert np.array_equal(test_positions[spec["idx"]],
                              view["payment"][test]), \
            f"{bank}: the scored paper and the ledger's test rows differ"
        y = spec["y"].astype(bool)
        assert np.array_equal(y, view["y"][test]), \
            f"{bank}: labels differ between the scores and the ledger"
        never = view["never"][test]
        record = {"bank": bank, "n_test": len(y), "test_pos": int(y.sum())}
        for name, part in (("never_seen", never), ("rest", ~never)):
            positives = int(y[part].sum())
            record[f"{name}_rows"] = int(part.sum())
            record[f"{name}_pos"] = positives
            if positives == 0:
                record[f"{name}_solo_ap"] = np.nan
                record[f"{name}_global_ap"] = np.nan
                record[f"{name}_advantage"] = np.nan
                continue
            solo = [average_precision(y[part], spec["solo"][s][part])
                    for s in range(spec["solo"].shape[0])]
            glob = [average_precision(y[part], spec["global"][s][part])
                    for s in range(spec["global"].shape[0])]
            record[f"{name}_solo_ap"] = float(aggregate(solo))
            record[f"{name}_global_ap"] = float(aggregate(glob))
            record[f"{name}_advantage"] = float(aggregate(
                np.array(glob) - np.array(solo)))
        rows.append(record)
    table = pd.DataFrame(rows).set_index("bank")
    table["delta"] = banks.loc[table.index, "delta"]
    table["se"] = banks.loc[table.index, "se"]
    roster = dirs["roster"]
    table["train_rows"] = roster.loc[table.index, "train_rows"]
    table["train_pos"] = roster.loc[table.index, "train_pos"]

    delta = table["delta"].to_numpy()
    log_rows = np.log(table["train_rows"].to_numpy(np.float64))
    correlations = {name: correlation_block(
        table[f"{name}_advantage"].to_numpy(), delta, log_rows)
        for name in ("never_seen", "rest")}
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    levels = {name: {
        "advantage_min_q25_median_q75_max": [
            float(v) for v in table[f"{name}_advantage"].quantile(quantiles)],
        "banks_with_positives": int(table[f"{name}_pos"].gt(0).sum())}
        for name in ("never_seen", "rest")}
    return {"dataset_key": KEY, "family": family, "k": len(table),
            "pairing": exp1_analysis.PAIRING[family],
            "correlation_with_shortfall": correlations,
            "advantage_levels": levels, "frame_probe": FRAME_PROBE,
            "seconds": time.time() - t0,
            "artifacts": {"advantage": write_csv(run_dir, "advantage", table)}}


def cmd_advantage(family: str, done: dict | None = None) -> dict | None:
    done = done if done is not None else completed(EXPERIMENT)
    inputs = advantage_inputs(completed(exp1.EXPERIMENT), family)
    if inputs is None:
        print(f"advantage {family}: Experiment 1 inputs incomplete", flush=True)
        return None
    metrics = run(EXPERIMENT, advantage_config(family, inputs),
                  lambda config, run_dir: advantage_metrics(family, inputs,
                                                            run_dir),
                  done=done)
    c = metrics["correlation_with_shortfall"]
    print(f"advantage {family}: k {metrics['k']} never-seen rho "
          f"{c['never_seen']['rho']:+.3f} (p {c['never_seen']['p']:.4f}, "
          f"partial {c['never_seen']['partial_rho']:+.3f} p "
          f"{c['never_seen']['partial_p_conditional']:.4f}) rest rho "
          f"{c['rest']['rho']:+.3f} (p {c['rest']['p']:.4f}, partial "
          f"{c['rest']['partial_rho']:+.3f} p "
          f"{c['rest']['partial_p_conditional']:.4f})", flush=True)
    return metrics


# ------------------------------------------------------------- examples --

def examples_config(family: str, shortfall: tuple[Path, dict]) -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": EXPERIMENT, "check": "examples", "world": world,
            "dataset": dataset, "dataset_key": KEY, "family": family,
            "predictor": "the bank's training-window laundering payment "
                         "count (roster train_pos, either-endpoint "
                         "attribution)",
            "target": "the bank's Experiment 1 shortfall (banks.csv delta)",
            "statistics": CORRELATIONS,
            "quartiles": "pandas qcut over the count, four bins; per "
                         "quartile the count range, the unweighted mean "
                         "shortfall, and the banks below zero",
            "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(KEY)},
            "inputs": {"shortfall": _hash_of(shortfall[0])}}


def examples_metrics(family: str, shortfall: tuple[Path, dict],
                     run_dir: Path) -> dict:
    banks = banks_table(*shortfall)
    roster = data.read_roster(KEY)
    table = banks[["delta", "se", "test_rows", "test_pos"]].copy()
    table["train_pos"] = roster.loc[table.index, "train_pos"]
    table["train_rows"] = roster.loc[table.index, "train_rows"]
    train_pos = table["train_pos"].to_numpy(np.float64)
    delta = table["delta"].to_numpy()
    log_rows = np.log(table["train_rows"].to_numpy(np.float64))
    correlation = correlation_block(train_pos, delta, log_rows)
    table["quartile"] = pd.qcut(table["train_pos"], 4, labels=False)
    quartiles = []
    for q, group in table.groupby("quartile"):
        quartiles.append({
            "quartile": int(q) + 1,
            "count_range": [int(group["train_pos"].min()),
                            int(group["train_pos"].max())],
            "banks": int(len(group)),
            "mean_shortfall": float(group["delta"].mean()),
            "banks_negative": int((group["delta"] < 0).sum())})
    return {"dataset_key": KEY, "family": family, "k": len(table),
            "correlation_with_shortfall": correlation,
            "quartile_table": quartiles,
            "artifacts": {"examples": write_csv(run_dir, "examples", table)}}


def cmd_examples(family: str, done: dict | None = None) -> dict | None:
    done = done if done is not None else completed(EXPERIMENT)
    shortfall = exp1_analysis.shortfall_run(completed(exp1.EXPERIMENT), KEY, family)
    if shortfall is None:
        print(f"examples {family}: Experiment 1 shortfall Run not found", flush=True)
        return None
    metrics = run(EXPERIMENT, examples_config(family, shortfall),
                  lambda config, run_dir: examples_metrics(family, shortfall,
                                                           run_dir),
                  done=done)
    c = metrics["correlation_with_shortfall"]
    cells = ", ".join(f"{q['count_range'][0]}-{q['count_range'][1]}: "
                      f"{q['mean_shortfall']:+.3f} ({q['banks_negative']} neg)"
                      for q in metrics["quartile_table"])
    print(f"examples {family}: rho {c['rho']:+.3f} (p {c['p']:.4f}) partial "
          f"{c['partial_rho']:+.3f} (p {c['partial_p_conditional']:.4f}) "
          f"quartiles {cells}", flush=True)
    return metrics


# ------------------------------------------------------- partner census --

def census_config() -> dict:
    world, dataset = data.DATASETS[KEY]
    return {"experiment": EXPERIMENT, "check": "partner_census",
            "world": world, "dataset": dataset, "dataset_key": KEY,
            "never_seen": NEVER_SEEN,
            "window": "every census quantity reads the bank's pre-test rows "
                      "(split < 2), the window a bank holds when it chooses "
                      "partners; the counterparties whose history is sought "
                      "are the test window's never-seen laundering ones",
            "counterparties": "distinct never-seen laundering counterparty "
                              "accounts, each counted once however many "
                              "payments it sent",
            "candidates": "counterparty banks ranked by the bank's pre-test "
                          "cross-bank rows, the payment rail and the bank "
                          "itself excluded; a candidate outside the roster "
                          "is read from the rows the roster ledgers hold, "
                          "an undercount, and the roster share of each "
                          "bank's top candidates is reported",
            "rail": {"bank": RAIL, "reason": "excluded from the roster as a "
                     "payment rail; not a "
                     "bank a records agreement could be struck with"},
            "tops": list(TOPS),
            "pooled": "history at any roster bank other than the bank itself",
            "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(KEY)},
            "inputs": {"ledgers": f"data/interim/exp1/ledgers/{KEY}"}}


def history_holders(query: set[str]) -> dict[str, set[str]]:
    """For each queried (bank, account) key, the banks whose ledger holds a
    pre-test row with that account. Every row of every roster ledger sits in
    both endpoint banks' ledgers, so one pass over the roster ledgers reads
    a roster holder exactly and a non-roster holder as far as the roster's
    rows reveal it."""
    index = pd.Index(sorted(query))
    holders: dict[str, set[str]] = {key: set() for key in query}
    for bank in data.read_roster(KEY).index:
        ledger = data.read_ledger(KEY, bank)
        pre = ledger["split"].to_numpy() < 2
        fb = ledger["from_bank"].astype(str)[pre]
        tb = ledger["to_bank"].astype(str)[pre]
        from_key = (fb + ":" + ledger["from_acct"].astype(str)[pre]).to_numpy()
        to_key = (tb + ":" + ledger["to_acct"].astype(str)[pre]).to_numpy()
        fb, tb = fb.to_numpy(), tb.to_numpy()
        for keys in (from_key, to_key):
            hit = index.get_indexer(keys) >= 0
            for key, payer, payee in zip(keys[hit], fb[hit], tb[hit]):
                holders[key].update((payer, payee))
    return holders


def census_metrics(run_dir: Path) -> dict:
    t0 = time.time()
    roster = data.read_roster(KEY)
    targets = {bank: set(counterparty_view(bank)["never_laundering_keys"])
               for bank in roster.index}
    holders = history_holders(set().union(*targets.values()))
    roster_banks = set(roster.index)
    rows = []
    for bank in roster.index:
        view = counterparty_view(bank)
        counts = view["partner_counts"]
        shares = counts / counts.sum()
        cumulative = shares.cumsum()
        ranked = list(counts.index)
        keys = sorted(targets[bank])
        held = [holders[key] - {RAIL, bank} for key in keys]
        record = {
            "bank": bank,
            "partner_banks": int(len(counts)),
            "top1_share": float(shares.iloc[:1].sum()),
            "top5_share": float(shares.iloc[:5].sum()),
            "top10_share": float(shares.iloc[:10].sum()),
            "partners_to_half": int(np.searchsorted(cumulative.to_numpy(),
                                                    0.5) + 1),
            "rail_share": view["rail_share_pretest_cross"],
            "never_seen_laundering_counterparties": len(keys),
            "roster_share_top25": float(np.mean(
                [c in roster_banks for c in ranked[:25]])) if ranked else np.nan}
        for k in TOPS:
            top = set(ranked[:k])
            record[f"history_top{k}"] = (
                float(np.mean([bool(h & top) for h in held]))
                if keys else np.nan)
        record["history_all_roster"] = (
            float(np.mean([bool(h & roster_banks) for h in held]))
            if keys else np.nan)
        rows.append(record)
    table = pd.DataFrame(rows).set_index("bank")
    quantiles = [0.0, 0.25, 0.5, 0.75, 1.0]
    summary = {c: [float(v) for v in table[c].quantile(quantiles)]
               for c in table.columns if table[c].notna().any()}
    weights = table["never_seen_laundering_counterparties"]
    pooled = {c: float((table[c] * weights).sum() / weights.sum())
              for c in [f"history_top{k}" for k in TOPS]
              + ["history_all_roster"]}
    return {"dataset_key": KEY, "k": len(table),
            "queried_counterparties": int(weights.sum()),
            "pooled_history_share": pooled,
            "quantiles_min_q25_median_q75_max": summary,
            "seconds": time.time() - t0,
            "artifacts": {"census": write_csv(run_dir, "census", table)}}


def cmd_census(done: dict | None = None) -> dict:
    done = done if done is not None else completed(EXPERIMENT)
    metrics = run(EXPERIMENT, census_config(),
                  lambda config, run_dir: census_metrics(run_dir),
                  done=done)
    q = metrics["quantiles_min_q25_median_q75_max"]
    p = metrics["pooled_history_share"]
    print(f"census: k {metrics['k']} median partners {q['partner_banks'][2]:.0f} "
          f"top1 share {q['top1_share'][2]:.3f} to-half "
          f"{q['partners_to_half'][2]:.0f} pooled history top1 "
          f"{p['history_top1']:.3f} top10 {p['history_top10']:.3f} all "
          f"{p['history_all_roster']:.3f}", flush=True)
    return metrics


# ----------------------------------------------------------------- main --

def cmd_all() -> int:
    done = completed(EXPERIMENT)
    cmd_never_seen(done)
    for family in FAMILIES:
        cmd_advantage(family, done)
        cmd_examples(family, done)
    cmd_census(done)
    return 0


def cmd_status() -> int:
    done = completed(EXPERIMENT)
    done_exp1 = completed(exp1.EXPERIMENT)
    print(f"never_seen: {'analysed' if config_hash(never_seen_config()) in done else 'ready'}")
    for family in FAMILIES:
        inputs = advantage_inputs(done_exp1, family)
        state = ("inputs incomplete" if inputs is None else
                 "analysed" if config_hash(advantage_config(family, inputs))
                 in done else "ready")
        print(f"advantage {family:6s} {state}")
        shortfall = exp1_analysis.shortfall_run(done_exp1, KEY, family)
        state = ("inputs incomplete" if shortfall is None else
                 "analysed" if config_hash(examples_config(family, shortfall))
                 in done else "ready")
        print(f"examples  {family:6s} {state}")
    print(f"partner_census: {'analysed' if config_hash(census_config()) in done else 'ready'}")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    check_execution_settings()
    exp1.install_signal_handlers()
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "never_seen":
        return 0 if cmd_never_seen() else 1
    if cmd == "advantage":
        return 0 if cmd_advantage(args[0]) is not None else 1
    if cmd == "examples":
        return 0 if cmd_examples(args[0]) is not None else 1
    if cmd == "census":
        return 0 if cmd_census() else 1
    if cmd == "all":
        return cmd_all()
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
