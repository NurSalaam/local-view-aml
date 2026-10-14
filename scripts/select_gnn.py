#!/usr/bin/env python
"""The GNN selection — the GNN configuration search.

Six candidates on HI-Small in the global set-up, one Run per (candidate,
seed). Stages, each a pure function over the Run log:

  gate    candidate 1 at seeds 0-2 — the code-path equivalence gate against
          the anchor check's recorded HI-Small anchor; nothing else runs until it passes
  screen  candidates 2-6 at seed 0, ranked by best validation AP
  commit  the two highest screen APs at seeds 1-2 (seed 0 adopted)
  verdict the winner (3-seed mean validation AP; ties to fewer parameters),
          the anchor-role reading, and the frozen configuration

Usage:
  uv run python scripts/select_gnn.py plan <gate|screen|commit>
  uv run python scripts/select_gnn.py run <candidate:seed> [...]
  uv run python scripts/select_gnn.py launch <gate|screen|commit> [--shards 3]
  uv run python scripts/select_gnn.py gate | verdict | status

Execution settings come from the environment and are recorded in every Run
config: M_TORCH_THREADS (default 4), M_GNN_WORKERS (default 0), M_DEVICE
(default cpu). M_SMOKE=1 subsamples the world and shortens training for a
smoke test; it must be paired with M_EXPERIMENTS_DIR pointing away from the
real Run log.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("M_DEVICE", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import frame  # noqa: E402
from m import completed, device, run, seed_all  # noqa: E402
from m.artifacts import write_array, write_json, write_state  # noqa: E402
from m.features import VERSION as FEATURES_VERSION  # noqa: E402
from m.gnn import VERSION as GNN_VERSION  # noqa: E402
from m.features import usd_rate_table  # noqa: E402
from m.gnn import GNNConfig, edge_inputs, payment_graph, train_edge_scorer  # noqa: E402
from m.runlog import REPO_ROOT, config_hash, experiments_root, find, index  # noqa: E402
from m.stats import average_precision, minority_f1  # noqa: E402
from m.worlds.amlworld import GNN_AGGREGATES  # noqa: E402

EXPERIMENT = "select_gnn"
DATASET = "HI-Small"
SEEDS = [0, 1, 2]
SMOKE = os.environ.get("M_SMOKE") == "1"
SMOKE_FRAC = float(os.environ.get("M_SMOKE_FRAC", "0.03"))
if SMOKE and experiments_root().resolve() == (REPO_ROOT / "experiments").resolve():
    raise SystemExit("M_SMOKE=1 requires M_EXPERIMENTS_DIR pointing away from the "
                     "real Run log")
THREADS = int(os.environ.get("M_TORCH_THREADS", "4"))
WORKERS = int(os.environ.get("M_GNN_WORKERS", "0"))
EXECUTION = {"device": "cpu", "torch_threads": 4, "loader_workers": 0}
RELATIVE_COST = {1: 1.0, 2: 1.3, 3: 0.75, 4: 1.6, 5: 1.0, 6: 1.0}
ANCHOR_FLOOR = min(GNN_AGGREGATES[DATASET].values()) - 10.0
ANCHOR_CEILING = max(GNN_AGGREGATES[DATASET].values()) + 10.0
MEMORY_GUARD_GB = 8.0

BASE = dict(embed_dim=8, batch_size=4096, max_epochs=2 if SMOKE else 15,
            patience=3, loader_workers=WORKERS)
CANDIDATES: dict[int, GNNConfig] = {
    1: GNNConfig(hidden=64, fanout=(15, 15), lr=2e-3, pos_weight=6.0, **BASE),
    2: GNNConfig(hidden=128, fanout=(15, 15), lr=2e-3, pos_weight=6.0, **BASE),
    3: GNNConfig(hidden=64, fanout=(10, 10), lr=2e-3, pos_weight=6.0, **BASE),
    4: GNNConfig(hidden=64, fanout=(25, 25), lr=2e-3, pos_weight=6.0, **BASE),
    5: GNNConfig(hidden=64, fanout=(15, 15), lr=1e-3, pos_weight=6.0, **BASE),
    6: GNNConfig(hidden=64, fanout=(15, 15), lr=2e-3, pos_weight=12.0, **BASE),
}


def unit_config(candidate: int, seed: int) -> dict:
    config = {"experiment": EXPERIMENT, "check": "candidate_seed",
              "candidate": candidate, "world": "amlworld", "dataset": DATASET,
              "dataset_key": DATASET, "setup": "global", "head": "edge",
              "model": "graphsage_2layer_edgeattr",
              "gnn": CANDIDATES[candidate].as_dict(), "seed": seed,
              "features_version": FEATURES_VERSION, "gnn_version": GNN_VERSION,
              "device": os.environ.get("M_DEVICE", "cpu"),
              "torch_threads": THREADS}
    if SMOKE:
        config["smoke"] = {"frac": SMOKE_FRAC}
    return config


def verdict_config() -> dict:
    config = {"experiment": EXPERIMENT, "check": "verdict",
              "world": "amlworld", "dataset": DATASET, "dataset_key": DATASET,
              "setup": "global", "head": "edge",
              "model": "graphsage_2layer_edgeattr",
              "candidates": {str(k): v.as_dict() for k, v in CANDIDATES.items()},
              "seeds": SEEDS, "screen_seed": 0, "finalists": 2,
              "ranking": "best validation AP at early stop; commit winner by "
                         "3-seed mean; means within one pooled seed SD tie; "
                         "ties break towards fewer "
                         "parameters, then the lower candidate number",
              "anchor_role": {"dataset": DATASET, "metric": "test_f1_x100 mean "
                              "over seeds", "floor": ANCHOR_FLOOR,
                              "rule": "escalate only below the floor"},
              "features_version": FEATURES_VERSION, "gnn_version": GNN_VERSION,
              "device": os.environ.get("M_DEVICE", "cpu"),
              "torch_threads": THREADS}
    if SMOKE:
        config["smoke"] = {"frac": SMOKE_FRAC}
    return config


def parse_unit(text: str) -> tuple[int, int]:
    candidate, seed = text.split(":")
    return int(candidate), int(seed)


def unit_name(candidate: int, seed: int) -> str:
    return f"{candidate}:{seed}"


def done_index() -> dict:
    return completed(EXPERIMENT)


def recorded(done: dict, candidate: int, seed: int) -> dict | None:
    path = done.get(config_hash(unit_config(candidate, seed)))
    if path is None:
        return None
    return json.loads((path / "metrics.json").read_text())


def smoke_subsample(arrays, frac: float = SMOKE_FRAC):
    ts, payer, payee, numeric, codes, y, rows, n_nodes = arrays
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(ts), int(len(ts) * frac), replace=False))
    position = np.full(len(ts), -1, np.int64)
    position[keep] = np.arange(len(keep))
    rows = {k: position[v][position[v] >= 0] for k, v in rows.items()}
    return (ts[keep], payer[keep], payee[keep], numeric[keep], codes[keep],
            y[keep], rows, n_nodes)


def build_inputs() -> dict:
    df = frame(DATASET)
    arrays = edge_inputs(df, usd_rate_table(df))
    if SMOKE:
        arrays = smoke_subsample(arrays)
    ts, payer, payee, numeric, codes, y, rows, n_nodes = arrays
    return {"arrays": arrays,
            "data": payment_graph(ts, payer, payee, n_nodes)}


def make_seed_fn(inputs: dict, candidate: int, seed: int):
    ts, payer, payee, numeric, codes, y, rows, n_nodes = inputs["arrays"]

    def fn(config, run_dir: Path) -> dict:
        seed_all(seed)
        t0 = time.time()
        tag = f"[cand {candidate} seed {seed}]"
        result = train_edge_scorer(
            inputs["data"], payer, payee, ts, numeric, codes, y, rows,
            CANDIDATES[candidate], device(),
            log=lambda m: print(f"  {tag} {m}", flush=True),
            on_epoch=lambda trace: write_json(run_dir, "trace", trace))
        yte = y[rows["test"]]
        artifacts = {
            "val_probs": write_array(run_dir, "val_probs",
                                     result["val_probs"].astype(np.float32)),
            "test_probs": write_array(run_dir, "test_probs",
                                      result["test_probs"].astype(np.float32)),
            "best_state": write_state(run_dir, "best_state", result["best_state"]),
            "trace": write_json(run_dir, "trace", result["trace"]),
        }
        return {"candidate": candidate, "seed": seed,
                "val_ap": result["best_val_ap"],
                "test_ap": average_precision(yte, result["test_probs"]),
                "test_f1_x100": minority_f1(yte, result["test_probs"]) * 100,
                "epochs": len(result["trace"]),
                "n_parameters": result["n_parameters"],
                "test_positives": int(yte.sum()),
                "seconds": time.time() - t0,
                "trace": result["trace"],
                "artifacts": artifacts}
    return fn


def plan(stage: str, done: dict) -> list[tuple[int, int]]:
    """The (candidate, seed) units a stage still needs, in launch order."""
    if stage == "gate":
        wanted = [(1, s) for s in SEEDS]
    elif stage == "screen":
        wanted = [(c, 0) for c in CANDIDATES if c != 1]
    elif stage == "commit":
        wanted = [(c, s) for c in finalists(done) for s in SEEDS]
    else:
        raise SystemExit(f"unknown stage {stage}")
    return [(c, s) for c, s in wanted if recorded(done, c, s) is None]


def screen_table(done: dict) -> dict[int, dict]:
    table = {}
    for c in CANDIDATES:
        m = recorded(done, c, 0)
        if m is not None:
            table[c] = {"val_ap": m["val_ap"], "test_ap": m["test_ap"],
                        "test_f1_x100": m["test_f1_x100"],
                        "n_parameters": m["n_parameters"], "epochs": m["epochs"],
                        "seconds": m["seconds"]}
    return table


def finalists(done: dict) -> list[int]:
    table = screen_table(done)
    missing = [c for c in CANDIDATES if c not in table]
    if missing:
        raise SystemExit(f"screen incomplete: candidates {missing} have no "
                         f"seed-0 Run; the commit stage cannot be planned")
    ranked = sorted(table, key=lambda c: (-table[c]["val_ap"],
                                          table[c]["n_parameters"], c))
    return ranked[:2]


def gate_reading(done: dict) -> dict:
    """Candidate 1's three seeds under the new code path against the anchor check's
    recorded HI-Small anchor Run."""
    ref_dir = find(index("checks"), check="anchor_gnn", dataset_key=DATASET)
    ref = json.loads((ref_dir / "metrics.json").read_text())
    ref_ap = np.array([s["test_ap"] for s in ref["seeds"]])
    ref_f1 = np.array([s["test_f1_x100"] for s in ref["seeds"]])
    sd_ap, sd_f1 = ref_ap.std(ddof=1), ref_f1.std(ddof=1)
    band = {"test_ap": [float(ref_ap.min() - sd_ap), float(ref_ap.max() + sd_ap)],
            "test_f1_x100": [float(ref_f1.min() - sd_f1), float(ref_f1.max() + sd_f1)]}
    seeds = [recorded(done, 1, s) for s in SEEDS]
    reading = {"reference_run": ref_dir.name,
               "reference": {"test_ap": ref_ap.tolist(), "test_f1_x100": ref_f1.tolist(),
                             "sd_ap": float(sd_ap), "sd_f1": float(sd_f1)},
               "band": band,
               "complete": all(m is not None for m in seeds)}
    if reading["complete"]:
        ap = float(np.mean([m["test_ap"] for m in seeds]))
        f1 = float(np.mean([m["test_f1_x100"] for m in seeds]))
        reading |= {"seeds": [{"seed": m["seed"], "test_ap": m["test_ap"],
                               "test_f1_x100": m["test_f1_x100"],
                               "val_ap": m["val_ap"]} for m in seeds],
                    "test_ap_mean": ap, "test_f1_x100_mean": f1,
                    "pass": bool(band["test_ap"][0] <= ap <= band["test_ap"][1]
                                 and band["test_f1_x100"][0] <= f1 <= band["test_f1_x100"][1])}
    return reading


def make_verdict_fn():
    def fn(config, run_dir: Path) -> dict:
        done = done_index()
        table = screen_table(done)
        top = finalists(done)
        commit = {}
        for c in top:
            seeds = [recorded(done, c, s) for s in SEEDS]
            if any(m is None for m in seeds):
                raise SystemExit(f"commit incomplete for candidate {c}")
            commit[c] = {"val_ap_mean": float(np.mean([m["val_ap"] for m in seeds])),
                         "test_ap_mean": float(np.mean([m["test_ap"] for m in seeds])),
                         "test_f1_x100_mean": float(np.mean([m["test_f1_x100"] for m in seeds])),
                         "n_parameters": seeds[0]["n_parameters"],
                         "seeds": [{"seed": m["seed"], "val_ap": m["val_ap"],
                                    "test_ap": m["test_ap"],
                                    "test_f1_x100": m["test_f1_x100"],
                                    "epochs": m["epochs"]} for m in seeds]}
        seed_sd = {c: float(np.std([m["val_ap"] for m in commit[c]["seeds"]], ddof=1))
                   for c in top}
        pooled_sd = float(np.sqrt(np.mean([sd ** 2 for sd in seed_sd.values()])))
        difference = float(abs(commit[top[0]]["val_ap_mean"] - commit[top[1]]["val_ap_mean"]))
        tied_ap = difference <= pooled_sd
        if tied_ap:
            ranked = sorted(top, key=lambda c: (commit[c]["n_parameters"], c))
        else:
            ranked = sorted(top, key=lambda c: -commit[c]["val_ap_mean"])
        winner = ranked[0]
        tied_parameters = tied_ap and (commit[top[0]]["n_parameters"]
                                       == commit[top[1]]["n_parameters"])
        tie_break = ("none" if not tied_ap else
                     "fewer parameters" if not tied_parameters else
                     "lower candidate number")
        tie_rule = {"rule": "3-seed means within one pooled seed SD tie; "
                            "ties break towards fewer "
                            "parameters, then the lower candidate number",
                    "seed_sd": {str(c): v for c, v in seed_sd.items()},
                    "pooled_seed_sd": pooled_sd, "difference": difference,
                    "tied": bool(tied_ap),
                    "winner_under_original_four_decimal_rule":
                        sorted(top, key=lambda c: (-round(commit[c]["val_ap_mean"], 4),
                                                   commit[c]["n_parameters"], c))[0]}
        f1_winner = commit[winner]["test_f1_x100_mean"]
        anchor = {"dataset": DATASET, "published": GNN_AGGREGATES[DATASET],
                  "floor": ANCHOR_FLOOR, "band_ceiling": ANCHOR_CEILING,
                  "test_f1_x100_mean": f1_winner,
                  "below_floor": bool(f1_winner < ANCHOR_FLOOR),
                  "above_band": bool(f1_winner > ANCHOR_CEILING),
                  "escalation_authorised": bool(f1_winner < ANCHOR_FLOOR),
                  "source": "Altman 2024, Table 2 (the lowest "
                            "published HI-Small GNN aggregate minus 10 points "
                            "is the floor; the highest plus 10 the band ceiling)"}
        gate = gate_reading(done)
        if not gate["complete"]:
            raise SystemExit("gate incomplete: candidate 1 lacks a seed Run")
        return {"dataset_key": DATASET, "screen": {str(c): v for c, v in table.items()},
                "finalists": top, "commit": {str(c): v for c, v in commit.items()},
                "winner": winner, "tie_break": tie_break, "tie_rule": tie_rule,
                "frozen_configuration": CANDIDATES[winner].as_dict(),
                "anchor_role": anchor, "codepath_gate": gate,
                "frozen": bool(gate.get("pass", False)) and not anchor["below_floor"],
                "execution": {"device": os.environ.get("M_DEVICE", "cpu"),
                              "torch_threads": THREADS, "loader_workers": WORKERS}}
    return fn


def _exit_on_signal(signum, _frame):
    sys.exit(128 + signum)


def gate_required(done: dict, what: str) -> None:
    if SMOKE:
        return
    if not gate_reading(done).get("pass", False):
        raise SystemExit(f"{what} refused: the code-path equivalence gate has not "
                         f"passed; run `select_gnn.py launch gate` then `select_gnn.py gate`")


def cmd_run(units: list[str]) -> int:
    import torch

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, _exit_on_signal)
    torch.set_num_threads(THREADS)
    done = done_index()
    todo = [parse_unit(u) for u in units]
    todo = [(c, s) for c, s in todo if recorded(done, c, s) is None]
    if not todo:
        print("nothing to run: every unit is complete", flush=True)
        return 0
    print(f"building {DATASET} inputs for units {[unit_name(*u) for u in todo]}"
          f" (root {experiments_root()}, threads {THREADS}, workers {WORKERS}, "
          f"smoke {SMOKE})", flush=True)
    inputs = build_inputs()
    for candidate, seed in todo:
        if candidate != 1:
            gate_required(done, f"candidate {candidate}")
        metrics = run(EXPERIMENT, unit_config(candidate, seed),
                      make_seed_fn(inputs, candidate, seed), seed=seed, done=done)
        print(f"candidate {candidate} seed {seed}: val_ap {metrics['val_ap']:.4f} "
              f"test_ap {metrics['test_ap']:.4f} f1 {metrics['test_f1_x100']:.1f} "
              f"epochs {metrics['epochs']} ({metrics['seconds'] / 60:.1f} min)",
              flush=True)
    return 0


def free_memory_gb() -> float:
    import psutil

    return psutil.virtual_memory().available / 2**30


def balanced_lots(units: list[tuple[int, int]], shards: int) -> list[list[tuple[int, int]]]:
    """Deal units to shards greedily by relative cost, heaviest first."""
    lots: list[list[tuple[int, int]]] = [[] for _ in range(shards)]
    loads = [0.0] * shards
    for unit in sorted(units, key=lambda u: -RELATIVE_COST[u[0]]):
        i = loads.index(min(loads))
        lots[i].append(unit)
        loads[i] += RELATIVE_COST[unit[0]]
    return [lot for lot in lots if lot]


def cmd_launch(stage: str, shards: int) -> int:
    done = done_index()
    if stage in ("screen", "commit"):
        gate_required(done, f"launch {stage}")
    units = plan(stage, done)
    if not units:
        print(f"{stage}: nothing to run", flush=True)
        return 0
    lots = balanced_lots(units, shards)
    log_dir = Path(os.environ.get("M_LOG_DIR", REPO_ROOT / "logs" / "select_gnn"))
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{stage}: {len(units)} units in {len(lots)} shards -> {experiments_root()} "
          f"(device {os.environ.get('M_DEVICE', 'cpu')}, threads {THREADS}, "
          f"workers {WORKERS}, smoke {SMOKE}); logs in {log_dir}", flush=True)
    procs: list[tuple[int, subprocess.Popen, object]] = []

    def stop_children(signum, _frame):
        for _, proc, _ in procs:
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop_children)

    unlaunched: list[list[tuple[int, int]]] = []
    for i, lot in enumerate(lots):
        waited = 0
        while free_memory_gb() < MEMORY_GUARD_GB:
            if waited == 0:
                print(f"shard {i}: waiting for {MEMORY_GUARD_GB} GB free "
                      f"({free_memory_gb():.1f} GB now)", flush=True)
            time.sleep(30)
            waited += 30
            if waited > 3600:
                print(f"memory guard: waited an hour; shard {i} and later lots "
                      f"not launched", flush=True)
                unlaunched = lots[i:]
                break
        if unlaunched:
            break
        log_path = log_dir / f"{stage}-shard{i}.log"
        args = [sys.executable, str(Path(__file__).resolve()), "run",
                *[unit_name(*u) for u in lot]]
        print(f"shard {i}: {[unit_name(*u) for u in lot]} -> {log_path}", flush=True)
        handle = log_path.open("a")
        procs.append((i, subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT,
                                          env=os.environ.copy()), handle))
        if i + 1 < len(lots):
            time.sleep(20)
    failed = 0
    for i, proc, handle in procs:
        code = proc.wait()
        handle.close()
        print(f"shard {i}: exit {code}", flush=True)
        failed += code != 0
    if unlaunched:
        print(f"unlaunched lots: {[[unit_name(*u) for u in lot] for lot in unlaunched]}",
              flush=True)
    return 1 if failed or unlaunched else 0


def cmd_status() -> int:
    runs = experiments_root() / EXPERIMENT / "runs"
    if not runs.exists():
        print("no Runs yet")
        return 0
    rows = []
    for status_path in sorted(runs.glob("*/status.json")):
        status = json.loads(status_path.read_text())
        config = json.loads((status_path.parent / "config.json").read_text())
        last = ""
        trace_path = status_path.parent / "artifacts" / "trace.json"
        if trace_path.exists():
            trace = json.loads(trace_path.read_text())
            if trace:
                last = f"epoch {trace[-1]['epoch']} val_ap {trace[-1]['val_ap']:.4f}"
        rows.append((status_path.parent.name, config.get("check"),
                     config.get("candidate", "-"), config.get("seed", "-"),
                     status["status"], last, status.get("error", "")))
    for r in rows:
        print("  ".join(str(x) for x in r))
    return 0


def check_execution_settings() -> None:
    actual = {"device": os.environ.get("M_DEVICE", "cpu"), "torch_threads": THREADS,
              "loader_workers": WORKERS}
    if not SMOKE and actual != EXECUTION:
        raise SystemExit(f"execution settings {actual} differ from the fixed "
                         f"{EXECUTION}; set M_DEVICE, M_TORCH_THREADS and "
                         f"M_GNN_WORKERS accordingly")


def verdict_ready(done: dict) -> None:
    for c in finalists(done):
        missing = [s for s in SEEDS if recorded(done, c, s) is None]
        if missing:
            raise SystemExit(f"commit incomplete for candidate {c}: seeds {missing}")
    if not gate_reading(done)["complete"]:
        raise SystemExit("gate incomplete: candidate 1 lacks a seed Run")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    check_execution_settings()
    if cmd == "plan":
        units = plan(args[0], done_index())
        print(" ".join(unit_name(*u) for u in units))
        return 0
    if cmd == "run":
        return cmd_run(args)
    if cmd == "launch":
        shards = 3
        if "--shards" in args:
            shards = int(args[args.index("--shards") + 1])
        return cmd_launch(args[0], shards)
    if cmd == "gate":
        reading = gate_reading(done_index())
        print(json.dumps(reading, indent=2))
        return 0 if reading.get("pass") else 1
    if cmd == "verdict":
        verdict_ready(done_index())
        metrics = run(EXPERIMENT, verdict_config(), make_verdict_fn())
        print(json.dumps({k: metrics[k] for k in ("winner", "finalists", "frozen",
                                                  "tie_break")}, indent=2))
        print("anchor role:", json.dumps(metrics["anchor_role"], indent=2))
        return 0
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
