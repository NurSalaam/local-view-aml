#!/usr/bin/env python
"""The GNN selection: the search for the GNN family's configuration
(Appendix B).

Six candidates on HI-Small at (network, all), one Run per (candidate, seed).
Stages, each a pure function over the Run log:

  screen  candidates 1-6 at seed 0, ranked by their best validation average
          precision
  commit  the two candidates highest at the screen, fitted at seeds 1-2 (their
          seed-0 Runs reused)
  verdict the candidate kept: the higher three-seed mean validation average
          precision, unless the two means differ by no more than their pooled
          seed standard deviation, when the candidate with fewer parameters is
          kept, then the lower candidate number

Usage:
  uv run python scripts/select_gnn.py plan <screen|commit>
  uv run python scripts/select_gnn.py run <candidate:seed> [...]
  uv run python scripts/select_gnn.py launch <screen|commit> [--shards 3]
  uv run python scripts/select_gnn.py verdict | status

Execution settings come from the environment and are recorded in every Run
config: M_TORCH_THREADS (default 4), M_GNN_WORKERS (default 0), M_DEVICE
(default cpu); the script refuses other values.
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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import frame  # noqa: E402
from m import completed, device, run, seed_all  # noqa: E402
from m.artifacts import write_array, write_json, write_state  # noqa: E402
from m.features import VERSION as FEATURES_VERSION  # noqa: E402
from m.gnn import VERSION as GNN_VERSION  # noqa: E402
from m.features import usd_rate_table  # noqa: E402
from m.gnn import GNNConfig, edge_inputs, payment_graph, train_edge_scorer  # noqa: E402
from m.runlog import REPO_ROOT, config_hash, experiments_root  # noqa: E402
from m.stats import average_precision, minority_f1  # noqa: E402

EXPERIMENT = "select_gnn"
DATASET = "HI-Small"
SEEDS = [0, 1, 2]
THREADS = int(os.environ.get("M_TORCH_THREADS", "4"))
WORKERS = int(os.environ.get("M_GNN_WORKERS", "0"))
EXECUTION = {"device": "cpu", "torch_threads": 4, "loader_workers": 0}
RELATIVE_COST = {1: 1.0, 2: 1.3, 3: 0.75, 4: 1.6, 5: 1.0, 6: 1.0}
MEMORY_GUARD_GB = 8.0
TIE_RULE = ("best validation average precision at early stopping; the two "
            "finalists ranked by their three-seed means, which tie when they "
            "differ by no more than the pooled seed standard deviation; a tie "
            "goes to fewer parameters, then the lower candidate number")

BASE = dict(embed_dim=8, batch_size=4096, max_epochs=15,
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
    return {"experiment": EXPERIMENT, "check": "candidate_seed",
            "candidate": candidate, "world": "amlworld", "dataset": DATASET,
            "dataset_key": DATASET, "setup": "global", "head": "edge",
            "model": "graphsage_2layer_edgeattr",
            "gnn": CANDIDATES[candidate].as_dict(), "seed": seed,
            "features_version": FEATURES_VERSION, "gnn_version": GNN_VERSION,
            "device": os.environ.get("M_DEVICE", "cpu"),
            "torch_threads": THREADS}


def verdict_config() -> dict:
    return {"experiment": EXPERIMENT, "check": "verdict",
            "world": "amlworld", "dataset": DATASET, "dataset_key": DATASET,
            "setup": "global", "head": "edge",
            "model": "graphsage_2layer_edgeattr",
            "candidates": {str(k): v.as_dict() for k, v in CANDIDATES.items()},
            "seeds": SEEDS, "screen_seed": 0, "finalists": 2,
            "ranking": TIE_RULE,
            "features_version": FEATURES_VERSION, "gnn_version": GNN_VERSION,
            "device": os.environ.get("M_DEVICE", "cpu"),
            "torch_threads": THREADS}


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


def build_inputs() -> dict:
    df = frame(DATASET)
    arrays = edge_inputs(df, usd_rate_table(df))
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
    if stage == "screen":
        wanted = [(c, 0) for c in CANDIDATES]
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
        tie_rule = {"rule": TIE_RULE,
                    "seed_sd": {str(c): v for c, v in seed_sd.items()},
                    "pooled_seed_sd": pooled_sd, "difference": difference,
                    "tied": bool(tied_ap)}
        return {"dataset_key": DATASET, "screen": {str(c): v for c, v in table.items()},
                "finalists": top, "commit": {str(c): v for c, v in commit.items()},
                "winner": winner, "tie_break": tie_break, "tie_rule": tie_rule,
                "configuration": CANDIDATES[winner].as_dict(),
                "execution": {"device": os.environ.get("M_DEVICE", "cpu"),
                              "torch_threads": THREADS, "loader_workers": WORKERS}}
    return fn


def _exit_on_signal(signum, _frame):
    sys.exit(128 + signum)


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
          f" (root {experiments_root()}, threads {THREADS}, workers {WORKERS})",
          flush=True)
    inputs = build_inputs()
    for candidate, seed in todo:
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
    units = plan(stage, done)
    if not units:
        print(f"{stage}: nothing to run", flush=True)
        return 0
    lots = balanced_lots(units, shards)
    log_dir = Path(os.environ.get("M_LOG_DIR", REPO_ROOT / "logs" / "select_gnn"))
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{stage}: {len(units)} units in {len(lots)} shards -> {experiments_root()} "
          f"(device {os.environ.get('M_DEVICE', 'cpu')}, threads {THREADS}, "
          f"workers {WORKERS}); logs in {log_dir}", flush=True)
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
    if actual != EXECUTION:
        raise SystemExit(f"execution settings {actual} differ from the fixed "
                         f"{EXECUTION}; set M_DEVICE, M_TORCH_THREADS and "
                         f"M_GNN_WORKERS accordingly")


def verdict_ready(done: dict) -> None:
    for c in finalists(done):
        missing = [s for s in SEEDS if recorded(done, c, s) is None]
        if missing:
            raise SystemExit(f"commit incomplete for candidate {c}: seeds {missing}")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "status":
        return cmd_status()
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
    if cmd == "verdict":
        verdict_ready(done_index())
        metrics = run(EXPERIMENT, verdict_config(), make_verdict_fn())
        print(json.dumps({k: metrics[k] for k in ("winner", "finalists", "tie_break",
                                                  "configuration")}, indent=2))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
