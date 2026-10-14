#!/usr/bin/env python
"""The GNN anchor on HI-Small: the reference that the GNN selection's gate
reads (``select_gnn.py gate``).

The GNN family (src/m/gnn.py) at the configuration of the GNN selection's
first candidate, fitted on HI-Small at (network, all) under seeds 0, 1 and 2,
one Run per seed, so a stopped sweep loses at most one seed's training. A
second Run gathers the three seed Runs and their means of test average
precision and of F1 on the laundering class. The anchor trains on the CPU
with 2 loader workers and 10 torch threads.

Usage:  uv run python scripts/checks_anchor_gnn.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import frame  # noqa: E402
from m import completed, device, run, seed_all  # noqa: E402
from m.features import VERSION, usd_rate_table  # noqa: E402
from m.gnn import GNNConfig, edge_inputs, payment_graph, train_edge_scorer  # noqa: E402
from m.runlog import config_hash  # noqa: E402
from m.stats import average_precision, minority_f1  # noqa: E402

DATASET = "HI-Small"
CONFIG = GNNConfig(loader_workers=2)
TORCH_THREADS = 10
SEEDS = [0, 1, 2]


def base_config() -> dict:
    return {"experiment": "checks", "world": "amlworld", "dataset": DATASET,
            "dataset_key": DATASET, "setup": "global", "head": "edge",
            "model": "graphsage_2layer_edgeattr", "gnn": CONFIG.as_dict(),
            "features_version": VERSION, "device": str(device()),
            "torch_threads": TORCH_THREADS}


def seed_config(seed: int) -> dict:
    return base_config() | {"check": "anchor_gnn_seed", "seed": seed}


def means_config() -> dict:
    return base_config() | {"check": "anchor_gnn", "seeds": SEEDS,
                            "aggregation": "mean over the anchor_gnn_seed Runs"}


def make_seed_fn(inputs: dict, seed: int):
    ts, payer, payee, numeric, codes, y, rows, n_nodes = inputs["arrays"]

    def fn(config, run_dir) -> dict:
        seed_all(seed)
        t0 = time.time()
        result = train_edge_scorer(
            inputs["data"], payer, payee, ts, numeric, codes, y, rows,
            CONFIG, device(),
            log=lambda m: print(f"  [seed {seed}] {m}", flush=True))
        yte = y[rows["test"]]
        return {"seed": seed,
                "test_f1_x100": minority_f1(yte, result["test_probs"]) * 100,
                "test_ap": average_precision(yte, result["test_probs"]),
                "val_ap": result["best_val_ap"],
                "epochs": len(result["trace"]),
                "trace": result["trace"],
                "test_positives": int(yte.sum()),
                "seconds": time.time() - t0}
    return fn


def build_inputs() -> dict:
    df = frame(DATASET)
    arrays = edge_inputs(df, usd_rate_table(df))
    ts, payer, payee, numeric, codes, y, rows, n_nodes = arrays
    return {"arrays": arrays, "data": payment_graph(ts, payer, payee, n_nodes)}


def means(config, run_dir) -> dict:
    done = completed("checks")
    seeds = [json.loads((done[config_hash(seed_config(s))] / "metrics.json").read_text())
             for s in SEEDS]
    return {"dataset_key": DATASET, "seeds": seeds,
            "test_ap_mean": float(np.mean([r["test_ap"] for r in seeds])),
            "test_f1_x100_mean": float(np.mean([r["test_f1_x100"] for r in seeds])),
            "test_positives": seeds[0]["test_positives"]}


def main() -> int:
    if {"-h", "--help"} & set(sys.argv[1:]):
        print(__doc__)
        return 0
    import torch

    torch.set_num_threads(TORCH_THREADS)
    done = completed("checks")
    missing = [s for s in SEEDS if config_hash(seed_config(s)) not in done]
    if missing:
        print(f"building {DATASET} inputs for seeds {missing}", flush=True)
        inputs = build_inputs()
        for s in missing:
            metrics = run("checks", seed_config(s), make_seed_fn(inputs, s),
                          seed=s, done=done)
            print(f"anchor_gnn_seed {s}: F1 {metrics['test_f1_x100']:.2f} "
                  f"AP {metrics['test_ap']:.4f}", flush=True)
    metrics = run("checks", means_config(), means, done=done)
    print(f"anchor_gnn {DATASET}: F1 {metrics['test_f1_x100_mean']:.2f} "
          f"AP {metrics['test_ap_mean']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
