#!/usr/bin/env python
"""Anchor check, GNN half: anchors on every world.

One Run per (dataset, seed), so a killed sweep loses at most one seed's training; a cheap verdict Run per
dataset then aggregates its three seed Runs into the seed-mean and the
anchor comparison. The expensive graph build happens once per dataset and is
shared across that dataset's seed Runs by closure; a dataset whose verdict
is already recorded is skipped.

The model is the two-layer GraphSAGE-style inductive GNN
(src/m/gnn.py) at one fixed configuration, global set-up only.
Payment-labelled worlds get the edge head under strictly-earlier neighbour
sampling; the account-labelled world gets the node head under a protocol
matched to the published one (centralised, 60 per cent labels, node-level,
transductive) with the residual mismatch disclosed in the Run.

Anchors are family-adjacent: GraphSAGE has no published number on AMLworld
or TransXion, so AMLworld datasets must land within 10 F1 points of the
nearest published GNN aggregate (GIN / GIN+EU / PNA), TransXion within 0.10
AP of the published Multi-GIN edge-level reading, AMLgentex within 0.10 AP
of the paper's prose figure.

Usage:  uv run python scripts/checks_anchor_gnn.py [unit ...]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("M_DEVICE", "cpu")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract import AMLGENTEX_DATASET, DATASETS, frame
from m import completed, device, run, seed_all
from m.features import VERSION, account_codes, own_fields, usd_rate_table
from m.gnn import (GNNConfig, edge_inputs, payment_graph, train_edge_scorer,
                   train_node_scorer)
from m.runlog import config_hash
from m.stats import average_precision, minority_f1
from m.worlds.amlworld import GNN_AGGREGATES

CONFIG = GNNConfig(loader_workers=int(os.environ.get("M_GNN_WORKERS", "2")))
SEEDS = [0, 1, 2]

UNITS = ["HI-Small", "LI-Small", "transxion", AMLGENTEX_DATASET,
         "HI-Medium", "LI-Medium"]

TRANSXION_ANCHOR = {"ap": 0.2878, "tolerance": 0.10,
                    "source": "Chen 2026, Table 3 (Multi-GIN, edge-level)"}
AMLGENTEX_ANCHOR = {"ap": 0.15, "tolerance": 0.10,
                    "source": "Östman 2025, prose (centralised, 60 % labels)"}

NODE_PROTOCOL = {
    "task": "node-level SAR-account detection",
    "labels": "an account is positive iff it is an endpoint of at least one "
              "real-transfer SAR row (pseudo-bank rows dropped)",
    "split": "random 60/20/20 over accounts, redrawn per seed",
    "matched": "centralised, 60 % labels, node-level, transductive",
    "residual_mismatch": "the paper's node feature pipeline and architecture "
                         "are not documented in the release; here the "
                         "GraphSAGE-style model reads structure "
                         "and transfer attributes only, with full-history "
                         "neighbourhoods (no temporal restriction), and the "
                         "60/20/20 label split approximates the paper's "
                         "60 % label availability",
}


def head_of(key: str) -> str:
    return "node" if DATASETS[key][0] == "amlgentex" else "edge"


def seed_unit_config(key: str, seed: int) -> dict:
    world, dataset = DATASETS[key]
    return {"experiment": "checks", "check": "anchor_gnn_seed", "world": world,
            "dataset": dataset, "dataset_key": key, "seed": seed,
            "setup": "global", "head": head_of(key),
            "model": "graphsage_2layer_edgeattr", "gnn": CONFIG.as_dict(),
            "features_version": VERSION,
            "device": os.environ.get("M_DEVICE", "auto")}


def verdict_config(key: str) -> dict:
    world, dataset = DATASETS[key]
    return {"experiment": "checks", "check": "anchor_gnn", "world": world,
            "dataset": dataset, "dataset_key": key, "setup": "global",
            "head": head_of(key), "model": "graphsage_2layer_edgeattr",
            "gnn": CONFIG.as_dict(), "seeds": SEEDS,
            "aggregation": "mean over anchor_gnn_seed Runs",
            "features_version": VERSION,
            "device": os.environ.get("M_DEVICE", "auto")}


def node_inputs(key: str):
    df = frame(key)
    rates = usd_rate_table(df)
    fields = own_fields(df, rates)
    payer, payee = account_codes(df)
    n_nodes = int(max(payer.max(), payee.max())) + 1
    ts = df["ts"].to_numpy(np.int64)
    y_row = df["is_laundering"].to_numpy(np.int8)

    y_node = np.zeros(n_nodes, np.int8)
    y_node[payer[y_row == 1]] = 1
    y_node[payee[y_row == 1]] = 1

    numeric = np.column_stack([
        np.log1p(np.maximum(fields["amount_usd"], 0.0)),
        fields["day_of_week"] / 6.0]).astype(np.float32)
    numeric[:, 0] = (numeric[:, 0] - numeric[:, 0].mean()) \
        / max(numeric[:, 0].std(), 1e-6)
    codes = fields["payment_format"].astype(np.int64)[:, None]
    return ts, payer, payee, numeric, codes, y_node, n_nodes


def make_edge_seed_fn(inputs, seed: int):
    ts, payer, payee, numeric, codes, y, rows, n_nodes = inputs["arrays"]

    def fn(config, run_dir) -> dict:
        key = config["dataset_key"]
        seed_all(seed)
        t0 = time.time()
        result = train_edge_scorer(
            inputs["data"], payer, payee, ts, numeric, codes, y, rows,
            CONFIG, device(),
            log=lambda m: print(f"  [{key} seed {seed}] {m}", flush=True))
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


def make_node_seed_fn(inputs, seed: int):
    y_node, n_nodes = inputs["y_node"], inputs["n_nodes"]

    def fn(config, run_dir) -> dict:
        key = config["dataset_key"]
        seed_all(seed)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n_nodes)
        n_tr, n_va = int(0.6 * n_nodes), int(0.8 * n_nodes)
        rows = {"train": np.sort(perm[:n_tr]),
                "val": np.sort(perm[n_tr:n_va]),
                "test": np.sort(perm[n_va:])}
        t0 = time.time()
        result = train_node_scorer(
            inputs["data"], inputs["numeric"], inputs["codes"], y_node, rows,
            CONFIG, device(),
            log=lambda m: print(f"  [{key} seed {seed}] {m}", flush=True))
        yte = y_node[rows["test"]]
        return {"seed": seed,
                "test_ap": average_precision(yte, result["test_probs"]),
                "val_ap": result["best_val_ap"],
                "epochs": len(result["trace"]),
                "trace": result["trace"],
                "test_positives": int(yte.sum()),
                "seconds": time.time() - t0}
    return fn


def build_inputs(key: str) -> dict:
    if head_of(key) == "edge":
        df = frame(key)
        arrays = edge_inputs(df, usd_rate_table(df))
        ts, payer, payee, numeric, codes, y, rows, n_nodes = arrays
        return {"arrays": arrays,
                "data": payment_graph(ts, payer, payee, n_nodes)}
    ts, payer, payee, numeric, codes, y_node, n_nodes = node_inputs(key)
    return {"y_node": y_node, "n_nodes": n_nodes, "numeric": numeric,
            "codes": codes,
            "data": payment_graph(ts, payer, payee, n_nodes)}


def make_verdict_fn(key: str):
    def fn(config, run_dir) -> dict:
        done = completed("checks")
        seeds_out = []
        for s in SEEDS:
            path = done[config_hash(seed_unit_config(key, s))]
            seeds_out.append(json.loads((path / "metrics.json").read_text()))
        ap_mean = float(np.mean([r["test_ap"] for r in seeds_out]))
        metrics = {"dataset_key": key, "seeds": seeds_out,
                   "test_ap_mean": ap_mean}
        if head_of(key) == "node":
            metrics |= {"protocol": NODE_PROTOCOL, "anchor": AMLGENTEX_ANCHOR,
                        "distance_from_anchor": ap_mean - AMLGENTEX_ANCHOR["ap"],
                        "pass": bool(abs(ap_mean - AMLGENTEX_ANCHOR["ap"])
                                     <= AMLGENTEX_ANCHOR["tolerance"])}
            return metrics
        f1_mean = float(np.mean([r["test_f1_x100"] for r in seeds_out]))
        metrics["test_f1_x100_mean"] = f1_mean
        metrics["test_positives"] = seeds_out[0]["test_positives"]
        if key in GNN_AGGREGATES:
            published = GNN_AGGREGATES[key]
            nearest = min(published, key=lambda m: abs(published[m] - f1_mean))
            metrics["anchor"] = {"published": published, "nearest": nearest,
                                 "tolerance": 10.0,
                                 "source": "Altman 2024, Table 2",
                                 "note": "family-adjacent: no published "
                                         "GraphSAGE number on this world"}
            metrics["distance_from_nearest"] = f1_mean - published[nearest]
            metrics["pass"] = bool(abs(f1_mean - published[nearest]) <= 10.0)
        else:
            metrics["anchor"] = TRANSXION_ANCHOR | {
                "note": "family-adjacent: no published GraphSAGE number"}
            metrics["distance_from_anchor"] = ap_mean - TRANSXION_ANCHOR["ap"]
            metrics["pass"] = bool(abs(ap_mean - TRANSXION_ANCHOR["ap"])
                                   <= TRANSXION_ANCHOR["tolerance"])
        return metrics
    return fn


def recorded_verdicts(done: dict) -> set[str]:
    keys = set()
    for path in done.values():
        config = json.loads((path / "config.json").read_text())
        if config.get("check") == "anchor_gnn":
            keys.add(config.get("dataset_key"))
    return keys


def main() -> int:
    import torch

    torch.set_num_threads(10)
    units = sys.argv[1:] or UNITS
    done = completed("checks")
    already = recorded_verdicts(done)
    for key in units:
        if key in already:
            print(f"anchor_gnn {key}: verdict already recorded", flush=True)
            continue
        missing = [s for s in SEEDS
                   if config_hash(seed_unit_config(key, s)) not in done]
        if missing:
            print(f"[{key}] building inputs for seeds {missing}", flush=True)
            inputs = build_inputs(key)
            maker = (make_node_seed_fn if head_of(key) == "node"
                     else make_edge_seed_fn)
            for s in SEEDS:
                metrics = run("checks", seed_unit_config(key, s),
                              maker(inputs, s), seed=s, done=done)
                shown = metrics.get("test_f1_x100", metrics["test_ap"])
                print(f"anchor_gnn_seed {key} seed {s}: {shown:.4f}",
                      flush=True)
            del inputs
        metrics = run("checks", verdict_config(key), make_verdict_fn(key),
                      done=done)
        shown = metrics.get("test_f1_x100_mean", metrics["test_ap_mean"])
        print(f"anchor_gnn {key}: pass={metrics['pass']} mean={shown:.4f}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
