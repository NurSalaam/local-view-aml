#!/usr/bin/env python
"""Experiment 2 — mechanism: the consortium cells and the examples dial.

Part one runs the two consortium cells beside Experiment 1's solo and global cells:
C-local (trained on the 107 roster banks' local copies, features from each
bank's own ledger) and C-global (the same banks' training rows from the
pooled global-sight table). Part two subsamples the laundering rows handed
to the learner at the 20 largest roster banks (5/10/30/100/all, 5 nested
draws); the "all" level adopts Experiment 1's solo Runs by config identity.

  uv run python scripts/exp2.py plan                # jobs still to run
  uv run python scripts/exp2.py run <unit> [...]    # execute units in-process
  uv run python scripts/exp2.py launch [--budget 14] [--machine-gb 24]
  uv run python scripts/exp2.py status

Unit specs: ``caches:<key>``, ``tree_clocal:<key>``, ``blind_clocal:<key>``,
``tree_cglobal:<key>``, ``blind_cglobal:<key>``, ``gnn_clocal:<key>:<seed>``,
``gnn_cglobal:<key>:<seed>``, ``gnn_cglobaldup:<key>:<seed>``,
``dial_tree:<key>:<bank>:<level>:<draw>``,
``dial_gnn:<key>:<bank>:<level>:<draw>``, ``network_ledgers:<key>``,
``tree_nown:<key>:<bank>``, ``blind_nown:<key>:<bank>``,
``gnn_nown:<key>:<bank>:<seed>``.

The own-pool network-sight cell is Experiment 1's solo Run with
the whole network in view: the bank's own rows, labels, windows and
configuration, features from the pooled global-sight table (trees) and
messages over the whole payment graph (GNN).

Execution settings are Experiment 1's, recorded in every Run config; M_SMOKE=1 pairs
with M_EXPERIMENTS_DIR pointing away from the real Run log.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("M_DEVICE", "cpu")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

sys.path.insert(0, str(Path(__file__).resolve().parent))

# torch must load before lightgbm (see scripts/exp1.py).
import torch  # noqa: E402, F401
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

import exp1  # noqa: E402
import exp1_data as data  # noqa: E402
from extract import FEATURE_COLUMNS, load_column  # noqa: E402
from m import completed, device, run, seed_all  # noqa: E402
from m.artifacts import write_array, write_json, write_state  # noqa: E402
from m.features import CATEGORICAL_FIELDS, OWN_FIELDS, extract  # noqa: E402
from m.features import VERSION as FEATURES_VERSION  # noqa: E402
from m.gnn import ASSEMBLY_VERSION  # noqa: E402
from m.gnn import VERSION as GNN_VERSION  # noqa: E402
from m.gnn import edge_inputs, payment_graph, train_edge_scorer  # noqa: E402
from m.grid import EARLY_STOP, GRID  # noqa: E402
from m.lgbdata import (DS_PARAMS, RowGroupReader, construct, fit_grid,  # noqa: E402
                       predict_matrix, setting_records)
from m.runlog import REPO_ROOT, config_hash, experiments_root  # noqa: E402
from m.stats import average_precision, minority_f1  # noqa: E402

EXPERIMENT = "exp2"
SMOKE = exp1.SMOKE
SEEDS = exp1.SEEDS
KEY = os.environ.get("M_SMOKE_KEY", "sweden_100K_easy") if SMOKE else "HI-Medium"
FIXED_EXP2 = exp1.FIXED_EXP1
ROUNDS = exp1.ROUNDS
GNN = exp1.GNN
LGB_THREADS = exp1.LGB_THREADS_GLOBAL
LGB_THREADS_DIAL = exp1.LGB_THREADS_SOLO
LGB_THREADS_SOLO = exp1.LGB_THREADS_SOLO
TORCH_THREADS = exp1.TORCH_THREADS
DEVICE_NAME = exp1.DEVICE_NAME
COLUMNS = {"tree": FEATURE_COLUMNS, "blind": OWN_FIELDS}
ROW_RULE = "one row per (payment, roster endpoint bank), sorted by (payment, bank)"

DIAL_LEVELS = [2, 5] if SMOKE else [5, 10, 30, 100]
DIAL_DRAWS = 2 if SMOKE else 5
DIAL_BANKS = 2 if SMOKE else 20
DIAL_SALT = 20260823
DIAL_GNN_SEED = 0

LOG_DIR = Path(os.environ.get("M_LOG_DIR", REPO_ROOT / "logs" / "exp2"))


def cache_dir(key: str) -> Path:
    if SMOKE:
        return Path(os.environ["M_EXPERIMENTS_DIR"]) / "interim_exp2" / key
    return REPO_ROOT / "data" / "interim" / "exp2" / key


def clocal_table_path(key: str) -> Path:
    return cache_dir(key) / f"clocal_table_v{FEATURES_VERSION}.parquet"


def clocal_gnn_path(key: str) -> Path:
    return cache_dir(key) / f"clocal_gnn_inputs_v{FEATURES_VERSION}.npz"


def roster_copies_path(key: str) -> Path:
    return cache_dir(key) / "roster_copies.npy"


def caches_ready(key: str) -> bool:
    return (clocal_table_path(key).exists() and clocal_gnn_path(key).exists()
            and roster_copies_path(key).exists())


def network_ledgers_dir(key: str) -> Path:
    return cache_dir(key) / f"network_ledgers_v{FEATURES_VERSION}"


def network_ledger_path(key: str, bank: str) -> Path:
    return network_ledgers_dir(key) / f"{bank}.parquet"


def network_ledgers_ready(key: str) -> bool:
    return (network_ledgers_dir(key) / "roster.csv").exists()


# ----------------------------------------------------------------- units --

@dataclass(frozen=True)
class Unit:
    check: str
    key: str = "-"
    bank: str = "-"
    level: int = -1
    draw: int = -1
    seed: int = -1

    @property
    def spec(self) -> str:
        parts = [self.check, self.key]
        if self.bank != "-":
            parts.append(self.bank)
        if self.level >= 0:
            parts += [str(self.level), str(self.draw)]
        if self.seed >= 0:
            parts.append(str(self.seed))
        return ":".join(parts)

    @property
    def family(self) -> str:
        return {"dial_tree": "tree", "dial_gnn": "gnn"}.get(
            self.check, self.check.split("_")[0])

    @property
    def setup(self) -> str:
        return "dial" if self.check.startswith("dial") else self.check.split("_")[1]


def parse_unit(spec: str) -> Unit:
    parts = spec.split(":")
    check = parts[0]
    if check in ("caches", "network_ledgers"):
        return Unit(check, parts[1])
    if check in ("tree_nown", "blind_nown"):
        return Unit(check, parts[1], parts[2])
    if check == "gnn_nown":
        return Unit(check, parts[1], parts[2], seed=int(parts[3]))
    if check in ("tree_clocal", "blind_clocal", "tree_cglobal", "blind_cglobal"):
        return Unit(check, parts[1])
    if check in ("gnn_clocal", "gnn_cglobal", "gnn_cglobaldup"):
        return Unit(check, parts[1], seed=int(parts[2]))
    if check in ("dial_tree", "dial_gnn"):
        return Unit(check, parts[1], parts[2], int(parts[3]), int(parts[4]))
    raise SystemExit(f"unknown unit spec {spec}")


def dial_roster(key: str) -> pd.DataFrame:
    """The dial's banks: the largest of the roster by training-window
    laundering payments (77 to 265 on HI-Medium)."""
    table = exp1.roster_for(key).sort_values("train_pos", ascending=False)
    picked = table.head(DIAL_BANKS)
    if not SMOKE and key == "HI-Medium":
        assert int(picked["train_pos"].min()) == 77 and int(picked["train_pos"].max()) == 265
    return picked


def dial_levels_for(train_pos: int) -> list[int]:
    """Numeric levels strictly below the bank's own count; a level at or
    above it is the "all" cell, which adopts Experiment 1's solo Run."""
    return [lv for lv in DIAL_LEVELS if lv < train_pos]


# ---------------------------------------------------------------- config --

def base_config(unit: Unit) -> dict:
    world, dataset = data.DATASETS[unit.key]
    config = {"experiment": EXPERIMENT, "check": unit.check, "world": world,
              "dataset": dataset, "dataset_key": unit.key,
              "family": unit.family, "setup": unit.setup,
              "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(unit.key)},
              "features_version": FEATURES_VERSION}
    if SMOKE:
        config["smoke"] = {"banks": exp1.SMOKE_BANKS, "epochs": exp1.SMOKE_EPOCHS,
                           "rounds": exp1.SMOKE_ROUNDS}
    return config


def consortium_fields(key: str) -> dict:
    roster = exp1.roster_for(key)
    return {"pool": "roster", "n_pool_banks": int(len(roster)),
            "row_rule": ROW_RULE}


def tree_cell_config(unit: Unit) -> dict:
    sight = "own_ledger" if unit.setup == "clocal" else "whole_network"
    return base_config(unit) | consortium_fields(unit.key) | {
        "sight": sight,
        "grid": GRID, "fixed": FIXED_EXP2, "num_round": ROUNDS,
        "early_stop": EARLY_STOP, "dataset_params": DS_PARAMS,
        "columns": "full" if unit.family == "tree" else "own_fields",
        "n_feature_columns": len(COLUMNS[unit.family]),
        "table": "clocal_copies" if unit.setup == "clocal" else "pooled_repeated",
        "num_threads": LGB_THREADS}


def gnn_cell_config(unit: Unit) -> dict:
    sight = "own_ledger" if unit.setup == "clocal" else "whole_network"
    graph = ("disjoint union of the roster ledger graphs" if unit.setup == "clocal"
             else "whole payment graph; loss on the roster banks' payments")
    copy_weighted = {}
    if unit.setup == "cglobaldup":
        copy_weighted = {
            "seeding": "each training and validation payment seeded once per "
                       "roster bank it touches, as C-local's copies are; test "
                       "payments scored once"}
    return base_config(unit) | consortium_fields(unit.key) | copy_weighted | {
        "sight": sight, "graph": graph,
        "head": "edge", "model": "graphsage_2layer_edgeattr",
        "gnn": GNN.as_dict(), "seed": unit.seed, "gnn_version": GNN_VERSION,
        "assembly_version": ASSEMBLY_VERSION,
        "device": DEVICE_NAME, "torch_threads": TORCH_THREADS,
        "sampling": "in-process, prefetch thread"}


def dial_fields(unit: Unit, bank_code: int) -> dict:
    return {"bank": unit.bank, "level": unit.level, "draw": unit.draw,
            "subsample": {"salt": DIAL_SALT, "bank_code": bank_code,
                          "rng": "numpy default_rng((salt, bank_code, draw)); "
                                 "permutation of the ledger's training-window "
                                 "laundering rows; the level keeps the prefix",
                          "nested": True},
            "deletion": "rows leave the fitting set only; ledger, features, "
                        "graph, validation window and test payments unchanged"}


def dial_tree_config(unit: Unit, bank_code: int) -> dict:
    return base_config(unit) | dial_fields(unit, bank_code) | {
        "grid": GRID, "fixed": FIXED_EXP2, "num_round": ROUNDS,
        "early_stop": EARLY_STOP, "dataset_params": DS_PARAMS,
        "columns": "full", "n_feature_columns": len(COLUMNS["tree"]),
        "table": "ledger", "num_threads": LGB_THREADS_DIAL}


def dial_gnn_config(unit: Unit, bank_code: int) -> dict:
    return base_config(unit) | dial_fields(unit, bank_code) | {
        "head": "edge", "model": "graphsage_2layer_edgeattr",
        "gnn": GNN.as_dict(), "seed": DIAL_GNN_SEED, "gnn_version": GNN_VERSION,
        "assembly_version": ASSEMBLY_VERSION,
        "device": DEVICE_NAME, "torch_threads": TORCH_THREADS,
        "sampling": "in-process, prefetch thread"}


def nown_config(unit: Unit) -> dict:
    """Experiment 1's solo config for the same bank (and seed), verbatim, with the
    sight widened to the whole network; ``solo_config`` pins the pairing."""
    solo = exp1.unit_config(exp1.Unit(f"{unit.family}_solo", unit.key, unit.bank,
                                  unit.seed))
    source = ({"table": "the bank's ledger rows, features from the pooled "
                        "global-sight table"}
              if unit.family in ("tree", "blind") else
              {"graph": "whole payment graph, standardised on the whole "
                        "network's training window; loss on the bank's own "
                        "training payments"})
    return solo | source | {
        "experiment": EXPERIMENT, "check": unit.check, "setup": "nown",
        "sight": "whole_network", "pool": "own",
        "solo_config": config_hash(solo)}


def unit_config(unit: Unit, roster: pd.DataFrame | None = None) -> dict:
    if unit.check in ("tree_clocal", "blind_clocal", "tree_cglobal", "blind_cglobal"):
        return tree_cell_config(unit)
    if unit.check in ("tree_nown", "blind_nown", "gnn_nown"):
        return nown_config(unit)
    if unit.check in ("gnn_clocal", "gnn_cglobal", "gnn_cglobaldup"):
        return gnn_cell_config(unit)
    if unit.check in ("dial_tree", "dial_gnn"):
        table = roster if roster is not None else exp1.roster_for(unit.key)
        code = int(table.loc[unit.bank, "code"])
        if unit.check == "dial_tree":
            return dial_tree_config(unit, code)
        return dial_gnn_config(unit, code)
    raise SystemExit(f"no config for {unit.spec}")


# ---------------------------------------------------------------- caches --

def build_caches(key: str) -> None:
    """One pass over the roster ledgers: the C-local copies table (tree
    features from each bank's own ledger, canonical (payment, bank) order),
    the union-of-ledgers GNN inputs, and the per-payment roster copy counts."""
    out = cache_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    roster = exp1.roster_for(key)
    rates = data.read_rates(key)
    n_payments = len(load_column(key, "split"))
    total = int(roster["touching"].sum())

    X = np.empty((total, len(FEATURE_COLUMNS)), np.float32)
    payment = np.empty(total, np.int64)
    bank_code = np.empty(total, np.int32)
    y = np.empty(total, np.int8)
    split = np.empty(total, np.int8)
    g_ts, g_payer, g_payee, g_numeric, g_codes = [], [], [], [], []
    node_offset = 0
    filled = 0
    for bank, row in roster.iterrows():
        ledger = data.read_ledger(key, bank)
        n = len(ledger)
        feats = extract(ledger, rates)
        for j, c in enumerate(FEATURE_COLUMNS):
            X[filled:filled + n, j] = feats[c]
        payment[filled:filled + n] = ledger["payment"].to_numpy(np.int64)
        bank_code[filled:filled + n] = int(row["code"])
        y[filled:filled + n] = ledger["is_laundering"].to_numpy(np.int8)
        split[filled:filled + n] = ledger["split"].to_numpy(np.int8)
        ts, payer, payee, numeric, codes, _, _, n_nodes = edge_inputs(ledger, rates)
        g_ts.append(ts)
        g_payer.append(payer + node_offset)
        g_payee.append(payee + node_offset)
        g_numeric.append(numeric)
        g_codes.append(codes)
        node_offset += n_nodes
        filled += n
        print(f"  cache {key} {bank}: {n} rows", flush=True)
    assert filled == total

    counts = np.bincount(payment, minlength=n_payments).astype(np.uint8)
    np.save(out / "roster_copies.npy.tmp.npy", counts)
    (out / "roster_copies.npy.tmp.npy").rename(roster_copies_path(key))

    order = np.lexsort((bank_code, payment))
    # The GNN arrays stay in bank-major ledger order, matching the
    # concatenation above; only the tree table takes the (payment, bank) sort.
    gnn = {"ts": np.concatenate(g_ts), "payer": np.concatenate(g_payer),
           "payee": np.concatenate(g_payee), "numeric": np.vstack(g_numeric),
           "codes": np.vstack(g_codes), "y": y, "split": split,
           "payment": payment, "bank_code": bank_code,
           "n_nodes": np.int64(node_offset)}
    del g_ts, g_payer, g_payee, g_numeric, g_codes
    tmp = clocal_gnn_path(key).with_suffix(f".{os.getpid()}.npz.tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, **gnn)
    tmp.rename(clocal_gnn_path(key))
    del gnn

    columns = {c: X[order, j] for j, c in enumerate(FEATURE_COLUMNS)}
    columns["is_laundering"] = y[order]
    columns["split"] = split[order]
    columns["payment"] = payment[order]
    columns["bank_code"] = bank_code[order]
    tmp = clocal_table_path(key).with_suffix(f".{os.getpid()}.parquet.tmp")
    pq.write_table(pa.table(columns), tmp, compression="zstd",
                   row_group_size=1_000_000)
    tmp.rename(clocal_table_path(key))
    roster.to_csv(out / "roster.csv")
    print(f"caches {key}: {total} copies over {len(roster)} banks, "
          f"{node_offset} union nodes", flush=True)


def read_clocal(key: str, columns: list[str]) -> dict[str, np.ndarray]:
    table = pq.read_table(clocal_table_path(key),
                          columns=columns + ["is_laundering", "split",
                                             "payment", "bank_code"])
    X = np.empty((table.num_rows, len(columns)), np.float32)
    for j, c in enumerate(columns):
        X[:, j] = table.column(c).to_numpy()
    return {"X": X,
            "y": table.column("is_laundering").to_numpy(),
            "split": table.column("split").to_numpy(),
            "payment": table.column("payment").to_numpy(),
            "bank_code": table.column("bank_code").to_numpy()}


def read_clocal_gnn(key: str) -> dict[str, np.ndarray]:
    z = np.load(clocal_gnn_path(key))
    return {k: z[k] for k in z.files}


# ------------------------------------------------------- consortium cells --

def clocal_tree_metrics(unit: Unit, run_dir: Path) -> dict:
    t0 = time.time()
    columns = COLUMNS[unit.family]
    cat_idx = [columns.index(c) for c in CATEGORICAL_FIELDS]
    cache = read_clocal(unit.key, columns)
    X, y, split = cache["X"], cache["y"], cache["split"]
    tr, va, te = split == 0, split == 1, split == 2
    dtr = construct(X[tr], y[tr], cat_idx, num_threads=LGB_THREADS)
    dva = construct(X[va], y[va], cat_idx, reference=dtr, num_threads=LGB_THREADS)
    boosters = fit_grid(dtr, dva, GRID, FIXED_EXP2, ROUNDS, EARLY_STOP, LGB_THREADS,
                        progress=lambda g: exp1._progress(g, f"{unit.spec} grid"))
    del dtr, dva
    scores = predict_matrix(boosters, X[te])
    yte = y[te]
    settings = setting_records(boosters, GRID)
    for s, rec in enumerate(settings):
        rec["test_ap_copies"] = average_precision(yte, scores[s])
        rec["test_f1_copies"] = minority_f1(yte, scores[s])
    return {
        "settings": settings,
        "rows": {"train_copies": int(tr.sum()), "val_copies": int(va.sum()),
                 "test_copies": int(te.sum())},
        "positives": {"train": int(y[tr].sum()), "val": int(y[va].sum()),
                      "test": int(yte.sum())},
        "artifacts": {
            "test_scores": write_array(run_dir, "test_scores", scores),
            "test_payment": write_array(run_dir, "test_payment", cache["payment"][te]),
            "test_bank_code": write_array(run_dir, "test_bank_code",
                                          cache["bank_code"][te])},
        "seconds": time.time() - t0}


def cglobal_tree_metrics(unit: Unit, run_dir: Path) -> dict:
    t0 = time.time()
    key = unit.key
    columns = COLUMNS[unit.family]
    cat_idx = [columns.index(c) for c in CATEGORICAL_FIELDS]
    reader = RowGroupReader(data.global_table(key), columns)
    split = load_column(key, "split")
    y = load_column(key, "is_laundering")
    counts = np.load(roster_copies_path(key))
    assert len(counts) == len(split) == len(reader)

    ref = pq.read_table(clocal_table_path(key), columns=["payment", "split",
                                                         "is_laundering"])
    ref_payment = ref.column("payment").to_numpy()
    ref_split = ref.column("split").to_numpy()
    ref_y = ref.column("is_laundering").to_numpy()

    def copies(s):
        rows = np.flatnonzero((split == s) & (counts > 0))
        c = counts[rows]
        # The row-handling check behind the blind placebo: the repeated pooled
        # rows must be, payment for payment and label for label, the C-local
        # table's rows for the same split.
        assert np.array_equal(np.repeat(rows, c), ref_payment[ref_split == s])
        assert np.array_equal(np.repeat(y[rows], c), ref_y[ref_split == s])
        return np.repeat(reader.rows(rows), c, axis=0), np.repeat(y[rows], c)

    Xtr, ytr = copies(0)
    dtr = construct(Xtr, ytr, cat_idx, num_threads=LGB_THREADS)
    n_train = len(ytr)
    del Xtr
    Xva, yva = copies(1)
    dva = construct(Xva, yva, cat_idx, reference=dtr, num_threads=LGB_THREADS)
    n_val = len(yva)
    del Xva
    boosters = fit_grid(dtr, dva, GRID, FIXED_EXP2, ROUNDS, EARLY_STOP, LGB_THREADS,
                        progress=lambda g: exp1._progress(g, f"{unit.spec} grid"))
    del dtr, dva
    te_rows = np.flatnonzero((split == 2) & (counts > 0))
    scores = predict_matrix(boosters, reader.rows(te_rows))
    reader.release()
    yte = y[te_rows]
    settings = setting_records(boosters, GRID)
    for s, rec in enumerate(settings):
        rec["test_ap_payments"] = average_precision(yte, scores[s])
        rec["test_f1_payments"] = minority_f1(yte, scores[s])
    return {
        "settings": settings, "row_alignment": True,
        "rows": {"train_copies": int(n_train), "val_copies": int(n_val),
                 "test_payments": int(len(te_rows))},
        "test_positives": int(yte.sum()),
        "artifacts": {
            "test_scores": write_array(run_dir, "test_scores", scores),
            "test_payment": write_array(run_dir, "test_payment", te_rows)},
        "seconds": time.time() - t0}


def gnn_rows_and_record(unit: Unit, arrays, graph, rows, run_dir: Path,
                        extra_artifacts: dict | None = None) -> dict:
    ts, payer, payee, numeric, codes, y = arrays
    seed_all(unit.seed)
    t0 = time.time()
    result = train_edge_scorer(
        graph, payer, payee, ts, numeric, codes, y, rows, GNN, device(),
        log=lambda m: print(f"  [{unit.spec}] {m}", flush=True),
        on_epoch=lambda trace: write_json(run_dir, "trace", trace))
    yte = y[rows["test"]]
    artifacts = {
        "val_probs": write_array(run_dir, "val_probs",
                                 result["val_probs"].astype(np.float32)),
        "test_probs": write_array(run_dir, "test_probs",
                                  result["test_probs"].astype(np.float32)),
        "best_state": write_state(run_dir, "best_state", result["best_state"]),
        "trace": write_json(run_dir, "trace", result["trace"]),
    } | (extra_artifacts or {})
    return {"seed": unit.seed, "val_ap": result["best_val_ap"],
            "test_ap": average_precision(yte, result["test_probs"]),
            "test_f1_x100": minority_f1(yte, result["test_probs"]) * 100,
            "epochs": len(result["trace"]), "n_parameters": result["n_parameters"],
            "n_payments": int(len(ts)),
            "rows": {k: int(len(v)) for k, v in rows.items()},
            "positives": {k: int(y[v].sum()) for k, v in rows.items()},
            "test_positives": int(yte.sum()), "seconds": time.time() - t0,
            "trace": result["trace"], "artifacts": artifacts}


def gnn_clocal_metrics(unit: Unit, run_dir: Path) -> dict:
    z = read_clocal_gnn(unit.key)
    rows = {name: np.flatnonzero(z["split"] == k)
            for name, k in (("train", 0), ("val", 1), ("test", 2))}
    graph = payment_graph(z["ts"], z["payer"], z["payee"], int(z["n_nodes"]))
    extra = {"test_payment": write_array(run_dir, "test_payment",
                                         z["payment"][rows["test"]]),
             "test_bank_code": write_array(run_dir, "test_bank_code",
                                           z["bank_code"][rows["test"]])}
    arrays = (z["ts"], z["payer"], z["payee"], z["numeric"], z["codes"], z["y"])
    metrics = gnn_rows_and_record(unit, arrays, graph, rows, run_dir, extra)
    return metrics | {"n_nodes": int(z["n_nodes"])}


def gnn_cglobal_metrics(unit: Unit, run_dir: Path) -> dict:
    ts, payer, payee, numeric, codes, y, rows_all, n_nodes = exp1.read_gnn_inputs(unit.key)
    counts = np.load(roster_copies_path(unit.key))
    rows = {name: r[counts[r] > 0] for name, r in rows_all.items()}
    graph = payment_graph(ts, payer, payee, n_nodes)
    extra = {"test_payment": write_array(run_dir, "test_payment", rows["test"])}
    arrays = (ts, payer, payee, numeric, codes, y)
    metrics = gnn_rows_and_record(unit, arrays, graph, rows, run_dir, extra)
    return metrics | {"n_nodes": int(n_nodes)}


def gnn_cglobaldup_metrics(unit: Unit, run_dir: Path) -> dict:
    """C-global with C-local's example weighting: a payment between two
    roster banks is fitted and validated twice, as in the union of ledger
    graphs, so that the value of sight moves sight alone."""
    ts, payer, payee, numeric, codes, y, rows_all, n_nodes = exp1.read_gnn_inputs(unit.key)
    counts = np.load(roster_copies_path(unit.key))
    rows = {}
    for name, r in rows_all.items():
        r = r[counts[r] > 0]
        rows[name] = r if name == "test" else np.repeat(r, counts[r])
    graph = payment_graph(ts, payer, payee, n_nodes)
    extra = {"test_payment": write_array(run_dir, "test_payment", rows["test"])}
    arrays = (ts, payer, payee, numeric, codes, y)
    metrics = gnn_rows_and_record(unit, arrays, graph, rows, run_dir, extra)
    return metrics | {"n_nodes": int(n_nodes)}


# ------------------------------------------------ own-pool network sight --

def build_network_ledgers(key: str) -> None:
    """Each roster bank's ledger rows with their pooled global-sight feature
    values, row-aligned to the ledger (the rows C-global reads for the
    bank's payments); one ordered pass over the pooled table."""
    out = network_ledgers_dir(key)
    out.mkdir(parents=True, exist_ok=True)
    roster = exp1.roster_for(key)
    payments = {bank: pq.read_table(data.ledger_path(key, bank), columns=["payment"])
                .column("payment").to_numpy() for bank in roster.index}
    flat = np.concatenate(list(payments.values()))
    order = np.argsort(flat, kind="stable")
    reader = RowGroupReader(data.global_table(key), FEATURE_COLUMNS)
    X = np.empty((len(flat), len(FEATURE_COLUMNS)), np.float32)
    X[order] = reader.rows(flat[order])
    reader.release()
    start = 0
    for bank, payment in payments.items():
        block = X[start:start + len(payment)]
        start += len(payment)
        columns = {c: np.ascontiguousarray(block[:, j])
                   for j, c in enumerate(FEATURE_COLUMNS)}
        columns["payment"] = payment
        target = network_ledger_path(key, bank)
        tmp = target.with_suffix(f".{os.getpid()}.parquet.tmp")
        pq.write_table(pa.table(columns), tmp, compression="zstd")
        tmp.rename(target)
    roster.to_csv(out / "roster.csv")
    print(f"network ledgers {key}: {len(flat)} rows over {len(roster)} banks",
          flush=True)


def network_features(key: str, bank: str, ledger: pd.DataFrame) -> dict:
    table = pq.read_table(network_ledger_path(key, bank))
    assert np.array_equal(table.column("payment").to_numpy(),
                          ledger["payment"].to_numpy(np.int64))
    return {c: table.column(c).to_numpy() for c in FEATURE_COLUMNS}


def nown_tree_metrics(unit: Unit, ledger: pd.DataFrame, run_dir: Path) -> dict:
    feats = network_features(unit.key, unit.bank, ledger)
    return exp1.tree_solo_metrics(unit, ledger, feats, run_dir)


def nown_gnn_metrics(unit: Unit, ledger: pd.DataFrame, run_dir: Path) -> dict:
    ts, payer, payee, numeric, codes, y, _, n_nodes = exp1.read_gnn_inputs(unit.key)
    payment = ledger["payment"].to_numpy(np.int64)
    split = ledger["split"].to_numpy(np.int8)
    assert np.array_equal(y[payment], ledger["is_laundering"].to_numpy(np.int8))
    rows = {name: payment[split == k]
            for name, k in (("train", 0), ("val", 1), ("test", 2))}
    graph = payment_graph(ts, payer, payee, n_nodes)
    arrays = (ts, payer, payee, numeric, codes, y, rows, n_nodes)
    return exp1.train_and_record(arrays, graph, unit.seed, run_dir, unit.spec,
                               rows["test"])


# ------------------------------------------------------------------ dial --

def dial_subsample(ledger: pd.DataFrame, bank_code: int, level: int,
                   draw: int) -> tuple[np.ndarray, np.ndarray]:
    """Ledger row indices of the kept and removed training-window laundering
    payments; nested across levels by the prefix construction."""
    y = ledger["is_laundering"].to_numpy(np.int8)
    split = ledger["split"].to_numpy(np.int8)
    pos = np.flatnonzero((split == 0) & (y == 1))
    rng = np.random.default_rng([DIAL_SALT, bank_code, draw])
    perm = rng.permutation(len(pos))
    return pos[perm[:level]], pos[perm[level:]]


def dial_tree_metrics(unit: Unit, ledger: pd.DataFrame, feats: dict,
                      bank_code: int, run_dir: Path) -> dict:
    t0 = time.time()
    columns = COLUMNS["tree"]
    cat_idx = [columns.index(c) for c in CATEGORICAL_FIELDS]
    X = np.column_stack([feats[c] for c in columns]).astype(np.float32)
    y = ledger["is_laundering"].to_numpy(np.int8)
    split = ledger["split"].to_numpy(np.int8)
    kept, removed = dial_subsample(ledger, bank_code, unit.level, unit.draw)
    tr = split == 0
    tr[removed] = False
    va, te = split == 1, split == 2
    dtr = construct(X[tr], y[tr], cat_idx, num_threads=LGB_THREADS_DIAL)
    dva = construct(X[va], y[va], cat_idx, reference=dtr,
                    num_threads=LGB_THREADS_DIAL)
    boosters = fit_grid(dtr, dva, GRID, FIXED_EXP2, ROUNDS, EARLY_STOP,
                        LGB_THREADS_DIAL)
    scores = predict_matrix(boosters, X[te])
    yte = y[te]
    settings = setting_records(boosters, GRID)
    for s, rec in enumerate(settings):
        rec["test_ap"] = average_precision(yte, scores[s])
        rec["test_f1"] = minority_f1(yte, scores[s])
    payment = ledger["payment"].to_numpy(np.int64)
    return {
        "bank": unit.bank, "level": unit.level, "draw": unit.draw,
        "settings": settings,
        "kept_positives": int(len(kept)), "removed_positives": int(len(removed)),
        "kept_payments": sorted(int(p) for p in payment[kept]),
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "positives": {"train": int(y[tr].sum()), "val": int(y[va].sum()),
                      "test": int(yte.sum())},
        "artifacts": {
            "test_scores": write_array(run_dir, "test_scores", scores),
            "test_payment": write_array(run_dir, "test_payment", payment[te])},
        "seconds": time.time() - t0}


def dial_gnn_metrics(unit: Unit, ledger: pd.DataFrame, rates: dict,
                     bank_code: int, run_dir: Path) -> dict:
    ts, payer, payee, numeric, codes, y, rows, n_nodes = edge_inputs(ledger, rates)
    kept, removed = dial_subsample(ledger, bank_code, unit.level, unit.draw)
    keep = np.ones(len(ts), bool)
    keep[removed] = False
    rows = dict(rows)
    rows["train"] = rows["train"][keep[rows["train"]]]
    graph = payment_graph(ts, payer, payee, n_nodes)
    payment = ledger["payment"].to_numpy(np.int64)
    unit_seeded = Unit(unit.check, unit.key, unit.bank, unit.level, unit.draw,
                       seed=DIAL_GNN_SEED)
    extra = {"test_payment": write_array(run_dir, "test_payment",
                                         payment[rows["test"]])}
    arrays = (ts, payer, payee, numeric, codes, y)
    metrics = gnn_rows_and_record(unit_seeded, arrays, graph, rows, run_dir, extra)
    return metrics | {
        "bank": unit.bank, "level": unit.level, "draw": unit.draw,
        "kept_positives": int(len(kept)), "removed_positives": int(len(removed)),
        "kept_payments": sorted(int(p) for p in payment[kept]),
        "n_nodes": int(n_nodes)}


# --------------------------------------------------------------- runner --

def execute(unit: Unit, done: dict, cache: exp1.LedgerCache,
            roster: pd.DataFrame) -> dict | None:
    with exp1.RssSampler() as sampler:
        return _execute(unit, done, cache, roster, sampler)


def _execute(unit: Unit, done: dict, cache: exp1.LedgerCache,
             roster: pd.DataFrame, sampler: exp1.RssSampler) -> dict | None:
    if unit.check == "caches":
        if not caches_ready(unit.key):
            build_caches(unit.key)
        return None
    if unit.check == "network_ledgers":
        if not network_ledgers_ready(unit.key):
            build_network_ledgers(unit.key)
        return None

    config = unit_config(unit, roster)
    recorded = done.get(config_hash(config))
    if recorded is not None:
        return json.loads((recorded / "metrics.json").read_text())

    def sampled(fn):
        def wrapped(cfg, run_dir):
            return fn(cfg, run_dir) | exp1.memory_record(sampler)
        return wrapped

    if unit.check in ("tree_clocal", "blind_clocal"):
        fn = lambda cfg, rd: clocal_tree_metrics(unit, rd)  # noqa: E731
        return run(EXPERIMENT, config, sampled(fn), done=done)
    if unit.check in ("tree_cglobal", "blind_cglobal"):
        fn = lambda cfg, rd: cglobal_tree_metrics(unit, rd)  # noqa: E731
        return run(EXPERIMENT, config, sampled(fn), done=done)
    if unit.check == "gnn_clocal":
        fn = lambda cfg, rd: gnn_clocal_metrics(unit, rd)  # noqa: E731
        return run(EXPERIMENT, config, sampled(fn), seed=unit.seed, done=done)
    if unit.check == "gnn_cglobal":
        fn = lambda cfg, rd: gnn_cglobal_metrics(unit, rd)  # noqa: E731
        return run(EXPERIMENT, config, sampled(fn), seed=unit.seed, done=done)
    if unit.check == "gnn_cglobaldup":
        fn = lambda cfg, rd: gnn_cglobaldup_metrics(unit, rd)  # noqa: E731
        return run(EXPERIMENT, config, sampled(fn), seed=unit.seed, done=done)
    if unit.check in ("tree_nown", "blind_nown"):
        def fn(cfg, rd):
            return nown_tree_metrics(unit, cache.load(unit.key, unit.bank), rd)
        return run(EXPERIMENT, config, sampled(fn), done=done)
    if unit.check == "gnn_nown":
        def fn(cfg, rd):
            return nown_gnn_metrics(unit, cache.load(unit.key, unit.bank), rd)
        return run(EXPERIMENT, config, sampled(fn), seed=unit.seed, done=done)
    if unit.check == "dial_tree":
        def fn(cfg, rd):
            ledger = cache.load(unit.key, unit.bank)
            return dial_tree_metrics(unit, ledger, cache.features(unit.key, unit.bank),
                                     int(roster.loc[unit.bank, "code"]), rd)
        return run(EXPERIMENT, config, sampled(fn), done=done)
    if unit.check == "dial_gnn":
        def fn(cfg, rd):
            ledger = cache.load(unit.key, unit.bank)
            return dial_gnn_metrics(unit, ledger, cache.rates,
                                    int(roster.loc[unit.bank, "code"]), rd)
        return run(EXPERIMENT, config, sampled(fn), seed=DIAL_GNN_SEED, done=done)
    raise SystemExit(f"unknown unit {unit.spec}")


def summary_line(unit: Unit, metrics: dict | None) -> str:
    if metrics is None:
        return f"{unit.spec}: done"
    if "test_ap" in metrics:
        return (f"{unit.spec}: val_ap {metrics['val_ap']:.4f} test_ap "
                f"{metrics['test_ap']:.4f} epochs {metrics['epochs']} "
                f"({metrics['seconds'] / 60:.1f} min)")
    aps = [s.get("test_ap", s.get("test_ap_copies", s.get("test_ap_payments")))
           for s in metrics["settings"]]
    return (f"{unit.spec}: test_ap median {np.median(aps):.4f} "
            f"({metrics['seconds'] / 60:.1f} min)")


def cmd_run(specs: list[str]) -> int:
    import pyarrow as pa_mod

    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    torch.set_num_threads(TORCH_THREADS)
    pa_mod.set_cpu_count(2)
    done = completed(EXPERIMENT)
    cache = exp1.LedgerCache()
    roster = exp1.roster_for(KEY)
    print(f"run {len(specs)} units (root {experiments_root()}, smoke {SMOKE})",
          flush=True)
    failures = 0
    for spec in specs:
        unit = parse_unit(spec)
        t0 = time.time()
        try:
            metrics = execute(unit, done, cache, roster)
        except SystemExit:
            raise
        except Exception as exc:
            failures += 1
            print(f"{unit.spec}: FAILED {type(exc).__name__}: {exc} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            continue
        print(f"{summary_line(unit, metrics)} [{time.time() - t0:.0f}s]", flush=True)
    return 1 if failures else 0


# ---------------------------------------------------------------- jobs --

@dataclass(frozen=True)
class Job:
    name: str
    specs: tuple[str, ...]
    threads: int
    mem_gb: float
    deps: tuple[str, ...] = ()


MEM = {"caches": 9.0, "tree_clocal": 8.0, "tree_cglobal": 8.0,
       "blind_clocal": 3.0, "blind_cglobal": 3.0,
       "gnn_clocal": 10.0, "gnn_cglobal": 12.0, "gnn_cglobaldup": 12.0,
       "dial_tree": 2.5, "dial_gnn": 4.0,
       "network_ledgers": 7.0, "tree_nown": 2.0, "gnn_nown": 8.0}
GNN_THREADS = TORCH_THREADS + 1
NOWN_LOTS = {"tree": 5, "gnn": 3}


def dial_units(key: str, family: str) -> list[Unit]:
    check = f"dial_{family}"
    units = []
    for bank, row in dial_roster(key).iterrows():
        for level in dial_levels_for(int(row["train_pos"])):
            for draw in range(DIAL_DRAWS):
                units.append(Unit(check, key, bank, level, draw))
    return units


def plan_jobs(done: dict, exclude: frozenset[str] = frozenset()) -> list[Job]:
    """Launch order: GNN C-global (longest), GNN C-local, the tree
    cells, the blind tree, then the examples dial; the own-pool
    network-sight cell last, its tree and blind
    units in lots of banks, its GNN units in lots of banks with the
    three seeds together."""
    key = KEY
    roster = exp1.roster_for(key)
    todo = lambda u: (u.spec not in exclude  # noqa: E731
                      and config_hash(unit_config(u, roster)) not in done)
    jobs: list[Job] = []
    if not caches_ready(key):
        jobs.append(Job(f"caches:{key}", (f"caches:{key}",), 1, MEM["caches"]))
    caches_dep = (f"caches:{key}",)
    for seed in SEEDS:
        u = Unit("gnn_cglobal", key, seed=seed)
        if todo(u):
            jobs.append(Job(u.spec, (u.spec,), GNN_THREADS, MEM["gnn_cglobal"],
                            caches_dep))
    for seed in SEEDS:
        u = Unit("gnn_clocal", key, seed=seed)
        if todo(u):
            jobs.append(Job(u.spec, (u.spec,), GNN_THREADS, MEM["gnn_clocal"],
                            caches_dep))
    for check in ("tree_clocal", "tree_cglobal", "blind_clocal", "blind_cglobal"):
        u = Unit(check, key)
        if todo(u):
            jobs.append(Job(u.spec, (u.spec,), LGB_THREADS, MEM[check], caches_dep))
    table = dial_roster(key)
    banks = list(table.index)
    weights = [float(w) for w in table["touching"]]
    for family, n_lots, threads in (("tree", 5, LGB_THREADS_DIAL),
                                    ("gnn", 3, GNN_THREADS)):
        by_bank = {b: [u for u in dial_units(key, family) if u.bank == b]
                   for b in banks}
        for i, lot in enumerate(exp1._split_lots(banks, weights, n_lots)):
            specs = tuple(u.spec for b in lot for u in by_bank[b] if todo(u))
            if specs:
                jobs.append(Job(f"dial_{family}:{key}:lot{i}", specs, threads,
                                MEM[f"dial_{family}"]))
    return jobs + nown_jobs(key, roster, todo)


def nown_jobs(key: str, roster: pd.DataFrame, todo) -> list[Job]:
    jobs: list[Job] = []
    if not network_ledgers_ready(key):
        jobs.append(Job(f"network_ledgers:{key}", (f"network_ledgers:{key}",), 1,
                        MEM["network_ledgers"]))
    banks = list(roster.index)
    weights = [float(w) for w in roster["touching"]]
    for i, lot in enumerate(exp1._split_lots(banks, weights, NOWN_LOTS["tree"])):
        specs = tuple(u.spec for b in lot
                      for u in (Unit("tree_nown", key, b), Unit("blind_nown", key, b))
                      if todo(u))
        if specs:
            jobs.append(Job(f"tree_nown:{key}:lot{i}", specs, LGB_THREADS_SOLO,
                            MEM["tree_nown"], (f"network_ledgers:{key}",)))
    for i, lot in enumerate(exp1._split_lots(banks, weights, NOWN_LOTS["gnn"])):
        specs = tuple(u.spec for b in lot for s in SEEDS
                      for u in (Unit("gnn_nown", key, b, seed=s),) if todo(u))
        if specs:
            jobs.append(Job(f"gnn_nown:{key}:lot{i}", specs, GNN_THREADS,
                            MEM["gnn_nown"]))
    return jobs


def dep_satisfied(name: str) -> bool:
    kind, _, key = name.partition(":")
    if kind == "network_ledgers":
        return network_ledgers_ready(key)
    return caches_ready(key) if kind == "caches" else False


LOCK = LOG_DIR / "launcher.pid"
WATCHDOG_SECONDS = 3 * 3600
MAX_JOB_FAILURES = 3


def cmd_launch(budget: int, machine_gb: float, poll: int) -> int:
    exp1.check_execution_settings()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        pid = int(LOCK.read_text().strip() or 0)
        try:
            os.kill(pid, 0)
            raise SystemExit(f"a launcher is already running (pid {pid}, {LOCK})")
        except ProcessLookupError:
            pass
    LOCK.write_text(f"{os.getpid()}\n")
    started = time.time()
    running: dict[str, tuple[subprocess.Popen, Job, object, Path]] = {}
    failed: set[str] = set()
    failures: dict[str, int] = {}
    terminated: dict[str, float] = {}

    def stop_children():
        for proc, _, _, _ in running.values():
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        deadline = time.time() + 60
        for proc, _, _, _ in running.values():
            while proc.poll() is None and time.time() < deadline:
                time.sleep(1)
            if proc.poll() is None:
                proc.kill()

    def on_signal(signum, _frame):
        stop_children()
        sys.exit(128 + signum)

    exp1.install_signal_handlers(on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print(f"launch: budget {budget} threads, machine {machine_gb} GB, root "
          f"{experiments_root()}, smoke {SMOKE}; logs in {LOG_DIR}", flush=True)
    idle_polls = 0
    try:
        while True:
            for name in list(running):
                proc, job, handle, log_path = running[name]
                code = proc.poll()
                if code is None:
                    if name in terminated and time.time() - terminated[name] > 60:
                        proc.kill()
                    elif name not in terminated and \
                            time.time() - log_path.stat().st_mtime > WATCHDOG_SECONDS:
                        print(f"[{time.strftime('%H:%M:%S')}] WATCHDOG {name}: "
                              f"no output; terminating", flush=True)
                        proc.send_signal(signal.SIGTERM)
                        terminated[name] = time.time()
                    continue
                handle.close()
                del running[name]
                if code == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] done {name}", flush=True)
                else:
                    failures[name] = failures.get(name, 0) + 1
                    if failures[name] >= MAX_JOB_FAILURES:
                        failed.add(name)
                    print(f"[{time.strftime('%H:%M:%S')}] FAILED {name} (exit "
                          f"{code}, failure {failures[name]})", flush=True)
            done = completed(EXPERIMENT)
            in_flight = {spec for _, job, _, _ in running.values()
                         for spec in job.specs}
            exclude = in_flight | exp1.failed_units_since(started, in_flight)
            queue = [j for j in plan_jobs(done, frozenset(exclude))
                     if j.name not in running and j.name not in failed]
            if not queue and not running:
                break
            threads_used = sum(j.threads for _, j, _, _ in running.values())
            reserved = sum(j.mem_gb for _, j, _, _ in running.values())
            held_threads = 0
            launched_now = False
            for job in queue:
                if not all(dep_satisfied(d) for d in job.deps):
                    if any(d in failed for d in job.deps):
                        failed.add(job.name)
                    continue
                fits_threads = threads_used + held_threads + job.threads <= budget
                fits_memory = (reserved + job.mem_gb <= machine_gb - 3.0
                               and exp1.free_memory_gb() >= job.mem_gb + 2.0)
                if not (fits_threads and fits_memory):
                    if not fits_threads and held_threads == 0:
                        held_threads = job.threads
                    continue
                log_path = LOG_DIR / f"{job.name.replace(':', '_')}.log"
                handle = log_path.open("a")
                args = [sys.executable, str(Path(__file__).resolve()), "run",
                        *job.specs]
                proc = subprocess.Popen(args, stdout=handle,
                                        stderr=subprocess.STDOUT,
                                        env=os.environ.copy(),
                                        start_new_session=True)
                running[job.name] = (proc, job, handle, log_path)
                threads_used += job.threads
                reserved += job.mem_gb
                launched_now = True
                print(f"[{time.strftime('%H:%M:%S')}] start {job.name} "
                      f"({len(job.specs)} units, {job.threads} thr, "
                      f"{job.mem_gb} GB; running {len(running)}) -> "
                      f"{log_path.name}", flush=True)
                time.sleep(5)
            if not launched_now and not running:
                idle_polls += 1
                if idle_polls > 3:
                    print(f"stalled: {[j.name for j in queue]}; exiting", flush=True)
                    return 1
            else:
                idle_polls = 0
            time.sleep(poll)
    finally:
        stop_children()
        if LOCK.exists() and LOCK.read_text().strip() == str(os.getpid()):
            LOCK.unlink()
    print(f"launch complete; failed jobs: {sorted(failed)}", flush=True)
    return 1 if failed else 0


def cmd_plan() -> int:
    done = completed(EXPERIMENT)
    for job in plan_jobs(done):
        print(f"{job.name}: {len(job.specs)} units, {job.threads} thr, "
              f"{job.mem_gb} GB, deps {list(job.deps)}")
    return 0


def cmd_status() -> int:
    runs = experiments_root() / EXPERIMENT / "runs"
    if not runs.exists():
        print("no Runs yet")
        return 0
    counts: dict[tuple, int] = {}
    incomplete = []
    for status_path in sorted(runs.glob("*/status.json")):
        status = json.loads(status_path.read_text())
        config = json.loads((status_path.parent / "config.json").read_text())
        key = (config.get("check"), status["status"])
        counts[key] = counts.get(key, 0) + 1
        if status["status"] != "complete":
            incomplete.append((status_path.parent.name, config.get("check"),
                               config.get("bank"), config.get("level"),
                               config.get("draw"), status["status"],
                               status.get("error", "")))
    for (check, st), n in sorted(counts.items()):
        print(f"{check:16s} {st:9s} {n}")
    for row in incomplete:
        print("  ", *row)
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        return cmd_run(args)
    if cmd == "plan":
        return cmd_plan()
    if cmd == "launch":
        budget = int(args[args.index("--budget") + 1]) if "--budget" in args else 14
        machine = float(args[args.index("--machine-gb") + 1]) if "--machine-gb" in args else 24.0
        poll = int(args[args.index("--poll") + 1]) if "--poll" in args else 30
        return cmd_launch(budget, machine, poll)
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
