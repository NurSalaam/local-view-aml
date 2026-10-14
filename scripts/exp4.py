#!/usr/bin/env python
"""Experiment 4: Borrowing another bank's detector (Chapter 6). Tests
hypothesis 3.

Units, each one Run keyed by config hash (skip-if-done over the Run log):

  lender_tree:<key>:<bank>    refit one lender's eight boosted trees of
                              Experiment 1 and ship them
  gnn_identity_gate:<key>     the transfer inference path must reproduce every
                              (lender, seed) Experiment 1 solo Run's test_probs < 1e-6
  transfer_tree:<key>:<bank>  every lender's detector scores one bank's
  transfer_gnn:<key>:<bank>   validation window and test set

  uv run python scripts/exp4.py plan
  uv run python scripts/exp4.py run <unit> [...]
  uv run python scripts/exp4.py launch [--budget 14] [--machine-gb 24]
  uv run python scripts/exp4.py status

Only a fitted detector crosses between banks: a borrowing bank's features
come from its own ledger only. The lender's detector is its eight boosted trees
(tree family) or Experiment 1's three fitted GNNs plus the lender's numeric
standardisation constants (GNN family), which travel with the weights as a
deployed pipeline's scaler does.

The GNN identity gate replays each Experiment 1 solo Run's random stream — the
neighbour sampler is the only RNG consumer while a loader iterates, and its
consumption is model-independent, so draining the recorded number of epochs'
train and validation loaders positions the stream exactly where Experiment 1's test
inference ran — then loads the persisted weights and scores the lender's own
test window through the transfer path. Transfer cells seed the stream fresh
per (borrower, seed), so all candidates at one seed score identical sampled
contexts; the remaining variation from inference sampling falls within the
variation between seeds, which the interval of the weighted mean gain includes.

Execution: Experiment 1's settings (M_TORCH_THREADS 4, M_LGB_THREADS_SOLO 2,
M_DEVICE cpu).
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

import exp1  # noqa: E402  (imports torch before lightgbm)
import exp1_data as data  # noqa: E402
import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from m import completed, run, seed_all  # noqa: E402
from m.artifacts import load_array, verify, write_array, write_text  # noqa: E402
from m.features import CATEGORICAL_FIELDS  # noqa: E402
from m.features import VERSION as FEATURES_VERSION  # noqa: E402
from m.gnn import ASSEMBLY_VERSION  # noqa: E402
from m.gnn import VERSION as GNN_VERSION  # noqa: E402
from m.gnn import (PaymentGNN, edge_inputs, n_code_levels,  # noqa: E402
                   payment_graph, prefetch, shipped_constants)
from m.grid import EARLY_STOP, GRID, NUM_ROUND  # noqa: E402
from m.lgbdata import (DS_PARAMS, SEQUENCE_VERSION, construct, fit_grid,  # noqa: E402
                       predict_matrix, setting_records)
from m.runlog import REPO_ROOT, config_hash, experiments_root  # noqa: E402
from m.stats import average_precision  # noqa: E402

EXPERIMENT = "exp4"
KEYS = ["HI-Medium", "HI-Small"]
PRIMARY_KEY = "HI-Medium"
N_LENDERS = 5
SEEDS = [0, 1, 2]
EXPECTED_LENDERS = {"HI-Medium": ["020", "000", "011", "012", "01226"]}
IDENTITY_TOLERANCE = 1e-6
NUMERIC_DIM = 5
TORCH_THREADS = exp1.TORCH_THREADS
LGB_THREADS = exp1.LGB_THREADS_SOLO
CAT_COLUMNS = ("fmt", "pay_ccy", "recv_ccy")
LOG_DIR = Path(os.environ.get("M_LOG_DIR", REPO_ROOT / "logs" / "exp4"))

GNN = exp1.GNN

PROTOCOL = {
    "lenders": "the five qualifying banks that hold the most confirmed cases "
               "(training-window laundering payments); every other qualifying "
               "bank is offered the five lenders' detectors, and each lender "
               "those of the other four",
    "choice": "the lender whose detector has the highest validation average "
              "precision on the bank's validation window: the median over the "
              "eight configurations (tree family) or the mean over the three "
              "seeds (GNN family)",
    "comparison": "the comparison with the own detector: the bank borrows "
                  "the chosen detector only if its validation average "
                  "precision exceeds that of the bank's own detector from its "
                  "Experiment 1 Run; otherwise the bank keeps its own detector "
                  "and has a gain of zero",
    "pairing": {"tree": "median_over_grid", "gnn": "mean_over_seeds"},
    "movement": "only a fitted detector crosses between banks; the GNN "
                "family's detector carries the lender's numeric "
                "standardisation constants with the weights",
}


# ----------------------------------------------------------------- units --

@dataclass(frozen=True)
class Unit:
    check: str
    key: str
    bank: str = "-"

    @property
    def spec(self) -> str:
        parts = [self.check, self.key]
        if self.bank != "-":
            parts.append(self.bank)
        return ":".join(parts)

    @property
    def family(self) -> str:
        return {"lender_tree": "tree", "transfer_tree": "tree",
                "transfer_gnn": "gnn", "gnn_identity_gate": "gnn"}[self.check]


def parse_unit(spec: str) -> Unit:
    parts = spec.split(":")
    if parts[0] == "gnn_identity_gate":
        return Unit(parts[0], parts[1])
    return Unit(parts[0], parts[1], parts[2])


def lenders_for(key: str) -> list[str]:
    roster = data.read_roster(key)
    order = roster.sort_values("train_pos", ascending=False, kind="stable")
    lenders = list(order.head(N_LENDERS).index)
    if key in EXPECTED_LENDERS:
        assert lenders == EXPECTED_LENDERS[key], \
            f"{key}: lenders {lenders} differ from the expected lenders"
    return lenders


def menu_for(key: str, bank: str) -> list[str]:
    return [c for c in lenders_for(key) if c != bank]


# ------------------------------------------------- adoption from Experiment 1's log --

def exp1_run_dir(done1: dict, unit: exp1.Unit) -> Path:
    path = done1.get(config_hash(exp1.unit_config(unit)))
    if path is None:
        raise SystemExit(f"Experiment 1 Run for {unit.spec} not found; Experiment 4 adopts it by "
                         f"config identity")
    return path


def solo_dirs(done1: dict, key: str, family: str, bank: str) -> list[Path]:
    if family == "gnn":
        return [exp1_run_dir(done1, exp1.Unit("gnn_solo", key, bank, s)) for s in SEEDS]
    return [exp1_run_dir(done1, exp1.Unit(f"{family}_solo", key, bank))]


_HASHES: dict[str, str] = {}


def _hash_of(path: Path) -> str:
    """A completed Run's config hash; cached, since a Run never mutates."""
    key = str(path)
    if key not in _HASHES:
        _HASHES[key] = json.loads((path / "status.json").read_text())["config_hash"]
    return _HASHES[key]


def _metrics(path: Path) -> dict:
    return json.loads((path / "metrics.json").read_text())


def _artifact(path: Path, name: str) -> np.ndarray:
    return load_array(path, name, _metrics(path)["artifacts"][name]["sha256"])


# ------------------------------------------------------------- ledgers --

def categorical_categories(ledger) -> dict[str, list[str]]:
    return {col: [str(c) for c in ledger[col].cat.categories]
            for col in CAT_COLUMNS}


def feature_matrix(cache: exp1.LedgerCache, key: str, bank: str, family: str):
    ledger = cache.load(key, bank)
    feats = cache.features(key, bank)
    columns = exp1.COLUMNS[family]
    X = np.column_stack([feats[c] for c in columns]).astype(np.float32)
    return ledger, X


# --------------------------------------------------------------- lender --

def base_config(unit: Unit) -> dict:
    world, dataset = data.DATASETS[unit.key]
    config = {"experiment": EXPERIMENT, "check": unit.check, "world": world,
              "dataset": dataset, "dataset_key": unit.key,
              "family": unit.family, "protocol": PROTOCOL,
              "lenders": lenders_for(unit.key),
              "roster": exp1.ROSTER_RULE | {"exclude": data.excluded_banks(unit.key)},
              "features_version": FEATURES_VERSION}
    if unit.bank != "-":
        config["bank"] = unit.bank
    return config


def lender_config(unit: Unit, done1: dict) -> dict:
    solo = solo_dirs(done1, unit.key, unit.family, unit.bank)
    return base_config(unit) | {
        "role": "lender", "grid": GRID, "fixed": exp1.FIXED_EXP1,
        "num_round": NUM_ROUND, "early_stop": EARLY_STOP,
        "dataset_params": DS_PARAMS,
        "columns": "full",
        "n_feature_columns": len(exp1.COLUMNS[unit.family]),
        "table": "ledger", "sequence_version": SEQUENCE_VERSION,
        "num_threads": LGB_THREADS,
        "shipped": "the eight boosters at their selected iterations, as text",
        "solo_reference": _hash_of(solo[0])}


def lender_metrics(unit: Unit, cache: exp1.LedgerCache, done1: dict,
                   run_dir: Path) -> dict:
    t0 = time.time()
    ledger, X = feature_matrix(cache, unit.key, unit.bank, unit.family)
    columns = exp1.COLUMNS[unit.family]
    cat_idx = [columns.index(c) for c in CATEGORICAL_FIELDS]
    y = ledger["is_laundering"].to_numpy(np.int8)
    split = ledger["split"].to_numpy(np.int8)
    tr, va = split == 0, split == 1
    dtr = construct(X[tr], y[tr], cat_idx, num_threads=LGB_THREADS)
    dva = construct(X[va], y[va], cat_idx, reference=dtr, num_threads=LGB_THREADS)
    boosters = fit_grid(dtr, dva, GRID, exp1.FIXED_EXP1, NUM_ROUND, EARLY_STOP,
                        LGB_THREADS)
    settings = setting_records(boosters, GRID)

    artifacts = {}
    reload_identical = True
    for s, bst in enumerate(boosters):
        text = bst.model_to_string(num_iteration=bst.best_iteration)
        artifacts[f"booster_s{s}"] = write_text(run_dir, f"booster_s{s}", text)
        reloaded = lgb.Booster(model_str=text)
        reload_identical &= bool(np.array_equal(
            reloaded.predict(X[va]),
            bst.predict(X[va], num_iteration=bst.best_iteration)))
    assert reload_identical, "a shipped booster does not reproduce its own scores"

    solo_metrics = _metrics(solo_dirs(done1, unit.key, unit.family, unit.bank)[0])
    exp1_val = np.array([s["val_ap"] for s in solo_metrics["settings"]])
    val = np.array([s["val_ap"] for s in settings])
    comparison = {
        "val_ap_max_abs_diff": float(np.max(np.abs(val - exp1_val))),
        "best_iteration_equal": bool(all(
            a["best_iteration"] == b["best_iteration"]
            for a, b in zip(settings, solo_metrics["settings"])))}
    assert comparison["val_ap_max_abs_diff"] == 0.0 \
        and comparison["best_iteration_equal"], \
        (f"{unit.spec}: the retrained lender diverges from its Experiment 1 solo Run "
         f"({comparison}); the shipped object would not be the recorded "
         f"detector")
    return {"bank": unit.bank, "settings": settings,
            "rows": {"train": int(tr.sum()), "val": int(va.sum())},
            "positives": {"train": int(y[tr].sum()), "val": int(y[va].sum())},
            "categorical_categories": categorical_categories(ledger),
            "reload_identical": reload_identical,
            "exp1_solo_comparison": comparison,
            "artifacts": artifacts, "seconds": time.time() - t0}


# -------------------------------------------------------- identity gate --

def gnn_loader(graph, payer, payee, ts, rows_idx, sampler, shuffle: bool):
    from torch_geometric.loader import LinkNeighborLoader

    return LinkNeighborLoader(
        graph, num_neighbors=list(GNN.fanout), neighbor_sampler=sampler,
        edge_label_index=torch.from_numpy(np.stack([payer[rows_idx],
                                                    payee[rows_idx]])),
        edge_label=torch.arange(len(rows_idx)), batch_size=GNN.batch_size,
        shuffle=shuffle, num_workers=0,
        edge_label_time=torch.from_numpy(ts[rows_idx] - 1), time_attr="time")


@torch.no_grad()
def score_pass(loader, rows_t, jobs) -> None:
    """One sampling pass over the loader's label rows; every job — a
    (numeric_t, codes_t, model, out) tuple — scores the identical sampled
    batches, so candidates are compared on the same contexts."""
    for batch in prefetch(loader):
        pos = batch.edge_label.long()
        pid = batch.edge_pid.long()
        rr = rows_t[pos]
        rev = batch.edge_reverse
        for numeric_t, codes_t, model, out in jobs:
            batch.seed_numeric = torch.cat(
                [numeric_t[rr], torch.zeros(len(rr), 1)], dim=1)
            batch.seed_codes = codes_t[rr]
            batch.edge_numeric = torch.cat(
                [numeric_t[pid], rev.to(numeric_t.dtype).unsqueeze(1)], dim=1)
            batch.edge_codes = codes_t[pid]
            out[pos] = torch.sigmoid(model(batch)).float()


def load_state(path: Path) -> dict:
    record = _metrics(path)["artifacts"]["best_state"]
    return torch.load(verify(path, record), weights_only=True)


def state_n_codes(state: dict) -> list[int]:
    return [int(state[f"embeddings.{j}.weight"].shape[0])
            for j in range(len(CATEGORICAL_FIELDS))]


def gate_config(key: str, inputs: dict[str, dict[int, Path]]) -> dict:
    unit = Unit("gnn_identity_gate", key)
    return base_config(unit) | {
        "seeds": SEEDS, "gnn": GNN.as_dict(), "gnn_version": GNN_VERSION,
        "assembly_version": ASSEMBLY_VERSION, "tolerance": IDENTITY_TOLERANCE,
        "device": exp1.DEVICE_NAME, "torch_threads": TORCH_THREADS,
        "reproduction": "the recorded epochs' train and validation loader "
                        "stream is replayed (neighbour sampling is the only "
                        "RNG consumer and is model-independent), then the "
                        "persisted best_state scores the lender's own test "
                        "window through the transfer path",
        "constants": "the lender's standardisation constants are recomputed "
                     "from its ledger and shipped with the weights",
        "inputs": {c: {str(s): _hash_of(d) for s, d in seeds.items()}
                   for c, seeds in inputs.items()}}


def gate_metrics(key: str, inputs: dict[str, dict[int, Path]],
                 cache: exp1.LedgerCache, run_dir: Path) -> dict:
    t0 = time.time()
    results, lender_block = [], {}
    cats_ref = None
    for lender, seed_dirs in inputs.items():
        ledger = cache.load(key, lender)
        cats = categorical_categories(ledger)
        if cats_ref is None:
            cats_ref = cats
        assert cats == cats_ref, f"{lender}: categorical vocabularies differ"
        constants = shipped_constants(ledger, cache.rates)
        ts, payer, payee, numeric, codes, y, rows, n_nodes = edge_inputs(
            ledger, cache.rates, constants)
        graph = payment_graph(ts, payer, payee, n_nodes)
        numeric_t = torch.from_numpy(numeric)
        codes_t = torch.from_numpy(codes.astype(np.int32, copy=False))
        n_codes = n_code_levels(codes)
        test_t = torch.from_numpy(rows["test"])
        lender_block[lender] = {"constants": constants, "n_codes": n_codes,
                                "n_payments": int(len(ts))}
        for seed, exp1_dir in seed_dirs.items():
            t1 = time.time()
            m = _metrics(exp1_dir)
            ref = load_array(exp1_dir, "test_probs",
                             m["artifacts"]["test_probs"]["sha256"])
            state = load_state(exp1_dir)
            assert state_n_codes(state) == n_codes, \
                f"{lender} seed {seed}: embedding shapes differ from the ledger"
            seed_all(seed)
            from torch_geometric.sampler import NeighborSampler

            sampler = NeighborSampler(graph, num_neighbors=list(GNN.fanout),
                                      time_attr="time")
            loaders = {s: gnn_loader(graph, payer, payee, ts, rows[s], sampler,
                                     shuffle=(s == "train"))
                       for s in ("train", "val", "test")}
            model = PaymentGNN(n_codes, NUMERIC_DIM + 1, GNN)
            for _ in range(m["epochs"]):
                for _ in prefetch(loaders["train"]):
                    pass
                for _ in prefetch(loaders["val"]):
                    pass
            model.load_state_dict(state)
            model.eval()
            out = torch.empty(len(rows["test"]))
            score_pass(loaders["test"], test_t, [(numeric_t, codes_t, model, out)])
            probs = out.numpy()
            diff = float(np.max(np.abs(probs.astype(np.float64)
                                       - ref.astype(np.float64))))
            results.append({"lender": lender, "seed": seed,
                            "epochs_replayed": m["epochs"],
                            "max_abs_diff": diff,
                            "bit_identical": bool(np.array_equal(probs, ref)),
                            "seconds": time.time() - t1})
            print(f"  gate {lender} seed {seed}: max_abs_diff {diff:.2e} "
                  f"({time.time() - t1:.0f}s)", flush=True)
    return {"dataset_key": key, "results": results,
            "lenders": lender_block,
            "vocabularies_identical": True,
            "tolerance": IDENTITY_TOLERANCE,
            "pass": bool(all(r["max_abs_diff"] < IDENTITY_TOLERANCE
                             for r in results)),
            "seconds": time.time() - t0}


def gate_inputs(done1: dict, key: str) -> dict[str, dict[int, Path]]:
    return {c: {s: exp1_run_dir(done1, exp1.Unit("gnn_solo", key, c, s))
                for s in SEEDS} for c in lenders_for(key)}


def gate_run(done: dict, done1: dict, key: str) -> tuple[Path, dict] | None:
    """The gate Run for one dataset, by config identity."""
    path = done.get(config_hash(gate_config(key, gate_inputs(done1, key))))
    if path is None:
        return None
    return path, _metrics(path)


# --------------------------------------------------------- transfer, tree --

def lender_run_dir(done: dict, key: str, family: str, lender: str,
                   done1: dict) -> Path:
    unit = Unit(f"lender_{family}", key, lender)
    path = done.get(config_hash(lender_config(unit, done1)))
    if path is None:
        raise SystemExit(f"lender Run {unit.spec} not found")
    return path


def transfer_tree_config(unit: Unit, done: dict, done1: dict) -> dict:
    candidates = menu_for(unit.key, unit.bank)
    return base_config(unit) | {
        "role": "borrower", "candidates": candidates, "columns": "full",
        "inputs": {
            "lenders": {c: _hash_of(lender_run_dir(done, unit.key, unit.family,
                                                   c, done1))
                        for c in candidates},
            "solo": [_hash_of(d) for d in
                     solo_dirs(done1, unit.key, unit.family, unit.bank)]}}


class LenderModels:
    """Shipped lender boosters, loaded once per (key, family, lender) and
    verified against their recorded digests."""

    def __init__(self):
        self.cache: dict[tuple, list] = {}

    def boosters(self, done: dict, done1: dict, key: str, family: str,
                 lender: str) -> list:
        item = (key, family, lender)
        if item not in self.cache:
            path = lender_run_dir(done, key, family, lender, done1)
            m = _metrics(path)
            self.cache[item] = [
                lgb.Booster(model_str=verify(
                    path, m["artifacts"][f"booster_s{s}"]).read_text())
                for s in range(len(GRID))]
        return self.cache[item]


def transfer_tree_metrics(unit: Unit, cache: exp1.LedgerCache,
                          models: LenderModels, done: dict, done1: dict,
                          run_dir: Path) -> dict:
    t0 = time.time()
    key, bank, family = unit.key, unit.bank, unit.family
    candidates = menu_for(key, bank)
    ledger, X = feature_matrix(cache, key, bank, family)

    lender_dir = lender_run_dir(done, key, family, candidates[0], done1)
    assert categorical_categories(ledger) == \
        _metrics(lender_dir)["categorical_categories"], \
        f"{bank}: categorical vocabulary differs from the shipped model's"

    y = ledger["is_laundering"].to_numpy(np.int8)
    split = ledger["split"].to_numpy(np.int8)
    va, te = split == 1, split == 2
    yva, yte = y[va], y[te]
    Xva, Xte = X[va], X[te]

    test_payment = ledger["payment"].to_numpy(np.int64)[te]
    solo = solo_dirs(done1, key, family, bank)[0]
    assert np.array_equal(test_payment, _artifact(solo, "test_payment")), \
        f"{bank}: the transfer test set differs from Experiment 1's"

    val_scores = np.empty((len(candidates), len(GRID), int(va.sum())), np.float32)
    test_scores = np.empty((len(candidates), len(GRID), int(te.sum())), np.float32)
    per_candidate = {}
    for i, c in enumerate(candidates):
        boosters = models.boosters(done, done1, key, family, c)
        val_scores[i] = predict_matrix(boosters, Xva)
        test_scores[i] = predict_matrix(boosters, Xte)
        val_ap = ([average_precision(yva, val_scores[i, s])
                   for s in range(len(GRID))]
                  if yva.sum() else [0.0] * len(GRID))
        per_candidate[c] = {
            "val_ap": [float(v) for v in val_ap],
            "test_ap": [float(average_precision(yte, test_scores[i, s]))
                        for s in range(len(GRID))]}

    solo_metrics = _metrics(solo)
    return {"bank": bank, "candidates": candidates,
            "per_candidate": per_candidate,
            "own_val_ap": [s["val_ap"] for s in solo_metrics["settings"]],
            "rows": {"val": int(va.sum()), "test": int(te.sum())},
            "positives": {"val": int(yva.sum()), "test": int(yte.sum())},
            "artifacts": {
                "val_scores": write_array(run_dir, "val_scores", val_scores),
                "test_scores": write_array(run_dir, "test_scores", test_scores),
                "test_payment": write_array(run_dir, "test_payment",
                                            test_payment)},
            "seconds": time.time() - t0}


# ---------------------------------------------------------- transfer, GNN --

def transfer_gnn_config(unit: Unit, done: dict, done1: dict) -> dict:
    candidates = menu_for(unit.key, unit.bank)
    gate = gate_run(done, done1, unit.key)
    assert gate is not None, f"{unit.key}: no identity gate Run"
    return base_config(unit) | {
        "role": "borrower", "menu": candidates, "seeds": SEEDS,
        "availability": "a candidate is available only when its shipped "
                        "embedding tables cover every categorical code the "
                        "borrower's ledger uses; "
                        "exclusions are recorded beside the scores",
        "gnn": GNN.as_dict(), "gnn_version": GNN_VERSION,
        "assembly_version": ASSEMBLY_VERSION,
        "device": exp1.DEVICE_NAME, "torch_threads": TORCH_THREADS,
        "seeding": "seed_all(seed) before each (borrower, seed) sampling "
                   "pass, so every candidate at one seed scores identical "
                   "sampled contexts; the remaining variation from inference "
                   "sampling falls within the variation between seeds",
        "standardisation": "the borrower's tensors are standardised with "
                           "each lender's shipped constants",
        "inputs": {
            "gate": _hash_of(gate[0]),
            "lenders": {c: {str(s): _hash_of(
                exp1_run_dir(done1, exp1.Unit("gnn_solo", unit.key, c, s)))
                for s in SEEDS} for c in candidates},
            "solo": [_hash_of(d) for d in
                     solo_dirs(done1, unit.key, "gnn", unit.bank)]}}


def transfer_gnn_metrics(unit: Unit, cache: exp1.LedgerCache, done: dict,
                         done1: dict, run_dir: Path) -> dict:
    from torch_geometric.sampler import NeighborSampler

    t0 = time.time()
    key, bank = unit.key, unit.bank
    menu = menu_for(key, bank)
    gate = gate_run(done, done1, key)
    gate_codes = {c: gate[1]["lenders"][c]["n_codes"] for c in menu}

    ledger = cache.load(key, bank)
    max_codes = [int(ledger[col].cat.codes.max()) for col in CAT_COLUMNS]
    readable = lambda c: all(m < n for m, n in zip(max_codes, gate_codes[c]))  # noqa: E731
    candidates = [c for c in menu if readable(c)]
    excluded = {c: {"lender_n_codes": gate_codes[c],
                    "borrower_max_codes": max_codes}
                for c in menu if not readable(c)}
    assert candidates, f"{bank}: no candidate's shipped object can read its ledger"
    constants = {c: gate[1]["lenders"][c]["constants"] for c in candidates}

    parts = {c: edge_inputs(ledger, cache.rates, constants[c])
             for c in candidates}
    first = parts[candidates[0]]
    ts, payer, payee, _, codes, y, rows, n_nodes = first
    for c in candidates[1:]:
        assert np.array_equal(parts[c][4], codes) and \
            np.array_equal(parts[c][0], ts)
    graph = payment_graph(ts, payer, payee, n_nodes)
    codes_t = torch.from_numpy(codes.astype(np.int32, copy=False))
    numeric_t = {c: torch.from_numpy(parts[c][3]) for c in candidates}
    for c in candidates:
        assert all(int(codes[:, j].max()) < gate_codes[c][j]
                   for j in range(codes.shape[1])), \
            f"{bank}: a categorical code exceeds lender {c}'s vocabulary"

    solo = solo_dirs(done1, key, "gnn", bank)
    test_payment = ledger["payment"].to_numpy(np.int64)[rows["test"]]
    assert np.array_equal(test_payment, _artifact(solo[0], "test_payment")), \
        f"{bank}: the transfer test set differs from Experiment 1's"

    states = {c: {s: load_state(exp1_run_dir(done1, exp1.Unit("gnn_solo", key, c, s)))
                  for s in SEEDS} for c in candidates}
    n_val, n_test = len(rows["val"]), len(rows["test"])
    val_probs = np.empty((len(candidates), len(SEEDS), n_val), np.float32)
    test_probs = np.empty((len(candidates), len(SEEDS), n_test), np.float32)
    val_t = torch.from_numpy(rows["val"])
    test_t = torch.from_numpy(rows["test"])
    for j, seed in enumerate(SEEDS):
        seed_all(seed)
        sampler = NeighborSampler(graph, num_neighbors=list(GNN.fanout),
                                  time_attr="time")
        val_loader = gnn_loader(graph, payer, payee, ts, rows["val"], sampler,
                                shuffle=False)
        test_loader = gnn_loader(graph, payer, payee, ts, rows["test"], sampler,
                                 shuffle=False)
        models, outs_val, outs_test = {}, {}, {}
        for c in candidates:
            model = PaymentGNN(state_n_codes(states[c][seed]), NUMERIC_DIM + 1, GNN)
            model.load_state_dict(states[c][seed])
            model.eval()
            models[c] = model
            outs_val[c] = torch.empty(n_val)
            outs_test[c] = torch.empty(n_test)
        score_pass(val_loader, val_t,
                   [(numeric_t[c], codes_t, models[c], outs_val[c])
                    for c in candidates])
        score_pass(test_loader, test_t,
                   [(numeric_t[c], codes_t, models[c], outs_test[c])
                    for c in candidates])
        for i, c in enumerate(candidates):
            val_probs[i, j] = outs_val[c].numpy()
            test_probs[i, j] = outs_test[c].numpy()

    yva, yte = y[rows["val"]], y[rows["test"]]
    per_candidate = {}
    for i, c in enumerate(candidates):
        val_ap = ([float(average_precision(yva, val_probs[i, j]))
                   for j in range(len(SEEDS))]
                  if yva.sum() else [0.0] * len(SEEDS))
        per_candidate[c] = {
            "val_ap": val_ap,
            "test_ap": [float(average_precision(yte, test_probs[i, j]))
                        for j in range(len(SEEDS))]}
    return {"bank": bank, "menu": menu, "candidates": candidates,
            "excluded": excluded, "seeds": SEEDS,
            "per_candidate": per_candidate,
            "own_val_ap": {str(s): _metrics(d)["val_ap"]
                           for s, d in zip(SEEDS, solo)},
            "rows": {"val": n_val, "test": n_test},
            "positives": {"val": int(yva.sum()), "test": int(yte.sum())},
            "artifacts": {
                "val_probs": write_array(run_dir, "val_probs", val_probs),
                "test_probs": write_array(run_dir, "test_probs", test_probs),
                "test_payment": write_array(run_dir, "test_payment",
                                            test_payment)},
            "seconds": time.time() - t0}


# --------------------------------------------------------------- runner --

def unit_config(unit: Unit, done: dict, done1: dict) -> dict:
    if unit.check == "lender_tree":
        return lender_config(unit, done1)
    if unit.check == "gnn_identity_gate":
        return gate_config(unit.key, gate_inputs(done1, unit.key))
    if unit.check == "transfer_tree":
        return transfer_tree_config(unit, done, done1)
    if unit.check == "transfer_gnn":
        return transfer_gnn_config(unit, done, done1)
    raise SystemExit(f"unknown unit {unit.spec}")


def execute(unit: Unit, done: dict, done1: dict, cache: exp1.LedgerCache,
            models: LenderModels) -> dict:
    with exp1.RssSampler() as sampler:
        def sampled(fn):
            def wrapped(config, run_dir):
                return fn(config, run_dir) | exp1.memory_record(sampler)
            return wrapped

        if unit.check == "lender_tree":
            return run(EXPERIMENT, unit_config(unit, done, done1),
                       sampled(lambda config, run_dir:
                               lender_metrics(unit, cache, done1, run_dir)),
                       done=done)
        if unit.check == "gnn_identity_gate":
            inputs = gate_inputs(done1, unit.key)
            return run(EXPERIMENT, gate_config(unit.key, inputs),
                       sampled(lambda config, run_dir:
                               gate_metrics(unit.key, inputs, cache, run_dir)),
                       done=done)
        if unit.check == "transfer_tree":
            return run(EXPERIMENT, unit_config(unit, done, done1),
                       sampled(lambda config, run_dir:
                               transfer_tree_metrics(unit, cache, models, done,
                                                     done1, run_dir)),
                       done=done)
        if unit.check == "transfer_gnn":
            gate = gate_run(done, done1, unit.key)
            if gate is None or not gate[1]["pass"]:
                raise SystemExit(f"{unit.spec} refused: the identity gate has "
                                 f"not passed (a recorded pass False is "
                                 f"terminal until the code that keys it "
                                 f"changes)")
            return run(EXPERIMENT, unit_config(unit, done, done1),
                       sampled(lambda config, run_dir:
                               transfer_gnn_metrics(unit, cache, done, done1,
                                                    run_dir)),
                       done=done)
    raise SystemExit(f"unknown unit {unit.spec}")


def summary_line(unit: Unit, metrics: dict) -> str:
    if "pass" in metrics:
        return f"{unit.spec}: pass={metrics['pass']}"
    if unit.check.startswith("lender"):
        diff = metrics["exp1_solo_comparison"]["val_ap_max_abs_diff"]
        return (f"{unit.spec}: val_ap median "
                f"{np.median([s['val_ap'] for s in metrics['settings']]):.4f} "
                f"exp1 diff {diff:.2e} ({metrics['seconds'] / 60:.1f} min)")
    aps = {c: (np.median(v["test_ap"]) if unit.family != "gnn"
               else np.mean(v["test_ap"]))
           for c, v in metrics["per_candidate"].items()}
    best = max(aps, key=aps.get)
    return (f"{unit.spec}: best candidate {best} test_ap {aps[best]:.4f} "
            f"({metrics['seconds'] / 60:.1f} min)")


def cmd_run(specs: list[str]) -> int:
    import pyarrow as pa

    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    torch.set_num_threads(TORCH_THREADS)
    pa.set_cpu_count(2)
    done = completed(EXPERIMENT)
    done1 = completed("exp1")
    cache = exp1.LedgerCache()
    models = LenderModels()
    print(f"run {len(specs)} units (root {experiments_root()})", flush=True)
    failures = 0
    for spec in specs:
        unit = parse_unit(spec)
        t0 = time.time()
        try:
            metrics = execute(unit, done, done1, cache, models)
        except SystemExit:
            raise
        except Exception as exc:
            failures += 1
            print(f"{unit.spec}: FAILED {type(exc).__name__}: {exc} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            continue
        print(f"{summary_line(unit, metrics)} [{time.time() - t0:.0f}s]",
              flush=True)
    return 1 if failures else 0


# ---------------------------------------------------------------- jobs --

@dataclass(frozen=True)
class Job:
    name: str
    specs: tuple[str, ...]
    threads: int
    mem_gb: float
    deps: tuple[str, ...] = ()


GNN_THREADS = TORCH_THREADS + 1
MEM = {"lender": 3.0, "gate": 6.0, "transfer_tree": 2.5, "transfer_gnn": 4.5}
LOTS = {"transfer_tree": 3, "transfer_gnn": 3}


def plan_jobs(done: dict, done1: dict, exclude: set[str] = frozenset()) -> list[Job]:
    def todo(unit: Unit) -> bool:
        if unit.spec in exclude:
            return False
        try:
            return config_hash(unit_config(unit, done, done1)) not in done
        except (SystemExit, AssertionError):
            return True  # inputs not ready yet; the dep holds it back

    jobs: list[Job] = []
    for key in KEYS:
        roster = data.read_roster(key)
        lenders = lenders_for(key)
        specs = tuple(u.spec for c in lenders
                      for u in (Unit("lender_tree", key, c),) if todo(u))
        if specs:
            jobs.append(Job(f"lender_tree:{key}", specs, LGB_THREADS,
                            MEM["lender"]))
        if gate_run(done, done1, key) is None:
            jobs.append(Job(f"gnn_identity_gate:{key}",
                            (f"gnn_identity_gate:{key}",), GNN_THREADS,
                            MEM["gate"]))
        banks = list(roster.index)
        weights = [float(w) for w in roster["touching"]]
        for i, lot in enumerate(exp1._split_lots(banks, weights,
                                               LOTS["transfer_tree"])):
            specs = tuple(u.spec for b in lot
                          for u in (Unit("transfer_tree", key, b),) if todo(u))
            if specs:
                jobs.append(Job(f"transfer_tree:{key}:lot{i}", specs,
                                LGB_THREADS, MEM["transfer_tree"],
                                (f"lender_tree:{key}",)))
        for i, lot in enumerate(exp1._split_lots(banks, weights,
                                               LOTS["transfer_gnn"])):
            specs = tuple(u.spec for b in lot
                          for u in (Unit("transfer_gnn", key, b),) if todo(u))
            if specs:
                jobs.append(Job(f"transfer_gnn:{key}:lot{i}", specs,
                                GNN_THREADS, MEM["transfer_gnn"],
                                (f"gnn_identity_gate:{key}",)))
    return jobs


def dep_satisfied(name: str, done: dict, done1: dict) -> bool:
    kind, _, rest = name.partition(":")
    key = rest.split(":")[0]
    if kind == "gnn_identity_gate":
        gate = gate_run(done, done1, key)
        return gate is not None and bool(gate[1]["pass"])
    return all(config_hash(lender_config(Unit(kind, key, c), done1)) in done
               for c in lenders_for(key))


def cmd_launch(budget: int, machine_gb: float, poll: int) -> int:
    exp1.check_execution_settings()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    running: dict[str, tuple[subprocess.Popen, Job, object, Path]] = {}
    failed: set[str] = set()
    failures: dict[str, int] = {}

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
    started = time.time()
    print(f"launch: budget {budget} threads, machine {machine_gb} GB, root "
          f"{experiments_root()}; logs in {LOG_DIR}", flush=True)
    idle_polls = 0
    try:
        while True:
            for name in list(running):
                proc, job, handle, log_path = running[name]
                code = proc.poll()
                if code is None:
                    continue
                handle.close()
                del running[name]
                if code == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] done {name}", flush=True)
                else:
                    failures[name] = failures.get(name, 0) + 1
                    if failures[name] >= exp1.MAX_JOB_FAILURES or \
                            name.startswith("gnn_identity_gate"):
                        failed.add(name)
                    print(f"[{time.strftime('%H:%M:%S')}] FAILED {name} "
                          f"(exit {code}, failure {failures[name]})", flush=True)
            done = completed(EXPERIMENT)
            done1 = completed("exp1")
            for key in KEYS:
                gate = gate_run(done, done1, key)
                if gate is not None and not gate[1]["pass"]:
                    name = f"gnn_identity_gate:{key}"
                    if name not in failed:
                        failed.add(name)
                        print(f"{name}: recorded pass=False; its dependants "
                              f"are blocked until the code that keys the gate "
                              f"changes", flush=True)
            in_flight = {s for _, j, _, _ in running.values() for s in j.specs}
            exclude = in_flight | failed_units_since(started, in_flight)
            plan = plan_jobs(done, done1, exclude)
            queue = [j for j in plan if j.name not in running
                     and j.name not in failed]
            if not queue and not running:
                break
            threads_used = sum(j.threads for _, j, _, _ in running.values())
            reserved = sum(j.mem_gb for _, j, _, _ in running.values())
            launched_now = False
            for job in queue:
                if not all(dep_satisfied(d, done, done1) for d in job.deps):
                    if any(d in failed for d in job.deps):
                        failed.add(job.name)
                        print(f"blocked {job.name}: dependency failed", flush=True)
                    continue
                if threads_used + job.threads > budget or \
                        reserved + job.mem_gb > machine_gb - 3.0 or \
                        exp1.free_memory_gb() < job.mem_gb + 2.0:
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
                      f"{job.mem_gb} GB) -> {log_path.name}", flush=True)
                time.sleep(5)
            if not launched_now and not running:
                idle_polls += 1
                if idle_polls > 3:
                    print(f"stalled: {[j.name for j in queue]} cannot launch",
                          flush=True)
                    return 1
            else:
                idle_polls = 0
            time.sleep(poll)
    finally:
        stop_children()
    print(f"launch complete; failed jobs: {sorted(failed)}", flush=True)
    return 1 if failed else 0


def failed_units_since(started: float, in_flight: set[str]) -> set[str]:
    """Specs of Experiment 4 units whose Runs failed since the launcher started, or
    were left ``running`` by a child that is gone."""
    out: set[str] = set()
    runs = experiments_root() / EXPERIMENT / "runs"
    if not runs.exists():
        return out
    for status_path in runs.glob("*/status.json"):
        if status_path.stat().st_mtime < started:
            continue
        status = json.loads(status_path.read_text())
        if status["status"] == "complete":
            continue
        config = json.loads((status_path.parent / "config.json").read_text())
        unit = Unit(config["check"], config["dataset_key"],
                    config.get("bank", "-"))
        if status["status"] == "failed" or unit.spec not in in_flight:
            out.add(unit.spec)
    return out


def cmd_plan() -> int:
    done = completed(EXPERIMENT)
    done1 = completed("exp1")
    for job in plan_jobs(done, done1):
        print(f"{job.name}: {len(job.specs)} units, {job.threads} thr, "
              f"{job.mem_gb} GB, deps {list(job.deps)}")
    return 0


def cmd_status() -> int:
    runs = experiments_root() / EXPERIMENT / "runs"
    if not runs.exists():
        print("no Runs yet")
        return 0
    counts: dict[tuple, int] = {}
    bad = []
    for status_path in sorted(runs.glob("*/status.json")):
        status = json.loads(status_path.read_text())
        config = json.loads((status_path.parent / "config.json").read_text())
        key = (config.get("check"), config.get("dataset_key"), status["status"])
        counts[key] = counts.get(key, 0) + 1
        if status["status"] != "complete":
            bad.append((status_path.parent.name, config.get("check"),
                        config.get("bank"), status["status"],
                        status.get("error", "")))
    for (check, key, st), n in sorted(counts.items()):
        print(f"{check:18s} {str(key):20s} {st:9s} {n}")
    for row in bad:
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
        machine = float(args[args.index("--machine-gb") + 1]) \
            if "--machine-gb" in args else 24.0
        poll = int(args[args.index("--poll") + 1]) if "--poll" in args else 20
        return cmd_launch(budget, machine, poll)
    if cmd == "status":
        return cmd_status()
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
