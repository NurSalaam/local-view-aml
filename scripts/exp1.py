#!/usr/bin/env python
"""Experiment 1: The shortfall at each bank (Chapter 4).

Every trainable unit is one Run keyed by (family, scenario, dataset, bank,
seed); the two gates and the analysis are Runs too. Stages are pure
functions over the Run log and the derived caches, so an interrupted
launcher resumes where it stopped and a finished unit is never recomputed.

The tree and GNN families are fitted in the two scenarios of the shortfall:
``solo``, the bank's own detector at (ledger, own), and ``global``, the
detector with the global view at (network, all). Both run on HI-Medium,
HI-Small, the two controls and the eleven other AMLgentex datasets; LI-Small
and LI-Medium take the global scenario alone, for the comparison with
published scores. The blind tree runs solo at every qualifying bank of
HI-Medium and global on HI-Medium and the AMLgentex dataset.

  uv run python scripts/exp1.py plan                # jobs still to run
  uv run python scripts/exp1.py run <unit> [...]    # execute units in-process
  uv run python scripts/exp1.py launch [--budget 14] [--machine-gb 24]
  uv run python scripts/exp1.py status

Unit specs: ``ledgers:<key>``, ``gnn_inputs:<key>``, ``streamed_gate``,
``gnn_codepath_gate``, ``tree_solo:<key>:<bank>``, ``blind_solo:<key>:<bank>``,
``tree_global:<key>``, ``blind_global:<key>``, ``gnn_solo:<key>:<bank>:<seed>``,
``gnn_global:<key>:<seed>``.

Execution settings come from the environment and are recorded in every Run
config: M_TORCH_THREADS (4), M_LGB_THREADS_GLOBAL (10), M_LGB_THREADS_SOLO
(2), M_DEVICE (cpu).
"""
from __future__ import annotations

import json
import os
import resource
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("M_DEVICE", "cpu")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import platform  # noqa: E402
import subprocess as _sp  # noqa: E402

# torch must load before lightgbm: each bundles an OpenMP runtime, and torch's
# parallel reductions crash or hang when lightgbm's runtime loads first.
if "lightgbm" in sys.modules and "torch" not in sys.modules:
    raise SystemExit("lightgbm was imported before torch; import torch (or scripts/exp1.py) first")
import torch  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp1_data as data  # noqa: E402
from extract import FEATURE_COLUMNS, load_column  # noqa: E402
from m import completed, device, run, seed_all  # noqa: E402
from m.artifacts import load_array, write_array, write_json, write_state  # noqa: E402
from m.features import CATEGORICAL_FIELDS, OWN_FIELDS, extract  # noqa: E402
from m.features import VERSION as FEATURES_VERSION  # noqa: E402
from m.gnn import ASSEMBLY_VERSION  # noqa: E402
from m.gnn import VERSION as GNN_VERSION  # noqa: E402
from m.gnn import GNNConfig, edge_inputs, payment_graph, train_edge_scorer  # noqa: E402
from m.grid import EARLY_STOP, FIXED, GRID, NUM_ROUND  # noqa: E402
from m.lgbdata import SEQUENCE_VERSION  # noqa: E402
from m.lgbdata import (DS_PARAMS, RepeatedRowsSequence, RowGroupReader,  # noqa: E402
                       construct, fit_grid, predict_matrix, predict_rows,
                       setting_records)
from m.runlog import REPO_ROOT, config_hash, experiments_root, find, index, select  # noqa: E402
from m.stats import average_precision, minority_f1  # noqa: E402

EXPERIMENT = "exp1"
SEEDS = [0, 1, 2]
TORCH_THREADS = int(os.environ.get("M_TORCH_THREADS", "4"))
LGB_THREADS_GLOBAL = int(os.environ.get("M_LGB_THREADS_GLOBAL", "10"))
LGB_THREADS_SOLO = int(os.environ.get("M_LGB_THREADS_SOLO", "2"))
DEVICE_NAME = os.environ.get("M_DEVICE", "cpu")
EXECUTION = {"torch_threads": 4, "lgb_threads_global": 10,
             "lgb_threads_solo": 2, "device": "cpu"}

FIXED_EXP1 = {k: v for k, v in FIXED.items() if k != "num_threads"}
ROUNDS = NUM_ROUND
GNN = GNNConfig(hidden=64, embed_dim=8, fanout=(10, 10), batch_size=4096,
                lr=2e-3, pos_weight=6.0, max_epochs=15, patience=3, loader_workers=0)
ROSTER_RULE = {"min_test_pos": data.MIN_TEST_POS}
FAMILIES = ("tree", "blind", "gnn")
COLUMNS = {"tree": FEATURE_COLUMNS, "blind": OWN_FIELDS}
BLIND_CHECKS = {"HI-Medium": ("blind_solo", "blind_global"),
                data.AMLGENTEX_DATASET: ("blind_global",)}

GATE_KEY = "HI-Small"
CODEPATH_GATE = "gnn_codepath_gate"
STREAMED_GATE = "streamed_gate"
PRIMARY_KEY = "HI-Medium"
CODEPATH_REFERENCE = {"experiment": "select_gnn", "seed": 0}
CODEPATH_TOLERANCE = 1e-6

GNN_INPUTS = REPO_ROOT / "data" / "interim" / "exp1" / "gnn_inputs"
LOG_DIR = Path(os.environ.get("M_LOG_DIR", REPO_ROOT / "logs" / "exp1"))
MEDIUM = ("HI-Medium", "LI-Medium")


def dataset_checks(key: str) -> tuple[str, ...]:
    """The unit checks run on one dataset. The blind tree runs where the
    thesis reads it: solo and global on HI-Medium (pairing against each
    scenario's own best configuration, and Experiment 2's own-pool placebo),
    global on the AMLgentex dataset (the comparison with published scores)."""
    trained = (("tree_global", "gnn_global") if key in data.ANCHOR_ONLY else
               ("tree_solo", "tree_global", "gnn_solo", "gnn_global"))
    return trained + BLIND_CHECKS.get(key, ())


def shortfall_families(key: str) -> tuple[str, ...]:
    """The families run in both scenarios on one dataset, so measured by a
    shortfall."""
    return tuple(f for f in FAMILIES
                 if {f"{f}_solo", f"{f}_global"} <= set(dataset_checks(key)))


# ----------------------------------------------------------------- units --

@dataclass(frozen=True)
class Unit:
    check: str
    key: str = "-"
    bank: str = "-"
    seed: int = -1

    @property
    def spec(self) -> str:
        parts = [self.check]
        if self.key != "-":
            parts.append(self.key)
        if self.bank != "-":
            parts.append(self.bank)
        if self.seed >= 0:
            parts.append(str(self.seed))
        return ":".join(parts)

    @property
    def family(self) -> str:
        return self.check.split("_")[0]

    @property
    def setup(self) -> str:
        return self.check.split("_")[1]


def parse_unit(spec: str) -> Unit:
    parts = spec.split(":")
    check = parts[0]
    if check in (STREAMED_GATE, CODEPATH_GATE):
        return Unit(check)
    if check in ("ledgers", "gnn_inputs", "tree_global", "blind_global"):
        return Unit(check, parts[1])
    if check in ("tree_solo", "blind_solo"):
        return Unit(check, parts[1], parts[2])
    if check == "gnn_global":
        return Unit(check, parts[1], seed=int(parts[2]))
    if check == "gnn_solo":
        return Unit(check, parts[1], parts[2], int(parts[3]))
    raise SystemExit(f"unknown unit spec {spec}")


def base_config(unit: Unit) -> dict:
    world, dataset = data.DATASETS[unit.key]
    config = {"experiment": EXPERIMENT, "check": unit.check, "world": world,
              "dataset": dataset, "dataset_key": unit.key,
              "family": unit.family, "setup": unit.setup,
              "roster": ROSTER_RULE | {"exclude": data.excluded_banks(unit.key)},
              "features_version": FEATURES_VERSION}
    if unit.bank != "-":
        config["bank"] = unit.bank
    return config


def tree_config(unit: Unit) -> dict:
    solo = unit.setup == "solo"
    return base_config(unit) | {
        "grid": GRID, "fixed": FIXED_EXP1, "num_round": ROUNDS,
        "early_stop": EARLY_STOP, "dataset_params": DS_PARAMS,
        "columns": "full" if unit.family == "tree" else "own_fields",
        "n_feature_columns": len(COLUMNS[unit.family]),
        "table": "ledger" if solo else "copies_streamed",
        "sequence_version": SEQUENCE_VERSION,
        "num_threads": LGB_THREADS_SOLO if solo else LGB_THREADS_GLOBAL}


def gnn_config(unit: Unit) -> dict:
    return base_config(unit) | {
        "head": "edge", "model": "graphsage_2layer_edgeattr",
        "gnn": GNN.as_dict(), "seed": unit.seed, "gnn_version": GNN_VERSION,
        "assembly_version": ASSEMBLY_VERSION,
        "device": DEVICE_NAME, "torch_threads": TORCH_THREADS,
        "sampling": "in-process, prefetch thread"}


def gate_config(unit: Unit) -> dict:
    if unit.check == STREAMED_GATE:
        world, dataset = data.DATASETS[GATE_KEY]
        config = {"experiment": EXPERIMENT, "check": unit.check, "world": world,
                  "dataset": dataset, "dataset_key": GATE_KEY, "family": "tree",
                  "setup": "global", "grid": GRID, "fixed": FIXED_EXP1,
                  "num_round": ROUNDS, "early_stop": EARLY_STOP,
                  "dataset_params": DS_PARAMS, "columns": "full",
                  "sequence_version": SEQUENCE_VERSION,
                  "num_threads": LGB_THREADS_GLOBAL,
                  "criterion": "bit-identical test predictions, every setting",
                  "features_version": FEATURES_VERSION}
    else:
        config = {"experiment": EXPERIMENT, "check": unit.check,
                  "world": "amlworld", "dataset": "HI-Small",
                  "dataset_key": "HI-Small", "family": "gnn", "setup": "global",
                  "gnn": GNN.as_dict(), "seed": CODEPATH_REFERENCE["seed"],
                  "reference": CODEPATH_REFERENCE, "tolerance": CODEPATH_TOLERANCE,
                  "gnn_version": GNN_VERSION, "assembly_version": ASSEMBLY_VERSION,
                  "device": DEVICE_NAME,
                  "torch_threads": TORCH_THREADS,
                  "sampling": "in-process, prefetch thread",
                  "features_version": FEATURES_VERSION}
    return config


def unit_config(unit: Unit) -> dict:
    if unit.family in ("tree", "blind"):
        return tree_config(unit)
    if unit.check == "gnn_solo" or unit.check == "gnn_global":
        return gnn_config(unit)
    return gate_config(unit)


def roster_for(key: str) -> pd.DataFrame:
    return data.read_roster(key)


# ------------------------------------------------------------- run log --

def done_index() -> dict:
    return completed(EXPERIMENT)


def selected_candidate() -> int:
    """The GNN selection verdict's winner, whose configuration must be ``GNN``."""
    runs = index(CODEPATH_REFERENCE["experiment"])
    if not select(runs, check="verdict"):
        raise SystemExit("the GNN selection has no verdict yet; run the GNN selection "
                         "(scripts/select_gnn.py) through its verdict first")
    verdict = find(runs, check="verdict")
    metrics = json.loads((verdict / "metrics.json").read_text())
    assert metrics["configuration"] == GNN.as_dict(), \
        f"the GNN selection chose candidate {metrics['winner']}, not Experiment 1's GNN"
    return metrics["winner"]


def adopted_dir(seed: int) -> Path:
    """The GNN selection's Run of its winning candidate at one seed, matched by
    config identity."""
    return find(index(CODEPATH_REFERENCE["experiment"]), check="candidate_seed",
                candidate=selected_candidate(), dataset_key="HI-Small",
                seed=seed, gnn=GNN.as_dict(), gnn_version=GNN_VERSION,
                features_version=FEATURES_VERSION, torch_threads=TORCH_THREADS,
                device=DEVICE_NAME)


def adopted_run(unit: Unit) -> Path | None:
    """The GNN selection's winning Runs stand in for HI-Small's GNN global
    side."""
    if unit.check != "gnn_global" or unit.key != "HI-Small":
        return None
    return adopted_dir(unit.seed)


def recorded_dir(done: dict, unit: Unit) -> Path | None:
    path = done.get(config_hash(unit_config(unit)))
    if path is not None:
        return path
    return adopted_run(unit)


def recorded(done: dict, unit: Unit) -> dict | None:
    path = recorded_dir(done, unit)
    if path is None:
        return None
    return json.loads((path / "metrics.json").read_text())


def gate_state(done: dict, check: str) -> str:
    """``unrun`` (no Run), ``passed``, or ``failed`` (a complete Run with
    pass False — terminal until the code that re-keys the gate changes)."""
    metrics = recorded(done, Unit(check))
    if metrics is None:
        return "unrun"
    return "passed" if metrics.get("pass") else "failed"


def peak_rss_gb() -> float:
    """The process's lifetime peak (bytes on Darwin, KiB elsewhere)."""
    scale = 2**30 if platform.system() == "Darwin" else 2**20
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def rss_gb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / 2**30


class RssSampler:
    """High-water resident memory of this process during one unit, sampled
    on a daemon thread; ``peak_gb`` after ``stop``."""

    def __init__(self, interval: float = 0.5):
        import threading

        self.interval = interval
        self.peak = self.start = rss_gb()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while not self._stop.wait(self.interval):
            self.peak = max(self.peak, rss_gb())

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, rss_gb())


def memory_record(sampler: RssSampler | None = None) -> dict:
    out = {"process_peak_rss_gb": peak_rss_gb(), "rss_gb_at_end": rss_gb()}
    if sampler is not None:
        out |= {"unit_peak_rss_gb": sampler.peak, "rss_gb_at_start": sampler.start}
    return out


def committed_tree() -> list[str]:
    """Uncommitted paths: the code that produces a Run must be committed."""
    out = _sp.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                  capture_output=True, text=True)
    if out.returncode:
        raise SystemExit("Runs launch only from a git clone of the repository: "
                         + out.stderr.strip())
    return [line for line in out.stdout.splitlines() if line.strip()]


def check_execution_settings() -> None:
    actual = {"torch_threads": TORCH_THREADS, "lgb_threads_global": LGB_THREADS_GLOBAL,
              "lgb_threads_solo": LGB_THREADS_SOLO, "device": DEVICE_NAME}
    if actual != EXECUTION:
        raise SystemExit(f"execution settings {actual} differ from the fixed "
                         f"{EXECUTION}; set M_TORCH_THREADS, "
                         f"M_LGB_THREADS_GLOBAL, M_LGB_THREADS_SOLO and M_DEVICE "
                         f"accordingly")
    if committed_tree():
        raise SystemExit("Runs launch only from a committed tree; uncommitted: "
                         + ", ".join(committed_tree()[:12]))


# --------------------------------------------------------- tree family --

def tree_solo_metrics(unit: Unit, ledger: pd.DataFrame, feats: dict,
                      run_dir: Path) -> dict:
    t0 = time.time()
    columns = COLUMNS[unit.family]
    cat_idx = [columns.index(c) for c in CATEGORICAL_FIELDS]
    X = np.column_stack([feats[c] for c in columns]).astype(np.float32)
    y = ledger["is_laundering"].to_numpy(np.int8)
    split = ledger["split"].to_numpy(np.int8)
    tr, va, te = split == 0, split == 1, split == 2
    dtr = construct(X[tr], y[tr], cat_idx, num_threads=LGB_THREADS_SOLO)
    dva = construct(X[va], y[va], cat_idx, reference=dtr, num_threads=LGB_THREADS_SOLO)
    boosters = fit_grid(dtr, dva, GRID, FIXED_EXP1, ROUNDS, EARLY_STOP, LGB_THREADS_SOLO)
    scores = predict_matrix(boosters, X[te])
    yte = y[te]
    settings = setting_records(boosters, GRID)
    for s, rec in enumerate(settings):
        rec["test_ap"] = average_precision(yte, scores[s])
        rec["test_f1"] = minority_f1(yte, scores[s])
    return {
        "bank": unit.bank, "settings": settings,
        "rows": {"train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum())},
        "positives": {"train": int(y[tr].sum()), "val": int(y[va].sum()),
                      "test": int(yte.sum())},
        "artifacts": {
            "test_scores": write_array(run_dir, "test_scores", scores),
            "test_payment": write_array(run_dir, "test_payment",
                                        ledger["payment"].to_numpy(np.int64)[te])},
        "seconds": time.time() - t0}


def tree_global_metrics(unit: Unit, run_dir: Path) -> dict:
    t0 = time.time()
    key = unit.key
    columns = COLUMNS[unit.family]
    cat_idx = [columns.index(c) for c in CATEGORICAL_FIELDS]
    reader = RowGroupReader(data.global_table(key), columns)
    split = load_column(key, "split")
    y = load_column(key, "is_laundering")
    counts = np.load(data.ledger_dir(key) / "copies.npy")
    assert len(counts) == len(split) == len(reader)

    def sequence(s, counts):
        rows = np.flatnonzero(split == s)
        return (RepeatedRowsSequence(reader, rows, counts[rows]),
                np.repeat(y[rows], counts[rows]))

    seq_tr, ytr = sequence(0, counts)
    seq_va, yva = sequence(1, counts)
    n_train, n_val = len(ytr), len(yva)
    dtr = construct(seq_tr, ytr, cat_idx, num_threads=LGB_THREADS_GLOBAL)
    dva = construct(seq_va, yva, cat_idx, reference=dtr, num_threads=LGB_THREADS_GLOBAL)
    del seq_tr, seq_va, ytr, yva, counts
    reader.release()
    t_construct = time.time() - t0
    boosters = fit_grid(dtr, dva, GRID, FIXED_EXP1, ROUNDS, EARLY_STOP,
                        LGB_THREADS_GLOBAL,
                        progress=lambda g: _progress(g, f"{unit.spec} grid"))
    t_fit = time.time() - t0 - t_construct
    del dtr, dva
    te = split == 2
    scores = predict_rows(boosters, reader, te)
    yte = y[te]
    settings = setting_records(boosters, GRID)
    for s, rec in enumerate(settings):
        rec["test_ap"] = average_precision(yte, scores[s])
        rec["test_f1"] = minority_f1(yte, scores[s])
    selected = int(np.argmax([s["val_ap"] for s in settings]))
    return {
        "settings": settings, "selected": selected,
        "test_ap": settings[selected]["test_ap"],
        "test_f1_x100": settings[selected]["test_f1"] * 100,
        "rows": {"train_copies": int(n_train), "val_copies": int(n_val),
                 "train_payments": int((split == 0).sum()),
                 "val_payments": int((split == 1).sum()), "test": int(te.sum())},
        "test_positives": int(yte.sum()),
        "artifacts": {"test_scores": write_array(run_dir, "test_scores", scores)},
        "seconds": {"construct": t_construct, "fit": t_fit,
                    "total": time.time() - t0}}


def _progress(items, desc):
    from m import track

    return track(items, desc=desc)


def streamed_gate_metrics(config: dict, run_dir: Path) -> dict:
    """In-memory against streamed construction of the copies table: every
    setting's test predictions must be bit-identical."""
    t0 = time.time()
    key = config["dataset_key"]
    columns = FEATURE_COLUMNS
    cat_idx = [columns.index(c) for c in CATEGORICAL_FIELDS]
    reader = RowGroupReader(data.global_table(key), columns)
    split = load_column(key, "split")
    y = load_column(key, "is_laundering")
    counts = np.load(data.ledger_dir(key) / "copies.npy")
    rows = {s: np.flatnonzero(split == s) for s in (0, 1, 2)}
    X = np.empty((len(reader), len(columns)), np.float32)
    for g in range(len(reader.bounds) - 1):
        X[reader.bounds[g]:reader.bounds[g + 1]] = reader.group(g)
    reader.release()
    Xte = X[rows[2]]

    def in_memory(X):
        Xtr = np.repeat(X[rows[0]], counts[rows[0]], axis=0)
        dtr = construct(Xtr, np.repeat(y[rows[0]], counts[rows[0]]), cat_idx,
                        num_threads=LGB_THREADS_GLOBAL)
        del Xtr
        Xva = np.repeat(X[rows[1]], counts[rows[1]], axis=0)
        dva = construct(Xva, np.repeat(y[rows[1]], counts[rows[1]]), cat_idx,
                        reference=dtr, num_threads=LGB_THREADS_GLOBAL)
        del Xva
        return fit_grid(dtr, dva, GRID, FIXED_EXP1, ROUNDS, EARLY_STOP,
                        LGB_THREADS_GLOBAL,
                        progress=lambda g: _progress(g, "gate in-memory"))

    def streamed():
        seq_tr = RepeatedRowsSequence(reader, rows[0], counts[rows[0]])
        seq_va = RepeatedRowsSequence(reader, rows[1], counts[rows[1]])
        dtr = construct(seq_tr, np.repeat(y[rows[0]], counts[rows[0]]), cat_idx,
                        num_threads=LGB_THREADS_GLOBAL)
        dva = construct(seq_va, np.repeat(y[rows[1]], counts[rows[1]]), cat_idx,
                        reference=dtr, num_threads=LGB_THREADS_GLOBAL)
        reader.release()
        return fit_grid(dtr, dva, GRID, FIXED_EXP1, ROUNDS, EARLY_STOP,
                        LGB_THREADS_GLOBAL,
                        progress=lambda g: _progress(g, "gate streamed"))

    mem = in_memory(X)
    pred_mem = [b.predict(Xte, num_iteration=b.best_iteration) for b in mem]
    rec_mem = setting_records(mem, GRID)
    del mem, X
    stream = streamed()
    pred_stream = [b.predict(Xte, num_iteration=b.best_iteration) for b in stream]
    rec_stream = setting_records(stream, GRID)
    settings = []
    for s, cfg in enumerate(GRID):
        identical = bool(np.array_equal(pred_mem[s], pred_stream[s]))
        settings.append({"params": cfg, "predictions_identical": identical,
                         "max_abs_diff": float(np.max(np.abs(pred_mem[s] - pred_stream[s]))),
                         "best_iteration": [rec_mem[s]["best_iteration"],
                                            rec_stream[s]["best_iteration"]],
                         "val_ap": [rec_mem[s]["val_ap"], rec_stream[s]["val_ap"]]})
        print(f"  gate {cfg}: identical={identical}", flush=True)
    return {"dataset_key": key,
            "n_copies": {"train": int(counts[rows[0]].sum()),
                         "val": int(counts[rows[1]].sum())},
            "n_test": int(len(rows[2])), "settings": settings,
            "pass": all(s["predictions_identical"] for s in settings),
            "seconds": time.time() - t0}


# ---------------------------------------------------------- GNN family --

def gnn_inputs_path(key: str) -> Path:
    return GNN_INPUTS / f"{key}_v{FEATURES_VERSION}.npz"


def write_gnn_inputs(key: str) -> Path:
    df = data.frame(key)
    ts, payer, payee, numeric, codes, y, rows, n_nodes = edge_inputs(
        df, data.read_rates(key))
    path = gnn_inputs_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.npz.tmp")
    with tmp.open("wb") as handle:
        np.savez(handle, ts=ts, payer=payer, payee=payee, numeric=numeric,
                 codes=codes, y=y, split=df["split"].to_numpy(np.int8),
                 n_nodes=np.int64(n_nodes))
    tmp.rename(path)
    return path


def read_gnn_inputs(key: str):
    z = np.load(gnn_inputs_path(key))
    split = z["split"]
    rows = {name: np.flatnonzero(split == k)
            for name, k in (("train", 0), ("val", 1), ("test", 2))}
    return (z["ts"], z["payer"], z["payee"], z["numeric"], z["codes"], z["y"],
            rows, int(z["n_nodes"]))


def train_and_record(arrays, graph, seed: int, run_dir: Path, tag: str,
                     test_payment: np.ndarray | None = None) -> dict:
    ts, payer, payee, numeric, codes, y, rows, n_nodes = arrays
    seed_all(seed)
    t0 = time.time()
    result = train_edge_scorer(
        graph, payer, payee, ts, numeric, codes, y, rows, GNN, device(),
        log=lambda m: print(f"  [{tag}] {m}", flush=True),
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
    if test_payment is not None:
        artifacts["test_payment"] = write_array(run_dir, "test_payment", test_payment)
    return {"seed": seed, "val_ap": result["best_val_ap"],
            "test_ap": average_precision(yte, result["test_probs"]),
            "test_f1_x100": minority_f1(yte, result["test_probs"]) * 100,
            "epochs": len(result["trace"]), "n_parameters": result["n_parameters"],
            "n_nodes": int(n_nodes), "n_payments": int(len(ts)),
            "rows": {k: int(len(v)) for k, v in rows.items()},
            "positives": {k: int(y[v].sum()) for k, v in rows.items()},
            "test_positives": int(yte.sum()), "seconds": time.time() - t0,
            "trace": result["trace"], "artifacts": artifacts}


def codepath_gate_metrics(config: dict, run_dir: Path) -> dict:
    """The GNN selection's winner at seed 0 on HI-Small under Experiment 1's tensor assembly
    must reproduce the adopted Run's per-payment test scores."""
    ref_dir = adopted_dir(CODEPATH_REFERENCE["seed"])
    ref_metrics = json.loads((ref_dir / "metrics.json").read_text())
    ref_probs = load_array(ref_dir, "test_probs",
                           ref_metrics["artifacts"]["test_probs"]["sha256"])
    arrays = read_gnn_inputs("HI-Small")
    graph = payment_graph(arrays[0], arrays[1], arrays[2], arrays[7])
    metrics = train_and_record(arrays, graph, CODEPATH_REFERENCE["seed"], run_dir,
                               "codepath gate")
    probs = load_array(run_dir, "test_probs")
    diff = float(np.max(np.abs(probs.astype(np.float64) - ref_probs.astype(np.float64))))
    metrics |= {"reference": CODEPATH_REFERENCE | {
                    "candidate": selected_candidate(), "run": ref_dir.name, "test_ap": ref_metrics["test_ap"], "val_ap": ref_metrics["val_ap"],
                    "epochs": ref_metrics["epochs"]},
                "max_abs_diff": diff,
                "bit_identical": bool(np.array_equal(probs, ref_probs)),
                "tolerance": CODEPATH_TOLERANCE,
                "pass": bool(diff < CODEPATH_TOLERANCE)}
    return metrics


# --------------------------------------------------------------- runner --

def _exit_on_signal(signum, _frame):
    sys.exit(128 + signum)


def install_signal_handlers(handler=_exit_on_signal) -> None:
    """SIGTERM always; SIGHUP only when nobody (nohup) asked to ignore it."""
    signal.signal(signal.SIGTERM, handler)
    if signal.getsignal(signal.SIGHUP) is not signal.SIG_IGN:
        signal.signal(signal.SIGHUP, handler)


class LedgerCache:
    """One dataset's ledgers, USD table, and per-bank feature columns,
    loaded on demand and kept for the bank being processed."""

    def __init__(self):
        self.current = None
        self.ledger = self.feats = self.rates = None

    def load(self, key: str, bank: str):
        if (key, bank) != self.current:
            self.rates = data.read_rates(key)
            self.ledger = data.read_ledger(key, bank)
            self.feats = None
            self.current = (key, bank)
        return self.ledger

    def features(self, key: str, bank: str) -> dict:
        self.load(key, bank)
        if self.feats is None:
            self.feats = extract(self.ledger, self.rates)
        return self.feats


def execute(unit: Unit, done: dict, cache: LedgerCache) -> dict | None:
    """Run one unit; preprocessing happens inside the Run so a failure is
    recorded against the unit's config hash, and the unit's own peak resident
    memory is sampled while it runs."""
    with RssSampler() as sampler:
        return _execute(unit, done, cache, sampler)


def _execute(unit: Unit, done: dict, cache: LedgerCache, sampler: RssSampler) -> dict | None:
    if unit.check == "ledgers":
        if not data.ledgers_ready(unit.key):
            table = data.write_ledgers(unit.key)
            print(f"ledgers {unit.key}: {len(table)} banks", flush=True)
        return None
    if unit.check == "gnn_inputs":
        if not gnn_inputs_path(unit.key).exists():
            write_gnn_inputs(unit.key)
        return None
    if recorded(done, unit) is not None:
        return recorded(done, unit)
    def sampled(fn):
        def wrapped(config, run_dir):
            return fn(config, run_dir) | memory_record(sampler)
        return wrapped

    if unit.check == STREAMED_GATE:
        return run(EXPERIMENT, unit_config(unit), sampled(streamed_gate_metrics), done=done)
    if unit.check == CODEPATH_GATE:
        return run(EXPERIMENT, unit_config(unit), sampled(codepath_gate_metrics),
                   seed=CODEPATH_REFERENCE["seed"], done=done)
    if unit.check in ("tree_solo", "blind_solo"):
        def fn(config, run_dir):
            ledger = cache.load(unit.key, unit.bank)
            return tree_solo_metrics(unit, ledger, cache.features(unit.key, unit.bank),
                                     run_dir)
        return run(EXPERIMENT, unit_config(unit), sampled(fn), done=done)
    if unit.check in ("tree_global", "blind_global"):
        if unit.key in MEDIUM and gate_state(done, STREAMED_GATE) != "passed":
            raise SystemExit(f"{unit.spec} refused: the streamed gate has not passed")
        return run(EXPERIMENT, unit_config(unit),
                   sampled(lambda config, run_dir: tree_global_metrics(unit, run_dir)),
                   done=done)
    if gate_state(done, CODEPATH_GATE) != "passed":
        raise SystemExit(f"{unit.spec} refused: the GNN code-path gate has not passed")
    if unit.check == "gnn_solo":
        def fn(config, run_dir):
            ledger = cache.load(unit.key, unit.bank)
            arrays = edge_inputs(ledger, cache.rates)
            graph = payment_graph(arrays[0], arrays[1], arrays[2], arrays[7])
            test_payment = ledger["payment"].to_numpy(np.int64)[arrays[6]["test"]]
            return train_and_record(arrays, graph, unit.seed, run_dir, unit.spec,
                                    test_payment)
        return run(EXPERIMENT, unit_config(unit), sampled(fn), seed=unit.seed, done=done)
    if unit.check == "gnn_global":
        def fn(config, run_dir):
            arrays = read_gnn_inputs(unit.key)
            graph = payment_graph(arrays[0], arrays[1], arrays[2], arrays[7])
            return train_and_record(arrays, graph, unit.seed, run_dir, unit.spec)
        return run(EXPERIMENT, unit_config(unit), sampled(fn), seed=unit.seed, done=done)
    raise SystemExit(f"unknown unit {unit.spec}")


def summary_line(unit: Unit, metrics: dict | None) -> str:
    if metrics is None:
        return f"{unit.spec}: done"
    if "pass" in metrics:
        return f"{unit.spec}: pass={metrics['pass']}"
    if unit.family == "gnn":
        return (f"{unit.spec}: val_ap {metrics['val_ap']:.4f} test_ap "
                f"{metrics['test_ap']:.4f} epochs {metrics['epochs']} "
                f"({metrics['seconds'] / 60:.1f} min)")
    aps = [s["test_ap"] for s in metrics["settings"]]
    secs = metrics["seconds"]
    secs = secs["total"] if isinstance(secs, dict) else secs
    return f"{unit.spec}: test_ap median {np.median(aps):.4f} ({secs / 60:.1f} min)"


def cmd_run(specs: list[str]) -> int:
    import pyarrow as pa

    check_execution_settings()
    install_signal_handlers()
    torch.set_num_threads(TORCH_THREADS)
    pa.set_cpu_count(2)
    done = done_index()
    cache = LedgerCache()
    print(f"run {len(specs)} units (root {experiments_root()}, torch threads "
          f"{TORCH_THREADS}, lgb threads {LGB_THREADS_GLOBAL}/{LGB_THREADS_SOLO})",
          flush=True)
    failures = 0
    for spec in specs:
        unit = parse_unit(spec)
        t0 = time.time()
        try:
            metrics = execute(unit, done, cache)
        except SystemExit:
            raise
        except Exception as exc:  # one unit's failure is recorded; the lot goes on
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
    cost: float = 1.0
    trailing: bool = False


def _split_lots(items: list, weights: list[float], n: int) -> list[list]:
    lots: list[list] = [[] for _ in range(n)]
    loads = [0.0] * n
    for item, w in sorted(zip(items, weights), key=lambda t: (-t[1], str(t[0]))):
        i = loads.index(min(loads))
        lots[i].append(item)
        loads[i] += w
    return lots


def size_class(key: str) -> str:
    if key in MEDIUM:
        return "medium"
    if key in ("HI-Small", "LI-Small"):
        return "small"
    return "tiny"


GNN_THREADS = TORCH_THREADS + 1  # intra-op threads plus the sampling thread
MEM = {  # declared peak resident GB per job kind and size class
    "ledgers": {"medium": 6.0, "small": 2.5, "tiny": 1.5},
    "gnn_inputs": {"medium": 12.0, "small": 5.0, "tiny": 2.0},
    "tree_global": {"medium": 15.0, "small": 6.5, "tiny": 2.5},
    "gnn_global": {"medium": 12.0, "small": 7.0, "tiny": 9.0},
    "tree_solo": {"medium": 1.5, "small": 1.5, "tiny": 1.5},
    "gnn_solo": {"medium": 3.5, "small": 3.5, "tiny": 4.0},
    STREAMED_GATE: 9.0, CODEPATH_GATE: 7.0,
}
LOTS = {"tree_solo": {"medium": 5, "small": 1, "tiny": 2},
        "gnn_solo": {"medium": 3, "small": 1, "tiny": 1}}


def dataset_jobs(key: str, done: dict, exclude: set[str]) -> dict[str, list[Job]]:
    """Jobs for one dataset by kind; empty lists where nothing remains. Lots
    partition the whole roster deterministically, so a lot's membership and
    name are stable across re-plans; each lot carries only its units still
    to run and not in ``exclude`` (units in flight or failed this session)."""
    sc = size_class(key)
    cells = dataset_checks(key)
    out: dict[str, list[Job]] = {k: [] for k in
                                 ("ledgers", "tree_solo", "tree_global", "gnn_inputs",
                                  "gnn_global", "gnn_solo")}
    if not data.ledgers_ready(key):
        out["ledgers"].append(Job(f"ledgers:{key}", (f"ledgers:{key}",), 1,
                                  MEM["ledgers"][sc]))
    todo = lambda u: u.spec not in exclude and recorded_dir(done, u) is None  # noqa: E731
    tree_solo = [c for c in ("tree_solo", "blind_solo") if c in cells]

    if not data.ledgers_ready(key):
        if tree_solo:
            out["tree_solo"].append(Job(f"tree_solo:{key}:pending", (), LGB_THREADS_SOLO,
                                        MEM["tree_solo"][sc], (f"ledgers:{key}",)))
        if "gnn_solo" in cells:
            out["gnn_solo"].append(Job(f"gnn_solo:{key}:pending", (), GNN_THREADS,
                                       MEM["gnn_solo"][sc],
                                       (f"ledgers:{key}", CODEPATH_GATE)))
    else:
        table = roster_for(key)
        banks = list(table.index)
        weights = [float(w) for w in table["touching"]]
        if tree_solo:
            for i, lot in enumerate(_split_lots(banks, weights, LOTS["tree_solo"][sc])):
                specs = tuple(u.spec for b in lot
                              for u in (Unit(c, key, b) for c in tree_solo) if todo(u))
                if specs:
                    out["tree_solo"].append(Job(f"tree_solo:{key}:lot{i}", specs,
                                                LGB_THREADS_SOLO, MEM["tree_solo"][sc],
                                                (f"ledgers:{key}",)))
        if "gnn_solo" in cells:
            for i, lot in enumerate(_split_lots(banks, weights, LOTS["gnn_solo"][sc])):
                specs = tuple(u.spec for b in lot for s in SEEDS
                              for u in (Unit("gnn_solo", key, b, s),) if todo(u))
                if specs:
                    out["gnn_solo"].append(Job(f"gnn_solo:{key}:lot{i}", specs,
                                               GNN_THREADS, MEM["gnn_solo"][sc],
                                               (f"ledgers:{key}", CODEPATH_GATE)))

    glob = [u.spec for u in (Unit(c, key) for c in ("tree_global", "blind_global")
                             if c in cells) if todo(u)]
    if glob:
        deps = (f"ledgers:{key}",) + ((STREAMED_GATE,) if key in MEDIUM else ())
        out["tree_global"].append(Job(f"tree_global:{key}", tuple(glob),
                                      LGB_THREADS_GLOBAL, MEM["tree_global"][sc], deps))
    seeds = [s for s in SEEDS if todo(Unit("gnn_global", key, seed=s))]
    if seeds:
        if not gnn_inputs_path(key).exists():
            out["gnn_inputs"].append(Job(f"gnn_inputs:{key}", (f"gnn_inputs:{key}",),
                                         1, MEM["gnn_inputs"][sc], (f"ledgers:{key}",)))
        deps = (CODEPATH_GATE, f"gnn_inputs:{key}")
        if sc == "medium":
            for s in seeds:
                out["gnn_global"].append(Job(f"gnn_global:{key}:{s}",
                                             (f"gnn_global:{key}:{s}",), GNN_THREADS,
                                             MEM["gnn_global"][sc], deps))
        else:
            out["gnn_global"].append(Job(f"gnn_global:{key}",
                                         tuple(f"gnn_global:{key}:{s}" for s in seeds),
                                         GNN_THREADS, MEM["gnn_global"][sc], deps))
    return out


def plan_jobs(done: dict, exclude: set[str] = frozenset()) -> list[Job]:
    """Every job still to run, in launch priority (the
    tree family and the gates lead, HI-Medium's GNN global seeds follow,
    LI-Medium last among the six datasets of extract.py, the sensitivity GNN
    trailing)."""
    sensitivity = data.SENSITIVITY
    per = {key: dataset_jobs(key, done, exclude) for key in data.PRIMARY + data.ANCHOR_ONLY}
    sens = {key: dataset_jobs(key, done, exclude) for key in sensitivity}
    lead = ["HI-Small", "HI-Medium", "transxion", data.AMLGENTEX_DATASET, "LI-Small"]

    def take(keys, kind, table=per):
        return [j for key in keys for j in table[key][kind]]

    queue: list[Job] = take(lead, "ledgers")
    if not gnn_inputs_path("HI-Small").exists():
        queue += [j for j in per["HI-Small"]["gnn_inputs"]] or [
            Job("gnn_inputs:HI-Small", ("gnn_inputs:HI-Small",), 1,
                MEM["gnn_inputs"]["small"], ("ledgers:HI-Small",))]
    if gate_state(done, STREAMED_GATE) == "unrun":
        queue.append(Job(STREAMED_GATE, (STREAMED_GATE,), LGB_THREADS_GLOBAL,
                         MEM[STREAMED_GATE], (f"ledgers:{GATE_KEY}",)))
    if gate_state(done, CODEPATH_GATE) == "unrun":
        queue.append(Job(CODEPATH_GATE, (CODEPATH_GATE,), GNN_THREADS,
                         MEM[CODEPATH_GATE], ("ledgers:HI-Small", "gnn_inputs:HI-Small")))
    queue += take(lead, "tree_solo")
    queue += take(("HI-Small", "transxion", data.AMLGENTEX_DATASET, "LI-Small",
                   "HI-Medium"), "tree_global")
    queue += take(("HI-Medium",), "gnn_inputs") + take(("HI-Medium",), "gnn_global")
    queue += take(lead, "gnn_solo")
    for key in ("HI-Small", "transxion", data.AMLGENTEX_DATASET, "LI-Small"):
        queue += [j for j in take((key,), "gnn_inputs") if j not in queue]
        queue += take((key,), "gnn_global")
    for kind in ("ledgers", "tree_solo", "tree_global", "gnn_solo", "gnn_inputs",
                 "gnn_global"):
        queue += take(("LI-Medium",), kind)
    for key in sensitivity:
        queue += take((key,), "ledgers", sens) + take((key,), "tree_solo", sens) \
            + take((key,), "tree_global", sens)
    for key in sensitivity:
        for kind in ("gnn_inputs", "gnn_global", "gnn_solo"):
            queue += [Job(j.name, j.specs, j.threads, j.mem_gb, j.deps, j.cost, True)
                      for j in take((key,), kind, sens)]
    seen: set[str] = set()
    ordered = []
    for job in queue:
        if job.name not in seen and (job.specs or job.name.endswith(":pending")):
            seen.add(job.name)
            ordered.append(job)
    return ordered


def dep_satisfied(name: str, done: dict) -> bool:
    kind, _, key = name.partition(":")
    if kind == "ledgers":
        return data.ledgers_ready(key)
    if kind == "gnn_inputs":
        return gnn_inputs_path(key).exists()
    if kind in (STREAMED_GATE, CODEPATH_GATE):
        return gate_state(done, kind) == "passed"
    return False


def free_memory_gb() -> float:
    """Memory a new process could take without paging: the machine minus
    wired, anonymous and compressor pages (file-backed pages are
    reclaimable, and macOS keeps gigabytes of them after the parquet
    reads); psutil's ``available`` when ``vm_stat`` is unavailable."""
    import psutil

    total = psutil.virtual_memory().total
    try:
        out = _sp.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    except (OSError, _sp.CalledProcessError):
        return psutil.virtual_memory().available / 2**30
    page = 16384
    fields = {}
    for line in out.splitlines():
        if "page size of" in line:
            page = int(line.split("page size of")[1].split()[0])
        elif ":" in line:
            name, value = line.rsplit(":", 1)
            fields[name.strip()] = int(value.strip().rstrip("."))
    held = (fields.get("Pages wired down", 0) + fields.get("Anonymous pages", 0)
            + fields.get("Pages occupied by compressor", 0)) * page
    return (total - held) / 2**30


def failed_units_since(started: float, in_flight: set[str]) -> set[str]:
    """Specs of units whose Runs failed since the launcher started, or were
    left ``running`` by a child that is gone (a crash record); neither is
    re-dealt into a lot this session."""
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
        unit = Unit(config["check"], config.get("dataset_key", "-"),
                    config.get("bank", "-"), config.get("seed", -1))
        if status["status"] == "failed" or unit.spec not in in_flight:
            out.add(unit.spec)
    return out


MAX_JOB_FAILURES = 3


LOCK = LOG_DIR / "launcher.pid"
WATCHDOG_SECONDS = 3 * 3600


def acquire_lock() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK.exists():
        pid = int(LOCK.read_text().strip() or 0)
        try:
            os.kill(pid, 0)
            raise SystemExit(f"a launcher is already running (pid {pid}, {LOCK})")
        except ProcessLookupError:
            pass
    LOCK.write_text(f"{os.getpid()}\n")


def cmd_launch(budget: int, machine_gb: float, poll: int) -> int:
    check_execution_settings()
    acquire_lock()
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

    install_signal_handlers(on_signal)
    signal.signal(signal.SIGINT, on_signal)
    print(f"launch: budget {budget} threads, machine {machine_gb} GB, root "
          f"{experiments_root()}; logs in {LOG_DIR}", flush=True)
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
                        print(f"[{time.strftime('%H:%M:%S')}] WATCHDOG {name}: no output "
                              f"for {WATCHDOG_SECONDS // 3600} h; terminating", flush=True)
                        proc.send_signal(signal.SIGTERM)
                        terminated[name] = time.time()
                    continue
                handle.close()
                del running[name]
                if code == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] done {name}", flush=True)
                else:
                    failures[name] = failures.get(name, 0) + 1
                    if failures[name] >= MAX_JOB_FAILURES or name in (STREAMED_GATE, CODEPATH_GATE):
                        failed.add(name)
                    print(f"[{time.strftime('%H:%M:%S')}] FAILED {name} (exit {code}, "
                          f"failure {failures[name]}); its failed units are excluded, "
                          f"the rest re-planned", flush=True)
            done = done_index()
            for gate in (STREAMED_GATE, CODEPATH_GATE):
                if gate_state(done, gate) == "failed" and gate not in failed:
                    failed.add(gate)
                    print(f"{gate}: recorded pass=False; its dependants are blocked "
                          f"until the code that re-keys the gate changes", flush=True)
            in_flight = {spec for _, job, _, _ in running.values() for spec in job.specs}
            exclude = in_flight | failed_units_since(started, in_flight)
            plan = plan_jobs(done, exclude)
            for job in plan:
                if job.name.endswith(":pending") and any(d in failed for d in job.deps):
                    failed.add(job.name)
            pending = [j for j in plan if j.name.endswith(":pending") and j.name not in failed]
            queue = [j for j in plan if not j.name.endswith(":pending")
                     and j.name not in running and j.name not in failed]
            primary_open = any(not j.trailing for j in queue + pending) or any(
                not job.trailing for _, job, _, _ in running.values())
            if not queue and not running and not pending:
                break
            threads_used = sum(j.threads for _, j, _, _ in running.values())
            reserved = sum(j.mem_gb for _, j, _, _ in running.values())
            trailing_reserved = sum(j.mem_gb for _, j, _, _ in running.values() if j.trailing)
            primary_fits_threads = any(not j.trailing and threads_used + j.threads <= budget
                                       and reserved + j.mem_gb <= machine_gb - 3.0
                                       and all(dep_satisfied(d, done) for d in j.deps)
                                       for j in queue)
            held_threads = 0
            launched_now = False
            for job in queue:
                if job.trailing and primary_open:
                    # Trailing work may use the machine while the last
                    # primary jobs are memory-bound one at a time, but never
                    # holds more memory than the largest primary job needs
                    # to start (12 GB of the 21 GB admitted) and never takes
                    # threads a launchable primary job could use.
                    if primary_fits_threads or trailing_reserved + job.mem_gb > machine_gb - 3.0 - 12.0:
                        continue
                if not all(dep_satisfied(d, done) for d in job.deps):
                    if any(d in failed for d in job.deps):
                        failed.add(job.name)
                        print(f"blocked {job.name}: dependency failed", flush=True)
                    continue
                fits_threads = threads_used + held_threads + job.threads <= budget
                fits_memory = (reserved + job.mem_gb <= machine_gb - 3.0
                               and free_memory_gb() >= job.mem_gb + 2.0)
                if not (fits_threads and fits_memory):
                    # A thread-blocked head keeps its threads in hand so
                    # smaller jobs cannot starve it; a memory-blocked head
                    # waits for the running job whose memory it needs, and
                    # smaller jobs may use the machine meanwhile.
                    if not fits_threads and held_threads == 0:
                        held_threads = job.threads
                    continue
                log_path = LOG_DIR / f"{job.name.replace(':', '_')}.log"
                handle = log_path.open("a")
                args = [sys.executable, str(Path(__file__).resolve()), "run", *job.specs]
                proc = subprocess.Popen(args, stdout=handle, stderr=subprocess.STDOUT,
                                        env=os.environ.copy(), start_new_session=True)
                running[job.name] = (proc, job, handle, log_path)
                threads_used += job.threads
                reserved += job.mem_gb
                if job.trailing:
                    trailing_reserved += job.mem_gb
                launched_now = True
                print(f"[{time.strftime('%H:%M:%S')}] start {job.name} "
                      f"({len(job.specs)} units, {job.threads} thr, {job.mem_gb} GB; "
                      f"running {len(running)}, threads {threads_used}, reserved "
                      f"{reserved:.1f} GB) -> {log_path.name}", flush=True)
                time.sleep(5)
            if not launched_now and not running:
                idle_polls += 1
                if idle_polls > 3:
                    print(f"stalled: {[j.name for j in queue + pending]} cannot launch "
                          f"(dependencies or resources); exiting", flush=True)
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
    done = done_index()
    for job in plan_jobs(done):
        print(f"{job.name}: {len(job.specs)} units, {job.threads} thr, "
              f"{job.mem_gb} GB, deps {list(job.deps)}"
              + (" [trailing]" if job.trailing else ""))
    return 0


def cmd_status() -> int:
    runs = experiments_root() / EXPERIMENT / "runs"
    if not runs.exists():
        print("no Runs yet")
        return 0
    counts: dict[tuple, int] = {}
    failed = []
    for status_path in sorted(runs.glob("*/status.json")):
        status = json.loads(status_path.read_text())
        config = json.loads((status_path.parent / "config.json").read_text())
        key = (config.get("check"), config.get("dataset_key"), status["status"])
        counts[key] = counts.get(key, 0) + 1
        if status["status"] != "complete":
            failed.append((status_path.parent.name, config.get("check"),
                           config.get("dataset_key"), config.get("bank"),
                           config.get("seed"), status["status"], status.get("error", "")))
    for (check, key, st), n in sorted(counts.items()):
        print(f"{check:16s} {str(key):36s} {st:9s} {n}")
    for row in failed:
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
