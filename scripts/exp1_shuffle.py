#!/usr/bin/env python
"""Experiment 1 — the shuffled-bank-label control.

One uniform permutation of the bank-label multiset over the primary
dataset's account nodes — every bank identifier, stubs included — so each
bank keeps exactly its account count. Only the label moves: the payment
network is byte-identically unchanged, because the original "bank:account"
string travels with each node as its account identity (raw account numbers
repeat across banks and would silently merge distinct nodes otherwise). The
new label is used for ledger membership, either-endpoint attribution and
the qualifying rule only; it is never a model input. Ledgers are rebuilt
into a derived cache keyed by the permutation seed; the global scenario's
Runs and their persisted test scores are reused by config identity, and
only the solo scenario retrains — the tree and blind grids and the GNN at
three seeds, per qualifying bank of the relabelled frame.

  uv run python scripts/exp1_shuffle.py plan
  uv run python scripts/exp1_shuffle.py run <unit> [...]
  uv run python scripts/exp1_shuffle.py launch [--budget 14] [--machine-gb 24]
  uv run python scripts/exp1_shuffle.py status

Unit specs: ``ledgers:<key>``, ``tree_solo:<key>:<bank>``,
``blind_solo:<key>:<bank>``, ``gnn_solo:<key>:<bank>:<seed>``. Every Run
lands in Experiment 1's Run log, keyed by Experiment 1's unit config plus the permutation block.
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

import exp1  # noqa: E402
import exp1_data as data  # noqa: E402
from m import completed, run  # noqa: E402
from m.gnn import edge_inputs, payment_graph  # noqa: E402
from m.runlog import REPO_ROOT, config_hash, experiments_root  # noqa: E402

EXPERIMENT = exp1.EXPERIMENT
SMOKE = exp1.SMOKE
SEEDS = exp1.SEEDS
KEY = os.environ.get("M_SMOKE_KEY", "sweden_100K_easy") if SMOKE else "HI-Medium"
SHUFFLE_SEED = 0
PERMUTATION = {
    "seed": SHUFFLE_SEED,
    "scope": "all account nodes of the frame, every bank identifier included",
    "node": "an account node is a (bank, account-number) pair, as "
            "m.features.account_codes keys it",
    "rng": "numpy default_rng(seed); one uniform permutation of the "
           "bank-label multiset over account nodes, so each bank keeps "
           "exactly its account count",
    "identity": "the original bank:account string is carried as the account "
                "column, so no two nodes merge where raw account numbers "
                "repeat across banks",
    "label_use": "ledger membership, either-endpoint attribution and the "
                 "qualifying rule only; never a model input",
}
READING = "shuffled bank labels"

LOG_DIR = Path(os.environ.get("M_LOG_DIR", REPO_ROOT / "logs" / "exp1_shuffle"))


def cache_root() -> Path:
    if SMOKE:
        return Path(os.environ["M_EXPERIMENTS_DIR"]) / "interim_exp1_shuffle"
    return REPO_ROOT / "data" / "interim" / "exp1" / "ledgers_shuffled"


def ledger_dir(key: str) -> Path:
    return cache_root() / f"seed{SHUFFLE_SEED}" / key


def ledgers_ready(key: str) -> bool:
    out = ledger_dir(key)
    return (out / "roster.csv").exists() and (out / "copies.npy").exists()


def read_roster(key: str) -> pd.DataFrame:
    return pd.read_csv(ledger_dir(key) / "roster.csv",
                       dtype={"bank": str}).set_index("bank")


def read_rates(key: str) -> dict[str, float]:
    return json.loads((ledger_dir(key) / "rates.json").read_text())


def read_ledger(key: str, bank: str) -> pd.DataFrame:
    return pd.read_parquet(ledger_dir(key) / f"{bank}.parquet")


def shuffled_frame(key: str) -> pd.DataFrame:
    """The canonical frame with the bank-label multiset permuted over its
    account nodes; the network itself is unchanged, each node keeping its
    payments under a globally unique identity string."""
    df = data.frame(key)
    n = len(df)
    fb = df["from_bank"].cat.codes.to_numpy(np.int64)
    tb = df["to_bank"].cat.codes.to_numpy(np.int64)
    bank_cats = df["from_bank"].cat.categories
    accts = df["from_acct"].cat.categories.union(df["to_acct"].cat.categories)
    n_acct = len(accts)
    fa = accts.get_indexer(df["from_acct"].cat.categories).astype(np.int64)[
        df["from_acct"].cat.codes.to_numpy()]
    ta = accts.get_indexer(df["to_acct"].cat.categories).astype(np.int64)[
        df["to_acct"].cat.codes.to_numpy()]
    nodes, codes = np.unique(np.concatenate([fb * n_acct + fa, tb * n_acct + ta]),
                             return_inverse=True)
    payer, payee = codes[:n], codes[n:]
    node_bank = (nodes // n_acct).astype(np.int64)
    node_acct = (nodes % n_acct).astype(np.int64)
    assert np.array_equal(payer == payee, df["same_account"].to_numpy())

    rng = np.random.default_rng(SHUFFLE_SEED)
    new_bank = node_bank[rng.permutation(len(nodes))]
    assert np.array_equal(np.bincount(node_bank, minlength=len(bank_cats)),
                          np.bincount(new_bank, minlength=len(bank_cats)))

    identity = pd.Index(bank_cats[node_bank].astype(str)) + ":" \
        + pd.Index(accts[node_acct].astype(str))
    assert identity.is_unique
    df["from_bank"] = pd.Categorical.from_codes(new_bank[payer], bank_cats)
    df["to_bank"] = pd.Categorical.from_codes(new_bank[payee], bank_cats)
    df["from_acct"] = pd.Categorical.from_codes(payer, identity)
    df["to_acct"] = pd.Categorical.from_codes(payee, identity)
    return df


def build_ledgers(key: str) -> pd.DataFrame:
    df = shuffled_frame(key)
    table = data.write_ledgers(key, df=df, out=ledger_dir(key))
    print(f"shuffled ledgers {key} (seed {SHUFFLE_SEED}): {len(table)} "
          f"qualifying banks, test positives {int(table['test_pos'].min())} to "
          f"{int(table['test_pos'].max())}, "
          f"{int((table['val_pos'] == 0).sum())} with no validation-window "
          f"laundering payment", flush=True)
    return table


def roster_for(key: str) -> pd.DataFrame:
    table = read_roster(key)
    if SMOKE and len(table) > exp1.SMOKE_BANKS:
        picks = np.unique(np.linspace(0, len(table) - 1, exp1.SMOKE_BANKS)
                          .round().astype(int))
        table = table.iloc[picks]
    return table


class ShuffledLedgerCache(exp1.LedgerCache):
    def load(self, key: str, bank: str):
        if (key, bank) != self.current:
            self.rates = read_rates(key)
            self.ledger = read_ledger(key, bank)
            self.feats = None
            self.current = (key, bank)
        return self.ledger


def parse_unit(spec: str) -> exp1.Unit:
    parts = spec.split(":")
    check = parts[0]
    if check == "ledgers":
        return exp1.Unit(check, parts[1])
    if check in ("tree_solo", "blind_solo"):
        return exp1.Unit(check, parts[1], parts[2])
    if check == "gnn_solo":
        return exp1.Unit(check, parts[1], parts[2], int(parts[3]))
    raise SystemExit(f"unknown unit spec {spec}")


def unit_config(unit: exp1.Unit) -> dict:
    return exp1.unit_config(unit) | {"control": "shuffled_bank_labels",
                                   "permutation": PERMUTATION,
                                   "reading": READING}


# --------------------------------------------------------------- runner --

def execute(unit: exp1.Unit, done: dict, cache: ShuffledLedgerCache) -> dict | None:
    with exp1.RssSampler() as sampler:
        return _execute(unit, done, cache, sampler)


def _execute(unit: exp1.Unit, done: dict, cache: ShuffledLedgerCache,
             sampler: exp1.RssSampler) -> dict | None:
    if unit.check == "ledgers":
        if not ledgers_ready(unit.key):
            build_ledgers(unit.key)
        return None

    config = unit_config(unit)
    recorded = done.get(config_hash(config))
    if recorded is not None:
        return json.loads((recorded / "metrics.json").read_text())

    def sampled(fn):
        def wrapped(cfg, run_dir):
            return fn(cfg, run_dir) | exp1.memory_record(sampler)
        return wrapped

    if unit.check in ("tree_solo", "blind_solo"):
        def fn(cfg, run_dir):
            ledger = cache.load(unit.key, unit.bank)
            return exp1.tree_solo_metrics(unit, ledger,
                                        cache.features(unit.key, unit.bank),
                                        run_dir)
        return run(EXPERIMENT, config, sampled(fn), done=done)
    if unit.check == "gnn_solo":
        if exp1.gate_state(done, exp1.CODEPATH_GATE) != "passed" and not SMOKE:
            raise SystemExit(f"{unit.spec} refused: the GNN code-path gate "
                             f"has not passed")

        def fn(cfg, run_dir):
            ledger = cache.load(unit.key, unit.bank)
            arrays = edge_inputs(ledger, cache.rates)
            graph = payment_graph(arrays[0], arrays[1], arrays[2], arrays[7])
            test_payment = ledger["payment"].to_numpy(np.int64)[arrays[6]["test"]]
            return exp1.train_and_record(arrays, graph, unit.seed, run_dir,
                                       f"shuffled {unit.spec}", test_payment)
        return run(EXPERIMENT, config, sampled(fn), seed=unit.seed, done=done)
    raise SystemExit(f"unknown unit {unit.spec}")


def cmd_run(specs: list[str]) -> int:
    import pyarrow as pa

    exp1.check_execution_settings()
    exp1.install_signal_handlers()
    torch.set_num_threads(exp1.TORCH_THREADS)
    pa.set_cpu_count(2)
    done = completed(EXPERIMENT)
    cache = ShuffledLedgerCache()
    print(f"run {len(specs)} shuffled-control units (root {experiments_root()}, "
          f"seed {SHUFFLE_SEED}, smoke {SMOKE})", flush=True)
    failures = 0
    for spec in specs:
        unit = parse_unit(spec)
        t0 = time.time()
        try:
            metrics = execute(unit, done, cache)
        except SystemExit:
            raise
        except Exception as exc:
            failures += 1
            print(f"{unit.spec}: FAILED {type(exc).__name__}: {exc} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
            continue
        print(f"{exp1.summary_line(unit, metrics)} [{time.time() - t0:.0f}s]",
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


MEM = {"ledgers": 12.0, "tree_solo": 3.0, "gnn_solo": 7.0}
LOTS = {"tree_solo": 5, "gnn_solo": 3}
GNN_THREADS = exp1.TORCH_THREADS + 1


def plan_jobs(done: dict, exclude: frozenset[str] = frozenset()) -> list[Job]:
    key = KEY
    jobs: list[Job] = []
    if not ledgers_ready(key):
        jobs.append(Job(f"ledgers:{key}", (f"ledgers:{key}",), 1, MEM["ledgers"]))
        jobs.append(Job(f"tree_solo:{key}:pending", (), exp1.LGB_THREADS_SOLO,
                        MEM["tree_solo"], (f"ledgers:{key}",)))
        jobs.append(Job(f"gnn_solo:{key}:pending", (), GNN_THREADS,
                        MEM["gnn_solo"], (f"ledgers:{key}",)))
        return jobs
    todo = lambda u: (u.spec not in exclude  # noqa: E731
                      and config_hash(unit_config(u)) not in done)
    table = roster_for(key)
    banks = list(table.index)
    weights = [float(w) for w in table["touching"]]
    for i, lot in enumerate(exp1._split_lots(banks, weights, LOTS["tree_solo"])):
        specs = tuple(u.spec for b in lot
                      for u in (exp1.Unit("tree_solo", key, b),
                                exp1.Unit("blind_solo", key, b)) if todo(u))
        if specs:
            jobs.append(Job(f"tree_solo:{key}:lot{i}", specs,
                            exp1.LGB_THREADS_SOLO, MEM["tree_solo"]))
    for i, lot in enumerate(exp1._split_lots(banks, weights, LOTS["gnn_solo"])):
        specs = tuple(u.spec for b in lot for s in SEEDS
                      for u in (exp1.Unit("gnn_solo", key, b, s),) if todo(u))
        if specs:
            jobs.append(Job(f"gnn_solo:{key}:lot{i}", specs, GNN_THREADS,
                            MEM["gnn_solo"]))
    return jobs


def dep_satisfied(name: str) -> bool:
    kind, _, key = name.partition(":")
    return ledgers_ready(key) if kind == "ledgers" else False


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
            plan = plan_jobs(done, frozenset(exclude))
            for job in plan:
                if job.name.endswith(":pending") and any(d in failed for d in job.deps):
                    failed.add(job.name)
            pending = [j for j in plan if j.name.endswith(":pending")
                       and j.name not in failed]
            queue = [j for j in plan if not j.name.endswith(":pending")
                     and j.name not in running and j.name not in failed]
            if not queue and not running and not pending:
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
                    print(f"stalled: {[j.name for j in queue + pending]}; "
                          f"exiting", flush=True)
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
    done = completed(EXPERIMENT)
    if not ledgers_ready(KEY):
        print(f"{KEY}: shuffled ledgers not built")
        return 0
    table = roster_for(KEY)
    counts: dict[str, int] = {}
    for bank in table.index:
        for unit in (exp1.Unit("tree_solo", KEY, bank), exp1.Unit("blind_solo", KEY, bank),
                     *(exp1.Unit("gnn_solo", KEY, bank, s) for s in SEEDS)):
            state = "complete" if config_hash(unit_config(unit)) in done else "todo"
            counts[f"{unit.check} {state}"] = counts.get(f"{unit.check} {state}", 0) + 1
    print(f"{KEY} (shuffle seed {SHUFFLE_SEED}): {len(table)} qualifying banks")
    for name, n in sorted(counts.items()):
        print(f"  {name}: {n}")
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
