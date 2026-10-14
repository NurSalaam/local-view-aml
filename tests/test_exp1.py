"""Experiment 1 machinery: the streamed copies table
against in-memory construction, the GNN tensor assembly and prefetch
against the stored layout, the row rule's copies, and the statistics.

Run:  uv run python -m unittest tests.test_exp1 -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("M_DEVICE", "cpu")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402  (before lightgbm: OpenMP runtimes, see scripts/exp1.py)
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from m.gnn import (GNNConfig, _attach_edge_features, payment_graph,  # noqa: E402
                   prefetch, train_edge_scorer)
from m.lgbdata import (RepeatedRowsSequence, RowGroupReader, construct,  # noqa: E402
                       fit_grid, predict_matrix, predict_rows, repeat_index)
from m.stats import (aggregate_shortfalls, heterogeneity, joint_bootstrap,  # noqa: E402
                     leave_one_out, null_q_check, spearman_bank_bootstrap,
                     weighted_average_precision, weighted_roc_auc)

HI_SMALL_CACHE = REPO / "data" / "interim" / "features" / "HI-Small_global.parquet"


class TestStreamedCopies(unittest.TestCase):
    def setUp(self):
        from m.grid import FIXED, GRID

        self.grid, self.fixed = GRID[:2], {k: v for k, v in FIXED.items() if k != "num_threads"}
        rng = np.random.default_rng(0)
        n, d = 120_000, 9
        X = rng.standard_normal((n, d)).astype(np.float32)
        X[:, 0] = rng.integers(0, 4, n)
        X[:, 1] = (rng.random(n) < 0.03) * rng.random(n)
        y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 2] + X[:, 3] * X[:, 4] - 3)))).astype(np.int8)
        self.X, self.y = X, y
        self.split = np.where(np.arange(n) < 72_000, 0, np.where(np.arange(n) < 96_000, 1, 2))
        self.counts = rng.integers(1, 3, n)
        self.tmp = tempfile.TemporaryDirectory()
        cols = [f"f{j}" for j in range(d)]
        path = Path(self.tmp.name) / "t.parquet"
        pq.write_table(pa.table({c: X[:, j] for j, c in enumerate(cols)}), path,
                       row_group_size=25_000)
        self.reader = RowGroupReader(path, cols)

    def tearDown(self):
        self.tmp.cleanup()

    def copies(self, s):
        rows = np.flatnonzero(self.split == s)
        return np.repeat(self.X[rows], self.counts[rows], axis=0), np.repeat(self.y[rows], self.counts[rows])

    def test_sequence_rows_match_repeated_matrix(self):
        rows = np.flatnonzero(self.split == 0)
        seq = RepeatedRowsSequence(self.reader, rows, self.counts[rows])
        Xtr, _ = self.copies(0)
        self.assertEqual(len(seq), len(Xtr))
        for i in (0, 1, 24_999, 25_000, 50_001, len(seq) - 1):
            self.assertTrue(np.array_equal(seq[i], Xtr[i].astype(np.float64)))
            self.assertEqual(seq[i].dtype, np.float64)
        self.assertTrue(np.array_equal(seq[24_990:25_010], Xtr[24_990:25_010]))
        self.assertTrue(np.array_equal(seq[0:len(seq)], Xtr))

    def test_repeat_index(self):
        starts = repeat_index(np.array([3, 7, 9]), np.array([2, 1, 2]))
        self.assertEqual(starts.tolist(), [0, 2, 3, 5])

    def test_streamed_construction_reproduces_in_memory_predictions(self):
        Xtr, ytr = self.copies(0)
        Xva, yva = self.copies(1)
        dtr = construct(Xtr, ytr, [0])
        dva = construct(Xva, yva, [0], reference=dtr)
        mem = fit_grid(dtr, dva, self.grid, self.fixed, 60, 10, 2)
        r0, r1 = np.flatnonzero(self.split == 0), np.flatnonzero(self.split == 1)
        s_tr = RepeatedRowsSequence(self.reader, r0, self.counts[r0])
        s_va = RepeatedRowsSequence(self.reader, r1, self.counts[r1])
        d_tr = construct(s_tr, ytr, [0])
        d_va = construct(s_va, yva, [0], reference=d_tr)
        streamed = fit_grid(d_tr, d_va, self.grid, self.fixed, 60, 10, 2)
        Xte = self.X[self.split == 2]
        for a, b in zip(mem, streamed):
            self.assertEqual(a.best_iteration, b.best_iteration)
            self.assertTrue(np.array_equal(a.predict(Xte, num_iteration=a.best_iteration),
                                           b.predict(Xte, num_iteration=b.best_iteration)))
        self.assertTrue(np.array_equal(predict_rows(mem, self.reader, self.split == 2),
                                       predict_matrix(mem, Xte)))


class TestRowCopies(unittest.TestCase):
    def test_copies_per_endpoint_bank(self):
        from m.features import row_copies

        payment, group = row_copies(np.array([0, 1, 2, 2]), np.array([1, 1, 0, 2]))
        self.assertEqual(payment.tolist(), [0, 0, 1, 2, 2, 3])
        self.assertEqual(group.tolist(), [0, 1, 1, 0, 2, 2])


class TestGNNAssembly(unittest.TestCase):
    def synthetic(self, seed=0, n_nodes=300, e=6000):
        rng = np.random.default_rng(seed)
        ts = np.sort(rng.integers(0, 10_000, e)).astype(np.int64)
        payer = rng.integers(0, n_nodes, e).astype(np.int64)
        payee = rng.integers(0, n_nodes, e).astype(np.int64)
        numeric = rng.standard_normal((e, 5)).astype(np.float32)
        codes = np.stack([rng.integers(0, k, e) for k in (4, 6, 6)], 1).astype(np.int32)
        y = (rng.random(e) < 0.05).astype(np.int8)
        rows = {"train": np.arange(0, 3600), "val": np.arange(3600, 4800),
                "test": np.arange(4800, e)}
        return ts, payer, payee, numeric, codes, y, rows, n_nodes

    def stored_layout(self, ts, payer, payee, numeric, codes, n_nodes):
        from torch_geometric.data import Data

        e = len(ts)
        edge_index = np.empty((2, 2 * e), np.int64)
        edge_index[0, :e], edge_index[1, :e] = payer, payee
        edge_index[0, e:], edge_index[1, e:] = payee, payer
        direction = np.zeros((2 * e, 1), np.float32)
        direction[e:] = 1.0
        numeric2 = np.hstack([np.vstack([numeric, numeric]), direction])
        codes2 = np.vstack([codes, codes])
        time2 = np.concatenate([ts, ts])
        order = np.argsort(time2, kind="stable")
        return Data(num_nodes=n_nodes,
                    edge_index=torch.from_numpy(np.ascontiguousarray(edge_index[:, order])),
                    time=torch.from_numpy(time2[order]),
                    edge_numeric=torch.from_numpy(numeric2[order]),
                    edge_codes=torch.from_numpy(codes2[order].astype(np.int32)))

    def test_gathered_batch_equals_stored_batch(self):
        from torch_geometric.loader import LinkNeighborLoader

        ts, payer, payee, numeric, codes, y, rows, n_nodes = self.synthetic()
        old = self.stored_layout(ts, payer, payee, numeric, codes, n_nodes)
        new = payment_graph(ts, payer, payee, n_nodes)
        r = rows["train"]

        def loader(data):
            return LinkNeighborLoader(
                data, num_neighbors=[10, 10],
                edge_label_index=torch.from_numpy(np.stack([payer[r], payee[r]])),
                edge_label=torch.arange(len(r)), batch_size=512, shuffle=True,
                num_workers=0, edge_label_time=torch.from_numpy(ts[r] - 1), time_attr="time")

        torch.manual_seed(5)
        bo = next(iter(loader(old)))
        torch.manual_seed(5)
        bn = next(iter(loader(new)))
        _attach_edge_features(bn, torch.from_numpy(numeric), torch.from_numpy(codes))
        self.assertTrue(torch.equal(bo.edge_index, bn.edge_index))
        self.assertTrue(torch.equal(bo.edge_numeric, bn.edge_numeric))
        self.assertTrue(torch.equal(bo.edge_codes, bn.edge_codes))
        self.assertTrue(torch.equal(bo.edge_label_index, bn.edge_label_index))

    def test_prefetch_preserves_order_and_propagates_errors(self):
        self.assertEqual(list(prefetch(iter(range(50)), depth=3)), list(range(50)))

        def broken():
            yield 1
            raise ValueError("boom")

        with self.assertRaises(ValueError):
            list(prefetch(broken()))

    def test_training_is_reproducible_run_to_run(self):
        ts, payer, payee, numeric, codes, y, rows, n_nodes = self.synthetic()
        cfg = GNNConfig(hidden=16, fanout=(5, 5), batch_size=512, max_epochs=2,
                        loader_workers=0)
        graph = payment_graph(ts, payer, payee, n_nodes)
        outs = []
        for _ in range(2):
            torch.manual_seed(0)
            outs.append(train_edge_scorer(graph, payer, payee, ts, numeric, codes, y,
                                          rows, cfg, torch.device("cpu"), log=lambda m: None))
        self.assertTrue(np.array_equal(outs[0]["test_probs"], outs[1]["test_probs"]))
        self.assertEqual(outs[0]["trace"], outs[1]["trace"])


class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_weighted_metrics_match_sklearn(self):
        from sklearn.metrics import average_precision_score, roc_auc_score

        for _ in range(3):
            n = 400
            y = (self.rng.random(n) < 0.1).astype(np.int8)
            s = np.round(self.rng.random(n) + 0.5 * y, 2)
            w = self.rng.integers(0, 4, n).astype(float)
            w[y == 1] += 1
            self.assertAlmostEqual(weighted_average_precision(y, s, w[None])[0],
                                   average_precision_score(y, s, sample_weight=w), places=9)
            self.assertAlmostEqual(weighted_roc_auc(y, s, w[None])[0],
                                   roc_auc_score(y, s, sample_weight=w), places=9)

    def test_null_check_calibrates_under_the_null(self):
        k, R = 30, 2000
        ses = self.rng.random(k) * 0.05 + 0.02
        deltas = 0.2 + self.rng.standard_normal(k) * ses
        reps = deltas[None, :] + self.rng.standard_normal((R, k)) * ses[None, :]
        chk = null_q_check(reps, deltas, ses)
        self.assertAlmostEqual(chk["null_q95"], 42.56, delta=3.0)
        self.assertFalse(chk["passes"])
        wide = 0.2 + self.rng.standard_normal(k) * 0.1
        chk = null_q_check(reps - deltas[None, :] + wide[None, :], wide, ses)
        self.assertTrue(chk["passes"])
        self.assertGreater(heterogeneity(wide, ses)["I2"], 0.5)

    def test_leave_one_out_drops_the_most_weighted(self):
        labels = ["a", "b", "c"]
        out = leave_one_out(labels, np.array([0.1, 0.2, 0.3]), np.array([0.05, 0.01, 0.05]))
        self.assertEqual(out["dropped"], "b")
        self.assertEqual(out["k"], 2)

    def test_joint_bootstrap_shapes_and_aggregates(self):
        n_test = 2000
        y_all = (self.rng.random(n_test) < 0.1).astype(np.int8)
        groups = {}
        for g in range(4):
            idx = np.sort(self.rng.choice(n_test, 300, replace=False))
            y = y_all[idx]
            solo = self.rng.random((3, 300)).astype(np.float32) + 0.2 * y
            glob = self.rng.random((3, 300)).astype(np.float32) + 0.4 * y
            groups[str(g)] = {"idx": idx, "y": y, "solo": solo, "global": glob,
                              "neg_weight": 0.5}
        boot = joint_bootstrap(n_test, groups, replicates=40, seed=0, chunk=16)
        self.assertEqual(boot["reps"]["ap"]["0"].shape, (40, 3))
        for aggregate in ("median", "mean"):
            agg = aggregate_shortfalls(boot, "ap", aggregate)
            for g in groups:
                self.assertEqual(agg[g]["delta_reps"].shape, (40,))
                self.assertGreater(agg[g]["delta"], 0)
                self.assertGreater(agg[g]["se_boot"], 0)
        again = joint_bootstrap(n_test, groups, replicates=40, seed=0, chunk=16)
        self.assertTrue(np.array_equal(again["reps"]["auc"]["1"], boot["reps"]["auc"]["1"]))

    def test_spearman_bootstrap_reads_direction(self):
        a = self.rng.random(40)
        out = spearman_bank_bootstrap(a, a + 0.1 * self.rng.random(40), draws=200,
                                      permutations=500)
        self.assertTrue(out["positive"])
        out = spearman_bank_bootstrap(a, self.rng.random(40), draws=200, permutations=500)
        self.assertLess(out["boot_q05"], out["rho"] + 1e-9)


if __name__ == "__main__":
    unittest.main()


class TestAnalysisRules(unittest.TestCase):
    def test_common_base_rate_weight_moves_every_bank_to_the_minimum(self):
        rates = {"a": (83, 42_215), "b": (45, 45_506), "c": (38, 35_961)}
        common = min(p / n for p, n in rates.values())
        for pos, rows in rates.values():
            pi = pos / rows
            w = (pi / (1 - pi)) / (common / (1 - common))
            self.assertGreaterEqual(w, 1.0)
            self.assertAlmostEqual(pos / (pos + w * (rows - pos)), common, places=12)

    def test_lots_are_deterministic_and_cover_every_bank(self):
        sys.path.insert(0, str(REPO / "scripts"))
        os.environ.setdefault("M_SMOKE", "1")
        os.environ.setdefault("M_EXPERIMENTS_DIR", "/tmp/exp1-test-scratch")
        from exp1 import _split_lots

        banks = [f"b{i}" for i in range(23)]
        weights = [float((i * 7919) % 101 + 1) for i in range(23)]
        a = _split_lots(banks, weights, 5)
        b = _split_lots(list(reversed(banks)), list(reversed(weights)), 5)
        self.assertEqual(a, b)
        self.assertEqual(sorted(x for lot in a for x in lot), sorted(banks))
        self.assertEqual(len(a), 5)
