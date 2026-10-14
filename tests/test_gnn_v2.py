"""GNN code path version 2 against a version-1 reference (unit test for the
GNN selection's code-path equivalence gate).

Run:  uv run python -m unittest tests.test_gnn_v2 -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

os.environ.setdefault("M_DEVICE", "cpu")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from m.gnn import (GNNConfig, PaymentGNN, _attach_edge_features,  # noqa: E402
                   edge_inputs, n_code_levels, payment_graph)


class EdgeSAGEv1(nn.Module):
    def __init__(self, in_dim, edge_dim, out_dim):
        super().__init__()
        self.message = nn.Sequential(nn.Linear(in_dim + edge_dim, out_dim),
                                     nn.ReLU())
        self.root = nn.Linear(in_dim, out_dim)

    def forward(self, x, edge_index, edge_attr):
        src, dst = edge_index
        msg = self.message(torch.cat([x[src], edge_attr], dim=1))
        agg = torch.zeros(x.size(0), msg.size(1), dtype=msg.dtype)
        agg.index_add_(0, dst, msg)
        deg = torch.zeros(x.size(0), dtype=msg.dtype)
        deg.index_add_(0, dst, torch.ones(len(dst), dtype=msg.dtype))
        agg = agg / deg.clamp(min=1).unsqueeze(1)
        return torch.relu(self.root(x) + agg)


class PaymentGNNv1(nn.Module):
    def __init__(self, n_codes, numeric_dim, cfg, head):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, cfg.embed_dim) for n in n_codes])
        edge_dim = numeric_dim + cfg.embed_dim * len(n_codes)
        self.node0 = nn.Parameter(torch.zeros(cfg.hidden))
        self.conv1 = EdgeSAGEv1(cfg.hidden, edge_dim, cfg.hidden)
        self.conv2 = EdgeSAGEv1(cfg.hidden, edge_dim, cfg.hidden)
        head_in = 2 * cfg.hidden + edge_dim if head == "edge" else cfg.hidden
        self.head_kind = head
        self.head = nn.Sequential(nn.Linear(head_in, cfg.hidden), nn.ReLU(),
                                  nn.Linear(cfg.hidden, 1))

    def edge_features(self, numeric, codes):
        parts = [numeric] + [emb(codes[:, j].long())
                             for j, emb in enumerate(self.embeddings)]
        return torch.cat(parts, dim=1)

    def forward(self, batch):
        e = self.edge_features(batch.edge_numeric, batch.edge_codes)
        x = self.node0.unsqueeze(0).expand(batch.num_nodes, -1)
        x = self.conv1(x, batch.edge_index, e)
        x = self.conv2(x, batch.edge_index, e)
        if self.head_kind == "edge":
            src, dst = batch.edge_label_index
            seed = self.edge_features(batch.seed_numeric, batch.seed_codes)
            return self.head(torch.cat([x[src], x[dst], seed], dim=1)).squeeze(1)
        return self.head(x[:batch.batch_size]).squeeze(1)


def v1_state_from_v2(model: PaymentGNN) -> dict:
    return {k.replace(".lin.", ".message.0."): v
            for k, v in model.state_dict().items()}


def synthetic_batch(rng, n_nodes=2_000, n_edges=20_000, n_seeds=256,
                    numeric_dim=6, levels=(7, 15, 15)):
    from torch_geometric.data import Data

    batch = Data(num_nodes=n_nodes)
    batch.edge_index = torch.from_numpy(rng.integers(0, n_nodes, (2, n_edges)))
    batch.edge_numeric = torch.from_numpy(rng.standard_normal((n_edges, numeric_dim)).astype(np.float32))
    batch.edge_codes = torch.from_numpy(np.stack([rng.integers(0, n, n_edges) for n in levels], 1).astype(np.int32))
    batch.edge_label_index = torch.from_numpy(rng.integers(0, n_nodes, (2, n_seeds)))
    batch.seed_numeric = torch.from_numpy(rng.standard_normal((n_seeds, numeric_dim)).astype(np.float32))
    batch.seed_codes = torch.from_numpy(np.stack([rng.integers(0, n, n_seeds) for n in levels], 1).astype(np.int32))
    batch.batch_size = n_seeds
    return batch, list(levels), numeric_dim


class TestCodePathV2(unittest.TestCase):
    def setUp(self):
        self.cfg = GNNConfig()
        self.rng = np.random.default_rng(0)

    def test_initialisation_draws_are_identical(self):
        batch, levels, numeric_dim = synthetic_batch(self.rng)
        torch.manual_seed(7)
        v1 = PaymentGNNv1(levels, numeric_dim, self.cfg, head="edge")
        torch.manual_seed(7)
        v2 = PaymentGNN(levels, numeric_dim, self.cfg, head="edge")
        for a, b in zip(v1.state_dict().values(), v2.state_dict().values()):
            self.assertTrue(torch.equal(a, b))

    def test_logits_equal_within_rounding_on_synthetic_batch(self):
        batch, levels, numeric_dim = synthetic_batch(self.rng)
        torch.manual_seed(7)
        v2 = PaymentGNN(levels, numeric_dim, self.cfg, head="edge")
        v1 = PaymentGNNv1(levels, numeric_dim, self.cfg, head="edge")
        v1.load_state_dict(v1_state_from_v2(v2))
        v1.eval(); v2.eval()
        with torch.no_grad():
            diff = (v1(batch) - v2(batch)).abs().max().item()
        self.assertLess(diff, 1e-6)

    def test_gradients_agree_within_rounding(self):
        batch, levels, numeric_dim = synthetic_batch(self.rng)
        torch.manual_seed(7)
        v2 = PaymentGNN(levels, numeric_dim, self.cfg, head="edge")
        v1 = PaymentGNNv1(levels, numeric_dim, self.cfg, head="edge")
        v1.load_state_dict(v1_state_from_v2(v2))
        target = torch.from_numpy(self.rng.integers(0, 2, batch.batch_size).astype(np.float32))
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(self.cfg.pos_weight))
        loss_fn(v1(batch), target).backward()
        loss_fn(v2(batch), target).backward()
        g1 = torch.cat([p.grad.flatten() for p in v1.parameters()])
        g2 = torch.cat([p.grad.flatten() for p in v2.parameters()])
        rel = ((g1 - g2).abs().max() / g1.abs().max().clamp(min=1e-12)).item()
        self.assertLess(rel, 1e-4)

    def test_node_head_matches(self):
        batch, levels, numeric_dim = synthetic_batch(self.rng)
        torch.manual_seed(7)
        v2 = PaymentGNN(levels, numeric_dim, self.cfg, head="node")
        v1 = PaymentGNNv1(levels, numeric_dim, self.cfg, head="node")
        v1.load_state_dict(v1_state_from_v2(v2))
        v1.eval(); v2.eval()
        with torch.no_grad():
            diff = (v1(batch) - v2(batch)).abs().max().item()
        self.assertLess(diff, 1e-6)

    @unittest.skipUnless((REPO / "data" / "interim" / "features" / "HI-Small_global.parquet").exists(),
                         "HI-Small feature cache absent")
    def test_logits_equal_within_rounding_on_one_real_batch(self):
        from torch_geometric.loader import LinkNeighborLoader

        from extract import frame
        from m.features import usd_rate_table

        df = frame("HI-Small")
        ts, payer, payee, numeric, codes, y, rows, n_nodes = edge_inputs(df, usd_rate_table(df))
        data = payment_graph(ts, payer, payee, n_nodes)
        r = rows["train"]
        loader = LinkNeighborLoader(
            data, num_neighbors=list(self.cfg.fanout),
            edge_label_index=torch.from_numpy(np.stack([payer[r], payee[r]])),
            edge_label=torch.arange(len(r)), batch_size=self.cfg.batch_size,
            shuffle=True, num_workers=0,
            edge_label_time=torch.from_numpy(ts[r] - 1), time_attr="time")
        torch.manual_seed(1)
        batch = next(iter(loader))
        _attach_edge_features(batch, torch.from_numpy(numeric), torch.from_numpy(codes))
        rr = torch.from_numpy(r)[batch.edge_label.long()]
        seed_numeric = torch.from_numpy(np.hstack([numeric, np.zeros((len(numeric), 1), np.float32)]))
        batch.seed_numeric = seed_numeric[rr]
        batch.seed_codes = torch.from_numpy(codes.astype(np.int32))[rr]
        torch.manual_seed(7)
        v2 = PaymentGNN(n_code_levels(codes), seed_numeric.size(1), self.cfg, head="edge")
        v1 = PaymentGNNv1(n_code_levels(codes), seed_numeric.size(1), self.cfg, head="edge")
        v1.load_state_dict(v1_state_from_v2(v2))
        v1.eval(); v2.eval()
        with torch.no_grad():
            diff = (v1(batch) - v2(batch)).abs().max().item()
        self.assertGreater(batch.edge_index.size(1), 100_000)
        self.assertLess(diff, 1e-6)


class TestArtifacts(unittest.TestCase):
    def test_array_round_trip_with_hash(self):
        import tempfile

        from m.artifacts import load_array, write_array

        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "artifacts").mkdir()
            arr = np.arange(10, dtype=np.float32)
            rec = write_array(run_dir, "probs", arr)
            self.assertEqual(rec["file"], "probs.npy")
            self.assertTrue(np.array_equal(load_array(run_dir, "probs", rec["sha256"]), arr))
            with self.assertRaises(ValueError):
                load_array(run_dir, "probs", "0" * 64)


if __name__ == "__main__":
    unittest.main()
