"""The GNN family: a two-layer GraphSAGE-style inductive GNN
(Hamilton 2018).

Payments are directed, so the message-passing graph holds each payment twice
— once forward, once reversed, with a direction flag in the numeric edge
features — and one mean aggregation over in-edges then covers both
directions. Neighbour sampling is time-respecting: the sampler admits edges
with time at most the seed's query time, and the query time is the payment's
timestamp minus one second, so on integer timestamps the context is exactly
the strictly-earlier payments — the same discipline as the tree family's
windows. Nodes carry no observed features (a learned constant vector), so
everything the model knows arrives through structure and edge attributes,
which keeps it inductive.

The edge head scores payment-labelled worlds from both endpoint embeddings
plus the payment's own fields; the node head scores account-labelled worlds
from the account embedding alone.

Code path version 2 (recorded in Run configs as ``gnn_version``): the layer
message relu(W [x_src ; e] + b) is computed as relu((x W_x)[src] + e W_e + b)
— the node part projected once per node and gathered per edge, a single
broadcast vector in layer 1 where every node carries the constant vector —
and the tiny code vocabularies are looked up as one-hot products with the
embedding tables. Both are algebraically identical to version 1 and differ
only in floating-point summation order; the parameters, their shapes, and
their initialisation draws are unchanged.

Tensor assembly (value-preserving, so the version stays 2; ``ASSEMBLY_VERSION``
re-keys the gates and Runs that depend on it): the graph stores
one payment identifier and one direction flag per edge, and each batch's
edge features are gathered from the payment table by that identifier —
the same float32 values the earlier twice-stored layout held, without the
second copy. Batches are drawn on a prefetch thread so neighbour sampling
overlaps the model step; the sampler is the only consumer of the random
stream while a loader is being iterated, so the sampled subgraphs and every
downstream number are unchanged.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

VERSION = 2
ASSEMBLY_VERSION = 1

PREFETCH_DEPTH = 4


@dataclass(frozen=True)
class GNNConfig:
    hidden: int = 64
    embed_dim: int = 8
    fanout: tuple[int, int] = (15, 15)
    batch_size: int = 4096
    lr: float = 2e-3
    pos_weight: float = 6.0
    max_epochs: int = 15
    patience: int = 3
    loader_workers: int = 2
    temporal: bool = True

    def as_dict(self) -> dict:
        out = {"hidden": self.hidden, "embed_dim": self.embed_dim,
               "fanout": list(self.fanout), "batch_size": self.batch_size,
               "lr": self.lr, "pos_weight": self.pos_weight,
               "max_epochs": self.max_epochs, "patience": self.patience,
               "loader_workers": self.loader_workers}
        if not self.temporal:
            out["temporal"] = False
        return out


class EdgeSAGE(nn.Module):
    """One SAGE-style layer with edge features and mean aggregation.

    ``x`` is either one row per node or a single row (a constant node vector)
    that broadcasts over nodes and edges; ``deg`` is the in-degree of every
    node in the batch graph, computed once per batch by the caller.
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim + edge_dim, out_dim)
        self.root = nn.Linear(in_dim, out_dim)
        self.in_dim = in_dim

    def forward(self, x, edge_index, edge_attr, deg):
        src, dst = edge_index
        w_x = self.lin.weight[:, :self.in_dim]
        w_e = self.lin.weight[:, self.in_dim:]
        node_part = x @ w_x.t()
        gathered = node_part if node_part.size(0) == 1 else node_part[src]
        msg = torch.relu(gathered + torch.addmm(self.lin.bias, edge_attr, w_e.t()))
        agg = torch.zeros(deg.size(0), msg.size(1), device=msg.device,
                          dtype=msg.dtype)
        agg.index_add_(0, dst, msg)
        return torch.relu(self.root(x) + agg / deg.clamp(min=1).unsqueeze(1))


class PaymentGNN(nn.Module):
    def __init__(self, n_codes: list[int], numeric_dim: int, cfg: GNNConfig,
                 head: str):
        super().__init__()
        self.n_codes = list(n_codes)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(n, cfg.embed_dim) for n in n_codes])
        edge_dim = numeric_dim + cfg.embed_dim * len(n_codes)
        self.node0 = nn.Parameter(torch.zeros(cfg.hidden))
        self.conv1 = EdgeSAGE(cfg.hidden, edge_dim, cfg.hidden)
        self.conv2 = EdgeSAGE(cfg.hidden, edge_dim, cfg.hidden)
        head_in = 2 * cfg.hidden + edge_dim if head == "edge" else cfg.hidden
        self.head_kind = head
        self.head = nn.Sequential(nn.Linear(head_in, cfg.hidden), nn.ReLU(),
                                  nn.Linear(cfg.hidden, 1))

    def edge_features(self, numeric, codes):
        parts = [numeric]
        for j, (n, emb) in enumerate(zip(self.n_codes, self.embeddings)):
            onehot = F.one_hot(codes[:, j].long(), n).to(numeric.dtype)
            parts.append(onehot @ emb.weight)
        return torch.cat(parts, dim=1)

    def forward(self, batch):
        e = self.edge_features(batch.edge_numeric, batch.edge_codes)
        deg = torch.bincount(batch.edge_index[1],
                             minlength=batch.num_nodes).to(e.dtype)
        x = self.conv1(self.node0.unsqueeze(0), batch.edge_index, e, deg)
        x = self.conv2(x, batch.edge_index, e, deg)
        if self.head_kind == "edge":
            src, dst = batch.edge_label_index
            seed = self.edge_features(batch.seed_numeric, batch.seed_codes)
            return self.head(torch.cat([x[src], x[dst], seed], dim=1)).squeeze(1)
        return self.head(x[:batch.batch_size]).squeeze(1)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def raw_numeric(fields: dict[str, np.ndarray]) -> np.ndarray:
    """The unstandardised numeric edge block: log amounts, cross-currency
    flag, scaled hour and weekday."""
    return np.column_stack([
        np.log1p(np.maximum(fields["amount_usd"], 0.0)),
        np.log1p(np.maximum(fields["amount_received_usd"], 0.0)),
        fields["cross_currency"],
        fields["hour_of_day"] / 23.0,
        fields["day_of_week"] / 6.0,
    ]).astype(np.float32)


def standardise_numeric(numeric: np.ndarray, constants: dict) -> None:
    """Standardise the two log-amount columns in place with the given
    constants, exactly as the training-frame computation applies its own."""
    for j in (0, 1):
        mean, std = np.float32(constants["mean"][j]), np.float32(constants["std"][j])
        numeric[:, j] = (numeric[:, j] - mean) / max(std, 1e-6)


def shipped_constants(df: pd.DataFrame, rates: dict[str, float]) -> dict:
    """The frame's standardisation constants — the log-amount means and
    standard deviations over its training rows. These are the aggregate
    summaries that travel with a trained model as a deployed pipeline's
    scaler does; they are not payment records."""
    from m.features import own_fields

    numeric = raw_numeric(own_fields(df, rates))
    train = df["split"].to_numpy() == 0
    return {"mean": [float(numeric[train, j].mean()) for j in (0, 1)],
            "std": [float(numeric[train, j].std()) for j in (0, 1)]}


def edge_inputs(df: pd.DataFrame, rates: dict[str, float],
                constants: dict | None = None):
    """The GNN family's inputs for one sight's frame: timestamps, endpoint
    node codes compact within the frame, the standardised numeric block
    (log amounts standardised on the frame's training rows, cross-currency
    flag, hour and weekday), the code columns, labels, split rows, and the
    node count. ``rates`` is the dataset-level USD table. ``constants``
    standardises with another frame's shipped constants instead of this
    frame's own training rows (cross-bank model transfer)."""
    from m.features import account_codes, own_fields

    fields = own_fields(df, rates)
    payer, payee = account_codes(df)
    n_nodes = int(max(payer.max(), payee.max())) + 1
    ts = df["ts"].to_numpy(np.int64)
    split = df["split"].to_numpy()
    y = df["is_laundering"].to_numpy(np.int8)

    train = split == 0
    numeric = raw_numeric(fields)
    if constants is None:
        constants = {"mean": [float(numeric[train, j].mean()) for j in (0, 1)],
                     "std": [float(numeric[train, j].std()) for j in (0, 1)]}
    standardise_numeric(numeric, constants)
    codes = np.column_stack([
        fields["payment_format"], fields["payment_currency"],
        fields["receiving_currency"]]).astype(np.int32)
    rows = {name: np.flatnonzero(split == k)
            for name, k in (("train", 0), ("val", 1), ("test", 2))}
    return ts, payer, payee, numeric, codes, y, rows, n_nodes


def payment_graph(ts: np.ndarray, payer: np.ndarray, payee: np.ndarray,
                  n_nodes: int):
    """Augmented directed graph: every payment forward and reversed, sorted
    by time so temporal sampling can binary-search. Each edge carries its
    payment identifier and a reversed flag; the numeric and code features
    are gathered from the payment table per batch. Arrays are built in
    sorted order directly, one at a time, to keep the transient footprint
    near the graph's own size."""
    from torch_geometric.data import Data

    e = len(ts)
    order = np.argsort(np.concatenate([ts, ts]), kind="stable")
    forward = order < e
    pid = np.where(forward, order, order - e).astype(np.int32)
    edge_index = np.empty((2, 2 * e), np.int64)
    edge_index[0] = np.where(forward, payer[pid], payee[pid])
    edge_index[1] = np.where(forward, payee[pid], payer[pid])
    time = ts[pid]
    reverse = (~forward).astype(np.uint8)
    del order, forward
    return Data(num_nodes=n_nodes,
                edge_index=torch.from_numpy(edge_index),
                time=torch.from_numpy(time),
                edge_pid=torch.from_numpy(pid),
                edge_reverse=torch.from_numpy(reverse))


def n_code_levels(codes: np.ndarray) -> list[int]:
    return [int(codes[:, j].max()) + 1 for j in range(codes.shape[1])]


def prefetch(loader: Iterable, depth: int = PREFETCH_DEPTH) -> Iterator:
    """Iterate ``loader`` on a background thread, ``depth`` batches ahead."""
    items: queue.Queue = queue.Queue(maxsize=depth)
    end = object()

    def produce():
        try:
            for item in loader:
                items.put(item)
        except BaseException as exc:
            items.put(exc)
        items.put(end)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    while True:
        item = items.get()
        if item is end:
            break
        if isinstance(item, BaseException):
            thread.join()
            raise item
        yield item
    thread.join()


def _snapshot(model: nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _attach_edge_features(batch, numeric_t, codes_t):
    pid = batch.edge_pid.long()
    batch.edge_numeric = torch.cat(
        [numeric_t[pid], batch.edge_reverse.to(numeric_t.dtype).unsqueeze(1)],
        dim=1)
    batch.edge_codes = codes_t[pid]


def train_edge_scorer(data, payer, payee, ts, numeric, codes, y,
                      rows: dict[str, np.ndarray], cfg: GNNConfig, device,
                      log=print,
                      on_epoch: Callable[[list[dict]], None] | None = None) -> dict:
    """Early stopping on validation AP; returns the best epoch's validation
    probabilities and the test probabilities in ``rows`` order, the best
    state dictionary, and the epoch trace. ``rows`` maps split name to
    payment indices; seed query times are shifted to t - 1 so the sampled
    context is strictly earlier. ``on_epoch`` receives the trace after every
    epoch."""
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.sampler import NeighborSampler

    from m.stats import average_precision

    numeric_t = torch.from_numpy(numeric)
    codes_t = torch.from_numpy(codes.astype(np.int32, copy=False))
    y_t = torch.from_numpy(y.astype(np.float32))
    rows_t = {k: torch.from_numpy(v) for k, v in rows.items()}
    zero_col = {k: torch.zeros(1, 1) for k in rows}

    sampler = NeighborSampler(data, num_neighbors=list(cfg.fanout),
                              time_attr="time" if cfg.temporal else None)

    def loader(split, shuffle):
        r = rows[split]
        timing = ({"edge_label_time": torch.from_numpy(ts[r] - 1),
                   "time_attr": "time"} if cfg.temporal else {})
        return LinkNeighborLoader(
            data, num_neighbors=list(cfg.fanout), neighbor_sampler=sampler,
            edge_label_index=torch.from_numpy(np.stack([payer[r], payee[r]])),
            edge_label=torch.arange(len(r)),
            batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.loader_workers,
            persistent_workers=cfg.loader_workers > 0, **timing)

    loaders = {"train": loader("train", shuffle=True),
               "val": loader("val", shuffle=False),
               "test": loader("test", shuffle=False)}

    def attach(batch, split):
        r = rows_t[split][batch.edge_label.long()]
        batch.seed_numeric = torch.cat(
            [numeric_t[r], zero_col[split].expand(len(r), 1)], dim=1)
        batch.seed_codes = codes_t[r]
        _attach_edge_features(batch, numeric_t, codes_t)
        return r

    model = PaymentGNN(n_code_levels(codes), numeric.shape[1] + 1, cfg,
                       head="edge").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(cfg.pos_weight, device=device))

    @torch.no_grad()
    def infer(split):
        model.eval()
        probs = torch.empty(len(rows[split]))
        for batch in prefetch(loaders[split]):
            pos = batch.edge_label.long()
            attach(batch, split)
            probs[pos] = torch.sigmoid(model(batch.to(device))).float().cpu()
        return probs.numpy()

    best_ap, best_state, best_val_probs, since_best, trace = -1.0, None, None, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        total_loss = n_batches = 0
        for batch in prefetch(loaders["train"]):
            r = attach(batch, "train")
            target = y_t[r].to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch.to(device)), target)
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1
        val_probs = infer("val")
        val_ap = average_precision(y[rows["val"]], val_probs)
        trace.append({"epoch": epoch, "loss": total_loss / max(n_batches, 1),
                      "val_ap": val_ap})
        log(f"epoch {epoch}: loss {trace[-1]['loss']:.4f} val_ap {val_ap:.4f}")
        if val_ap > best_ap:
            best_ap, since_best = val_ap, 0
            best_state, best_val_probs = _snapshot(model), val_probs
        else:
            since_best += 1
        if on_epoch is not None:
            on_epoch(trace)
        if since_best >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.to(device)
    return {"val_probs": best_val_probs, "test_probs": infer("test"),
            "best_val_ap": best_ap, "best_state": best_state,
            "n_parameters": n_parameters(model), "trace": trace}


def train_node_scorer(data, numeric, codes, y_node: np.ndarray,
                      rows: dict[str, np.ndarray], cfg: GNNConfig, device,
                      log=print,
                      on_epoch: Callable[[list[dict]], None] | None = None) -> dict:
    """Node-level variant for the account-labelled world; the anchor
    protocol is transductive, so sampling here is not time-restricted."""
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.sampler import NeighborSampler

    from m.stats import average_precision

    numeric_t = torch.from_numpy(numeric)
    codes_t = torch.from_numpy(codes.astype(np.int32, copy=False))
    y_t = torch.from_numpy(y_node.astype(np.float32))
    sampler = NeighborSampler(data, num_neighbors=list(cfg.fanout))

    def loader(split, shuffle):
        return NeighborLoader(
            data, num_neighbors=list(cfg.fanout), neighbor_sampler=sampler,
            input_nodes=torch.from_numpy(rows[split]),
            batch_size=cfg.batch_size, shuffle=shuffle,
            num_workers=cfg.loader_workers,
            persistent_workers=cfg.loader_workers > 0)

    loaders = {"train": loader("train", shuffle=True),
               "val": loader("val", shuffle=False),
               "test": loader("test", shuffle=False)}

    model = PaymentGNN(n_code_levels(codes), numeric.shape[1] + 1, cfg,
                       head="node").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(cfg.pos_weight, device=device))

    @torch.no_grad()
    def infer(split):
        model.eval()
        out = np.empty(len(rows[split]), np.float32)
        cursor = 0
        for batch in prefetch(loaders[split]):
            _attach_edge_features(batch, numeric_t, codes_t)
            probs = torch.sigmoid(model(batch.to(device))).float().cpu().numpy()
            out[cursor:cursor + len(probs)] = probs
            cursor += len(probs)
        return out

    best_ap, best_state, best_val_probs, since_best, trace = -1.0, None, None, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        total_loss = n_batches = 0
        for batch in prefetch(loaders["train"]):
            _attach_edge_features(batch, numeric_t, codes_t)
            target = y_t[batch.n_id[:batch.batch_size]].to(device)
            opt.zero_grad()
            loss = loss_fn(model(batch.to(device)), target)
            loss.backward()
            opt.step()
            total_loss += float(loss)
            n_batches += 1
        val_probs = infer("val")
        val_ap = average_precision(y_node[rows["val"]], val_probs)
        trace.append({"epoch": epoch, "loss": total_loss / max(n_batches, 1),
                      "val_ap": val_ap})
        log(f"epoch {epoch}: loss {trace[-1]['loss']:.4f} val_ap {val_ap:.4f}")
        if val_ap > best_ap:
            best_ap, since_best = val_ap, 0
            best_state, best_val_probs = _snapshot(model), val_probs
        else:
            since_best += 1
        if on_epoch is not None:
            on_epoch(trace)
        if since_best >= cfg.patience:
            break
    model.load_state_dict(best_state)
    model.to(device)
    return {"val_probs": best_val_probs, "test_probs": infer("test"),
            "best_val_ap": best_ap, "best_state": best_state,
            "n_parameters": n_parameters(model), "trace": trace}
