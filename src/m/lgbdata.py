"""The tree family's LightGBM canon.

One discretisation for every table: a Dataset is constructed eagerly with
``DS_PARAMS`` and LightGBM's construction defaults, validation sets by
``reference=``, before any ``lgb.train``. The eight frozen settings train
from the constructed Dataset; ``num_threads`` is an execution parameter
passed beside the frozen dictionary, never inside it.

The row rule's copies table for a global-sight set-up is the pooled
global-sight table with each payment's row repeated once per endpoint bank.
``RepeatedRowsSequence`` streams exactly that table from the pooled parquet
cache — sampled rows for the bin mappers by random access, then the rows in
order in batches — so the Medium datasets' 60 M-row tables never exist in
memory. LightGBM samples the same row indices whether the data arrives as a
matrix or a Sequence (both call the library's seeded sampler), and bins the
same values, so the two constructions agree bit for bit; the streamed gate
in Experiment 1 asserts it on HI-Small.
"""

from __future__ import annotations

import numbers
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

DS_PARAMS = {"max_bin": 63}
SEQUENCE_VERSION = 1


def repeat_index(pooled_rows: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """``starts[k]`` = copies before pooled row ``k``; length len + 1."""
    starts = np.zeros(len(pooled_rows) + 1, np.int64)
    np.cumsum(counts, out=starts[1:])
    return starts


class RowGroupReader:
    """Feature columns of a parquet table by row group, one group cached."""

    def __init__(self, path: Path, columns: list[str]):
        self.file = pq.ParquetFile(path)
        self.columns = columns
        meta = self.file.metadata
        sizes = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        self.bounds = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        self._cached = -1
        self._matrix: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.bounds[-1])

    def group(self, g: int) -> np.ndarray:
        if g != self._cached:
            table = self.file.read_row_group(g, columns=self.columns)
            matrix = np.empty((table.num_rows, len(self.columns)), np.float32)
            for j, c in enumerate(self.columns):
                matrix[:, j] = table.column(c).to_numpy()
            self._matrix = matrix
            self._cached = g
        return self._matrix

    def release(self) -> None:
        self._matrix, self._cached = None, -1

    def rows(self, indices: np.ndarray) -> np.ndarray:
        """Feature rows for increasing table row indices, float32."""
        out = np.empty((len(indices), len(self.columns)), np.float32)
        if len(indices) == 0:
            return out
        g0 = int(np.searchsorted(self.bounds, indices[0], "right") - 1)
        g1 = int(np.searchsorted(self.bounds, indices[-1], "right") - 1)
        filled = 0
        for g in range(g0, g1 + 1):
            lo, hi = self.bounds[g], self.bounds[g + 1]
            sel = indices[(indices >= lo) & (indices < hi)]
            out[filled:filled + len(sel)] = self.group(g)[sel - lo]
            filled += len(sel)
        return out

    def batches(self, rows_mask: np.ndarray):
        """Yield (table row indices, feature rows) per row group for the
        rows where ``rows_mask`` holds."""
        for g in range(len(self.bounds) - 1):
            lo, hi = int(self.bounds[g]), int(self.bounds[g + 1])
            local = np.flatnonzero(rows_mask[lo:hi])
            if len(local):
                yield local + lo, self.group(g)[local]


class RepeatedRowsSequence(lgb.Sequence):
    """The rows ``pooled_rows`` of a parquet table, each repeated
    ``counts[k]`` times, in order — the copies table of one split."""

    batch_size = 500_000

    def __init__(self, reader: RowGroupReader, pooled_rows: np.ndarray,
                 counts: np.ndarray):
        self.reader = reader
        self.pooled_rows = pooled_rows
        self.counts = counts
        self.starts = repeat_index(pooled_rows, counts)

    def __len__(self) -> int:
        return int(self.starts[-1])

    def _range(self, a: int, b: int) -> np.ndarray:
        k0 = int(np.searchsorted(self.starts, a, "right") - 1)
        k1 = int(np.searchsorted(self.starts, b - 1, "right") - 1)
        block = self.reader.rows(self.pooled_rows[k0:k1 + 1])
        expanded = np.repeat(block, self.counts[k0:k1 + 1], axis=0)
        offset = a - int(self.starts[k0])
        return expanded[offset:offset + (b - a)]

    def __getitem__(self, idx):
        if isinstance(idx, numbers.Integral):
            return self._range(int(idx), int(idx) + 1)[0].astype(np.float64)
        if isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            return self._range(start, stop)
        raise TypeError(f"unsupported index {type(idx).__name__}")


def construct(data, label: np.ndarray, categorical: list[int],
              reference: lgb.Dataset | None = None,
              num_threads: int | None = None) -> lgb.Dataset:
    """A Dataset under the canon, constructed now; ``num_threads`` bounds the
    construction's OpenMP threads and touches no binning parameter."""
    params = dict(DS_PARAMS)
    if num_threads is not None:
        params["num_threads"] = num_threads
    ds = lgb.Dataset(data, label=label, categorical_feature=categorical,
                     params=params, reference=reference, free_raw_data=True)
    ds.construct()
    return ds


def fit_grid(dtr: lgb.Dataset, dva: lgb.Dataset, grid: list[dict], fixed: dict,
             num_round: int, early_stop: int, num_threads: int,
             progress=None) -> list[lgb.Booster]:
    boosters = []
    settings = grid if progress is None else progress(grid)
    for cfg in settings:
        boosters.append(lgb.train(
            {**fixed, **cfg, "num_threads": num_threads}, dtr,
            num_boost_round=num_round, valid_sets=[dva],
            callbacks=[lgb.early_stopping(early_stop, verbose=False)]))
    return boosters


def setting_records(boosters: list[lgb.Booster], grid: list[dict]) -> list[dict]:
    return [{"params": cfg, "best_iteration": int(b.best_iteration),
             "val_ap": float(b.best_score["valid_0"]["average_precision"])}
            for b, cfg in zip(boosters, grid)]


def predict_matrix(boosters: list[lgb.Booster], X: np.ndarray) -> np.ndarray:
    """(settings, rows) float32 scores of one feature matrix."""
    return np.stack([b.predict(X, num_iteration=b.best_iteration)
                     for b in boosters]).astype(np.float32)


def predict_rows(boosters: list[lgb.Booster], reader: RowGroupReader,
                 rows_mask: np.ndarray) -> np.ndarray:
    """(settings, rows) float32 scores of the masked table rows, in table
    order, one row group at a time."""
    n = int(rows_mask.sum())
    out = np.empty((len(boosters), n), np.float32)
    filled = 0
    for _, X in reader.batches(rows_mask):
        out[:, filled:filled + len(X)] = predict_matrix(boosters, X)
        filled += len(X)
    return out
