"""The counterparty history behind the never-seen-counterparty cue.

The vectorised earlier-appearance count against a brute-force definition and
against an independent reference implementation, the count over both sides
of an account, and the choice of the counterparty's side. Neither torch nor
LightGBM is imported here, as in the probe itself.

Run:  uv run python -m unittest tests.test_frame -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from m.frame import earlier_counts, endpoint_history, external_history  # noqa: E402


def reference_earlier_counts(keys: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Per key block, the index of the first row sharing the queried
    timestamp."""
    order = np.lexsort((ts, keys))
    k, t = keys[order], ts[order]
    starts = np.flatnonzero(np.concatenate([[True], k[1:] != k[:-1]]))
    first_equal = np.empty(len(k), np.int64)
    for s, e in zip(starts, np.concatenate([starts[1:], [len(k)]])):
        first_equal[s:e] = np.searchsorted(t[s:e], t[s:e], "left")
    out = np.empty(len(k), np.int64)
    out[order] = first_equal
    return out


def brute_force_counts(rows: list[tuple], ts: np.ndarray) -> np.ndarray:
    return np.array([sum(1 for j in range(len(rows))
                         if rows[j] == rows[i] and ts[j] < ts[i])
                     for i in range(len(rows))], np.int64)


class TestCounterpartyHistory(unittest.TestCase):
    def test_matches_the_reference_over_seeded_random_frames(self):
        for seed in range(20):
            rng = np.random.default_rng(seed)
            n = int(rng.integers(1, 400))
            keys = rng.integers(0, max(2, n // 5), n).astype(np.int64)
            ts = np.sort(rng.integers(0, max(2, n // 3), n)).astype(np.int64)
            np.testing.assert_array_equal(earlier_counts((keys,), ts),
                                          reference_earlier_counts(keys, ts))

    def test_counts_strictly_earlier_rows_with_the_same_key(self):
        rng = np.random.default_rng(7)
        keys = rng.integers(0, 4, 60).astype(np.int64)
        ts = np.sort(rng.integers(0, 8, 60)).astype(np.int64)
        np.testing.assert_array_equal(
            earlier_counts((keys,), ts),
            brute_force_counts([(int(k),) for k in keys], ts))

    def test_ties_count_as_not_earlier(self):
        keys = np.array([1, 1, 1, 2], np.int64)
        ts = np.array([5, 5, 9, 5], np.int64)
        np.testing.assert_array_equal(earlier_counts((keys,), ts),
                                      np.array([0, 0, 2, 0]))

    def test_the_input_order_need_not_be_sorted_by_time(self):
        keys = np.array([1, 1, 1], np.int64)
        ts = np.array([9, 1, 5], np.int64)
        np.testing.assert_array_equal(earlier_counts((keys,), ts),
                                      np.array([2, 0, 1]))

    def test_endpoint_history_counts_both_sides_of_an_account(self):
        payer = np.array([0, 1, 0, 2], np.int64)
        payee = np.array([1, 0, 2, 0], np.int64)
        ts = np.array([1, 2, 3, 4], np.int64)
        payer_hist, payee_hist = endpoint_history(payer, payee, ts)
        # account 0 appears at t = 1, 2, 3, 4; account 1 at 1, 2; account 2 at 3, 4
        np.testing.assert_array_equal(payer_hist, np.array([0, 1, 2, 1]))
        np.testing.assert_array_equal(payee_hist, np.array([0, 1, 0, 3]))

    def test_external_history_takes_the_side_that_is_not_the_bank_s(self):
        payer_hist = np.array([3, 7], np.int64)
        payee_hist = np.array([5, 2], np.int64)
        payer_is_own = np.array([True, False])
        np.testing.assert_array_equal(
            external_history(payer_hist, payee_hist, payer_is_own),
            np.array([5, 7]))


if __name__ == "__main__":
    unittest.main()
