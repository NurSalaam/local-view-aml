"""The frame probe's rules.

The vectorised earlier-appearance count against a brute-force definition and
against an independent reference implementation, the ordered-pair count, and
the three frame-invariance identities the probe asserts at run time.
Neither torch nor LightGBM is imported here, as in the probe itself.

Run:  uv run python -m unittest tests.test_frame -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from m.frame import (earlier_counts, endpoint_history, external_history,  # noqa: E402
                     pair_history, rule_scores)


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


class TestEarlierCounts(unittest.TestCase):
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

    def test_pair_history_counts_the_ordered_pair(self):
        rng = np.random.default_rng(3)
        payer = rng.integers(0, 5, 80).astype(np.int64)
        payee = rng.integers(0, 5, 80).astype(np.int64)
        ts = np.sort(rng.integers(0, 10, 80)).astype(np.int64)
        np.testing.assert_array_equal(
            pair_history(payer, payee, ts),
            brute_force_counts(list(zip(payer.tolist(), payee.tolist())), ts))

    def test_endpoint_history_counts_both_sides_of_an_account(self):
        payer = np.array([0, 1, 0, 2], np.int64)
        payee = np.array([1, 0, 2, 0], np.int64)
        ts = np.array([1, 2, 3, 4], np.int64)
        payer_hist, payee_hist = endpoint_history(payer, payee, ts)
        # account 0 appears at t = 1, 2, 3, 4; account 1 at 1, 2; account 2 at 3, 4
        np.testing.assert_array_equal(payer_hist, np.array([0, 1, 2, 1]))
        np.testing.assert_array_equal(payee_hist, np.array([0, 1, 0, 3]))


class TestFrameInvariance(unittest.TestCase):
    """A bank's own endpoint, an ordered pair with one endpoint at the bank,
    and the payment's own amount read the same in the bank's ledger as on the
    global frame. The external endpoint need not."""

    def setUp(self):
        rng = np.random.default_rng(11)
        n = 4000
        self.accounts = 60
        self.bank_of = rng.integers(0, 6, self.accounts)
        payer = rng.integers(0, self.accounts, n)
        payee = rng.integers(0, self.accounts, n)
        keep = payer != payee
        self.payer, self.payee = payer[keep].astype(np.int64), payee[keep].astype(np.int64)
        self.ts = np.sort(rng.integers(0, 500, keep.sum())).astype(np.int64)
        self.amount = rng.random(keep.sum())

    def _scores(self, idx: np.ndarray, bank: int, hist: tuple) -> dict:
        payer_hist, payee_hist, pair = hist
        return rule_scores(payer_hist, payee_hist, pair,
                           self.bank_of[self.payer[idx]] == bank, self.amount[idx])

    def test_own_pair_and_amount_agree_between_the_two_frames(self):
        payer_hist, payee_hist = endpoint_history(self.payer, self.payee, self.ts)
        pair = pair_history(self.payer, self.payee, self.ts)
        differing = 0
        for bank in range(6):
            touches = ((self.bank_of[self.payer] == bank)
                       | (self.bank_of[self.payee] == bank))
            idx = np.flatnonzero(touches)
            ledger_payer, ledger_payee = endpoint_history(
                self.payer[idx], self.payee[idx], self.ts[idx])
            ledger_pair = pair_history(self.payer[idx], self.payee[idx], self.ts[idx])
            ledger = self._scores(idx, bank, (ledger_payer, ledger_payee, ledger_pair))
            globally = self._scores(idx, bank, (payer_hist[idx], payee_hist[idx], pair[idx]))
            for rule in ("own", "pair", "amount"):
                np.testing.assert_array_equal(ledger[rule], globally[rule],
                                              err_msg=f"bank {bank}, rule {rule}")
            differing += int((ledger["external"] != globally["external"]).sum())
        self.assertGreater(differing, 0,
                           "the external rule must be able to differ between frames")

    def test_external_history_takes_the_side_that_is_not_the_bank_s(self):
        payer_hist = np.array([3, 7], np.int64)
        payee_hist = np.array([5, 2], np.int64)
        payer_is_own = np.array([True, False])
        np.testing.assert_array_equal(
            external_history(payer_hist, payee_hist, payer_is_own),
            np.array([5, 7]))


if __name__ == "__main__":
    unittest.main()
