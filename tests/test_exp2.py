"""Experiment 2 machinery on toy ledgers: the split of a bank's test set by
whether the counterparty appears in its ledger before the test window, and
the case reduction's nested draws.

Run:  uv run python -m unittest tests.test_exp2 -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("M_DEVICE", "cpu")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402, F401  (before lightgbm: OpenMP runtimes, see scripts/exp1.py)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import exp2  # noqa: E402
import exp2_examples  # noqa: E402


def toy_ledger() -> pd.DataFrame:
    """Bank A's ledger. Row 0 is pre-test; rows 1 to 4 are test rows: a
    counterparty seen before the test window, one first met on row 2 and
    met again on row 3, and a payment between two of A's own accounts."""
    return pd.DataFrame({
        "from_bank": ["A", "A", "A", "C", "A"],
        "from_acct": ["1", "1", "2", "9", "1"],
        "to_bank": ["B", "B", "C", "A", "A"],
        "to_acct": ["7", "7", "9", "2", "2"],
        "split": [0, 2, 2, 2, 2],
        "payment": [10, 11, 12, 13, 14],
    })


class TestCounterpartyView(unittest.TestCase):
    def test_first_part_of_the_test_set(self):
        exp2_examples.counterparty_view.cache_clear()
        with mock.patch.object(exp2_examples.data, "read_ledger",
                               return_value=toy_ledger()):
            view = exp2_examples.counterparty_view("A")
        exp2_examples.counterparty_view.cache_clear()
        np.testing.assert_array_equal(view["test"], [False, True, True, True, True])
        np.testing.assert_array_equal(view["never"], [False, False, True, True, False])
        np.testing.assert_array_equal(view["payment"], [10, 11, 12, 13, 14])


class TestCaseReduction(unittest.TestCase):
    def test_draws_are_nested_across_levels(self):
        rng = np.random.default_rng(0)
        n = 400
        ledger = pd.DataFrame({
            "is_laundering": (rng.random(n) < 0.2).astype(np.int8),
            "split": rng.choice([0, 1, 2], n, p=[0.6, 0.2, 0.2]).astype(np.int8)})
        positives = np.flatnonzero((ledger["split"] == 0) & (ledger["is_laundering"] == 1))
        for draw in range(exp2.DIAL_DRAWS):
            kept = []
            for level in exp2.dial_levels_for(len(positives)):
                k, r = exp2.dial_subsample(ledger, 7, level, draw)
                self.assertEqual(len(k), level)
                self.assertEqual(sorted(np.concatenate([k, r])), list(positives))
                kept.append(set(k))
            for small, big in zip(kept, kept[1:]):
                self.assertLessEqual(small, big)


if __name__ == "__main__":
    unittest.main()
