"""Experiment 4 machinery on toy values: the choice and the abstain rule at one
bank, and the independence of each reading's seed-noise draws.

Run:  uv run python -m unittest tests.test_exp4 -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("M_DEVICE", "cpu")
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import torch  # noqa: E402, F401  (before lightgbm: OpenMP runtimes, see scripts/exp1.py)
import numpy as np  # noqa: E402

import exp4_analysis  # noqa: E402


class TestChoice(unittest.TestCase):
    def test_borrows_the_lender_with_the_highest_validation_score(self):
        self.assertEqual(
            exp4_analysis.choose({"020": 0.3, "000": 0.6, "011": 0.5}, 0.4),
            ("000", True))

    def test_abstains_when_its_own_detector_scores_as_high(self):
        self.assertEqual(exp4_analysis.choose({"020": 0.3, "000": 0.6}, 0.6),
                         ("000", False))
        self.assertEqual(exp4_analysis.choose({"020": 0.3, "000": 0.6}, 0.7),
                         ("000", False))

    def test_a_tie_between_lenders_goes_to_the_first(self):
        self.assertEqual(exp4_analysis.choose({"020": 0.6, "000": 0.6}, 0.1),
                         ("020", True))


class TestReadings(unittest.TestCase):
    def test_a_reading_does_not_depend_on_the_readings_before_it(self):
        rng = np.random.default_rng(1)
        banks = [f"b{i}" for i in range(8)]
        est = {b: {"policy": {"delta": float(rng.normal()),
                              "delta_reps": rng.normal(size=exp4_analysis.REPLICATES),
                              "seed_var": float(rng.random())}}
               for b in banks}
        weights = rng.random(len(banks)) + 0.5
        alone = exp4_analysis.weighted_mean_reading(banks, est, weights, "gnn")
        exp4_analysis.weighted_mean_reading(banks[:3], est, weights[:3], "gnn")
        after = exp4_analysis.weighted_mean_reading(banks, est, weights, "gnn")
        self.assertEqual(alone, after)


if __name__ == "__main__":
    unittest.main()
