"""The derived numbers of facts.py on toy values: the census table's ranges and
laundering rates, and the share of the shortfall that borrowing recovers.

Run:  uv run python -m unittest tests.test_facts -v
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
import pandas as pd  # noqa: E402

import facts  # noqa: E402


class TestCensusTable(unittest.TestCase):
    def test_ledger_and_confirmed_case_ranges(self):
        roster = pd.DataFrame({"touching": [9_610, 226_188, 50_000],
                               "train_pos": [4, 265, 40]},
                              index=pd.Index(["a", "b", "c"], name="bank"))
        self.assertEqual(facts.ledger_facts(roster),
                         {"banks": 3, "ledger_payments_range": [9_610, 226_188],
                          "train_pos_range": [4, 265]})

    def test_laundering_rate(self):
        out = facts.census_facts(
            {"HI-Small": {"n_rows": 981_000, "n_laundering": 1_000}},
            {"n_rows": 6_530, "n_laundering": 10},
            {"n_real_transfers": 2_200, "n_sar_transfers": 100})
        self.assertEqual(out["laundering_one_in"],
                         {"HI-Small": 981.0, "transxion": 653.0,
                          facts.data.AMLGENTEX_DATASET: 22.0})


class TestRecovery(unittest.TestCase):
    def test_shares_are_weighted_gain_over_weighted_yardstick(self):
        """Banks a and b form the first quarter, with weights 1/0.1^2 = 100
        and 1/0.2^2 = 25. The weighted gain is (100 x 0.5 + 25 x 0.1)/125 =
        0.42; the weighted shortfall (100 x 0.6 + 25 x 0.2)/125 = 0.52; the
        weighted value of training examples (100 x 0.7 + 25 x 0.3)/125 = 0.62.
        Bank c, in the second quarter, changes nothing."""
        index = pd.Index(["a", "b", "c"], name="bank")
        policy = pd.DataFrame({"quartile": [1, 1, 2], "train_pos": [4, 22, 30],
                               "weight_se_exp1": [0.1, 0.2, 0.05],
                               "policy_gain": [0.5, 0.1, 0.9],
                               "delta_exp1": [0.6, 0.2, -0.3]}, index=index)
        components = pd.DataFrame({"examples_delta": [0.7, 0.3, 0.0]}, index=index)
        out = facts.recovery_facts(policy, components,
                                   {"weighted_mean": 0.42,
                                    "share_of_shortfall_recovered": 0.42 / 0.52})
        self.assertEqual((out["banks"], out["count_range"]), (2, [4, 22]))
        self.assertAlmostEqual(out["gain"]["weighted"], 0.42, places=12)
        self.assertAlmostEqual(out["shortfall"]["weighted_mean"], 0.52, places=12)
        self.assertAlmostEqual(out["shortfall"]["share_weighted"], 0.42 / 0.52, places=12)
        self.assertAlmostEqual(out["examples"]["share_weighted"], 0.42 / 0.62, places=12)
        self.assertAlmostEqual(out["checks"]["gain_vs_verdict_run"], 0.0, places=12)
        self.assertAlmostEqual(out["checks"]["share_vs_verdict_run"], 0.0, places=12)


if __name__ == "__main__":
    unittest.main()
