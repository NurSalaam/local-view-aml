"""The derived numbers of facts.py on toy values: the census table's ranges and
laundering rates, the payment rail and the qualifying banks, the median bank,
and the share of the shortfall that borrowing recovers.

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
import numpy as np  # noqa: E402
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


def toy_ledger() -> pd.DataFrame:
    """Eighty payments among banks 001 and 002 and the rail 070: 35 laundering
    payments within 001 and 5 from 002 to the rail in the test window; 20
    laundering payments within 001 and 10 card payments from the rail to 002
    in the training window; 10 cash payments from the rail to 001 in the test
    window."""
    blocks = [(35, "001", "a1", "001", "a2", 2, 1, "Wire"),
              (5, "002", "b1", "070", "r2", 2, 1, "ACH"),
              (20, "001", "a1", "001", "a2", 0, 1, "Wire"),
              (10, "070", "r1", "002", "b1", 0, 0, "Cheque"),
              (10, "070", "r1", "001", "a1", 2, 0, "Cash")]
    rows = [row[1:] for row in blocks for _ in range(row[0])]
    df = pd.DataFrame(rows, columns=["from_bank", "from_acct", "to_bank", "to_acct",
                                     "split", "is_laundering", "fmt"])
    banks = pd.CategoricalDtype(["001", "002", "070"])
    return df.astype({"from_bank": banks, "to_bank": banks, "fmt": "category",
                      "split": np.int8, "is_laundering": np.int8})


class TestQualifyingAndRail(unittest.TestCase):
    def setUp(self):
        self.df = toy_ledger()
        self.counts = facts.ledger_counts(self.df)

    def test_qualifying_banks_coverage_and_selection(self):
        """Only 001 holds thirty test-window laundering payments. It touches
        65 of the 80 payments and 55 of the 60 laundering ones; in the
        training window 20 of 30 payments and all 20 laundering ones. Its
        training window holds 20 laundering payments for 35 in its test
        window, against 20 for 40 over 001 and 002. Bank 002 holds laundering
        in its test window and no confirmed case."""
        out = facts.qualifying_facts(self.counts)
        self.assertEqual(out["banks_by_threshold"], {"10": 1, "20": 1, "30": 1, "50": 0})
        self.assertAlmostEqual(out["coverage"]["all_windows"]["payments"], 65 / 80)
        self.assertAlmostEqual(out["coverage"]["all_windows"]["laundering_payments"], 55 / 60)
        self.assertAlmostEqual(out["coverage"]["train"]["payments"], 20 / 30)
        self.assertAlmostEqual(out["coverage"]["train"]["laundering_payments"], 1.0)
        self.assertAlmostEqual(out["train_over_test"]["qualifying"], 20 / 35)
        self.assertAlmostEqual(out["train_over_test"]["all_non_rail"], 0.5)
        self.assertEqual(out["no_confirmed_case"], 1)

    def test_rail_profile(self):
        """The rail touches 25 payments, 5 of them laundering; 20 go out, half
        by cheque and half in cash; it holds two accounts. 001 touches 65
        payments, and the rail's 5 test-window laundering payments rank second
        to 001's 35. Of 001's 45 test-window payments 10 touch the rail, and
        none of its laundering ones."""
        out = facts.rail_facts(self.df, self.counts)
        self.assertEqual(out["accounts"], 2)
        self.assertAlmostEqual(out["share_of_payments"], 25 / 80)
        self.assertAlmostEqual(out["share_of_laundering_payments"], 5 / 60)
        self.assertEqual(out["between_rail_accounts"], 0)
        self.assertAlmostEqual(out["outgoing_share"], 0.8)
        self.assertEqual(out["outgoing_formats"], {"Cash": 0.5, "Cheque": 0.5})
        self.assertAlmostEqual(out["payments_over_next_largest"], 25 / 65)
        self.assertEqual((out["test_laundering_payments"],
                          out["next_bank_test_laundering_payments"],
                          out["rank_by_test_laundering"]), (5, 35, 2))
        self.assertAlmostEqual(out["qualifying_test_sets"]["share_of_payments"], 10 / 45)
        self.assertAlmostEqual(out["qualifying_test_sets"]["share_of_laundering"], 0.0)


class TestMedianBank(unittest.TestCase):
    def test_lower_median_ties_by_identifier(self):
        roster = pd.DataFrame({"train_rows": [5, 3, 9, 3]},
                              index=pd.Index(["d", "a", "c", "b"], name="bank"))
        self.assertEqual(facts.median_bank(roster), "b")


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


class TestQuarterIntervals(unittest.TestCase):
    def test_first_quarter_matches_the_policy_runs_own_draw(self):
        """The quarters draw their seed noise from one stream; the first
        quarter draws first, so its interval is the one the policy Run draws
        from a fresh generator for its primary reading."""
        import exp4_analysis

        rng = np.random.default_rng(3)
        labels = [f"b{i}" for i in range(8)]
        banks = pd.DataFrame({"quartile": [1, 1, 2, 2, 3, 3, 4, 4],
                              "weight_se_exp1": rng.uniform(0.05, 0.2, 8),
                              "seed_var": rng.uniform(0.0, 0.01, 8)},
                             index=pd.Index(labels, name="bank"))
        reps = rng.normal(0.2, 0.05, (200, 8))
        q1 = labels[:2]
        est = {b: {"policy": {"delta": 0.0, "delta_reps": reps[:, i],
                              "seed_var": banks.loc[b, "seed_var"]}}
               for i, b in enumerate(labels)}
        weights = 1.0 / banks["weight_se_exp1"].to_numpy() ** 2
        primary = exp4_analysis.weighted_mean_reading(q1, est, weights[:2], "gnn")
        every = exp4_analysis.weighted_mean_reading(labels, est, weights, "gnn")
        out = facts.quarter_intervals(banks, reps, "gnn",
                                      {"primary": primary, "all_banks": every})
        np.testing.assert_allclose(out["1"], primary["ci95"], rtol=0, atol=1e-12)
        self.assertEqual(out["checks"]["quartile_1_vs_primary"], 0.0)


if __name__ == "__main__":
    unittest.main()
