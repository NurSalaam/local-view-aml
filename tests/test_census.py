"""The census statistics on a hand-worked example of five payments.

Run:  uv run python -m unittest tests.test_census -v
"""

from __future__ import annotations

import unittest

import numpy as np

from m.census import either_endpoint_counts, within_bank

FROM_BANK = np.array([0, 0, 1, 2, 2], np.int64)
TO_BANK = np.array([0, 1, 1, 0, 2], np.int64)
IS_LAUNDERING = np.array([0, 0, 0, 0, 1], np.int8)


class TestCensus(unittest.TestCase):
    def test_a_payment_within_one_bank_counts_once(self):
        np.testing.assert_array_equal(
            either_endpoint_counts(FROM_BANK, TO_BANK, np.ones(5, bool), 3),
            np.array([3, 2, 2]))

    def test_within_bank_share_against_the_sum_of_squared_endpoint_shares(self):
        # Normal payments: 0-0, 0-1, 1-1 and 2-0, so two of four join one bank.
        # Their eight endpoints lie 4, 3 and 1 at banks 0, 1 and 2, so a random
        # assignment joins one bank with chance 0.5^2 + 0.375^2 + 0.125^2.
        licit = within_bank(FROM_BANK, TO_BANK, IS_LAUNDERING)["licit"]
        self.assertEqual(licit["within_share"], 0.5)
        self.assertAlmostEqual(licit["chance"], 0.40625)
        self.assertAlmostEqual(licit["over_chance"], 0.5 / 0.40625)

    def test_each_class_reads_its_own_endpoints(self):
        laundering = within_bank(FROM_BANK, TO_BANK, IS_LAUNDERING)["laundering"]
        self.assertEqual(laundering, {"within_share": 1.0, "chance": 1.0,
                                      "over_chance": 1.0})

    def test_bank_codes_need_not_be_contiguous(self):
        sparse = within_bank(FROM_BANK * 7 + 3, TO_BANK * 7 + 3, IS_LAUNDERING)
        self.assertEqual(sparse, within_bank(FROM_BANK, TO_BANK, IS_LAUNDERING))


if __name__ == "__main__":
    unittest.main()
