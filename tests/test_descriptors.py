"""The descriptors, the rule machinery, and Experiment 3's rank statistics.

Every descriptor against a hand-computed value on a ledger small enough to
check by eye; the folds against their definition; the freeze rule
against each line it can be decided on; the partial rank correlation against
an independent route through the precision matrix; Benjamini-Hochberg against
scipy's; and the conditional permutation's size under a null it is meant to
survive. Neither torch nor LightGBM is imported here, as in the scripts.

Run:  uv run python -m unittest tests.test_descriptors -v
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from m import ceiling as lab  # noqa: E402
from m import descriptors as desc  # noqa: E402
from m import rank, rule  # noqa: E402


def toy() -> desc.Ledger:
    """Bank B holds accounts 0 and 1; accounts 2 and 3 sit at banks 10 and 11.

        row  payer  payee  own?            note
        0    0      1      both            internal
        1    0      2      payer           to bank 10
        2    2      0      payee           back from bank 10, so the pair reciprocates
        3    1      2      payer           to bank 10 again
        4    0      3      payer           to bank 11
    """
    return desc.Ledger(
        payer=np.array([0, 0, 2, 1, 0]),
        payee=np.array([1, 2, 0, 2, 3]),
        payer_is_own=np.array([True, True, False, True, True]),
        payee_is_own=np.array([True, False, True, False, False]),
        counterparty_bank=np.array([-1, 10, 10, 10, 11]),
        fmt=np.array([0, 0, 1, 1, 1]),
        pay_ccy=np.array([0, 0, 0, 0, 1]),
        recv_ccy=np.array([0, 0, 0, 0, 1]),
    )


class Descriptors(unittest.TestCase):
    def setUp(self) -> None:
        self.values = desc.compute(toy())

    def test_counts(self) -> None:
        self.assertEqual(self.values["payment_count"], 5.0)
        self.assertEqual(self.values["active_accounts"], 2.0)

    def test_internal_and_boundary(self) -> None:
        self.assertAlmostEqual(self.values["internal_share"], 1 / 5)
        # both own accounts appear on a cross-bank row
        self.assertAlmostEqual(self.values["boundary_share"], 1.0)

    def test_external_counterparties(self) -> None:
        # account 2 appears on rows 1, 2 and 3; account 3 on row 4
        self.assertAlmostEqual(self.values["external_encounters"], 2.0)
        self.assertAlmostEqual(self.values["external_recurrence"], 0.0)

    def test_external_banks_and_concentration(self) -> None:
        self.assertEqual(self.values["external_banks"], 2.0)
        self.assertAlmostEqual(self.values["external_hhi"], (3 / 4) ** 2 + (1 / 4) ** 2)

    def test_activity_and_reciprocity(self) -> None:
        # account 0 appears on rows 0, 1, 2, 4; account 1 on rows 0 and 3
        self.assertAlmostEqual(self.values["activity_gini"], desc.gini(np.array([4, 2])))
        # pairs (0,1) (0,2) (2,0) (1,2) (0,3); only (0,2) and (2,0) reciprocate
        self.assertAlmostEqual(self.values["reciprocity"], 2 / 5)

    def test_entropies(self) -> None:
        shares = np.array([2 / 5, 3 / 5])
        self.assertAlmostEqual(self.values["format_entropy"],
                               float(-(shares * np.log(shares)).sum()))
        currency = np.array([8 / 10, 2 / 10])
        self.assertAlmostEqual(self.values["currency_entropy"],
                               float(-(currency * np.log(currency)).sum()))

    def test_gini_bounds(self) -> None:
        self.assertEqual(desc.gini(np.array([7])), 0.0)
        self.assertEqual(desc.gini(np.array([3, 3, 3])), 0.0)
        self.assertGreater(desc.gini(np.array([0, 0, 0, 100])), 0.7)

    def test_transform_is_defined_at_zero(self) -> None:
        self.assertEqual(desc.transform(np.array([0.0]))[0], 0.0)
        self.assertAlmostEqual(desc.transform(np.array([1e6]))[0], np.log(1e6), places=5)

    def test_groups_are_complete_and_disjoint(self) -> None:
        grouped = [name for members in desc.GROUPS.values() for name in members]
        self.assertEqual(sorted(grouped), sorted(desc.NAMES))
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(set(desc.DESCRIPTIONS), set(desc.NAMES))


class Candidates(unittest.TestCase):
    def test_construction(self) -> None:
        built = rule.candidates(desc.GROUPS, desc.BASELINE)
        self.assertEqual(len(built), 16)
        self.assertEqual(built[0].name, desc.BASELINE)
        self.assertTrue(all(len(c.descriptors) <= 3 for c in built))
        self.assertEqual(len({c.name for c in built}), 16)

    def test_folds_cover_every_bank_once(self) -> None:
        ids = rule.fold_ids(107, repeat=3)
        self.assertEqual(len(ids), 107)
        self.assertEqual(sorted(set(ids)), list(range(rule.N_FOLDS)))
        self.assertTrue(all(10 <= (ids == f).sum() <= 11 for f in range(rule.N_FOLDS)))

    def test_folds_are_a_function_of_the_repeat(self) -> None:
        self.assertTrue(np.array_equal(rule.fold_ids(50, 2), rule.fold_ids(50, 2)))
        self.assertFalse(np.array_equal(rule.fold_ids(50, 2), rule.fold_ids(50, 3)))

    def test_weighted_fit_recovers_a_planted_rule(self) -> None:
        rng = np.random.default_rng(0)
        design = rng.normal(size=(200, 2))
        target = 0.5 + 2.0 * design[:, 0] - 1.0 * design[:, 1]
        intercept, coefficients = rule.fit(design, target, np.ones(200),
                                           np.ones(2, bool))
        self.assertAlmostEqual(intercept, 0.5)
        np.testing.assert_allclose(coefficients, [2.0, -1.0], atol=1e-10)

    def test_constant_column_is_dropped(self) -> None:
        design = np.column_stack([np.arange(20.0), np.ones(20)])
        keep = ~rule.constant_columns(design)
        self.assertEqual(list(keep), [True, False])
        _, coefficients = rule.fit(design, np.arange(20.0), np.ones(20), keep)
        self.assertEqual(coefficients[1], 0.0)


class Freeze(unittest.TestCase):
    """The baseline freezes unless something beats it outright; among the rules
    that do, parsimony and candidate order decide."""

    order = ["scale", "rich", "other", "richer"]
    size = {"scale": 2, "rich": 2, "other": 2, "richer": 3}
    none_beat = dict.fromkeys(order, False)
    all_beat = {"scale": False, "rich": True, "other": True, "richer": True}

    def freeze(self, score, paired, beats):
        return rule.freeze(self.order, score, paired, self.size, "scale", beats)

    def test_the_baseline_stands_when_nothing_beats_it(self) -> None:
        out = self.freeze({"scale": 0.50, "rich": 0.90, "other": 0.40, "richer": 0.80},
                          dict.fromkeys(self.order, 0.01), self.none_beat)
        self.assertEqual(out["winner"], "scale")
        self.assertEqual(out["decided_by"], "the baseline stands unbeaten")
        self.assertEqual(out["beaters"], [])

    def test_a_clear_winner_that_beats_the_baseline_wins_on_score(self) -> None:
        out = self.freeze({"scale": 0.50, "rich": 0.90, "other": 0.40, "richer": 0.45},
                          dict.fromkeys(self.order, 0.01), self.all_beat)
        self.assertEqual(out["winner"], "rich")
        self.assertEqual(out["decided_by"], "score")

    def test_parsimony_breaks_a_tie(self) -> None:
        out = self.freeze({"scale": 0.50, "rich": 0.51, "other": 0.30, "richer": 0.52},
                          {"scale": 0.0, "rich": 0.05, "other": 0.0, "richer": 0.0},
                          self.all_beat)
        self.assertEqual(out["tied"], ["rich", "richer"])
        self.assertEqual(out["winner"], "rich")
        self.assertEqual(out["decided_by"], "parsimony")

    def test_candidate_order_breaks_a_remaining_tie(self) -> None:
        out = self.freeze({"scale": 0.50, "rich": 0.51, "other": 0.515, "richer": 0.40},
                          {"scale": 0.0, "rich": 0.05, "other": 0.0, "richer": 0.0},
                          self.all_beat)
        self.assertEqual(out["tied"], ["rich", "other"])
        self.assertEqual(out["winner"], "rich")
        self.assertEqual(out["decided_by"], "candidate order")

    def test_the_guard_keeps_a_stably_closer_baseline(self) -> None:
        """The band is per candidate, so without the guard a noisier and
        lower-scoring candidate ties with the leader while the baseline, stably
        closer, does not."""
        score = {"scale": 0.620, "rich": 0.619, "other": 0.300, "richer": 0.644}
        paired = {"scale": 0.022, "rich": 0.030, "other": 0.0, "richer": 0.0}
        unguarded = rule.freeze_unguarded(self.order, score, paired, self.size)
        self.assertNotIn("scale", unguarded["tied"])
        self.assertIn("rich", unguarded["tied"])
        guarded = self.freeze(score, paired, self.none_beat)
        self.assertEqual(guarded["winner"], "scale")

    def test_the_marginal_rule_is_wider(self) -> None:
        score = {"scale": 0.50, "rich": 0.52, "other": 0.40, "richer": 0.40}
        paired = dict.fromkeys(self.order, 0.005)
        marginal = dict.fromkeys(self.order, 0.10)
        current = self.freeze(score, paired, self.all_beat)
        marginal_band = rule.freeze_marginal(self.order, score, marginal, self.size)
        self.assertEqual(current["winner"], "rich")
        self.assertEqual(marginal_band["winner"], "scale")


class RankStatistics(unittest.TestCase):
    def test_spearman_matches_scipy(self) -> None:
        from scipy.stats import spearmanr

        rng = np.random.default_rng(1)
        for _ in range(50):
            x, y = rng.normal(size=40), rng.normal(size=40)
            self.assertAlmostEqual(rank.spearman(x, y),
                                   float(spearmanr(x, y).statistic), places=12)

    def test_partial_matches_the_precision_matrix(self) -> None:
        from scipy.stats import rankdata

        rng = np.random.default_rng(2)
        for _ in range(50):
            n, q = int(rng.integers(15, 60)), int(rng.integers(1, 4))
            controls = rng.normal(size=(n, q))
            x = controls @ rng.normal(size=q) + rng.normal(size=n)
            y = controls @ rng.normal(size=q) + rng.normal(size=n)
            got = rank.partial_spearman(x, y, controls)["rho"]
            ranks = np.column_stack([rankdata(x), rankdata(y)]
                                    + [rankdata(c) for c in controls.T])
            precision = np.linalg.inv(np.corrcoef(ranks, rowvar=False))
            want = -precision[0, 1] / np.sqrt(precision[0, 0] * precision[1, 1])
            self.assertAlmostEqual(got, want, places=10)

    def test_partial_kendall_is_free_of_removal_order(self) -> None:
        rng = np.random.default_rng(3)
        for _ in range(30):
            n = int(rng.integers(20, 60))
            controls = rng.normal(size=(n, 2))
            x = controls @ [1.0, 0.5] + rng.normal(size=n)
            y = controls @ [0.5, 1.0] + rng.normal(size=n)
            forward = rank.partial_kendall(x, y, controls)
            backward = rank.partial_kendall(x, y, controls[:, ::-1])
            self.assertAlmostEqual(forward, backward, places=10)

    def test_benjamini_hochberg_matches_scipy(self) -> None:
        from scipy.stats import false_discovery_control

        rng = np.random.default_rng(4)
        for _ in range(200):
            p = rng.random(int(rng.integers(2, 50)))
            got = rank.benjamini_hochberg(p, 0.10)["declared"]
            want = false_discovery_control(p, method="bh") <= 0.10
            np.testing.assert_array_equal(got, want)

    def test_permutation_p_is_never_zero(self) -> None:
        x = np.arange(30.0)
        out = rank.permutation_p(x, x, alternative="greater", draws=199, seed=0)
        self.assertGreater(out["p"], 0.0)
        self.assertAlmostEqual(out["p"], 1 / 200)

    def test_an_undefined_correlation_is_not_evidence(self) -> None:
        """A variable with no variation, and one the controls explain exactly,
        both give NaN and a p-value of 1 — never the smallest p the permutation
        resolution allows."""
        z = np.arange(60.0)
        y = np.random.default_rng(8).normal(size=60)
        flat = rank.permutation_p(np.ones(60), y, draws=99, seed=0)
        self.assertTrue(np.isnan(flat["rho"]))
        self.assertEqual(flat["p"], 1.0)
        for x in (np.ones(60), z):
            out = rank.conditional_permutation_p(x, y, z, draws=99, seed=0)
            self.assertTrue(np.isnan(out["rho"]))
            self.assertEqual(out["p_conditional"], 1.0)
        self.assertTrue(np.isnan(rank.partial_spearman(y, 3 * z + 1, z)["rho"]))
        self.assertEqual(rank.permutation_p_difference(np.ones(60), y, z, draws=99,
                                                       seed=0), 1.0)

    def test_out_of_fold_never_sees_the_bank_it_predicts(self) -> None:
        rng = np.random.default_rng(9)
        design = rng.normal(size=(107, 3))
        target = rng.normal(size=107)
        weights = np.ones(107)
        before = rule.out_of_fold(design, target, weights)
        moved = target.copy()
        moved[5] += 100.0
        after = rule.out_of_fold(design, moved, weights)
        np.testing.assert_allclose(before[:, 5], after[:, 5], atol=1e-12)
        self.assertGreater(np.abs(before - after).max(), 1.0)

    def test_blocks_are_contiguous_in_the_control(self) -> None:
        control = np.random.default_rng(5).normal(size=100)
        blocks = rank.control_blocks(control, width=5)
        order = np.argsort(control)
        self.assertTrue(np.all(np.diff(blocks[order]) >= 0))
        self.assertEqual(len(np.unique(blocks)), 20)

    def test_conditional_permutation_holds_its_size(self) -> None:
        """Under a null where the two variables are independent given the control
        but both driven by it through a hostile monotone transform, the
        conditional test keeps its level where the t-approximation does not."""
        rng = np.random.default_rng(6)
        conditional, asymptotic = 0, 0
        trials = 200
        for _ in range(trials):
            z = rng.normal(size=107)
            driver = np.exp(3.0 * z)
            x = driver + rng.normal(0, 0.5 * driver.std(), 107)
            y = driver + rng.normal(0, 0.5 * driver.std(), 107)
            out = rank.conditional_permutation_p(x, y, z, draws=299,
                                                 seed=int(rng.integers(1 << 30)))
            conditional += out["p_conditional"] <= 0.10
            asymptotic += out["p_asymptotic"] <= 0.10
        self.assertLess(conditional / trials, 0.20)
        self.assertGreater(asymptotic / trials, conditional / trials)


class Neighbourhood(unittest.TestCase):
    def test_counts_against_brute_force(self) -> None:
        rng = np.random.default_rng(7)
        payer = rng.integers(0, 12, 60)
        payee = rng.integers(0, 12, 60)
        keep = payer != payee
        payer, payee = payer[keep], payee[keep]
        y = (rng.random(len(payer)) < 0.3).astype(np.float64)
        total, laundering = lab.neighbourhood(payer, payee, y)
        for i in range(len(payer)):
            touching = {j for j in range(len(payer)) if j != i
                        and {payer[j], payee[j]} & {payer[i], payee[i]}}
            self.assertEqual(total[i], len(touching))
            self.assertEqual(laundering[i], sum(y[j] for j in touching))

    def test_entropy_of_one_family_is_zero(self) -> None:
        self.assertEqual(lab.entropy(np.zeros(9, np.int64), 8), 0.0)
        self.assertAlmostEqual(lab.entropy(np.array([0, 1]), 8), np.log(2))


if __name__ == "__main__":
    unittest.main()
