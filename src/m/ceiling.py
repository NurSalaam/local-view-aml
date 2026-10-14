"""The label-using quantities behind Experiment 3's ceiling.

Experiment 3's label-free rule is what a bank could actually compute. The ceiling is what
a predictor could reach if it were handed the labels and the whole network, and
the distance between them is the price of label-freeness. Six quantities in
three groups: how much evidence a bank's laundering leaves (`laundering_count`,
`test_base_rate`), where the generator put it (`laundering_internal_share`,
`chain_pattern_share`), and what surrounds it (`class_homophily`,
`pattern_entropy`).

Data. The whole dataset's training window arrives as parallel arrays in
canonical order — the two endpoints' account codes over one (bank, account)
dictionary, the label, and the roster membership of each endpoint. Two payments
are **adjacent** when they share an account, so a payment's neighbourhood is
read off endpoint degrees by inclusion and exclusion rather than by building any
neighbour list; that is what keeps a 19-million-row training window a matter of
seconds.

The payment rail's accounts are excluded from the neighbourhood counts. The bank
roster already excludes AMLworld's bank 070 — fifteen hyperactive
accounts with no internal payments, a payment rail in all but name — and its
accounts hold 5.06 % of every training endpoint slot, the largest at a degree of
619,047 against a global mean of 20.0. Left in, `class_homophily` ranks banks
by their exposure to those fifteen accounts instead of by class mixing. The
rail-included reading is computed beside it, as is the mean-of-ratios form,
which weights a payment's neighbourhood by its size not at all.
"""

from __future__ import annotations

import numpy as np

VERSION = 1

NAMES = ("laundering_count", "test_base_rate", "laundering_internal_share",
         "chain_pattern_share", "class_homophily", "pattern_entropy")

GROUPS = {
    "evidence": ("laundering_count", "test_base_rate"),
    "placement": ("laundering_internal_share", "chain_pattern_share"),
    "neighbourhood": ("class_homophily", "pattern_entropy"),
}

DESCRIPTIONS = {
    "laundering_count": "the bank's training-window laundering payments",
    "test_base_rate": "laundering share of the bank's exam paper",
    "laundering_internal_share": "share of the bank's training-window laundering "
                                 "payments whose two endpoints both belong to it",
    "chain_pattern_share": "share of the bank's matched training-window laundering "
                           "payments in a chain-shaped injected pattern — STACK, "
                           "CYCLE or RANDOM — rather than a star-shaped one",
    "class_homophily": "share of the payments adjacent to the bank's training-window "
                       "laundering payments, on the global frame, that are themselves "
                       "laundering",
    "pattern_entropy": "Shannon entropy, in nats, of the bank's matched "
                       "training-window laundering payments over the eight injected "
                       "pattern families",
}


def pair_keys(payer: np.ndarray, payee: np.ndarray) -> np.ndarray:
    """One key per payment for the unordered account pair it joins."""
    span = int(max(payer.max(), payee.max())) + 1
    low = np.minimum(payer, payee)
    high = np.maximum(payer, payee)
    return low.astype(np.int64) * span + high


def neighbourhood(payer: np.ndarray, payee: np.ndarray,
                  laundering: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per payment, how many other payments share an account with it, and how
    many of those are laundering.

    Both counts are inclusion and exclusion on endpoint degrees: the payments
    touching either endpoint, less those touching both, less the payment
    itself.
    """
    n_accounts = int(max(payer.max(), payee.max())) + 1
    degree = (np.bincount(payer, minlength=n_accounts)
              + np.bincount(payee, minlength=n_accounts))
    degree_laundering = (np.bincount(payer, weights=laundering, minlength=n_accounts)
                         + np.bincount(payee, weights=laundering, minlength=n_accounts))

    keys, inverse = np.unique(pair_keys(payer, payee), return_inverse=True)
    shared = np.bincount(inverse, minlength=len(keys))[inverse]
    shared_laundering = np.bincount(inverse, weights=laundering,
                                    minlength=len(keys))[inverse]

    total = degree[payer] + degree[payee] - shared - 1
    laundering_total = (degree_laundering[payer] + degree_laundering[payee]
                        - shared_laundering - laundering)
    return total, laundering_total


def homophily(total: np.ndarray, laundering_total: np.ndarray,
              focal: np.ndarray) -> float:
    """The pooled ratio over one bank's focal payments: the laundering among all
    their neighbours, over all their neighbours. Undefined when the focal
    payments have no neighbours at all."""
    denominator = float(total[focal].sum())
    if denominator <= 0:
        return np.nan
    return float(laundering_total[focal].sum() / denominator)


def mean_of_ratios(total: np.ndarray, laundering_total: np.ndarray,
                   focal: np.ndarray) -> float:
    """The same quantity averaged the other way: each focal payment's own share
    of laundering neighbours, averaged over the payments. The pooled ratio
    weights a payment by the size of its neighbourhood and this one does not,
    so the two answer different questions about the same graph and both are
    recorded."""
    counted = focal & (total > 0)
    if not counted.any():
        return np.nan
    return float(np.mean(laundering_total[counted] / total[counted]))


def entropy(codes: np.ndarray, n_families: int) -> float:
    """Shannon entropy in nats over the pattern families present."""
    if len(codes) == 0:
        return np.nan
    counts = np.bincount(codes, minlength=n_families)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())
