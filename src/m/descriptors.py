"""The label-free properties of a bank's ledger.

A *label-free property* is one number a bank can compute about itself from its
own ledger, over the training window, without reading a label. Twelve of them,
in six groups, are defined here (Chapter 5); Experiment 3 asks whether they
order the banks the way the shortfall does.

Data. A bank's ledger arrives here as :class:`Ledger`, parallel arrays over the
rows of that bank's training window with same-account rows already dropped
(an account paying itself is bookkeeping, not structure). Accounts carry identity codes over the (bank, account) pair, since
account strings recur across banks; ``payer_is_own`` and ``payee_is_own`` mark
the endpoints belonging to the bank whose ledger this is, and at least one of
them is true on every row. ``counterparty_bank`` is the bank code of the
endpoint that is not the bank's own, and −1 on the rows where both endpoints
are. Nothing here holds a DataFrame, reads a label, or mutates its input.

Every function is a pure function of those arrays. A property that has no
value on a ledger — a share over an empty set — is NaN, and the caller records
how many banks that touched; it is never imputed.

The transform is the same for all twelve: a value *v* enters a regression
model as log(1 + v). It is defined at zero and needs no reference population. How far it
departs from the plain logarithm depends on the size of the value: over
HI-Medium's qualifying banks the two differ by at most 2e-4 on `payment_count` and by
0.41 on `external_encounters`, whose smallest value is 2, so the transform is
cosmetic for the large counts and substantive for the small one.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

VERSION = 1

NAMES = ("payment_count", "active_accounts", "internal_share", "boundary_share",
         "external_recurrence", "external_encounters", "external_banks",
         "external_hhi", "activity_gini", "reciprocity",
         "format_entropy", "currency_entropy")

GROUPS = {
    "scale": ("payment_count", "active_accounts"),
    "openness": ("internal_share", "boundary_share"),
    "knowledge": ("external_recurrence", "external_encounters"),
    "spread": ("external_banks", "external_hhi"),
    "structure": ("activity_gini", "reciprocity"),
    "mix": ("format_entropy", "currency_entropy"),
}

BASELINE = "scale"
RECURRENCE_THRESHOLD = 5

DESCRIPTIONS = {
    "payment_count": "rows in the bank's training ledger",
    "active_accounts": "distinct own accounts appearing on either side",
    "internal_share": "share of rows whose two endpoints both belong to the bank",
    "boundary_share": "share of active own accounts with at least one external "
                      "counterparty",
    "external_recurrence": f"share of distinct external counterparty accounts "
                           f"appearing on {RECURRENCE_THRESHOLD} or more rows",
    "external_encounters": "median, over distinct external counterparty accounts, "
                           "of the rows each appears on",
    "external_banks": "distinct banks other than the bank's own appearing on its rows",
    "external_hhi": "Herfindahl-Hirschman index of the cross-bank rows over those "
                    "banks: the sum of squared shares",
    "activity_gini": "Gini coefficient of the row counts of the bank's active own "
                     "accounts",
    "reciprocity": "share of distinct ordered account pairs (a, b) on the bank's "
                   "rows for which (b, a) also appears",
    "format_entropy": "Shannon entropy, in nats, of the payment-format distribution "
                      "over the bank's rows",
    "currency_entropy": "Shannon entropy, in nats, over the multiset formed by "
                        "taking both the paid and the received currency of every row",
}


class Ledger(NamedTuple):
    """One bank's training-window rows, same-account rows already dropped."""

    payer: np.ndarray
    payee: np.ndarray
    payer_is_own: np.ndarray
    payee_is_own: np.ndarray
    counterparty_bank: np.ndarray
    fmt: np.ndarray
    pay_ccy: np.ndarray
    recv_ccy: np.ndarray


def transform(values: np.ndarray) -> np.ndarray:
    """The transform: log(1 + v), the same for every property."""
    return np.log1p(np.asarray(values, np.float64))


def entropy(codes: np.ndarray) -> float:
    """Shannon entropy in nats of the distribution of ``codes``; zero on an
    empty or single-valued input, where there is nothing to be uncertain
    about."""
    if len(codes) == 0:
        return 0.0
    counts = np.bincount(codes[codes >= 0])
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def gini(counts: np.ndarray) -> float:
    """Gini coefficient of a non-negative activity vector: 0 when every account
    is equally active, approaching 1 when one account does everything. Zero on
    a single account, where there is no inequality to measure."""
    m = len(counts)
    if m <= 1:
        return 0.0
    x = np.sort(np.asarray(counts, np.float64))
    total = x.sum()
    if total == 0:
        return 0.0
    rank = np.arange(1, m + 1, dtype=np.float64)
    return float((2.0 * (rank * x).sum()) / (m * total) - (m + 1.0) / m)


def _counts_over(values: np.ndarray) -> np.ndarray:
    """Occurrences of each distinct value, discarding the values themselves."""
    if len(values) == 0:
        return np.zeros(0, np.int64)
    return np.unique(values, return_counts=True)[1]


def compute(ledger: Ledger) -> dict[str, float]:
    """Every label-free property of one bank's training ledger."""
    n = len(ledger.payer)
    internal = ledger.payer_is_own & ledger.payee_is_own
    crosses = ~internal

    own_endpoint = np.where(ledger.payer_is_own, ledger.payer, ledger.payee)
    own_accounts = np.concatenate([ledger.payer[ledger.payer_is_own],
                                   ledger.payee[ledger.payee_is_own]])
    active = np.unique(own_accounts)

    external_endpoint = np.where(ledger.payer_is_own, ledger.payee, ledger.payer)[crosses]
    external_counts = _counts_over(external_endpoint)
    boundary = np.unique(own_endpoint[crosses])

    bank_counts = _counts_over(ledger.counterparty_bank[crosses])
    bank_shares = bank_counts / bank_counts.sum() if len(bank_counts) else bank_counts

    span = int(max(ledger.payer.max(), ledger.payee.max())) + 1
    pairs = np.unique(ledger.payer.astype(np.int64) * span + ledger.payee)
    reverse = (pairs % span) * span + (pairs // span)
    position = np.searchsorted(pairs, reverse)
    mutual = pairs[np.minimum(position, len(pairs) - 1)] == reverse

    return {
        "payment_count": float(n),
        "active_accounts": float(len(active)),
        "internal_share": float(internal.mean()),
        "boundary_share": float(len(boundary) / len(active)),
        "external_recurrence": (float((external_counts >= RECURRENCE_THRESHOLD).mean())
                                if len(external_counts) else np.nan),
        "external_encounters": (float(np.median(external_counts))
                                if len(external_counts) else np.nan),
        "external_banks": float(len(bank_counts)),
        "external_hhi": (float((bank_shares ** 2).sum())
                         if len(bank_counts) else np.nan),
        "activity_gini": gini(np.bincount(np.searchsorted(active, own_accounts))),
        "reciprocity": float(mutual.mean()) if len(pairs) else np.nan,
        "format_entropy": entropy(ledger.fmt),
        "currency_entropy": entropy(np.concatenate([ledger.pay_ccy, ledger.recv_ccy])),
    }
