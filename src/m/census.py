"""The statistics the three census checks share (Chapter 3, Census).

A dataset arrives here as aligned arrays with one entry per payment: the
paying bank's code, the receiving bank's code and, where needed, the
laundering label. Nothing here holds a DataFrame or mutates its input.
"""

from __future__ import annotations

import numpy as np


def either_endpoint_counts(from_bank: np.ndarray, to_bank: np.ndarray,
                           mask: np.ndarray, n_banks: int) -> np.ndarray:
    """Per bank code, the masked payments in that bank's ledger: a payment
    between two banks counts at both, a payment within one bank once."""
    same = mask & (from_bank == to_bank)
    return (np.bincount(from_bank[mask], minlength=n_banks)
            + np.bincount(to_bank[mask], minlength=n_banks)
            - np.bincount(from_bank[same], minlength=n_banks))


def within_bank(from_bank: np.ndarray, to_bank: np.ndarray,
                is_laundering: np.ndarray) -> dict:
    """For laundering and for normal payments separately, the share that
    joins two accounts at the same bank, beside the share expected if
    accounts were assigned to banks at random: the sum over banks of the
    square of the bank's share of the class's payment endpoints."""
    banks, codes = np.unique(np.concatenate([from_bank, to_bank]),
                             return_inverse=True)
    n = len(from_bank)
    payer, payee = codes[:n], codes[n:]
    out = {}
    for label, mask in (("licit", is_laundering == 0),
                        ("laundering", is_laundering == 1)):
        f, t = payer[mask], payee[mask]
        endpoints = (np.bincount(f, minlength=len(banks))
                     + np.bincount(t, minlength=len(banks)))
        p = endpoints / endpoints.sum()
        observed = float((f == t).mean())
        chance = float((p ** 2).sum())
        out[label] = {"within_share": observed, "chance": chance,
                      "over_chance": observed / chance}
    return out
