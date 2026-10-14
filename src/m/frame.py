"""Counterparty history within a frame, for the never-seen-counterparty cue
(Chapter 3, Census).

A *frame* is a set of payments: the whole payment table, or the payments in
one bank's ledger. A payment's counterparty history is the number of
strictly earlier payments in the frame in which its counterparty, the
account that is not the bank's own, appears on either side. A counterparty
with no such payment is never seen.

Data. A frame arrives here as arrays in canonical (time-sorted) order: the two
endpoints' account identity codes from :func:`m.features.account_codes` — one
dictionary over both columns, so an account keeps one identity whichever side
it pays from — and the timestamp. A bank's ledger is the same arrays
restricted to the rows touching that bank, which leaves every identity
unchanged. Nothing here holds a DataFrame or mutates its input.
"""

from __future__ import annotations

import numpy as np


def earlier_counts(keys: tuple[np.ndarray, ...], ts: np.ndarray) -> np.ndarray:
    """Per row, the number of strictly earlier rows sharing its key.

    Rows are sorted once by key then by time, so a key's rows form a
    contiguous block and equal timestamps form a contiguous run inside it.
    The count is then the distance from the start of the row's key block to
    the start of its timestamp run: every row before that run shares the key
    and is strictly earlier, and every row inside it ties.
    """
    order = np.lexsort((ts, *reversed(keys)))
    t = ts[order]
    new_key = np.zeros(len(t), bool)
    new_key[0] = True
    for key in keys:
        sorted_key = key[order]
        new_key[1:] |= sorted_key[1:] != sorted_key[:-1]
    new_run = new_key.copy()
    new_run[1:] |= t[1:] != t[:-1]
    position = np.arange(len(t))
    key_start = np.maximum.accumulate(np.where(new_key, position, 0))
    run_start = np.maximum.accumulate(np.where(new_run, position, 0))
    out = np.empty(len(t), np.int64)
    out[order] = run_start - key_start
    return out


def endpoint_history(payer: np.ndarray, payee: np.ndarray,
                     ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Earlier appearances of each payment's payer and payee within the frame,
    counting an account's appearances on either side."""
    n = len(ts)
    counts = earlier_counts((np.concatenate([payer, payee]),),
                            np.concatenate([ts, ts]))
    return counts[:n], counts[n:]


def external_history(payer_hist: np.ndarray, payee_hist: np.ndarray,
                     payer_is_own: np.ndarray) -> np.ndarray:
    """The counterparty's earlier appearances: the payee's where the payer is
    the bank's own account, the payer's otherwise."""
    return np.where(payer_is_own, payee_hist, payer_hist)
