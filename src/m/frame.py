"""The frame probe's trivial rules.

A *frame* is a set of payments: the whole payment table (the global frame), or
the payments touching one bank (that bank's ledger). Every rule here scores a
payment from its frame alone, trains nothing, and counts only payments strictly
earlier than the one it scores. Scored on one exam paper under both frames, a
rule measures what the local frame manufactures: the gap between its ledger
reading and its global reading is predictive signal that exists only because
the bank sees a part of the network rather than the whole of it.

Three rules cannot vary with the frame, and stand as the probe's placebo. The
bank's own endpoint carries the same history in the bank's ledger as in the
whole table, because every payment touching that account touches the bank. So
does the ordered pair of endpoints, one of which is always the bank's. So does
the payment's own amount, which counts no history at all. The probe scores all
three on both frames and requires identical readings, which checks that the
exam paper's rows and the global frame's rows line up.

Data. A frame arrives here as arrays in canonical (time-sorted) order: the two
endpoints' account identity codes from :func:`m.features.account_codes` — one
dictionary over both columns, so an account keeps one identity whichever side
it pays from — the timestamp, and the amount. A bank's ledger is the same
arrays restricted to the rows touching that bank, which leaves every identity
unchanged. Nothing here holds a DataFrame or mutates its input.
"""

from __future__ import annotations

import numpy as np

VERSION = 1

RULES = ("external", "own", "payer", "payee", "min", "pair", "amount")
FRAME_INVARIANT = ("own", "pair", "amount")
RULE_DESCRIPTIONS = {
    "external": "minus the earlier appearances in the frame of the endpoint "
                "that is not the bank's own account",
    "own": "minus the earlier appearances in the frame of the bank's own endpoint",
    "payer": "minus the earlier appearances in the frame of the paying account",
    "payee": "minus the earlier appearances in the frame of the receiving account",
    "min": "minus the smaller of the two endpoints' earlier appearances in the frame",
    "pair": "minus the earlier appearances in the frame of the ordered "
            "(payer, payee) pair",
    "amount": "the payment's own amount in US dollars, which reads no history",
}


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


def pair_history(payer: np.ndarray, payee: np.ndarray,
                 ts: np.ndarray) -> np.ndarray:
    """Earlier appearances of each payment's ordered (payer, payee) pair."""
    return earlier_counts((payer, payee), ts)


def rule_scores(payer_hist: np.ndarray, payee_hist: np.ndarray,
                pair_hist: np.ndarray, payer_is_own: np.ndarray,
                amount_usd: np.ndarray) -> dict[str, np.ndarray]:
    """Every trivial rule's score, higher meaning more suspicious. The
    history arrays are counted within one frame; ``payer_is_own`` marks the
    rows whose paying account belongs to the bank whose paper is scored."""
    external = np.where(payer_is_own, payee_hist, payer_hist)
    own = np.where(payer_is_own, payer_hist, payee_hist)
    return {"external": -external.astype(np.float64),
            "own": -own.astype(np.float64),
            "payer": -payer_hist.astype(np.float64),
            "payee": -payee_hist.astype(np.float64),
            "min": -np.minimum(payer_hist, payee_hist).astype(np.float64),
            "pair": -pair_hist.astype(np.float64),
            "amount": amount_usd.astype(np.float64)}


def external_history(payer_hist: np.ndarray, payee_hist: np.ndarray,
                     payer_is_own: np.ndarray) -> np.ndarray:
    """The external endpoint's earlier appearances, as a count."""
    return np.where(payer_is_own, payee_hist, payer_hist)
