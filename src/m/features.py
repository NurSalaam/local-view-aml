"""The tree family's feature vocabulary.

Every quantity is computed from payments strictly earlier than the queried
payment, within sliding windows, from the payments visible to one sight — the
frame handed to :func:`graph_features` *is* the sight. Same-account payments
contribute to no degree, sum, or fan feature but remain query rows.

The USD rate table is a dataset-level constant computed from the global
training window and shared by every scenario: a sight-dependent rate would make
the payment's own USD amounts differ between the ledger-sight and network-sight
tables, and the blind-tree placebo could not read exactly zero.

Window intervals are open on both ends, (t - w, t): a payment at exactly
t - w or t contributes nothing.

The vectorised core sorts contributor payments once per blocking (payer,
payee, ordered pair) into ``code * span + offset`` key arrays; every windowed
count, sum, and distinct-counterparty query is then two ``searchsorted``
calls. Distinct counting uses the interval identity: a repeat payment j is a
within-window duplicate at query time t iff t lies in
(ts_j, prev_same_pair_ts_j + w).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VERSION = 2

WINDOWS = {"6h": 21_600, "24h": 86_400, "7d": 604_800}
FAN_KS = (2, 4, 8, 16)

OWN_FIELDS = [
    "amount_usd", "amount_received_usd", "cross_currency", "payment_format",
    "payment_currency", "receiving_currency", "hour_of_day", "day_of_week",
]
CATEGORICAL_FIELDS = ["payment_format", "payment_currency", "receiving_currency"]

_USD_NAMES = ("US Dollar", "USD")


def graph_feature_names() -> list[str]:
    names = []
    for win in WINDOWS:
        for side in ("payer", "payee"):
            names += [f"{side}_{f}_{win}" for f in
                      ("out_degree", "in_degree", "out_count", "in_count",
                       "out_sum_usd", "in_sum_usd")]
            names += [f"{side}_fan_out_ge_{k}_{win}" for k in FAN_KS]
            names += [f"{side}_fan_in_ge_{k}_{win}" for k in FAN_KS]
        names += [f"{side}_reciprocity_{win}" for side in ("payer", "payee")]
        names += [f"recurrence_{win}"]
    return names


def usd_rate_table(df: pd.DataFrame) -> dict[str, float]:
    """Median implied cross-rate to US dollars per currency, from the
    training window's cross-currency payments, pivoting through the
    most-traded currency where no direct US-dollar cross exists."""
    currencies = sorted(set(df["pay_ccy"].cat.categories)
                        | set(df["recv_ccy"].cat.categories))
    usd = [c for c in currencies if c in _USD_NAMES]
    if not usd:
        return {c: 1.0 for c in currencies}
    (usd,) = usd

    tr = df[df["split"] == 0]
    pay = tr["pay_ccy"].astype(str).to_numpy()
    recv = tr["recv_ccy"].astype(str).to_numpy()
    paid = tr["amount_paid"].to_numpy(np.float64)
    received = tr["amount_received"].to_numpy(np.float64)
    ok = (pay != recv) & (paid > 0) & (received > 0)
    pay, recv, paid, received = pay[ok], recv[ok], paid[ok], received[ok]

    traded = pd.Series(np.concatenate([pay, recv])).value_counts()
    pivot = traded.index[0]

    def pooled_rate(c: str, target: str) -> float:
        forward = (pay == c) & (recv == target)
        backward = (pay == target) & (recv == c)
        obs = np.concatenate([received[forward] / paid[forward],
                              paid[backward] / received[backward]])
        return float(np.median(obs)) if len(obs) else np.nan

    rates = {usd: 1.0}
    pivot_to_usd = 1.0 if pivot == usd else pooled_rate(pivot, usd)
    for c in currencies:
        if c == usd:
            continue
        direct = pooled_rate(c, usd)
        if np.isfinite(direct):
            rates[c] = direct
            continue
        via = pooled_rate(c, pivot)
        if not (np.isfinite(via) and np.isfinite(pivot_to_usd)):
            raise ValueError(f"currency {c!r} has no cross to {usd!r},"
                             f" directly or through {pivot!r}")
        rates[c] = via * pivot_to_usd
    return rates


def own_fields(df: pd.DataFrame, rates: dict[str, float]) -> dict[str, np.ndarray]:
    pay_rate = np.array([rates[c] for c in df["pay_ccy"].cat.categories])
    recv_rate = np.array([rates[c] for c in df["recv_ccy"].cat.categories])
    pay_codes = df["pay_ccy"].cat.codes.to_numpy()
    recv_codes = df["recv_ccy"].cat.codes.to_numpy()
    ts = df["ts"].to_numpy()
    stamps = pd.to_datetime(ts, unit="s")
    return {
        "amount_usd": (df["amount_paid"].to_numpy(np.float64)
                       * pay_rate[pay_codes]).astype(np.float32),
        "amount_received_usd": (df["amount_received"].to_numpy(np.float64)
                                * recv_rate[recv_codes]).astype(np.float32),
        "cross_currency": (pay_codes != recv_codes).astype(np.float32),
        "payment_format": df["fmt"].cat.codes.to_numpy(np.float32),
        "payment_currency": pay_codes.astype(np.float32),
        "receiving_currency": recv_codes.astype(np.float32),
        "hour_of_day": stamps.hour.to_numpy(np.float32),
        "day_of_week": stamps.dayofweek.to_numpy(np.float32),
    }


def account_codes(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Codes over (bank, account) pairs, compact within this frame. The
    loaders categorise the two endpoint account columns independently (the
    bank columns share one roster), so both are re-coded onto their union
    dictionary before pairing — mixing the raw per-column codes would give
    an account different identities as payer and as payee."""
    accts = df["from_acct"].cat.categories.union(df["to_acct"].cat.categories)
    from_acct = accts.get_indexer(df["from_acct"].cat.categories).astype(np.int64)[
        df["from_acct"].cat.codes.to_numpy()]
    to_acct = accts.get_indexer(df["to_acct"].cat.categories).astype(np.int64)[
        df["to_acct"].cat.codes.to_numpy()]
    n_acct = len(accts)
    from_key = df["from_bank"].cat.codes.to_numpy(np.int64) * n_acct + from_acct
    to_key = df["to_bank"].cat.codes.to_numpy(np.int64) * n_acct + to_acct
    nodes, codes = np.unique(np.concatenate([from_key, to_key]),
                             return_inverse=True)
    n = len(df)
    return codes[:n], codes[n:]


def graph_features(ts: np.ndarray, payer: np.ndarray, payee: np.ndarray,
                   paid_usd: np.ndarray, recv_usd: np.ndarray,
                   contribute: np.ndarray) -> dict[str, np.ndarray]:
    """The 93 windowed graph features for every row of one sight's frame."""
    n = len(ts)
    t0 = int(ts.min()) if n else 0
    span = (int(ts.max()) - t0 + 2) if n else 2
    off = (ts - t0).astype(np.int64)

    idx = np.flatnonzero(contribute)
    if len(idx) == 0:
        return {name: np.zeros(n, np.float32) for name in graph_feature_names()}
    offc = off[idx]
    pc = payer[idx].astype(np.int64)
    qc = payee[idx].astype(np.int64)
    paidc = paid_usd[idx].astype(np.float64)
    recvc = recv_usd[idx].astype(np.float64)

    n_nodes = int(max(payer.max(), payee.max())) + 1 if n else 1
    assert n_nodes * span < 2**62

    pair64 = pc * n_nodes + qc
    order_pair = np.lexsort((offc, pair64))
    pair_sorted = pair64[order_pair]
    off_pair = offc[order_pair]
    unique_pairs = np.unique(pair_sorted)
    pair_code = np.searchsorted(unique_pairs, pair_sorted)
    key_pair = pair_code * span + off_pair
    n_pairs = len(unique_pairs)
    assert n_pairs * span < 2**62

    prev_off = np.full(len(idx), np.iinfo(np.int64).min // 4, np.int64)
    same = pair_sorted[1:] == pair_sorted[:-1]
    prev_off[1:][same] = off_pair[:-1][same]
    prev_c = np.empty_like(prev_off)
    prev_c[order_pair] = prev_off

    sides = {"payer": payer.astype(np.int64), "payee": payee.astype(np.int64)}

    def block(codes: np.ndarray, amounts: np.ndarray):
        order = np.lexsort((offc, codes))
        keys = codes[order] * span + offc[order]
        cum = np.concatenate([[0.0], np.cumsum(amounts[order])])
        prev_b = prev_c[order]
        return keys, cum, prev_b

    out_keys, out_cum, out_prev = block(pc, paidc)
    in_keys, in_cum, in_prev = block(qc, recvc)

    def window_counts(keys, cum, account, w):
        qhi = account * span + off
        qlo = account * span + np.maximum(off - w, -1)
        hi = np.searchsorted(keys, qhi, "left")
        lo = np.searchsorted(keys, qlo, "right")
        return hi - lo, (cum[hi] - cum[lo])

    def duplicates(keys, prev_b, account, w):
        kept = prev_b + w > keys % span
        end_keys = np.sort((keys - keys % span)[kept]
                           + np.minimum(prev_b[kept] + w, span - 1))
        start = np.searchsorted(keys[kept], account * span + off, "left")
        ended = np.searchsorted(end_keys, account * span + off, "right")
        return start - ended

    def pair_window(pfrom, pto, w):
        q64 = pfrom * n_nodes + pto
        code = np.searchsorted(unique_pairs, q64)
        code = np.minimum(code, n_pairs - 1)
        exists = unique_pairs[code] == q64
        qhi = code * span + off
        qlo = code * span + np.maximum(off - w, -1)
        count = (np.searchsorted(key_pair, qhi, "left")
                 - np.searchsorted(key_pair, qlo, "right"))
        return np.where(exists, count, 0)

    feats: dict[str, np.ndarray] = {}
    for win, w in WINDOWS.items():
        for side, account in sides.items():
            out_count, out_sum = window_counts(out_keys, out_cum, account, w)
            in_count, in_sum = window_counts(in_keys, in_cum, account, w)
            out_deg = out_count - duplicates(out_keys, out_prev, account, w)
            in_deg = in_count - duplicates(in_keys, in_prev, account, w)
            feats[f"{side}_out_degree_{win}"] = out_deg.astype(np.float32)
            feats[f"{side}_in_degree_{win}"] = in_deg.astype(np.float32)
            feats[f"{side}_out_count_{win}"] = out_count.astype(np.float32)
            feats[f"{side}_in_count_{win}"] = in_count.astype(np.float32)
            feats[f"{side}_out_sum_usd_{win}"] = out_sum.astype(np.float32)
            feats[f"{side}_in_sum_usd_{win}"] = in_sum.astype(np.float32)
            for k in FAN_KS:
                feats[f"{side}_fan_out_ge_{k}_{win}"] = (out_deg >= k).astype(np.float32)
                feats[f"{side}_fan_in_ge_{k}_{win}"] = (in_deg >= k).astype(np.float32)
        payer64, payee64 = sides["payer"], sides["payee"]
        rc_payer = pair_window(payee64, payer64, w)
        rc_payee = pair_window(payer64, payee64, w)
        feats[f"payer_reciprocity_{win}"] = rc_payer.astype(np.float32)
        feats[f"payee_reciprocity_{win}"] = rc_payee.astype(np.float32)
        feats[f"recurrence_{win}"] = (rc_payer + rc_payee).astype(np.float32)
    return feats


def extract(df: pd.DataFrame, rates: dict[str, float]) -> dict[str, np.ndarray]:
    """Own fields plus graph features for one sight's frame, canonical order."""
    fields = own_fields(df, rates)
    payer, payee = account_codes(df)
    feats = graph_features(df["ts"].to_numpy(np.int64), payer, payee,
                           fields["amount_usd"], fields["amount_received_usd"],
                           ~df["same_account"].to_numpy())
    return fields | feats


def row_copies(from_group: np.ndarray, to_group: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
    """One copy per (payment, endpoint group): a payment between groups B and
    C yields B's copy and C's copy; a within-group payment yields one copy.
    Returned sorted by (payment, group) for a deterministic table order."""
    n = len(from_group)
    idx = np.arange(n, dtype=np.int64)
    cross = from_group != to_group
    payment = np.concatenate([idx, idx[cross]])
    group = np.concatenate([from_group, to_group[cross]])
    order = np.lexsort((group, payment))
    return payment[order], group[order]
