"""Scores and uncertainty machinery for Experiment 1 and its successors.

The joint bootstrap resamples the whole test window with replacement and
recomputes every group's shortfall from the same resample — never
group-by-group — so the dependence induced by shared cross-group payments is
carried into every statistic. Heterogeneity is the statistic Q with the I-squared
transform; its assumption-free null centres each bootstrap replicate on the
point shortfall, one common shortfall with the empirical joint dependence
kept.

Weighted average precision is vectorised over bootstrap replicates: scores
are sorted once per (group, setting, side), and each replicate contributes a
row of multiplicity weights whose cumulative sums give the precision at every
distinct-score threshold.
"""

from __future__ import annotations

import numpy as np

F1_THRESHOLD = 0.5
METRICS = ("ap", "ap_common")


def minority_f1(y: np.ndarray, score: np.ndarray) -> float:
    pred = score >= F1_THRESHOLD
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    if tp == 0:
        return 0.0
    precision, recall = tp / (tp + fp), tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    return float(average_precision_score(y, score))


def weighted_average_precision(y: np.ndarray, score: np.ndarray,
                               weights: np.ndarray) -> np.ndarray:
    """AP per replicate for a (replicates, n) weight matrix; ties grouped as
    one threshold, matching the unweighted sklearn definition at unit
    weights. Replicates with no positive mass return NaN."""
    order = np.argsort(-score, kind="stable")
    y_s = y[order].astype(np.float64)
    score_s = score[order]
    w_s = weights[:, order].astype(np.float64)
    ends = np.flatnonzero(np.diff(score_s) != 0)
    ends = np.concatenate([ends, [len(score_s) - 1]])
    cum_tp = np.cumsum(w_s * y_s, axis=1)[:, ends]
    cum_all = np.cumsum(w_s, axis=1)[:, ends]
    tp_inc = np.diff(cum_tp, axis=1, prepend=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(cum_all > 0, cum_tp / cum_all, 0.0)
        total = cum_tp[:, -1]
        ap = (tp_inc * precision).sum(axis=1) / total
    return np.where(total > 0, ap, np.nan)


_METRIC_FN = {"ap": weighted_average_precision, "ap_common": weighted_average_precision}


def _row_weights(spec: dict, metric: str) -> np.ndarray | None:
    if metric != "ap_common":
        return None
    return np.where(spec["y"] == 1, 1.0, spec["neg_weight"]).astype(np.float32)


def joint_bootstrap(n_test: int, groups: dict, *, replicates: int = 1000,
                    seed: int = 0, chunk: int = 100,
                    metrics: tuple[str, ...] = METRICS) -> dict:
    """Per-setting scores on both sides and their joint-bootstrap
    replicates.

    ``groups`` maps a label to ``{"idx": test-window positions of the
    group's test set, "y": labels, "solo": (settings, n_g) scores,
    "global": (settings, n_g) scores, "neg_weight": float}``; the
    ``neg_weight`` reweights negatives for the common-base-rate AP. Each
    replicate draws one multiplicity vector over all ``n_test`` payments and
    re-reads every group from it. Returns ``point[metric][label][side]`` as
    (settings,) arrays and ``reps[metric][label]`` as (replicates, settings)
    arrays of global minus solo.
    """
    rng = np.random.default_rng(seed)
    labels = list(groups)
    n_settings = next(iter(groups.values()))["solo"].shape[0]

    point = {m: {} for m in metrics}
    for g in labels:
        spec = groups[g]
        for m in metrics:
            base = _row_weights(spec, m)
            ones = np.ones((1, len(spec["y"]))) if base is None else base[None, :]
            fn = _METRIC_FN[m]
            point[m][g] = {
                side: np.array([fn(spec["y"], spec[side][s], ones)[0]
                                for s in range(n_settings)])
                for side in ("solo", "global")}

    reps = {m: {g: np.empty((replicates, n_settings)) for g in labels}
            for m in metrics}
    done = 0
    while done < replicates:
        r = min(chunk, replicates - done)
        multiplicity = np.zeros((r, n_test), np.float32)
        for j in range(r):
            draw = rng.integers(0, n_test, n_test)
            multiplicity[j] = np.bincount(draw, minlength=n_test)
        for g in labels:
            spec = groups[g]
            w = multiplicity[:, spec["idx"]]
            for m in metrics:
                base = _row_weights(spec, m)
                wm = w if base is None else w * base[None, :]
                fn = _METRIC_FN[m]
                for s in range(n_settings):
                    reps[m][g][done:done + r, s] = (
                        fn(spec["y"], spec["global"][s], wm)
                        - fn(spec["y"], spec["solo"][s], wm))
        done += r
    return {"point": point, "reps": reps, "labels": labels,
            "n_settings": n_settings}


def aggregate_shortfalls(boot: dict, metric: str, aggregate: str) -> dict:
    """Point shortfall, per-setting shortfalls, replicate shortfalls, and
    the bootstrap standard error and interval per label. ``aggregate`` is
    ``median`` (settings-paired, tree family) or ``mean`` (seed-mean, GNN
    family); the replicate aggregate ignores NaN settings."""
    reduce = {"median": np.nanmedian, "mean": np.nanmean}[aggregate]
    out = {}
    for g in boot["labels"]:
        p = boot["point"][metric][g]
        per_setting = p["global"] - p["solo"]
        delta_reps = reduce(boot["reps"][metric][g], axis=1)
        valid = np.isfinite(delta_reps)
        lo, hi = np.percentile(delta_reps[valid], [2.5, 97.5])
        out[g] = {"delta": float(reduce(per_setting)),
                  "per_setting": per_setting.tolist(),
                  "solo_by_setting": p["solo"].tolist(),
                  "global_by_setting": p["global"].tolist(),
                  "delta_reps": delta_reps,
                  "se_boot": float(np.std(delta_reps[valid], ddof=1)),
                  "ci95": [float(lo), float(hi)],
                  "replicates_valid": int(valid.sum())}
    return out


def heterogeneity(deltas: np.ndarray, ses: np.ndarray) -> dict:
    """The statistic Q and the share I-squared; groups with an error bar of zero are
    excluded rather than given infinite weight."""
    usable = ses > 0
    d, s = deltas[usable], ses[usable]
    k = len(d)
    w = 1.0 / s**2
    mean = float((w * d).sum() / w.sum())
    q = float((w * (d - mean) ** 2).sum())
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"Q": q, "I2": i2, "k": k, "weighted_mean": mean,
            "excluded_degenerate": int((~usable).sum())}


def null_q_replicates(delta_reps: np.ndarray, deltas: np.ndarray, ses: np.ndarray,
                      seed_var: np.ndarray | None = None, *,
                      seed: int = 0) -> np.ndarray:
    """Q on every centred replicate: each replicate's shortfalls centred on
    their point values (one common shortfall, the empirical joint dependence
    kept), plus an independent normal seed-noise draw of variance
    ``seed_var / 3`` per group when given. Replicates holding a non-finite
    shortfall are dropped. ``delta_reps`` is (replicates, groups)."""
    usable = ses > 0
    reps = delta_reps[:, usable] - deltas[usable][None, :]
    if seed_var is not None:
        rng = np.random.default_rng(seed)
        reps = reps + rng.standard_normal(reps.shape) * np.sqrt(seed_var[usable] / 3.0)[None, :]
    w = 1.0 / ses[usable] ** 2
    reps = reps[np.all(np.isfinite(reps), axis=1)]
    means = (reps * w).sum(axis=1) / w.sum()
    return ((reps - means[:, None]) ** 2 * w).sum(axis=1)


def null_q_check(delta_reps: np.ndarray, deltas: np.ndarray, ses: np.ndarray,
                 seed_var: np.ndarray | None = None, *, seed: int = 0,
                 level: float = 0.95) -> dict:
    """The assumption-free null for Q (:func:`null_q_replicates`); the check
    passes when the observed Q exceeds the null's ``level`` quantile."""
    q_null = null_q_replicates(delta_reps, deltas, ses, seed_var, seed=seed)
    observed = heterogeneity(deltas, ses)["Q"]
    q95 = float(np.quantile(q_null, level))
    return {"observed_Q": observed, "null_q95": q95,
            "null_q_mean": float(q_null.mean()),
            "null_replicates": int(len(q_null)),
            "exceedance": float((q_null >= observed).mean()),
            "passes": bool(observed > q95)}


def leave_one_out(labels: list[str], deltas: np.ndarray, ses: np.ndarray) -> dict:
    """I-squared without the single most-weighted group."""
    weights = np.where(ses > 0, 1.0 / np.where(ses > 0, ses, 1.0) ** 2, 0.0)
    drop = int(np.argmax(weights))
    keep = np.ones(len(deltas), bool)
    keep[drop] = False
    het = heterogeneity(deltas[keep], ses[keep])
    return {"dropped": labels[drop], "dropped_weight_share":
            float(weights[drop] / weights.sum()), **het}


def tau_reading(deltas: np.ndarray, ses: np.ndarray) -> dict:
    """tau, the estimated standard deviation of the true shortfalls, and its
    two terms: the excess Q - (k - 1) and the weight term
    sum w - sum w^2 / sum w, with w = 1/s^2; tau^2 is their ratio, floored
    at zero."""
    w = 1.0 / ses ** 2
    excess = heterogeneity(deltas, ses)["Q"] - (len(deltas) - 1)
    weight_term = float(w.sum() - (w ** 2).sum() / w.sum())
    return {"excess": float(excess), "weight_term": weight_term,
            "tau": float(np.sqrt(max(0.0, excess / weight_term)))}


def leave_each_out(labels: list[str], deltas: np.ndarray, ses: np.ndarray,
                   delta_reps: np.ndarray | None = None,
                   seed_var: np.ndarray | None = None) -> dict:
    """Q, I-squared and tau with each group removed in turn, beside each
    group's share of the weight and of Q on the full set. With replicates,
    each removal also carries the null check of :func:`null_q_check` on the
    kept columns (seed noise drawn afresh at seed 0 per removal)."""
    k = len(deltas)
    w = 1.0 / ses ** 2
    full = heterogeneity(deltas, ses)
    terms = w * (deltas - full["weighted_mean"]) ** 2
    removals = []
    for i in range(k):
        keep = np.arange(k) != i
        het = heterogeneity(deltas[keep], ses[keep])
        row = {"bank": labels[i], "weight_share": float(w[i] / w.sum()),
               "q_share": float(terms[i] / full["Q"]), "Q": het["Q"],
               "I2": het["I2"], "tau": tau_reading(deltas[keep], ses[keep])["tau"]}
        if delta_reps is not None:
            null = null_q_check(delta_reps[:, keep], deltas[keep], ses[keep],
                                None if seed_var is None else seed_var[keep])
            row |= {"null_q95": null["null_q95"], "exceedance": null["exceedance"],
                    "passes": null["passes"]}
        removals.append(row)
    lowest = min(removals, key=lambda r: r["I2"])
    return {"full": {**full, **tau_reading(deltas, ses)},
            "min_I2": lowest["I2"], "min_bank": lowest["bank"],
            "removals": removals}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def spearman_bank_bootstrap(a: np.ndarray, b: np.ndarray, *, draws: int = 1000,
                            seed: int = 0) -> dict:
    """Spearman rho over groups with a group-resampling bootstrap interval
    (one-sided positive reading: the 5th percentile), drawn from a fresh
    generator at ``seed``."""
    from scipy.stats import rankdata

    rng = np.random.default_rng(seed)
    k = len(a)
    ra, rb = rankdata(a), rankdata(b)

    def rho(x, y):
        x, y = x - x.mean(), y - y.mean()
        denom = np.sqrt((x * x).sum() * (y * y).sum())
        return float((x * y).sum() / denom) if denom > 0 else np.nan

    observed = rho(ra, rb)
    boot = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, k, k)
        boot[i] = rho(rankdata(a[idx]), rankdata(b[idx]))
    boot = boot[np.isfinite(boot)]
    return {"rho": observed, "k": int(k),
            "boot_q05": float(np.percentile(boot, 5)),
            "boot_ci95": [float(np.percentile(boot, 2.5)),
                          float(np.percentile(boot, 97.5))],
            "boot_draws_valid": int(len(boot)),
            "positive": bool(np.percentile(boot, 5) > 0)}
