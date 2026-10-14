"""Rank statistics for Experiment 3: does a ledger measurement track the
shortfall?

Experiment 3 asks whether a measurement *orders* the banks the way the shortfall does,
so every statistic here is a function of ranks: units and freak banks cannot
dominate, and taking logs changes nothing.

Data. Every function takes parallel one-dimensional arrays over banks — one
value per bank, in one shared bank order — and returns plain floats or a small
dictionary. Controls arrive as a (banks, controls) matrix, which is what makes
the base-rate partial and the two-control scale partial the same call. Nothing
here holds a DataFrame or mutates its input.

An undefined correlation is not evidence: where a variable carries no
variation the statistic is NaN and the p-value is 1, never the smallest the
permutation resolution allows.

Two null distributions are used, and the choice is fixed in advance rather
than convenient. A bare rank correlation is tested by **permuting the banks**, which
assumes only exchangeability under the null and gives a p-value that is never
zero. A partial rank correlation is tested by permuting the target **within
blocks of banks alike in the controls**, which leaves the controls' hold on the
target in place and breaks only the association under test; the t-approximation
on n − 2 − q degrees of freedom is reported beside it, and is not the test,
because rank residualisation removes only the linear-in-rank part of a
control's effect.
"""

from __future__ import annotations

import numpy as np

VERSION = 2


def _ranks(values: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(np.asarray(values, np.float64))


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denominator) if denominator > 0 else np.nan


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return _pearson(_ranks(x), _ranks(y))


def permutation_p(x: np.ndarray, y: np.ndarray, *, alternative: str = "two-sided",
                  draws: int = 10_000, seed: int = 0) -> dict:
    """Spearman's rho with a bank-permutation p-value.

    The p-value is (1 + the number of permutations at least as extreme) /
    (1 + draws), so it is never zero and reports its own resolution. The
    alternative is ``two-sided`` (the screen) or ``greater`` (the held-out
    test, whose direction is fixed in advance).
    """
    rx, ry = _ranks(x), _ranks(y)
    observed = _pearson(rx, ry)
    if not np.isfinite(observed):
        return {"rho": np.nan, "k": int(len(rx)), "alternative": alternative,
                "permutations": int(draws), "p": 1.0,
                "p_resolution": float(1.0 / (1 + draws))}
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        null[i] = _pearson(rx, rng.permutation(ry))
    extreme = (np.abs(null) >= abs(observed)) if alternative == "two-sided" \
        else (null >= observed)
    return {"rho": observed, "k": int(len(rx)), "alternative": alternative,
            "permutations": int(draws),
            "p": float((1 + int(extreme.sum())) / (1 + draws)),
            "p_resolution": float(1.0 / (1 + draws))}


def asymptotic_p(rho: float, k: int, *, controls: int = 0,
                 alternative: str = "two-sided") -> float:
    """The t-approximation to a (partial) rank correlation's p-value, on
    k − 2 − ``controls`` degrees of freedom."""
    from scipy.stats import t as student_t

    df = k - 2 - controls
    if df <= 0 or not np.isfinite(rho) or abs(rho) >= 1.0:
        return np.nan
    statistic = rho * np.sqrt(df / (1.0 - rho * rho))
    tail = float(student_t.sf(abs(statistic), df))
    return 2.0 * tail if alternative == "two-sided" else \
        (tail if statistic >= 0 else 1.0 - tail)


def _as_columns(controls: np.ndarray) -> np.ndarray:
    """Controls as a (banks, controls) matrix, however they arrived."""
    controls = np.asarray(controls, np.float64)
    return controls[:, None] if controls.ndim == 1 else controls


def _residual_maker(controls: np.ndarray) -> np.ndarray:
    """The projection that removes an intercept and the controls' ranks."""
    columns = _as_columns(controls)
    design = np.column_stack([np.ones(len(columns))]
                             + [_ranks(column) for column in columns.T])
    return np.eye(len(columns)) - design @ np.linalg.pinv(design)


RESIDUAL_TOLERANCE = 1e-12


def partial_spearman(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> dict:
    """Partial rank correlation by rank residualisation: rank-transform the two
    variables and every control, regress the two on the controls, and correlate
    what is left. With one control this is the ordinary partial Spearman; with
    the scale pair it removes both at once.

    A variable that carries no variation at all, and a residual smaller than
    :data:`RESIDUAL_TOLERANCE` of the variation it came from, are both undefined
    rather than whatever the rounding happens to make them: the first has no
    ordering to correlate, and the second is a variable the controls explain
    exactly.
    """
    remove = _residual_maker(controls)
    q = _as_columns(controls).shape[1]
    ranked_x, ranked_y = _ranks(x), _ranks(y)
    residual_x, residual_y = remove @ ranked_x, remove @ ranked_y
    spread = np.sqrt(((ranked_x - ranked_x.mean()) ** 2).sum()
                     * ((ranked_y - ranked_y.mean()) ** 2).sum())
    left = np.sqrt((residual_x @ residual_x) * (residual_y @ residual_y))
    rho = _pearson(residual_x, residual_y) \
        if spread > 0 and left > RESIDUAL_TOLERANCE * spread else np.nan
    return {"rho": rho, "controls": int(q), "k": int(len(x)),
            "p": asymptotic_p(rho, len(x), controls=q)}


def control_blocks(controls: np.ndarray, width: int = 5) -> np.ndarray:
    """Blocks of banks alike in the controls: consecutive runs of ``width`` in
    the order of the controls' rank sum, the last block absorbing the
    remainder.

    A partial correlation asks whether two things still move together among
    banks that are alike in the controls, and these blocks are those banks.
    Permuting a target within them holds its dependence on the controls —
    linear or not — and breaks only its dependence on the descriptor, which is
    the null the partial tests. Ordering by the rank sum reduces the controls
    to one dimension, which is exact for a single control and close to it
    wherever the controls rank the banks alike; how close is a measured
    quantity, reported as the within-block spread of each control.
    """
    key = np.sum([_ranks(column) for column in _as_columns(controls).T], axis=0)
    order = np.argsort(key, kind="stable")
    k = len(key)
    blocks = np.empty(k, np.int64)
    blocks[order] = np.minimum(np.arange(k) // width, max(1, k // width) - 1)
    return blocks


def conditional_permutation_p(x: np.ndarray, y: np.ndarray, controls: np.ndarray, *,
                              width: int = 5, draws: int = 10_000,
                              seed: int = 0) -> dict:
    """The partial rank correlation with a within-block permutation p-value.

    The statistic is :func:`partial_spearman`'s; the null permutes the target
    inside the blocks :func:`control_blocks` builds, so the controls' hold on
    the target survives the permutation and only the association under test is
    destroyed. This replaces the t-approximation, whose size is inflated
    wherever the controls act on both variables other than linearly in rank.
    """
    blocks = control_blocks(controls, width)
    observed = partial_spearman(x, y, controls)
    members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    if not np.isfinite(observed["rho"]):
        return {**observed, "p_conditional": 1.0, "p_asymptotic": observed["p"],
                "blocks": int(len(members)), "block_width": int(width),
                "permutations": int(draws)}

    remove = _residual_maker(controls)
    residual_x = remove @ _ranks(x)
    ry = _ranks(y)

    rng = np.random.default_rng(seed)
    permuted = np.repeat(ry[None, :], draws, axis=0)
    for member in members:
        order = np.argsort(rng.random((draws, len(member))), axis=1)
        permuted[:, member] = ry[member][order]

    projected = permuted @ remove
    numerator = projected @ residual_x
    denominator = np.sqrt(np.einsum("ij,ij->i", projected, permuted)
                          * (residual_x @ residual_x))
    null = np.where(denominator > 0, numerator / denominator, np.nan)
    extreme = np.abs(null) >= abs(observed["rho"])
    return {**observed, "p_conditional": float((1 + int(extreme.sum())) / (1 + draws)),
            "p_asymptotic": observed["p"], "blocks": int(len(members)),
            "block_width": int(width), "permutations": int(draws)}


def partial_size(controls: np.ndarray, *, replicates: int = 500, draws: int = 999,
                 levels: tuple[float, ...] = (0.05, 0.10),
                 widths: tuple[int, ...] = (3, 5, 10), width: int = 5,
                 seed: int = 0) -> dict:
    """The realised size of :func:`conditional_permutation_p` on one set of real
    control values, under a deliberately hostile null.

    No distribution-free conditional independence test exists, so the size of
    this one is a measured quantity rather than a guarantee. The null draws two
    variables that are independent *given* the controls but both driven by them
    through exp(3z), z being the standardised rank sum — the harshest monotone
    transform of a rank-uniform quantity, which is the shape rank
    residualisation handles worst. The rate at which the test rejects is
    reported at every nominal level, on the very controls the screen uses, at the
    chosen block width and at the widths either side of it, with the
    t-approximation's own rate beside them.
    """
    columns = _as_columns(controls)
    z = np.sum([_ranks(column) for column in columns.T], axis=0)
    z = (z - z.mean()) / z.std()
    driver = np.exp(3.0 * z)
    scale = 0.5 * driver.std()

    rng = np.random.default_rng(seed)
    conditional = {w: np.empty(replicates) for w in sorted(set(widths) | {width})}
    asymptotic = np.empty(replicates)
    for i in range(replicates):
        x = driver + rng.normal(0.0, scale, len(z))
        y = driver + rng.normal(0.0, scale, len(z))
        for w in conditional:
            out = conditional_permutation_p(x, y, columns, width=w, draws=draws,
                                            seed=int(rng.integers(1 << 30)))
            conditional[w][i] = out["p_conditional"]
        asymptotic[i] = out["p_asymptotic"]

    def rates(values: np.ndarray) -> dict:
        return {str(level): float((values <= level).mean()) for level in levels}

    return {"replicates": int(replicates), "permutations": int(draws),
            "block_width": int(width),
            "size": rates(conditional[width]),
            "size_by_block_width": {str(w): rates(values)
                                    for w, values in sorted(conditional.items())},
            "size_of_the_t_approximation": rates(asymptotic),
            "null": "x and y independent given the controls, both driven by "
                    "exp(3z) with z the standardised rank sum"}


def _kendall(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import kendalltau

    return float(kendalltau(x, y).statistic)


def partial_kendall(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Kendall's partial tau (Kendall 1942), applied recursively when more than
    one control is removed. The recursion removes the controls in the order
    given; the order is fixed so the reading is reproducible."""
    controls = np.atleast_2d(np.asarray(controls, np.float64))
    if controls.shape[0] != len(x):
        controls = controls.T

    taus = {("x", "y"): _kendall(x, y)}
    variables = {"x": x, "y": y}
    for i, column in enumerate(controls.T):
        variables[f"z{i}"] = column
    keys = ["x", "y"] + [f"z{i}" for i in range(controls.shape[1])]
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            taus[(a, b)] = _kendall(variables[a], variables[b])

    def tau(a: str, b: str) -> float:
        return taus[(a, b)] if (a, b) in taus else taus[(b, a)]

    current = {(a, b): tau(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]}
    remaining = keys[2:]
    while remaining:
        z, remaining = remaining[0], remaining[1:]
        others = ["x", "y"] + remaining

        def joint(a: str, b: str) -> float:
            return current[(a, b)] if (a, b) in current else current[(b, a)]

        updated = {}
        for i, a in enumerate(others):
            for b in others[i + 1:]:
                numerator = joint(a, b) - joint(a, z) * joint(b, z)
                denominator = np.sqrt((1 - joint(a, z) ** 2) * (1 - joint(b, z) ** 2))
                updated[(a, b)] = float(numerator / denominator) if denominator > 0 else np.nan
        current = updated
    return current[("x", "y")]


def benjamini_hochberg(p: np.ndarray, q: float = 0.10) -> dict:
    """The step-up procedure over one family: sort the p-values, compare each
    with its rising bar (position / total x q), and declare everything up to
    and including the largest position still under its bar."""
    p = np.asarray(p, np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    bars = np.arange(1, m + 1, dtype=np.float64) / m * q
    under = p[order] <= bars
    cut = int(np.flatnonzero(under).max()) + 1 if under.any() else 0
    declared = np.zeros(m, bool)
    declared[order[:cut]] = True
    position = np.empty(m, np.int64)
    position[order] = np.arange(1, m + 1)
    return {"declared": declared, "position": position,
            "bar": position / m * q, "q": float(q), "m": int(m),
            "declared_count": int(cut),
            "threshold": float(p[order][cut - 1]) if cut else 0.0}


def repeat_mean_rho(predictions: np.ndarray, target: np.ndarray) -> float:
    """The ranking metric: the mean over repeats of the out-of-fold
    rank correlation. ``predictions`` is (repeats, banks)."""
    return float(np.mean([_pearson(_ranks(row), _ranks(target))
                          for row in np.atleast_2d(predictions)]))


def bootstrap_repeat_mean(predictions: np.ndarray, target: np.ndarray, *,
                          draws: int = 1000, seed: int = 0) -> dict:
    """A bank-resampling interval for the ranking metric itself, so the wobble
    that bands a candidate is the wobble of the quantity it was ranked by."""
    predictions = np.atleast_2d(predictions)
    rng = np.random.default_rng(seed)
    k = len(target)
    values = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, k, k)
        values[i] = repeat_mean_rho(predictions[:, idx], target[idx])
    values = values[np.isfinite(values)]
    return {"sd": float(np.std(values, ddof=1)),
            "q05": float(np.percentile(values, 5)),
            "ci95": [float(np.percentile(values, 2.5)),
                     float(np.percentile(values, 97.5))],
            "draws_valid": int(len(values))}


def bootstrap_repeat_mean_difference(a: np.ndarray, b: np.ndarray,
                                     target: np.ndarray, *, draws: int = 1000,
                                     seed: int = 0) -> dict:
    """The paired reading of one candidate's ranking metric minus another's:
    the same resample scores both, so the wobble they share cancels."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    rng = np.random.default_rng(seed)
    k = len(target)
    values = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, k, k)
        values[i] = (repeat_mean_rho(a[:, idx], target[idx])
                     - repeat_mean_rho(b[:, idx], target[idx]))
    values = values[np.isfinite(values)]
    return {"delta": repeat_mean_rho(a, target) - repeat_mean_rho(b, target),
            "sd": float(np.std(values, ddof=1)),
            "q05": float(np.percentile(values, 5)),
            "ci95": [float(np.percentile(values, 2.5)),
                     float(np.percentile(values, 97.5))],
            "draws_valid": int(len(values)),
            "positive": bool(np.percentile(values, 5) > 0)}


def bootstrap_rho(x: np.ndarray, y: np.ndarray, *, draws: int = 1000,
                  seed: int = 0) -> dict:
    """A bank-resampling interval for one rank correlation: resample the banks
    with replacement, re-rank inside each draw, and read the percentiles."""
    rng = np.random.default_rng(seed)
    k = len(x)
    values = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, k, k)
        values[i] = spearman(x[idx], y[idx])
    values = values[np.isfinite(values)]
    return {"sd": float(np.std(values, ddof=1)),
            "q05": float(np.percentile(values, 5)),
            "ci95": [float(np.percentile(values, 2.5)),
                     float(np.percentile(values, 97.5))],
            "draws_valid": int(len(values))}


def permutation_p_difference(a: np.ndarray, b: np.ndarray, y: np.ndarray, *,
                             draws: int = 10_000, seed: int = 0) -> float:
    """One-sided p for rho(a, y) − rho(b, y) under bank permutation of the
    target, which breaks both associations at once and so leaves the two arms'
    shared structure in place."""
    ry = _ranks(y)
    ra, rb = _ranks(a), _ranks(b)
    observed = _pearson(ra, ry) - _pearson(rb, ry)
    if not np.isfinite(observed):
        return 1.0
    rng = np.random.default_rng(seed)
    null = np.empty(draws)
    for i in range(draws):
        permuted = rng.permutation(ry)
        null[i] = _pearson(ra, permuted) - _pearson(rb, permuted)
    return float((1 + int((null >= observed).sum())) / (1 + draws))


def target_resampling_rho(prediction: np.ndarray, replicates: np.ndarray) -> dict:
    """The rank agreement recomputed on each of Experiment 1's joint-bootstrap replicates
    of the target, holding the prediction fixed.

    The bank bootstrap prices a different sample of *banks*; this prices a
    different draw of the *test window* the targets were measured on, which is
    the noise Experiment 1 recorded and Experiment 3 inherits. ``replicates`` is (replicates,
    banks), Experiment 1's ``replicates.npy``, in the banks' shared order.
    """
    values = np.array([spearman(prediction, replicate) for replicate in replicates])
    values = values[np.isfinite(values)]
    return {"median": float(np.median(values)),
            "sd": float(np.std(values, ddof=1)),
            "q05": float(np.percentile(values, 5)),
            "ci95": [float(np.percentile(values, 2.5)),
                     float(np.percentile(values, 97.5))],
            "replicates_valid": int(len(values))}
