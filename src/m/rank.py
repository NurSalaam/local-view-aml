"""Rank statistics: does one quantity order the banks the way another does?

Every statistic here is a function of ranks, so units and freak banks cannot
dominate, and taking logs changes nothing. Experiment 2 correlates confirmed
cases with the shortfall, size held fixed; Experiment 3 scores the regression
models by the rank correlation of their out-of-fold predictions with the
shortfall.

Data. Every function takes parallel one-dimensional arrays over banks — one
value per bank, in one shared bank order — and returns plain floats or a small
dictionary. Controls arrive as a (banks, controls) matrix, which is what makes
the one-control partial and the two-control scale partial the same call.
Nothing here holds a DataFrame or mutates its input.

An undefined correlation is not evidence: where a variable carries no
variation the statistic is NaN and the p-value is 1, never the smallest the
permutation resolution allows.

A partial rank correlation is tested by permuting the target **within blocks
of banks alike in the controls**, which leaves the controls' hold on the
target in place and breaks only the association under test. A t-approximation
would not do, because rank residualisation removes only the linear-in-rank
part of a control's effect.
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
    return {"rho": rho, "controls": int(q), "k": int(len(x))}


def control_blocks(controls: np.ndarray, width: int = 5) -> np.ndarray:
    """Blocks of banks alike in the controls: consecutive runs of ``width`` in
    the order of the controls' rank sum, the last block absorbing the
    remainder.

    A partial correlation asks whether two things still move together among
    banks that are alike in the controls, and these blocks are those banks.
    Permuting a target within them holds its dependence on the controls —
    linear or not — and breaks only its dependence on ``x``, which is
    the null the partial tests. Ordering by the rank sum reduces the controls
    to one dimension, which is exact for a single control and close to it
    wherever the controls rank the banks alike.
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
    destroyed. A t-approximation's size would be inflated wherever the
    controls act on both variables other than linearly in rank.
    """
    blocks = control_blocks(controls, width)
    observed = partial_spearman(x, y, controls)
    members = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    if not np.isfinite(observed["rho"]):
        return {**observed, "p_conditional": 1.0,
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
            "blocks": int(len(members)),
            "block_width": int(width), "permutations": int(draws)}


def repeat_mean_rho(predictions: np.ndarray, target: np.ndarray) -> float:
    """The ranking metric: the mean over repeats of the out-of-fold
    rank correlation. ``predictions`` is (repeats, banks)."""
    return float(np.mean([_pearson(_ranks(row), _ranks(target))
                          for row in np.atleast_2d(predictions)]))


def bootstrap_repeat_mean(predictions: np.ndarray, target: np.ndarray, *,
                          draws: int = 1000, seed: int = 0) -> dict:
    """A bank-resampling interval for the ranking metric itself, so the
    variation that bands a candidate is the variation of the quantity it was
    ranked by."""
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
    the same resample scores both, so the variation they share cancels."""
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
