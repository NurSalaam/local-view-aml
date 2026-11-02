"""Regression models on a bank's properties: candidates, fitting and folds.

A regression model is linear in transformed property values, and its
coefficients are the only fitted quantities, so a bank plugs its own numbers
into the formula and needs no reference population. That is what makes the
model usable by a bank, and it is why nothing here standardises or centres a
property.

Data. A candidate is a name and an ordered tuple of property names. A design
matrix is (banks, properties) of already-transformed values in the candidate's
order; a target is (banks,) of measured shortfalls; weights are (banks,) of
1/s_B². A fitted model is the intercept and one coefficient per property, in
the candidate's order — small enough to print and apply by hand.

Fold identity is a function of the repeat index alone: repeat *r* permutes the
roster's canonical order with ``numpy.random.default_rng(r)`` and cuts the
permutation into contiguous blocks. Every candidate therefore sees the same
folds, which is what makes the selection budget identical without an audit.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

VERSION = 3
N_FOLDS = 10
REPEATS = 10


class Candidate(NamedTuple):
    name: str
    descriptors: tuple[str, ...]


def candidates(groups: dict[str, tuple[str, ...]], baseline: str) -> tuple[Candidate, ...]:
    """The construction: the baseline group alone, then each other
    group alone, then the baseline plus one further property."""
    others = [name for name in groups if name != baseline]
    built = [Candidate(baseline, groups[baseline])]
    built += [Candidate(name, groups[name]) for name in others]
    built += [Candidate(f"{baseline}+{descriptor}", groups[baseline] + (descriptor,))
              for name in others for descriptor in groups[name]]
    return tuple(built)


def constant_columns(design: np.ndarray) -> np.ndarray:
    """Columns with no variation across the fitting roster. Such a column is
    collinear with the intercept, so it is dropped and its coefficient recorded
    as zero rather than left to a minimum-norm solver to split arbitrarily."""
    return np.ptp(design, axis=0) == 0.0


def fit(design: np.ndarray, target: np.ndarray, weights: np.ndarray,
        keep: np.ndarray) -> tuple[float, np.ndarray]:
    """Weighted least squares of ``target`` on ``design[:, keep]`` plus an
    intercept. Returns the intercept and a full-width coefficient vector, zero
    in the dropped positions."""
    root = np.sqrt(weights)[:, None]
    columns = np.column_stack([np.ones(len(target)), design[:, keep]])
    solution, *_ = np.linalg.lstsq(columns * root, target * root[:, 0], rcond=None)
    coefficients = np.zeros(design.shape[1])
    coefficients[keep] = solution[1:]
    return float(solution[0]), coefficients


def predict(design: np.ndarray, intercept: float, coefficients: np.ndarray) -> np.ndarray:
    return intercept + design @ coefficients


def fold_ids(k: int, repeat: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold membership of every bank under one repeat: the roster's canonical
    order permuted at seed ``repeat``, cut into contiguous blocks."""
    permutation = np.random.default_rng(repeat).permutation(k)
    ids = np.empty(k, np.int64)
    ids[permutation] = np.arange(k) * n_folds // k
    return ids


def out_of_fold(design: np.ndarray, target: np.ndarray, weights: np.ndarray, *,
                repeats: int = REPEATS, n_folds: int = N_FOLDS) -> np.ndarray:
    """Every bank's prediction from a fit that never saw it, once per repeat;
    returns (repeats, banks). Which columns are constant is decided once on the
    whole fitting roster, so every fold fits the same model shape."""
    keep = ~constant_columns(design)
    k = len(target)
    predictions = np.empty((repeats, k))
    for repeat in range(repeats):
        ids = fold_ids(k, repeat, n_folds)
        for fold in range(n_folds):
            held = ids == fold
            intercept, coefficients = fit(design[~held], target[~held],
                                          weights[~held], keep)
            predictions[repeat, held] = predict(design[held], intercept, coefficients)
    return predictions
