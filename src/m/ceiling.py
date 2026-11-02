"""The two quantities computed from labels, on which Experiment 3 builds the
models that use labels (Appendix D.4).

The models on label-free properties are what a bank could compute from its
ledger without reading a label. The two quantities here read the labels a bank
holds before its test window: its number of confirmed cases
(`laundering_count`) and the share of them whose two accounts are both at the
bank (`laundering_internal_share`).

Data. The whole dataset's training window arrives as parallel arrays in
canonical order: each payment's two bank identifiers and its label.
"""

from __future__ import annotations

import numpy as np

VERSION = 2

NAMES = ("laundering_count", "laundering_internal_share")

DESCRIPTIONS = {
    "laundering_count": "the bank's training-window laundering payments",
    "laundering_internal_share": "share of the bank's training-window laundering "
                                 "payments whose two endpoints both belong to it",
}


def quantities(from_bank: np.ndarray, to_bank: np.ndarray, laundering: np.ndarray,
               bank: str) -> dict[str, float]:
    """One bank's two quantities; the internal share is undefined at a bank
    with no confirmed case."""
    cases = ((from_bank == bank) | (to_bank == bank)) & laundering
    internal = (from_bank == bank) & (to_bank == bank) & laundering
    count = float(cases.sum())
    return {"laundering_count": count,
            "laundering_internal_share": float(internal.sum()) / count if count else np.nan}
