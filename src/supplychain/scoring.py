"""Problem 1: rank suppliers by how much they matter to production.

Five supply indicators are extracted from five years of order and delivery
history, weighted with the Analytic Hierarchy Process (AHP), and combined into
one score with TOPSIS.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import Dataset

INDICATORS = [
    "total_supply",  # m3 delivered over 240 weeks (scale)
    "valid_orders",  # orders that were delivered (activity)
    "fulfillment_rate",  # delivered / ordered (reliability)
    "valid_order_rate",  # valid / (valid + invalid) orders (robustness)
    "fluctuation",  # mean |delivered - ordered| / ordered (lower is better)
]

# Pairwise comparison matrix from the paper, in INDICATORS order.
AHP_MATRIX = np.array(
    [
        [1, 3, 3, 3, 2],
        [1 / 3, 1, 2 / 3, 2 / 3, 1 / 2],
        [1 / 3, 3 / 2, 1, 3 / 2, 1],
        [1 / 3, 3 / 2, 2 / 3, 1, 2 / 3],
        [1 / 2, 2, 1, 3 / 2, 1],
    ]
)
# Saaty's random consistency index, indexed by matrix size.
RANDOM_INDEX = [0, 0, 0.0001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49]


def indicators(ds: Dataset) -> pd.DataFrame:
    orders, supplies = ds.orders.to_numpy(), ds.supplies.to_numpy()

    # An order with nothing delivered is treated as a recording error ("invalid")
    # and zeroed out. Following the original code, a "valid" order is one with a
    # delivery that differs from the order, so exact fills are not counted.
    invalid = (orders > supplies) & (supplies == 0)
    valid = (orders != supplies) & (supplies != 0)
    orders = np.where(invalid, 0, orders)

    with np.errstate(divide="ignore", invalid="ignore"):
        deviation = np.where(supplies != 0, np.abs(supplies - orders) / orders, 0)
        frame = pd.DataFrame(
            {
                "total_supply": supplies.sum(1),
                "valid_orders": valid.sum(1),
                "fulfillment_rate": supplies.sum(1) / orders.sum(1),
                "valid_order_rate": valid.sum(1) / (valid.sum(1) + invalid.sum(1)),
                "fluctuation": deviation.sum(1) / valid.sum(1),
            },
            index=ds.supplies.index,
        )
    return frame


@dataclass(frozen=True)
class AhpResult:
    weights: pd.Series
    lambda_max: float
    consistency_ratio: float


def ahp(matrix: np.ndarray = AHP_MATRIX, names: list[str] = INDICATORS) -> AhpResult:
    """Principal-eigenvector weights and Saaty's consistency ratio."""
    n = len(matrix)
    eigvals, eigvecs = np.linalg.eig(matrix)
    k = np.argmax(eigvals.real)
    lam = eigvals[k].real
    vec = eigvecs[:, k].real
    ci = (lam - n) / (n - 1)
    return AhpResult(pd.Series(vec / vec.sum(), index=names), lam, ci / RANDOM_INDEX[n])


def topsis(values: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Weighted TOPSIS closeness, normalised to sum to 1.

    Every column must already be "larger is better".
    """
    z = values / np.sqrt((values**2).sum())
    w = weights.loc[values.columns]
    d_best = np.sqrt((w * (z - z.max()) ** 2).sum(axis=1))
    d_worst = np.sqrt((w * (z - z.min()) ** 2).sum(axis=1))
    closeness = d_worst / (d_best + d_worst)
    return closeness / closeness.sum()


def rank_suppliers(ds: Dataset) -> pd.DataFrame:
    """All suppliers with their indicators and score, best first."""
    raw = indicators(ds)
    benefit = raw.assign(fluctuation=1 / raw["fluctuation"])  # reciprocal, as in the paper
    score = topsis(benefit, ahp().weights)
    ranked = raw.assign(material=ds.material, score=score).sort_values(
        "score", ascending=False, kind="stable"
    )
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked
