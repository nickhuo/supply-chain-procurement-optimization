"""Monte Carlo check of an order plan under supplier delivery noise."""

import numpy as np
import pandas as pd

from .data import WEEKLY_CAPACITY
from .forecast import NoiseModel


def shortfall_weeks(
    order: pd.DataFrame,
    consumption: pd.Series,
    noise: NoiseModel,
    capacity: float = WEEKLY_CAPACITY,
    runs: int = 100,
    seed: int = 2021,
) -> float:
    """Average number of weeks in which cumulative deliveries fall behind production.

    Each run scales every order by a draw of that supplier's irregular
    component, converts deliveries to product volume, and counts the weeks k
    where the running total is below k * capacity.
    """
    rng = np.random.default_rng(seed)
    product = order.div(consumption.loc[order.index], axis=0).to_numpy()
    k = np.arange(1, order.shape[1] + 1)
    counts = [
        (np.cumsum((noise.sample(order.index, order.shape[1], rng) * product).sum(0)) < capacity * k).sum()
        for _ in range(runs)
    ]
    return float(np.mean(counts))


def historical_shortfall_weeks(production: pd.DataFrame, capacity: float = WEEKLY_CAPACITY) -> int:
    """Weeks of the 240-week history where cumulative supply was behind production."""
    k = np.arange(1, production.shape[1] + 1)
    return int((production.sum().cumsum().to_numpy() < capacity * k).sum())
