"""Forecasts for the next 24-week cycle.

Deliveries repeat on a 24-week cycle with no significant trend, so the 240
weeks of history are folded into ten cycles. Supplier capacity for week w of
the next cycle is the largest delivery seen in week w of any past cycle;
transporter loss for week w is the mean of its non-zero past losses.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data import CYCLE_WEEKS


def fold(frame: pd.DataFrame, cycle: int = CYCLE_WEEKS) -> np.ndarray:
    """Reshape rows x (cycles*cycle) to rows x cycles x cycle."""
    return frame.to_numpy().reshape(len(frame), -1, cycle)


def _cycle_frame(values: np.ndarray, index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(values, index=index, columns=pd.RangeIndex(1, values.shape[1] + 1, name="week"))


def capacity_upper_bound(supplies: pd.DataFrame) -> pd.DataFrame:
    """Best delivery each supplier achieved in each week of the cycle (m3)."""
    return _cycle_frame(fold(supplies).max(axis=1), supplies.index)


def expected_loss_rates(loss_rates: pd.DataFrame) -> pd.DataFrame:
    """Mean non-zero loss rate per transporter and cycle week; NaN if never active."""
    folded = fold(loss_rates)
    active = folded != 0
    with np.errstate(invalid="ignore"):
        mean = folded.sum(axis=1) / active.sum(axis=1)
    return _cycle_frame(np.where(active.any(axis=1), mean, np.nan), loss_rates.index)


def trending_series(supplies: pd.DataFrame, alpha: float = 0.05) -> list[str]:
    """Suppliers whose deliveries correlate with time at the given significance."""
    t = np.arange(supplies.shape[1])
    return [s for s, row in supplies.iterrows() if row.std() > 0 and stats.pearsonr(t, row).pvalue < alpha]


@dataclass(frozen=True)
class NoiseModel:
    """Multiplicative seasonal model Y = mean * season * I, with I ~ N(mu, sigma)."""

    mu: pd.Series
    sigma: pd.Series

    def sample(self, suppliers: pd.Index, weeks: int, rng: np.random.Generator) -> np.ndarray:
        return rng.normal(self.mu.loc[suppliers].to_numpy()[:, None], self.sigma.loc[suppliers].to_numpy()[:, None], (len(suppliers), weeks))


def fit_noise(supplies: pd.DataFrame) -> NoiseModel:
    series = supplies.to_numpy()
    level = series.mean(axis=1, keepdims=True)
    season = fold(supplies).mean(axis=1) / level
    season /= season.mean(axis=1, keepdims=True)
    season = np.tile(season, series.shape[1] // CYCLE_WEEKS)
    # The original replaces zeros by 0.1 before dividing, to avoid 0/0.
    irregular = np.where(series == 0, 0.1, series) / np.where(season == 0, 0.1, season) / level
    return NoiseModel(
        pd.Series(irregular.mean(axis=1), index=supplies.index),
        pd.Series(irregular.std(axis=1, ddof=1), index=supplies.index),
    )
