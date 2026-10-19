"""Problem constants and loaders for the CUMCM 2021 Problem C datasets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"

WEEKLY_CAPACITY = 28_200  # m3 of product per week
TRANSPORTER_CAPACITY = 6_000  # m3 of raw material per transporter per week
CYCLE_WEEKS = 24  # planning horizon; supply data repeats on this cycle
BUFFER_WEEKS = 2  # target inventory, in weeks of production

# m3 of raw material consumed per m3 of product
CONSUMPTION = {"A": 0.60, "B": 0.66, "C": 0.72}
# Unit purchase price relative to type C (A is 20% and B is 10% dearer)
PRICE = {"A": 1.2, "B": 1.1, "C": 1.0}


@dataclass(frozen=True)
class Dataset:
    orders: pd.DataFrame  # supplier x week, m3 ordered by the company
    supplies: pd.DataFrame  # supplier x week, m3 actually delivered
    material: pd.Series  # supplier -> "A" | "B" | "C"
    loss_rates: pd.DataFrame  # transporter x week, fraction lost in transit (0 = idle)

    @property
    def consumption(self) -> pd.Series:
        return self.material.map(CONSUMPTION)

    @property
    def price(self) -> pd.Series:
        return self.material.map(PRICE)

    def to_production(self, volumes: pd.DataFrame) -> pd.DataFrame:
        """Convert raw-material volumes (m3) to the product volume they can make."""
        return volumes.div(self.consumption.loc[volumes.index], axis=0)


def _weekly(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.astype(float)
    frame.columns = pd.RangeIndex(1, frame.shape[1] + 1, name="week")
    return frame


def load(raw_dir: Path = RAW_DIR) -> Dataset:
    sheets = pd.read_excel(raw_dir / "suppliers.xlsx", sheet_name=None, index_col=0)
    orders, supplies = sheets.values()  # sheet order: orders, then supplies
    material = orders.iloc[:, 0].rename("material")
    material.index.name = orders.index.name = supplies.index.name = "supplier"
    assert material.equals(supplies.iloc[:, 0].rename("material"))

    loss = pd.read_excel(raw_dir / "transporters.xlsx", index_col=0)
    loss.index.name = "transporter"
    return Dataset(
        orders=_weekly(orders.iloc[:, 1:]),
        supplies=_weekly(supplies.iloc[:, 1:]),
        material=material,
        loss_rates=_weekly(loss) / 100,
    )


def cumulative_requirement(
    weeks: int = CYCLE_WEEKS,
    capacity: float = WEEKLY_CAPACITY,
    buffer_weeks: float = BUFFER_WEEKS,
) -> np.ndarray:
    """Cumulative product volume that must be supplied by the end of each week.

    Production consumes `capacity` per week, and from week 3 on the company also
    holds `buffer_weeks` of production in stock (the original model exempts the
    first two weeks while the buffer is being built).
    """
    k = np.arange(1, weeks + 1)
    return capacity * k + np.where(k >= 3, buffer_weeks * capacity, 0.0)
