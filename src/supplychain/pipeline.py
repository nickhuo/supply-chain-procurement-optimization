"""Run Problems 1-4 end to end."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import forecast, planning, scoring, simulation
from .data import WEEKLY_CAPACITY, Dataset, load

TOP_N = 50
Q4_BUFFER_WEEKS = 1.95  # the paper relaxes the buffer to make the Q4 plan feasible


@dataclass(frozen=True)
class Results:
    ranking: pd.DataFrame
    ahp: scoring.AhpResult
    top: pd.Index
    top_share: float
    trending: list[str]
    capacity: pd.DataFrame
    loss: pd.DataFrame
    min_set: pd.Index
    q2_cost: float
    q2_order: pd.DataFrame
    q2_transport: planning.TransportPlan
    q3: planning.JointPlan
    q3_order: pd.DataFrame
    q4_increase: float
    q4: planning.JointPlan
    q4_order: pd.DataFrame
    shortfall: dict[str, float]


def run(ds: Dataset | None = None, runs: int = 100) -> Results:
    ds = ds or load()

    # Problem 1
    ranking = scoring.rank_suppliers(ds)
    top = ranking.index[:TOP_N]
    top_share = ds.supplies.loc[top].to_numpy().sum() / ds.supplies.to_numpy().sum()

    # Forecast inputs for the next cycle
    cap = forecast.capacity_upper_bound(ds.supplies.loc[top])
    loss = forecast.expected_loss_rates(ds.loss_rates)
    cons, price, material = ds.consumption.loc[top], ds.price.loc[top], ds.material.loc[top]

    # Problem 2
    min_set = planning.min_supplier_set(cap, cons)
    q2 = planning.cheapest_order_plan(cap, cons, price, planning.open_capacity(loss))
    q2_order = np.floor(q2.order + 1e-6)
    q2_transport = planning.min_loss_transport(q2_order, loss)

    # Problem 3
    q3 = planning.goal_program(cap, cons, material, loss)
    q3_order = q3.order.order.round()

    # Problem 4
    increase = planning.max_capacity_increase(cap, cons, loss)
    q4 = planning.goal_program(cap, cons, material, loss, WEEKLY_CAPACITY + increase, Q4_BUFFER_WEEKS)
    q4_order = q4.order.order.round()

    noise = forecast.fit_noise(ds.supplies.loc[top])
    shortfall = {
        "history": simulation.historical_shortfall_weeks(ds.to_production(ds.supplies)),
        "q2": simulation.shortfall_weeks(q2_order, cons, noise, runs=runs),
        "q3": simulation.shortfall_weeks(q3_order, cons, noise, runs=runs),
        "q4": simulation.shortfall_weeks(q4_order, cons, noise, WEEKLY_CAPACITY + increase, runs=runs),
    }

    return Results(
        ranking=ranking,
        ahp=scoring.ahp(),
        top=top,
        top_share=top_share,
        trending=forecast.trending_series(ds.supplies.loc[top]),
        capacity=cap,
        loss=loss,
        min_set=min_set,
        q2_cost=q2.objective,
        q2_order=q2_order,
        q2_transport=q2_transport,
        q3=q3,
        q3_order=q3_order,
        q4_increase=increase,
        q4=q4,
        q4_order=q4_order,
        shortfall=shortfall,
    )
