"""Problems 2-4: ordering and transshipment models, solved with HiGHS via linopy.

Shared notation (all over the next 24-week cycle):
    cap[s, w]   capacity upper bound of supplier s in week w (m3 raw material)
    order[s, w] quantity ordered from supplier s in week w, 0 <= order <= cap
    flow[s,w,t] quantity of supplier s's week-w delivery carried by transporter t
    loss[t, w]  expected loss rate of transporter t in week w (NaN = unavailable)
"""

from dataclasses import dataclass

import linopy
import numpy as np
import pandas as pd
import xarray as xr

from .data import BUFFER_WEEKS, TRANSPORTER_CAPACITY, WEEKLY_CAPACITY, cumulative_requirement


def _da(frame: pd.DataFrame | pd.Series) -> xr.DataArray:
    return xr.DataArray(frame)


def _week_da(values: np.ndarray, weeks: pd.Index) -> xr.DataArray:
    return xr.DataArray(values, coords=[weeks])


def _solve(m: linopy.Model) -> None:
    status, condition = m.solve("highs", io_api="direct", output_flag=False, log_to_console=False)
    if status != "ok":
        raise RuntimeError(f"solver failed: {status} / {condition}")


def _cumulative_production(order, consumption: pd.Series):
    return (order / _da(consumption)).sum("supplier").cumsum("week")


def open_capacity(loss: pd.DataFrame) -> pd.Series:
    """Weekly transport capacity of the transporters that have a loss history."""
    return loss.notna().sum() * TRANSPORTER_CAPACITY


# --- Problem 2 -------------------------------------------------------------


def min_supplier_set(cap: pd.DataFrame, consumption: pd.Series) -> pd.Index:
    """Smallest set of suppliers that can meet demand at full capacity (0-1 ILP)."""
    m = linopy.Model()
    pick = m.add_variables(binary=True, coords=[cap.index], name="pick")
    production = (pick * _da(cap.div(consumption, axis=0))).sum("supplier").cumsum("week")
    m.add_constraints(production >= _week_da(cumulative_requirement(), cap.columns), name="demand")
    m.add_objective(pick.sum())
    _solve(m)
    return cap.index[pick.solution.to_numpy() > 0.5]


@dataclass(frozen=True)
class OrderPlan:
    order: pd.DataFrame  # supplier x week, m3
    objective: float


def cheapest_order_plan(
    cap: pd.DataFrame, consumption: pd.Series, price: pd.Series, transport: pd.Series
) -> OrderPlan:
    """Minimise purchase cost subject to demand and weekly transport capacity (LP).

    `transport` is the weekly volume the fleet can carry. The paper used 8 x 6000
    for every week; passing `open_capacity(loss)` instead keeps the plan
    shippable when a transporter has no history for a week, at the same cost.
    """
    m = linopy.Model()
    order = m.add_variables(lower=0, upper=_da(cap), name="order")
    m.add_constraints(
        _cumulative_production(order, consumption) >= _week_da(cumulative_requirement(), cap.columns),
        name="demand",
    )
    m.add_constraints(order.sum("supplier") <= _da(transport), name="transport")
    m.add_objective((order * _da(price)).sum())
    _solve(m)
    return OrderPlan(order.solution.to_pandas(), m.objective.value)


@dataclass(frozen=True)
class TransportPlan:
    flow: xr.DataArray  # supplier x week x transporter, m3
    loss: float  # expected m3 lost in transit

    def as_frame(self) -> pd.DataFrame:
        """supplier x (week, transporter), matching the competition's template."""
        frame = self.flow.to_dataframe("m3")["m3"].unstack(["week", "transporter"])
        return frame.sort_index(axis=1)


def _add_flows(m: linopy.Model, suppliers: pd.Index, loss: pd.DataFrame):
    available = _da(loss.notna().T)  # week x transporter
    flow = m.add_variables(
        lower=0,
        upper=xr.where(available, np.inf, 0.0),
        coords=[suppliers, loss.columns, loss.index],
        name="flow",
    )
    m.add_constraints(flow.sum("supplier") <= TRANSPORTER_CAPACITY, name="fleet")
    expected_loss = (flow * _da(loss.fillna(0).T)).sum()
    return flow, expected_loss


def min_loss_transport(order: pd.DataFrame, loss: pd.DataFrame) -> TransportPlan:
    """Route a fixed order plan through the transporters with least expected loss (LP)."""
    m = linopy.Model()
    flow, expected_loss = _add_flows(m, order.index, loss)
    m.add_constraints(flow.sum("transporter") == _da(order), name="ship_all")
    m.add_objective(expected_loss)
    _solve(m)
    return TransportPlan(flow.solution, m.objective.value)


# --- Problems 3 and 4 ------------------------------------------------------


@dataclass(frozen=True)
class JointPlan:
    order: OrderPlan
    transport: TransportPlan
    deviations: dict[str, float]


def goal_program(
    cap: pd.DataFrame,
    consumption: pd.Series,
    material: pd.Series,
    loss: pd.DataFrame,
    capacity: float = WEEKLY_CAPACITY,
    buffer_weeks: float = BUFFER_WEEKS,
) -> JointPlan:
    """Order and route together, preferring type A, avoiding type C, and losing little.

    Goals with deviation variables (d-, d+), all weighted equally:
        G1  sum(order of A) + d1- - d1+ = sum(order)   minimise d1-  (as much A as possible)
        G2  sum(order of C) + d2- - d2+ = 0            minimise d2+  (as little C as possible)
        G3  expected transit loss                      minimise it directly
    """
    m = linopy.Model()
    order = m.add_variables(lower=0, upper=_da(cap), name="order")
    flow, expected_loss = _add_flows(m, cap.index, loss)
    m.add_constraints(flow.sum("transporter") == order, name="ship_all")
    requirement = cumulative_requirement(capacity=capacity, buffer_weeks=buffer_weeks)
    m.add_constraints(
        _cumulative_production(order, consumption) >= _week_da(requirement, cap.columns),
        name="demand",
    )

    d = {k: m.add_variables(lower=0, name=k) for k in ["d1_minus", "d1_plus", "d2_minus", "d2_plus"]}
    is_a, is_c = (_da((material == x).astype(float)) for x in "AC")
    m.add_constraints((order * is_a).sum() + d["d1_minus"] - d["d1_plus"] - order.sum() == 0, name="goal_A")
    m.add_constraints((order * is_c).sum() + d["d2_minus"] - d["d2_plus"] == 0, name="goal_C")
    m.add_objective(d["d1_minus"] + d["d2_plus"] + expected_loss)
    _solve(m)

    return JointPlan(
        OrderPlan(order.solution.to_pandas(), m.objective.value),
        TransportPlan(flow.solution, float(expected_loss.solution)),
        {k: float(v.solution) for k, v in d.items()},
    )


def max_capacity_increase(cap: pd.DataFrame, consumption: pd.Series, loss: pd.DataFrame) -> float:
    """Largest weekly capacity increase D the suppliers and fleet can sustain (LP).

    As in the original, the extra demand enters week k as (k + 2) * D, i.e.
    production plus a two-week buffer, including in weeks 1 and 2.
    """
    m = linopy.Model()
    order = m.add_variables(lower=0, upper=_da(cap), name="order")
    flow, _ = _add_flows(m, cap.index, loss)
    m.add_constraints(flow.sum("transporter") == order, name="ship_all")
    extra = m.add_variables(lower=0, name="extra_capacity")
    k = _week_da(np.arange(1, len(cap.columns) + 1, dtype=float), cap.columns)
    m.add_constraints(
        _cumulative_production(order, consumption) - (k + 2) * extra
        >= _week_da(cumulative_requirement(), cap.columns),
        name="demand",
    )
    m.add_objective(-1 * extra)
    _solve(m)
    return float(extra.solution)
