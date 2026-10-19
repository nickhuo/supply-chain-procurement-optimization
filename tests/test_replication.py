"""Pin the numbers this replication shares with the 2021 paper."""

import numpy as np
import pandas as pd
import pytest

from supplychain import pipeline, planning
from supplychain.data import TRANSPORTER_CAPACITY, cumulative_requirement, load

PAPER_TOP50 = [229, 361, 275, 108, 140, 340, 329, 282, 268, 151, 306, 131, 356, 194, 330, 308, 139, 352, 247, 143, 348, 395, 365, 284, 31, 201, 307, 364, 40, 374, 346, 367, 294, 55, 80, 37, 218, 266, 244, 123, 7, 338, 114, 150, 314, 126, 291, 86, 3, 189]  # fmt: skip
PAPER_TOP10_SCORES = [0.021369, 0.01956, 0.014278, 0.014137, 0.013818, 0.013506, 0.013494, 0.012907, 0.012008, 0.011864]  # fmt: skip


@pytest.fixture(scope="module")
def ds():
    return load()


@pytest.fixture(scope="module")
def results(ds):
    return pipeline.run(ds, runs=20)


def test_ahp_weights_match_paper(results):
    assert results.ahp.weights.round(2).tolist() == [0.40, 0.11, 0.17, 0.13, 0.19]
    assert results.ahp.consistency_ratio < 0.1


def test_top50_matches_paper_in_order(results):
    assert [int(s[1:]) for s in results.top] == PAPER_TOP50
    assert results.ranking["score"].iloc[:10].round(6).tolist() == PAPER_TOP10_SCORES


def test_top50_supply_share(results):
    assert round(results.top_share, 4) == 0.9721


def test_historical_shortfall_weeks(results):
    assert results.shortfall["history"] == 175


def test_min_supplier_count(results):
    assert len(results.min_set) == 21


def test_capacity_increase(results):
    assert results.q4_increase == pytest.approx(434.626, abs=1e-3)


def test_q2_cost_is_published_code_optimum(results):
    # The published code's LP optimum; the paper's text reports 570,600 (see README).
    assert results.q2_cost == pytest.approx(528_612.75, abs=0.01)


def test_plans_are_feasible(results, ds):
    cons = ds.consumption.loc[results.top]
    produced = results.q3.order.order.div(cons, axis=0).sum().cumsum().to_numpy()
    assert (produced >= cumulative_requirement() - 1e-6).all()
    for order, plan in [
        (results.q2_order, results.q2_transport),
        (results.q3.order.order, results.q3.transport),
        (results.q4.order.order, results.q4.transport),
    ]:
        assert (order <= results.capacity + 1e-6).to_numpy().all()
        assert (plan.flow.sum("supplier") <= TRANSPORTER_CAPACITY + 1e-6).all()
        assert np.allclose(plan.flow.sum("transporter").to_pandas(), order, atol=1e-6)


def test_q3_shifts_mix_toward_a(results, ds):
    material = ds.material.loc[results.top]
    share = lambda order, m: order[material == m].to_numpy().sum() / order.to_numpy().sum()  # noqa: E731
    assert share(results.q3_order, "A") > share(results.q2_order, "A")
    assert share(results.q3_order, "C") < share(results.q2_order, "C")
    assert results.q3.transport.loss < results.q2_transport.loss


def test_open_capacity_does_not_change_q2_cost(ds, results):
    """The paper capped every week at 8 x 6000; capping at the open fleet costs nothing extra."""
    cap = results.capacity
    full_fleet = pd.Series(8 * TRANSPORTER_CAPACITY, index=cap.columns)
    paper = planning.cheapest_order_plan(cap, ds.consumption.loc[results.top], ds.price.loc[results.top], full_fleet)
    assert paper.objective == pytest.approx(results.q2_cost, rel=1e-9)
