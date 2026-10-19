"""Write result tables, plan workbooks and figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .data import ROOT, WEEKLY_CAPACITY, Dataset, cumulative_requirement  # noqa: E402
from .pipeline import Q4_BUFFER_WEEKS, Results  # noqa: E402

OUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, validated all-pairs
MATERIAL_COLOR = dict(zip("ABC", SERIES))
SCENARIOS = {"Problem 2": "q2", "Problem 3": "q3", "Problem 4": "q4"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "font.size": 10,
        "lines.linewidth": 2,
        "legend.frameon": False,
        "legend.labelcolor": INK,
    }
)


def _orders(r: Results) -> dict[str, pd.DataFrame]:
    return {"q2": r.q2_order, "q3": r.q3_order, "q4": r.q4_order}


def _transports(r: Results):
    return {"q2": r.q2_transport, "q3": r.q3.transport, "q4": r.q4.transport}


def summary(r: Results, ds: Dataset) -> dict:
    by_material = {
        key: order.groupby(ds.material.loc[order.index]).sum().sum(axis=1).round().to_dict()
        for key, order in _orders(r).items()
    }
    return {
        "q1": {
            "ahp_weights": r.ahp.weights.round(4).to_dict(),
            "ahp_consistency_ratio": round(r.ahp.consistency_ratio, 4),
            "top50": list(r.top),
            "top50_supply_share": round(r.top_share, 4),
            "suppliers_with_trend_p05": len(r.trending),
        },
        "q2": {
            "min_suppliers": len(r.min_set),
            "min_supplier_set": list(r.min_set),
            "purchase_cost_lp": round(r.q2_cost, 2),
            "suppliers_ordered_from": int((r.q2_order.sum(axis=1) > 0).sum()),
            "transit_loss_m3": round(r.q2_transport.loss, 1),
        },
        "q3": {
            "deviations": {k: round(v, 1) for k, v in r.q3.deviations.items()},
            "transit_loss_m3": round(r.q3.transport.loss, 1),
        },
        "q4": {
            "capacity_increase_m3_per_week": round(r.q4_increase, 3),
            "new_weekly_capacity": round(WEEKLY_CAPACITY + r.q4_increase, 3),
            "buffer_weeks_used_for_plan": Q4_BUFFER_WEEKS,
            "transit_loss_m3": round(r.q4.transport.loss, 1),
        },
        "order_volume_by_material_m3": by_material,
        "shortfall_weeks": r.shortfall,
    }


def write_tables(r: Results, ds: Dataset, out: Path = OUT_DIR) -> None:
    out.mkdir(exist_ok=True)
    r.ranking.to_csv(out / "supplier_ranking.csv", float_format="%.6g")

    # Same layout as the competition templates: every supplier, blank = no order.
    with pd.ExcelWriter(out / "order_plans.xlsx") as xl:
        for name, key in SCENARIOS.items():
            plan = _orders(r)[key].reindex(ds.supplies.index).fillna(0)
            plan.columns = [f"Week {w:02d}" for w in plan.columns]
            plan.replace(0, np.nan).to_excel(xl, sheet_name=name)
    with pd.ExcelWriter(out / "transport_plans.xlsx") as xl:
        for name, key in SCENARIOS.items():
            plan = _transports(r)[key].as_frame().round(2).reindex(ds.supplies.index).fillna(0)
            plan.columns = pd.MultiIndex.from_tuples([(f"Week {w:02d}", t) for w, t in plan.columns])
            plan.where(plan > 0).to_excel(xl, sheet_name=name)

    (out / "summary.json").write_text(json.dumps(summary(r, ds), indent=2) + "\n")


def _label_end(ax, x, y, text, color=INK):
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", color=color, fontsize=9)


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_history(ds: Dataset, path: Path):
    """Weekly product volume the company ordered vs what suppliers delivered."""
    weeks = ds.supplies.columns
    ordered = ds.to_production(ds.orders).sum()
    supplied = ds.to_production(ds.supplies).sum()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(weeks, ordered / 1e3, color=SERIES[0], linewidth=1.5, label="Ordered")
    ax.plot(weeks, supplied / 1e3, color=SERIES[1], linewidth=1.5, label="Delivered")
    ax.axhline(WEEKLY_CAPACITY / 1e3, color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(2, WEEKLY_CAPACITY / 1e3 - 1.5, "Weekly capacity 28.2k", color=INK_2, fontsize=9, va="top")
    ax.set(xlim=(1, 240), xlabel="Week", ylabel="Product equivalent (k m³)")
    ax.set_title("Five years of orders and deliveries, in weeks of production")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)


def fig_top50(ds: Dataset, r: Results, path: Path):
    weeks = ds.supplies.columns
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(weeks, ds.supplies.sum() / 1e3, color=SERIES[0], linewidth=1.5, label="All 402 suppliers")
    ax.plot(weeks, ds.supplies.loc[r.top].sum() / 1e3, color=SERIES[1], linewidth=1.5, label="Top 50 suppliers")
    ax.set(xlim=(1, 240), xlabel="Week", ylabel="Delivered (k m³)")
    ax.set_title(f"The top 50 suppliers deliver {r.top_share:.2%} of all material")
    ax.legend(loc="upper left", ncol=2)
    _save(fig, path)


def fig_seasonality(ds: Dataset, path: Path, suppliers=("S139", "S140")):
    fig, axes = plt.subplots(len(suppliers), 1, figsize=(10, 4.6), sharex=True)
    for ax, s in zip(axes, suppliers):
        ax.plot(ds.supplies.columns, ds.supplies.loc[s], color=SERIES[0], linewidth=1.5)
        for b in range(24, 240, 24):
            ax.axvline(b + 0.5, color=GRID, linewidth=1)
        ax.set_ylabel("m³")
        ax.set_title(f"{s} weekly deliveries (gridlines mark 24-week cycles)", fontsize=10)
    axes[-1].set(xlim=(1, 240), xlabel="Week")
    _save(fig, path)


def fig_plan_mix(r: Results, ds: Dataset, path: Path):
    """Ordered volume by material for each scenario, as stacked horizontal bars."""
    mix = pd.DataFrame(
        {name: _orders(r)[key].groupby(ds.material.loc[r.top]).sum().sum(axis=1) for name, key in SCENARIOS.items()}
    ).T.reindex(columns=list("ABC")).fillna(0)
    fig, ax = plt.subplots(figsize=(10, 2.8))
    left = np.zeros(len(mix))
    for m in mix.columns:
        widths = mix[m].to_numpy() / 1e3
        ax.barh(mix.index, widths, left=left, color=MATERIAL_COLOR[m], height=0.55,
                edgecolor=SURFACE, linewidth=2, label=f"Type {m}")
        for y, (x0, w) in enumerate(zip(left, widths)):
            if w > 25:
                ax.text(x0 + w / 2, y, f"{m} {w:.0f}k", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        left += widths
    for y, total in enumerate(left):
        _label_end(ax, total, y, f"{total:.0f}k m³", INK_2)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set(xlabel="Ordered over 24 weeks (k m³)", xlim=(0, left.max() * 1.12))
    ax.set_title("Problem 3's goal program shifts orders from type C to type A")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_inventory(r: Results, ds: Dataset, path: Path):
    """Planned stock beyond current production need, in weeks of production."""
    cons = ds.consumption.loc[r.top]
    capacities = {"q2": WEEKLY_CAPACITY, "q3": WEEKLY_CAPACITY, "q4": WEEKLY_CAPACITY + r.q4_increase}
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for color, (name, key) in zip(SERIES, SCENARIOS.items()):
        order = _orders(r)[key]
        cap = capacities[key]
        stock = (order.div(cons, axis=0).sum().cumsum() - cumulative_requirement(capacity=cap, buffer_weeks=0)) / cap
        ax.plot(stock.index, stock, color=color, marker="o", markersize=4, label=name)
    ax.axhline(2, color=INK_2, linestyle="--", linewidth=1.2)
    ax.text(24.4, 1.93, "Two-week buffer", color=INK_2, fontsize=9, va="top", ha="right")
    ax.set(xlim=(0.5, 24.5), xlabel="Planned week", ylabel="Stock (weeks of production)")
    ax.set_title("Planned stock reaches the two-week buffer by week 3 (Problem 4 targets 1.95)")
    ax.legend(loc="upper left", ncol=3)
    _save(fig, path)


def write_figures(r: Results, ds: Dataset, out: Path = FIG_DIR) -> None:
    out.mkdir(exist_ok=True)
    fig_history(ds, out / "history.png")
    fig_top50(ds, r, out / "top50_share.png")
    fig_seasonality(ds, out / "seasonality.png")
    fig_plan_mix(r, ds, out / "plan_mix.png")
    fig_inventory(r, ds, out / "inventory.png")
