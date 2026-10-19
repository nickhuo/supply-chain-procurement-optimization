# Raw-Material Ordering and Transshipment Optimization

Our solution to **CUMCM 2021 Problem C** (China Undergraduate Mathematical Contest in
Modeling, provincial **First Prize**), rebuilt in Python from the original MATLAB.

**What it solves.** A manufacturer makes 28,200 m³ of board a week from three raw materials
(A, B and C), wants two weeks of production in stock, and buys from 402 unreliable suppliers.
Eight transporters carry the material, each with weekly capacity and a transit loss rate. Using
five years of order, delivery and loss data, the project answers four questions:

| Question | Method |
|---|---|
| 1. Which 50 suppliers matter most? | 5 supply indicators weighted by **AHP**, scored with **TOPSIS** |
| 2. What is the fewest number of suppliers that can meet demand, and what are the cheapest orders and the least-loss routing for 24 weeks? | **0-1 integer program**, then two **linear programs** |
| 3. How can the plan buy more A, less C, and lose less in transit? | **Goal program** that orders and routes together |
| 4. How much can weekly capacity grow? | **LP** that maximises the capacity increase |

Supplier capacity and transit loss for the next 24 weeks are forecast from the 24-week
seasonality in the data. Every plan is stress-tested with a **Monte Carlo** simulation of
supplier noise. The models are built with [linopy](https://github.com/PyPSA/linopy) and solved
with [HiGHS](https://highs.dev). A full run takes about 5 seconds.

**What the results show**

- **Few suppliers matter.** The top 50 suppliers (12% of 402) deliver **97.2%** of all
  material, and **21** suppliers are enough to meet demand.
- **The history is badly short.** Between them, the 402 suppliers fell behind production in
  **175 of 240** past weeks. Every plan here reaches the two-week buffer by week 3 and keeps it,
  to within rounding (Problem 4 targets 1.95 weeks, as the paper does).
- **Joint planning pays off.** Problem 3 cuts type C orders by **92%** and lowers expected
  transit loss by **30%** compared with planning orders and routes separately (Problem 2).
- **Capacity can grow.** Weekly output can rise by **434.6 m³** (+1.5%) with the current
  suppliers and transporters.
- **The replication is verified.** It matches the published supplier ranking (same order,
  scores to 6 decimals), the minimum supplier count and the capacity increase, and tests pin
  these numbers. It also fixes an infeasible transport step in the original code at no extra cost.

**Limitations**

- **Optimistic capacity.** Capacity is each supplier's best delivery in that week of past
  cycles. That is an upper bound, and it assumes no trend, yet 12 of the top 50 suppliers show
  a significant trend (p < 0.05).
- **Plans assume orders arrive in full.** Noise is only simulated afterwards. Problem 3 keeps
  its buffer tight, so it averages **4.9** simulated short weeks out of 24, against 3.0 for Problem 2.
- **Problem 2 orders from all 50 top suppliers,** not only the 21 that the integer program says suffice.
- **The single-transporter rule is ignored.** The model does not enforce the rule that a
  supplier's weekly delivery should preferably use a single transporter.
- **Weights are judgement calls.** The AHP comparisons are subjective, and the goal program
  weights its three goals equally.

## Results

| | Problem 2 | Problem 3 | Problem 4 |
|---|---|---|---|
| Weekly capacity (m³) | 28,200 | 28,200 | 28,634.6 |
| Ordered over 24 weeks, A / B / C (k m³) | 202 / 78 / 200 | 307 / 132 / 15 | 310 / 134 / 16 |
| Expected transit loss (m³) | 3,143 | 2,196 | 2,288 |
| Simulated short weeks (of 24) | 3.0 | 4.9 | 5.1 |

The cheapest 24-week purchase cost is 528,612.75, in units of the type C price. All numbers
are in [`outputs/summary.json`](outputs/summary.json).

![Order mix by scenario](figures/plan_mix.png)
![Planned stock in weeks of production](figures/inventory.png)

## Method

**Supplier scoring.** Orders with no delivery are treated as recording errors. Each supplier
gets five indicators: total supply, valid orders, fulfillment rate, valid-order rate, and
fluctuation $\frac{1}{E_i}\sum_j |O_{ij}-S_{ij}|/O_{ij}$. AHP turns a pairwise comparison matrix
into weights (0.40 / 0.11 / 0.17 / 0.13 / 0.19, CR = 0.009). Weighted TOPSIS then ranks
suppliers by their closeness to an ideal supplier.

**Forecast.** Deliveries repeat every 24 weeks, the same length as the plan:

![Seasonality](figures/seasonality.png)

Capacity is $\mathrm{cap}_{iw} = \max_p S_{i,24p+w}$. Transit loss $\ell_{tw}$ is the mean of
past non-zero losses, and a transporter with no history in a week is treated as unavailable.

**Models.** $x_{iw}$ is the volume ordered from supplier $i$ in week $w$, with
$0 \le x_{iw} \le \mathrm{cap}_{iw}$. $z_{iwt}$ is the share of it that transporter $t$ carries.
$m_i \in \{0.60, 0.66, 0.72\}$ is the material needed per m³ of product. Demand is cumulative
production plus the two-week buffer from week 3 on,
$R_k = 28200\,k + 2\cdot 28200\cdot[k \ge 3]$:

$$\sum_{w\le k}\sum_i x_{iw}/m_i \ge R_k,\qquad \sum_t z_{iwt} = x_{iw},\qquad \sum_i z_{iwt} \le 6000$$

| Problem | Objective |
|---|---|
| 2a, fewest suppliers | $\min \sum_i y_i$, with $y_i \in \{0,1\}$ and every selected supplier at full capacity |
| 2b, cheapest orders | $\min \sum p_i x_{iw}$, where $p_i$ is 1.2 / 1.1 / 1.0 for A / B / C |
| 2c, least-loss routing | $\min \sum \ell_{tw} z_{iwt}$ for the orders fixed in 2b |
| 3, goal program | $\min d_1^- + d_2^+ + \sum \ell_{tw} z_{iwt}$, where $d_1^-$ is the volume that is not A and $d_2^+$ is the volume of C |
| 4, capacity growth | $\max D$, with $R_k$ raised by $(k+2)D$ |

**Simulation.** Each supplier's history is split into level × seasonal index × noise. Each plan
is replayed 100 times with normal noise, counting the weeks in which cumulative deliveries fall
behind production.

## Running it

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run supplychain   # solve Problems 1-4, write outputs/ and figures/
uv run pytest        # check the reproduced numbers
```

The outputs are the supplier ranking (`supplier_ranking.csv`), order plans and transport plans
in the competition's answer-sheet layout (`order_plans.xlsx`, `transport_plans.xlsx`), and
`summary.json`. Source is in `src/supplychain/`, with one module per step. The original problem
statement and paper (in Chinese) are in `docs/`.
