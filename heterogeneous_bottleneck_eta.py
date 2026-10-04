#!/usr/bin/env python3
"""Minimal four-type robustness check for the Bay Bridge bottleneck model.

Computes only:
  1. approximately revenue-optimal static toll,
  2. an upper bound on unrestricted dynamic-toll revenue, and
  3. a first-best lower bound on dynamically achievable system cost.

Dependencies: numpy, pandas, scipy
"""

from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog, minimize_scalar

ETA_VALUES = [1.5, 2.1, 2.5, 3.0, 4.0, 5.0]


def make_case(eta, h_v=0.20, h_s=0.20, dt_minutes=15):
    """Build the four-type Bay Bridge discretization."""
    dt = dt_minutes / 60.0
    demand, capacity = 70_000.0, 9_600.0
    desired = np.arange(5.0 + dt / 2, 10.0, dt)

    alpha_bar = 22.0
    beta_e_bar, beta_l_bar = 0.61 * alpha_bar, 2.40 * alpha_bar
    alpha = np.array([
        alpha_bar * (1 - h_v), alpha_bar * (1 - h_v),
        alpha_bar * (1 + h_v), alpha_bar * (1 + h_v),
    ])
    beta_e = np.array([
        beta_e_bar * (1 - h_s), beta_e_bar * (1 + h_s),
        beta_e_bar * (1 - h_s), beta_e_bar * (1 + h_s),
    ])
    beta_l = np.array([
        beta_l_bar * (1 - h_s), beta_l_bar * (1 + h_s),
        beta_l_bar * (1 - h_s), beta_l_bar * (1 + h_s),
    ])

    # Bay Bridge generalized-cost calibration used in the paper.
    parking, transit_fare = 30.0, 6.14
    car_time = 21.0 / 60.0
    transit_time = (20.0 + 10.0 + 32.0) / 60.0
    car_cost_type = parking + alpha * car_time
    transit_cost_type = transit_fare + alpha * eta * transit_time
    gap_type = transit_cost_type - car_cost_type

    # Extend the crossing horizon until no type would drive farther out even
    # with zero toll and zero queue.
    max_gap = max(float(gap_type.max()), 0.0)
    early_radius = max_gap / beta_e.min() + 0.5
    late_radius = max_gap / beta_l.min() + 0.5
    service_start = np.floor((5.0 - early_radius) / dt) * dt
    service_end = np.ceil((10.0 + late_radius) / dt) * dt
    service = np.arange(service_start + dt / 2, service_end, dt)

    # A group is a (preference type, desired-arrival-time bin) pair.
    n_desired = len(desired)
    group_type = np.repeat(np.arange(4), n_desired)
    group_desired = np.tile(desired, 4)
    group_mass = np.full(4 * n_desired, demand / (4 * n_desired))

    alpha_g = alpha[group_type]
    beta_e_g, beta_l_g = beta_e[group_type], beta_l[group_type]
    gap_g, car_cost_g = gap_type[group_type], car_cost_type[group_type]

    early = np.maximum(group_desired[:, None] - service[None, :], 0.0)
    late = np.maximum(service[None, :] - group_desired[:, None], 0.0)
    schedule = beta_e_g[:, None] * early + beta_l_g[:, None] * late
    value = gap_g[:, None] - schedule  # zero-queue willingness to pay

    G, S = len(group_mass), len(service)
    slot_capacity = np.full(S, capacity * dt)
    A_cap_x = sparse.kron(np.ones((1, G)), sparse.eye(S), format="csr")
    A_dem_x = sparse.kron(sparse.eye(G), np.ones((1, S)), format="csr")
    A_cap = sparse.hstack([A_cap_x, sparse.csr_matrix((S, G))], format="csr")
    A_dem = sparse.hstack([A_dem_x, sparse.eye(G)], format="csr")

    return dict(
        alpha_type=alpha, gap_type=gap_type,
        alpha=alpha_g, gap=gap_g, car_cost=car_cost_g,
        schedule=schedule, value=value, group_mass=group_mass,
        slot_capacity=slot_capacity, G=G, S=S,
        A_cap_x=A_cap_x, A_dem_x=A_dem_x, A_cap=A_cap, A_dem=A_dem,
    )


def equilibrium(case, toll):
    """Finite-grid heterogeneous equilibrium for a static toll."""
    G, S = case["G"], case["S"]
    car_obj = (case["schedule"] + toll) / case["alpha"][:, None]
    transit_obj = case["gap"] / case["alpha"]
    res = linprog(
        np.r_[car_obj.ravel(), transit_obj],
        A_ub=case["A_cap"], b_ub=case["slot_capacity"],
        A_eq=case["A_dem"], b_eq=case["group_mass"],
        bounds=(0, None), method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    x = res.x[: G * S].reshape(G, S)
    y = res.x[G * S :]
    w = -np.asarray(res.ineqlin.marginals)  # hours
    return dict(toll=toll, revenue=toll * x.sum(), x=x, y=y, w=w)


def optimal_static(case, n_coarse=101, tol=0.005, eps=1e-5):
    """Coarse global search followed by local refinement."""
    p_max = max(0.0, float(case["gap_type"].max()))
    if p_max == 0:
        return equilibrium(case, 0.0)

    grid = np.linspace(0.0, p_max, n_coarse)
    vals = [equilibrium(case, p) for p in grid]
    i = int(np.argmax([z["revenue"] for z in vals]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_coarse - 1)]
    opt = minimize_scalar(
        lambda p: -equilibrium(case, p)["revenue"],
        bounds=(lo, hi), method="bounded", options={"xatol": tol},
    )
    prices = [lo, hi, grid[i], opt.x,
              max(0.0, opt.x - eps), max(0.0, hi - eps),
              max(0.0, grid[i] - eps)]
    return max((equilibrium(case, p) for p in prices), key=lambda z: z["revenue"])


def dynamic_revenue_upper_bound(case):
    """Conservative upper bound on unrestricted dynamic-toll revenue.

    Any group g assigned to slot s must satisfy p_s <= v_gs, where v_gs is
    its zero-queue willingness to pay. We further relax anonymous pricing and
    allow each assigned group to pay v_gs itself. Maximizing these payments
    subject only to group masses and bottleneck capacity gives an upper bound.
    """
    A = sparse.vstack([case["A_cap_x"], case["A_dem_x"]], format="csr")
    b = np.r_[case["slot_capacity"], case["group_mass"]]
    res = linprog(-case["value"].ravel(), A_ub=A, b_ub=b,
                  bounds=(0, None), method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return -float(res.fun)


def system_costs(case, static_eq):
    """Static system cost and first-best lower bound on dynamic system cost."""
    G, S = case["G"], case["S"]
    x, y, w = static_eq["x"], static_eq["y"], static_eq["w"]
    transit_cost = case["car_cost"] + case["gap"]
    static_cost = (
        np.dot(case["car_cost"], x.sum(axis=1))
        + np.dot(transit_cost, y)
        + np.sum(case["schedule"] * x)
        + np.sum(case["alpha"][:, None] * w[None, :] * x)
    )

    obj = np.r_[
        (case["car_cost"][:, None] + case["schedule"]).ravel(),
        transit_cost,
    ]
    res = linprog(
        obj, A_ub=case["A_cap"], b_ub=case["slot_capacity"],
        A_eq=case["A_dem"], b_eq=case["group_mass"],
        bounds=(0, None), method="highs",
    )
    if not res.success:
        raise RuntimeError(res.message)
    return float(static_cost), float(res.fun)


def run_experiment(etas=ETA_VALUES, h_v=0.50, h_s=0.50, dt_minutes=15):
    rows = []
    for eta in etas:
        case = make_case(eta, h_v, h_s, dt_minutes)
        static = optimal_static(case)
        dynamic_ub = dynamic_revenue_upper_bound(case)
        sc_static, sc_fb = system_costs(case, static)
        rows.append(dict(
            eta=eta, h_V=h_v, h_S=h_s, dt_minutes=dt_minutes,
            static_toll=static["toll"], static_revenue=static["revenue"],
            dynamic_revenue_upper_bound=dynamic_ub,
            certified_revenue_ratio=static["revenue"] / dynamic_ub,
            static_system_cost=sc_static, first_best_system_cost=sc_fb,
            system_cost_upper_ratio=sc_static / sc_fb,
        ))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dt-minutes", type=float, default=15)
    parser.add_argument("--h-v", type=float, default=0.5)
    parser.add_argument("--h-s", type=float, default=0.5)
    parser.add_argument("--etas", type=float, nargs="+", default=ETA_VALUES)
    parser.add_argument("--output", type=Path, default=Path("heterogeneity_eta_results_v2.csv"))
    args = parser.parse_args()

    results = run_experiment(args.etas, args.h_v, args.h_s, args.dt_minutes)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
