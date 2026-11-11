#!/usr/bin/env python3
"""Poiseuille 2D convergence order study with wall position corrections.

This benchmark measures the spatial convergence order of 2D Poiseuille flow
under four wall treatments (see ARMS below):
1. halfway         - pre-streaming half-way bounce-back, wall truly at 0.5 delta
2. subcell_halfway - the wall really sits at 0.25 delta but the BC still uses the
                     mid-point (this is the wall-position error being measured)
3. effective       - the same flow, scored against the effective wall (0.5 delta)
                     that the mid-point BC actually enforces
4. bouzidi         - Bouzidi interpolated bounce-back honouring q = 0.25 delta

Physics
-------
Fully-developed 2D laminar channel flow between parallel plates, driven by
a constant body force (equivalently, a constant pressure gradient). The
analytic solution is:

    u(y) = 4*u_max*(y/H)*(1 - y/H)

where H is the channel height and u_max is the centerline velocity.

Convergence measurement
-----------------------
Three grids with refinement ratio r = 2. L2 error of velocity profile vs
analytic solution. Observed order: p = ln(E_coarse/E_fine) / ln(r)

Acceptance criteria (pre-declared, corrected by AUDIT-RUNTEST-6):
  (a) halfway (wall truly at q = 0.5 delta): p ~ 2 -- mid-point BB is exact for
      that placement;
  (b) subcell_halfway (wall at q = 0.25 delta but the BC still uses the
      mid-point): p ~ 1 -- the wall-position error is O(delta);
  (c) effective (identical flow, scored against the effective wall the BC
      enforces, H_eff = ny - 2): p ~ 2 -- this is what the "effective wall
      position" correction buys;
  (d) bouzidi (interpolated BC honouring q = 0.25 delta): p >= 1.8 vs the TRUE
      wall.

Before AUDIT-RUNTEST-6 all three arms ("halfway"/"effective"/"bouzidi") set
q_wall = 0.5 and only the label "bouzidi" reached the interpolated kernel, which
at q = 0.5 is algebraically identical to half-way bounce-back; "effective" fell
through to the same apply_halfway_bb call.  The three-arm comparison in
result.json was therefore one computation reported three times (bit-identical
l2_rel down to the last digit).  Arms are now genuinely distinct.

References
----------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). "Momentum transfer of
    a Boltzmann-lattice fluid with boundaries." Phys. Fluids 13, 3452.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tensorlbm.d2q9 import C, W, OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import stream

# ---------------------------------------------------------------------------
# Wall-treatment arms (added by AUDIT-RUNTEST-6; before this, all arms collapsed
# onto the same code path -- see the module docstring).
#
# Geometry convention: ghost rows 0 and ny-1, fluid rows 1..ny-2.  The no-slip
# surface sits a sub-cell distance q below the first fluid row
# (y_wall,bottom = 1 - q) and q above the last one (y_wall,top = ny - 2 + q),
# so the TRUE channel height is H_true = ny - 3 + 2q.  q = 0.5 is the textbook
# half-way placement, for which H_true = ny - 2 = H.
#
# `reference` selects the analytic profile the measured profile is scored
# against:
#   "true"      -- parabola spanning the true walls (height H_true, wall at 1 - q)
#   "effective" -- parabola spanning the placement the *mid-point* BC actually
#                  enforces (height H_eff = ny - 2, wall at 0.5) with the centre
#                  velocity that geometry implies -- the classic "effective wall
#                  position" correction.
# ---------------------------------------------------------------------------
ARMS: dict[str, dict] = {
    # control: wall genuinely at the mid-point, BC exact for it
    "halfway": dict(q=0.5, kernel="halfway", reference="true"),
    # sub-cell wall ignored by the BC -> first-order wall-position error
    "subcell_halfway": dict(q=0.25, kernel="halfway", reference="true"),
    # same flow, scored against the effective wall the BC enforces
    "effective": dict(q=0.25, kernel="halfway", reference="effective"),
    # interpolated (Bouzidi) BC honouring q -> second order vs the true wall
    "bouzidi": dict(q=0.25, kernel="bouzidi", reference="true"),
}


DEVICE = torch.device("cpu")


def guo_source_1d(ux: torch.Tensor, accel: float, tau: float) -> torch.Tensor:
    """Guo source for x-directed body force (channel flow)."""
    c = C.to(dtype=torch.float32)
    w = W.to(dtype=torch.float32).view(9, 1, 1)
    cx = c[:, 0].view(9, 1, 1)
    ux_b = ux.unsqueeze(0)
    cdotu = cx * ux_b
    coeff = 1.0 - 1.0 / (2.0 * tau)
    S = coeff * w * (3.0 * (cx - ux_b) + 9.0 * cdotu * cx) * accel
    return S


def apply_halfway_bb(f: torch.Tensor) -> torch.Tensor:
    """Pre-streaming half-way bounce-back on top/bottom walls."""
    ny = f.shape[1]
    f_out = f.clone()
    # Bottom wall (y=0)
    for i in [4, 7, 8]:  # c_y < 0
        f_out[OPPOSITE[i], 0, :] = f[i, 1, :]
    # Top wall (y=ny-1)
    for i in [2, 5, 6]:  # c_y > 0
        f_out[OPPOSITE[i], ny - 1, :] = f[i, ny - 2, :]
    return f_out


def apply_bouzidi_bb(f: torch.Tensor, f_prev: torch.Tensor, q_bottom: float, q_top: float) -> torch.Tensor:
    """Bouzidi interpolated bounce-back with fractional distances q.
    
    q = distance from fluid node to wall / delta
    q = 0.5 corresponds to half-way bounce-back.
    """
    ny = f.shape[1]
    f_out = f.clone()
    
    # Bottom wall
    for i in [4, 7, 8]:  # c_y < 0
        opp = OPPOSITE[i]
        if q_bottom < 0.5:
            # Linear interpolation
            f_out[opp, 0, :] = 2.0 * q_bottom * f[i, 1, :] + (1.0 - 2.0 * q_bottom) * f_prev[i, 2, :]
        else:
            # Quadratic interpolation
            f_out[opp, 0, :] = f[i, 1, :] / (2.0 * q_bottom) + (2.0 * q_bottom - 1.0) / (2.0 * q_bottom) * f_prev[opp, 1, :]
    
    # Top wall
    for i in [2, 5, 6]:  # c_y > 0
        opp = OPPOSITE[i]
        if q_top < 0.5:
            f_out[opp, ny - 1, :] = 2.0 * q_top * f[i, ny - 2, :] + (1.0 - 2.0 * q_top) * f_prev[i, ny - 3, :]
        else:
            f_out[opp, ny - 1, :] = f[i, ny - 2, :] / (2.0 * q_top) + (2.0 * q_top - 1.0) / (2.0 * q_top) * f_prev[opp, ny - 2, :]
    
    return f_out


def run_poiseuille_case(
    H: int,
    tau: float,
    u_max_target: float,
    wall_model: str,
    steps: int,
) -> dict:
    """Run one Poiseuille case.
    
    wall_model: 'halfway' | 'effective' | 'bouzidi'
    """
    torch.manual_seed(0)
    nu = (tau - 0.5) / 3.0

    if wall_model not in ARMS:
        raise ValueError(f"unknown wall_model {wall_model!r}; known: {sorted(ARMS)}")
    spec = ARMS[wall_model]
    q_wall = float(spec["q"])
    kernel = spec["kernel"]

    # Channel geometry.  ny is chosen so that the TRUE height H_true equals the
    # requested H for *every* arm (H_true = ny - 3 + 2q), which keeps the
    # refinement ratio between grids at exactly 2.
    ny = int(round(H + 3.0 - 2.0 * q_wall))
    H_true = ny - 3.0 + 2.0 * q_wall
    H_eff = float(ny - 2)  # height implied by a mid-point (q = 0.5) placement

    nx = 4 * H  # L = 4H (long enough for fully-developed flow)

    # Body force to achieve target u_max over the TRUE height
    # u_max = accel * H^2 / (8*nu)  =>  accel = 8*nu*u_max/H^2
    accel = 8.0 * nu * u_max_target / (H_true * H_true)

    # Analytic solutions, evaluated at the fluid node rows j = 1 .. ny-2
    y_fluid = torch.arange(1, ny - 1, dtype=torch.float64)

    # (a) parabola spanning the true walls:  u = 4 u_max (y/H_t)(1 - y/H_t)
    y_norm_true = (y_fluid - (1.0 - q_wall)) / H_true
    u_analytic_true = 4.0 * u_max_target * y_norm_true * (1.0 - y_norm_true)

    # (b) parabola spanning the mid-point placement the BC actually enforces,
    #     with the centre velocity that placement implies at this body force
    u_max_eff = accel * H_eff * H_eff / (8.0 * nu)
    y_norm_eff = (y_fluid - 0.5) / H_eff
    u_analytic_eff = 4.0 * u_max_eff * y_norm_eff * (1.0 - y_norm_eff)

    u_analytic = u_analytic_true if spec["reference"] == "true" else u_analytic_eff
    
    # Initialize
    rho = torch.ones((ny, nx), dtype=torch.float32)
    ux = torch.zeros((ny, nx), dtype=torch.float32)
    uy = torch.zeros((ny, nx), dtype=torch.float32)
    f = equilibrium(rho, ux, uy)
    
    t0 = time.perf_counter()
    f_prev = f.clone()
    u_mid: torch.Tensor | None = None
    for step in range(steps):
        rho, ux, uy = macroscopic(f)
        # Guo forcing
        ux_star = ux + 0.5 * accel / rho
        feq = equilibrium(rho, ux_star, uy)
        S = guo_source_1d(ux_star, accel, tau)
        f = f + (feq - f) / tau + S
        
        # Wall boundary
        if kernel == "bouzidi":
            f = apply_bouzidi_bb(f, f_prev, q_wall, q_wall)
        else:
            f = apply_halfway_bb(f)
        
        f_prev = f.clone()
        f = stream(f)
        if step == steps // 2:
            rho_m, ux_m, uy_m = macroscopic(f)
            u_mid = (ux_m + 0.5 * accel / rho_m)[1 : ny - 1, nx // 2].double().clone()
    
    elapsed = time.perf_counter() - t0
    
    # Extract velocity profile (middle of channel, away from inlet/outlet)
    rho, ux, uy = macroscopic(f)
    ux_star = ux + 0.5 * accel / rho
    u_profile = ux_star[1:ny-1, nx // 2].double()  # fluid rows only
    
    # Compute errors
    # Steady-state guard: a fixed step count does not guarantee the parabolic
    # profile has been reached -- the transient time grows like H^2 while the
    # step budget here is constant, which silently destroys the observed order
    # for large H (AUDIT-RUNTEST-6 finding D10).  Compare the second half of the
    # run with the endpoint instead of trusting the step count.
    if u_mid is not None:
        steady_rel_change = (u_profile - u_mid).abs().max().item() / u_max_target
    else:
        steady_rel_change = float("nan")
    steady = bool(steady_rel_change < 1e-3)
    if not steady:
        print(
            f"  WARNING: H={H} arm={wall_model} not steady after {steps} steps "
            f"(max |du|/u_max = {steady_rel_change:.3e} over the last half of the run); "
            "increase --steps (roughly ~H^2) before reading the order off"
        )

    err = u_profile - u_analytic
    l2_err = torch.sqrt(torch.mean(err ** 2)).item()
    linf_err = err.abs().max().item()
    l2_rel = l2_err / u_max_target
    
    return {
        "H": H,
        "H_true": H_true,
        "H_eff": H_eff,
        "q_wall": q_wall,
        "bc_kernel": kernel,
        "reference": spec["reference"],
        "ny": ny,
        "nx": nx,
        "tau": tau,
        "nu": nu,
        "accel": accel,
        "u_max_target": u_max_target,
        "wall_model": wall_model,
        "steps": steps,
        "steady_rel_change_last_half": steady_rel_change,
        "steady": steady,
        "elapsed_s": round(elapsed, 2),
        "l2_err": l2_err,
        "l2_rel": l2_rel,
        "linf_err": linf_err,
        "linf_rel": linf_err / u_max_target,
        "l2_rel_true_wall": (u_profile - u_analytic_true).pow(2).mean().sqrt().item()
        / u_max_target,
        "l2_rel_effective_wall": (u_profile - u_analytic_eff).pow(2).mean().sqrt().item()
        / u_max_target,
        "u_profile": u_profile.tolist(),
        "u_analytic": u_analytic.tolist(),
        "finite": bool(torch.isfinite(f).all().item()),
    }


def compute_order(errors: list[float], ratios: list[float]) -> list[float]:
    """Compute observed convergence order."""
    orders = []
    for i in range(len(errors) - 1):
        if errors[i] > 0 and errors[i + 1] > 0 and ratios[i] > 1:
            p = math.log(errors[i] / errors[i + 1]) / math.log(ratios[i])
            orders.append(p)
        else:
            orders.append(float("nan"))
    return orders


def run_convergence_study(
    H_list: list[int],
    tau: float,
    u_max: float,
    steps: int,
    wall_models: list[str],
) -> dict:
    """Run convergence study for multiple wall models."""
    results = {}
    
    for model in wall_models:
        print(f"\n=== Wall model: {model} ===")
        cases = []
        for H in H_list:
            r = run_poiseuille_case(H, tau, u_max, model, steps)
            cases.append(r)
            print(f"H={H:3d} l2_rel={r['l2_rel']:.6e} linf_rel={r['linf_rel']:.6e} finite={r['finite']}")
        
        errors_l2 = [c["l2_rel"] for c in cases]
        errors_linf = [c["linf_rel"] for c in cases]
        # Refinement ratio on the TRUE channel height: for q != 0.5 the integer
        # row count can only approximate H, and using the nominal H would bias
        # the observed order.
        ratios = [cases[i]["H_true"] / cases[i - 1]["H_true"] for i in range(1, len(cases))]
        
        orders_l2 = compute_order(errors_l2, ratios)
        orders_linf = compute_order(errors_linf, ratios)
        
        results[model] = {
            "per_grid": cases,
            "convergence": {
                "errors_l2_rel": errors_l2,
                "errors_linf_rel": errors_linf,
                "grid_ratios": ratios,
                "h_true_list": [c["H_true"] for c in cases],
                "observed_orders_l2": orders_l2,
                "observed_orders_linf": orders_linf,
            },
        }
    
    return {
        "case": "poiseuille_convergence_order",
        "lattice": "D2Q9",
        "collision": "bgk",
        "forcing": "guo",
        "driving": "body_force",
        "tau": tau,
        "u_max_target": u_max,
        "H_list": H_list,
        "steps": steps,
        "wall_models": results,
    }


def main():
    ap = argparse.ArgumentParser(description="Poiseuille 2D convergence order study")
    ap.add_argument("--H", type=int, nargs="+", default=[20, 40, 80])
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--umax", type=float, default=0.04)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument(
        "--models", nargs="+", default=["halfway", "subcell_halfway", "effective", "bouzidi"]
    )
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    
    result = run_convergence_study(args.H, args.tau, args.umax, args.steps, args.models)
    
    out_json = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(out_json)
    print("\n" + out_json)


if __name__ == "__main__":
    main()
