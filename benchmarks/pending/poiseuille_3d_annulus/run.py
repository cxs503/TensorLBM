#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""B14-annulus: 3D annular-pipe (concentric ring) Poiseuille flow (D3Q19) — analytic validation.

Physics: steady, fully-developed laminar flow in a straight annular pipe
(outer radius R_o, inner solid cylinder radius a, axis ratio a/R_o = 0.5)
driven by a uniform-velocity inlet (Zou-He) with a pressure outlet
(Zou-He).  Both walls (outer pipe wall AND inner solid cylinder) are
no-slip via post-streaming half-way bounce-back (the inner cylinder is a
solid column of bounce-back cells).

Analytic solution (steady, fully developed, no-slip at r=a and r=R_o):
    u(r) = G/(4*nu) * [ R_o^2 - r^2 + (R_o^2-a^2)/ln(R_o/a) * ln(r/R_o) ]
    Q    = pi*G/(8*nu) * [ R_o^4 - a^4 - (R_o^2-a^2)^2/ln(R_o/a) ]
where G = -dp/dx is the (positive) driving pressure gradient, nu=(tau-0.5)/3.
NOTE on the task-sheet formula u(r)=U_max*(1-(r/R_o)^2+(a^2/(R_o^2-a^2))*ln(r/R_o)):
that parameterisation does NOT satisfy the inner no-slip condition
(u(a) != 0, e.g. u(a)/U_max = 0.52 for R_o=30, a=15), so it is NOT the
annular Poiseuille solution; the Q formula above matches the standard
result (White, Viscous Fluid Flow) and is used as given.  The profile
above is the exact solution with u(a)=u(R_o)=0, peak at
r*^2 = (R_o^2-a^2)/(2*ln(R_o/a)).

Reference geometry problem (same family as the ellipse, two-parameter
geometry family): the digital staircase walls have an effective radius
that differs from the nominal one (circular pipe: R_eff^Q = R+0.11), but
for the annulus the two parameters (R_o, a) are NOT identifiable from a
single integral observable (Q or dp) — R_eff^Q-style single-observable
inversion is underdetermined (ellipse lesson 2026-08-20).  The primary
comparison therefore uses the NOMINAL geometry (R_o, a) with the nominal
gradient G_nom fixed by mass conservation from the imposed u_in:
    G_nom = 8*nu*u_in*(R_o^2-a^2) / [R_o^4-a^4-(R_o^2-a^2)^2/ln(R_o/a)]
(equivalent to u_mean = u_in on the annular inlet).  All observed
quantities (Q, dp, u_center) are reported separately.  Shape-normalized,
measured-dp, and per-cell variants are disclosed as secondary diagnostics.

True simulation, no extrapolation:
  - library primitives only: solver3d.collide_bgk3d / stream3d,
    d3q19.equilibrium3d / macroscopic3d,
    boundaries3d.zou_he_inlet_velocity_3d / zou_he_outlet_pressure_3d /
    bounce_back_cells_3d
  - post-streaming half-way bounce-back at ALL solid cells (outer wall
    d > R_o and inner cylinder d < a)
  - no correction factors, no result tuning, extrap: none

Usage:
    run.py single R_o out.json [--a-ratio 0.5] [--tau T] [--u-in U]
        [--min-steps N] [--max-steps N] [--device cuda:2] [--seed 0]
    run.py scan out_dir [--R 30 45] [--a-ratio 0.5] [--min-steps N]
        [--max-steps N] [--device cuda:2]
    run.py summarize out_dir [--R 30 45]   # re-aggregate from case JSONs
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

from compile_route import add_compile_mode_arg, compile_mode_from_args, route_step  # noqa: E402

import numpy as np
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.boundaries3d import (
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# analytic annular Poiseuille
# ---------------------------------------------------------------------------
def annulus_geom_factor(R_o: float, a: float) -> float:
    """Phi = R_o^4 - a^4 - (R_o^2-a^2)^2/ln(R_o/a)  (Q = pi*G/(8 nu) * Phi)."""
    ln = math.log(R_o / a)
    return R_o ** 4 - a ** 4 - (R_o ** 2 - a ** 2) ** 2 / ln


def annulus_u(r: np.ndarray, G: float, R_o: float, a: float, nu: float) -> np.ndarray:
    """Exact annular Poiseuille profile u(r), u(a)=u(R_o)=0."""
    ln = math.log(R_o / a)
    return G / (4.0 * nu) * (
        R_o ** 2 - r ** 2 + (R_o ** 2 - a ** 2) / ln * np.log(r / R_o)
    )


def annulus_peak(r_star: float, G: float, R_o: float, a: float, nu: float) -> float:
    return float(annulus_u(np.array([r_star]), G, R_o, a, nu)[0])


def annulus_setup(R_o: int, a: float, L_over_R: int, device: torch.device):
    """Build the annular geometry: domain (nz, ny, nx), axis, fluid/wall masks.

    Flow along +x.  Cross-section (y,z): nz=ny=2*R_o+3, axis at (R_o+1, R_o+1).
    Fluid cells: a <= d <= R_o.  Solid (bounce-back): d > R_o (outer wall) and
    d < a (inner solid cylinder).
    """
    ny = nz = 2 * R_o + 3
    nx = L_over_R * R_o
    yc = zc = R_o + 1

    iz = torch.arange(nz, device=device, dtype=torch.float32).view(-1, 1)
    iy = torch.arange(ny, device=device, dtype=torch.float32).view(1, -1)
    d2 = (iy - yc) ** 2 + (iz - zc) ** 2          # (nz, ny)
    d = torch.sqrt(d2)
    fluid2d = (d <= R_o) & (d >= a)               # annular fluid cross-section
    wall2d = ~fluid2d                             # solid cells (outer wall + inner column)
    wall_mask = wall2d.unsqueeze(-1).expand(nz, ny, nx).contiguous()
    return ny, nz, nx, yc, zc, d, fluid2d, wall_mask


def annular_profile(
    ux_plane: torch.Tensor,   # (nz, ny) time-averaged ux at measurement plane
    d: torch.Tensor,          # (nz, ny) distance from axis
    R_o: int,
    a: float,
    G_ref: float,             # gradient used for the analytic profile
    nu: float,
    U_max_ref: float,         # peak velocity used for the central-region mask
    Q_nom: float,             # nominal flow rate (u_in * pi*(Ro^2-a^2))
) -> dict:
    """Bin the annular plane by radius (bin k: k<=d<k+1, k=floor(a)..R_o) and
    compare with the exact annular Poiseuille profile (per-cell average of the
    analytic u(r) inside each bin, removing binning bias)."""
    d_np = d.cpu().numpy()
    u_np = ux_plane.cpu().numpy().astype(np.float64)
    fluid = (d_np <= R_o) & (d_np >= a)
    u_ana_cell = annulus_u(d_np, G_ref, float(R_o), a, nu)

    k0 = int(math.floor(a))
    u_num, u_ana, cells, d_avg = [], [], [], []
    for k in range(k0, R_o + 1):
        m = fluid & (d_np >= k) & (d_np < k + 1)
        if m.sum() == 0:
            continue
        u_num.append(float(u_np[m].mean()))
        u_ana.append(float(u_ana_cell[m].mean()))
        cells.append(int(m.sum()))
        d_avg.append(float(d_np[m].mean()))

    u_num = np.array(u_num)
    u_ana = np.array(u_ana)
    l2_rel = float(np.linalg.norm(u_num - u_ana) / np.linalg.norm(u_ana))

    # Per-cell central-region max relative error (|u_ana| > 20% of U_max_ref)
    mask_c = fluid & (np.abs(u_ana_cell) > 0.2 * abs(U_max_ref))
    if mask_c.sum() > 0:
        max_rel = float(np.max(np.abs(u_np[mask_c] - u_ana_cell[mask_c])
                               / np.abs(u_ana_cell[mask_c])) * 100.0)
    else:
        max_rel = float("nan")

    # Per-bin central-region max relative error (radially-averaged profile)
    rel_bin = np.abs(u_num - u_ana) / np.abs(u_ana)
    mask_bin = np.abs(u_ana) > 0.2 * abs(U_max_ref)
    if mask_bin.sum() > 0:
        max_rel_bin = float(np.max(rel_bin[mask_bin])) * 100.0
    else:
        max_rel_bin = float("nan")

    # Near-wall bins (first bin at the inner wall, last bin at the outer wall)
    def _bin_err(idx):
        if idx < 0 or idx >= len(u_ana) or abs(u_ana[idx]) < 1e-12:
            return float("nan")
        return float(rel_bin[idx]) * 100.0

    # Flow-rate diagnostics (nominal annulus area A = pi*(R_o^2-a^2))
    Q = float(u_np[fluid].sum())
    N_cells = int(fluid.sum())
    u_center = float(u_np[fluid].max())  # peak velocity (at r~r*)
    u_peak_err_pct = (u_center - U_max_ref) / abs(U_max_ref) * 100.0

    return {
        "l2_rel_err": l2_rel,
        "max_rel_err_central_pct": max_rel,
        "max_rel_bin_central_pct": max_rel_bin,
        "u_peak_err_pct": u_peak_err_pct,
        "u_center": u_center,
        "Q": Q,
        "Q_nom": Q_nom,
        "Q_ratio": Q / Q_nom if Q_nom else float("nan"),
        "N_fluid_cells": N_cells,
        "bin_inner_err_pct": _bin_err(0),
        "bin_outer_err_pct": _bin_err(len(u_ana) - 1),
        "bins": [round(float(v), 8) for v in u_num],
        "bins_ana": [round(float(v), 8) for v in u_ana],
        "bins_d": [round(float(v), 4) for v in d_avg],
        "bins_cells": cells,
    }


def run_case(
    R_o: int,
    a_ratio: float,
    tau: float,
    u_in: float,
    min_steps: int,
    max_steps: int,
    out_path: str,
    device: torch.device,
    seed: int = 0,
    L_over_R: int = 6,
    compile_mode: str | None = "default",
) -> dict:
    torch.manual_seed(seed)
    nu = (tau - 0.5) / 3.0
    a = a_ratio * R_o
    ny, nz, nx, yc, zc, d, fluid2d, wall_mask = annulus_setup(R_o, a, L_over_R, device)

    # --- nominal analytic reference (nominal geometry + mass-conservation G) ---
    Phi = annulus_geom_factor(float(R_o), a)
    A_nom = math.pi * (float(R_o) ** 2 - a ** 2)
    G_nom = 8.0 * nu * u_in * (float(R_o) ** 2 - a ** 2) / Phi   # u_mean = u_in
    r_star = math.sqrt((float(R_o) ** 2 - a ** 2) / (2.0 * math.log(R_o / a)))
    U_max_nom = annulus_peak(r_star, G_nom, float(R_o), a, nu)

    rho_out = 1.0
    Re = u_in * 2.0 * (R_o - a) / nu        # Re = u_mean*D_h/nu, D_h = 2(R_o-a)
    Ma = U_max_nom / math.sqrt(CS2)

    # --- initial condition: rest density + annular Poiseuille profile ---
    d3 = d.unsqueeze(-1)                             # (nz, ny, 1)
    u_ana_cell = annulus_u(d.cpu().numpy(), G_nom, float(R_o), a, nu)
    ux0_np = np.where(fluid2d.cpu().numpy(), u_ana_cell, 0.0).astype(np.float32)
    ux0 = torch.from_numpy(ux0_np).to(device).view(nz, ny, 1).expand(nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)
    initial_mass = float(f.sum().item())

    # ---- whole-step function (shared compile path; step index & steady-state
    # monitoring stay outside the compiled region, per compile_utils rules) ----
    def _step(f):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        return bounce_back_cells_3d(f, wall_mask)

    step_fn = route_step(_step, compile_mode, name=f"poiseuille_3d_annulus[Ro{R_o}]")

    x_meas = nx // 2                    # mid-pipe measurement plane
    x_dev = nx - 8                      # fully-developed check plane (near outlet)

    t0 = time.time()
    umax_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            umax_hist.append(float(ux[:, :, x_meas][fluid2d].max().item()))
            if step >= min_steps and len(umax_hist) >= 10:
                recent = umax_hist[-10:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    # --- time-average ux at both planes over the last 400 steps ---
    acc_meas = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_dev = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_in = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    for _ in range(400):
        f = step_fn(f)
        rho, ux, _, _ = macroscopic3d(f)
        acc_meas += ux[:, :, x_meas]
        acc_dev += ux[:, :, x_dev]
        acc_rho_in += rho[:, :, 0]
    acc_meas /= 400.0
    acc_dev /= 400.0
    acc_rho_in /= 400.0

    rho, ux, _, _ = macroscopic3d(f)

    # --- radial profile analysis at the measurement plane ---
    Q_nom = u_in * A_nom
    # PRIMARY: nominal-geometry absolute normalization (G_nom from mass
    # conservation of the imposed u_in; pure analytic prediction).
    prof = annular_profile(acc_meas, d, R_o, a, G_nom, nu, U_max_nom, Q_nom)
    # shape-normalized variant: scale the analytic profile so its peak equals
    # the measured u_center (secondary diagnostic)
    u_center_meas = prof["u_center"]
    G_shape = G_nom * (u_center_meas / U_max_nom) if U_max_nom > 0 else G_nom
    prof_shape = annular_profile(acc_meas, d, R_o, a, G_shape, nu, u_center_meas, Q_nom)
    # measured-pressure-gradient variant: G from the measured Zou-He inlet
    # density (independent integral observable)
    rho_in_meas = float(acc_rho_in[fluid2d].mean().item())
    dp_meas = (rho_in_meas - rho_out) * CS2
    G_meas = dp_meas / nx
    prof_dp = annular_profile(acc_meas, d, R_o, a, G_meas, nu, U_max_nom, Q_nom)
    # flow-rate variant: G back-solved from the measured Q with the NOMINAL
    # geometry (Q carries the magnitude; profile shape still nominal)
    G_Q = 8.0 * nu * prof["Q"] / (math.pi * Phi) if Phi > 0 else float("nan")
    prof_Q = annular_profile(acc_meas, d, R_o, a, G_Q, nu, U_max_nom, Q_nom)

    # --- fully-developed check: profile at x_dev vs x_meas (normalized) ---
    fd_dev = annular_profile(acc_dev, d, R_o, a, G_nom, nu, U_max_nom, Q_nom)
    fd_max_dev = float(np.max(np.abs(np.array(prof["bins"]) - np.array(fd_dev["bins"]))
                              / np.maximum(np.abs(np.array(prof["bins"])), 1e-12)))

    # --- pressure diagnostics ---
    u_max_dp = annulus_peak(r_star, G_meas, float(R_o), a, nu)
    u_max_dp_err_pct = (prof["u_center"] - u_max_dp) / abs(u_max_dp) * 100.0 if u_max_dp > 0 else float("nan")

    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0

    result = {
        "case": "poiseuille_3d_annulus",
        "collision": "bgk",
        "lattice": "D3Q19",
        "boundary": (
            "zou_he_velocity_inlet(x=0) + zou_he_pressure_outlet(x=nx-1) + "
            "half-way bounce-back at ALL solid cells (outer wall d>R_o and "
            "inner solid cylinder d<a, post-streaming)"
        ),
        "driving": f"uniform velocity inlet u_in={u_in}",
        "R_o": R_o,
        "a": a,
        "a_ratio": a_ratio,
        "ny": ny, "nz": nz, "nx": nx,
        "L_over_R": float(nx / R_o),
        "tau": tau,
        "nu_lb": nu,
        "u_in": u_in,
        "rho_out": rho_out,
        "Phi": Phi,
        "G_nom": G_nom,
        "r_star": r_star,
        "U_max_nom": U_max_nom,
        "Re": Re,
        "Ma": Ma,
        "min_steps": min_steps,
        "n_steps": step,
        "compile_mode": compile_mode,
        "steady": steady,
        # --- PRIMARY: nominal-geometry absolute normalization ---
        "l2_rel_err": prof["l2_rel_err"],
        "max_rel_bin_central_pct": prof["max_rel_bin_central_pct"],
        "max_rel_err_central_pct": prof["max_rel_err_central_pct"],
        # --- shape-normalized variant (secondary) ---
        "l2_rel_err_shape": prof_shape["l2_rel_err"],
        "max_rel_bin_central_shape_pct": prof_shape["max_rel_bin_central_shape_pct"],
        "max_rel_err_central_shape_pct": prof_shape["max_rel_err_central_shape_pct"],
        # --- measured-dp variant (secondary) ---
        "l2_rel_err_dp": prof_dp["l2_rel_err"],
        "max_rel_bin_central_dp_pct": prof_dp["max_rel_bin_central_pct"],
        # --- Q-normalized variant (secondary) ---
        "l2_rel_err_Q": prof_Q["l2_rel_err"],
        "max_rel_bin_central_Q_pct": prof_Q["max_rel_bin_central_pct"],
        "G_Q": G_Q,
        # --- peak / flow / pressure diagnostics ---
        "u_center": prof["u_center"],
        "u_peak_err_pct": prof["u_peak_err_pct"],
        "u_max_dp_err_pct": u_max_dp_err_pct,
        "Q": prof["Q"],
        "Q_nom": prof["Q_nom"],
        "Q_ratio": prof["Q_ratio"],
        "bin_inner_err_pct": prof["bin_inner_err_pct"],
        "bin_outer_err_pct": prof["bin_outer_err_pct"],
        "N_fluid_cells": prof["N_fluid_cells"],
        "rho_in_measured": rho_in_meas,
        "G_meas": G_meas,
        "dp_meas": dp_meas,
        "fd_max_rel_dev_pct": fd_max_dev * 100.0,
        "mass_drift_pct": mass_drift_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "bins_d": prof["bins_d"],
        "bins_cells": prof["bins_cells"],
        "u_profile": prof["bins"],
        "u_analytic": prof["bins_ana"],
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


def build_summary(cases: list[dict], out_dir: str) -> dict:
    """Aggregate per-grid cases into result.json (verdict + full disclosure).

    Separate from run_case so a crashed/expensive simulation can be
    re-summarized from the saved case JSONs (--summarize-only).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    convergence = [
        {
            "R_o": c["R_o"], "a": c["a"], "Re": c["Re"],
            "l2_rel_err": c["l2_rel_err"],
            "max_rel_bin_central_pct": c["max_rel_bin_central_pct"],
            "max_rel_err_central_pct": c["max_rel_err_central_pct"],
            "l2_rel_err_shape": c["l2_rel_err_shape"],
            "max_rel_bin_central_shape_pct": c["max_rel_bin_central_shape_pct"],
            "l2_rel_err_dp": c["l2_rel_err_dp"],
            "max_rel_bin_central_dp_pct": c["max_rel_bin_central_dp_pct"],
            "l2_rel_err_Q": c["l2_rel_err_Q"],
            "max_rel_bin_central_Q_pct": c["max_rel_bin_central_Q_pct"],
            "u_peak_err_pct": c["u_peak_err_pct"],
            "Q_ratio": c["Q_ratio"],
            "bin_inner_err_pct": c["bin_inner_err_pct"],
            "bin_outer_err_pct": c["bin_outer_err_pct"],
            "G_Q": c["G_Q"],
            "n_steps": c["n_steps"], "steady": c["steady"],
        }
        for c in cases
    ]
    errs = [c["max_rel_bin_central_pct"] for c in convergence]
    converged = len(errs) >= 2 and errs[-1] < errs[0]
    passed = all(e <= 3.0 for e in errs) and converged
    errs_cell = [c["max_rel_err_central_pct"] for c in convergence]
    errs_shape = [c["max_rel_bin_central_shape_pct"] for c in convergence]
    errs_dp = [c["max_rel_bin_central_dp_pct"] for c in convergence]
    errs_Q = [c["max_rel_bin_central_Q_pct"] for c in convergence]
    qr = [c["Q_ratio"] for c in convergence]

    notes = (
        f"Nominal-geometry absolute comparison (u_ana from exact annular "
        f"Poiseuille u(r)=G/(4nu)[Ro^2-r^2+(Ro^2-a^2)/ln(Ro/a)*ln(r/Ro)] with "
        f"G_nom from mass conservation of the imposed u_in, no measured "
        f"quantity in the reference): per-bin radially-averaged profile max: "
        f"{' -> '.join(f'{e:.2f}%' for e in errs)} — "
        f"{'both <=3%' if all(e <= 3.0 for e in errs) else 'NOT both <=3%'}, "
        f"{'monotone' if converged else 'NOT monotone'} grid convergence. "
        f"Q_ratio (measured/nominal): {' / '.join(f'{v:.4f}' for v in qr)}. "
        f"Per-cell (stricter) central max: "
        f"{' -> '.join(f'{e:.2f}%' for e in errs_cell)}. Shape-normalized "
        f"per-bin: {' -> '.join(f'{e:.2f}%' for e in errs_shape)}; measured-dp "
        f"per-bin: {' -> '.join(f'{e:.2f}%' for e in errs_dp)}; Q-normalized "
        f"per-bin: {' -> '.join(f'{e:.2f}%' for e in errs_Q)} (all disclosed, "
        f"secondary). Root cause if failing: the annulus is a TWO-parameter "
        f"geometry family (R_o, a) whose digital staircase walls are "
        f"anisotropic (inner concave vs outer convex wall); a single integral "
        f"observable (Q or dp) cannot identify (R_o,eff, a_eff) — R_eff^Q-style "
        f"inversion is underdetermined (same failure mode as the ellipse "
        f"2026-08-20). No extrapolation, no tuning."
    )

    summary = {
        "case": "poiseuille_3d_annulus_convergence",
        "lattice": "D3Q19",
        "collision": "bgk",
        "boundary": cases[0]["boundary"],
        "driving": cases[0]["driving"],
        "R_o_list": [c["R_o"] for c in cases],
        "a_ratio": cases[0]["a_ratio"],
        "L_over_R": cases[0]["L_over_R"],
        "tau": cases[0]["tau"],
        "u_in": cases[0]["u_in"],
        "min_steps": cases[0]["min_steps"],
        "max_steps": cases[0]["max_steps"],
        "extrap": "none",
        "comparison_method": (
            "exact annular Poiseuille profile u(r)=G/(4nu)[Ro^2-r^2+"
            "(Ro^2-a^2)/ln(Ro/a)*ln(r/Ro)] with NOMINAL geometry (R_o, a) and "
            "nominal gradient G_nom = 8*nu*u_in*(Ro^2-a^2)/Phi (mass "
            "conservation from the imposed u_in; pure analytic prediction, no "
            "measured quantity in the reference). The annulus is a two-parameter "
            "geometry family: R_eff^Q-style single-observable inversion of "
            "(R_o,eff, a_eff) is underdetermined (ellipse lesson 2026-08-20), so "
            "no effective-geometry reference is used. Shape-normalized, "
            "measured-dp and Q-normalized variants are disclosed as secondary "
            "diagnostics."
        ),
        "primary_metric": (
            "max relative error of the radially-averaged profile in the central "
            "region (|u_ana| > 0.2*U_max_nom), vs exact annular Poiseuille "
            "u(r) with nominal geometry and nominal G_nom (absolute "
            "normalization)"
        ),
        "per_grid": convergence,
        "converged": converged,
        "passed_3pct_and_converged": passed,
        "verdict": "verified" if passed else "not_verified",
        "verified": passed,
        "notes": notes,
        "saved_to": f"benchmarks/verified/poiseuille_3d_annulus/",
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def scan(R_o_list, a_ratio, tau, u_in, min_steps, max_steps, out_dir: str,
         device: torch.device, seed: int = 0,
         compile_mode: str | None = "default") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for R_o in R_o_list:
        p = out_dir / f"case_Ro{R_o}.json"
        r = run_case(R_o, a_ratio, tau, u_in, min_steps, max_steps, str(p),
                     device, seed, compile_mode=compile_mode)
        cases.append(r)
        print(
            f"R_o={r['R_o']:3d} a={r['a']:.1f} Re={r['Re']:7.2f} steps={r['n_steps']:6d} "
            f"steady={r['steady']} Q_ratio={r['Q_ratio']:.4f} "
            f"max_bin={r['max_rel_bin_central_pct']:.4f}% cell={r['max_rel_err_central_pct']:.4f}% "
            f"shape={r['max_rel_bin_central_shape_pct']:.4f}% dp={r['max_rel_bin_central_dp_pct']:.4f}% "
            f"inner_bin={r['bin_inner_err_pct']:.4f}% outer_bin={r['bin_outer_err_pct']:.4f}% "
            f"u_peak_err={r['u_peak_err_pct']:+.4f}%",
            flush=True,
        )
    summary = build_summary(cases, str(out_dir))
    print(f"verdict={summary['verdict']} "
          f"max_bin: {' -> '.join(f'{e:.2f}%' for e in [c['max_rel_bin_central_pct'] for c in summary['per_grid']])}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="3D annular-pipe Poiseuille (D3Q19)")
    sub = ap.add_subparsers(dest="mode_cmd", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("R_o", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--a-ratio", type=float, default=0.5)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--u-in", type=float, default=0.02)
    p1.add_argument("--min-steps", type=int, default=20000)
    p1.add_argument("--max-steps", type=int, default=60000)
    p1.add_argument("--device", type=str, default="cuda:2")
    p1.add_argument("--seed", type=int, default=0)
    add_compile_mode_arg(p1)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--R", type=int, nargs="+", default=[30, 45])
    p2.add_argument("--a-ratio", type=float, default=0.5)
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--u-in", type=float, default=0.02)
    p2.add_argument("--min-steps", type=int, default=20000)
    p2.add_argument("--max-steps", type=int, default=60000)
    p2.add_argument("--device", type=str, default="cuda:2")
    p2.add_argument("--seed", type=int, default=0)
    add_compile_mode_arg(p2)

    p3 = sub.add_parser("summarize")
    p3.add_argument("out_dir", type=str)
    p3.add_argument("--R", type=int, nargs="+", default=[30, 45])

    args = ap.parse_args()
    device = torch.device(args.device)
    if args.mode_cmd == "summarize":
        cases = []
        for R_o in args.R:
            p = Path(args.out_dir) / f"case_Ro{R_o}.json"
            if p.exists():
                cases.append(json.loads(p.read_text()))
        if len(cases) == 0:
            print("no case JSONs found")
            return
        build_summary(cases, args.out_dir)
        return
    compile_mode = compile_mode_from_args(args)
    if args.mode_cmd == "single":
        r = run_case(args.R_o, args.a_ratio, args.tau, args.u_in,
                     args.min_steps, args.max_steps, args.out_json, device,
                     args.seed, compile_mode=compile_mode)
        print(json.dumps({k: r[k] for k in
                          ["R_o", "a", "nx", "ny", "nz", "Re", "Ma",
                           "n_steps", "steady", "u_peak_err_pct", "l2_rel_err",
                           "max_rel_bin_central_pct", "max_rel_err_central_pct",
                           "l2_rel_err_shape", "max_rel_bin_central_shape_pct",
                           "max_rel_bin_central_dp_pct", "Q_ratio",
                           "bin_inner_err_pct", "bin_outer_err_pct",
                           "fd_max_rel_dev_pct", "mass_drift_pct", "finite",
                           "elapsed_s"]}, indent=2))
    else:
        scan(args.R, args.a_ratio, args.tau, args.u_in,
             args.min_steps, args.max_steps, args.out_dir, device, args.seed,
             compile_mode)


if __name__ == "__main__":
    main()
