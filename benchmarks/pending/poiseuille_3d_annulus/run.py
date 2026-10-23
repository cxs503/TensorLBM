#!/usr/bin/env python
"""3D annular-pipe (concentric ring) Poiseuille flow (D3Q19) — analytic validation.

W3-B revival of pending/poiseuille_3d_annulus (Wave-3, 2026-09-20).

Physics: steady fully-developed laminar flow in a straight annulus
(outer radius R_o, inner solid cylinder radius a, ratio a/R_o = 0.5),
driven by a uniform-velocity Zou-He inlet with a Zou-He pressure outlet;
both walls no-slip via the library swap-bounce-back on all solid cells
(outer wall d > R_o, inner solid column d < a).

Exact solution (self-derived, cross-checked numerically in validate.py):
    u(r) = G/(4 nu) * [ R_o^2 - r^2 + (R_o^2-a^2)/ln(R_o/a) * ln(r/R_o) ]
    Q    = pi*G/(8 nu) * Phi,  Phi = R_o^4 - a^4 - (R_o^2-a^2)^2/ln(R_o/a)
equivalently the Stokes-ODE family u = c0 + c1*ln(r) - c2*r^2 (c2 = G/4nu),
whose two no-slip roots define an effective annulus (delta_i, delta_o).

Comparison — TWO reported channels (pre-registered in NOTES.md):
  CH1 effective frame (primary): the digital staircase annulus is not the
      nominal one (two-parameter geometry: single-observable R_eff^Q-style
      inversion is underdetermined, ellipse lesson).  Instead the FULL
      binned profile is fit with the exact 3-parameter family
      u = c0 + c1 ln r - c2 r^2 (weighted LSQ on central bins selected by
      the NOMINAL analytic profile, |u_nom| > 0.2 U_max_nom); the fitted
      family's two roots are the effective radii delta_i, delta_o
      (no-slip inversion, no nominal radius enters).  The gradient is
      anchored by the imposed flux: G_Q = 8 nu Q_meas / (pi Phi(delta_o,
      delta_i)) with Q_meas the measured mid-plane flow rate (set by the
      Zou-He inlet mass conservation).  Reference u_ref(r) = exact annulus
      profile of (delta_i, delta_o, G_Q); metric = central-region binned
      max / weighted-L2 relative error (u_ref > 0.2 U_max_ref).
      Disclosures: G_fit = 4 nu c2 vs G_Q, interior pressure gradient
      G_int (rho at x=nx/4 vs 3nx/4), all-bins-fit delta variant.
  CH2 nominal frame (direct channel, pipe strict-review precedent): the
      exact profile of the NOMINAL (a, R_o) with G_nom = 8 nu u_in
      (R_o^2-a^2)/Phi(R_o,a) (u_mean = u_in on the nominal annulus) — no
      measured quantity in the reference at all.  Reported separately;
      passing is desired but not required for CH1.

No extrapolation, no correction factors, no tuning (extrap: none).

Library primitives only (grep self-check: zero hand-written kernels):
  tensorlbm.solver3d.collide_bgk3d / stream3d,
  tensorlbm.d3q19.equilibrium3d / macroscopic3d,
  tensorlbm.boundaries3d.zou_he_inlet_velocity_3d / zou_he_outlet_pressure_3d /
  bounce_back_cells_3d, benchmarks/compile_route.route_step.

Usage:
    run.py single R_o out.json [--a-ratio 0.5] [--tau 0.8] [--u-in 0.02]
        [--min-steps N] [--max-steps N] [--device cuda:0] [--seed 0]
        [--L-over-R 6]
    run.py scan out_dir --R 20 40 80 [--a-ratio 0.5] ...
    run.py summarize out_dir --R 20 40 80
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch
from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    ensure_tensorlbm_importable,
    route_step,
)

ensure_tensorlbm_importable()

from tensorlbm.boundaries3d import (  # noqa: E402
    bounce_back_cells_3d,
    zou_he_inlet_velocity_3d,
    zou_he_outlet_pressure_3d,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

CS2 = 1.0 / 3.0


# ---------------------------------------------------------------------------
# exact annular Poiseuille solution (float64 analysis; self-derived)
# ---------------------------------------------------------------------------
def annulus_geom_factor(R_o: float, a: float) -> float:
    """Phi = R_o^4 - a^4 - (R_o^2-a^2)^2/ln(R_o/a);  Q = pi G/(8 nu) Phi."""
    ln = math.log(R_o / a)
    return R_o**4 - a**4 - (R_o**2 - a**2) ** 2 / ln


def annulus_u(r, G: float, R_o: float, a: float, nu: float):
    """Exact profile u(r) with u(a)=u(R_o)=0 (numpy array or scalar).

    r is floored at 1e-9 so solid cells at the axis (d=0, always masked out
    downstream) do not produce log(0) warnings."""
    ln = math.log(R_o / a)
    rr = np.maximum(np.asarray(r, dtype=np.float64), 1e-9)
    return G / (4.0 * nu) * (R_o**2 - rr**2 + (R_o**2 - a**2) / ln * np.log(rr / R_o))


def annulus_G_from_Q(Q: float, R_o: float, a: float, nu: float) -> float:
    """Gradient implied by flow rate Q through the annulus (R_o, a)."""
    return 8.0 * nu * Q / (math.pi * annulus_geom_factor(R_o, a))


def annulus_peak(G: float, R_o: float, a: float, nu: float) -> float:
    r_star = math.sqrt((R_o**2 - a**2) / (2.0 * math.log(R_o / a)))
    return float(annulus_u(r_star, G, R_o, a, nu))


# ---------------------------------------------------------------------------
# exact-family fit + no-slip root inversion (the two-parameter revival)
# ---------------------------------------------------------------------------
def family_fit(r: np.ndarray, u: np.ndarray, w: np.ndarray, sel: np.ndarray):
    """Weighted LSQ of u ~= c0 + c1*ln r - c2*r^2 on selected bins.

    Weights are the per-bin cell counts (bin-mean variance ~ 1/w), applied
    as sqrt(w) row scaling; columns are normalised before the solve for
    conditioning (the {1, ln r, r^2} basis is strongly correlated over a
    half-decade radial band).  Returns (c0, c1, c2, cond).
    """
    rr = np.asarray(r, dtype=np.float64)[sel]
    uu = np.asarray(u, dtype=np.float64)[sel]
    ww = np.asarray(w, dtype=np.float64)[sel]
    X = np.stack([np.ones_like(rr), np.log(rr), -(rr**2)], axis=1)
    sw = np.sqrt(ww)
    scale = np.linalg.norm(X * sw[:, None], axis=0)
    scale[scale == 0] = 1.0
    A = (X / scale) * sw[:, None]
    beta_s, res, rank, sv = np.linalg.lstsq(A, uu * sw, rcond=None)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")
    beta = beta_s / scale
    return float(beta[0]), float(beta[1]), float(beta[2]), cond


def family_eval(r, c):
    return (
        c[0]
        + c[1] * np.log(np.asarray(r, dtype=np.float64))
        - c[2] * np.asarray(r, dtype=np.float64) ** 2
    )


def family_roots(c, r_lo: float, r_hi: float) -> tuple[float, float]:
    """The two positive roots of the family (no-slip inversion) in
    [r_lo, r_hi] by dense sign scan + bisection (numpy-only)."""
    n = 20001
    grid = np.linspace(r_lo, r_hi, n)
    g = family_eval(grid, c)
    sign = np.sign(g)
    idx = np.nonzero(np.diff(sign))[0]
    roots = []
    for i in idx:
        lo, hi = grid[i], grid[i + 1]
        glo = g[i]
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            gm = c[0] + c[1] * math.log(mid) - c[2] * mid**2
            if (gm > 0) == (glo > 0):
                lo, glo = mid, gm
            else:
                hi = mid
        roots.append(0.5 * (lo + hi))
    if len(roots) != 2:
        raise RuntimeError(
            f"family_roots: expected 2 roots in [{r_lo},{r_hi}], got {len(roots)}: {roots}"
        )
    return float(roots[0]), float(roots[1])


# ---------------------------------------------------------------------------
# geometry + radial binning
# ---------------------------------------------------------------------------
def annulus_setup(R_o: int, a: float, L_over_R: int, device):
    """Cross-section ny=nz=2R_o+3, axis at (R_o+1, R_o+1); fluid a<=d<=R_o."""
    ny = nz = 2 * R_o + 3
    nx = L_over_R * R_o
    yc = zc = R_o + 1
    iz = torch.arange(nz, device=device, dtype=torch.float32).view(-1, 1)
    iy = torch.arange(ny, device=device, dtype=torch.float32).view(1, -1)
    d = torch.sqrt((iy - yc) ** 2 + (iz - zc) ** 2)  # (nz, ny)
    fluid2d = (d <= R_o) & (d >= a)
    wall2d = ~fluid2d
    wall_mask = wall2d.unsqueeze(-1).expand(nz, ny, nx).contiguous()
    return ny, nz, nx, yc, zc, d, fluid2d, wall_mask


def radial_bins(d_np: np.ndarray, fluid: np.ndarray, r_inner: float, r_outer: float):
    """Bin fluid cells by radius: bin k holds k <= d < k+1 (float64)."""
    k = np.floor(d_np).astype(int)
    ks, rs, us, ws = [], [], [], []
    for kk in range(int(math.floor(r_inner)), int(math.ceil(r_outer)) + 1):
        m = fluid & (k == kk)
        if m.sum() == 0:
            continue
        ks.append(kk)
        rs.append(float(d_np[m].mean()))
        ws.append(int(m.sum()))
    return np.array(ks), np.array(rs, dtype=np.float64), np.array(ws, dtype=np.float64)


def bin_means(u_np: np.ndarray, d_np: np.ndarray, fluid: np.ndarray, ks: np.ndarray) -> np.ndarray:
    k = np.floor(d_np).astype(int)
    return np.array([float(u_np[fluid & (k == kk)].mean()) for kk in ks], dtype=np.float64)


def profile_metrics(u_np, d_np, fluid, u_ref_cell, U_max_ref, ks):
    """Binned + per-cell central-region errors vs an analytic reference."""
    ub = bin_means(u_np, d_np, fluid, ks)
    wb = np.array(
        [int((fluid & (np.floor(d_np).astype(int) == kk)).sum()) for kk in ks], dtype=np.float64
    )
    rb = np.array([float(d_np[fluid & (np.floor(d_np).astype(int) == kk)].mean()) for kk in ks])
    uref_b = np.array(
        [float(u_ref_cell[fluid & (np.floor(d_np).astype(int) == kk)].mean()) for kk in ks]
    )

    sel = uref_b > 0.2 * abs(U_max_ref)  # central region from the reference itself
    err_bin = np.abs(ub - uref_b)
    max_bin = float(err_bin[sel].max() / abs(U_max_ref) * 100.0)
    l2_bin = float(np.linalg.norm(err_bin[sel]) / np.linalg.norm(uref_b[sel]) * 100.0)
    mc = fluid & (u_ref_cell > 0.2 * abs(U_max_ref))
    max_cell = float(np.abs(u_np[mc] - u_ref_cell[mc]).max() / abs(U_max_ref) * 100.0)
    Q = float(u_np[fluid].sum())
    return {
        "max_bin_central_pct": max_bin,
        "l2_bin_central_pct": l2_bin,
        "max_cell_central_pct": max_cell,
        "u_center": float(u_np[fluid].max()),
        "Q": Q,
        "r_bins": [round(float(v), 5) for v in rb],
        "u_bins": [round(float(v), 8) for v in ub],
        "uref_bins": [round(float(v), 8) for v in uref_b],
        "w_bins": [int(v) for v in wb],
        "_raw": (ub, uref_b, sel),
    }


# ---------------------------------------------------------------------------
# per-grid simulation
# ---------------------------------------------------------------------------
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
    avg_steps: int = 400,
) -> dict:
    torch.manual_seed(seed)
    nu = (tau - 0.5) / 3.0
    a = a_ratio * R_o
    ny, nz, nx, yc, zc, d, fluid2d, wall_mask = annulus_setup(R_o, a, L_over_R, device)

    Phi = annulus_geom_factor(float(R_o), a)
    A_nom = math.pi * (float(R_o) ** 2 - a**2)
    G_nom = 8.0 * nu * u_in * (float(R_o) ** 2 - a**2) / Phi  # u_mean = u_in (nominal)
    U_max_nom = annulus_peak(G_nom, float(R_o), a, nu)

    rho_out = 1.0
    Re = u_in * 2.0 * (R_o - a) / nu

    # initial condition: rest + NOMINAL analytic profile (transient only;
    # the steady state is fixed by the boundary conditions) — disclosed
    u_ana_cell = annulus_u(d.cpu().numpy(), G_nom, float(R_o), a, nu)
    ux0_np = np.where(fluid2d.cpu().numpy(), u_ana_cell, 0.0).astype(np.float32)
    ux0 = torch.from_numpy(ux0_np).to(device).view(nz, ny, 1).expand(nz, ny, nx)
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    initial_mass = float(f.sum().item())

    def _step(f):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        return bounce_back_cells_3d(f, wall_mask)

    step_fn = route_step(_step, compile_mode, name=f"poiseuille_3d_annulus[Ro{R_o}]")

    x_meas = nx // 2
    x_dev = nx - 8
    x_p1, x_p2 = nx // 4, (3 * nx) // 4

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

    acc_meas = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_dev = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_p1 = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_p2 = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_in = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    for _ in range(avg_steps):
        f = step_fn(f)
        rho, ux, _, _ = macroscopic3d(f)
        acc_meas += ux[:, :, x_meas]
        acc_dev += ux[:, :, x_dev]
        acc_rho_p1 += rho[:, :, x_p1]
        acc_rho_p2 += rho[:, :, x_p2]
        acc_rho_in += rho[:, :, 0]
    acc_meas /= avg_steps
    acc_dev /= avg_steps
    acc_rho_p1 /= avg_steps
    acc_rho_p2 /= avg_steps
    acc_rho_in /= avg_steps
    elapsed = time.time() - t0

    d_np = d.cpu().numpy().astype(np.float64)
    fluid_np = fluid2d.cpu().numpy()
    u_meas = acc_meas.cpu().numpy().astype(np.float64)
    u_dev = acc_dev.cpu().numpy().astype(np.float64)

    ks, r_bins, w_bins = radial_bins(d_np, fluid_np, a, float(R_o))
    u_bins = bin_means(u_meas, d_np, fluid_np, ks)
    Q_meas = float(u_meas[fluid_np].sum())
    rho_p1 = float(acc_rho_p1[fluid_np].mean().item())
    rho_p2 = float(acc_rho_p2[fluid_np].mean().item())
    rho_in = float(acc_rho_in[fluid_np].mean().item())
    G_int = (rho_p1 - rho_p2) * CS2 / float(x_p2 - x_p1)  # interior pressure gradient
    dp_inlet_outlet = (rho_in - rho_out) * CS2

    # ---- CH1: full-profile family fit -> no-slip root inversion -----------
    u_nom_bins = annulus_u(r_bins, G_nom, float(R_o), a, nu)
    sel_fit = u_nom_bins > 0.2 * U_max_nom  # central bins (nominal selection)
    c0, c1, c2, cond = family_fit(r_bins, u_bins, w_bins, sel_fit)
    G_fit = 4.0 * nu * c2
    delta_i, delta_o = family_roots((c0, c1, c2), a - 1.5, R_o + 1.5)
    # all-bins robustness variant
    c0a, c1a, c2a, _ = family_fit(r_bins, u_bins, w_bins, np.ones_like(sel_fit))
    delta_i_ab, delta_o_ab = family_roots((c0a, c1a, c2a), a - 1.5, R_o + 1.5)

    # anchor the reference gradient with the imposed flux (effective geom)
    G_Q = annulus_G_from_Q(Q_meas, delta_o, delta_i, nu)
    U_max_ref = annulus_peak(G_Q, delta_o, delta_i, nu)
    u_ref_cell_eff = annulus_u(d_np, G_Q, delta_o, delta_i, nu)
    m_eff = profile_metrics(u_meas, d_np, fluid_np, u_ref_cell_eff, U_max_ref, ks)

    # ---- CH2: nominal-frame direct channel --------------------------------
    u_ref_cell_nom = annulus_u(d_np, G_nom, float(R_o), a, nu)
    m_nom = profile_metrics(u_meas, d_np, fluid_np, u_ref_cell_nom, U_max_nom, ks)
    # measured-dp variant on the nominal geometry (secondary disclosure)
    G_dp = dp_inlet_outlet / float(nx)
    m_nom_dp = profile_metrics(
        u_meas,
        d_np,
        fluid_np,
        annulus_u(d_np, G_dp, float(R_o), a, nu),
        annulus_peak(G_dp, float(R_o), a, nu),
        ks,
    )
    # fully-developed check at the near-outlet plane
    u_dev_bins = bin_means(u_dev, d_np, fluid_np, ks)
    fd_max_dev = float(np.max(np.abs(u_bins - u_dev_bins) / np.maximum(np.abs(u_bins), 1e-12)))

    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0
    u_mean_eff = Q_meas / (math.pi * (delta_o**2 - delta_i**2))

    result = {
        "case": "poiseuille_3d_annulus",
        "lattice": "D3Q19",
        "collision": "bgk",
        "boundary": (
            "zou_he_velocity_inlet(x=0) + zou_he_pressure_outlet(x=nx-1) + "
            "swap bounce-back on ALL solid cells (outer d>R_o and inner column "
            "d<a, post-streaming)"
        ),
        "R_o": R_o,
        "a": a,
        "a_ratio": a_ratio,
        "ny": ny,
        "nz": nz,
        "nx": nx,
        "L_over_R": float(nx / R_o),
        "tau": tau,
        "nu_lb": nu,
        "u_in": u_in,
        "rho_out": rho_out,
        "Re": Re,
        "Ma": U_max_nom / math.sqrt(CS2),
        "G_nom": G_nom,
        "U_max_nom": U_max_nom,
        "min_steps": min_steps,
        "n_steps": step,
        "avg_steps": avg_steps,
        "steady": steady,
        "compile_mode": compile_mode,
        # CH1 effective frame
        "fit_c": [c0, c1, c2],
        "fit_cond": cond,
        "G_fit": G_fit,
        "delta_i": delta_i,
        "delta_o": delta_o,
        "delta_i_minus_a": delta_i - a,
        "delta_o_minus_Ro": delta_o - float(R_o),
        "delta_i_allbins": delta_i_ab,
        "delta_o_allbins": delta_o_ab,
        "delta_i_minus_a_allbins": delta_i_ab - a,
        "delta_o_minus_Ro_allbins": delta_o_ab - float(R_o),
        "G_Q": G_Q,
        "G_fit_over_G_Q": G_fit / G_Q,
        "G_int_over_G_Q": G_int / G_Q,
        "G_int": G_int,
        "dp_inlet_outlet": dp_inlet_outlet,
        "U_max_ref_eff": U_max_ref,
        "u_mean_eff": u_mean_eff,
        "eff_max_bin_central_pct": m_eff["max_bin_central_pct"],
        "eff_l2_bin_central_pct": m_eff["l2_bin_central_pct"],
        "eff_max_cell_central_pct": m_eff["max_cell_central_pct"],
        # CH2 nominal frame
        "nom_max_bin_central_pct": m_nom["max_bin_central_pct"],
        "nom_l2_bin_central_pct": m_nom["l2_bin_central_pct"],
        "nom_max_cell_central_pct": m_nom["max_cell_central_pct"],
        "nom_max_bin_central_dp_pct": m_nom_dp["max_bin_central_pct"],
        # diagnostics
        "Q_meas": Q_meas,
        "N_fluid_cells": int(fluid_np.sum()),
        "A_nom": A_nom,
        "N_over_A_nom": fluid_np.sum() / A_nom,
        "u_center": m_eff["u_center"],
        "u_peak_err_vs_nom_pct": (m_eff["u_center"] - U_max_nom) / U_max_nom * 100.0,
        "fd_max_rel_dev_pct": fd_max_dev * 100.0,
        "mass_drift_pct": mass_drift_pct,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "r_bins": m_eff["r_bins"],
        "w_bins": m_eff["w_bins"],
        "u_bins": m_eff["u_bins"],
        "uref_bins_eff": m_eff["uref_bins"],
        "uref_bins_nom": m_nom["uref_bins"],
    }
    Path(out_path).write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _mono(errs: list[float]) -> bool:
    return all(errs[i + 1] <= errs[i] + 1e-9 for i in range(len(errs) - 1))


def build_summary(cases: list[dict], out_dir: str) -> dict:
    out_dir = Path(out_dir)
    convergence = []
    for c in cases:
        convergence.append(
            {
                "R_o": c["R_o"],
                "a": c["a"],
                "Re": c["Re"],
                "n_steps": c["n_steps"],
                "steady": c["steady"],
                "eff_max_bin_central_pct": round(c["eff_max_bin_central_pct"], 4),
                "eff_l2_bin_central_pct": round(c["eff_l2_bin_central_pct"], 4),
                "eff_max_cell_central_pct": round(c["eff_max_cell_central_pct"], 4),
                "nom_max_bin_central_pct": round(c["nom_max_bin_central_pct"], 4),
                "nom_l2_bin_central_pct": round(c["nom_l2_bin_central_pct"], 4),
                "delta_i_minus_a": round(c["delta_i_minus_a"], 4),
                "delta_o_minus_Ro": round(c["delta_o_minus_Ro"], 4),
                "delta_i_minus_a_allbins": round(c["delta_i_minus_a_allbins"], 4),
                "delta_o_minus_Ro_allbins": round(c["delta_o_minus_Ro_allbins"], 4),
                "G_fit_over_G_Q": round(c["G_fit_over_G_Q"], 5),
                "G_int_over_G_Q": round(c["G_int_over_G_Q"], 5),
                "Q_ratio_N_over_A_nom": round(c["N_over_A_nom"], 5),
                "fd_max_rel_dev_pct": round(c["fd_max_rel_dev_pct"], 5),
                "mass_drift_pct": round(c["mass_drift_pct"], 6),
                "elapsed_s": c["elapsed_s"],
            }
        )
    errs_eff = [c["eff_max_bin_central_pct"] for c in convergence]
    errs_nom = [c["nom_max_bin_central_pct"] for c in convergence]
    di = [c["delta_i_minus_a"] for c in convergence]
    do = [c["delta_o_minus_Ro"] for c in convergence]
    delta_spread_i = max(di) - min(di)
    delta_spread_o = max(do) - min(do)
    delta_grid_indep = (delta_spread_i <= 0.05) and (delta_spread_o <= 0.05)

    passed_eff = len(errs_eff) >= 2 and all(e <= 3.0 for e in errs_eff) and _mono(errs_eff)
    passed_nom = len(errs_nom) >= 2 and all(e <= 3.0 for e in errs_nom) and _mono(errs_nom)

    summary = {
        "case": "poiseuille_3d_annulus_convergence",
        "lattice": "D3Q19",
        "collision": "bgk",
        "a_ratio": cases[0]["a_ratio"],
        "R_o_list": [c["R_o"] for c in cases],
        "tau": cases[0]["tau"],
        "u_in": cases[0]["u_in"],
        "extrap": "none",
        "comparison_method": (
            "CH1 (primary): 3-parameter weighted LSQ of the exact family "
            "u=c0+c1 ln r-c2 r^2 on central bins (selected by the nominal "
            "analytic profile |u|>0.2 U_max_nom), no-slip roots -> effective "
            "radii (delta_i, delta_o); gradient anchored by the imposed flux "
            "G_Q=8 nu Q_meas/(pi Phi(delta_o,delta_i)); metric = central "
            "binned max rel err vs the exact profile of (delta_i,delta_o,G_Q). "
            "CH2 (direct channel): nominal geometry (a,R_o) with "
            "G_nom=8 nu u_in (R_o^2-a^2)/Phi (u_mean=u_in, no measured "
            "quantity), same metric. Disclosures: G_fit vs G_Q vs G_int "
            "(interior pressure gradient), all-bins delta variant, per-cell "
            "max, dp-variant nominal frame."
        ),
        "per_grid": convergence,
        "eff_errors_pct": errs_eff,
        "nom_errors_pct": errs_nom,
        "eff_monotone": _mono(errs_eff),
        "nom_monotone": _mono(errs_nom),
        "delta_i_minus_a": di,
        "delta_o_minus_Ro": do,
        "delta_spread_i": delta_spread_i,
        "delta_spread_o": delta_spread_o,
        "delta_grid_independent_(bound_0.05)": delta_grid_indep,
        "passed_eff_3pct_monotone": passed_eff,
        "passed_nom_3pct_monotone": passed_nom,
        "verdict": "verified" if (passed_eff and delta_grid_indep) else "not_verified",
        "notes": (
            f"CH1 eff-frame binned central max: {' -> '.join(f'{e:.3f}%' for e in errs_eff)} "
            f"(all<=3%: {all(e <= 3 for e in errs_eff)}, monotone: {_mono(errs_eff)}). "
            f"delta_i-a: {' / '.join(f'{v:+.4f}' for v in di)}; "
            f"delta_o-R_o: {' / '.join(f'{v:+.4f}' for v in do)} "
            f"(spreads {delta_spread_i:.4f}/{delta_spread_o:.4f}, "
            f"grid-independent bound 0.05: {delta_grid_indep}). "
            f"CH2 nominal-frame direct: {' -> '.join(f'{e:.3f}%' for e in errs_nom)} "
            f"(all<=3%: {all(e <= 3 for e in errs_nom)}, monotone: {_mono(errs_nom)}). "
            f"G_fit/G_Q: {' / '.join(f'{v:.4f}' for v in [c['G_fit_over_G_Q'] for c in cases])}; "
            f"G_int/G_Q: {' / '.join(f'{v:.4f}' for v in [c['G_int_over_G_Q'] for c in cases])}."
        ),
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def default_steps(R_o: int) -> tuple[int, int]:
    if R_o <= 20:
        return 20000, 60000
    if R_o <= 40:
        return 30000, 90000
    return 40000, 150000


def scan(
    R_list,
    a_ratio,
    tau,
    u_in,
    out_dir,
    device,
    seed=0,
    compile_mode="default",
    L_over_R=6,
    min_steps=0,
    max_steps=0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for R_o in R_list:
        dmin, dmax = default_steps(R_o)
        dmin = min_steps if min_steps > 0 else dmin
        dmax = max_steps if max_steps > 0 else dmax
        p = out_dir / f"case_Ro{R_o}.json"
        r = run_case(
            R_o,
            a_ratio,
            tau,
            u_in,
            dmin,
            dmax,
            str(p),
            device,
            seed=seed,
            L_over_R=L_over_R,
            compile_mode=compile_mode,
        )
        cases.append(r)
        print(
            f"R_o={r['R_o']:3d} steps={r['n_steps']:6d} steady={r['steady']} "
            f"eff={r['eff_max_bin_central_pct']:.4f}% nom={r['nom_max_bin_central_pct']:.4f}% "
            f"di={r['delta_i_minus_a']:+.4f} do={r['delta_o_minus_Ro']:+.4f} "
            f"Gfit/GQ={r['G_fit_over_G_Q']:.4f} fint={r['elapsed_s']:.0f}s",
            flush=True,
        )
    summary = build_summary(cases, str(out_dir))
    print(
        f"verdict={summary['verdict']} eff: {summary['eff_errors_pct']} nom: {summary['nom_errors_pct']}"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="3D annular Poiseuille (D3Q19)")
    sub = ap.add_subparsers(dest="mode_cmd", required=True)

    p1 = sub.add_parser("single")
    p1.add_argument("R_o", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--a-ratio", type=float, default=0.5)
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--u-in", type=float, default=0.02)
    p1.add_argument("--min-steps", type=int, default=0)
    p1.add_argument("--max-steps", type=int, default=0)
    p1.add_argument("--L-over-R", type=int, default=6)
    p1.add_argument("--device", type=str, default="cuda:0")
    p1.add_argument("--seed", type=int, default=0)
    add_compile_mode_arg(p1)

    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--R", type=int, nargs="+", default=[20, 40, 80])
    p2.add_argument("--a-ratio", type=float, default=0.5)
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--u-in", type=float, default=0.02)
    p2.add_argument("--min-steps", type=int, default=0)
    p2.add_argument("--max-steps", type=int, default=0)
    p2.add_argument("--L-over-R", type=int, default=6)
    p2.add_argument("--device", type=str, default="cuda:0")
    p2.add_argument("--seed", type=int, default=0)
    add_compile_mode_arg(p2)

    p3 = sub.add_parser("summarize")
    p3.add_argument("out_dir", type=str)
    p3.add_argument("--R", type=int, nargs="+", default=[20, 40, 80])

    args = ap.parse_args()
    device = torch.device(args.device)
    if args.mode_cmd == "summarize":
        cases = [
            json.loads((Path(args.out_dir) / f"case_Ro{R}.json").read_text())
            for R in args.R
            if (Path(args.out_dir) / f"case_Ro{R}.json").exists()
        ]
        if not cases:
            print("no case JSONs found")
            return
        s = build_summary(cases, args.out_dir)
        print(f"verdict={s['verdict']} eff: {s['eff_errors_pct']} nom: {s['nom_errors_pct']}")
        return
    compile_mode = compile_mode_from_args(args)
    if args.mode_cmd == "single":
        dmin, dmax = default_steps(args.R_o)
        run_case(
            args.R_o,
            args.a_ratio,
            args.tau,
            args.u_in,
            args.min_steps if args.min_steps > 0 else dmin,
            args.max_steps if args.max_steps > 0 else dmax,
            args.out_json,
            device,
            L_over_R=args.L_over_R,
            compile_mode=compile_mode,
        )
    else:
        scan(
            args.R,
            args.a_ratio,
            args.tau,
            args.u_in,
            args.out_dir,
            device,
            seed=args.seed,
            compile_mode=compile_mode,
            L_over_R=args.L_over_R,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
