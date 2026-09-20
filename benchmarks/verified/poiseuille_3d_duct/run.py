#!/usr/bin/env python
"""3D rectangular-duct Poiseuille flow (D3Q19) — analytic validation (Shah-London).

W3-B task 2 (Wave-3, 2026-09-20).  Square / rectangular straight duct, flow
along +x, lattice-aligned walls, driven by a uniform-velocity Zou-He inlet
with a Zou-He pressure outlet; all four walls no-slip via the library swap
bounce-back on the wall cells.

Exact solution (self-derived double odd-sine series, independently
cross-checked in ../validate.py against a hand-written DST-diagonalised
5-point FD solve + Shah-London invariants):
    u(y,z) = (16 G/(nu pi^2)) * sum_{m,n odd}
             sin(m pi (z+a)/(2a)) sin(n pi (y+b)/(2b)) / (m n lambda_mn)
    lambda_mn = (m pi/(2a))^2 + (n pi/(2b))^2,  a=W/2, b=H/2
Square invariants: Po = 56.9083, u_max/u_mean = 2.09626 (Shah-London
1978: 56.91 / 2.0962).  2:1 rectangle: Po = 62.1922.

Comparison — two channels (same pre-registration structure as the annulus):
  CH1 effective frame (primary): halfway bounce-back walls sit at
      lattice-dependent effective positions; the effective half-widths
      (a_eff = a + da, b_eff = b + db) are inverted from the FULL measured
      2D profile by a 3-parameter weighted least squares over (G, da, db)
      (G analytic given (da,db); (da,db) by coarse grid scan + alternating
      golden-section; fit region: u_meas > 0.2 u_max,meas — data-side
      selection).  G then anchored by the imposed flux:
      G_Q = nu * Q_meas / I0(a_eff, b_eff), Q_meas the measured mid-plane
      flow rate (= u_in * W * H by inlet mass conservation, lattice-aligned
      walls make digital area = nominal area exactly).
      Metric: per-cell central-region (u_ref > 0.2 U_max_ref) max and
      weighted-L2 relative error vs the series of (a_eff, b_eff, G_Q).
  CH2 nominal frame (direct channel): series of the NOMINAL (a, b) with
      G_nom = nu u_in 4ab / I0(a,b) (u_mean = u_in on the nominal duct;
      no measured quantity).  Reported separately.
  Cross-check observable: Po = 2 G D_h^2/(nu u_mean) in both frames vs
      Shah-London; plus interior pressure gradient G_int.

No extrapolation, no correction factors, no tuning (extrap: none).
Library primitives only (grep self-check: zero hand-written kernels):
  tensorlbm.solver3d.collide_bgk3d / stream3d,
  tensorlbm.d3q19.equilibrium3d / macroscopic3d,
  tensorlbm.boundaries3d.zou_he_inlet_velocity_3d / zou_he_outlet_pressure_3d /
  bounce_back_cells_3d, benchmarks/compile_route.route_step.

Usage:
    run.py single W out.json [--aspect 1.0] [--tau 0.8] [--u-in 0.02]
        [--min-steps N] [--max-steps N] [--device cuda:0] [--L-over-W 4]
    run.py scan out_dir --W 32 64 128 [--aspect 1.0]
    run.py summarize out_dir --W 32 64 128
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
_M = 121  # series truncation (odd modes 1.._M); converged to <1e-8


# ---------------------------------------------------------------------------
# exact series solution (float64)
# ---------------------------------------------------------------------------
def duct_S(yy: np.ndarray, zz: np.ndarray, a: float, b: float, M: int = _M) -> np.ndarray:
    """Series solution S of -lap S = 1 (i.e. u = G/nu * S) on (-a,a)x(-b,b)."""
    m = np.arange(1, M + 1, 2, dtype=np.float64)
    n = np.arange(1, M + 1, 2, dtype=np.float64)
    sy = np.sin(np.pi * n[None, :] * (yy[..., None] + b) / (2.0 * b))
    sz = np.sin(np.pi * m[None, :] * (zz[..., None] + a) / (2.0 * a))
    lam = (np.pi * m / (2.0 * a))[:, None] ** 2 + (np.pi * n / (2.0 * b))[None, :] ** 2
    coef = 16.0 / (np.pi**2 * np.outer(m, n) * lam)
    # m-modes (z, half-width a) contract coef's FIRST axis, n-modes (y,
    # half-width b) the SECOND — for a!=b the pairing is load-bearing
    # (a previous "...j,jk,...k" contracted sy against the m-axis, which is
    # a transpose: harmless for a=b, wrong shape for a!=b).
    return np.einsum("...m,mn,...n->...", sz, coef, sy)


def duct_I0(a: float, b: float, M: int = 1201) -> float:
    """I0 = integral of duct_S over the rectangle (analytic, term-wise)."""
    m = np.arange(1, M + 1, 2, dtype=np.float64)
    n = np.arange(1, M + 1, 2, dtype=np.float64)
    lam = (np.pi * m / (2.0 * a))[:, None] ** 2 + (np.pi * n / (2.0 * b))[None, :] ** 2
    coef = 16.0 / (np.pi**2 * np.outer(m, n) * lam)
    proj = np.outer(4.0 * a / (m * np.pi), 4.0 * b / (n * np.pi))
    return float((coef * proj).sum())


def duct_u(yy, zz, a, b, G, nu):
    return G / nu * duct_S(yy, zz, a, b)


def duct_Po(a: float, b: float) -> dict:
    """Analytic invariants of the exact duct solution (Shah-London)."""
    m = np.arange(1, 1201, 2, dtype=np.float64)
    n = np.arange(1, 1201, 2, dtype=np.float64)
    lam = (np.pi * m / (2.0 * a))[:, None] ** 2 + (np.pi * n / (2.0 * b))[None, :] ** 2
    coef = 16.0 / (np.outer(m, n) * lam)
    umean = float((coef * np.outer(2 / (m * np.pi), 2 / (n * np.pi))).sum() / np.pi**2)
    umax = float((coef * np.outer(np.sin(np.pi * m / 2), np.sin(np.pi * n / 2))).sum() / np.pi**2)
    dh = 4.0 * a * b / (a + b)
    return {"Po": 2.0 * dh**2 / umean, "umax_over_umean": umax / umean}


# ---------------------------------------------------------------------------
# effective-geometry inversion: (da, db) grid scan + alternating golden section
# ---------------------------------------------------------------------------
def _G_lsq(u: np.ndarray, S: np.ndarray, w: np.ndarray) -> float:
    num = float((w * u * S).sum())
    den = float((w * S * S).sum())
    return num / den


def _sse(u, S, w):
    r = (u - _G_lsq(u, S, w) * S)[w > 0]
    return float((r * r).sum())


def _golden(fn, lo, hi, tol=1e-6, iters=200):
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = fn(c), fn(d)
    for _ in range(iters):
        if hi - lo < tol:
            break
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = fn(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = fn(d)
    return 0.5 * (lo + hi)


def duct_invert(
    u: np.ndarray, yy: np.ndarray, zz: np.ndarray, a: float, b: float, w: np.ndarray | None = None
):
    """Fit (da, db) [G analytic per candidate] of the series family to the
    measured 2D profile.  Coarse 2D grid scan (M=61) then alternating
    golden section; final scale at M=121."""
    if w is None:
        w = (u > 0.2 * u.max()).astype(np.float64)

    def S_of(da, db, M=_M):
        return duct_S(yy, zz, a + da, b + db, M=M)

    best = (None, None, float("inf"))
    for da in np.arange(-0.6, 0.61, 0.1):
        for db in np.arange(-0.6, 0.61, 0.1):
            s = _sse(u, S_of(da, db, M=61), w)
            if s < best[2]:
                best = (da, db, s)
    da, db = best[0], best[1]
    for _ in range(12):
        da = _golden(lambda x: _sse(u, S_of(x, db, M=61), w), da - 0.1, da + 0.1)
        db = _golden(lambda x: _sse(u, S_of(da, x, M=61), w), db - 0.1, db + 0.1)
    return float(da), float(db), _G_lsq(u, S_of(da, db, M=_M), w)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def duct_setup(W: int, H: int, L_over_W: int, device):
    """Fluid y in [1,H], z in [1,W]; wall planes y=0/H+1, z=0/W+1 (BB cells).

    Cross-section arrays are indexed [z, y] to match the tensor layout
    (19, nz, ny, nx): rows are z (length nz=W+2), cols are y (ny=H+2)."""
    ny, nz = H + 2, W + 2
    nx = L_over_W * W
    yy = np.arange(1, ny - 1, dtype=np.float64) - (H + 1) / 2.0  # (-H/2, H/2)
    zz = np.arange(1, nz - 1, dtype=np.float64) - (W + 1) / 2.0
    wall2d = np.zeros((nz, ny), dtype=bool)  # rows z, cols y
    wall2d[:, 0] = wall2d[:, -1] = True  # y-walls (columns)
    wall2d[0, :] = wall2d[-1, :] = True  # z-walls (rows)
    wall_mask = torch.from_numpy(wall2d).to(device).unsqueeze(-1).expand(nz, ny, nx).contiguous()
    return ny, nz, nx, yy, zz, wall_mask


def cell_max_l2(u: np.ndarray, uref: np.ndarray, U_max_ref: float):
    sel = uref > 0.2 * abs(U_max_ref)
    err = np.abs(u - uref)
    max_rel = float(err[sel].max() / abs(U_max_ref) * 100.0)
    l2 = float(np.linalg.norm(err[sel]) / np.linalg.norm(uref[sel]) * 100.0)
    return max_rel, l2, int(sel.sum())


# ---------------------------------------------------------------------------
# per-grid simulation
# ---------------------------------------------------------------------------
def run_case(
    W: int,
    aspect: float,
    tau: float,
    u_in: float,
    min_steps: int,
    max_steps: int,
    out_path: str,
    device,
    seed: int = 0,
    L_over_W: int = 4,
    compile_mode: str | None = "default",
    avg_steps: int = 400,
) -> dict:
    torch.manual_seed(seed)
    nu = (tau - 0.5) / 3.0
    H = int(round(W * aspect))
    a, b = W / 2.0, H / 2.0
    ny, nz, nx, yy, zz, wall_mask = duct_setup(W, H, L_over_W, device)

    inv_exact = duct_Po(a, b)
    G_nom = nu * u_in * (4.0 * a * b) / duct_I0(a, b)
    # cross-section arrays indexed [z, y] to match the tensor layout (nz, ny)
    zz2, yy2 = np.meshgrid(zz, yy, indexing="ij")  # both (W, H)
    u0_cell = duct_u(yy2, zz2, a, b, G_nom, nu)  # (W, H)

    rho_out = 1.0
    Dh = 4.0 * a * b / (a + b)
    Re = u_in * Dh / nu

    ux3 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
    u0_t = torch.from_numpy(u0_cell.astype(np.float32)).to(device).unsqueeze(-1)
    ux3[1:-1, 1:-1, :] = u0_t.expand(nz - 2, ny - 2, nx)
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    f = equilibrium3d(rho0, ux3, torch.zeros_like(rho0), torch.zeros_like(rho0), device=device)
    initial_mass = float(f.sum().item())

    def _step(f):
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        f = zou_he_inlet_velocity_3d(f, u_in)
        f = zou_he_outlet_pressure_3d(f, rho_out)
        return bounce_back_cells_3d(f, wall_mask)

    step_fn = route_step(_step, compile_mode, name=f"duct3d[W{W}xH{H}]")

    x_meas = nx // 2
    x_p1, x_p2 = nx // 4, (3 * nx) // 4

    t0 = time.time()
    umax_hist: list[float] = []
    step = 0
    steady = False
    for step in range(1, max_steps + 1):
        f = step_fn(f)
        if step % 200 == 0:
            _, ux, _, _ = macroscopic3d(f)
            umax_hist.append(float(ux[1:-1, 1:-1, x_meas].max().item()))
            if step >= min_steps and len(umax_hist) >= 10:
                recent = umax_hist[-10:]
                mean = sum(recent) / len(recent)
                drift = (max(recent) - min(recent)) / max(abs(mean), 1e-12)
                if drift < 1e-5:
                    steady = True
                    break
    elapsed = time.time() - t0

    acc = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho1 = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho2 = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    acc_rho_in = torch.zeros((nz, ny), dtype=torch.float32, device=device)
    for _ in range(avg_steps):
        f = step_fn(f)
        rho, ux, _, _ = macroscopic3d(f)
        acc += ux[:, :, x_meas]
        acc_rho1 += rho[:, :, x_p1]
        acc_rho2 += rho[:, :, x_p2]
        acc_rho_in += rho[:, :, 0]
    acc /= avg_steps
    acc_rho1 /= avg_steps
    acc_rho2 /= avg_steps
    acc_rho_in /= avg_steps
    elapsed = time.time() - t0

    u_meas = acc[1:-1, 1:-1].cpu().numpy().astype(np.float64)  # (H, W)
    rho_p1 = float(acc_rho1[1:-1, 1:-1].mean().item())
    rho_p2 = float(acc_rho2[1:-1, 1:-1].mean().item())
    rho_in = float(acc_rho_in[1:-1, 1:-1].mean().item())
    G_int = (rho_p1 - rho_p2) * CS2 / float(x_p2 - x_p1)
    Q_meas = float(u_meas.sum())
    u_mean_meas = Q_meas / (W * H)

    # ---- CH1: effective geometry inversion ---------------------------------
    da, db, G_fit = duct_invert(u_meas, yy2, zz2, a, b)
    a_eff, b_eff = a + da, b + db
    G_Q = nu * Q_meas / duct_I0(a_eff, b_eff)
    uref_eff = duct_u(yy2, zz2, a_eff, b_eff, G_Q, nu)
    U_max_ref = float(uref_eff.max())
    eff_max, eff_l2, n_sel = cell_max_l2(u_meas, uref_eff, U_max_ref)

    # ---- CH2: nominal frame ------------------------------------------------
    uref_nom = duct_u(yy2, zz2, a, b, G_nom, nu)
    U_max_nom = float(uref_nom.max())
    nom_max, nom_l2, _ = cell_max_l2(u_meas, uref_nom, U_max_nom)

    # Po channels.  Po_sim is the pure MEASUREMENT (interior pressure
    # gradient + measured mean + nominal Dh of the built duct; no inversion,
    # no nominal-anchor circularity) — the Shah-London cross-check.
    # Po_Q (= 2 G_Q Dh^2/(nu u_mean)) mixes the flux-anchored gradient with
    # the nominal Dh: it equals Po_exact * G_Q/G_nom and diagnoses the
    # effective-width bias.  Po_effgeom is Po_exact evaluated at the
    # inverted (a_eff, b_eff) — an identity of the inversion, disclosed.
    Po_sim = 2.0 * G_int * Dh**2 / (nu * u_mean_meas)
    Po_Q = 2.0 * G_Q * Dh**2 / (nu * u_mean_meas)
    Po_effgeom = duct_Po(a_eff, b_eff)["Po"]
    u_max_meas = float(u_meas.max())

    mass_drift_pct = (float(f.sum().item()) - initial_mass) / initial_mass * 100.0

    result = {
        "case": "poiseuille_3d_duct_rect",
        "lattice": "D3Q19",
        "collision": "bgk",
        "boundary": (
            "zou_he_velocity_inlet(x=0) + zou_he_pressure_outlet(x=nx-1) + "
            "swap bounce-back on the four lattice-aligned wall planes "
            "(y=0/H+1, z=0/W+1, post-streaming)"
        ),
        "W": W,
        "H": H,
        "aspect": aspect,
        "a": a,
        "b": b,
        "ny": ny,
        "nz": nz,
        "nx": nx,
        "L_over_W": float(nx / W),
        "tau": tau,
        "nu_lb": nu,
        "u_in": u_in,
        "rho_out": rho_out,
        "Re": Re,
        "Ma": U_max_nom / math.sqrt(CS2),
        "Po_exact": inv_exact["Po"],
        "umax_over_umean_exact": inv_exact["umax_over_umean"],
        "G_nom": G_nom,
        "U_max_nom": U_max_nom,
        "min_steps": min_steps,
        "n_steps": step,
        "avg_steps": avg_steps,
        "steady": steady,
        "compile_mode": compile_mode,
        # CH1
        "da": da,
        "db": db,
        "a_eff": a_eff,
        "b_eff": b_eff,
        "G_fit": G_fit * nu,
        "G_Q": G_Q,
        "G_fit_over_G_Q": G_fit * nu / G_Q,
        "G_int": G_int,
        "G_int_over_G_Q": G_int / G_Q,
        "eff_max_cell_central_pct": eff_max,
        "eff_l2_cell_central_pct": eff_l2,
        "n_central_cells": n_sel,
        # CH2
        "nom_max_cell_central_pct": nom_max,
        "nom_l2_cell_central_pct": nom_l2,
        # observables
        "Q_meas": Q_meas,
        "u_mean_meas": u_mean_meas,
        "u_max_meas": u_max_meas,
        "Po_sim": Po_sim,
        "Po_Q": Po_Q,
        "Po_effgeom": Po_effgeom,
        "Po_sim_err_pct": (Po_sim - inv_exact["Po"]) / inv_exact["Po"] * 100.0,
        "Po_Q_err_pct": (Po_Q - inv_exact["Po"]) / inv_exact["Po"] * 100.0,
        "umax_over_umean_meas": u_max_meas / u_mean_meas,
        "dp_inlet_outlet": (rho_in - rho_out) * CS2,
        "mass_drift_pct": mass_drift_pct,
        "mass_drift_expected_pct": G_Q * nx / (2.0 * CS2) * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": round(elapsed, 1),
        "u_plane": [[round(float(v), 7) for v in row] for row in u_meas],
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
                "W": c["W"],
                "H": c["H"],
                "Re": c["Re"],
                "n_steps": c["n_steps"],
                "steady": c["steady"],
                "eff_max_cell_central_pct": round(c["eff_max_cell_central_pct"], 4),
                "eff_l2_cell_central_pct": round(c["eff_l2_cell_central_pct"], 4),
                "nom_max_cell_central_pct": round(c["nom_max_cell_central_pct"], 4),
                "nom_l2_cell_central_pct": round(c["nom_l2_cell_central_pct"], 4),
                "da": round(c["da"], 4),
                "db": round(c["db"], 4),
                "G_fit_over_G_Q": round(c["G_fit_over_G_Q"], 5),
                "G_int_over_G_Q": round(c["G_int_over_G_Q"], 5),
                "Po_sim_err_pct": round(c["Po_sim_err_pct"], 4),
                "Po_Q_err_pct": round(c["Po_Q_err_pct"], 4),
                "mass_drift_pct": round(c["mass_drift_pct"], 4),
                "elapsed_s": c["elapsed_s"],
            }
        )
    errs_eff = [c["eff_max_cell_central_pct"] for c in convergence]
    errs_nom = [c["nom_max_cell_central_pct"] for c in convergence]
    das = [c["da"] for c in convergence]
    dbs = [c["db"] for c in convergence]
    spread_da, spread_db = max(das) - min(das), max(dbs) - min(dbs)
    delta_grid_indep = spread_da <= 0.05 and spread_db <= 0.05
    passed_eff = len(errs_eff) >= 2 and all(e <= 3.0 for e in errs_eff) and _mono(errs_eff)
    passed_nom = len(errs_nom) >= 2 and all(e <= 3.0 for e in errs_nom) and _mono(errs_nom)
    summary = {
        "case": "poiseuille_3d_duct_rect_convergence",
        "lattice": "D3Q19",
        "collision": "bgk",
        "aspect": cases[0]["aspect"],
        "W_list": [c["W"] for c in cases],
        "tau": cases[0]["tau"],
        "u_in": cases[0]["u_in"],
        "extrap": "none",
        "comparison_method": (
            "CH1 (primary): full 2D profile weighted LSQ of the exact double-"
            "sine series over (G, da, db) (da/db = effective half-width "
            "offsets, G analytic; coarse grid + alternating golden section), "
            "fit region u_meas>0.2 u_max; G anchored by imposed flux "
            "G_Q = nu Q_meas/I0(a_eff,b_eff); metric = per-cell central max "
            "rel err. CH2 (direct): nominal (a,b) series with G_nom from "
            "u_mean=u_in. Cross-checks: Po vs Shah-London, G_int, G_fit/G_Q."
        ),
        "per_grid": convergence,
        "eff_errors_pct": errs_eff,
        "nom_errors_pct": errs_nom,
        "eff_monotone": _mono(errs_eff),
        "nom_monotone": _mono(errs_nom),
        "da": das,
        "db": dbs,
        "da_spread": spread_da,
        "db_spread": spread_db,
        "delta_grid_independent_(bound_0.05)": delta_grid_indep,
        "passed_eff_3pct_monotone": passed_eff,
        "passed_nom_3pct_monotone": passed_nom,
        "verdict": "verified" if (passed_eff and delta_grid_indep) else "not_verified",
        "notes": (
            f"CH1 eff per-cell central max: {' -> '.join(f'{e:.3f}%' for e in errs_eff)}; "
            f"da: {' / '.join(f'{v:+.4f}' for v in das)}; "
            f"db: {' / '.join(f'{v:+.4f}' for v in dbs)} "
            f"(spreads {spread_da:.4f}/{spread_db:.4f}, bound 0.05: {delta_grid_indep}). "
            f"CH2 nominal: {' -> '.join(f'{e:.3f}%' for e in errs_nom)}. "
            f"Po_sim err: {' / '.join(f'{v:+.3f}%' for v in [c['Po_sim_err_pct'] for c in convergence])}."
        ),
    }
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    return summary


def default_steps(W: int) -> tuple[int, int]:
    if W <= 32:
        return 20000, 60000
    if W <= 64:
        return 30000, 90000
    return 40000, 150000


def scan(
    W_list,
    aspect,
    tau,
    u_in,
    out_dir,
    device,
    seed=0,
    compile_mode="default",
    L_over_W=4,
    min_steps=0,
    max_steps=0,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    for W in W_list:
        dmin, dmax = default_steps(W)
        dmin = min_steps if min_steps > 0 else dmin
        dmax = max_steps if max_steps > 0 else dmax
        p = out_dir / f"case_W{W}.json"
        r = run_case(
            W,
            aspect,
            tau,
            u_in,
            dmin,
            dmax,
            str(p),
            device,
            seed=seed,
            L_over_W=L_over_W,
            compile_mode=compile_mode,
        )
        cases.append(r)
        print(
            f"W={r['W']:4d} H={r['H']:4d} steps={r['n_steps']:6d} steady={r['steady']} "
            f"eff={r['eff_max_cell_central_pct']:.4f}% nom={r['nom_max_cell_central_pct']:.4f}% "
            f"da={r['da']:+.4f} db={r['db']:+.4f} Po_sim_err={r['Po_sim_err_pct']:+.3f}% "
            f"t={r['elapsed_s']:.0f}s",
            flush=True,
        )
    s = build_summary(cases, str(out_dir))
    print(f"verdict={s['verdict']} eff: {s['eff_errors_pct']} nom: {s['nom_errors_pct']}")
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="3D rectangular-duct Poiseuille (D3Q19)")
    sub = ap.add_subparsers(dest="mode_cmd", required=True)
    p1 = sub.add_parser("single")
    p1.add_argument("W", type=int)
    p1.add_argument("out_json", type=str)
    p1.add_argument("--aspect", type=float, default=1.0, help="H/W")
    p1.add_argument("--tau", type=float, default=0.8)
    p1.add_argument("--u-in", type=float, default=0.02)
    p1.add_argument("--min-steps", type=int, default=0)
    p1.add_argument("--max-steps", type=int, default=0)
    p1.add_argument("--L-over-W", type=int, default=4)
    p1.add_argument("--device", type=str, default="cuda:0")
    p1.add_argument("--seed", type=int, default=0)
    add_compile_mode_arg(p1)
    p2 = sub.add_parser("scan")
    p2.add_argument("out_dir", type=str)
    p2.add_argument("--W", type=int, nargs="+", default=[32, 64, 128])
    p2.add_argument("--aspect", type=float, default=1.0)
    p2.add_argument("--tau", type=float, default=0.8)
    p2.add_argument("--u-in", type=float, default=0.02)
    p2.add_argument("--min-steps", type=int, default=0)
    p2.add_argument("--max-steps", type=int, default=0)
    p2.add_argument("--L-over-W", type=int, default=4)
    p2.add_argument("--device", type=str, default="cuda:0")
    p2.add_argument("--seed", type=int, default=0)
    add_compile_mode_arg(p2)
    p3 = sub.add_parser("summarize")
    p3.add_argument("out_dir", type=str)
    p3.add_argument("--W", type=int, nargs="+", default=[32, 64, 128])
    args = ap.parse_args()
    if args.mode_cmd == "summarize":
        cases = [
            json.loads((Path(args.out_dir) / f"case_W{W}.json").read_text())
            for W in args.W
            if (Path(args.out_dir) / f"case_W{W}.json").exists()
        ]
        if not cases:
            print("no case JSONs found")
            return
        s = build_summary(cases, args.out_dir)
        print(f"verdict={s['verdict']} eff: {s['eff_errors_pct']} nom: {s['nom_errors_pct']}")
        return
    device = torch.device(args.device)
    compile_mode = compile_mode_from_args(args)
    if args.mode_cmd == "single":
        dmin, dmax = default_steps(args.W)
        run_case(
            args.W,
            args.aspect,
            args.tau,
            args.u_in,
            args.min_steps if args.min_steps > 0 else dmin,
            args.max_steps if args.max_steps > 0 else dmax,
            args.out_json,
            device,
            L_over_W=args.L_over_W,
            compile_mode=compile_mode,
        )
    else:
        scan(
            args.W,
            args.aspect,
            args.tau,
            args.u_in,
            args.out_dir,
            device,
            seed=args.seed,
            compile_mode=compile_mode,
            L_over_W=args.L_over_W,
            min_steps=args.min_steps,
            max_steps=args.max_steps,
        )


if __name__ == "__main__":
    main()
