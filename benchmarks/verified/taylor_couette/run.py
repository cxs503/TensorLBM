#!/usr/bin/env python
"""Annular Taylor-Couette laminar flow (D2Q9, 2D r-theta plane) — analytic validation.

Problem
-------
Concentric cylinders, inner cylinder rotating at omega_i (counter-clockwise),
outer cylinder at rest, radius ratio eta = r_i/r_o = 0.5, axially periodic
(the 2D r-theta plane is the exact representation: the laminar solution has
no axial dependence).

    Re_i = omega_i * r_i^2 / nu = 20  <<  Re_i,crit(eta=0.5) ~= 68.2
    (Esser & Grossmann 1996; for eta=0.5 the gap-based Re, Omega_i*r_i*d/nu,
    equals Re_i because d = r_o - r_i = r_i) — deep in the laminar
    axisymmetric (stable) regime, no Taylor vortices.

Analytic solution (steady, no-slip at both walls; exact, no literature data
needed):

    u_theta(r) = A*r + B/r,
    A = (Omega_o r_o^2 - Omega_i r_i^2)/(r_o^2 - r_i^2),
    B = r_i^2 r_o^2 (Omega_i - Omega_o)/(r_o^2 - r_i^2),   Omega_o = 0

    wall torque per unit axial length (identical magnitude on both walls):
    M = 4*pi*mu*L*B,  mu = rho0*nu, L = 1 (2D per-unit-depth lattice units).

Comparison method — effective staircase radii (R_eff^Q methodology of the
verified poiseuille_3d_pipe benchmark, transplanted to the annulus)
--------------------------------------------------------------------------------
The digital (staircase) bounce-back annulus is NOT an annulus of the nominal
radii (r_i, r_o).  Following the approved precedent, the geometric constants
are INVERTED from the flow itself instead of assuming the nominal radii
(which would be circular reasoning):

  1. Measure the azimuthally-averaged profile u_theta(r) (radial bins).
  2. Weighted 2-parameter least-squares fit of the exact Couette family
     u = A*r + B/r over the central bins (|u| > 20% of the wall speed,
     region selected with the NOMINAL analytic profile).
  3. Effective wall radii from the fitted solution's no-slip conditions:
        r_o_eff = sqrt(-B/A)                (u_theta(r_o_eff) = 0)
        r_i_eff = sqrt(B/(omega_i - A))     (u_theta(r_i_eff) = omega_i*r_i_eff,
                                             omega_i is the imposed quantity)
     The offsets delta_i = r_i_eff - r_i, delta_o = r_o_eff - r_o must come
     out as GRID-INDEPENDENT constants in lattice units across the mesh
     levels (the analog of poiseuille_3d_pipe's R_eff^Q - R = +0.11 at both
     R = 20 and R = 40).
  4. PRIMARY metric: binned profile vs the analytic Couette profile of the
     EFFECTIVE geometry, u_ref(r) = A*r + B/r, absolute normalization
     U_max_ref = omega_i * r_i_eff (imposed wall speed at the effective
     wall; equals A*r_i_eff + B/r_i_eff identically).  Max and weighted-L2
     relative error over the central region (u_ref > 0.2*U_max_ref).
  5. Independent cross-validation channel — wall torque, M_ref = 4*pi*nu*B.
     Primary: exact interface-flux bookkeeping — per boundary link the ACTUAL
     outgoing population g[q, x_s] and the ACTUAL returning population
     g[qbar, x_f] (whatever the collide/bounce-back scheme really reflects),
     momentum to the fluid with the link-midpoint lever.  A pure measurement
     with no reflection-model assumption, valid symmetrically for the moving
     and the static wall.  Secondary: full-way plain MEM on the static outer
     wall (the per-cell generalisation of boundaries.compute_obstacle_forces,
     the library's documented convention for stationary obstacles).
     Additionally disclosed: the library primitive d3q27.
     moving_wall_linkwise_me_force_torque (ideal halfway reflection model)
     overestimates the torque by ~2x on this scheme because the returning
     populations have been relaxed by the BGK collision — quantified in the
     results as a convention-mismatch disclosure.

  No extrapolation, no correction factors, no tuning (extrap: none).
  The NOMINAL-radii comparison is additionally disclosed to show the bias
  the effective-radius method removes.

Entry (common-module path, no hand-written solver kernels)
----------------------------------------------------------
  - tensorlbm.rotating_cylinder.rotating_wall_velocity        (wall velocity)
  - tensorlbm.rotating_cylinder.moving_wall_bounce_back       (Ladd moving wall)
  - tensorlbm.solver.collide_bgk / tensorlbm.solver.stream    (D2Q9 BGK path)
  - tensorlbm.d2q9.equilibrium / macroscopic
  - tensorlbm.boundaries.cylinder_mask / bounce_back_cells / compute_obstacle_forces
  - tensorlbm.d3q27.moving_wall_linkwise_me_force_torque      (torque diagnostic)
  - benchmarks/compile_route.route_step                       (shared compile wrapper)

Geometry: square grid n = 2*r_o + 3, centre ((n-1)/2, (n-1)/2); fluid =
{r_i < d <= r_o}; inner disk (d <= r_i) = rotating solid (moving-wall
bounce-back, the whole obstacle mask exactly as run_rotating_cylinder uses
it); exterior (d > r_o) = static solid (bounce-back).  Streaming is periodic
over the box; the whole box border is solid, so the wrap connects solid to
solid only.

Usage
-----
    run.py single --ri 24 --out case_ri24.json [--re-i 20] [--nu 0.05]
        [--eta 0.5] [--device cuda] [--min-mult 3.5] [--max-mult 10.0]
        [--sample-interval 200] [--avg-steps 1000] [--seed 0]
    run.py scan --out-dir DIR --ri 24 48 96 [same options]   (sequential)
    run.py aggregate --out-dir DIR --ri 24 48 96             (from case jsons)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # <repo>/benchmarks

import numpy as np
import torch  # noqa: E402
from compile_route import (  # noqa: E402
    add_compile_mode_arg,
    compile_mode_from_args,
    route_step,
)

from tensorlbm.boundaries import (  # noqa: E402
    bounce_back_cells,
    compute_obstacle_forces,
    cylinder_mask,
)
from tensorlbm.d2q9 import C, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.d3q27 import moving_wall_linkwise_me_force_torque  # noqa: E402
from tensorlbm.rotating_cylinder import (  # noqa: E402
    moving_wall_bounce_back,
    rotating_wall_velocity,
)
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

CS2 = 1.0 / 3.0


# --------------------------------------------------------------------------- #
# geometry / link bookkeeping
# --------------------------------------------------------------------------- #
def annulus_setup(ri: int, ro: int, omega_i: float, device: torch.device):
    n = 2 * ro + 3
    cc = (n - 1) / 2.0
    inner = cylinder_mask(n, n, cc, cc, float(ri), device=device)  # d <= r_i
    outer = ~cylinder_mask(n, n, cc, cc, float(ro), device=device)  # d > r_o
    fluid = ~inner & ~outer
    yy, xx = torch.meshgrid(
        torch.arange(n, device=device, dtype=torch.float32),
        torch.arange(n, device=device, dtype=torch.float32),
        indexing="ij",
    )
    d = torch.sqrt((xx - cc) ** 2 + (yy - cc) ** 2)
    # rigid-body wall velocity field of the rotating inner cylinder (module fn)
    ux_w, uy_w = rotating_wall_velocity(inner, cc, cc, omega_i)
    return SimpleNamespace(
        n=n,
        c=cc,
        inner=inner,
        outer=outer,
        fluid=fluid,
        d=d,
        xx=xx,
        yy=yy,
        ux_w=ux_w,
        uy_w=uy_w,
    )


def _opposite_map() -> dict[int, int]:
    """Direction index map q -> qbar with c_qbar = -c_q (from tensorlbm.d2q9 C)."""
    c_cpu = C.cpu()
    opp: dict[int, int] = {}
    for a in range(9):
        for b in range(9):
            if int(c_cpu[a, 0]) == -int(c_cpu[b, 0]) and int(c_cpu[a, 1]) == -int(c_cpu[b, 1]):
                opp[a] = b
    return opp


OPPOSITE_Q = _opposite_map()


def build_wall_links(geo, solid: torch.Tensor):
    """Enumerate fluid->solid boundary links for the wall-torque diagnostics.

    One link per (fluid cell x_f, lattice direction q in 1..8) whose neighbour
    x_f + c_q is solid.  Kept per direction: flat indices of the fluid cell
    (for the returning population g[qbar, x_f]) and of the solid endpoint
    (for the outgoing population g[q, x_s]).  Also precomputed per link:
    the lever coefficients cx*ly - cy*lx of the link midpoint about the
    annulus centre (for the z-torque) and the link tables consumed by the
    library's linkwise MEM primitive (dirs3/weights/pos3/sol_all).
    """
    n = geo.n
    device = geo.fluid.device
    c_f = C.to(device).to(torch.float32)
    w_f = W.to(device).to(torch.float32)
    fluid_by_q: dict[int, torch.Tensor] = {}
    sol_by_q: dict[int, torch.Tensor] = {}
    coef_by_q: dict[int, torch.Tensor] = {}
    parts_sol, parts_px, parts_py, parts_q = [], [], [], []
    for q in range(1, 9):
        cxq = int(C.cpu()[q, 0].item())
        cyq = int(C.cpu()[q, 1].item())
        # shifted[y, x] = solid[y + cyq, x + cxq]
        shifted = torch.roll(solid, shifts=(-cyq, -cxq), dims=(0, 1))
        lm = geo.fluid & shifted
        flat_f = lm.reshape(-1).nonzero(as_tuple=False).squeeze(1)
        fluid_by_q[q] = flat_f
        sol_by_q[q] = torch.empty(0, dtype=torch.int64, device=device)
        coef_by_q[q] = torch.empty(0, device=device)
        if flat_f.numel() == 0:
            continue
        ys = flat_f // n
        xs = flat_f % n
        sol = (ys + cyq) * n + (xs + cxq)
        sol_by_q[q] = sol
        lx = xs.to(torch.float32) + 0.5 * cxq - geo.c
        ly = ys.to(torch.float32) + 0.5 * cyq - geo.c
        coef_by_q[q] = cxq * ly - cyq * lx
        parts_sol.append(sol)
        parts_px.append(xs.to(torch.float32) + 0.5 * cxq)
        parts_py.append(ys.to(torch.float32) + 0.5 * cyq)
        parts_q.append(torch.full((flat_f.numel(),), q, dtype=torch.int64, device=device))
    if not parts_sol:
        return SimpleNamespace(
            sol_all=None, fluid_by_q=fluid_by_q, sol_by_q=sol_by_q, coef_by_q=coef_by_q
        )
    sol_all = torch.cat(parts_sol)
    qx = torch.cat(parts_q)
    dirs3 = torch.stack([c_f[qx, 0], c_f[qx, 1], torch.zeros_like(c_f[qx, 0])], dim=1)
    weights = w_f[qx]
    pos3 = torch.stack(
        [torch.cat(parts_px), torch.cat(parts_py), torch.zeros_like(sol_all, dtype=torch.float32)],
        dim=1,
    )
    return SimpleNamespace(
        sol_all=sol_all,
        fluid_by_q=fluid_by_q,
        sol_by_q=sol_by_q,
        coef_by_q=coef_by_q,
        dirs3=dirs3,
        weights=weights,
        pos3=pos3,
    )


def interface_wall_torque(g: torch.Tensor, links) -> float:
    """z-torque (CCW+) ON THE FLUID from one wall — exact interface bookkeeping.

    Per boundary link (fluid x_f, direction q into the solid) the actual
    outgoing population g[q, x_s]*c_q leaves the fluid and the ACTUAL
    returning population g[qbar, x_f]*c_qbar arrives (whatever the collide/
    bounce-back scheme really reflects); momentum to the fluid per link is
    g[qbar, x_f]*c_qbar - g[q, x_s]*c_q, taken with the link-midpoint lever.
    This is a pure measurement (no reflection-model assumption) and works
    symmetrically for the static and the moving wall.
    """
    if links.sol_all is None:
        return 0.0
    total = 0.0
    for q in range(1, 9):
        idx_s = links.sol_by_q[q]
        if idx_s.numel() == 0:
            continue
        s = g[q].reshape(-1)[idx_s] + g[OPPOSITE_Q[q]].reshape(-1)[links.fluid_by_q[q]]
        total += float((s * links.coef_by_q[q]).sum().item())
    return total


def link_wall_torque(g: torch.Tensor, links, uw3: torch.Tensor, cc: float) -> float:
    """DISCLOSED variant: halfway linkwise moving-wall MEM torque via the
    library primitive tensorlbm.d3q27.moving_wall_linkwise_me_force_torque.

    The primitive models the reflected population as the ideal halfway
    reflection f_r = f_o - 2*rho*w*(c.u_w)/cs^2.  In this scheme the
    populations returning from the solid have been RELAXED by the BGK
    collision (collide runs over all cells before the post-streaming
    bounce-back swap), so the ideal-reflection model does not describe the
    actual reflection; this channel overestimates the torque (quantified in
    the results) and is reported only to document the convention mismatch.
    """
    if links.sol_all is None:
        return 0.0
    outs = [g[q].reshape(-1)[links.sol_by_q[q]] for q in range(1, 9)]
    out = torch.cat([o for o in outs if o.numel() > 0])
    rho_s = g.sum(dim=0).reshape(-1)[links.sol_all]
    _, _, _, tq = moving_wall_linkwise_me_force_torque(
        out,
        links.dirs3,
        links.weights,
        uw3,
        links.pos3,
        origin=(cc, cc, 0.0),
        density=rho_s,
    )
    return float(tq[2].item())


def make_plain_torque_outer(geo):
    """Full-way (wet-node) plain MEM z-torque on the STATIC outer wall — the
    per-cell generalisation of boundaries.compute_obstacle_forces
    (F_alpha = 2*sum_q c_qa f_q at solid cells, post-stream pre-bounce, the
    library's documented convention for STATIONARY obstacles) with the lever
    arm r x F.  Returns the torque ON THE WALL."""
    c_f = C.to(geo.fluid.device).to(torch.float32)
    lev = c_f[:, 1].view(9, 1, 1) * (geo.xx - geo.c).unsqueeze(0) - c_f[:, 0].view(9, 1, 1) * (
        geo.yy - geo.c
    ).unsqueeze(0)
    outer3 = geo.outer.unsqueeze(0).to(torch.float32)

    def _tq_outer(g: torch.Tensor) -> float:
        return float((2.0 * (lev * g * outer3).sum()).item())

    return _tq_outer


# --------------------------------------------------------------------------- #
# profile analysis
# --------------------------------------------------------------------------- #
def couette_fit(r: np.ndarray, u: np.ndarray, w: np.ndarray, sel: np.ndarray):
    """Weighted least-squares fit u ~= A*r + B/r on the selected bins."""
    x = np.stack([r[sel], 1.0 / r[sel]], axis=1)
    ws = w[sel]
    beta, *_ = np.linalg.lstsq(x * ws[:, None], u[sel] * ws, rcond=None)
    return float(beta[0]), float(beta[1])


def analyse_profile(
    u_th: np.ndarray,  # per-cell tangential velocity (time-averaged)
    d: np.ndarray,  # per-cell radius
    fluid: np.ndarray,  # fluid mask
    ri: float,
    ro: float,
    omega_i: float,
) -> dict:
    # --- radial bins: bin k holds cells with k <= d < k+1 ------------------
    k = np.floor(d).astype(int)
    rs, us, ws, ks = [], [], [], []
    for kk in range(int(np.floor(ri)) - 1, int(np.ceil(ro)) + 2):
        m = fluid & (k == kk)
        if m.sum() == 0:
            continue
        ks.append(kk)
        rs.append(float(d[m].mean()))
        us.append(float(u_th[m].mean()))
        ws.append(int(m.sum()))
    r = np.asarray(rs)
    ub = np.asarray(us)
    w = np.asarray(ws, dtype=np.float64)

    # --- nominal analytic solution (for fit-region selection + disclosure) -
    a_nom = -omega_i * ri**2 / (ro**2 - ri**2)
    b_nom = omega_i * ri**2 * ro**2 / (ro**2 - ri**2)
    u_nom = a_nom * r + b_nom / r
    umax_nom = omega_i * ri

    # --- fit on central bins (region from the NOMINAL analytic profile) ----
    sel_fit = u_nom > 0.2 * umax_nom
    a_fit, b_fit = couette_fit(r, ub, w, sel_fit)
    if not (a_fit < 0.0 and b_fit > 0.0 and omega_i - a_fit > 0.0):
        raise RuntimeError(f"unphysical Couette fit: A={a_fit}, B={b_fit}")
    ro_eff = math.sqrt(-b_fit / a_fit)
    ri_eff = math.sqrt(b_fit / (omega_i - a_fit))
    umax_ref = omega_i * ri_eff  # == A*r_i_eff + B/r_i_eff identically
    ident_check = a_fit * ri_eff + b_fit / ri_eff

    # all-bins fit variant (robustness disclosure)
    sel_all = np.ones_like(sel_fit)
    a_all, b_all = couette_fit(r, ub, w, sel_all)
    ro_eff_ab = math.sqrt(-b_all / a_all) if a_all < 0 else float("nan")
    ri_eff_ab = math.sqrt(b_all / (omega_i - a_all)) if omega_i - a_all > 0 else float("nan")

    # --- primary metrics: binned profile vs effective-geometry analytic -----
    u_ref = a_fit * r + b_fit / r
    sel = u_ref > 0.2 * umax_ref
    err_abs = np.abs(ub - u_ref)
    max_bin = float(err_abs[sel].max() / umax_ref * 100.0)
    l2_bin = float(
        np.linalg.norm(np.sqrt(w[sel]) * (ub - u_ref)[sel])
        / np.linalg.norm(np.sqrt(w[sel]) * u_ref[sel])
        * 100.0
    )
    # shape-normalized variant (rescale reference by innermost-bin amplitude)
    i0 = int(np.argmax(u_ref))
    s_shape = ub[i0] / u_ref[i0]
    u_sh = u_ref * s_shape
    max_bin_sh = float(np.abs(ub[sel] - u_sh[sel]).max() / abs(ub[i0]) * 100.0)
    l2_bin_sh = float(
        np.linalg.norm(np.sqrt(w[sel]) * (ub - u_sh)[sel])
        / np.linalg.norm(np.sqrt(w[sel]) * u_sh[sel])
        * 100.0
    )
    # per-cell (stricter) disclosure
    u_ref_cell = a_fit * d + b_fit / d
    mc = fluid & (u_ref_cell > 0.2 * umax_ref)
    max_cell = float(np.abs(u_th[mc] - u_ref_cell[mc]).max() / umax_ref * 100.0)
    imax_cell = int(np.argmax(np.abs(u_th[mc] - u_ref_cell[mc]) / umax_ref))
    cells_r = np.asarray(np.nonzero(mc))
    worst_cell = (
        float(d[cells_r[0, imax_cell], cells_r[1, imax_cell]]),
        float(u_ref_cell[cells_r[0, imax_cell], cells_r[1, imax_cell]]),
    )
    # nominal-radii comparison (disclosure: the bias R_eff removes)
    err_nom = np.abs(ub - u_nom)
    sel_nom = u_nom > 0.2 * umax_nom
    max_bin_nom = float(err_nom[sel_nom].max() / umax_nom * 100.0)
    l2_bin_nom = float(
        np.linalg.norm(np.sqrt(w[sel_nom]) * (ub - u_nom)[sel_nom])
        / np.linalg.norm(np.sqrt(w[sel_nom]) * u_nom[sel_nom])
        * 100.0
    )
    # mid-gap value vs nominal analytic (single-number disclosure)
    r_mid = 0.5 * (ri + ro)
    imid = int(np.argmin(np.abs(r - r_mid)))
    u_mid_nom = a_nom * r[imid] + b_nom / r[imid]
    mid_nom_err = float((ub[imid] - u_mid_nom) / u_mid_nom * 100.0)
    u_mid_eff = a_fit * r[imid] + b_fit / r[imid]
    mid_eff_err = float((ub[imid] - u_mid_eff) / u_mid_eff * 100.0)

    return {
        "A_fit": a_fit,
        "B_fit": b_fit,
        "ri_eff": ri_eff,
        "ro_eff": ro_eff,
        "delta_i": ri_eff - ri,
        "delta_o": ro_eff - ro,
        "eta_eff": ri_eff / ro_eff,
        "u_max_ref": umax_ref,
        "identity_check_umax": ident_check,
        "fit_bins_used": int(sel_fit.sum()),
        "fit_bins_total": int(len(r)),
        "ri_eff_allbins": ri_eff_ab,
        "ro_eff_allbins": ro_eff_ab,
        "delta_i_allbins": (ri_eff_ab - ri) if math.isfinite(ri_eff_ab) else float("nan"),
        "delta_o_allbins": (ro_eff_ab - ro) if math.isfinite(ro_eff_ab) else float("nan"),
        "profile_max_rel_bin_central_pct": max_bin,
        "profile_l2_rel_bin_central_pct": l2_bin,
        "profile_max_rel_bin_central_shape_pct": max_bin_sh,
        "profile_l2_rel_bin_central_shape_pct": l2_bin_sh,
        "percell_max_rel_central_pct": max_cell,
        "percell_worst_radius": worst_cell[0],
        "percell_worst_uref": worst_cell[1],
        "nominal_max_rel_bin_central_pct": max_bin_nom,
        "nominal_l2_rel_bin_central_pct": l2_bin_nom,
        "midgap_radius": float(r[imid]),
        "midgap_u_sim": float(ub[imid]),
        "midgap_err_vs_nominal_pct": mid_nom_err,
        "midgap_err_vs_effective_pct": mid_eff_err,
        "bins_k": ks,
        "bins_r": [round(float(v), 5) for v in r],
        "bins_u": [round(float(v), 8) for v in ub],
        "bins_uref": [round(float(v), 8) for v in u_ref],
        "bins_unom": [round(float(v), 8) for v in u_nom],
        "bins_cells": [int(v) for v in w],
        "n_fluid_cells": int(fluid.sum()),
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def run_case(
    ri: int,
    eta: float,
    re_i: float,
    nu: float,
    out_json: str,
    device: torch.device,
    min_mult: float = 3.5,
    max_mult: float = 10.0,
    sample_interval: int = 200,
    avg_steps: int = 1000,
    seed: int = 0,
    compile_mode: str | None = "default",
) -> dict:
    ro = int(round(ri / eta))
    if abs(ri / ro - eta) > 1e-9:
        raise ValueError(f"ri={ri} does not give eta={eta} with integer ro")
    omega_i = re_i * nu / ri**2
    tau = 3.0 * nu + 0.5
    t_gap = (ro - ri) ** 2 / nu
    min_steps = int(min_mult * t_gap)
    max_steps = int(max_mult * t_gap)
    u_wall = omega_i * ri
    ma = u_wall / math.sqrt(CS2)

    torch.manual_seed(seed)
    geo = annulus_setup(ri, ro, omega_i, device)

    links_in = build_wall_links(geo, geo.inner)
    links_out = build_wall_links(geo, geo.outer)
    zeros3 = torch.zeros_like(links_out.pos3) if links_out.sol_all is not None else None
    uw3_in = (
        torch.stack(
            [
                geo.ux_w.reshape(-1)[links_in.sol_all],
                geo.uy_w.reshape(-1)[links_in.sol_all],
                torch.zeros_like(links_in.sol_all.to(torch.float32)),
            ],
            dim=1,
        )
        if links_in.sol_all is not None
        else None
    )
    plain_tq_out = make_plain_torque_outer(geo)

    n = geo.n
    rho0 = torch.ones((n, n), device=device, dtype=torch.float32)
    zeros = torch.zeros_like(rho0)
    f = equilibrium(rho0, zeros, zeros)  # start from REST
    mass0 = float(f.sum().item())

    inner_m, outer_m, uxw, uyw = geo.inner, geo.outer, geo.ux_w, geo.uy_w

    def _pre(f):
        return stream(collide_bgk(f, tau))

    def _post(f):
        f2 = moving_wall_bounce_back(f, inner_m, uxw, uyw)
        return bounce_back_cells(f2, outer_m)

    pre = route_step(_pre, compile_mode, name=f"taylor_couette[ri{ri}] collide+stream")
    post = route_step(_post, compile_mode, name=f"taylor_couette[ri{ri}] walls")

    print(
        f"[tc ri={ri}] n={n} tau={tau:.4f} omega_i={omega_i:.6e} u_wall={u_wall:.5f} "
        f"Re_i={re_i} Ma={ma:.4f} min_steps={min_steps} max_steps={max_steps} "
        f"fluid_cells={int(geo.fluid.sum())} links_in={links_in.sol_all.numel() if links_in.sol_all is not None else 0} "
        f"links_out={links_out.sol_all.numel() if links_out.sol_all is not None else 0}",
        flush=True,
    )

    def _torque_channels(g):
        return (
            interface_wall_torque(g, links_in),
            interface_wall_torque(g, links_out),
            plain_tq_out(g),
            link_wall_torque(g, links_in, uw3_in, geo.c),
            link_wall_torque(g, links_out, zeros3, geo.c),
        )

    # ---- spin-up to steady state (start from rest) ------------------------
    umax_hist: list[float] = []
    steady = False
    step = 0
    t0 = time.time()
    while step < max_steps:
        g = None
        for _ in range(sample_interval):
            g = pre(f)
            f = post(g)
            step += 1
        _, ux, uy = macroscopic(g)
        u_th = ((geo.xx - geo.c) * uy - (geo.yy - geo.c) * ux) / geo.d
        umax = float(u_th[geo.fluid].abs().max().item())
        if not math.isfinite(umax):
            raise RuntimeError(f"non-finite state at step {step}")
        umax_hist.append(umax)
        if step % (sample_interval * 10) == 0:
            print(f"[tc ri={ri}] step={step} umax={umax:.6e}", flush=True)
        if step >= min_steps and len(umax_hist) >= 10:
            rec = umax_hist[-10:]
            drift = (max(rec) - min(rec)) / max(abs(sum(rec) / len(rec)), 1e-12)
            if drift < 1e-5:
                steady = True
                print(
                    f"[tc ri={ri}] steady at step={step} (drift={drift:.2e}, umax={umax:.6e})",
                    flush=True,
                )
                break
    elapsed = time.time() - t0

    # ---- averaging window: fields + per-step torque channels ---------------
    acc_rho = torch.zeros((n, n), device=device)
    acc_ux = torch.zeros((n, n), device=device)
    acc_uy = torch.zeros((n, n), device=device)
    tq_if_in: list[float] = []
    tq_if_out: list[float] = []
    tq_fw_out: list[float] = []
    tq_lw_in: list[float] = []
    tq_lw_out: list[float] = []
    for _ in range(avg_steps):
        g = pre(f)
        a, b, c, d_, e = _torque_channels(g)
        tq_if_in.append(a)
        tq_if_out.append(b)
        tq_fw_out.append(c)
        tq_lw_in.append(d_)
        tq_lw_out.append(e)
        f = post(g)
        rho, ux, uy = macroscopic(g)
        acc_rho += rho
        acc_ux += ux
        acc_uy += uy
    acc_rho /= avg_steps
    acc_ux /= avg_steps
    acc_uy /= avg_steps
    sec_per_step = elapsed / max(step, 1)

    # net plain-MEM force on the inner cylinder (library fn; ~0 sanity check)
    fx, fy = compute_obstacle_forces(g, geo.inner)

    u_th_mean = (
        (((geo.xx - geo.c) * acc_uy - (geo.yy - geo.c) * acc_ux) / geo.d)
        .cpu()
        .numpy()
        .astype(np.float64)
    )
    d_np = geo.d.cpu().numpy().astype(np.float64)
    fluid_np = geo.fluid.cpu().numpy()

    prof = analyse_profile(u_th_mean, d_np, fluid_np, float(ri), float(ro), omega_i)

    # --- torque channels -----------------------------------------------------
    nu_eff = nu  # mu = rho0*nu with rho0 = 1, L = 1 (per unit depth)
    m_ref = 4.0 * math.pi * nu_eff * prof["B_fit"]
    # reference built explicitly from the inverted effective radii (identity
    # check against B_fit):
    b_from_radii = (
        omega_i
        * prof["ri_eff"] ** 2
        * prof["ro_eff"] ** 2
        / (prof["ro_eff"] ** 2 - prof["ri_eff"] ** 2)
    )
    m_from_radii = 4.0 * math.pi * nu_eff * b_from_radii
    t_if_in = float(np.mean(tq_if_in))
    t_if_out = float(np.mean(tq_if_out))
    t_fw_out = float(np.mean(tq_fw_out))
    t_lw_in = float(np.mean(tq_lw_in))
    t_lw_out = float(np.mean(tq_lw_out))
    torque = {
        "M_ref_from_Bfit": m_ref,
        "M_ref_from_radii": m_from_radii,
        "M_identity_rel_diff_pct": abs(m_from_radii - m_ref) / m_ref * 100.0,
        "T_if_inner_on_fluid_mean": t_if_in,
        "T_if_outer_on_fluid_mean": t_if_out,
        "T_fw_outer_on_solid_mean": t_fw_out,
        "T_lw_inner_mean_disclosed": t_lw_in,
        "T_lw_outer_mean_disclosed": t_lw_out,
        "T_if_inner_err_pct": (t_if_in - m_ref) / m_ref * 100.0,
        "T_if_outer_err_pct": (t_if_out + m_ref) / m_ref * 100.0,
        "T_fw_outer_err_pct": (t_fw_out - m_ref) / m_ref * 100.0,
        "T_lw_inner_ratio_disclosed": t_lw_in / m_ref,
        "T_lw_outer_ratio_disclosed": t_lw_out / m_ref,
        "T_if_balance_pct": (t_if_in + t_if_out) / m_ref * 100.0,
        "T_if_inner_std": float(np.std(tq_if_in)),
        "net_force_inner_fx": float(fx.item()),
        "net_force_inner_fy": float(fy.item()),
        "net_force_inner_mag_rel_M": (math.hypot(float(fx.item()), float(fy.item())) / abs(m_ref)),
    }

    mass_drift_pct = (float(f.sum().item()) - mass0) / mass0 * 100.0
    finite = bool(torch.isfinite(f).all().item())

    result = {
        "case": "taylor_couette_annulus",
        "lattice": "D2Q9",
        "collision": "bgk",
        "boundary": (
            "inner cylinder d<=r_i: Ladd moving-wall bounce-back "
            "(tensorlbm.rotating_cylinder.moving_wall_bounce_back, whole obstacle "
            "mask, u_w from rotating_wall_velocity); outer wall d>r_o: static "
            "bounce_back_cells; streaming periodic (box border fully solid)"
        ),
        "entry": (
            "tensorlbm.rotating_cylinder public functions (rotating_wall_velocity, "
            "moving_wall_bounce_back) + tensorlbm.solver collide_bgk/stream + "
            "d2q9 equilibrium/macroscopic + boundaries cylinder_mask/"
            "bounce_back_cells/compute_obstacle_forces + d3q27."
            "moving_wall_linkwise_me_force_torque (torque diagnostic); "
            "compile via benchmarks/compile_route"
        ),
        "ri": ri,
        "ro": ro,
        "eta": eta,
        "n_grid": n,
        "re_i": re_i,
        "re_i_critical_note": "Re_i,crit(eta=0.5) ~= 68.2 (Esser & Grossmann 1996)",
        "nu": nu,
        "tau": tau,
        "omega_i": omega_i,
        "u_wall_nominal": u_wall,
        "Ma": ma,
        "t_gap_steps": t_gap,
        "min_steps": min_steps,
        "max_steps": max_steps,
        "n_steps": step,
        "avg_steps": avg_steps,
        "sample_interval": sample_interval,
        "steady": steady,
        "sec_per_step": round(sec_per_step, 6),
        "elapsed_s": round(elapsed, 1),
        "seed": seed,
        "device": str(device),
        "compile_mode": compile_mode,
        "extrap": "none",
        "mass_drift_pct": mass_drift_pct,
        "finite": finite,
        "comparison_method": (
            "effective staircase radii: 2-param weighted LSQ fit of the exact "
            "Couette family A*r+B/r on central bins; r_o_eff=sqrt(-B/A), "
            "r_i_eff=sqrt(B/(omega_i-A)) from the fitted no-slip conditions; "
            "primary comparison u_ref=A*r+B/r with absolute normalization "
            "U_max_ref=omega_i*r_i_eff (imposed omega_i)"
        ),
        "primary_metric": (
            "max & weighted-L2 relative error of the azimuthally-averaged binned "
            "profile vs the effective-geometry Couette solution over the central "
            "region (u_ref > 0.2*U_max_ref)"
        ),
        **prof,
        **torque,
    }
    Path(out_json).write_text(json.dumps(result, indent=2))
    print(
        f"[tc ri={ri}] DONE steps={step} steady={steady} "
        f"max_bin={prof['profile_max_rel_bin_central_pct']:.3f}% "
        f"l2_bin={prof['profile_l2_rel_bin_central_pct']:.3f}% "
        f"delta_i={prof['delta_i']:+.3f} delta_o={prof['delta_o']:+.3f} "
        f"T_if_in_err={torque['T_if_inner_err_pct']:+.3f}% "
        f"T_if_out_err={torque['T_if_outer_err_pct']:+.3f}% "
        f"balance={torque['T_if_balance_pct']:+.3f}% "
        f"T_fw_out_err={torque['T_fw_outer_err_pct']:+.3f}%",
        flush=True,
    )
    return result


# --------------------------------------------------------------------------- #
# scan / aggregate
# --------------------------------------------------------------------------- #
def summarize(cases: list[dict]) -> dict:
    errs_max = [c["profile_max_rel_bin_central_pct"] for c in cases]
    errs_l2 = [c["profile_l2_rel_bin_central_pct"] for c in cases]
    err_decreased = all(errs_max[i + 1] < errs_max[i] for i in range(len(errs_max) - 1)) and all(
        errs_l2[i + 1] < errs_l2[i] for i in range(len(errs_l2) - 1)
    )
    passed_profile = all(e <= 3.0 for e in errs_max) and all(e <= 3.0 for e in errs_l2)
    di = [c["delta_i"] for c in cases]
    do = [c["delta_o"] for c in cases]
    t_errs = [c["T_if_inner_err_pct"] for c in cases]
    balance = [c["T_if_balance_pct"] for c in cases]
    torque_ok = all(abs(e) <= 3.0 for e in t_errs)
    per_grid = [
        {
            "ri": c["ri"],
            "ro": c["ro"],
            "n_grid": c["n_grid"],
            "Re_i": c["re_i"],
            "tau": c["tau"],
            "omega_i": c["omega_i"],
            "n_steps": c["n_steps"],
            "steady": c["steady"],
            "ri_eff": round(c["ri_eff"], 4),
            "ro_eff": round(c["ro_eff"], 4),
            "delta_i": round(c["delta_i"], 4),
            "delta_o": round(c["delta_o"], 4),
            "delta_i_allbins": round(c["delta_i_allbins"], 4),
            "delta_o_allbins": round(c["delta_o_allbins"], 4),
            "profile_max_rel_bin_central_pct": round(c["profile_max_rel_bin_central_pct"], 4),
            "profile_l2_rel_bin_central_pct": round(c["profile_l2_rel_bin_central_pct"], 4),
            "percell_max_rel_central_pct": round(c["percell_max_rel_central_pct"], 4),
            "profile_max_rel_bin_central_shape_pct": round(
                c["profile_max_rel_bin_central_shape_pct"], 4
            ),
            "nominal_max_rel_bin_central_pct": round(c["nominal_max_rel_bin_central_pct"], 4),
            "nominal_l2_rel_bin_central_pct": round(c["nominal_l2_rel_bin_central_pct"], 4),
            "midgap_err_vs_nominal_pct": round(c["midgap_err_vs_nominal_pct"], 4),
            "T_if_inner_err_pct": round(c["T_if_inner_err_pct"], 4),
            "T_if_outer_err_pct": round(c["T_if_outer_err_pct"], 4),
            "T_fw_outer_err_pct": round(c["T_fw_outer_err_pct"], 4),
            "T_if_balance_pct": round(c["T_if_balance_pct"], 4),
            "T_lw_inner_ratio_disclosed": round(c["T_lw_inner_ratio_disclosed"], 4),
            "T_lw_outer_ratio_disclosed": round(c["T_lw_outer_ratio_disclosed"], 4),
            "M_ref_from_Bfit": round(c["M_ref_from_Bfit"], 6),
        }
        for c in cases
    ]
    passed = passed_profile and err_decreased and len(cases) >= 2
    summary = {
        "case": "taylor_couette_annulus_convergence",
        "lattice": "D2Q9",
        "collision": "bgk",
        "problem": (
            "annular Taylor-Couette laminar flow, eta=0.5, inner rotating / outer "
            "at rest, Re_i=20 (<< Re_crit~68), axial-periodic 2D r-theta plane"
        ),
        "boundary": cases[0]["boundary"],
        "entry": cases[0]["entry"],
        "extrap": "none",
        "comparison_method": cases[0]["comparison_method"],
        "primary_metric": cases[0]["primary_metric"],
        "ri_list": [c["ri"] for c in cases],
        "per_grid": per_grid,
        "convergence": {
            "err_decreased": err_decreased,
            "profile_max_pct": errs_max,
            "profile_l2_pct": errs_l2,
            "delta_i_values": di,
            "delta_o_values": do,
            "delta_i_spread_lattice": max(di) - min(di),
            "delta_o_spread_lattice": max(do) - min(do),
            "geometry_constants_grid_independent": (
                (max(di) - min(di) <= 0.25) and (max(do) - min(do) <= 0.25)
            ),
        },
        "torque_crosscheck": {
            "T_if_inner_err_pct": t_errs,
            "T_if_outer_err_pct": [c["T_if_outer_err_pct"] for c in cases],
            "T_fw_outer_err_pct": [c["T_fw_outer_err_pct"] for c in cases],
            "T_if_balance_pct": balance,
            "all_within_3pct": torque_ok,
        },
        "passed_3pct_and_converged": passed,
        "verdict": "verified" if passed else "not_verified",
        "notes": (
            f"effective-geometry Couette comparison: per-bin profile max "
            f"{' -> '.join(f'{e:.2f}%' for e in errs_max)}, L2 "
            f"{' -> '.join(f'{e:.2f}%' for e in errs_l2)}; "
            f"{'monotone' if err_decreased else 'NOT monotone'} convergence; "
            f"geometry offsets delta_i={' / '.join(f'{v:+.3f}' for v in di)}, "
            f"delta_o={' / '.join(f'{v:+.3f}' for v in do)} lattice units "
            f"(spread {max(di) - min(di):.3f} / {max(do) - min(do):.3f}); "
            f"torque cross-check (interface-flux MEM vs 4*pi*nu*B): "
            f"{' / '.join(f'{e:+.2f}%' for e in t_errs)}, wall balance "
            f"{' / '.join(f'{e:+.2f}%' for e in balance)}. Nominal-radii "
            f"comparison (disclosure): "
            f"{' -> '.join(f'{e:.2f}%' for e in [c['nominal_max_rel_bin_central_pct'] for c in cases])}"
        ),
    }
    return summary


def cmd_scan(args) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cases = []
    for ri in args.ri:
        p = out_dir / f"case_ri{ri}.json"
        cases.append(
            run_case(
                ri,
                args.eta,
                args.re_i,
                args.nu,
                str(p),
                device,
                args.min_mult,
                args.max_mult,
                args.sample_interval,
                args.avg_steps,
                args.seed,
                compile_mode_from_args(args),
            )
        )
    summary = summarize(cases)
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("convergence", "verdict")}, indent=2))


def cmd_aggregate(args) -> None:
    out_dir = Path(args.out_dir)
    cases = [json.loads((out_dir / f"case_ri{ri}.json").read_text()) for ri in args.ri]
    summary = summarize(cases)
    (out_dir / "result.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description="Annular Taylor-Couette laminar benchmark (D2Q9)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _common(p):
        p.add_argument("--ri", type=int, nargs="+", default=[24, 48, 96])
        p.add_argument("--eta", type=float, default=0.5)
        p.add_argument("--re-i", type=float, default=20.0)
        p.add_argument("--nu", type=float, default=0.05)
        p.add_argument("--min-mult", type=float, default=3.5)
        p.add_argument("--max-mult", type=float, default=10.0)
        p.add_argument("--sample-interval", type=int, default=200)
        p.add_argument("--avg-steps", type=int, default=1000)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", type=str, default="cuda")
        add_compile_mode_arg(p)

    p1 = sub.add_parser("single")
    p1.add_argument("--out", type=str, required=True)
    _common(p1)

    p2 = sub.add_parser("scan")
    p2.add_argument("--out-dir", type=str, required=True)
    _common(p2)

    p3 = sub.add_parser("aggregate")
    p3.add_argument("--out-dir", type=str, required=True)
    _common(p3)

    args = ap.parse_args()
    if args.cmd == "single":
        device = torch.device(args.device)
        run_case(
            args.ri[0],
            args.eta,
            args.re_i,
            args.nu,
            args.out,
            device,
            args.min_mult,
            args.max_mult,
            args.sample_interval,
            args.avg_steps,
            args.seed,
            compile_mode_from_args(args),
        )
    elif args.cmd == "scan":
        cmd_scan(args)
    else:
        cmd_aggregate(args)


if __name__ == "__main__":
    main()
