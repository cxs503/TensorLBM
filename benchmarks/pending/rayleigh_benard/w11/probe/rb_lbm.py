#!/usr/bin/env python
"""Wave-11 Track C: Rayleigh-Benard probe runner (Phase 0, CPU).

Library-faithful double-distribution thermal LBM on the Wave-11 worktree
(read-only import). Grid (Q, ny, nx); x periodic, wall rows y=0 / y=ny-1.

Machines reused verbatim from src/tensorlbm (verified/thermal_cavity lineage):
  - D2Q9 equilibrium/macroscopic/stream (periodic gather)
  - collide_bgk_force (Guo 2002)
  - pre_streaming_bounce_back (V3 no-slip recipe, wall rows)
  - temperature_collision / temperature_stream (D2Q5, W5 fp32-normalized)
Thermal wall BC = rotated (rows) mirror of thermal.apply_temperature_boundaries
isothermal-column construction: freeze w_i*T_wall + non-equilibrium correction
that pins the effective isotherm at the wall node.

Modes:
  conduction  - no flow, no buoyancy: invariance of the linear profile (d_T test)
  decay       - no buoyancy, seeded cos(kx)*sin(pi z): diffusive decay rate
                (thermal operator + d_T test, no momentum coupling)
  poiseuille  - body-forced channel: effective no-slip wall planes (d_v test)
  growth      - full RBC linear growth: seeded eigenmode amplitude series
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, "/nfs/wangxi/worktrees/bm_w11/src")

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.thermal import (  # noqa: E402
    W5,
    collide_bgk_force,
    pre_streaming_bounce_back,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)

TAU = 0.855  # nu = 0.118333..., lattice
TAU_T = 1.0  # alpha = 1/6 exactly
NU = (TAU - 0.5) / 3.0
ALPHA = (TAU_T - 0.5) / 3.0
PR = NU / ALPHA
T_HOT = 1.0
T_COLD = 0.0


def apply_rb_temp_bc(g: torch.Tensor, construction: str = "node") -> torch.Tensor:
    """Isothermal bottom (y=0, T_hot) / top (y=ny-1, T_cold) rows.

    Rotated mirror of thermal.apply_temperature_boundaries isothermal-column
    construction: freeze equilibrium w_i*T_wall over the wall row, then add to
    the into-fluid population (dir 3 at bottom / dir 4 at top) the
    non-equilibrium carried by the first fluid row (from the pre-BC field),
    zero-summed against the rest population (wall-row Sum(T) stays T_wall).

    construction="node": T_wall pins the wall NODE row (isotherm planes at
    y=0 and y=ny-1, d_T=ny-1).  Base profile = linspace(T_hot,T_cold,ny).
    construction="half": anti-symmetric mirror about the half-planes
    y=0.5 / y=ny-1.5 (T_wall_node = 2*T_plane - T_first_fluid_row), placing
    the isotherm planes exactly at the no-slip planes measured by P1
    (d_v=ny-2) -> d_T=d_v=ny-2 coincident layers.
    """
    w = W5.to(g.device)
    w3, w4 = float(w[3]), float(w[4])
    t1 = g[:, 1, :].sum(dim=0)  # first fluid row (pre-BC input field)
    tn2 = g[:, -2, :].sum(dim=0)
    if construction == "node":
        t_bot, t_top = T_HOT, T_COLD
    elif construction == "half":
        t_bot = 2.0 * T_HOT - t1
        t_top = 2.0 * T_COLD - tn2
    else:
        raise ValueError(construction)
    neq_bot = g[3, 1, :] - w3 * t1
    neq_top = g[4, -2, :] - w4 * tn2
    g = g.clone()
    g[:, 0, :] = w.view(5, 1) * t_bot
    g[:, -1, :] = w.view(5, 1) * t_top
    g[3, 0, :] = g[3, 0, :] + neq_bot
    g[0, 0, :] = g[0, 0, :] - neq_bot
    g[4, -1, :] = g[4, -1, :] + neq_top
    g[0, -1, :] = g[0, -1, :] - neq_top
    return g


def d_thermal(ny, construction):
    return (ny - 1) if construction == "node" else (ny - 2)


def buoyancy_theta(T, g_beta, Tbar):
    """F = (0, rho*g_beta*theta), theta = T - Tbar(y): zero on the exact
    conductive base state (no hydrostatic term, pure linear coupling)."""
    fy = g_beta * (T - Tbar)
    return torch.stack([torch.zeros_like(fy), fy], dim=0)


def base_state(ny, nx, device, construction="node"):
    y = torch.arange(ny, device=device, dtype=torch.float64)
    if construction == "node":
        Tbar = T_HOT + (T_COLD - T_HOT) * y / (ny - 1)
    else:
        d = ny - 2
        Tbar = T_HOT - (y - 0.5) / d
    return Tbar.view(ny, 1).expand(ny, nx).contiguous().float()


def seed_fields(ny, nx, device, amp, kind, eigen=None, construction="node"):
    """Return (T_seed_add, ux, uy) perturbation fields (fp32).

    kind=analytic: theta = amp*S(j)*cos(2*pi*i/nx), u=0, with S the
                     slowest vertical eigenmode consistent with the thermal
                     construction: node -> sin(pi*j/(ny-1)) (zero at wall
                     nodes); half -> sin(pi*(j-0.5)/(ny-2)) (zero at the
                     half-planes, anti-symmetric wall-node values).
    kind=eigen:    eigen dict with z-normalized eigenfunctions sampled on
                     node rows: th[j,i] = amp*Th[j]*cos(k*x_i),
                     ux = amp*Phder[j]*sin(k*x_i), uy = -amp*k*Ph[j]*cos(k*x_i).
    """
    y = torch.arange(ny, device=device, dtype=torch.float64)
    x = torch.arange(nx, device=device, dtype=torch.float64)
    if kind == "analytic":
        if construction == "node":
            shape_y = torch.sin(math.pi * y / (ny - 1))
        else:
            shape_y = torch.sin(math.pi * (y - 0.5) / (ny - 2))
        th = amp * shape_y.view(ny, 1) * torch.cos(2.0 * math.pi * x / nx).view(1, nx)
        ux = torch.zeros((ny, nx), dtype=torch.float64, device=device)
        uy = torch.zeros((ny, nx), dtype=torch.float64, device=device)
    elif kind == "eigen":
        Th = torch.tensor(eigen["Theta_node"], dtype=torch.float64, device=device)
        Ph = torch.tensor(eigen["Phi_node"], dtype=torch.float64, device=device)
        Phd = torch.tensor(eigen["Phider_node"], dtype=torch.float64, device=device)
        kx = 2.0 * math.pi * x / nx
        k = 2.0 * math.pi / nx
        # lattice u = u_nondim * (alpha/d): the eigenfunctions (Phi, Phi',
        # Theta) are one globally-scaled eigenvector in nondim units; only the
        # velocity fields carry the alpha/d factor (theta_lat = theta_nondim
        # for DeltaT = 1 across the layer).
        d = ny - 2
        vs = ALPHA / d
        th = amp * Th.view(ny, 1) * torch.cos(kx).view(1, nx)
        ux = amp * vs * Phd.view(ny, 1) * torch.sin(kx).view(1, nx)
        uy = -amp * vs * k * Ph.view(ny, 1) * torch.cos(kx).view(1, nx)
    else:
        raise ValueError(kind)
    return th.float(), ux.float(), uy.float()


def project_amplitudes(T, ux, uy, Tbar, shape, k, device):
    """Signed projections (fp64 accumulation): A_T = <theta, Th cos>,
    A_ux = <ux, Phider sin>, A_uy = <-uy, k Ph cos> (each normalized so that
    a pure seeded mode of amplitude A gives exactly A)."""
    Th = torch.tensor(shape["Theta_node"], dtype=torch.float64, device=device).view(-1, 1)
    Ph = torch.tensor(shape["Phi_node"], dtype=torch.float64, device=device).view(-1, 1)
    Phd = torch.tensor(shape["Phider_node"], dtype=torch.float64, device=device).view(-1, 1)
    nx = T.shape[1]
    x = torch.arange(nx, device=device, dtype=torch.float64)
    c = torch.cos(k * x).view(1, -1)
    s = torch.sin(k * x).view(1, -1)
    th = T.double() - Tbar.double()
    nT = float((Th**2 * c**2).sum())
    nU = float((Phd**2 * s**2).sum())
    nV = float(((k * Ph) ** 2 * c**2).sum())
    aT = float((th * Th * c).sum()) / max(nT, 1e-300)
    aU = float((ux.double() * Phd * s).sum()) / max(nU, 1e-300)
    aV = float((-uy.double() * (k * Ph) * c).sum()) / max(nV, 1e-300)
    return aT, aU, aV


def run_conduction(ny, nx, steps, device, construction="node"):
    """P0.5: base conductive state must be an exact invariant; quantifies the
    discrete steady profile (endpoint pinning, interior linearity)."""
    Tbar = base_state(ny, nx, device, construction)
    f = equilibrium(
        torch.ones((ny, nx), device=device),
        torch.zeros((ny, nx), device=device),
        torch.zeros((ny, nx), device=device),
    )
    g = temperature_equilibrium(Tbar, torch.zeros_like(Tbar), torch.zeros_like(Tbar))
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    max_drift = 0.0
    t_hist = []
    for step in range(1, steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_theta(T, 0.0, Tbar)
        g = temperature_collision(g, TAU_T, ux, uy)
        g = temperature_stream(g)
        g = apply_rb_temp_bc(g, construction)
        f_pre = f
        f = collide_bgk_force(f, TAU, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
        if step % max(1, steps // 20) == 0 or step == steps:
            T = g.sum(dim=0)
            drift = float((T - Tbar).abs().max())
            umax = float((ux**2 + uy**2).sqrt().max())
            t_hist.append({"step": step, "max_drift_T": drift, "u_max": umax})
            max_drift = max(max_drift, drift)
    T = g.sum(dim=0).cpu()
    yref = base_state(ny, 1, "cpu", construction)[:, 0].double()
    dev_lin = float((T.double() - yref.view(ny, 1)).abs().max())
    return {
        "mode": "conduction",
        "ny": ny,
        "nx": nx,
        "steps": steps,
        "construction": construction,
        "d_thermal": d_thermal(ny, construction),
        "max_drift_T": max_drift,
        "final_dev_from_base": dev_lin,
        "T0_exact": float(T[0, :].mean()),
        "Ttop_exact": float(T[-1, :].mean()),
        "history": t_hist,
    }


def run_decay(ny, nx, steps, amp, device, sample_every, construction="node"):
    """P0.7: pure diffusive decay of seeded cos(kx)*S(j) mode (no buoyancy).
    Lattice prediction (tau_T=1 collapses collision to full relaxation):
    sigma = ln[(1+cos k+cos q)/3] with q = pi/d_T per construction."""
    Tbar = base_state(ny, nx, device, construction)
    th, _, _ = seed_fields(ny, nx, device, amp, "analytic", construction=construction)
    f = equilibrium(
        torch.ones((ny, nx), device=device),
        torch.zeros((ny, nx), device=device),
        torch.zeros((ny, nx), device=device),
    )
    g = temperature_equilibrium(Tbar + th, torch.zeros_like(Tbar), torch.zeros_like(Tbar))
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    k = 2.0 * math.pi / nx
    y = torch.arange(ny, device=device, dtype=torch.float64)
    sy = (
        (
            torch.sin(math.pi * y / (ny - 1))
            if construction == "node"
            else torch.sin(math.pi * (y - 0.5) / (ny - 2))
        )
        .cpu()
        .numpy()
        .tolist()
    )
    shape = {"Theta_node": sy, "Phi_node": [0.0] * ny, "Phider_node": [0.0] * ny}
    samples = []
    for step in range(0, steps + 1):
        if step % sample_every == 0:
            T = g.sum(dim=0)
            aT, aU, aV = project_amplitudes(
                T, torch.zeros_like(T), torch.zeros_like(T), Tbar, shape, k, device
            )
            samples.append({"step": step, "A_T": aT, "A_ux": aU, "A_uy": aV})
        if step == steps:
            break
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_theta(T, 0.0, Tbar)
        g = temperature_collision(g, TAU_T, ux, uy)
        g = temperature_stream(g)
        g = apply_rb_temp_bc(g, construction)
        f_pre = f
        f = collide_bgk_force(f, TAU, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
    q = math.pi / d_thermal(ny, construction)
    pred = {
        "sigma_lattice_ideal": math.log((1.0 + math.cos(k) + math.cos(q)) / 3.0),
        "sigma_continuum": -ALPHA * (k * k + q * q),
    }
    return {
        "mode": "decay",
        "ny": ny,
        "nx": nx,
        "steps": steps,
        "amp": amp,
        "construction": construction,
        "d_thermal": d_thermal(ny, construction),
        "k_lat": k,
        "alpha": ALPHA,
        "prediction": pred,
        "samples": samples,
    }


def run_poiseuille(ny, nx, steps, f_body, device, sample_every):
    """P1: body-forced channel, wall rows 0/ny-1, tau=TAU (same as RBC).
    Steady u_x(y) parabola fit -> effective no-slip wall planes."""
    f = equilibrium(
        torch.ones((ny, nx), device=device),
        torch.zeros((ny, nx), device=device),
        torch.zeros((ny, nx), device=device),
    )
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    F = torch.zeros((2, ny, nx), device=device)
    F[0] = f_body
    hist = []
    prof = None
    for step in range(1, steps + 1):
        f_pre = f
        f = collide_bgk_force(f, TAU, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
        if step % sample_every == 0 or step == steps:
            rho, ux, uy = macroscopic(f)
            prof = ux.mean(dim=1).detach().cpu().numpy()
            hist.append({"step": step, "u_max": float(ux.max()), "u_mean": float(ux.mean())})
    import numpy as np

    def parab(yv, uv):
        A = np.vstack([yv**2, yv, np.ones_like(yv)]).T
        coef, *_ = np.linalg.lstsq(A, uv, rcond=None)
        a2, b1, c0 = coef
        disc = b1**2 - 4 * a2 * c0
        y0 = (-b1 - math.sqrt(disc)) / (2 * a2)
        y1 = (-b1 + math.sqrt(disc)) / (2 * a2)
        resid = float(np.abs(uv - (a2 * yv**2 + b1 * yv + c0)).max())
        return {"a2": a2, "b1": b1, "c0": c0, "y0": y0, "y1": y1, "d": y1 - y0, "resid": resid}

    y = np.arange(ny, dtype=np.float64)
    u = np.asarray(prof, dtype=np.float64)
    fit_all = parab(y, u)
    fit_fluid = parab(y[1:-1], u[1:-1])  # fluid rows only: the robust fit
    # analytic centerline check with each candidate depth
    umax = float(u[1:-1].max())
    theory = {f"d={dy}": f_body * dy**2 / (8.0 * NU) for dy in (ny - 2, ny - 1, ny)}
    return {
        "mode": "poiseuille",
        "ny": ny,
        "nx": nx,
        "steps": steps,
        "f_body": f_body,
        "tau": TAU,
        "nu": NU,
        "u_max_fluid": umax,
        "u_max_theory": theory,
        "u_max_rel_err": {kk: abs(umax - vv) / vv for kk, vv in theory.items()},
        "shape_ratio_u1_over_uc": float(u[1] / umax),
        "shape_ratio_pred": {
            "d=ny-2_halfway": 0.5 * (ny - 3) / ((ny - 2) ** 2 / 4.0),
            "d=ny-1_nodeplane": 1.0 * (ny - 2) / ((ny - 1) ** 2 / 4.0),
        },
        "fit_all_rows": fit_all,
        "fit_fluid_rows": fit_fluid,
        "profile": u.tolist(),
        "history": hist,
    }


def run_growth(
    ny,
    nx,
    steps,
    g_beta,
    amp,
    device,
    sample_every,
    eigen_npz,
    eigen_tag,
    construction="node",
    ra=None,
):
    """P2/P3: full RBC seeded-eigenmode growth; amplitude time series.
    g_beta = Ra*nu*alpha/d^2 for the construction's layer depth d (G = 1/d)."""
    import numpy as np

    eig = np.load(eigen_npz)
    tag = eigen_tag
    eigen = {
        "Theta_node": eig[f"{tag}/Theta_node"].tolist(),
        "Phi_node": eig[f"{tag}/Phi_node"].tolist(),
        "Phider_node": eig[f"{tag}/Phider_node"].tolist(),
    }
    Tbar = base_state(ny, nx, device, construction)
    th, ux0, uy0 = seed_fields(ny, nx, device, amp, "eigen", eigen)
    f = equilibrium(torch.ones((ny, nx), device=device), ux0, uy0)
    g = temperature_equilibrium(Tbar + th, ux0, uy0)
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True
    k = 2.0 * math.pi / nx
    samples = []
    t0 = time.time()
    for step in range(0, steps + 1):
        if step % sample_every == 0 or step == steps:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            aT, aU, aV = project_amplitudes(T, ux, uy, Tbar, eigen, k, device)
            umax = float((ux**2 + uy**2).sqrt()[1:-1, :].max())
            samples.append(
                {
                    "step": step,
                    "A_T": aT,
                    "A_ux": aU,
                    "A_uy": aV,
                    "u_max": umax,
                    "rho_mean": float(rho.mean()),
                    "heat_ledger": float((T.double() - Tbar.double()).sum()),
                }
            )
        if step == steps:
            break
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_theta(T, g_beta, Tbar)
        g = temperature_collision(g, TAU_T, ux, uy)
        g = temperature_stream(g)
        g = apply_rb_temp_bc(g, construction)
        f_pre = f
        f = collide_bgk_force(f, TAU, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
    return {
        "mode": "growth",
        "ny": ny,
        "nx": nx,
        "steps": steps,
        "g_beta": g_beta,
        "ra": ra,
        "amp": amp,
        "k_lat": k,
        "eigen_tag": eigen_tag,
        "construction": construction,
        "d_thermal": d_thermal(ny, construction),
        "nu": NU,
        "alpha": ALPHA,
        "pr": PR,
        "tau": TAU,
        "tau_T": TAU_T,
        "elapsed_s": round(time.time() - t0, 1),
        "samples": samples,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["conduction", "decay", "poiseuille", "growth"])
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--nx", type=int, default=0, help="0 -> 2*ny")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--amp", type=float, default=1e-4)
    ap.add_argument("--g-beta", type=float, default=0.0)
    ap.add_argument("--f-body", type=float, default=1e-6)
    ap.add_argument("--sample-every", type=int, default=50)
    ap.add_argument("--eigen-npz", default="")
    ap.add_argument("--eigen-tag", default="")
    ap.add_argument("--construction", choices=["node", "half"], default="node")
    ap.add_argument(
        "--ra",
        type=float,
        default=0.0,
        help="if >0 and mode=growth: g_beta = Ra*nu*alpha/d^3 "
        "(Ra = g_beta*DeltaT*d^3/(nu alpha), DeltaT=1)",
    )
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    nx = args.nx if args.nx > 0 else 2 * args.ny
    if args.mode == "conduction":
        res = run_conduction(args.ny, nx, args.steps, device, args.construction)
    elif args.mode == "decay":
        res = run_decay(
            args.ny, nx, args.steps, args.amp, device, args.sample_every, args.construction
        )
    elif args.mode == "poiseuille":
        res = run_poiseuille(
            args.ny, nx, args.steps, args.f_body, device, max(1000, args.steps // 10)
        )
    else:
        g_beta = args.g_beta
        if args.ra > 0:
            d = d_thermal(args.ny, args.construction)
            g_beta = args.ra * NU * ALPHA / (d**3)
        res = run_growth(
            args.ny,
            nx,
            args.steps,
            g_beta,
            args.amp,
            device,
            args.sample_every,
            args.eigen_npz,
            args.eigen_tag,
            args.construction,
            args.ra,
        )
    Path(args.out).write_text(json.dumps(res, indent=1))
    slim = {k: v for k, v in res.items() if k not in ("samples", "history", "profile")}
    print(json.dumps(slim))
    print("DONE")


if __name__ == "__main__":
    main()
