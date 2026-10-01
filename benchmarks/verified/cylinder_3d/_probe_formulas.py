#!/usr/bin/env python3
"""Extended multi-formula friction/drag probe for 3D extruded cylinder Re=40.

Runs ONE 3D simulation (z-periodic extruded cylinder) identical in physics to
run.py, and evaluates every force estimator on the SAME field every sample:

  pressure : drag_pressure_integration(extrap='none', p0='far_field')
  friction : standard / lagrange / faces / mix50 / bfl_smooth / bfl_lag_exact
  MEM      : momentum_exchange_{standard,galilean,background_subtracted}

Key additions vs run_compare_d20.py:
  * --nz exposes the periodic span (cheap 2D-equivalent reduction),
  * --checkpoints prints windowed means at 25/50/75/100% of the run so a true
    plateau (not step-3000 transient) can be seen,
  * time series of Cd_p / each Cd_f written into the JSON,
  * 'ratio' calibrated weight (SUBOFF recipe) is reported.

Usage:
  python _probe_formulas.py --D 20 --lateral 16 --nz 8 --steps 60000 \
      --device sdaa:0 --out /tmp/cyl3d_probe_D20.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    get_near_wall_2d,
)
from tensorlbm.momentum_exchange import (
    momentum_exchange_background_subtracted,
    momentum_exchange_galilean,
    momentum_exchange_standard,
)
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll

REF_CD = 1.54
REF_CD_DC = 1.522


def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def face_counts(solid):
    fluid = ~solid
    nfx = torch.zeros_like(solid, dtype=torch.float32)
    nfy = torch.zeros_like(solid, dtype=torch.float32)
    nfz = torch.zeros_like(solid, dtype=torch.float32)
    nfx[:, :, 1:-1] += (solid[:, :, 2:] & fluid[:, :, 1:-1]).float()
    nfx[:, :, 1:-1] += (solid[:, :, :-2] & fluid[:, :, 1:-1]).float()
    nfy[:, 1:-1, :] += (solid[:, 2:, :] & fluid[:, 1:-1, :]).float()
    nfy[:, 1:-1, :] += (solid[:, :-2, :] & fluid[:, 1:-1, :]).float()
    nfz[1:-1, :, :] += (solid[2:, :, :] & fluid[1:-1, :, :]).float()
    nfz[1:-1, :, :] += (solid[:-2, :, :] & fluid[1:-1, :, :]).float()
    return nfx, nfy, nfz


def smooth_q(solid, cx, cy, R, near):
    nz, ny, nx = solid.shape
    dev = solid.device
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    r_c = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    q = (r_c - R).clamp(0.05, 1.0)
    return q * near.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--out", default="/tmp/cyl3d_probe.json")
    ap.add_argument("--lateral", type=float, default=16.0)
    ap.add_argument("--D", type=int, default=20)
    ap.add_argument("--nz", type=int, default=0, help="0 -> 2*D")
    ap.add_argument("--u-in", type=float, default=0.08)
    ap.add_argument("--sample", type=int, default=100)
    a = ap.parse_args()

    dev = torch.device(a.device)
    D = a.D
    R = D / 2.0
    nz = a.nz if a.nz > 0 else 2 * D
    nx = ny = int(round(a.lateral * D))
    cx = cy = nx // 2
    Re = 40.0
    u_in = a.u_in
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in**2 * (D * nz)

    tag = f"[cyl3d D={D} {nx}x{ny}x{nz}]"
    print(f"{tag} Re={Re} u_in={u_in} nu={nu:.6f} tau={tau:.6f}", flush=True)
    t0 = time.time()

    solid = cylinder3d_mask(nx, ny, nz, cx, cy, R, dev)
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    n_near = int(near.sum().item())
    nfx, nfy, nfz = face_counts(solid)
    n_faces = int((nfx + nfy + nfz).sum().item())
    ratio = n_faces / max(n_near, 1)
    q = smooth_q(solid, cx, cy, R, near)
    q_vals = q[near]
    print(
        f"{tag} near={n_near} faces={n_faces} ratio={ratio:.4f} "
        f"q_smooth mean={float(q_vals.mean()):.4f} min={float(q_vals.min()):.4f} "
        f"max={float(q_vals.max()):.4f}",
        flush=True,
    )

    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, ux0, uy0, uz0)
    del rho0, ux0, uy0, uz0
    im0 = float(f.sum().item())
    print(f"{tag} init done ({time.time() - t0:.0f}s)", flush=True)

    bc_config = {"far_field_faces": ["y-", "y+"], "periodic_faces": ["z-", "z+"]}

    FORMULAS = ["standard", "lagrange", "faces", "mix50", "bfl_smooth", "bfl_lag_exact"]
    hist = {k: [] for k in FORMULAS}
    cd_p_hist, mass_hist = [], []
    step = 0
    for step in range(1, a.steps + 1):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d_roll(f)
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % a.sample == 0:
            fx_p, _, _ = drag_pressure_integration(
                f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid
            )
            cd_p_hist.append(fx_p)
            hist["standard"].append(
                drag_friction_integration(f, mesh, dpS, nu, formula="standard")[0]
            )
            hist["lagrange"].append(
                drag_friction_integration(f, mesh, dpS, nu, formula="lagrange")[0]
            )
            hist["faces"].append(
                drag_friction_integration(f, mesh, dpS, nu, formula="faces", solid=solid)[0]
            )
            hist["mix50"].append(
                drag_friction_integration(f, mesh, dpS, nu, formula="mix50", solid=solid)[0]
            )
            hist["bfl_smooth"].append(
                drag_friction_integration(f, mesh, dpS, nu, q_wall=q, formula="bfl")[0]
            )
            hist["bfl_lag_exact"].append(
                drag_friction_integration(f, mesh, dpS, nu, q_wall=q, formula="bfl_lagrange")[0]
            )
            mass_hist.append(float(f.sum().item()))

        if step % 2500 == 0:
            n_avg = min(200, len(cd_p_hist))
            if n_avg:
                cdp = sum(cd_p_hist[-n_avg:]) / n_avg
                parts = " ".join(f"{k}={sum(hist[k][-n_avg:]) / n_avg:.4f}" for k in FORMULAS)
                print(
                    f"{tag} step={step} Cd_p={cdp:.4f} {parts} ({time.time() - t0:.0f}s)",
                    flush=True,
                )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(cd_p_hist)
    win = min(n_tot, 200)

    cdp = sum(cd_p_hist[-win:]) / win
    cdf = {k: sum(hist[k][-win:]) / win for k in FORMULAS}
    cdt = {k: cdp + cdf[k] for k in FORMULAS}

    # calibrated ratio weight (SUBOFF recipe)
    gain = cdf["faces"] / cdf["standard"] - 1.0
    w_ratio = 1.0 - gain / (ratio - 1.0) if abs(ratio - 1.0) > 1e-9 else float("nan")
    cd_f_wr = (
        w_ratio * cdf["standard"] + (1 - w_ratio) * cdf["faces"]
        if w_ratio == w_ratio
        else float("nan")
    )
    cd_t_wr = cdp + cd_f_wr if cd_f_wr == cd_f_wr else float("nan")

    # MEM diagnostics on final field
    mem_cd = {}
    try:
        ms = momentum_exchange_standard(f, solid, near)[0] / dpS
        mg = momentum_exchange_galilean(f, solid, near, tau)[0] / dpS
        mb = (
            momentum_exchange_background_subtracted(f, solid, near, rho0=1.0, u0=(u_in, 0.0, 0.0))[
                0
            ]
            / dpS
        )
        mem_cd = {"standard": ms, "galilean": mg, "bg_sub": mb}
    except Exception as exc:  # pragma: no cover
        mem_cd = {"error": str(exc)}

    # drift windows
    def wmean(k, frac):
        kk = int(n_tot * frac)
        seg = hist[k][max(0, kk - win) : kk]
        if not seg:
            return float("nan")
        pseg = cd_p_hist[max(0, kk - win) : kk]
        return sum(seg) / len(seg) + sum(pseg) / len(pseg)

    plateau = {
        f"{k}@{int(f * 100)}%": wmean(k, f) for f in (0.25, 0.5, 0.75, 1.0) for k in FORMULAS
    }

    print(f"{tag} === FINAL win={win} ===", flush=True)
    for k in FORMULAS:
        print(
            f"  {k:14s} Cd_f={cdf[k]:.4f} Cd={cdt[k]:.4f} "
            f"err_vs_DC={(cdt[k] - REF_CD_DC) / REF_CD_DC * 100:+.2f}% "
            f"err_vs_Tr={(cdt[k] - REF_CD) / REF_CD * 100:+.2f}% "
            f"(f/std {100 * (cdf[k] / cdf['standard'] - 1):+.1f}%)",
            flush=True,
        )
    print(f"  ratio_w={w_ratio:.4f} Cd_f_weighted={cd_f_wr:.4f} Cd={cd_t_wr:.4f}", flush=True)
    print(f"  MEM {mem_cd}", flush=True)

    res = {
        "case": "cylinder_3d_re40_probe_formulas",
        "D_cells": D,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "lateral_D": a.lateral,
        "blockage_pct": 100.0 * D / nx,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "n_near_cells": n_near,
        "n_wall_faces": n_faces,
        "face_cell_ratio": ratio,
        "q_smooth_mean": float(q_vals.mean()),
        "q_smooth_min": float(q_vals.min()),
        "q_smooth_max": float(q_vals.max()),
        "n_steps": a.steps,
        "n_finished": step,
        "avg_window_samples": win,
        "cd_pressure": cdp,
        "cd_friction": cdf,
        "cd_total": cdt,
        "err_vs_DC_pct": {k: (v - REF_CD_DC) / REF_CD_DC * 100 for k, v in cdt.items()},
        "err_vs_Tritton_pct": {k: (v - REF_CD) / REF_CD * 100 for k, v in cdt.items()},
        "ratio_weight": w_ratio,
        "cd_f_weighted_ratio": cd_f_wr,
        "cd_tot_weighted_ratio": cd_t_wr,
        "mem_cd": mem_cd,
        "plateau_windows": plateau,
        "cd_p_series": cd_p_hist,
        "cd_f_series": hist,
        "mass_drift_pct": (mass_hist[-1] - im0) / im0 * 100.0 if mass_hist else float("nan"),
        "finite": bool(torch.isfinite(f).all().item()),
        "wall_s": elapsed,
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(f"{tag} saved {a.out} ({elapsed:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
