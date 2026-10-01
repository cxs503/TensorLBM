#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""D20 7-formula friction comparison for 3D cylinder Re=40 (infinite span).

Runs ONE D20 simulation (320x320x40, 50000 steps, identical physics to
run.py) and evaluates all 7 friction formulas on the SAME flow history
every sample step.  Drag integration is purely diagnostic (no feedback),
so all formulas share identical fields.

Formulas (drag_friction_integration):
  standard     formula='standard'                     tau = 2*nu*u_t
  lagrange     formula='lagrange'                     tau = nu*(3u1 - u2/3)
  bfl_smooth   formula='bfl',        q=q_smooth       tau = nu*u1/q
  bfl_lag_exact formula='bfl_lagrange', q=q_smooth    tau = nu*(u1(q+1)/q - u2*q/(q+1))
  faces        formula='faces' (per wall face, dA=1)  tau = 2*nu*u_t(face)
  dA_scale     standard x (wall_faces / near_cells)   geometric area fix
  u05          formula='bfl', q=0.5 constant          (=standard; q-sensitivity baseline)

q_smooth = r_c - R at each near-wall cell (distance to the smooth cylinder
surface along the radial), clamped to [0.05, 1.0].

Output: JSON with time-averaged Cd_p / Cd_f / Cd for all 7 formulas,
face/cell counts, q_smooth stats, and a per-cell decomposition of the
faces-vs-standard difference on the final field (u_t check).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/wxsc/cxs/TensorLBM/src")

import torch

from tensorlbm.boundaries3d import bounce_back_cells_3d, far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.drag_pressure import (
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    get_near_wall_2d,
)
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll

REF_CD = 1.54


def cylinder3d_mask(nx, ny, nz, cx, cy, radius, device):
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def face_counts(solid):
    """Per-cell wall-face counts (nfx, nfy, nfz) with verified slicing."""
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
    """q_smooth = distance from near-wall cell center to the smooth
    cylinder surface along the radial, clamped to [0.05, 1.0]."""
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
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--steps", type=int, default=50000)
    ap.add_argument("--out", default="/tmp/cyl3d_d20_compare.json")
    ap.add_argument("--lateral", type=float, default=None)
    ap.add_argument("--D", type=int, default=20)
    a = ap.parse_args()

    dev = torch.device(a.device)
    D = a.D
    R = D / 2.0
    nz = 2 * D
    if a.lateral is None:
        a.lateral = 16.0 if D <= 40 else 12.0
    nx = ny = int(round(a.lateral * D))
    cx = cy = nx // 2
    Re = 40.0
    u_in = 0.08
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
    ratio = n_faces / n_near
    q = smooth_q(solid, cx, cy, R, near)
    q_half = torch.full_like(solid, 0.5, dtype=torch.float32) * near.float()
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

    bc_config = {
        "far_field_faces": ["y-", "y+"],
        "periodic_faces": ["z-", "z+"],
    }

    FORMULAS = ["standard", "lagrange", "bfl_smooth", "bfl_lag_exact", "faces", "dA_scale", "u05"]
    hist = {k: [] for k in FORMULAS}
    cd_p_hist, cd_tot_hist, mass_hist = [], [], []
    sample_interval = 100
    step = 0
    for step in range(1, a.steps + 1):
        f_pre_solid = f[:, solid].clone()
        f = collide_bgk3d(f, tau)
        f[:, solid] = f_pre_solid
        f = bounce_back_cells_3d(f, solid)
        f = stream3d_roll(f)
        f = far_field_bc_3d(f, u_in, bc_config=bc_config)

        if step % sample_interval == 0:
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
            hist["bfl_smooth"].append(
                drag_friction_integration(f, mesh, dpS, nu, q_wall=q, formula="bfl")[0]
            )
            hist["bfl_lag_exact"].append(
                drag_friction_integration(f, mesh, dpS, nu, q_wall=q, formula="bfl_lagrange")[0]
            )
            hist["faces"].append(
                drag_friction_integration(f, mesh, dpS, nu, formula="faces", solid=solid)[0]
            )
            hist["dA_scale"].append(
                drag_friction_integration(f, mesh, dpS, nu, formula="standard")[0] * ratio
            )
            hist["u05"].append(
                drag_friction_integration(f, mesh, dpS, nu, q_wall=q_half, formula="bfl")[0]
            )
            cd_tot_hist.append(fx_p + hist["standard"][-1])
            mass_hist.append(float(f.sum().item()))

        if step % 5000 == 0:
            n_avg = min(200, len(cd_tot_hist))
            cd = sum(cd_tot_hist[-n_avg:]) / n_avg
            print(
                f"{tag} step={step} Cd={cd:.4f} "
                f"(Cd_p={sum(cd_p_hist[-n_avg:]) / n_avg:.4f} "
                f"Cd_f_std={sum(hist['standard'][-n_avg:]) / n_avg:.4f} "
                f"Cd_f_faces={sum(hist['faces'][-n_avg:]) / n_avg:.4f}) "
                f"({time.time() - t0:.0f}s)",
                flush=True,
            )

        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

    elapsed = time.time() - t0
    n_tot = len(cd_tot_hist)
    win = min(n_tot, 200)
    cd_p = sum(cd_p_hist[-win:]) / win
    cd_f = {}
    for k in FORMULAS:
        cd_f[k] = sum(hist[k][-win:]) / win
    cd_tot = {k: cd_p + v for k, v in cd_f.items()}
    half = win // 2
    drift = (
        (sum(cd_tot_hist[-half:]) / half - sum(cd_tot_hist[-2 * half : -half]) / half)
        / REF_CD
        * 100.0
    )

    print(
        f"{tag} === FINAL (win={win} samples) === Cd_p={cd_p:.4f} drift={drift:+.3f}% "
        f"({elapsed:.0f}s)",
        flush=True,
    )
    for k in FORMULAS:
        err = (cd_tot[k] - REF_CD) / REF_CD * 100.0
        print(
            f"  {k:14s} Cd_f={cd_f[k]:.4f}  Cd={cd_tot[k]:.4f}  err={err:+.2f}% "
            f"(vs std {100.0 * (cd_f[k] / cd_f['standard'] - 1.0):+.1f}%)",
            flush=True,
        )

    # --- per-cell u_t diagnostic on the final field -------------------
    diag = {}
    rho, ux, uy, uz = macroscopic3d(f)
    u_dot_n = ux * mesh.nx_n + uy * mesh.ny_n + uz * mesh.nz_n
    ut_x = ux - u_dot_n * mesh.nx_n
    ut_y = uy - u_dot_n * mesh.ny_n
    ut_z = uz - u_dot_n * mesh.nz_n
    near_f = near.float()
    # faces contribution per cell (x-force): 2*nu*ux*(nfy+nfz)
    face_x = 2.0 * nu * ux * (nfy + nfz) * near_f
    # standard contribution per cell: 2*nu*ut_x
    std_x = 2.0 * nu * ut_x * near_f
    # decomposition by face type
    yf = (nfy > 0.5) & near
    xf_only = (nfy <= 0.5) & (nfx > 0.5) & near
    two_face = (nfx > 0.5) & (nfy > 0.5) & near
    diag = {
        "n_y_face_cells": int(yf.sum()),
        "n_x_only_cells": int(xf_only.sum()),
        "n_two_face_cells": int(two_face.sum()),
        "sum_std_x": float(std_x.sum().item()),
        "sum_faces_x": float(face_x.sum().item()),
        "sum_std_at_yface_cells": float((std_x * yf.float()).sum().item()),
        "sum_std_at_x_only_cells": float((std_x * xf_only.float()).sum().item()),
        "sum_std_at_two_face_cells": float((std_x * two_face.float()).sum().item()),
        "sum_facex_at_yface_cells": float((face_x * yf.float()).sum().item()),
        "sum_facex_at_two_face_cells": float((face_x * two_face.float()).sum().item()),
        "sum_utx_at_yface": float((ut_x * yf.float()).sum().item()),
        "sum_ux_at_yface": float((ux * yf.float()).sum().item()),
        "mean_ux_yface": float((ux * yf.float()).sum().item() / max(int(yf.sum()), 1)),
        "mean_ux_twface": float((ux * two_face.float()).sum().item() / max(int(two_face.sum()), 1)),
        "mean_ux_xonly": float((ux * xf_only.float()).sum().item() / max(int(xf_only.sum()), 1)),
        "mean_utx_xonly": float((ut_x * xf_only.float()).sum().item() / max(int(xf_only.sum()), 1)),
    }
    print(f"{tag} per-cell diag: {json.dumps(diag, indent=1)}", flush=True)

    res = {
        "case": "cylinder_3d_re40_d20_formula_compare",
        "lattice": "D3Q19",
        "collision": "bgk",
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "D_cells": D,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "n_near_cells": n_near,
        "n_wall_faces": n_faces,
        "face_cell_ratio": ratio,
        "q_smooth_mean": float(q_vals.mean()),
        "q_smooth_min": float(q_vals.min()),
        "q_smooth_max": float(q_vals.max()),
        "n_steps": a.steps,
        "n_finished": step,
        "sample_interval": sample_interval,
        "avg_window_samples": win,
        "cd_pressure": cd_p,
        "cd_friction": cd_f,
        "cd_total": cd_tot,
        "err_cd_pct": {k: (v - REF_CD) / REF_CD * 100.0 for k, v in cd_tot.items()},
        "ref_cd": REF_CD,
        "drift_cd_pct_std": drift,
        "mass_drift_pct": (mass_hist[-1] - im0) / im0 * 100.0,
        "finite": bool(torch.isfinite(f).all().item()),
        "per_cell_diag": diag,
        "formula_notes": {
            "standard": "2*nu*u_t (cell-based, dA=1)",
            "lagrange": "nu*(3u1-u2/3)",
            "bfl_smooth": "nu*u1/q_smooth, q_smooth=r_c-R clamped [0.05,1]",
            "bfl_lag_exact": "nu*(u1(q+1)/q - u2*q/(q+1))",
            "faces": "per-wall-face 2*nu*u_t(face), dA=1/face",
            "dA_scale": "standard * (wall_faces/near_cells) geometric area fix",
            "u05": "nu*u1/0.5 (bfl with q=0.5, =standard identity)",
        },
        "wall_s": elapsed,
    }
    Path(a.out).write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
