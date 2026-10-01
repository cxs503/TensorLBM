#!/usr/bin/env python3
"""Independent 2D free-stream cylinder Re=40 cross-check (library D2Q9 + MEM).

Purpose: settle whether the 3D-extruded case's Cd=1.42 deficit is a
*force-integration* artefact or a *flow-resolution* artefact.  This script is a
completely different discretisation (D2Q9 + Ladd momentum-exchange force) on a
much larger domain (40D x 40D, blockage 2.5%) and much finer grid, run to
steady state.  If it lands on the literature cluster ~1.5 while the 3D
pressure+friction split sits at 1.42, the deficit is in the friction integral.

Also reports a 2D pressure/friction split identical in convention to
tensorlbm.drag_pressure (dA=1 per near cell, p0=far-field, extrap='none') so the
3D and 2D numbers are directly comparable.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tensorlbm.boundaries import compute_obstacle_forces, cylinder_mask, far_field_bc_2d
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import collide_bgk, stream


def near_mask_2d(solid):
    fluid = ~solid
    n = torch.zeros_like(solid)
    n[:, 1:-1] |= (solid[:, 2:] | solid[:, :-2]) & fluid[:, 1:-1]
    n[1:-1, :] |= (solid[2:, :] | solid[:-2, :]) & fluid[1:-1, :]
    return n


def face_counts_2d(solid):
    fluid = ~solid
    nfx = torch.zeros_like(solid, dtype=torch.float32)
    nfy = torch.zeros_like(solid, dtype=torch.float32)
    nfx[:, 1:-1] += (solid[:, 2:] & fluid[:, 1:-1]).float()
    nfx[:, 1:-1] += (solid[:, :-2] & fluid[:, 1:-1]).float()
    nfy[1:-1, :] += (solid[2:, :] & fluid[1:-1, :]).float()
    nfy[1:-1, :] += (solid[:-2, :] & fluid[1:-1, :]).float()
    return nfx, nfy


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="sdaa:0")
    ap.add_argument("--D", type=int, default=40)
    ap.add_argument("--domain-D", type=float, default=40.0)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--u-in", type=float, default=0.05)
    ap.add_argument("--out", default="/tmp/cyl2d_re40.json")
    ap.add_argument("--sample", type=int, default=200)
    a = ap.parse_args()

    dev = torch.device(a.device)
    D = a.D
    R = D / 2.0
    nx = ny = int(a.domain_D * D)
    cx = nx / 2.0
    cy = ny / 2.0
    Re = 40.0
    u_in = a.u_in
    nu = u_in * D / Re
    tau = 3.0 * nu + 0.5
    dpS = 0.5 * u_in**2 * D  # 2D: frontal length D

    print(f"[cyl2d Re=40 D={D} {nx}x{ny}] nu={nu} tau={tau:.6f}", flush=True)
    t0 = time.time()

    solid = cylinder_mask(nx, ny, cx, cy, R, dev)
    near = near_mask_2d(solid)
    n_near = int(near.sum().item())
    nfx, nfy = face_counts_2d(solid)
    n_faces = int((nfx + nfy).sum().item())
    print(
        f"[cyl2d] solid={int(solid.sum())} near={n_near} faces={n_faces} "
        f"ratio={n_faces / n_near:.4f}",
        flush=True,
    )

    rho0 = torch.ones((ny, nx), device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium(rho0, ux0, torch.zeros_like(rho0))
    del rho0, ux0

    yy, xx = torch.meshgrid(
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    r_c = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    nxn = ((xx - cx) / R).clamp(-1, 1)
    nyn = ((yy - cy) / R).clamp(-1, 1)
    norm = torch.sqrt(nxn**2 + nyn**2).clamp(min=1e-10)
    nxn = (nxn / norm) * near.float()
    nyn = (nyn / norm) * near.float()
    q_sm = (r_c - R).clamp(0.05, 1.0) * near.float()

    mem_hist, cdp_hist, cdf_std, cdf_fc, cdf_bfl = [], [], [], [], []
    step = 0
    for step in range(1, a.steps + 1):
        f = stream(collide_bgk(f, tau))
        # MEM raw force (post-stream, pre-BB)
        mem_hist.append(compute_obstacle_forces(f, solid)[0])
        f = far_field_bc_2d(f, u_in, solid)

        if step % a.sample == 0:
            rho, ux, uy = macroscopic(f)
            p = (rho - 1.0) / 3.0
            nm = near.float()
            far = (~solid).float() * (1.0 - nm)
            p0 = (p * far).sum() / far.sum().clamp(min=1.0)
            p_corr = p - p0
            cdp_hist.append(float((-(p_corr * nxn * nm)).sum().item() / dpS))
            u_dot_n = ux * nxn + uy * nyn
            ut_x = ux - u_dot_n * nxn
            cdf_std.append(float((2.0 * nu * ut_x * nm).sum().item() / dpS))
            cdf_fc.append(float((2.0 * nu * ux * nfy).sum().item() / dpS))
            cdf_bfl.append(
                float((nu * ut_x * (1.0 / q_sm.clamp(min=1e-6)) * nm).sum().item() / dpS)
            )

        if step % 5000 == 0:
            k = min(200, len(mem_hist))
            mm = sum(mem_hist[-k:]) / k / dpS
            print(
                f"[cyl2d] step={step} Cd_mem={mm:.4f} "
                f"Cd_p={sum(cdp_hist[-k:]) / k:.4f} "
                f"Cd_f_std={sum(cdf_std[-k:]) / k:.4f} "
                f"Cd_f_faces={sum(cdf_fc[-k:]) / k:.4f} "
                f"Cd_f_bfl={sum(cdf_bfl[-k:]) / k:.4f} ({time.time() - t0:.0f}s)",
                flush=True,
            )

    k = min(200, len(mem_hist))
    out = {
        "case": "cylinder_2d_re40_independent_crosscheck",
        "D": D,
        "nx": nx,
        "ny": ny,
        "domain_D": a.domain_D,
        "Re": Re,
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "steps": a.steps,
        "n_finished": step,
        "n_near": n_near,
        "n_faces": n_faces,
        "face_cell_ratio": n_faces / n_near,
        "cd_mem": sum(mem_hist[-k:]) / k / dpS,
        "cd_mem_last": float(mem_hist[-1].item() / dpS),
        "cd_pressure": sum(cdp_hist[-k:]) / k,
        "cd_friction_standard": sum(cdf_std[-k:]) / k,
        "cd_friction_faces": sum(cdf_fc[-k:]) / k,
        "cd_friction_bfl_smooth": sum(cdf_bfl[-k:]) / k,
        "plateau_mem": {
            f"{int(fr * 100)}%": float(
                sum(mem_hist[int(len(mem_hist) * fr) - k : int(len(mem_hist) * fr)]) / k / dpS
            )
            for fr in (0.25, 0.5, 0.75, 1.0)
        },
        "wall_s": time.time() - t0,
    }
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
