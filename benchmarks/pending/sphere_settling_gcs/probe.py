#!/usr/bin/env python3
"""Probe: periodic settling cell driven by gravity — anchor calibration."""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "src"))

import torch

from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, sphere_mask
from tensorlbm.d3q19 import C, W, equilibrium3d
from tensorlbm.force_control_surface import control_surface_force, hasimoto_factor, stokes_drag
from tensorlbm.solver3d import collide_bgk3d, macroscopic3d, stream3d

torch.set_num_threads(8)
CS2, CS4 = 1.0 / 3.0, 1.0 / 9.0


def guo_source(rho, ux, uy, uz, fx, fy, fz, tau):
    c = C.to(fx.device).float() if isinstance(fx, torch.Tensor) else C.to(rho.device).float()
    w = W.to(rho.device).float().view(19, 1, 1, 1)
    cu = c[:, 0].view(19, 1, 1, 1) * ux + c[:, 1].view(19, 1, 1, 1) * uy + c[:, 2].view(19, 1, 1, 1) * uz
    cF = c[:, 0].view(19, 1, 1, 1) * fx + c[:, 1].view(19, 1, 1, 1) * fy + c[:, 2].view(19, 1, 1, 1) * fz
    return (1.0 - 0.5 / tau) * w * (cF / CS2 + cu * cF / CS4)


def main(R=6.0, L=48, u_target=0.02, re=0.05, steps=4000, sample=50):
    L = int(L)
    centre = (L / 2.0, L / 2.0, L / 2.0)
    nu = u_target * 2 * R / re
    tau = 0.5 + 3 * nu
    solid = sphere_mask(L, L, L, *centre, R, device=torch.device("cpu"))
    n_solid = int(solid.sum().item())
    v_cell = float(L**3)
    v_fluid = v_cell - n_solid
    f_stokes = stokes_drag(R, nu, u_target)
    g = f_stokes / v_fluid  # force density (rho=1)
    print(f"R={R} L={L} nu={nu} tau={tau:.4f} n_solid={n_solid} g={g:.6e} F_stokes={f_stokes:.6f}")

    rho0 = torch.ones(L, L, L)
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))
    del rho0
    lam = R / L
    print("hasimoto(lam) =", hasimoto_factor(lam))

    t0 = time.time()
    for step in range(1, steps + 1):
        rho, ux, uy, uz = macroscopic3d(f)
        fx = rho * g
        f_c = collide_bgk3d(f, tau) + guo_source(rho, ux, uy, uz, fx, torch.zeros_like(fx), torch.zeros_like(fx), tau)
        f_c[:, solid] = f[:, solid]
        f = stream3d(bounce_back_cells_3d(f_c, solid))
        if step % sample == 0 and step % 500 == 0:
            rho2, ux2, uy2, uz2 = macroscopic3d(f)
            fluid = ~solid
            u_drift = float((rho2 * ux2)[fluid].sum() / rho2[fluid].sum())
            res = control_surface_force(
                f, centre, R + 2.5, periodic=True, body_force=(g, 0.0, 0.0)
            )
            res0 = control_surface_force(f, centre, R + 2.5, periodic=True)
            print(
                f"step={step} U_drift={u_drift:.6f} "
                f"F_gcs_corr={res.fx:.5f} F_gcs_raw={res0.fx:.5f} anchor={g * v_fluid:.5f} "
                f"(({time.time() - t0:.0f}s))",
                flush=True,
            )
    rho2, ux2, uy2, uz2 = macroscopic3d(f)
    fluid = ~solid
    u_drift = float((rho2 * ux2)[fluid].sum() / rho2[fluid].sum())
    print(f"U_drift/U_stokes_0 = {u_drift / u_target:.5f}")
    print(f"U_drift*hasimoto/U_stokes_0 = {u_drift * hasimoto_factor(lam) / u_target:.5f}")


if __name__ == "__main__":
    main(*[float(v) if i < 4 else int(v) for i, v in enumerate(sys.argv[1:])])