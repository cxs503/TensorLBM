#!/usr/bin/env python3
"""Post-process a saved cylinder_3d final field: decompose the Ladd MEM force
by mask (all solid cells vs. only wall-adjacent surface cells) and cross-check
against the pressure/friction diagnostics on the SAME field.

    python _analyze_field_mem.py <field.pt> <D_cells> [--nz N] [--lateral L]

The MEM convention is identical to run.py: called on the post-stream field at
the solid cells (== the populations the next half-way bounce-back reverses).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
for _p in (_REPO / "src", _REPO / "benchmarks"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from tensorlbm.d3q19 import C  # noqa: E402
from tensorlbm.drag_pressure import (  # noqa: E402
    SurfaceMesh,
    drag_friction_integration,
    drag_pressure_integration,
    get_near_wall_2d,
)


def mem_force(f, mask):
    """Ladd MEM: F_x = 2 * sum_solid c_ix f_i  (see obstacles.compute_obstacle_forces_3d)."""
    c = C.to(f.device).float()
    cx = c[:, 0].view(19, 1, 1, 1)
    return 2.0 * (cx * (f * mask.unsqueeze(0))).sum()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("field", type=Path)
    ap.add_argument("D", type=int)
    ap.add_argument("--nz", type=int, default=1)
    ap.add_argument("--lateral", type=float, default=16.0)
    ap.add_argument("--u-in", type=float, default=0.08)
    a = ap.parse_args()

    dev = torch.device("cpu")
    f = torch.load(a.field, map_location=dev).float()
    _, nz, ny, nx = f.shape
    D = a.D
    R = D / 2.0
    cx, cy = nx // 2, ny // 2
    Re, u_in = 40.0, a.u_in
    nu = u_in * D / Re
    dpS = 0.5 * u_in**2 * (D * nz)
    print(f"field {tuple(f.shape)}  D={D} nz={nz} nx={nx} blockage={100 * D / nx:.2f}%")

    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    solid = (xx - cx) ** 2 + (yy - cy) ** 2 <= R**2
    fluid = ~solid
    # wall-adjacent solid cells (4-neighbour + z-rolls, z periodic)
    surf = solid & (
        torch.roll(fluid, 1, 1)
        | torch.roll(fluid, -1, 1)  # y neighbours
        | torch.roll(fluid, 1, 2)
        | torch.roll(fluid, -1, 2)  # x neighbours
    )
    interior = solid & ~surf

    fx_all = float(mem_force(f, solid).item()) / dpS
    fx_surf = float(mem_force(f, surf).item()) / dpS
    fx_int = float(mem_force(f, interior).item()) / dpS
    print(
        f"n_solid={int(solid.sum())} n_surface={int(surf.sum())} n_interior={int(interior.sum())}"
    )
    print(f"Cd_mem(all solid)   = {fx_all:.4f}   err = {(fx_all - 1.5) / 1.5 * 100:+.2f}%")
    print(f"Cd_mem(surface only)= {fx_surf:.4f}   err = {(fx_surf - 1.5) / 1.5 * 100:+.2f}%")
    print(f"Cd_mem(interior)    = {fx_int:.4f}   (spurious/self-cancelling term)")

    # pressure/friction diagnostic on the same field
    near = get_near_wall_2d(solid, axis="z")
    mesh = SurfaceMesh.from_cylinder(solid, near, cx, cy, R, axis="z")
    q = (torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) - R).clamp(0.05, 1.0) * near.float()
    cdp = float(
        drag_pressure_integration(f, mesh, dpS, extrap="none", p0_method="far_field", solid=solid)[
            0
        ]
    )
    for k in ("standard", "lagrange", "faces", "mix50"):
        cdf = float(
            drag_friction_integration(f, mesh, dpS, nu, q_wall=q, formula=k, solid=solid)[0]
        )
        print(
            f"  [diag] Cd_p={cdp:.4f} Cd_f_{k}={cdf:.4f} Cd_p+f={cdp + cdf:.4f} "
            f"err={(cdp + cdf - 1.5) / 1.5 * 100:+.2f}%"
        )

    # ---- independent control-volume momentum balance -----------------------
    c = C.to(dev).float()
    rho = f.sum(dim=0)  # (nz, ny, nx)
    ux = (c[:, 0].view(19, 1, 1, 1) * f).sum(0) / rho
    uy = (c[:, 1].view(19, 1, 1, 1) * f).sum(0) / rho
    p = (rho - 1.0) / 3.0

    def plane(ax, idx):
        return rho[:, :, idx], ux[:, :, idx], uy[:, :, idx], p[:, :, idx]

    # x-faces: flux = rho*ux^2 + p ; y-faces: rho*ux*uy
    ro, uxo, _, po = plane(2, nx - 1)
    ri, uxi, _, pi = plane(2, 0)
    # average over the periodic z planes
    Fxo = float((ro * uxo**2 + po).mean(0).sum())
    Fxi = float((ri * uxi**2 + pi).mean(0).sum())
    _, uxt, uyt, _ = plane(1, ny - 1)
    _, uxb, uyb, _ = plane(1, 0)
    rt = rho[:, -1, :].mean(0)
    rb = rho[:, 0, :].mean(0)
    Fyt = float((rt * uxt * uyt).mean(0).sum())
    Fyb = float((rb * uxb * uyb).mean(0).sum())
    D_mom = (Fxo - Fxi) + (Fyt - Fyb)
    cd_mom = D_mom / (0.5 * u_in**2 * D * nz)
    print(
        f"CV momentum balance: Fx_out={Fxo:.4f} Fx_in={Fxi:.4f} Fy_top={Fyt:.4f} Fy_bot={Fyb:.4f}"
    )
    print(
        f"Cd_momentum_balance = {cd_mom:.4f}   err = {(cd_mom - 1.5) / 1.5 * 100:+.2f}%   "
        f"(independent of both MEM and the pressure/friction split)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
