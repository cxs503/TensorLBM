"""End-to-end 3-D hybrid DG-LBM sphere-flow, body-fitted prism band.

Drop-in curvilinear successor of ``dg_lbm_sphere_hybrid``: instead of a
staircased Cartesian DG shell, the near-wall band is built as a *body-fitted*
prism band that hugs the true sphere — each element an affine hex from a local
orthonormal frame (radial / azimuthal / streamwise), so the inner face normal is
exactly the radial direction.  The DG advection uses the contravariant metric
``ĉ = J^{-T} c`` and face-area scaling; solid walls use a specular reflection
map about the curved wall normal.  Drag is measured by the momentum-exchange on
the curved DG wall faces (``compute_dg_solid_force_geo``).

CPU smoke (tiny grid, few steps):

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=src python examples/dg_lbm_sphere_prism_hybrid.py 30

A real Cd run needs a larger grid on GPU.
"""

from __future__ import annotations

import math
import sys

import torch

from tensorlbm.boundaries3d import (
    apply_simple_channel_boundaries_3d,
    make_channel_wall_mask_3d,
    sphere_mask,
)
from tensorlbm.d3q19 import OPPOSITE as OPP3D
from tensorlbm.d3q19 import C as C3D
from tensorlbm.d3q19 import W as W3D
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.dg_advection import get_ops
from tensorlbm.dg_curv import (
    compute_dg_solid_force_geo,
    hybrid_step_geo,
    init_dg_from_lbm,
    make_sphere_prism_topology,
)
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import correct_mass3d


def run(
    nz=32,
    ny=32,
    nx=64,
    radius=5.0,
    n_layers=1,
    first_height=1.0,
    growth=1.1,
    n_az=24,
    n_stream=16,
    polar_cap=0.985,
    u_in=0.1,
    tau_lbm=0.9,
    n_steps=30,
    device="cpu",
    dtype=torch.float64,
):
    torch.manual_seed(0)
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5
    solid = sphere_mask(nx, ny, nz, cx, cy, cz, radius, device=device)

    ops = get_ops(degree=1, dx=1.0, dtype=dtype, device=device)
    C = C3D.to(dtype)
    W = W3D.to(dtype)
    opp = OPP3D.to(device)

    # ---- body-fitted prism band (curvilinear) ----
    topo, geo, meta = make_sphere_prism_topology(
        solid,
        center=(cx, cy, cz),
        R=radius,
        n_layers=n_layers,
        first_height=first_height,
        growth=growth,
        n_az=n_az,
        n_stream=n_stream,
        polar_cap=polar_cap,
        vel=C,
        dtype=dtype,
        device=device,
    )
    print(
        f"[prism band] n_band={topo.n_band}  n_az={meta['n_az']}  "
        f"n_stream={meta['n_stream']}  n_layers={meta['n_layers']}"
    )
    if topo.n_band == 0:
        print("  -> no band elements built; abort")
        return False

    rho0 = torch.ones(nz, ny, nx, dtype=dtype, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, dtype=dtype, device=device)
    ux0[solid] = 0.0
    uz0 = torch.zeros_like(ux0)
    f_lbm = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), uz0).to(dtype)

    # seed DG nodal DOFs from the P0 LBM value at each prism element centre
    f_dg = init_dg_from_lbm(f_lbm, topo, ops, dtype)

    wall_mask = make_channel_wall_mask_3d(nz, ny, nx, solid, device=device)
    initial_mass = float(f_lbm.sum().item())

    for step in range(1, n_steps + 1):
        f_lbm, f_dg = hybrid_step_geo(
            f_lbm,
            f_dg,
            C,
            W,
            ops,
            topo,
            geo,
            tau_lbm=tau_lbm,
            dt=1.0,
            n_substeps=16,
            scheme="rk3",
        )
        f_lbm = apply_simple_channel_boundaries_3d(f_lbm, u_in, wall_mask, solid)
        if step % 10 == 0:
            f_lbm = correct_mass3d(f_lbm, initial_mass)
        if step % 5 == 0 or step == n_steps:
            rho, ux, uy, uz = macroscopic3d(f_lbm)
            ux[solid] = 0.0
            ms = float(torch.sqrt(ux * ux + uy * uy + uz * uz).max().item())
            fdg_max = float(f_dg.abs().max().item())
            fdg_mean = float(f_dg.mean().item())
            # drag from the curved DG wall (momentum exchange)
            fdg_f = compute_dg_solid_force_geo(f_dg, topo, geo, C, ops)
            drag = float(fdg_f[0])  # x-component = streamwise
            fxc, _, _ = compute_obstacle_forces_3d(f_lbm, solid)
            # proper drag coefficient: Cd = Fx / (0.5 * u^2 * A), A = pi*R^2
            nu = (tau_lbm - 0.5) / 3.0
            re = u_in * (2.0 * radius) / nu
            area = math.pi * radius * radius
            dyn = 0.5 * u_in * u_in * area
            ref_cd = 24.0 / re * (1.0 + 0.15 * re**0.687) if re > 1e-6 else 100.0
            cd_dg = abs(drag) / dyn
            cd_st = abs(float(fxc)) / dyn
            err_dg = abs(cd_dg - ref_cd) / ref_cd * 100
            err_st = abs(cd_st - ref_cd) / ref_cd * 100
            print(
                f"step {step:3d}: max|u|={ms:.4f} "
                f"Cd(DG curved)={cd_dg:.3f} Cd(staircase)={cd_st:.3f} "
                f"ref(S-N)={ref_cd:.3f}  err_DG={err_dg:.0f}% err_st={err_st:.0f}%  "
                f"[Re={re:.2f}]"
            )
            if not math.isfinite(ms) or ms > 5.0 or not math.isfinite(drag):
                print("  -> UNSTABLE")
                return False
    return True


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ok = run(n_steps=n)
    print("STABLE" if ok else "FAILED")
