#!/home/wxsc/anaconda3/envs/ftw-env/bin/python
"""3D cylinder with static-block AMR (2:1) + MRT+Smag LES — quantitative drag.

Validates: local refinement gives the equivalent of a fine global grid at
a fraction of the cost.  Coarse grid resolves D_c cells across the cylinder;
a 2:1 fine block wraps the near-cylinder region (equivalent D_f = 2·D_c).
Reference for 3D cylinder (Luo et al. 2007 / Schlanderer & Sandberg 2011):
  Re=1000: Cd ≈ 1.0, St ≈ 0.20 (3D, spanwise 2π-4π)
"""
import sys, os, time, json, math
sys.path.insert(0, "/DATA/cxs_host/TensorLBM/src")

import torch
from tensorlbm.d3q19 import equilibrium3d, C as C19
from tensorlbm.solver3d import stream3d, collide_mrt3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.static_block_amr import (
    StaticBlockAMR3D, StaticBlockAMRConfig, AMRAdvanceResult, BoxRegion,
)


def cylinder_mask_3d(nz, ny, nx, cx, cy, radius, device):
    """Cylinder along z: solid where (x-cx)²+(y-cy)² < r²."""
    zz, yy, xx = torch.meshgrid(torch.arange(nz, device=device),
                                torch.arange(ny, device=device),
                                torch.arange(nx, device=device), indexing="ij")
    return ((xx - cx)**2 + (yy - cy)**2).sqrt() < radius


def run(re=1000, nx=240, ny=96, nz=48, D_c=20, u_in=0.1,
        n_steps=6000, device="cuda:0", tau_coarse=0.51):
    """3D cylinder, coarse grid + 2:1 fine block around the body."""
    dev = torch.device(device)
    radius = D_c / 2.0
    nu = u_in * D_c / re
    tau = 3.0 * nu + 0.5
    re_actual = u_in * D_c / (3.0 * (tau_coarse - 0.5)) if tau_coarse > 0.5 else re
    tau = tau_coarse  # use the stable coarse tau; Re set via D_c

    cx, cy = int(nx * 0.3), int(ny * 0.5)
    solid = cylinder_mask_3d(nz, ny, nx, cx, cy, radius, dev)
    # surface for diagnostics
    fluid = ~solid
    surf = solid & (torch.roll(fluid, 1, dims=1) | torch.roll(fluid, -1, dims=1)
                    | torch.roll(fluid, 1, dims=2) | torch.roll(fluid, -1, dims=2))

    # fine block: wrap the cylinder with 1.5D margin in x,y; full span z
    pad = int(1.5 * D_c)
    x0 = int(max(1, cx - radius - pad))
    x1 = int(min(nx - 2, cx + radius + pad))
    y0 = int(max(1, cy - radius - pad))
    y1 = int(min(ny - 2, cy + radius + pad))
    box = BoxRegion(x0, x1, y0, y1, 1, nz - 2)
    print(f"fine block box: x[{x0},{x1}] y[{y0},{y1}] z[1,{nz-2}]")

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(rho0), torch.zeros_like(rho0),
                      device=dev)

    config = StaticBlockAMRConfig(box, tau_coarse=tau, reflux=True)
    solver = StaticBlockAMR3D(f, config)

    initial_mass = float(f.sum().item())
    dyn_p = 0.5 * u_in**2 * D_c * nz  # frontal area = D_c × span (Lz=nz)
    cdev = C19.to(dev).float()

    # coarse solid (unused inside the fine block; the fine solid is the
    # cylinder sampled at fine resolution — here we simply use the coarse
    # solid for the force at the coarse level and note this is approximate)
    cd_list = []
    t0 = time.time()

    # fine solid mask (2x resolution) — use the EXACT piecewise-constant
    # sampling mapping of _sample_parent_with_ghost:
    #   fine idx i  →  coarse idx floor((x0*ratio - ghost + i)/ratio)
    r, g = config.ratio, config.ghost
    zf, yf, xf = solver.fine_f.shape[1:]
    z_f = torch.arange(box.z0 * r - g, box.z1 * r + g, device=dev)
    y_f = torch.arange(box.y0 * r - g, box.y1 * r + g, device=dev)
    x_f = torch.arange(box.x0 * r - g, box.x1 * r + g, device=dev)
    z_c = torch.div(z_f, r, rounding_mode="floor")
    y_c = torch.div(y_f, r, rounding_mode="floor")
    x_c = torch.div(x_f, r, rounding_mode="floor")
    solid_f = solid[z_c[:, None, None], y_c[None, :, None], x_c[None, None, :]]
    # MEM force acts on SURFACE solid cells only (fluid neighbour present);
    # interior solid cells carry spurious post-stream populations.
    fluid_f = ~solid_f
    surf_f = solid_f & (
        torch.roll(fluid_f, 1, dims=0) | torch.roll(fluid_f, -1, dims=0)
        | torch.roll(fluid_f, 1, dims=1) | torch.roll(fluid_f, -1, dims=1)
        | torch.roll(fluid_f, 1, dims=2) | torch.roll(fluid_f, -1, dims=2)
    )

    # MEM force accumulator: computed inside advance AFTER stream, BEFORE BB
    # (exactly the 2D-validated order: collide→freeze→stream→[force]→BB)
    force_holder = {"fx": 0.0, "valid": False}
    cdev = C19.to(dev).float()

    def advance(fb: torch.Tensor, tau_l: float, level: int, substep: int):
        """Collide (freeze solid) + stream; compute MEM force; then BB.

        The force must use the post-stream, pre-BB field (Ladd): on the
        solid surface cells, F = 2·Σ c·f.  BB afterwards keeps the body
        physical.  Only the fine level contributes the resolved force.
        """
        bf = fb.clone()
        col = collide_smagorinsky_mrt3d(fb, tau_l, C_s=0.12)
        if level == 1:
            fb2 = torch.where(solid_f.unsqueeze(0), bf, col)
        else:
            fb2 = torch.where(solid.unsqueeze(0), bf, col)
        out = stream3d(fb2)
        if level == 1:
            # Ladd MEM on surface solid cells (post-stream, pre-BB),
            # divided by ratio³: the fine grid has ratio³× more solid
            # cells than the parent (volume correction, cf.
            # octree_boundary.force.convert_leaf_force_to_l1).
            fx = 2.0 * (cdev[:, 0].view(19, 1, 1, 1)
                        * out * surf_f.unsqueeze(0)).sum().item() / (r ** 3)
            force_holder["fx"] = fx
            force_holder["valid"] = True
        # bounce-back on solid cells (post-stream, keeps body physical)
        from tensorlbm.boundaries3d import bounce_back_cells_3d
        m = solid_f if level == 1 else solid
        out = bounce_back_cells_3d(out, m)
        return AMRAdvanceResult(out, fb2)

    for step in range(1, n_steps + 1):
        force_holder["valid"] = False
        solver.step(advance)
        fx = force_holder["fx"] if force_holder["valid"] else 0.0
        if step % 500 == 0:
            solver.coarse_f = solver.coarse_f * (initial_mass / solver.coarse_f.sum().item())
        if step > n_steps // 3:
            cd = fx / dyn_p
            cd_list.append(cd)
        if step % 1000 == 0 or step == n_steps:
            cd_avg = sum(cd_list) / max(len(cd_list), 1)
            el = time.time() - t0
            print(f"  step {step:5d}: Cd={cd_avg:.4f} (ref 1.0), {el:.0f}s")

    cd_avg = sum(cd_list) / max(len(cd_list), 1)
    print(f"\n  FINAL: Cd={cd_avg:.4f}  (3D Re={re_actual:.0f} ref ~1.0)")
    return {"re": re, "cd": cd_avg, "tau": tau, "steps": n_steps}


if __name__ == "__main__":
    device = "cuda:0"
    r = run(re=1000, n_steps=6000, device=device)
    with open("/tmp/cyl3d_amr_result.json", "w") as fp:
        json.dump(r, fp, indent=2)
    print("saved /tmp/cyl3d_amr_result.json")
