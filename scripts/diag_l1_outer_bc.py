#!/usr/bin/env python3
"""Probe the L1-block velocity field at the octree shell's OUTER boundary.

Runs the L1 block alone (frozen solid sphere, no octree shell) to near-steady
near-wall state, then reports the L1 tangential velocity at the radial
positions where the octree shell's ghost-fed outer boundary sits
(r = R + bl + transition + ~0.5 L1, i.e. the exterior L1 cells just outside
the covered band) vs the free stream.  Compares R6 and R8.

If the L1 field is significantly depressed there (u << U0), the shell's outer
BC underdrives the near-wall shear layer -> uniform drag deficit.
"""
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.cascaded_collision import collide_cascaded_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry, root_advance,
)
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult, NestedStaticBlockAMR3D, StaticBlockAMRConfig,
)

RATIO, GHOST, Q = 2, 1, 19


def probe(radius, nx, ny, nz, bl, steps=600, ramp=200):
    device = torch.device("cuda:0")
    shape = (nz, ny, nx)
    cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
    nu = 0.06 * (2.0 * radius) / 100.0
    tau_coarse = 0.5 + 3.0 * nu
    solid_coarse, solid_coarse_q = build_sphere_geometry(
        nx, ny, nz, cx, cy, cz, radius, device)
    plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
    box1 = plan.box
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, 0.06)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)
    s1, fc1, radius1, _l1 = build_fine_block_geometry(
        box1, (cx, cy, cz), radius, RATIO, GHOST, device)
    nz1, ny1, nx1 = s1
    config1 = StaticBlockAMRConfig(
        box1, tau_coarse=tau_coarse, reflux=True,
        ghost_interpolation="injection")
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST),
                   float(fc1[2] - GHOST))
    # octree only for its geometry (solid mask + shell mask + leaf coords)
    octree = build_octree_shell(s1, phys_center, radius1,
                                bl_thickness_cells=bl, d_max=1, transition=1,
                                device=device)
    l1_solid_phys = octree._solid
    l1_solid_g = torch.zeros((nz1 + 2 * GHOST, ny1 + 2 * GHOST, nx1 + 2 * GHOST),
                             dtype=torch.bool, device=device)
    l1_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = l1_solid_phys
    l1_solid_q = l1_solid_g.unsqueeze(0).expand(Q, *l1_solid_g.shape).contiguous()
    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(shape, width=16, max_strength=0.2,
                                  device=device, faces=sponge_faces)
    l1_posts = []

    def advance(f, tau, level, substep):
        if level == 0:
            out, post, _ = root_advance(
                f, tau, solid_coarse_q, sigma, 0.06, collision="cascaded")
            return AMRAdvanceResult(out, post)
        if level == 1:
            before = f
            collided = collide_cascaded_d3q19(f, tau)
            post = torch.where(l1_solid_q, before, collided)
            out = stream3d(post)
            l1_posts.append(post[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST])
            return AMRAdvanceResult(out, post)
        raise ValueError

    for _ in range(steps):
        l1_posts.clear()
        amr.step(advance)

    l1_fine = amr.interfaces[0].fine_f
    f_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
    rho_l, ux_l, uy_l, uz_l = macroscopic3d(f_phys)
    # sample at shell-outer-boundary ghost positions: L1 cells just outside the
    # covered band (dist in (R+bl+transition, R+bl+transition+1) L1)
    center = torch.tensor(phys_center, dtype=torch.float64, device=device)
    nz_, ny_, nx_ = f_phys.shape[1:]
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz_, device=device, dtype=torch.float64) + 0.5,
        torch.arange(ny_, device=device, dtype=torch.float64) + 0.5,
        torch.arange(nx_, device=device, dtype=torch.float64) + 0.5,
        indexing="ij")
    rr = torch.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2
                    + (zz - center[2]) ** 2)
    outer = (rr >= radius1 + bl) & (rr < radius1 + bl + 1.5) & (~l1_solid_phys)
    inner = (rr >= radius1 - 0.5) & (rr < radius1 + 0.5) & (~l1_solid_phys)
    ct = (xx - center[0]) / rr.clamp_min(1e-12)
    print(f"=== R{int(radius)} bl={bl} (R_l1={radius1:.1f}) ===")
    print(f"  L1 tau={config1.tau_fine:.4f}")
    for name, sel in (("inner (r~R+0..0.5)", inner), ("outer (r~R+{:.1f})".format(bl), outer)):
        ux_s = ux_l[sel]
        uy_s = uy_l[sel]
        uz_s = uz_l[sel]
        rv = torch.stack((xx[sel] - center[0], yy[sel] - center[1],
                          zz[sel] - center[2]), dim=1)
        rn = rv.norm(dim=1).clamp_min(1e-12)
        u_dot_r = (ux_s * rv[:, 0] + uy_s * rv[:, 1] + uz_s * rv[:, 2]) / rn
        u_t = torch.sqrt((ux_s - u_dot_r * rv[:, 0] / rn) ** 2
                         + (uy_s - u_dot_r * rv[:, 1] / rn) ** 2
                         + (uz_s - u_dot_r * rv[:, 2] / rn) ** 2)
        ct_s = ct[sel]
        print(f"  {name}: n={int(sel.sum())} mean_u_x={ux_s.mean().item():+.5f} "
              f"mean_u_t={u_t.mean().item():.5f} mean|u|={torch.sqrt(ux_s**2+uy_s**2+uz_s**2).mean().item():.5f}")
        for lo, hi in ((-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)):
            s2 = (ct_s >= lo) & (ct_s < hi)
            if bool(s2.any()):
                print(f"    ct[{lo:+.1f},{hi:+.1f}): n={int(s2.sum())} "
                      f"u_x={ux_s[s2].mean().item():+.5f} u_t={u_t[s2].mean().item():.5f}")
    print()


for bl in (3.0, 5.0):
    probe(6.0, 96, 64, 64, bl)
    probe(8.0, 128, 88, 88, bl)
