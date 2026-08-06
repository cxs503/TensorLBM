#!/usr/bin/env python3
"""Deep R8-vs-R6 geometry audit: ghost fallback pairing, BFL upstream donors, q branches.

Focus: R8-specific mechanisms behind the -22% drag deficit vs uniform R8.
"""
import math
import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.d3q19 import C, OPPOSITE, equilibrium3d
from tensorlbm.octree_boundary.geometry import (
    DOMAIN_OUT, FANOUT, SHELL_OUTSIDE, SOLID, build_octree_shell,
)
from tensorlbm.octree_boundary.stepping import build_ghost_plan
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry,
)
from tensorlbm.static_block_amr import (
    NestedStaticBlockAMR3D, StaticBlockAMRConfig,
)

RATIO, GHOST = 2, 1


def diag(radius, nx, ny, nz):
    dev = torch.device("cpu")
    shape = (nz, ny, nx)
    cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
    solid_coarse, _ = build_sphere_geometry(nx, ny, nz, cx, cy, cz, radius, dev)
    plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
    box1 = plan.box
    rho = torch.ones(shape, device=dev)
    ux = torch.full_like(rho, 0.06)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=dev)
    s1, fc1, radius1, _l1 = build_fine_block_geometry(
        box1, (cx, cy, cz), radius, RATIO, GHOST, dev)
    nz1, ny1, nx1 = s1
    config1 = StaticBlockAMRConfig(box1, tau_coarse=0.5288, reflux=True,
                                   ghost_interpolation="injection")
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
    octree = build_octree_shell(s1, phys_center, radius1, bl_thickness_cells=3.0,
                                d_max=1, transition=1, device=dev)
    gp = build_ghost_plan(octree, s1)
    n_link = gp.n_ghost
    solid = octree._solid

    # --- 1. ghost fallback: how many ghost slots sample an L1-solid cell ---
    # rebuild the fallback decision exactly as build_ghost_plan does
    leaf = gp.leaf
    d_link = octree.interface_links[:, 1]
    opp = OPPOSITE.to(dev)
    c_vec = C.to(dev)
    level_i = octree.leaf_level[leaf]
    dx = 2.0 ** (-level_i.to(torch.float64))
    coords = octree._l1_coords
    centers64 = (coords.to(torch.float64) + 0.5) / (
        2.0 ** octree.leaf_level.to(torch.float64))[:, None]
    p_xyz = centers64[leaf] + c_vec[d_link].to(torch.float64) * dx[:, None]
    p = p_xyz[:, [2, 1, 0]]
    cell_p = torch.stack((
        p[:, 0].floor().to(torch.int64).clamp(0, nz1 - 1),
        p[:, 1].floor().to(torch.int64).clamp(0, ny1 - 1),
        p[:, 2].floor().to(torch.int64).clamp(0, nx1 - 1),
    ), dim=1)
    solid_host = solid[cell_p[:, 0], cell_p[:, 1], cell_p[:, 2]]
    n_fb = int(solid_host.sum().item())
    print(f"=== R{int(radius)} ===")
    print(f"  interface links = {n_link}, ghost fallback (L1-solid host) = {n_fb} "
          f"({100*n_fb/max(n_link,1):.1f}%)")

    # --- 2. BFL upstream donors: how many need ghosts, and of those how many
    #        ghost slots are themselves fallback links -----------------------
    m = octree.bfl_mask
    nt = octree.neighbor_table
    n_bfl = int(m.sum().item())
    n_up_ghost = 0
    n_up_ghost_fb = 0
    n_up_solid = 0
    n_up_leaf = 0
    q_lin = torch.zeros(3, dtype=torch.float64)
    q_quad = torch.zeros(3, dtype=torch.float64)
    n_lin = n_quad = 0
    fb_by_dir = {}
    for d in range(1, 19):
        idx = torch.nonzero(m[d], as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        up = nt[int(OPPOSITE[d].item()), idx]
        n_up_leaf += int((up >= 0).sum().item())
        n_up_ghost += int((up == SHELL_OUTSIDE).sum().item())
        n_up_solid += int((up == SOLID).sum().item())
        qq = octree.q_field[d, idx].to(torch.float64)
        lin = qq < 0.5
        n_lin += int(lin.sum().item())
        n_quad += int((~lin).sum().item())
        # per-link "force magnitude weight" = 1 (each link contributes c_d*(f_d+f_bc));
        # track q means per branch
        q_lin += torch.tensor([qq[lin].mean().item(), 0, 0]) if lin.any() else 0
        q_quad += torch.tensor([qq[~lin].mean().item(), 0, 0]) if (~lin).any() else 0
        if int((up == SHELL_OUTSIDE).sum().item()):
            # do these ghost slots fall back to solid host?
            slots = gp.slot[d, idx[up == SHELL_OUTSIDE]]
            gl = gp.leaf[slots]
            gd = octree.interface_links[slots, 1]
            gdx = 2.0 ** (-octree.leaf_level[gl].to(torch.float64))
            gp_xyz = centers64[gl] + c_vec[gd].to(torch.float64) * gdx[:, None]
            gpp = gp_xyz[:, [2, 1, 0]]
            gcell = torch.stack((
                gpp[:, 0].floor().to(torch.int64).clamp(0, nz1 - 1),
                gpp[:, 1].floor().to(torch.int64).clamp(0, ny1 - 1),
                gpp[:, 2].floor().to(torch.int64).clamp(0, nx1 - 1),
            ), dim=1)
            gsh = solid[gcell[:, 0], gcell[:, 1], gcell[:, 2]]
            n_up_ghost_fb += int(gsh.sum().item())
            fb_by_dir[int(d)] = (int((up == SHELL_OUTSIDE).sum().item()),
                                 int(gsh.sum().item()))
    print(f"  BFL links = {n_bfl}, upstream: leaf={n_up_leaf} ghost={n_up_ghost} "
          f"solid={n_up_solid} fanout={n_bfl-n_up_leaf-n_up_ghost-n_up_solid}")
    print(f"  BFL upstream-ghost links whose ghost slot is fallback: {n_up_ghost_fb}")
    print(f"  q branches: lin(q<0.5)={n_lin} ({100*n_lin/n_bfl:.1f}%) "
          f"quad={n_quad} ({100*n_quad/n_bfl:.1f}%)")
    print(f"  q mean: lin={q_lin[0]:.4f} quad={q_quad[0]:.4f}")
    print(f"  per-dir ghost-upstream (count, fallback): {fb_by_dir}")

    # --- 3. force-per-branch estimate with a synthetic wall field -------------
    # use f = equilibrium(1, u, 0, 0): wall normal at rest. Compute the MEM
    # force per link for lin vs quad branches (leaf lattice units, unweighted)
    from tensorlbm.d3q19 import W
    f_eq = equilibrium3d(
        torch.ones(octree.n_leaf, device=dev),
        torch.full((octree.n_leaf,), 0.06, device=dev),
        torch.zeros(octree.n_leaf, device=dev),
        torch.zeros(octree.n_leaf, device=dev), device=dev,
    )
    f_eq = f_eq.view(19, octree.n_leaf)
    c = C.to(dev)
    F = torch.zeros(3, dtype=torch.float64)
    n_lin_l = n_quad_l = 0
    F_lin = torch.zeros(3, dtype=torch.float64)
    F_quad = torch.zeros(3, dtype=torch.float64)
    for d in range(1, 19):
        idx = torch.nonzero(m[d], as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        od = int(OPPOSITE[d].item())
        qq = octree.q_field[d, idx].to(torch.float64)
        fp_d = f_eq[d, idx].to(torch.float64)
        fp_opp = f_eq[od, idx].to(torch.float64)
        up = nt[od, idx]
        fp_up = torch.zeros_like(fp_d)
        valid = up >= 0
        fp_up[valid] = f_eq[d, up[valid]].to(torch.float64)
        gh = up == SHELL_OUTSIDE
        if bool(gh.any()):
            fp_up[gh] = fp_d[gh]  # approximate: ghost ~ leaf value (far field ~ uniform)
        lin = qq < 0.5
        f_lin_v = 2.0 * qq * fp_d + (1.0 - 2.0 * qq) * fp_up
        safe_q = torch.where(lin, torch.ones_like(qq), qq)
        f_quad_v = fp_d / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
        f_bc = torch.where(lin, f_lin_v, f_quad_v)
        ex = (fp_d + f_bc).unsqueeze(1) * c[d].to(torch.float64)
        F += ex.sum(dim=0)
        F_lin += ex[lin].sum(dim=0)
        F_quad += ex[~lin].sum(dim=0)
        n_lin_l += int(lin.sum().item())
        n_quad_l += int((~lin).sum().item())
    print(f"  synthetic MEM Fx total={F[0]:.4f} per-link={F[0]/n_bfl:.6f}")
    print(f"    lin: n={n_lin_l} Fx={F_lin[0]:.4f} per-link={F_lin[0]/max(n_lin_l,1):.6f}")
    print(f"    quad: n={n_quad_l} Fx={F_quad[0]:.4f} per-link={F_quad[0]/max(n_quad_l,1):.6f}")
    print(f"    Fy={F[1]:.4f} Fz={F[2]:.4f} (should be ~0)")
    print()


for (r, nx, ny, nz) in ((6.0, 96, 64, 64), (8.0, 128, 88, 88)):
    diag(r, nx, ny, nz)
