#!/usr/bin/env python3
"""Replicate run_case geometry exactly (R6 vs R8) and print shell/CV coverage stats."""
import math
import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.force import build_shell_control_volume
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry,
)
from tensorlbm.static_block_amr import (
    NestedStaticBlockAMR3D, StaticBlockAMRConfig,
)
from tensorlbm.d3q19 import equilibrium3d

RATIO, GHOST, Q = 2, 1, 19

def diag(radius, nx, ny, nz, device="cpu"):
    dev = torch.device(device)
    shape = (nz, ny, nx)
    cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
    solid_coarse, solid_coarse_q = build_sphere_geometry(
        nx, ny, nz, cx, cy, cz, radius, dev)
    plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
    box1 = plan.box
    print(f"  coarse box1={[box1.x0, box1.x1, box1.y0, box1.y1, box1.z0, box1.z1]}")
    rho = torch.ones(shape, device=dev)
    ux = torch.full_like(rho, 0.06)
    zero = torch.zeros_like(rho)
    coarse_f = equilibrium3d(rho, ux, zero, zero, device=dev)
    s1, fc1, radius1, _l1 = build_fine_block_geometry(
        box1, (cx, cy, cz), radius, RATIO, GHOST, dev)
    nz1, ny1, nx1 = s1
    print(f"  L1 shape s1={s1} fc1={[float(v) for v in fc1]} radius1={radius1}")
    config1 = StaticBlockAMRConfig(box1, tau_coarse=0.5216, reflux=True,
                                   ghost_interpolation="injection")
    amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
    phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
    radius_l1 = radius1
    octree = build_octree_shell(s1, phys_center, radius_l1,
                                bl_thickness_cells=3.0, d_max=1,
                                transition=1, device=dev)
    st = octree.stats
    shell_band = octree.meta["delta_mask"]
    solid = octree._solid
    shell = octree._shell_mask
    centers = octree.leaf_center.double()
    c = torch.tensor(phys_center, dtype=torch.float64)
    d2 = ((centers - c) ** 2).sum(dim=1)
    n_ci = int((d2 <= radius_l1 ** 2).sum().item())
    bfl_sum = octree.bfl_mask.sum(dim=0)
    n_ci_no_bfl = int(((d2 <= radius_l1 ** 2) & (bfl_sum == 0)).sum().item())
    n_ci_with_bfl = int(((d2 <= radius_l1 ** 2) & (bfl_sum > 0)).sum().item())
    q = octree.q_field[octree.bfl_mask].double()
    q_lo = int((q < 0.5).sum().item()) if q.numel() else 0
    q_hi = int((q >= 0.5).sum().item()) if q.numel() else 0
    # leaves with centre INSIDE sphere that HAVE bfl: q distribution
    ci_mask = (d2 <= radius_l1 ** 2) & (bfl_sum > 0)
    q_ci = octree.q_field[:, ci_mask]
    bfl_ci = octree.bfl_mask[:, ci_mask]
    q_ci_lo = int((q_ci[bfl_ci] < 0.5).sum().item())
    q_ci_hi = int((q_ci[bfl_ci] >= 0.5).sum().item())
    print(f"  n_leaf={st['n_leaf']} shell_cells={st['n_shell_cells']}")
    print(f"  leaf_volume={st['leaf_volume']:.4f} analytic={st['analytic_shell_volume']:.4f} "
          f"vol_err={st['volume_error']*100:.4f}%")
    print(f"  n_center_inside={n_ci} no_bfl={n_ci_no_bfl} with_bfl={n_ci_with_bfl}")
    print(f"  bfl_links={int(octree.bfl_mask.sum().item())} q<0.5={q_lo} q>=0.5={q_hi}")
    print(f"  bfl links on centre-inside leaves: q<0.5={q_ci_lo} q>=0.5={q_ci_hi}")
    # CV box bounds (with-ghost), and distances
    filter_shell = amr.interfaces[0]._interface_filter_blend
    l1_solid_g = torch.zeros((nz1 + 2 * GHOST, ny1 + 2 * GHOST, nx1 + 2 * GHOST),
                             dtype=torch.bool, device=dev)
    l1_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = solid
    cv_w = build_shell_control_volume(
        (int(l1_solid_g.shape[0]), int(l1_solid_g.shape[1]), int(l1_solid_g.shape[2])),
        fc1, radius_l1, shell_band, 6,
        covered=octree._shell_mask, filter_shell=filter_shell,
        solid=l1_solid_g, device=dev)
    # find cv bounds
    nzcv, nycv, nxcv = cv_w.shape
    nz_ = torch.nonzero(cv_w, as_tuple=False).amin(dim=0)
    nx_ = torch.nonzero(cv_w, as_tuple=False).amax(dim=0)
    print(f"  CV with-ghost bounds z[{nz_[0]},{nx_[0]}] y[{nz_[1]},{nx_[1]}] "
          f"x[{nz_[2]},{nx_[2]}] of ({nzcv},{nycv},{nxcv})")
    print(f"  CV cells={int(cv_w.sum().item())}")
    # shell coverage: are there shell-band L1 cells (centre dist<=R+4, fluid)
    # NOT covered by shell mask? and solid cells with centre outside R?
    zz = torch.arange(nz1, device=dev).double().unsqueeze(1).unsqueeze(2) + 0.5
    yy = torch.arange(ny1, device=dev).double().unsqueeze(0).unsqueeze(2) + 0.5
    xx = torch.arange(nx1, device=dev).double().unsqueeze(0).unsqueeze(1) + 0.5
    dfield = torch.sqrt(
        (xx - c[0]) ** 2 + (yy - c[1]) ** 2 + (zz - c[2]) ** 2)
    near_fluid = (~solid) & (dfield <= radius_l1 + 4.0)
    print(f"  fluid cells with dist<=R+4: {int(near_fluid.sum().item())} "
          f"shell covered: {int(shell.sum().item())} "
          f"uncovered: {int((near_fluid & ~shell).sum().item())}")
    # solid cells with centre distance > R (mask error)
    print(f"  solid cells with dist>R: {int((solid & (dfield > radius_l1)).sum().item())}")
    # BFL link normals distribution on the streamwise axis
    return octree

if __name__ == "__main__":
    torch.manual_seed(0)
    print("=== R6 (coarse radius 6) ===")
    diag(6.0, 96, 64, 64)
    print("=== R8 (coarse radius 8) ===")
    diag(8.0, 128, 88, 88)
