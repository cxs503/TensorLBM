#!/usr/bin/env python3
"""Run the exact run_case stepping for N steps; instrument freeze effectiveness."""
import argparse
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.octree_boundary.bfl import bfl_apply_gather, bfl_ramp_wall_velocity, leaf_force_weights
from tensorlbm.octree_boundary.force import ShellForceLedger, build_shell_control_volume
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import build_ghost_plan, step_octree_shell
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import build_fine_block_geometry, build_sphere_geometry, root_advance
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import AMRAdvanceResult, NestedStaticBlockAMR3D, StaticBlockAMRConfig

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--nx", type=int, default=96)
p.add_argument("--ny", type=int, default=64)
p.add_argument("--nz", type=int, default=64)
p.add_argument("--radius", type=float, default=6.0)
p.add_argument("--reynolds", type=float, default=100.0)
p.add_argument("--lattice-speed", type=float, default=0.06)
p.add_argument("--steps", type=int, default=60)
p.add_argument("--ramp-steps", type=int, default=200)
p.add_argument("--d-max", type=int, default=1)
p.add_argument("--collision", default="cumulant")
p.add_argument("--bl-thickness", type=float, default=3.0)
args = p.parse_args()

device = torch.device(args.device)
shape = (args.nz, args.ny, args.nx)
cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0
RATIO, GHOST, Q = 2, 1, 19
radius_coarse = args.radius
nu = args.lattice_speed * (2.0 * radius_coarse) / args.reynolds
tau_coarse = 0.5 + 3.0 * nu

solid_coarse, solid_coarse_q = build_sphere_geometry(
    args.nx, args.ny, args.nz, cx, cy, cz, radius_coarse, device)
plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
box1 = plan.box
rho = torch.ones(shape, device=device)
ux = torch.full_like(rho, args.lattice_speed)
zero = torch.zeros_like(rho)
coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

s1, fc1, radius1, _l1 = build_fine_block_geometry(
    box1, (cx, cy, cz), radius_coarse, RATIO, GHOST, device)
nz1, ny1, nx1 = s1
config1 = StaticBlockAMRConfig(box1, tau_coarse=tau_coarse, reflux=True, ghost_interpolation="injection")
amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))
l1_fine = amr.interfaces[0].fine_f

phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
radius_l1 = radius1
octree = build_octree_shell(
    s1, phys_center, radius_l1,
    bl_thickness_cells=args.bl_thickness, d_max=args.d_max,
    transition=1, device=device)
shell_band = octree.meta["delta_mask"]
host = octree.leaf_host_cell
octree.f_leaf = l1_fine[:, host[:, 0] + GHOST, host[:, 1] + GHOST, host[:, 2] + GHOST].clone()
leaf_weights = leaf_force_weights(octree)
ghost_plan = build_ghost_plan(octree, s1)
dx_leaf = 2.0 ** (-octree.d_max)
dt_leaf = dx_leaf

l1_solid_phys = octree._solid
l1_solid_g = torch.zeros((nz1 + 2 * GHOST, ny1 + 2 * GHOST, nx1 + 2 * GHOST),
                         dtype=torch.bool, device=device)
l1_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = l1_solid_phys
l1_solid_q = l1_solid_g.unsqueeze(0).expand(Q, *l1_solid_g.shape).contiguous()

filter_shell = amr.interfaces[0]._interface_filter_blend
cv_w = build_shell_control_volume(
    (int(l1_solid_g.shape[0]), int(l1_solid_g.shape[1]), int(l1_solid_g.shape[2])),
    fc1, radius_l1, shell_band, 6,
    covered=octree._shell_mask, filter_shell=filter_shell,
    solid=l1_solid_g, device=device)
cv = cv_w[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]

sponge_faces = ("x+", "y-", "y+", "z-", "z+")
sigma = build_sponge_sigma_3d(shape, width=16, max_strength=0.2, device=device, faces=sponge_faces)
dynamic_area_cv = 0.5 * args.lattice_speed ** 2 * math.pi * radius_l1 ** 2
radius_leaf = radius_l1 / dx_leaf
dynamic_area_mem = 0.5 * args.lattice_speed ** 2 * math.pi * radius_leaf ** 2

import math
l1_posts = []
ledger = ShellForceLedger(octree)
current_step = 0

# reference frozen snapshot: L1 solid + L0 solid + leaf values at init
l1_solid_snapshot = l1_fine[:, l1_solid_g].clone()
l0_solid_snapshot = coarse_f[:, solid_coarse].clone()
leaf_snapshot = octree.f_leaf.clone()

def advance(f, tau, level, substep):
    if level == 0:
        out, post_collision, _ = root_advance(
            f, tau, solid_coarse_q, sigma, args.lattice_speed, collision=args.collision)
        return AMRAdvanceResult(out, post_collision)
    if level == 1:
        before = f
        collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
        post = torch.where(l1_solid_q, before, collided)
        out = stream3d(post)
        l1_posts.append(post[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST])
        return AMRAdvanceResult(out, post)
    raise ValueError(f"unexpected level {level}")

def shell_advance(f, tau, level, substep):
    collided = collide_cumulant_d3q19(f.view(Q, 1, 1, -1), tau, C_s=0.0)
    post = collided.view_as(f)
    return AMRAdvanceResult(post.clone(), post)

def bfl_callback(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
    rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(octree_, post, current_step, args.ramp_steps)
    return bfl_apply_gather(
        octree_, out, post,
        ghost_plan=ghost_plan_, ghost_vals=ghost_vals,
        wall_velocity=(uwx, uwy, uwz), wall_density=rho_w,
        force_weights=leaf_weights, return_force=True)

print(f"L1 solid cells: {int(l1_solid_phys.sum().item())}  BFL links: {int(octree.bfl_mask.sum().item())}")

for current_step in range(1, args.steps + 1):
    l1_fine = amr.interfaces[0].fine_f
    l1_old_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST].clone()
    l1_posts.clear()
    amr.step(advance)
    l1_fine = amr.interfaces[0].fine_f
    l1_f_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
    shell_ledger = step_octree_shell(
        octree, shell_advance, l1_old_phys, l1_f_phys,
        tau_coarse=config1.tau_fine,
        l1_post=l1_posts if config1.reflux else None,
        shell_level=1, ghost_plan=ghost_plan,
        bfl_fn=bfl_callback, force_ledger=ledger,
    )
    ledger.observe_cv_force(l1_old_phys, l1_f_phys, l1_posts, cv, solid=l1_solid_phys)
    mem = ledger.mem_force
    cvf = ledger.cv_force
    ledger.reset()
    if current_step in (1, 5, 20, 60) or current_step % 20 == 0:
        # freeze drift checks
        l1_solid_now = l1_fine[:, l1_solid_g]
        l1_drift = float((l1_solid_now - l1_solid_snapshot).abs().max().item())
        l0_drift = float((coarse_f[:, solid_coarse] - l0_solid_snapshot).abs().max().item())
        leaf_drift = float((octree.f_leaf - leaf_snapshot).abs().max().item())
        cd_mem = float(mem[0].item()) / dynamic_area_mem
        cd_cv = float(cvf[0].item()) / dynamic_area_cv
        print(
            f"step={current_step:4d} Cd_mem={cd_mem:+.4f} Cd_cv={cd_cv:+.4f} "
            f"L1solid_drift={l1_drift:.3e} L0solid_drift={l0_drift:.3e} "
            f"leaf_drift={leaf_drift:.3e} ref_res={shell_ledger.mass_residual:.2e}",
            flush=True,
        )

# wake velocity check: u at L1 cells behind the sphere (x > center+R+2, same y,z)
l1_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
_, ux1, uy1, uz1 = macroscopic3d(l1_phys)
cx1, cy1, cz1 = (int(v) for v in phys_center)
wake = ux1[cz1, cy1, cx1 + int(radius1) + 4: cx1 + int(radius1) + 16]
side = ux1[cz1, cy1 - int(radius1) - 4, cx1]
print(f"wake ux (x=R+4..R+16 at centre line): {wake.tolist()}")
print(f"side ux (y=R+4): {float(side.item()):.4f}")
# velocity inside sphere should be ~0 if frozen
u_inside = ux1[cz1, cy1, cx1]
print(f"u at sphere centre (L1): {float(u_inside.item()):.4f}")
