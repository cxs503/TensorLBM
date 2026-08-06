#!/usr/bin/env python3
"""Decompose the leaf MEM force per BFL link at R6 vs R8 (post-ramp quasi-steady).

Runs the hybrid sphere to quasi-steady state (short), then on the last root
step reports:
  n_bfl_links, per-direction link counts, q distribution,
  per-link x-exchange stats (mean/abs), upstream donor type split,
  and the MEM force vs the CV force in L1 units.
"""
import argparse
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.cascaded_collision import collide_cascaded_d3q19
from tensorlbm.d3q19 import C, equilibrium3d
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather,
    bfl_ramp_wall_velocity,
    leaf_force_weights,
)
from tensorlbm.octree_boundary.force import (
    ShellForceLedger,
    build_shell_control_volume,
    convert_leaf_force_to_l1,
)
from tensorlbm.octree_boundary.geometry import (
    SHELL_OUTSIDE, SOLID, FANOUT, build_octree_shell,
)
from tensorlbm.octree_boundary.stepping import build_ghost_plan, step_octree_shell
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry,
    build_sphere_geometry,
    root_advance,
)
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--nx", type=int, default=96)
p.add_argument("--ny", type=int, default=64)
p.add_argument("--nz", type=int, default=64)
p.add_argument("--radius", type=float, default=6.0)
p.add_argument("--steps", type=int, default=400)
p.add_argument("--ramp-steps", type=int, default=150)
p.add_argument("--report", type=int, default=100)
args = p.parse_args()

device = torch.device(args.device)
shape = (args.nz, args.ny, args.nx)
cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0
RATIO, GHOST, Q = 2, 1, 19
radius_coarse = args.radius
nu = 0.06 * (2.0 * radius_coarse) / 100.0
tau_coarse = 0.5 + 3.0 * nu

solid_coarse, solid_coarse_q = build_sphere_geometry(
    args.nx, args.ny, args.nz, cx, cy, cz, radius_coarse, device)
plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
box1 = plan.box
rho = torch.ones(shape, device=device)
ux = torch.full_like(rho, 0.06)
zero = torch.zeros_like(rho)
coarse_f = equilibrium3d(rho, ux, zero, zero, device=device)

s1, fc1, radius1, _l1 = build_fine_block_geometry(
    box1, (cx, cy, cz), radius_coarse, RATIO, GHOST, device)
nz1, ny1, nx1 = s1
config1 = StaticBlockAMRConfig(
    box1, tau_coarse=tau_coarse, reflux=True, ghost_interpolation="injection")
amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))

phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
radius_l1 = radius1
octree = build_octree_shell(
    s1, phys_center, radius_l1,
    bl_thickness_cells=3.0, d_max=1, transition=1, device=device)
shell_band = octree.meta["delta_mask"]
host = octree.leaf_host_cell
l1_fine = amr.interfaces[0].fine_f
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
covered_phys = octree._shell_mask

filter_shell = amr.interfaces[0]._interface_filter_blend
cv_w = build_shell_control_volume(
    (int(l1_solid_g.shape[0]), int(l1_solid_g.shape[1]), int(l1_solid_g.shape[2])),
    fc1, radius_l1, shell_band, 6,
    covered=octree._shell_mask, filter_shell=filter_shell,
    solid=l1_solid_g, device=device)
cv = cv_w[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]

sponge_faces = ("x+", "y-", "y+", "z-", "z+")
sigma = build_sponge_sigma_3d(shape, width=16, max_strength=0.2,
                              device=device, faces=sponge_faces)
dynamic_area_cv = 0.5 * 0.06 ** 2 * math.pi * radius_l1 ** 2
radius_leaf = radius_l1 / dx_leaf
dynamic_area_mem = 0.5 * 0.06 ** 2 * math.pi * radius_leaf ** 2

l1_posts = []
ledger = ShellForceLedger(octree)
current_step = 0


def advance(f, tau, level, substep):
    if level == 0:
        out, post_collision, _ = root_advance(
            f, tau, solid_coarse_q, sigma, 0.06, collision="cascaded")
        return AMRAdvanceResult(out, post_collision)
    if level == 1:
        before = f
        collided = collide_cascaded_d3q19(f, tau)
        post = torch.where(l1_solid_q, before, collided)
        out = stream3d(post)
        l1_posts.append(post[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST])
        return AMRAdvanceResult(out, post)
    raise ValueError(f"unexpected level {level}")


def shell_advance(f, tau, level, substep):
    collided = collide_cascaded_d3q19(f.view(Q, 1, 1, -1), tau)
    post = collided.view_as(f)
    return AMRAdvanceResult(post.clone(), post)


def bfl_callback(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
    rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(
        octree_, post, current_step, args.ramp_steps)
    return bfl_apply_gather(
        octree_, out, post,
        ghost_plan=ghost_plan_, ghost_vals=ghost_vals,
        wall_velocity=(uwx, uwy, uwz), wall_density=rho_w,
        force_weights=leaf_weights, return_force=True)


c = C.to(device)
c0 = c[:, 0].view(Q, 1, 1, 1)

mem_tail = []
cv_tail = []
for current_step in range(1, args.steps + 1):
    l1_fine = amr.interfaces[0].fine_f
    l1_old_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST].clone()
    l1_posts.clear()
    amr.step(advance)
    l1_fine = amr.interfaces[0].fine_f
    l1_f_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
    l1_pre_shell = l1_f_phys.clone()
    shell_ledger = step_octree_shell(
        octree, shell_advance, l1_old_phys, l1_f_phys,
        tau_coarse=config1.tau_fine,
        l1_post=l1_posts if config1.reflux else None,
        shell_level=1, ghost_plan=ghost_plan,
        bfl_fn=bfl_callback, force_ledger=ledger,
    )
    ledger.observe_cv_force(
        l1_old_phys, l1_pre_shell, l1_posts, cv, solid=l1_solid_phys,
        wall_mom_l1=ledger.wall_momentum_l1(dx_leaf, dt_leaf),
    )
    if current_step > args.steps - args.report:
        mem_tail.append(float(ledger.mem_force[0].item()))
        cv_tail.append(float(ledger.cv_force[0].item()))
    ledger.reset()

mem_mean = sum(mem_tail) / len(mem_tail)
cv_mean = sum(cv_tail) / len(cv_tail)
print(f"R{int(args.radius)}: Cd_mem={mem_mean/dynamic_area_mem:.4f} "
      f"Cd_cv={cv_mean/dynamic_area_cv:.4f} "
      f"F_mem_l1={convert_leaf_force_to_l1(mem_mean, dx_leaf, dt_leaf).item():.4f} "
      f"F_cv_l1={cv_mean:.4f}", flush=True)

# ---- per-link MEM decomposition on the current leaf state ------------------
f_prev = octree.f_leaf
mask = octree.bfl_mask
q_field = octree.q_field
nt = octree.neighbor_table
opp = octree._opp.to(device)
n_link = int(mask.sum().item())
print(f"n_bfl_links={n_link}", flush=True)
x_exch = []
q_hist = []
up_type = {"leaf": 0, "ghost": 0, "solid": 0, "fanout": 0}
for d in range(1, Q):
    m = mask[d]
    if not bool(m.any()):
        continue
    od = int(opp[d].item())
    idx = torch.nonzero(m, as_tuple=False).squeeze(1)
    qq = q_field[d, idx].to(torch.float64)
    fp_d = f_prev[d, idx].to(torch.float64)
    fp_opp = f_prev[od, idx].to(torch.float64)
    up = nt[od, idx]
    # approximate fp_up: real leaves only (ignore ghost values here)
    fp_up = torch.zeros_like(fp_d)
    valid = up >= 0
    if bool(valid.any()):
        fp_up[valid] = f_prev[d, up[valid]].to(torch.float64)
    lin = qq < 0.5
    f_bc_lin = 2.0 * qq * fp_d + (1.0 - 2.0 * qq) * fp_up
    safe_q = torch.where(lin, torch.ones_like(qq), qq)
    f_bc_quad = fp_d / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
    f_bc = torch.where(lin, f_bc_lin, f_bc_quad)
    exchange = fp_d + f_bc
    x_exch.append((exchange * float(c[d, 0].item())).detach().cpu())
    q_hist.append(qq.detach().cpu())
    up_type["leaf"] += int(valid.sum().item())
    up_type["ghost"] += int((up == SHELL_OUTSIDE).sum().item())
    up_type["solid"] += int((up == SOLID).sum().item())
    up_type["fanout"] += int((up == FANOUT).sum().item())

x_exch = torch.cat(x_exch)
q_all = torch.cat(q_hist)
print(f"per-link x-exchange: mean={x_exch.mean():+.6f} absmean={x_exch.abs().mean():.6f} "
      f"sum={x_exch.sum():+.6f}", flush=True)
print(f"q: <0.5 {(q_all<0.5).sum().item()}/{q_all.numel()}  mean={q_all.mean():.3f}", flush=True)
print(f"upstream donors: {up_type}", flush=True)
