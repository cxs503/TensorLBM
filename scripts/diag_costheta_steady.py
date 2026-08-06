#!/usr/bin/env python3
"""Steady-state cos_theta drag segmentation + near-wall u_t profile, R6 vs R8.

Window-averaged (over the last --report steps, every substep) per-link drag
force vs cos_theta on the octree BFL links, plus the near-wall tangential
velocity profile per cos_theta band and per radial ring.  Uses the same
post-collision populations the MEM force is computed from.

Also probes the shell outer-boundary coupling: mean ghost-fed inflow velocity
at the shell interface vs the local L1 velocity, per cos_theta band.
"""
import argparse
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.cascaded_collision import collide_cascaded_d3q19
from tensorlbm.d3q19 import C, equilibrium3d, macroscopic3d
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather, bfl_ramp_wall_velocity, leaf_force_weights,
)
from tensorlbm.octree_boundary.force import (
    ShellForceLedger, build_shell_control_volume, convert_leaf_force_to_l1,
)
from tensorlbm.octree_boundary.geometry import (
    SHELL_OUTSIDE, SOLID, build_octree_shell,
)
from tensorlbm.octree_boundary.stepping import build_ghost_plan, step_octree_shell
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry, root_advance,
)
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult, NestedStaticBlockAMR3D, StaticBlockAMRConfig,
)

p = argparse.ArgumentParser()
p.add_argument("--device", default="cpu")
p.add_argument("--nx", type=int, default=96)
p.add_argument("--ny", type=int, default=64)
p.add_argument("--nz", type=int, default=64)
p.add_argument("--radius", type=float, default=6.0)
p.add_argument("--steps", type=int, default=2500)
p.add_argument("--ramp-steps", type=int, default=300)
p.add_argument("--report", type=int, default=700)
p.add_argument("--bl-thickness", type=float, default=3.0)
args = p.parse_args()

device = torch.device(args.device)
shape = (args.nz, args.ny, args.nx)
cx, cy, cz = args.nx * 0.5, args.ny / 2.0, args.nz / 2.0
RATIO, GHOST, Q = 2, 1, 19
radius_coarse = args.radius
nu = 0.06 * (2.0 * radius_coarse) / 100.0
tau_coarse = 0.5 + 3.0 * nu
tau_leaf = 0.5 + 2.0 * (tau_coarse - 0.5)

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
    bl_thickness_cells=args.bl_thickness, d_max=1, transition=1, device=device)
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

# ---- per-leaf geometry: cos_theta, dist_wall (leaf dx units), rings --------
center = torch.tensor(phys_center, dtype=torch.float64, device=device)
centers64 = (octree._l1_coords.to(torch.float64) + 0.5) / (
    2.0 ** octree.leaf_level.to(torch.float64))[:, None]
r_vec = centers64 - center
r_norm = r_vec.norm(dim=1).clamp_min(1e-12)
cos_theta = (r_vec[:, 0] / r_norm).clamp(-1.0, 1.0)
dwl = (r_norm - radius_l1) / (2.0 ** (-octree.leaf_level.to(torch.float64)))
ring = dwl.round().to(torch.int64).clamp(0, 9)
SEGS = ((-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0))
seg_idx = torch.full((octree.n_leaf,), -1, dtype=torch.int64, device=device)
for k, (lo, hi) in enumerate(SEGS):
    seg_idx[(cos_theta >= lo) & (cos_theta < hi)] = k
bfl_leaf = octree.bfl_mask.sum(dim=0) > 0

# windowed accumulators
acc_seg_fx = torch.zeros(5, dtype=torch.float64, device=device)
acc_seg_n = torch.zeros(5, dtype=torch.int64, device=device)
acc_ut = torch.zeros(5, dtype=torch.float64, device=device)
acc_ut_n = torch.zeros(5, dtype=torch.int64, device=device)
# radial u_t profile per segment: rings 0..9
acc_ring = torch.zeros((5, 10), dtype=torch.float64, device=device)
acc_ring_n = torch.zeros((5, 10), dtype=torch.int64, device=device)

l1_posts = []
ledger = ShellForceLedger(octree)
current_step = 0
c = C.to(device)
opp = octree._opp.to(device)


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


def accumulate(post, f_out):
    """Accumulate per-link drag vs cos_theta + near-wall u_t from the same
    populations the MEM force uses (post = pre-stream, f_out = BFL-applied)."""
    mask = octree.bfl_mask
    qf = octree.q_field
    nt = octree.neighbor_table
    for d in range(1, Q):
        m = mask[d]
        if not bool(m.any()):
            continue
        od = int(opp[d].item())
        idx = torch.nonzero(m, as_tuple=False).squeeze(1)
        qq = qf[d, idx].to(torch.float64)
        fp_d = post[d, idx].to(torch.float64)
        fp_bc = f_out[od, idx].to(torch.float64)
        fx = (fp_d + fp_bc) * float(c[d, 0].item())
        sg = seg_idx[idx]
        for k in range(5):
            sel = sg == k
            if bool(sel.any()):
                acc_seg_fx[k] += float(fx[sel].sum().item())
                acc_seg_n[k] += int(sel.sum().item())
    # near-wall u_t per segment + radial profile (boundary leaves)
    rho_l, ux_l, uy_l, uz_l = macroscopic3d(post.view(Q, 1, 1, -1))
    ux_l = ux_l.view(-1).to(torch.float64)
    uy_l = uy_l.view(-1).to(torch.float64)
    uz_l = uz_l.view(-1).to(torch.float64)
    u_dot_r = (ux_l * r_vec[:, 0] + uy_l * r_vec[:, 1] + uz_l * r_vec[:, 2]) / r_norm
    u_t = torch.sqrt(
        (ux_l - u_dot_r * r_vec[:, 0] / r_norm) ** 2
        + (uy_l - u_dot_r * r_vec[:, 1] / r_norm) ** 2
        + (uz_l - u_dot_r * r_vec[:, 2] / r_norm) ** 2)
    for i in torch.nonzero(bfl_leaf, as_tuple=False).squeeze(1):
        k = int(seg_idx[i].item())
        if k < 0:
            continue
        acc_ut[k] += float(u_t[i].item())
        acc_ut_n[k] += 1
        r = int(ring[i].item())
        acc_ring[k, r] += float(u_t[i].item())
        acc_ring_n[k, r] += 1


def bfl_callback(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
    rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(
        octree_, post, current_step, args.ramp_steps)
    f_out, force = bfl_apply_gather(
        octree_, out, post,
        ghost_plan=ghost_plan_, ghost_vals=ghost_vals,
        wall_velocity=(uwx, uwy, uwz), wall_density=rho_w,
        force_weights=leaf_weights, return_force=True)
    if current_step > args.steps - args.report:
        accumulate(post, f_out)
    return f_out, force


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
cd_mem = mem_mean / dynamic_area_mem
print(f"R{int(args.radius)} bl={args.bl_thickness}: tau_c={tau_coarse:.4f} "
      f"tau_L1={config1.tau_fine:.4f} tau_shell={0.5+2.0*(config1.tau_fine-0.5):.4f} "
      f"Cd_mem={cd_mem:.4f} Cd_cv={cv_mean/dynamic_area_cv:.4f}", flush=True)

n_win = int(acc_seg_n.sum().item())
print(f"window={args.report} steps, accumulated links={n_win}")
print("window-averaged per-link drag force vs cos_theta (leaf units):")
tot = float(acc_seg_fx.sum().item())
for k, (lo, hi) in enumerate(SEGS):
    n = int(acc_seg_n[k].item())
    fx = float(acc_seg_fx[k].item())
    print(f"  cos_theta [{lo:+.1f},{hi:+.1f}): n={n} fx={fx:+.4f} "
          f"({100*fx/tot if tot else 0:5.1f}% of total) per-link={fx/max(n,1):+.6f}",
          flush=True)
print("near-wall (BFL leaves) tangential u per cos_theta segment:")
for k, (lo, hi) in enumerate(SEGS):
    n = int(acc_ut_n[k].item())
    ut = float(acc_ut[k].item()) / max(n, 1)
    print(f"  cos_theta [{lo:+.1f},{hi:+.1f}): n={n} mean_u_t={ut:.5f} "
          f"({100*ut/0.06:.1f}% U0)", flush=True)
print("near-wall u_t radial profile (leaf dx from wall) per cos_theta band:")
print(f"  {'ring':>5}" + "".join(f"{f'ct[{lo:+.1f},{hi:+.1f})':>14}" for lo, hi in SEGS))
for r in range(10):
    row = []
    for k in range(5):
        n = int(acc_ring_n[k, r].item())
        row.append(f"{float(acc_ring[k, r].item())/max(n,1):.5f} (n={n})" if n else "-")
    print(f"  {r:>5}" + "".join(f"{v:>14}" for v in row))
