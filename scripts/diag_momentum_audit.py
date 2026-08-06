#!/usr/bin/env python3
"""Full momentum closure audit of the hybrid CV.

Per root step measures, on the streamwise axis:
  import       : L1 streaming import through CV faces
  dP_ext       : exterior L1 cell momentum change (CV region)
  dP_cov       : covered L1 cell momentum change (restriction)
  dP_leaf      : leaf population momentum change
  dP_core      : L1 frozen solid cell momentum change
  restr_cons   : dP_cov - dP_leaf (should be ~0 if restriction is conservative
                 and the covered cells start from the previous restriction)
  wall_mom     : total BFL wall exchange per root step (sum of substeps,
                 converted to L1 units)  [force on fluid]
  reflux_app   : reflux momentum applied to exterior cells
  fine/coarse  : interface transfer net momenta
Closure identities:
  (1) import + dP_ext + dP_cov + dP_core + wall_mom + reflux_app == 0
      (total momentum conservation of the joint system; reflux_app is the
       correction, wall_mom the wall force on the fluid, so the residual
       should be roundoff)
  (2) F_cv = import - (dP_ext + dP_cov)  ==  -(wall_mom + dP_core) + O(res)
      i.e. the CV measures the sum of the leaf-BFL force and the frozen-core
      force on the body.
"""
import argparse
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.control_volume_force import fluid_momentum, streaming_momentum_import
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
from tensorlbm.octree_boundary.geometry import build_octree_shell
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
p.add_argument("--reynolds", type=float, default=100.0)
p.add_argument("--lattice-speed", type=float, default=0.06)
p.add_argument("--steps", type=int, default=120)
p.add_argument("--warmup", type=int, default=60)
p.add_argument("--ramp-steps", type=int, default=200)
p.add_argument("--d-max", type=int, default=1)
p.add_argument("--bl-thickness", type=float, default=3.0)
p.add_argument("--cv-margin", type=int, default=6)
p.add_argument("--report", type=int, default=30)
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
config1 = StaticBlockAMRConfig(
    box1, tau_coarse=tau_coarse, reflux=True, ghost_interpolation="injection")
amr = NestedStaticBlockAMR3D(coarse_f, (config1,), fine_solids=(None,))

phys_center = (float(fc1[0] - GHOST), float(fc1[1] - GHOST), float(fc1[2] - GHOST))
radius_l1 = radius1
octree = build_octree_shell(
    s1, phys_center, radius_l1,
    bl_thickness_cells=args.bl_thickness, d_max=args.d_max,
    transition=1, device=device)
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
    fc1, radius_l1, shell_band, args.cv_margin,
    covered=octree._shell_mask, filter_shell=filter_shell,
    solid=l1_solid_g, device=device)
cv = cv_w[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]

sponge_faces = ("x+", "y-", "y+", "z-", "z+")
sigma = build_sponge_sigma_3d(shape, width=16, max_strength=0.2,
                              device=device, faces=sponge_faces)
dynamic_area_cv = 0.5 * args.lattice_speed ** 2 * math.pi * radius_l1 ** 2
radius_leaf = radius_l1 / dx_leaf
dynamic_area_mem = 0.5 * args.lattice_speed ** 2 * math.pi * radius_leaf ** 2

fluid_in_cv = cv & ~l1_solid_phys
covered_in_cv = covered_phys & fluid_in_cv
exterior_in_cv = fluid_in_cv & ~covered_phys
solid_in_cv = cv & l1_solid_phys
print(f"cv={int(cv.sum())} fluid={int(fluid_in_cv.sum())} "
      f"covered={int(covered_in_cv.sum())} exterior={int(exterior_in_cv.sum())} "
      f"solid_in_cv={int(solid_in_cv.sum())} n_leaf={octree.n_leaf}", flush=True)

l1_posts = []
ledger = ShellForceLedger(octree)
current_step = 0


def advance(f, tau, level, substep):
    if level == 0:
        out, post_collision, _ = root_advance(
            f, tau, solid_coarse_q, sigma, args.lattice_speed,
            collision="cascaded")
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
cxv = c[:, 0].view(Q, 1, 1, 1)


def mx(t, mask):
    """streamwise momentum over a physical-grid mask."""
    return float((t * cxv.to(t.dtype))[:, mask].sum().item())


def leaf_mx(f_leaf):
    return float((f_leaf * c[:, 0].view(Q, 1).to(f_leaf.dtype)).sum().item())


def inv_mx(inv):
    return float((inv * c[:, 0]).sum().item())


stats = {k: [] for k in (
    "import", "dP_ext", "dP_cov", "dP_leaf", "dP_core", "restr_cons",
    "wall_mom", "reflux_app", "closure", "F_cv", "F_body", "F_mem_l1",
)}
for current_step in range(1, args.steps + 1):
    l1_fine = amr.interfaces[0].fine_f
    l1_old_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST].clone()
    l1_posts.clear()
    f_leaf_old = octree.f_leaf.clone()
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
    imported = streaming_momentum_import(
        l1_posts[0] + (l1_posts[1] if len(l1_posts) > 1 else 0), cv)
    mem = ledger.mem_force
    mem_l1 = convert_leaf_force_to_l1(mem, dx_leaf, dt_leaf)
    wall_mom = 2.0 * float(mem_l1[0].item())   # sum over 2 substeps, force on fluid

    dP_ext = mx(l1_f_phys, exterior_in_cv) - mx(l1_old_phys, exterior_in_cv)
    dP_cov = mx(l1_f_phys, covered_in_cv) - mx(l1_old_phys, covered_in_cv)
    dP_core = mx(l1_f_phys, solid_in_cv) - mx(l1_old_phys, solid_in_cv)
    dP_leaf = leaf_mx(octree.f_leaf) - leaf_mx(f_leaf_old)
    restr_cons = dP_cov - dP_leaf
    reflux_app = inv_mx(shell_ledger.applied_shell_correction)

    ft = octree.meta.get("last_fine_transfer")
    fine_net_x = inv_mx(ft.net_outgoing) if ft is not None else 0.0
    coarse_net_x = inv_mx(shell_ledger.raw_kinetic_mismatch) if (
        shell_ledger.raw_kinetic_mismatch is not None) else 0.0
    coarse_net_x = fine_net_x - coarse_net_x

    F_cv = float(imported[0].item()) - (dP_ext + dP_cov)
    # expected body force = -(force on fluid) = -(wall + core)
    F_body = -(wall_mom + dP_core)
    closure = (
        float(imported[0].item()) + dP_ext + dP_cov + dP_core
        + wall_mom + reflux_app
    )

    for k, v in (
        ("import", float(imported[0].item())), ("dP_ext", dP_ext),
        ("dP_cov", dP_cov), ("dP_leaf", dP_leaf), ("dP_core", dP_core),
        ("restr_cons", restr_cons), ("wall_mom", wall_mom),
        ("reflux_app", reflux_app), ("closure", closure), ("F_cv", F_cv),
        ("F_body", F_body), ("F_mem_l1", float(mem_l1[0].item())),
    ):
        stats[k].append(v)
    ledger.reset()

    if current_step % args.report == 0 or current_step == 1:
        tail = slice(max(0, len(stats["F_cv"]) - args.warmup), len(stats["F_cv"]))
        if tail.start == tail.stop:
            tail = slice(-1, None)
        avg = {k: sum(stats[k][tail]) / len(stats[k][tail]) for k in stats}
        print(
            f"step={current_step:4d} import={avg['import']:+.3f} "
            f"dP_ext={avg['dP_ext']:+.3f} dP_cov={avg['dP_cov']:+.3f} "
            f"dP_leaf={avg['dP_leaf']:+.3f} dP_core={avg['dP_core']:+.3f} "
            f"restr_cons={avg['restr_cons']:+.3f} wall_mom={avg['wall_mom']:+.3f} "
            f"reflux_app={avg['reflux_app']:+.3f} closure={avg['closure']:+.3f}",
            flush=True,
        )
        print(
            f"          F_cv={avg['F_cv']:+.3f} (Cd {avg['F_cv']/dynamic_area_cv:+.3f}) "
            f"F_body=wall+core={avg['F_body']:+.3f} (Cd {avg['F_body']/dynamic_area_cv:+.3f}) "
            f"F_mem_l1={avg['F_mem_l1']:+.3f} (Cd {avg['F_mem_l1']/dynamic_area_cv:+.3f}) "
            f"fine_net={fine_net_x:+.3f} coarse_net={coarse_net_x:+.3f}",
            flush=True,
        )

print("\n=== tail averages ===", flush=True)
tail = slice(max(0, len(stats["F_cv"]) - args.warmup), len(stats["F_cv"]))
avg = {k: sum(stats[k][tail]) / len(stats[k][tail]) for k in stats}
for k in stats:
    print(f"  {k:10s} = {avg[k]:+.6f}")
print(f"  Cd_cv   = {avg['F_cv']/dynamic_area_cv:+.6f}")
print(f"  Cd_body = {avg['F_body']/dynamic_area_cv:+.6f}")
print(f"  Cd_mem  = {avg['F_mem_l1']/dynamic_area_cv:+.6f}")
