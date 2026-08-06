#!/usr/bin/env python3
"""Decompose the CV force: import, change, restriction delta, reflux delta, MEM.

Per root step we record:
  import     : streaming momentum import through the CV faces (sum over the
               L1 post-collision states of all `ratio` substeps)
  dP_old     : p(cv) after L1 advance  - p(cv) at root-step start  (pure L1
               collision+stream change; restriction/reflux NOT yet applied)
  dP_restr   : covered-cell momentum delta caused by the shell restriction
               (in-place rewrite of the covered L1 cells)
  dP_reflux  : exterior-cell momentum delta caused by the face-local reflux
               correction (in-place rewrite)
  dP_shell   : dP_restr + dP_reflux  (total in-place shell rewrite)
  F_cv_full  : import - (dP_old + dP_shell)        -- current observer
  F_cv_clean : import - dP_old                     -- L1-advance-only balance
  F_mem      : leaf MEM force converted to L1 units
  reflux momentum actually applied (from report.applied_inventory_correction)
  fine/coarse net transfer momentum (streamwise)
"""
import argparse
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.control_volume_force import fluid_momentum, streaming_momentum_import
from tensorlbm.cumulant import collide_cumulant_d3q19
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
p.add_argument("--steps", type=int, default=400)
p.add_argument("--warmup", type=int, default=60)
p.add_argument("--ramp-steps", type=int, default=200)
p.add_argument("--d-max", type=int, default=1)
p.add_argument("--bl-thickness", type=float, default=3.0)
p.add_argument("--cv-margin", type=int, default=6)
p.add_argument("--shell-margin", type=int, default=6)
p.add_argument("--wake-cells", type=int, default=32)
p.add_argument("--wall-margin", type=int, default=8)
p.add_argument("--collision", choices=("cumulant", "cascaded"), default="cumulant")
p.add_argument("--report", type=int, default=20)
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
plan = plan_body_shell_box(
    solid_coarse, args.shell_margin, args.wake_cells, pad=args.wall_margin)
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

# region masks on the physical L1 grid
fluid_in_cv = cv & ~l1_solid_phys
covered_in_cv = covered_phys & fluid_in_cv          # restriction target
exterior_in_cv = fluid_in_cv & ~covered_phys        # reflux target
print(f"cv cells={int(cv.sum())} fluid={int(fluid_in_cv.sum())} "
      f"covered_in_cv={int(covered_in_cv.sum())} exterior_in_cv={int(exterior_in_cv.sum())} "
      f"covered_total={int(covered_phys.sum())} n_leaf={octree.n_leaf}",
      flush=True)

l1_posts = []
ledger = ShellForceLedger(octree)
current_step = 0


def advance(f, tau, level, substep):
    if level == 0:
        out, post_collision, _ = root_advance(
            f, tau, solid_coarse_q, sigma, args.lattice_speed,
            collision=args.collision)
        return AMRAdvanceResult(out, post_collision)
    if level == 1:
        before = f
        if args.collision == "cascaded":
            from tensorlbm.cascaded_collision import collide_cascaded_d3q19
            collided = collide_cascaded_d3q19(f, tau)
        else:
            collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
        post = torch.where(l1_solid_q, before, collided)
        out = stream3d(post)
        l1_posts.append(post[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST])
        return AMRAdvanceResult(out, post)
    raise ValueError(f"unexpected level {level}")


def shell_advance(f, tau, level, substep):
    if args.collision == "cascaded":
        from tensorlbm.cascaded_collision import collide_cascaded_d3q19
        collided = collide_cascaded_d3q19(f.view(Q, 1, 1, -1), tau)
    else:
        collided = collide_cumulant_d3q19(f.view(Q, 1, 1, -1), tau, C_s=0.0)
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


def mom(t, mask):
    """streamwise momentum of populations over a physical mask."""
    return float(fluid_momentum(t, mask, solid=l1_solid_phys)[0].item())


def mom_region(t, region):
    """streamwise momentum over a physical region mask (all 19 directions)."""
    return float(
        (t * C.to(device=t.device, dtype=t.dtype)[:, 0].view(Q, 1, 1, 1))[:, region].sum().item()
    )


def inventory_momentum_x(inv):
    c = C.to(device=inv.device, dtype=inv.dtype)
    return float((inv * c[:, 0]).sum().item())

stats = {k: [] for k in (
    "import", "dP_old", "dP_restr", "dP_reflux", "dP_shell",
    "F_full", "F_clean", "F_mem_l1", "ref_applied_x", "fine_net_x", "coarse_net_x",
)}
for current_step in range(1, args.steps + 1):
    l1_fine = amr.interfaces[0].fine_f
    l1_old_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST].clone()
    l1_posts.clear()
    amr.step(advance)
    l1_fine = amr.interfaces[0].fine_f
    l1_f_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
    p_old = mom(l1_old_phys, cv)
    # state right after the L1 advance, before the shell rewrites
    p_pre_shell = mom(l1_f_phys, cv)
    cov_pre = mom_region(l1_f_phys, covered_in_cv)
    ext_pre = mom_region(l1_f_phys, exterior_in_cv)

    shell_ledger = step_octree_shell(
        octree, shell_advance, l1_old_phys, l1_f_phys,
        tau_coarse=config1.tau_fine,
        l1_post=l1_posts if config1.reflux else None,
        shell_level=1, ghost_plan=ghost_plan,
        bfl_fn=bfl_callback, force_ledger=ledger,
    )
    p_post_shell = mom(l1_f_phys, cv)
    cov_post = mom_region(l1_f_phys, covered_in_cv)
    ext_post = mom_region(l1_f_phys, exterior_in_cv)

    imported = streaming_momentum_import(
        l1_posts[0] + (l1_posts[1] if len(l1_posts) > 1 else 0), cv)
    mem = ledger.mem_force
    mem_l1 = convert_leaf_force_to_l1(mem, dx_leaf, dt_leaf)

    dP_old = p_pre_shell - p_old
    dP_restr = cov_post - cov_pre
    dP_reflux = ext_post - ext_pre
    dP_shell = dP_restr + dP_reflux
    F_full = float(imported[0].item()) - (dP_old + dP_shell)
    F_clean = float(imported[0].item()) - dP_old

    ft = octree.meta.get("last_fine_transfer")
    fine_net_x = inventory_momentum_x(ft.net_outgoing) if ft is not None else 0.0
    coarse_net_x = inventory_momentum_x(
        shell_ledger.raw_kinetic_mismatch) if shell_ledger is not None else 0.0
    # raw_kinetic_mismatch is fine - coarse; coarse = fine - mismatch
    coarse_net_x = fine_net_x - coarse_net_x
    ref_applied_x = inventory_momentum_x(shell_ledger.applied_shell_correction)

    stats["import"].append(float(imported[0].item()))
    stats["dP_old"].append(dP_old)
    stats["dP_restr"].append(dP_restr)
    stats["dP_reflux"].append(dP_reflux)
    stats["dP_shell"].append(dP_shell)
    stats["F_full"].append(F_full)
    stats["F_clean"].append(F_clean)
    stats["F_mem_l1"].append(float(mem_l1[0].item()))
    stats["ref_applied_x"].append(ref_applied_x)
    stats["fine_net_x"].append(fine_net_x)
    stats["coarse_net_x"].append(coarse_net_x)
    ledger.reset()

    if current_step % args.report == 0 or current_step == 1:
        cd = lambda f_: f_ / dynamic_area_cv
        n = max(1, len(stats["F_full"]) - args.warmup)
        tail = slice(max(0, len(stats["F_full"]) - n), len(stats["F_full"]))
        avg = {k: sum(stats[k][tail]) / len(stats[k][tail]) for k in stats}
        print(
            f"step={current_step:4d} "
            f"Cd_full={cd(avg['F_full']):+.4f} Cd_clean={cd(avg['F_clean']):+.4f} "
            f"Cd_mem={avg['F_mem_l1']/dynamic_area_cv:+.4f} "
            f"import={avg['import']:+.4f} dP_old={avg['dP_old']:+.4f} "
            f"dP_restr={avg['dP_restr']:+.4f} dP_reflux={avg['dP_reflux']:+.4f} "
            f"ref_app={avg['ref_applied_x']:+.4f} "
            f"fine_net={avg['fine_net_x']:+.4f} coarse_net={avg['coarse_net_x']:+.4f} "
            f"ref_res={shell_ledger.mass_residual:.2e}",
            flush=True,
        )

print("\n=== averages over post-warmup tail ===", flush=True)
n = max(1, len(stats["F_full"]) - args.warmup)
tail = slice(max(0, len(stats["F_full"]) - n), len(stats["F_full"]))
avg = {k: sum(stats[k][tail]) / len(stats[k][tail]) for k in stats}
for k in ("import", "dP_old", "dP_restr", "dP_reflux", "dP_shell",
          "F_full", "F_clean", "F_mem_l1", "ref_applied_x",
          "fine_net_x", "coarse_net_x"):
    print(f"  {k:12s} = {avg[k]:+.6f}")
print(f"  Cd_full  = {avg['F_full']/dynamic_area_cv:+.6f}")
print(f"  Cd_clean = {avg['F_clean']/dynamic_area_cv:+.6f}")
print(f"  Cd_mem   = {avg['F_mem_l1']/dynamic_area_cv:+.6f}")
