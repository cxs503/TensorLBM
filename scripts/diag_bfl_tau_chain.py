#!/usr/bin/env python3
"""Per-link BFL force decomposition by q-segment, R6 vs R8 (tau-chain focus).

Runs the hybrid octree sphere to quasi-steady state (short), then on the last
root step decomposes the BFL link momentum exchange:

  * q segments: [0,0.3) [0.3,0.5) [0.5,0.7) [0.7,1.0]
  * lin vs quad branch totals
  * upstream donor type (leaf/ghost/solid/fanout) per segment
  * q histogram density near q=0.5 (the lin/quad switch)

Also probes the tau dependence of the reconstruction: for every link it
recomputes f_bc under the hypothesis that the *leaf* non-equilibrium part
should carry an extra factor (tau_leaf-0.5) relative to the interpolated
baseline — i.e. the sensitivity dF/d(tau_f-0.5) — and reports the implied
force change if the shell ran at tau_coarse instead of tau_leaf.
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
p.add_argument("--shell-tau-override", type=float, default=None,
               help="Force the shell relaxation time (diagnostic). "
                    "None = production convective chain.")
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
        tau_shell_override=args.shell_tau_override,
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
tau_shell_used = (args.shell_tau_override if args.shell_tau_override is not None
                  else 0.5 + 2.0 * (config1.tau_fine - 0.5))
print(f"R{int(args.radius)}: tau_c={tau_coarse:.4f} tau_L1={config1.tau_fine:.4f} "
      f"tau_shell={tau_shell_used:.4f} "
      f"Cd_mem={cd_mem:.4f} Cd_cv={cv_mean/dynamic_area_cv:.4f}", flush=True)

# ---- per-link MEM decomposition on the current leaf state ------------------
f_prev = octree.f_leaf
mask = octree.bfl_mask
q_field = octree.q_field
nt = octree.neighbor_table
opp = octree._opp.to(device)
n_link = int(mask.sum().item())

seg_bounds = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
seg = {b: {"n": 0, "fx": 0.0, "fx_abs": 0.0, "qsum": 0.0,
           "up_leaf": 0, "up_ghost": 0, "up_solid": 0}
       for b in seg_bounds}
n_lin = n_quad = 0
fx_lin = fx_quad = 0.0
n_near_half = 0          # |q-0.5| < 0.01
fx_near_half = 0.0
all_q = []
# tau probe: sensitivity of f_bc to (tau_leaf-0.5) via the non-equilibrium
# part, evaluated as |f_bc(tau_leaf) - f_bc(tau_coarse)| under the hypothesis
# that the leaf non-equilibrium scales as (tau-0.5) / (tau_ref-0.5).
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
    fp_up = torch.zeros_like(fp_d)
    valid = up >= 0
    if bool(valid.any()):
        fp_up[valid] = f_prev[d, up[valid]].to(torch.float64)
    ghost = up == SHELL_OUTSIDE
    # NOTE: ghost values not stored here; use fp_d as proxy for the
    # upstream-ghost links (error only in the up_ghost split of fp_up)
    if bool(ghost.any()):
        fp_up[ghost] = fp_d[ghost]
    lin = qq < 0.5
    n_lin += int(lin.sum().item())
    n_quad += int((~lin).sum().item())
    f_bc_lin = 2.0 * qq * fp_d + (1.0 - 2.0 * qq) * fp_up
    safe_q = torch.where(lin, torch.ones_like(qq), qq)
    f_bc_quad = fp_d / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
    f_bc = torch.where(lin, f_bc_lin, f_bc_quad)
    exchange = fp_d + f_bc
    fx = exchange * float(c[d, 0].item())
    fx_lin += float(fx[lin].sum().item())
    fx_quad += float(fx[~lin].sum().item())
    for b in seg_bounds:
        sel = (qq >= b[0]) & (qq < b[1])
        if bool(sel.any()):
            s = seg[b]
            s["n"] += int(sel.sum().item())
            s["fx"] += float(fx[sel].sum().item())
            s["fx_abs"] += float(fx[sel].abs().sum().item())
            s["qsum"] += float(qq[sel].sum().item())
            s["up_leaf"] += int((valid & sel).sum().item())
            s["up_ghost"] += int((ghost & sel).sum().item())
            s["up_solid"] += int(((up == SOLID) & sel).sum().item())
    near = (qq - 0.5).abs() < 0.01
    n_near_half += int(near.sum().item())
    fx_near_half += float(fx[near].sum().item())
    all_q.append(qq.detach().cpu())

all_q = torch.cat(all_q)
print(f"n_bfl_links={n_link}  q_mean={all_q.mean():.3f} "
      f"lin={n_lin} ({100*n_lin/n_link:.1f}%) quad={n_quad} ({100*n_quad/n_link:.1f}%)",
      flush=True)
print(f"branch fx: lin={fx_lin:+.4f} ({100*fx_lin/max(abs(fx_lin)+abs(fx_quad),1e-30):.1f}% of |fx|) "
      f"quad={fx_quad:+.4f}", flush=True)
print(f"links with |q-0.5|<0.01: {n_near_half} ({100*n_near_half/n_link:.2f}%) "
      f"fx={fx_near_half:+.4f}", flush=True)
print(f"{'segment':<12}{'n':>8}{'fx':>12}{'per-link':>12}{'q_mean':>8}{'up_leaf':>9}{'up_ghost':>10}{'up_solid':>10}", flush=True)
for b in seg_bounds:
    s = seg[b]
    pl = s["fx"] / max(s["n"], 1)
    print(f"[{b[0]:.1f},{b[1]:.1f})  {s['n']:>8}{s['fx']:>12.4f}{pl:>12.6f}"
          f"{s['qsum']/max(s['n'],1):>8.3f}{s['up_leaf']:>9}{s['up_ghost']:>10}{s['up_solid']:>10}", flush=True)
# q histogram around 0.5
for lo, hi in ((0.40, 0.45), (0.45, 0.48), (0.48, 0.50), (0.50, 0.52), (0.52, 0.55), (0.55, 0.60)):
    n = int(((all_q >= lo) & (all_q < hi)).sum().item())
    print(f"  q in [{lo:.2f},{hi:.2f}): {n} ({100*n/n_link:.2f}%)", flush=True)

# ---- near-wall flow: per-link force vs polar angle, leaf tangential u ------
# polar angle theta from the +x axis (streamwise), cos_theta = x/r
from tensorlbm.d3q19 import macroscopic3d
leaf_rho, leaf_ux, leaf_uy, leaf_uz = macroscopic3d(
    f_prev.view(Q, 1, 1, -1))
leaf_ux = leaf_ux.view(-1).to(torch.float64)
leaf_uy = leaf_uy.view(-1).to(torch.float64)
leaf_uz = leaf_uz.view(-1).to(torch.float64)
center = torch.tensor(phys_center, dtype=torch.float64, device=device)
centers64 = (octree._l1_coords.to(torch.float64) + 0.5) / (
    2.0 ** octree.leaf_level.to(torch.float64))[:, None]
r_vec = centers64 - center
r_norm = r_vec.norm(dim=1).clamp_min(1e-12)
cos_theta = (r_vec[:, 0] / r_norm).clamp(-1.0, 1.0)
# tangential velocity magnitude at the boundary leaves (u - (u.r_hat)r_hat)
u_dot_r = (leaf_ux * r_vec[:, 0] + leaf_uy * r_vec[:, 1]
           + leaf_uz * r_vec[:, 2]) / r_norm
u_t = torch.sqrt(
    (leaf_ux - u_dot_r * r_vec[:, 0] / r_norm) ** 2
    + (leaf_uy - u_dot_r * r_vec[:, 1] / r_norm) ** 2
    + (leaf_uz - u_dot_r * r_vec[:, 2] / r_norm) ** 2
)
per_leaf = mask.sum(dim=0)
bfl_leaves = per_leaf > 0
print("near-wall flow (boundary leaves):", flush=True)
print(f"  n_bfl_leaves={int(bfl_leaves.sum().item())} "
      f"mean|u_t|={u_t[bfl_leaves].mean().item():.5f} "
      f"mean|u|={torch.sqrt(leaf_ux**2+leaf_uy**2+leaf_uz**2)[bfl_leaves].mean().item():.5f} "
      f"(U0={0.06})", flush=True)
for lo, hi in ((-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2), (0.2, 0.6), (0.6, 1.0)):
    sel = bfl_leaves & (cos_theta >= lo) & (cos_theta < hi)
    if not bool(sel.any()):
        continue
    print(f"  cos_theta in [{lo:+.1f},{hi:+.1f}): n={int(sel.sum().item())} "
          f"mean_u_t={u_t[sel].mean().item():.5f} "
          f"mean_u_x={leaf_ux[sel].mean().item():+.5f}", flush=True)
# per-link force vs cos_theta (drag distribution)
fx_vs_ct = torch.zeros(5, dtype=torch.float64)
n_vs_ct = torch.zeros(5, dtype=torch.int64)
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
    fp_up = torch.zeros_like(fp_d)
    valid = up >= 0
    if bool(valid.any()):
        fp_up[valid] = f_prev[d, up[valid]].to(torch.float64)
    gh = up == SHELL_OUTSIDE
    if bool(gh.any()):
        fp_up[gh] = fp_d[gh]
    lin = qq < 0.5
    f_lin = 2.0 * qq * fp_d + (1.0 - 2.0 * qq) * fp_up
    safe_q = torch.where(lin, torch.ones_like(qq), qq)
    f_quad = fp_d / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp
    f_bc = torch.where(lin, f_lin, f_quad)
    fx = (fp_d + f_bc) * float(c[d, 0].item())
    ct = cos_theta[idx]
    for k, (lo, hi) in enumerate(((-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2),
                                  (0.2, 0.6), (0.6, 1.0))):
        sel = (ct >= lo) & (ct < hi)
        fx_vs_ct[k] += float(fx[sel].sum().item())
        n_vs_ct[k] += int(sel.sum().item())
print("per-link drag force vs cos_theta (leaf units):", flush=True)
for k, (lo, hi) in enumerate(((-1.0, -0.6), (-0.6, -0.2), (-0.2, 0.2),
                              (0.2, 0.6), (0.6, 1.0))):
    print(f"  cos_theta in [{lo:+.1f},{hi:+.1f}): n={int(n_vs_ct[k].item())} "
          f"fx={fx_vs_ct[k].item():+.4f} per-link={fx_vs_ct[k].item()/max(int(n_vs_ct[k].item()),1):+.6f}",
          flush=True)
