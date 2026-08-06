#!/usr/bin/env python3
"""Probe the ghost-feed at the shell OUTER boundary in the REAL coupled run.

Runs the full hybrid validation setup (L0 + L1 + octree shell with BFL +
restriction + reflux) for a short warmup, then reports, at the ghost-cell
positions of the shell's interface links:

  * the L1 field the ghost TRILINEAR-SAMPLES (u at the ghost position),
  * the ghost population the shell actually INJECTS (post-collision value,
    i.e. after the tau_f/(2 tau_c) * (1 - 1/tau_f) neq rescale),
  * the shell leaves' own velocity at the same radius,

split by radial band (near-wall / mid / outer) and by cos_theta, for R6 and
R8.  If the outer-boundary ghosts sample an under-driven L1 field (u << U0),
or the injected ghost velocity is systematically below the L1 sample, the
shell's outer drive is weak -> uniform drag deficit.
"""
import argparse
import math

import torch

from tensorlbm.amr_shell_planning import plan_body_shell_box
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.octree_boundary.bfl import (
    bfl_apply_gather, bfl_ramp_wall_velocity, leaf_force_weights,
)
from tensorlbm.octree_boundary.force import ShellForceLedger
from tensorlbm.octree_boundary.geometry import build_octree_shell
from tensorlbm.octree_boundary.stepping import (
    build_ghost_plan, fill_ghost, step_octree_shell,
)
from tensorlbm.solver3d import stream3d
from tensorlbm.sphere_amr_common import (
    build_fine_block_geometry, build_sphere_geometry, root_advance,
)
from tensorlbm.sponge_layer import build_sponge_sigma_3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult, NestedStaticBlockAMR3D, StaticBlockAMRConfig,
)

Q = 19
GHOST = 1
RATIO = 2


def run_probe(radius, nx, ny, nz, bl=3.0, steps=500, warmup=250, ramp=200):
    device = torch.device("cuda:0")
    shape = (nz, ny, nx)
    cx, cy, cz = nx * 0.5, ny / 2.0, nz / 2.0
    U0 = 0.06
    nu = U0 * (2.0 * radius) / 100.0
    tau_coarse = 0.5 + 3.0 * nu
    solid_coarse, solid_coarse_q = build_sphere_geometry(
        nx, ny, nz, cx, cy, cz, radius, device)
    plan = plan_body_shell_box(solid_coarse, 6, 32, pad=8)
    box1 = plan.box
    rho = torch.ones(shape, device=device)
    ux = torch.full_like(rho, U0)
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
    octree = build_octree_shell(s1, phys_center, radius1,
                                bl_thickness_cells=bl, d_max=1, transition=1,
                                device=device)
    host = octree.leaf_host_cell
    l1_fine = amr.interfaces[0].fine_f
    octree.f_leaf = l1_fine[
        :, host[:, 0] + GHOST, host[:, 1] + GHOST, host[:, 2] + GHOST
    ].clone()
    leaf_weights = leaf_force_weights(octree)
    ghost_plan = build_ghost_plan(octree, s1, solid_fallback=True)
    dx_leaf = 2.0 ** (-octree.d_max)
    dt_leaf = dx_leaf

    assert octree._solid is not None
    l1_solid_phys = octree._solid
    l1_solid_g = torch.zeros(
        (nz1 + 2 * GHOST, ny1 + 2 * GHOST, nx1 + 2 * GHOST),
        dtype=torch.bool, device=device)
    l1_solid_g[GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST] = l1_solid_phys
    l1_solid_q = l1_solid_g.unsqueeze(0).expand(Q, *l1_solid_g.shape).contiguous()

    sponge_faces = ("x+", "y-", "y+", "z-", "z+")
    sigma = build_sponge_sigma_3d(shape, width=16, max_strength=0.2,
                                  device=device, faces=sponge_faces)
    ledger = ShellForceLedger(octree)
    l1_posts = []
    current_step = 0

    def advance(f, tau, level, substep):
        nonlocal current_step
        if level == 0:
            out, post, _ = root_advance(
                f, tau, solid_coarse_q, sigma, U0, collision="cumulant")
            return AMRAdvanceResult(out, post)
        if level == 1:
            before = f
            collided = collide_cumulant_d3q19(f, tau, C_s=0.0)
            post = torch.where(l1_solid_q, before, collided)
            out = stream3d(post)
            l1_posts.append(post[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST])
            return AMRAdvanceResult(out, post)
        raise ValueError

    def shell_advance(f, tau, level, substep):
        collided = collide_cumulant_d3q19(f.view(Q, 1, 1, -1), tau, C_s=0.0)
        post = collided.view_as(f)
        return AMRAdvanceResult(post.clone(), post)

    def bfl_callback(octree_, out, post, ghost_plan_, ghost_vals, *, substep):
        rho_w, uwx, uwy, uwz = bfl_ramp_wall_velocity(
            octree_, post, current_step, ramp)
        return bfl_apply_gather(
            octree_, out, post,
            ghost_plan=ghost_plan_, ghost_vals=ghost_vals,
            wall_velocity=(uwx, uwy, uwz), wall_density=rho_w,
            force_weights=leaf_weights, return_force=True, q_min=None)

    for current_step in range(1, steps + 1):
        l1_fine = amr.interfaces[0].fine_f
        l1_old_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST].clone()
        l1_posts.clear()
        amr.step(advance)
        l1_fine = amr.interfaces[0].fine_f
        l1_f_phys = l1_fine[:, GHOST:-GHOST, GHOST:-GHOST, GHOST:-GHOST]
        step_octree_shell(
            octree, shell_advance, l1_old_phys, l1_f_phys,
            tau_coarse=config1.tau_fine,
            l1_post=l1_posts if config1.reflux else None,
            shell_level=1, ghost_plan=ghost_plan,
            bfl_fn=bfl_callback, force_ledger=ledger)
        ledger.reset()

    # ------------------------------------------------------------------
    # Diagnostics: sample the L1 field AT the ghost positions and compare
    # with what fill_ghost injects.
    # ------------------------------------------------------------------
    taus = [config1.tau_fine,
            0.5 + 2.0 * (config1.tau_fine - 0.5)]  # convective chain d=1
    tau_shell = taus[1]
    l1_phys = l1_f_phys  # current pre-collision L1 state (ghost source)
    # time-lerp anchors: use current state (steady regime) for the sample
    parent_t = l1_phys
    ghost_vals = fill_ghost(octree, ghost_plan, parent_t, taus)

    center = torch.tensor(phys_center, dtype=torch.float64, device=device)
    leaf = ghost_plan.leaf
    lev = octree.leaf_level[leaf]
    dx = 2.0 ** (-lev.to(torch.float64))
    coords = octree._l1_coords
    centers64 = (coords.to(torch.float64) + 0.5) / (
        2.0 ** octree.leaf_level.to(torch.float64))[:, None]
    r_vec = centers64[leaf] - center
    r_norm = r_vec.norm(dim=1).clamp_min(1e-12)
    dist_wall_dx = (r_norm - radius1) / dx          # leaf dx units
    ct = (r_vec[:, 0] / r_norm).clamp(-1.0, 1.0)

    # ghost position = leaf centre + c_vec[d_link] * dx
    c_vec = octree._c_vec.to(torch.float64)
    d_link = octree.interface_links[:, 1]
    gpos = centers64[leaf] + c_vec[d_link] * dx[:, None]
    gr = (gpos - center).norm(dim=1).clamp_min(1e-12)
    gdist_wall_dx = (gr - radius1) / dx

    # L1 velocity at the ghost position: trilinear from the plan stencil
    rho_l, ux_l, uy_l, uz_l = macroscopic3d(l1_phys)
    wx = ghost_plan.wx.unsqueeze(0)
    wy = ghost_plan.wy.unsqueeze(0)
    wz = ghost_plan.wz.unsqueeze(0)
    gz0, gy0, gx0 = ghost_plan.z0, ghost_plan.y0, ghost_plan.x0
    gz1, gy1, gx1 = ghost_plan.z1, ghost_plan.y1, ghost_plan.x1

    def _tri(u):
        wdt = u.dtype
        wx_c = wx.to(wdt)
        wy_c = wy.to(wdt)
        wz_c = wz.to(wdt)
        v00 = torch.lerp(u[gz0, gy0, gx0], u[gz0, gy0, gx1], wx_c)
        v01 = torch.lerp(u[gz0, gy1, gx0], u[gz0, gy1, gx1], wx_c)
        v10 = torch.lerp(u[gz1, gy0, gx0], u[gz1, gy0, gx1], wx_c)
        v11 = torch.lerp(u[gz1, gy1, gx0], u[gz1, gy1, gx1], wx_c)
        return torch.lerp(torch.lerp(v00, v01, wy_c),
                          torch.lerp(v10, v11, wy_c), wz_c)

    su_x = _tri(ux_l)
    su_y = _tri(uy_l)
    su_z = _tri(uz_l)

    # ghost-injected macroscopic (post-collision populations)
    rg, ugx, ugy, ugz = macroscopic3d(ghost_vals.view(Q, 1, 1, -1))
    ugx, ugy, ugz = ugx.view(-1), ugy.view(-1), ugz.view(-1)

    # tangential components at the ghost position (radial direction from
    # the sphere centre)
    rv = gpos - center
    rn = rv.norm(dim=1).clamp_min(1e-12)
    for name, ux_c, uy_c, uz_c in (
            ("L1_sample", su_x, su_y, su_z),
            ("ghost_inject", ugx, ugy, ugz)):
        ux_c = ux_c.reshape(-1)
        uy_c = uy_c.reshape(-1)
        uz_c = uz_c.reshape(-1)
        u_dot_r = (ux_c * rv[:, 0] + uy_c * rv[:, 1] + uz_c * rv[:, 2]) / rn
        u_t = torch.sqrt(
            (ux_c - u_dot_r * rv[:, 0] / rn) ** 2
            + (uy_c - u_dot_r * rv[:, 1] / rn) ** 2
            + (uz_c - u_dot_r * rv[:, 2] / rn) ** 2)
        print(f"  [{name}] n={int(leaf.shape[0])} "
              f"mean_u_x={ux_c.mean().item():+.5f} "
              f"mean_u_t={u_t.mean().item():.5f} "
              f"mean|u|={torch.sqrt(ux_c**2+uy_c**2+uz_c**2).mean().item():.5f}")
        for lo, hi in ((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 1e9)):
            s2 = (gdist_wall_dx >= lo) & (gdist_wall_dx < hi)
            if bool(s2.any()):
                print(f"      gdist[{lo:.0f},{hi:.0f})dx: n={int(s2.sum())} "
                      f"u_x={ux_c[s2].mean().item():+.5f} "
                      f"u_t={u_t[s2].mean().item():.5f}")

    # leaf tangential velocity profile (shell leaves, pre-stream)
    rho_lf, uxl, uyl, uzl = macroscopic3d(
        octree.f_leaf.view(Q, 1, 1, -1))
    uxl, uyl, uzl = uxl.view(-1), uyl.view(-1), uzl.view(-1)
    rv_l = centers64 - center
    rn_l = rv_l.norm(dim=1).clamp_min(1e-12)
    u_dot_r_l = (uxl * rv_l[:, 0] + uyl * rv_l[:, 1] + uzl * rv_l[:, 2]) / rn_l
    u_t_l = torch.sqrt(
        (uxl - u_dot_r_l * rv_l[:, 0] / rn_l) ** 2
        + (uyl - u_dot_r_l * rv_l[:, 1] / rn_l) ** 2
        + (uzl - u_dot_r_l * rv_l[:, 2] / rn_l) ** 2)
    dl = (rn_l - radius1) / (2.0 ** (-octree.leaf_level.to(torch.float64)))
    print("  leaf u_t profile (leaf dx from wall):")
    for lo, hi in ((0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 4.0),
                   (4.0, 6.0), (6.0, 1e9)):
        s2 = (dl >= lo) & (dl < hi)
        if bool(s2.any()):
            print(f"      dist[{lo:.0f},{hi:.0f})dx: n={int(s2.sum())} "
                  f"u_t={u_t_l[s2].mean().item():.5f} "
                  f"u_x={uxl[s2].mean().item():+.5f}")
    # ghost sample vs inject ratio (stress proxy): neq amplitude
    rho_c = rho_l[gz0, gy0, gx0].squeeze(0)
    ux_c = ux_l[gz0, gy0, gx0].squeeze(0)
    uy_c = uy_l[gz0, gy0, gx0].squeeze(0)
    uz_c = uz_l[gz0, gy0, gx0].squeeze(0)
    feq_sample = equilibrium3d(
        rho_c.view(1, 1, 1, -1), ux_c.view(1, 1, 1, -1),
        uy_c.view(1, 1, 1, -1), uz_c.view(1, 1, 1, -1))
    f_sample_neq = (
        parent_t[:, gz0, gy0, gx0].view(Q, 1, 1, -1) - feq_sample
    )
    feq_ghost = equilibrium3d(
        rg.view(1, 1, 1, -1), ugx.view(1, 1, 1, -1), ugy.view(1, 1, 1, -1),
        ugz.view(1, 1, 1, -1))
    f_ghost_neq = ghost_vals.view(Q, 1, 1, -1) - feq_ghost
    inj_ratio = (
        f_ghost_neq.norm(dim=0) / f_sample_neq.norm(dim=0).clamp_min(1e-12)
    ).view(-1)
    s_out = gdist_wall_dx >= 4.0
    print(f"  neq|ghost|/neq|L1| (outer, gdist>=4dx): "
          f"mean={inj_ratio[s_out].mean().item():.4f} "
          f"median={inj_ratio[s_out].median().item():.4f} "
          f"(expected {(tau_shell-1.0)/(2.0*taus[0]):.4f} abs, "
          f"tau_L1={taus[0]:.4f} tau_shell={tau_shell:.4f})")
    return dict(tau_l1=taus[0], tau_shell=tau_shell,
                expected_scale=(tau_shell - 1.0) / (2.0 * taus[0]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--radius", type=float, default=8.0)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--bl", type=float, default=3.0)
    a = p.parse_args()
    if a.radius == 8.0:
        nx, ny, nz = 128, 88, 88
    else:
        nx, ny, nz = 96, 64, 64
    print(f"=== R{a.radius} bl={a.bl} steps={a.steps} ===", flush=True)
    info = run_probe(a.radius, nx, ny, nz, bl=a.bl, steps=a.steps)
    print(f"=== R{a.radius} done: {info} ===", flush=True)
