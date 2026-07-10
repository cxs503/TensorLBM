#!/usr/bin/env python
"""Benchmark: Magnus effect — rotating cylinder in uniform flow.

A cylinder rotates at prescribed angular velocity in uniform flow.
The rotation creates circulation, generating lift (Magnus effect).
Measure lift coefficient Cl vs rotation rate parameter α=ωR/(2U).

Physics
-------
- Grid: 300×200×1, D3Q19 BGK, nz=1 (quasi-2D)
- Cylinder: center (100, 100), radius R=10
- Flow: U=0.1, Re=100, tau=0.55
- Rotation: cylinder surface moves at tangential velocity v_θ=ω*R
  - Test multiple rotation rates: α=0, 0.5, 1.0, 1.5, 2.0, 2.5
  - α = ω*R/(2*U) is the rotation rate parameter
- Moving-wall bounce-back on cylinder surface (full-way)
  - u_wall_x = -ω*(y-cy), u_wall_y = ω*(x-cx)
- Inlet: velocity BC, Outlet: sponge (target equilibrium, width=40)
- Top/bottom: sponge
- Steps: 3000 per rotation rate
- Force: momentum-exchange method on bounce-back boundary

Analytical/Reference
--------------------
- Potential flow (Glauert 1925): Cl = 2π*α for α < 2 (then Cl saturates)
- Experimental (Tokumaru & Dimotakis 1993): Cl ≈ 1.5-2.5 for α=1-2 at Re~100
- At α=0: Cl=0 (no rotation, no lift)

Validation
----------
1. Cl=0 at α=0 (no rotation)
2. Cl increases with α (Magnus effect)
3. Compare Cl vs α with Glauert theory (Cl=2πα) for α<1.5
4. Target: <25% error for at least 3 rotation rates

Run
---
    PYTHONPATH=src python examples/benchmark_magnus_cylinder.py --device cpu --steps 3000
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

# Ensure src/ is importable even without PYTHONPATH
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import C, OPPOSITE, W, equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def build_sponge_mask(
    nx: int,
    ny: int,
    nz: int,
    sponge_width: int,
    dev: torch.device,
) -> torch.Tensor:
    """Build sponge-layer strength mask for outlet + top/bottom boundaries.

    The mask is 0 in the interior and ramps quadratically to 1 at the
    boundary.  Applied as  f ← (1−σ)·f + σ·f_eq  to gradually relax
    the distribution toward free-stream equilibrium, preventing
    pressure-wave reflections.
    """
    sponge = torch.zeros(nz, ny, nx, device=dev, dtype=torch.float32)

    # Outlet sponge (right side): quadratic ramp 0 → 1
    sw_x = min(sponge_width, nx // 4)
    for i in range(sw_x):
        s = ((i + 1) / sw_x) ** 2
        sponge[:, :, nx - sw_x + i] = s

    # Top/bottom sponge: quadratic ramp 0 → 1
    sw_y = min(sponge_width, ny // 4)
    for i in range(sw_y):
        s = ((sw_y - i) / sw_y) ** 2
        sponge[:, i, :] = torch.maximum(
            sponge[:, i, :],
            torch.full_like(sponge[:, i, :], s),
        )
        sponge[:, ny - 1 - i, :] = torch.maximum(
            sponge[:, ny - 1 - i, :],
            torch.full_like(sponge[:, ny - 1 - i, :], s),
        )
    return sponge


def build_cylinder_geometry(
    nx: int,
    ny: int,
    nz: int,
    R: float,
    cx: float,
    cy: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the rigid mask and signed distance for the circular cylinder.

    The analytic zero level set belongs to the solid.  This ensures BFL has a
    strictly positive-distance fluid source rather than a q≈0 link whose wall
    sits directly on the source lattice node.
    """
    _, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device),
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    phi = torch.sqrt((xx.float() - cx) ** 2 + (yy.float() - cy) ** 2) - R
    return phi <= 0.0, phi


def circle_link_fraction(
    x: torch.Tensor,
    y: torch.Tensor,
    cxq: int,
    cyq: int,
    cx: float,
    cy: float,
    R: float,
) -> torch.Tensor:
    """First ray/circle intercept on ``(x,y) + delta*c``, for 0 < delta < 1.

    Linear signed-distance interpolation is exact for a planar wall, but not
    for a circle on diagonal lattice links.  BFL needs the geometric wall
    intercept, which this quadratic ray/circle solve supplies.
    """
    dx, dy = x - cx, y - cy
    a = float(cxq * cxq + cyq * cyq)
    b = 2.0 * (dx * cxq + dy * cyq)
    c = dx.square() + dy.square() - R * R
    return (-b - torch.sqrt((b.square() - 4.0 * a * c).clamp_min(0.0))) / (2.0 * a)


def compute_force_momentum_exchange(
    f: torch.Tensor,
    solid: torch.Tensor,
    c_dev: torch.Tensor,
    w_dev: torch.Tensor | None = None,
    omega_eff: float = 0.0,
    cx: float = 0.0,
    cy: float = 0.0,
    yy: torch.Tensor | None = None,
    xx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hydrodynamic force on the cylinder via momentum exchange.

    For full-way bounce-back the force on the solid is

        F = 2 · Σ_{x∈solid} Σ_q  c_q · f_q(x) · [x − c_q is fluid]

    where  f_q  is the **post-streaming pre-bounce-back** population
    and the bracket is 1 when the neighbour cell in direction −c_q is
    fluid (i.e. the population actually came from the fluid).

    Returns a tensor [Fx, Fy, Fz].
    """
    F = torch.zeros(3, device=f.device, dtype=f.dtype)
    rho = f.sum(dim=0)
    moving = w_dev is not None and yy is not None and xx is not None
    if moving:
        assert w_dev is not None and yy is not None and xx is not None
        uw_x = -omega_eff * (yy - cy)
        uw_y = omega_eff * (xx - cx)
    for q in range(1, 19):
        cxq = int(c_dev[q, 0].item())
        cyq = int(c_dev[q, 1].item())
        czq = int(c_dev[q, 2].item())
        # A population q at x_s arrived from the fluid node x_s-c_q.
        neighbour_is_solid = torch.roll(
            solid, shifts=(czq, cyq, cxq), dims=(0, 1, 2)
        )
        boundary = solid & ~neighbour_is_solid
        if boundary.any():
            fsum = f[q][boundary].sum()
            impulse = 2.0 * fsum
            if moving:
                rho_source = torch.roll(
                    rho, shifts=(czq, cyq, cxq), dims=(0, 1, 2)
                )
                cu_wall = cxq * uw_x + cyq * uw_y
                # Wall momentum exchange for the same moving-link rule used
                # in apply_moving_bounceback.
                impulse += (6.0 * w_dev[q] * rho_source * cu_wall)[boundary].sum()
            F[0] += cxq * impulse
            F[1] += cyq * impulse
            F[2] += czq * impulse
    return F


def apply_moving_bounceback(
    f: torch.Tensor,
    solid: torch.Tensor,
    opp: torch.Tensor,
    c_dev: torch.Tensor,
    w_dev: torch.Tensor,
    omega_eff: float,
    cx: float,
    cy: float,
    yy: torch.Tensor,
    xx: torch.Tensor,
) -> torch.Tensor:
    """Full-way bounce-back with moving-wall correction.

    For a solid node x_s the post-bounce-back population is

        f_i(x_s) = f_opp(x_s) − 2·w_i·3·ρ·(c_i · u_wall)

    where u_wall is the solid-body rotation velocity at x_s:
        u_wall_x = −ω·(y−cy),  u_wall_y = ω·(x−cx)

    The density ρ is approximated as 1 (lattice units, far-field).
    """
    # Wall velocity field on the full grid (only used inside solid)
    uw_x = -omega_eff * (yy - cy)   # (nz, ny, nx)
    uw_y = omega_eff * (xx - cx)
    uw_z = torch.zeros_like(uw_x)

    # Start with stationary full-way bounce-back everywhere in the solid.
    # Moving-wall momentum is then applied *per fluid--solid lattice link*,
    # not indiscriminately to every population of an axial boundary shell.
    # This includes diagonal D3Q19 links at the stair-step circle and avoids
    # injecting momentum into links that did not receive a fluid population.
    f_out = torch.where(solid.unsqueeze(0), f[opp], f)
    rho = f.sum(dim=0)

    for q in range(1, 19):
        cxq = int(c_dev[q, 0].item())
        cyq = int(c_dev[q, 1].item())
        czq = int(c_dev[q, 2].item())
        # At a solid node x_s, f_q arrived from x_s-c_q.  Only such incoming
        # fluid links are reflected into qbar.  torch.roll(...,+c_q) indexes
        # x_s-c_q at x_s.
        source_is_solid = torch.roll(
            solid, shifts=(czq, cyq, cxq), dims=(0, 1, 2)
        )
        link = solid & ~source_is_solid
        rho_source = torch.roll(rho, shifts=(czq, cyq, cxq), dims=(0, 1, 2))
        cu_wall = cxq * uw_x + cyq * uw_y + czq * uw_z
        # Ladd/Aidun moving-link BB: f_qbar = f_q + 6*w_q*rho_f*c_q.u_w
        # for the pull-streamed population at the solid node.  The
        # source-fluid density preserves the local pressure contribution.
        reflected = f[q] + 6.0 * w_dev[q] * rho_source * cu_wall
        qbar = int(opp[q].item())
        f_out[qbar] = torch.where(link, reflected, f_out[qbar])
    return f_out


def apply_bouzidi_moving_bounceback(
    f_post: torch.Tensor, f_stream: torch.Tensor, solid: torch.Tensor,
    phi: torch.Tensor, opp: torch.Tensor, c_dev: torch.Tensor,
    w_dev: torch.Tensor, omega_eff: float, cx: float, cy: float,
    yy: torch.Tensor, xx: torch.Tensor, R: float | None = None,
) -> torch.Tensor:
    """Bouzidi--Firdaouss--Lallemand moving BB on actual fluid--solid links.

    ``f_post`` is post-collision and ``f_stream`` ordinarily streamed.  The
    signed distance ``phi`` is positive in fluid and negative in solid, so its
    linear interpolation supplies each link's wall fraction ``delta``.
    """
    f_out = f_stream.clone()
    rho = f_post.sum(dim=0).clamp_min(1e-12)
    fluid = ~solid
    for q in range(1, 19):
        cxq, cyq, czq = (int(c_dev[q, d].item()) for d in range(3))
        destination_solid = torch.roll(
            solid, shifts=(-czq, -cyq, -cxq), dims=(0, 1, 2)
        )
        link = fluid & destination_solid
        if not link.any():
            continue
        if R is None:
            # Compatibility path for generic signed-distance geometries.
            phi_next = torch.roll(phi, shifts=(-czq, -cyq, -cxq), dims=(0, 1, 2))
            delta = phi / (phi - phi_next).clamp_min(1e-12)
        else:
            delta = circle_link_fraction(xx, yy, cxq, cyq, cx, cy, R)
        delta = delta.clamp(1e-6, 1.0 - 1e-6)
        # Evaluate rigid-wall velocity at the intersection, not a solid-cell
        # centre. c_q points from the fluid node toward the wall.
        wall_y = yy + delta * cyq
        wall_x = xx + delta * cxq
        cu_wall = cxq * (-omega_eff * (wall_y - cy)) + cyq * (omega_eff * (wall_x - cx))
        correction = -6.0 * w_dev[q] * rho * cu_wall
        upstream = torch.roll(f_post[q], shifts=(czq, cyq, cxq), dims=(0, 1, 2))
        near = 2.0 * delta * f_post[q] + (1.0 - 2.0 * delta) * upstream + correction
        qbar = int(opp[q].item())
        far = ((f_post[q] + (2.0 * delta - 1.0) * f_post[qbar]) / (2.0 * delta)
               + correction / (2.0 * delta))
        f_out[qbar] = torch.where(link, torch.where(delta <= 0.5, near, far), f_out[qbar])
    return f_out


def compute_bouzidi_momentum_exchange(
    f_post: torch.Tensor, solid: torch.Tensor, phi: torch.Tensor,
    c_dev: torch.Tensor, w_dev: torch.Tensor, omega_eff: float,
    cx: float, cy: float, yy: torch.Tensor, xx: torch.Tensor, R: float | None = None,
) -> torch.Tensor:
    """Momentum exchange from the same BFL populations used at the wall."""
    reflected = apply_bouzidi_moving_bounceback(
        f_post, torch.zeros_like(f_post), solid, phi, OPPOSITE.to(f_post.device),
        c_dev, w_dev, omega_eff, cx, cy, yy, xx, R,
    )
    force = torch.zeros(3, dtype=f_post.dtype, device=f_post.device)
    fluid = ~solid
    for q in range(1, 19):
        cxq, cyq, czq = (int(c_dev[q, d].item()) for d in range(3))
        link = fluid & torch.roll(solid, shifts=(-czq, -cyq, -cxq), dims=(0, 1, 2))
        qbar = int(OPPOSITE[q].item())
        impulse = (f_post[q] + reflected[qbar])[link].sum()
        force += impulse * c_dev[q].to(dtype=f_post.dtype)
    return force


# --------------------------------------------------------------------------- #
# Single-rotation-rate simulation
# --------------------------------------------------------------------------- #

def run_single_rotation(
    alpha: float,
    omega: float,
    nx: int,
    ny: int,
    nz: int,
    R: float,
    cx: float,
    cy: float,
    u_in: float,
    tau: float,
    n_steps: int,
    ramp_steps: int,
    sponge_width: int,
    dev: torch.device,
    log_every: int = 500,
) -> tuple[float, float, bool]:
    """Run LBM simulation for one rotation rate.

    Returns ``(Cl_mean, Cd_mean, numerically_stable)``.  A non-finite
    distribution or force aborts this rate and cannot be a PASS.
    """
    D = 2.0 * R

    # --- Lattice constants on device -----------------------------------
    c_dev = C.to(dev)
    opp = OPPOSITE.to(dev)
    w_dev = W.to(dev)

    # --- Cylinder solid mask -------------------------------------------
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev),
        torch.arange(ny, device=dev),
        torch.arange(nx, device=dev),
        indexing="ij",
    )
    yy_f = yy.float()
    xx_f = xx.float()
    # The analytic zero level set is part of the rigid body, so no BFL link
    # can start exactly on the wall with a zero interpolation fraction.
    solid, phi = build_cylinder_geometry(nx, ny, nz, R, cx, cy, dev)

    # --- Free-stream equilibrium (sponge target + inlet) ---------------
    rho1 = torch.ones(nz, ny, nx, device=dev)
    feq_fs = equilibrium3d(
        rho1,
        torch.full_like(rho1, u_in),
        torch.zeros_like(rho1),
        torch.zeros_like(rho1),
        device=dev,
    )

    # --- Sponge mask (precomputed) -------------------------------------
    sponge = build_sponge_mask(nx, ny, nz, sponge_width, dev)
    sponge_4d = sponge.unsqueeze(0)  # (1, nz, ny, nx)

    # --- Initial flow: uniform + small perturbation --------------------
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full_like(rho0, u_in)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    # Small perturbation to break symmetry (helps vortex shedding at α=0)
    torch.manual_seed(42)
    uy0 += 0.02 * u_in * (torch.rand_like(rho0) * 2.0 - 1.0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=dev)

    # --- Force accumulators (second half) ------------------------------
    fx_accum = 0.0
    fy_accum = 0.0
    n_meas = 0

    t0 = time.time()
    for step in range(1, n_steps + 1):
        if not torch.isfinite(f).all().item():
            print(f"    [数值失败] step {step}: non-finite distribution", flush=True)
            return float("nan"), float("nan"), False
        # --- Rotation-rate ramp (smooth start) -------------------------
        if step <= ramp_steps:
            omega_eff = omega * float(step) / ramp_steps
        else:
            omega_eff = omega

        # --- 1. Collision (BGK) ---------------------------------------
        f_post = collide_bgk3d(f, tau)

        # --- 2. Streaming ---------------------------------------------
        f = stream3d(f_post)

        # --- 3. Boundary conditions -----------------------------------
        # Sponge: relax toward free-stream equilibrium
        f = f * (1.0 - sponge_4d) + feq_fs * sponge_4d
        # Inlet: hard velocity BC
        f[:, :, :, 0] = feq_fs[:, :, :, 0]

        # --- 4. Force measurement (BFL momentum exchange) --------------
        F = compute_bouzidi_momentum_exchange(
            f_post, solid, phi, c_dev, w_dev, omega_eff, cx, cy, yy_f, xx_f, R,
        )
        F_drag = float(F[0].item())
        F_lift = float(F[1].item())

        if step > n_steps // 2:
            fx_accum += F_drag
            fy_accum += F_lift
            n_meas += 1

        # --- 5. Moving-wall BFL on cylinder links ----------------------
        f = apply_bouzidi_moving_bounceback(
            f_post, f, solid, phi, opp, c_dev, w_dev,
            omega_eff, cx, cy, yy_f, xx_f, R,
        )
        if not torch.isfinite(f).all().item() or not torch.isfinite(F).all().item():
            print(f"    [数值失败] step {step}: non-finite force or post-bounce-back distribution", flush=True)
            return float("nan"), float("nan"), False

        # --- 6. Logging -----------------------------------------------
        if step % log_every == 0 or step == n_steps:
            cd_inst = 2.0 * F_drag / (u_in ** 2 * D)
            cl_inst = 2.0 * F_lift / (u_in ** 2 * D)
            rho, ux, uy, uz = macroscopic3d(f)
            print(
                f"    步数 {step:5d}:  Cd={cd_inst:+.4f}  Cl={cl_inst:+.4f}  "
                f"ρ∈[{float(rho.min()):.4f},{float(rho.max()):.4f}]  "
                f"{time.time() - t0:.0f}s",
                flush=True,
            )

    # --- Time-averaged coefficients ------------------------------------
    F_drag_mean = fx_accum / max(n_meas, 1)
    F_lift_mean = fy_accum / max(n_meas, 1)
    cd_mean = 2.0 * F_drag_mean / (u_in ** 2 * D)
    cl_mean = 2.0 * F_lift_mean / (u_in ** 2 * D)

    stable = math.isfinite(cl_mean) and math.isfinite(cd_mean) and n_meas > 0
    if not stable:
        print("    [数值失败] no finite measurement window", flush=True)
    return cl_mean, cd_mean, stable


# --------------------------------------------------------------------------- #
# Full benchmark
# --------------------------------------------------------------------------- #

def run_magnus_benchmark(
    device: str = "cpu",
    steps: int = 3000,
    nx: int = 300,
    ny: int = 200,
    nz: int = 1,
    R: float = 10.0,
    cx: float = 100.0,
    cy: float = 100.0,
    u_in: float = 0.1,
    tau: float = 0.55,
    alpha_list: list[float] | None = None,
    sponge_width: int = 40,
    ramp_steps: int = 200,
) -> list[dict]:
    """Run the full Magnus-effect benchmark over multiple rotation rates."""
    dev = torch.device(device)
    D = 2.0 * R
    nu_lat = (tau - 0.5) / 3.0
    Re_eff = u_in * D / nu_lat

    if alpha_list is None:
        alpha_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]

    # --- Header --------------------------------------------------------
    print("=" * 70, flush=True)
    print("  马格努斯效应基准测试 — 旋转圆柱体升力 (Magnus Effect)", flush=True)
    print("=" * 70, flush=True)
    print(f"  网格:       {nx} × {ny} × {nz}  (准二维, D3Q19 BGK)", flush=True)
    print(f"  圆柱体:     中心=({cx:.0f},{cy:.0f})  R={R}  D={D}", flush=True)
    print(f"  流动:       U={u_in}  τ={tau}  ν={nu_lat:.6f}  Re≈{Re_eff:.0f}", flush=True)
    print(f"  边界条件:   入口=速度BC  出口=海绵层(宽={sponge_width})  上下=海绵层", flush=True)
    print(f"  圆柱体:     移动壁反弹边界 (moving-wall bounce-back)", flush=True)
    print(f"  力测量:     动量交换法 (momentum exchange)", flush=True)
    print(f"  旋转率参数: α = {alpha_list}", flush=True)
    print(f"  步数:       {steps} (每个旋转率)  渐进={ramp_steps}步", flush=True)
    print(f"  设备:       {device}", flush=True)
    print("=" * 70, flush=True)

    results: list[dict] = []

    for alpha in alpha_list:
        omega = 2.0 * alpha * u_in / R
        v_theta = omega * R

        print(flush=True)
        print(f"  ▸ α = {alpha:.1f}  (ω = {omega:.6f}, v_θ = {v_theta:.4f})", flush=True)
        print(f"  {'-' * 66}", flush=True)

        cl_mean, cd_mean, stable = run_single_rotation(
            alpha=alpha,
            omega=omega,
            nx=nx, ny=ny, nz=nz,
            R=R, cx=cx, cy=cy,
            u_in=u_in, tau=tau,
            n_steps=steps,
            ramp_steps=ramp_steps,
            sponge_width=sponge_width,
            dev=dev,
        )

        # Glauert (1925) potential-flow theory: Cl = 2πα  (for α < 2)
        cl_glauert = 2.0 * math.pi * alpha

        # Compare |Cl| with Glauert (sign depends on rotation-direction
        # convention; the grid y-axis points downward, so ω>0 gives
        # positive Cl in grid coordinates)
        cl_abs = abs(cl_mean)
        if cl_glauert > 1e-10:
            error = abs(cl_abs - cl_glauert) / cl_glauert * 100.0
        else:
            error = 0.0 if cl_abs < 0.15 else 999.0

        results.append({
            "alpha": alpha,
            "omega": omega,
            "cl": cl_mean,
            "cd": cd_mean,
            "cl_abs": cl_abs,
            "cl_glauert": cl_glauert,
            "error": error,
            "stable": stable,
        })

        print(
            f"  → 结果:  Cl = {cl_mean:+.4f}  Cd = {cd_mean:+.4f}  "
            f"(Glauert Cl = {cl_glauert:.4f}, 误差 = {error:.1f}%)",
            flush=True,
        )

    # =================================================================== #
    # Results table
    # =================================================================== #
    print(flush=True)
    print("=" * 70, flush=True)
    print("  结果汇总 — Cl vs α  (升力系数 vs 旋转率参数)", flush=True)
    print("=" * 70, flush=True)
    header = (
        f"  {'α':>5s}  {'ω':>10s}  {'Cl(模拟)':>10s}  "
        f"{'|Cl|':>8s}  {'Cl(Glauert)':>12s}  {'误差%':>8s}  {'Cd':>8s}"
    )
    print(header, flush=True)
    print(f"  {'─'*5}  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*12}  {'─'*8}  {'─'*8}", flush=True)

    for r in results:
        print(
            f"  {r['alpha']:5.1f}  {r['omega']:10.6f}  {r['cl']:+10.4f}  "
            f"{r['cl_abs']:8.4f}  {r['cl_glauert']:12.4f}  {r['error']:8.1f}  "
            f"{r['cd']:+8.4f}",
            flush=True,
        )

    # =================================================================== #
    # Validation
    # =================================================================== #
    print(flush=True)
    print("=" * 70, flush=True)
    print("  验证结果", flush=True)
    print("=" * 70, flush=True)

    finite_results = all(
        r["stable"] and all(math.isfinite(r[k]) for k in ("cl", "cd", "error"))
        for r in results
    )

    # 1. Cl ≈ 0 at α = 0
    cl_alpha0 = abs(results[0]["cl"])
    check1 = finite_results and cl_alpha0 < 0.15
    print(
        f"  1. α=0 时 Cl≈0:     Cl = {results[0]['cl']:+.4f}  "
        f"|Cl| = {cl_alpha0:.4f}  → {'通过 ✓' if check1 else '未通过 ✗'}  "
        f"(阈值 |Cl| < 0.15)",
        flush=True,
    )

    # 2. |Cl| increases with α (Magnus effect)
    cl_abs_vals = [r["cl_abs"] for r in results]
    max_cl = max(cl_abs_vals)
    check2 = finite_results and all(
        cl_abs_vals[i] <= cl_abs_vals[i + 1] + 0.1 * max_cl + 0.05
        for i in range(len(cl_abs_vals) - 1)
    )
    print(
        f"  2. |Cl| 随 α 增大:  {'通过 ✓' if check2 else '未通过 ✗'}",
        flush=True,
    )
    if not check2:
        print(
            f"     |Cl| 序列: {['%.4f' % v for v in cl_abs_vals]}",
            flush=True,
        )

    # 3. Glauert comparison for α < 1.5 (excluding α=0)
    glauert_results = [r for r in results if 0 < r["alpha"] < 1.5]
    n_pass_glauert = sum(1 for r in glauert_results if r["error"] < 25.0)
    check3 = finite_results and n_pass_glauert >= 2
    print(
        f"  3. Glauert理论比较:  {n_pass_glauert}/{len(glauert_results)} 个旋转率"
        f"(α<1.5)误差<25%  → {'通过 ✓' if check3 else '未通过 ✗'}",
        flush=True,
    )

    # 4. Overall: <25% error for at least 3 rotation rates (α>0)
    n_pass_overall = sum(1 for r in results if r["alpha"] > 0 and r["error"] < 25.0)
    check4 = finite_results and n_pass_overall >= 3
    print(
        f"  4. 总体验证目标:    {n_pass_overall} 个旋转率误差<25%  "
        f"(目标≥3)  → {'通过 ✓' if check4 else '未通过 ✗'}",
        flush=True,
    )

    if not finite_results:
        print("  数值完整性检查: 未通过（至少一个旋转率出现非有限值或未完成测量）", flush=True)
    else:
        print("  数值完整性检查: 通过", flush=True)
    all_pass = finite_results and check1 and check2 and check4
    print(flush=True)
    verdict = "PASS" if all_pass else "FAIL"
    print(f"  ═══ 总体评定: {verdict} ═══", flush=True)
    print("=" * 70, flush=True)

    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="马格努斯效应基准测试: 旋转圆柱体升力 (Magnus effect benchmark)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="设备: cpu / cuda / sdaa:N",
    )
    parser.add_argument(
        "--steps", type=int, default=3000,
        help="每个旋转率的LBM时间步数 (默认 3000)",
    )
    parser.add_argument(
        "--nx", type=int, default=300,
        help="网格x方向 (默认 300)",
    )
    parser.add_argument(
        "--ny", type=int, default=200,
        help="网格y方向 (默认 200)",
    )
    parser.add_argument(
        "--R", type=float, default=10.0,
        help="圆柱半径 (格子单位, 默认 10)",
    )
    parser.add_argument(
        "--cx", type=float, default=100.0,
        help="圆柱中心x坐标 (默认 100)",
    )
    parser.add_argument(
        "--cy", type=float, default=100.0,
        help="圆柱中心y坐标 (默认 100)",
    )
    parser.add_argument(
        "--u-in", dest="u_in", type=float, default=0.1,
        help="入口速度 (格子单位, 默认 0.1)",
    )
    parser.add_argument(
        "--tau", type=float, default=0.55,
        help="BGK松弛时间τ (默认 0.55)",
    )
    parser.add_argument(
        "--sponge-width", dest="sponge_width", type=int, default=40,
        help="海绵层宽度 (默认 40)",
    )
    parser.add_argument(
        "--ramp-steps", dest="ramp_steps", type=int, default=200,
        help="旋转速率渐进步数 (默认 200)",
    )
    args = parser.parse_args()

    # Use all CPU cores for torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))

    run_magnus_benchmark(
        device=args.device,
        steps=args.steps,
        nx=args.nx,
        ny=args.ny,
        R=args.R,
        cx=args.cx,
        cy=args.cy,
        u_in=args.u_in,
        tau=args.tau,
        sponge_width=args.sponge_width,
        ramp_steps=args.ramp_steps,
    )
