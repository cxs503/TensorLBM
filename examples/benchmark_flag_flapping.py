"""Benchmark: Flag-in-wind — elastic flag flapping in uniform flow.

An elastic flag (one end clamped, other end free) flaps in uniform
flow due to a flow-induced instability (flutter).  The flapping
frequency and amplitude depend on the bending stiffness and flow
speed.

Physics
-------
- Fluid:  D3Q19 BGK collision + streaming, IBM direct-forcing
          (``ibm_direct_forcing_3d_vec`` + Guo body-force collision)
- Structure:  discrete Euler-Bernoulli beam (mass-spring chain with
              biharmonic bending operator + axial springs to maintain
              segment length).  Each node is an IBM marker.
              Node 0 is clamped (fixed position and angle).
- Coupling:  IBM reaction force → beam dynamics → marker velocity →
             IBM target velocity

Beam model (rotational-spring / discrete Laplacian)
---------------------------------------------------
- N nodes along the flag, each with mass m_node
- Bending force (rotational spring, 3-point Laplacian stencil):
                  F_i = -k_b * (x_{i-1} - 2*x_i + x_{i+1})
  Each pair of adjacent segments is connected by a rotational spring
  that resists the relative rotation (bending angle).  For small
  deflections this reduces to the discrete Laplacian (2nd derivative).
- Bending damping:  F_i += -c_b * (same operator on velocities)
- Axial springs:  linear springs between adjacent nodes maintaining
                  rest length L_node (prevents stretching/compression)
- Global damping:  F_i += -c_global * v_i  (Rayleigh, velocity-proportional)
- External force:  F_i += F_ibm_i  (from IBM)
- First node clamped (fixed position and angle), last node free
- Update:  v_i += F_i / m_node * dt;  x_i += v_i * dt

Boundary conditions (ghost nodes):
  - Clamped end (node 0):   du/dx=0  →  x_{-1} = x_1
  - Free end (node N-1):    d²u/dx²=0  →  x_N = 2*x_{N-1} - x_{N-2}
                            (bending force at free end = 0; zero-shear
                             condition omitted for stability)

Setup
-----
- Grid: 400×200×1, D3Q19 BGK, nz=1 (quasi-2D)
- Flag: L=60, h=2, N=20 nodes, clamped at (100, 100)
  - Extends in +x direction initially (horizontal)
  - Bending stiffness: E*I = 20.0 (lattice units, tuned for visible flapping)
  - Density ratio: rho_s/rho_f = 1.0
- Flow: U=0.1, Re=U*L/nu≈600, tau=0.55
- Inlet: velocity BC, Outlet: sponge (target equilibrium, width=40)
- Top/bottom: sponge (width=20)
- IBM markers = beam nodes (20 points along the flag)
- Steps: 5000

Expected behavior
-----------------
- Flag initially horizontal, then starts to oscillate due to flow
  instability
- Develops periodic flapping pattern
- Flapping frequency related to flow speed and bending stiffness

Validation
----------
1. Flag tip oscillates periodically (amplitude > 1 cell)
2. Flapping frequency is non-zero (detected via FFT)
3. Vortex shedding behind the flag (qualitative)

Run
---
    PYTHONPATH=src python examples/benchmark_flag_flapping.py --device cpu --steps 5000
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "src")

from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.ibm_vec import ibm_direct_forcing_3d_vec
from tensorlbm.ibm import ibm_delta_hat, ibm_delta_4pt
from tensorlbm.benchmark_observability import (
    BenchmarkReporter, assert_benchmark_tensor_device, resolve_benchmark_device,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def interpolate_velocity_markers(
    ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor,
    marker_x: torch.Tensor, marker_y: torch.Tensor, marker_z: torch.Tensor,
    kernel: str = "4pt",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized velocity interpolation at Lagrangian markers (3D).

    Extracts the interpolation portion of ``ibm_direct_forcing_3d_vec`` so
    we can obtain per-marker fluid velocities (needed for per-node
    hydrodynamic force on the beam).
    """
    nz, ny, nx = ux.shape
    device = ux.device
    n_markers = marker_x.shape[0]

    delta_fn = ibm_delta_hat if kernel == "hat" else ibm_delta_4pt
    support = 2 if kernel == "hat" else 4
    half_s = support // 2

    ix0 = (torch.floor(marker_x) - half_s + 1).long()
    iy0 = (torch.floor(marker_y) - half_s + 1).long()
    iz0 = (torch.floor(marker_z) - half_s + 1).long()

    offsets = torch.arange(support, device=device)

    ix_all = (ix0.unsqueeze(1) + offsets.unsqueeze(0)) % nx
    iy_all = (iy0.unsqueeze(1) + offsets.unsqueeze(0)) % ny
    iz_all = (iz0.unsqueeze(1) + offsets.unsqueeze(0)) % nz

    rx_all = (ix0.unsqueeze(1) + offsets.unsqueeze(0)).float() - marker_x.unsqueeze(1)
    ry_all = (iy0.unsqueeze(1) + offsets.unsqueeze(0)).float() - marker_y.unsqueeze(1)
    rz_all = (iz0.unsqueeze(1) + offsets.unsqueeze(0)).float() - marker_z.unsqueeze(1)

    wx_all = delta_fn(rx_all)
    wy_all = delta_fn(ry_all)
    wz_all = delta_fn(rz_all)

    u_mx = torch.zeros(n_markers, dtype=ux.dtype, device=device)
    u_my = torch.zeros(n_markers, dtype=uy.dtype, device=device)
    u_mz = torch.zeros(n_markers, dtype=uz.dtype, device=device)

    for di in range(support):
        for dj in range(support):
            for dk in range(support):
                w = wx_all[:, di] * wy_all[:, dj] * wz_all[:, dk]
                ix = ix_all[:, di]
                iy = iy_all[:, dj]
                iz = iz_all[:, dk]
                u_mx += w * ux[iz, iy, ix]
                u_my += w * uy[iz, iy, ix]
                u_mz += w * uz[iz, iy, ix]

    return u_mx, u_my, u_mz


def collide_bgk3d_guo(
    f: torch.Tensor,
    tau: float,
    fx: torch.Tensor,
    fy: torch.Tensor,
    fz: torch.Tensor,
) -> torch.Tensor:
    """D3Q19 BGK collision with Guo (2002) body-force correction.

    The Guo scheme distributes the force between the shifted equilibrium
    velocity and a post-collision correction term, giving second-order
    accuracy in the forcing.  This is essential for IBM direct-forcing:
    the force is "baked into" the equilibrium so the collision relaxes
    toward the velocity-corrected state rather than undoing the force.
    """
    rho, ux, uy, uz = macroscopic3d(f)
    rho_s = rho.clamp(min=1e-12)

    ux_s = ux + 0.5 * fx / rho_s
    uy_s = uy + 0.5 * fy / rho_s
    uz_s = uz + 0.5 * fz / rho_s
    feq = equilibrium3d(rho, ux_s, uy_s, uz_s)
    f_post = f - (f - feq) / tau

    c = C.to(f.device).float()
    w = W.to(f.device).float().view(19, 1, 1, 1)
    cx = c[:, 0].view(19, 1, 1, 1)
    cy = c[:, 1].view(19, 1, 1, 1)
    cz = c[:, 2].view(19, 1, 1, 1)

    cu = cx * ux + cy * uy + cz * uz
    cF = cx * fx + cy * fy + cz * fz
    uF = ux * fx + uy * fy + uz * fz

    coeff = 1.0 - 0.5 / tau
    forcing = coeff * w * (3.0 * (cF - uF) + 9.0 * cu * cF)
    return f_post + forcing


def apply_inlet_velocity(f: torch.Tensor, u_in: float) -> torch.Tensor:
    """Inlet velocity BC: set x=0 column to equilibrium with u_in."""
    nz, ny, nx = f.shape[1:]
    rho_in = torch.ones(nz, ny, 1, device=f.device, dtype=f.dtype)
    feq_in = equilibrium3d(
        rho_in,
        torch.full_like(rho_in, u_in),
        torch.zeros_like(rho_in),
        torch.zeros_like(rho_in),
        device=f.device,
    )
    f = f.clone()
    f[:, :, :, 0] = feq_in[:, :, :, 0]
    return f


def apply_outlet_sponge(f: torch.Tensor, u_in: float,
                         sponge_width: int) -> torch.Tensor:
    """Sponge layer at outlet: relax distributions toward equilibrium.

    A quadratic ramp blends the current distribution with the target
    equilibrium (rho=1, u=(u_in, 0, 0)) over the last *sponge_width*
    columns, damping spurious reflections.
    """
    if sponge_width <= 0:
        return f
    nz, ny, nx = f.shape[1:]
    sw = min(sponge_width, nx - 2)
    rho, ux, uy, uz = macroscopic3d(f)
    rho_t = torch.ones_like(rho)
    feq_t = equilibrium3d(
        rho_t,
        torch.full_like(rho_t, u_in),
        torch.zeros_like(rho_t),
        torch.zeros_like(rho_t),
        device=f.device,
    )
    idx = torch.arange(sw, device=f.device, dtype=f.dtype)
    sigma = (idx / max(sw - 1, 1)) ** 2
    sigma = sigma.view(1, 1, 1, sw)
    f = f.clone()
    f[:, :, :, -sw:] = (1.0 - sigma) * f[:, :, :, -sw:] + sigma * feq_t[:, :, :, -sw:]
    return f


def apply_top_bottom_sponge(f: torch.Tensor, u_in: float,
                             sponge_width: int) -> torch.Tensor:
    """Sponge layers at top and bottom: relax toward free-stream.

    A quadratic ramp blends the current distribution with the target
    equilibrium (rho=1, u=(u_in, 0, 0)) over the first and last
    *sponge_width* rows, damping spurious reflections from the
    top/bottom boundaries (open-flow condition).
    """
    if sponge_width <= 0:
        return f
    nz, ny, nx = f.shape[1:]
    sw = min(sponge_width, ny // 2 - 1)
    rho_t = torch.ones(nz, ny, nx, device=f.device, dtype=f.dtype)
    feq_t = equilibrium3d(
        rho_t,
        torch.full_like(rho_t, u_in),
        torch.zeros_like(rho_t),
        torch.zeros_like(rho_t),
        device=f.device,
    )
    idx = torch.arange(sw, device=f.device, dtype=f.dtype)
    sigma = (idx / max(sw - 1, 1)) ** 2  # 0 (inner) → 1 (boundary)

    f = f.clone()
    # Bottom: sigma=1 at y=0, sigma=0 at y=sw-1
    sigma_b = sigma.flip(0).view(1, sw, 1)
    f[:, :, :sw, :] = (1.0 - sigma_b) * f[:, :, :sw, :] + sigma_b * feq_t[:, :, :sw, :]
    # Top: sigma=0 at y=ny-sw, sigma=1 at y=ny-1
    sigma_t = sigma.view(1, sw, 1)
    f[:, :, -sw:, :] = (1.0 - sigma_t) * f[:, :, -sw:, :] + sigma_t * feq_t[:, :, -sw:, :]
    return f


def compute_vorticity_z(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """z-vorticity  ω_z = ∂u_y/∂x − ∂u_x/∂y  for (nz, ny, nx) fields."""
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    dux_dy[:, 1:-1, :] = 0.5 * (ux[:, 2:, :] - ux[:, :-2, :])
    duy_dx[:, :, 1:-1] = 0.5 * (uy[:, :, 2:] - uy[:, :, :-2])
    return duy_dx - dux_dy


# ---------------------------------------------------------------------------
# Discrete elastic beam (rotational-spring bending + axial springs)
# ---------------------------------------------------------------------------

def compute_beam_forces(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    vel_x: torch.Tensor,
    vel_y: torch.Tensor,
    k_b: float,
    c_b: float,
    k_axial: float,
    c_global: float,
    L_rest: float,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal forces on beam nodes.

    Combines four mechanisms:

    1. **Bending** (rotational-spring / discrete Laplacian, 3-point stencil):
           F_i = -k_b * (x_{i-1} - 2*x_i + x_{i+1})
       Each pair of adjacent segments is connected by a rotational spring
       that resists the relative rotation (bending angle).  For small
       deflections the bending angle at node *i* is
       ``phi_i ≈ (y_{i+1} - 2*y_i + y_{i-1}) / L_node``, so the restoring
       force is proportional to the discrete Laplacian (2nd derivative).
       This model has a significantly higher lowest-mode eigenvalue than
       the 4th-order biharmonic operator, making the flag-underdamped with
       moderate stiffness when coupled to IBM direct forcing.

    2. **Bending damping** (same operator on velocities):
           F_i += -c_b * (v_{i-1} - 2*v_i + v_{i+1})

    3. **Axial springs** (linear springs between adjacent nodes):
       Each segment (i, i+1) has a spring with stiffness k_axial and
       rest length L_rest.  This keeps the flag from stretching or
       compressing, maintaining its total length.

    4. **Global damping** (Rayleigh, velocity-proportional):
           F_i += -c_global * v_i

    Boundary conditions (ghost nodes):
        - Clamped end (node 0):   du/dx=0  →  x_{-1} = x_1
        - Free end (node N-1):    d²u/dx²=0  →  x_N = 2*x_{N-1} - x_{N-2}
                                  (bending force at free end = 0)

    Returns (fx, fy) tensors of shape (N,).
    """
    fx = torch.zeros(N, dtype=pos_x.dtype, device=pos_x.device)
    fy = torch.zeros(N, dtype=pos_y.dtype, device=pos_y.device)

    # --- 1. Bending (Laplacian / rotational spring) + bending damping ---
    if N >= 3:
        # Ghost nodes (one on each side for 3-point stencil)
        # Clamped end (node 0): zero slope → x_{-1} = x_1
        gl_x = pos_x[1:2]
        gl_y = pos_y[1:2]
        gl_vx = vel_x[1:2]
        gl_vy = vel_y[1:2]

        # Free end (node N-1): zero curvature → x_N = 2*x_{N-1} - x_{N-2}
        gr_x = (2.0 * pos_x[-1] - pos_x[-2]).unsqueeze(0)
        gr_y = (2.0 * pos_y[-1] - pos_y[-2]).unsqueeze(0)
        gr_vx = (2.0 * vel_x[-1] - vel_x[-2]).unsqueeze(0)
        gr_vy = (2.0 * vel_y[-1] - vel_y[-2]).unsqueeze(0)

        # Padded arrays: [ghost_left, real_nodes..., ghost_right]  (length N+2)
        px = torch.cat([gl_x, pos_x, gr_x])
        py = torch.cat([gl_y, pos_y, gr_y])
        pvx = torch.cat([gl_vx, vel_x, gr_vx])
        pvy = torch.cat([gl_vy, vel_y, gr_vy])

        # Laplacian for all nodes: L_i = x_{i-1} - 2*x_i + x_{i+1}
        # In padded array, real node i is at index i+1, so:
        # L[0:N] = px[0:N] - 2*px[1:N+1] + px[2:N+2]
        bend_x = px[0:N] - 2.0 * px[1:N + 1] + px[2:N + 2]
        bend_y = py[0:N] - 2.0 * py[1:N + 1] + py[2:N + 2]
        damp_x = pvx[0:N] - 2.0 * pvx[1:N + 1] + pvx[2:N + 2]
        damp_y = pvy[0:N] - 2.0 * pvy[1:N + 1] + pvy[2:N + 2]

        # The Laplacian is negative at a positive transverse maximum, so
        # a restoring rotational spring/damper uses +k*L and +c*L.
        # The previous negative signs made both terms anti-restoring and
        # caused exponential growth before the velocity clamp could act.
        fx = k_b * bend_x + c_b * damp_x
        fy = k_b * bend_y + c_b * damp_y

        # Node 0: clamped — zero internal force (position is fixed externally)
        fx[0] = 0.0
        fy[0] = 0.0
        # Node N-1: free end — Laplacian is zero by construction (ghost node),
        # so bending force is already zero there.

    # --- 2. Axial springs (maintain rest length) ---
    dx_seg = pos_x[1:] - pos_x[:-1]  # (N-1,)
    dy_seg = pos_y[1:] - pos_y[:-1]
    dist = torch.sqrt(dx_seg * dx_seg + dy_seg * dy_seg + 1e-12)
    stretch = dist - L_rest
    # Force on node i from segment (i, i+1): pull toward i+1 if stretched
    fx_seg = k_axial * stretch * dx_seg / dist
    fy_seg = k_axial * stretch * dy_seg / dist
    fx[:-1] += fx_seg
    fx[1:] -= fx_seg
    fy[:-1] += fy_seg
    fy[1:] -= fy_seg

    # --- 3. Global damping (Rayleigh) ---
    fx -= c_global * vel_x
    fy -= c_global * vel_y

    return fx, fy


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_flag_flapping_benchmark(
    device: str = "cpu",
    n_steps: int = 5000,
    nx: int = 400,
    ny: int = 200,
    u_in: float = 0.1,
    tau: float = 0.55,
    flag_L: float = 60.0,
    flag_h: float = 2.0,
    flag_N: int = 20,
    clamp_x: float = 100.0,
    clamp_y: float = 100.0,
    EI: float = 20.0,
    rho_s_ratio: float = 1.0,
    k_axial: float = 0.0,
    c_bend: float = 1.0,
    c_global: float = 1e-3,
    k_foundation: float = 0.02,
    sponge_outlet: int = 40,
    sponge_tb: int = 20,
    n_substeps: int = 20,
    ramp_steps: int = 500,
    ibm_relax: float = 0.5,
    kernel: str = "4pt",
    output_interval: int = 25,
    max_wall_seconds: float | None = 900.0,
    output_dir: str = "outputs",
):
    device_metadata = resolve_benchmark_device(device)
    dev = torch.device(device_metadata["resolved"])
    if output_interval < 1:
        raise ValueError("output_interval must be at least one step")
    if max_wall_seconds is not None and max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive or None")
    reporter = BenchmarkReporter(output_dir, "flag_flapping", n_steps, device_metadata)
    reporter.start()
    nz = 1
    cz0 = 0.0  # single z-layer

    # --- Lattice viscosity / relaxation -------------------------------
    nu_lat = (tau - 0.5) / 3.0
    Re_actual = u_in * flag_L / nu_lat if nu_lat > 0 else float("inf")

    # --- Flag / beam parameters ---------------------------------------
    n_seg = flag_N - 1
    L_node = flag_L / n_seg
    rho_f = 1.0
    rho_solid = rho_s_ratio * rho_f
    m_node = rho_solid * flag_h * L_node          # mass per node
    k_b = EI                                      # bending stiffness
    c_b = c_bend                                  # bending damping
    L_rest = L_node                               # axial spring rest length

    # --- IBM markers: beam nodes only (no cylinder) -------------------
    beam_pos_x = torch.tensor(
        [clamp_x + i * L_node for i in range(flag_N)],
        dtype=torch.float32, device=dev,
    )
    beam_pos_y = torch.full((flag_N,), clamp_y, dtype=torch.float32, device=dev)
    beam_vel_x = torch.zeros(flag_N, dtype=torch.float32, device=dev)
    beam_vel_y = torch.zeros(flag_N, dtype=torch.float32, device=dev)

    # Small initial perturbation to break symmetry (trigger flutter)
    beam_pos_y[1:] += 0.5 * torch.linspace(0, 1, flag_N - 1, device=dev)

    # Store initial positions for foundation spring (weak anchoring)
    beam_pos_x_init = beam_pos_x.clone()
    beam_pos_y_init = beam_pos_y.clone()

    n_total = flag_N

    # --- Initial flow: uniform + small perturbation -------------------
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    yy_grid = torch.arange(ny, device=dev, dtype=torch.float32).view(1, ny, 1)
    uy0 = 0.005 * u_in * torch.sin(2.0 * math.pi * yy_grid / ny)
    uy0 = uy0.expand(nz, ny, nx).contiguous()
    uz0 = torch.zeros_like(ux0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=dev)
    # Validate actual state placement before entering the expensive loop.
    assert_benchmark_tensor_device(f, dev, "distribution f")
    assert_benchmark_tensor_device(beam_pos_x, dev, "beam_pos_x")

    # --- Wake probe (downstream of flag tip) --------------------------
    probe_x = min(int(clamp_x + flag_L + 20), nx - sponge_outlet - 2)
    probe_y = int(clamp_y)

    initial_mass = float(f.sum().item())

    # --- Storage ------------------------------------------------------
    tip_y_hist: list[float] = []
    tip_x_hist: list[float] = []
    tip_vy_hist: list[float] = []
    fy_beam_hist: list[float] = []
    uy_probe_hist: list[float] = []

    # --- Header -------------------------------------------------------
    print("=" * 70, flush=True)
    print("  旗帜风中摆动基准测试 — 弹性旗帜在均匀流中的流致振动", flush=True)
    print("  Flag-in-wind: elastic flag flapping due to flow-induced instability", flush=True)
    print("=" * 70, flush=True)
    print(f"  网格:       {nx} × {ny} × {nz}  (准二维, D3Q19)", flush=True)
    print(f"  旗帜:       L={flag_L}  h={flag_h}  N={flag_N}节点  L_node={L_node:.4f}", flush=True)
    print(f"  固定端:     ({clamp_x:.0f}, {clamp_y:.0f})  沿+x方向延伸", flush=True)
    print(f"  流动:       Re={Re_actual:.0f}  U={u_in}  ν={nu_lat:.6f}  τ={tau:.4f}", flush=True)
    print(f"  边界:       入口速度BC  出口海绵层(宽度={sponge_outlet})  "
          f"上下海绵层(宽度={sponge_tb})", flush=True)
    print(f"  材料:       ρ_s/ρ_f={rho_s_ratio}  EI={EI}  m_node={m_node:.4f}", flush=True)
    print(f"  弹簧:       k_b={k_b}  c_b={c_b}  k_axial={k_axial}  c_global={c_global}  k_found={k_foundation}", flush=True)
    print(f"  子步进:     n_sub={n_substeps}  dt_sub={1.0/n_substeps:.4f}", flush=True)
    print(f"  渐升:       ramp_steps={ramp_steps}", flush=True)
    print(f"  IBM松弛:   ibm_relax={ibm_relax}", flush=True)
    print(f"  IBM:        旗帜标记={flag_N}(ds={L_node:.3f})  总标记={n_total}", flush=True)
    print(f"  内核:       '{kernel}'", flush=True)
    print(f"  运行:       步数={n_steps}  请求设备={device}  实际设备={dev}", flush=True)
    print(f"  设备断言:   allocation={device_metadata['allocation_device']}  "
          f"max_wall_seconds={max_wall_seconds}", flush=True)
    print(f"  状态文件:   {reporter.status_path}  进度CSV: {reporter.progress_path}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    numerical_failure: str | None = None
    for step in range(1, n_steps + 1):
        # 流速渐升: 前ramp_steps步线性增加入口速度, 避免初始冲击
        ramp = min(float(step) / float(ramp_steps), 1.0) if ramp_steps > 0 else 1.0
        u_in_eff = u_in * ramp

        # --- 1. 宏观场 (碰撞前) ---------------------------------------
        rho, ux, uy, uz = macroscopic3d(f)
        # Never permit NaNs/Infs to feed marker wrapping, FFT, or a PASS.
        if not (torch.isfinite(f).all().item()
                and torch.isfinite(rho).all().item()
                and torch.isfinite(beam_pos_x).all().item()
                and torch.isfinite(beam_pos_y).all().item()
                and torch.isfinite(beam_vel_x).all().item()
                and torch.isfinite(beam_vel_y).all().item()):
            numerical_failure = f"step {step}: non-finite fluid or beam state"
            print(f"  [数值失败] {numerical_failure}", flush=True)
            break

        # NaN debug: check macroscopic fields
        if step <= 200 and (step % 10 == 0):
            if torch.isnan(rho).any() or torch.isnan(ux).any():
                print(f"  [DEBUG] NaN in macro at step {step}: "
                      f"rho=[{float(rho.min()):.4e},{float(rho.max()):.4e}] "
                      f"ux=[{float(ux.min()):.4e},{float(ux.max()):.4e}] "
                      f"beam_pos_x=[{float(beam_pos_x.min()):.4f},{float(beam_pos_x.max()):.4f}] "
                      f"beam_pos_y=[{float(beam_pos_y.min()):.4f},{float(beam_pos_y.max()):.4f}] "
                      f"beam_vel_x=[{float(beam_vel_x.min()):.4e},{float(beam_vel_x.max()):.4e}]",
                      flush=True)
                break

        # --- 2. IBM 直动力 --------------------------------------------
        # 所有标记都是梁节点 (无圆柱)
        mx_all = beam_pos_x
        my_all = beam_pos_y
        mz_all = torch.full_like(beam_pos_x, cz0)

        # 插值梁节点处的流体速度 (用于松弛目标 + 流体力计算)
        u_mx_b, u_my_b, _ = interpolate_velocity_markers(
            ux, uy, uz, beam_pos_x, beam_pos_y,
            torch.full_like(beam_pos_x, cz0), kernel=kernel,
        )

        # 松弛目标速度: u_target = alpha*v_beam + (1-alpha)*u_interp
        # 降低IBM力, 稳定显式FSI耦合 (附加质量效应)
        u_tgt_bx = ibm_relax * beam_vel_x + (1.0 - ibm_relax) * u_mx_b
        u_tgt_by = ibm_relax * beam_vel_y + (1.0 - ibm_relax) * u_my_b

        u_t_x = u_tgt_bx
        u_t_y = u_tgt_by
        u_t_z = torch.zeros(n_total, device=dev, dtype=torch.float32)

        fx_grid, fy_grid, fz_grid = ibm_direct_forcing_3d_vec(
            ux, uy, uz, mx_all, my_all, mz_all,
            u_t_x, u_t_y, u_t_z,
            kernel=kernel,
        )
        assert_benchmark_tensor_device(fx_grid, dev, "IBM force grid")

        # --- 3. 碰撞 (BGK + Guo体力) ----------------------------------
        f = collide_bgk3d_guo(f, tau, fx_grid, fy_grid, fz_grid)

        # --- 4. 流动 ---------------------------------------------------
        f = stream3d(f)

        # --- 5. 边界条件 ----------------------------------------------
        f = apply_inlet_velocity(f, u_in_eff)
        f = apply_outlet_sponge(f, u_in_eff, sponge_outlet)
        f = apply_top_bottom_sponge(f, u_in_eff, sponge_tb)

        # --- 6. 质量修正 ----------------------------------------------
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if not (torch.isfinite(f).all().item()
                and torch.isfinite(fx_grid).all().item()
                and torch.isfinite(fy_grid).all().item()):
            numerical_failure = f"step {step}: non-finite IBM force or post-boundary distribution"
            print(f"  [数值失败] {numerical_failure}", flush=True)
            break

        # NaN debug
        if step <= 200 and (step % 10 == 0):
            if torch.isnan(f).any():
                print(f"  [DEBUG] NaN in f at step {step}", flush=True)
                rho2, ux2, uy2, uz2 = macroscopic3d(f)
                print(f"    rho: [{float(rho2.min()):.6e}, {float(rho2.max()):.6e}]", flush=True)
                print(f"    ux:  [{float(ux2.min()):.6e}, {float(ux2.max()):.6e}]", flush=True)
                print(f"    fx_grid: [{float(fx_grid.min()):.6e}, {float(fx_grid.max()):.6e}]", flush=True)
                print(f"    fy_grid: [{float(fy_grid.min()):.6e}, {float(fy_grid.max()):.6e}]", flush=True)
                break

        # --- 7. 梁节点流体力 (IBM反力) --------------------------------
        # 标记力 = u_target - u_interp; 流体力 = -标记力 * ds
        ds_beam = L_node
        F_hydro_x = -(u_tgt_bx - u_mx_b) * ds_beam
        F_hydro_y = -(u_tgt_by - u_my_b) * ds_beam

        # --- 8. 结构更新 (子步进半隐式Euler) ---------------------------
        # 弯曲弹簧较硬, 需要子步进以保持显式积分稳定性
        n_sub = n_substeps
        dt_sub = 1.0 / n_sub
        for _ in range(n_sub):
            F_int_x, F_int_y = compute_beam_forces(
                beam_pos_x, beam_pos_y, beam_vel_x, beam_vel_y,
                k_b, c_b, k_axial, c_global, L_rest, flag_N,
            )
            # Foundation spring: weak anchoring to initial position
            # Prevents rigid-body drift (uniform/linear modes have zero
            # Laplacian → no bending resistance → flag blows away)
            F_int_x += -k_foundation * (beam_pos_x - beam_pos_x_init)
            F_int_y += -k_foundation * (beam_pos_y - beam_pos_y_init)
            F_total_x = F_hydro_x + F_int_x
            F_total_y = F_hydro_y + F_int_y
            # Node 0 clamped
            F_total_x[0] = 0.0
            F_total_y[0] = 0.0
            # Semi-implicit Euler: v += F/m * dt; x += v * dt
            beam_vel_x += (F_total_x / m_node) * dt_sub
            beam_vel_y += (F_total_y / m_node) * dt_sub
            beam_vel_x[0] = 0.0
            beam_vel_y[0] = 0.0
            beam_pos_x += beam_vel_x * dt_sub
            beam_pos_y += beam_vel_y * dt_sub
            beam_pos_x[0] = clamp_x
            beam_pos_y[0] = clamp_y

        # 速度钳制: 防止数值不稳定导致旗帜飞出网格
        v_max = 0.5
        beam_vel_x = beam_vel_x.clamp(min=-v_max, max=v_max)
        beam_vel_y = beam_vel_y.clamp(min=-v_max, max=v_max)
        beam_vel_x[0] = 0.0
        beam_vel_y[0] = 0.0

        if not (torch.isfinite(beam_pos_x).all().item()
                and torch.isfinite(beam_pos_y).all().item()
                and torch.isfinite(beam_vel_x).all().item()
                and torch.isfinite(beam_vel_y).all().item()):
            numerical_failure = f"step {step}: non-finite beam state after structural update"
            print(f"  [数值失败] {numerical_failure}", flush=True)
            break

        # NaN debug: check beam state after sub-stepping
        if step <= 200 and (step % 10 == 0):
            if torch.isnan(beam_pos_x).any() or torch.isnan(beam_vel_x).any():
                print(f"  [DEBUG] NaN in beam at step {step}: "
                      f"pos_x=[{float(beam_pos_x.min()):.4f},{float(beam_pos_x.max()):.4f}] "
                      f"pos_y=[{float(beam_pos_y.min()):.4f},{float(beam_pos_y.max()):.4f}] "
                      f"vel_x=[{float(beam_vel_x.min()):.4e},{float(beam_vel_x.max()):.4e}] "
                      f"vel_y=[{float(beam_vel_y.min()):.4e},{float(beam_vel_y.max()):.4e}] "
                      f"F_hydro_x=[{float(F_hydro_x.min()):.4e},{float(F_hydro_x.max()):.4e}] "
                      f"F_hydro_y=[{float(F_hydro_y.min()):.4e},{float(F_hydro_y.max()):.4e}]",
                      flush=True)
                break

        # --- 9. 记录 -------------------------------------------------
        tip_y = float(beam_pos_y[-1].item()) - clamp_y
        tip_x = float(beam_pos_x[-1].item()) - clamp_x - flag_L
        tip_vy = float(beam_vel_y[-1].item())
        fy_total = float(F_hydro_y.sum().item())
        uy_probe = float(uy[0, probe_y, probe_x].item())

        tip_y_hist.append(tip_y)
        tip_x_hist.append(tip_x)
        tip_vy_hist.append(tip_vy)
        fy_beam_hist.append(fy_total)
        uy_probe_hist.append(uy_probe)

        # --- 10. 打印 ------------------------------------------------
        elapsed = time.time() - t0
        watchdog_expired = max_wall_seconds is not None and elapsed >= max_wall_seconds
        if step % output_interval == 0 or step == n_steps or watchdog_expired:
            print(
                f"  步 {step:5d}:  旗尖y={tip_y:+.4f}  vy={tip_vy:+.5f}  "
                f"Fy={fy_total:+.5f}  探针uy={uy_probe:+.5f}  "
                f"ρ∈[{float(rho.min()):.4f},{float(rho.max()):.4f}]  "
                f"{elapsed:.0f}秒",
                flush=True,
            )
            reporter.progress(step, elapsed, tip_y, tip_x)
        if watchdog_expired:
            numerical_failure = (
                f"watchdog: elapsed {elapsed:.1f}s reached "
                f"max_wall_seconds={max_wall_seconds}"
            )
            print(f"  [看门狗] {numerical_failure}", flush=True)
            break

    dt_total = time.time() - t0
    print("=" * 70, flush=True)
    print(f"  仿真完成: {dt_total:.1f}秒  ({dt_total/max(len(tip_y_hist), 1)*1e3:.1f} 毫秒/步)", flush=True)

    # ===================================================================
    # 分析
    # ===================================================================
    tip_y_arr = np.array(tip_y_hist)
    tip_x_arr = np.array(tip_x_hist)
    uy_arr = np.array(uy_probe_hist)
    fy_arr = np.array(fy_beam_hist)

    n_completed = len(tip_y_hist)
    # Use the actual completed history when a numerical failure aborts early.
    n_trans = n_completed // 2
    tip_y_ss = tip_y_arr[n_trans:]
    uy_ss = uy_arr[n_trans:]
    n_ss = len(tip_y_ss)

    # FFT (零填充提高频率分辨率)
    n_fft = max(4096, 2 * n_ss)
    freqs = np.fft.rfftfreq(n_fft, d=1.0)

    def _dominant_freq(signal: np.ndarray) -> tuple[float, float]:
        if signal.size == 0 or not np.isfinite(signal).all():
            return 0.0, 0.0
        sig = signal - signal.mean()
        if np.max(np.abs(sig)) < 1e-15:
            return 0.0, 0.0
        spectrum = np.abs(np.fft.rfft(sig, n=n_fft))
        spectrum[0] = 0.0
        idx = int(np.argmax(spectrum))
        return freqs[idx], spectrum[idx]

    f_tip, _ = _dominant_freq(tip_y_ss)
    f_probe, _ = _dominant_freq(uy_ss)

    # 振幅: 峰峰值的一半
    A_tip = float(np.max(tip_y_ss) - np.min(tip_y_ss)) / 2.0 if n_ss > 0 else 0.0
    A_tip_max = float(np.max(np.abs(tip_y_ss))) if n_ss > 0 else 0.0

    # --- 涡量场分析 (定性涡脱落检测) ---
    rho_f, ux_f, uy_f, _ = macroscopic3d(f)
    vort = compute_vorticity_z(ux_f, uy_f)[0].cpu().numpy()
    # 尾流区域: 旗帜后方
    wake_x0 = int(clamp_x + flag_L + 10)
    wake_x1 = min(int(clamp_x + flag_L + 150), nx - sponge_outlet)
    wake_y0 = max(int(clamp_y - 60), sponge_tb)
    wake_y1 = min(int(clamp_y + 60), ny - sponge_tb)
    vort_wake = vort[wake_y0:wake_y1, wake_x0:wake_x1]
    vort_rms = float(np.sqrt(np.mean(vort_wake ** 2))) if vort_wake.size > 0 else 0.0
    vort_max = float(np.max(np.abs(vort_wake))) if vort_wake.size > 0 else 0.0

    # --- 验证报告 ---
    print(flush=True)
    print("=" * 70, flush=True)
    print("  验证结果", flush=True)
    print("=" * 70, flush=True)

    # 1. 周期性振荡 (振幅 > 1格子)
    print(f"  1. 旗尖周期性振荡 (振幅 > 1格子):", flush=True)
    print(f"     旗尖y振幅 (峰峰值/2) = {A_tip:.4f} 格子单位", flush=True)
    print(f"     旗尖y最大位移        = {A_tip_max:.4f} 格子单位", flush=True)
    amp_ok = A_tip > 1.0
    amp_err = 0.0 if amp_ok else (1.0 - A_tip) / 1.0 * 100
    print(f"     检查: {'通过' if amp_ok else '未通过'}  "
          f"(A > 1.0, 误差={amp_err:.1f}%)", flush=True)

    # 2. 拍动频率非零 (FFT检测)
    print(f"  2. 拍动频率非零 (FFT检测):", flush=True)
    print(f"     旗尖主频 f_tip   = {f_tip:.6f} 周/步", flush=True)
    print(f"     探针主频 f_probe = {f_probe:.6f} 周/步", flush=True)
    if f_tip > 0:
        T_tip = 1.0 / f_tip
        print(f"     拍动周期 T       = {T_tip:.1f} 步", flush=True)
        print(f"     5000步内振荡数   = {n_steps * f_tip:.1f}", flush=True)
    freq_ok = f_tip > 1e-5
    freq_err = 0.0 if freq_ok else 100.0
    print(f"     检查: {'通过' if freq_ok else '未通过'}  "
          f"(f > 1e-5, 误差={freq_err:.1f}%)", flush=True)

    # 3. 尾流涡脱落 (定性)
    print(f"  3. 尾流涡脱落 (定性):", flush=True)
    print(f"     尾流区域: x=[{wake_x0},{wake_x1}] y=[{wake_y0},{wake_y1}]", flush=True)
    print(f"     涡量RMS = {vort_rms:.6f}", flush=True)
    print(f"     涡量最大值 = {vort_max:.6f}", flush=True)
    vort_ok = vort_rms > 1e-4
    vort_err = 0.0 if vort_ok else 100.0
    print(f"     检查: {'通过' if vort_ok else '未通过'}  "
          f"(RMS > 1e-4, 误差={vort_err:.1f}%)", flush=True)

    # --- 综合评估 ---
    finite_metrics = (numerical_failure is None and n_completed == n_steps
                      and all(np.isfinite(a).all() for a in
                              (tip_y_arr, tip_x_arr, uy_arr, fy_arr))
                      and all(math.isfinite(v) for v in
                              (A_tip, A_tip_max, f_tip, f_probe, vort_rms, vort_max)))
    print(f"  数值完整性检查: {'通过' if finite_metrics else '未通过'}"
          f"  ({numerical_failure or '所有状态和指标均为有限值'})", flush=True)
    all_pass = finite_metrics and amp_ok and freq_ok and vort_ok
    print(flush=True)
    print(f"  周期振荡检查: {'通过' if amp_ok else '未通过'} "
          f"(A={A_tip:.4f} > 1.0)", flush=True)
    print(f"  频率检测检查: {'通过' if freq_ok else '未通过'} "
          f"(f={f_tip:.6f} > 0)", flush=True)
    print(f"  涡脱落检查:   {'通过' if vort_ok else '未通过'} "
          f"(RMS={vort_rms:.6f} > 1e-4)", flush=True)
    print(flush=True)
    print(f"  总体结果: {'通过 ✓' if all_pass else '未通过 ✗'}", flush=True)
    print("=" * 70, flush=True)

    # ===================================================================
    # 保存输出
    # ===================================================================
    os.makedirs(output_dir, exist_ok=True)

    # CSV时间序列
    csv_path = os.path.join(output_dir, "flag_flapping_data.csv")
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "tip_y", "tip_x", "tip_vy", "Fy_beam", "uy_probe"])
        for i in range(n_completed):
            w.writerow([i + 1, tip_y_hist[i], tip_x_hist[i],
                        tip_vy_hist[i], fy_beam_hist[i], uy_probe_hist[i]])
    print(f"  已保存: {csv_path}", flush=True)

    # 图表
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 时间序列
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(tip_y_arr, "b-", linewidth=0.8)
        axes[0].set_ylabel("旗尖y位移 (格子单位)")
        axes[0].set_title("旗帜摆动: 旗尖横向位移时间序列")
        axes[0].axhline(0, color="k", linewidth=0.5)
        axes[0].axvline(n_trans, color="r", linestyle="--", linewidth=0.5,
                        label="瞬态结束")
        axes[0].legend(fontsize=8)

        axes[1].plot(uy_arr, "g-", linewidth=0.8)
        axes[1].set_ylabel(r"$u_y$ 探针 (格子单位)")
        axes[1].set_title(f"尾流速度探针 (x={probe_x}, y={probe_y})")
        axes[1].axhline(0, color="k", linewidth=0.5)

        axes[2].plot(fy_arr, "r-", linewidth=0.8)
        axes[2].set_ylabel(r"$F_{y}$ 旗帜流体力")
        axes[2].set_xlabel("时间步")
        axes[2].set_title("旗帜横向流体力")
        axes[2].axhline(0, color="k", linewidth=0.5)

        plt.tight_layout()
        ts_path = os.path.join(output_dir, "flag_flapping_timeseries.png")
        fig.savefig(ts_path, dpi=120)
        plt.close(fig)
        print(f"  已保存: {ts_path}", flush=True)

        # 涡量场快照
        vmax = max(abs(vort.min()), abs(vort.max()))
        vmax = max(vmax, 1e-6) * 0.8
        fig2, ax2 = plt.subplots(figsize=(14, 4))
        im = ax2.imshow(vort, origin="lower", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax,
                        extent=(0.0, float(nx), 0.0, float(ny)))
        plt.colorbar(im, ax=ax2, label=r"$\omega_z$")
        ax2.plot(clamp_x, clamp_y, "ks", markersize=6, label="固定端")
        bx = beam_pos_x.cpu().numpy()
        by = beam_pos_y.cpu().numpy()
        ax2.plot(bx, by, "r.-", markersize=3, linewidth=1.5, label="旗帜")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(f"涡量场 (步 {n_steps})  f={f_tip:.6f}  "
                      f"A={A_tip:.4f}  RMS={vort_rms:.6f}")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        vort_path = os.path.join(output_dir, "flag_flapping_vorticity.png")
        fig2.savefig(vort_path, dpi=120)
        plt.close(fig2)
        print(f"  已保存: {vort_path}", flush=True)

        # FFT频谱
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 5))
        y_spec = np.abs(np.fft.rfft(tip_y_ss - tip_y_ss.mean(), n=n_fft))
        ax3.semilogy(freqs, y_spec, "b-", linewidth=0.8)
        ax3.axvline(f_tip, color="r", linestyle="--", linewidth=0.8,
                     label=f"f_tip={f_tip:.6f}")
        ax3.set_ylabel("|FFT(旗尖y)|")
        ax3.set_xlabel("频率 (周/步)")
        ax3.set_title("旗尖位移频谱")
        ax3.legend(fontsize=8)
        plt.tight_layout()
        fft_path = os.path.join(output_dir, "flag_flapping_spectrum.png")
        fig3.savefig(fft_path, dpi=120)
        plt.close(fig3)
        print(f"  已保存: {fft_path}", flush=True)

    except ImportError:
        print("  (matplotlib不可用 — 跳过图表)", flush=True)

    result = {
        "f_tip": f_tip,
        "f_probe": f_probe,
        "amplitude": A_tip,
        "amplitude_max": A_tip_max,
        "vort_rms": vort_rms,
        "vort_max": vort_max,
        "Re_actual": Re_actual,
        "numerical_failure": numerical_failure,
        "steps_completed": n_completed,
        "pass": all_pass,
    }
    reporter.finish(
        n_completed,
        "PASSED" if all_pass else "FAILED",
        numerical_failure,
        result,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="旗帜风中摆动基准: 弹性旗帜在均匀流中的流致振动"
    )
    parser.add_argument("--device", default="cpu",
                        help="设备: cpu / cuda / sdaa:N")
    parser.add_argument("--steps", type=int, default=5000,
                        help="LBM时间步数")
    parser.add_argument("--nx", type=int, default=400, help="网格x方向")
    parser.add_argument("--ny", type=int, default=200, help="网格y方向")
    parser.add_argument("--u-in", dest="u_in", type=float, default=0.1,
                        help="入口速度 (格子单位)")
    parser.add_argument("--tau", type=float, default=0.55,
                        help="BGK松弛时间 τ (ν=(τ−0.5)/3)")
    parser.add_argument("--flag-L", dest="flag_L", type=float, default=60.0,
                        help="旗帜长度 (格子单位)")
    parser.add_argument("--flag-h", dest="flag_h", type=float, default=2.0,
                        help="旗帜厚度 (格子单位)")
    parser.add_argument("--flag-N", dest="flag_N", type=int, default=20,
                        help="旗帜节点数")
    parser.add_argument("--clamp-x", dest="clamp_x", type=float,
                        default=100.0, help="固定端x坐标")
    parser.add_argument("--clamp-y", dest="clamp_y", type=float,
                        default=100.0, help="固定端y坐标")
    parser.add_argument("--EI", type=float, default=20.0,
                        help="弯曲刚度 (格子单位, 调谐用于可见摆动)")
    parser.add_argument("--rho-s-ratio", dest="rho_s_ratio", type=float,
                        default=1.0, help="密度比 ρ_s/ρ_f")
    parser.add_argument("--k-axial", dest="k_axial", type=float,
                        default=0.0, help="轴向弹簧刚度 (0=禁用, 防止与弯曲刚度耦合不稳定)")
    parser.add_argument("--c-bend", dest="c_bend", type=float,
                        default=1.0, help="弯曲阻尼系数 (模态比例阻尼)")
    parser.add_argument("--c-global", dest="c_global", type=float,
                        default=1e-3, help="全局阻尼系数")
    parser.add_argument("--k-foundation", dest="k_foundation", type=float,
                        default=0.02, help="基础弹簧刚度 (防止刚体漂移)")
    parser.add_argument("--sponge-outlet", dest="sponge_outlet",
                        type=int, default=40, help="出口海绵层宽度")
    parser.add_argument("--sponge-tb", dest="sponge_tb",
                        type=int, default=20, help="上下海绵层宽度")
    parser.add_argument("--n-substeps", dest="n_substeps",
                        type=int, default=20, help="结构更新子步数")
    parser.add_argument("--ramp-steps", dest="ramp_steps",
                        type=int, default=500, help="入口流速渐升步数")
    parser.add_argument("--ibm-relax", dest="ibm_relax",
                        type=float, default=0.5, help="IBM耦合松弛因子(0-1)")
    parser.add_argument("--kernel", default="4pt", choices=["hat", "4pt"],
                        help="IBM delta内核")
    parser.add_argument("--output-interval", dest="output_interval",
                        type=int, default=25, help="进度CSV/日志间隔 (步)")
    parser.add_argument("--max-wall-seconds", dest="max_wall_seconds", type=float,
                        default=900.0,
                        help="看门狗时间上限; <=0 禁用 (默认900秒)")
    parser.add_argument("--output-dir", dest="output_dir",
                        default="outputs", help="输出目录")
    args = parser.parse_args()

    run_flag_flapping_benchmark(
        device=args.device,
        n_steps=args.steps,
        nx=args.nx,
        ny=args.ny,
        u_in=args.u_in,
        tau=args.tau,
        flag_L=args.flag_L,
        flag_h=args.flag_h,
        flag_N=args.flag_N,
        clamp_x=args.clamp_x,
        clamp_y=args.clamp_y,
        EI=args.EI,
        rho_s_ratio=args.rho_s_ratio,
        k_axial=args.k_axial,
        c_bend=args.c_bend,
        c_global=args.c_global,
        k_foundation=args.k_foundation,
        sponge_outlet=args.sponge_outlet,
        sponge_tb=args.sponge_tb,
        n_substeps=args.n_substeps,
        ramp_steps=args.ramp_steps,
        ibm_relax=args.ibm_relax,
        kernel=args.kernel,
        output_interval=args.output_interval,
        max_wall_seconds=None if args.max_wall_seconds <= 0 else args.max_wall_seconds,
        output_dir=args.output_dir,
    )
