"""Benchmark: Turek-Hron FSI2 — cylinder with elastic beam in channel flow.

The classic Turek & Hron (2006) fluid-structure interaction benchmark:
a rigid cylinder with an elastic beam attached behind it, placed in
channel flow.  Vortex shedding from the cylinder drives the beam into
periodic oscillation.

Physics
-------
- Fluid:  D3Q19 BGK collision + streaming, IBM direct-forcing
          (``ibm_direct_forcing_3d_vec`` + Guo body-force collision)
- Structure:  discrete Euler-Bernoulli beam (mass-spring chain with
              biharmonic bending operator).  Each node is an IBM marker.
              Node 0 is clamped to the cylinder trailing edge.
- Coupling:  IBM reaction force → beam dynamics → marker velocity →
             IBM target velocity

Beam model (discrete Euler-Bernoulli / biharmonic operator)
----------------------------------------------------------
- N nodes along the beam, each with mass m_node
- Bending force (discrete biharmonic, 5-point stencil):
                  F_i = -k_b * (x_{i-2} - 4*x_{i-1} + 6*x_i - 4*x_{i+1} + x_{i+2})
- Damping:        F_i += -c_b * (same operator on velocities)
- External force: F_i += F_ibm_i  (from IBM)
- First node fixed (attached to cylinder), last node free
- Update:  v_i += F_i / m_node * dt;  x_i += v_i * dt

Boundary conditions (ghost nodes):
  - Clamped end (node 0):   du/dx=0  →  x_{-1} = x_1
  - Free end (node N-1):    d²u/dx²=0  →  x_N = 2*x_{N-1} - x_{N-2}
                            (bending force at free end = 0; zero-shear
                             condition omitted for stability)

Setup (FSI2 case, Re≈100)
-------------------------
- Grid: 300×100×1, D3Q19 BGK, nz=1 (quasi-2D)
- Channel: height=100, inlet velocity U=0.1
- Cylinder: centre (60, 50), radius R=5
- Beam: L=35, h=2, N=20 nodes
- Beam properties: rho_s=10 (density ratio), E_bend=1e4 (bending stiffness)
- tau=0.53  (ν = (τ−0.5)/3 = 0.01, Re = U·2R/ν = 100)
- Far-field: inlet velocity BC, outlet sponge, walls bounce-back
- Steps: 5000

Validation (Turek-Hron FSI2 case, Re=100)
-----------------------------------------
1. Beam oscillates periodically (vortex-induced)
2. Tip amplitude A_y in range 0.02–0.06 (lattice units)
3. Oscillation frequency ≈ vortex shedding frequency

Run
---
    PYTHONPATH=src python examples/benchmark_turek_hron.py --device cpu --steps 5000
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cylinder_markers(n_markers: int, R: float, cx: float, cy: float,
                     cz: float, device: torch.device):
    """Lagrangian marker points on a circular cylinder cross-section."""
    theta = torch.linspace(0.0, 2.0 * math.pi, n_markers + 1, device=device)[:-1]
    mx = cx + R * torch.cos(theta)
    my = cy + R * torch.sin(theta)
    mz = torch.full_like(mx, float(cz))
    return mx, my, mz


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


def bounce_back_y_walls(f: torch.Tensor) -> torch.Tensor:
    """Full-way bounce-back on top/bottom walls (y=0, y=ny-1).

    For quasi-2D (nz=1) only the y-boundaries are treated; z-boundaries
    are skipped to avoid overwriting the entire domain.
    """
    opp = OPPOSITE.to(f.device)
    f_opp = f[opp]
    f_out = f.clone()
    f_out[:, :, 0, :] = f_opp[:, :, 0, :]
    f_out[:, :, -1, :] = f_opp[:, :, -1, :]
    return f_out


def compute_vorticity_z(ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """z-vorticity  ω_z = ∂u_y/∂x − ∂u_x/∂y  for (nz, ny, nx) fields."""
    dux_dy = torch.zeros_like(ux)
    duy_dx = torch.zeros_like(uy)
    dux_dy[:, 1:-1, :] = 0.5 * (ux[:, 2:, :] - ux[:, :-2, :])
    duy_dx[:, :, 1:-1] = 0.5 * (uy[:, :, 2:] - uy[:, :, :-2])
    return duy_dx - dux_dy


# ---------------------------------------------------------------------------
# Discrete elastic beam (biharmonic / Laplacian operator)
# ---------------------------------------------------------------------------

def compute_beam_forces(
    pos_x: torch.Tensor,
    pos_y: torch.Tensor,
    vel_x: torch.Tensor,
    vel_y: torch.Tensor,
    k_b: float,
    c_b: float,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Internal forces on beam nodes from discrete biharmonic operator.

    Bending (discrete biharmonic, 5-point stencil for 4th derivative):
        F_i = -k_b * (x_{i-2} - 4*x_{i-1} + 6*x_i - 4*x_{i+1} + x_{i+2})

    Damping (same operator on velocities):
        F_i += -c_b * (v_{i-2} - 4*v_{i-1} + 6*v_i - 4*v_{i+1} + v_{i+2})

    Boundary conditions (ghost nodes):
        - Clamped end (node 0):   du/dx=0  →  x_{-1} = x_1
        - Free end (node N-1):    d²u/dx²=0  →  x_N = 2*x_{N-1} - x_{N-2}
                                  (bending force at free end = 0)

    Note: The zero-shear (d³u/dx³=0) condition is NOT enforced because it
    produces a negative-eigenvalue (unstable) mode in the discrete operator.
    The zero-moment condition alone gives a stable, positive-semi-definite
    stiffness matrix.

    The formula is applied independently to x and y components.

    Returns (fx, fy) tensors of shape (N,) with zero at node 0 (clamped)
    and node N-1 (free end — only external IBM force applies there).
    """
    fx = torch.zeros(N, dtype=pos_x.dtype, device=pos_x.device)
    fy = torch.zeros(N, dtype=pos_y.dtype, device=pos_y.device)

    if N < 4:
        # Too few nodes for biharmonic stencil; use Laplacian fallback
        if N >= 3:
            fx[1:N - 1] = -k_b * (2.0 * pos_x[1:N - 1] - pos_x[0:N - 2] - pos_x[2:N])
            fy[1:N - 1] = -k_b * (2.0 * pos_y[1:N - 1] - pos_y[0:N - 2] - pos_y[2:N])
            fx[1:N - 1] -= c_b * (2.0 * vel_x[1:N - 1] - vel_x[0:N - 2] - vel_x[2:N])
            fy[1:N - 1] -= c_b * (2.0 * vel_y[1:N - 1] - vel_y[0:N - 2] - vel_y[2:N])
        return fx, fy

    # --- Ghost nodes ---
    # Clamped end (node 0): zero slope → x_{-1} = x_1
    gl_x = pos_x[1:2]
    gl_y = pos_y[1:2]
    gl_vx = vel_x[1:2]
    gl_vy = vel_y[1:2]

    # Free end (node N-1): zero moment → x_N = 2*x_{N-1} - x_{N-2}
    gr_x = (2.0 * pos_x[-1] - pos_x[-2]).unsqueeze(0)
    gr_y = (2.0 * pos_y[-1] - pos_y[-2]).unsqueeze(0)
    gr_vx = (2.0 * vel_x[-1] - vel_x[-2]).unsqueeze(0)
    gr_vy = (2.0 * vel_y[-1] - vel_y[-2]).unsqueeze(0)

    # --- Padded arrays: [ghost_left, real_nodes..., ghost_right] ---
    # Length = N + 2
    px = torch.cat([gl_x, pos_x, gr_x])
    py = torch.cat([gl_y, pos_y, gr_y])
    pvx = torch.cat([gl_vx, vel_x, gr_vx])
    pvy = torch.cat([gl_vy, vel_y, gr_vy])

    # --- Biharmonic operator for nodes 1 to N-2 ---
    # B_i = x_{i-2} - 4*x_{i-1} + 6*x_i - 4*x_{i+1} + x_{i+2}
    # In padded array, real node i is at index i+1, so:
    # B[1:N-1] = px[0:N-2] - 4*px[1:N-1] + 6*px[2:N] - 4*px[3:N+1] + px[4:N+2]
    bend_x = (
        px[0:N - 2] - 4.0 * px[1:N - 1] + 6.0 * px[2:N]
        - 4.0 * px[3:N + 1] + px[4:N + 2]
    )
    bend_y = (
        py[0:N - 2] - 4.0 * py[1:N - 1] + 6.0 * py[2:N]
        - 4.0 * py[3:N + 1] + py[4:N + 2]
    )
    damp_x = (
        pvx[0:N - 2] - 4.0 * pvx[1:N - 1] + 6.0 * pvx[2:N]
        - 4.0 * pvx[3:N + 1] + pvx[4:N + 2]
    )
    damp_y = (
        pvy[0:N - 2] - 4.0 * pvy[1:N - 1] + 6.0 * pvy[2:N]
        - 4.0 * pvy[3:N + 1] + pvy[4:N + 2]
    )

    # The biharmonic operator is positive at a positive deflection maximum;
    # -k*B is restoring.  Its velocity counterpart must have the same sign
    # for damping, not the former anti-damping sign.
    fx[1:N - 1] = -k_b * bend_x - c_b * damp_x
    fy[1:N - 1] = -k_b * bend_y - c_b * damp_y

    # Node 0: clamped, force = 0
    # Node N-1: free end, bending force = 0 (only external IBM force)

    return fx, fy


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------

def run_turek_hron_benchmark(
    device: str = "cpu",
    n_steps: int = 5000,
    nx: int = 300,
    ny: int = 100,
    R: float = 5.0,
    u_in: float = 0.1,
    tau: float = 0.53,
    beam_L: float = 35.0,
    beam_h: float = 2.0,
    beam_N: int = 20,
    rho_s: float = 10.0,
    E_bend: float = 1e4,
    c_bend: float = 100.0,
    n_cyl_markers: int = 32,
    sponge_width: int = 40,
    n_substeps: int = 50,
    ramp_steps: int = 500,
    ibm_relax: float = 0.5,
    kernel: str = "4pt",
    output_interval: int = 500,
    output_dir: str = "outputs",
):
    dev = torch.device(device)
    nz = 1
    D = 2.0 * R
    cx0 = 60.0           # cylinder centre x
    cy0 = ny * 0.5       # cylinder centre y  (=50)
    cz0 = 0.0            # single z-layer

    # --- Lattice viscosity / relaxation -------------------------------
    nu_lat = (tau - 0.5) / 3.0
    Re_actual = u_in * D / nu_lat if nu_lat > 0 else float("inf")

    # --- Beam parameters ----------------------------------------------
    n_seg = beam_N - 1
    L_node = beam_L / n_seg
    rho_f = 1.0
    rho_solid = rho_s * rho_f
    m_node = rho_solid * beam_h * L_node          # mass per node
    k_b = E_bend                                   # bending stiffness
    c_b = c_bend                                   # damping coefficient

    # Expected vortex shedding (Strouhal ≈ 0.2 for cylinder at Re~100-600)
    St_ref = 0.2
    f_shed = St_ref * u_in / D
    T_shed = 1.0 / f_shed if f_shed > 0 else float("inf")

    # --- IBM markers: cylinder (fixed) + beam (elastic) ---------------
    mx_cyl, my_cyl, mz_cyl = cylinder_markers(
        n_cyl_markers, R, cx0, cy0, cz0, dev
    )
    ds_cyl = 2.0 * math.pi * R / n_cyl_markers

    # Beam nodes: from cylinder trailing edge, extending downstream
    beam_x0 = cx0 + R          # trailing edge x
    beam_y0 = cy0              # trailing edge y
    beam_pos_x = torch.tensor(
        [beam_x0 + i * L_node for i in range(beam_N)],
        dtype=torch.float32, device=dev,
    )
    beam_pos_y = torch.full((beam_N,), beam_y0, dtype=torch.float32, device=dev)
    beam_vel_x = torch.zeros(beam_N, dtype=torch.float32, device=dev)
    beam_vel_y = torch.zeros(beam_N, dtype=torch.float32, device=dev)

    # Small initial perturbation to break symmetry (trigger vortex shedding)
    beam_pos_y[1:] += 0.05 * torch.linspace(0, 1, beam_N - 1, device=dev)

    n_total = n_cyl_markers + beam_N

    # --- Initial flow: uniform + small perturbation -------------------
    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    # Small transverse perturbation to break symmetry and trigger shedding
    # Wavelength = cylinder diameter (2R) — matches vortex shedding scale
    yy_grid = torch.arange(ny, device=dev, dtype=torch.float32).view(1, ny, 1)
    uy0 = 0.01 * u_in * torch.sin(2.0 * math.pi * yy_grid / (2.0 * R))
    uy0 = uy0.expand(nz, ny, nx).contiguous()
    uz0 = torch.zeros_like(ux0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=dev)

    # --- Wake probe (downstream of beam, ~5D from cylinder) -----------
    probe_x = min(int(cx0 + 5.0 * D), nx - 2)
    probe_y = int(cy0)

    initial_mass = float(f.sum().item())

    # --- Storage ------------------------------------------------------
    tip_y_hist: list[float] = []
    tip_x_hist: list[float] = []
    tip_vy_hist: list[float] = []
    fy_beam_hist: list[float] = []
    uy_probe_hist: list[float] = []

    # --- Header -------------------------------------------------------
    print("=" * 70, flush=True)
    print("  Turek-Hron FSI2 基准测试 — 圆柱+弹性梁在通道流中的流固耦合", flush=True)
    print("  Turek & Hron (2006) — 刚性圆柱后附弹性梁的涡激振动", flush=True)
    print("=" * 70, flush=True)
    print(f"  网格:       {nx} × {ny} × {nz}  (准二维, D3Q19)", flush=True)
    print(f"  圆柱:       中心=({cx0:.0f},{cy0:.0f})  R={R}  D={D}", flush=True)
    print(f"  流动:       Re={Re_actual:.0f}  U={u_in}  ν={nu_lat:.6f}  τ={tau:.4f}", flush=True)
    print(f"  通道:       高度={ny}  入口速度BC  出口海绵层(宽度={sponge_width})", flush=True)
    print(f"  弹性梁:     L={beam_L}  h={beam_h}  N={beam_N}节点  L_node={L_node:.4f}", flush=True)
    print(f"  材料:       ρ_s={rho_s}  E_bend={E_bend}  m_node={m_node:.4f}", flush=True)
    print(f"  弹簧:       k_b={k_b:.1f}  c_b={c_b:.1f}", flush=True)
    print(f"  子步进:     n_sub={n_substeps}  dt_sub={1.0/n_substeps:.4f}", flush=True)
    print(f"  渐升:       ramp_steps={ramp_steps}", flush=True)
    print(f"  IBM松弛:   ibm_relax={ibm_relax}", flush=True)
    print(f"  预期:       St≈{St_ref}  f_shed={f_shed:.6f}  T≈{T_shed:.0f}步", flush=True)
    print(f"  IBM:        圆柱标记={n_cyl_markers}(ds={ds_cyl:.3f})  "
          f"梁标记={beam_N}(ds={L_node:.3f})  总标记={n_total}", flush=True)
    print(f"  内核:       '{kernel}'", flush=True)
    print(f"  运行:       步数={n_steps}  设备={device}", flush=True)
    print("=" * 70, flush=True)

    t0 = time.time()
    numerical_failure: str | None = None
    for step in range(1, n_steps + 1):
        # 流速渐升: 前ramp_steps步线性增加入口速度, 避免初始冲击
        ramp = min(float(step) / float(ramp_steps), 1.0) if ramp_steps > 0 else 1.0
        u_in_eff = u_in * ramp

        # --- 1. 宏观场 (碰撞前) ---------------------------------------
        rho, ux, uy, uz = macroscopic3d(f)
        if not (torch.isfinite(f).all().item()
                and torch.isfinite(rho).all().item()
                and torch.isfinite(beam_pos_x).all().item()
                and torch.isfinite(beam_pos_y).all().item()
                and torch.isfinite(beam_vel_x).all().item()
                and torch.isfinite(beam_vel_y).all().item()):
            numerical_failure = f"step {step}: non-finite fluid or beam state"
            print(f"  [数值失败] {numerical_failure}", flush=True)
            break

        # --- 2. IBM 直接力 --------------------------------------------
        # 合并圆柱+梁标记
        mx_all = torch.cat([mx_cyl, beam_pos_x], dim=0)
        my_all = torch.cat([my_cyl, beam_pos_y], dim=0)
        mz_all = torch.cat([mz_cyl, torch.full_like(beam_pos_x, cz0)], dim=0)

        # 插值梁节点处的流体速度 (用于松弛目标 + 流体力计算)
        u_mx_b, u_my_b, _ = interpolate_velocity_markers(
            ux, uy, uz, beam_pos_x, beam_pos_y,
            torch.full_like(beam_pos_x, cz0), kernel=kernel,
        )

        # 松弛目标速度: u_target = alpha*v_beam + (1-alpha)*u_interp
        # 降低IBM力, 稳定显式FSI耦合 (附加质量效应)
        u_tgt_bx = ibm_relax * beam_vel_x + (1.0 - ibm_relax) * u_mx_b
        u_tgt_by = ibm_relax * beam_vel_y + (1.0 - ibm_relax) * u_my_b

        # 目标速度: 圆柱=0 (固定), 梁=松弛目标
        u_t_x = torch.cat([
            torch.zeros(n_cyl_markers, device=dev, dtype=torch.float32),
            u_tgt_bx,
        ], dim=0)
        u_t_y = torch.cat([
            torch.zeros(n_cyl_markers, device=dev, dtype=torch.float32),
            u_tgt_by,
        ], dim=0)
        u_t_z = torch.zeros(n_total, device=dev, dtype=torch.float32)

        fx_grid, fy_grid, fz_grid = ibm_direct_forcing_3d_vec(
            ux, uy, uz, mx_all, my_all, mz_all,
            u_t_x, u_t_y, u_t_z,
            kernel=kernel,
        )

        # --- 3. 碰撞 (BGK + Guo体力) ----------------------------------
        f = collide_bgk3d_guo(f, tau, fx_grid, fy_grid, fz_grid)

        # --- 4. 流动 ---------------------------------------------------
        f = stream3d(f)

        # --- 5. 边界条件 ----------------------------------------------
        f = apply_inlet_velocity(f, u_in_eff)
        f = apply_outlet_sponge(f, u_in_eff, sponge_width)
        f = bounce_back_y_walls(f)

        # --- 6. 质量修正 ----------------------------------------------
        if step % 100 == 0:
            f = correct_mass3d(f, initial_mass)

        if not (torch.isfinite(f).all().item()
                and torch.isfinite(fx_grid).all().item()
                and torch.isfinite(fy_grid).all().item()):
            numerical_failure = f"step {step}: non-finite IBM force or post-boundary distribution"
            print(f"  [数值失败] {numerical_failure}", flush=True)
            break

        # --- 7. 梁节点流体力 (IBM反力) --------------------------------
        # 标记力 = u_target - u_interp; 流体力 = -标记力 * ds
        ds_beam = L_node
        F_hydro_x = -(u_tgt_bx - u_mx_b) * ds_beam
        F_hydro_y = -(u_tgt_by - u_my_b) * ds_beam

        # --- 8. 结构更新 (子步进半隐式Euler) ---------------------------
        # 弯曲弹簧较硬, 需要子步进以保持显式积分稳定性
        # (Courant条件: dt_sub < 1/sqrt(k_b/m_node))
        n_sub = n_substeps
        dt_sub = 1.0 / n_sub
        for _ in range(n_sub):
            F_int_x, F_int_y = compute_beam_forces(
                beam_pos_x, beam_pos_y, beam_vel_x, beam_vel_y,
                k_b, c_b, beam_N,
            )
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
            beam_pos_x[0] = beam_x0
            beam_pos_y[0] = beam_y0

        # 速度钳制: 防止数值不稳定导致梁飞出网格
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

        # --- 9. 记录 -------------------------------------------------
        tip_y = float(beam_pos_y[-1].item()) - beam_y0
        tip_x = float(beam_pos_x[-1].item()) - beam_x0 - beam_L
        tip_vy = float(beam_vel_y[-1].item())
        fy_total = float(F_hydro_y.sum().item())
        uy_probe = float(uy[0, probe_y, probe_x].item())

        tip_y_hist.append(tip_y)
        tip_x_hist.append(tip_x)
        tip_vy_hist.append(tip_vy)
        fy_beam_hist.append(fy_total)
        uy_probe_hist.append(uy_probe)

        # --- 10. 打印 ------------------------------------------------
        if step % output_interval == 0 or step == n_steps:
            print(
                f"  步 {step:5d}:  梁尖y={tip_y:+.4f}  vy={tip_vy:+.5f}  "
                f"Fy={fy_total:+.5f}  探针uy={uy_probe:+.5f}  "
                f"ρ∈[{float(rho.min()):.4f},{float(rho.max()):.4f}]  "
                f"{time.time()-t0:.0f}秒",
                flush=True,
            )

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
    # Analyse only completed samples; an aborted run is forced to FAIL below.
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
    f_shed_meas, _ = _dominant_freq(uy_ss)
    St_meas = f_shed_meas * D / u_in if u_in > 0 else 0.0

    A_tip = 0.5 * float(np.ptp(tip_y_ss)) if n_ss > 0 else 0.0
    freq_ratio = f_tip / f_shed_meas if f_shed_meas > 1e-12 else float("nan")

    # --- 验证报告 ---
    print(flush=True)
    print("=" * 70, flush=True)
    print("  验证结果", flush=True)
    print("=" * 70, flush=True)

    # 1. 周期性振荡
    print(f"  1. 周期性振荡 (涡致振动):", flush=True)
    print(f"     梁尖振幅 A_y = {A_tip:.4f} 格子单位", flush=True)
    oscillation_ok = A_tip > 0.005
    osc_err = 0.0 if oscillation_ok else (0.005 - A_tip) / 0.005 * 100
    print(f"     检查: {'通过' if oscillation_ok else '未通过'}  "
          f"(A_y > 0.005, 误差={osc_err:.1f}%)", flush=True)

    # 2. 振幅范围
    print(f"  2. 梁尖振幅范围:", flush=True)
    A_target_low = 0.02
    A_target_high = 0.06
    A_target_mid = (A_target_low + A_target_high) / 2.0
    print(f"     目标范围: [{A_target_low}, {A_target_high}] 格子单位", flush=True)
    print(f"     测量值:   A_y = {A_tip:.4f}", flush=True)
    if A_target_low <= A_tip <= A_target_high:
        amp_err = 0.0
        amp_ok = True
    else:
        nearest = A_target_low if A_tip < A_target_low else A_target_high
        amp_err = abs(A_tip - nearest) / nearest * 100
        amp_ok = False
    print(f"     检查: {'通过' if amp_ok else '未通过'}  (误差={amp_err:.1f}%)", flush=True)

    # 3. 频率匹配
    print(f"  3. 振荡频率 ≈ 涡脱落频率:", flush=True)
    print(f"     f_tip  = {f_tip:.6f} 周/步", flush=True)
    print(f"     f_shed = {f_shed_meas:.6f} 周/步  (St={St_meas:.4f})", flush=True)
    if not math.isnan(freq_ratio) and f_shed_meas > 1e-12:
        freq_err = abs(freq_ratio - 1.0) * 100
        freq_ok = freq_err < 50.0
    else:
        freq_err = 100.0
        freq_ok = False
    print(f"     比值   = {freq_ratio:.4f}  (≈1.0表示频率锁定)", flush=True)
    print(f"     检查: {'通过' if freq_ok else '未通过'}  "
          f"(|f_tip/f_shed−1|={abs(freq_ratio-1.0) if not math.isnan(freq_ratio) else 1.0:.3f}, "
          f"误差={freq_err:.1f}%)", flush=True)

    # --- 综合评估 ---
    finite_metrics = (numerical_failure is None and n_completed == n_steps
                      and all(np.isfinite(a).all() for a in
                              (tip_y_arr, tip_x_arr, uy_arr, fy_arr))
                      and all(math.isfinite(v) for v in
                              (A_tip, f_tip, f_shed_meas, St_meas, freq_ratio)))
    print(f"  数值完整性检查: {'通过' if finite_metrics else '未通过'}"
          f"  ({numerical_failure or '所有状态和指标均为有限值'})", flush=True)
    all_pass = finite_metrics and oscillation_ok and amp_ok and freq_ok
    print(flush=True)
    print(f"  周期振荡检查: {'通过' if oscillation_ok else '未通过'} "
          f"(A_y={A_tip:.4f} > 0.005)", flush=True)
    print(f"  振幅范围检查: {'通过' if amp_ok else '未通过'} "
          f"(A_y={A_tip:.4f}, 目标=[{A_target_low},{A_target_high}])", flush=True)
    print(f"  频率匹配检查: {'通过' if freq_ok else '未通过'} "
          f"(误差={freq_err:.1f}%)", flush=True)
    print(flush=True)
    print(f"  总体结果: {'通过 ✓' if all_pass else '未通过 ✗'}", flush=True)
    print("=" * 70, flush=True)

    # ===================================================================
    # 保存输出
    # ===================================================================
    os.makedirs(output_dir, exist_ok=True)

    # CSV时间序列
    csv_path = os.path.join(output_dir, "turek_hron_fsi2_data.csv")
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
        axes[0].set_ylabel("梁尖y位移 (格子单位)")
        axes[0].set_title("Turek-Hron FSI2: 弹性梁尖端横向位移")
        axes[0].axhline(0, color="k", linewidth=0.5)
        axes[0].axvline(n_trans, color="r", linestyle="--", linewidth=0.5,
                        label="瞬态结束")
        axes[0].legend(fontsize=8)

        axes[1].plot(uy_arr, "g-", linewidth=0.8)
        axes[1].set_ylabel(r"$u_y$ 探针 (格子单位)")
        axes[1].set_title(f"尾流速度探针 (x={probe_x}, y={probe_y})")
        axes[1].axhline(0, color="k", linewidth=0.5)

        axes[2].plot(fy_arr, "r-", linewidth=0.8)
        axes[2].set_ylabel(r"$F_{y}$ 梁流体力")
        axes[2].set_xlabel("时间步")
        axes[2].set_title("弹性梁横向流体力")
        axes[2].axhline(0, color="k", linewidth=0.5)

        plt.tight_layout()
        ts_path = os.path.join(output_dir, "turek_hron_timeseries.png")
        fig.savefig(ts_path, dpi=120)
        plt.close(fig)
        print(f"  已保存: {ts_path}", flush=True)

        # 涡量场快照
        rho_f, ux_f, uy_f, _ = macroscopic3d(f)
        vort = compute_vorticity_z(ux_f, uy_f)[0].cpu().numpy()
        vmax = max(abs(vort.min()), abs(vort.max()))
        vmax = max(vmax, 1e-6) * 0.8
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        im = ax2.imshow(vort, origin="lower", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax,
                        extent=(0.0, float(nx), 0.0, float(ny)))
        plt.colorbar(im, ax=ax2, label=r"$\omega_z$")
        ax2.plot(cx0, cy0, "ko", markersize=5, label="圆柱中心")
        bx = beam_pos_x.cpu().numpy()
        by = beam_pos_y.cpu().numpy()
        ax2.plot(bx, by, "r.-", markersize=3, linewidth=1, label="弹性梁")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_title(f"涡量场 (步 {n_steps})  St={St_meas:.3f}  "
                      f"A={A_tip:.4f}")
        ax2.legend(fontsize=8)
        plt.tight_layout()
        vort_path = os.path.join(output_dir, "turek_hron_vorticity.png")
        fig2.savefig(vort_path, dpi=120)
        plt.close(fig2)
        print(f"  已保存: {vort_path}", flush=True)

        # FFT频谱
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        y_spec = np.abs(np.fft.rfft(tip_y_ss - tip_y_ss.mean(), n=n_fft))
        uy_spec = np.abs(np.fft.rfft(uy_ss - uy_ss.mean(), n=n_fft))
        ax3a.semilogy(freqs, y_spec, "b-", linewidth=0.8)
        ax3a.axvline(f_tip, color="r", linestyle="--", linewidth=0.8,
                     label=f"f_tip={f_tip:.6f}")
        ax3a.set_ylabel("|FFT(梁尖y)|")
        ax3a.set_title("梁尖位移频谱")
        ax3a.legend(fontsize=8)

        ax3b.semilogy(freqs, uy_spec, "g-", linewidth=0.8)
        ax3b.axvline(f_shed_meas, color="r", linestyle="--", linewidth=0.8,
                     label=f"f_shed={f_shed_meas:.6f}")
        ax3b.set_ylabel(r"|FFT($u_y$)|")
        ax3b.set_xlabel("频率 (周/步)")
        ax3b.set_title("尾流速度频谱")
        ax3b.legend(fontsize=8)

        plt.tight_layout()
        fft_path = os.path.join(output_dir, "turek_hron_spectra.png")
        fig3.savefig(fft_path, dpi=120)
        plt.close(fig3)
        print(f"  已保存: {fft_path}", flush=True)

    except ImportError:
        print("  (matplotlib不可用 — 跳过图表)", flush=True)

    return {
        "St_measured": St_meas,
        "St_expected": St_ref,
        "f_tip": f_tip,
        "f_shed": f_shed_meas,
        "freq_ratio": freq_ratio,
        "amplitude": A_tip,
        "amplitude_D": A_tip / D,
        "Re_actual": Re_actual,
        "numerical_failure": numerical_failure,
        "steps_completed": n_completed,
        "pass": all_pass,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Turek-Hron FSI2基准: 圆柱+弹性梁在通道流中的流固耦合"
    )
    parser.add_argument("--device", default="cpu",
                        help="设备: cpu / cuda / sdaa:N")
    parser.add_argument("--steps", type=int, default=5000,
                        help="LBM时间步数")
    parser.add_argument("--nx", type=int, default=300, help="网格x方向")
    parser.add_argument("--ny", type=int, default=100, help="网格y方向")
    parser.add_argument("--R", type=float, default=5.0,
                        help="圆柱半径 (格子单位)")
    parser.add_argument("--u-in", dest="u_in", type=float, default=0.1,
                        help="入口速度 (格子单位)")
    parser.add_argument("--tau", type=float, default=0.53,
                        help="BGK松弛时间 τ (ν=(τ−0.5)/3, Re=100时τ=0.53)")
    parser.add_argument("--beam-L", dest="beam_L", type=float, default=35.0,
                        help="梁长度 (格子单位)")
    parser.add_argument("--beam-h", dest="beam_h", type=float, default=2.0,
                        help="梁厚度 (格子单位)")
    parser.add_argument("--beam-N", dest="beam_N", type=int, default=20,
                        help="梁节点数")
    parser.add_argument("--rho-s", dest="rho_s", type=float,
                        default=10.0, help="密度比 ρ_s/ρ_f")
    parser.add_argument("--E-bend", dest="E_bend", type=float,
                        default=1e4, help="弯曲刚度 k_b")
    parser.add_argument("--c-bend", dest="c_bend", type=float,
                        default=100.0, help="弯曲阻尼系数 c_b")
    parser.add_argument("--n-cyl-markers", dest="n_cyl_markers",
                        type=int, default=32, help="圆柱表面IBM标记数")
    parser.add_argument("--sponge-width", dest="sponge_width",
                        type=int, default=40, help="出口海绵层宽度")
    parser.add_argument("--n-substeps", dest="n_substeps",
                        type=int, default=50, help="结构更新子步数")
    parser.add_argument("--ramp-steps", dest="ramp_steps",
                        type=int, default=500, help="入口流速渐升步数")
    parser.add_argument("--ibm-relax", dest="ibm_relax",
                        type=float, default=0.5, help="IBM耦合松弛因子(0-1)")
    parser.add_argument("--kernel", default="4pt", choices=["hat", "4pt"],
                        help="IBM delta内核")
    parser.add_argument("--output-interval", dest="output_interval",
                        type=int, default=500, help="打印间隔 (步)")
    parser.add_argument("--output-dir", dest="output_dir",
                        default="outputs", help="输出目录")
    args = parser.parse_args()

    run_turek_hron_benchmark(
        device=args.device,
        n_steps=args.steps,
        nx=args.nx,
        ny=args.ny,
        R=args.R,
        u_in=args.u_in,
        tau=args.tau,
        beam_L=args.beam_L,
        beam_h=args.beam_h,
        beam_N=args.beam_N,
        rho_s=args.rho_s,
        E_bend=args.E_bend,
        c_bend=args.c_bend,
        n_cyl_markers=args.n_cyl_markers,
        sponge_width=args.sponge_width,
        n_substeps=args.n_substeps,
        ramp_steps=args.ramp_steps,
        ibm_relax=args.ibm_relax,
        kernel=args.kernel,
        output_interval=args.output_interval,
        output_dir=args.output_dir,
    )
