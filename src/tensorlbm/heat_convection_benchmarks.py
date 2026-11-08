"""LBM 热对流验证面：Rayleigh-Bénard 自然对流 + 强制对流绕圆柱。

战役 P67-N01 交付：双分布函数 (DDF) 热 LBM 在两个经典传热基准上的验证。

基准 1: Rayleigh-Bénard 自然对流
  - 几何：2D 方腔，底壁 T_hot，顶壁 T_cold，侧壁周期
  - 物理：Boussinesq 近似，Ra = gβΔTH³/(να)
  - 验证：Nu(Ra) 标度律，层流区 Nu ~ Ra^{1/3}（或 Nu~Ra^{1/4} 取决于区域）
  - 对标：de Vahl Davis (1983) 基准值

基准 2: 强制对流绕圆柱
  - 几何：2D 通道，中心放置圆柱，来流 U∞
  - 物理：Re = U∞D/ν, Pr = ν/α
  - 验证：平均 Nu vs Churchill-Bernstein 经验式
  - Churchill-Bernstein (1977):
    Nu = 0.3 + 0.62 Re^{1/2} Pr^{1/3} / [1+(0.4/Pr)^{2/3}]^{1/4}
         × [1 + (Re/282000)^{5/8}]^{4/5}

References
----------
de Vahl Davis, G. (1983). Natural convection of air in a square cavity.
    Int. J. Numer. Meth. Fluids, 3, 249-264.
Churchill, S. W., & Bernstein, M. (1977). A correlated equation for the
    forced convection from gases and liquids to a circular cylinder in
    crossflow. J. Heat Transfer, 99(2), 300-306.
"""

from __future__ import annotations

import time
from typing import Any

import torch

from .d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from .solver import stream
from .thermal import (
    C5,
    W5,
    OPPOSITE5,
    apply_buoyancy_force,
    buoyancy_force,
    collide_bgk_force,
    collide_thermal_bgk,
    pre_streaming_bounce_back,
    temperature_equilibrium,
    temperature_stream,
)

__all__ = [
    "simulate_rayleigh_benard",
    "simulate_forced_convection_cylinder",
    "churchill_bernstein",
    "de_vahl_davis_reference",
]


# ============================================================================
# Reference data
# ============================================================================

def de_vahl_davis_reference(ra: float) -> dict[str, float]:
    """de Vahl Davis (1983) 基准值：自然对流方腔 Nu(Ra)。

    注意：de Vahl Davis 的基准是左热右冷方腔（与 RB 不同几何），
    此处提供的是其 Nu 值作为量级参考。RB 的 Nu 标度律不同。

    Ra=1e3: Nu ≈ 1.118
    Ra=1e4: Nu ≈ 2.243
    Ra=1e5: Nu ≈ 4.519
    Ra=1e6: Nu ≈ 8.800
    """
    ref = {
        1e3: 1.118,
        1e4: 2.243,
        1e5: 4.519,
        1e6: 8.800,
    }
    ra_key = min(ref.keys(), key=lambda k: abs(k - ra))
    return {"ra": ra_key, "nu_ref": ref[ra_key]}


def churchill_bernstein(re: float, pr: float) -> float:
    """Churchill-Bernstein (1977) 经验式：强制对流绕圆柱平均 Nu。

    Nu = 0.3 + 0.62 Re^{1/2} Pr^{1/3}
         / [1 + (0.4/Pr)^{2/3}]^{1/4}
         × [1 + (Re/282000)^{5/8}]^{4/5}

    适用范围：Re Pr > 0.2（低 Re 也适用，Re < 282000 时末项 ≈ 1）。
    """
    if re * pr < 0.2:
        return 1.0
    term1 = 0.62 * re**0.5 * pr**(1.0/3.0)
    term2 = (1.0 + (0.4 / pr)**(2.0/3.0))**(0.25)
    term3 = (1.0 + (re / 282000.0)**(5.0/8.0))**(4.0/5.0)
    return 0.3 + term1 / term2 * term3


# ============================================================================
# Rayleigh-Bénard convection
# ============================================================================

def _rb_wall_mask(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """RB 几何：上下壁 no-slip，左右周期（无壁）。"""
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True   # bottom wall
    mask[-1, :] = True  # top wall
    return mask


def _rb_thermal_bc_zou_he(
    g: torch.Tensor, t_hot: float, t_cold: float, 
    ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """RB 温度边界：底壁 T_hot，顶壁 T_cold，左右周期。

    使用非平衡修正的 Dirichlet BC（与 thermal.py 的等温壁同构）：
    - 壁节点取 equilibrium(T_wall, u=0)
    - 指向流体的群体叠加非平衡修正（从相邻流体节点获取）
    """
    w = W5.to(device=g.device, dtype=g.dtype)
    g = g.clone()
    
    # Bottom wall (y=0): T = t_hot
    # Get non-equilibrium from first fluid row (y=1)
    T_fluid_bot = g[:, 1, :].sum(dim=0)  # temperature at y=1
    neq_bot = g[:, 1, :] - w.view(5, 1) * T_fluid_bot.unsqueeze(0)
    
    # Set wall to equilibrium + non-equilibrium correction
    g[:, 0, :] = w.view(5, 1) * t_hot
    # Add non-equilibrium from fluid (for directions pointing into fluid: i=3)
    g[3, 0, :] = g[3, 0, :] + neq_bot[3, :]
    # Subtract from rest direction to conserve sum
    g[0, 0, :] = g[0, 0, :] - neq_bot[3, :]
    
    # Top wall (y=ny-1): T = t_cold
    T_fluid_top = g[:, -2, :].sum(dim=0)  # temperature at y=ny-2
    neq_top = g[:, -2, :] - w.view(5, 1) * T_fluid_top.unsqueeze(0)
    
    g[:, -1, :] = w.view(5, 1) * t_cold
    # Add non-equilibrium from fluid (for directions pointing into fluid: i=4)
    g[4, -1, :] = g[4, -1, :] + neq_top[4, :]
    g[0, -1, :] = g[0, -1, :] - neq_top[4, :]
    
    return g


def _nusselt_rb(T: torch.Tensor, ny: int, dT: float) -> dict[str, float]:
    """RB Nusselt 数：Nu = -∂T/∂y · H / ΔT，在底壁和顶壁分别计算后平均。

    H = ny - 1（壁面间距，壁面位于节点 y=0 和 y=ny-1）。
    使用二阶单侧差分。
    """
    H = float(ny - 1)  # Wall-to-wall distance in lattice units
    
    # Bottom wall: ∂T/∂y ≈ (−3T[0] + 4T[1] − T[2]) / (2·Δy)
    dTdy_bot = (-1.5 * T[0, :] + 2.0 * T[1, :] - 0.5 * T[2, :])
    nu_bot = -dTdy_bot.mean().item() * H / dT

    # Top wall: ∂T/∂y ≈ (3T[-1] − 4T[-2] + T[-3]) / (2·Δy)
    dTdy_top = (1.5 * T[-1, :] - 2.0 * T[-2, :] + 0.5 * T[-3, :])
    nu_top = -dTdy_top.mean().item() * H / dT

    return {
        "nu_bot": nu_bot,
        "nu_top": nu_top,
        "nu": 0.5 * (nu_bot + nu_top),
    }


def _energy_balance_rb(g: torch.Tensor, ny: int, nx: int) -> dict[str, float]:
    """RB 能量守恒诊断：底壁入热流 vs 顶壁出热流。"""
    T = g.sum(dim=0)
    H = float(ny - 1)
    
    # Bottom wall heat flux (into domain): q_bot = -∂T/∂y|_{y=0}
    dTdy_bot = (-1.5 * T[0, :] + 2.0 * T[1, :] - 0.5 * T[2, :]).mean().item()
    q_bot = -dTdy_bot

    # Top wall heat flux (out of domain): q_top = -∂T/∂y|_{y=ny-1}
    dTdy_top = (1.5 * T[-1, :] - 2.0 * T[-2, :] + 0.5 * T[-3, :]).mean().item()
    q_top = -dTdy_top

    # Relative imbalance
    q_avg = 0.5 * (abs(q_bot) + abs(q_top))
    if q_avg > 1e-10:
        imbalance = abs(q_bot - q_top) / q_avg
    else:
        imbalance = 0.0

    return {"q_bot": q_bot, "q_top": q_top, "imbalance": imbalance}


def simulate_rayleigh_benard(
    nx: int,
    ny: int | None = None,
    ra: float = 1e4,
    pr: float = 0.71,
    tau: float = 0.6,
    t_hot: float = 1.0,
    t_cold: float = 0.0,
    steps: int = 100000,
    device: str = "cpu",
    report_every: int = 5000,
    enable_buoyancy: bool = True,
) -> dict[str, Any]:
    """Rayleigh-Bénard 自然对流 DDF-LBM 模拟。

    几何：2D 矩形，底壁 T_hot，顶壁 T_cold，左右周期。
    重力向下，浮力向上（热流体上升）。

    Args:
        nx: 水平方向格点数。
        ny: 垂直方向格点数（默认 = nx，即方腔）。
        ra: Rayleigh 数。
        pr: Prandtl 数。
        tau: 动量松弛时间。
        t_hot: 底壁温度。
        t_cold: 顶壁温度。
        steps: 总步数。
        device: 计算设备。
        report_every: 报告间隔步数。
        enable_buoyancy: 是否开启浮力（False = 纯导热负控）。

    Returns:
        dict: 包含 Nu、能量平衡、收敛历史等。
    """
    device_t = torch.device(device)
    if ny is None:
        ny = nx
    H = float(ny - 1)  # Wall-to-wall distance
    dT = t_hot - t_cold

    # Lattice parameters
    nu_lat = (tau - 0.5) / 3.0
    alpha_lat = nu_lat / pr
    tau_T = 3.0 * alpha_lat + 0.5
    g_beta = ra * nu_lat * alpha_lat / (H**3)  # ΔT = 1

    # Initialize fields
    rho0 = torch.ones((ny, nx), device=device_t)
    u0 = torch.zeros((ny, nx), device=device_t)
    f = equilibrium(rho0, u0, u0)

    # Temperature: linear profile (conductive base state) + small perturbation
    T0 = torch.linspace(t_hot, t_cold, ny, device=device_t).unsqueeze(1).expand(ny, nx).contiguous()
    # Add small random perturbation to break symmetry and trigger convection
    torch.manual_seed(42)
    T0 = T0 + 0.01 * (torch.rand(ny, nx, device=device_t) - 0.5) * dT
    g = temperature_equilibrium(T0, u0, u0)

    wall = _rb_wall_mask(ny, nx, device_t)
    interior = ~wall

    nu_history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)

        # Buoyancy force
        if enable_buoyancy:
            F = buoyancy_force(rho, T, g_beta, t_ref=0.5 * (t_hot + t_cold))
        else:
            F = torch.zeros(2, ny, nx, device=device_t)

        # Temperature: collide → stream → BC
        g = collide_thermal_bgk(g, T, ux, uy, tau_T)
        g = temperature_stream(g)
        g = _rb_thermal_bc_zou_he(g, t_hot, t_cold, ux, uy)

        # Velocity: collide (with Guo force) → pre-streaming bounce-back → stream
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)

        # Diagnostics
        if step % report_every == 0 or step == steps:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nu_rb = _nusselt_rb(T, ny, dT)
            eb = _energy_balance_rb(g, ny, nx)
            u_max = float(torch.sqrt(ux**2 + uy**2).max().item())
            nu_history.append({
                "step": step,
                "nu": nu_rb["nu"],
                "nu_bot": nu_rb["nu_bot"],
                "nu_top": nu_rb["nu_top"],
                "energy_imbalance": eb["imbalance"],
                "u_max": u_max,
            })

    elapsed = time.time() - t0

    # Final diagnostics
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    nu_final = _nusselt_rb(T, ny, dT)
    eb_final = _energy_balance_rb(g, ny, nx)

    return {
        "nx": nx,
        "ny": ny,
        "ra": ra,
        "pr": pr,
        "tau": tau,
        "tau_T": tau_T,
        "nu_lat": nu_lat,
        "alpha_lat": alpha_lat,
        "g_beta": g_beta,
        "H": H,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "nu": nu_final["nu"],
        "nu_bot": nu_final["nu_bot"],
        "nu_top": nu_final["nu_top"],
        "energy_imbalance": eb_final["imbalance"],
        "u_max": float(torch.sqrt(ux**2 + uy**2).max().item()),
        "T_min": float(T.min().item()),
        "T_max": float(T.max().item()),
        "nu_history": nu_history,
        "enable_buoyancy": enable_buoyancy,
    }


# ============================================================================
# Forced convection around a cylinder
# ============================================================================

def _cylinder_mask(ny: int, nx: int, cy: int, cx: int, R: int,
                   device: torch.device) -> torch.Tensor:
    """圆柱固体掩码。"""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij",
    )
    return ((xx - cx)**2 + (yy - cy)**2) <= R**2


def _cylinder_thermal_bc(
    g: torch.Tensor,
    solid: torch.Tensor,
    t_cyl: float,
) -> torch.Tensor:
    """圆柱表面等温边界：固体节点取 equilibrium(T_cyl, u=0)。

    使用非平衡修正把有效等温面拉到壁面。
    """
    w = W5.to(device=g.device, dtype=g.dtype)
    g = g.clone()
    
    # Find surface nodes: solid nodes adjacent to fluid
    fluid = ~solid
    # For each solid node, check if it has fluid neighbors
    has_fluid_neighbor = torch.zeros_like(solid)
    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        shifted_fluid = torch.roll(fluid, shifts=(di, dj), dims=(0, 1))
        has_fluid_neighbor |= shifted_fluid
    
    surface = solid & has_fluid_neighbor
    
    # Set surface nodes to equilibrium at T_cyl
    T_wall = torch.full_like(g[0], t_cyl)
    geq_wall = w.view(5, 1, 1) * T_wall.unsqueeze(0)
    surface_exp = surface.unsqueeze(0).expand(5, -1, -1)
    g = torch.where(surface_exp, geq_wall, g)
    
    return g


def _channel_inlet_thermal(
    g: torch.Tensor, t_in: float, ny: int, device: torch.device
) -> torch.Tensor:
    """通道入口温度边界：T = t_in。"""
    w = W5.to(device=g.device, dtype=g.dtype)
    g = g.clone()
    g[:, :, 0] = w.view(5, 1) * t_in
    return g


def _channel_outlet_thermal(g: torch.Tensor) -> torch.Tensor:
    """通道出口：零梯度（复制倒数第二列）。"""
    g = g.clone()
    g[:, :, -1] = g[:, :, -2].clone()
    return g


def _channel_wall_thermal_adiabatic(g: torch.Tensor) -> torch.Tensor:
    """通道上下壁绝热：反射法向分布。"""
    g = g.clone()
    # Bottom wall (y=0): reflect downward → upward
    g[3, 0, :] = g[4, 0, :].clone()
    # At top wall, reflect upward → downward
    g[4, -1, :] = g[3, -1, :].clone()
    return g


def _avg_nusselt_cylinder(
    T: torch.Tensor,
    solid: torch.Tensor,
    t_cyl: float,
    t_in: float,
    D: float,
) -> float:
    """圆柱平均 Nusselt 数。

    通过固体节点相邻流体节点的温度梯度计算局部热流，
    然后沿圆柱表面积分平均。

    Nu = h·D/k，其中 h = q/(T_cyl - T_in)，q = -k ∂T/∂n|_wall
    """
    dT = t_cyl - t_in
    if abs(dT) < 1e-12:
        return 0.0

    ny, nx = T.shape
    fluid = ~solid

    # Find fluid nodes adjacent to solid (surface-adjacent fluid nodes)
    fluid_near_solid = fluid.clone()
    fluid_near_solid[1:-1, 1:-1] &= (
        solid[2:, 1:-1] | solid[:-2, 1:-1] |
        solid[1:-1, 2:] | solid[1:-1, :-2]
    )

    # Compute temperature gradient at these nodes
    dTdx = torch.zeros_like(T)
    dTdy = torch.zeros_like(T)
    dTdx[:, 1:-1] = (T[:, 2:] - T[:, :-2]) / 2.0
    dTdx[:, 0] = T[:, 1] - T[:, 0]
    dTdx[:, -1] = T[:, -1] - T[:, -2]
    dTdy[1:-1, :] = (T[2:, :] - T[:-2, :]) / 2.0
    dTdy[0, :] = T[1, :] - T[0, :]
    dTdy[-1, :] = T[-1, :] - T[-2, :]

    # Heat flux magnitude at surface
    grad_mag = torch.sqrt(dTdx**2 + dTdy**2)
    q_surface = grad_mag[fluid_near_solid]

    if q_surface.numel() > 0:
        q_mean = q_surface.mean().item()
        h = q_mean / abs(dT)
        nu = h * D
    else:
        nu = 0.0

    return nu


def simulate_forced_convection_cylinder(
    nx: int,
    ny: int,
    re: float = 100.0,
    pr: float = 0.7,
    tau: float = 0.55,
    u_in: float = 0.1,
    t_cyl: float = 1.0,
    t_in: float = 0.0,
    steps: int = 50000,
    device: str = "cpu",
    report_every: int = 2000,
    cyl_frac_x: float = 0.25,
    cyl_frac_y: float = 0.5,
    cyl_frac_r: float = 0.05,
) -> dict[str, Any]:
    """强制对流绕圆柱 DDF-LBM 模拟。

    几何：2D 通道，中心放置圆柱。
    入口：均匀来流 U∞，T = t_in
    出口：零梯度（outflow）
    上下壁：无滑移 + 绝热
    圆柱表面：无滑移 + 等温 T = t_cyl
    """
    device_t = torch.device(device)
    dT = t_cyl - t_in

    # Cylinder geometry
    R = int(cyl_frac_r * ny)
    D = 2.0 * R
    cx = int(cyl_frac_x * nx)
    cy = int(cyl_frac_y * ny)

    # Lattice parameters from Re
    nu_lat = u_in * D / re
    tau_calc = 3.0 * nu_lat + 0.5
    if abs(tau - 0.5) < 1e-6:
        tau = tau_calc
    nu_lat = (tau - 0.5) / 3.0
    alpha_lat = nu_lat / pr
    tau_T = 3.0 * alpha_lat + 0.5

    # Solid mask
    solid = _cylinder_mask(ny, nx, cy, cx, R, device_t)
    # Wall mask: top and bottom walls + cylinder
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device_t)
    wall[0, :] = True
    wall[-1, :] = True
    wall |= solid

    # Initialize
    rho0 = torch.ones((ny, nx), device=device_t)
    ux0 = torch.full((ny, nx), u_in, device=device_t)
    uy0 = torch.zeros((ny, nx), device=device_t)
    f = equilibrium(rho0, ux0, uy0)

    T0 = torch.full((ny, nx), t_in, device=device_t)
    T0 = torch.where(solid, torch.full_like(T0, t_cyl), T0)
    g = temperature_equilibrium(T0, ux0, uy0)

    nu_history = []
    t0 = time.time()

    for step in range(1, steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)

        # Temperature: collide → stream → BC
        g = collide_thermal_bgk(g, T, ux, uy, tau_T)
        g = temperature_stream(g)
        # Thermal BCs
        g = _cylinder_thermal_bc(g, solid, t_cyl)
        g = _channel_inlet_thermal(g, t_in, ny, device_t)
        g = _channel_outlet_thermal(g)
        g = _channel_wall_thermal_adiabatic(g)

        # Velocity: collide → bounce-back on walls/cylinder → stream
        f_pre = f
        feq = equilibrium(rho, ux, uy)
        f = f - (f - feq) / tau
        opp = OPPOSITE.to(device_t)
        f = torch.where(wall.unsqueeze(0), f_pre[opp], f)
        f = stream(f)

        # Inlet BC for velocity (Zou-He style)
        rho_in = torch.ones(ny, 1, device=device_t)
        ux_in = torch.full((ny, 1), u_in, device=device_t)
        uy_in = torch.zeros(ny, 1, device=device_t)
        f[:, :, 0] = equilibrium(rho_in, ux_in, uy_in).squeeze(-1)
        
        # Outlet: zero gradient
        f[:, :, -1] = f[:, :, -2].clone()

        # Diagnostics
        if step % report_every == 0 or step == steps:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nu_cyl = _avg_nusselt_cylinder(T, solid, t_cyl, t_in, D)
            u_max = float(torch.sqrt(ux**2 + uy**2).max().item())
            nu_history.append({
                "step": step,
                "nu": nu_cyl,
                "u_max": u_max,
            })

    elapsed = time.time() - t0

    # Final
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    nu_final = _avg_nusselt_cylinder(T, solid, t_cyl, t_in, D)
    nu_cb = churchill_bernstein(re, pr)

    return {
        "nx": nx,
        "ny": ny,
        "re": re,
        "pr": pr,
        "tau": tau,
        "tau_T": tau_T,
        "nu_lat": nu_lat,
        "alpha_lat": alpha_lat,
        "u_in": u_in,
        "D": D,
        "R": R,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "nu_avg": nu_final,
        "nu_churchill_bernstein": nu_cb,
        "nu_relative_error": abs(nu_final - nu_cb) / nu_cb if nu_cb > 0 else 0.0,
        "u_max": float(torch.sqrt(ux**2 + uy**2).max().item()),
        "T_min": float(T.min().item()),
        "T_max": float(T.max().item()),
        "nu_history": nu_history,
    }
