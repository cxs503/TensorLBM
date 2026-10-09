"""W2-A 共享驱动库（纯胶水）：de Vahl Davis 自然对流方腔。

物理全部调用 tensorlbm 共性模块入口（d2q9 / solver / thermal），
本文件只做编排、参数换算与观测量提取。口径约定见 NOTES.md 预注册：
- H = nx−1（等温壁位于节点 x=0 / x=nx−1 上，与 thermal3d 的 length=nx−1 同构）
- Nu：库 nusselt_number mode='grad1'（壁节点−首内邻差分），整壁含角点平均
- u_max/v_max：竖直/水平中线（x=y=H/2，相邻节点线性插值）上的分量峰值，
  无量纲化用扩散尺度 u_nd = u_lu·(H_lu/α_lu)（DVD 约定）
"""

from __future__ import annotations

import math

import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.solver import stream
from tensorlbm.thermal import (
    apply_temperature_boundaries,
    buoyancy_force,
    cavity_wall_mask,
    collide_bgk_force,
    nusselt_number,
    pre_streaming_bounce_back,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)
from tensorlbm.thermal import thermal_params as _lib_thermal_params


def lattice_params(nx: int, ra: float, pr: float, tau: float) -> dict[str, float]:
    """Ra/Pr/τ → 格子参数（H = nx−1 约定，见模块 docstring 与 NOTES.md）。

    ν/α/τ_T 与库 thermal_params 逐项一致；仅 H 与由此反解的 gβ 采用
    nx−1（等温节点间距），并与 Nu 的 H 保持同一约定。
    """
    p = _lib_thermal_params(nx, ra, pr, tau)
    h = float(nx - 1)
    return {
        "nu": p["nu"],
        "alpha": p["alpha"],
        "tau_T": p["tau_T"],
        "H": h,
        "g_beta": ra * p["nu"] * p["alpha"] / (h * h * h),
    }


def make_fields(
    nx: int,
    ra: float,
    pr: float,
    tau: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    t_hot: float = 1.0,
    t_cold: float = 0.0,
):
    """初场：ρ=1、u=0、T 线性导热态 T(x)=t_hot−ΔT·x/H（与 BC 相容）。"""
    ny = nx
    p = lattice_params(nx, ra, pr, tau)
    x = torch.arange(nx, device=device, dtype=dtype)
    T0 = (t_hot - (t_hot - t_cold) * x / p["H"]).unsqueeze(0).expand(ny, nx).contiguous()
    zero = torch.zeros_like(T0)
    f = equilibrium(torch.ones_like(T0), zero, zero)
    g = temperature_equilibrium(T0, zero, zero)
    g = apply_temperature_boundaries(g, t_hot, t_cold)
    wall = cavity_wall_mask(ny, nx, device)
    return f, g, wall, p


def advance(
    f: torch.Tensor,
    g: torch.Tensor,
    wall: torch.Tensor,
    tau: float,
    tau_T: float,
    g_beta: float,
    t_hot: float = 1.0,
    t_cold: float = 0.0,
    mask_wall_u: bool = True,
):
    """单步推进（与库 simulate_natural_convection 主循环同序）。

    mask_wall_u=True：温度碰撞前把壁环节点的宏观 u 置零（方案选择，
    见 NOTES.md 诊断链）。速度场半程反弹壁节点的宏观 u 是反弹分布的
    组合值（实测热壁列 mean u_x=−8e-5、上下壁行 u_y=−3.6e-5/+1.3e-4，
    非零且上下不对称）；固壁不输运温度，将其清零后再进温度碰撞的
    平衡态 (1+3cu)。否则热壁列交付被 3w1·T_w·u_x(0)/τ_T 系统污染
    （T_cold=0 侧天然免疫 → 左右壁不对称），造成 grad1 读数偏置
    （+0.16 Nu @n64）与净能量抽取（−1.1e-4/步 @n64，Ra 无关），
    内区以恒速慢冷、稳态永不达成。置零后读数与真实链路通量逐位
    吻合、300k 步钉死（sym 0.0019、intT 恒定）。
    """
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    F = buoyancy_force(rho, T, g_beta, t_ref=t_cold)
    if mask_wall_u:
        fluid = 1.0 - wall.to(ux.dtype)
        ux = ux * fluid
        uy = uy * fluid
    g = temperature_collision(g, tau_T, ux, uy)
    g = temperature_stream(g)
    g = apply_temperature_boundaries(g, t_hot, t_cold)
    f_pre = f
    f = collide_bgk_force(f, tau, F)
    f = pre_streaming_bounce_back(f_pre, f, wall)
    f = stream(f)
    return f, g


def observables(
    f: torch.Tensor, g: torch.Tensor, p: dict[str, float], t_hot: float = 1.0, t_cold: float = 0.0
) -> dict[str, float]:
    """直接观测量（稳态最终场；无拟合/外推/修正）。口径见 NOTES.md 预注册。"""
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    ny, nx = T.shape
    H = p["H"]
    dT = t_hot - t_cold
    scale = H / p["alpha"]  # 扩散尺度无量纲化（DVD 约定）

    nu = nusselt_number(T, H, dT, t_hot, t_cold, mode="grad1")

    # 中线：x = H/2（nx 偶数时恰为两列中点，线性插值权重 0.5）
    xc = H / 2.0
    i0 = int(math.floor(xc))
    frac = xc - i0
    i1 = min(i0 + 1, nx - 1)
    u_line = (1.0 - frac) * ux[:, i0] + frac * ux[:, i1]
    v_line = (1.0 - frac) * uy[i0, :] + frac * uy[i1, :]
    ju = int(u_line.argmax())
    jv = int(v_line.argmax())

    prof_hot = (T[:, 0] - T[:, 1]) * H / dT
    jn_max = int(prof_hot.argmax())
    jn_min = int(prof_hot.argmin())
    u_absmax = float(torch.maximum(ux.abs().max(), uy.abs().max()))
    # Nu_{1/2}：竖直中线 x=H/2 处（两列中点二阶差分 (T[:,i0]−T[:,i1])/1·H）
    nu_mid = float(((T[:, i0] - T[:, i1]) * H / dT).mean())

    return {
        "nu": nu["nu"],
        "nu_left": nu["nu_left"],
        "nu_right": nu["nu_right"],
        "nu_sym": abs(nu["nu_left"] - nu["nu_right"]),
        "u_max_nd": float(u_line[ju]) * scale,
        "v_max_nd": float(v_line[jv]) * scale,
        "u_max_lu": float(u_line[ju]),
        "v_max_lu": float(v_line[jv]),
        "u_max_y": ju / H,
        "v_max_x": jv / H,
        "nu_wall_max": float(prof_hot[jn_max]),
        "nu_wall_max_y": jn_max / H,
        "nu_wall_min": float(prof_hot[jn_min]),
        "nu_wall_min_y": jn_min / H,
        "nu_mid": nu_mid,
        "T_min": float(T.min()),
        "T_max": float(T.max()),
        "u_absmax_lu": u_absmax,
        "mach": u_absmax * math.sqrt(3.0),
    }
