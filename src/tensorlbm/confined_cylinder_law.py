"""受限域圆柱绕流双参数偏差定律（P187-W1-N08）。

两个**系统性**偏差源把任何受限域算例从"无界基准"拉偏，本模块把它们分离成
可乘、可验的两条律：

    Nu(H/D, tau) = Nu_inf * f(H/D) * t(tau)
    f(H/D) = 1 + a * (D/H)**m
    t(tau) = 1 + b * tau**(-beta)

* ``H/D`` — 堵塞比（槽道有效宽度 / 圆柱有效直径）；
* ``tau`` — 统计窗长（时间平均窗口长度，单位 D/U）。

三层用法：

1. :func:`simulate_confined_cylinder` — 双分布函数（D2Q9 速度 + D2Q5 温度）
   受限槽道内等温加热圆柱绕流，返回 Nu(t) 时间序列（可后处理成任意窗长）。
2. :func:`window_table` / :func:`fit_multiplicative_law` — 窗均值抽取与可分离
   乘法律拟合（含加法模型 AIC 对照、交互项残差、负控 F 检验）。
3. :func:`evaluate_criteria` — 把 C1..C6 判据程序化判定，产出可序列化 dict。

数值口径（写死在代码里、报告里复述）
------------------------------------
* 有效直径 ``D_eff = 2R + 1``：半程反弹壁面位于固体节点与流体节点之间，故
  几何直径比名义 ``2R`` 大 1 个格子；``Re = U_in*D_eff/nu`` 与 ``H/D =
  H_eff/D_eff`` 全部用有效量，避免 1/D 的系统性几何误差。
* ``H_eff = ny - 2``：上下壁半程位置 y=0.5 与 y=ny-1.5 之间距。
* 圆柱等温 ``T_w = 1``，来流 ``T_in = 0``，上下壁绝热（零法向通量），故总热流
  等于圆柱放热量，可用进出口焓通量差完全独立地校核表面热流积分。
* ``Nu = Q / (pi * alpha * dT)``（单位深度，表面积 ``A = pi*D``，``k = alpha``）。

基准：Churchill & Bernstein (1977) 圆柱平均 Nu 经验式（模块内自带纯公式实现，
避免与既有模块耦合）。
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from .d2q9 import C, OPPOSITE, equilibrium, macroscopic
from .solver import stream
from .thermal import (
    C5,
    OPPOSITE5,
    W5,
    collide_thermal_bgk,
    temperature_equilibrium,
    temperature_stream,
)

__all__ = [
    "ConfinedCylinderConfig",
    "ConfinedGeometry",
    "build_geometry",
    "churchill_bernstein",
    "simulate_confined_cylinder",
    "_bc_heat_flux",
    "window_means",
    "window_table",
    "fit_multiplicative_law",
    "fit_additive_law",
    "interaction_residuals",
    "negative_control_ftest",
    "evaluate_criteria",
    "MISSION_WINDOWS",
    "STATIONARY_WINDOWS",
]

# 战役规定的四个统计窗（单位 D/U，左闭右开）
MISSION_WINDOWS: tuple[tuple[float, float], ...] = (
    (15.0, 40.0),
    (25.0, 60.0),
    (40.0, 80.0),
    (60.0, 105.0),
)
# 补充窗：全部落在起转瞬变之后（t0 >= 40 D/U），用于把"统计窗截断"与"起转
# 瞬变"两个物理机制分开（见报告第 5 节）。
STATIONARY_WINDOWS: tuple[tuple[float, float], ...] = (
    (40.0, 55.0),
    (40.0, 65.0),
    (40.0, 80.0),
    (40.0, 105.0),
    (50.0, 105.0),
    (60.0, 105.0),
)


# ============================================================================
# 经验式基准
# ============================================================================

def churchill_bernstein(re: float, pr: float) -> float:
    """Churchill-Bernstein (1977) 圆柱平均 Nu（强制对流横流）。"""
    if re * pr < 0.2:
        return 1.0
    term1 = 0.62 * re ** 0.5 * pr ** (1.0 / 3.0)
    term2 = (1.0 + (0.4 / pr) ** (2.0 / 3.0)) ** 0.25
    term3 = (1.0 + (re / 282000.0) ** (5.0 / 8.0)) ** (4.0 / 5.0)
    return 0.3 + term1 / term2 * term3


# ============================================================================
# 配置与几何
# ============================================================================

@dataclass
class ConfinedCylinderConfig:
    """单算例配置。长度用格子单位，时间用 D/U。"""

    h_over_d: float = 8.0
    diameter: int = 16                 # 名义格子直径 2R
    re: float = 100.0
    pr: float = 0.7
    u_in: float = 0.1
    upstream_d: float = 6.0            # 入口到柱心距离 [D]
    downstream_d: float = 16.0         # 柱心到出口距离 [D]
    t_end_d: float = 110.0             # 总物理时长 [D/U]
    dt_sample_d: float = 0.1           # 采样间隔 [D/U]
    t_cyl: float = 1.0
    t_in: float = 0.0
    rho0: float = 1.0
    save_fields: bool = False
    dtype: str = "float32"            # float32 实测比 float64 快 2-3x（loong64）
    device: str = "cpu"
    threads: int = 0                   # 0 = 保持环境默认
    label: str = ""
    checkpoint: str = ""               # 断点文件（存在则续跑）
    time_budget_s: float = 0.0         # >0: 到预算即落盘返回 partial
    ckpt_every: int = 2000             # 每多少步落盘

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConfinedGeometry:
    """静态几何 + 预计算的边界索引（构造一次，循环内零分配）。"""

    ny: int
    nx: int
    cy: int
    cx: int
    R: int
    D_eff: float
    H_eff: float
    h_over_d: float
    solid: torch.Tensor
    wall: torch.Tensor
    # 热边界配对：[(i, opp_i, ys, xs)]，半程反反弹 Dirichlet / 绝热反射
    thermal_cyl_pairs: list = field(default_factory=list)
    thermal_wall_pairs: list = field(default_factory=list)
    # 热流配对：[(i, ys, xs)]，圆柱外表面边界链
    flux_pairs: list = field(default_factory=list)


def _roll_to(A: torch.Tensor, c: Sequence[int]) -> torch.Tensor:
    """返回 ``B(x) = A(x - c)``。"""
    return torch.roll(A, shifts=(int(c[1]), int(c[0])), dims=(A.ndim - 2, A.ndim - 1))


def build_geometry(cfg: ConfinedCylinderConfig) -> ConfinedGeometry:
    """构造圆柱 + 上下壁掩码与半程壁边界索引。"""
    R = max(2, int(cfg.diameter) // 2)
    D_eff = 2.0 * R + 1.0                       # 半程壁有效直径
    ny = int(round(cfg.h_over_d * D_eff)) + 2
    H_eff = float(ny - 2)
    nx = int(round((cfg.upstream_d + cfg.downstream_d) * D_eff)) + 2
    cx = 1 + int(round(cfg.upstream_d * D_eff))
    cy = ny // 2

    dev = torch.device(cfg.device)
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=dev), torch.arange(nx, device=dev), indexing="ij"
    )
    solid = ((xx - cx) ** 2 + (yy - cy) ** 2) <= R * R
    wall = torch.zeros((ny, nx), dtype=torch.bool, device=dev)
    wall[0, :] = True
    wall[-1, :] = True
    wall |= solid

    geo = ConfinedGeometry(
        ny=ny, nx=nx, cy=cy, cx=cx, R=R, D_eff=D_eff, H_eff=H_eff,
        h_over_d=H_eff / D_eff, solid=solid, wall=wall,
    )

    opp5 = OPPOSITE5.to(dev)
    for i in range(1, 5):
        ci = (int(C5[i, 0].item()), int(C5[i, 1].item()))
        # 壁节点 w 的 i 向群体来自 w - c_i；若该邻居是流体则需回填
        nb_fluid = ~_roll_to(wall, ci)
        m_cyl = solid & nb_fluid
        if bool(m_cyl.any()):
            ys, xs = torch.nonzero(m_cyl, as_tuple=True)
            geo.thermal_cyl_pairs.append((i, int(opp5[i].item()), ys, xs))
            geo.flux_pairs.append((i, ys, xs))
        m_wall = (wall & ~solid) & nb_fluid
        if bool(m_wall.any()):
            ys, xs = torch.nonzero(m_wall, as_tuple=True)
            geo.thermal_wall_pairs.append((i, int(opp5[i].item()), ys, xs))
    return geo


# ============================================================================
# 求解器
# ============================================================================

def _stream9(f: torch.Tensor) -> torch.Tensor:
    """D2Q9 周期迁移（逐方向 roll）。

    与 :func:`tensorlbm.solver.stream` 的 gather 实现语义完全等价（周期边界、
    ``out[i](x) = f[i](x - c_i)``），但在 loong64 上实测快 3-4 倍（gather 的
    高级索引在 9 方向上退化为随机访存）。等价性由
    ``tests/test_confined_cylinder_law.py::test_stream_roll_matches_solver_stream``
    锁定。
    """
    c = C
    out = torch.empty_like(f)
    out[0] = f[0]
    for i in range(1, 9):
        out[i] = torch.roll(f[i], shifts=(int(c[i, 1]), int(c[i, 0])), dims=(0, 1))
    return out


def _stream5(g: torch.Tensor) -> torch.Tensor:
    """D2Q5 周期迁移（逐方向 roll），等价于 ``thermal.temperature_stream``。"""
    c = C5
    out = torch.empty_like(g)
    out[0] = g[0]
    for i in range(1, 5):
        out[i] = torch.roll(g[i], shifts=(int(c[i, 1]), int(c[i, 0])), dims=(0, 1))
    return out


def _bc_heat_flux(g: torch.Tensor, geo: ConfinedGeometry, t_wall: float, w5) -> float:
    """由等温壁**反反弹边界本身**计量的瞬时放热量（精确，无梯度插值误差）。

    对每条边界链，流体在 ``x_f`` 处失去 ``g_i``（沿 +c_i 流向壁面）并从壁面得到
    回填后的 ``g_opp(i)(w) = -g_i(w) + 2 w_i T_w``，故净得
    ``2 (w_i T_w - g_i(w))``（``g`` 取**迁移后、施加边界前**的状态）。
    求和即圆柱瞬时放热率（每格步能量）；``Nu = Q / (pi * alpha * dT)``。
    """
    total = 0.0
    for i, _oi, ys, xs in geo.thermal_cyl_pairs:
        total += 2.0 * (w5[i] * t_wall - float(g[i, ys, xs].sum().item()))
    return total


def _surface_flux(T: torch.Tensor, geo: ConfinedGeometry, t_wall: float) -> float:
    """圆柱外表面法向热流积分（二阶单侧差分，缺第二点时回退一阶）。

    对每条边界链沿外法向（壁 -> 流体）取 ``-dT/dn``，壁面温度用**给定**的
    ``t_wall``（不是固体节点的 ghost 值：半程反反弹下 ghost 温度并不等于壁温），
    第一内点距壁 0.5 格、第二内点距壁 1.5 格：

        -dT/dn = (8/3) T_w - 3 T_1 + (1/3) T_2     （二阶，点距 0 / 0.5 / 1.5）
        -dT/dn = 2 (T_w - T_1)                     （一阶回退）

    闭合轮廓上总通量 = 圆柱总放热（Gauss），与轮廓形状无关；每链面长 1（单位
    深度）。返回 ``-oint dT/dn ds``（k 已约去），``Nu = 该值 / (pi * dT)``。
    """
    total = 0.0
    ny, nx = T.shape
    for i, ys, xs in geo.flux_pairs:
        ci = (int(C5[i, 0].item()), int(C5[i, 1].item()))
        ys1 = torch.clamp(ys - ci[1], 0, ny - 1)
        xs1 = torch.clamp(xs - ci[0], 0, nx - 1)
        t1 = T[ys1, xs1]
        ys2 = torch.clamp(ys - 2 * ci[1], 0, ny - 1)
        xs2 = torch.clamp(xs - 2 * ci[0], 0, nx - 1)
        has2 = (~_roll_to(geo.wall, ci))[ys, xs] & (
            ~_roll_to(geo.wall, (2 * ci[0], 2 * ci[1]))
        )[ys, xs]
        # 非均匀三点 (0, 0.5, 1.5) 的一侧二阶导数：dT/dn = -(8/3)T0 + 3T1 - (1/3)T2
        q2 = (8.0 / 3.0) * t_wall - 3.0 * t1 + (1.0 / 3.0) * T[ys2, xs2]
        q1 = 2.0 * (t_wall - t1)
        total += float(torch.where(has2, q2, q1).sum().item())
    return total


def _save_checkpoint(path, step, f, g, times, nu_local, nu_global, umax, diag, nu_bc=()):
    torch.save({
        "step": int(step), "f": f.detach().cpu(), "g": g.detach().cpu(),
        "times": times, "nu_local": nu_local, "nu_global": nu_global,
        "nu_bc": list(nu_bc), "umax": umax, "diag": diag,
    }, path)


@torch.no_grad()
def simulate_confined_cylinder(cfg: ConfinedCylinderConfig) -> dict[str, Any]:
    """受限槽道内等温加热圆柱 DDF-LBM 求解，返回 Nu(t) 序列与诊断量（支持断点续跑）。

    每步：D2Q9 碰撞 -> 壁面半程反弹（仅壁节点索引赋值）-> 周期迁移 -> 进出流
    覆盖；D2Q5 碰撞 -> 迁移 -> 反反弹 Dirichlet（圆柱）/ 绝热反射（上下壁）->
    进出流覆盖。统计量 $\mathrm{Nu}(t)$ 用表面热流积分（主）+ 进出口焓通量差
    （独立校核，仅在热尾流充分出流后有效）。
    """
    if cfg.threads and cfg.threads > 0:
        torch.set_num_threads(int(cfg.threads))
    dev = torch.device(cfg.device)
    dtype = torch.float32 if cfg.dtype == "float32" else torch.float64
    geo = build_geometry(cfg)

    D_eff = geo.D_eff
    nu_lat = cfg.u_in * D_eff / cfg.re
    tau_f = 3.0 * nu_lat + 0.5
    alpha_lat = nu_lat / cfg.pr
    tau_T = 3.0 * alpha_lat + 0.5

    ny, nx = geo.ny, geo.nx
    solid, wall = geo.solid, geo.wall
    steps_per_du = D_eff / cfg.u_in                # 每 D/U 的格子步数
    n_steps = int(round(cfg.t_end_d * steps_per_du))
    stride = max(1, int(round(cfg.dt_sample_d * steps_per_du)))

    # 半程反弹索引（只对壁节点赋值，避免 9x ny x nx 全阵列 gather + where）
    wall_ys, wall_xs = torch.nonzero(wall, as_tuple=True)
    opp_list = [int(v) for v in OPPOSITE.tolist()]

    rho0 = torch.full((ny, nx), cfg.rho0, device=dev, dtype=dtype)
    ux0 = torch.full((ny, nx), cfg.u_in, device=dev, dtype=dtype)
    uy0 = torch.zeros((ny, nx), device=dev, dtype=dtype)
    f = equilibrium(rho0, ux0, uy0)
    T0 = torch.full((ny, nx), cfg.t_in, device=dev, dtype=dtype)
    T0 = torch.where(solid, torch.full_like(T0, cfg.t_cyl), T0)
    g = temperature_equilibrium(T0, ux0, uy0)

    # ---- 断点续跑 ----
    state = None
    if cfg.checkpoint and os.path.exists(cfg.checkpoint):
        state = torch.load(cfg.checkpoint, map_location=dev, weights_only=False)
        f = state["f"].to(dev)
        g = state["g"].to(dev)
        start_step = int(state["step"]) + 1
        times = list(state["times"])
        nu_local_hist = list(state["nu_local"])
        nu_bc_hist = list(state.get("nu_bc", []))
        nu_glob_hist = list(state["nu_global"])
        umax_hist = list(state["umax"])
        diag_hist = list(state["diag"])
    else:
        times, nu_local_hist, nu_glob_hist, umax_hist, diag_hist = [], [], [], [], []
        nu_bc_hist = []
        start_step = 1

    w5 = [float(v) for v in W5.view(5).tolist()]
    t_cyl, t_in = float(cfg.t_cyl), float(cfg.t_in)
    dT = t_cyl - t_in
    inv_tau = 1.0 / tau_f
    flux_scale_local = 1.0 / (math.pi * dT)               # Nu = -oint dT/dn ds / (pi dT)
    flux_scale_global = 1.0 / (math.pi * alpha_lat * dT)  # Nu = Q / (pi k dT), k = alpha

    ux_in_col = torch.full((ny, 1), cfg.u_in, device=dev, dtype=dtype)
    uy_zero_col = torch.zeros((ny, 1), device=dev, dtype=dtype)
    g_in = temperature_equilibrium(
        torch.full((ny, 1), t_in, device=dev, dtype=dtype),
        torch.full((ny, 1), cfg.u_in, device=dev, dtype=dtype),
        torch.zeros((ny, 1), device=dev, dtype=dtype),
    ).view(5, ny)

    t0 = time.time()
    status = "done"
    for step in range(start_step, n_steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)

        # ---- 温度场：碰撞 -> 迁移 -> 边界 ----
        g = collide_thermal_bgk(g, T, ux, uy, tau_T)
        g = _stream5(g)
        q_bc = _bc_heat_flux(g, geo, t_cyl, w5)
        for i, oi, ys, xs in geo.thermal_cyl_pairs:
            g[oi, ys, xs] = -g[i, ys, xs] + 2.0 * w5[i] * t_cyl
        for i, oi, ys, xs in geo.thermal_wall_pairs:
            g[oi, ys, xs] = g[i, ys, xs]
        g[:, :, 0] = g_in
        g[:, :, -1] = g[:, :, -2]

        # ---- 速度场：碰撞 -> 半程反弹 -> 迁移 -> 进出流 ----
        f_pre = f
        feq = equilibrium(rho, ux, uy)
        f = f - (f - feq) * inv_tau
        for i in range(9):
            f[i, wall_ys, wall_xs] = f_pre[opp_list[i], wall_ys, wall_xs]
        f = _stream9(f)
        # 速度入口：密度从内点外推（定压入口会把质量流量压偏 ~10%，见报告第 5 节）
        f[:, :, 0] = equilibrium(rho[:, 1:2].contiguous(), ux_in_col, uy_zero_col).view(9, ny)
        f[:, :, -1] = f[:, :, -2]
        # 不可压 LBM 的压力水平是自由的：每步把总质量归一到 rho0*N，防止缓慢漂移
        if step % 5 == 0:
            f = f * (rho0 * (ny * nx) / f.sum())

        if step % stride == 0 or step == n_steps:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nu_loc = _surface_flux(T, geo, t_cyl) * flux_scale_local
            nu_bc = q_bc * flux_scale_global
            # 全局焓通量差（上下壁绝热 => 进出口差 = 圆柱放热；仅稳态有效）
            q_in = (rho[1:-1, 1] * ux[1:-1, 1] * T[1:-1, 1]).sum().item()
            q_out = (rho[1:-1, -2] * ux[1:-1, -2] * T[1:-1, -2]).sum().item()
            nu_glob = (q_out - q_in) * flux_scale_global
            x_probe = min(geo.cx + int(round(8.0 * D_eff)), nx - 2)
            x_probe4 = min(geo.cx + int(round(4.0 * D_eff)), nx - 2)
            x_probe_n3 = nx - 3
            q_plane = (rho[1:-1, x_probe] * ux[1:-1, x_probe] * T[1:-1, x_probe]).sum().item()
            q_plane4 = (rho[1:-1, x_probe4] * ux[1:-1, x_probe4] * T[1:-1, x_probe4]).sum().item()
            q_planen3 = (rho[1:-1, x_probe_n3] * ux[1:-1, x_probe_n3] * T[1:-1, x_probe_n3]).sum().item()
            umass_in = (rho[1:-1, 1] * ux[1:-1, 1]).sum().item()
            umass_out = (rho[1:-1, -2] * ux[1:-1, -2]).sum().item()
            times.append(step / steps_per_du)
            nu_local_hist.append(nu_loc)
            nu_bc_hist.append(nu_bc)
            nu_glob_hist.append(nu_glob)
            umax_hist.append(float(torch.sqrt(ux * ux + uy * uy).max().item()))
            diag_hist.append({
                "t": step / steps_per_du,
                "sum_T": float(T.sum().item()),
                "q_loc_step": nu_loc * math.pi * alpha_lat * dT,
                "q_out_step": q_out,
                "q_in_step": q_in,
                "q_plane_step": q_plane,
                "q_plane4_step": q_plane4,
                "q_planen3_step": q_planen3,
                "mass_in": umass_in,
                "mass_out": umass_out,
            })

        if cfg.time_budget_s and step % 50 == 0 and (time.time() - t0) > cfg.time_budget_s:
            status = "partial"
            break
        if cfg.checkpoint and step % cfg.ckpt_every == 0:
            _save_checkpoint(cfg.checkpoint, step, f, g, times, nu_local_hist,
                             nu_glob_hist, umax_hist, diag_hist, nu_bc_hist)
    if cfg.checkpoint:
        _save_checkpoint(cfg.checkpoint, step, f, g, times, nu_local_hist,
                         nu_glob_hist, umax_hist, diag_hist, nu_bc_hist)

    elapsed = time.time() - t0
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)

    out: dict[str, Any] = {
        "label": cfg.label or f"hd{geo.h_over_d:.2f}_D{cfg.diameter}",
        "config": cfg.as_dict(),
        "geometry": {
            "ny": ny, "nx": nx, "cx": geo.cx, "cy": geo.cy, "R": geo.R,
            "D_eff": D_eff, "H_eff": geo.H_eff, "h_over_d": geo.h_over_d,
        },
        "lattice": {
            "nu": nu_lat, "tau_f": tau_f, "alpha": alpha_lat, "tau_T": tau_T,
            "steps_per_du": steps_per_du, "n_steps": n_steps, "sample_stride": stride,
            "dtype": cfg.dtype,
        },
        "nu_churchill_bernstein": churchill_bernstein(cfg.re, cfg.pr),
        "times": [round(v, 5) for v in times],
        "nu_local": [round(v, 6) for v in nu_local_hist],
        "nu_bc": [round(v, 6) for v in nu_bc_hist],
        "nu_global": [round(v, 6) for v in nu_glob_hist],
        "umax": [round(v, 5) for v in umax_hist],
        "diag": diag_hist,
        "elapsed_s": round(elapsed, 2),
        "node_steps": int(ny * nx * n_steps),
        "final": {
            "u_max": float(torch.sqrt(ux * ux + uy * uy).max().item()),
            "T_min": float(T.min().item()),
            "T_max": float(T.max().item()),
        },
        "stable": bool(np.isfinite(np.array(nu_local_hist[-5:], dtype=float)).all()),
        "status": status,
        "step_reached": int(step),
        "t_reached": float(times[-1]) if times else 0.0,
    }

    # 尾段统计：按时间窗（最后 15 D/U）而不是按样本数，避免短跑时把起转段算进来
    t_arr = np.asarray(times, dtype=float)
    tail_m = t_arr >= (t_arr[-1] - 15.0) if t_arr.size else np.zeros(0, dtype=bool)
    nu_l = float(np.mean(np.asarray(nu_local_hist)[tail_m])) if tail_m.any() else float("nan")
    nu_g = float(np.mean(np.asarray(nu_glob_hist)[tail_m])) if tail_m.any() else float("nan")
    nu_b = float(np.mean(np.asarray(nu_bc_hist)[tail_m])) if (tail_m.any() and nu_bc_hist) else float("nan")
    out["final"]["nu_local_mean_tail"] = round(nu_l, 6)   # 表面热流积分（诊断，离散误差 ~15-100%）
    out["final"]["nu_balance_mean_tail"] = round(nu_g, 6)  # 进出口焓平衡（含入口 Dirichlet 热沉，备用校核）
    out["final"]["nu_bc_mean_tail"] = round(nu_b, 6)       # 壁面反反弹边界计量（精确，主指标）
    out["final"]["surf_vs_global_pct"] = round(
        abs(nu_l - nu_g) / max(abs(nu_g), 1e-12) * 100.0, 3
    )

    if cfg.save_fields:
        out["fields"] = {
            "T": np.asarray(T.cpu(), dtype=np.float32),
            "ux": np.asarray(ux.cpu(), dtype=np.float32),
            "uy": np.asarray(uy.cpu(), dtype=np.float32),
            "solid": np.asarray(solid.cpu()),
        }
    return out

# ============================================================================
# 窗均值与定律拟合
# ============================================================================

def window_means(
    times: Sequence[float], series: Sequence[float], windows: Sequence[tuple[float, float]]
) -> dict[tuple[float, float], float]:
    """对每个时间窗 [t0, t1) 求序列平均值。"""
    t = np.asarray(times, dtype=float)
    s = np.asarray(series, dtype=float)
    out: dict[tuple[float, float], float] = {}
    for w in windows:
        m = (t >= w[0]) & (t < w[1])
        if not m.any():
            out[tuple(w)] = float("nan")
            continue
        out[tuple(w)] = float(s[m].mean())
    return out


def window_table(
    run: dict[str, Any],
    windows: Sequence[tuple[float, float]],
    series_key: str = "nu_local",
) -> dict[tuple[float, float], float]:
    """从单算例结果中抽取窗均值。"""
    return window_means(run["times"], run[series_key], windows)


def _mul_model(p: np.ndarray, hd: np.ndarray, tau: np.ndarray) -> np.ndarray:
    n_inf, a, m, b, beta = p
    return n_inf * (1.0 + a * hd ** (-m)) * (1.0 + b * tau ** (-beta))


def _add_model(p: np.ndarray, hd: np.ndarray, tau: np.ndarray) -> np.ndarray:
    n_inf, a, m, b, beta = p
    return n_inf + a * hd ** (-m) + b * tau ** (-beta)


def _linear_seed(
    hd: np.ndarray, tau: np.ndarray, nu: np.ndarray, m: float, beta: float, additive: bool
):
    """给定 (m, beta) 时用线性最小二乘解 (N_inf, a, b)（对数残差意义近似）。"""
    if additive:
        X = np.column_stack([np.ones_like(hd), hd ** (-m), tau ** (-beta)])
        coef, *_ = np.linalg.lstsq(X, nu, rcond=None)
    else:
        X = np.column_stack([np.ones_like(hd), hd ** (-m), tau ** (-beta)])
        coef, *_ = np.linalg.lstsq(X, np.log(np.maximum(nu, 1e-12)), rcond=None)
        coef = np.array([math.exp(coef[0]), coef[1], coef[2]])
    return np.array([coef[0], coef[1], m, coef[2], beta])


def _fit(model: str, hd: np.ndarray, tau: np.ndarray, nu: np.ndarray) -> dict[str, Any]:
    """网格播种 + Levenberg-Marquardt 精修（对数残差最小二乘）。"""
    from scipy.optimize import least_squares

    additive = model == "additive"
    fmod = _add_model if additive else _mul_model
    log_obs = np.log(np.maximum(nu, 1e-12))

    def resid(p):
        pred = fmod(p, hd, tau)
        return np.log(np.maximum(pred, 1e-12)) - log_obs

    best = None
    for m0 in (1.0, 1.5):
        for b0 in (0.5, 1.0):
            p0 = _linear_seed(hd, tau, nu, m0, b0, additive)
            try:
                r = least_squares(
                    resid, p0, method="lm", max_nfev=3000,
                    xtol=1e-14, ftol=1e-14, gtol=1e-14,
                )
            except Exception:
                continue
            if not np.all(np.isfinite(r.x)):
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        raise RuntimeError(f"fit failed for model={model}")
    p = best.x
    sse = float(np.sum(best.fun ** 2))
    n = int(nu.size)
    pnum = 5
    aic = n * math.log(sse / n) + 2 * pnum
    aicc = aic + (2 * pnum * (pnum + 1)) / max(n - pnum - 1, 1)
    return {
        "model": model,
        "params": {
            "nu_inf": float(p[0]), "a": float(p[1]), "m": float(p[2]),
            "b": float(p[3]), "beta": float(p[4]),
        },
        "sse_log": sse,
        "rmse_log_pct": float(math.sqrt(sse / n) * 100.0),
        "n_points": n,
        "aic": float(aic),
        "aicc": float(aicc),
        "grad_norm": float(np.linalg.norm(best.jac.T @ best.fun)),
    }


def fit_multiplicative_law(
    hd: Sequence[float], tau: Sequence[float], nu: Sequence[float]
) -> dict[str, Any]:
    """拟合 ``Nu = Nu_inf (1 + a (D/H)^m)(1 + b tau^-beta)``（对数残差 LS）。"""
    return _fit("multiplicative", np.asarray(hd, float), np.asarray(tau, float), np.asarray(nu, float))


def fit_additive_law(
    hd: Sequence[float], tau: Sequence[float], nu: Sequence[float]
) -> dict[str, Any]:
    """加法对照模型 ``Nu = Nu_inf + a (D/H)^m + b tau^-beta``（同一残差度量）。"""
    return _fit("additive", np.asarray(hd, float), np.asarray(tau, float), np.asarray(nu, float))


def interaction_residuals(
    hd: Sequence[float], tau: Sequence[float], nu: Sequence[float]
) -> dict[str, Any]:
    """可分离性检验：参数乘法律残差 + 双向 ANOVA（对数域）交互项。

    * ``law_max_abs_pct`` / ``law_rms_pct``：参数律残差（相对 %）；
    * ``anova_interaction_max_abs_pct``：去掉行/列主效应后的交互项（相对 %）。
    """
    hd_a = np.asarray(hd, float)
    tau_a = np.asarray(tau, float)
    nu_a = np.asarray(nu, float)
    fit = fit_multiplicative_law(hd_a, tau_a, nu_a)
    pred = _mul_model(np.array([
        fit["params"]["nu_inf"], fit["params"]["a"], fit["params"]["m"],
        fit["params"]["b"], fit["params"]["beta"],
    ]), hd_a, tau_a)
    rel = pred / nu_a - 1.0

    # 双向 ANOVA（对数域、加性主效应）
    y = np.log(nu_a)
    hd_u = np.unique(np.round(hd_a, 9))
    tau_u = np.unique(np.round(tau_a, 9))
    mu = y.mean()
    row = np.array([y[np.round(hd_a, 9) == v].mean() - mu for v in hd_u])
    col = np.array([y[np.round(tau_a, 9) == v].mean() - mu for v in tau_u])
    fit_row = np.array([row[list(hd_u).index(v)] for v in np.round(hd_a, 9)])
    fit_col = np.array([col[list(tau_u).index(v)] for v in np.round(tau_a, 9)])
    inter = y - (mu + fit_row + fit_col)
    return {
        "law_max_abs_pct": float(np.max(np.abs(rel)) * 100.0),
        "law_rms_pct": float(np.sqrt(np.mean(rel ** 2)) * 100.0),
        "anova_interaction_max_abs_pct": float(np.max(np.abs(inter)) * 100.0),
        "anova_interaction_rms_pct": float(np.sqrt(np.mean(inter ** 2)) * 100.0),
        "anova_table_shape": [len(hd_u), len(tau_u)],
        "fit": fit,
    }


def _fit_reduced(
    hd: np.ndarray, tau: np.ndarray, log_obs: np.ndarray, use_hd: bool, use_tau: bool
) -> dict[str, float]:
    """只在 ``use_*`` 指定的因子上拟合乘法律，返回 SSE 与自由参数个数。"""
    from scipy.optimize import least_squares

    names = ["nu_inf"]
    if use_hd:
        names += ["a", "m"]
    if use_tau:
        names += ["b", "beta"]

    def unpack(pv: np.ndarray) -> dict[str, float]:
        d = dict(zip(names, pv))
        d.setdefault("a", 0.0)
        d.setdefault("m", 1.0)
        d.setdefault("b", 0.0)
        d.setdefault("beta", 1.0)
        return d

    def resid(pv: np.ndarray) -> np.ndarray:
        d = unpack(pv)
        pred = d["nu_inf"] * (1.0 + d["a"] * hd ** (-d["m"])) * (1.0 + d["b"] * tau ** (-d["beta"]))
        return np.log(np.maximum(pred, 1e-12)) - log_obs

    best = None
    for m0 in (1.0, 1.5):
        for b0 in (0.5, 1.0):
            p0 = [float(np.exp(np.mean(log_obs)))] + ([0.0, m0] if use_hd else []) + ([0.0, b0] if use_tau else [])
            try:
                r = least_squares(resid, p0, method="lm", max_nfev=3000)
            except Exception:
                continue
            if not np.all(np.isfinite(r.x)):
                continue
            if best is None or r.cost < best.cost:
                best = r
    if best is None:
        raise RuntimeError("reduced fit failed")
    return {
        "sse": float(np.sum(best.fun ** 2)),
        "k": len(names),
        "params": {k: float(v) for k, v in unpack(best.x).items()},
    }


def negative_control_ftest(
    hd: Sequence[float], tau: Sequence[float], nu: Sequence[float], factor: str = "hd"
) -> dict[str, Any]:
    """负控：把某因子取值全置同值后，用嵌套 F 检验判定该因子项不显著。

    全模型 ``Nu = Nu_inf (1 + a (D/H)^m)(1 + b tau^-beta)``（5 参数）对去掉该因子
    项的嵌套模型（hd: 3 参数；tau: 3 参数）作对数残差 F 检验。因子无变异时
    F 应不显著（p > 0.05），说明估计器不会凭空造出该项。
    """
    from scipy import stats

    hd_a = np.asarray(hd, float).copy()
    tau_a = np.asarray(tau, float).copy()
    nu_a = np.asarray(nu, float)
    if factor == "hd":
        hd_a[:] = float(np.mean(hd_a))
        # 被置常数的项去掉，保留另一项（嵌套模型）
        use_hd, use_tau = False, True
    elif factor == "tau":
        tau_a[:] = float(np.mean(tau_a))
        use_hd, use_tau = True, False
    else:
        raise ValueError("factor must be 'hd' or 'tau'")

    log_obs = np.log(np.maximum(nu_a, 1e-12))
    n = int(nu_a.size)
    full = _fit_reduced(hd_a, tau_a, log_obs, True, True)
    red = _fit_reduced(hd_a, tau_a, log_obs, use_hd, use_tau)
    df1 = max(full["k"] - red["k"], 1)
    df2 = max(n - full["k"], 1)
    num = red["sse"] - full["sse"]
    if num <= 0.0 or full["sse"] <= 0.0:
        f_stat, p_val = 0.0, 1.0
    else:
        f_stat = (num / df1) / (full["sse"] / df2)
        p_val = float(1.0 - stats.f.cdf(f_stat, df1, df2))
    return {
        "factor": factor,
        "F": float(f_stat),
        "p_value": float(p_val),
        "df": [int(df1), int(df2)],
        "sse_full": full["sse"],
        "sse_reduced": red["sse"],
        "significant": bool(p_val < 0.05),
        "n_points": n,
    }


# ============================================================================
# 判据程序化评估
# ============================================================================

def _tables_to_arrays(rows: Sequence[dict[str, Any]]):
    hd = np.array([float(r["h_over_d"]) for r in rows])
    tau = np.array([float(r["tau"]) for r in rows])
    nu = np.array([float(r["nu"]) for r in rows])
    return hd, tau, nu


def evaluate_criteria(
    rows_primary: Sequence[dict[str, Any]],
    rows_mission: Sequence[dict[str, Any]],
    rows_coarse: Sequence[dict[str, Any]],
    nu_cb: float,
    test_windows: Sequence[Sequence[float]] = ((25.0, 60.0), (40.0, 105.0)),
    predicted_prior_pct: float = 2.5,
) -> dict[str, Any]:
    """按 C1..C5 逐条判定（C6 由调用脚本的 provenance 保证）。

    参数
    ----
    rows_primary : 细网格 × 平稳窗（τ 起点固定在起转之后）数据行，含
        ``h_over_d`` / ``tau`` / ``nu`` / ``window``；
    rows_mission : 细网格 × 战役规定窗数据行（用于 C3 与旁证）；
    rows_coarse  : 粗网格 × 同一组窗（用于 C4 网格正交性）；
    nu_cb        : Churchill-Bernstein 基准值。
    """
    out: dict[str, Any] = {}

    # ---- C1 可分离乘法律 -------------------------------------------------
    hd, tau, nu = _tables_to_arrays(rows_primary)
    inter = interaction_residuals(hd, tau, nu)
    mul = inter["fit"]
    add = fit_additive_law(hd, tau, nu)
    c1_ok = (inter["law_max_abs_pct"] < 2.0) and (mul["aic"] < add["aic"])
    out["C1"] = {
        "dataset": "fine x stationary-windows",
        "n_points": int(nu.size),
        "multiplicative": mul,
        "additive": add,
        "law_max_abs_residual_pct": inter["law_max_abs_pct"],
        "law_rms_residual_pct": inter["law_rms_pct"],
        "anova_interaction_max_abs_pct": inter["anova_interaction_max_abs_pct"],
        "aic_multiplicative": mul["aic"],
        "aic_additive": add["aic"],
        "delta_aic_mul_minus_add": mul["aic"] - add["aic"],
        "pass": bool(c1_ok),
        "threshold": "残差<2% 且 AIC(乘)<AIC(加)",
    }

    # ---- C2 零堵塞外推 vs Churchill-Bernstein ----------------------------
    nu_inf = mul["params"]["nu_inf"]
    hd_arr = np.array(sorted({float(r["h_over_d"]) for r in rows_primary}))
    hd_max = float(hd_arr.max())
    sel = [r for r in rows_primary if abs(float(r["h_over_d"]) - hd_max) < 1e-9]
    baseline_nu = float(np.mean([float(r["nu"]) for r in sel]))
    dev_inf = abs(nu_inf - nu_cb) / nu_cb * 100.0
    dev_base = abs(baseline_nu - nu_cb) / nu_cb * 100.0
    out["C2"] = {
        "nu_churchill_bernstein": nu_cb,
        "nu_inf_extrapolated": nu_inf,
        "deviation_pct": dev_inf,
        "baseline_widest_domain": {"h_over_d": hd_max, "nu": baseline_nu, "deviation_pct": dev_base},
        "improvement_pp": dev_base - dev_inf,
        "pass": bool(dev_inf < 5.0),
        "threshold": "外推 Nu_inf vs C-B 偏差 <5%",
    }

    # ---- C3 窗长律外推 ---------------------------------------------------
    tw = [(float(w[0]), float(w[1])) for w in test_windows]
    fit_rows = [
        r for r in rows_mission
        if not any(abs(float(r["window"][0]) - w[0]) < 1e-9 and abs(float(r["window"][1]) - w[1]) < 1e-9 for w in tw)
    ]
    hd_f, tau_f, nu_f = _tables_to_arrays(fit_rows)
    law = fit_multiplicative_law(hd_f, tau_f, nu_f)
    p = law["params"]

    def pred_mul(tau_w: float, hd_w: float) -> float:
        return p["nu_inf"] * (1.0 + p["a"] * hd_w ** (-p["m"])) * (1.0 + p["b"] * tau_w ** (-p["beta"]))

    def obs_mean(win) -> float:
        vals = []
        for r in rows_mission:
            if abs(float(r["window"][0]) - win[0]) < 1e-9 and abs(float(r["window"][1]) - win[1]) < 1e-9:
                vals.append(float(r["nu"]))
        return float(np.mean(vals))

    tau_short = tw[0][1] - tw[0][0]
    tau_long = tw[1][1] - tw[1][0]
    hd_ref = float(np.mean(hd_f))
    obs_short, obs_long = obs_mean(tw[0]), obs_mean(tw[1])
    pred_short, pred_long = pred_mul(tau_short, hd_ref), pred_mul(tau_long, hd_ref)
    obs_diff = (obs_short - obs_long) / obs_long * 100.0
    pred_diff = (pred_short - pred_long) / pred_long * 100.0
    out["C3"] = {
        "test_windows": tw,
        "fit_windows": sorted({tuple(float(x) for x in r["window"]) for r in fit_rows}),
        "law_from_holdout_fit": law,
        "tau_short": tau_short, "tau_long": tau_long,
        "observed_diff_pct": obs_diff,
        "predicted_diff_pct": pred_diff,
        "abs_diff_pp": abs(pred_diff - obs_diff),
        "prior_expectation_pct": predicted_prior_pct,
        "dev_from_prior_pp": abs(obs_diff - predicted_prior_pct),
        "pass": bool(abs(pred_diff - obs_diff) < 0.5),
        "threshold": "|律预测差值-实测差值| < 0.5pp",
    }

    # ---- C4 网格正交性 ---------------------------------------------------
    hd_c, tau_c, nu_c = _tables_to_arrays(rows_coarse)
    law_c = fit_multiplicative_law(hd_c, tau_c, nu_c)
    drift = {}
    for key in ("a", "m", "beta"):
        vf, vc = law["params"][key], law_c["params"][key]
        drift[key] = abs(vf - vc) / max(abs(vf), 1e-12) * 100.0
    out["C4"] = {
        "fine": law["params"],
        "coarse": law_c["params"],
        "drift_pct": drift,
        "note": "两网格级均用 mission-window 数据集，同级同窗比较",
        "pass": bool(all(v < 10.0 for v in drift.values())),
        "threshold": "(a,m,beta) 漂移 <10%",
    }

    # ---- C5 负控 ---------------------------------------------------------
    nc_hd = negative_control_ftest(hd, tau, nu, factor="hd")
    nc_tau = negative_control_ftest(hd, tau, nu, factor="tau")
    out["C5"] = {
        "hd_constant": nc_hd,
        "tau_constant": nc_tau,
        "pass": bool((nc_hd["p_value"] > 0.05) and (nc_tau["p_value"] > 0.05)),
        "threshold": "两负控 p>0.05（对应项不显著）",
    }
    out["all_pass"] = bool(all(out[k]["pass"] for k in ("C1", "C2", "C3", "C4", "C5")))
    return out