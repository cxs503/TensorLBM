"""双分布函数热 LBM（D2Q9 速度 + D2Q5 温度 advection-diffusion）。

模型（DDF，Guo 2002 力格式 + Boussinesq 近似，c_s² = 1/3）：

- 速度场：D2Q9 BGK 碰撞，浮力 F = ρ·g·β·(T − T_ref) 以 Guo 力格式注入
  （速度修正 u* = u + F/(2ρ)，平衡分布用 u*，碰撞后叠加力项 F_i）。
- 温度场：D2Q5 BGK advection-diffusion（g_eq = w_i·T·(1 + 3 c_i·u)），
  热扩散率 α = c_s²·(τ_T − 1/2) = (τ_T − 1/2)/3。
- 无量纲参数：Ra = g·β·ΔT·H³/(ν·α)，Pr = ν/α，
  ν = (τ − 1/2)/3。格子中取 ΔT = T_hot − T_cold，由 Ra 反解 g·β。
- 边界：
  * 四壁 no-slip —— pre-streaming 半程反弹（与 verified/cavity_re100 的
    V3 配方一致：碰撞前分布反射，动量不进壁面行）
  * 温度边界（post-streaming）：上/下绝热壁（y=∓0.5）半程 bounce-back
    反射（从周期 stream 的对侧绕行落点取回本壁出射分布），左/右等温
    列整列取平衡分布 w_i·T_wall（u=0；等温壁位于节点上），指向流体的
    群体叠加第一流体列非平衡修正（等温面精确定位壁节点）
- Nusselt 数：Nu = −∂T/∂x·H/ΔT 沿壁面平均（多种差分口径可选）。

兼容接口（tensorlbm.physics 命名空间约定，与 thermal3d.py 同构）：
``C_D2Q5``、``W_D2Q5``、``equilibrium_thermal``、``collide_thermal_bgk``、
``stream_thermal``、``macroscopic_thermal``、``apply_buoyancy_force``。

格子排列：分布张量 (Q, ny, nx)，与 solver.py 一致。
"""

from __future__ import annotations

from typing import Any

import torch

from .d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from .solver import stream

CS2 = 1.0 / 3.0

# ── D2Q5 温度格子 ────────────────────────────────────────────────────────────
# 方向：0:(0,0) 1:(1,0) 2:(-1,0) 3:(0,1) 4:(0,-1)
C5 = torch.tensor([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], dtype=torch.int64)
W5 = torch.tensor([1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0], dtype=torch.float32)
OPPOSITE5 = torch.tensor([0, 2, 1, 4, 3], dtype=torch.int64)

# physics 命名空间兼容别名（与 thermal3d 的 C_D3Q7 / W_D3Q7 同构）
C_D2Q5 = C5
W_D2Q5 = W5

# D2Q5 流索引缓存，keyed by (ny, nx, device.type, device.index)
_stream5_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


def temperature_equilibrium(T: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q5 温度平衡分布 g_eq = w_i·T·(1 + 3·c_i·u)（advection-diffusion 一阶展开）。"""
    device = T.device
    c = C5.to(device)
    w = W5.to(device).view(5, 1, 1)
    cu = c[:, 0].view(5, 1, 1) * ux + c[:, 1].view(5, 1, 1) * uy
    return w * T.unsqueeze(0) * (1.0 + 3.0 * cu)


# physics 兼容名（与 thermal3d.equilibrium_thermal_3d 同构）
equilibrium_thermal = temperature_equilibrium


def temperature_collision(
    g: torch.Tensor, tau_T: float, ux: torch.Tensor, uy: torch.Tensor
) -> torch.Tensor:
    """D2Q5 温度 BGK 碰撞：g_new = g − (g − g_eq)/τ_T。

    热扩散率 α = (τ_T − 1/2)/3。零阶矩（温度）守恒。
    """
    T = g.sum(dim=0)
    return collide_thermal_bgk(g, T, ux, uy, tau_T)


def collide_thermal_bgk(
    g: torch.Tensor, T: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, tau_T: float
) -> torch.Tensor:
    """D2Q5 温度 BGK 碰撞（T 由调用方提供，physics 命名空间签名）。

    g_new = g − (g − g_eq)/τ_T，热扩散率 α = (τ_T − 1/2)/3。
    """
    geq = temperature_equilibrium(T, ux, uy)
    return g - (g - geq) / tau_T


def temperature_stream(g: torch.Tensor) -> torch.Tensor:
    """D2Q5 周期 streaming（单次 advanced-index gather，索引按 (shape, device) 缓存）。"""
    ny, nx = g.shape[1], g.shape[2]
    device = g.device
    key = (ny, nx, device.type, device.index)
    if key not in _stream5_cache:
        c = C5.to(device)
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(5, device=device).view(5, 1, 1).expand(5, ny, nx)
        y_idx = y_src.unsqueeze(2).expand(5, ny, nx)
        x_idx = x_src.unsqueeze(1).expand(5, ny, nx)
        _stream5_cache[key] = (q_idx, y_idx, x_idx)
    q_idx, y_idx, x_idx = _stream5_cache[key]
    return g[q_idx, y_idx, x_idx]


# physics 兼容名
stream_thermal = temperature_stream


def macroscopic_thermal(g: torch.Tensor) -> torch.Tensor:
    """宏观温度 T = Σ_i g_i（physics 命名空间签名）。"""
    return g.sum(dim=0)


def apply_temperature_boundaries(g: torch.Tensor, t_hot: float, t_cold: float) -> torch.Tensor:
    """post-streaming 温度边界（绝热壁解缠反射 + 等温列冻结平衡）。

    - 绝热下/上壁（壁面 y=∓0.5，与速度场 no-slip 半程壁同位）：周期
      ``temperature_stream`` 把从下壁行向下离开的分布 g_pre[4,0,:] 绕行到
      顶行（= post-stream 的 g[4,ny−1,:]），从顶行向上离开的绕行到底行
      （= g[3,0,:]）。此处将其从对侧取回并按半程 bounce-back 反射回原壁行：
        g[3,0,:] = g_pre[4,0,:]（= g[4,−1,:]），g[4,−1,:] = g_pre[3,ny−1,:]（= g[3,0,:]）
      净绝热通量逐位恒等于 0（两壁 ΔΣT 互为相反数）。旧版
      g[3,0,:]=g[4,0,:] 取的是 post-stream 自行 1 到达的 g_pre[4,1,:]，
      并非应反射的 g_pre[4,0,:]——这是旧规则绝热壁假热汇的根因
      （对流场下两者之差即泄漏，实测 Ra=1e4 N=64 约 −2.2e-2 步⁻¹）。
    - 左右 Dirichlet 列（等温壁位于节点 x=0 / x=nx−1，间距 nx−1 格，与
      速度场 no-slip 壁同位）：整列取温度平衡分布（u=0，g_i = w_i·T_wall，
      thermal3d 同构），指向流体的群体（左壁 g_1、右壁 g_2）叠加第一
      流体列的非平衡修正，把有效等温面从距第一流体节点 τ_T 格处拉回
      壁节点（详见下方实现注记；修正量自静止群体 g_0 零和扣除，壁列
      ΣT = s_w·T_wall 与纯冻结一致）。x 向周期绕行的落点在被整列覆写
      范围内，不参与能量账目。
    - 纯扩散 u=0 时全场均匀温度保持不动（tests/test_thermal.py 自洽测试）。
    """
    w = W5.to(device=g.device, dtype=g.dtype)
    w1, w2 = float(w[1]), float(w[2])
    # 非平衡修正项（取自 BC 前输入场：第一流体列未被任何壁规则修改）
    T1 = g[:, :, 1].sum(dim=0)  # 第一流体列温度
    Tn2 = g[:, :, -2].sum(dim=0)  # 倒数第二列温度
    neq_left = g[1, :, 1] - w1 * T1  # 指向流体群体所携非平衡（左壁）
    neq_right = g[2, :, -2] - w2 * Tn2  # （右壁）
    g = g.clone()
    # 分布张量形状 (Q, ny, nx)：g[方向, y, x]。
    # 解缠：被周期 stream 绕行到对侧壁行的本壁出射分布
    refl_from_bottom = g[4, -1, :].clone()  # = g_pre[4, 0, :]（下壁向下出射，绕至顶行）
    refl_from_top = g[3, 0, :].clone()  # = g_pre[3, ny−1, :]（顶壁向上出射，绕至底行）
    # 半程 bounce-back 反射回原壁行（角点随后被 Dirichlet 覆写）
    g[3, 0, :] = refl_from_bottom
    g[4, -1, :] = refl_from_top
    # 左壁 x=0 整列、右壁 x=nx−1 整列 = 温度 equilibrium（u=0 ⇒ w_i·T_wall）
    g[:, :, 0] = w.view(5, 1) * t_hot
    g[:, :, -1] = w.view(5, 1) * t_cold
    # 指向流体的群体叠加第一流体列的非平衡修正（把有效等温面拉回壁节点）：
    # 纯冻结把 D2Q5 BGK 的有效等温面放到距第一流体节点 τ_T 格处
    # （1D 导热探针实测 x_hot = 1−τ_T），壁面 Nu 系统性偏低；
    # 修正量 = 该群体在体节点稳态所携非平衡（纯导热斜率 s 时恰为 w·τ_T·s），
    # 均匀场恒为 0（纯扩散自洽性保持）。g_1 平衡只含 c_x·u_x 项，近壁
    # u_x≈0，忽略 u 的误差为二阶小量。
    # 修正量同时从静止群体 g_0 扣除（g_0 永不离开壁列）：壁列 ΣT 保持
    # s_w·T_wall，节点温度读数与纯冻结一致；交付值只依赖 g_1(BC) 与
    # 碰撞平衡（由 T(0) 决定），两者均不变——物理与逐位交付不变。
    g[1, :, 0] = g[1, :, 0] + neq_left
    g[0, :, 0] = g[0, :, 0] - neq_left
    g[2, :, -1] = g[2, :, -1] + neq_right
    g[0, :, -1] = g[0, :, -1] - neq_right
    return g


def buoyancy_force(
    rho: torch.Tensor, T: torch.Tensor, g_beta: float, t_ref: float = 0.0
) -> torch.Tensor:
    """Boussinesq 浮力 F = (0, ρ·g·β·(T − T_ref))（重力向下，浮力沿 +y）。"""
    f_y = rho * g_beta * (T - t_ref)
    return torch.stack([torch.zeros_like(f_y), f_y], dim=0)


def apply_buoyancy_force(
    f: torch.Tensor,
    T: torch.Tensor,
    T_ref: float,
    beta: float,
    g_y: float = -1.0,
) -> torch.Tensor:
    """向 D2Q9 分布注入 Boussinesq 浮力（physics 命名空间签名，与 3D 同构）。

    一阶 Guo 力注入：F_y = −ρ·β·(T − T_ref)·g_y，f_i += w_i·3·c_yi·F_y。
    *g_y* 为无量纲重力加速度（负 = 向下）；*beta* 为合并系数 g·β
    （格子单位）。完整二阶 Guo 力格式见 :func:`collide_bgk_force`。
    """
    rho, _, _ = macroscopic(f)
    F_y = -rho * beta * (T - T_ref) * g_y
    c = C.to(f.device)
    w = W.to(f.device).view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    return f + w * 3.0 * cy * F_y.unsqueeze(0)


def collide_bgk_force(f: torch.Tensor, tau: float, F: torch.Tensor) -> torch.Tensor:
    """含 Guo 力格式的 D2Q9 BGK 碰撞。

    力注入（Guo, Zheng & Shi 2002，c_s² = 1/3）：
      u* = u + F/(2ρ)（碰撞后宏观速度），
      f_new = f − (f − f_eq(u*))/τ + F_i，
      F_i = w_i·(1 − 1/(2τ))·[3·(c_i·F − u*·F) + 9·(c_i·u*)·(c_i·F)]。
    """
    rho, ux, uy = macroscopic(f)
    rho_safe = rho.clamp(min=1e-12)
    fx = F[0]
    fy = F[1]
    ux_star = ux + 0.5 * fx / rho_safe
    uy_star = uy + 0.5 * fy / rho_safe

    feq = equilibrium(rho, ux_star, uy_star)
    f_new = f - (f - feq) / tau

    c = C.to(f.device)
    w = W.to(f.device).view(9, 1, 1)
    cx = c[:, 0].view(9, 1, 1)
    cy = c[:, 1].view(9, 1, 1)
    fxv = fx.unsqueeze(0)
    fyv = fy.unsqueeze(0)
    cF = cx * fxv + cy * fyv  # (9, ny, nx) c_i·F
    uF = ux_star * fx + uy_star * fy  # (ny, nx)     u*·F
    cu = cx * ux_star + cy * uy_star  # (9, ny, nx)  c_i·u*
    Fi = w * (1.0 - 0.5 / tau) * (3.0 * (cF - uF.unsqueeze(0)) + 9.0 * cu * cF)
    return f_new + Fi


def pre_streaming_bounce_back(
    f_pre: torch.Tensor, f: torch.Tensor, wall_mask: torch.Tensor
) -> torch.Tensor:
    """pre-streaming 半程反弹（静止壁，V3 配方，见 verified/cavity_re100）。

    在 collide 之后、stream 之前，用碰撞前的分布 f_pre 在壁面行做
    方向反射（f[opp]），动量不进入壁面行。配合周期 stream，等价于
    half-way no-slip 壁面。
    """
    opp = OPPOSITE.to(f.device)
    return torch.where(wall_mask.unsqueeze(0), f_pre[opp], f)


def cavity_wall_mask(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """四壁（底/顶/左/右）no-slip 掩码。"""
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def nusselt_number(
    T: torch.Tensor,
    H: float,
    dT: float,
    t_hot: float = 1.0,
    t_cold: float = 0.0,
    mode: str = "grad2",
) -> dict[str, float]:
    """壁面平均 Nusselt 数：Nu = −∂T/∂n·H/ΔT，沿整壁（含角点）平均。

    mode 选择 ∂T/∂x 的数值差分口径：
      'halfway' : 一阶差分取在 half-way 壁面（x=∓0.5）：
                  ∂T/∂x ≈ (T(0) − T_hot)/0.5（左壁）、(T_cold − T(nx−1))/0.5（右壁）
      'grad1'   : 节点一阶单侧差分（壁面近似在节点上）
      'grad2'   : 节点二阶单侧差分（默认，壁面近似在节点上）
    """
    if mode == "halfway":
        nu_left = 2.0 * (t_hot - T[:, 0]) * H / dT
        nu_right = 2.0 * (T[:, -1] - t_cold) * H / dT
    elif mode == "grad1":
        nu_left = (T[:, 0] - T[:, 1]) * H / dT
        nu_right = (T[:, -2] - T[:, -1]) * H / dT
    elif mode == "grad2":
        nu_left = (1.5 * T[:, 0] - 2.0 * T[:, 1] + 0.5 * T[:, 2]) * H / dT
        nu_right = (-1.5 * T[:, -1] + 2.0 * T[:, -2] - 0.5 * T[:, -3]) * H / dT
    else:
        raise ValueError(f"unknown mode: {mode}")
    nu_left = float(nu_left.mean().item())
    nu_right = float(nu_right.mean().item())
    return {"nu_left": nu_left, "nu_right": nu_right, "nu": 0.5 * (nu_left + nu_right)}


def thermal_params(nx: int, ra: float, pr: float, tau: float) -> dict[str, float]:
    """由 Ra、Pr、τ 反解格子参数。

    方腔物理长度 L = nx（half-way 壁面位于 x=∓0.5，间距 = nx 个格子单位，
    与 verified/cavity_re100 的 H = nx 约定一致），ν = (τ − 1/2)/3，
    α = ν/Pr，τ_T = 3α + 1/2，g·β = Ra·ν·α/(ΔT·L³)（ΔT = 1）。
    """
    nu = (tau - 0.5) / 3.0
    alpha = nu / pr
    tau_T = 3.0 * alpha + 0.5
    H = float(nx)
    g_beta = ra * nu * alpha / (H**3)  # ΔT = 1
    return {"nu": nu, "alpha": alpha, "tau_T": tau_T, "H": H, "g_beta": g_beta}


def simulate_natural_convection(
    nx: int,
    ra: float = 1e4,
    pr: float = 0.71,
    tau: float = 0.6,
    t_hot: float = 1.0,
    t_cold: float = 0.0,
    steps: int = 100000,
    device: torch.device | str = "cpu",
    seed_t: str = "linear",
    report_every: int = 10000,
) -> dict[str, Any]:
    """自然对流方腔（左热右冷、上下绝热）双分布 LBM 主循环。

    返回 dict：格子参数、Nu（多口径）、稳态残差、运行时间、宏观统计。
    """
    device = torch.device(device)
    ny = nx
    p = thermal_params(nx, ra, pr, tau)
    nu_lat, alpha_lat, tau_T, H, g_beta = (p["nu"], p["alpha"], p["tau_T"], p["H"], p["g_beta"])
    dT = t_hot - t_cold

    # 初始场：静止、均匀密度；温度线性分布（x 方向 T_cold→T_hot）加速收敛
    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)
    if seed_t == "linear":
        T0 = (
            torch.linspace(t_cold, t_hot, nx, device=device)
            .unsqueeze(0)
            .expand(ny, nx)
            .contiguous()
        )
    else:
        T0 = torch.full((ny, nx), 0.5 * (t_hot + t_cold), device=device)
    g = temperature_equilibrium(T0, u0, u0)

    wall = cavity_wall_mask(ny, nx, device)
    interior = ~wall
    resid_mask = interior.clone()

    def _macros():
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        return rho, ux, uy, T

    rho, ux, uy, T = _macros()
    ux_prev = ux.detach().clone()
    uy_prev = uy.detach().clone()
    T_prev = T.detach().clone()
    last_resid_u = None
    last_resid_T = None
    nu_history = []

    import time

    t0 = time.time()
    for step in range(1, steps + 1):
        rho, ux, uy, T = _macros()
        F = buoyancy_force(rho, T, g_beta, t_ref=t_cold)
        # 温度：碰撞（用当前 u）→ 流 → 边界
        g = temperature_collision(g, tau_T, ux, uy)
        g = temperature_stream(g)
        g = apply_temperature_boundaries(g, t_hot, t_cold)
        # 速度：碰撞（Guo 力）→ pre-streaming 半程反弹 → 流
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)

        if step % report_every == 0 or step == steps:
            rho, ux, uy, T = _macros()
            du = (
                torch.max(
                    torch.abs(ux[resid_mask] - ux_prev[resid_mask]),
                    torch.abs(uy[resid_mask] - uy_prev[resid_mask]),
                )
                .max()
                .item()
            )
            dT_res = torch.max(torch.abs(T[resid_mask] - T_prev[resid_mask])).item()
            last_resid_u, last_resid_T = du, dT_res
            nu_cur = nusselt_number(T, H, dT, t_hot, t_cold, mode="grad2")
            nu_history.append({"step": step, "resid_u": du, "resid_T": dT_res, **nu_cur})
            ux_prev = ux.detach().clone()
            uy_prev = uy.detach().clone()
            T_prev = T.detach().clone()
    elapsed = time.time() - t0

    rho, ux, uy, T = _macros()
    T_cpu = T.detach().cpu()
    u_mag = (ux.detach().cpu() ** 2 + uy.detach().cpu() ** 2).sqrt()

    nu_grad2 = nusselt_number(T_cpu, H, dT, t_hot, t_cold, mode="grad2")
    nu_grad1 = nusselt_number(T_cpu, H, dT, t_hot, t_cold, mode="grad1")
    nu_halfway = nusselt_number(T_cpu, H, dT, t_hot, t_cold, mode="halfway")

    return {
        "nx": nx,
        "ra": ra,
        "pr": pr,
        "tau": tau,
        "tau_T": tau_T,
        "nu": nu_lat,
        "alpha": alpha_lat,
        "g_beta": g_beta,
        "H": H,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "last_resid_u": last_resid_u,
        "last_resid_T": last_resid_T,
        "nu_grad2": nu_grad2["nu"],
        "nu_left_grad2": nu_grad2["nu_left"],
        "nu_right_grad2": nu_grad2["nu_right"],
        "nu_grad1": nu_grad1["nu"],
        "nu_halfway": nu_halfway["nu"],
        "u_max": float(u_mag.max().item()),
        "T_min": float(T_cpu.min().item()),
        "T_max": float(T_cpu.max().item()),
        "nu_history": nu_history,
    }


__all__ = [
    "CS2",
    "C5",
    "W5",
    "OPPOSITE5",
    "C_D2Q5",
    "W_D2Q5",
    "temperature_equilibrium",
    "temperature_collision",
    "temperature_stream",
    "equilibrium_thermal",
    "collide_thermal_bgk",
    "stream_thermal",
    "macroscopic_thermal",
    "apply_buoyancy_force",
    "apply_temperature_boundaries",
    "buoyancy_force",
    "collide_bgk_force",
    "pre_streaming_bounce_back",
    "cavity_wall_mask",
    "nusselt_number",
    "thermal_params",
    "simulate_natural_convection",
]
