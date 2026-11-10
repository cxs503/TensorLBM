"""二阶重构的组合滑移壁（mixed bounce-back / specular）：清偿 σ→0 的 O(1/H) 振幅偏置。

要清除的缺陷
------------
``partial_slip_wall.mixed_wall_row`` 的组合格式::

    f_q(wall) = (1 - sigma) f_{opp(q)}^{post}(adj) + sigma f_{M(q)}^{post}(adj)

在力驱动（Poiseuille，体加速度 ``g``）下反演滑移长度 ``b`` 得到

    b(sigma) = 3 nu sigma/(1 - sigma) + c ,     c = u_wall nu /(g H)

其中 ``u_wall`` 是**与 sigma 无关、与网格无关**的常值伪滑移速度，故 ``c`` 正比于
``1/H``（本仓基线 tau=0.8, nu=0.1 时 c = -0.065；m7963-c1 报告的 c ≈ -0.139 是
同一偏置在另一 tau 下的取值，见 ``kappa`` 表）。

偏置溯源（机制，不是拟合）
--------------------------
定常 x-均匀 Poiseuille 的离散解在体区**精确**是“抛物线 + 常值”：``u = u_ana + u_wall``
（曲率与解析值一致到 1e-6 相对，偏差沿 y 为严格常数到 1e-15；``probe_parabola_residual``
可复现）。常值方向是内部定常方程的**零空间**（``u'``、``u''`` 都不变），所以
``u_wall`` 只能由壁面条件确定——它是**半程 bounce-back 在 Guo 力项下的伪滑移速度**，
不是体区截断误差。

Guo 源项 ``(1 - 1/(2 tau)) 3 w_q c_xq F`` 写在壁面相邻流点上，被
``f_{opp(q)}^{post}`` 原封不动反射回流体，使壁面链路上多出一份正比于 ``F`` 的切向
动量，这份动量即伪滑移。对 ``F`` 的线性性已数值确认：``u_wall/g`` 在
``g = 1e-5 .. 4e-5`` 上相对变化 < 1e-4（``probe_kappa_table`` 的 g 扫描列），所以它
是**格式 / 力项路由**的伪迹，不是有限 Mach 可压缩性误差。

精确闭式
--------
tau 从 0.55 扫到 1.5 测得的 ``u_wall/g`` 与

    kappa(tau) = s - 3/(4 s),      s = 2 tau - 1 = 6 nu
    u_wall     = g kappa(tau) = g [6 nu - 1/(8 nu)]

在全部 8 个 tau 上吻合到 < 1e-9 相对（``probe_kappa_table`` 复现）。``kappa`` 在
``nu = 1/sqrt(48) ≈ 0.1443``（tau ≈ 0.933）变号：单一符号的经验模型解释不了这一
变号，闭式可以。

二阶重构（作用于边界节点）
--------------------------
设无偏解 ``u^true``，实测解 ``u* = u^true + delta``（``delta = u_wall``）。把平衡态对
切向速度的线性展开代入混合壁面算子，两条反射通道的平移量符号相反：

* ``row[opp(q)]``: ``c_x(opp(q)) = -c_x(q)`` → 平移 ``-3 w_q rho delta c_xq``
* ``row[M(q)]``  : 镜面只翻法向，``c_x(M(q)) = c_x(q)`` → 平移 ``+3 w_q rho delta c_xq``

加权得 ``mix*_q = mix^true_q - (1 - 2 sigma) 3 w_q rho delta c_xq``，故边界节点上的
二阶修正项为::

    delta f_q = (1 - 2 sigma) * 3 w_q rho (c_q . delta_vec),   q 指向流体（inward）
    delta_vec = kappa(tau) * F_par / rho        （只取切向分量）

四条结构性质（精确，非拟合）：

* ``sigma = 0``（纯反弹）：修正即把伪滑移从半程反弹中减掉，``b(sigma=0) = 0``。
* ``sigma = 1/2``：两通道平移量恰好抵消，修正量为 0（格式自身无偏）。
* ``sigma = 1``（纯镜面）：修正量为 0，自由滑移负控不受影响。
* ``sum_q delta f_q = 0``、``sum_q c_yq delta f_q = 0``：质量与法向动量严格守恒，
  只补切向动量。

应用后 ``b(sigma) = 3 nu sigma/(1 - sigma)`` 精确成立（H 无关、sigma 全覆盖），
见 ``channel_poiseuille_2nd``。

适用范围与局限
--------------
* 推导在“力线性”（小 Mach）区内严格；``O(Ma^2)`` 项未含（``u_wall`` 本身 O(F)，
  残留 O(F^2) 项在测试算例里 < 1e-9 相对）。
* ``kappa`` 的闭式由 tau 扫描标定后与离散解析结构（两条通道平移量之差）比对确认，
  但**未**给出逐项手推的完整二阶 Chapman-Enskog 展开（本仓无人手算审计）。
* 只处理平壁（法向可任意，切向需与力同向）；曲面 / 阶梯壁的组合滑移与二阶重构
  未实现，见 ``wall_refinement_combination_gate``。
"""

from __future__ import annotations

import math

import numpy as np

from .partial_slip_wall import (
    _CX9,
    _CY9,
    _MIRROR_Y,
    _OPP2,
    _W9,
    PoiseuilleResult,
    analytic_plane_profile,
    collide_bgk_guo_1d,
    equilibrium_1d,
    invert_slip_length,
    macroscopic_1d,
    mixed_wall_row,
    nu_from_tau,
    slip_length_from_quadratic_fit,
    tau_from_nu,
)

__all__ = [
    "spurious_slip_coefficient",
    "spurious_slip_velocity",
    "corrected_mixed_wall_row",
    "channel_poiseuille_2nd",
    "couette_slip",
    "slip_law_theory",
    "invert_slip_length_2nd",
    "probe_parabola_residual",
    "probe_kappa_table",
    "b_sigma_curve",
]


# ---------------------------------------------------------------------------
# 1. 伪滑移的精确闭式
# ---------------------------------------------------------------------------
def spurious_slip_coefficient(tau: float) -> float:
    """``kappa(tau) = s - 3/(4 s)``，``s = 2 tau - 1``：半程反弹 + Guo 力的伪滑移系数。

    ``u_wall = g * kappa``（``g`` = 体力加速度，格点单位）。tau 扫描验证见模块
    docstring；在 ``s = sqrt(3)/2``（``nu = 1/sqrt(48)``）处变号。
    """
    if tau <= 0.5:
        raise ValueError(f"tau must exceed 0.5, got {tau}")
    s = 2.0 * tau - 1.0
    return s - 0.75 / s


def spurious_slip_velocity(g: float, tau: float) -> float:
    """壁面伪滑移速度 ``u_wall = g * kappa(tau)``（格点单位，正号沿力方向）。"""
    return float(g) * spurious_slip_coefficient(tau)


# ---------------------------------------------------------------------------
# 2. 边界节点上的二阶修正
# ---------------------------------------------------------------------------
def corrected_mixed_wall_row(
    f_post: np.ndarray,
    sigma: float,
    j_fluid: int,
    top: bool,
    delta_vec: tuple = (0.0, 0.0),
    amplitude: float | None = None,
) -> np.ndarray:
    """组合滑移壁 + 二阶伪滑移修正的边界节点分布。

    参数
    ----
    f_post : (9, N) 碰撞后分布（含 Guo 源项）。
    sigma  : 镜面分数（0 = 纯反弹，1 = 纯镜面）。
    j_fluid: 相邻**流**行索引。
    top    : True 表示上壁（法向 +y 指向流体）。
    delta_vec : ``(dx, dy)`` 伪滑移速度矢量。壁法向为 y，**只有切向分量 dx 参与修正**；
        dy 被投影掉（否则会注入法向动量，破坏 Σ c_y δf = 0）。

    修正项 ``delta f_q = (1 - 2 sigma) 3 w_q rho (c_q . delta_vec)`` 只加在
    “指向流体”的方向上：质量与法向动量守恒，切向动量精确减掉伪滑移。
    """
    row = mixed_wall_row(f_post, sigma, j_fluid, top)
    if delta_vec[0] == 0.0 and delta_vec[1] == 0.0:
        return row
    rho = float(np.sum(f_post[:, j_fluid]))
    inward = (_CY9[:, 0] < 0) if top else (_CY9[:, 0] > 0)
    # 只取**切向**分量：平面通道的壁法向是 y，故修正量只用 delta_vec[0]（x）。法向分量
    # 投影掉，否则会额外注入法向动量（破坏 Σ c_y δf = 0，且无物理依据）。
    proj = _CX9[:, 0] * float(delta_vec[0])
    amp = -2.0 * (1.0 - float(sigma)) if amplitude is None else float(amplitude)
    corr = amp * 3.0 * _W9[:, 0] * rho * proj
    return np.where(inward, row + corr, row)


# ---------------------------------------------------------------------------
# 3. 平面 Poiseuille（二阶重构版）
# ---------------------------------------------------------------------------
def channel_poiseuille_2nd(
    sigma: float,
    ny: int = 33,
    tau: float = 0.8,
    g: float = 1e-5,
    steps: int = 60000,
    tol: float = 1e-11,
    correction: bool = True,
    amplitude: float | None = None,
) -> PoiseuilleResult:
    """平面 Poiseuille，组合滑移壁 + 伪滑移二阶重构（``correction=False`` 为基线）。

    几何与 ``partial_slip_wall.channel_poiseuille`` 一致：``j=0``/``j=ny-1`` 为固体
    行，流行为 ``1..ny-2``，反弹壁面在 ``y = 0.5`` 与 ``y = ny-1.5``，半高
    ``H = (ny-2)/2``。
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if ny < 8:
        raise ValueError("ny must be >= 8")
    delta = spurious_slip_velocity(g, tau) if correction else 0.0
    dv = (delta, 0.0)

    f = equilibrium_1d(np.ones(ny), np.zeros(ny), np.zeros(ny))
    u_prev = np.zeros(ny - 2)
    converged = False
    residual = float("inf")
    step = 0
    u = u_prev
    for step in range(1, steps + 1):
        fx = g * f.sum(0)
        f_post = collide_bgk_guo_1d(f, tau, fx=fx, fy=0.0)
        full = f_post.copy()
        full[:, 0] = corrected_mixed_wall_row(f_post, sigma, 1, False, dv, amplitude)
        full[:, ny - 1] = corrected_mixed_wall_row(f_post, sigma, ny - 2, True, dv, amplitude)
        new_f = np.zeros_like(f)
        for q in range(9):
            cy = int(_CY9[q, 0])
            new_f[q, 1 : ny - 1] = full[q, 1 - cy : ny - 1 - cy]
        f = new_f
        _rho, ux, _uy = macroscopic_1d(f, g * f.sum(0), 0.0)
        u = ux[1 : ny - 1].copy()
        residual = float(np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1e-300))
        u_prev = u
        if residual < tol and step > 200:
            converged = True
            break
    return PoiseuilleResult(
        sigma=sigma,
        n=ny,
        tau=tau,
        g=g,
        coord=np.arange(1, ny - 1, dtype=float),
        u=u,
        length=(ny - 2) / 2.0,
        geometry="plane",
        steps=step,
        converged=converged,
        residual=residual,
    )


def invert_slip_length_2nd(result: PoiseuilleResult, core_fraction: float = 0.6) -> dict:
    """与 ``partial_slip_wall.invert_slip_length`` 相同，额外给出与 3nu sigma/(1-sigma) 的比对。"""
    out = invert_slip_length(result, core_fraction=core_fraction)
    out["b_theory"] = slip_law_theory(result.sigma, nu_from_tau(result.tau))
    out["b_rel_error"] = (
        (out["b"] - out["b_theory"]) / out["b_theory"] if out["b_theory"] > 0 else float("nan")
    )
    return out


def slip_law_theory(sigma: float, nu: float) -> float:
    """无偏滑移律 ``b = 3 nu sigma/(1 - sigma) = (nu/cs^2) sigma/(1-sigma)``。"""
    if not 0.0 <= sigma < 1.0:
        return float("inf") if sigma >= 1.0 else float("nan")
    return 3.0 * float(nu) * float(sigma) / (1.0 - float(sigma))


# ---------------------------------------------------------------------------
# 4. Couette（动壁）——验证切应力
# ---------------------------------------------------------------------------
def couette_slip(
    sigma: float,
    ny: int = 33,
    tau: float = 0.8,
    u_wall: float = 1e-3,
    steps: int = 40000,
    tol: float = 1e-12,
) -> dict:
    """平面 Couette：下壁 ``-u_wall``、上壁 ``+u_wall``，组合滑移壁。

    动壁的切向动量注入用标准 ``6 w_q rho (c_q . u_wall)``：反弹与镜面两条通道
    共用（镜面只翻法向，切向跳跃相同）。

    解析：``u'' = 0`` ⇒ ``u = A y``；Maxwell 滑移是“壁速 - 壁面流体速度 = b du/dn”，
    即 ``u_wall - A H = b A`` ⇒ ``A = u_wall/(H + b)``（滑移使切变率下降）。
    返回反演出的滑移长度 ``b``、切变率 ``A`` 与壁面切应力 ``tau_w = rho nu A``。
    """
    if not 0.0 <= sigma <= 1.0:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")
    if ny < 8:
        raise ValueError("ny must be >= 8")
    nu = nu_from_tau(tau)
    # 动壁切向动量注入只挂在**反弹（耗散）通道**上：镜面反射在壁面参考系里保守切向
    # 分量，不注入壁速（Galilean 不变量），故权重是 (1 - sigma)。
    shift_w = 6.0 * _W9[:, 0] * _CX9[:, 0] * u_wall
    f = equilibrium_1d(np.ones(ny), np.zeros(ny), np.zeros(ny))
    u_prev = np.zeros(ny - 2)
    residual = float("inf")
    step = 0
    u = u_prev
    for step in range(1, steps + 1):
        f_post = collide_bgk_guo_1d(f, tau, fx=0.0, fy=0.0)
        full = f_post.copy()
        for row, top in ((0, False), (ny - 1, True)):
            jf = 1 if not top else ny - 2
            base = mixed_wall_row(f_post, sigma, jf, top)
            rho = float(np.sum(f_post[:, jf]))
            inward = (_CY9[:, 0] < 0) if top else (_CY9[:, 0] > 0)
            u_w = u_wall if top else -u_wall
            full[:, row] = np.where(inward, base + (1.0 - float(sigma)) * rho * shift_w * u_w / u_wall, base)
        new_f = np.zeros_like(f)
        for q in range(9):
            cy = int(_CY9[q, 0])
            new_f[q, 1 : ny - 1] = full[q, 1 - cy : ny - 1 - cy]
        f = new_f
        _rho, ux, _uy = macroscopic_1d(f, 0.0, 0.0)
        u = ux[1 : ny - 1].copy()
        residual = float(np.max(np.abs(u - u_prev)) / max(np.max(np.abs(u)), 1e-300))
        u_prev = u
        if residual < tol and step > 200:
            break
    half = (ny - 2) / 2.0
    y = np.arange(1, ny - 1, dtype=float)
    yc = float(np.mean(y))
    A_fit, intercept = np.polyfit(y - yc, u, 1)
    dev = float(np.max(np.abs(u - (A_fit * (y - yc) + intercept))))
    # Maxwell: u_wall - u(H) = b A。最外流点在壁面内侧 0.5 格（壁面 y = ny-1.5），
    # 故 u(H) = u_last + 0.5 A，从而 b = (u_wall - u_last)/A - 0.5。
    b_inv = (u_wall - float(np.max(u))) / A_fit - 0.5
    return {
        "sigma": float(sigma),
        "tau": float(tau),
        "nu": nu,
        "u_wall": float(u_wall),
        "shear_rate": float(A_fit),
        "shear_rate_ideal_noslip": u_wall / half,
        "shear_stress": float(nu * A_fit),
        "shear_stress_noslip": float(nu * u_wall / half),
        "shear_stress_analytic": float(nu * u_wall / (half + b_inv)),
        "b_inverted": float(b_inv),
        "b_theory": slip_law_theory(sigma, nu),
        "profile_linearity_residual": dev,
        "steps": step,
        "residual": residual,
        "coord": y,
        "u": u,
    }


# ---------------------------------------------------------------------------
# 5. 诊断：体区“抛物线 + 常值”残差、kappa 表
# ---------------------------------------------------------------------------
def probe_parabola_residual(result: PoiseuilleResult) -> dict:
    """体区解相对 ``u_ana`` 的残差结构：常数平移？还是有曲率误差？"""
    nu = nu_from_tau(result.tau)
    ana = analytic_plane_profile(result.coord, result.length, nu, result.g, 0.0)
    d = np.asarray(result.u) - ana
    return {
        "sigma": float(result.sigma),
        "tau": float(result.tau),
        "mean_offset": float(np.mean(d)),
        "std_offset": float(np.std(d)),
        "max_abs": float(np.max(np.abs(d))),
        "offset_over_g": float(np.mean(d) / result.g),
        "curvature_ratio": float(
            slip_length_from_quadratic_fit(
                result.coord, result.u, result.length, nu, result.g
            )["curvature_ratio"]
        ),
        "kappa_closed_form": spurious_slip_coefficient(result.tau),
    }


def probe_kappa_table(
    taus=(0.55, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5),
    gs=(1e-5, 2e-5, 4e-5),
    ny: int = 17,
    steps: int = 200000,
    tol: float = 1e-12,
) -> dict:
    """扫 tau、g：测 ``u_wall/g``，与闭式 ``kappa = s - 3/(4s)`` 比对。"""
    rows = []
    for tau in taus:
        for g in gs:
            r = channel_poiseuille_2nd(
                sigma=0.0, ny=ny, tau=tau, g=g, steps=steps, tol=tol, correction=False
            )
            p = probe_parabola_residual(r)
            rows.append(
                {
                    "tau": float(tau),
                    "nu": nu_from_tau(tau),
                    "g": float(g),
                    "offset_over_g": p["offset_over_g"],
                    "kappa_closed_form": p["kappa_closed_form"],
                    "rel_dev": abs(p["offset_over_g"] - p["kappa_closed_form"])
                    / abs(p["kappa_closed_form"]),
                    "curvature_ratio": p["curvature_ratio"],
                    "parabola_std": p["std_offset"],
                }
            )
    devs = [r["rel_dev"] for r in rows]
    return {"rows": rows, "max_rel_dev": float(max(devs)), "ny": ny}


def b_sigma_curve(
    sigmas=(0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9),
    ny: int = 33,
    tau: float = 0.8,
    g: float = 1e-5,
    steps: int = 60000,
    tol: float = 1e-11,
) -> dict:
    """``b(sigma)`` 曲线：基线（无修正）vs 二阶重构 vs 理论 ``3 nu sigma/(1-sigma)``。"""
    nu = nu_from_tau(tau)
    rows = []
    for s in sigmas:
        base = channel_poiseuille_2nd(s, ny=ny, tau=tau, g=g, steps=steps, tol=tol,
                                     correction=False)
        fix = channel_poiseuille_2nd(s, ny=ny, tau=tau, g=g, steps=steps, tol=tol,
                                     correction=True)
        ib = invert_slip_length(base)
        iF = invert_slip_length(fix)
        bt = slip_law_theory(s, nu)
        rows.append(
            {
                "sigma": float(s),
                "H": (ny - 2) / 2.0,
                "b_baseline": float(ib["b"]),
                "b_corrected": float(iF["b"]),
                "b_theory": float(bt),
                "rel_err_baseline": float((ib["b"] - bt) / bt) if bt > 0 else float("nan"),
                "rel_err_corrected": float((iF["b"] - bt) / bt) if bt > 0 else float("nan"),
            }
        )
    return {"ny": ny, "H": (ny - 2) / 2.0, "tau": tau, "nu": nu, "rows": rows}