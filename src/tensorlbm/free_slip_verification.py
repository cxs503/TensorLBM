"""Analytic verification harness for the D3Q19 free-slip (specular) wall.

背景
----
``tensorlbm.boundaries3d.free_slip_cells_3d`` 自落地起一直处于 ``IMPLEMENTATION_ONLY``
（见 ``docs/boundary_capability_contract_r1.md``）：docstring 引用 waLBerla ``FreeSlip.h``，
但没有任何测试，更没有任何解析解判据。本模块把这条欠账变成可机器裁决的证据面。

三个可写出的解析事实（全部与网格无关、不含拟合自由度，除显式声明的拟合项）：

1. **代数层**：镜面反射算子必须把方向 q 的速度 ``c_q`` 映射到 ``(c_x, -c_y, c_z)``
   （axis=1），即只翻法向分量、保切向分量；且必须是 19 个方向上的对合（involutive）双射。
2. **不变量层**：算子是一个置换，因此单格点的质量 ``Σ_q f_q``、切向动量
   ``Σ_q c_t f_q`` 逐格点精确守恒，法向动量 ``Σ_q c_n f_q`` 精确取反；
   于是全局 x 动量在两侧自由滑移壁的通道里**精确守恒**（无壁面动量汇），
   而同样初始场的无滑移通道会持续吸走动量。这是两条方向相反的判据，互相做负控。
3. **谱层**：均匀来流（plug flow）是自由滑移壁的精确不动点（无寄生阻力）；
   剪切波模态在两侧自由滑移壁间是余弦驻波，衰减率由 ν=(τ-½)/3 决定。

本次验证真正测出来的、值得写进文档的**非平凡结论**：
离散剪切本征模态是 ``cos(k_eff (y - δ))``，其表观自由滑移面位于壁格点中心**内侧**
δ ≈ 0.32 lu（H=32 与 H=64 两次独立拟合分别 0.310 / 0.320，稳定且与分辨率无关）。
因此若把壁面当成"格点中心"（δ=0）来算衰减率，误差是 **一阶 O(k)**
（H=32 时 4.3%，H=64 时 2.1%）；而按拟合出的表观壁面计，ν 的恢复误差
降到 0.25% / 0.10%（<1%）。这两条都是判据（一正一负），不是免责声明。

用法
----
    from tensorlbm.free_slip_verification import build_results
    results = build_results()          # 纯 dict，可 json.dump

``tests/test_free_slip_d3q19_analytic.py`` 与
``artifacts/free_slip_d3q19_analytic/run_verification.py`` 共用本模块，
保证"文档里的数字"与"测试里的判据"是同一份计算，不存在手抄漂移。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch

from tensorlbm.boundaries3d import (
    _D3Q19_MIRROR_X,
    _D3Q19_MIRROR_Y,
    _D3Q19_MIRROR_Z,
    bounce_back_cells_3d,
    free_slip_cells_3d,
)
from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

#: 全场景统一判据：解析解相对误差上限（比平台 15% 门槛严得多）。
REL_TOL = 0.01  # 1%

#: 精确守恒类判据（置换算子，double 下应在 1e-13 量级）。
EXACT_TOL = 1e-12

#: 表观自由滑移面内移量的合法区间（lu）。实测 0.310/0.320；
#: 上界 <0.5 表示它既不是"格点中心"(0) 也不是"半路"(0.5)。
INSET_RANGE = (0.25, 0.42)

_MIRRORS = {0: _D3Q19_MIRROR_X, 1: _D3Q19_MIRROR_Y, 2: _D3Q19_MIRROR_Z}


# ---------------------------------------------------------------------------
# 几何与推进
# ---------------------------------------------------------------------------


def wall_mask(shape: Sequence[int], axis: int) -> torch.Tensor:
    """两侧壁（沿 ``axis`` 的两个端面）的布尔掩码，shape=(nz, ny, nx)。"""
    mask = torch.zeros(shape, dtype=torch.bool)
    if axis == 0:
        mask[:, :, 0] = True
        mask[:, :, -1] = True
    elif axis == 1:
        mask[:, 0, :] = True
        mask[:, -1, :] = True
    elif axis == 2:
        mask[0, :, :] = True
        mask[-1, :, :] = True
    else:  # pragma: no cover - 参数错误
        raise ValueError(f"axis must be 0/1/2, got {axis!r}")
    return mask


def specular_step(
    f: torch.Tensor,
    tau: float,
    solid: torch.Tensor,
    axis: int = 1,
) -> torch.Tensor:
    """一步自由滑移壁 D3Q19 BGK：碰撞 → NoDynamics → 镜面反射 → 周期 stream。

    顺序与 ``tensorlbm.lbm_step_correct``（BB 模式）完全一致，测的是库的**实际组合**：
    碰撞(全体) → 固体格点回填碰撞前快照 → 在固体格点做反射 → 周期 streaming。
    """
    pre = f
    f = collide_bgk3d(f, tau)
    f = torch.where(solid.unsqueeze(0), pre, f)
    f = free_slip_cells_3d(f, solid, axis=axis)
    return stream3d(f)


def noslip_step(
    f: torch.Tensor,
    tau: float,
    solid: torch.Tensor,
) -> torch.Tensor:
    """同序的无滑移对照：反射算子换成半路反弹（bounce-back）。"""
    pre = f
    f = collide_bgk3d(f, tau)
    f = torch.where(solid.unsqueeze(0), pre, f)
    f = bounce_back_cells_3d(f, solid)
    return stream3d(f)


def _profile_field(
    shape: Sequence[int],
    axis: int,
    profile: torch.Tensor,
) -> torch.Tensor:
    """把沿 ``axis`` 的 1-D 剖面（长度 = shape[2-axis]）广播成 (nz, ny, nx) 场。

    torch 张量的空间轴顺序是 (z, y, x)，而本模块 ``axis`` 用物理命名
    axis=0→x、1→y、2→z，故索引映射为 ``2 - axis``。
    """
    nz, ny, nx = shape
    dim_len = (nx, ny, nz)[axis]
    if profile.numel() != dim_len:
        raise ValueError(f"profile length {profile.numel()} != axis {axis} length {dim_len}")
    view = {
        0: lambda p: p.view(1, 1, nx),
        1: lambda p: p.view(1, ny, 1),
        2: lambda p: p.view(nz, 1, 1),
    }[axis](profile)
    return view.expand(nz, ny, nx).clone()


def _field_equilibrium(ux: torch.Tensor) -> torch.Tensor:
    rho = torch.ones_like(ux)
    zero = torch.zeros_like(ux)
    return equilibrium3d(rho, ux, zero, zero)


def _axial_ux(f: torch.Tensor, axis: int) -> torch.Tensor:
    """取沿壁法向（axis）的一维 u_x 剖面，只保留内部流体格点（去掉两端壁面）。"""
    _, ux, _, _ = macroscopic3d(f)
    nz, ny, nx = ux.shape
    if axis == 0:
        vec = ux[nz // 2, ny // 2, :]
    elif axis == 1:
        vec = ux[nz // 2, :, nx // 2]
    else:
        vec = ux[:, ny // 2, nx // 2]
    return vec[1:-1]


# ---------------------------------------------------------------------------
# 案例 1：镜面反射算子的代数正确性
# ---------------------------------------------------------------------------


def specular_map_error(axis: int) -> dict:
    """实测镜像表 vs 解析镜面映射 ``c → (c_x, -c_y, c_z)`` 的逐方向一致性。"""
    mirror = _MIRRORS[axis]
    sign = torch.ones(3, dtype=C.dtype)
    sign[axis] = -1
    expected = C * sign.unsqueeze(0)  # (19, 3)
    got = C[mirror]  # (19, 3)
    mismatched = int((got != expected).any(dim=1).sum())
    # 对合性：mirror[mirror[q]] == q；双射性：像集大小 == 19
    involution_ok = bool(torch.equal(mirror[mirror], torch.arange(C.shape[0])))
    bijective = int(torch.unique(mirror).numel()) == int(C.shape[0])
    return {
        "axis": axis,
        "mismatched_directions": mismatched,
        "involution": involution_ok,
        "bijective": bijective,
    }


# ---------------------------------------------------------------------------
# 案例 2：置换算子的逐格点不变量 + 全局动量守恒
# ---------------------------------------------------------------------------


def operator_invariants(axis: int = 1, seed: int = 0) -> dict:
    """单壁格点上的质量/切向动量/法向动量三项不变量（非平衡随机场）。"""
    g = torch.Generator().manual_seed(seed)
    shape = (3, 5, 4) if axis != 2 else (5, 3, 4)
    f = torch.rand(19, *shape, generator=g, dtype=torch.get_default_dtype()) + 0.1
    mask = wall_mask(shape, axis)
    out = free_slip_cells_3d(f, mask, axis=axis)
    sel = mask.unsqueeze(0).expand_as(f)
    f_wall, o_wall = f[sel].view(19, -1), out[sel].view(19, -1)

    tangential = [a for a in range(3) if a != axis]
    def moment(vals: torch.Tensor, comp: int) -> torch.Tensor:
        return (C[:, comp].to(vals.dtype).unsqueeze(1) * vals).sum(dim=0)

    before_mass = f_wall.sum(dim=0)
    after_mass = o_wall.sum(dim=0)
    res = {
        "mass_rel": float((after_mass - before_mass).abs().max() / before_mass.abs().max()),
        "normal_momentum_flip_rel": float(
            (moment(o_wall, axis) + moment(f_wall, axis)).abs().max()
            / moment(f_wall, axis).abs().max()
        ),
    }
    for a in tangential:
        before = moment(f_wall, a)
        after = moment(o_wall, a)
        res[f"tangential_momentum_{'xyz'[a]}_rel"] = float(
            (after - before).abs().max() / before.abs().max()
        )
    return res


def momentum_conservation_case(
    n_wall: int = 32,
    axis: int = 1,
    tau: float = 1.0,
    amplitude: float = 1e-2,
    steps: int = 1000,
    free_slip: bool = True,
) -> dict:
    """``u_x = A(1 + cos(k_axis))`` 的通道：自由滑移壁应精确守恒总 x 动量。

    均匀分量 A 是精确不动点，余弦分量是 ν 衰减的驻波；总 x 动量
    ``Σ ρ u_x`` 在自由滑移壁下没有任何壁面动量汇（切向动量逐格点守恒），
    故必须**精确**守恒；无滑移壁则持续把它吸走。
    """
    ny = n_wall + 1 if axis == 1 else 4
    nz = n_wall + 1 if axis == 2 else 4
    nx = n_wall + 1 if axis == 0 else 4
    shape = (nz, ny, nx)
    solid = wall_mask(shape, axis)
    n_axis = shape[2 - axis]
    k = math.pi / (n_axis - 1)
    coord = torch.arange(n_axis, dtype=torch.get_default_dtype())
    profile = amplitude * (1.0 + torch.cos(k * coord))
    ux = _profile_field(shape, axis, profile)

    f = _field_equilibrium(ux)
    m0 = float((macroscopic3d(f)[0] * macroscopic3d(f)[1]).sum())
    step = specular_step if free_slip else (lambda ff, t, s, axis=1: noslip_step(ff, t, s))
    for _ in range(steps):
        f = step(f, tau, solid, axis=axis) if free_slip else step(f, tau, solid)

    rho, uxf, _, _ = macroscopic3d(f)
    m1 = float((rho * uxf).sum())
    u_axial = _axial_ux(f, axis)
    return {
        "axis": axis,
        "free_slip": free_slip,
        "n_wall": n_wall,
        "steps": steps,
        "momentum_initial": m0,
        "momentum_final": m1,
        "momentum_rel_change": abs(m1 - m0) / abs(m0),
        # 均匀分量的存活率：自由滑移下应≈1，无滑移下被壁面吸走
        "mean_profile_ratio": float(u_axial.mean() / amplitude),
    }


# ---------------------------------------------------------------------------
# 案例 3：均匀流（plug flow）是精确不动点
# ---------------------------------------------------------------------------


def plug_flow_case(n_wall: int = 32, axis: int = 1, u0: float = 1e-2, steps: int = 2000) -> dict:
    ny = n_wall + 1 if axis == 1 else 4
    nz = n_wall + 1 if axis == 2 else 4
    nx = n_wall + 1 if axis == 0 else 4
    shape = (nz, ny, nx)
    solid = wall_mask(shape, axis)
    ux = torch.full(shape, u0, dtype=torch.get_default_dtype())
    f = _field_equilibrium(ux)
    for _ in range(steps):
        f = specular_step(f, tau=1.0, solid=solid, axis=axis)
    _, uxf, uyf, _ = macroscopic3d(f)
    dev = float((uxf - u0).abs().max() / u0)
    return {
        "axis": axis,
        "steps": steps,
        "max_relative_deviation": dev,
        "max_normal_velocity": float(uyf.abs().max()) if axis == 1 else None,
    }


# ---------------------------------------------------------------------------
# 案例 4：剪切波驻波 → 表观壁面位置与黏度恢复
# ---------------------------------------------------------------------------


def fit_mode(
    profile: torch.Tensor,
    k_min: float,
    k_max: float,
    coords: torch.Tensor | None = None,
    n_delta: int = 161,
    n_k: int = 201,
    delta_range: tuple[float, float] = (-0.5, 0.9),
) -> dict:
    """把轴向剖面拟合成 ``a·cos(k (x - δ))``（δ 与 k 双自由度最小二乘网格搜索）。

    ``coords`` 是剖面各点**在格子坐标下**的坐标（缺省 = 内部流体格点 1..N，
    即两端壁格点中心在 0 与 N+1）。这条不能省：若误用 0..N-1，坐标原点整体错位
    一个格子，δ 会被这个错位吸收掉（实测会把 δ 顶到搜索区间的边界上）。

    返回 ``delta``（表观自由滑移面相对壁格点中心、朝流体侧的内移量）、``k``、
    ``amplitude`` 与配准残差 ``residual``（max|u - fit| / max|u|）。
    """
    n = profile.numel()
    if coords is None:
        coords = torch.arange(1, n + 1, dtype=torch.get_default_dtype())
    best: tuple[float, float, float, float] | None = None
    for delta in torch.linspace(delta_range[0], delta_range[1], n_delta):
        for k in torch.linspace(k_min, k_max, n_k):
            basis = torch.cos(k * (coords - delta))
            denom = float((basis * basis).sum())
            if denom == 0.0:
                continue
            a = float((profile * basis).sum() / denom)
            if a == 0.0:
                continue
            resid = float((profile - a * basis).abs().max() / profile.abs().max())
            if best is None or resid < best[0]:
                best = (resid, float(delta), float(k), a)
    assert best is not None
    resid, delta, k, a = best
    return {"residual": resid, "delta": delta, "k": k, "amplitude": a}


def _channel_shape(n_wall: int, axis: int) -> tuple[tuple[int, int, int], torch.Tensor, int]:
    ny = n_wall + 1 if axis == 1 else 4
    nz = n_wall + 1 if axis == 2 else 4
    nx = n_wall + 1 if axis == 0 else 4
    shape = (nz, ny, nx)
    return shape, wall_mask(shape, axis), shape[2 - axis]


def shear_wave_case(
    n_wall: int = 32,
    axis: int = 1,
    tau: float = 1.0,
    amplitude: float = 1e-2,
    t_fit: int = 2000,
    t_window: int = 1000,
) -> dict:
    """两侧自由滑移壁之间的剪切驻波：测表观壁面位置 δ 与黏度恢复误差。

    初始场 ``u_x = A cos(k_nominal·coord)``，``k_nominal = π/n_wall``。
    先跑到 ``t_fit``（非本征分量按 ``exp(-ν m² k² t)`` 衰减，m=2 快 4 倍），
    再把剖面拟合成 ``a·cos(k_eff(y-δ))``；随后用拟合波数算衰减率 → ``nu_fitted``（判据），
    用标称波数算 → ``nu_nominal``（= "把壁面当成格点中心" 的做法，负判据）。
    """
    shape, solid, n_axis = _channel_shape(n_wall, axis)
    k_nom = math.pi / (n_axis - 1)
    coord = torch.arange(n_axis, dtype=torch.get_default_dtype())
    ux = _profile_field(shape, axis, amplitude * torch.cos(k_nom * coord))
    f = _field_equilibrium(ux)

    t2 = t_fit + t_window
    amps: dict[int, torch.Tensor] = {}
    for t in range(1, t2 + 1):
        f = specular_step(f, tau=tau, solid=solid, axis=axis)
        if t in (t_fit, t2):
            amps[t] = _axial_ux(f, axis)

    coords = torch.arange(1, n_axis - 1, dtype=torch.get_default_dtype())
    fit = fit_mode(amps[t_fit], 0.9 * k_nom, 1.1 * k_nom, coords=coords)
    basis = torch.cos(fit["k"] * (coords - fit["delta"]))
    denom = float((basis * basis).sum())
    a1 = float((amps[t_fit] * basis).sum() / denom)
    a2 = float((amps[t2] * basis).sum() / denom)
    nu_ana = (tau - 0.5) / 3.0
    nu_fit = -math.log(a2 / a1) / (fit["k"] ** 2 * t_window)
    nu_nom = -math.log(a2 / a1) / (k_nom ** 2 * t_window)
    return {
        "axis": axis,
        "n_wall": n_wall,
        "nominal_gap": float(n_axis - 1),
        "k_nominal": k_nom,
        "delta": fit["delta"],
        "k_fitted": fit["k"],
        "effective_gap": math.pi / fit["k"],
        "gap_minus_twice_delta": float(n_axis - 1) - 2.0 * fit["delta"],
        "shape_residual": fit["residual"],
        "amplitude_t1": a1,
        "amplitude_t2": a2,
        "nu_analytic": nu_ana,
        "nu_fitted": nu_fit,
        "nu_nominal": nu_nom,
        "err_fitted": abs(nu_fit - nu_ana) / nu_ana,
        "err_nominal": abs(nu_nom - nu_ana) / nu_ana,
        "window": [t_fit, t2],
    }


def noslip_control_case(
    n_wall: int = 32,
    axis: int = 1,
    tau: float = 1.0,
    amplitude: float = 1e-2,
    t_fit: int = 2000,
    t_window: int = 1000,
) -> dict:
    """负控：同样初始场、同样测量流程，只把反射算子换成半路反弹（无滑移）。

    余弦驻波在无滑移壁上不满足 ``∂u_x/∂n = 0``，真实壁面剪应力会额外抽走能量，
    因而按同一套流程量出来的表观 ν 必然远离 (τ-½)/3、剖面也不再是余弦。
    若这条判据不红，说明剪切波案例的判据没有鉴别力。
    """
    shape, solid, n_axis = _channel_shape(n_wall, axis)
    k_nom = math.pi / (n_axis - 1)
    coord = torch.arange(n_axis, dtype=torch.get_default_dtype())
    ux = _profile_field(shape, axis, amplitude * torch.cos(k_nom * coord))
    f = _field_equilibrium(ux)
    t2 = t_fit + t_window
    amps: dict[int, torch.Tensor] = {}
    for t in range(1, t2 + 1):
        f = noslip_step(f, tau=tau, solid=solid)
        if t in (t_fit, t2):
            amps[t] = _axial_ux(f, axis)

    coords = torch.arange(1, n_axis - 1, dtype=torch.get_default_dtype())
    fit = fit_mode(amps[t_fit], 0.9 * k_nom, 1.1 * k_nom, coords=coords)
    basis = torch.cos(fit["k"] * (coords - fit["delta"]))
    denom = float((basis * basis).sum())
    a1 = float((amps[t_fit] * basis).sum() / denom)
    a2 = float((amps[t2] * basis).sum() / denom)
    nu_ana = (tau - 0.5) / 3.0
    nu_app = -math.log(a2 / a1) / (fit["k"] ** 2 * t_window)
    return {
        "axis": axis,
        "n_wall": n_wall,
        "nu_analytic": nu_ana,
        "nu_apparent": nu_app,
        "err_apparent": abs(nu_app - nu_ana) / nu_ana,
        "shape_residual": fit["residual"],
        "amplitude_t1": a1,
        "amplitude_t2": a2,
    }


# ---------------------------------------------------------------------------
# 汇总（测试与工件共用的唯一数字来源）
# ---------------------------------------------------------------------------


def build_results(
    n_wall_list: Sequence[int] = (32, 64),
    t_fit_map: dict[int, int] | None = None,
    t_window: int = 1000,
) -> dict:
    """跑完全部案例，返回可直接 ``json.dump`` 的结果字典。"""
    torch.set_default_dtype(torch.float64)
    if t_fit_map is None:
        # 拟合时刻：m≥2 的高阶模态按 exp(-ν m² k² t) 衰减（k∝1/H，衰减速率 ∝ H⁻²）。
        # 实测 δ / 残差在 t_fit 翻倍后不变（H=32: 1000 vs 2000，H=64: 2000 vs 4000），
        # 故取较小值以控 CI 时长。
        t_fit_map = {32: 1000, 64: 2000}
    results: dict = {"cases": {}, "settings": {"dtype": "float64", "t_fit_map": dict(t_fit_map)}}

    results["cases"]["algebra"] = {f"axis{a}": specular_map_error(a) for a in (0, 1, 2)}
    results["cases"]["invariants"] = {f"axis{a}": operator_invariants(a) for a in (0, 1, 2)}

    results["cases"]["plug_flow"] = {
        f"gap{n}": plug_flow_case(n_wall=n) for n in (n_wall_list[0],)
    }
    results["cases"]["plug_flow"]["z_axis"] = plug_flow_case(n_wall=n_wall_list[0], axis=2)

    results["cases"]["momentum_free_slip"] = momentum_conservation_case(
        n_wall=n_wall_list[0], free_slip=True
    )
    results["cases"]["momentum_no_slip_control"] = momentum_conservation_case(
        n_wall=n_wall_list[0], free_slip=False
    )

    results["cases"]["shear_wave"] = {
        f"gap{n}": shear_wave_case(n_wall=n, t_fit=t_fit_map[n], t_window=t_window)
        for n in n_wall_list
    }
    results["cases"]["shear_wave"]["z_axis"] = shear_wave_case(
        n_wall=n_wall_list[0], axis=2, t_fit=t_fit_map[n_wall_list[0]], t_window=t_window
    )
    results["cases"]["noslip_control"] = noslip_control_case(
        n_wall=n_wall_list[0], t_fit=t_fit_map[n_wall_list[0]], t_window=t_window
    )
    results["tolerances"] = {
        "relative": REL_TOL,
        "exact": EXACT_TOL,
        "inset_range": list(INSET_RANGE),
    }
    return results


__all__ = [
    "EXACT_TOL",
    "INSET_RANGE",
    "REL_TOL",
    "build_results",
    "fit_mode",
    "momentum_conservation_case",
    "noslip_control_case",
    "noslip_step",
    "operator_invariants",
    "plug_flow_case",
    "shear_wave_case",
    "specular_map_error",
    "specular_step",
    "wall_mask",
]