#!/usr/bin/env python3
"""D3Q19 BGK 核心的五步解析解验证（pytest 形态，设备无关）。

五个案例都有可写出的精确解，通过判据一律为相对误差 < 1%：

  Step A 剪切波衰减   u_y(x,t)=A·cos(kx)·exp(-νk²t)          → 校验 ν=(τ-½)/3
  Step B Couette 流   u_x(y)=U·(y-½)/(H-½)（半路反弹无滑移）  → 校验壁面动量交换
  Step C Poiseuille 流 u(y)=G/(2ν)·(y-½)(H+½-y)               → 校验体力注入与剖面
  Step D 壁面摩擦     Cf=2ν/(L·U)（Couette 型摩擦系数）        → 校验壁面剪应力提取
  Step E 方管 Poiseuille 三维两向受限的 Boussinesq 级数解      → 校验 3D 角区（y、z 同时受限）

r3（2026-11-07）：加 Step E。A–D 全是"单方向受限 + 另一方向周期"的通道流，
即二维几何；D3Q19 在三维里最容易露馅的恰恰是方管角区（两条壁的反弹算子叠加、
剪应力由两向共同贡献），此前完全没有解析解覆盖。Step E 同时给出文献积分量
f·Re=56.91（⇔ R=28.455）的独立对表。

历史背景（2026-11-03 复核）：本模块原先以「脚本 + 硬件钉扎默认参数」的形态存在，
`def test_step_x(device="sdaa:0")` 使 CPU-only runner 在构造 torch.device 时直接报错，
加上函数 `return bool`（pytest 视为非法返回值），四个案例长期躺在
`tests/ci_quarantine.txt` 里从未真正裁决过。本版本把它改成可被 CI 真跑的用例：
设备默认 CPU（`TENSORLBM_TEST_DEVICE` 可覆盖），断言而非返回布尔，
并把逐分量 Python 循环改成整张量布尔掩码/广播（数学逐点等价，不改变任何数值定义）。

命令行形态保留：`python tests/test_lbm_verification.py` 打印全部测量值并汇总。

Step F（2026-11-08）：同一条梯子从"BGK 专用"扩成"算子不变量"——A/B/C/E 四个算例
都对碰撞算子参数化，MRT / TRT / 正则化 BGK 必须给出同一组解析解（同样 1% 门）。
意义：`docs/d3q19-d3q27-mrt-consistency-audit.md` 明确只证明"Mrt 与 BGK 互相一致"、
"不外推到物理解精度"；Step F 把这句话补上——高级碰撞算子现在有解析解锚，
不再只有算子间自洽。TRT（Λ=3/16）在 Step C 上把半路反弹的壁面位置误差压到
0 量级、在 Step E 角区把误差降到 BGK 的 0.57 倍，这是"残差归属=壁面位置误差"
的独立证据（BGK 的 0.34% 不是强迫项或可压缩性引入的）。

r2（2026-11-07）：把这条梯子从"能红能绿"升级为"能出证据"——
同一段代码既做断言（pytest 门），也能把实测数字写成 JSON（`--json`），
再由 `scripts/lbm_verification_report.py` 程序化生成数字表与对照图，
避免报告里的数字靠手抄漂移。dtype 由 `TENSORLBM_TEST_DTYPE` 解析
（缺省 float64：验证的是数值方法本身，不是 float32 舍入地板），
并新增 Step C 的 float32 对照用例，把 dtype 敏感度变成被判据约束的事实。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

try:  # 直接以脚本运行时也能找到 src/
    from tensorlbm.boundaries3d import bounce_back_cells_3d
    from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import (
        collide_bgk3d,
        collide_mrt3d,
        collide_rlbm3d,
        collide_trt3d,
        stream3d,
    )
except ImportError:  # pragma: no cover - 脚本运行路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
    from tensorlbm.boundaries3d import bounce_back_cells_3d
    from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import (
        collide_bgk3d,
        collide_mrt3d,
        collide_rlbm3d,
        collide_trt3d,
        stream3d,
    )

pytestmark = [pytest.mark.physics]

# 全部案例的统一判据：相对误差上限。
REL_TOL = 0.01  # 1%


TOL_PCT = 1.0  # 统一判据的另一写法（百分比）：报告里逐项列"误差% vs 判据%"

# ---------------------------------------------------------------------------
# 碰撞算子注册表：解析解梯子不绑定 BGK（见 Step F）
# ---------------------------------------------------------------------------
COLLISIONS = ("bgk", "mrt", "trt", "rlbm")


def collide(f: torch.Tensor, tau: float, collision: str = "bgk") -> torch.Tensor:
    """按名字调用同一条量纲约定下的碰撞算子（ν 一律 = (τ-½)/3）。

    * ``bgk``  —— 单弛豫 BGK（原梯子的默认路径，数值不变）
    * ``mrt``  —— D3Q19 多弛豫，剪切矩弛豫率 = 1/τ ⇒ 剪切粘度与 BGK 一致，
                  非流体力学矩用库默认率（s_e=1.19, s_eps=1.4, s_q=1.2）
    * ``trt``  —— 双弛豫，对称部分 τ₊=τ，反称部分由魔术参数 Λ=3/16 定
                  （Ginzburg 2008：Λ=3/16 消除半路反弹的壁面位置误差）
    * ``rlbm`` —— 正则化 BGK（二阶 Hermite 投影掉鬼模）
    """
    if collision == "bgk":
        return collide_bgk3d(f, tau=tau)
    if collision == "mrt":
        return collide_mrt3d(f, tau=tau)
    if collision == "trt":
        return collide_trt3d(f, tau_plus=tau)
    if collision == "rlbm":
        return collide_rlbm3d(f, tau=tau)
    raise ValueError(f"未知碰撞算子 {collision!r}，可选 {COLLISIONS}")


def _device() -> torch.device:
    """验证用设备：默认 CPU，可用 TENSORLBM_TEST_DEVICE=cuda:0 / sdaa:0 覆盖。

    解析解验证与设备无关，默认 CPU 是为了让 CPU-only runner 也能裁决，
    不再因为 `sdaa:0` 这种硬件钉扎而跳过。显式请求的加速器不可用时 *skip*
    而不是伪绿：环境缺硬件不等于验证通过，但也不该把整套输入判成 FAIL。
    """
    name = os.environ.get("TENSORLBM_TEST_DEVICE", "cpu")
    if name.startswith("sdaa"):
        sdaa = getattr(torch, "sdaa", None)
        if sdaa is None or not sdaa.is_available():
            pytest.skip(f"TENSORLBM_TEST_DEVICE={name!r} 但 torch.sdaa 不可用")
    elif name.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip(f"TENSORLBM_TEST_DEVICE={name!r} 但 CUDA 不可用")
    return torch.device(name)


def resolve_dtype(explicit: torch.dtype | None = None) -> torch.dtype:
    """dtype 解析：explicit > ``TENSORLBM_TEST_DTYPE`` > float64。

    缺省 float64 是刻意的：这条梯子裁决的是数值方法（碰撞核/边界/强迫）是否正确，
    float32 的单精度舍入地板会把方法误差淹没（见 docs/lbm_verification_ladder_r2.md
    的 dtype 对照表）。需要 float32 视点时用环境变量切换，同一份断言仍然生效。
    """
    if explicit is not None:
        return explicit
    name = os.environ.get("TENSORLBM_TEST_DTYPE", "float64").lower()
    if name not in {"float32", "float64"}:
        raise ValueError(f"TENSORLBM_TEST_DTYPE 只接受 float32/float64，收到 {name!r}")
    return getattr(torch, name)


@pytest.fixture(autouse=True)
def _verification_dtype():
    """把 torch 默认 dtype 钉到 :func:`resolve_dtype` 的解析结果（用后恢复）。

    四个案例的张量都由默认 dtype 构造，因此这一个钩子就能让整条梯子换精度，
    不必在每个 case 里传 dtype 参数（也保证断言与报告数字走的是同一个精度）。
    """
    old = torch.get_default_dtype()
    torch.set_default_dtype(resolve_dtype())
    try:
        yield
    finally:
        torch.set_default_dtype(old)


@pytest.fixture(autouse=True)
def _single_thread(monkeypatch):
    """把 torch 线程数钉到 1（用后恢复）。

    这些案例的格点场只有 ~10³–10⁴ 个单元，逐分量小张量算子在多线程下线程调度
    开销远大于计算本身：实测在本机（nproc=128、load15>100）不钉线程时三个案例
    115 s 跑不完（user 11m29s），钉 1 线程后 79 s 全部跑完。
    """
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def _no_dynamics_and_bounce(
    f: torch.Tensor, f_pre: torch.Tensor, solid: torch.Tensor
) -> torch.Tensor:
    """NoDynamics（固体格点保留**碰撞前**分布）+ 半路反弹。

    等价于历史实现里的 ``for q in range(19): f[q] = where(sm[q], f_pre[q], f[q])``
    ——逐点相同，只是用整张量布尔掩码一次性完成。

    注意 f_pre 必须是**碰撞调用之前**的快照：若误用碰撞后的分布回填，固体格点每步
    都会带着碰撞结果再反弹，等效于往壁面注入伪动量（实测 50 步后场量偏差 4e-3，
    Couette 剖面误差从 0% 涨到 4.8%，Poiseuille 涨到 17%）。
    """
    f = torch.where(solid.unsqueeze(0).expand_as(f), f_pre, f)
    return bounce_back_cells_3d(f, solid)


def _friction_coefficient(nu: float, gap: float, u_top: float) -> float:
    """Couette 型摩擦系数 Cf = 2τ_w/(ρU²)，其中 τ_w = ν·U/L。"""
    return 2.0 * nu / (gap * u_top)


# ---------------------------------------------------------------------------
# Step A：剪切波衰减 → 粘度
# ---------------------------------------------------------------------------
def shear_wave_case(
    device: torch.device | None = None,
    *,
    nx: int = 64,
    steps: int = 100,
    tau: float = 1.0,
    collision: str = "bgk",
) -> dict:
    """测量剪切波衰减给出的有效粘度，并与 ν=(τ-½)/3 比对。

    默认参数即 ``test_step_a_shear_wave`` 断言的算例；``nx`` / ``steps`` / ``tau``
    只是为 ``scripts/verify_lbm_analytic_convergence.py`` 的网格收敛与 τ 负控留的
    入口，走的是同一段代码，published 数字因此不会与断言漂移。
    """
    d = device if device is not None else _device()
    ny, nz = 4, 4
    nu_expected = (tau - 0.5) / 3.0
    k = 2 * math.pi / nx
    a0 = 0.01

    uy = torch.zeros(nz, ny, nx, device=d)
    # 跟随 torch 默认 dtype：默认 float32（CI 断言用），脚本做阶数测量时切 float64
    xs = torch.arange(nx, device=d, dtype=torch.get_default_dtype())
    uy[:, :] = (a0 * torch.cos(k * xs)).expand(nz, ny, nx)
    zero = torch.zeros(nz, ny, nx, device=d)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, uy, zero)

    _, _, uy0, _ = macroscopic3d(f)
    a_init = float(uy0[0, 0, 0])

    history = [a_init]
    for _ in range(steps):
        f = collide(f, tau, collision)
        f = stream3d(f)
        _, _, uy_t, _ = macroscopic3d(f)
        history.append(float(uy_t[0, 0, 0]))

    a_final = history[-1]
    nu_measured = -math.log(abs(a_final / a_init)) / (k**2 * steps)
    err = abs(nu_measured - nu_expected) / nu_expected

    return {
        "case": "A_shear_wave",
        "collision": collision,
        "tau": tau,
        "steps": steps,
        "nx": nx,
        "history": history,
        "nu_expected": nu_expected,
        "nu_measured": nu_measured,
        "amplitude_init": a_init,
        "amplitude_final": a_final,
        "rel_err": err,
    }


def test_step_a_shear_wave() -> None:
    """剪切波衰减给出 ν=(τ-½)/3，相对误差 < 1%。"""
    r = shear_wave_case()
    assert abs(r["amplitude_final"]) < abs(r["amplitude_init"]), "剪切波未衰减，碰撞核有问题"
    assert r["rel_err"] < REL_TOL, (
        f"Step A 粘度相对误差 {r['rel_err']:.3%} ≥ {REL_TOL:.0%}："
        f"ν_num={r['nu_measured']:.6f} vs ν_exact={r['nu_expected']:.6f}"
    )


# ---------------------------------------------------------------------------
# Step B：Couette 流 → 半路反弹无滑移
# ---------------------------------------------------------------------------
def couette_case(
    device: torch.device | None = None,
    *,
    ny: int = 12,
    steps: int = 1000,
    bounce_back: bool = True,
    collision: str = "bgk",
) -> dict:
    """Couette 剖面算例。``bounce_back=False`` 只用于负控（关掉壁面反弹）。"""
    d = device if device is not None else _device()
    nx, nz = 80, 4
    tau = 1.0
    u_top = 0.05

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    zero = torch.zeros(nz, ny, nx, device=d)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, zero, zero)
    rho1 = torch.ones(nz, ny, nx, device=d)
    feq_top = equilibrium3d(rho1, torch.full_like(rho1, u_top), zero, zero)

    for _ in range(steps):
        f_pre = f.clone()
        f = collide(f, tau, collision)
        if bounce_back:
            f = _no_dynamics_and_bounce(f, f_pre, solid)
        else:  # 负控：NoDynamics 保留碰撞前分布，但不做壁面反弹
            f = torch.where(solid.unsqueeze(0).expand_as(f), f_pre, f)
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        f = stream3d(f)
        # 进出口周期性（x 方向复制倒数第二列）
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy().astype(float)
    y = np.arange(1, ny - 1)
    u_exact = u_top * (y - 0.5) / (ny - 1 - 0.5)
    rel_err = float(np.max(np.abs(u - u_exact)) / u_top)

    return {
        "case": "B_couette",
        "collision": collision,
        "tau": tau,
        "steps": steps,
        "bounce_back": bounce_back,
        "u_top": u_top,
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
        "y": y.tolist(),
        "rel_err": rel_err,
    }


def test_step_b_couette() -> None:
    """Couette 线性剖面（半路反弹 ⇒ 壁面在 y=½ 与 y=ny-1+½），相对误差 < 1%。"""
    r = couette_case()
    assert r["rel_err"] < REL_TOL, (
        f"Step B 剖面相对误差 {r['rel_err']:.3%} ≥ {REL_TOL:.0%}："
        f"u_num={np.round(r['u_num'], 6).tolist()} u_exact={np.round(r['u_exact'], 6).tolist()}"
    )


# ---------------------------------------------------------------------------
# Step C：Poiseuille 流 → 体力 + 剖面
# ---------------------------------------------------------------------------
def poiseuille_case(
    device: torch.device | None = None,
    *,
    ny: int = 12,
    steps: int = 1000,
    collision: str = "bgk",
) -> dict:
    d = device if device is not None else _device()
    nx, nz = 80, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    height = ny - 2
    u_max = 0.05
    g_force = 2 * nu * u_max / height**2
    # 收敛性实测：1000/1500/2000/3000 步的相对误差同为 0.3378% ⇒ 该残差是
    # 半路反弹 + Guo 离散在 ny=12 下的稳态离散误差（不随步数下降），取 1000 步。

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    # 离散速度/权重跟当前默认 dtype 走：写死 .float() 会让 float64 场量里混进
    # float32 的体力增量（类型提升后仍能跑，但把 float32 舍入引进 float64 判据）
    c = C.to(d).to(torch.get_default_dtype())
    w = W.to(d).to(torch.get_default_dtype())
    # Guo 体力的动量项：w_q·3·c_qx·G（逐格点常数，可广播到整张量）
    accel = (w * 3.0 * c[:, 0]).view(19, 1, 1, 1) * g_force

    zero = torch.zeros(nz, ny, nx, device=d)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, zero, zero)

    for _ in range(steps):
        f_pre = f.clone()
        f = collide(f, tau, collision)
        f = _no_dynamics_and_bounce(f, f_pre, solid)
        f = f + accel
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy().astype(float)
    y = np.arange(1, ny - 1)
    u_exact = g_force / (2 * nu) * (y - 0.5) * (height + 0.5 - y)
    rel_err = float(np.max(np.abs(u - u_exact)) / max(u_exact.max(), 1e-12))

    return {
        "case": "C_poiseuille",
        "collision": collision,
        "tau": tau,
        "steps": steps,
        "nu": nu,
        "body_force": g_force,
        "height": height,
        "u_max_num": float(u.max()),
        "u_max_exact": float(u_exact.max()),
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
        "y": y.tolist(),
        "rel_err": rel_err,
    }


def test_step_c_poiseuille() -> None:
    """Poiseuille 抛物线剖面（Guo 体力），相对误差 < 1%。"""
    r = poiseuille_case()
    assert r["rel_err"] < REL_TOL, (
        f"Step C 剖面相对误差 {r['rel_err']:.3%} ≥ {REL_TOL:.0%}："
        f"u_max_num={r['u_max_num']:.6f} u_max_exact={r['u_max_exact']:.6f}"
    )


@pytest.mark.physics
def test_step_c_poiseuille_float32_error_is_not_rounding_bound() -> None:
    """dtype 敏感度：Step C 的 float32 误差必须仍 < 1%，且与 float64 同量级。

    Step C 的残差（~0.34%）来自半路反弹 + Guo 离散的可压缩性离散误差，属物理/离散
    误差而非舍入误差。因此 float32 与 float64 必须给出同量级误差；若 float32 明显
    更差，说明场量尺度已贴到单精度舍入地板，该 dtype 下"误差 < 1%"的判据不再有
    鉴别力（这条用例就是那个前提的守卫）。
    """
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        r32 = poiseuille_case()
    finally:
        torch.set_default_dtype(old)
    r64 = poiseuille_case()
    assert r32["rel_err"] < REL_TOL, f"float32 下 Step C 误差 {r32['rel_err']:.3%} ≥ {REL_TOL:.0%}"
    ratio = r32["rel_err"] / max(r64["rel_err"], 1e-12)
    assert 0.3 <= ratio <= 3.0, (
        f"float32/float64 误差比 {ratio:.3g} 超出 [0.3, 3]：float32 误差已由舍入主导"
        f"（f32={r32['rel_err']:.4%} f64={r64['rel_err']:.4%}）"
    )


def poiseuille_slip_metrics(ny: int, steps: int) -> dict:
    """把 Step C 剖面误差分解为「均匀滑移偏移」与「形状误差」两支。

    Step C 的 ~0.34%（ny=12）残差**不是剖面形状算错**：数值剖面是「精确抛物线 +
    一个与 y 无关的均匀速度偏移」（半路反弹壁面上的有限滑移），形状偏差只有 1e-5
    量级。只盯 ``max|u-u_ex|`` 会把滑移当成形状误差，从而看不出加密网格是否真在
    收敛（步数扫描本身也无法区分二者）。
    """
    r = poiseuille_case(ny=ny, steps=steps)
    u = np.asarray(r["u_num"], dtype=float)
    e = np.asarray(r["u_exact"], dtype=float)
    u_max = float(np.max(np.abs(e)))
    offset = float(np.mean(u - e))
    return {
        "ny": ny,
        "h": ny - 2,
        "steps": steps,
        "rel_err": float(np.max(np.abs(u - e)) / u_max),
        "rel_err_shape": float(np.max(np.abs((u - offset) - e)) / u_max),
        "offset": offset,
        "u_max_exact": u_max,
    }


def test_step_c_poiseuille_residual_is_second_order_slip() -> None:
    """验证欠账：Step C 的残差必须随网格加密**二阶收敛**，且主体是均匀滑移。

    此前只有「步数扫描」证明该残差不随步数下降（1000–3000 步同为 0.3378%），据此
    断言它是离散误差；但「不随步数下降」同样可能意味着代码有 bug（稳态解本身错）。
    密网格裁决才是唯一证据：若残差是二阶收敛的滑移，加密 h=6→h=10 的总量误差应降
    ~2.8 倍（实测 0.9534%→0.3378%），且去掉均匀偏移后的形状误差已到 1e-5 量级。

    判据（全部机器裁）：形状误差 < 总量误差的 10%、观测阶 p∈[1.5, 2.6]、
    |offset|·h² 在 ±20% 内为常数（滑移律 offset≈G/4）、ny=12 时判据余量 ≥2 倍。
    """
    coarse = poiseuille_slip_metrics(8, 500)   # h=6，步数按 h² 缩放（500∝6²）
    fine = poiseuille_slip_metrics(12, 1000)   # h=10

    # 1) 残差主体是均匀偏移，不是形状误差
    for r in (coarse, fine):
        assert r["rel_err_shape"] < 0.1 * r["rel_err"], (
            f"ny={r['ny']}：形状误差 {r['rel_err_shape']:.3%} 已达总量误差 "
            f"{r['rel_err']:.3%} 的 10%，残差不再能用「均匀滑移」解释"

        )

    # 2) 总量误差二阶收敛
    p = math.log(coarse["rel_err"] / fine["rel_err"]) / math.log(fine["h"] / coarse["h"])
    assert 1.5 <= p <= 2.6, (
        f"观测收敛阶 p={p:.2f} 落在 [1.5, 2.6] 之外：Step C 残差不是二阶离散误差"
        f"（ny=8:{coarse['rel_err']:.4%} ny=12:{fine['rel_err']:.4%}）"
    )

    # 3) 滑移服从 offset·h²≈const（⇒ offset≈G/4，G=2νu_max/h²）
    ratio = (abs(coarse["offset"]) * coarse["h"] ** 2) / (abs(fine["offset"]) * fine["h"] ** 2)
    assert 0.8 <= ratio <= 1.2, (
        f"|offset|·h² 在 h=6 与 h=10 之比为 {ratio:.3f}，偏离 1 超过 ±20%："
        f"滑移标度律 offset≈G/4 不成立"
    )

    # 4) 判据余量：ny=12 的默认网格上 1% 判据至少还有 2 倍余量
    assert fine["rel_err"] < 0.5 * REL_TOL, (
        f"ny=12 误差 {fine['rel_err']:.3%} 已吃掉 1% 判据的一半以上"
    )


# ---------------------------------------------------------------------------
# Step D：壁面摩擦 → 剪应力提取
# ---------------------------------------------------------------------------
def friction_case(
    device: torch.device | None = None,
    *,
    ny: int = 12,
    steps: int = 1000,
) -> dict:
    d = device if device is not None else _device()
    nx, nz = 80, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_top = 0.05
    gap = ny - 1.5  # 底壁 y=½ 到顶壁 y=ny-1+½（半路反弹）
    cf_exact = _friction_coefficient(nu, gap, u_top)
    # 收敛性实测：500 步 0.11%、800 步 0.0041%、1000 步 0.0040% ⇒ 取 1000 步。

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    zero = torch.zeros(nz, ny, nx, device=d)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, zero, zero)
    rho1 = torch.ones(nz, ny, nx, device=d)
    feq_top = equilibrium3d(rho1, torch.full_like(rho1, u_top), zero, zero)

    for _ in range(steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        f = _no_dynamics_and_bounce(f, f_pre, solid)
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u_wall = float(ux[0, 1, nx // 2])
    du_dy = u_wall / 0.5  # 半路反弹：壁面位于 y=½，首个流体节点在 y=1
    tau_w = nu * du_dy
    cf_num = 2 * tau_w / u_top**2
    rel_err = abs(cf_num - cf_exact) / cf_exact

    return {
        "case": "D_friction",
        "tau": tau,
        "steps": steps,
        "nu": nu,
        "u_top": u_top,
        "gap": gap,
        "u_wall": u_wall,
        "du_dy": du_dy,
        "tau_w": tau_w,
        "cf_num": cf_num,
        "cf_exact": cf_exact,
        "rel_err": rel_err,
    }


def test_step_d_friction() -> None:
    """壁面剪应力给出的 Cf=2ν/(L·U)（Couette 型摩擦系数），相对误差 < 1%。

    注意定义：这里 Cf=2τ_w/(ρU²) 且 τ_w=νU/L，因此 Cf≡2ν/(L·U) 是**摩擦系数**，
    不是工程阻力系数 C_D（后者要按湿面积/迎流面积归一）。判据只在此定义下成立。
    """
    r = friction_case()
    assert r["rel_err"] < REL_TOL, (
        f"Step D 摩擦系数相对误差 {r['rel_err']:.3%} ≥ {REL_TOL:.0%}："
        f"Cf_num={r['cf_num']:.6f} vs Cf_exact={r['cf_exact']:.6f}"
    )


# ---------------------------------------------------------------------------
# Step E：3D 方管 Poiseuille → D3Q19 的二维受限（含角区）解析解
# ---------------------------------------------------------------------------
# A–D 四个案例都是"单方向受限"的通道流：y 向一层壁 + x 向周期。它们证明了 D3Q19 的
# 碰撞核/半路反弹/Guo 强迫在**二维几何**下正确，却完全没碰到"两个方向同时受限"
# 的情形——方管的角区才是 D3Q19 在 3D 里最容易露馅的地方（角点上两条壁的反弹算子
# 叠加、角区剪应力由两个方向共同贡献）。Step E 用 Boussinesq 级数精确解补这一级。
def _duct_coords(length: int) -> np.ndarray:
    """方管截面流体格点的相对坐标（单位 lu，管心为 0）。

    半路反弹把壁面放在 $j=\\pm(L+1)/2$，流体格点为 $j=1..L$，故
    $Y,Z\\in\\{\\pm\\frac12,\\pm\\frac32,\\dots,\\pm(a-\\frac12)\\}$，$a=L/2$。
    """
    return np.arange(1, length + 1, dtype=float) - 0.5 * (length + 1)


def duct_series(
    Y: np.ndarray,
    Z: np.ndarray,
    length: int,
    body_force: float,
    nu: float,
    *,
    n_terms: int = 401,
    order: int = 0,
) -> np.ndarray:
    r"""方管 Poiseuille 的 Boussinesq 级数解（``order=2`` 时返回其解析 Laplacian）。

    截面 $[-a,a]^2$（边长 $L=2a$）、四壁无滑移、沿管轴均匀体力 $g$ 下，充分发展流满足

    .. math:: \nabla^2 u = -g/\nu,\qquad u=0\ \text{on walls}

    的级数解

    .. math::

        u(Y,Z)=\frac{16\,(g/\nu)\,a^2}{\pi^3}\sum_{n\ \text{odd}}
        \frac{(-1)^{(n-1)/2}}{n^3}
        \left[1-\frac{\cosh(n\pi Z/2a)}{\cosh(n\pi/2)}\right]\cos\frac{n\pi Y}{2a}

    ``order=2`` 利用 $\partial^2_Y+\partial^2_Z$ 作用后方括号项恰好相消（等于 1），
    逐项求导得 $-\sum_n (-1)^{(n-1)/2}k_n^2\cos(k_nY)/n^3\cdot 16(g/\nu)a^2/\pi^3$，
    即**不依赖差分格式**的独立自检量（见 ``test_step_e_duct_series_satisfies_poisson``）。

    ``cosh`` 比值一律用指数形式计算，避免 $n$ 大时 ``cosh`` 溢出：
    $\cosh(kZ)/\cosh(ka)=e^{-k(a-|Z|)}(1+e^{-2k|Z|})/(1+e^{-2ka})$。
    """
    a = 0.5 * length
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=float)
    out = np.zeros(np.broadcast(Y, Z).shape)
    for n in range(1, n_terms + 1, 2):
        k = n * math.pi / (2.0 * a)
        sign = (-1) ** ((n - 1) // 2) / n**3
        cos_kY = np.cos(k * Y)
        if order == 2:
            out += sign * (-k * k) * cos_kY
        else:
            d = a - np.abs(Z)
            ratio = np.exp(-k * d) * (1.0 + np.exp(-2.0 * k * np.abs(Z))) / (
                1.0 + np.exp(-2.0 * k * a)
            )
            out += sign * (1.0 - ratio) * cos_kY
    return out * (16.0 * (body_force / nu) * a * a / math.pi**3)


def duct_profile_exact(
    length: int,
    body_force: float,
    nu: float,
    *,
    n_terms: int = 401,
) -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(u_exact[iz, iy], coord)``：与 :func:`duct_poiseuille_case` 的格点一一对应。"""
    coord = _duct_coords(length)
    Z, Y = np.meshgrid(coord, coord, indexing="ij")
    return duct_series(Y, Z, length, body_force, nu, n_terms=n_terms), coord


#: 方管（正方形截面）无量纲阻力的文献值：$f\cdot Re_{D_h}=56.91$（Darcy 摩擦因子、
#: 按水力直径 $D_h=L$ 归一），等价于 $R\equiv gL^2/(\nu\bar u)=28.455$。
#: 来源：Shah & London, *Laminar Flow Forced Convection in Ducts* (1978) 方管条目；
#: 本文件的 28.455 由 $56.91/2$ 得到（$f=2R/Re$ 的定义下）。
SQUARE_DUCT_RESISTANCE = 28.455


def duct_poiseuille_case(
    device: torch.device | None = None,
    *,
    length: int = 10,
    nx: int = 4,
    steps: int = 2000,
    tau: float = 1.0,
    u_nominal: float = 0.05,
    collision: str = "bgk",
) -> dict:
    """方管 Poiseuille 算例（y、z 两向同时受限，x 向周期）。

    ``u_nominal`` 只用于定标体力 $g=8\\nu u_{nominal}/L^2$（该式对平面通道精确：
    它给出 $u_{center}=gL^2/8\\nu$；方管实测中心速度约为其 0.589 倍）。取 0.05 是为
    让方管实际 Mach 数与 Step C 同量级（~0.05），便于横向比较两者的离散误差。

    默认 $L=10$、2000 步；``scripts`` 侧要网格/步数敏感性时改这两个参数即可，
    走的是同一段代码（published 数字因此不会与断言漂移）。
    """
    d = device if device is not None else _device()
    ny = nz = length + 2
    nu = (tau - 0.5) / 3.0
    body_force = 8.0 * nu * u_nominal / length**2

    c = C.to(d).to(torch.get_default_dtype())
    w = W.to(d).to(torch.get_default_dtype())
    accel = (w * 3.0 * c[:, 0]).view(19, 1, 1, 1) * body_force

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[0, :, :] = True   # z 向底壁
    solid[-1, :, :] = True  # z 向顶壁
    solid[:, 0, :] = True   # y 向侧壁
    solid[:, -1, :] = True  # y 向侧壁

    dflt = torch.get_default_dtype()
    zero = torch.zeros(nz, ny, nx, device=d, dtype=dflt)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d, dtype=dflt), zero, zero, zero)

    for _ in range(steps):
        f_pre = f.clone()
        f = collide(f, tau, collision)
        f = _no_dynamics_and_bounce(f, f_pre, solid)
        f = f + accel
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]   # 进出口周期性
        f[:, :, :, -1] = f[:, :, :, -2]

    _, ux, _, _ = macroscopic3d(f)
    u = ux.numpy()[1:-1, 1:-1, nx // 2]  # (iz, iy)
    u_exact, coord = duct_profile_exact(length, body_force, nu)

    u_max_exact = float(u_exact.max())
    rel_err = float(np.max(np.abs(u - u_exact)) / u_max_exact)
    # 无量纲阻力 R = gL²/(ν·ū)：与截面剖面的节点值不同泛函（积分量），
    # 可与文献 f·Re=56.91（⇔R=28.455）独立对表。
    u_mean_num = float(u.mean())
    u_mean_exact = float(u_exact.mean())
    resistance_num = body_force * length**2 / (nu * u_mean_num)
    resistance_exact = body_force * length**2 / (nu * u_mean_exact)

    return {
        "case": "E_duct_poiseuille",
        "collision": collision,
        "length": length,
        "nx": nx,
        "steps": steps,
        "tau": tau,
        "nu": nu,
        "body_force": body_force,
        "u_max_num": float(u.max()),
        "u_max_exact": u_max_exact,
        "u_mean_num": u_mean_num,
        "u_mean_exact": u_mean_exact,
        "resistance_num": resistance_num,
        # 自洽基准：级数解在**同一批格点上**取算术平均（与 LBM 的取平均方式一致），
        # 这样比较量是求解器误差，不含"节点平均 vs 连续积分"的求积偏差。
        "resistance_exact_nodal": resistance_exact,
        "resistance_literature": SQUARE_DUCT_RESISTANCE,
        "resistance_err_nodal": abs(resistance_num - resistance_exact) / resistance_exact,
        # 对文献值 28.455 的偏差：其中 ~1.2% 来自节点平均代替截面积分（口径差，加密网格
        # 会自行收敛：L=10 → 1.40%，L=14 → 0.73%），剩余才是求解器误差。故对文献只设 3% 门。
        "resistance_err_literature": abs(resistance_num - SQUARE_DUCT_RESISTANCE)
        / SQUARE_DUCT_RESISTANCE,
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
        "u_err": np.abs(u - u_exact).tolist(),
        "diag_coord": coord.tolist(),
        "diag_num": np.diagonal(u).tolist(),
        "diag_exact": np.diagonal(u_exact).tolist(),
        "rel_err": rel_err,
        "err_center": float(
            abs(u[length // 2 - 1, length // 2 - 1] - u_exact[length // 2 - 1, length // 2 - 1])
            / u_max_exact
        ),
        "err_l2": float(np.linalg.norm(u - u_exact) / np.linalg.norm(u_exact)),
    }


def test_step_e_duct_series_satisfies_poisson() -> None:
    """先验证"基准本身"：级数解的 Laplacian（解析求导）必须等于 −g/ν，且壁面为零。

    这是 PLAYBOOK 的"先给工具造解析基准"：若级数写错（系数/符号/坐标原点错位），
    后面 0.4% 的 LBM 误差就没有意义。判据用 $\\max|\\nabla^2u+g/\\nu|/(g/\\nu)$，
    网格取 Step E 的流体格点 $|Y|,|Z|\\le a-\\frac12$（壁面 $\\pm a$ 处级数本身为零，不在其中）。
    """
    length, nu = 10, (1.0 - 0.5) / 3.0
    body_force = 8.0 * nu * 0.05 / length**2
    coord = _duct_coords(length)
    Z, Y = np.meshgrid(coord, coord, indexing="ij")

    # 1) 壁面无滑移：级数在 |Y|=a 或 |Z|=a 处必须为 0（相对尺度 1e-12）
    a = 0.5 * length
    wall = np.linspace(-a, a, 7)
    for Yw, Zw in ((np.full_like(wall, a), wall), (wall, np.full_like(wall, a))):
        val = np.abs(duct_series(Yw, Zw, length, body_force, nu, n_terms=401)).max()
        assert val < 1e-12 * abs(duct_series(np.zeros((1, 1)), np.zeros((1, 1)),
                                             length, body_force, nu, n_terms=401))[0, 0], (
            f"级数解在壁面不为零（max|u_wall|={val:.3e}），基准本身有误"
        )

    # 2) 对称性 u(Y,Z)=u(-Y,Z)=u(Y,-Z)
    u = duct_series(Y, Z, length, body_force, nu, n_terms=401)
    assert np.allclose(u, duct_series(-Y, Z, length, body_force, nu, n_terms=401), atol=1e-18)
    assert np.allclose(u, duct_series(Y, -Z, length, body_force, nu, n_terms=401), atol=1e-18)

    # 3) Poisson：解析 Laplacian 与 −g/ν 的最大相对偏差随项数单调下降（⇒ 残差是截断）
    resid = {}
    for n_terms in (201, 801):
        lap = duct_series(Y, Z, length, body_force, nu, n_terms=n_terms, order=2)
        resid[n_terms] = float(np.abs(lap + body_force / nu).max() / (body_force / nu))
    assert resid[801] < 1e-2, (
        f"级数解 Laplacian 残差 {resid[801]:.3e} ≥ 1e-2：基准解不满足 Poisson 方程"
    )
    assert resid[201] > resid[801], (
        f"残差随项数未下降（201 项 {resid[201]:.3e} vs 801 项 {resid[801]:.3e}）："
        "残差不是级数截断，基准形式可能有误"
    )


def test_step_e_duct_poiseuille() -> None:
    """3D 方管（两向受限）Poiseuille 剖面 vs Boussinesq 级数解，相对误差 < 1%。

    判据取**截面逐格点最大偏差 / u_max**（比中心点更严）：实测 0.416%，中心点误差
    0.223%。另用积分量（截面质量通量）交叉验证：$R=gL^2/(\\nu\\bar u)$ 与级数解在同一
    批格点上取平均的偏差须 <1%（实测 0.233%，该口径把求积偏差消掉，只留求解器误差）；
    与文献 $f\\cdot Re=56.91$（⇔28.455）的偏差另有约 1.2% 是节点平均的求积偏差
    （L=14 降到 0.73%），故对文献只设 3% 门——不拿口径差冒充求解器误差。
    """
    r = duct_poiseuille_case()
    assert r["rel_err"] < REL_TOL, (
        f"Step E 方管剖面最大偏差 {r['rel_err']:.3%} ≥ {REL_TOL:.0%}："
        f"u_max_num={r['u_max_num']:.6f} u_max_exact={r['u_max_exact']:.6f}"
    )
    assert r["resistance_err_nodal"] < REL_TOL, (
        f"Step E 质量通量口径 R_num={r['resistance_num']:.4f} vs 级数解同批格点平均 "
        f"{r['resistance_exact_nodal']:.4f}，相对偏差 {r['resistance_err_nodal']:.3%} ≥ {REL_TOL:.0%}"
    )
    assert r["resistance_err_literature"] < 0.03, (
        f"Step E 无量纲阻力 R_num={r['resistance_num']:.4f} vs 文献 f·Re=56.91（⇔R=28.455）"
        f"偏差 {r['resistance_err_literature']:.3%} ≥ 3%：超出节点求积偏差可解释范围"
    )


@pytest.mark.physics
def test_step_e_duct_error_is_discretization_limited() -> None:
    """Step E 的 0.42% 是离散误差，不是"步数不够"：网格加密则降、加步数则不动。

    实测（2000 步）：L=6 → 0.827%，L=8 → 0.585%，L=10 → 0.416%，L=14 → 0.235%；
    L=10 在 1000/2000/3000 步下误差恒为 0.4160%。本用例断言 L=10 比 L=6 至少好 1.5×
    —— 若哪天这条失败，先查网格没查步数，别用放宽 1% 阈值掩盖。
    """
    e6 = duct_poiseuille_case(length=6)["rel_err"]
    e10 = duct_poiseuille_case(length=10)["rel_err"]
    assert e10 < REL_TOL and e6 < REL_TOL, f"L=6 {e6:.3%} / L=10 {e10:.3%} 均须 < {REL_TOL:.0%}"
    assert e6 / e10 > 1.5, (
        f"加密 L=6→10 误差只降 {e6 / e10:.2f}×（{e6:.3%} → {e10:.3%}）："
        "残差可能不是离散误差（或基准/几何映射错位）"
    )

# ---------------------------------------------------------------------------
# Step F：算子不变量——A/B/C/E 在 MRT / TRT / RLBM 下同样满足解析解判据
# ---------------------------------------------------------------------------
OPERATOR_STEPS = ("A_shear_wave", "B_couette", "C_poiseuille", "E_duct_poiseuille")


def _run_operator_step(collision: str, step: str) -> dict:
    """按 (算子, 算例) 跑一次，返回该算例的 metrics dict（与默认 BGK 路径同一段代码）。"""
    if step == "A_shear_wave":
        return shear_wave_case(collision=collision)
    if step == "B_couette":
        return couette_case(collision=collision)
    if step == "C_poiseuille":
        return poiseuille_case(collision=collision)
    if step == "E_duct_poiseuille":
        return duct_poiseuille_case(collision=collision)
    raise ValueError(f"未知算例 {step!r}")


@pytest.mark.parametrize("collision", [c for c in COLLISIONS if c != "bgk"])
@pytest.mark.parametrize("step", OPERATOR_STEPS)
def test_step_f_operator_reproduces_analytic_anchors(collision: str, step: str) -> None:
    """MRT / TRT / 正则化 BGK 必须复现同一组解析解（相对误差 < 1%）。

    MRT 的剪切矩弛豫率是 1/τ（与 BGK 同粘度），TRT 的对称部分 τ₊=τ，
    正则化 BGK 只投影非平衡高阶矩——三者在**流体力学层面**都应与 BGK 等价，
    因此 Step A（ν）、B（Couette 剖面）、C（Poiseuille 剖面）、E（方管级数解）
    的解析解判据一条都不能豁免。任何算子跑不过即说明该算子的尺度/粘度约定
    或它与半路反弹/强迫项的耦合有问题，而不是"高级算子精度本来就不同"。
    """
    r = _run_operator_step(collision, step)
    assert r["collision"] == collision
    assert r["rel_err"] < REL_TOL, (
        f"算子 {collision} 在 {step} 上相对误差 {r['rel_err']:.4%} ≥ {REL_TOL:.0%}"
    )


def test_step_f_trt_magic_parameter_removes_halfway_bounceback_slip() -> None:
    """鉴别力守卫：TRT(Λ=3/16) 消掉 Step C 的壁面位置误差，BGK 必须**没能**消掉。

    若把断言只写成「TRT 误差 < 1%」，它对"TRT 其实退化成 BGK"这种缺陷毫无鉴别力
    （BGK 本身就是 0.34% < 1%）。因此这里做**双向判据**：
    负控 BGK 的残差必须仍 > 1e-3，TRT 必须 < 1e-3 且小于 BGK 的 1%。
    这也把 Step C 残差的归属钉死为半路反弹的壁面位置误差（Ginzburg 2008 的
    Λ=3/16 正是为此设计），而不是强迫项/可压缩性误差。
    """
    bgk = poiseuille_case(collision="bgk")
    trt = poiseuille_case(collision="trt")
    assert bgk["rel_err"] > 1e-3, (
        f"负控失效：BGK 的 Step C 残差 {bgk['rel_err']:.3%} 已 ≤ 0.1%，"
        "壁面位置误差的判据不再有鉴别力，需重新标定本用例"
    )
    assert trt["rel_err"] < 1e-3, (
        f"TRT(Λ=3/16) 未消除半路反弹的壁面位置误差：{trt['rel_err']:.3%} ≥ 0.1%"
    )
    assert trt["rel_err"] < 0.01 * bgk["rel_err"], (
        f"TRT 残差 {trt['rel_err']:.3%} 未降到 BGK {bgk['rel_err']:.3%} 的 1% 以下"
    )


def test_step_f_trt_shear_wave_error_is_second_order_in_k() -> None:
    """TRT 在剪切波上有一个 O(k²) 的有效粘度偏差（BGK 没有）——把它钉住。

    实测（100 步、τ=1、float64）：nx=32 → 0.1601%，64 → 0.0399%，128 → 0.0100%，
    k 减半误差降 4.0×（二阶），而 BGK 在 nx=64 上是 1.6e-7（舍入量级）。
    归因：这是 TRT 反称通道 τ₋=½+Λ/(τ₊−½) 在有限 k 下与剪切模的耦合（截断项），
    不是粘度尺度错——因此它是**分辨率偏差**，做 TRT 的粘度标定时要随 k 一起外推。
    若哪天这条失败（比值离开 [2.5,6]），先查 TRT 的 τ₋ 定义，别改判据阈值。
    """
    e64 = shear_wave_case(collision="trt", nx=64)["rel_err"]
    e128 = shear_wave_case(collision="trt", nx=128)["rel_err"]
    bgk64 = shear_wave_case(collision="bgk", nx=64)["rel_err"]
    assert e64 < REL_TOL and e128 < REL_TOL, f"TRT 剪切波误差 {e64:.3%} / {e128:.3%} 须 < 1%"
    ratio = e64 / e128
    assert 2.5 < ratio < 6.0, (
        f"TRT 剪切波误差 nx=64→128 只降 {ratio:.2f}×（{e64:.3%} → {e128:.3%}），不是二阶 k 收敛"
    )
    assert bgk64 < 1e-5, f"对照失效：BGK 在 nx=64 上误差 {bgk64:.2e} 已不小，无法归因到 TRT 的 τ₋ 通道"


def test_step_f_trt_reduces_duct_corner_error() -> None:
    """3D 角区同样受壁面位置误差支配：TRT 的方管误差必须显著低于 BGK。

    实测（L=10、2000 步）BGK 0.416% → TRT 0.238%（比 0.57）。判据取"比 < 0.75"
    留出余量；若 TRT 在三维角区反而不优于 BGK，说明双壁叠加处的处理有问题。
    """
    bgk = duct_poiseuille_case(collision="bgk")
    trt = duct_poiseuille_case(collision="trt")
    ratio = trt["rel_err"] / bgk["rel_err"]
    assert ratio < 0.75, (
        f"TRT 方管角区误差 {trt['rel_err']:.3%} 相对 BGK {bgk['rel_err']:.3%} 的比 {ratio:.2f} ≥ 0.75"
    )


# ---------------------------------------------------------------------------
# 负控（判据的鉴别力）：NoDynamics 回填必须用碰撞前快照
# ---------------------------------------------------------------------------
@pytest.mark.regression
def test_no_dynamics_restore_must_use_pre_collision_snapshot() -> None:
    """负控：故意用碰撞后分布回填固体格点，场量必须出现可测偏差。

    这条用例证明上面四个解析解判据确实有鉴别力——若回填顺序写错，Couette 剖面误差
    会从 0.0013% 涨到 4.8%（实测，2000 步），Poiseuille 涨到 17%，判据会红。
    本用例直接比较两种回填顺序 200 步后的场量差，阈值 1e-3。
    """
    d = _device()
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    u_top = 0.05
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    zero = torch.zeros(nz, ny, nx, device=d)
    feq_top = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.full_like(torch.ones(nz, ny, nx, device=d), u_top),
        zero,
        zero,
    )

    def advance(f: torch.Tensor, steps: int, correct_order: bool) -> torch.Tensor:
        for _ in range(steps):
            f_pre = f.clone() if correct_order else None
            f = collide_bgk3d(f, tau=tau)
            if not correct_order:
                f_pre = f.clone()  # 错误顺序：碰撞后才取快照（固体格点回填碰撞结果）
            f = _no_dynamics_and_bounce(f, f_pre, solid)
            f[:, :, -1, :] = feq_top[:, :, -1, :]
            f = stream3d(f)
            f[:, :, :, 0] = f[:, :, :, -2]
            f[:, :, :, -1] = f[:, :, :, -2]
        return f

    zero_f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, zero, zero)
    f_ok = advance(zero_f.clone(), 200, True)
    f_bad = advance(zero_f.clone(), 200, False)
    gap = float((f_ok - f_bad).abs().max())
    assert gap > 1e-3, (
        f"两种 NoDynamics 回填顺序在 200 步后场量差仅 {gap:.3e} ≤ 1e-3，"
        "解析解判据可能已失去对该缺陷的鉴别力"
    )


CASES = (
    ("A", shear_wave_case),
    ("B", couette_case),
    ("C", poiseuille_case),
    ("D", friction_case),
    ("E", duct_poiseuille_case),
)

#: 每个 Step 的"关键量 vs 基准"字段名（报告表格的取数口径，写死在这里便于审阅）
KEY_QUANTITY = {
    "A": ("运动粘度 ν", "nu_measured", "nu_expected"),
    "B": ("Couette 剖面 max|Δu|/u_top", None, None),
    "C": ("Poiseuille 峰值 u_max", "u_max_num", "u_max_exact"),
    "D": ("壁面摩擦系数 Cf", "cf_num", "cf_exact"),
    "E": ("方管截面最大速度 u_max", "u_max_num", "u_max_exact"),
}


def collect_metrics(device: torch.device | None = None, *, refine: bool = False) -> dict:
    """跑完全部案例，返回可序列化的指标（数值全部程序化注入报告，不手抄）。

    ``refine=True`` 额外跑 Step E 的网格/步数敏感性（★只给报告 CLI 用：
    pytest 的鉴别力由 ``test_step_e_duct_error_is_discretization_limited`` 独立裁决，
    不在这条路径上重复付算力）。
    """
    d = device if device is not None else _device()
    steps: dict[str, dict] = {}
    for step, case in CASES:
        r = case(d)
        r["step"] = step
        r["err_pct"] = 100.0 * float(r["rel_err"])
        r["tolerance_pct"] = TOL_PCT
        r["passed"] = bool(r["rel_err"] < REL_TOL)
        steps[step] = r
    refinement = None
    if refine:
        refinement = {
            "E_lengths": [
                {"length": L, "err_pct": 100.0 * float(duct_poiseuille_case(d, length=L)["rel_err"])}
                for L in (6, 10, 14)
            ],
            "E_steps": [
                {
                    "steps": n,
                    "err_pct": 100.0
                    * float(duct_poiseuille_case(d, steps=n)["rel_err"]),
                }
                for n in (1000, 2000)
            ],
        }
    out = {
        "device": str(d),
        "dtype": str(torch.get_default_dtype()).replace("torch.", ""),
        "tol_pct": TOL_PCT,
        "n_pass": sum(1 for r in steps.values() if r["passed"]),
        "n_total": len(steps),
        "steps": steps,
    }
    if refinement is not None:
        out["refinement"] = refinement
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="TensorLBM 解析解验证梯 Step A-E（打印表格，可选写 metrics JSON）"
    )
    ap.add_argument("--json", type=Path, default=None, help="指标 JSON 输出路径")
    ap.add_argument(
        "--refine",
        action="store_true",
        help="额外跑 Step E 的网格/步数敏感性（多约 25 s，报告数字用）",
    )
    args = ap.parse_args(argv)

    torch.set_num_threads(1)
    torch.set_default_dtype(resolve_dtype())
    metrics = collect_metrics(torch.device("cpu"), refine=args.refine)

    print("\n" + "=" * 66)
    print(f"  LBM 解析解验证：5 案例，判据 = 相对误差 < {TOL_PCT:g}%，dtype={metrics['dtype']}")
    print("=" * 66)
    for step, r in metrics["steps"].items():
        flag = "PASS" if r["passed"] else "FAIL"
        print(f"  {r['case']:<14} rel_err={r['rel_err']:.4%}  {flag}")
        label, num_key, exact_key = KEY_QUANTITY[step]
        if num_key:
            print(f"      {label}: num={r[num_key]:.8g} exact={r[exact_key]:.8g}")
    print(f"\n  通过 {metrics['n_pass']}/{metrics['n_total']}")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(metrics, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"  metrics -> {args.json}")

    return 0 if metrics["n_pass"] == metrics["n_total"] else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
