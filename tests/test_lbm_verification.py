#!/usr/bin/env python3
"""D3Q19 BGK 核心的四步解析解验证（pytest 形态，设备无关）。

四个案例都有可写出的精确解，通过判据一律为相对误差 < 1%：

  Step A 剪切波衰减   u_y(x,t)=A·cos(kx)·exp(-νk²t)          → 校验 ν=(τ-½)/3
  Step B Couette 流   u_x(y)=U·(y-½)/(H-½)（半路反弹无滑移）  → 校验壁面动量交换
  Step C Poiseuille 流 u(y)=G/(2ν)·(y-½)(H+½-y)               → 校验体力注入与剖面
  Step D 壁面摩擦     Cf=2ν/(L·U)（Couette 型摩擦系数）        → 校验壁面剪应力提取

历史背景（2026-11-03 复核）：本模块原先以「脚本 + 硬件钉扎默认参数」的形态存在，
`def test_step_x(device="sdaa:0")` 使 CPU-only runner 在构造 torch.device 时直接报错，
加上函数 `return bool`（pytest 视为非法返回值），四个案例长期躺在
`tests/ci_quarantine.txt` 里从未真正裁决过。本版本把它改成可被 CI 真跑的用例：
设备默认 CPU（`TENSORLBM_TEST_DEVICE` 可覆盖），断言而非返回布尔，
并把逐分量 Python 循环改成整张量布尔掩码/广播（数学逐点等价，不改变任何数值定义）。

命令行形态保留：`python tests/test_lbm_verification.py` 打印全部测量值并汇总。
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import torch

try:  # 直接以脚本运行时也能找到 src/
    from tensorlbm.boundaries3d import bounce_back_cells_3d
    from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import collide_bgk3d, stream3d
except ImportError:  # pragma: no cover - 脚本运行路径
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
    from tensorlbm.boundaries3d import bounce_back_cells_3d
    from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import collide_bgk3d, stream3d

pytestmark = [pytest.mark.physics]

# 全部案例的统一判据：相对误差上限。
REL_TOL = 0.01  # 1%


def _device() -> torch.device:
    """验证用设备：默认 CPU，可用 TENSORLBM_TEST_DEVICE=cuda:0 / sdaa:0 覆盖。

    解析解验证与设备无关，默认 CPU 是为了让 CPU-only runner 也能裁决，
    不再因为 `sdaa:0` 这种硬件钉扎而跳过。
    """
    return torch.device(os.environ.get("TENSORLBM_TEST_DEVICE", "cpu"))


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
def shear_wave_case(device: torch.device | None = None) -> dict:
    """测量剪切波衰减给出的有效粘度，并与 ν=(τ-½)/3 比对。"""
    d = device if device is not None else _device()
    nx, ny, nz = 64, 4, 4
    tau = 1.0
    nu_expected = (tau - 0.5) / 3.0
    k = 2 * math.pi / nx
    a0 = 0.01
    steps = 100

    uy = torch.zeros(nz, ny, nx, device=d)
    xs = torch.arange(nx, device=d, dtype=torch.float32)
    uy[:, :] = (a0 * torch.cos(k * xs)).expand(nz, ny, nx)
    zero = torch.zeros(nz, ny, nx, device=d)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, uy, zero)

    _, _, uy0, _ = macroscopic3d(f)
    a_init = float(uy0[0, 0, 0])

    for _ in range(steps):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)

    _, _, uy_f, _ = macroscopic3d(f)
    a_final = float(uy_f[0, 0, 0])
    nu_measured = -math.log(abs(a_final / a_init)) / (k**2 * steps)
    err = abs(nu_measured - nu_expected) / nu_expected

    return {
        "case": "A_shear_wave",
        "tau": tau,
        "steps": steps,
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
def couette_case(device: torch.device | None = None) -> dict:
    d = device if device is not None else _device()
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    u_top = 0.05
    # 收敛性实测（2026-11-03，CPU 单线程，ny=12）：500 步 0.036%、800 步 0.0013%、
    # 1000/1200 步同为 0.0013% ⇒ 800 步已到稳态离散残差，取 1000 步留安全裕度。
    steps = 1000

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
        "tau": tau,
        "steps": steps,
        "u_top": u_top,
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
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
def poiseuille_case(device: torch.device | None = None) -> dict:
    d = device if device is not None else _device()
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    height = ny - 2
    u_max = 0.05
    g_force = 2 * nu * u_max / height**2
    # 收敛性实测：1000/1500/2000/3000 步的相对误差同为 0.3378% ⇒ 该残差是
    # 半路反弹 + Guo 离散在 ny=12 下的稳态离散误差（不随步数下降），取 1000 步。
    steps = 1000

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    c = C.to(d).float()
    w = W.to(d).float()
    # Guo 体力的动量项：w_q·3·c_qx·G（逐格点常数，可广播到整张量）
    accel = (w * 3.0 * c[:, 0]).view(19, 1, 1, 1) * g_force

    zero = torch.zeros(nz, ny, nx, device=d)
    f = equilibrium3d(torch.ones(nz, ny, nx, device=d), zero, zero, zero)

    for _ in range(steps):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
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
        "tau": tau,
        "steps": steps,
        "nu": nu,
        "body_force": g_force,
        "height": height,
        "u_max_num": float(u.max()),
        "u_max_exact": float(u_exact.max()),
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
        "rel_err": rel_err,
    }


def test_step_c_poiseuille() -> None:
    """Poiseuille 抛物线剖面（Guo 体力），相对误差 < 1%。"""
    r = poiseuille_case()
    assert r["rel_err"] < REL_TOL, (
        f"Step C 剖面相对误差 {r['rel_err']:.3%} ≥ {REL_TOL:.0%}："
        f"u_max_num={r['u_max_num']:.6f} u_max_exact={r['u_max_exact']:.6f}"
    )


# ---------------------------------------------------------------------------
# Step D：壁面摩擦 → 剪应力提取
# ---------------------------------------------------------------------------
def friction_case(device: torch.device | None = None) -> dict:
    d = device if device is not None else _device()
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_top = 0.05
    gap = 10.5  # 底壁 y=½ 到顶壁 y=ny-1+½
    cf_exact = _friction_coefficient(nu, gap, u_top)
    # 收敛性实测：500 步 0.11%、800 步 0.0041%、1000 步 0.0040% ⇒ 取 1000 步。
    steps = 1000

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


CASES = (shear_wave_case, couette_case, poiseuille_case, friction_case)


def main() -> int:
    torch.set_num_threads(1)
    print("\n" + "=" * 66)
    print("  LBM 解析解验证：4 案例，判据 = 相对误差 < 1%")
    print("=" * 66)
    results = []
    for case in CASES:
        r = case(torch.device("cpu"))
        ok = r["rel_err"] < REL_TOL
        results.append((r["case"], ok, r["rel_err"]))
        print(f"  {r['case']:<14} rel_err={r['rel_err']:.4%}  {'PASS' if ok else 'FAIL'}")
        for key in (
            "nu_expected",
            "nu_measured",
            "u_max_num",
            "u_max_exact",
            "cf_num",
            "cf_exact",
            "u_wall",
        ):
            if key in r:
                print(f"      {key} = {r[key]:.8g}")
    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n  通过 {n_pass}/{len(results)}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
