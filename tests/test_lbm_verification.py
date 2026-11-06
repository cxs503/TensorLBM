#!/usr/bin/env python3
"""LBM 100% 解析解验证脚本 (analytic benchmarks for the D3Q19 core)

运行所有4个验证案例, 每个都有精确解析解:
  Step A: 剪切波衰减 → 验证粘度ν (误差<1%)
  Step B: Couette流 → 验证反弹+无滑移 (误差<1%)
  Step C: Poiseuille流 → 验证体力+速度剖面 (误差<1%)
  Step D: 摩擦阻力 → 验证壁面剪切应力 (误差<1%)

用法: PYTHONPATH=src python tests/test_lbm_verification.py

历史 (2026-11-03, m3643): 本文件此前被列入 ``tests/ci_quarantine.txt``,
四个案例在 CI 上从未真正跑过。两个独立原因:

1. 四个函数签名写死 ``device="sdaa:0"`` —— CPU-only host 上 torch.device()
   直接抛错;
2. pytest 把 ``device`` 形参当作 fixture 请求 —— 收集阶段即报
   ``fixture 'device' not found``, 与数值是否正确无关。

修复方式: 设备默认值改为 ``tensorlbm.utils.default_device_name()`` (探测
SDAA/CUDA/插件加速器, 回落 CPU), 数值内核搬进 ``run_step_*`` 纯函数并返回
度量字典, pytest 侧只留零形参薄壳。阈值 (1%) 与网格/步数一律未放宽。

度量可直接落盘: ``python scripts/lbm_analytic_verification_report.py``
会跑四个案例并写 JSON + 图 (docs/lbm_analytic_verification_r1.md 的数据源)。
"""

import math
import sys

import numpy as np
import torch

# 添加src到路径
sys.path.insert(0, "src")

from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import C, W, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d
from tensorlbm.utils import default_device_name

THRESHOLD_PCT = 1.0


def _resolve(device):
    """None → this host's preferred device; explicit value passed through."""
    return default_device_name() if device is None else device


def run_step_a_shear_wave(device=None):
    """Step A: 剪切波衰减 — 验证粘度ν

    物理: u_y(x,t) = A·cos(kx)·exp(-νk²t)
    精确解: ν = -ln(A_t/A_0) / (k²·t)
    """
    d = torch.device(_resolve(device))
    nx, ny, nz = 64, 4, 4
    tau = 1.0
    nu_expected = (tau - 0.5) / 3.0
    k = 2 * math.pi / nx
    A0 = 0.01
    n_steps = 100

    # 初始化: u_y(x) = A·cos(kx)
    uy = torch.zeros(nz, ny, nx, device=d)
    for i in range(nx):
        uy[:, :, i] = A0 * math.cos(k * i)
    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        uy,
        torch.zeros(nz, ny, nx, device=d),
    )

    # 测量初始振幅
    _, _, uy0, _ = macroscopic3d(f)
    A_init = uy0[0, 0, 0].item()

    # 运行并采样振幅 (供衰减曲线图)
    samples = [(0, A_init)]
    for step in range(1, n_steps + 1):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)
        if step % 10 == 0:
            samples.append((step, macroscopic3d(f)[2][0, 0, 0].item()))

    # 测量最终振幅
    A_final = samples[-1][1]

    # 计算粘度
    nu_measured = -math.log(abs(A_final / A_init)) / (k**2 * n_steps)
    err = abs(nu_measured - nu_expected) / nu_expected * 100

    return {
        "case": "A:shear_wave",
        "device": str(d),
        "tau": tau,
        "nu_expected": nu_expected,
        "nu_measured": nu_measured,
        "A_init": A_init,
        "A_final": A_final,
        "steps": n_steps,
        "decay_samples": samples,
        "err_pct": err,
        "threshold_pct": THRESHOLD_PCT,
        "passed": bool(err < THRESHOLD_PCT),
    }


def run_step_b_couette(device=None):
    """Step B: Couette流 — 验证反弹+无滑移

    物理: u(y) = U·(y-0.5)/(ny-1-0.5)
    半路反弹: 壁面在y=0.5
    """
    d = torch.device(_resolve(device))
    nx, ny, nz = 80, 12, 4
    u_top = 0.05
    tau = 1.0

    # 底壁固体(反弹), 顶壁用平衡态(运动)
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
    )

    rho1 = torch.ones(nz, ny, nx, device=d)
    for _step in range(2000):
        f_pre = f.clone()
        # 1. 碰撞
        f = collide_bgk3d(f, tau=tau)
        # 2. NoDynamics: 恢复固体格点
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # 3. 半路反弹 (流之前)
        f = bounce_back_cells_3d(f, solid)
        # 4. 顶壁运动平衡态
        feq_top = equilibrium3d(
            rho1, torch.full_like(rho1, u_top), torch.zeros_like(rho1), torch.zeros_like(rho1)
        )
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        # 5. 迁移
        f = stream3d(f)
        # 6. 进出口周期性
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    # 验证速度剖面
    _rho, ux, _uy, _uz = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = u_top * (y - 0.5) / (ny - 1 - 0.5)
    max_err = np.max(np.abs(u - u_exact) / u_top) * 100

    return {
        "case": "B:couette",
        "device": str(d),
        "u_top": u_top,
        "ny": ny,
        "tau": tau,
        "y": y.tolist(),
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
        "err_pct": float(max_err),
        "threshold_pct": THRESHOLD_PCT,
        "passed": bool(max_err < THRESHOLD_PCT),
    }


def run_step_c_poiseuille(device=None):
    """Step C: Poiseuille流 — 验证体力+速度剖面

    物理: u(y) = G/(2ν)·(y-0.5)·(H+0.5-y)
    """
    d = torch.device(_resolve(device))
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    H = ny - 2
    u_max = 0.05
    G = 2 * nu * u_max / H**2

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)
    c = C.to(d).float()
    w = W.to(d).float()

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
    )

    for _step in range(3000):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        # Guo体力
        for q in range(19):
            f[q] = f[q] + w[q] * 3 * c[q, 0] * G
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _rho, ux, _uy, _uz = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = G / (2 * nu) * (y - 0.5) * (H + 0.5 - y)
    max_err = np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-10)) * 100

    return {
        "case": "C:poiseuille",
        "device": str(d),
        "nu": nu,
        "G": G,
        "H": H,
        "y": y.tolist(),
        "u_num": u.tolist(),
        "u_exact": u_exact.tolist(),
        "err_pct": float(max_err),
        "threshold_pct": THRESHOLD_PCT,
        "passed": bool(max_err < THRESHOLD_PCT),
    }


def run_step_d_friction(device=None):
    """Step D: 摩擦阻力 — 验证壁面剪切应力

    物理: τ_w = ν·du/dy, Cf = 2ν/((H-0.5)·U)
    """
    d = torch.device(_resolve(device))
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_top = 0.05
    gap = ny - 1 - 0.5  # 底壁 y=0.5 → 顶壁 y=ny-1
    # 壁面间距: 底壁y=0.5到顶壁y=ny-1=11 → 距离=10.5
    cf_exact = 2 * nu / (gap * u_top)

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
    )

    rho1 = torch.ones(nz, ny, nx, device=d)
    for _step in range(3000):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        feq_top = equilibrium3d(
            rho1, torch.full_like(rho1, u_top), torch.zeros_like(rho1), torch.zeros_like(rho1)
        )
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    _rho, ux, _uy, _uz = macroscopic3d(f)
    u_wall = ux[0, 1, nx // 2].item()
    du_dy = u_wall / 0.5  # 半路反弹: 壁面在0.5
    tau_w = nu * du_dy
    cf_num = 2 * tau_w / u_top**2
    err = abs(cf_num - cf_exact) / cf_exact * 100

    return {
        "case": "D:friction",
        "device": str(d),
        "nu": nu,
        "u_top": u_top,
        "gap": gap,
        "u_wall": u_wall,
        "du_dy": du_dy,
        "tau_w": tau_w,
        "cf_num": cf_num,
        "cf_exact": cf_exact,
        "err_pct": float(err),
        "threshold_pct": THRESHOLD_PCT,
        "passed": bool(err < THRESHOLD_PCT),
    }


# --- pytest 薄壳 (零形参: 避免 pytest 把 device 当 fixture) ---


def test_step_a_shear_wave():
    m = run_step_a_shear_wave()
    assert m["passed"], f"shear-wave viscosity error {m['err_pct']:.3f}% >= 1%"


def test_step_b_couette():
    m = run_step_b_couette()
    assert m["passed"], f"Couette profile error {m['err_pct']:.3f}% >= 1%"


def test_step_c_poiseuille():
    m = run_step_c_poiseuille()
    assert m["passed"], f"Poiseuille profile error {m['err_pct']:.3f}% >= 1%"


def test_step_d_friction():
    m = run_step_d_friction()
    assert m["passed"], f"wall-shear Cf error {m['err_pct']:.3f}% >= 1%"


CASES = (
    ("Step A: 剪切波衰减", run_step_a_shear_wave),
    ("Step B: Couette流", run_step_b_couette),
    ("Step C: Poiseuille流", run_step_c_poiseuille),
    ("Step D: 摩擦阻力", run_step_d_friction),
)


def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " LBM 100% 解析解验证体系 ".center(58) + "║")
    print("║" + " 每个案例都有精确解, 误差<1%才算通过 ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    device = default_device_name()
    print(f"  device = {device}")

    results = []
    for name, runner in CASES:
        m = runner(device)
        print(f"  {name}: err={m['err_pct']:.3f}%  {'PASS ✓' if m['passed'] else 'FAIL ✗'}")
        results.append((name, m))

    print("\n" + "=" * 60)
    print("  验证汇总")
    print("=" * 60)
    n_pass = sum(1 for _n, m in results if m["passed"])
    for name, m in results:
        status = "PASS ✓" if m["passed"] else "FAIL ✗"
        print(f"  {name}: {status} ({m['err_pct']:.3f}%)")

    print(f"\n  通过: {n_pass}/{len(results)}")
    if n_pass == len(results):
        print("  ★ LBM求解器基础物理100%正确!")
    else:
        print("  ✗ 有验证未通过,请检查代码")

    return n_pass == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
