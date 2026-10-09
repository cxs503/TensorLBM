#!/usr/bin/env python3
"""LBM 100% 解析解验证脚本

运行所有4个验证案例, 每个都有精确解析解:
  Step A: 剪切波衰减 → 验证粘度ν (误差<1%)
  Step B: Couette流 → 验证反弹+无滑移 (误差<1%)
  Step C: Poiseuille流 → 验证体力+速度剖面 (误差<1%)
  Step D: 摩擦阻力 → 验证壁面剪切应力 (误差<1%)

用法: PYTHONPATH=src python tests/test_lbm_verification.py
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


def test_step_a_shear_wave(device="sdaa:0"):
    """Step A: 剪切波衰减 — 验证粘度ν

    物理: u_y(x,t) = A·cos(kx)·exp(-νk²t)
    精确解: ν = -ln(A_t/A_0) / (k²·t)
    """
    print("\n" + "=" * 60)
    print("  Step A: 剪切波衰减 (粘度验证)")
    print("=" * 60)

    d = torch.device(device)
    nx, ny, nz = 64, 4, 4
    tau = 1.0
    nu_expected = (tau - 0.5) / 3.0
    k = 2 * math.pi / nx
    A0 = 0.01

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

    # 运行100步
    for _ in range(100):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)

    # 测量最终振幅
    _, _, uy_f, _ = macroscopic3d(f)
    A_final = uy_f[0, 0, 0].item()

    # 计算粘度
    nu_measured = -math.log(abs(A_final / A_init)) / (k**2 * 100)
    err = abs(nu_measured - nu_expected) / nu_expected * 100

    print(f"  τ={tau}, ν_expected={nu_expected:.6f}")
    print(f"  A_init={A_init:.6f}, A_final={A_final:.6f}")
    print(f"  ν_measured={nu_measured:.6f}")
    print(f"  误差={err:.2f}%")

    passed = err < 1.0
    print(f"  结果: {'PASS ✓' if passed else 'FAIL ✗'} (阈值<1%)")
    return passed


def test_step_b_couette(device="sdaa:0"):
    """Step B: Couette流 — 验证反弹+无滑移

    物理: u(y) = U·(y-0.5)/(ny-1-0.5)
    半路反弹: 壁面在y=0.5
    """
    print("\n" + "=" * 60)
    print("  Step B: Couette流 (反弹+无滑移验证)")
    print("=" * 60)

    d = torch.device(device)
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

    for step in range(2000):
        f_pre = f.clone()
        # 1. 碰撞
        f = collide_bgk3d(f, tau=tau)
        # 2. NoDynamics: 恢复固体格点
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        # 3. 半路反弹 (流之前)
        f = bounce_back_cells_3d(f, solid)
        # 4. 顶壁运动平衡态
        rho1 = torch.ones(nz, ny, nx, device=d)
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
    rho, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = u_top * (y - 0.5) / (ny - 1 - 0.5)
    max_err = np.max(np.abs(u - u_exact) / u_top) * 100

    print(f"  u_top={u_top}, ny={ny}, τ={tau}")
    print(f"  u_num  ={np.round(u, 6)}")
    print(f"  u_exact={np.round(u_exact, 6)}")
    print(f"  max_err={max_err:.2f}%")

    passed = max_err < 1.0
    print(f"  结果: {'PASS ✓' if passed else 'FAIL ✗'} (阈值<1%)")
    return passed


def test_step_c_poiseuille(device="sdaa:0"):
    """Step C: Poiseuille流 — 验证体力+速度剖面

    物理: u(y) = G/(2ν)·(y-0.5)·(H+0.5-y)
    """
    print("\n" + "=" * 60)
    print("  Step C: Poiseuille流 (体力+速度剖面验证)")
    print("=" * 60)

    d = torch.device(device)
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

    for step in range(3000):
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

    rho, ux, _, _ = macroscopic3d(f)
    u = ux[0, 1:-1, nx // 2].cpu().numpy()
    y = np.arange(1, ny - 1)
    u_exact = G / (2 * nu) * (y - 0.5) * (H + 0.5 - y)
    max_err = np.max(np.abs(u - u_exact) / max(u_exact.max(), 1e-10)) * 100

    print(f"  ν={nu:.4f}, G={G:.6e}, H={H}")
    print(f"  u_max_num={u.max():.6f}, u_max_exact={u_exact.max():.6f}")
    print(f"  max_err={max_err:.2f}%")

    passed = max_err < 1.0
    print(f"  结果: {'PASS ✓' if passed else 'FAIL ✗'} (阈值<1%)")
    return passed


def test_step_d_friction(device="sdaa:0"):
    """Step D: 摩擦阻力 — 验证壁面剪切应力

    物理: τ_w = ν·du/dy, Cf = 2ν/((H-0.5)·U)
    """
    print("\n" + "=" * 60)
    print("  Step D: 摩擦阻力 (壁面剪切应力验证)")
    print("=" * 60)

    d = torch.device(device)
    nx, ny, nz = 80, 12, 4
    tau = 1.0
    nu = (tau - 0.5) / 3.0
    u_top = 0.05
    # 壁面间距: 底壁y=0.5到顶壁y=ny-1=11 → 距离=10.5
    cf_exact = 2 * nu / (10.5 * u_top)

    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=d)
    solid[:, 0, :] = True
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    f = equilibrium3d(
        torch.ones(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
        torch.zeros(nz, ny, nx, device=d),
    )

    for step in range(3000):
        f_pre = f.clone()
        f = collide_bgk3d(f, tau=tau)
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])
        f = bounce_back_cells_3d(f, solid)
        rho1 = torch.ones(nz, ny, nx, device=d)
        feq_top = equilibrium3d(
            rho1, torch.full_like(rho1, u_top), torch.zeros_like(rho1), torch.zeros_like(rho1)
        )
        f[:, :, -1, :] = feq_top[:, :, -1, :]
        f = stream3d(f)
        f[:, :, :, 0] = f[:, :, :, -2]
        f[:, :, :, -1] = f[:, :, :, -2]

    rho, ux, _, _ = macroscopic3d(f)
    u_wall = ux[0, 1, nx // 2].item()
    du_dy = u_wall / 0.5  # 半路反弹: 壁面在0.5
    tau_w = nu * du_dy
    cf_num = 2 * tau_w / u_top**2
    err = abs(cf_num - cf_exact) / cf_exact * 100

    print(f"  ν={nu:.4f}, U={u_top}, 壁面间距=10.5")
    print(f"  u_wall={u_wall:.6f}, du/dy={du_dy:.6f}")
    print(f"  τ_w={tau_w:.6f}")
    print(f"  Cf_num={cf_num:.4f}, Cf_exact={cf_exact:.4f}")
    print(f"  误差={err:.2f}%")

    passed = err < 1.0
    print(f"  结果: {'PASS ✓' if passed else 'FAIL ✗'} (阈值<1%)")
    return passed


def main():
    print("\n" + "╔" + "═" * 58 + "╗")
    print("║" + " LBM 100% 解析解验证体系 ".center(58) + "║")
    print("║" + " 每个案例都有精确解, 误差<1%才算通过 ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")

    device = "sdaa:0" if torch.sdaa.is_available() else "cpu"

    results = []
    results.append(("Step A: 剪切波衰减", test_step_a_shear_wave(device)))
    results.append(("Step B: Couette流", test_step_b_couette(device)))
    results.append(("Step C: Poiseuille流", test_step_c_poiseuille(device)))
    results.append(("Step D: 摩擦阻力", test_step_d_friction(device)))

    print("\n" + "=" * 60)
    print("  验证汇总")
    print("=" * 60)
    n_pass = 0
    for name, passed in results:
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name}: {status}")
        if passed:
            n_pass += 1

    print(f"\n  通过: {n_pass}/{len(results)}")
    if n_pass == len(results):
        print("  ★ LBM求解器基础物理100%正确!")
    else:
        print("  ✗ 有验证未通过,请检查代码")

    return n_pass == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
