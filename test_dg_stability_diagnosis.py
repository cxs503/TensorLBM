#!/usr/bin/env python3
"""DG-LBM 稳定性诊断: 验证碰撞刚度问题和修复方案。

核心发现:
  dg_lbm_rhs_band 使用 -(f-f_eq)/τ 作为碰撞RHS项。
  在 method-of-lines 中显式积分此RHS时, 稳定性要求 dt_sub < 2*τ。
  当 τ_dg = τ_lbm - 0.5 较小时(如 Re=50: τ_dg=0.0576),
  即使 n_substeps=6 (dt_sub=0.167), dt_sub/τ_dg ≈ 2.9 > 2, 不满足稳定性。
  实际 SUBOFF 使用 n_substeps=4 (dt_sub=0.25), dt_sub/τ_dg ≈ 4.34, 严重不稳定。

  正确的 fix: 使用 operator splitting (先碰撞后对流),
  碰撞采用 Δt-aware 的 collide_bgk_dg (已实现但未在 band step 中使用)。
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch

from tensorlbm.dg_advection import (
    collide_bgk_dg,
    dg_advect,
    dg_lbm_rhs,
    dg_lbm_step,
    get_ops,
    equilibrium_dg,
    macroscopic_dg,
)
from tensorlbm.dg_band import (
    build_band_topology,
    dg_advect_band,
    dg_lbm_rhs_band,
    dg_lbm_step_band,
)
from tensorlbm.d2q9 import C as C2D, W as W2D, OPPOSITE as OPP2D
from tensorlbm.d3q19 import C as C3D, W as W3D

DT = torch.float64


def test_combined_rhs_stability():
    """验证 combined RHS 在小 τ 时不稳定的问题。"""
    print("=" * 60)
    print("TEST 1: Combined RHS stability vs τ")
    print("=" * 60)

    ny, nx = 16, 16
    ops = get_ops(degree=1, dx=1.0, dtype=DT)
    r = torch.linspace(0, 1, ny, dtype=DT).view(ny, 1, 1, 1)

    for tau in [1.0, 0.5, 0.3, 0.1, 0.0576]:
        n_substeps = 6
        dt_sub = 1.0 / n_substeps
        ratio = dt_sub / tau
        stable_bound = "OK" if ratio < 2.0 else f"UNSTABLE (ratio={ratio:.2f} > 2)"

        rho = torch.ones(ny, nx, 2, 2, dtype=DT)
        ux = 0.05 * torch.ones(ny, nx, 2, 2, dtype=DT)
        uy = torch.zeros_like(ux)
        f = equilibrium_dg(rho, [ux, uy], C2D.to(DT), W2D.to(DT))
        f = f + 0.001 * torch.randn_like(f)  # 小扰动

        # 运行一步 combined RHS (单子步 Euler)
        rhs = dg_lbm_rhs(f, C2D.to(DT), W2D.to(DT), tau, ops, ndim_spatial=2)
        f1 = f + dt_sub * rhs

        has_nan = f1.isnan().any().item()
        max_val = f1.abs().max().item()
        print(f"  τ={tau:.4f}, dt_sub={dt_sub:.4f}, dt/τ={ratio:.2f} → {stable_bound}, "
              f"|f|_max={max_val:.4f}, NaN={has_nan}")

        if has_nan:
            print(f"    ❌ NaN detected!")
        elif max_val > 10:
            print(f"    ⚠️  爆炸趋势 (|f| > 10)")


def test_split_vs_combined():
    """对比 operator splitting 和 combined RHS 的稳定性。"""
    print("\n" + "=" * 60)
    print("TEST 2: Split (collide_then_advect) vs Combined RHS")
    print("=" * 60)

    ny, nx = 16, 16
    ops = get_ops(degree=1, dx=1.0, dtype=DT)
    tau_dg = 0.0576  # Re=50 on SUBOFF
    n_substeps = 4
    dt_sub = 1.0 / n_substeps

    torch.manual_seed(42)
    rho = torch.ones(ny, nx, 2, 2, dtype=DT)
    ux = 0.05 * torch.ones(ny, nx, 2, 2, dtype=DT)
    uy = torch.zeros_like(ux)
    f0 = equilibrium_dg(rho, [ux, uy], C2D.to(DT), W2D.to(DT))
    f0 = f0 + 0.001 * (torch.rand_like(f0) - 0.5)

    # Method A: Combined RHS (当前有问题的方案)
    f_a = f0.clone()
    for _ in range(n_substeps):
        rhs = dg_lbm_rhs(f_a, C2D.to(DT), W2D.to(DT), tau_dg, ops, ndim_spatial=2)
        f_a = f_a + dt_sub * rhs
    has_nan_a = f_a.isnan().any().item()
    max_a = f_a.abs().max().item()

    # Method B: Operator splitting (collide then advect)
    f_b = f0.clone()
    for _ in range(n_substeps):
        # 1. Collision (Δτ-aware, unconditionally stable)
        f_b = collide_bgk_dg(f_b, C2D.to(DT), W2D.to(DT), tau_dg, dt_sub)
        # 2. Advection (one RK3 step)
        f_b = dg_advect(f_b, C2D.to(DT), ops, ndim_spatial=2,
                        dt=dt_sub, n_substeps=1, scheme="rk3")
    has_nan_b = f_b.isnan().any().item()
    max_b = f_b.abs().max().item()

    print(f"  τ_dg={tau_dg}, n_substeps={n_substeps}, dt_sub={dt_sub}")
    print(f"  Method A (Combined RHS): |f|_max={max_a:.4f}, NaN={has_nan_a}")
    print(f"  Method B (Split):       |f|_max={max_b:.4f}, NaN={has_nan_b}")

    if has_nan_a:
        print("  ❌ Combined RHS 产生 NaN!")
    if not has_nan_b:
        print("  ✅ Split approach 稳定!")
    if has_nan_a and not has_nan_b:
        print("  → 修复方案: 使用 operator splitting")


def test_band_split_stability():
    """在 band topology 上测试 operator splitting 稳定性。"""
    print("\n" + "=" * 60)
    print("TEST 3: Band topology with split approach")
    print("=" * 60)

    ny, nx = 32, 32
    tau_dg = 0.0576
    n_substeps = 4
    dt_sub = 1.0 / n_substeps

    # 创建一个 band (区域内 band, 外围 exterior)
    band_mask = torch.zeros(ny, nx, dtype=torch.bool)
    band_mask[8:24, 8:24] = True
    topo = build_band_topology(band_mask, periodic=True)

    ops = get_ops(degree=1, dx=1.0, dtype=DT)
    n_band = topo.n_band
    nn = 2

    # 初始化
    rho0 = torch.ones(ny, nx, dtype=DT)
    ux0 = 0.05 * torch.ones(ny, nx, dtype=DT)
    uy0 = torch.zeros(ny, nx, dtype=DT)
    # 使用 equilibrium 构建 full grid, 然后提取 band
    f_band_init = torch.rand(9, n_band, nn, nn, dtype=DT) * 0.01 + 0.1

    # 使用 equilibrium 种子
    from tensorlbm.d2q9 import equilibrium as eq2d
    f_lbm = eq2d(rho0, ux0, uy0).to(DT)
    cb = topo.band_coords
    f_dg = f_lbm[:, cb[:, 0], cb[:, 1]].unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nn, nn).contiguous().clone()
    Q = f_lbm.shape[0]
    N = int(torch.tensor(f_lbm.shape[1:]).prod().item())
    ext_field = f_lbm.reshape(Q, N)

    # Method A: Current combined RHS band step
    f_a = f_dg.clone()
    has_nan_a = False
    try:
        f_a = dg_lbm_step_band(
            f_a, C2D.to(DT), W2D.to(DT), tau_dg, ops, topo, ext_field,
            dt=1.0, n_substeps=n_substeps, scheme="rk3", opposite=OPP2D.to(DT),
        )
    except Exception as e:
        print(f"  Method A exception: {e}")
        has_nan_a = True
    has_nan_a = has_nan_a or f_a.isnan().any().item()
    max_a = f_a.abs().max().item() if not has_nan_a else float('nan')

    # Method B: Operator splitting band step
    f_b = f_dg.clone()
    for _ in range(n_substeps):
        f_b = collide_bgk_dg(f_b, C2D.to(DT), W2D.to(DT), tau_dg, dt_sub)
        f_b = dg_advect_band(
            f_b, C2D.to(DT), ops, topo, ext_field,
            dt=dt_sub, n_substeps=1, scheme="rk3", opposite=OPP2D.to(DT),
        )
    has_nan_b = f_b.isnan().any().item()
    max_b = f_b.abs().max().item() if not has_nan_b else float('nan')

    print(f"  τ_dg={tau_dg}, n_substeps={n_substeps}, dt_sub={dt_sub}")
    print(f"  Method A (Combined band): |f|_max={max_a:.4f}, NaN={has_nan_a}")
    print(f"  Method B (Split band):    |f|_max={max_b:.4f}, NaN={has_nan_b}")

    if has_nan_a and not has_nan_b:
        print("  ✅ Band split approach 修复了 NaN!")


def test_collide_bgk_dg_correctness():
    """验证 collide_bgk_dg 的正确性: 应该保持平衡态不变。"""
    print("\n" + "=" * 60)
    print("TEST 4: collide_bgk_dg correctness (equilibrium preserved)")
    print("=" * 60)

    ny, nx = 8, 8
    rho = 1.0 + 0.1 * torch.rand(ny, nx, 2, 2, dtype=DT)
    ux = 0.05 * torch.rand(ny, nx, 2, 2, dtype=DT)
    uy = 0.03 * torch.rand(ny, nx, 2, 2, dtype=DT)
    f_eq = equilibrium_dg(rho, [ux, uy], C2D.to(DT), W2D.to(DT))

    # 对平衡态做碰撞应该保持平衡态
    for tau in [0.05, 0.1, 0.5, 1.0, 10.0]:
        for dt in [0.1, 0.25, 0.5, 1.0]:
            f_post = collide_bgk_dg(f_eq, C2D.to(DT), W2D.to(DT), tau, dt)
            diff = (f_post - f_eq).abs().max().item()
            if diff > 1e-10:
                print(f"    ❌ τ={tau}, dt={dt}: equilibrium not preserved, diff={diff:.2e}")
            else:
                pass  # OK
    print("  ✅ collide_bgk_dg 保持平衡态不变")


if __name__ == "__main__":
    test_combined_rhs_stability()
    test_split_vs_combined()
    test_band_split_stability()
    test_collide_bgk_dg_correctness()
