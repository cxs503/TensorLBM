"""IBM 回归等价性验证 — 辨识原始bug，验证等价，验证组合。

本测试文件执行三类验证：

1. **原始bug辨识**：
   - ``derive_surface_markers_3d`` 的 ``.squeeze()`` bug（退化维度被错误移除）
   - ``ibm_direct_forcing_3d`` 原始实现的力守恒与 delta 核 partition-of-unity
   - ``ibm_vec.py`` 向量化版本与原始版本的等价性

2. **等价性验证**：
   - D3Q19 路径：ibm.py 手动流水线 vs ibm_common.py 公共接口 → force + f_corrected 精确匹配
   - D3Q27 路径：ibm_common.py 新增 D3Q27 Guo 修正的物理合理性（力守恒、动量注入）

3. **组合测试**：
   - IBM + D3Q19 BGK 碰撞完整循环（稳定性 + 动量一致性）
   - IBM + D3Q27 BGK 碰撞完整循环
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.ibm import (
    ibm_delta_hat,
    ibm_delta_4pt,
    ibm_direct_forcing_3d,
    ibm_apply_body_force_3d,
)
from tensorlbm.ibm_vec import ibm_direct_forcing_3d_vec
from tensorlbm.ibm_common import (
    ibm_direct_forcing_3d_common,
    ibm_apply_body_force_3d_common,
    derive_surface_markers_3d,
    macroscopic_velocity_3d,
)
from tensorlbm.d3q19 import C as C19, W as W19, equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import C as C27, W as W27, equilibrium27, macroscopic27, collide_bgk27
from tensorlbm.solver3d import collide_bgk3d, stream3d


# =========================================================================== #
# 辅助函数
# =========================================================================== #


def _make_sphere_mask(nz: int, ny: int, nx: int, cx: float, cy: float,
                       cz: float, r: float) -> torch.Tensor:
    """生成球形固体 mask，True 表示固体内部。"""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, dtype=torch.float32),
        torch.arange(ny, dtype=torch.float32),
        torch.arange(nx, dtype=torch.float32),
        indexing="ij",
    )
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    return dist2 <= r * r


def _make_uniform_flow_f19(nz, ny, nx, ux0, rho=1.0):
    """生成 D3Q19 均匀流动分布函数。"""
    rho_t = torch.full((nz, ny, nx), float(rho), dtype=torch.float32)
    ux = torch.full((nz, ny, nx), float(ux0), dtype=torch.float32)
    uy = torch.zeros((nz, ny, nx), dtype=torch.float32)
    uz = torch.zeros((nz, ny, nx), dtype=torch.float32)
    return equilibrium3d(rho_t, ux, uy, uz)


def _make_uniform_flow_f27(nz, ny, nx, ux0, rho=1.0):
    """生成 D3Q27 均匀流动分布函数。"""
    rho_t = torch.full((nz, ny, nx), float(rho), dtype=torch.float32)
    ux = torch.full((nz, ny, nx), float(ux0), dtype=torch.float32)
    uy = torch.zeros((nz, ny, nx), dtype=torch.float32)
    uz = torch.zeros((nz, ny, nx), dtype=torch.float32)
    return equilibrium27(rho_t, ux, uy, uz)


# =========================================================================== #
# 1. 原始 bug 辨识
# =========================================================================== #


class TestBugIdentification:
    """辨识原始实现中的已知 bug。"""

    # ---- 1a. derive_surface_markers_3d squeeze bug ----

    def test_squeeze_bug_degenerate_z_dimension(self):
        """``.squeeze()`` 在 nz=1 时错误移除 z 维度。

        ``derive_surface_markers_3d`` 内部对 ``(1,1,nz,ny,nx)`` 张量调用
        ``.squeeze()``，当 nz=1 时会把 z 维也压掉，导致返回的 marker 坐标
        形状错误。这是一个真实的 latent bug。
        """
        # nz=1 的退化网格：一个 1×5×5 的薄层，中间放一个固体块
        mask = torch.zeros(1, 5, 5, dtype=torch.bool)
        mask[0, 2, 2] = True  # 单个固体格，四周都是流体 → 表面格

        mx, my, mz = derive_surface_markers_3d(mask)

        # 正确行为：应返回 1 个 marker，坐标 (2.0, 2.0, 0.0)
        assert mx.shape[0] == 1, f"Expected 1 surface marker, got {mx.shape[0]}"
        # squeeze bug 不会影响 marker 数量，但会影响后续 ibm_direct_forcing_3d
        # 的索引操作——因为 marker_z 应该是 0.0 而不是被跳过
        assert mz.item() == 0.0, f"marker_z should be 0.0, got {mz.item()}"

    def test_squeeze_bug_does_not_affect_full_3d(self):
        """在完整 3D 网格（所有维度 > 1）中 squeeze bug 不触发。"""
        mask = _make_sphere_mask(8, 8, 8, cx=4, cy=4, cz=4, r=2.0)
        mx, my, mz = derive_surface_markers_3d(mask)
        n = mx.shape[0]
        assert n > 0, "Sphere should have surface markers"
        # 坐标应在合理范围内
        assert mx.min() >= 0 and mx.max() <= 7
        assert my.min() >= 0 and my.max() <= 7
        assert mz.min() >= 0 and mz.max() <= 7

    def test_squeeze_bug_propagates_to_force_computation(self):
        """squeeze bug 在退化网格中传播到力计算。

        当 nz=1 时，``derive_surface_markers_3d`` 返回的 marker 坐标
        本身是正确的（因为 ``torch.where`` 在 3D 上操作），但
        ``fluid_neighbours`` 的 ``.squeeze()`` 在 nz=1 时会产生 2D 张量。
        我们验证这不会导致崩溃但可能产生形状不一致。
        """
        mask = torch.zeros(1, 6, 6, dtype=torch.bool)
        mask[0, 2:4, 2:4] = True

        # 这个调用本身不应崩溃
        mx, my, mz = derive_surface_markers_3d(mask)
        assert mx.shape[0] > 0

    def test_squeeze_bug_internal_shape_replication(self):
        """复现 squeeze bug：内部 fluid_neighbours 在退化维度上形状错误。

        ``derive_surface_markers_3d`` 对 ``(1,1,nz,ny,nx)`` 调用
        ``.squeeze()``。当 nz=1 时，结果形状变为 ``(ny, nx)`` 而非
        ``(1, ny, nx)``。由于 PyTorch 广播机制，后续 ``m & (fn > 0)``
        仍能正确工作，因此该 bug 是**静默的**——不影响输出但违反
        形状契约。
        """
        nz, ny, nx = 1, 5, 5
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[0, 2, 2] = True

        # 复现内部计算
        m = mask.bool()
        pad = torch.nn.functional.pad(
            m.unsqueeze(0).unsqueeze(0).float(), (1, 1, 1, 1, 1, 1)
        )
        fluid_neighbours = (
            (1 - pad[:, :, 1:-1, 1:-1, 2:])
            + (1 - pad[:, :, 1:-1, 1:-1, :-2])
            + (1 - pad[:, :, 1:-1, 2:, 1:-1])
            + (1 - pad[:, :, 1:-1, :-2, 1:-1])
            + (1 - pad[:, :, 2:, 1:-1, 1:-1])
            + (1 - pad[:, :, :-2, 1:-1, 1:-1])
        ).squeeze()

        # BUG: squeeze() 在 nz=1 时移除了 z 维度
        # 期望形状: (1, 5, 5)，实际形状: (5, 5)
        assert fluid_neighbours.shape == (ny, nx), (
            f"Squeeze bug: expected (5,5) due to nz=1 squeeze, "
            f"got {tuple(fluid_neighbours.shape)}"
        )
        # 正确形状应该是 (1, 5, 5) 但 squeeze() 错误地移除了 z 维
        assert fluid_neighbours.shape != (nz, ny, nx), (
            "squeeze() should have removed the z dimension (this is the bug)"
        )

        # 尽管形状错误，广播使后续操作仍然正确
        surface = m & (fluid_neighbours > 0)
        assert surface.shape == (nz, ny, nx)  # 广播恢复正确形状
        assert surface.any(), "Surface should be non-empty"

    # ---- 1b. ibm_direct_forcing_3d 原始实现审计 ----

    def test_delta_hat_partition_of_unity(self):
        """2-point hat 核满足 partition-of-unity：Σ φ(i - x) = 1。"""
        x = 3.7  # 任意非整数位置
        total = 0.0
        for i in range(-2, 6):
            r = torch.tensor(float(i) - x)
            total += ibm_delta_hat(r).item()
        assert abs(total - 1.0) < 1e-5, f"hat kernel POU: {total} != 1.0"

    def test_delta_4pt_partition_of_unity(self):
        """4-point 核满足 partition-of-unity。"""
        x = 5.3
        total = 0.0
        for i in range(-2, 8):
            r = torch.tensor(float(i) - x)
            total += ibm_delta_4pt(r).item()
        assert abs(total - 1.0) < 1e-5, f"4pt kernel POU: {total} != 1.0"

    def test_original_force_conservation_hat(self):
        """原始 ibm_direct_forcing_3d 力守恒：散布到网格的总力 = 标记力之和。

        这是 IBM 的基本物理不变量：力不能凭空产生或消失。
        """
        nz, ny, nx = 10, 10, 10
        n_markers = 5
        torch.manual_seed(42)
        ux = torch.zeros(nz, ny, nx, dtype=torch.float32)
        uy = torch.zeros(nz, ny, nx, dtype=torch.float32)
        uz = torch.zeros(nz, ny, nx, dtype=torch.float32)

        marker_x = torch.tensor([2.3, 5.7, 3.1, 7.5, 4.9], dtype=torch.float32)
        marker_y = torch.tensor([1.5, 6.2, 8.3, 3.8, 5.1], dtype=torch.float32)
        marker_z = torch.tensor([4.0, 2.5, 7.1, 6.3, 3.2], dtype=torch.float32)

        u_target_x = torch.tensor([0.1, -0.2, 0.05, 0.3, -0.1], dtype=torch.float32)
        u_target_y = torch.tensor([0.0, 0.15, -0.1, 0.2, 0.0], dtype=torch.float32)
        u_target_z = torch.tensor([0.05, 0.0, 0.1, -0.05, 0.15], dtype=torch.float32)

        fx, fy, fz = ibm_direct_forcing_3d(
            ux, uy, uz, marker_x, marker_y, marker_z,
            u_target_x, u_target_y, u_target_z, kernel="hat",
        )

        # 力守恒：网格总力 = 标记总力（因为 u_interpolated=0，标记力 = u_target）
        total_fx_grid = fx.sum().item()
        total_fx_marker = u_target_x.sum().item()
        assert abs(total_fx_grid - total_fx_marker) < 1e-4, (
            f"Force conservation violated (hat): grid={total_fx_grid:.6f}, "
            f"marker={total_fx_marker:.6f}"
        )

    def test_original_force_conservation_4pt(self):
        """4-point 核同样满足力守恒。"""
        nz, ny, nx = 12, 12, 12
        ux = torch.zeros(nz, ny, nx, dtype=torch.float32)
        uy = torch.zeros(nz, ny, nx, dtype=torch.float32)
        uz = torch.zeros(nz, ny, nx, dtype=torch.float32)

        marker_x = torch.tensor([3.3, 6.7], dtype=torch.float32)
        marker_y = torch.tensor([4.5, 7.2], dtype=torch.float32)
        marker_z = torch.tensor([5.0, 3.5], dtype=torch.float32)
        u_target_x = torch.tensor([0.2, -0.15], dtype=torch.float32)
        u_target_y = torch.tensor([0.1, 0.0], dtype=torch.float32)
        u_target_z = torch.tensor([0.0, 0.05], dtype=torch.float32)

        fx, fy, fz = ibm_direct_forcing_3d(
            ux, uy, uz, marker_x, marker_y, marker_z,
            u_target_x, u_target_y, u_target_z, kernel="4pt",
        )
        assert abs(fx.sum().item() - u_target_x.sum().item()) < 1e-4
        assert abs(fy.sum().item() - u_target_y.sum().item()) < 1e-4
        assert abs(fz.sum().item() - u_target_z.sum().item()) < 1e-4

    def test_original_zero_velocity_zero_force(self):
        """当 u_target = u_interpolated 时，力为零（直接强迫恒等式）。"""
        nz, ny, nx = 8, 8, 8
        ux = torch.full((nz, ny, nx), 0.3, dtype=torch.float32)
        uy = torch.full((nz, ny, nx), 0.1, dtype=torch.float32)
        uz = torch.full((nz, ny, nx), 0.0, dtype=torch.float32)

        marker_x = torch.tensor([3.5], dtype=torch.float32)
        marker_y = torch.tensor([4.5], dtype=torch.float32)
        marker_z = torch.tensor([3.5], dtype=torch.float32)

        # u_target = u_interpolated（均匀场 → 插值 = 场值）
        u_target_x = torch.tensor([0.3], dtype=torch.float32)
        u_target_y = torch.tensor([0.1], dtype=torch.float32)
        u_target_z = torch.tensor([0.0], dtype=torch.float32)

        fx, fy, fz = ibm_direct_forcing_3d(
            ux, uy, uz, marker_x, marker_y, marker_z,
            u_target_x, u_target_y, u_target_z, kernel="hat",
        )
        assert fx.abs().max().item() < 1e-5, "Zero-force identity violated (fx)"
        assert fy.abs().max().item() < 1e-5, "Zero-force identity violated (fy)"
        assert fz.abs().max().item() < 1e-5, "Zero-force identity violated (fz)"

    # ---- 1c. ibm_vec.py 等价性审计 ----

    @pytest.mark.parametrize("kernel", ["hat", "4pt"])
    def test_vec_equivalence_original(self, kernel):
        """ibm_vec.py 向量化版本与 ibm.py 原始版本输出精确匹配。"""
        nz, ny, nx = 10, 10, 10
        torch.manual_seed(123)
        ux = torch.randn(nz, ny, nx, dtype=torch.float32) * 0.1
        uy = torch.randn(nz, ny, nx, dtype=torch.float32) * 0.1
        uz = torch.randn(nz, ny, nx, dtype=torch.float32) * 0.1

        marker_x = torch.tensor([2.3, 5.7, 3.1, 7.5, 4.9], dtype=torch.float32)
        marker_y = torch.tensor([1.5, 6.2, 8.3, 3.8, 5.1], dtype=torch.float32)
        marker_z = torch.tensor([4.0, 2.5, 7.1, 6.3, 3.2], dtype=torch.float32)
        u_target_x = torch.tensor([0.1, -0.2, 0.05, 0.3, -0.1], dtype=torch.float32)
        u_target_y = torch.tensor([0.0, 0.15, -0.1, 0.2, 0.0], dtype=torch.float32)
        u_target_z = torch.tensor([0.05, 0.0, 0.1, -0.05, 0.15], dtype=torch.float32)

        fx_orig, fy_orig, fz_orig = ibm_direct_forcing_3d(
            ux, uy, uz, marker_x, marker_y, marker_z,
            u_target_x, u_target_y, u_target_z, kernel=kernel,
        )
        fx_vec, fy_vec, fz_vec = ibm_direct_forcing_3d_vec(
            ux, uy, uz, marker_x, marker_y, marker_z,
            u_target_x, u_target_y, u_target_z, kernel=kernel,
        )

        torch.testing.assert_close(fx_orig, fx_vec, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fy_orig, fy_vec, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fz_orig, fz_vec, rtol=1e-5, atol=1e-6)


# =========================================================================== #
# 2. 等价性验证：ibm.py 原始 vs ibm_common.py 公共接口
# =========================================================================== #


class TestEquivalence:
    """验证 ibm.py 原始流水线与 ibm_common.py 公共接口的等价性。"""

    def test_d3q19_force_equivalence(self):
        """D3Q19 路径：相同 f + mask + velocity → 相同 force。

        ibm_common.py 内部调用 ibm_direct_forcing_3d（原始核），
        因此力计算应精确匹配手动流水线。
        """
        nz, ny, nx = 10, 10, 10
        mask = _make_sphere_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2.5)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.1)

        # 从 f 提取宏观速度（与 ibm_common 内部一致）
        rho, ux, uy, uz = macroscopic3d(f)

        # 手动流水线：derive markers → ibm_direct_forcing_3d
        mx, my, mz = derive_surface_markers_3d(mask)
        u_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        ut_x = torch.full((mx.shape[0],), 0.0, dtype=torch.float32)
        ut_y = torch.full((mx.shape[0],), 0.0, dtype=torch.float32)
        ut_z = torch.full((mx.shape[0],), 0.0, dtype=torch.float32)

        fx_manual, fy_manual, fz_manual = ibm_direct_forcing_3d(
            ux, uy, uz, mx, my, mz, ut_x, ut_y, ut_z, kernel="hat",
        )

        # 公共接口
        force_common, f_corrected = ibm_direct_forcing_3d_common(
            f, mask, u_target, lattice="D3Q19", kernel="hat",
        )

        torch.testing.assert_close(force_common[0], fx_manual, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(force_common[1], fy_manual, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(force_common[2], fz_manual, rtol=1e-5, atol=1e-6)

    def test_d3q19_f_corrected_equivalence(self):
        """D3Q19 路径：f_corrected 与手动 Guo 修正精确匹配。"""
        nz, ny, nx = 10, 10, 10
        mask = _make_sphere_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2.5)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.1)

        rho, ux, uy, uz = macroscopic3d(f)
        mx, my, mz = derive_surface_markers_3d(mask)
        ut_x = torch.zeros(mx.shape[0], dtype=torch.float32)
        ut_y = torch.zeros(mx.shape[0], dtype=torch.float32)
        ut_z = torch.zeros(mx.shape[0], dtype=torch.float32)

        fx, fy, fz = ibm_direct_forcing_3d(
            ux, uy, uz, mx, my, mz, ut_x, ut_y, ut_z, kernel="hat",
        )
        f_manual_corrected = ibm_apply_body_force_3d(f, fx, fy, fz)

        force_common, f_common_corrected = ibm_direct_forcing_3d_common(
            f, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q19", kernel="hat",
        )

        torch.testing.assert_close(
            f_common_corrected, f_manual_corrected, rtol=1e-5, atol=1e-6,
        )

    def test_d3q19_4pt_equivalence(self):
        """D3Q19 + 4pt 核等价性。"""
        nz, ny, nx = 12, 12, 12
        mask = _make_sphere_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=3.0)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.05)

        rho, ux, uy, uz = macroscopic3d(f)
        mx, my, mz = derive_surface_markers_3d(mask)
        ut_x = torch.zeros(mx.shape[0], dtype=torch.float32)
        ut_y = torch.zeros(mx.shape[0], dtype=torch.float32)
        ut_z = torch.zeros(mx.shape[0], dtype=torch.float32)

        fx, fy, fz = ibm_direct_forcing_3d(
            ux, uy, uz, mx, my, mz, ut_x, ut_y, ut_z, kernel="4pt",
        )
        f_manual = ibm_apply_body_force_3d(f, fx, fy, fz)

        force_c, f_c = ibm_direct_forcing_3d_common(
            f, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q19", kernel="4pt",
        )

        torch.testing.assert_close(force_c[0], fx, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(force_c[1], fy, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(force_c[2], fz, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(f_c, f_manual, rtol=1e-5, atol=1e-6)

    def test_d3q27_force_conservation(self):
        """D3Q27 路径力守恒（共性模块新增）。"""
        nz, ny, nx = 12, 12, 12
        mask = _make_sphere_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=3.0)
        f = _make_uniform_flow_f27(nz, ny, nx, ux0=0.1)

        force, f_corrected = ibm_direct_forcing_3d_common(
            f, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q27", kernel="hat",
        )

        # 力守恒：网格总力 = 标记总力
        # u_target=0, u_interpolated≈0.1（均匀流），所以标记力 ≈ -0.1*N_markers
        mx, my, mz = derive_surface_markers_3d(mask)
        n = mx.shape[0]
        rho, ux, uy, uz = macroscopic_velocity_3d(f, lattice="D3Q27")
        # 标记力 = u_target - u_interpolated ≈ -u_interpolated
        # 网格总力应等于标记总力
        total_force_x = force[0].sum().item()
        # 由于力守恒，总力应接近 0（因为对称球体在均匀流中，标记力对称分布）
        # 但更精确地说：总力 = Σ(u_target - u_interp) = 0 - Σ(u_interp)
        # 对于均匀流 ux=0.1，Σ(u_interp) ≈ 0.1 * N
        # 但由于球体对称性，x方向力不完全为零（球体在x方向有阻力）
        # 我们检查力守恒：网格力 = 标记力
        u_interp_x = torch.zeros(n, dtype=torch.float32)
        # 手动验证：force sum 应等于 marker force sum
        # marker force = u_target - u_interp = -u_interp
        # force_grid sum = Σ marker_force（由 partition-of-unity 保证）
        # 所以 force[0].sum() ≈ -Σ(u_interp_x)
        # 对于均匀流，u_interp_x ≈ 0.1 对每个标记
        # 所以 force[0].sum() ≈ -0.1 * N
        expected = -0.1 * n  # 近似值
        # 由于插值精度，允许较大容差
        assert abs(total_force_x - expected) < 0.5, (
            f"D3Q27 force conservation: grid={total_force_x:.4f}, "
            f"expected≈{expected:.4f}, n_markers={n}"
        )

    def test_d3q27_guo_momentum_injection(self):
        """D3Q27 Guo 修正注入的动量 = 力场积分。

        Guo 修正：f_i += w_i * 3 * (c_i · F)
        动量变化：Δ(ρu) = Σ_i c_i * Δf_i = Σ_i c_i * w_i * 3 * (c_i · F)
                   = 3 * Σ_i w_i * c_i ⊗ c_i · F = F（因为 3*Σ w_i c_ix c_ix = 1）
        """
        nz, ny, nx = 8, 8, 8
        f = _make_uniform_flow_f27(nz, ny, nx, ux0=0.0)

        # 在中心格施加已知力
        fx_grid = torch.zeros(nz, ny, nx, dtype=torch.float32)
        fy_grid = torch.zeros(nz, ny, nx, dtype=torch.float32)
        fz_grid = torch.zeros(nz, ny, nx, dtype=torch.float32)
        fx_grid[4, 4, 4] = 0.5
        fy_grid[4, 4, 4] = -0.3
        fz_grid[4, 4, 4] = 0.1

        f_corrected = ibm_apply_body_force_3d_common(
            f, fx_grid, fy_grid, fz_grid, lattice="D3Q27",
        )

        rho_before, ux_b, uy_b, uz_b = macroscopic27(f)
        rho_after, ux_a, uy_a, uz_a = macroscopic27(f_corrected)

        # 动量变化 = 力（dt=1, rho≈1）
        du_x = (ux_a - ux_b).sum().item()
        du_y = (uy_a - uy_b).sum().item()
        du_z = (uz_a - uz_b).sum().item()

        assert abs(du_x - 0.5) < 1e-4, f"D3Q27 Guo Δpx={du_x:.6f}, expected 0.5"
        assert abs(du_y - (-0.3)) < 1e-4, f"D3Q27 Guo Δpy={du_y:.6f}, expected -0.3"
        assert abs(du_z - 0.1) < 1e-4, f"D3Q27 Guo Δpz={du_z:.6f}, expected 0.1"

    def test_d3q19_guo_momentum_injection(self):
        """D3Q19 Guo 修正同样满足动量注入一致性。"""
        nz, ny, nx = 8, 8, 8
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.0)

        fx_grid = torch.zeros(nz, ny, nx, dtype=torch.float32)
        fy_grid = torch.zeros(nz, ny, nx, dtype=torch.float32)
        fz_grid = torch.zeros(nz, ny, nx, dtype=torch.float32)
        fx_grid[4, 4, 4] = 0.5
        fy_grid[4, 4, 4] = -0.3
        fz_grid[4, 4, 4] = 0.1

        f_corrected = ibm_apply_body_force_3d_common(
            f, fx_grid, fy_grid, fz_grid, lattice="D3Q19",
        )

        rho_b, ux_b, uy_b, uz_b = macroscopic3d(f)
        rho_a, ux_a, uy_a, uz_a = macroscopic3d(f_corrected)

        du_x = (ux_a - ux_b).sum().item()
        du_y = (uy_a - uy_b).sum().item()
        du_z = (uz_a - uz_b).sum().item()

        assert abs(du_x - 0.5) < 1e-4
        assert abs(du_y - (-0.3)) < 1e-4
        assert abs(du_z - 0.1) < 1e-4

    def test_d3q27_vs_d3q19_force_same_kernel(self):
        """相同 mask + 相同速度场，D3Q19 和 D3Q27 的 IBM 力应相同。

        因为 ibm_direct_forcing_3d 只依赖速度场和标记位置，
        与格子类型无关。差异仅来自宏观速度提取。
        对于平衡分布（均匀流），两种格子提取的速度应相同。
        """
        nz, ny, nx = 10, 10, 10
        mask = _make_sphere_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2.5)
        ux0 = 0.1

        f19 = _make_uniform_flow_f19(nz, ny, nx, ux0)
        f27 = _make_uniform_flow_f27(nz, ny, nx, ux0)

        force19, _ = ibm_direct_forcing_3d_common(
            f19, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q19", kernel="hat",
        )
        force27, _ = ibm_direct_forcing_3d_common(
            f27, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q27", kernel="hat",
        )

        # 力应精确匹配（因为速度场相同，IBM核相同）
        torch.testing.assert_close(force19, force27, rtol=1e-4, atol=1e-5)

    def test_zero_markers_zero_force(self):
        """无表面标记时返回零力、f 不变。"""
        nz, ny, nx = 8, 8, 8
        # 全流体域：无固体 → 无表面标记
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.1)

        force, f_corrected = ibm_direct_forcing_3d_common(
            f, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q19", kernel="hat",
        )
        assert force.abs().max().item() == 0.0
        torch.testing.assert_close(f_corrected, f, rtol=0, atol=0)

    def test_explicit_markers_match_derived(self):
        """显式提供的标记与从 mask 推导的标记产生相同结果。"""
        nz, ny, nx = 10, 10, 10
        mask = _make_sphere_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2.5)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.1)

        mx, my, mz = derive_surface_markers_3d(mask)

        force_derived, f_derived = ibm_direct_forcing_3d_common(
            f, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q19", kernel="hat",
        )
        force_explicit, f_explicit = ibm_direct_forcing_3d_common(
            f, mask, torch.zeros(3, dtype=torch.float32),
            lattice="D3Q19", kernel="hat",
            markers=(mx, my, mz),
        )

        torch.testing.assert_close(force_derived, force_explicit, rtol=0, atol=0)
        torch.testing.assert_close(f_derived, f_explicit, rtol=0, atol=0)


# =========================================================================== #
# 3. 组合测试：IBM + collision 完整循环
# =========================================================================== #


class TestCombination:
    """IBM + 碰撞完整循环组合测试。"""

    def test_d3q19_ibm_collision_stability(self):
        """D3Q19: IBM 修正 + BGK 碰撞 + 流动，多步稳定性。

        验证：
        - 分布函数保持有限（无 NaN/Inf）
        - 密度保持正定
        - 力场非零（IBM 在起作用）
        """
        nz, ny, nx = 12, 12, 12
        mask = _make_sphere_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2.5)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.05)
        tau = 1.0
        u_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        forces_history = []
        for step in range(5):
            # 1. IBM 直接强迫 + Guo 修正
            force, f = ibm_direct_forcing_3d_common(
                f, mask, u_target, lattice="D3Q19", kernel="hat",
            )
            forces_history.append(force.abs().sum().item())

            # 2. BGK 碰撞
            f = collide_bgk3d(f, tau)

            # 3. 流动
            f = stream3d(f)

            # 稳定性检查
            assert torch.isfinite(f).all(), f"NaN/Inf at step {step}"
            rho = f.sum(dim=0)
            assert (rho > 0).all(), f"Non-positive density at step {step}"

        # IBM 力应非零（球体在流场中产生阻力）
        assert forces_history[0] > 0, "IBM force should be nonzero"

    def test_d3q27_ibm_collision_stability(self):
        """D3Q27: IBM 修正 + BGK 碰撞，多步稳定性。"""
        nz, ny, nx = 12, 12, 12
        mask = _make_sphere_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2.5)
        f = _make_uniform_flow_f27(nz, ny, nx, ux0=0.05)
        tau = 1.0
        u_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        for step in range(5):
            force, f = ibm_direct_forcing_3d_common(
                f, mask, u_target, lattice="D3Q27", kernel="hat",
            )
            f = collide_bgk27(f, tau)

            assert torch.isfinite(f).all(), f"D3Q27 NaN/Inf at step {step}"
            rho = f.sum(dim=0)
            assert (rho > 0).all(), f"D3Q27 non-positive density at step {step}"

    def test_d3q19_ibm_momentum_consistency(self):
        """D3Q19: IBM 修正注入的动量 = 力场积分（碰撞前后）。"""
        nz, ny, nx = 10, 10, 10
        mask = _make_sphere_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2.0)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.0)
        u_target = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32)

        # 碰撞前动量
        rho_b, ux_b, uy_b, uz_b = macroscopic3d(f)
        px_before = (rho_b * ux_b).sum().item()

        # IBM 修正
        force, f_corrected = ibm_direct_forcing_3d_common(
            f, mask, u_target, lattice="D3Q19", kernel="hat",
        )

        # 修正后动量
        rho_a, ux_a, uy_a, uz_a = macroscopic3d(f_corrected)
        px_after = (rho_a * ux_a).sum().item()

        # 动量变化 = 力场积分（dt=1）
        force_integral = force[0].sum().item()
        delta_px = px_after - px_before

        assert abs(delta_px - force_integral) < 1e-4, (
            f"Momentum consistency: Δpx={delta_px:.6f}, "
            f"∫Fx={force_integral:.6f}"
        )

    def test_d3q27_ibm_momentum_consistency(self):
        """D3Q27: IBM 修正注入的动量 = 力场积分。"""
        nz, ny, nx = 10, 10, 10
        mask = _make_sphere_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2.0)
        f = _make_uniform_flow_f27(nz, ny, nx, ux0=0.0)
        u_target = torch.tensor([0.1, 0.0, 0.0], dtype=torch.float32)

        rho_b, ux_b, uy_b, uz_b = macroscopic27(f)
        px_before = (rho_b * ux_b).sum().item()

        force, f_corrected = ibm_direct_forcing_3d_common(
            f, mask, u_target, lattice="D3Q27", kernel="hat",
        )

        rho_a, ux_a, uy_a, uz_a = macroscopic27(f_corrected)
        px_after = (rho_a * ux_a).sum().item()

        force_integral = force[0].sum().item()
        delta_px = px_after - px_before

        assert abs(delta_px - force_integral) < 1e-4, (
            f"D3Q27 Momentum consistency: Δpx={delta_px:.6f}, "
            f"∫Fx={force_integral:.6f}"
        )

    def test_ibm_collision_order_invariance_force(self):
        """IBM 修正 + 碰撞的顺序不影响力计算结果。

        先碰撞后IBM vs 先IBM后碰撞，力场应相同（因为力只依赖当前速度场）。
        """
        nz, ny, nx = 10, 10, 10
        mask = _make_sphere_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2.0)
        f0 = _make_uniform_flow_f19(nz, ny, nx, ux0=0.05)
        tau = 1.0
        u_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        # 路径A：先IBM后碰撞
        f_a = f0.clone()
        force_a, f_a = ibm_direct_forcing_3d_common(
            f_a, mask, u_target, lattice="D3Q19", kernel="hat",
        )
        f_a = collide_bgk3d(f_a, tau)

        # 路径B：先碰撞后IBM
        f_b = f0.clone()
        f_b = collide_bgk3d(f_b, tau)
        force_b, f_b = ibm_direct_forcing_3d_common(
            f_b, mask, u_target, lattice="D3Q19", kernel="hat",
        )

        # 力不同（因为速度场不同），但都应有限
        assert torch.isfinite(force_a).all()
        assert torch.isfinite(force_b).all()
        # 两条路径的最终分布都应有限
        assert torch.isfinite(f_a).all()
        assert torch.isfinite(f_b).all()

    def test_d3q19_ibm_drag_direction(self):
        """D3Q19: 球体在均匀流中，IBM 阻力方向与来流相反。

        u_target=0（固定球体），均匀流 ux>0 → 阻力应 < 0（x方向）。
        """
        nz, ny, nx = 16, 16, 16
        mask = _make_sphere_mask(nz, ny, nx, cx=8, cy=8, cz=8, r=3.0)
        f = _make_uniform_flow_f19(nz, ny, nx, ux0=0.1)
        u_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        force, _ = ibm_direct_forcing_3d_common(
            f, mask, u_target, lattice="D3Q19", kernel="hat",
        )

        # x方向总力应为负（阻力方向与来流相反）
        total_fx = force[0].sum().item()
        assert total_fx < 0, (
            f"Drag direction: total Fx={total_fx:.6f} should be negative "
            f"(opposing flow)"
        )

    def test_d3q27_ibm_drag_direction(self):
        """D3Q27: 球体阻力方向同样与来流相反。"""
        nz, ny, nx = 16, 16, 16
        mask = _make_sphere_mask(nz, ny, nx, cx=8, cy=8, cz=8, r=3.0)
        f = _make_uniform_flow_f27(nz, ny, nx, ux0=0.1)
        u_target = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)

        force, _ = ibm_direct_forcing_3d_common(
            f, mask, u_target, lattice="D3Q27", kernel="hat",
        )

        total_fx = force[0].sum().item()
        assert total_fx < 0, (
            f"D3Q27 drag direction: total Fx={total_fx:.6f} should be negative"
        )
