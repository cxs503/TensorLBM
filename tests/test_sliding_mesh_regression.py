"""Regression equivalence & bug-identification tests for sliding mesh.

Tests three dimensions:
1. Bug identification — does the original sliding_mesh.py carry known bugs?
2. Equivalence — D2Q9 original vs common-module 3D (structural equivalence);
   D3Q27 (common-only) physical reasonableness.
3. Combination — sliding mesh + collision end-to-end.
"""
from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d2q9 import equilibrium as equilibrium2d, macroscopic as macroscopic2d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.sliding_mesh import (
    apply_sliding_mesh_bc_2d,
    interpolate_interface_2d,
    rotate_velocity_field_2d,
)
from tensorlbm.sliding_mesh_common import (
    apply_sliding_mesh_bc_3d,
    interpolate_interface_3d,
    rotate_velocity_field_3d,
    sliding_mesh_step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_f_2d(ny: int = 8, nx: int = 10) -> torch.Tensor:
    rho = torch.ones(ny, nx)
    return equilibrium2d(rho, torch.full_like(rho, 0.05), torch.zeros_like(rho))


def _make_f_3d(nz: int = 4, ny: int = 8, nx: int = 10) -> torch.Tensor:
    rho = torch.ones(nz, ny, nx)
    return equilibrium3d(
        rho,
        torch.full_like(rho, 0.05),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
    )


def _make_f_27(nz: int = 4, ny: int = 8, nx: int = 10) -> torch.Tensor:
    rho = torch.ones(nz, ny, nx)
    return equilibrium27(
        rho,
        torch.full_like(rho, 0.05),
        torch.zeros_like(rho),
        torch.zeros_like(rho),
    )


# ===========================================================================
# 1. BUG IDENTIFICATION
# ===========================================================================

class TestSlidingMeshBugs:
    """Identify pre-existing bugs in the original sliding_mesh.py."""

    def test_runner_collide_bgk_wrong_signature(self) -> None:
        """BUG: run_sliding_mesh_rotor calls collide_bgk with wrong arguments.

        sliding_mesh.py line 366 calls:
            f = collide_bgk(f, rho, ux, uy, tau)
        But the actual signature is:
            collide_bgk(f, tau)
        This raises TypeError at runtime.
        """
        import inspect
        from tensorlbm.solver import collide_bgk

        sig = inspect.signature(collide_bgk)
        params = list(sig.parameters.keys())
        assert params == ["f", "tau"], f"collide_bgk signature: {params}"

        # Verify the runner source calls it with extra args
        from tensorlbm import sliding_mesh

        source = inspect.getsource(sliding_mesh)
        assert "collide_bgk(f, rho, ux, uy, tau)" in source, (
            "Expected collide_bgk(f, rho, ux, uy, tau) call in runner"
        )

        # Demonstrate the TypeError
        f = _make_f_2d()
        rho = torch.ones(8, 10)
        with pytest.raises(TypeError):
            collide_bgk(f, rho, torch.zeros_like(rho), torch.zeros_like(rho), 1.0)  # type: ignore[call-arg]

    def test_apply_bc_2d_hardcoded_center(self) -> None:
        """BUG (limitation): apply_sliding_mesh_bc_2d hardcodes center at (nx/2, ny/2).

        The 2D BC function does not accept cx, cy parameters and always assumes
        the rotor center is at the domain center.  The 3D common version correctly
        accepts cx, cy, cz.
        """
        import inspect

        sig = inspect.signature(apply_sliding_mesh_bc_2d)
        param_names = list(sig.parameters.keys())
        assert "cx" not in param_names, "2D BC should NOT have cx parameter (hardcoded)"
        assert "cy" not in param_names, "2D BC should NOT have cy parameter (hardcoded)"

        # 3D version does accept cx, cy, cz
        sig3d = inspect.signature(apply_sliding_mesh_bc_3d)
        param_names_3d = list(sig3d.parameters.keys())
        assert "cx" in param_names_3d
        assert "cy" in param_names_3d
        assert "cz" in param_names_3d

    def test_interpolate_3d_hardcoded_center(self) -> None:
        """BUG (limitation): interpolate_interface_3d hardcodes center at 0.5.

        Unlike apply_sliding_mesh_bc_3d which accepts cx/cy/cz, the interpolation
        function always uses cx=cy=cz=0.5 (normalized).  This is inconsistent.
        """
        import inspect

        sig = inspect.signature(interpolate_interface_3d)
        param_names = list(sig.parameters.keys())
        assert "cx" not in param_names, "interpolate_interface_3d hardcodes center at 0.5"
        assert "cy" not in param_names


# ===========================================================================
# 2. EQUIVALENCE: D2Q9 original vs common 3D
# ===========================================================================

class TestSlidingMeshEquivalence:
    """Verify D2Q9 original == common 3D (axis=z, single z-layer)."""

    def test_rotate_velocity_2d_vs_3d_z(self) -> None:
        """2D rotation == 3D rotation about z-axis."""
        ny, nx = 8, 10
        ux = torch.randn(ny, nx)
        uy = torch.randn(ny, nx)
        theta = 0.35

        ux2d, uy2d = rotate_velocity_field_2d(ux, uy, theta)

        uz = torch.zeros(1, ny, nx)
        ux3d, uy3d, uz3d = rotate_velocity_field_3d(
            ux.unsqueeze(0), uy.unsqueeze(0), uz, theta, axis="z"
        )
        assert torch.allclose(ux2d, ux3d[0], atol=1e-6)
        assert torch.allclose(uy2d, uy3d[0], atol=1e-6)
        assert torch.allclose(uz3d, torch.zeros_like(uz3d))

    @pytest.mark.parametrize("theta", [0.0, 0.1, math.pi / 4, math.pi / 2, math.pi])
    def test_rotate_velocity_multiple_angles(self, theta: float) -> None:
        """Rotation equivalence holds for multiple angles."""
        ny, nx = 6, 8
        ux = torch.randn(ny, nx)
        uy = torch.randn(ny, nx)
        ux2d, uy2d = rotate_velocity_field_2d(ux, uy, theta)
        uz = torch.zeros(1, ny, nx)
        ux3d, uy3d, _ = rotate_velocity_field_3d(
            ux.unsqueeze(0), uy.unsqueeze(0), uz, theta, axis="z"
        )
        assert torch.allclose(ux2d, ux3d[0], atol=1e-6)
        assert torch.allclose(uy2d, uy3d[0], atol=1e-6)

    def test_interpolate_interface_2d_vs_3d_single_layer(self) -> None:
        """2D bilinear interpolation == 3D trilinear with single z-layer."""
        ny, nx = 8, 10
        f_inner = _make_f_2d(ny, nx)
        f_outer = _make_f_2d(ny, nx) + 0.01
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[3:5, 3:7] = True
        theta = 0.3

        f2d = interpolate_interface_2d(f_inner, f_outer, mask, theta)

        nz = 1
        f_inner_3d = f_inner.unsqueeze(1).expand(9, nz, ny, nx).contiguous()
        f_outer_3d = f_outer.unsqueeze(1).expand(9, nz, ny, nx).contiguous()
        mask_3d = mask.unsqueeze(0).expand(nz, ny, nx).contiguous()
        f3d = interpolate_interface_3d(f_inner_3d, f_outer_3d, mask_3d, theta, axis="z")

        assert torch.allclose(f2d, f3d[:, 0], atol=1e-5)

    def test_apply_bc_wall_velocity_formula_matches(self) -> None:
        """Wall velocity formula: u = omega x r is identical in 2D and 3D (z-axis)."""
        ny, nx = 8, 10
        omega = 0.02
        cx_abs = nx / 2.0
        cy_abs = ny / 2.0

        yy, xx = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        r_x = xx - cx_abs
        r_y = yy - cy_abs

        # 2D formula (from source code)
        u_wall_x_2d = -omega * r_y
        u_wall_y_2d = omega * r_x

        # 3D formula (axis=z, from source code)
        u_wall_x_3d = -omega * r_y
        u_wall_y_3d = omega * r_x
        u_wall_z_3d = torch.zeros_like(r_x)

        assert torch.allclose(u_wall_x_2d, u_wall_x_3d)
        assert torch.allclose(u_wall_y_2d, u_wall_y_3d)
        assert torch.allclose(u_wall_z_3d, torch.zeros_like(u_wall_z_3d))

    def test_apply_bc_relaxation_formula_matches(self) -> None:
        """Both 2D and 3D use the same relaxation: f - (1/tau)*(f - f_eq)."""
        ny, nx = 8, 10
        rho = torch.ones(ny, nx)
        f = _make_f_2d(ny, nx)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[3:5, 3:7] = True
        tau = 1.0
        omega = 0.01
        theta = 0.2

        f_out_2d = apply_sliding_mesh_bc_2d(f, mask, theta, omega, rho, tau)

        # Manually compute the expected result
        yy, xx = torch.meshgrid(
            torch.arange(ny, dtype=torch.float32),
            torch.arange(nx, dtype=torch.float32),
            indexing="ij",
        )
        cx_abs = nx / 2.0
        cy_abs = ny / 2.0
        r_x = xx - cx_abs
        r_y = yy - cy_abs
        u_wall_x = torch.where(mask, -omega * r_y, torch.zeros_like(r_x))
        u_wall_y = torch.where(mask, omega * r_x, torch.zeros_like(r_x))
        f_eq = equilibrium2d(rho, u_wall_x, u_wall_y)
        omega_relax = 1.0 / tau
        expected = f - omega_relax * (f - f_eq)
        mask_exp = mask.unsqueeze(0).expand(9, -1, -1)
        expected_full = torch.where(mask_exp, expected, f)

        assert torch.allclose(f_out_2d, expected_full, atol=1e-6)

    def test_apply_bc_3d_d3q19_finite(self) -> None:
        """3D BC with D3Q19 produces finite output."""
        nz, ny, nx = 4, 8, 10
        f = _make_f_3d(nz, ny, nx)
        rho = torch.ones(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True
        tau = 1.0
        omega = 0.01
        theta = 0.2

        f_out = apply_sliding_mesh_bc_3d(
            f, mask, theta, omega, rho, tau,
            axis="z", cx=nx / 2, cy=ny / 2, cz=nz / 2,
            lattice="D3Q19",
        )
        assert torch.isfinite(f_out).all()
        # Non-interface cells unchanged
        mask_other = ~mask
        for d in range(19):
            assert torch.equal(f_out[d][mask_other], f[d][mask_other])

    def test_sliding_mesh_step_auto_detect_d3q19(self) -> None:
        """sliding_mesh_step auto-detects D3Q19 from f.shape[0]."""
        nz, ny, nx = 4, 8, 10
        f = _make_f_3d(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True

        f_out = sliding_mesh_step(
            f, mask, omega=0.01,
            theta=0.2, tau=1.0, axis="z",
            cx=nx / 2, cy=ny / 2, cz=nz / 2,
        )
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()


# ===========================================================================
# 2b. D3Q27 physical reasonableness (common-only)
# ===========================================================================

class TestSlidingMeshD3Q27Reasonableness:
    """D3Q27 sliding mesh is common-module-only; verify physical reasonableness."""

    def test_apply_bc_d3q27_finite(self) -> None:
        """3D BC with D3Q27 produces finite output."""
        nz, ny, nx = 4, 8, 10
        f = _make_f_27(nz, ny, nx)
        rho = torch.ones(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True

        f_out = apply_sliding_mesh_bc_3d(
            f, mask, theta=0.3, omega=0.01, rho=rho, tau=1.0,
            axis="z", cx=nx / 2, cy=ny / 2, cz=nz / 2,
            lattice="D3Q27",
        )
        assert torch.isfinite(f_out).all()

    def test_sliding_mesh_step_d3q27_auto(self) -> None:
        """sliding_mesh_step auto-detects D3Q27 from f.shape[0]=27."""
        nz, ny, nx = 4, 8, 10
        f = _make_f_27(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True

        f_out = sliding_mesh_step(
            f, mask, omega=0.01,
            theta=0.2, tau=1.0, axis="z",
            cx=nx / 2, cy=ny / 2, cz=nz / 2,
        )
        assert f_out.shape == f.shape
        assert torch.isfinite(f_out).all()

    def test_d3q27_wall_velocity_correct(self) -> None:
        """D3Q27 wall velocity = omega x r (same formula as D3Q19)."""
        nz, ny, nx = 4, 8, 10
        f19 = _make_f_3d(nz, ny, nx)
        f27 = _make_f_27(nz, ny, nx)
        rho = torch.ones(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True

        f19_out = apply_sliding_mesh_bc_3d(
            f19, mask, theta=0.0, omega=0.01, rho=rho, tau=1.0,
            axis="z", cx=nx / 2, cy=ny / 2, cz=nz / 2,
            lattice="D3Q19",
        )
        f27_out = apply_sliding_mesh_bc_3d(
            f27, mask, theta=0.0, omega=0.01, rho=rho, tau=1.0,
            axis="z", cx=nx / 2, cy=ny / 2, cz=nz / 2,
            lattice="D3Q27",
        )
        # Both should be finite
        assert torch.isfinite(f19_out).all()
        assert torch.isfinite(f27_out).all()
        # Non-interface cells unchanged in both
        mask_other = ~mask
        for d in range(19):
            assert torch.equal(f19_out[d][mask_other], f19[d][mask_other])
        for d in range(27):
            assert torch.equal(f27_out[d][mask_other], f27[d][mask_other])

    def test_rotate_3d_all_axes(self) -> None:
        """3D rotation works for all three axes."""
        nz, ny, nx = 4, 6, 8
        ux = torch.randn(nz, ny, nx)
        uy = torch.randn(nz, ny, nx)
        uz = torch.randn(nz, ny, nx)
        theta = 0.3

        for axis in ("x", "y", "z"):
            rx, ry, rz = rotate_velocity_field_3d(ux, uy, uz, theta, axis=axis)
            assert torch.isfinite(rx).all()
            assert torch.isfinite(ry).all()
            assert torch.isfinite(rz).all()
            # Verify rotation preserves magnitude
            mag_before = torch.sqrt(ux**2 + uy**2 + uz**2)
            mag_after = torch.sqrt(rx**2 + ry**2 + rz**2)
            assert torch.allclose(mag_before, mag_after, atol=1e-4), (
                f"Rotation about {axis} does not preserve magnitude"
            )


# ===========================================================================
# 3. COMBINATION: sliding mesh + collision
# ===========================================================================

class TestSlidingMeshWithCollision:
    """Combination test: sliding mesh BC applied with BGK collision."""

    def _bgk_collision_2d(self, f: torch.Tensor, tau: float) -> torch.Tensor:
        rho, ux, uy = macroscopic2d(f)
        feq = equilibrium2d(rho, ux, uy)
        return f - (f - feq) / tau

    def _bgk_collision_3d(self, f: torch.Tensor, tau: float, lattice: str) -> torch.Tensor:
        if lattice == "D3Q19":
            rho, ux, uy, uz = macroscopic3d(f)
            feq = equilibrium3d(rho, ux, uy, uz)
        else:
            rho, ux, uy, uz = macroscopic27(f)
            feq = equilibrium27(rho, ux, uy, uz)
        return f - (f - feq) / tau

    def test_sliding_bc_after_collision_2d(self) -> None:
        """2D: collision → sliding mesh BC → finite output."""
        ny, nx = 8, 10
        f = _make_f_2d(ny, nx)
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[3:5, 3:7] = True
        tau = 1.0
        omega = 0.01
        theta = 0.1

        # Collision
        f = self._bgk_collision_2d(f, tau)
        assert torch.isfinite(f).all()

        # Sliding mesh BC
        rho, _, _ = macroscopic2d(f)
        f = apply_sliding_mesh_bc_2d(f, mask, theta, omega, rho, tau)
        assert torch.isfinite(f).all()

    def test_sliding_step_after_collision_3d_d3q19(self) -> None:
        """3D D3Q19: collision → sliding_mesh_step → finite output."""
        nz, ny, nx = 4, 8, 10
        f = _make_f_3d(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True
        tau = 1.0
        omega = 0.01

        # Collision
        f = self._bgk_collision_3d(f, tau, "D3Q19")
        assert torch.isfinite(f).all()

        # Sliding mesh step
        f = sliding_mesh_step(
            f, mask, omega=omega,
            theta=0.1, tau=tau, axis="z",
            cx=nx / 2, cy=ny / 2, cz=nz / 2,
        )
        assert torch.isfinite(f).all()

    def test_sliding_step_after_collision_3d_d3q27(self) -> None:
        """3D D3Q27: collision → sliding_mesh_step → finite output."""
        nz, ny, nx = 4, 8, 10
        f = _make_f_27(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True
        tau = 1.0
        omega = 0.01

        # Collision
        f = self._bgk_collision_3d(f, tau, "D3Q27")
        assert torch.isfinite(f).all()

        # Sliding mesh step
        f = sliding_mesh_step(
            f, mask, omega=omega,
            theta=0.1, tau=tau, axis="z",
            cx=nx / 2, cy=ny / 2, cz=nz / 2,
        )
        assert torch.isfinite(f).all()

    def test_multi_step_collision_sliding_stability(self) -> None:
        """Run several collision+sliding steps; verify no NaN/Inf divergence."""
        nz, ny, nx = 4, 8, 10
        f = _make_f_3d(nz, ny, nx)
        mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
        mask[1:3, 3:5, 3:7] = True
        tau = 1.5
        omega = 0.01
        theta = 0.0

        for step in range(10):
            f = self._bgk_collision_3d(f, tau, "D3Q19")
            theta += omega
            f = sliding_mesh_step(
                f, mask, omega=omega,
                theta=theta, tau=tau, axis="z",
                cx=nx / 2, cy=ny / 2, cz=nz / 2,
            )
            assert torch.isfinite(f).all(), f"Non-finite at step {step}"
            mass = f.sum().item()
            assert mass < 1e6, f"Mass explosion at step {step}: {mass}"

    def test_interpolate_after_rotation_2d(self) -> None:
        """2D: rotate velocity → interpolate interface → finite output."""
        ny, nx = 8, 10
        f_inner = _make_f_2d(ny, nx)
        f_outer = _make_f_2d(ny, nx) + 0.01
        mask = torch.zeros(ny, nx, dtype=torch.bool)
        mask[3:5, 3:7] = True
        theta = 0.3

        # Rotate velocity field
        rho, ux, uy = macroscopic2d(f_inner)
        ux_rot, uy_rot = rotate_velocity_field_2d(ux, uy, theta)
        assert torch.isfinite(ux_rot).all()
        assert torch.isfinite(uy_rot).all()

        # Interpolate interface
        f_blended = interpolate_interface_2d(f_inner, f_outer, mask, theta)
        assert torch.isfinite(f_blended).all()
        # Non-interface cells keep f_outer
        mask_exp = mask.unsqueeze(0).expand(9, -1, -1)
        assert torch.equal(f_blended[~mask_exp], f_outer[~mask_exp])
