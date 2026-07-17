"""Contract tests for the common IBM direct-forcing module.

These tests verify operator algebra (shape, force conservation, equilibrium
fixed-point, zero-force identity, D3Q19/D3Q27 parity), NOT immersed-boundary
physics correctness or moving-body validation.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q27 import equilibrium27
from tensorlbm.ibm_common import (
    IBMCapabilityWithheldError,
    derive_surface_markers_3d,
    ibm_apply_body_force_3d_common,
    ibm_direct_forcing_3d_common,
    macroscopic_velocity_3d,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _solid_mask(nz: int, ny: int, nx: int, cx: int, cy: int, cz: int, r: int) -> torch.Tensor:
    """Return a boolean solid mask for a small sphere of radius ``r``."""
    iz, iy, ix = torch.meshgrid(
        torch.arange(nz), torch.arange(ny), torch.arange(nx), indexing="ij"
    )
    return (((ix - cx).float() ** 2 + (iy - cy).float() ** 2 + (iz - cz).float() ** 2) <= r ** 2)


# --------------------------------------------------------------------------- #
# Shape tests
# --------------------------------------------------------------------------- #


class TestIBMCommonShape:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_force_and_f_shapes(self, lattice, q, equilibrium) -> None:
        nz, ny, nx = 8, 10, 12
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=6, cy=5, cz=4, r=2)
        u_target = torch.zeros(3)
        force, f_corr = ibm_direct_forcing_3d_common(f, mask, u_target, lattice=lattice)
        assert force.shape == (3, nz, ny, nx)
        assert f_corr.shape == f.shape == (q, nz, ny, nx)

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_force_is_finite(self, lattice) -> None:
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        u = torch.full_like(rho, 0.01)
        zero = torch.zeros_like(rho)
        if lattice == "D3Q19":
            f = equilibrium3d(rho, u, zero, zero)
        else:
            f = equilibrium27(rho, u, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=4, cy=4, cz=4, r=2)
        u_target = torch.zeros(3)
        force, f_corr = ibm_direct_forcing_3d_common(f, mask, u_target, lattice=lattice)
        assert torch.isfinite(force).all()
        assert torch.isfinite(f_corr).all()


# --------------------------------------------------------------------------- #
# Zero-force identity: zero target in zero flow → zero force, f unchanged
# --------------------------------------------------------------------------- #


class TestIBMCommonZeroForceIdentity:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_zero_target_zero_flow_produces_zero_force(self, lattice, q, equilibrium) -> None:
        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        u_target = torch.zeros(3)
        force, f_corr = ibm_direct_forcing_3d_common(f, mask, u_target, lattice=lattice)
        assert torch.allclose(force, torch.zeros_like(force), atol=1e-7)
        # f should be unchanged (zero force → zero correction).
        assert torch.allclose(f_corr, f, atol=1e-7)


# --------------------------------------------------------------------------- #
# Equilibrium fixed-point: equilibrium with matching target velocity → ~zero force
# --------------------------------------------------------------------------- #


class TestIBMCommonEquilibriumFixedPoint:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_equilibrium_with_matching_velocity_small_force(self, lattice, q, equilibrium) -> None:
        """When the target velocity matches the fluid velocity, the IBM force
        should be near zero (the interpolated marker velocity ≈ field velocity
        for a uniform field)."""
        nz, ny, nx = 12, 12, 12
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.02)
        uy = torch.full_like(rho, 0.0)
        uz = torch.full_like(rho, 0.0)
        f = equilibrium(rho, ux, uy, uz)
        mask = _solid_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2)
        # Target = fluid velocity → force should be ~0.
        u_target = torch.tensor([0.02, 0.0, 0.0])
        force, f_corr = ibm_direct_forcing_3d_common(f, mask, u_target, lattice=lattice)
        # The interpolated velocity at markers should match the uniform field,
        # so the direct-forcing F = u_target - u_interp ≈ 0.
        assert force.abs().max().item() < 1e-4


# --------------------------------------------------------------------------- #
# Force conservation: total Eulerian force = total marker force
# --------------------------------------------------------------------------- #


class TestIBMCommonForceConservation:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_force_conservation_uniform_target(self, lattice, q, equilibrium) -> None:
        """For a uniform target velocity in zero flow, the total Eulerian IBM
        force should equal N_markers × u_target (force = u_target - u_interp,
        and u_interp = 0 in zero flow)."""
        nz, ny, nx = 12, 12, 12
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=6, cy=6, cz=6, r=2)
        u_target = torch.tensor([0.05, 0.0, 0.0])
        force, _ = ibm_direct_forcing_3d_common(f, mask, u_target, lattice=lattice)
        # Derive markers to count them.
        mx, my, mz = derive_surface_markers_3d(mask)
        n_markers = mx.shape[0]
        expected_x = n_markers * 0.05
        assert force[0].sum().item() == pytest.approx(expected_x, abs=1e-4)
        assert force[1].sum().item() == pytest.approx(0.0, abs=1e-6)
        assert force[2].sum().item() == pytest.approx(0.0, abs=1e-6)


# --------------------------------------------------------------------------- #
# D3Q19 / D3Q27 parity: same physics, different Q
# --------------------------------------------------------------------------- #


class TestIBMCommonLatticeParity:
    def test_d3q19_and_d3q27_both_produce_consistent_force(self) -> None:
        """Both lattices should produce finite, non-NaN forces for the same
        mask and target.  The magnitudes need not match exactly (different
        weights), but both must be well-formed."""
        nz, ny, nx = 10, 10, 10
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        u_target = torch.tensor([0.03, 0.0, 0.0])
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f19 = equilibrium3d(rho, zero, zero, zero)
        f27 = equilibrium27(rho, zero, zero, zero)
        force19, _ = ibm_direct_forcing_3d_common(f19, mask, u_target, lattice="D3Q19")
        force27, _ = ibm_direct_forcing_3d_common(f27, mask, u_target, lattice="D3Q27")
        assert torch.isfinite(force19).all()
        assert torch.isfinite(force27).all()
        # Both should have non-zero total x-force (target ≠ 0 in zero flow).
        assert force19[0].sum().abs().item() > 1e-6
        assert force27[0].sum().abs().item() > 1e-6


# --------------------------------------------------------------------------- #
# Kernel selection
# --------------------------------------------------------------------------- #


class TestIBMCommonKernelSelection:
    def test_hat_and_4pt_both_work(self) -> None:
        nz, ny, nx = 10, 10, 10
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = _solid_mask(nz, ny, nx, cx=5, cy=5, cz=5, r=2)
        u_target = torch.tensor([0.02, 0.0, 0.0])
        force_hat, _ = ibm_direct_forcing_3d_common(f, mask, u_target, kernel="hat")
        force_4pt, _ = ibm_direct_forcing_3d_common(f, mask, u_target, kernel="4pt")
        assert torch.isfinite(force_hat).all()
        assert torch.isfinite(force_4pt).all()


# --------------------------------------------------------------------------- #
# Fail-closed: unknown lattice / kernel
# --------------------------------------------------------------------------- #


class TestIBMCommonFailClosed:
    def test_unknown_lattice_raises(self) -> None:
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        with pytest.raises(IBMCapabilityWithheldError, match="WITHHELD_UNKNOWN_LATTICE"):
            ibm_direct_forcing_3d_common(f, mask, torch.zeros(3), lattice="D2Q9")

    def test_unknown_kernel_raises(self) -> None:
        nz, ny, nx = 8, 8, 8
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium3d(rho, zero, zero, zero)
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        with pytest.raises(IBMCapabilityWithheldError, match="WITHHELD_UNKNOWN_KERNEL"):
            ibm_direct_forcing_3d_common(f, mask, torch.zeros(3), kernel="3pt")


# --------------------------------------------------------------------------- #
# Macroscopic velocity extraction
# --------------------------------------------------------------------------- #


class TestMacroscopicVelocity3D:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_equilibrium_velocity_recovery(self, lattice, q, equilibrium) -> None:
        """macroscopic_velocity_3d should recover the velocity used to build
        the equilibrium distribution."""
        nz, ny, nx = 6, 6, 6
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, 0.02)
        f = equilibrium(rho, ux, uy, uz)
        rho_out, uxo, uyo, uzo = macroscopic_velocity_3d(f, lattice=lattice)
        assert torch.allclose(rho_out, rho, atol=1e-5)
        assert torch.allclose(uxo, ux, atol=1e-5)
        assert torch.allclose(uyo, uy, atol=1e-5)
        assert torch.allclose(uzo, uz, atol=1e-5)


# --------------------------------------------------------------------------- #
# Surface marker derivation
# --------------------------------------------------------------------------- #


class TestDeriveSurfaceMarkers:
    def test_empty_mask_gives_no_markers(self) -> None:
        mask = torch.zeros((4, 5, 6), dtype=torch.bool)
        mx, my, mz = derive_surface_markers_3d(mask)
        assert mx.shape[0] == 0
        assert my.shape[0] == 0
        assert mz.shape[0] == 0

    def test_solid_sphere_has_surface_markers(self) -> None:
        mask = _solid_mask(10, 10, 10, cx=5, cy=5, cz=5, r=2)
        mx, my, mz = derive_surface_markers_3d(mask)
        assert mx.shape[0] > 0
        assert mx.shape[0] == my.shape[0] == mz.shape[0]

    def test_fully_solid_border_is_surface(self) -> None:
        """A fully solid domain has surface cells only at the border (adjacent
        to the padded fluid region); interior cells are not surface."""
        mask = torch.ones((6, 6, 6), dtype=torch.bool)
        mx, my, mz = derive_surface_markers_3d(mask)
        # All surface cells should be on the border (at least one index is 0 or 5).
        assert mx.shape[0] > 0
        for x, y, z in zip(mx.tolist(), my.tolist(), mz.tolist()):
            on_border = (x in (0, 5)) or (y in (0, 5)) or (z in (0, 5))
            assert on_border, f"interior cell ({x},{y},{z}) should not be surface"


# --------------------------------------------------------------------------- #
# Body-force application (lattice-neutral)
# --------------------------------------------------------------------------- #


class TestApplyBodyForce3DCommon:
    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_zero_force_is_identity(self, lattice, q, equilibrium) -> None:
        nz, ny, nx = 6, 6, 6
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        fz = torch.zeros((nz, ny, nx))
        out = ibm_apply_body_force_3d_common(f, fz, fz, fz, lattice=lattice)
        assert torch.allclose(out, f, atol=1e-7)

    @pytest.mark.parametrize("lattice,q,equilibrium", [
        ("D3Q19", 19, equilibrium3d),
        ("D3Q27", 27, equilibrium27),
    ])
    def test_preserves_shape(self, lattice, q, equilibrium) -> None:
        nz, ny, nx = 6, 6, 6
        rho = torch.ones((nz, ny, nx))
        zero = torch.zeros_like(rho)
        f = equilibrium(rho, zero, zero, zero)
        fx = torch.full((nz, ny, nx), 1e-4)
        out = ibm_apply_body_force_3d_common(f, fx, zero, zero, lattice=lattice)
        assert out.shape == f.shape
