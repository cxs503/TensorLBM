"""TDD contract tests for the common wall-function module (wall_function_common.py).

These tests specify the solver-agnostic, lattice-agnostic wall-function
interface before the implementation is written.  The common module must:

* Provide wall_function(f, mask, u_tau, y_plus, ...) → f_corrected.
* Be combinable with any collision/turbulence (takes pre-computed u_tau, y_plus).
* Not be bound to a specific solver.
* Support D3Q19 and D3Q27.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.wall_function_common import (
    SUPPORTED_LATTICES,
    compute_u_tau,
    compute_y_plus,
    wall_function,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equilibrium(lattice: str, nz: int = 6, ny: int = 8, nx: int = 10) -> torch.Tensor:
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.1)
    uy = torch.zeros(nz, ny, nx)
    uz = torch.zeros(nz, ny, nx)
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz)
    elif lattice == "D3Q27":
        return equilibrium27(rho, ux, uy, uz)
    raise ValueError(f"Unknown lattice: {lattice}")


def _make_channel_mask(nz: int = 6, ny: int = 8, nx: int = 10) -> torch.Tensor:
    """Solid walls at z=0 and z=nz-1."""
    mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    return mask


def _q_for(lattice: str) -> int:
    return {"D3Q19": 19, "D3Q27": 27}[lattice]


# ---------------------------------------------------------------------------
# compute_u_tau / compute_y_plus
# ---------------------------------------------------------------------------

class TestComputeUTauYPlus:
    """Wall-quantity computation helpers are lattice-agnostic."""

    def test_compute_u_tau_positive(self) -> None:
        u_mag = torch.full((6, 8, 10), 0.1)
        u_tau = compute_u_tau(u_mag, nu=0.02, y_val=0.5, wall_law="log")
        assert u_tau.shape == u_mag.shape
        assert (u_tau > 0).all()

    def test_compute_y_plus_positive(self) -> None:
        u_tau = torch.full((6, 8, 10), 0.01)
        y_plus = compute_y_plus(u_tau, nu=0.02, y_val=0.5)
        assert y_plus.shape == u_tau.shape
        assert (y_plus > 0).all()

    def test_compute_u_tau_reichardt(self) -> None:
        u_mag = torch.full((6, 8, 10), 0.1)
        u_tau = compute_u_tau(u_mag, nu=0.02, y_val=0.5, wall_law="reichardt")
        assert u_tau.shape == u_mag.shape
        assert (u_tau > 0).all()

    def test_compute_u_tau_rejects_unknown_law(self) -> None:
        u_mag = torch.full((6, 8, 10), 0.1)
        with pytest.raises(ValueError, match="Unknown wall_law"):
            compute_u_tau(u_mag, nu=0.02, y_val=0.5, wall_law="bogus")


# ---------------------------------------------------------------------------
# wall_function — shape and type contract
# ---------------------------------------------------------------------------

class TestWallFunctionShape:
    """wall_function returns f_corrected with the same shape as f."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_returns_same_shape(self, lattice: str) -> None:
        q = _q_for(lattice)
        f = _make_equilibrium(lattice)
        mask = _make_channel_mask()
        u_tau = torch.full_like(mask, 0.01, dtype=torch.float32)
        y_plus = torch.full_like(mask, 50.0, dtype=torch.float32)
        f_corrected = wall_function(f, mask, u_tau, y_plus, lattice=lattice, nu=0.02)
        assert f_corrected.shape == f.shape

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_returns_finite(self, lattice: str) -> None:
        f = _make_equilibrium(lattice)
        mask = _make_channel_mask()
        u_tau = torch.full_like(mask, 0.01, dtype=torch.float32)
        y_plus = torch.full_like(mask, 50.0, dtype=torch.float32)
        f_corrected = wall_function(f, mask, u_tau, y_plus, lattice=lattice, nu=0.02)
        assert torch.isfinite(f_corrected).all()

    def test_rejects_unsupported_lattice(self) -> None:
        f = torch.rand(19, 6, 8, 10)
        mask = _make_channel_mask()
        u_tau = torch.full_like(mask, 0.01, dtype=torch.float32)
        y_plus = torch.full_like(mask, 50.0, dtype=torch.float32)
        with pytest.raises(ValueError, match="Unsupported lattice"):
            wall_function(f, mask, u_tau, y_plus, lattice="D2Q9", nu=0.02)


# ---------------------------------------------------------------------------
# wall_function — physics contract
# ---------------------------------------------------------------------------

class TestWallFunctionPhysics:
    """wall_function applies a body force on near-wall cells only."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_no_correction_far_from_wall(self, lattice: str) -> None:
        """With u_tau=0 everywhere, f should be unchanged."""
        f = _make_equilibrium(lattice)
        mask = _make_channel_mask()
        u_tau = torch.zeros_like(mask, dtype=torch.float32)
        y_plus = torch.zeros_like(mask, dtype=torch.float32)
        f_corrected = wall_function(f, mask, u_tau, y_plus, lattice=lattice, nu=0.02)
        assert torch.allclose(f_corrected, f, atol=1e-7)

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_correction_only_near_wall(self, lattice: str) -> None:
        """Interior cells (far from wall) should be unchanged."""
        f = _make_equilibrium(lattice)
        mask = _make_channel_mask()
        u_tau = torch.full_like(mask, 0.01, dtype=torch.float32)
        y_plus = torch.full_like(mask, 50.0, dtype=torch.float32)
        f_corrected = wall_function(f, mask, u_tau, y_plus, lattice=lattice, nu=0.02)
        # Interior cells (not near wall) should be unchanged
        interior = slice(2, -2)
        assert torch.allclose(
            f_corrected[:, interior, :, :],
            f[:, interior, :, :],
            atol=1e-7,
        )

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_mass_is_conserved_or_decreased(self, lattice: str) -> None:
        """Wall function should not create mass (it's a dissipative force)."""
        f = _make_equilibrium(lattice)
        mask = _make_channel_mask()
        u_tau = torch.full_like(mask, 0.01, dtype=torch.float32)
        y_plus = torch.full_like(mask, 50.0, dtype=torch.float32)
        f_corrected = wall_function(f, mask, u_tau, y_plus, lattice=lattice, nu=0.02)
        # Mass should be approximately conserved (body force redistributes momentum)
        mass_before = f.sum().item()
        mass_after = f_corrected.sum().item()
        assert mass_after == pytest.approx(mass_before, rel=0.1)


# ---------------------------------------------------------------------------
# Solver-agnostic combination test
# ---------------------------------------------------------------------------

class TestSolverAgnosticCombination:
    """wall_function can be combined with any collision/turbulence operator."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_wall_function_after_identity_collision(self, lattice: str) -> None:
        """The common module does not call any specific collision operator."""
        f = _make_equilibrium(lattice)
        mask = _make_channel_mask()
        u_tau = torch.full_like(mask, 0.01, dtype=torch.float32)
        y_plus = torch.full_like(mask, 50.0, dtype=torch.float32)
        # Simulate: collide (identity) → wall function
        f_post = f  # identity collision
        f_corrected = wall_function(f_post, mask, u_tau, y_plus, lattice=lattice, nu=0.02)
        assert f_corrected.shape == f.shape
        assert torch.isfinite(f_corrected).all()

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_wall_function_with_precomputed_u_tau_from_any_model(self, lattice: str) -> None:
        """u_tau can come from any turbulence model (RANS, LES, etc.)."""
        f = _make_equilibrium(lattice)
        mask = _make_channel_mask()
        # Simulate u_tau from a turbulence model (just a field)
        u_tau = torch.rand_like(mask, dtype=torch.float32) * 0.02
        y_plus = u_tau * 0.5 / 0.02
        f_corrected = wall_function(f, mask, u_tau, y_plus, lattice=lattice, nu=0.02)
        assert f_corrected.shape == f.shape
        assert torch.isfinite(f_corrected).all()
