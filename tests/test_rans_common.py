"""Contract tests for the RANS common collision module.

Verifies that k-epsilon, Spalart-Allmaras, and k-omega SST RANS models
delegate to the common BGK/MRT collision via the shared _nu_t_to_tau_eff
interface (same as Smagorinsky/WALE/Vreman), for D3Q19 and D3Q27 lattices.

Contract tests verify operator algebra (shape, finite, mass, momentum,
equilibrium identity), NOT turbulence physics correctness.

Hot-path invariants verified:
    - nu_t is a per-cell field (ndim == spatial dims), never a scalar
    - No .item() / .mean().item() / float(tensor) host syncs in collision
"""
from __future__ import annotations

import inspect

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.rans_common import (
    collide_rans_3d,
    collide_rans_bgk27,
    collide_rans_bgk3d,
    collide_rans_mrt27,
    collide_rans_mrt3d,
)
from tensorlbm.turbulence import _nu_t_to_tau_eff

DEVICE = torch.device("cpu")
TAU = 0.7  # > 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f3d19(nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium3d(rho, ux, uy, uz)


def _f3d27(nz: int = 4, ny: int = 6, nx: int = 8, u_mag: float = 0.04) -> torch.Tensor:
    rho = torch.rand((nz, ny, nx)) + 0.5
    ux = torch.rand_like(rho) * u_mag
    uy = torch.rand_like(rho) * u_mag
    uz = torch.rand_like(rho) * u_mag
    return equilibrium27(rho, ux, uy, uz)


def _nu_t_field_3d(nz: int = 4, ny: int = 6, nx: int = 8) -> torch.Tensor:
    """A non-trivial per-cell eddy viscosity field (not a scalar)."""
    return torch.rand((nz, ny, nx)) * 0.05


# ---------------------------------------------------------------------------
# Per-cell nu_t invariant
# ---------------------------------------------------------------------------

class TestNuTIsPerCell:
    """nu_t must be a per-cell field, never a scalar."""

    def test_nu_t_field_has_spatial_dims(self) -> None:
        nu_t = _nu_t_field_3d()
        assert nu_t.ndim == 3
        assert nu_t.shape == (4, 6, 8)

    def test_tau_eff_preserves_shape(self) -> None:
        nu_t = _nu_t_field_3d()
        tau_eff = _nu_t_to_tau_eff(TAU, nu_t)
        assert tau_eff.shape == nu_t.shape

    def test_tau_eff_exceeds_half(self) -> None:
        nu_t = torch.zeros(4, 6, 8)
        tau_eff = _nu_t_to_tau_eff(TAU, nu_t)
        assert (tau_eff >= 0.5001).all()

    def test_tau_eff_varies_spatially(self) -> None:
        """With non-uniform nu_t, tau_eff must vary across cells."""
        nu_t = torch.zeros(4, 6, 8)
        nu_t[0, 0, 0] = 0.1
        tau_eff = _nu_t_to_tau_eff(TAU, nu_t)
        assert tau_eff[0, 0, 0] > tau_eff[0, 0, 1]


# ---------------------------------------------------------------------------
# D3Q19 BGK
# ---------------------------------------------------------------------------

class TestRansBgk3d19:
    def test_shape(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        assert collide_rans_bgk3d(f, TAU, nu_t).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        assert torch.isfinite(collide_rans_bgk3d(f, TAU, nu_t)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        rho_in, _, _, _ = macroscopic3d(f)
        rho_out, _, _, _ = macroscopic3d(collide_rans_bgk3d(f, TAU, nu_t))
        assert torch.allclose(rho_out, rho_in, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        _, ux, uy, uz = macroscopic3d(f)
        _, uxo, uyo, uzo = macroscopic3d(collide_rans_bgk3d(f, TAU, nu_t))
        assert torch.allclose(uxo, ux, atol=1e-5)
        assert torch.allclose(uyo, uy, atol=1e-5)
        assert torch.allclose(uzo, uz, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium3d(rho, ux, uy, uz)
        nu_t = _nu_t_field_3d()
        assert torch.allclose(collide_rans_bgk3d(feq, TAU, nu_t), feq, atol=1e-5)

    def test_zero_nu_t_matches_plain_bgk(self) -> None:
        """With nu_t=0, RANS BGK == plain BGK with tau."""
        from tensorlbm.solver3d import collide_bgk3d

        f = _f3d19()
        nu_t = torch.zeros(f.shape[1:])
        rans_out = collide_rans_bgk3d(f, TAU, nu_t)
        bgk_out = collide_bgk3d(f, TAU)
        assert torch.allclose(rans_out, bgk_out, atol=1e-6)


# ---------------------------------------------------------------------------
# D3Q19 MRT
# ---------------------------------------------------------------------------

class TestRansMrt3d19:
    def test_shape(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        assert collide_rans_mrt3d(f, TAU, nu_t).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        assert torch.isfinite(collide_rans_mrt3d(f, TAU, nu_t)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        rho_in, _, _, _ = macroscopic3d(f)
        rho_out, _, _, _ = macroscopic3d(collide_rans_mrt3d(f, TAU, nu_t))
        assert torch.allclose(rho_out, rho_in, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        _, ux, uy, uz = macroscopic3d(f)
        _, uxo, uyo, uzo = macroscopic3d(collide_rans_mrt3d(f, TAU, nu_t))
        assert torch.allclose(uxo, ux, atol=1e-5)
        assert torch.allclose(uyo, uy, atol=1e-5)
        assert torch.allclose(uzo, uz, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium3d(rho, ux, uy, uz)
        nu_t = _nu_t_field_3d()
        assert torch.allclose(collide_rans_mrt3d(feq, TAU, nu_t), feq, atol=1e-5)


# ---------------------------------------------------------------------------
# D3Q27 BGK
# ---------------------------------------------------------------------------

class TestRansBgk3d27:
    def test_shape(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        assert collide_rans_bgk27(f, TAU, nu_t).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        assert torch.isfinite(collide_rans_bgk27(f, TAU, nu_t)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        rho_in, _, _, _ = macroscopic27(f)
        rho_out, _, _, _ = macroscopic27(collide_rans_bgk27(f, TAU, nu_t))
        assert torch.allclose(rho_out, rho_in, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        _, ux, uy, uz = macroscopic27(f)
        _, uxo, uyo, uzo = macroscopic27(collide_rans_bgk27(f, TAU, nu_t))
        assert torch.allclose(uxo, ux, atol=1e-5)
        assert torch.allclose(uyo, uy, atol=1e-5)
        assert torch.allclose(uzo, uz, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        nu_t = _nu_t_field_3d()
        assert torch.allclose(collide_rans_bgk27(feq, TAU, nu_t), feq, atol=1e-5)

    def test_zero_nu_t_matches_plain_bgk(self) -> None:
        from tensorlbm.d3q27 import collide_bgk27

        f = _f3d27()
        nu_t = torch.zeros(f.shape[1:])
        rans_out = collide_rans_bgk27(f, TAU, nu_t)
        bgk_out = collide_bgk27(f, TAU)
        assert torch.allclose(rans_out, bgk_out, atol=1e-6)


# ---------------------------------------------------------------------------
# D3Q27 MRT
# ---------------------------------------------------------------------------

class TestRansMrt3d27:
    def test_shape(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        assert collide_rans_mrt27(f, TAU, nu_t).shape == f.shape

    def test_finite(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        assert torch.isfinite(collide_rans_mrt27(f, TAU, nu_t)).all()

    def test_conserves_mass(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        rho_in, _, _, _ = macroscopic27(f)
        rho_out, _, _, _ = macroscopic27(collide_rans_mrt27(f, TAU, nu_t))
        assert torch.allclose(rho_out, rho_in, atol=1e-5)

    def test_conserves_momentum(self) -> None:
        f = _f3d27()
        nu_t = _nu_t_field_3d()
        _, ux, uy, uz = macroscopic27(f)
        _, uxo, uyo, uzo = macroscopic27(collide_rans_mrt27(f, TAU, nu_t))
        assert torch.allclose(uxo, ux, atol=1e-5)
        assert torch.allclose(uyo, uy, atol=1e-5)
        assert torch.allclose(uzo, uz, atol=1e-5)

    def test_equilibrium_is_identity(self) -> None:
        nz, ny, nx = 4, 6, 8
        rho = torch.ones((nz, ny, nx))
        ux = torch.full_like(rho, 0.03)
        uy = torch.full_like(rho, 0.01)
        uz = torch.full_like(rho, -0.01)
        feq = equilibrium27(rho, ux, uy, uz)
        nu_t = _nu_t_field_3d()
        assert torch.allclose(collide_rans_mrt27(feq, TAU, nu_t), feq, atol=1e-5)


# ---------------------------------------------------------------------------
# Unified dispatch: collide_rans_3d
# ---------------------------------------------------------------------------

class TestRansDispatch:
    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [
            ("D3Q19", "BGK"), ("D3Q19", "MRT"),
            ("D3Q27", "BGK"), ("D3Q27", "MRT"),
        ],
    )
    def test_dispatch_shape_and_finite(self, lattice: str, collision: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        nu_t = _nu_t_field_3d()
        out = collide_rans_3d(lattice, collision, f, tau=TAU, nu_t=nu_t)
        assert out.shape == f.shape
        assert torch.isfinite(out).all()

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [
            ("D3Q19", "BGK"), ("D3Q19", "MRT"),
            ("D3Q27", "BGK"), ("D3Q27", "MRT"),
        ],
    )
    def test_dispatch_conserves_mass(self, lattice: str, collision: str) -> None:
        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        nu_t = _nu_t_field_3d()
        out = collide_rans_3d(lattice, collision, f, tau=TAU, nu_t=nu_t)
        macro = macroscopic3d if lattice == "D3Q19" else macroscopic27
        rho_in, _, _, _ = macro(f)
        rho_out, _, _, _ = macro(out)
        assert torch.allclose(rho_out, rho_in, atol=1e-5)

    def test_dispatch_case_insensitive(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        out_lower = collide_rans_3d("d3q19", "bgk", f, tau=TAU, nu_t=nu_t)
        out_upper = collide_rans_3d("D3Q19", "BGK", f, tau=TAU, nu_t=nu_t)
        assert torch.allclose(out_lower, out_upper, atol=1e-7)

    def test_dispatch_rejects_unknown_lattice(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        with pytest.raises(ValueError, match="lattice"):
            collide_rans_3d("D2Q9", "BGK", f, tau=TAU, nu_t=nu_t)

    def test_dispatch_rejects_unknown_collision(self) -> None:
        f = _f3d19()
        nu_t = _nu_t_field_3d()
        with pytest.raises(ValueError, match="collision"):
            collide_rans_3d("D3Q19", "TRT", f, tau=TAU, nu_t=nu_t)

    def test_dispatch_rejects_wrong_population_size(self) -> None:
        f = _f3d19()  # 19 populations
        nu_t = _nu_t_field_3d()
        with pytest.raises(ValueError, match="27"):
            collide_rans_3d("D3Q27", "BGK", f, tau=TAU, nu_t=nu_t)


# ---------------------------------------------------------------------------
# Hot-path: no host sync in collision source
# ---------------------------------------------------------------------------

class TestNoHostSync:
    """Source-level check that collision functions contain no GPU→CPU syncs."""

    SYNC_PATTERNS = (".item()", ".mean().item()", "float(", "bool(tensor", "bool(f")

    @pytest.mark.parametrize(
        "func",
        [
            collide_rans_bgk3d,
            collide_rans_mrt3d,
            collide_rans_bgk27,
            collide_rans_mrt27,
            collide_rans_3d,
        ],
    )
    def test_no_sync_patterns_in_source(self, func) -> None:
        src = inspect.getsource(func)
        for pat in self.SYNC_PATTERNS:
            assert pat not in src, f"{func.__name__}: found '{pat}' in source"

    @pytest.mark.parametrize(
        "func",
        [
            collide_rans_bgk3d,
            collide_rans_mrt3d,
            collide_rans_bgk27,
            collide_rans_mrt27,
        ],
    )
    def test_no_mask_bool_allocation_in_source(self, func) -> None:
        """mask.bool() per-call allocation must not appear in collision."""
        src = inspect.getsource(func)
        assert ".bool()" not in src, f"{func.__name__}: found '.bool()' allocation"


# ---------------------------------------------------------------------------
# RANS solver integration: k-epsilon, SA, k-omega SST
# ---------------------------------------------------------------------------

class TestRansKeIntegration:
    """k-epsilon solver → collide_rans_3d → common collision."""

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [
            ("D3Q19", "BGK"), ("D3Q19", "MRT"),
            ("D3Q27", "BGK"), ("D3Q27", "MRT"),
        ],
    )
    def test_ke_collision_shape_finite_mass(self, lattice: str, collision: str) -> None:
        from tensorlbm.rans_ke import KESolver, collide_rans_ke

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        macro = macroscopic3d if lattice == "D3Q19" else macroscopic27
        _, ux, uy, uz = macro(f)
        solver = KESolver(nu=0.01)
        solver.initialize(ux, uy, uz)
        out = collide_rans_ke(f, TAU, solver, lattice=lattice, collision=collision)
        assert out.shape == f.shape
        assert torch.isfinite(out).all()
        rho_in, _, _, _ = macro(f)
        rho_out, _, _, _ = macro(out)
        assert torch.allclose(rho_out, rho_in, atol=1e-4)

    def test_ke_nu_t_is_per_cell_field(self) -> None:
        from tensorlbm.rans_ke import KESolver

        f = _f3d19()
        _, ux, uy, uz = macroscopic3d(f)
        solver = KESolver(nu=0.01)
        solver.initialize(ux, uy, uz)
        nu_t = solver.compute_nu_t()
        assert nu_t.ndim == 3  # per-cell, not scalar
        assert nu_t.shape == ux.shape


class TestRansSaIntegration:
    """Spalart-Allmaras solver → collide_rans_3d → common collision."""

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [
            ("D3Q19", "BGK"), ("D3Q19", "MRT"),
            ("D3Q27", "BGK"), ("D3Q27", "MRT"),
        ],
    )
    def test_sa_collision_shape_finite_mass(self, lattice: str, collision: str) -> None:
        from tensorlbm.rans_ke import SASolver, collide_rans_sa

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        macro = macroscopic3d if lattice == "D3Q19" else macroscopic27
        _, ux, uy, uz = macro(f)
        nz, ny, nx = ux.shape
        wall_dist = torch.full((nz, ny, nx), 5.0)
        solver = SASolver(nu=0.01)
        solver.initialize(ux, uy, uz)
        out = collide_rans_sa(f, TAU, solver, wall_dist, lattice=lattice, collision=collision)
        assert out.shape == f.shape
        assert torch.isfinite(out).all()
        rho_in, _, _, _ = macro(f)
        rho_out, _, _, _ = macro(out)
        assert torch.allclose(rho_out, rho_in, atol=1e-4)

    def test_sa_no_scalar_averaging_in_source(self) -> None:
        """collide_rans_sa must not contain .mean().item() scalar averaging."""
        from tensorlbm.rans_ke import collide_rans_sa

        src = inspect.getsource(collide_rans_sa)
        assert ".mean().item()" not in src
        assert "nu_eff = nu_lam + nu_t.mean()" not in src

    def test_sa_nu_t_is_per_cell_field(self) -> None:
        from tensorlbm.rans_ke import SASolver

        f = _f3d19()
        _, ux, uy, uz = macroscopic3d(f)
        solver = SASolver(nu=0.01)
        solver.initialize(ux, uy, uz)
        nu_t = solver.compute_nu_t()
        assert nu_t.ndim == 3
        assert nu_t.shape == ux.shape


class TestKOmegaSstIntegration:
    """k-omega SST solver → collide_rans_3d → common collision."""

    @pytest.mark.parametrize(
        ("lattice", "collision"),
        [
            ("D3Q19", "BGK"), ("D3Q19", "MRT"),
            ("D3Q27", "BGK"), ("D3Q27", "MRT"),
        ],
    )
    def test_sst_collision_shape_finite_mass(self, lattice: str, collision: str) -> None:
        from tensorlbm.rans_ke import KOmegaSSTSolver, collide_rans_komega_sst

        f = _f3d19() if lattice == "D3Q19" else _f3d27()
        macro = macroscopic3d if lattice == "D3Q19" else macroscopic27
        _, ux, uy, uz = macro(f)
        nz, ny, nx = ux.shape
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        wall_dist = torch.full((nz, ny, nx), 5.0)
        solver = KOmegaSSTSolver(mask=mask, nu_lbm=0.01)
        out = collide_rans_komega_sst(
            f, solver, ux, uy, uz, tau=TAU, wall_dist=wall_dist,
            lattice=lattice, collision=collision,
        )
        assert out.shape == f.shape
        assert torch.isfinite(out).all()
        rho_in, _, _, _ = macro(f)
        rho_out, _, _, _ = macro(out)
        assert torch.allclose(rho_out, rho_in, atol=1e-4)

    def test_sst_nu_t_is_per_cell_field(self) -> None:
        from tensorlbm.rans_ke import KOmegaSSTSolver

        nz, ny, nx = 4, 6, 8
        mask = torch.zeros((nz, ny, nx), dtype=torch.bool)
        solver = KOmegaSSTSolver(mask=mask, nu_lbm=0.01)
        nu_t = solver.compute_nu_t()
        assert nu_t.ndim == 3
        assert nu_t.shape == (nz, ny, nx)
