"""TDD contract tests for the common AMR module (amr_common.py).

These tests specify the solver-agnostic, lattice-agnostic AMR interface
before the implementation is written.  The common module must:

* Provide AMRPatch3D as a public interface combinable with any collision/turbulence.
* Provide refine/coarsen operations that are not bound to a specific solver.
* Provide halo exchange between patches.
* Support D3Q19 and D3Q27.
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.amr_common import (
    AMRPatch3D,
    coarsen,
    halo_exchange,
    refine,
    SUPPORTED_LATTICES,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.refinement import BoxRegion


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_equilibrium_3d(lattice: str, nz: int = 4, ny: int = 6, nx: int = 8) -> torch.Tensor:
    """Create an equilibrium distribution for the given lattice."""
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.05)
    uy = torch.zeros(nz, ny, nx)
    uz = torch.zeros(nz, ny, nx)
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz)
    elif lattice == "D3Q27":
        return equilibrium27(rho, ux, uy, uz)
    raise ValueError(f"Unknown lattice: {lattice}")


def _q_for(lattice: str) -> int:
    return {"D3Q19": 19, "D3Q27": 27}[lattice]


# ---------------------------------------------------------------------------
# AMRPatch3D dataclass
# ---------------------------------------------------------------------------

class TestAMRPatch3D:
    """AMRPatch3D is a public data container combinable with any solver."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_patch_holds_distribution_and_metadata(self, lattice: str) -> None:
        q = _q_for(lattice)
        f = torch.rand(q, 8, 8, 8)
        box = BoxRegion(0, 4, 0, 4, 0, 4)
        patch = AMRPatch3D(f=f, box=box, ratio=2, level=1, parent_level=0, tau=0.55,
                           lattice=lattice)
        assert patch.f is f
        assert patch.box is box
        assert patch.ratio == 2
        assert patch.level == 1
        assert patch.parent_level == 0
        assert patch.tau == 0.55
        assert patch.lattice == lattice

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_patch_shape_properties(self, lattice: str) -> None:
        q = _q_for(lattice)
        f = torch.rand(q, 8, 10, 12)
        box = BoxRegion(0, 4, 0, 5, 0, 6)
        patch = AMRPatch3D(f=f, box=box, lattice=lattice)
        assert patch.nz == 8
        assert patch.ny == 10
        assert patch.nx == 12
        assert patch.cells == 8 * 10 * 12

    def test_patch_default_lattice_is_d3q19(self) -> None:
        f = torch.rand(19, 4, 4, 4)
        box = BoxRegion(0, 2, 0, 2, 0, 2)
        patch = AMRPatch3D(f=f, box=box)
        assert patch.lattice == "D3Q19"

    def test_patch_rejects_unsupported_lattice(self) -> None:
        f = torch.rand(19, 4, 4, 4)
        box = BoxRegion(0, 2, 0, 2, 0, 2)
        with pytest.raises(ValueError, match="Unsupported lattice"):
            AMRPatch3D(f=f, box=box, lattice="D2Q9")


# ---------------------------------------------------------------------------
# refine / coarsen operations (solver-agnostic)
# ---------------------------------------------------------------------------

class TestRefineCoarsen:
    """refine/coarsen are not bound to a specific solver."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_refine_doubles_resolution(self, lattice: str) -> None:
        q = _q_for(lattice)
        f = torch.rand(q, 4, 6, 8)
        f_fine = refine(f, lattice=lattice, tau_c=1.0, tau_f=0.75, ratio=2)
        assert f_fine.shape == (q, 8, 12, 16)

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_coarsen_halves_resolution(self, lattice: str) -> None:
        q = _q_for(lattice)
        f = torch.rand(q, 8, 12, 16)
        f_coarse = coarsen(f, lattice=lattice, tau_f=0.75, tau_c=1.0, ratio=2)
        assert f_coarse.shape == (q, 4, 6, 8)

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_refine_coarsen_roundtrip_preserves_mass(self, lattice: str) -> None:
        """refine then coarsen should approximately preserve total mass."""
        q = _q_for(lattice)
        f = _make_equilibrium_3d(lattice, nz=4, ny=6, nx=8)
        f_fine = refine(f, lattice=lattice, tau_c=1.0, tau_f=0.75, ratio=2)
        f_back = coarsen(f_fine, lattice=lattice, tau_f=0.75, tau_c=1.0, ratio=2)
        mass_orig = f.sum().item()
        mass_back = f_back.sum().item()
        assert mass_back == pytest.approx(mass_orig, rel=0.05)

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_refine_without_fh_uses_plain_interpolation(self, lattice: str) -> None:
        q = _q_for(lattice)
        f = torch.rand(q, 4, 4, 4)
        f_fine = refine(f, lattice=lattice, tau_c=1.0, tau_f=0.75, ratio=2, use_fh=False)
        assert f_fine.shape == (q, 8, 8, 8)

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_coarsen_without_fh_uses_plain_restriction(self, lattice: str) -> None:
        q = _q_for(lattice)
        f = torch.rand(q, 8, 8, 8)
        f_coarse = coarsen(f, lattice=lattice, tau_f=0.75, tau_c=1.0, ratio=2, use_fh=False)
        assert f_coarse.shape == (q, 4, 4, 4)

    def test_refine_rejects_unsupported_lattice(self) -> None:
        f = torch.rand(19, 4, 4, 4)
        with pytest.raises(ValueError, match="Unsupported lattice"):
            refine(f, lattice="D2Q9", tau_c=1.0, tau_f=0.75)

    def test_coarsen_rejects_unsupported_lattice(self) -> None:
        f = torch.rand(19, 8, 8, 8)
        with pytest.raises(ValueError, match="Unsupported lattice"):
            coarsen(f, lattice="D2Q9", tau_f=0.75, tau_c=1.0)


# ---------------------------------------------------------------------------
# Halo exchange between patches
# ---------------------------------------------------------------------------

class TestHaloExchange:
    """halo_exchange copies parent-level data into patch boundary cells."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_halo_exchange_preserves_interior(self, lattice: str) -> None:
        """Interior cells should be unchanged after halo exchange."""
        q = _q_for(lattice)
        parent_f = torch.rand(q, 8, 8, 8)
        patch_f = torch.rand(q, 8, 8, 8)
        box = BoxRegion(0, 4, 0, 4, 0, 4)
        r = 2
        interior_before = patch_f[:, r:-r, r:-r, r:-r].clone()
        halo_exchange(
            patch_f, parent_f, box=box, ratio=r,
            lattice=lattice, tau_p=1.0, tau_c=0.75,
        )
        assert torch.equal(patch_f[:, r:-r, r:-r, r:-r], interior_before)

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_halo_exchange_modifies_border(self, lattice: str) -> None:
        """Border cells should be overwritten by upsampled parent values."""
        q = _q_for(lattice)
        parent_f = torch.rand(q, 8, 8, 8)
        patch_f = torch.zeros(q, 8, 8, 8)
        box = BoxRegion(0, 4, 0, 4, 0, 4)
        r = 2
        halo_exchange(
            patch_f, parent_f, box=box, ratio=r,
            lattice=lattice, tau_p=1.0, tau_c=0.75,
        )
        # Border should now be non-zero (parent was random)
        border = torch.ones(8, 8, 8, dtype=torch.bool)
        border[r:-r, r:-r, r:-r] = False
        assert patch_f[:, border].abs().sum() > 0

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_halo_exchange_without_fh(self, lattice: str) -> None:
        """Halo exchange should also work with plain interpolation."""
        q = _q_for(lattice)
        parent_f = torch.rand(q, 8, 8, 8)
        patch_f = torch.zeros(q, 8, 8, 8)
        box = BoxRegion(0, 4, 0, 4, 0, 4)
        halo_exchange(
            patch_f, parent_f, box=box, ratio=2,
            lattice=lattice, tau_p=1.0, tau_c=0.75, use_fh=False,
        )
        # Should not crash and border should be modified
        r = 2
        border = torch.ones(8, 8, 8, dtype=torch.bool)
        border[r:-r, r:-r, r:-r] = False
        assert patch_f[:, border].abs().sum() > 0


# ---------------------------------------------------------------------------
# Solver-agnostic combination test
# ---------------------------------------------------------------------------

class TestSolverAgnosticCombination:
    """AMR operations work with arbitrary collide/stream/boundary callables."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_refine_works_with_identity_collision(self, lattice: str) -> None:
        """The common module does not call any specific collision operator."""
        q = _q_for(lattice)
        f = _make_equilibrium_3d(lattice)
        # Add perturbation
        f = f + 0.01 * torch.rand_like(f)
        f_fine = refine(f, lattice=lattice, tau_c=1.0, tau_f=0.75, ratio=2)
        # Simulate a collision step (identity)
        f_fine_post = f_fine  # identity collision
        f_coarse = coarsen(f_fine_post, lattice=lattice, tau_f=0.75, tau_c=1.0, ratio=2)
        assert f_coarse.shape == f.shape
        assert torch.isfinite(f_coarse).all()

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_amr_patch_can_be_created_from_any_solver_output(self, lattice: str) -> None:
        """AMRPatch3D can wrap output from any collision/streaming operator."""
        q = _q_for(lattice)
        f = torch.rand(q, 6, 8, 10)
        box = BoxRegion(0, 3, 0, 4, 0, 5)
        patch = AMRPatch3D(f=f, box=box, ratio=2, level=1, lattice=lattice)
        # The patch is just a data container — no solver binding
        assert patch.f.shape == (q, 6, 8, 10)
