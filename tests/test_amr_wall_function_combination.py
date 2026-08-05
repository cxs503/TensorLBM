"""TDD combination tests for AMR × wall-function common modules.

These tests verify that the common AMR module (amr_common) and the common
wall-function module (wall_function_common) can be combined in a single
simulation step, and that the wall×refinement combination gate provides a
clear admission path for this combination.
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
from tensorlbm.wall_function_common import (
    compute_u_tau,
    compute_y_plus,
    wall_function,
)
from tensorlbm.wall_refinement_combination_gate import (
    CollisionFamily,
    CombinationEvidence,
    GateStatus,
    GeometryOwnership,
    Lattice,
    PhysicsModel,
    RefinementType,
    WallRefinementCombination,
    WallTreatment,
    assess_wall_refinement_combination,
)
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.refinement import BoxRegion


def _make_equilibrium(lattice: str, nz: int = 6, ny: int = 8, nx: int = 10) -> torch.Tensor:
    rho = torch.ones(nz, ny, nx)
    ux = torch.full((nz, ny, nx), 0.1)
    uy = torch.zeros(nz, ny, nx)
    uz = torch.zeros(nz, ny, nx)
    if lattice == "D3Q19":
        return equilibrium3d(rho, ux, uy, uz)
    return equilibrium27(rho, ux, uy, uz)


def _make_channel_mask(nz: int = 6, ny: int = 8, nx: int = 10) -> torch.Tensor:
    mask = torch.zeros(nz, ny, nx, dtype=torch.bool)
    mask[0, :, :] = True
    mask[-1, :, :] = True
    return mask


def _q_for(lattice: str) -> int:
    return {"D3Q19": 19, "D3Q27": 27}[lattice]


# ---------------------------------------------------------------------------
# Gate: common_wf + AMR admission path
# ---------------------------------------------------------------------------

class TestGateCombinationPath:
    """The wall×refinement gate has a clear admission path for common modules."""

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_common_wf_plus_amr_allowed_with_evidence(self, lattice: str) -> None:
        decision = assess_wall_refinement_combination(
            WallRefinementCombination(
                lattice=Lattice(lattice),
                collision=CollisionFamily.MRT,
                wall_treatment=WallTreatment.COMMON_WALL_FUNCTION,
                refinement=RefinementType.DYNAMIC_AMR,
                geometry_ownership=GeometryOwnership.FINE_LEVEL,
                evidence=CombinationEvidence(
                    wall_distance_dy=0.5,
                    y_plus=50.0,
                    level_link_owner="fine",
                    wall_geometry_owner="fine",
                    interface_transfer_proof="FH proof",
                ),
            )
        )
        assert decision.status is GateStatus.ALLOWED

    @pytest.mark.parametrize("lattice", ["D3Q19", "D3Q27"])
    def test_common_wf_plus_amr_withheld_without_evidence(self, lattice: str) -> None:
        decision = assess_wall_refinement_combination(
            WallRefinementCombination(
                lattice=Lattice(lattice),
                collision=CollisionFamily.MRT,
                wall_treatment=WallTreatment.COMMON_WALL_FUNCTION,
                refinement=RefinementType.DYNAMIC_AMR,
                geometry_ownership=GeometryOwnership.FINE_LEVEL,
            )
        )
        assert decision.status is GateStatus.WITHHELD


# ---------------------------------------------------------------------------
# Functional combination: AMR refine + wall function
# ---------------------------------------------------------------------------

class TestAmrWallFunctionCombination:
    """AMR refine/coarsen and wall_function can be used together."""

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_refine_then_wall_function_then_coarsen(self, lattice: str) -> None:
        """End-to-end: refine → wall function → coarsen roundtrip."""
        q = _q_for(lattice)
        f = _make_equilibrium(lattice, nz=4, ny=6, nx=8)
        mask = _make_channel_mask(nz=4, ny=6, nx=8)

        # Step 1: Refine the coarse field
        f_fine = refine(f, lattice=lattice, tau_c=1.0, tau_f=0.75, ratio=2)
        assert f_fine.shape == (q, 8, 12, 16)

        # Step 2: Compute wall quantities on the fine grid
        mask_fine = _make_channel_mask(nz=8, ny=12, nx=16)
        if lattice == "D3Q19":
            rho, ux, uy, uz = macroscopic3d(f_fine)
        else:
            rho, ux, uy, uz = macroscopic27(f_fine)
        u_mag = torch.sqrt(ux**2 + uy**2 + uz**2)
        u_tau = compute_u_tau(u_mag, nu=0.02, y_val=0.5, wall_law="log")
        y_plus = compute_y_plus(u_tau, nu=0.02, y_val=0.5)

        # Step 3: Apply wall function on the fine grid
        f_fine_corrected = wall_function(
            f_fine, mask_fine, u_tau, y_plus,
            lattice=lattice, nu=0.02,
        )
        assert f_fine_corrected.shape == f_fine.shape
        assert torch.isfinite(f_fine_corrected).all()

        # Step 4: Coarsen back
        f_back = coarsen(f_fine_corrected, lattice=lattice, tau_f=0.75, tau_c=1.0, ratio=2)
        assert f_back.shape == f.shape
        assert torch.isfinite(f_back).all()

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_amr_patch_with_wall_function(self, lattice: str) -> None:
        """AMRPatch3D can hold a wall-function-corrected distribution."""
        q = _q_for(lattice)
        f = _make_equilibrium(lattice, nz=4, ny=6, nx=8)
        mask = _make_channel_mask(nz=4, ny=6, nx=8)

        # Refine and create a patch
        f_fine = refine(f, lattice=lattice, tau_c=1.0, tau_f=0.75, ratio=2)
        box = BoxRegion(0, 4, 0, 3, 0, 4)
        patch = AMRPatch3D(
            f=f_fine, box=box, ratio=2, level=1,
            parent_level=0, tau=0.75, lattice=lattice,
        )

        # Apply wall function on the patch
        mask_fine = _make_channel_mask(nz=8, ny=12, nx=16)
        if lattice == "D3Q19":
            rho, ux, uy, uz = macroscopic3d(patch.f)
        else:
            rho, ux, uy, uz = macroscopic27(patch.f)
        u_mag = torch.sqrt(ux**2 + uy**2 + uz**2)
        u_tau = compute_u_tau(u_mag, nu=0.02, y_val=0.5, wall_law="log")
        y_plus = compute_y_plus(u_tau, nu=0.02, y_val=0.5)

        patch.f = wall_function(
            patch.f, mask_fine, u_tau, y_plus,
            lattice=lattice, nu=0.02,
        )
        assert patch.f.shape == (q, 8, 12, 16)
        assert torch.isfinite(patch.f).all()

    @pytest.mark.parametrize("lattice", SUPPORTED_LATTICES)
    def test_halo_exchange_preserves_wall_correction(self, lattice: str) -> None:
        """Halo exchange after wall function should preserve interior correction."""
        q = _q_for(lattice)
        f = _make_equilibrium(lattice, nz=4, ny=6, nx=8)
        parent_f = _make_equilibrium(lattice, nz=8, ny=12, nx=16)

        # Create a patch with wall-corrected distribution
        mask = _make_channel_mask(nz=8, ny=12, nx=16)
        if lattice == "D3Q19":
            rho, ux, uy, uz = macroscopic3d(f)
        else:
            rho, ux, uy, uz = macroscopic27(f)
        u_mag = torch.sqrt(ux**2 + uy**2 + uz**2)
        u_tau = compute_u_tau(u_mag, nu=0.02, y_val=0.5, wall_law="log")
        y_plus = compute_y_plus(u_tau, nu=0.02, y_val=0.5)

        f_corrected = wall_function(
            f, mask[:4, :6, :8], u_tau[:4, :6, :8], y_plus[:4, :6, :8],
            lattice=lattice, nu=0.02,
        )
        f_fine = refine(f_corrected, lattice=lattice, tau_c=1.0, tau_f=0.75, ratio=2)

        box = BoxRegion(0, 4, 0, 3, 0, 4)
        r = 2
        interior_before = f_fine[:, r:-r, r:-r, r:-r].clone()

        halo_exchange(
            f_fine, parent_f, box=box, ratio=r,
            lattice=lattice, tau_p=1.0, tau_c=0.75,
        )
        # Interior should be preserved
        assert torch.equal(f_fine[:, r:-r, r:-r, r:-r], interior_before)
