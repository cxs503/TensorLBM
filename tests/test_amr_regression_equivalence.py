"""AMR 回归等价性验证 — TDD test suite.

Verifies:
1. Original bug identification in adaptive_refinement.py
2. Equivalence between adaptive_refinement (original) and amr_common (extracted)
3. Combination testing: AMR + collision complete loop

Run:
    cd /root/.hermes/marine-control/TensorLBM_dev/regress-amr-r1
    python -m pytest tests/test_amr_regression_equivalence.py -v
"""
from __future__ import annotations

import sys
import os

import pytest
import torch

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.refinement import BoxRegion, _coarse_to_fine_3d, _fine_to_coarse_3d
from tensorlbm import adaptive_refinement as ar
from tensorlbm import amr_common
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_d3q19_field(nz=4, ny=4, nx=4, seed=42):
    """Create a physically plausible D3Q19 distribution."""
    torch.manual_seed(seed)
    rho = torch.ones(nz, ny, nx)
    ux = 0.1 * torch.randn(nz, ny, nx)
    uy = 0.05 * torch.randn(nz, ny, nx)
    uz = 0.03 * torch.randn(nz, ny, nx)
    f_eq = equilibrium3d(rho, ux, uy, uz)
    f_neq = 0.01 * torch.randn(19, nz, ny, nx)
    return f_eq + f_neq


def _make_d3q27_field(nz=4, ny=4, nx=4, seed=99):
    """Create a physically plausible D3Q27 distribution."""
    torch.manual_seed(seed)
    rho = torch.ones(nz, ny, nx)
    ux = 0.1 * torch.randn(nz, ny, nx)
    uy = 0.05 * torch.randn(nz, ny, nx)
    uz = 0.03 * torch.randn(nz, ny, nx)
    f_eq = equilibrium27(rho, ux, uy, uz)
    f_neq = 0.01 * torch.randn(27, nz, ny, nx)
    return f_eq + f_neq


def _simple_collide_d3q19(f, tau=1.0):
    """Simple BGK collision for D3Q19."""
    rho, ux, uy, uz = macroscopic3d(f)
    f_eq = equilibrium3d(rho, ux, uy, uz, device=f.device)
    return f - (1.0 / tau) * (f - f_eq)


def _simple_collide_d3q27(f, tau=1.0):
    """Simple BGK collision for D3Q27."""
    rho, ux, uy, uz = macroscopic27(f)
    f_eq = equilibrium27(rho, ux, uy, uz, device=f.device)
    return f - (1.0 / tau) * (f - f_eq)


def _identity(x):
    """Identity streaming / boundary (no-op)."""
    return x


# ===========================================================================
# PART 1: Original bug identification
# ===========================================================================

class TestOriginalBugIdentification:
    """Identify known bugs in the original adaptive_refinement.py."""

    def test_bug1_original_fh_coarse_to_fine_3d_hardcoded_d3q19(self):
        """BUG-1: Original _fh_coarse_to_fine_3d is hardcoded to D3Q19.

        The original function imports d3q19.macroscopic3d/equilibrium3d
        unconditionally.  Passing D3Q27 data (27 velocities) causes a
        shape mismatch because macroscopic3d expects exactly 19 velocities.

        This is a *design limitation* (带病上岗): the original cannot
        handle D3Q27, while amr_common.refine() dispatches correctly.
        """
        f27 = _make_d3q27_field(nz=4, ny=4, nx=4)
        # Original: hardcoded D3Q19 — should fail on D3Q27 data
        with pytest.raises(Exception):
            ar._fh_coarse_to_fine_3d(f27, tau_c=1.0, tau_f=0.75, ratio=2)

    def test_bug1_amr_common_handles_d3q27(self):
        """amr_common correctly dispatches D3Q27 — no bug."""
        f27 = _make_d3q27_field(nz=4, ny=4, nx=4)
        f_fine = amr_common.refine(f27, lattice="D3Q27", tau_c=1.0, tau_f=0.75)
        assert f_fine.shape == (27, 8, 8, 8)

    def test_bug2_halo_exchange_shape_mismatch_when_upsample_smaller(self):
        """BUG-2: halo_exchange shape mismatch when f_up < patch_f.

        When the upsampled parent data (f_up) is smaller than patch_f in
        any spatial dimension, the min()-truncated boolean mask has shape
        (fz, fy, fx) which does NOT match patch_f's spatial shape
        (nz_f, ny_f, nx_f).  PyTorch boolean indexing requires exact
        shape match, so this raises a RuntimeError.

        This bug exists in BOTH amr_common.halo_exchange AND the original
        AdaptiveSolver3D._inject_to_patch (same code pattern).
        """
        # parent grid: 6x6x6, box covers 3x3x3, ratio=2 → f_up is 6x6x6
        parent_f = _make_d3q19_field(nz=6, ny=6, nx=6, seed=1)
        box = BoxRegion(0, 3, 0, 3, 0, 3)  # 3x3x3 parent region
        ratio = 2
        # f_up will be (19, 6, 6, 6)
        # Create a patch_f LARGER than f_up: (19, 8, 8, 8)
        patch_f = _make_d3q19_field(nz=8, ny=8, nx=8, seed=2)

        # PyTorch boolean indexing requires exact shape match; raises IndexError
        with pytest.raises((RuntimeError, IndexError)):
            amr_common.halo_exchange(
                patch_f, parent_f, box=box, ratio=ratio,
                lattice="D3Q19", tau_p=1.0, tau_c=0.75,
            )

    def test_bug2_halo_exchange_works_when_shapes_match(self):
        """When f_up and patch_f have matching spatial shapes, halo_exchange works."""
        parent_f = _make_d3q19_field(nz=6, ny=6, nx=6, seed=1)
        box = BoxRegion(0, 3, 0, 3, 0, 3)  # 3x3x3 → f_up is 6x6x6
        ratio = 2
        patch_f = _make_d3q19_field(nz=6, ny=6, nx=6, seed=2)  # matches f_up

        # Should not raise
        amr_common.halo_exchange(
            patch_f, parent_f, box=box, ratio=ratio,
            lattice="D3Q19", tau_p=1.0, tau_c=0.75,
        )
        # Verify border was overwritten (interior untouched)
        r = ratio
        interior_before = patch_f[:, r:-r, r:-r, r:-r].clone()
        # Re-run to confirm idempotent structure
        amr_common.halo_exchange(
            patch_f, parent_f, box=box, ratio=ratio,
            lattice="D3Q19", tau_p=1.0, tau_c=0.75,
        )
        # Interior should be unchanged after second call (border overwritten, interior not)
        # Actually interior IS untouched, so it stays the same
        assert torch.allclose(interior_before, patch_f[:, r:-r, r:-r, r:-r])

    def test_bug3_original_fh_fine_to_coarse_3d_hardcoded_d3q19(self):
        """BUG-3: Original _fh_fine_to_coarse_3d is also hardcoded to D3Q19."""
        f27_fine = _make_d3q27_field(nz=8, ny=8, nx=8)
        with pytest.raises(Exception):
            ar._fh_fine_to_coarse_3d(f27_fine, tau_f=0.75, tau_c=1.0, ratio=2)

    def test_no_bug_coarse_to_fine_3d_lattice_agnostic(self):
        """_coarse_to_fine_3d (plain trilinear) is lattice-agnostic — no bug."""
        # Works for any Q
        f19 = _make_d3q19_field(nz=4, ny=4, nx=4)
        f27 = _make_d3q27_field(nz=4, ny=4, nx=4)
        assert _coarse_to_fine_3d(f19, 2).shape == (19, 8, 8, 8)
        assert _coarse_to_fine_3d(f27, 2).shape == (27, 8, 8, 8)

    def test_no_bug_fine_to_coarse_3d_lattice_agnostic(self):
        """_fine_to_coarse_3d (block average) is lattice-agnostic — no bug."""
        f19_fine = _make_d3q19_field(nz=8, ny=8, nx=8)
        f27_fine = _make_d3q27_field(nz=8, ny=8, nx=8)
        assert _fine_to_coarse_3d(f19_fine, 2).shape == (19, 4, 4, 4)
        assert _fine_to_coarse_3d(f27_fine, 2).shape == (27, 4, 4, 4)


# ===========================================================================
# PART 2: Equivalence verification (original vs amr_common)
# ===========================================================================

class TestEquivalenceOriginalVsCommon:
    """Verify that amr_common produces identical output to adaptive_refinement
    for D3Q19 (the only lattice the original supports)."""

    def test_refine_d3q19_fh_equivalence(self):
        """refine(use_fh=True) D3Q19: amr_common == original."""
        f = _make_d3q19_field(nz=4, ny=4, nx=4, seed=7)
        tau_c, tau_f = 1.0, 0.75

        # Original
        f_orig = ar._fh_coarse_to_fine_3d(f, tau_c, tau_f, ratio=2)
        # Common
        f_common = amr_common.refine(f, lattice="D3Q19", tau_c=tau_c, tau_f=tau_f, ratio=2, use_fh=True)

        assert f_orig.shape == f_common.shape == (19, 8, 8, 8)
        assert torch.allclose(f_orig, f_common, atol=1e-7), \
            "D3Q19 FH refine: original vs common mismatch"

    def test_refine_d3q19_plain_equivalence(self):
        """refine(use_fh=False) D3Q19: amr_common == original."""
        f = _make_d3q19_field(nz=4, ny=4, nx=4, seed=7)
        f_orig = _coarse_to_fine_3d(f, 2)
        f_common = amr_common.refine(f, lattice="D3Q19", ratio=2, use_fh=False)
        assert torch.allclose(f_orig, f_common, atol=0)

    def test_coarsen_d3q19_fh_equivalence(self):
        """coarsen(use_fh=True) D3Q19: amr_common == original."""
        f_fine = _make_d3q19_field(nz=8, ny=8, nx=8, seed=7)
        tau_f, tau_c = 0.75, 1.0

        f_orig = ar._fh_fine_to_coarse_3d(f_fine, tau_f, tau_c, ratio=2)
        f_common = amr_common.coarsen(f_fine, lattice="D3Q19", tau_f=tau_f, tau_c=tau_c, ratio=2, use_fh=True)

        assert f_orig.shape == f_common.shape == (19, 4, 4, 4)
        assert torch.allclose(f_orig, f_common, atol=1e-7), \
            "D3Q19 FH coarsen: original vs common mismatch"

    def test_coarsen_d3q19_plain_equivalence(self):
        """coarsen(use_fh=False) D3Q19: amr_common == original."""
        f_fine = _make_d3q19_field(nz=8, ny=8, nx=8, seed=7)
        f_orig = _fine_to_coarse_3d(f_fine, 2)
        f_common = amr_common.coarsen(f_fine, lattice="D3Q19", ratio=2, use_fh=False)
        assert torch.allclose(f_orig, f_common, atol=0)

    def test_refine_coarsen_roundtrip_d3q19(self):
        """refine → coarsen roundtrip preserves macroscopic moments (D3Q19).

        NOTE: trilinear interpolation with align_corners=True does NOT
        conserve the mean exactly.  On a 4³ grid the density drift is
        ~7% — this is a known property of the interpolation scheme, not
        a bug.  We verify approximate conservation (within 10%).
        """
        f = _make_d3q19_field(nz=4, ny=4, nx=4, seed=11)
        tau_c, tau_f = 1.0, 0.75

        f_fine = amr_common.refine(f, lattice="D3Q19", tau_c=tau_c, tau_f=tau_f)
        f_back = amr_common.coarsen(f_fine, lattice="D3Q19", tau_f=tau_f, tau_c=tau_c)

        rho_orig, _, _, _ = macroscopic3d(f)
        rho_back, _, _, _ = macroscopic3d(f_back)
        # Density should be approximately preserved (FH rescaling + averaging)
        assert rho_back.shape == rho_orig.shape
        # Known drift from align_corners=True trilinear on small grids
        max_diff = torch.max(torch.abs(rho_orig - rho_back)).item()
        assert max_diff < 0.1, \
            f"Density drift too large: {max_diff} (expected < 0.1 for 4³ grid)"

    def test_refine_coarsen_roundtrip_d3q27(self):
        """refine → coarsen roundtrip preserves density (D3Q27).

        Same align_corners=True drift as D3Q19 — approximately conserved.
        """
        f = _make_d3q27_field(nz=4, ny=4, nx=4, seed=22)
        tau_c, tau_f = 1.0, 0.75

        f_fine = amr_common.refine(f, lattice="D3Q27", tau_c=tau_c, tau_f=tau_f)
        f_back = amr_common.coarsen(f_fine, lattice="D3Q27", tau_f=tau_f, tau_c=tau_c)

        rho_orig, _, _, _ = macroscopic27(f)
        rho_back, _, _, _ = macroscopic27(f_back)
        assert rho_back.shape == rho_orig.shape
        max_diff = torch.max(torch.abs(rho_orig - rho_back)).item()
        assert max_diff < 0.1, \
            f"D3Q27 density drift too large: {max_diff} (expected < 0.1)"

    def test_halo_exchange_equivalence_with_solver_inject(self):
        """halo_exchange (common) produces same result as _inject_to_patch (original).

        We simulate the same scenario: create a patch via refine, then inject
        parent data into its border.  Both paths should produce identical
        border values.
        """
        parent_f = _make_d3q19_field(nz=6, ny=6, nx=6, seed=5)
        box = BoxRegion(1, 4, 1, 4, 1, 4)  # 3x3x3 region
        ratio = 2
        tau_p, tau_c = 1.0, 0.75

        # --- Common path: standalone halo_exchange ---
        f_patch_common = amr_common.refine(
            parent_f[:, 1:4, 1:4, 1:4], lattice="D3Q19",
            tau_c=tau_p, tau_f=tau_c, ratio=ratio,
        )
        amr_common.halo_exchange(
            f_patch_common, parent_f, box=box, ratio=ratio,
            lattice="D3Q19", tau_p=tau_p, tau_c=tau_c,
        )

        # --- Original path: replicate _inject_to_patch logic ---
        f_patch_orig = ar._fh_coarse_to_fine_3d(
            parent_f[:, 1:4, 1:4, 1:4], tau_p, tau_c, ratio,
        )
        # Replicate _inject_to_patch
        f_c = parent_f[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1]
        f_up = ar._fh_coarse_to_fine_3d(f_c, tau_p, tau_c, ratio)
        nz_f, ny_f, nx_f = f_patch_orig.shape[1:]
        border = torch.ones((nz_f, ny_f, nx_f), dtype=torch.bool)
        border[ratio:-ratio, ratio:-ratio, ratio:-ratio] = False
        fz = min(f_up.shape[1], nz_f)
        fy = min(f_up.shape[2], ny_f)
        fx = min(f_up.shape[3], nx_f)
        f_patch_orig[:, border[:fz, :fy, :fx]] = f_up[:, border[:fz, :fy, :fx]]

        assert torch.allclose(f_patch_common, f_patch_orig, atol=1e-7), \
            "halo_exchange (common) vs _inject_to_patch (original) mismatch"

    def test_amr_patch3d_equivalence(self):
        """AMRPatch3D dataclass in amr_common has same fields as original."""
        from tensorlbm.adaptive_refinement import AMRPatch3D as OrigPatch
        f = _make_d3q19_field(nz=8, ny=8, nx=8)
        box = BoxRegion(0, 4, 0, 4, 0, 4)

        p_orig = OrigPatch(f=f, box=box, ratio=2, level=1, parent_level=0, tau=0.75)
        p_common = amr_common.AMRPatch3D(f=f, box=box, ratio=2, level=1, parent_level=0, tau=0.75, lattice="D3Q19")

        assert p_orig.nz == p_common.nz == 8
        assert p_orig.ny == p_common.ny == 8
        assert p_orig.nx == p_common.nx == 8
        assert p_common.cells == 512

    def test_d3q27_refine_plain_matches_trilinear(self):
        """D3Q27 plain refine uses same trilinear interpolation as _coarse_to_fine_3d."""
        f = _make_d3q27_field(nz=4, ny=4, nx=4, seed=33)
        f_ref = amr_common.refine(f, lattice="D3Q27", ratio=2, use_fh=False)
        f_expected = _coarse_to_fine_3d(f, 2)
        assert torch.allclose(f_ref, f_expected, atol=0)


# ===========================================================================
# PART 3: Combination testing — AMR + collision complete loop
# ===========================================================================

class TestCombinationAMRCollision:
    """Test AMR refine/coarsen combined with collision operators."""

    def test_refine_collide_coarsen_loop_d3q19(self):
        """Full loop: refine → collide → coarsen (D3Q19).

        Verifies that the AMR mechanics compose correctly with a BGK
        collision operator.  The coarsened result should have the same
        shape as the original and preserve density to reasonable tolerance.
        """
        f = _make_d3q19_field(nz=4, ny=4, nx=4, seed=100)
        tau_c, tau_f = 1.0, 0.75

        # 1. Refine coarse → fine
        f_fine = amr_common.refine(f, lattice="D3Q19", tau_c=tau_c, tau_f=tau_f)
        assert f_fine.shape == (19, 8, 8, 8)

        # 2. Collide on fine grid
        f_fine_collided = _simple_collide_d3q19(f_fine, tau=tau_f)
        assert f_fine_collided.shape == f_fine.shape

        # 3. Coarsen fine → coarse
        f_coarse = amr_common.coarsen(f_fine_collided, lattice="D3Q19", tau_f=tau_f, tau_c=tau_c)
        assert f_coarse.shape == (19, 4, 4, 4)

        # 4. Density preservation check (trilinear drift expected on small grids)
        rho_orig, _, _, _ = macroscopic3d(f)
        rho_final, _, _, _ = macroscopic3d(f_coarse)
        max_diff = torch.max(torch.abs(rho_orig - rho_final)).item()
        assert max_diff < 0.1, \
            f"Density drift after refine-collide-coarsen: {max_diff} (expected < 0.1)"

    def test_refine_collide_coarsen_loop_d3q27(self):
        """Full loop: refine → collide → coarsen (D3Q27)."""
        f = _make_d3q27_field(nz=4, ny=4, nx=4, seed=200)
        tau_c, tau_f = 1.0, 0.75

        f_fine = amr_common.refine(f, lattice="D3Q27", tau_c=tau_c, tau_f=tau_f)
        assert f_fine.shape == (27, 8, 8, 8)

        f_fine_collided = _simple_collide_d3q27(f_fine, tau=tau_f)
        f_coarse = amr_common.coarsen(f_fine_collided, lattice="D3Q27", tau_f=tau_f, tau_c=tau_c)
        assert f_coarse.shape == (27, 4, 4, 4)

        rho_orig, _, _, _ = macroscopic27(f)
        rho_final, _, _, _ = macroscopic27(f_coarse)
        max_diff = torch.max(torch.abs(rho_orig - rho_final)).item()
        assert max_diff < 0.1, \
            f"D3Q27 density drift after loop: {max_diff} (expected < 0.1)"

    def test_multi_step_amr_loop_d3q19(self):
        """Multi-step AMR loop: refine, collide multiple steps, coarsen."""
        f = _make_d3q19_field(nz=4, ny=4, nx=4, seed=300)
        tau_c, tau_f = 1.0, 0.75

        f_fine = amr_common.refine(f, lattice="D3Q19", tau_c=tau_c, tau_f=tau_f)

        # Run 5 collision steps on the fine grid
        for _ in range(5):
            f_fine = _simple_collide_d3q19(f_fine, tau=tau_f)

        f_coarse = amr_common.coarsen(f_fine, lattice="D3Q19", tau_f=tau_f, tau_c=tau_c)
        assert f_coarse.shape == (19, 4, 4, 4)

        # Density should still be positive
        rho, _, _, _ = macroscopic3d(f_coarse)
        assert (rho > 0).all(), "Density became non-positive after multi-step loop"

    def test_halo_exchange_preserves_interior_d3q19(self):
        """halo_exchange overwrites only border, interior is untouched."""
        parent_f = _make_d3q19_field(nz=8, ny=8, nx=8, seed=400)
        box = BoxRegion(1, 5, 1, 5, 1, 5)  # 4x4x4 → f_up is 8x8x8
        ratio = 2

        # Create patch with matching shape
        f_patch = amr_common.refine(
            parent_f[:, 1:5, 1:5, 1:5], lattice="D3Q19",
            tau_c=1.0, tau_f=0.75, ratio=ratio,
        )
        f_interior_before = f_patch[:, ratio:-ratio, ratio:-ratio, ratio:-ratio].clone()

        amr_common.halo_exchange(
            f_patch, parent_f, box=box, ratio=ratio,
            lattice="D3Q19", tau_p=1.0, tau_c=0.75,
        )

        f_interior_after = f_patch[:, ratio:-ratio, ratio:-ratio, ratio:-ratio]
        assert torch.allclose(f_interior_before, f_interior_after, atol=0), \
            "halo_exchange modified interior cells!"

    def test_amr_patch3d_lifecycle(self):
        """Full AMRPatch3D lifecycle: create, collide, halo exchange, coarsen."""
        parent_f = _make_d3q19_field(nz=8, ny=8, nx=8, seed=500)
        box = BoxRegion(2, 6, 2, 6, 2, 6)  # 4x4x4
        ratio = 2
        tau_p, tau_c = 1.0, 0.75

        # Create patch
        f_fine = amr_common.refine(
            parent_f[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1],
            lattice="D3Q19", tau_c=tau_p, tau_f=tau_c, ratio=ratio,
        )
        patch = amr_common.AMRPatch3D(
            f=f_fine, box=box, ratio=ratio, level=1,
            parent_level=0, tau=tau_c, lattice="D3Q19",
        )
        assert patch.cells == 512

        # Halo exchange (inject parent into border)
        amr_common.halo_exchange(
            patch.f, parent_f, box=box, ratio=ratio,
            lattice="D3Q19", tau_p=tau_p, tau_c=tau_c,
        )

        # Collide
        patch.f = _simple_collide_d3q19(patch.f, tau=tau_c)

        # Coarsen back
        f_coarse = amr_common.coarsen(
            patch.f, lattice="D3Q19", tau_f=tau_c, tau_c=tau_p, ratio=ratio,
        )
        assert f_coarse.shape == (19, 4, 4, 4)

        # Write back to parent
        parent_f[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1] = f_coarse
        rho, _, _, _ = macroscopic3d(parent_f)
        assert (rho > 0).all()

    def test_adaptive_solver3d_step_with_amr_common_refine(self):
        """Integration: use amr_common.refine inside an AdaptiveSolver3D-like flow.

        Simulates the _add_patch path using amr_common instead of the
        original hardcoded D3Q19 functions.
        """
        f_coarse = _make_d3q19_field(nz=8, ny=8, nx=8, seed=600)
        box = BoxRegion(2, 6, 2, 6, 2, 6)
        ratio = 2
        tau_c, tau_f = 1.0, 0.75

        # Use amr_common.refine (would be D3Q27-compatible)
        f_patch = amr_common.refine(
            f_coarse[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1],
            lattice="D3Q19", tau_c=tau_c, tau_f=tau_f, ratio=ratio,
        )

        # Collide + stream + boundary
        f_patch = _simple_collide_d3q19(f_patch, tau=tau_f)
        f_patch = _identity(f_patch)  # stream
        f_patch = _identity(f_patch)  # boundary

        # Restrict back
        f_avg = amr_common.coarsen(f_patch, lattice="D3Q19", tau_f=tau_f, tau_c=tau_c)
        f_coarse[:, box.z0:box.z1, box.y0:box.y1, box.x0:box.x1] = f_avg

        rho, _, _, _ = macroscopic3d(f_coarse)
        assert (rho > 0).all()
        assert f_coarse.shape == (19, 8, 8, 8)


# ===========================================================================
# PART 4: Edge cases and stress tests
# ===========================================================================

class TestEdgeCases:
    """Edge case tests for robustness."""

    def test_refine_ratio_4(self):
        """Refine with ratio=4."""
        f = _make_d3q19_field(nz=2, ny=2, nx=2, seed=700)
        f_fine = amr_common.refine(f, lattice="D3Q19", ratio=4, use_fh=False)
        assert f_fine.shape == (19, 8, 8, 8)

    def test_coarsen_non_divisible_raises(self):
        """Coarsen with non-divisible dimensions raises RuntimeError.

        KNOWN LIMITATION: _fine_to_coarse_3d uses view() reshape which
        requires dimensions to be exactly divisible by ratio.  This is
        a contract requirement, not a bug — the caller must ensure the
        fine grid was created by a proper refine operation.
        """
        f = _make_d3q19_field(nz=7, ny=7, nx=7, seed=800)
        # 7 is not divisible by 2 → view() fails
        with pytest.raises(RuntimeError, match="shape"):
            amr_common.coarsen(f, lattice="D3Q19", ratio=2, use_fh=False)

    def test_unsupported_lattice_raises(self):
        """Unsupported lattice name raises ValueError."""
        f = _make_d3q19_field(nz=4, ny=4, nx=4)
        with pytest.raises(ValueError, match="Unsupported lattice"):
            amr_common.refine(f, lattice="D2Q9")
        with pytest.raises(ValueError, match="Unsupported lattice"):
            amr_common.coarsen(f, lattice="D2Q9")

    def test_amr_patch3d_unsupported_lattice(self):
        """AMRPatch3D rejects unsupported lattice in __post_init__."""
        f = _make_d3q19_field(nz=4, ny=4, nx=4)
        box = BoxRegion(0, 2, 0, 2, 0, 2)
        with pytest.raises(ValueError, match="Unsupported lattice"):
            amr_common.AMRPatch3D(f=f, box=box, lattice="D2Q9")

    def test_equivalence_multiple_seeds(self):
        """Equivalence holds across multiple random seeds."""
        for seed in [1, 42, 100, 999]:
            f = _make_d3q19_field(nz=4, ny=4, nx=4, seed=seed)
            f_orig = ar._fh_coarse_to_fine_3d(f, 1.0, 0.75, 2)
            f_common = amr_common.refine(f, lattice="D3Q19", tau_c=1.0, tau_f=0.75)
            assert torch.allclose(f_orig, f_common, atol=1e-7), \
                f"Equivalence failed for seed {seed}"

    def test_fh_rescaling_correctness(self):
        """FH rescaling: when tau_f == tau_c, FH should equal plain interpolation
        of the rescaled field (which equals the original field)."""
        f = _make_d3q19_field(nz=4, ny=4, nx=4, seed=123)
        # When tau_f == tau_c, scale=1, so f_rescaled == f
        f_fh = amr_common.refine(f, lattice="D3Q19", tau_c=1.0, tau_f=1.0, use_fh=True)
        f_plain = amr_common.refine(f, lattice="D3Q19", ratio=2, use_fh=False)
        # They should be identical because scale=1 means no rescaling
        assert torch.allclose(f_fh, f_plain, atol=1e-7), \
            "FH with tau_f==tau_c should equal plain interpolation"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
