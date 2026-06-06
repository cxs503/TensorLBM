"""Tests for adaptive mesh refinement (AMR) — adaptive_refinement.py."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.adaptive_refinement import (
    AdaptationSchedule,
    AdaptiveSolver2D,
    AdaptiveSolver3D,
    _coarse_to_fine_2d,
    _fine_to_coarse_2d,
    _group_refine_boxes_2d,
    _group_refine_boxes_3d,
    gradient_indicator_2d,
    gradient_indicator_3d,
    mark_cells_for_refinement,
    nonequilibrium_indicator_2d,
    nonequilibrium_indicator_3d,
    vorticity_indicator_2d,
    vorticity_indicator_3d,
)
from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.refinement import BoxRegion

# ---------------------------------------------------------------------------
# Upsampling / restriction helpers
# ---------------------------------------------------------------------------

class TestCoarseFine2D:
    def test_upsample_shape(self) -> None:
        f = torch.rand(9, 8, 10)
        out = _coarse_to_fine_2d(f, ratio=2)
        assert out.shape == (9, 16, 20)

    def test_upsample_ratio4(self) -> None:
        f = torch.rand(9, 4, 6)
        out = _coarse_to_fine_2d(f, ratio=4)
        assert out.shape == (9, 16, 24)

    def test_restrict_shape(self) -> None:
        f = torch.rand(9, 16, 20)
        out = _fine_to_coarse_2d(f, ratio=2)
        assert out.shape == (9, 8, 10)

    def test_restrict_constant(self) -> None:
        """Restricting a constant field should give the same constant."""
        f = torch.full((9, 8, 8), 0.5)
        out = _fine_to_coarse_2d(f, ratio=2)
        assert out.shape == (9, 4, 4)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=1e-5)


# ---------------------------------------------------------------------------
# Error indicators — 2-D
# ---------------------------------------------------------------------------

class TestIndicators2D:
    def _make_f(self, ny: int = 8, nx: int = 10) -> tuple:
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        return f, rho, ux, uy

    def test_neq_zero_for_equilibrium(self) -> None:
        f, rho, ux, uy = self._make_f()
        ind = nonequilibrium_indicator_2d(f, rho, ux, uy)
        assert ind.shape == (8, 10)
        assert float(ind.max()) == pytest.approx(0.0, abs=1e-6)

    def test_neq_positive_for_nonequilibrium(self) -> None:
        f, rho, ux, uy = self._make_f()
        f = f + 0.01 * torch.rand_like(f)
        ind = nonequilibrium_indicator_2d(f, rho, ux, uy)
        assert float(ind.max()) > 0.0

    def test_vorticity_zero_for_uniform(self) -> None:
        ux = torch.full((8, 10), 0.05)
        uy = torch.zeros(8, 10)
        ind = vorticity_indicator_2d(ux, uy)
        assert ind.shape == (8, 10)
        assert float(ind.abs().max()) == pytest.approx(0.0, abs=1e-6)

    def test_vorticity_nonzero_for_shear(self) -> None:
        ny, nx = 16, 16
        y = torch.linspace(0, 1, ny)
        ux = y.unsqueeze(1).expand(ny, nx)
        uy = torch.zeros(ny, nx)
        ind = vorticity_indicator_2d(ux, uy)
        # Interior vorticity should be non-zero
        assert float(ind[1:-1, 1:-1].abs().max()) > 0.0

    def test_gradient_indicator_zero_for_uniform(self) -> None:
        phi = torch.full((8, 10), 1.0)
        ind = gradient_indicator_2d(phi)
        assert ind.shape == (8, 10)
        # Interior should be zero
        assert float(ind[1:-1, 1:-1].abs().max()) == pytest.approx(0.0, abs=1e-6)

    def test_gradient_indicator_nonzero_for_ramp(self) -> None:
        nx = 20
        x = torch.linspace(0, 1, nx)
        phi = x.unsqueeze(0).expand(8, nx)
        ind = gradient_indicator_2d(phi)
        assert float(ind.max()) > 0.0


# ---------------------------------------------------------------------------
# Error indicators — 3-D
# ---------------------------------------------------------------------------

class TestIndicators3D:
    def _make_f3d(self, nz: int = 4, ny: int = 6, nx: int = 8) -> tuple:
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        f = equilibrium3d(rho, ux, uy, uz)
        return f, rho, ux, uy, uz

    def test_neq_zero_for_equilibrium(self) -> None:
        f, rho, ux, uy, uz = self._make_f3d()
        ind = nonequilibrium_indicator_3d(f, rho, ux, uy, uz)
        assert ind.shape == (4, 6, 8)
        assert float(ind.max()) == pytest.approx(0.0, abs=1e-6)

    def test_neq_positive_for_perturbed(self) -> None:
        f, rho, ux, uy, uz = self._make_f3d()
        f = f + 0.01 * torch.rand_like(f)
        ind = nonequilibrium_indicator_3d(f, rho, ux, uy, uz)
        assert float(ind.max()) > 0.0

    def test_vorticity_zero_for_uniform(self) -> None:
        ux = torch.full((4, 6, 8), 0.05)
        uy = torch.zeros(4, 6, 8)
        uz = torch.zeros(4, 6, 8)
        ind = vorticity_indicator_3d(ux, uy, uz)
        assert ind.shape == (4, 6, 8)
        assert float(ind.abs().max()) == pytest.approx(0.0, abs=1e-6)

    def test_gradient_indicator_3d_shape(self) -> None:
        phi = torch.rand(4, 6, 8)
        ind = gradient_indicator_3d(phi)
        assert ind.shape == (4, 6, 8)
        assert torch.isfinite(ind).all()


# ---------------------------------------------------------------------------
# Cell marking
# ---------------------------------------------------------------------------

class TestMarkCells:
    def test_basic_marking(self) -> None:
        ind = torch.tensor([[0.001, 0.01, 0.0001],
                             [0.005, 0.02, 0.0000001]])
        refine, coarsen = mark_cells_for_refinement(
            ind, refine_threshold=0.005, coarsen_threshold=0.001
        )
        assert refine.shape == ind.shape
        assert coarsen.shape == ind.shape
        # 0.01, 0.02 > 0.005 → should be flagged for refinement
        assert bool(refine[0, 1])
        assert bool(refine[1, 2]) is False   # 0.0000001 < 0.001 → coarsen not refine
        # 0.0001, 0.0000001 < 0.001 → should be flagged for coarsening
        assert bool(coarsen[0, 2])

    def test_invalid_thresholds_raises(self) -> None:
        ind = torch.rand(4, 4)
        with pytest.raises(ValueError):
            mark_cells_for_refinement(ind, refine_threshold=0.001, coarsen_threshold=0.01)

    def test_all_refine(self) -> None:
        ind = torch.full((4, 4), 1.0)
        refine, coarsen = mark_cells_for_refinement(ind, 0.5, 0.1)
        assert refine.all()
        assert not coarsen.any()

    def test_all_coarsen(self) -> None:
        ind = torch.full((4, 4), 0.01)
        refine, coarsen = mark_cells_for_refinement(ind, 0.5, 0.1)
        assert not refine.any()
        assert coarsen.all()


# ---------------------------------------------------------------------------
# Box grouping helpers
# ---------------------------------------------------------------------------

class TestGroupBoxes2D:
    def test_empty_mask_returns_no_boxes(self) -> None:
        mask = torch.zeros(10, 20, dtype=torch.bool)
        boxes = _group_refine_boxes_2d(mask)
        assert boxes == []

    def test_full_mask_returns_boxes(self) -> None:
        mask = torch.ones(10, 20, dtype=torch.bool)
        boxes = _group_refine_boxes_2d(mask, max_patches=4)
        assert len(boxes) <= 4
        assert all(isinstance(b, BoxRegion) for b in boxes)

    def test_single_active_region(self) -> None:
        mask = torch.zeros(20, 40, dtype=torch.bool)
        mask[5:15, 10:30] = True
        boxes = _group_refine_boxes_2d(mask, pad=0, max_patches=8)
        assert len(boxes) >= 1
        # All boxes should cover the active region
        covered_x = set()
        for b in boxes:
            covered_x.update(range(b.x0, b.x1))
        assert all(x in covered_x for x in range(10, 30))


class TestGroupBoxes3D:
    def test_empty_mask_returns_no_boxes(self) -> None:
        mask = torch.zeros(4, 8, 16, dtype=torch.bool)
        boxes = _group_refine_boxes_3d(mask)
        assert boxes == []

    def test_active_region(self) -> None:
        mask = torch.zeros(8, 12, 24, dtype=torch.bool)
        mask[2:6, 3:9, 8:20] = True
        boxes = _group_refine_boxes_3d(mask, pad=0, max_patches=4)
        assert len(boxes) >= 1
        assert all(isinstance(b, BoxRegion) for b in boxes)


# ---------------------------------------------------------------------------
# AdaptationSchedule
# ---------------------------------------------------------------------------

class TestAdaptationSchedule:
    def test_adapts_at_interval(self) -> None:
        sch = AdaptationSchedule(interval=10, warmup=0)
        assert sch.should_adapt(0) is True
        assert sch.should_adapt(10) is True
        assert sch.should_adapt(20) is True
        assert sch.should_adapt(5) is False

    def test_warmup_respected(self) -> None:
        sch = AdaptationSchedule(interval=5, warmup=20)
        assert sch.should_adapt(0) is False
        assert sch.should_adapt(19) is False
        assert sch.should_adapt(20) is True
        assert sch.should_adapt(25) is True


# ---------------------------------------------------------------------------
# AdaptiveSolver2D — unit tests
# ---------------------------------------------------------------------------

class TestAdaptiveSolver2D:
    def _make_solver(self, ny: int = 16, nx: int = 24) -> AdaptiveSolver2D:
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)
        sch = AdaptationSchedule(
            interval=5, warmup=0,
            refine_threshold=1e-4,
            coarsen_threshold=1e-8,
            max_patches=4,
        )
        return AdaptiveSolver2D(f, schedule=sch)

    # Identity operators used to test the framework without a real solver
    @staticmethod
    def _identity(f: torch.Tensor) -> torch.Tensor:
        return f

    def test_initial_state(self) -> None:
        solver = self._make_solver()
        assert solver.n_patches == 0
        assert solver.total_cells == 16 * 24

    def test_step_without_patches(self) -> None:
        solver = self._make_solver()
        solver.step(self._identity, self._identity, self._identity)
        assert solver.coarse_f.shape == (9, 16, 24)

    def test_adapt_adds_patches(self) -> None:
        solver = self._make_solver()
        # Create a non-equilibrium perturbation to trigger refinement
        solver.coarse_f = solver.coarse_f + 0.01 * torch.rand_like(solver.coarse_f)
        rho, ux, uy = macroscopic(solver.coarse_f)
        indicator = nonequilibrium_indicator_2d(solver.coarse_f, rho, ux, uy)
        solver.adapt(indicator)
        # Some patches should have been added (perturbation is large)
        assert solver.n_patches >= 1

    def test_step_with_patches(self) -> None:
        solver = self._make_solver()
        # Manually add a patch
        box = BoxRegion(4, 10, 4, 10, 0, 0)
        solver._add_patch(box, ratio=2)
        assert solver.n_patches == 1
        solver.step(self._identity, self._identity, self._identity)
        assert solver.coarse_f.shape == (9, 16, 24)

    def test_coarsen_removes_patch(self) -> None:
        solver = self._make_solver()
        box = BoxRegion(4, 10, 4, 10, 0, 0)
        solver._add_patch(box, ratio=2)
        assert solver.n_patches == 1
        # indicator = 0 everywhere → should coarsen
        indicator = torch.zeros(16, 24)
        solver.adapt(indicator)
        assert solver.n_patches == 0

    def test_patch_info_structure(self) -> None:
        solver = self._make_solver()
        box = BoxRegion(2, 8, 2, 8, 0, 0)
        solver._add_patch(box, ratio=2)
        info = solver.patch_info()
        assert len(info) == 1
        assert "box" in info[0]
        assert info[0]["cells"] == info[0]["ny"] * info[0]["nx"]

    def test_total_cells_increases_with_patch(self) -> None:
        solver = self._make_solver()
        base = solver.total_cells
        box = BoxRegion(2, 8, 2, 8, 0, 0)
        solver._add_patch(box, ratio=2)
        assert solver.total_cells > base

    def test_max_patches_respected(self) -> None:
        solver = self._make_solver()
        solver.coarse_f = solver.coarse_f + 0.1 * torch.rand_like(solver.coarse_f)
        rho, ux, uy = macroscopic(solver.coarse_f)
        indicator = nonequilibrium_indicator_2d(solver.coarse_f, rho, ux, uy)
        solver.adapt(indicator)
        assert solver.n_patches <= solver.schedule.max_patches

    def test_should_adapt_delegates_to_schedule(self) -> None:
        solver = self._make_solver()
        assert solver.should_adapt(0) is True
        assert solver.should_adapt(3) is False

    def test_coarse_f_shape_preserved_after_steps(self) -> None:
        solver = self._make_solver()
        box = BoxRegion(4, 10, 4, 10, 0, 0)
        solver._add_patch(box, ratio=2)
        for _ in range(3):
            solver.step(self._identity, self._identity, self._identity)
        assert solver.coarse_f.shape == (9, 16, 24)


# ---------------------------------------------------------------------------
# AdaptiveSolver3D — unit tests
# ---------------------------------------------------------------------------

class TestAdaptiveSolver3D:
    def _make_solver(
        self, nz: int = 6, ny: int = 8, nx: int = 10
    ) -> AdaptiveSolver3D:
        rho = torch.ones(nz, ny, nx)
        ux = torch.full((nz, ny, nx), 0.05)
        uy = torch.zeros(nz, ny, nx)
        uz = torch.zeros(nz, ny, nx)
        f = equilibrium3d(rho, ux, uy, uz)
        sch = AdaptationSchedule(
            interval=5, warmup=0,
            refine_threshold=1e-4,
            coarsen_threshold=1e-8,
            max_patches=4,
        )
        return AdaptiveSolver3D(f, schedule=sch)

    @staticmethod
    def _identity(f: torch.Tensor) -> torch.Tensor:
        return f

    def test_initial_state(self) -> None:
        solver = self._make_solver()
        assert solver.n_patches == 0
        assert solver.total_cells == 6 * 8 * 10

    def test_step_without_patches(self) -> None:
        solver = self._make_solver()
        solver.step(self._identity, self._identity, self._identity)
        assert solver.coarse_f.shape == (19, 6, 8, 10)

    def test_adapt_adds_patches(self) -> None:
        solver = self._make_solver()
        solver.coarse_f = solver.coarse_f + 0.01 * torch.rand_like(solver.coarse_f)
        rho, ux, uy, uz = macroscopic3d(solver.coarse_f)
        indicator = nonequilibrium_indicator_3d(solver.coarse_f, rho, ux, uy, uz)
        solver.adapt(indicator)
        assert solver.n_patches >= 1

    def test_step_with_patch(self) -> None:
        solver = self._make_solver()
        box = BoxRegion(2, 6, 2, 5, 1, 4)
        solver._add_patch(box, ratio=2)
        solver.step(self._identity, self._identity, self._identity)
        assert solver.coarse_f.shape == (19, 6, 8, 10)

    def test_coarsen_removes_patch(self) -> None:
        solver = self._make_solver()
        box = BoxRegion(2, 6, 2, 5, 1, 4)
        solver._add_patch(box, ratio=2)
        assert solver.n_patches == 1
        indicator = torch.zeros(6, 8, 10)
        solver.adapt(indicator)
        assert solver.n_patches == 0

    def test_total_cells_with_patch(self) -> None:
        solver = self._make_solver()
        base = solver.total_cells
        box = BoxRegion(2, 5, 2, 5, 1, 4)
        solver._add_patch(box, ratio=2)
        assert solver.total_cells > base

    def test_max_patches_respected(self) -> None:
        solver = self._make_solver()
        solver.coarse_f = solver.coarse_f + 0.1 * torch.rand_like(solver.coarse_f)
        rho, ux, uy, uz = macroscopic3d(solver.coarse_f)
        indicator = nonequilibrium_indicator_3d(solver.coarse_f, rho, ux, uy, uz)
        solver.adapt(indicator)
        assert solver.n_patches <= solver.schedule.max_patches


# ---------------------------------------------------------------------------
# Integration: end-to-end adapt+step cycle (2-D)
# ---------------------------------------------------------------------------

class TestEndToEnd2D:
    def test_multiple_adapt_step_cycles(self) -> None:
        """Simulate several adapt+step cycles and verify solver remains stable."""
        from tensorlbm.solver import collide_bgk, stream

        ny, nx = 20, 30
        rho = torch.ones(ny, nx)
        ux = torch.full((ny, nx), 0.05)
        uy = torch.zeros(ny, nx)
        f = equilibrium(rho, ux, uy)

        sch = AdaptationSchedule(
            interval=5, warmup=0,
            refine_threshold=1e-3,
            coarsen_threshold=1e-7,
            max_patches=4,
        )
        solver = AdaptiveSolver2D(f, schedule=sch)
        omega = 1.0

        # A no-op boundary function is sufficient for testing AMR mechanics
        def identity_bc(f: torch.Tensor) -> torch.Tensor:
            return f

        for step in range(15):
            solver.step(
                lambda f: collide_bgk(f, omega),
                stream,
                identity_bc,
            )
            if solver.should_adapt(step):
                rho_f, ux_f, uy_f = macroscopic(solver.coarse_f)
                ind = nonequilibrium_indicator_2d(solver.coarse_f, rho_f, ux_f, uy_f)
                solver.adapt(ind)

        assert solver.coarse_f.shape == (9, ny, nx)
        assert torch.isfinite(solver.coarse_f).all()
        assert solver.n_patches <= sch.max_patches


# ---------------------------------------------------------------------------
# Public import check
# ---------------------------------------------------------------------------

def test_public_import() -> None:
    """Verify all AMR symbols are importable from the top-level package."""
    import tensorlbm as tlbm

    for name in [
        "AdaptationSchedule",
        "AMRPatch2D",
        "AMRPatch3D",
        "AdaptiveSolver2D",
        "AdaptiveSolver3D",
        "nonequilibrium_indicator_2d",
        "vorticity_indicator_2d",
        "gradient_indicator_2d",
        "nonequilibrium_indicator_3d",
        "vorticity_indicator_3d",
        "gradient_indicator_3d",
        "mark_cells_for_refinement",
    ]:
        assert hasattr(tlbm, name), f"Missing: {name}"
