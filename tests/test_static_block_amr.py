"""Conservation and scheduling tests for the production static block AMR."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.d3q19 import C as C19
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_mrt3d, stream3d
from tensorlbm.static_block_amr import (
    StaticBlockAMR3D,
    StaticBlockAMRConfig,
    convective_refined_tau,
)


def _uniform_equilibrium(shape: tuple[int, int, int]) -> torch.Tensor:
    rho = torch.full(shape, 1.03, dtype=torch.float64)
    ux = torch.full(shape, 0.031, dtype=torch.float64)
    uy = torch.full(shape, -0.012, dtype=torch.float64)
    uz = torch.full(shape, 0.007, dtype=torch.float64)
    return equilibrium3d(rho, ux, uy, uz)


def _config(*, reflux: bool = True) -> StaticBlockAMRConfig:
    return StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=7, y0=2, y1=6, z0=2, z1=5),
        tau_coarse=0.56,
        reflux=reflux,
    )


def test_convective_tau_scaling() -> None:
    assert convective_refined_tau(0.56) == pytest.approx(0.62)
    assert convective_refined_tau(0.5001) == pytest.approx(0.5002)
    with pytest.raises(ValueError):
        convective_refined_tau(0.5)


def test_uniform_moving_equilibrium_survives_nested_step_exactly() -> None:
    coarse = _uniform_equilibrium((8, 9, 11))
    solver = StaticBlockAMR3D(coarse.clone(), _config())
    coarse_before = solver.coarse_f.clone()
    fine_before = solver.fine_f.clone()
    calls: list[tuple[int, int, float]] = []

    def identity(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        calls.append((level, substep, tau))
        return f.clone()

    ledger = solver.step(identity)
    # D3Q19 weights are stored as float32 constants, so a float64
    # equilibrium→macroscopic→equilibrium round trip carries ~1e-8 noise.
    assert torch.allclose(solver.coarse_f, coarse_before, rtol=0.0, atol=1e-8)
    assert torch.allclose(solver.fine_f, fine_before, rtol=0.0, atol=1e-8)
    assert [(level, substep) for level, substep, _ in calls] == [
        (0, -1), (1, 0), (1, 1),
    ]
    assert [tau for _, _, tau in calls] == pytest.approx([0.56, 0.62, 0.62])
    assert ledger.mass_residual == pytest.approx(0.0, abs=1e-14)


def test_population_reflux_preserves_every_direction_inventory() -> None:
    torch.manual_seed(7)
    coarse = torch.rand((19, 8, 9, 11), dtype=torch.float64) * 0.01 + 0.02
    solver = StaticBlockAMR3D(coarse.clone(), _config(reflux=True))
    inventory_before = solver.coarse_f.sum(dim=(1, 2, 3))

    def perturb_fine(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        del tau, substep
        return f.clone() if level == 0 else f + 1e-5

    ledger = solver.step(perturb_fine)
    inventory_after = solver.coarse_f.sum(dim=(1, 2, 3))
    assert torch.allclose(inventory_after, inventory_before, rtol=0.0, atol=2e-13)
    assert ledger.shell_cells > 0
    assert torch.count_nonzero(ledger.replacement_mismatch).item() == 19
    assert torch.allclose(ledger.residual, torch.zeros_like(ledger.residual), atol=1e-15)


def test_without_reflux_replacement_changes_inventory() -> None:
    coarse = _uniform_equilibrium((8, 9, 11))
    solver = StaticBlockAMR3D(coarse.clone(), _config(reflux=False))
    before = solver.coarse_f.sum(dim=(1, 2, 3))

    def perturb(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        del tau, substep
        return f if level == 0 else f + 1e-5

    ledger = solver.step(perturb)
    after = solver.coarse_f.sum(dim=(1, 2, 3))
    assert not torch.allclose(after, before, rtol=0.0, atol=1e-12)
    assert ledger.shell_cells == 0


def test_reflux_is_population_proportional_and_does_not_create_negatives() -> None:
    coarse = _uniform_equilibrium((8, 9, 11)).float()
    solver = StaticBlockAMR3D(coarse, _config(reflux=True))

    def drain_fine(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        del tau, substep
        return f if level == 0 else f * 0.999

    ledger = solver.step(drain_fine)
    assert float(solver.coarse_f.min()) > 0.0
    assert ledger.limited_directions == 0
    assert abs(ledger.mass_residual) < 2e-5


def test_extreme_reflux_is_limited_and_exposes_residual() -> None:
    coarse = _uniform_equilibrium((8, 9, 11)).float()
    solver = StaticBlockAMR3D(coarse, _config(reflux=True))

    def inflate_fine(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        del tau, substep
        return f if level == 0 else f * 10.0

    ledger = solver.step(inflate_fine)
    assert float(solver.coarse_f.min()) >= 0.0
    assert ledger.limited_directions > 0
    assert torch.count_nonzero(ledger.residual).item() > 0


def test_local_block_saves_cells_against_uniform_refinement() -> None:
    coarse = _uniform_equilibrium((8, 9, 11))
    solver = StaticBlockAMR3D(coarse, _config())
    assert solver.physical_fine_shape == (6, 8, 8)
    assert solver.total_allocated_cells < solver.uniform_fine_equivalent_cells
    assert 0.0 < solver.cell_saving_fraction < 1.0


def test_fine_solid_gets_a_fluid_ghost_layer() -> None:
    coarse = _uniform_equilibrium((8, 9, 11))
    fine_solid = torch.zeros((6, 8, 8), dtype=torch.bool)
    fine_solid[1:-1, 2:-2, 2:-2] = True
    solver = StaticBlockAMR3D(coarse, _config(), fine_solid=fine_solid)
    padded = solver.fine_solid_with_ghost
    assert padded is not None
    assert padded.shape == (8, 10, 10)
    assert torch.equal(padded[1:-1, 1:-1, 1:-1], fine_solid)
    assert not bool(padded[0].any())
    assert not bool(padded[-1].any())
    assert not bool(padded[:, 0].any())
    assert not bool(padded[:, -1].any())
    assert not bool(padded[:, :, 0].any())
    assert not bool(padded[:, :, -1].any())


def test_block_must_be_strictly_interior() -> None:
    coarse = _uniform_equilibrium((8, 9, 11))
    bad = StaticBlockAMRConfig(
        BoxRegion(x0=0, x1=4, y0=2, y1=6, z0=2, z1=5),
        tau_coarse=0.56,
    )
    with pytest.raises(ValueError, match="strictly interior"):
        StaticBlockAMR3D(coarse, bad)


def test_static_block_runtime_is_publicly_exported() -> None:
    import tensorlbm

    assert tensorlbm.StaticBlockAMR3D is StaticBlockAMR3D
    assert tensorlbm.StaticBlockAMRConfig is StaticBlockAMRConfig


def test_real_mrt_stream_subcycling_preserves_global_mass_and_momentum() -> None:
    torch.manual_seed(11)
    coarse = _uniform_equilibrium((8, 9, 11)).float()
    coarse = coarse + 1e-5 * torch.randn_like(coarse)
    reference = stream3d(collide_mrt3d(coarse.clone(), tau=0.56))
    solver = StaticBlockAMR3D(coarse.clone(), _config(reflux=True))

    def advance(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        del level, substep
        return stream3d(collide_mrt3d(f, tau=tau))

    ledger = solver.step(advance)
    reference_by_q = reference.sum(dim=(1, 2, 3))
    actual_by_q = solver.coarse_f.sum(dim=(1, 2, 3))
    reference_mass = reference_by_q.sum()
    actual_mass = actual_by_q.sum()
    c = C19.to(reference_by_q)
    reference_momentum = (reference_by_q[:, None] * c).sum(dim=0)
    actual_momentum = (actual_by_q[:, None] * c).sum(dim=0)
    assert actual_mass == pytest.approx(float(reference_mass), abs=2e-5)
    assert torch.allclose(actual_momentum, reference_momentum, atol=2e-5, rtol=0.0)
    assert abs(ledger.mass_residual) < 1e-5
