"""Multi-level scheduling and conservation tests for nested static AMR."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_mrt3d, stream3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
    NestedStaticBlockAMR3D,
    StaticBlockAMRConfig,
)


def _equilibrium(shape: tuple[int, int, int]) -> torch.Tensor:
    rho = torch.full(shape, 1.01, dtype=torch.float64)
    ux = torch.full(shape, 0.025, dtype=torch.float64)
    uy = torch.full(shape, -0.006, dtype=torch.float64)
    uz = torch.full(shape, 0.003, dtype=torch.float64)
    return equilibrium3d(rho, ux, uy, uz)


def _configs() -> tuple[StaticBlockAMRConfig, StaticBlockAMRConfig]:
    outer = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=11, y0=3, y1=9, z0=3, z1=9),
        tau_coarse=0.56,
    )
    inner = StaticBlockAMRConfig(
        # Coordinates are in the outer fine grid including its ghost layer.
        BoxRegion(x0=4, x1=14, y0=4, y1=10, z0=4, z1=10),
        tau_coarse=outer.tau_fine,
    )
    return outer, inner


def test_three_levels_follow_exact_recursive_subcycling() -> None:
    hierarchy = NestedStaticBlockAMR3D(
        _equilibrium((12, 12, 14)), _configs(),
    )
    before = tuple(level.clone() for level in hierarchy.level_populations)
    calls: list[tuple[int, int, float]] = []

    def identity(
        state: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        calls.append((level, substep, tau))
        return AMRAdvanceResult(state.clone(), state.clone())

    ledgers = hierarchy.step(identity)

    assert [(level, substep) for level, substep, _ in calls] == [
        (0, -1),
        (1, 0), (2, 0), (2, 1),
        (1, 1), (2, 2), (2, 3),
    ]
    assert [tau for _, _, tau in calls] == pytest.approx([
        0.56, 0.62, 0.74, 0.74, 0.62, 0.74, 0.74,
    ])
    assert len(ledgers) == 2
    assert all(abs(ledger.mass_residual) < 1.0e-13 for ledger in ledgers)
    for actual, expected in zip(hierarchy.level_populations, before, strict=True):
        assert torch.allclose(actual, expected, rtol=0.0, atol=2.0e-8)


def test_three_level_mrt_step_remains_finite_and_conservative() -> None:
    torch.manual_seed(23)
    coarse = _equilibrium((12, 12, 14)).float()
    coarse += 1.0e-6 * torch.randn_like(coarse)
    hierarchy = NestedStaticBlockAMR3D(coarse, _configs())
    initial_mass = float(hierarchy.coarse_f.sum())

    def advance(
        state: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        del level, substep
        post = collide_mrt3d(state, tau=tau)
        return AMRAdvanceResult(stream3d(post), post)

    maximum_residual = 0.0
    for _ in range(50):
        ledgers = hierarchy.step(advance)
        maximum_residual = max(
            maximum_residual,
            *(float(ledger.residual.abs().max()) for ledger in ledgers),
        )

    final_mass = float(hierarchy.coarse_f.sum())
    assert all(
        bool(torch.isfinite(level).all()) for level in hierarchy.level_populations
    )
    assert min(float(level.min()) for level in hierarchy.level_populations) > 0.0
    assert abs(final_mass - initial_mass) / initial_mass < 1.0e-7
    assert maximum_residual < 2.0e-11


def test_nested_hierarchy_requires_tau_chain_and_reflux() -> None:
    coarse = _equilibrium((12, 12, 14))
    outer, inner = _configs()
    wrong_tau = StaticBlockAMRConfig(inner.box, tau_coarse=0.63)
    with pytest.raises(ValueError, match="tau_coarse"):
        NestedStaticBlockAMR3D(coarse, (outer, wrong_tau))

    replacement_only = StaticBlockAMRConfig(
        inner.box, tau_coarse=outer.tau_fine, reflux=False,
    )
    with pytest.raises(ValueError, match="reflux on every interface"):
        NestedStaticBlockAMR3D(coarse, (outer, replacement_only))


def test_nested_hierarchy_reports_cell_savings() -> None:
    hierarchy = NestedStaticBlockAMR3D(
        _equilibrium((12, 12, 14)), _configs(),
    )
    assert len(hierarchy.level_populations) == 3
    assert hierarchy.total_allocated_cells < hierarchy.uniform_finest_equivalent_cells
    assert 0.0 < hierarchy.cell_saving_fraction < 1.0


def test_restore_level_populations_relinks_nested_parent_state() -> None:
    hierarchy = NestedStaticBlockAMR3D(
        _equilibrium((12, 12, 14)), _configs(),
    )
    restored = tuple(level.clone() * 0.999 for level in hierarchy.level_populations)

    hierarchy.restore_level_populations(restored)

    assert hierarchy.coarse_f is restored[0]
    assert hierarchy.interfaces[0].fine_f is restored[1]
    assert hierarchy.interfaces[1].coarse_f is restored[1]
    assert hierarchy.finest_f is restored[2]
    with pytest.raises(ValueError, match="one population tensor per level"):
        hierarchy.restore_level_populations(restored[:-1])


def test_nested_runtime_is_publicly_exported() -> None:
    import tensorlbm

    assert tensorlbm.NestedStaticBlockAMR3D is NestedStaticBlockAMR3D
