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
    _merge_reflux_ledgers,
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


def _four_level_configs() -> tuple[
    StaticBlockAMRConfig,
    StaticBlockAMRConfig,
    StaticBlockAMRConfig,
]:
    outer, inner = _configs()
    deepest = StaticBlockAMRConfig(
        # The level-2 allocation is (14, 14, 22), including ghost cells.
        BoxRegion(x0=3, x1=19, y0=3, y1=11, z0=3, z1=11),
        tau_coarse=inner.tau_fine,
    )
    return outer, inner, deepest


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


def test_four_levels_follow_eight_finest_substeps_and_conserve() -> None:
    hierarchy = NestedStaticBlockAMR3D(
        _equilibrium((12, 12, 14)), _four_level_configs(),
    )
    before = tuple(level.clone() for level in hierarchy.level_populations)
    calls: list[tuple[int, int]] = []

    def identity(
        state: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        del tau
        calls.append((level, substep))
        return AMRAdvanceResult(state.clone(), state.clone())

    ledgers = hierarchy.step(identity)

    assert [sum(level == expected for level, _ in calls) for expected in range(4)] == [
        1, 2, 4, 8,
    ]
    assert [substep for level, substep in calls if level == 3] == list(range(8))
    assert len(ledgers) == 3
    assert all(abs(ledger.mass_residual) < 1.0e-12 for ledger in ledgers)
    for actual, expected in zip(hierarchy.level_populations, before, strict=True):
        assert torch.allclose(actual, expected, rtol=0.0, atol=3.0e-8)


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


def test_nested_hierarchy_accepts_dynamic_convective_tau_chain() -> None:
    hierarchy = NestedStaticBlockAMR3D(
        _equilibrium((12, 12, 14)), _configs(),
    )
    observed: dict[int, set[float]] = {0: set(), 1: set(), 2: set()}

    def identity(
        state: torch.Tensor,
        tau: float,
        level: int,
        substep: int,
    ) -> AMRAdvanceResult:
        del substep
        observed[level].add(tau)
        return AMRAdvanceResult(state.clone(), state.clone())

    hierarchy.step(identity, tau_by_level=(0.55, 0.60, 0.70))
    assert observed == {0: {0.55}, 1: {0.60}, 2: {0.70}}
    with pytest.raises(ValueError, match="interface 1"):
        hierarchy.step(identity, tau_by_level=(0.55, 0.60, 0.69))


def test_nested_interface_filters_preserve_uniform_state_and_conservation() -> None:
    outer = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=11, y0=3, y1=9, z0=3, z1=9),
        tau_coarse=0.56,
        interface_filter_width=1,
        interface_filter_strength=0.25,
    )
    inner = StaticBlockAMRConfig(
        BoxRegion(x0=4, x1=14, y0=4, y1=10, z0=4, z1=10),
        tau_coarse=outer.tau_fine,
        interface_filter_width=1,
        interface_filter_strength=0.25,
    )
    hierarchy = NestedStaticBlockAMR3D(
        _equilibrium((12, 12, 14)), (outer, inner),
    )
    before = tuple(level.clone() for level in hierarchy.level_populations)

    def identity(
        state: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        del tau, level, substep
        return AMRAdvanceResult(state.clone(), state.clone())

    ledgers = hierarchy.step(identity)

    for actual, expected in zip(hierarchy.level_populations, before, strict=True):
        assert torch.allclose(actual, expected, rtol=0.0, atol=3.0e-8)
    assert all(abs(ledger.mass_residual) < 1.0e-12 for ledger in ledgers)


def test_nested_hierarchy_reports_cell_savings() -> None:
    hierarchy = NestedStaticBlockAMR3D(
        _equilibrium((12, 12, 14)), _configs(),
    )
    assert len(hierarchy.level_populations) == 3
    assert hierarchy.total_allocated_cells < hierarchy.uniform_finest_equivalent_cells
    assert 0.0 < hierarchy.cell_saving_fraction < 1.0


def test_repeated_child_ledgers_accumulate_over_the_root_step() -> None:
    from tensorlbm.static_block_amr import PopulationRefluxLedger

    first = PopulationRefluxLedger(
        torch.tensor([1.0]),
        torch.tensor([0.75]),
        4,
        torch.tensor([0.25]),
        1,
        torch.tensor([2.0]),
        0.1,
        0.8,
        0.2,
        0.7,
        0.04,
    )
    second = PopulationRefluxLedger(
        torch.tensor([3.0]),
        torch.tensor([2.5]),
        5,
        torch.tensor([0.5]),
        2,
        torch.tensor([4.0]),
        0.3,
        0.6,
        0.1,
        0.9,
        0.07,
    )

    merged = _merge_reflux_ledgers(first, second)

    torch.testing.assert_close(merged.replacement_mismatch, torch.tensor([4.0]))
    torch.testing.assert_close(merged.applied_shell_correction, torch.tensor([3.25]))
    torch.testing.assert_close(merged.residual, torch.tensor([0.75]))
    torch.testing.assert_close(merged.raw_kinetic_mismatch, torch.tensor([6.0]))
    assert merged.shell_cells == 9
    assert merged.limited_directions == 3
    assert merged.restriction_limited_fraction == 0.3
    assert merged.restriction_minimum_alpha == 0.6
    assert merged.prolongation_limited_fraction == 0.2
    assert merged.prolongation_minimum_alpha == 0.7
    assert merged.maximum_applied_correction_fraction == 0.07


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
