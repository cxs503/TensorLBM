"""Conservation and scheduling tests for the production static block AMR."""
from __future__ import annotations

import pytest
import torch

from tensorlbm.d3q19 import C as C19
from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.refinement import BoxRegion
from tensorlbm.solver3d import collide_mrt3d, stream3d
from tensorlbm.static_block_amr import (
    AMRAdvanceResult,
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
    with pytest.raises(ValueError, match="maximum_reflux_correction_fraction"):
        StaticBlockAMRConfig(
            BoxRegion(x0=3, x1=7, y0=2, y1=6, z0=2, z1=5),
            tau_coarse=0.56,
            maximum_reflux_correction_fraction=0.0,
        )
    with pytest.raises(ValueError, match="ghost_interpolation"):
        StaticBlockAMRConfig(
            BoxRegion(x0=3, x1=7, y0=2, y1=6, z0=2, z1=5),
            tau_coarse=0.56,
            ghost_interpolation="cubic",
        )


def test_single_interface_accepts_a_convectively_scaled_dynamic_tau_pair() -> None:
    solver = StaticBlockAMR3D(_uniform_equilibrium((8, 9, 11)), _config())
    observed: list[float] = []

    def identity(
        f: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        del level, substep
        observed.append(tau)
        return AMRAdvanceResult(f.clone(), f.clone())

    solver.step(identity, tau_pair=(0.58, 0.66))
    assert observed == pytest.approx([0.58, 0.66, 0.66])
    with pytest.raises(ValueError, match="convective scaling"):
        solver.step(identity, tau_pair=(0.58, 0.65))


def test_restriction_regularization_is_an_explicit_common_option() -> None:
    config = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=7, y0=2, y1=6, z0=2, z1=5),
        tau_coarse=0.56,
        regularize_restriction=True,
    )
    solver = StaticBlockAMR3D(_uniform_equilibrium((8, 9, 11)), config)

    def identity(
        f: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        del tau, level, substep
        return AMRAdvanceResult(f.clone(), f.clone())

    ledger = solver.step(identity)
    assert bool(torch.isfinite(solver.coarse_f).all())
    assert abs(ledger.mass_residual) < 1e-12


def test_transfer_positivity_limits_amplified_restriction_before_parent_use() -> None:
    config = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=7, y0=2, y1=6, z0=2, z1=5),
        tau_coarse=0.5002,
        enforce_transfer_positivity=True,
    )
    solver = StaticBlockAMR3D(_uniform_equilibrium((8, 9, 11)).float(), config)
    # Inject a zero-mass/momentum stress large enough that fine-to-coarse
    # non-equilibrium amplification would otherwise create negatives.
    physical = solver.fine_physical
    physical[0] -= 0.2
    physical[1] += 0.1
    physical[2] += 0.1

    restricted = solver._restrict_physical()

    assert float(restricted.min()) >= 0.0
    diagnostic = solver.last_restriction_positivity
    assert diagnostic is not None
    assert diagnostic.limited_fraction > 0.0
    assert diagnostic.minimum_alpha < 1.0


def test_uniform_moving_equilibrium_survives_nested_step_exactly() -> None:
    coarse = _uniform_equilibrium((8, 9, 11))
    solver = StaticBlockAMR3D(coarse.clone(), _config())
    coarse_before = solver.coarse_f.clone()
    fine_before = solver.fine_f.clone()
    calls: list[tuple[int, int, float]] = []

    def identity(f: torch.Tensor, tau: float, level: int, substep: int) -> AMRAdvanceResult:
        calls.append((level, substep, tau))
        return AMRAdvanceResult(f.clone(), f.clone())

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


def test_flux_reflux_does_not_hide_nonconservative_collision_source() -> None:
    torch.manual_seed(7)
    coarse = torch.rand((19, 8, 9, 11), dtype=torch.float64) * 0.01 + 0.02
    solver = StaticBlockAMR3D(coarse.clone(), _config(reflux=True))
    inventory_before = solver.coarse_f.sum(dim=(1, 2, 3))

    def perturb_fine(f: torch.Tensor, tau: float, level: int, substep: int) -> AMRAdvanceResult:
        del tau, substep
        out = f.clone() if level == 0 else f + 1e-5
        return AMRAdvanceResult(out, out)

    ledger = solver.step(perturb_fine)
    inventory_after = solver.coarse_f.sum(dim=(1, 2, 3))
    # Reflux corrects interface transport, not an artificial volume source
    # injected throughout the fine block by this callback.
    assert not torch.allclose(inventory_after, inventory_before, rtol=0.0, atol=1e-12)
    assert ledger.shell_cells > 0


def test_reflux_requires_postcollision_flux_state() -> None:
    solver = StaticBlockAMR3D(_uniform_equilibrium((8, 9, 11)), _config(reflux=True))

    def opaque_update(f: torch.Tensor, tau: float, level: int, substep: int) -> torch.Tensor:
        del tau, level, substep
        return f.clone()

    with pytest.raises(TypeError, match="post-collision/pre-stream"):
        solver.step(opaque_update)


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

    def drain_fine(f: torch.Tensor, tau: float, level: int, substep: int) -> AMRAdvanceResult:
        del tau, substep
        out = f if level == 0 else f * 0.999
        return AMRAdvanceResult(out, out)

    ledger = solver.step(drain_fine)
    assert float(solver.coarse_f.min()) > 0.0
    assert ledger.limited_directions == 0
    assert abs(ledger.mass_residual) < 2e-5


def test_large_positive_interface_correction_does_not_create_negatives() -> None:
    coarse = _uniform_equilibrium((8, 9, 11)).float()
    solver = StaticBlockAMR3D(coarse, _config(reflux=True))

    def inflate_fine(f: torch.Tensor, tau: float, level: int, substep: int) -> AMRAdvanceResult:
        del tau, substep
        out = f if level == 0 else f * 10.0
        return AMRAdvanceResult(out, out)

    ledger = solver.step(inflate_fine)
    assert float(solver.coarse_f.min()) >= 0.0
    assert ledger.shell_cells > 0
    assert ledger.limited_directions > 0
    assert float(ledger.residual.abs().max()) > 0.0


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


def test_cell_centered_trilinear_ghost_fill_is_exact_for_linear_density() -> None:
    shape = (8, 9, 11)
    z, y, x = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float64) for size in shape),
        indexing="ij",
    )
    rho = 1.0 + 1e-3 * x + 2e-3 * y + 3e-3 * z
    zero = torch.zeros_like(rho)
    parent = equilibrium3d(rho, zero, zero, zero)
    config = StaticBlockAMRConfig(
        BoxRegion(x0=3, x1=7, y0=2, y1=6, z0=2, z1=5),
        tau_coarse=0.56,
        ghost_interpolation="trilinear",
    )
    solver = StaticBlockAMR3D(parent, config)
    interior_before = solver.fine_f[:, 1:-1, 1:-1, 1:-1].clone()
    solver.fine_f[:, 0] = 0.0
    solver.fine_f[:, -1] = 0.0
    solver.fine_f[:, :, 0] = 0.0
    solver.fine_f[:, :, -1] = 0.0
    solver.fine_f[:, :, :, 0] = 0.0
    solver.fine_f[:, :, :, -1] = 0.0

    solver._fill_ghost(parent)

    torch.testing.assert_close(
        solver.fine_f[:, 1:-1, 1:-1, 1:-1], interior_before,
    )
    plan = solver._ghost_sampling_plan
    filled_density = solver.fine_f.reshape(19, -1)[:, plan.target_flat].sum(dim=0)
    _, nz, ny, nx = solver.fine_f.shape
    local_z = torch.div(plan.target_flat, ny * nx, rounding_mode="floor")
    remainder = plan.target_flat % (ny * nx)
    local_y = torch.div(remainder, nx, rounding_mode="floor")
    local_x = remainder % nx
    fine_z = (config.box.z0 * 2 - 1 + local_z + 0.5) / 2.0 - 0.5
    fine_y = (config.box.y0 * 2 - 1 + local_y + 0.5) / 2.0 - 0.5
    fine_x = (config.box.x0 * 2 - 1 + local_x + 0.5) / 2.0 - 0.5
    expected = 1.0 + 1e-3 * fine_x + 2e-3 * fine_y + 3e-3 * fine_z
    expected = expected.to(filled_density)
    # The package lattice weights are intentionally stored in float32.
    torch.testing.assert_close(filled_density, expected, atol=2e-7, rtol=0.0)


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

    def advance(f: torch.Tensor, tau: float, level: int, substep: int) -> AMRAdvanceResult:
        del level, substep
        post = collide_mrt3d(f, tau=tau)
        return AMRAdvanceResult(stream3d(post), post)

    ledger = solver.step(advance)
    reference_by_q = reference.sum(dim=(1, 2, 3))
    actual_by_q = solver.coarse_f.sum(dim=(1, 2, 3))
    reference_mass = reference_by_q.sum()
    actual_mass = actual_by_q.sum()
    c = C19.to(reference_by_q)
    reference_momentum = (reference_by_q[:, None] * c).sum(dim=0)
    actual_momentum = (actual_by_q[:, None] * c).sum(dim=0)
    assert actual_mass == pytest.approx(float(reference_mass), abs=1e-4)
    assert torch.allclose(actual_momentum, reference_momentum, atol=1e-4, rtol=0.0)
    assert abs(ledger.mass_residual) < 1e-4


def test_face_local_interface_remains_finite_over_repeated_pulse_crossing() -> None:
    """A smooth convected perturbation may cross the interface repeatedly."""
    shape = (12, 14, 18)
    z, y, x = torch.meshgrid(
        torch.arange(shape[0]), torch.arange(shape[1]), torch.arange(shape[2]),
        indexing="ij",
    )
    rho = 1.0 + 1e-3 * torch.exp(
        -((x - 5.0) ** 2 + (y - 7.0) ** 2 + (z - 6.0) ** 2) / 8.0,
    )
    ux = torch.full(shape, 0.03)
    zero = torch.zeros(shape)
    coarse = equilibrium3d(rho, ux, zero, zero)
    solver = StaticBlockAMR3D(
        coarse,
        StaticBlockAMRConfig(
            BoxRegion(x0=6, x1=13, y0=3, y1=11, z0=3, z1=9),
            tau_coarse=0.56,
        ),
    )
    initial_mass = float(solver.coarse_f.sum())
    maximum_residual = 0.0

    def advance(
        state: torch.Tensor, tau: float, level: int, substep: int,
    ) -> AMRAdvanceResult:
        del level, substep
        post = collide_mrt3d(state, tau=tau)
        return AMRAdvanceResult(stream3d(post), post)

    for _ in range(50):
        ledger = solver.step(advance)
        maximum_residual = max(maximum_residual, abs(ledger.mass_residual))
    final_mass = float(solver.coarse_f.sum())
    assert bool(torch.isfinite(solver.coarse_f).all())
    assert float(solver.coarse_f.min()) > 0.0
    assert abs(final_mass - initial_mass) / initial_mass < 3e-7
    assert maximum_residual < 2e-10
