"""BFL vs Staircase comparison for SUBOFF bare_hull (D3Q19 MRT+Smag).

Uses fast vectorized operations from solver3d/boundaries3d.
Compares standard bounce_back_cells_3d (staircase) vs BFL interpolated
bounce-back with analytical DARPA SUBOFF q-values.

Reports Ct_total at steps 200/400/600/800/1000.
"""

from __future__ import annotations

import math
import time

import pytest
import torch

from tensorlbm.d3q19 import C as C19, equilibrium3d, OPPOSITE as OPP19
from tensorlbm.solver3d import collide_mrt3d, correct_mass3d, stream3d
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask
from tensorlbm.interpolated_bc import bouzidi_bounce_back_3d
from tensorlbm.interpolated_bc_suboff import compute_q_suboff
from tensorlbm.interpolated_bc_suboff import _suboff_radius_norm_torch
from tensorlbm.bfl_d3q19 import bouzidi_bounce_back_d3q19
from tensorlbm.wall_model import compute_bfl_link_normal
from tensorlbm.suboff_cad import suboff_radius_profile
from tensorlbm.suboff_resistance import _voxel_wetted_area

C19_SHIFTS = [(int(C19[q, 0]), int(C19[q, 1]), int(C19[q, 2])) for q in range(19)]
OPP_LIST = [int(x) for x in OPP19.tolist()]


def test_bfl_analytical_profile_matches_real_suboff_geometry() -> None:
    """BFL intersections and the CAD mask must describe the same hull."""
    xi = torch.linspace(-0.05, 1.05, 4097, dtype=torch.float64)
    actual = _suboff_radius_norm_torch(xi, SuboffConfig()).cpu().numpy()
    expected = suboff_radius_profile(xi.cpu().numpy(), SuboffConfig())
    assert abs(actual - expected).max() < 1e-10


def test_bfl_link_masks_follow_d3q19_xyz_directions() -> None:
    """Each BFL mask must mark the solid neighbour in that D3Q19 link."""
    nx, ny, nz, length = 80, 40, 40, 32.0
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(
        SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=length, device="cpu",
        config=SuboffConfig(),
    )
    masks, _ = compute_q_suboff(
        nx, ny, nz, cx, cy, cz, length, device="cpu",
    )
    for direction in range(1, 19):
        dcx, dcy, dcz = (int(v) for v in C19[direction].tolist())
        neighbour = torch.roll(
            solid, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2),
        )
        expected = ~solid & neighbour
        assert torch.equal(masks[direction], expected), direction


def test_bfl_q_reuses_solver_cad_mask_without_changing_links() -> None:
    nx, ny, nz, length = 80, 40, 40, 32.0
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid, _ = build_suboff_mask(
        "bare_hull", nx, ny, nz, cx=cx, cy=cy, cz=cz,
        length=length, device="cpu",
    )
    built_mask, built_q = compute_q_suboff(
        nx, ny, nz, cx, cy, cz, length, device="cpu",
    )
    reused_mask, reused_q = compute_q_suboff(
        nx, ny, nz, cx, cy, cz, length, device="cpu",
        solid_mask=solid,
    )

    assert torch.equal(reused_mask, built_mask)
    assert torch.equal(reused_q, built_q)
    with pytest.raises(ValueError, match="solid_mask"):
        compute_q_suboff(
            nx, ny, nz, cx, cy, cz, length, device="cpu",
            solid_mask=solid.float(),
        )


def test_full_hull_uses_halfway_links_on_voxel_appendages() -> None:
    nx, ny, nz, length = 120, 60, 60, 48.0
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    bare, _ = build_suboff_mask(
        "bare_hull", nx, ny, nz, cx=cx, cy=cy, cz=cz,
        length=length, device="cpu",
    )
    full, _ = build_suboff_mask(
        "full", nx, ny, nz, cx=cx, cy=cy, cz=cz,
        length=length, device="cpu",
    )
    masks, q = compute_q_suboff(
        nx, ny, nz, cx, cy, cz, length, hull_type="full", device="cpu",
    )
    appendage_links = 0
    for direction in range(1, 19):
        dcx, dcy, dcz = (int(v) for v in C19[direction].tolist())
        full_nb = torch.roll(full, (-dcz, -dcy, -dcx), (0, 1, 2))
        bare_nb = torch.roll(bare, (-dcz, -dcy, -dcx), (0, 1, 2))
        links = masks[direction] & full_nb & ~bare_nb
        appendage_links += int(links.sum())
        assert torch.all(q[direction][links] == 0.5)
    assert appendage_links > 0


def test_mature_bfl_reconstructs_unknown_from_outgoing_fluid_population() -> None:
    """At q=.5, BFL is exact half-way bounce-back, not solid streaming."""
    f = torch.zeros((19, 3, 3, 5), dtype=torch.float64)
    f_prev = torch.zeros_like(f)
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    direction = 1  # c=(+1,0,0), unknown is direction 2
    masks[direction, 1, 1, 2] = True
    f[2, 1, 1, 2] = 99.0  # streamed-from-solid value must be discarded
    f_prev[direction, 1, 1, 2] = 7.0

    out = bouzidi_bounce_back_d3q19(f, f_prev, masks, q)
    assert out[2, 1, 1, 2].item() == 7.0


def test_mature_bfl_uses_bouzidi_branches() -> None:
    f = torch.zeros((19, 3, 3, 6), dtype=torch.float64)
    f_prev = torch.zeros_like(f)
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    direction = 1
    # q<.5 at x=2: 2q*f_d(x)+(1-2q)*f_d(x-c_d)
    masks[direction, 1, 1, 2] = True
    q[direction, 1, 1, 2] = 0.25
    f_prev[direction, 1, 1, 2] = 8.0
    f_prev[direction, 1, 1, 1] = 4.0
    # q>.5 at x=4: f_d/(2q)+(2q-1)/(2q)*f_opp
    masks[direction, 1, 1, 4] = True
    q[direction, 1, 1, 4] = 0.75
    f_prev[direction, 1, 1, 4] = 9.0
    f_prev[2, 1, 1, 4] = 3.0

    out = bouzidi_bounce_back_d3q19(f, f_prev, masks, q)
    assert out[2, 1, 1, 2].item() == 6.0
    assert out[2, 1, 1, 4].item() == 7.0


def test_wall_model_slip_bfl_preserves_uniform_tangential_flow() -> None:
    """Moving BFL at local tangent speed is slip, not a second no-slip wall."""
    shape = (3, 3, 3)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f_prev = equilibrium3d(rho, ux, zero, zero)
    f_streamed = f_prev.clone()
    masks = torch.zeros_like(f_prev, dtype=torch.bool)
    q = torch.full_like(f_prev, 0.5)
    cell = (1, 1, 1)
    # Plane wall normal +y: outgoing links c_y>0 cross the wall.
    for direction in (3, 7, 10):
        masks[(direction,) + cell] = True

    slip = bouzidi_bounce_back_d3q19(
        f_streamed, f_prev, masks, q,
        wall_velocity=(ux, zero, zero), wall_density=rho,
    )
    stationary = bouzidi_bounce_back_d3q19(
        f_streamed, f_prev, masks, q,
    )

    assert torch.allclose(slip[(slice(None),) + cell], f_prev[(slice(None),) + cell], atol=1e-14)
    assert not torch.allclose(stationary[(slice(None),) + cell], f_prev[(slice(None),) + cell])

    _, slip_force = bouzidi_bounce_back_d3q19(
        f_streamed, f_prev, masks, q,
        wall_velocity=(ux, zero, zero), wall_density=rho,
        return_force=True, force_frame="wall",
    )
    _, stationary_force = bouzidi_bounce_back_d3q19(
        f_streamed, f_prev, masks, q, return_force=True,
    )
    assert slip_force[0] == pytest.approx(0.0, abs=1e-14)
    assert stationary_force[0] > 0.0


def test_bfl_link_force_decomposition_closes_flat_wall_impulse() -> None:
    shape = (3, 3, 3)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f_prev = equilibrium3d(rho, ux, zero, zero)
    masks = torch.zeros_like(f_prev, dtype=torch.bool)
    q = torch.full_like(f_prev, 0.5)
    cell = (1, 1, 1)
    for direction in (3, 7, 10):
        masks[(direction,) + cell] = True
    nx_n = torch.zeros(shape, dtype=torch.float64)
    ny_n = torch.zeros(shape, dtype=torch.float64)
    nz_n = torch.zeros(shape, dtype=torch.float64)
    ny_n[cell] = 1.0

    _, force, decomposition = bouzidi_bounce_back_d3q19(
        f_prev.clone(),
        f_prev,
        masks,
        q,
        return_force=True,
        force_normals=(nx_n, ny_n, nz_n),
        return_force_decomposition=True,
    )

    assert decomposition.active_links == 3
    assert decomposition.decomposed_links == 3
    assert decomposition.undecomposed_links == 0
    assert decomposition.coverage_fraction == pytest.approx(1.0)
    assert decomposition.unresolved_force == pytest.approx((0.0, 0.0, 0.0))
    assert decomposition.stationary_interpolation_force == pytest.approx(force)
    assert decomposition.moving_wall_population_correction_force == pytest.approx(
        (0.0, 0.0, 0.0),
    )
    assert decomposition.frame_correction_force == pytest.approx(
        (0.0, 0.0, 0.0),
    )
    assert decomposition.total_force == pytest.approx(force, abs=1e-14)
    assert decomposition.normal_force[0] == pytest.approx(0.0, abs=1e-14)
    assert decomposition.tangential_force[0] == pytest.approx(force[0], abs=1e-14)
    assert decomposition.normal_force[1] == pytest.approx(force[1], abs=1e-14)
    assert decomposition.tangential_force[1] == pytest.approx(0.0, abs=1e-14)
    assert decomposition.maximum_closure_error == pytest.approx(0.0, abs=1e-14)
    assert decomposition.maximum_relative_closure_error == pytest.approx(0.0)
    assert decomposition.maximum_component_closure_error == pytest.approx(0.0)


def test_bfl_link_force_decomposition_requires_normals_and_force() -> None:
    f = torch.zeros((19, 2, 2, 2))
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    with pytest.raises(ValueError, match="return_force"):
        bouzidi_bounce_back_d3q19(
            f,
            f,
            masks,
            q,
            return_force_decomposition=True,
        )
    with pytest.raises(ValueError, match="force_normals"):
        bouzidi_bounce_back_d3q19(
            f,
            f,
            masks,
            q,
            return_force=True,
            return_force_decomposition=True,
        )

    masks[1, 1, 1, 1] = True
    _, force, decomposition = bouzidi_bounce_back_d3q19(
        f,
        f,
        masks,
        q,
        return_force=True,
        force_normals=(f[0], f[0], f[0]),
        return_force_decomposition=True,
    )
    assert decomposition.active_links == 1
    assert decomposition.decomposed_links == 0
    assert decomposition.coverage_fraction == pytest.approx(0.0)
    assert decomposition.unresolved_force == pytest.approx(force)
    assert decomposition.maximum_closure_error == pytest.approx(0.0)
    assert decomposition.maximum_relative_closure_error == pytest.approx(0.0)


def test_moving_bfl_laboratory_force_closes_nonequilibrium_control_volume() -> None:
    """Wall-frame correction must not replace the conservative force ledger."""
    from tensorlbm.boundaries3d import sphere_mask
    from tensorlbm.control_volume_force import box_control_volume, observe_control_volume_force
    from tensorlbm.interpolated_bc import compute_q_sphere
    from tensorlbm.solver3d import stream3d
    from tensorlbm.wall_model import compute_bfl_link_normal

    shape = (20, 20, 28)
    nx, ny, nz = 28, 20, 20
    cx, cy, cz, radius = 11.0, 10.0, 10.0, 3.5
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros_like(rho)
    old = equilibrium3d(rho, ux, zero, zero)
    solid = sphere_mask(
        nx, ny, nz, cx, cy, cz, radius, device=torch.device("cpu"),
    )
    masks, q = compute_q_sphere(
        nx, ny, nz, cx, cy, cz, radius, device=torch.device("cpu"),
    )
    # A deterministic non-equilibrium perturbation makes the two force frames
    # observably different while keeping every population positive.
    old = old.clone()
    old[7][masks[7]] *= 1.03
    streamed = stream3d(old)
    nx_n, ny_n, nz_n = compute_bfl_link_normal(masks)
    normal_speed = ux * nx_n
    wall_velocity = (
        ux - normal_speed * nx_n,
        -normal_speed * ny_n,
        -normal_speed * nz_n,
    )
    updated, laboratory_force = bouzidi_bounce_back_d3q19(
        streamed, old, masks, q,
        wall_velocity=wall_velocity, wall_density=rho,
        return_force=True, force_frame="laboratory",
    )
    _, wall_frame_force = bouzidi_bounce_back_d3q19(
        streamed, old, masks, q,
        wall_velocity=wall_velocity, wall_density=rho,
        return_force=True, force_frame="wall",
    )
    cv = box_control_volume(
        shape, x0=5, x1=18, y0=4, y1=17, z0=4, z1=17,
    )
    cv_force = observe_control_volume_force(
        old, updated, old, cv, solid=solid,
    ).force_on_body

    assert cv_force[0].item() == pytest.approx(laboratory_force[0], abs=1e-11)
    assert wall_frame_force[0] != pytest.approx(laboratory_force[0], abs=1e-7)


def test_wall_model_startup_ramps_relative_normal_velocity_not_bfl_population() -> None:
    """Activation zero must be a co-moving impermeable wall, not a leaky blend."""
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 3, 3)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.04, dtype=torch.float64)
    uy = torch.full(shape, 0.03, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f_prev = equilibrium3d(rho, ux, uy, zero)
    masks = torch.zeros_like(f_prev, dtype=torch.bool)
    q = torch.full_like(f_prev, 0.5)
    solid = torch.zeros(shape, dtype=torch.bool)
    near = torch.zeros(shape, dtype=torch.bool)
    cell = (1, 1, 1)
    near[cell] = True
    # Plane wall normal +y: all positive-y links enter the solid.
    for direction in (3, 7, 10, 15, 17):
        masks[(direction,) + cell] = True

    out, friction, pressure = bfl_wall_function_3d(
        f_prev.clone(), f_prev, solid, 0.02, masks, q,
        near_mask=near, bfl_wall_mode="wall_model_slip",
        wall_activation=0.0,
    )
    assert torch.allclose(
        out[(slice(None),) + cell], f_prev[(slice(None),) + cell], atol=1e-14,
    )
    assert friction == pytest.approx(0.0, abs=1e-14)
    # D3Q19 weights are float32 constants, even for this float64 state.
    assert pressure == pytest.approx(0.0, abs=2e-9)


def test_wall_normal_and_shear_activation_are_independent() -> None:
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 3, 3)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.04, dtype=torch.float64)
    uy = torch.full(shape, 0.03, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f_prev = equilibrium3d(rho, ux, uy, zero)
    masks = torch.zeros_like(f_prev, dtype=torch.bool)
    q = torch.full_like(f_prev, 0.5)
    solid = torch.zeros(shape, dtype=torch.bool)
    near = torch.zeros(shape, dtype=torch.bool)
    cell = (1, 1, 1)
    near[cell] = True
    for direction in (3, 7, 10, 15, 17):
        masks[(direction,) + cell] = True

    out, friction, _ = bfl_wall_function_3d(
        f_prev.clone(), f_prev, solid, 0.02, masks, q,
        near_mask=near,
        bfl_wall_mode="wall_model_slip",
        wall_activation=0.0,
        wall_normal_activation=1.0,
        wall_shear_activation=0.0,
    )

    assert not torch.allclose(
        out[(slice(None),) + cell], f_prev[(slice(None),) + cell], atol=1e-14,
    )
    assert friction == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize(
    ("keyword", "message"),
    (
        ("wall_normal_activation", "wall_normal_activation"),
        ("wall_shear_activation", "wall_shear_activation"),
    ),
)
def test_independent_wall_activation_rejects_out_of_range(
    keyword: str,
    message: str,
) -> None:
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 3, 3)
    rho = torch.ones(shape)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero)
    options = {keyword: 1.1}

    with pytest.raises(ValueError, match=message):
        bfl_wall_function_3d(
            f.clone(), f, torch.zeros(shape, dtype=torch.bool), 0.02,
            torch.zeros_like(f, dtype=torch.bool), torch.full_like(f, 0.5),
            **options,
        )


def test_guo_wall_source_momentum_equals_reported_wall_traction() -> None:
    """Wall distance affects u_tau, but must not multiply integrated force."""
    from tensorlbm.d3q19 import C as C19
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 5, 5)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium3d(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    near = torch.zeros(shape, dtype=torch.bool)
    near[:, 1, :] = True
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)

    out, friction, _ = bfl_wall_function_3d(
        f.clone(), f, solid, 1e-3, masks, q,
        near_mask=near, apply_bfl=False, use_guo=True,
        wall_activation=1.0, y_val=0.5,
    )
    population_change = (out - f).sum(dim=(1, 2, 3))
    fluid_momentum_change = (
        population_change[:, None] * C19.to(f)
    ).sum(dim=0)
    assert fluid_momentum_change[0].item() == pytest.approx(
        -friction, abs=5e-11,
    )


def test_exchange_location_wall_source_is_conservative_and_changes_stress() -> None:
    """Exchange sampling changes the law input without changing force accounting."""
    from tensorlbm.d3q19 import C as C19
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 8, 7)
    rho = torch.ones(shape, dtype=torch.float64)
    y = torch.arange(shape[1], dtype=torch.float64).view(1, -1, 1)
    ux = (0.01 + 0.01 * y).expand(shape).clone()
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium3d(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    near = torch.zeros(shape, dtype=torch.bool)
    near[:, 1, :] = True
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    for direction in (4, 8, 9, 16, 18):
        masks[direction, :, 1, :] = True
    nx = torch.zeros(shape, dtype=torch.float64)
    ny = torch.zeros(shape, dtype=torch.float64)
    nz = torch.zeros(shape, dtype=torch.float64)
    ny[:, 1, :] = 1.0

    local_out, local_friction, _ = bfl_wall_function_3d(
        f.clone(), f, solid, 1e-3, masks, q,
        near_mask=near, apply_bfl=False, wall_normals=(nx, ny, nz),
        wall_law="reichardt", y_val=0.5,
    )
    exchange_out, exchange_friction, _, diagnostics = bfl_wall_function_3d(
        f.clone(), f, solid, 1e-3, masks, q,
        near_mask=near, apply_bfl=False, wall_normals=(nx, ny, nz),
        wall_law="reichardt", stress_exchange_distance=2.0,
        return_wall_diagnostics=True,
    )
    assert exchange_friction != pytest.approx(local_friction, rel=1e-5)
    population_change = (exchange_out - f).sum(dim=(1, 2, 3))
    momentum_change = (population_change[:, None] * C19.to(f)).sum(dim=0)
    assert momentum_change[0].item() == pytest.approx(
        -exchange_friction, abs=5e-11,
    )
    assert torch.isfinite(exchange_out).all()
    assert torch.isfinite(local_out).all()
    assert diagnostics.mode == "exchange_location_guo"
    assert diagnostics.requested_nodes == diagnostics.active_nodes == 21
    assert diagnostics.rejected_fraction == pytest.approx(0.0)
    assert diagnostics.wall_distance_mean == pytest.approx(2.0)
    assert diagnostics.y_plus_min is not None and diagnostics.y_plus_min > 0.0
    assert diagnostics.y_plus_max is not None
    assert diagnostics.y_plus_summary is not None
    assert diagnostics.y_plus_summary["requested_samples"] == 21
    assert diagnostics.y_plus_summary["finite_samples"] == 21
    assert diagnostics.pressure_gradient_parameter_mean == pytest.approx(0.0)
    assert diagnostics.pressure_gradient_parameter_p95 == pytest.approx(0.0)
    assert diagnostics.pressure_gradient_parameter_max == pytest.approx(0.0)
    assert diagnostics.shear_force[0] == pytest.approx(exchange_friction)
    assert diagnostics.wall_shear_axial_profile is not None
    assert sum(
        item["signed_shear_x_sum_lu"]
        for item in diagnostics.wall_shear_axial_profile
    ) == pytest.approx(exchange_friction)
    assert diagnostics.link_force_decomposition is None


def test_wall_diagnostics_include_actual_bfl_link_force_decomposition() -> None:
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 5, 5)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.04, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium3d(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    near = torch.zeros(shape, dtype=torch.bool)
    near[:, 1, :] = True
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    for direction in (4, 8, 9, 16, 18):
        masks[direction, :, 1, :] = True
    nx_n = torch.zeros(shape, dtype=torch.float64)
    ny_n = torch.zeros(shape, dtype=torch.float64)
    nz_n = torch.zeros(shape, dtype=torch.float64)
    ny_n[:, 1, :] = 1.0
    ny_n[1, 1, 2] = 0.0

    _, _, pressure, diagnostics = bfl_wall_function_3d(
        f.clone(),
        f,
        solid,
        1.0e-3,
        masks,
        q,
        near_mask=near,
        wall_normals=(nx_n, ny_n, nz_n),
        bfl_wall_mode="wall_model_slip",
        return_wall_diagnostics=True,
    )

    decomposition = diagnostics.link_force_decomposition
    assert decomposition is not None
    assert decomposition["active_links"] == int(masks.sum().item())
    assert decomposition["coverage_fraction"] == pytest.approx(1.0)
    completion = decomposition["normal_completion"]
    assert completion["scheme"] == "geometry_normal_with_bfl_link_fallback_v1"
    assert completion["fallback_nodes"] == 1
    assert completion["fallback_links"] == 5
    assert completion["unresolved_nodes"] == 0
    total = decomposition["total_force"]
    normal = decomposition["normal_force"]
    tangential = decomposition["tangential_force"]
    assert total[0] == pytest.approx(pressure, abs=1e-12)
    for component in range(3):
        assert total[component] == pytest.approx(
            normal[component] + tangential[component], abs=1e-12,
        )
        assert total[component] == pytest.approx(
            decomposition["stationary_interpolation_force"][component]
            + decomposition["moving_wall_population_correction_force"][component]
            + decomposition["frame_correction_force"][component],
            abs=1e-12,
        )
    assert decomposition["maximum_closure_error"] < 1.0e-12
    assert decomposition["maximum_component_closure_error"] < 1.0e-12


def test_exchange_location_requires_positive_distance() -> None:
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 3, 3)
    rho = torch.ones(shape)
    zero = torch.zeros(shape)
    f = equilibrium3d(rho, zero, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    with pytest.raises(ValueError, match="stress_exchange_distance"):
        bfl_wall_function_3d(
            f, f, solid, 0.01, masks, q,
            stress_exchange_distance=0.0,
        )


def test_exchange_diagnostics_measure_tangential_pressure_gradient() -> None:
    from tensorlbm.wall_model import bfl_wall_function_3d

    shape = (3, 8, 7)
    x = torch.arange(shape[2], dtype=torch.float64).view(1, 1, -1)
    rho = (1.0 + 1.0e-3 * x).expand(shape).clone()
    ux = torch.full(shape, 0.03, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium3d(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    near = torch.zeros(shape, dtype=torch.bool)
    near[:, 1, :] = True
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    masks[4, :, 1, :] = True
    nx = torch.zeros(shape, dtype=torch.float64)
    ny = torch.zeros(shape, dtype=torch.float64)
    nz = torch.zeros(shape, dtype=torch.float64)
    ny[:, 1, :] = 1.0

    _, _, _, diagnostics = bfl_wall_function_3d(
        f.clone(), f, solid, 1.0e-3, masks, q,
        near_mask=near,
        apply_bfl=False,
        wall_normals=(nx, ny, nz),
        wall_law="reichardt",
        stress_exchange_distance=2.0,
        return_wall_diagnostics=True,
    )

    assert diagnostics.pressure_gradient_parameter_mean is not None
    assert diagnostics.pressure_gradient_parameter_mean > 0.0
    assert diagnostics.pressure_gradient_parameter_p95 is not None
    assert diagnostics.pressure_gradient_parameter_p95 > 0.0
    assert diagnostics.pressure_gradient_parameter_max is not None
    assert diagnostics.pressure_gradient_parameter_max >= (
        diagnostics.pressure_gradient_parameter_p95
    )


def test_d3q27_guo_wall_source_momentum_equals_reported_wall_traction() -> None:
    """D3Q27 uses the same area/volume traction contract as D3Q19."""
    from tensorlbm.d3q27 import C as C27, equilibrium27
    from tensorlbm.wall_model import bfl_wall_function_d3q27

    shape = (3, 5, 5)
    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium27(rho, ux, zero, zero)
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0, :] = True
    near = torch.zeros(shape, dtype=torch.bool)
    near[:, 1, :] = True
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    area = torch.full(shape, 0.75, dtype=torch.float64)

    out, friction, _ = bfl_wall_function_d3q27(
        f.clone(), f, solid, 1e-3, masks, q,
        near_mask=near, apply_bfl=False, area_weight=area,
        wall_activation=0.4,
    )
    population_change = (out - f).sum(dim=(1, 2, 3))
    fluid_momentum_change = (
        population_change[:, None] * C27.to(f)
    ).sum(dim=0)
    assert fluid_momentum_change[0].item() == pytest.approx(
        -friction, abs=5e-11,
    )


def test_mature_wall_solver_implements_finite_distinct_musker_law() -> None:
    from tensorlbm.wall_model import _solve_wall_law

    speed = torch.tensor([0.06], dtype=torch.float64)
    near = torch.tensor([True])
    musker = _solve_wall_law(speed, 1e-5, 0.5, "musker", near)
    log = _solve_wall_law(speed, 1e-5, 0.5, "log", near)
    assert torch.isfinite(musker).all()
    assert musker.item() > 0.0
    assert musker.item() != pytest.approx(log.item(), rel=1e-5)
    with pytest.raises(ValueError, match="wall_law"):
        _solve_wall_law(speed, 1e-5, 0.5, "unknown", near)

    high_y_plus = _solve_wall_law(
        speed.float(), 5e-7, 0.5, "musker", near,
    )
    assert torch.isfinite(high_y_plus).all()
    assert high_y_plus.item() > 0.0


def test_bfl_link_normal_recovers_flat_wall_direction() -> None:
    masks = torch.zeros((19, 3, 3, 3), dtype=torch.bool)
    cell = (1, 1, 1)
    for direction in (3, 7, 10, 15, 17):  # all links with c_y=+1
        masks[(direction,) + cell] = True
    nx_n, ny_n, nz_n = compute_bfl_link_normal(masks)
    assert nx_n[cell].item() == pytest.approx(0.0, abs=1e-7)
    assert ny_n[cell].item() == pytest.approx(1.0, abs=1e-7)
    assert nz_n[cell].item() == pytest.approx(0.0, abs=1e-7)


def test_moving_bfl_requires_density() -> None:
    f = torch.zeros((19, 2, 2, 2))
    masks = torch.zeros_like(f, dtype=torch.bool)
    q = torch.full_like(f, 0.5)
    velocity = (torch.zeros_like(f[0]),) * 3
    with pytest.raises(ValueError, match="wall_density"):
        bouzidi_bounce_back_d3q19(
            f, f, masks, q, wall_velocity=velocity,
        )


def test_zero_boundary_fraction_is_transparent() -> None:
    f = torch.rand((19, 3, 3, 3))
    f_prev = torch.rand_like(f)
    masks = torch.zeros_like(f, dtype=torch.bool)
    masks[7, 1, 1, 1] = True
    q = torch.full_like(f, 0.31)
    out = bouzidi_bounce_back_d3q19(
        f, f_prev, masks, q, boundary_fraction=0.0,
    )
    assert torch.equal(out, f)


def far_field_bc_19(f: torch.Tensor, u_in: float = 0.06) -> torch.Tensor:
    nz, ny, nx = f.shape[1], f.shape[2], f.shape[3]
    rho1 = torch.ones(nz, ny, nx, dtype=f.dtype, device=f.device)
    feq = equilibrium3d(
        rho1, torch.full_like(rho1, u_in), torch.zeros_like(rho1), torch.zeros_like(rho1)
    )
    f = f.clone()
    f[:, :, :, 0] = feq[:, :, :, 0]
    f[:, :, :, -1] = f[:, :, :, -2]
    f[:, 0, :, :] = feq[:, 0, :, :]
    f[:, -1, :, :] = feq[:, -1, :, :]
    f[:, :, 0, :] = feq[:, :, 0, :]
    f[:, :, -1, :] = feq[:, :, -1, :]
    return f


def compute_forces_me(
    f_post: torch.Tensor,
    f_pre: torch.Tensor,
    solid: torch.Tensor,
    c_dev: torch.Tensor,
) -> float:
    """Momentum exchange drag force on solid."""
    total_fx = 0.0
    for d in range(1, 19):
        sx, sy, sz = C19_SHIFTS[d]
        nb_solid = torch.roll(solid, shifts=(-sz, -sy, -sx), dims=(0, 1, 2))
        fluid_bdry = ~solid & nb_solid
        if not fluid_bdry.any():
            continue
        delta = f_pre[d][fluid_bdry] - f_post[d][fluid_bdry]
        total_fx += (delta * c_dev[d, 0]).sum().item()
    return total_fx


def run_simulation(
    *,
    use_bfl: bool,
    nx: int,
    ny: int,
    nz: int,
    hull_length: float,
    u_in: float = 0.06,
    re: float = 1e5,
    cs_smag: float = 0.05,
    n_steps: int = 1000,
    device: str = "cpu",
) -> dict:
    dev = torch.device(device)
    nu_lat = u_in * hull_length / re
    tau = 3.0 * nu_lat + 0.5
    cx_g, cy_g, cz_g = nx * 0.35, ny / 2.0, nz / 2.0
    config = SuboffConfig()

    solid, _ = build_suboff_mask(
        SuboffHullType.BARE_HULL,
        nx=nx,
        ny=ny,
        nz=nz,
        cx=cx_g,
        cy=cy_g,
        cz=cz_g,
        length=hull_length,
        device="cpu",
        config=config,
    )
    solid = solid.to(dev)

    S = _voxel_wetted_area(solid, 1.0)
    dyn_p_S = 0.5 * 1.0 * u_in**2 * S
    c_dev = C19.to(dev).float()[:19]

    bfl_mask = None
    bfl_q = None
    if use_bfl:
        print("  BFL suboff q-field...")
        t_q = time.time()
        bfl_mask, bfl_q = compute_q_suboff(
            nx,
            ny,
            nz,
            cx_g,
            cy_g,
            cz_g,
            hull_length,
            device=dev,
        )
        n_links = int(bfl_mask.sum().item())
        print(f"  Q-field: {n_links} links ({time.time() - t_q:.1f}s)")

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(rho0.sum().item())

    label = "BFL" if use_bfl else "STAIRCASE"
    print(
        f"\n=== {label}: Re={re:.0e} tau={tau:.4f} {nx}x{ny}x{nz} L={hull_length} Cs={cs_smag} ==="
    )

    results: dict[int, dict[str, float]] = {}
    t0 = time.time()
    fx_samples: list[float] = []

    for step in range(1, n_steps + 1):
        # Reset solid
        f_eq = equilibrium3d(
            rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0)
        )
        f[:, solid] = f_eq[:, solid]

        # Collide
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=cs_smag)
        f_pre = f.clone()

        # Stream
        f = stream3d(f)

        # Far-field BC
        f = far_field_bc_19(f, u_in=u_in)

        # Wall BC
        if use_bfl and bfl_mask is not None:
            # BFL on boundary links
            for d in range(1, 19):
                if bfl_mask[d].any():
                    f = bouzidi_bounce_back_3d(f, f_pre, bfl_mask[d], bfl_q[d], d)
        else:
            # Standard bounce-back on solid cells
            f = bounce_back_cells_3d(f, solid)

        # Mass correction
        if step % 200 == 0:
            f = correct_mass3d(f, initial_mass)

        if step > 50:
            fx = compute_forces_me(f, f_pre, solid, c_dev)
            fx_samples.append(fx)

        if step in (200, 400, 500, 600, 800, 1000):
            ct = sum(fx_samples[-50:]) / max(len(fx_samples[-50:]), 1) / dyn_p_S
            results[int(step)] = {"Ct_total": float(ct)}
            print(f"  step {step:4d}: Ct={float(ct):.6f} ({time.time() - t0:.0f}s)")

    print(f"  Done in {time.time() - t0:.1f}s")
    return results


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cpu")
    p.add_argument("--nx", type=int, default=96)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--nz", type=int, default=48)
    p.add_argument("--hull-length", type=float, default=None)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--re", type=float, default=1e5)
    p.add_argument("--bfl-only", action="store_true")
    p.add_argument("--staircase-only", action="store_true")
    args = p.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    hull_length = args.hull_length or (0.6 * nx)

    print(f"SUBOFF BFL vs Staircase: {nx}x{ny}x{nz} L={hull_length} Re={args.re:.0e}")

    all_results: dict[str, dict] = {}

    if not args.bfl_only:
        all_results["staircase"] = run_simulation(
            use_bfl=False,
            nx=nx,
            ny=ny,
            nz=nz,
            hull_length=hull_length,
            re=args.re,
            n_steps=args.steps,
            device=args.device,
        )

    if not args.staircase_only:
        all_results["bfl"] = run_simulation(
            use_bfl=True,
            nx=nx,
            ny=ny,
            nz=nz,
            hull_length=hull_length,
            re=args.re,
            n_steps=args.steps,
            device=args.device,
        )

    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    steps_list = [200, 400, 500, 600, 800, 1000]
    keys = sorted(all_results.keys())
    header = f"{'Step':>6}  " + "  ".join(f"{k.upper():>18}" for k in keys)
    print(header)
    for step in steps_list:
        parts = [f"{step:6d}"]
        for key in keys:
            ct = all_results[key].get(step, {}).get("Ct_total", float("nan"))
            parts.append(f"{ct:18.6f}")
        print("  ".join(parts))

    if len(keys) == 2:
        k0, k1 = keys[0], keys[1]
        print(f"\nStaircase vs BFL Ct_total:")
        for step in steps_list:
            if step in all_results[k0] and step in all_results[k1]:
                cs = all_results[k0][step]["Ct_total"]
                cb = all_results[k1][step]["Ct_total"]
                delta = cs - cb
                pct = delta / max(abs(cs), 1e-12) * 100
                print(
                    f"  step {step:4d}: stair={cs:.6f}  bfl={cb:.6f}  Δ={delta:.6f}  ({pct:+.1f}%)"
                )
