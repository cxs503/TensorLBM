"""Moment-balance gates for the Spalding exchange wall model (ladder rung 1).

The validation ladder in ``docs/mature-lbm-gap-analysis-2026-08-01.md`` requires
"manufactured wall-law states, plus zero-mass/exact-momentum force-source
moments on D3Q19 and D3Q27" before any wall-modelled force may be admitted.

The population assimilation in :func:`apply_spalding_exchange_wall_model` is

    f' = f_eq(rho, u_target) + s * (f - f_eq(rho, u))               (s = scale)
    target velocity: u_target = (u.n) n + u1 t

with ``u1`` the Spalding-law velocity at the first off-wall distance and ``n``
the outward wall normal.  Because

* ``sum_i W_i = 1`` and ``sum_i W_i c_i = 0`` (lattice symmetry) give
  ``sum_i f_eq(rho, u) = rho`` and ``sum_i c_i f_eq(rho, u) = rho u``,
* the non-equilibrium part ``f - f_eq`` carries zero mass and zero momentum,
  since ``rho, u`` were recovered from ``f`` itself,

the source has provably zero zeroth moment and first moment
``rho (u_target - u)``, *given exact lattice weights*.  These tests assert both
identities independently of the implementation: the expected target velocity is
rebuilt in the test from the public primitives
(:func:`sample_wall_exchange_velocity`, :func:`solve_spalding_friction_velocity`,
:func:`spalding_u_plus_from_y_plus`).

Measured residuals are recorded in
``docs/spalding_wall_moment_balance_2026-11-03.md``.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.spalding_wall_model import (
    apply_spalding_exchange_wall_model,
    sample_wall_exchange_velocity,
    solve_spalding_friction_velocity,
    spalding_u_plus_from_y_plus,
)

# The shared lattice constants are float32; after the wall module renormalizes
# them the only remaining deviation is D3Q19 second-moment quantization
# (sum W c_i c_i - 1/3 = -2.5e-9).  That sets the floor of both identities.
MASS_ATOL = 1.0e-9  # per boundary node, unnormalized rho ~ 1
MOMENTUM_ATOL = 1.0e-8  # relative to rho * |u_target - u| ~ 1e-1
TARGET_RTOL = 1.0e-12
# Relative leakage of the non-equilibrium part into the source momentum, set by
# the float32 lattice-weight second moment: |S2_ii - 1/3| = 2.5e-9 (D3Q19),
# ~0 for D3Q27.  Measured 2.9e-10 for D3Q19, 0 for D3Q27.
QUANTIZATION_FLOOR = {19: 1.0e-8, 27: 1.0e-13}


def _lattice(q: int):
    if q == 19:
        from tensorlbm.d3q19 import C, W, macroscopic3d

        return C, W, macroscopic3d
    from tensorlbm.d3q27 import C, W, macroscopic27

    return C, W, macroscopic27


def _wall_state(q: int, *, seed: int = 26, shape=(5, 7, 9)):
    """Flat wall below a single fluid boundary cell, with non-equilibrium content."""
    C, _, _ = _lattice(q)
    c = C.double()
    generator = torch.Generator().manual_seed(seed + q)
    mask = torch.zeros((q,) + shape, dtype=torch.bool)
    cell = (2, 3, 4)
    for direction in range(q):
        if c[direction, 1] < 0.0:  # links pointing into the solid below
            mask[(direction,) + cell] = True
    nx = torch.zeros(shape, dtype=torch.float64)
    ny = torch.zeros(shape, dtype=torch.float64)
    nz = torch.zeros(shape, dtype=torch.float64)
    ny[cell] = 1.0  # outward normal points up into the fluid
    q_field = torch.full((q,) + shape, 0.5, dtype=torch.float64)
    return c, mask, (nx, ny, nz), q_field, cell, generator


def _manufactured_distributions(q: int, generator: torch.Generator, shape=(5, 7, 9)):
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.d3q27 import equilibrium27

    rho = torch.ones(shape, dtype=torch.float64)
    ux = torch.full(shape, 0.06, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    equilibrium = equilibrium3d if q == 19 else equilibrium27
    f_eq = equilibrium(rho, ux, zero, zero).double()
    # Genuine non-equilibrium content: the assimilation must not lose it in a way
    # that breaks the source moments.
    noise = 0.02 * torch.randn((q,) + shape, generator=generator, dtype=torch.float64)
    return f_eq + noise


def _expected_target(q, f, mask, normals, q_field, nu, exchange_distance):
    """Rebuild the modelled target velocity from public primitives only."""
    _, _, macroscopic = _lattice(q)
    _, ux, uy, uz = macroscopic(f)
    samples = sample_wall_exchange_velocity(
        (ux, uy, uz),
        mask,
        q_field,
        normals,
        exchange_distance=exchange_distance,
    )
    boundary = samples.boundary
    normal = torch.stack((samples.normal_x, samples.normal_y, samples.normal_z))
    sampled = torch.stack(
        (samples.velocity_x, samples.velocity_y, samples.velocity_z),
    )
    u_old = torch.stack((ux[boundary], uy[boundary], uz[boundary]))
    rho_old = f.sum(dim=0)[boundary]
    normal_flux = (sampled * normal).sum(dim=0)
    tangential = sampled - normal_flux[None] * normal
    tangential_speed = tangential.norm(dim=0).clamp_min(1e-20)
    tangent = tangential / tangential_speed
    u_tau = solve_spalding_friction_velocity(tangential_speed, samples.y2, nu)
    u1 = spalding_u_plus_from_y_plus(samples.y1 * u_tau / nu) * u_tau
    target = (u_old * normal).sum(dim=0)[None] * normal + u1[None] * tangent
    return boundary, rho_old, u_old, target


@pytest.mark.parametrize("q_count", (19, 27))
def test_population_source_has_zero_zeroth_moment(q_count: int) -> None:
    """The assimilation injects no net mass: sum_i (f' - f) = 0 per node."""
    c, mask, normals, q_field, cell, generator = _wall_state(q_count)
    f = _manufactured_distributions(q_count, generator)
    nu = 1.0e-4

    out, diagnostics = apply_spalding_exchange_wall_model(
        f,
        mask,
        q_field,
        normals,
        nu=nu,
        exchange_distance=2.0,
    )

    assert diagnostics.boundary_nodes == 1
    mass_source = (out - f).sum(dim=0)
    # Zero on the boundary node and everywhere else (only that node is touched).
    assert float(mass_source.abs().max()) <= MASS_ATOL
    assert float(mass_source[cell]) == pytest.approx(0.0, abs=MASS_ATOL)
    # Densities re-read from the updated populations agree with the source
    # identity: no systematic mass drift.
    assert float(out.sum() - f.sum()) == pytest.approx(0.0, abs=MASS_ATOL)


@pytest.mark.parametrize("q_count", (19, 27))
def test_population_source_momentum_equals_rho_delta_velocity(q_count: int) -> None:
    """First moment of the source equals rho (u_target - u) exactly."""
    c, mask, normals, q_field, cell, generator = _wall_state(q_count)
    f = _manufactured_distributions(q_count, generator)
    nu = 1.0e-4

    out, _ = apply_spalding_exchange_wall_model(
        f,
        mask,
        q_field,
        normals,
        nu=nu,
        exchange_distance=2.0,
    )

    boundary, rho_old, u_old, target = _expected_target(
        q_count,
        f,
        mask,
        normals,
        q_field,
        nu,
        2.0,
    )
    injected = torch.stack(
        [(c[:, axis, None, None, None] * (out - f)).sum(dim=0) for axis in range(3)],
    )[:, boundary]
    expected = rho_old[None] * (target - u_old)
    scale = float(expected.abs().max())
    assert scale > 0.05  # the manufactured state carries real shear
    assert float((injected - expected).abs().max()) <= MOMENTUM_ATOL * scale
    # Normal momentum is preserved (wall model acts tangentially only).
    assert float((injected - expected).abs().max()) < 1.0e-7 * scale


@pytest.mark.parametrize("q_count", (19, 27))
def test_source_moments_independent_of_nonequilibrium_scale(q_count: int) -> None:
    """``nonequilibrium_scale`` must not leak into the source moments.

    The non-equilibrium part carries zero mass/momentum only if the lattice
    weights reproduce the second moment exactly.  The shared constants are
    float32, so D3Q19 admits a residual of ``rho |S2 - I/3| |u| ~ 3e-10``
    (see ``QUANTIZATION_FLOOR``); D3Q27 is exact.  This test pins that floor
    instead of hiding it behind a loose global tolerance.
    """
    c, mask, normals, q_field, _, generator = _wall_state(q_count)
    f = _manufactured_distributions(q_count, generator)

    def moments(scale: float):
        out, _ = apply_spalding_exchange_wall_model(
            f,
            mask,
            q_field,
            normals,
            nu=1.0e-4,
            exchange_distance=2.0,
            nonequilibrium_scale=scale,
        )
        delta = out - f
        return (
            float(delta.sum(dim=0).abs().max()),
            torch.stack(
                [(c[:, axis, None, None, None] * delta).sum(dim=0) for axis in range(3)],
            ),
        )

    mass_zero, momentum_zero = moments(0.0)
    mass_half, momentum_half = moments(0.5)
    mass_full, momentum_full = moments(1.0)

    for mass in (mass_zero, mass_half, mass_full):
        assert mass <= MASS_ATOL
    reference = float(momentum_full.abs().max())
    assert reference > 0.05
    floor = QUANTIZATION_FLOOR[q_count] * reference
    for momentum in (momentum_zero, momentum_half):
        assert float((momentum - momentum_full).abs().max()) <= floor


@pytest.mark.parametrize("q_count", (19, 27))
@pytest.mark.parametrize("activation", (0.25, 0.5, 0.75))
def test_source_moments_are_linear_in_activation(
    q_count: int,
    activation: float,
) -> None:
    """Mass source stays zero and momentum scales linearly with ``activation``."""
    c, mask, normals, q_field, _, generator = _wall_state(q_count)
    f = _manufactured_distributions(q_count, generator)

    def moments(act: float):
        out, _ = apply_spalding_exchange_wall_model(
            f,
            mask,
            q_field,
            normals,
            nu=1.0e-4,
            exchange_distance=2.0,
            activation=act,
        )
        delta = out - f
        return (
            float(delta.sum(dim=0).abs().max()),
            torch.stack(
                [(c[:, axis, None, None, None] * delta).sum(dim=0) for axis in range(3)],
            ),
        )

    mass_full, momentum_full = moments(1.0)
    mass_act, momentum_act = moments(activation)
    assert mass_full <= MASS_ATOL
    assert mass_act <= MASS_ATOL
    assert momentum_full.abs().max() > 0.05
    error = (momentum_act - activation * momentum_full).abs().max()
    assert float(error) <= MOMENTUM_ATOL * float(momentum_full.abs().max())


@pytest.mark.parametrize("q_count", (19, 27))
def test_manufactured_spalding_profile_is_a_fixed_point(q_count: int) -> None:
    """A field already satisfying the Spalding law is left untouched.

    The wall model solves for ``u_tau`` from the exchange-location velocity and
    the first-cell distance.  If both sample points lie on one Spalding profile,
    the recovered first-cell velocity must reproduce the existing first-cell
    tangential velocity, so the assimilation is a no-op while still reporting
    the wall shear ``rho u_tau^2``.

    Geometry: boundary cell centre at ``y = 3``, wall plane at ``y = 2.5``
    (halfway BFL), so ``y1 = 0.5``.  With ``exchange_distance = 2.5`` the
    exchange sample sits on the lattice node ``y = 5`` (``y2 = 2.5`` exactly),
    which keeps the manufactured field interpolation-free.
    """
    c, mask, normals, q_field, cell, _ = _wall_state(q_count)
    from tensorlbm.d3q19 import equilibrium3d
    from tensorlbm.d3q27 import equilibrium27

    shape = (5, 7, 9)
    nu = 1.0e-4
    u_tau_manufactured = 0.0035
    y1, y2 = 0.5, 2.5
    u1 = (
        float(
            spalding_u_plus_from_y_plus(
                torch.tensor(y1 * u_tau_manufactured / nu, dtype=torch.float64)
            )
        )
        * u_tau_manufactured
    )
    u2 = (
        float(
            spalding_u_plus_from_y_plus(
                torch.tensor(y2 * u_tau_manufactured / nu, dtype=torch.float64)
            )
        )
        * u_tau_manufactured
    )
    ux = torch.zeros(shape, dtype=torch.float64)
    ux[cell] = u1  # boundary node, y1 above the wall
    ux[cell[0], 5, :] = u2  # exchange node, y2 above the wall
    rho = torch.ones(shape, dtype=torch.float64)
    zero = torch.zeros(shape, dtype=torch.float64)
    equilibrium = equilibrium3d if q_count == 19 else equilibrium27
    f = equilibrium(rho, ux, zero, zero).double()

    out, diagnostics = apply_spalding_exchange_wall_model(
        f,
        mask,
        q_field,
        normals,
        nu=nu,
        exchange_distance=2.5,
    )

    assert diagnostics.boundary_nodes == 1
    assert diagnostics.mean_y1 == pytest.approx(y1, abs=1.0e-6)
    assert diagnostics.mean_u_tau == pytest.approx(u_tau_manufactured, rel=2.0e-6)
    assert diagnostics.shear_force[1] == pytest.approx(0.0, abs=1.0e-18)
    _, _, macroscopic = _lattice(q_count)
    _, ux_out, _, _ = macroscopic(out)
    # The modelled first-cell velocity reproduces the manufactured value, i.e.
    # the assimilation is a fixed point of the manufactured profile.
    assert float(ux_out[cell]) == pytest.approx(u1, rel=1.0e-6)
    # Fixed-point residual is bounded by the weight-quantization floor in
    # relative terms (D3Q19: 2.5e-8 of u1, D3Q27: machine zero).
    assert float((out - f).abs().max()) <= 1.0e-7 * u1
    assert float((out - f).sum(dim=0).abs().max()) <= MASS_ATOL
