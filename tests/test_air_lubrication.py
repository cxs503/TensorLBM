"""Tests for the air-lubrication (PAL) module.

Structure
---------
* analytic-reference checks (exact stratified channel, PAL law),
* lattice-versus-analytic verification,
* conservation audits with **negative controls**: every audit is paired with a
  deliberately corrupted case that must fail the same threshold, so the test has
  teeth instead of merely passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.air_lubrication import (
    LAMBDA_AIR_WATER,
    ChannelAirLubrication2D,
    VoidFractionTransport2D,
    alpha_similarity_profile,
    curvature_drift_velocity,
    mixture_viscosity,
    pal_drag_reduction,
    rise_velocity_schiller_naumann,
    stratified_channel_exact,
    stratified_drag_reduction_from_alpha,
    stokes_rise_velocity,
)


# --------------------------------------------------------------- exact solution
def test_exact_uniform_channel_is_classical_laminar():
    """Uniform viscosity: tau_w = rho G H and DR = 0 (no air, no reduction)."""
    n = 64
    out = stratified_channel_exact(np.ones(n), u_mean=1.0, H=float(n))
    assert out["tau_w"] == pytest.approx(out["tau_w_reference"], rel=1e-14)
    assert out["dr"] == pytest.approx(0.0, abs=1e-14)
    # ... and the discrete G agrees with the classical tau_w = 3 mu U / H
    assert out["tau_w"] == pytest.approx(3.0 * 1.0 / float(n), rel=1e-3)
    u = out["u_profile"]
    # half-channel laminar: u(y) = G (H y - y^2/2); mean must equal u_mean
    assert float(np.mean(u)) == pytest.approx(1.0, rel=1e-12)
    assert u[0] > 0.0 and u[-1] > u[0]


def test_exact_drag_reduction_is_positive_and_bounded():
    y = np.arange(64) + 0.5
    alpha = alpha_similarity_profile(y, 0.2, 8.0)
    out = stratified_drag_reduction_from_alpha(alpha, 64.0, lam=0.05)
    assert 0.0 < out["dr"] < 1.0
    # more air (thicker layer) must reduce drag more
    thick = stratified_drag_reduction_from_alpha(
        alpha_similarity_profile(y, 0.2, 16.0), 64.0, lam=0.05
    )
    assert thick["dr"] > out["dr"]


def test_pal_law_monotone_in_alpha_wall():
    drs = [pal_drag_reduction(a, 8.0, 64.0, lam=0.05)["dr"] for a in (0.05, 0.2, 0.4)]
    assert drs[0] < drs[1] < drs[2]
    assert pal_drag_reduction(0.0, 8.0, 64.0, lam=0.05)["dr"] == pytest.approx(0.0, abs=1e-12)


def test_mixture_viscosity_bounds():
    alpha = np.array([0.0, 0.25, 0.5])
    harm = mixture_viscosity(alpha, 1.0, 0.05, model="harmonic")
    ein = mixture_viscosity(alpha, 1.0, model="einstein_roscoe")
    assert harm[0] == pytest.approx(1.0) and ein[0] == pytest.approx(1.0)
    assert np.all(harm[1:] < 1.0)  # a film lowers the viscosity
    assert np.all(ein[1:] > 1.0)  # a dilute bubbly mixture raises it
    with pytest.raises(ValueError):
        mixture_viscosity(np.array([1.5]), 1.0, 0.05)


def test_schiller_naumann_reduces_to_stokes():
    d = 5e-6
    assert rise_velocity_schiller_naumann(d) == pytest.approx(
        stokes_rise_velocity(d), rel=0.02
    )
    # and is smaller than Stokes once the bubble Reynolds number is O(10)
    assert rise_velocity_schiller_naumann(1e-3) < stokes_rise_velocity(1e-3)


def test_curvature_drift_sign_and_scaling():
    assert curvature_drift_velocity(5.0, 0.0, 300e-6) == pytest.approx(0.0)
    up = curvature_drift_velocity(5.0, 1.0, 300e-6)
    down = curvature_drift_velocity(5.0, -1.0, 300e-6)
    assert up > 0.0 and down < 0.0 and up == pytest.approx(-down)
    # quadratic in speed, linear in curvature
    assert curvature_drift_velocity(10.0, 1.0, 300e-6) == pytest.approx(4.0 * up, rel=1e-12)


# ------------------------------------------------------- lattice verification
def _lbm(nu, alpha, lam, steps=500, ny=32, nx=8, cs=0.0):
    sim = ChannelAirLubrication2D(
        nx=nx,
        ny=ny,
        u_mean=0.05,
        nu_liquid=nu,
        lam=lam,
        alpha_profile=alpha,
        sgs=cs > 0.0,
        cs_smag=cs,
    )
    return sim, sim.run(steps)


def test_lattice_reproduces_exact_stratified_solution():
    """Delta C_f from the lattice must match the exact stratified law."""
    y = np.arange(32) + 0.5
    alpha = alpha_similarity_profile(y, 0.30, 8.0)
    _, base = _lbm(2.0e-3, None, 1.0)
    _, air = _lbm(2.0e-3, alpha, 0.05)
    dr_num = 1.0 - air.tau_w / base.tau_w
    dr_ref = stratified_drag_reduction_from_alpha(alpha, 32.0, lam=0.05)["dr"]
    assert dr_ref > 0.1
    assert abs(dr_num - dr_ref) / dr_ref < 0.20


def test_lattice_wall_shear_matches_force_balance():
    sim, st = _lbm(2.0e-3, None, 1.0)
    assert st.tau_w == pytest.approx(sim.force * sim.H, rel=0.05)


def test_lattice_momentum_budget_closes():
    _, st = _lbm(2.0e-3, alpha_similarity_profile(np.arange(32) + 0.5, 0.2, 6.0), 0.05)
    assert st.momentum_residual_max < 1e-8


def test_negative_control_no_air_no_drag_reduction():
    """With alpha = 0 the lattice must return zero drag reduction."""
    _, base = _lbm(2.0e-3, None, 1.0)
    _, same = _lbm(2.0e-3, None, 0.05)  # lam is irrelevant when there is no air
    assert abs(1.0 - same.tau_w / base.tau_w) < 1e-10


def test_negative_control_equal_viscosity_no_drag_reduction():
    """alpha > 0 but lambda = 1 (air as viscous as water) must give DR ~ 0."""
    y = np.arange(32) + 0.5
    alpha = alpha_similarity_profile(y, 0.4, 8.0)
    _, base = _lbm(2.0e-3, None, 1.0)
    _, same = _lbm(2.0e-3, alpha, 1.0)
    assert abs(1.0 - same.tau_w / base.tau_w) < 1e-10
    assert stratified_drag_reduction_from_alpha(alpha, 32.0, lam=1.0)["dr"] == pytest.approx(
        0.0, abs=1e-12
    )


def test_lattice_rejects_bad_alpha_profile():
    with pytest.raises(ValueError):
        ChannelAirLubrication2D(nx=8, ny=16, alpha_profile=np.zeros(5))


# ------------------------------------------------------ transport / conservation
def _transport(nx=24, ny=24, flux=0.004, steps=200):
    y = np.arange(ny) + 0.5
    u = 0.08 * (y / (ny - 0.5)) ** (1 / 7.0)
    disp = 0.05 * 0.08 * (ny - 0.5) * np.clip((y / (ny - 0.5)) / 0.05, 0.15, 1.0)
    inj = np.zeros(nx)
    inj[:4] = flux
    tr = VoidFractionTransport2D(
        nx=nx,
        ny=ny,
        u_profile=u,
        rise_velocity=1e-3,
        disp_y=disp,
        injection=inj,
        dt=0.25,
    )
    U = np.repeat(u[:, None], nx, axis=1)
    err = 0.0
    for _ in range(steps):
        tr.step(U)
        err = max(err, tr.conservation_error())
    return tr, err


def test_transport_conserves_species_mass():
    tr, err = _transport()
    assert err < 1e-8
    assert tr.mass() == pytest.approx(tr.injected_total - tr.outflow_total, rel=1e-9, abs=1e-12)
    assert tr.alpha.min() >= 0.0 and tr.alpha.max() <= 1.0


def test_negative_control_transport_leak_is_detected():
    """The audit must fail when the field is corrupted by a small amount."""
    tr, _ = _transport()
    tr.alpha[0, 0] += 1e-3
    assert tr.conservation_error() > 1e-8


def test_negative_control_zero_injection_keeps_field_empty():
    y = np.arange(24) + 0.5
    u = 0.08 * (y / 23.5) ** (1 / 7.0)
    tr = VoidFractionTransport2D(nx=24, ny=24, u_profile=u, rise_velocity=1e-3)
    U = np.repeat(u[:, None], 24, axis=1)
    for _ in range(50):
        tr.step(U)
    assert tr.mass() == 0.0
    assert tr.alpha.max() == 0.0


def test_transport_injection_reaches_the_wall_row():
    tr, _ = _transport()
    assert tr.alpha[0, :4].mean() > 0.0
    assert tr.alpha[0, :4].mean() > tr.alpha[0, 10:].mean()


def test_air_lubrication_constants_are_physical():
    # mu_air/mu_water = 1.8e-5 / 1.0e-3
    assert LAMBDA_AIR_WATER == pytest.approx(1.8e-2, rel=0.02)