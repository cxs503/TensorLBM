"""Tests for :mod:`tensorlbm.free_surface_dispersion` (p485w1-n05).

The suite is deliberately layered:

* *unit* — the closed forms and their limits (no solver execution);
* *physics* — exactness of the von Neumann lattice modes against the continuum
  sound speed, and the estimator-bias identity against a direct finite-difference
  evaluation of the standing-wave field;
* *regression* — ``interface_pressure_policy="const"`` must stay bit-identical to
  the legacy Körner reconstruction (the calibration operator is opt-in);
* *integration* — a short periodic-rig run end to end.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm import free_surface_dispersion as fs
from tensorlbm.d2q9 import W_EXACT
from tensorlbm.d2q9 import C as C_TORCH
from tensorlbm.wave_tank2d import stokes2


# ---------------------------------------------------------------------------
# lattice identity
# ---------------------------------------------------------------------------
def test_lattice_constants_match_d2q9_module():
    assert np.allclose(fs.W, np.asarray(W_EXACT))
    assert np.allclose(fs.C, C_TORCH.numpy())
    assert fs.CS2 == pytest.approx(1.0 / 3.0)


def test_lattice_modes_shape_and_shear_branch():
    """The shear branch must reproduce ``omega = -i nu k^2`` to machine precision."""
    tau = 0.7
    nu = fs.CS2 * (tau - 0.5)
    k = 1e-3
    omega = fs.lattice_linear_modes(k, 0.0, tau)
    shear = [w for w in omega if abs(w.real) < 1e-6 and abs(w.imag) > 0]
    assert shear, omega
    rate = -min(abs(w.imag) for w in shear)
    assert rate == pytest.approx(-nu * k * k, rel=1e-6)


@pytest.mark.physics
def test_lattice_acoustic_branch_is_close_to_continuum():
    """At lattice-resolved wave numbers the D2Q9 sound speed error must be small."""
    for k in (0.02, 0.05, 0.1, 0.2):
        info = fs.lattice_phase_speed_error(k, 0.51)
        assert info["rel_error"] < 5e-3, info
        assert info["damping"] > 0.0
    # and the error must grow with k (second order leading term)
    e1 = abs(fs.lattice_phase_speed_error(0.1, 0.51)["rel_error"])
    e2 = abs(fs.lattice_phase_speed_error(0.2, 0.51)["rel_error"])
    assert e2 > e1


def test_lattice_acoustic_branch_is_isotropic_at_fourth_order():
    """D2Q9 isotropy: the axis and diagonal sound speeds agree to O(k^4)."""
    k = 0.1
    a = fs.lattice_phase_speed_error(k, 0.6, angle=0.0)["c_lattice"]
    d = fs.lattice_phase_speed_error(k, 0.6, angle=math.pi / 4)["c_lattice"]
    assert abs(a / d - 1.0) < 2e-3


def test_acoustic_mode_rejects_unstable_tau():
    with pytest.raises(ValueError):
        fs.lattice_linear_modes(0.1, 0.0, 0.5)


# ---------------------------------------------------------------------------
# closed-form gravity-wave relations
# ---------------------------------------------------------------------------
def test_continuum_limits():
    g, h = 9.81, 100.0
    # shallow water: c -> sqrt(g h)
    k = 1e-6
    assert fs.continuous_phase_speed(k, h, g) == pytest.approx(math.sqrt(g * h), rel=1e-5)
    # deep water: c -> sqrt(g/k)
    k = 10.0
    assert fs.continuous_phase_speed(k, h, g) == pytest.approx(math.sqrt(g / k), rel=1e-5)
    # group speed: n -> 1 in shallow water, n -> 1/2 in deep water
    assert fs.continuous_group_speed(1e-6, h, g) / fs.continuous_phase_speed(1e-6, h, g) == (
        pytest.approx(1.0, rel=1e-4)
    )
    assert fs.continuous_group_speed(10.0, h, g) / fs.continuous_phase_speed(10.0, h, g) == (
        pytest.approx(0.5, rel=1e-4)
    )


def test_group_speed_matches_numerical_derivative():
    k, h, g = 0.075, 20.0, 1.3e-4
    dk = 1e-6
    num = (fs.continuous_omega(k + dk, h, g) - fs.continuous_omega(k - dk, h, g)) / (2 * dk)
    assert fs.continuous_group_speed(k, h, g) == pytest.approx(num, rel=1e-8)


def test_depth_sensitivity_matches_finite_difference():
    k, h = 0.075, 20.0
    dlnh = 1e-7
    c0 = fs.continuous_phase_speed(k, h, 1.0)
    c1 = fs.continuous_phase_speed(k, h * (1 + dlnh), 1.0)
    num = (c1 / c0 - 1.0) / dlnh
    assert fs.depth_sensitivity(k, h) == pytest.approx(num, rel=1e-6)


def test_laplace_phase_shift_sign_and_zero():
    k, h, g = 0.075, 20.0, 1.3e-4
    assert fs.laplace_phase_shift(k, h, g, 0.0) == 0.0
    pos = fs.laplace_phase_shift(k, h, g, 1e-3)
    neg = fs.laplace_phase_shift(k, h, g, -1e-3)
    assert pos > 0.0 > neg
    # quadratic in k: doubling k roughly quadruples the shift
    r1 = fs.laplace_phase_shift(k, 10.0 * h, g, 1e-3)
    r2 = fs.laplace_phase_shift(2 * k, 10.0 * h, g, 1e-3)
    assert r2 / r1 == pytest.approx(4.0, rel=5e-2)


def test_stokes2_agreement():
    """Module must agree with the wave-tank theory module bit-for-bit in spirit."""
    k, h, g = 0.075, 20.0, 1.3e-4
    assert fs.continuous_phase_speed(k, h, g) == pytest.approx(
        stokes2.phase_speed(k, h, g), rel=1e-12
    )


# ---------------------------------------------------------------------------
# error budget
# ---------------------------------------------------------------------------
def test_budget_parts_sum_to_total():
    b = fs.phase_error_budget(
        0.075, 20.0, 1.3e-4, 0.51, interface_offset_cells=0.25, sigma_cal=-1e-3
    )
    parts = b.e_lattice + b.e_interface + b.e_laplace + b.e_viscous
    assert b.e_total == pytest.approx(parts, rel=1e-12)
    assert b.c_predicted == pytest.approx(b.c_cont * (1 + b.e_total), rel=1e-12)


def test_budget_lattice_term_is_second_order():
    g = 1.3e-4
    b1 = fs.phase_error_budget(0.075, 20.0, g, 0.51)
    b2 = fs.phase_error_budget(0.15, 20.0, g, 0.51)
    assert b2.e_lattice / b1.e_lattice == pytest.approx(4.0, rel=1e-12)


def test_budget_rejects_bad_arguments():
    with pytest.raises(ValueError):
        fs.phase_error_budget(0.0, 20.0, 1e-4, 0.51)
    with pytest.raises(ValueError):
        fs.phase_error_budget(0.1, 0.0, 1e-4, 0.51)


# ---------------------------------------------------------------------------
# estimator bias
# ---------------------------------------------------------------------------
def test_estimator_bias_vanishes_without_reflection():
    info = fs.standing_wave_estimator_bias(0.075, 0.0)
    assert abs(info["bias_min"]) < 1e-12
    assert abs(info["bias_max"]) < 1e-12


def test_estimator_bias_closed_form_matches_direct_evaluation():
    """``bias_at`` must equal a direct ``Im(Z'/Z)`` evaluation of the field."""
    k, r, x = 0.075, 0.12, 3.3

    def phasor(y: float) -> complex:
        return math.e ** (-1j * k * y) + r * math.e ** (1j * k * y)

    h = 1e-6
    local_slope = ((phasor(x + h) / phasor(x - h)).imag) / (2 * h)
    k_app = -(local_slope.real * 0 + local_slope)
    # d/dx arg Z = Im(Z'/Z); with Z' ~ central difference
    dz = (phasor(x + h) - phasor(x - h)) / (2 * h)
    darg = (dz / phasor(x)).imag
    k_app = -darg
    closed = fs.bias_at(k, r, 2 * k * x)
    assert k / k_app - 1.0 == pytest.approx(closed, rel=1e-5)


def test_estimator_bias_does_not_vanish_with_probe_spacing():
    """The headline result: refining the probe spacing does not remove the bias."""
    fine = fs.synthetic_standing_wave_bias(0.075, 0.1, probe_spacing_wavelengths=0.05)
    coarse = fs.synthetic_standing_wave_bias(0.075, 0.1, probe_spacing_wavelengths=0.5)
    assert fine["bias_two_point_max"] > 0.10
    assert abs(coarse["bias_two_point_max"]) < 1e-6  # degenerate ds = lambda/2
    assert fine["bias_two_point_span"] > coarse["bias_two_point_span"]


def test_reflection_needed_for_reported_wave_tank_errors():
    """A ~9 % residual reflection reproduces the reported 19 % bias."""
    info = fs.standing_wave_estimator_bias(0.075, 0.0868)
    assert 0.17 < info["bias_max"] < 0.21
    assert 0.02 < info["r_for_10pct_bias"] < 0.06


def test_synthetic_bias_rejects_negative_reflection():
    with pytest.raises(ValueError):
        fs.standing_wave_estimator_bias(0.1, -0.1)


# ---------------------------------------------------------------------------
# calibration operator
# ---------------------------------------------------------------------------
def test_subcell_pressure_numpy_semantics():
    fill = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    g = 1e-3
    out = fs.subcell_interface_pressure(fill, g)
    assert out[2] == pytest.approx(0.0, abs=1e-15)          # fill = 1/2 -> no offset
    assert out[0] == pytest.approx(-1.5e-3)                  # empty cell -> below p_atm
    assert out[4] == pytest.approx(1.5e-3)                   # full cell -> above p_atm
    assert np.allclose(out, 3.0 * g * (fill - 0.5))


def test_subcell_pressure_torch_and_clip():
    torch = pytest.importorskip("torch")
    fill = torch.tensor([0.0, 0.5, 1.0])
    out = fs.subcell_interface_pressure(fill, 1.0)          # 3*(f-0.5) beyond the 0.1 band
    assert torch.allclose(out, torch.tensor([-0.1, 0.0, 0.1]))


def test_calibrate_interface_offset_recovers_synthetic_offset():
    h, offset_true = 20.0, 0.37
    ms = []
    for k in (0.02, 0.05, 0.075, 0.12):
        ms.append(
            {"k": k, "rel_error": fs.depth_sensitivity(k, h) * offset_true / h}
        )
    ident = fs.calibrate_interface_offset(ms, h=h)
    assert ident["offset_cells"] == pytest.approx(offset_true, rel=1e-6)
    assert ident["rms_residual"] < 1e-12


# ---------------------------------------------------------------------------
# estimators
# ---------------------------------------------------------------------------
def test_fit_mode_2d_recovers_a_synthetic_mode():
    k_true, omega_true, amp = 0.075, 0.003, 0.05
    nx, nt = 84, 64
    x = np.arange(nx, dtype=float)
    t = np.arange(1, nt + 1, dtype=float) * 17.0
    eta = amp * np.cos(omega_true * t[:, None] - k_true * x[None, :] + 0.3)
    fit = fs.fit_mode_2d(eta, t, x, omega_true * 1.05, k_true * 0.95)
    assert fit["k"] == pytest.approx(k_true, rel=5e-3)
    assert fit["omega"] == pytest.approx(omega_true, rel=5e-3)
    assert fit["amplitude"] == pytest.approx(amp, rel=5e-3)


def test_group_speed_from_sweep_matches_continuum():
    h, g = 20.0, 1.3e-4
    k0 = 0.075
    ms = []
    for k in (k0 * 0.9, k0, k0 * 1.1):
        ms.append({"k": k, "c": fs.continuous_phase_speed(k, h, g), "h": h, "g": g})
    out = fs.group_speed_from_sweep(ms, k0)
    assert out["c_group_rel_error"] == pytest.approx(0.0, abs=1e-3)
    with pytest.raises(ValueError):
        fs.group_speed_from_sweep(ms[:2], k0)


# ---------------------------------------------------------------------------
# solver integration: opt-in calibration + bit-identical legacy path
# ---------------------------------------------------------------------------
def _tiny_state(nx: int = 24, ny: int = 16, h: float = 8.0, g: float = 1e-3):
    import torch

    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.free_surface_lbm_2d import (
        GAS,
        init_flags_from_fill_2d,
        init_mass_from_fill_2d,
    )

    torch.set_default_dtype(torch.float32)
    y = torch.arange(ny, dtype=torch.float32).unsqueeze(1).expand(ny, nx)
    x = torch.arange(nx, dtype=torch.float32).unsqueeze(0).expand(ny, nx)
    fill = (h + 0.3 * torch.cos(2 * math.pi * x / nx) - (y - 0.5)).clamp(0.0, 1.0)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = True
    solid[-1, :] = True
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags)
    active = flags != GAS
    rho = torch.where(active, (1.0 + 3.0 * g * (h - y)).clamp(min=1.0), torch.ones_like(x))
    f = equilibrium(rho, torch.zeros_like(x), torch.zeros_like(x))
    return f, mass, flags, solid, g


@pytest.mark.regression
def test_default_policy_is_bit_identical_to_legacy():
    import torch

    from tensorlbm.free_surface_lbm_2d import koerner_step_2d

    f0, m0, fl0, solid, g = _tiny_state()
    a = koerner_step_2d(f0.clone(), m0.clone(), fl0.clone(), solid, tau=0.52, gy=-g)
    b = koerner_step_2d(
        f0.clone(),
        m0.clone(),
        fl0.clone(),
        solid,
        tau=0.52,
        gy=-g,
        interface_pressure_policy="const",
    )
    for lhs, rhs in zip(a, b):
        assert torch.equal(lhs, rhs)


def test_subcell_policy_runs_and_changes_the_reconstruction():
    import torch

    from tensorlbm.free_surface_lbm_2d import koerner_step_2d, total_liquid_inventory_2d

    f0, m0, fl0, solid, g = _tiny_state()
    a = koerner_step_2d(f0.clone(), m0.clone(), fl0.clone(), solid, tau=0.52, gy=-g)
    b = koerner_step_2d(
        f0.clone(),
        m0.clone(),
        fl0.clone(),
        solid,
        tau=0.52,
        gy=-g,
        interface_pressure_policy="subcell",
    )
    assert not torch.equal(a[0], b[0])  # populations differ at interface cells
    for state in (a, b):
        assert all(torch.isfinite(t).all() for t in state[:1])
    inv_a = total_liquid_inventory_2d(*a)
    inv_b = total_liquid_inventory_2d(*b)
    assert inv_b == pytest.approx(inv_a, rel=5e-3)


def test_invalid_policy_raises():
    from tensorlbm.free_surface_lbm_2d import koerner_step_2d

    f0, m0, fl0, solid, g = _tiny_state()
    with pytest.raises(ValueError):
        koerner_step_2d(
            f0, m0, fl0, solid, tau=0.52, gy=-g, interface_pressure_policy="bogus"
        )


@pytest.mark.slow
def test_run_linear_wave_smoke():
    """End-to-end short rig run: finite output, small mass drift, keys present."""
    case = fs.LinearWaveCase(
        h=6.0,
        kh=1.0,
        amplitude=0.05,
        c_target=0.15,
        tau=0.52,
        wavelengths=1,
        headroom=6,
        n_periods=1.5,
        warmup_periods=0.5,
        record_per_period=16,
    )
    res = fs.run_linear_wave(case)
    assert math.isfinite(res["discrete"]["c_measured"])
    assert res["discrete"]["c_measured"] > 0.0
    assert abs(res["mass"]["relative_drift_total"]) < 0.05
    assert res["discrete"]["k_measured"] == pytest.approx(case.k, rel=0.2)
    assert res["budget"]["c_cont"] == pytest.approx(res["discrete"]["c_continuum"])

# ---------------------------------------------------------------------------
# two-wave fit and the replayed tank estimator (the campaign's estimator)
# ---------------------------------------------------------------------------
def _synthetic_field(k=0.075, omega=0.0045, amp=0.25, reflection=0.0, nx=84, nt=110, dt=44.0):
    x = np.arange(nx, dtype=float)
    t = np.arange(1, nt + 1, dtype=float) * dt
    eta = amp * (
        np.cos(omega * t[:, None] - k * x[None, :])
        + reflection * np.cos(omega * t[:, None] + k * x[None, :])
    )
    return eta, t, x


def test_fit_two_wave_is_unbiased_by_a_counter_wave():
    """A single-wave fit is biased by r; the two-wave fit must not be."""
    k, omega = 0.075, 0.0045
    eta, t, x = _synthetic_field(k=k, omega=omega, reflection=0.10)
    two = fs.fit_two_wave(eta, t, x, omega * 1.05, k * 0.95)
    assert two["k"] == pytest.approx(k, rel=2e-3)
    assert two["omega"] == pytest.approx(omega, rel=2e-3)
    assert two["reflection_ratio"] == pytest.approx(0.10, rel=2e-3)
    # A whole-box (one-wavelength) fit averages the standing pattern and is only
    # mildly biased; the strongly amplified estimator is the *local slope* one,
    # which the following test pins down.
    single = fs.fit_traveling_wave(eta, t, x, omega * 1.05, k * 0.95)
    assert abs(single["omega"] / omega - 1.0) < 1e-2


def test_two_point_phase_speed_envelope_grows_with_reflection():
    quiet = fs.two_point_phase_speed(*_synthetic_field(reflection=0.0)[0:3:1][:1][0:0] or
                                     _synthetic_field(reflection=0.0), 0.0045) \
        if False else None  # placeholder to keep linters quiet
    del quiet
    eta0, t0, x0 = _synthetic_field(reflection=0.0)
    eta1, t1, x1 = _synthetic_field(reflection=0.10)
    e0 = fs.two_point_phase_speed(eta0, t0, x0, 0.0045)
    e1 = fs.two_point_phase_speed(eta1, t1, x1, 0.0045)
    assert abs(e0["rel_error_max"]) < 0.02
    assert e1["rel_error_span"] > 0.15
    assert e1["rel_error_min"] < -0.05 < 0.05 < e1["rel_error_max"]


def test_group_speed_needs_fixed_gravity():
    """A gravity override must reproduce the requested target phase speed."""
    case = fs.LinearWaveCase(h=20.0, kh=1.5, c_target=0.06)
    g = case.gravity()
    assert fs.continuous_phase_speed(case.k, case.h, g) == pytest.approx(0.06, rel=1e-12)
    fixed = fs.LinearWaveCase(h=20.0, kh=1.65, c_target=0.06, g_override=g)
    assert fixed.gravity() == g
    # with g fixed, a shorter wave is slower and c_g < c (deep-water behaviour)
    c = fs.continuous_phase_speed(fixed.k, fixed.h, g)
    cg = fs.continuous_group_speed(fixed.k, fixed.h, g)
    assert c < 0.06 and cg < c
