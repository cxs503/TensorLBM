"""P88W1-N01: forced-sway LNG tank sloshing verification surface.

Fast tests (pure analytic anchors) run everywhere; the three solver-backed tests
use a deliberately coarse tank (``cells_per_height = 16``, a few forcing periods)
and therefore assert *loose* physical tolerances — the tight numbers of the node
come from the campaign at ``cells_per_height = 32/64`` in ``artifacts/p88w1``,
not from these unit runs.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.sloshing_forced import (
    TankSpec,
    abramson_frequency,
    bore_reflection_pressure,
    build_case,
    gauge_height_from_swl,
    harmonic_metrics,
    hydrostatic_runup_pressure,
    identify_resonance,
    linear_wall_pressure_amplitude,
    mode_excitation_coefficient,
    modal_forcing_amplitude,
    odd_mode_fraction,
    peak_frequency,
    pressure_metrics,
    resonance_ratio,
    run_decay_case,
    run_forced_case,
    stagnation_impact_pressure,
    static_tilt_elevation,
    steady_modal_amplitude,
    steady_wall_amplitude,
    wall_runup_series,
)
from tensorlbm.sloshing_forced import theory

L, H, B = 1.0, 0.4, 0.5
H_FILL = 0.2  # 50 % of H
G = 9.81


# ---------------------------------------------------------------------------
# geometry / scaling
# ---------------------------------------------------------------------------


def test_tank_geometry_matches_campaign_book():
    """L x B x H = 1.0 x 0.5 x 0.4 m at 50 % fill -> still-water depth 0.2 m."""
    spec = TankSpec()
    assert spec.length == pytest.approx(1.0)
    assert spec.breadth == pytest.approx(0.5)
    assert spec.height == pytest.approx(0.4)
    assert spec.depth == pytest.approx(H_FILL)
    assert spec.freeboard == pytest.approx(0.2)
    assert spec.gas_headroom == pytest.approx(0.2)


def test_spec_validation_rejects_bad_inputs():
    with pytest.raises(ValueError):
        TankSpec(cells_per_height=8).check()
    with pytest.raises(ValueError):
        TankSpec(tau=0.5).check()
    with pytest.raises(ValueError):
        TankSpec(fill_ratio=1.0).check()
    with pytest.raises(ValueError):
        TankSpec(fill_ratio=0.0).check()


def test_lattice_scaling_is_froude_and_mach_consistent():
    spec = TankSpec(cells_per_height=32)
    sc = spec.scaling()
    assert sc.dx == pytest.approx(H / 32.0)
    # dt is fixed by the lattice Mach number: u_lat = u_char dt / dx
    assert sc.u_lat == pytest.approx(spec.u_char * sc.dt / sc.dx)
    # lattice gravity is the Froude-scaled physical gravity
    assert sc.g_lattice == pytest.approx(G * sc.dt * sc.dt / sc.dx)
    # weak-compressibility budget stays small (no gravity-wave phase-speed bias > 3 %)
    assert 3.0 * sc.g_lattice * (spec.depth / sc.dx) < 0.05


def test_gauge_heights_straddle_the_still_water_line():
    dx = 0.0125
    h_cells = 16.0
    assert gauge_height_from_swl(16, h_cells, dx) == pytest.approx(-0.5 * dx)
    assert gauge_height_from_swl(17, h_cells, dx) == pytest.approx(+0.5 * dx)
    assert gauge_height_from_swl(1, h_cells, dx) == pytest.approx(-(h_cells - 0.5) * dx)


# ---------------------------------------------------------------------------
# analytic anchors
# ---------------------------------------------------------------------------


def test_abramson_first_and_second_mode_values():
    """Abramson (1966) dispersion for L = 1 m, h = 0.2 m, g = 9.81 m/s^2."""
    f1 = abramson_frequency(1, H_FILL, L, G)
    f2 = abramson_frequency(2, H_FILL, L, G)
    assert f1 == pytest.approx(0.659349, abs=1e-5)
    assert f2 == pytest.approx(1.1520951, abs=1e-6)
    assert f2 / f1 == pytest.approx(1.7473, abs=2e-3)
    # deep-water limit must be approached from below
    deep = math.sqrt(G * math.pi / L) / (2.0 * math.pi)
    assert f1 < deep


def test_excitation_parity_rule():
    """Horizontal translation excites odd modes only: (-1)^n - 1 vanishes for even n."""
    assert mode_excitation_coefficient(1) == pytest.approx(-2.0)
    assert mode_excitation_coefficient(2) == pytest.approx(0.0)
    assert mode_excitation_coefficient(3) == pytest.approx(-2.0)
    assert mode_excitation_coefficient(4) == pytest.approx(0.0)
    for n in (2, 4, 6):
        assert modal_forcing_amplitude(n, 0.002, 2 * math.pi * 0.66, H_FILL, L) == 0.0
    assert modal_forcing_amplitude(1, 0.002, 2 * math.pi * 0.66, H_FILL, L) > 0.0


def test_static_limit_reproduces_rigid_body_tilt():
    """omega -> 0 with fixed acceleration must recover eta = -Xddot (x - L/2)/g."""
    xddot = 1.0
    omega = 1e-3 * 2.0 * math.pi * abramson_frequency(1, H_FILL, L, G)
    sway = xddot / omega**2  # hold the *acceleration* amplitude fixed
    for x in (0.0, 0.25, 0.5, 0.75, 1.0):
        eta = 0.0
        for n in range(1, 400, 2):
            a = steady_modal_amplitude(n, sway, omega, 0.0, H_FILL, L, G)
            eta += a.real * math.cos(n * math.pi * x / L)
        assert eta == pytest.approx(static_tilt_elevation(x, xddot, G, L), rel=2e-2, abs=2e-5)


def test_resonance_ratio_and_peak_frequency():
    wn = 2.0 * math.pi * 0.659349
    zeta = 0.01
    assert resonance_ratio(wn, wn, zeta) == pytest.approx(1.0 / (2.0 * zeta), rel=1e-12)
    assert resonance_ratio(0.0, wn, zeta) == pytest.approx(1.0, rel=1e-12)
    assert peak_frequency(wn, zeta) == pytest.approx(wn * math.sqrt(1 - 2 * zeta**2))
    assert peak_frequency(wn, 0.9) == 0.0  # overdamped: no interior maximum


def test_finite_time_reference_converges_to_steady_state():
    """Long forced run of the ODE reference -> closed-form steady amplitude."""
    omega = 2.0 * math.pi * abramson_frequency(1, H_FILL, L, G)
    zeta = 0.05
    X0 = 0.002
    steady = abs(steady_wall_amplitude(X0, omega, zeta, H_FILL, L, G, modes=1))
    ref = theory.finite_time_reference_amplitude(
        1, X0, omega, zeta, H_FILL, L, G, n_periods=60.0, ramp_periods=1.0
    )
    assert ref["a_peak_half"] == pytest.approx(steady, rel=0.05)


def test_finite_time_reference_is_zero_for_unexcited_mode():
    omega = 2.0 * math.pi * abramson_frequency(1, H_FILL, L, G)
    ref = theory.finite_time_reference_amplitude(
        2, 0.002, omega, 0.01, H_FILL, L, G, n_periods=2.0
    )
    assert abs(ref["a_peak_overall"]) == pytest.approx(0.0, abs=1e-14)


def test_wall_pressure_decay_with_depth():
    """Linear wall pressure at the bottom is 1/cosh(kh) of its value at the SWL."""
    k = math.pi / L
    ratio = 1.0 / math.cosh(k * H_FILL)
    p_swl = linear_wall_pressure_amplitude(1, 0.01, 0.0, H_FILL, L, G, 1000.0)
    p_bot = linear_wall_pressure_amplitude(1, 0.01, -H_FILL, H_FILL, L, G, 1000.0)
    assert p_swl == pytest.approx(1000.0 * G * 0.01, rel=1e-12)
    assert p_bot / p_swl == pytest.approx(ratio, rel=1e-12)


def test_pressure_bounds_ordering():
    rho, v = 1000.0, 1.0
    stag = stagnation_impact_pressure(rho, v)
    bore = bore_reflection_pressure(rho, G, H_FILL, v)
    runup = hydrostatic_runup_pressure(rho, G, H_FILL, 0.05)
    assert stag == pytest.approx(0.5 * rho * v * v)
    assert bore > hydrostatic_runup_pressure(rho, G, H_FILL, 0.0)
    assert runup == pytest.approx(rho * G * (H_FILL + 0.05))
    assert stag > 0.0


def test_odd_mode_fraction_of_pure_modes():
    assert odd_mode_fraction({1: 1.0, 2: 0.0, 3: 0.0}, modes=3) == pytest.approx(1.0)
    assert odd_mode_fraction({1: 0.0, 2: 1.0, 3: 0.0}, modes=3) == pytest.approx(0.0)
    assert math.isnan(odd_mode_fraction({1: 0.0, 2: 0.0}, modes=2))


# ---------------------------------------------------------------------------
# identification helpers
# ---------------------------------------------------------------------------


def test_identify_resonance_recovers_parabolic_peak():
    ratios = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4])
    amps = 1.0 / np.sqrt((1.0 - ratios**2) ** 2 + (2 * 0.05 * ratios) ** 2)
    res = identify_resonance(ratios, amps)
    assert res["ratio_argmax"] == pytest.approx(1.0)
    assert res["ratio_parabolic"] == pytest.approx(0.998, abs=5e-3)
    assert res["curvature"] < 0.0


def test_identify_resonance_rejects_short_input():
    with pytest.raises(ValueError):
        identify_resonance([1.0, 1.1], [1.0, 2.0])


def test_harmonic_metrics_recovers_known_amplitude():
    t = np.linspace(0.0, 5.0, 501)
    f = 0.8
    y = 0.012 * np.cos(2 * math.pi * f * t + 0.3)
    met = harmonic_metrics(t, y, f, steady_frac=0.0)
    assert met["amplitude"] == pytest.approx(0.012, rel=1e-4)
    assert met["phase_deg"] == pytest.approx(math.degrees(0.3), abs=1e-2)


# ---------------------------------------------------------------------------
# solver-backed tests (coarse tank, loose tolerances)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def coarse_spec():
    return TankSpec(cells_per_height=16, u_lat=0.0035)


@pytest.fixture(scope="module")
def decay_run(coarse_spec):
    return run_decay_case(coarse_spec, mode=1, amplitude=1e-3, n_periods=2.0)


def test_free_decay_frequency_close_to_abramson(decay_run, coarse_spec):
    """Coarse-grid gate only: the node-level number comes from the 32/64 campaigns."""
    from tensorlbm.sloshing_resonance.identify import identify_mode

    f1 = abramson_frequency(1, coarse_spec.depth, coarse_spec.length, coarse_spec.g)
    fit = identify_mode(decay_run["t"], decay_run["a"][1], f1)
    assert abs(fit.f_damped / f1 - 1.0) < 0.15
    # coarse-grid ledger gate only; the campaign grid (H/32) reaches <0.5 %
    assert decay_run["max_mass_drift"] < 0.05


def test_free_decay_recorded_fields_are_consistent(decay_run):
    eta = decay_run["eta"]
    modal = decay_run["a"]
    assert eta.shape[0] == decay_run["t"].size
    assert modal[1].shape[0] == decay_run["t"].size
    # the mode-1 release must dominate the modal content
    amp = {n: float(np.max(np.abs(modal[n]))) for n in (1, 2, 3)}
    assert amp[1] > 2.0 * amp[2]


def test_forced_run_drives_mode_one_and_pressure(coarse_spec):
    case = run_forced_case(coarse_spec, ratio=1.0, sway_frac=0.002, n_periods=1.5)
    a1 = case["a"][1]
    assert np.max(np.abs(a1)) > 0.0
    # forced motion must drive the tank far above the 1e-4 m seed amplitude
    assert np.max(np.abs(a1)) > 5.0 * 1e-4
    pm = pressure_metrics(case, coarse_spec)
    assert len(pm["gauges"]) == len(set(coarse_spec.gauge_rows(case["depth_cells"])))
    assert pm["p_amp_Pa"] > 0.0
    assert math.isfinite(pm["impact_velocity_m_s"])
    runup = wall_runup_series(case)
    assert runup.shape == a1.shape
    # reconstruction must equal the odd-mode sum by construction
    manual = case["a"][1] + case["a"][3]
    assert np.allclose(runup, manual)


def test_response_smaller_at_second_mode_than_at_first(coarse_spec):
    """Horizontal sway: f1 response > f2 response (even modes are not excited)."""
    c1 = run_forced_case(coarse_spec, ratio=1.0, sway_frac=0.002, n_periods=1.5)
    c2 = run_forced_case(coarse_spec, ratio=abramson_frequency(2, H_FILL, L, G)
                         / abramson_frequency(1, H_FILL, L, G), sway_frac=0.002, n_periods=1.5)
    a1_first = float(np.max(np.abs(c1["a"][1])))
    a1_second = float(np.max(np.abs(c2["a"][1])))
    assert a1_first > a1_second


def test_build_case_rejects_unresolved_water_column():
    """A fill that resolves to fewer than 3 lattice cells must be refused."""
    with pytest.raises(ValueError):
        build_case(TankSpec(fill_ratio=0.02))
    with pytest.raises(ValueError):
        TankSpec(cells_per_height=4).scaling()