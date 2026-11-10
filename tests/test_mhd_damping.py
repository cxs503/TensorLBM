"""Magnetic damping of a shear wave: decay-rate anchor for the low-Rm MHD solver.

The exact solution ``u_x = U sin(k y) exp(-gamma t)`` with
``gamma = nu k^2 + nu Ha^2/h^2`` is an exact solution of the full nonlinear
incompressible low-Rm MHD system, so these tests check the solver against a
closed form rather than against a linearised reference.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.mhd.damping import (
    ShearWaveCase,
    damping_ladder,
    exact_amplitude,
    exact_profile,
    ha_scaling_exponent,
    magnetic_decay_rate,
    run_damping_case,
    viscous_decay_rate,
    wave_number,
)

# ---------------------------------------------------------------------------
# closed form
# ---------------------------------------------------------------------------


def test_magnetic_rate_is_quadratic_in_ha() -> None:
    """``gamma_magnetic = sigma B0^2/rho = nu Ha^2/h^2`` scales as ``Ha^2``."""
    nu, h = 0.05, 16.0
    rates = [magnetic_decay_rate(ha, nu, h) for ha in (1.0, 2.0, 4.0)]
    assert rates[1] / rates[0] == pytest.approx(4.0, rel=1e-12)
    assert rates[2] / rates[1] == pytest.approx(4.0, rel=1e-12)
    assert ha_scaling_exponent([1.0, 2.0, 4.0], rates) == pytest.approx(2.0, rel=1e-12)


def test_case_rates_add_up_and_match_the_hartmann_number() -> None:
    """``gamma_magnetic = nu Ha^2/h^2`` with ``h`` the same reference length."""
    case = ShearWaveCase(ny=32, ha=10.0, nu=0.05)
    assert case.wave_number == pytest.approx(2.0 * math.pi / 32.0)
    assert case.viscous_rate == pytest.approx(0.05 * case.wave_number**2)
    assert case.magnetic_rate == pytest.approx(0.05 * 100.0 / 16.0**2)
    assert case.rate == pytest.approx(case.viscous_rate + case.magnetic_rate)
    # consistency of the closure: sigma B0^2/rho must equal the rate
    assert case.force_coefficient == pytest.approx(case.magnetic_rate * case.rho0)


def test_exact_profile_is_zero_mean_and_peaks_at_the_amplitude() -> None:
    case = ShearWaveCase(ny=64, ha=5.0)
    u = exact_profile(case, 0.0)
    assert abs(float(u.mean())) < 1e-12 * case.amplitude
    assert float(np.max(np.abs(u))) == pytest.approx(case.amplitude, rel=1e-12)
    assert exact_amplitude(case, 0.0) == pytest.approx(case.amplitude)
    # one e-fold
    assert exact_amplitude(case, 1.0 / case.rate) == pytest.approx(
        case.amplitude * math.exp(-1.0), rel=1e-12
    )


def test_viscous_rate_scales_with_k_squared() -> None:
    nu = 0.05
    k1, k2 = wave_number(32, 1), wave_number(32, 2)
    assert viscous_decay_rate(nu, k2) / viscous_decay_rate(nu, k1) == pytest.approx(
        4.0, rel=1e-12
    )


# ---------------------------------------------------------------------------
# validation envelope
# ---------------------------------------------------------------------------


def test_case_validation_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError):
        ShearWaveCase(ny=2)
    with pytest.raises(ValueError):
        ShearWaveCase(ha=-1.0)
    with pytest.raises(ValueError):
        ShearWaveCase(nu=0.0)
    with pytest.raises(ValueError):
        ShearWaveCase(mode=0)
    with pytest.raises(ValueError):
        ShearWaveCase(amplitude=0.2)  # outside the low-Mach window
    with pytest.raises(ValueError):
        magnetic_decay_rate(5.0, 0.05, 0.0)
    with pytest.raises(ValueError):
        ha_scaling_exponent([5.0], [1.0])


# ---------------------------------------------------------------------------
# solver vs the exact rate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ha", [10.0, 20.0])
def test_measured_decay_rate_matches_the_exact_rate(ha: float) -> None:
    """Strong magnetic damping: the fitted rate must sit on the closed form."""
    run = run_damping_case(ShearWaveCase(ny=32, ha=ha, nu=0.05, amplitude=0.02))
    assert run["fit_r2"] > 0.9999
    assert run["rel_rate_error"] < 0.01
    assert run["rel_magnetic_error"] < 0.01
    # the energy decays at twice the amplitude rate
    assert run["gamma_energy_fit"] == pytest.approx(run["gamma_fit"], rel=0.02)


def test_zero_field_control_decays_at_the_viscous_rate_only() -> None:
    """``Ha = 0``: no Lorentz force, the mode must decay as ``nu k^2``."""
    run = run_damping_case(ShearWaveCase(ny=32, ha=0.0, nu=0.05, amplitude=0.02))
    assert run["gamma_magnetic"] == 0.0
    assert run["rel_rate_error"] < 0.01
    assert run["gamma_fit"] == pytest.approx(run["gamma_viscous"], rel=0.01)


def test_amplitude_history_is_a_pure_exponential() -> None:
    run = run_damping_case(ShearWaveCase(ny=32, ha=10.0, nu=0.05, steps=120, sample_every=4))
    hist = run["history"]
    assert len(hist) >= 10
    t = np.asarray([h["t"] for h in hist], dtype=float)
    a = np.asarray([h["amplitude"] for h in hist], dtype=float)
    exact = np.asarray([h["exact"] for h in hist], dtype=float)
    assert np.all(a > 0.0)
    assert float(np.max(np.abs(a - exact) / exact)) < 5e-3
    # monotone decay of a passively damped mode
    assert np.all(np.diff(a) < 0.0)
    assert run["rel_amplitude_end"] < 5e-3
    assert t[-1] == 120.0


def test_shear_wave_carries_no_net_momentum() -> None:
    """The momentum-conserving closure must keep the domain mean at zero."""
    run = run_damping_case(ShearWaveCase(ny=32, ha=10.0, nu=0.05, steps=80, sample_every=8))
    profile = np.asarray(run["profile"]["u_lbm"], dtype=float)
    assert abs(float(profile.mean())) < 1e-9 * run["amplitude0"]
    assert run["max_mach"] < 0.1


def test_ha_ladder_recovers_the_quadratic_scaling_law() -> None:
    """Criterion (d) of the node: ``gamma ~ Ha^2`` and rate error <= 5 %."""
    ladder = damping_ladder((5.0, 10.0, 20.0), ny=16, nu=0.05, amplitude=0.02, progress=False)
    assert ladder["scaling_exponent"] == pytest.approx(2.0, abs=0.05)
    for run in ladder["runs"].values():
        assert run["rel_rate_error"] < 0.05
    crit = ladder["criteria"]
    assert crit["c4_damping_rate"]["passed"]
    assert crit["c4b_scaling_exponent"]["passed"]
    assert crit["c4c_viscous_control"]["passed"]

def test_damping_rate_converges_with_grid() -> None:
    """Second-order convergence of the *measured* rate (criterion f on this branch).

    The shear wave is a single Fourier mode, so refining ``ny`` at fixed ``k``
    resolves the same physical decay with a smaller truncation error.
    """
    from tensorlbm.mhd.damping import damping_refinement

    ref = damping_refinement(ha=10.0, ny_list=(16, 32, 64), nu=0.05)
    errors = ref["rel_rate_errors"]
    assert errors[-1] < errors[0]
    assert min(ref["orders"]) >= 1.5, f"orders {ref['orders']} errors {errors}"
    assert ref["finest_order"] >= 1.5


def test_damping_amplitude_error_converges_with_grid() -> None:
    """The end-of-run amplitude error must fall under refinement as well."""
    errors = []
    for ny in (16, 32, 64):
        run = run_damping_case(ShearWaveCase(ny=ny, ha=10.0, nu=0.05, amplitude=0.02))
        errors.append(run["rel_amplitude_end"])
    assert errors[-1] < errors[0]
    order = float(np.log(errors[0] / errors[-1]) / np.log(4.0))
    assert order >= 1.5, f"amplitude-error order {order:.2f} (errors {errors})"
