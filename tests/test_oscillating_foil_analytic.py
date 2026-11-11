"""Tests for the oscillating-foil verification face: analytic anchors.

Covers the Theodorsen/Garrick layer (limits, independent algebraic form,
phasor-vs-quadrature), the kinematics helpers, the NACA 0012 geometry used by
the immersed boundary, and the machine-checkable criteria module (each criterion
is exercised both with data that must PASS and with data that must FAIL, i.e.
the criteria themselves have negative controls).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.flapping_propulsion import (
    FlappingKinematics,
    garrick_coefficients,
    theodorsen_C,
    theodorsen_C_bessel,
    thrust_reversal_heave,
)
from tensorlbm.oscillating_foil_criteria import (
    criterion_c1_linear_region,
    criterion_c1b_amplitude_scaling,
    criterion_c2_efficiency_peak,
    criterion_c3_degenerate_limits,
    criterion_c4_energy_ledger,
    criterion_c5_strouhal_monotone,
    criterion_c6_grid_convergence,
    criterion_c7_negative_controls,
    criterion_c8_reproducibility,
    summarise,
)
from tensorlbm.oscillating_foil_lbm import (
    FoilMotion,
    naca0012_half_thickness,
    naca0012_markers,
    k_from_strouhal,
    strouhal_from_k,
)

A_AXIS = -0.5


# ---------------------------------------------------------------------------
# Theodorsen function
# ---------------------------------------------------------------------------
def test_theodorsen_limits():
    assert abs(theodorsen_C(0.0) - 1.0) == 0.0
    assert abs(theodorsen_C(1e-9) - 1.0) < 1e-6
    assert abs(theodorsen_C(1e5) - 0.5) < 1e-4


@pytest.mark.parametrize("k", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_theodorsen_two_independent_paths(k):
    """Hankel-function path and modified-Bessel path must agree."""
    assert abs(theodorsen_C(k) - theodorsen_C_bessel(k)) < 1e-10


def test_theodorsen_monotone_real_part():
    ks = np.linspace(0.05, 10.0, 40)
    f = np.array([theodorsen_C(k).real for k in ks])
    assert np.all(np.diff(f) < 0.0)  # F decreases from 1 to 1/2
    assert f[-1] > 0.499


# ---------------------------------------------------------------------------
# Garrick: pure-heave closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("k", [0.25, 0.5, 1.0, 2.0])
@pytest.mark.parametrize("h0c", [0.02, 0.1, 0.25])
def test_pure_heave_closed_form(k, h0c):
    """``C_T = 4 pi k^2 (h0/c)^2 |C|^2``, ``C_P = 4 pi k^2 (h0/c)^2 F``.

    This is the algebraic form derived in the report (quasi-steady limit
    ``C_T -> pi (h0/b)^2 k^2`` checked separately below); it is independent of
    the phasor machinery in :func:`garrick_coefficients`.
    """
    kin = FlappingKinematics(U=1.0, c=1.0, h0=h0c, alpha0=0.0, phi=0.0, a=A_AXIS)
    g = garrick_coefficients(kin, k)
    C = theodorsen_C(k)
    ct = 4.0 * math.pi * k**2 * h0c**2 * abs(C) ** 2
    cp = 4.0 * math.pi * k**2 * h0c**2 * C.real
    assert g.C_T == pytest.approx(ct, rel=1e-12)
    assert g.C_P == pytest.approx(cp, rel=1e-12)
    assert g.eta == pytest.approx(abs(C) ** 2 / C.real, rel=1e-12)


@pytest.mark.parametrize("k", [1e-4, 1e-3])
def test_quasi_steady_limit_is_first_order_in_k(k):
    """``C_T -> pi (h0/b)^2 k^2`` with the known O(k) correction."""
    h0c = 0.1
    kin = FlappingKinematics(U=1.0, c=1.0, h0=h0c, alpha0=0.0, phi=0.0, a=A_AXIS)
    g = garrick_coefficients(kin, k)
    limit = math.pi * (h0c / 0.5) ** 2 * k**2
    rel = abs(g.C_T - limit) / limit
    assert rel == pytest.approx(math.pi * k, rel=0.05)


def test_phasor_versus_time_quadrature():
    for k in (0.5, 1.0, 2.0):
        kin = FlappingKinematics(
            U=1.0, c=1.0, h0=0.25, alpha0=math.radians(10.0), phi=math.pi / 2, a=A_AXIS
        )
        gp = garrick_coefficients(kin, k)
        gq = garrick_coefficients(kin, k, n_samples=720)
        assert gp.C_T == pytest.approx(gq.C_T, abs=1e-12)
        assert gp.C_P == pytest.approx(gq.C_P, abs=1e-12)


def test_efficiency_below_unity_and_energy_balance():
    """``eta <= 1`` for every harmonic motion: ``C_P >= C_T`` in inviscid theory."""
    for k in (0.25, 0.5, 1.0, 2.0, 3.0):
        for h0c, a0 in ((0.25, 0.0), (0.0, math.radians(10.0)), (0.25, math.radians(10.0))):
            kin = FlappingKinematics(U=1.0, c=1.0, h0=h0c, alpha0=a0, phi=math.pi / 2, a=A_AXIS)
            g = garrick_coefficients(kin, k)
            if g.C_P > 0:
                assert g.eta <= 1.0 + 1e-12


def test_thrust_reversal_heave_brackets_the_root():
    res = thrust_reversal_heave(0.5, math.radians(10.0), math.pi / 2, A_AXIS)
    assert res["exists"]
    h0c = res["h0_over_c_crit"]
    below = FlappingKinematics(
        U=1.0, c=1.0, h0=0.5 * h0c, alpha0=math.radians(10.0), phi=math.pi / 2, a=A_AXIS
    )
    above = FlappingKinematics(
        U=1.0, c=1.0, h0=1.5 * h0c, alpha0=math.radians(10.0), phi=math.pi / 2, a=A_AXIS
    )
    assert garrick_coefficients(below, 0.5).C_T < 0.0 < garrick_coefficients(above, 0.5).C_T


# ---------------------------------------------------------------------------
# kinematics / geometry
# ---------------------------------------------------------------------------
def test_strouhal_round_trip():
    for k in (0.5, 1.0, 2.0):
        for h0c in (0.02, 0.25):
            st = strouhal_from_k(k, h0c)
            assert k_from_strouhal(st, h0c) == pytest.approx(k, rel=1e-14)


def test_motion_modes_and_velocity_consistency():
    m = FoilMotion(c=40.0, U=0.1, h0=10.0, alpha0=math.radians(10.0), k=1.0)
    dt = 1e-3
    for t in (0.0, 137.0, 620.0):
        s0 = m.state(t)
        s1 = m.state(t + dt)
        num_h = (s1["h"] - s0["h"]) / dt
        num_a = (s1["alpha"] - s0["alpha"]) / dt
        assert num_h == pytest.approx(s0["h_dot"], abs=1e-4)
        assert num_a == pytest.approx(s0["alpha_dot"], abs=1e-6)


def test_naca0012_geometry():
    assert naca0012_half_thickness(0.0) == pytest.approx(0.0, abs=1e-12)
    assert 2.0 * naca0012_half_thickness(0.3) == pytest.approx(0.12, rel=0.01)
    # closed trailing edge
    assert naca0012_half_thickness(1.0) == pytest.approx(0.0, abs=2e-3)
    xs, ys = naca0012_markers(40.0, 0.5)
    assert xs.min() == pytest.approx(0.0, abs=0.2)
    assert xs.max() == pytest.approx(40.0, abs=0.5)
    assert ys.max() <= 0.06 * 40.0 + 0.1
    assert ys.min() >= -0.06 * 40.0 - 0.1
    # consecutive marker spacing does not exceed the requested spacing
    ds = np.hypot(np.diff(np.append(xs, xs[0])), np.diff(np.append(ys, ys[0])))
    assert ds.max() <= 1.2 * 0.5


# ---------------------------------------------------------------------------
# criteria module: PASS and FAIL controls for each criterion
# ---------------------------------------------------------------------------
def _lin_case(label, C_T, cd0, th, sigma=1e-4, h0_over_c=0.02, k=1.0):
    return {
        "label": label,
        "C_T": C_T,
        "cd0": cd0,
        "theory": {"C_T": th},
        "C_T_cycle_std": sigma,
        "h0_over_c": h0_over_c,
        "k": k,
        "strouhal": strouhal_from_k(k, h0_over_c),
    }


def test_c1_linear_region_pass_and_fail():
    good = [_lin_case("a", 0.0100 - 0.05, 0.05, 0.0100)]
    bad = [_lin_case("b", 0.0020 - 0.05, 0.05, 0.0100)]
    assert criterion_c1_linear_region(good)["verdict"] == "PASS"
    assert criterion_c1_linear_region(bad)["verdict"] == "FAIL"


def test_c1_withholds_below_noise_floor():
    noisy = [_lin_case("n", 0.0 - 0.05, 0.05, 5e-4, sigma=1e-3)]
    res = criterion_c1_linear_region(noisy)
    assert res["verdict"] == "WITHHELD"
    assert res["value"]["cases"][0]["resolvable"] is False


def test_c1b_amplitude_scaling():
    d = 1e-4
    ok = [
        _lin_case("lo", 4 * d - 0.05, 0.05, 4 * d, h0_over_c=0.02),
        _lin_case("hi", 16 * d - 0.05, 0.05, 16 * d, h0_over_c=0.04),
    ]
    res = criterion_c1b_amplitude_scaling(ok)
    assert res["verdict"] == "PASS"
    assert res["value"]["observed_ratio"] == pytest.approx(4.0, rel=1e-6)


def test_c2_efficiency_peak_detects_shift():
    ks = [0.5, 1.0, 1.5, 2.0]
    eta_th = [0.62, 0.66, 0.63, 0.58]
    good = [
        {"k": k, "eta": e, "theory": {"eta": e}} for k, e in zip(ks, eta_th, strict=True)
    ]
    assert criterion_c2_efficiency_peak(good)["verdict"] == "PASS"
    shifted = [
        {"k": k, "eta": e, "theory": {"eta": t}}
        for k, e, t in zip(ks, [0.35, 0.40, 0.45, 0.66], eta_th, strict=True)
    ]
    assert criterion_c2_efficiency_peak(shifted)["verdict"] == "FAIL"


def test_c3_degenerate_limits():
    analytic = {
        "quasi_steady_rel_error": 3e-4,
        "C_at_zero_minus_one": 0.0,
        "C_at_inf_minus_half": 1e-6,
        "pure_heave_rel_error": 1e-16,
    }
    z = {"C_T": 0.0, "C_P": 0.0}
    assert criterion_c3_degenerate_limits(analytic, z)["verdict"] == "PASS"
    assert (
        criterion_c3_degenerate_limits({**analytic, "quasi_steady_rel_error": 0.1}, z)[
            "verdict"
        ]
        == "FAIL"
    )
    assert criterion_c3_degenerate_limits(analytic, {"C_T": 1e-3, "C_P": 0.0})["verdict"] == "FAIL"


def test_c4_energy_ledger():
    good = {"work": 1.0, "dke": 0.1, "phi": 0.89, "flux": 0.0, "cycles": 8}
    assert criterion_c4_energy_ledger(good)["verdict"] == "PASS"
    bad = {"work": 1.0, "dke": 0.1, "phi": 0.5, "flux": 0.0, "cycles": 8}
    assert criterion_c4_energy_ledger(bad)["verdict"] == "FAIL"


def test_c5_monotone_and_non_monotone():
    mono = [
        {"strouhal": st, "C_T": 0.2 * st, "C_T_cycle_std": 1e-4}
        for st in (0.1, 0.2, 0.3, 0.4)
    ]
    assert criterion_c5_strouhal_monotone(mono)["verdict"] == "PASS"
    bad = [
        {"strouhal": st, "C_T": c, "C_T_cycle_std": 1e-4}
        for st, c in zip((0.1, 0.2, 0.3, 0.4), (0.1, 0.05, 0.3, 0.4), strict=True)
    ]
    assert criterion_c5_strouhal_monotone(bad)["verdict"] == "FAIL"


def test_c6_grid_convergence():
    exact = 0.2000
    # error ~ h^2 ~ 1/chord^2 (coarse 30, medium 40, fine 53)
    amp = 0.03 * exact * 53**2
    fine = exact + amp / 53**2
    med = exact + amp / 40**2
    coarse = exact + amp / 30**2
    ok = [
        {"chord": 30, "value": coarse},
        {"chord": 40, "value": med},
        {"chord": 53, "value": fine},
    ]
    res = criterion_c6_grid_convergence(ok)
    assert res["value"]["p_observed"] == pytest.approx(2.0, rel=0.2)
    assert res["verdict"] == "PASS"
    bad = [
        {"chord": 30, "value": exact + 1.5},
        {"chord": 40, "value": med},
        {"chord": 53, "value": fine},
    ]
    assert criterion_c6_grid_convergence(bad)["verdict"] == "FAIL"


def test_c7_negative_controls_pass_and_fail():
    locked = {"C_T": -0.06, "cd_independent": -0.055}
    ibm_off = {"C_T": 0.0, "C_P": 0.0}
    flip = {"C_T": 0.235}
    ref = {"C_T": 0.237}
    assert criterion_c7_negative_controls(locked, ibm_off, flip, ref)["verdict"] == "PASS"
    # a sign error in the force integration flips the thrust -> must FAIL
    assert (
        criterion_c7_negative_controls(locked, ibm_off, {"C_T": -0.235}, ref)["verdict"]
        == "FAIL"
    )
    # IBM that is not switched off -> must FAIL
    assert (
        criterion_c7_negative_controls(locked, {"C_T": 0.2, "C_P": 0.1}, flip, ref)["verdict"]
        == "FAIL"
    )


def test_c8_reproducibility():
    assert criterion_c8_reproducibility({"sha256": "a", "C_T": 1.0}, {"sha256": "a", "C_T": 1.0})[
        "verdict"
    ] == "PASS"
    assert criterion_c8_reproducibility({"sha256": "a", "C_T": 1.0}, {"sha256": "b", "C_T": 1.0})[
        "verdict"
    ] == "FAIL"


def test_summarise():
    s = summarise([{"id": "C1", "verdict": "PASS"}, {"id": "C2", "verdict": "FAIL"}])
    assert s["n_pass"] == 1 and s["n_fail"] == 1 and s["all_pass"] is False