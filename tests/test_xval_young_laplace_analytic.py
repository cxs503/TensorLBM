"""Gates for XVAL-LBM-YOUNG-LAPLACE-ANALYTIC-PLUS-DROPLET-CALI.

The tests pin the *analysis chain*, not the physics verdict: the physics verdict
lives in ``artifacts/summary.json`` and is produced by ``run_xval.py``.  Each test
here asserts an invariant that would otherwise be easy to break silently:

1. the two closed forms of the free-energy tension are algebraically the same
   quantity and reproduce the model's declared ``sigma`` (no hand-copied constant);
2. the gradient-energy form is verified against a *numerical* integral of the
   analytic tanh profile (the closed form is not taken on faith);
3. the measurement chain recovers a known radius and a known pressure jump from a
   synthetic field, and the interface shell is excluded from both masks;
4. the gradient-energy route, applied to the analytic 2D droplet profile,
   recovers the analytic tension - so a later mismatch on a simulated field is a
   statement about the simulation, not about the post-processing;
5. the double-count defect (integrating across a full diameter crosses two
   interfaces and returns 2 sigma) is detected by the same route used in the
   report - a regression guard for a defect that was actually shipped once;
6. the planted-defect machinery really rejects defects (asserts magnitudes, not
   just that the function runs).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XV = os.path.join(REPO, "xval", "XVAL-LBM-YOUNG-LAPLACE-ANALYTIC-PLUS-DROPLET-CALI")
sys.path.insert(0, os.path.join(REPO, "src"))
sys.path.insert(0, XV)

import xval_lib as XL  # noqa: E402


@pytest.mark.parametrize("sigma,xi", [(0.005, 3.0), (0.01, 3.0), (0.02, 3.0), (0.01, 2.0), (0.01, 4.0)])
def test_two_closed_forms_of_sigma_agree_with_the_declared_sigma(sigma: float, xi: float) -> None:
    c = XL.soy_constants(sigma, xi)
    assert c["sigma_identity_residual"] < 1e-12
    # recomputed from (kappa, A, B) - must equal the declared target, not by copy
    assert c["sigma_analytic"] == pytest.approx(sigma, rel=1e-12)
    assert c["xi_from_identity"] == pytest.approx(xi, rel=1e-12)
    assert c["psi0"] == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("sigma,xi", [(0.01, 3.0), (0.02, 2.0)])
def test_gradient_energy_closed_form_matches_numerical_integral(sigma: float, xi: float) -> None:
    c = XL.soy_constants(sigma, xi)
    x = np.linspace(-40.0 * xi, 40.0 * xi, 2_000_001)
    psi = XL.analytic_tanh_profile(np.abs(x), radius=0.0, xi=xi, psi0=c["psi0"])
    dpsi = np.gradient(psi, x)
    numerical = c["kappa"] * np.trapezoid(dpsi**2, x)
    assert numerical == pytest.approx(c["sigma_analytic"], rel=2e-3)


def test_equivalent_radius_recovers_a_known_disc() -> None:
    n = 96
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.hypot(xx - 40.3, yy - 51.7)
    psi = np.where(r < 12.0, 1.0, -1.0)
    R_eq, centre = XL.equivalent_radius(psi)
    assert R_eq == pytest.approx(12.0, rel=0.02)
    assert centre[0] == pytest.approx(40.3, abs=0.6)
    assert centre[1] == pytest.approx(51.7, abs=0.6)


def test_pressure_jump_excludes_the_interface_shell() -> None:
    n = 128
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    centre = (n / 2.0, n / 2.0)
    r = XL.radial_distance((n, n), centre)
    R, xi = 20.0, 3.0
    # +1e-3 inside, -1e-3 outside, and a huge (unphysical) value inside the shell:
    # if the shell leaked into a mask, dp would be wrong by ~1e3 relative.
    p = np.where(r < R - 1.5 * xi, 1.0e-3, np.where(r > R + 1.5 * xi, -1.0e-3, 1.0))
    dp, n_in, n_out = XL.pressure_jump(p, r, R, xi)
    assert dp == pytest.approx(2.0e-3, rel=1e-12)
    assert n_in > 0 and n_out > 0
    # the masks used must not contain any shell cell
    assert np.all(np.abs(p[r < R - 1.5 * xi]) < 0.5)


def test_gradient_route_recovers_analytic_sigma_on_the_analytic_profile() -> None:
    """The dp-free route is exercised end to end on a field with a known answer."""
    n = 256
    sigma, xi, R = 0.01, 3.0, 48.0
    c = XL.soy_constants(sigma, xi)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = np.hypot(xx - n / 2.0, yy - n / 2.0)
    psi = XL.analytic_tanh_profile(r, R, xi, c["psi0"])
    gy, gx = np.gradient(psi)
    s_cap = XL.gradient_energy_sigma(gx, gy, c["kappa"], R)
    assert s_cap == pytest.approx(c["sigma_analytic"], rel=0.05)
    # full-diameter integration crosses two interfaces -> 2 sigma (the shipped defect)
    s_double = XL.gradient_energy_sigma(2.0 * gx, 2.0 * gy, c["kappa"], R)
    assert s_double > 1.5 * c["sigma_analytic"]


def test_interface_width_fit_recovers_xi() -> None:
    n = 128
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    centre = (n / 2.0, n / 2.0)
    r = XL.radial_distance((n, n), centre)
    psi = XL.analytic_tanh_profile(r, 30.0, 3.0, 1.0)
    fit = XL.interface_width_fit(psi, 30.0, centre)
    assert fit["xi_meas"] == pytest.approx(3.0, rel=0.05)
    assert fit["R_fit"] == pytest.approx(30.0, abs=1.0)


def _synthetic_case(mult: float = 1.0) -> dict:
    sigma, xi, R, n = 0.01, 3.0, 15.0, 96
    c = XL.soy_constants(sigma, xi)
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    r = XL.radial_distance((n, n), (n / 2.0, n / 2.0))
    dp = XL.laplace_reference_2d(c["sigma_analytic"], R) * mult
    p = np.where(r < R, dp, 0.0)
    return {
        "case_id": "synthetic",
        "params": {"n": n, "radius": R, "sigma_target": sigma, "xi": xi},
        "analytic": c,
        "measures": {
            "dp_eq": dp,
            "dp_fixed": dp,
            "R_eq": R,
            "sigma_cap": c["sigma_analytic"],
        },
        "ref": {"dp_laplace": dp, "sigma_eff": dp * R},
        "_fields": {"p": p, "r": r, "psi": np.where(r < R, 1.0, -1.0)},
    }


def test_planted_defects_are_actually_rejected() -> None:
    """Falsifiability of the analysis chain: the defects must exceed the 15% band."""
    defects = XL.planted_defects([_synthetic_case()])
    assert defects["sphere_factor_in_2d"]["max_abs_err_pct"] == pytest.approx(100.0, rel=1e-6)
    assert defects["sigma_coef_half_kappa"]["max_abs_err_pct"] > 15.0
    assert defects["double_count_interface"]["max_abs_err_pct"] > 15.0
    caught = sum(1 for v in defects.values() if abs(v["max_abs_err_pct"]) > 15.0)
    assert caught >= 3


def test_calibration_recovers_the_sigma_used_to_build_the_data() -> None:
    sigma, xi = 0.01, 3.0
    c = XL.soy_constants(sigma, xi)
    radii = [10.0, 15.0, 20.0]
    dp = [XL.laplace_reference_2d(c["sigma_analytic"], r) for r in radii]
    fit = XL.fit_sigma_through_origin(np.array(dp), 1.0 / np.array(radii))
    assert fit == pytest.approx(c["sigma_analytic"], rel=1e-12)
    loo = XL.leave_one_radius_out(radii, dp)
    assert max(abs(r["err_pct"]) for r in loo) < 1e-9
    # and a 3D factor in the data is not absorbed by the 2D calibration
    dp3 = [2.0 * v for v in dp]
    fit3 = XL.fit_sigma_through_origin(np.array(dp3), 1.0 / np.array(radii))
    assert abs(fit3 - c["sigma_analytic"]) / c["sigma_analytic"] > 0.15


def test_laplace_reference_rejects_unknown_dimensionality() -> None:
    assert XL.laplace_reference_2d(0.04, 2.0, dims=2) == pytest.approx(0.02)
    assert XL.laplace_reference_2d(0.04, 2.0, dims=3) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        XL.laplace_reference_2d(0.04, 2.0, dims=4)


def test_kernel_smoke_static_droplet_is_finite_and_at_rest() -> None:
    """Tiny solver-backed smoke test: finiteness and rest, not physics."""
    from tensorlbm.multiphase_static_droplet import soy_free_energy_droplet

    out = soy_free_energy_droplet(
        nx=32, ny=32, radius=8.0, steps=40, backend="torch", dtype="float64"
    )
    for key in ("dp", "dp_analytical", "u_max", "sigma_from_capillary_energy"):
        assert math.isfinite(out[key])
    assert 0.0 < out["u_max"] < 1e-2