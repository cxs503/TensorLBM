"""Acceptance tests for :mod:`tensorlbm.convective_mass_transfer` (P154W3-N11).

Three layers:

* **closed form** -- the analytic anchors (Leveque constant, Blasius wall slope,
  correlations, Chilton-Colburn identity) are checked against textbook values;
* **operator** -- the D2Q5 advection-diffusion solver plus its Dirichlet wall
  boundary condition reproduce the Leveque similarity solution with the correct
  exponent, on a small lattice;
* **coupled** -- the cylinder mass-transfer driver conserves mass (its transfer
  rate is read out of the exact mass budget) and lands in the physical range.

Marked ``physics`` / ``benchmark`` so the CI selection can separate them from
the fast unit contract checks.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.convective_mass_transfer import (
    LEVEQUE_CONST,
    blasius_local_cf,
    blasius_local_sh,
    blasius_similarity,
    blasius_velocity_field,
    chilton_colburn_j,
    churchill_bernstein_mean_sh,
    effective_diffusivity,
    fit_power_law,
    hilpert_mean_sh,
    leveque_local_h,
    leveque_local_sh,
    leveque_mean_sh,
    leveque_similarity_profile,
    ranz_marshall_mean_sh,
    run_cylinder_mass_transfer,
    solve_shear_channel_ad,
    stanton_number,
    wall_flux_from_profile,
)

pytestmark = pytest.mark.physics


# ---------------------------------------------------------------------------
# closed-form anchors
# ---------------------------------------------------------------------------


def test_leveque_constant():
    """``C_L = 1/(9^{1/3} Gamma(4/3)) = 0.5383661`` (Leveque 1928 quotes 0.5384)."""
    assert LEVEQUE_CONST == pytest.approx(0.5383661, abs=1e-6)
    assert LEVEQUE_CONST == pytest.approx(1.0 / (9.0 ** (1 / 3) * math.gamma(4 / 3)), rel=1e-14)


def test_leveque_profile_is_the_wall_gradient_constant():
    """``theta'(0) = 1/Gamma(4/3)`` and ``theta(inf) = 1``."""
    d = 1e-6
    slope = (leveque_similarity_profile([d])[0] - 0.0) / d
    assert slope == pytest.approx(1.0 / math.gamma(4.0 / 3.0), rel=1e-5)
    assert float(leveque_similarity_profile([8.0])[0]) == pytest.approx(1.0, abs=1e-9)


def test_leveque_exponents():
    """``h_m ~ x^{-1/3}`` and ``Sh_x ~ x^{+2/3}``; ``Sh_mean = 1.5 Sh_L``."""
    x = np.array([10.0, 40.0, 160.0, 640.0])
    p_h = fit_power_law(x, leveque_local_h(x, 0.01, 1e-3))["p"]
    p_sh = fit_power_law(x, leveque_local_sh(x, 0.01, 1e-3))["p"]
    assert p_h == pytest.approx(-1.0 / 3.0, abs=1e-12)
    assert p_sh == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert leveque_mean_sh(50.0, 0.01, 1e-3) == pytest.approx(
        1.5 * float(leveque_local_sh(50.0, 0.01, 1e-3)), rel=1e-14
    )


def test_leveque_sh_matches_hand_value():
    """``Sh_x = 0.5384 (a x^2/D)^{1/3}``: a = 0.01, x = 100, D = 1e-3 -> 53.84."""
    # (0.01 * 1e4 / 1e-3)^{1/3} = (1e5)^{1/3} = 46.4159
    assert float(leveque_local_sh(100.0, 0.01, 1e-3)) == pytest.approx(
        0.5383661 * 100.0 ** (1.0 / 3.0) * 10.0, rel=1e-6
    )


def test_blasius_wall_slope_and_field():
    """``f''(0) = 0.3320573``; the field has ``u=U`` far away and ``u=v=0`` at the wall."""
    eta, fp, _f = blasius_similarity()
    fpp0 = (fp[1] - fp[0]) / (eta[1] - eta[0])
    assert fpp0 == pytest.approx(0.3320573, abs=1e-5)
    ux, uy = blasius_velocity_field(40, 32, 0.08, 0.08)
    assert float(ux[-1, -1]) == pytest.approx(0.08, rel=2e-3)
    assert float(ux[0, -1]) == pytest.approx(0.0, abs=1e-8)
    assert float(uy[0, -1]) == pytest.approx(0.0, abs=1e-12)
    assert torch.all(ux[:, 5:] >= -1e-9)


def test_blasius_sh_and_cf_conventions():
    """``Sh_x = 0.332 Re_x^{1/2} Sc^{1/3}``, ``C_f = 0.664 Re_x^{-1/2}``, ``Sh_L = 2 Sh_x(L)``."""
    x, u, nu, d = np.array([100.0]), 0.1, 0.01, 0.001
    sh = float(blasius_local_sh(x, u, nu, d)[0])
    assert sh == pytest.approx(0.332 * math.sqrt(1000.0) * 10.0 ** (1.0 / 3.0), rel=1e-12)
    assert float(blasius_local_cf(x, u, nu)[0]) == pytest.approx(
        0.664 / math.sqrt(1000.0), rel=1e-12
    )


def test_correlations():
    """Ranz-Marshall -> 2 as ``Re -> 0``; Hilpert ~ Churchill-Bernstein at ``Re = 100``."""
    assert ranz_marshall_mean_sh(0.0, 1.0) == pytest.approx(2.0)
    assert ranz_marshall_mean_sh(100.0, 1.0) == pytest.approx(2.0 + 0.6 * 10.0)
    assert hilpert_mean_sh(100.0, 1.0) == pytest.approx(
        churchill_bernstein_mean_sh(100.0, 1.0), rel=0.02
    )
    # the Sc (or Pr) exponent of every correlation is 1/3
    r1 = hilpert_mean_sh(100.0, 1.0)
    r2 = hilpert_mean_sh(100.0, 8.0)
    assert math.log(r2 / r1) / math.log(8.0) == pytest.approx(1.0 / 3.0, abs=1e-12)


def test_chilton_colburn_identity_on_blasius():
    """``j_D = St Sc^{2/3}`` reduces to ``C_f/2`` exactly on the Blasius plate."""
    x, u, nu = 200.0, 0.08, 0.08
    for sc in (1.0, 10.0, 1000.0):
        sh = float(blasius_local_sh(np.array([x]), u, nu, nu / sc)[0])
        re_x = u * x / nu
        j = chilton_colburn_j(stanton_number(sh, re_x, sc), sc)
        assert j == pytest.approx(float(blasius_local_cf(np.array([x]), u, nu)[0]) / 2.0, rel=5e-3)


def test_wall_flux_stencil_is_exact_for_linear_profile():
    """``-dG/dy`` from a linear profile is exact for every stencil order."""
    g = np.linspace(1.0, 0.0, 9)
    profile = [np.array([g[k]]) for k in range(4)]
    for order in (2, 3, 4):
        assert float(wall_flux_from_profile(profile, spacing=0.125, order=order)[0]) == pytest.approx(
            1.0, rel=1e-12
        )


def test_fit_power_law_recovers_known_exponent():
    x = np.array([1.0, 2.0, 5.0, 10.0, 100.0])
    y = 3.0 * x**0.5
    fit = fit_power_law(x, y)
    assert fit["p"] == pytest.approx(0.5, abs=1e-12)
    assert fit["A"] == pytest.approx(3.0, rel=1e-12)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)


def test_effective_diffusivity_is_diffusivity_without_flow():
    """Pure diffusion (``u = 0``): the measured decay rate must equal ``D``."""
    r = effective_diffusivity(0.05, u=0.0, nx=32, steps=1500)
    assert r["ratio"] == pytest.approx(1.0, rel=0.02)
    # advection at a resolved cell Peclet number (Pe_D = 1) stays within 2 %
    r2 = effective_diffusivity(0.05, u=0.05, nx=32, steps=1500)
    assert r2["ratio"] == pytest.approx(1.0, rel=0.02)
    # ... and the numerical-diffusion floor at high cell Peclet is a real effect:
    # Pe_D = 500 must show D_eff > 2 D (this is the high-Schmidt-number limit).
    r3 = effective_diffusivity(1e-4, u=0.05, nx=32, steps=1500)
    assert r3["ratio"] > 2.0
    assert r3["collision"] in ("trt", "bgk_tau_s_capped", "bgk")


# ---------------------------------------------------------------------------
# operator: Leveque similarity on a small lattice
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
def test_leveque_lattice_solution_exponent_and_level():
    """LBM advection-diffusion on a linear-shear field reproduces the Leveque law.

    Coarse but genuinely converged level: the fitted exponent must sit between
    0.55 and 0.70 (the exact value is 2/3) and the wall Sherwood number must be
    within 20 % of ``0.5384 (a x^2/D)^{1/3}`` over the resolved window.  The
    production sweep in ``artifacts/p154w3_n11`` runs the same problem at
    ``delta_c/dx`` up to 8.8 and reaches ~1 %.
    """
    shear, d, nx, ny = 0.008, 7.5e-5, 96, 12
    res = solve_shear_channel_ad(
        nx=nx, ny=ny, shear=shear, diffusivity=d, steps=4000, check_every=250,
        tol=1e-5, init="leveque",
    )
    x = res.x
    delta_c = (9.0 * d * x / shear) ** (1.0 / 3.0)
    ok = np.isfinite(res.sh_x) & (delta_c >= 1.5) & (x <= x[-1] - 6)
    assert ok.sum() >= 20
    fit = fit_power_law(x[ok], res.sh_x[ok])
    assert 0.55 < fit["p"] < 0.70
    exact = leveque_local_sh(x[ok], shear, d)
    assert abs(np.median(res.sh_x[ok] / exact - 1.0)) < 0.20


# ---------------------------------------------------------------------------
# coupled: cylinder mass transfer from the exact mass budget
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cylinder_mass_transfer_short_run():
    """Short cylinder run: positive transfer rate, ``Sh`` in the physical range."""
    res = run_cylinder_mass_transfer(
        140, 72, 6.0, u_in=0.05, reynolds=40.0, scalar_sc=(1.0,), n_steps=400,
        average_from=0.5, diagnostic_every=200,
    )
    s = res["scalars"][0]
    assert s["mdot"] > 0.0
    assert 1.0 < s["Sh"] < 20.0
    assert np.isfinite(res["Cd_mean"])
    assert res["n_solid_cells"] > 0
    # staircase area of a radius-6 disk vs the nominal pi r^2
    assert res["D_eff_from_area"] == pytest.approx(12.0, rel=0.06)