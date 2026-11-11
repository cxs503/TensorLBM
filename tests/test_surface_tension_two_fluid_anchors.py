"""Acceptance tests for the P284W3-N02 (v2) surface-tension anchors.

The v2 delta over ``test_free_surface_surface_tension.py`` is the *matched*
Rayleigh reference (two-fluid, equal density) plus the curvature-anisotropy
measurement and its binomial pre-filter:

* :func:`~tensorlbm.free_surface_tension.csf.rayleigh_period_2fluid` reduces to
  the classic vacuum formula at ``rho_outer = 0`` and is ``sqrt(2)`` longer for
  ``rho_outer = rho_inner`` -- a regression guard: the previous attempt
  compared a uniform-density solver against the *vacuum* formula;
* ``curvature(..., smooth=1)`` recovers the analytic ``cos 2 theta`` amplitude
  of ``kappa`` on a mode-2 ellipse where the raw operator does not;
* ``smooth=0`` (the default) keeps the historical operator bit-for-bit, so the
  already-verified Young-Laplace / Marangoni anchors cannot regress;
* the damped-sinusoid phase estimator, the geometric interface marker, the
  viscous parameter and the ``nu -> 0`` extrapolation are checked on synthetic
  data and on one short end-to-end oscillation run.

Long sweeps (radii, widths, viscosities, figures) live in
``scripts/run_p284w3_n02_surface_tension_v2.py``; see
``docs/P284W3_N02_SURFACE_TENSION_V2.md`` for the report.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.free_surface_tension import (
    CS2,
    CapillarySolver,
    SolverConfig,
    curvature,
    csf_force,
    gradient,
    rayleigh_period,
    rayleigh_period_2fluid,
    smooth_field,
    tanh_droplet_phi,
    tanh_ellipse_phi,
)
from tensorlbm.free_surface_tension import capillary_oscillation as co
from tensorlbm.free_surface_tension import verification as V

torch.set_num_threads(2)


# --------------------------------------------------------------------------- #
# analytic anchor: two-fluid Rayleigh period                                  #
# --------------------------------------------------------------------------- #
def test_two_fluid_period_reduces_to_vacuum_limit():
    """``rho_outer -> 0`` must reproduce the classical (vacuum) Rayleigh period.

    ``rayleigh_period`` is the 3-D / vacuum (``n(n-1)(n+2)``) formula; the 2-D
    cylinder vacuum period uses ``n(n^2-1)`` and is checked explicitly.
    """
    for mode in (2, 3):
        vac = rayleigh_period(0.04, 12.0, rho=1.0, mode=mode)
        two = rayleigh_period_2fluid(0.04, 12.0, 1.0, 0.0, mode=mode, ndim=3)
        assert vac == pytest.approx(two, rel=1e-12)
    t_cyl = rayleigh_period_2fluid(0.04, 12.0, 1.0, 0.0, mode=2, ndim=2)
    assert t_cyl == pytest.approx(2.0 * math.pi * math.sqrt(12.0**3 / (6 * 0.04)), rel=1e-12)


def test_two_fluid_period_equal_density_is_sqrt2_longer():
    """Uniform density on both sides: ``T`` grows by exactly ``sqrt(2)``."""
    for ndim, mode in ((2, 2), (3, 2)):
        vac = rayleigh_period_2fluid(0.04, 12.0, 1.0, 0.0, mode=mode, ndim=ndim)
        eq = rayleigh_period_2fluid(0.04, 12.0, 1.0, 1.0, mode=mode, ndim=ndim)
        assert eq / vac == pytest.approx(math.sqrt(2.0), rel=1e-12)
    # explicit literature values: 2-D n=2 -> omega^2 = 3 sigma / a^3
    t2 = rayleigh_period_2fluid(0.04, 16.0, 1.0, 1.0, mode=2, ndim=2)
    assert t2 == pytest.approx(2.0 * math.pi * math.sqrt(16.0**3 / (3 * 0.04)), rel=1e-12)
    # 3-D n=2 -> omega^2 = 4 sigma / a^3
    t3 = rayleigh_period_2fluid(0.04, 16.0, 1.0, 1.0, mode=2, ndim=3)
    assert t3 == pytest.approx(2.0 * math.pi * math.sqrt(16.0**3 / (4 * 0.04)), rel=1e-12)


def test_two_fluid_period_rejects_bad_ndim():
    with pytest.raises(ValueError):
        rayleigh_period_2fluid(0.1, 5.0, ndim=4)


# --------------------------------------------------------------------------- #
# curvature pre-filter                                                        #
# --------------------------------------------------------------------------- #
def test_smooth_field_zero_passes_is_identity():
    torch.manual_seed(0)
    f = torch.rand((16, 16), dtype=torch.float64)
    assert torch.equal(smooth_field(f, 0), f)


def test_smooth_field_preserves_constant_and_is_binomial():
    const = torch.full((12, 10), 0.37, dtype=torch.float64)
    assert torch.allclose(smooth_field(const, 3), const)
    torch.manual_seed(1)
    f = torch.rand((12, 10), dtype=torch.float64)
    ref = f
    for ax in range(2):
        ref = 0.25 * torch.roll(ref, 1, ax) + 0.5 * ref + 0.25 * torch.roll(ref, -1, ax)
    assert torch.allclose(smooth_field(f, 1), ref)


def _kappa_band_mean(phi: torch.Tensor, kappa: torch.Tensor) -> float:
    delta = torch.sqrt(sum(g * g for g in gradient(phi)) + 1e-30)
    band = delta > 0.25 * float(delta.max())
    return float(kappa[band].mean())


def test_curvature_smooth_keeps_circle_accuracy():
    """A static circle must stay ``kappa = 1/R`` within 2 % after filtering."""
    R, w, L = 12.0, 3.0, 48
    phi = tanh_droplet_phi((L, L), (0.5 * L, 0.5 * L), R, width=w)
    for smooth in (0, 1, 2):
        k = _kappa_band_mean(phi, curvature(phi, smooth=smooth))
        assert k / (1.0 / R) - 1.0 == pytest.approx(0.0, abs=0.02), f"smooth={smooth}"


def test_curvature_mode2_amplitude_is_accurate():
    """The band-weighted ``cos 2 theta`` force projection, not pointwise kappa.

    Pointwise ``kappa`` sampling is *not* evidence at ``w = 2`` (sub-cell
    sampling choices alone move it by tens of percent); the physically relevant
    quantity is the ``delta_s``-weighted projection, i.e. the generalised
    stiffness conjugate to a ``cos 2 theta`` displacement.  The raw operator
    overshoots it by ~3 %, one binomial pass brings it below 2 %.
    """
    R, w, eps = 16.0, 2.0, 0.02
    L = int(8 * R)
    a, b = R * (1 + eps), R * (1 - eps)
    phi = tanh_ellipse_phi((L, L), (0.5 * L, 0.5 * L), (a, b), width=w)
    raw = co.curvature_mode2_amplitude(phi, (a, b), smooth=0)
    filt = co.curvature_mode2_amplitude(phi, (a, b), smooth=1)
    assert 0.95 <= raw["mode2_ratio"] <= 1.10, raw
    assert 0.95 <= filt["mode2_ratio"] <= 1.02, filt
    assert abs(filt["mode2_ratio"] - 1.0) < abs(raw["mode2_ratio"] - 1.0)
    # the cos(0) (mean curvature) component stays accurate either way
    assert abs(raw["mean_ratio"] - 1.0) < 0.02
    assert abs(filt["mean_ratio"] - 1.0) < 0.01


# --------------------------------------------------------------------------- #
# defaults unchanged (regression guard for the already-verified anchors)      #
# --------------------------------------------------------------------------- #
def test_csf_force_default_path_unchanged():
    L, R, w = 40, 10.0, 3.0
    phi = tanh_droplet_phi((L, L), (0.5 * L, 0.5 * L), R, width=w)
    sigma = 0.05
    f_default = csf_force(phi, sigma)
    f_explicit = csf_force(phi, sigma, smooth=0)
    f_manual = torch.stack([sigma * curvature(phi) * g for g in gradient(phi)], dim=0)
    assert torch.equal(f_default, f_explicit)
    assert torch.allclose(f_default, f_manual, atol=0, rtol=0)


def test_curvature_default_equals_raw_operator():
    L = 32
    phi = tanh_droplet_phi((L, L), (0.5 * L, 0.5 * L), 8.0, width=3.0)
    assert torch.equal(curvature(phi), curvature(phi, smooth=0))


def test_young_laplace_2d_passes_with_filter():
    """The static anchor must not regress when the curvature filter is on."""
    out = V.young_laplace_2d(10.0, sigma=0.05, tau=0.9, width=3.0, steps=(300, 1500), curvature_smooth=1)
    assert abs(out["metrics"]["dp_rel_err"]) < 0.03
    assert out["params"]["curvature_smooth"] == 1


def test_marangoni_force_tangential_unchanged():
    """One Marangoni sample: the tangential stress equals ``|d sigma/dx|``."""
    out = V.marangoni_case(1e-5, Ly=24, Lx=4, steps=1200)
    assert abs(out["metrics"]["u_rel_err"]) < 0.30  # short run: loose sanity band
    assert out["metrics"]["u_interface"] > 0.0


# --------------------------------------------------------------------------- #
# phase estimation and viscous extrapolation                                  #
# --------------------------------------------------------------------------- #
def test_damped_sinusoid_fit_recovers_period_and_damping():
    T, gamma, A = 400.0, 0.0015, 0.011
    t = np.arange(0.0, 4.0 * T, 2.0)
    y = A * np.exp(-gamma * t) * np.cos(2 * math.pi * t / T + 0.4) + 0.003
    fit = co.damped_sinusoid_fit(t, y, T)
    assert fit["period"] == pytest.approx(T, rel=0.005)
    assert fit["gamma"] == pytest.approx(gamma, rel=0.15)
    assert fit["r2"] > 0.99


def test_interface_anisotropy_matches_ellipse_marker():
    R, eps, L = 12.0, 0.05, 64
    cfg = SolverConfig(width=3.0, init_laplace=False)
    solver = CapillarySolver((L, L), cfg)
    solver.set_phi(tanh_ellipse_phi((L, L), (0.5 * L, 0.5 * L), (R * (1 + eps), R * (1 - eps)), width=3.0))
    a = co.interface_anisotropy(solver, R)
    assert a == pytest.approx(2.0 * eps, rel=0.02)


def test_viscous_parameter_scaling():
    p1 = co.viscous_parameter(0.55, 1000.0, 16.0)
    p2 = co.viscous_parameter(0.60, 1000.0, 16.0)
    assert p2 / p1 == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert p1 == pytest.approx(math.sqrt((CS2 * 0.05) / ((2 * math.pi / 1000.0) * 256.0)), rel=1e-12)


def test_inviscid_extrapolation_recovers_intercept_and_rejects_unstable():
    pts = []
    for tau, err in ((0.55, 0.12), (0.60, 0.16), (0.70, 0.28)):
        period = 1000.0
        x = co.viscous_parameter(tau, period, 16.0)
        pts.append(
            {
                "tau": tau,
                "radius": 16.0,
                "period_ref_two_fluid": period / (1.0 + 0.05 + 3.0 * x),
                "period": period,
                "stable": True,
            }
        )
    pts.append({"tau": 0.505, "radius": 16.0, "period_ref_two_fluid": 1000.0, "period": 1010.0,
                "stable": False})
    out = co.inviscid_extrapolation(pts)
    assert out["n_points"] == 3
    assert out["rejected"] == [0.505]
    assert out["intercept"] == pytest.approx(0.05, abs=0.02)
    assert out["r2"] > 0.95


# --------------------------------------------------------------------------- #
# end-to-end (short) oscillation run                                          #
# --------------------------------------------------------------------------- #
def test_oscillation_case_short_run_smoke():
    out = co.oscillation_case(
        radius=8.0,
        L=64,
        width=2.0,
        tau=0.60,
        sigma=0.04,
        eps=0.02,
        periods=1.4,
        curvature_smooth=1,
        dtype=torch.float32,
        sample=4,
        marker_every=10,
    )
    m = out["metrics"]
    assert m["stable"] is True
    assert m["period"] > 0.0 and m["r2_fit"] > 0.9
    # marker and second-moment estimates must agree (same physics, two probes)
    assert abs(m["period_marker"] / m["period"] - 1.0) < 0.10
    # loose band: a short viscous run is not the acceptance anchor
    assert abs(m["period_rel_err"]) < 0.35
    assert m["period_ref_two_fluid"] == pytest.approx(2 * math.pi * math.sqrt(8.0**3 / (3 * 0.04)), rel=1e-12)
    assert 1.0 < m["period_ref_two_fluid"] / m["period_ref_vacuum"] < 1.5