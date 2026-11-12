"""Regression locks for the 2D/3D Laplace factor of the MCMP verification module.

``tensorlbm.mcmp_verification`` produces **2D** fields (``(n, n)``), so the
Young-Laplace law it must invert is the cylinder form ``Delta p = sigma / R``
(``sigma = Delta p R``).  Until 2026-11-12 ``measure_laplace`` returned
``Delta p R / 2`` - the *sphere* factor inherited from
``mcmp_verification_3d`` - i.e. every absolute surface tension was reported at
half its value (and every ``Ca_sp = mu |u| / sigma`` twice).  These tests pin
the factor so it cannot silently flip back.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.mcmp_verification import measure_laplace


def _synthetic_droplet(n: int = 48, radius: float = 12.0, width: float = 1.5):
    """A static-looking two-component droplet field (no solver in the loop)."""
    idx = torch.arange(n, dtype=torch.float32)
    yy, xx = torch.meshgrid(idx, idx, indexing="ij")
    r = torch.sqrt((xx - (n - 1) / 2.0) ** 2 + (yy - (n - 1) / 2.0) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / width))
    rho1 = 0.15 + 0.75 * s
    rho2 = 0.85 - 0.75 * s
    return {
        "G_12": -2.5,
        "n": n,
        "radius_init": radius,
        "steps_run": 1000,
        "converged": True,
        "u_max": 1e-3,
        "rho1": rho1,
        "rho2": rho2,
    }


def test_measure_laplace_2d_uses_cylinder_factor():
    run = _synthetic_droplet()
    case = measure_laplace(run)  # default dims=2
    assert case["laplace_dims"] == 2
    assert case["dp"] > 0 and case["radius"] > 0
    sigma_expected = case["dp"] * case["radius"]
    assert case["sigma"] == pytest.approx(sigma_expected, rel=1e-12)


def test_measure_laplace_3d_uses_sphere_factor():
    run = _synthetic_droplet()
    case2 = measure_laplace(run, dims=2)
    case3 = measure_laplace(run, dims=3)
    assert case3["sigma"] == pytest.approx(0.5 * case2["dp"] * case2["radius"], rel=1e-12)
    assert case3["sigma"] == pytest.approx(0.5 * case2["sigma"], rel=1e-12)


def test_measure_laplace_rejects_unknown_dimension():
    run = _synthetic_droplet()
    with pytest.raises(ValueError):
        measure_laplace(run, dims=4)


def test_young_laplace_calibration_fit_honours_dims():
    """The through-origin fit must invert the same factor as ``measure_laplace``."""
    from tensorlbm.mcmp_verification import _fit_laplace_level

    cases = []
    sigma_true = 0.2
    for R in (12.0, 16.0, 20.0):
        dp = sigma_true / R
        cases.append(
            {
                "G_12": -2.5,
                "radius_init": R,
                "radius": R,
                "dp": dp,
                "sigma": dp * R,
                "guardrails": [],
                "converged": True,
            }
        )
    level2 = _fit_laplace_level(-2.5, [dict(c) for c in cases], 0.05, dims=2)
    level3 = _fit_laplace_level(-2.5, [dict(c) for c in cases], 0.05, dims=3)
    assert level2["sigma_origin"] == pytest.approx(sigma_true, rel=1e-12)
    # Inverting the *sphere* factor on the same 2D data halves sigma: with
    # x = 2/R and y = sigma_true/R the through-origin slope is sigma_true/2.
    assert level3["sigma_origin"] == pytest.approx(0.5 * sigma_true, rel=1e-12)
    assert level2["max_rel_err"] == pytest.approx(0.0, abs=1e-12)
    assert level2["pass"] is True
    assert math.isfinite(level2["r2"])