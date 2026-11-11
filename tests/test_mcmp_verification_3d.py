"""Tests for 3D MCMP Young-Laplace calibration."""

from __future__ import annotations

import math

import torch

from tensorlbm.mcmp_verification_3d import (
    CS2,
    Guardrail,
    barycentric_velocity_3d,
    droplet_radius_3d,
    init_static_droplet_3d,
    measure_laplace_3d,
    sc_mcmp_pressure,
)


def test_sc_mcmp_pressure_3d_shape():
    """sc_mcmp_pressure works on 3D tensors."""
    rho1 = torch.full((10, 10, 10), 0.9)
    rho2 = torch.full((10, 10, 10), 0.1)
    p = sc_mcmp_pressure(rho1, rho2, G_12=-2.5)
    assert p.shape == (10, 10, 10)
    # With linear psi, p = CS2*(rho1+rho2) + G_12*CS2*rho1*rho2
    expected = CS2 * (0.9 + 0.1) + (-2.5) * CS2 * 0.9 * 0.1
    assert abs(float(p[0, 0, 0]) - expected) < 1e-6


def test_barycentric_velocity_3d():
    """Barycentric velocity returns 4 tensors of correct shape."""
    from tensorlbm.d3q19 import equilibrium3d

    rho = torch.full((8, 8, 8), 1.0)
    zero = torch.zeros((8, 8, 8))
    f = equilibrium3d(rho, zero, zero, zero)
    ux, uy, uz, rho_tot = barycentric_velocity_3d(f, f)
    assert ux.shape == (8, 8, 8)
    assert uy.shape == (8, 8, 8)
    assert uz.shape == (8, 8, 8)
    assert rho_tot.shape == (8, 8, 8)
    # At rest, velocity should be zero
    assert float(ux.max()) < 1e-10


def test_droplet_radius_3d_perfect_sphere():
    """droplet_radius_3d returns correct radius for a perfect tanh sphere."""
    n = 64
    radius = 15.0
    rho1_in, rho1_out = 0.9, 0.1
    rho1 = torch.zeros((n, n, n))
    centre = (n - 1) / 2.0
    for z in range(n):
        for y in range(n):
            for x in range(n):
                r = math.sqrt((x - centre) ** 2 + (y - centre) ** 2 + (z - centre) ** 2)
                s = 0.5 * (1.0 - math.tanh((r - radius) / 2.0))
                rho1[z, y, x] = rho1_out + s * (rho1_in - rho1_out)
    rho_av = 0.5 * (rho1_in + rho1_out)
    result = droplet_radius_3d(rho1, rho_av)
    assert result["radius"] is not None
    assert abs(result["radius"] - radius) < 1.0  # within 1 lattice unit
    assert result["crossings"] == 6


def test_init_static_droplet_3d_shape():
    """init_static_droplet_3d returns correct shapes."""
    f1, f2 = init_static_droplet_3d(
        n=32, radius=8.0,
        rho1_in=0.9, rho2_in=0.1,
        rho1_out=0.1, rho2_out=0.9,
    )
    assert f1.shape == (19, 32, 32, 32)
    assert f2.shape == (19, 32, 32, 32)


def test_measure_laplace_3d_guardrails_on_diverged():
    """measure_laplace_3d returns guardrails when density is non-finite."""
    run = {
        "G_12": -2.5, "n": 32, "radius_init": 8.0,
        "steps_run": 100, "converged": False, "u_max": float("nan"),
        "rho1": torch.full((32, 32, 32), float("nan")),
        "rho2": torch.full((32, 32, 32), 0.5),
    }
    case = measure_laplace_3d(run)
    assert case["sigma"] is None
    assert case["dp"] is None
    assert any(g["code"] == "non_finite_field" for g in case["guardrails"])
