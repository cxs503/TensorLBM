"""Tests for the LIFE-compatible D2Q9 central-moment partial-relaxation collision.

The numeric fixtures in ``tests/fixtures/life_cm_partial_oracle.json`` were
produced by automatically extracting and evaluating the upstream
``CENTRAL_MOMENTS`` block of joconnor22/LIFE ``src/Grid.cpp`` (see
``docs/XVAL-LBM-CENTRAL-MOMENT-PARTIAL-RELAXATION.md``); they pin the port to
the upstream operator at machine precision without redistributing GPL source.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tensorlbm.central_moment_d2q9 import (
    M0,
    M0INV,
    central_moments,
    collide_central_moment_d2q9,
    equilibrium_hermite4,
    macroscopic_guo,
    moments_to_populations,
)

FIXTURES = Path(__file__).parent / "fixtures" / "life_cm_partial_oracle.json"


def _f_from_list(vals: list[float]) -> torch.Tensor:
    return torch.tensor(vals, dtype=torch.float64).view(9, 1, 1)


def test_moment_matrices_are_inverse() -> None:
    """``M0`` (rows = directions) and ``M0INV`` = (M0ᵀ)⁻¹ must invert."""
    assert np.allclose(M0.T @ M0INV, np.eye(9), atol=1e-14)
    assert np.allclose(M0INV @ M0.T, np.eye(9), atol=1e-14)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_central_moment_roundtrip(seed: int) -> None:
    """κ → f must be the exact inverse of f → κ for arbitrary states."""
    rng = np.random.default_rng(seed)
    f = _f_from_list(list(0.05 + 0.1 * rng.random(9)))
    rho, ux, uy = macroscopic_guo(f)
    back = moments_to_populations(central_moments(f, ux, uy), ux, uy)
    assert torch.max(torch.abs(back - f)).item() < 1e-14


@pytest.mark.parametrize("u", [(0.0, 0.0), (0.1, 0.05), (-0.2, 0.15), (0.3, 0.0)])
def test_hermite4_equilibrium_central_moments(u: tuple[float, float]) -> None:
    """The upstream equilibrium has κ = (ρ,0,0,2ρc_s²,0,0,0,0,ρc_s⁴) exactly."""
    rho = 1.3
    ux = torch.tensor(u[0], dtype=torch.float64)
    uy = torch.tensor(u[1], dtype=torch.float64)
    feq = equilibrium_hermite4(torch.tensor(rho, dtype=torch.float64), ux, uy)
    kappa = [float(k.reshape(-1)[0]) for k in central_moments(feq, ux, uy)]
    target = [rho, 0.0, 0.0, 2.0 * rho / 3.0, 0.0, 0.0, 0.0, 0.0, rho / 9.0]
    assert np.allclose(kappa, target, atol=1e-14)


def test_equilibrium_is_collision_invariant() -> None:
    """Colliding the H4 equilibrium must return it unchanged (any ω)."""
    u = (0.07, -0.04)
    feq = equilibrium_hermite4(
        torch.tensor(1.0, dtype=torch.float64),
        torch.tensor(u[0], dtype=torch.float64),
        torch.tensor(u[1], dtype=torch.float64),
    )
    out = collide_central_moment_d2q9(feq, 1.4)
    assert torch.max(torch.abs(out - feq)).item() < 1e-15


def test_conservation_without_force() -> None:
    """Mass and momentum are conserved by the collision."""
    rng = np.random.default_rng(11)
    f = torch.tensor(
        np.stack([0.05 + 0.05 * rng.random((4, 5)) for _ in range(9)]), dtype=torch.float64
    )
    out = collide_central_moment_d2q9(f, 1.3)
    cx = torch.tensor([[0], [1], [0], [-1], [0], [1], [-1], [-1], [1]], dtype=torch.float64).view(9, 1, 1)
    cy = torch.tensor([[0], [0], [1], [0], [-1], [1], [1], [-1], [-1]], dtype=torch.float64).view(9, 1, 1).view(9,1,1)
    assert torch.allclose(out.sum(0), f.sum(0), atol=1e-13)
    assert torch.allclose((out * cx).sum(0), (f * cx).sum(0), atol=1e-13)
    assert torch.allclose((out * cy).sum(0), (f * cy).sum(0), atol=1e-13)


def test_force_accelerates_by_F_over_rho() -> None:
    """Upstream injects ``F/2`` into the ``κx``/``κy`` moments and the Guo
    velocity carries the other ``F/2``, so one collision + force advances the
    macroscopic velocity by exactly ``F/ρ`` (one step of acceleration)."""
    rng = np.random.default_rng(12)
    f = torch.tensor(
        np.stack([0.05 + 0.05 * rng.random((3, 4)) for _ in range(9)]), dtype=torch.float64
    )
    fx = torch.full((3, 4), 1e-3, dtype=torch.float64)
    fy = torch.full((3, 4), -2e-3, dtype=torch.float64)
    force = (fx, fy)
    rho_in, ux_in, uy_in = macroscopic_guo(f, force)
    out = collide_central_moment_d2q9(f, 1.3, force=force)
    rho_out, ux_out, uy_out = macroscopic_guo(out, force)
    assert torch.allclose(rho_out, rho_in, atol=1e-13)
    assert torch.allclose(ux_out - ux_in, fx / rho_out, atol=1e-15)
    assert torch.allclose(uy_out - uy_in, fy / rho_out, atol=1e-15)


def test_relaxation_targets_for_shear_moments() -> None:
    """Only κd/κxy carry (1-ω)·κ^neq; the rest are reset to equilibrium values."""
    rng = np.random.default_rng(13)
    f = _f_from_list(list(0.05 + 0.08 * rng.random(9)))
    omega = 1.7
    out = collide_central_moment_d2q9(f, omega)
    rho, ux, uy = macroscopic_guo(out)
    k = central_moments(out, ux, uy)
    k_in = [float(v.reshape(-1)[0]) for v in central_moments(f, *macroscopic_guo(f)[1:])]
    assert abs(float(k[3].item()) - 2.0 * float(rho.item()) / 3.0) < 1e-14  # κtr*
    assert abs(float(k[7].item())) < 1e-14  # κxyy* (no force)
    assert abs(float(k[8].item()) - float(rho.item()) / 9.0) < 1e-14  # κxxyy*
    assert abs(float(k[4].item()) - (1.0 - omega) * k_in[4]) < 1e-13  # κd*
    assert abs(float(k[5].item()) - (1.0 - omega) * k_in[5]) < 1e-13  # κxy*


def test_upstream_life_fixtures() -> None:
    """Bit-level agreement with the upstream LIFE CENTRAL_MOMENTS operator."""
    data = json.loads(FIXTURES.read_text())
    worst = 0.0
    for case in data["cases"]:
        f = _f_from_list(case["f"])
        fx = torch.tensor([[case["fx"]]], dtype=torch.float64)
        fy = torch.tensor([[case["fy"]]], dtype=torch.float64)
        out = collide_central_moment_d2q9(f, case["omega"], force=(fx, fy))
        worst = max(worst, float(torch.max(torch.abs(out.reshape(-1) - torch.tensor(case["expected"], dtype=torch.float64)))))
    assert worst < 1e-14, f"max |tensorlbm - LIFE| = {worst:g}"
    assert data["max_abs_error_tensorlbm_vs_life"] < 1e-14


def test_viscosity_matches_taylor_green_decay() -> None:
    """One-dimensional shear decay γ = ν k² identifies ν = (1/ω - 1/2)/3."""
    n = 64
    k = 2.0 * math.pi / n
    omega = 1.6
    nu = (1.0 / omega - 0.5) / 3.0
    u0 = 0.01
    x = torch.arange(n, dtype=torch.float64) + 0.5
    uy_theory = u0 * torch.sin(k * x)
    ux = torch.zeros((n, n), dtype=torch.float64)
    uy = uy_theory.view(1, n).repeat(n, 1)
    f = equilibrium_hermite4(torch.ones((n, n), dtype=torch.float64), ux, uy)

    from tensorlbm.solver import stream

    ampl = []
    for step in range(201):
        if step in (1, 200):
            _, _, uy_now = macroscopic_guo(f)
            ampl.append(float((uy_now * uy_theory.view(1, n)).sum() / (uy_theory**2).sum()))
        f = stream(collide_central_moment_d2q9(f, omega))
    rate = -math.log(ampl[1] / ampl[0]) / 199.0
    assert abs(rate - nu * k * k) / (nu * k * k) < 0.02, (rate, nu * k * k)