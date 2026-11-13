"""XVAL anchor XVAL-LBM-NOT-APPLICABLE-PHASE-FIELD-NOT-PSEUDOPOT.

Covers the transplant in :mod:`tensorlbm.phasefield.ternary_ac_oss` against the
OSS source semantics (``xval/.../oss_src``): coefficient identities, the
compression-source closure that makes phase mass exact, the recovered interface
width, the tension closure and the density-independence of the phase equation.
Acceptance numbers live in ``artifacts/xval/m18180`` (driver
``scripts/xval_cudaphasefield_not_pseudopot.py``); these tests pin the invariants
so a regression breaks a test rather than a report.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.phasefield.ternary_ac_oss import (
    OSS_SHIPPED_SIGMA,
    TernaryAC2D,
    analytic_interface_width,
    flat_interface_state,
    interface_width_10_90,
    lambda_from_sigma_matrix,
    oss_laplacian_is_stencil_identity,
    oss_relaxation_factor,
    rho_of_phases,
    tension_from_lambda,
    tension_from_profile,
    ternary_chemical_potential,
    ternary_compression_source,
)


def _shipped_sigma() -> list[list[float]]:
    s = OSS_SHIPPED_SIGMA
    return [[0.0, s, s], [s, 0.0, s], [s, s, 0.0]]


def test_lambda_combination_rule_degenerate() -> None:
    """``lambda_k = s_jk + s_ki - s_ij`` and the harmonic coupling."""
    lam, lambdat = lambda_from_sigma_matrix(_shipped_sigma())
    assert lam == pytest.approx((0.1, 0.1, 0.1), abs=1e-15)
    assert lambdat == pytest.approx(0.1, abs=1e-15)


def test_lambda_combination_rule_nondegenerate() -> None:
    lam, lambdat = lambda_from_sigma_matrix(
        [[0.0, 0.20, 0.12], [0.20, 0.0, 0.15], [0.12, 0.15, 0.0]]
    )
    assert lam == pytest.approx((0.17, 0.23, 0.07), abs=1e-15)
    assert lambdat == pytest.approx(3.0 / (1 / 0.17 + 1 / 0.23 + 1 / 0.07), abs=1e-15)


def test_inadmissible_tension_triple_yields_negative_lambda() -> None:
    """The OSS rule is not a triangle-inequality projection: it must be visible."""
    lam, _ = lambda_from_sigma_matrix(
        [[0.0, 0.40, 0.10], [0.40, 0.0, 0.12], [0.10, 0.12, 0.0]]
    )
    assert lam[2] < 0.0


def test_over_relaxation_factor_and_relaxation_time() -> None:
    w_c = oss_relaxation_factor(0.01)
    assert w_c == pytest.approx(1.0 / 0.53, rel=0, abs=1e-15)
    assert abs(1.0 - w_c) < 1.0  # stable over-relaxation
    assert 1.0 / w_c == pytest.approx(0.53, abs=1e-15)


def test_analytic_interface_width_matches_oss_documented_value() -> None:
    assert analytic_interface_width(4.0) == pytest.approx(4.394, abs=5e-4)


def test_laplacian_is_the_oss_nine_point_stencil() -> None:
    """Repo operator vs the explicit ``(S_diag + 4 S_edge - 20 phi)/6`` form."""
    assert oss_laplacian_is_stencil_identity() < 1e-6


def test_chemical_potential_bulk_term_vanishes_at_all_three_pure_phases() -> None:
    lam, lambdat = lambda_from_sigma_matrix(_shipped_sigma())
    w = 4.0
    zero = torch.zeros(2, 2, dtype=torch.float64)
    for k in range(3):
        phi = torch.zeros(3, 2, 2, dtype=torch.float64)
        phi[k] = 1.0
        mu = ternary_chemical_potential(phi, torch.stack([zero] * 3), lam, lambdat, w)
        assert torch.allclose(mu, torch.zeros_like(mu), atol=1e-15)


def test_compression_source_sums_to_zero_over_phases() -> None:
    """``sum_k efh_k == 0`` -- the reason the phase mass is exact."""
    g = torch.Generator().manual_seed(7)
    phi = torch.rand(3, 6, 6, generator=g, dtype=torch.float64)
    phi = phi / phi.sum(dim=0, keepdim=True)
    n = torch.randn(3, 6, 6, generator=g, dtype=torch.float64)
    nx = n / n.norm(dim=0, keepdim=True)
    ny = torch.randn(3, 6, 6, generator=g, dtype=torch.float64)
    ny = ny / ny.norm(dim=0, keepdim=True)
    efh = ternary_compression_source(phi, nx, ny, 4.0)
    assert float(efh.sum(dim=0).abs().max()) < 1e-14


def test_phase_mass_is_conserved_over_a_short_run() -> None:
    sim = TernaryAC2D(shape=(4, 96), w=4.0)
    sim.initialize(flat_interface_state((4, 96), w=4.0, initial_width=8.0))
    m0 = sim.phase_mass().clone()
    sim.run(60)
    drift = float((sim.phase_mass() - m0).abs().max())
    assert drift < 1e-11, drift


def test_interface_relaxes_to_the_analytic_width() -> None:
    sim = TernaryAC2D(shape=(4, 128), w=4.0)
    sim.initialize(flat_interface_state((4, 128), w=4.0, initial_width=6.0))
    sim.run(400)
    width = interface_width_10_90(sim.fields()[0][0])
    # 400 steps on a 128-node row is a *partially* relaxed state (the converged
    # acceptance numbers live in artifacts/xval/m18180).  What is pinned here is
    # the direction and the gross magnitude: the interface must sharpen from the
    # 6-node initial state towards the 4.394 analytic width, not sit or blow up.
    assert width < 5.6, width
    ref_sampled = 4.535  # estimator applied to the analytic sampled profile
    assert abs(width - ref_sampled) / ref_sampled < 0.25, width


def test_tension_closure_is_lambda_over_two() -> None:
    x = torch.arange(512, dtype=torch.float64)
    profile = 0.5 * (1.0 + torch.tanh(2.0 * (x - 128.0) / 4.0))
    sigma = tension_from_profile(profile, 4.0, OSS_SHIPPED_SIGMA, half_width=12.0)
    assert sigma == pytest.approx(tension_from_lambda(OSS_SHIPPED_SIGMA), rel=0.15)
    # the shipped matrix value is a lambda-scale number, not the tension
    assert abs(sigma - OSS_SHIPPED_SIGMA) / OSS_SHIPPED_SIGMA > 0.15


def test_phase_equation_is_density_independent() -> None:
    """The ``(1000,500,100)`` triple is inertial only -- the phase field ignores it."""
    build = lambda density: TernaryAC2D(shape=(4, 64), w=4.0, density=density)  # noqa: E731
    a, b = build((1000.0, 500.0, 100.0)), build((1.0, 1.0, 1.0))
    init = flat_interface_state((4, 64), w=4.0, initial_width=6.0)
    a.initialize(init)
    b.initialize(init)
    a.run(40)
    b.run(40)
    assert torch.equal(a.fields(), b.fields())
    # ... while the inertial mass does depend on the triple
    assert rho_of_phases(b.fields(), b.density).max() <= 1.0 + 1e-12
    assert rho_of_phases(a.fields(), a.density).max() > 100.0


def test_ternary_closure_is_algebraic() -> None:
    sim = TernaryAC2D(shape=(4, 32), w=4.0)
    sim.initialize(flat_interface_state((4, 32), w=4.0))
    done = sim.fields().sum(dim=0)
    assert float((done - 1.0).abs().max()) < 1e-12