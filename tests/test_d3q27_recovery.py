"""Acceptance tests for the D3Q27 recovery module (lattice audit + anchors).

Run:  python -m pytest tests/test_d3q27_recovery.py -q

The tests are deliberately small (H = 8 channel, D = 4 sphere) so the whole
file finishes in well under a minute; the production-resolution ladders live in
``benchmarks/frontier/p156w2_n03_d3q27_recovery/run.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.d3q27_recovery import (
    D3Q27Stepper,
    direction_liveness_audit,
    equilibrium27,
    equilibrium_third_moment_residual,
    hasimoto_correction,
    moment_audit,
    momentum_exchange_drag,
    run_channel_poiseuille,
    run_long_stability,
    sphere_mask,
)

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# 1. Lattice algebra / dead-enumeration audit
# ---------------------------------------------------------------------------


def test_moment_audit_is_exact_to_roundoff():
    a = moment_audit()
    assert a.n_directions == 27
    assert abs(a.weight_sum - 1.0) < 1e-12
    assert a.cs2_error < 1e-12
    assert a.rank1_residual < 1e-12
    assert a.rank2_residual < 1e-12
    assert a.rank3_residual < 1e-12
    assert a.rank4_isotropic_residual < 1e-12
    # D3Q27 is NOT isotropic at rank 6 -- the audit must say so, otherwise the
    # "isotropy" claim is unfalsifiable.
    assert a.rank6_isotropic_residual > 1e-2
    assert a.is_isotropic_rank4 and not a.is_isotropic_rank6


def test_third_moment_identity():
    """Equilibrium third moment == rho c_s^2 (delta u + delta u + delta u).

    The residual against the *truncated* analytic form must be roundoff; the
    residual against the full Maxwellian is exactly the rho u u u term, i.e.
    O(Ma^3), which the rank-6-anisotropic stencil cannot carry.
    """
    r = equilibrium_third_moment_residual(rho=1.3, u=(0.07, -0.05, 0.03))
    assert r["residual_vs_truncated_rel"] < 1e-14
    assert abs(r["residual_vs_full_maxwellian_abs"] - r["uuu_deficit_abs"]) < 1e-15


def test_all_27_directions_are_live():
    live = direction_liveness_audit()
    assert live["all_live"], f"dead directions: {live['failures']}"
    # the shift table must match the descriptor (cx, cy, cz) exactly
    assert live["shifts"][0] == (0, 0, 0)
    assert live["shifts"][19] == (1, 1, 1)


# ---------------------------------------------------------------------------
# 2. Three-dimensional Poiseuille anchor
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def channel():
    return run_channel_poiseuille(
        H=8, tau=0.8, force_density=1e-5, n_steps=3000, check_every=100
    )


def test_poiseuille_matches_analytic_profile(channel):
    assert channel.finite
    assert channel.converged
    # second-order convergence: 1.1 % at H = 8 (measured), so 3 % is the guard
    assert channel.rel_l2_error < 3e-2, channel.rel_l2_error
    assert channel.rel_linf_error < 3e-2, channel.rel_linf_error


def test_wall_is_halfway_not_at_solid_centres(channel):
    """Regression guard for the bounce-back wall position.

    A "reflect inside the solid cell" implementation reflects at the solid
    node centre, which widens the effective gap to H + 1 and over-predicts the
    peak velocity by (H+1)^2/H^2 = +26 % at H = 8.  The pull-form half-way
    bounce-back used here must stay within 2 %.
    """
    ratio = channel.u_max_lbm / channel.u_max_analytic
    assert abs(ratio - 1.0) < 0.02, ratio


def test_momentum_exchange_equals_applied_body_force(channel):
    """The force estimator is validated on plane walls, where the steady drag
    must equal the total applied body force exactly."""
    assert channel.force_balance_error < 1e-3, channel.force_balance_error
    # mass is conserved by construction
    mass = torch.tensor(channel.mass_history)
    assert float((mass - mass[0]).abs().max()) < 1e-8 * float(mass[0])


def test_poiseuille_velocity_divergence_is_zero(channel):
    assert channel.max_divergence < 1e-10, channel.max_divergence


# ---------------------------------------------------------------------------
# 3. Sphere drag machinery
# ---------------------------------------------------------------------------


def test_sphere_mask_volume_and_containment():
    n, d = 32, 8
    mask = sphere_mask(n, d)
    n_solid = int(mask.sum())
    exact = 4.0 / 3.0 * math.pi * (d / 2.0) ** 3
    assert abs(n_solid - exact) / exact < 0.1, (n_solid, exact)
    # every solid cell centre must be inside the nominal radius
    idx = torch.nonzero(mask, as_tuple=False).to(torch.float64) + 0.5
    r = (idx - 0.5 * n).norm(dim=1)
    assert float(r.max()) <= d / 2 + 1e-12
    # and the mask must not be empty in any coordinate slab through the centre
    assert mask[n // 2].any() and mask[:, n // 2].any() and mask[:, :, n // 2].any()


def test_hasimoto_correction_limits():
    assert abs(hasimoto_correction(1e-9) - 1.0) < 1e-6
    assert hasimoto_correction(0.1) > hasimoto_correction(0.05) > 1.0


def test_momentum_exchange_estimators_agree():
    """Standalone estimator == stepper link set (identical link definition)."""
    torch.manual_seed(0)
    n, d = 16, 6
    solid = sphere_mask(n, d)
    tau = 0.8
    stepper = D3Q27Stepper(solid, tau=tau, force=(0.0, 0.0, 0.0))
    rho = torch.ones((n, n, n), dtype=torch.float64)
    ux = 0.01 * torch.randn((n, n, n), dtype=torch.float64)
    f = equilibrium27(rho, ux, 0.003 * torch.randn_like(ux), 0.002 * torch.randn_like(ux))
    a = momentum_exchange_drag(f, solid)
    b = stepper.momentum_exchange_force(f)
    assert torch.allclose(a, b, atol=1e-14, rtol=0), (a, b)


def test_sphere_drag_direction_and_sign():
    """Drag must point along +x for a +x body force (no spurious lift)."""
    n, d = 16, 6
    solid = sphere_mask(n, d)
    stepper = D3Q27Stepper(solid, tau=0.8, force=(1e-4, 0.0, 0.0))
    ones = torch.ones((n, n, n), dtype=torch.float64)
    f = equilibrium27(ones, *[torch.zeros_like(ones)] * 3)
    for _ in range(30):
        box = []
        f = stepper.step(f, hook=lambda fp: box.append(stepper.momentum_exchange_force(fp)))
    fx, fy, fz = (float(v) for v in box[0])
    assert fx > 0.0
    assert abs(fy) < 1e-12 and abs(fz) < 1e-12


# ---------------------------------------------------------------------------
# 4. Stability gate (short version; the 1e5-step run is in the benchmark)
# ---------------------------------------------------------------------------


def test_short_stability_float32():
    res = run_long_stability(
        H=8, n_steps=4000, force_density=2e-5, dtype=torch.float32, probe_every=500
    )
    assert res.finite
    assert not res.diverged
    assert 0.9 < res.rho_min and res.rho_max < 1.1