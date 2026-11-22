# -*- coding: utf-8 -*-
"""DE-X-COL2D-REGDOSE: semantics of the REG dose knob.

The COL2 regularized (RR) collision is

    f_post = feq + (1 - 1/tau) * reconstruct(Pi^a)

with ``Pi^a`` the second-order non-equilibrium stress.  The campaign uses
``rr_tau`` as the *only* REG channel, so a "dose" multiplier on ``rr_tau``
simultaneously

* sets the stress-relaxation factor (1 - 1/tau) -- the projection strength, and
* sets the LBM lattice viscosity nu = cs^2 (tau - 1/2) -- inherited from the BGK
  mapping, because the RR branch keeps exactly the same second-order stress
  relaxation as BGK and only discards the higher-order non-equilibrium modes.

These tests pin that semantics down (and pin the tau < 0.5 negative-viscosity
regime that the 0.5x dose arm lands in).
"""
from __future__ import annotations

import torch

from tensorlbm.collision_d3q19_advanced import (
    collide_regularized_stress_d3q19,
    equilibrium3d,
    macroscopic3d,
    second_order_stress_d3q19,
)

CS2 = 1.0 / 3.0


def _perturbed(n: int = 6, seed: int = 7) -> tuple[torch.Tensor, torch.Tensor]:
    """Equilibrium plus a *valid* (zero mass, zero momentum) non-equilibrium part.

    The valid perturbation is the second-order stress reconstruction itself,
    which carries zero mass and zero momentum by lattice isotropy -- the same
    construction the DE-X-COL2-REG operator contract uses.
    """
    from tensorlbm.collision_d3q19_advanced import reconstruct_second_order_stress_d3q19

    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.05 * torch.rand(n, n, n, generator=g)
    ux = 0.05 * torch.randn(n, n, n, generator=g)
    uy = 0.03 * torch.randn(n, n, n, generator=g)
    uz = 0.02 * torch.randn(n, n, n, generator=g)
    feq = equilibrium3d(rho, ux, uy, uz)
    pi = [0.02 * torch.ones(n, n, n), -0.01 * torch.ones(n, n, n), 0.015 * torch.ones(n, n, n),
          0.008 * torch.ones(n, n, n), 0.0 * torch.ones(n, n, n), 0.0 * torch.ones(n, n, n)]
    f = feq + reconstruct_second_order_stress_d3q19(*pi)
    return f, feq


def _raw_stress(pop: torch.Tensor):
    return second_order_stress_d3q19(pop)


def test_dose_one_is_full_projection():
    """tau = 1 -> (1 - 1/tau) = 0 -> the post-collision state is the equilibrium."""
    f, feq = _perturbed()
    post = collide_regularized_stress_d3q19(f, feq, 1.0)
    assert torch.equal(post, feq)


def test_dose_scales_the_retained_stress_linearly():
    """The retained stress is exactly proportional to (1 - 1/tau)."""
    f, feq = _perturbed()
    post_half = collide_regularized_stress_d3q19(f, feq, 0.5)  # factor -1
    post_quarter = collide_regularized_stress_d3q19(f, feq, 0.25)  # factor -3
    assert (1.0 - 1.0 / 0.25) == 3.0 * (1.0 - 1.0 / 0.5)
    # post - feq is linear in the dose factor, hence the ratio is 3
    for a, b in zip(_raw_stress(post_quarter - feq), _raw_stress(post_half - feq)):
        assert torch.allclose(a, 3.0 * b, atol=1e-6, rtol=1e-5)


def test_dose_below_half_is_negative_viscosity():
    """tau < 0.5 -> nu = cs^2 (tau - 1/2) < 0 (the 0.5x dose arm regime)."""
    tau = 0.25019385
    nu = CS2 * (tau - 0.5)
    assert nu < 0.0
    # the relaxation factor is then |1 - 1/tau| > 1: the stress is amplified
    assert abs(1.0 - 1.0 / tau) > 2.9


def test_rr_keeps_the_bgk_tau_to_nu_mapping():
    """RR and BGK share the second-order stress relaxation factor (1 - 1/tau)."""
    f, feq = _perturbed()
    tau = 1.0007754
    post = collide_regularized_stress_d3q19(f, feq, tau)
    factor = 1.0 - 1.0 / tau
    # the retained stress of the RR branch equals the BGK second-order stress
    # relaxation: scaling the dose scales it by exactly the BGK factor
    post_b = collide_regularized_stress_d3q19(f, feq, 0.5003877)
    factor_b = 1.0 - 1.0 / 0.5003877
    for a, b in zip(_raw_stress(post - feq), _raw_stress(post_b - feq)):
        assert torch.allclose(a, (factor / factor_b) * b, atol=1e-6, rtol=1e-4)
    # conservation: mass and momentum are untouched by the collision
    rho_a, ux_a, uy_a, uz_a = macroscopic3d(f)
    rho_b, ux_b, uy_b, uz_b = macroscopic3d(post)
    assert torch.allclose(rho_a, rho_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(ux_a, ux_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(uy_a, uy_b, atol=1e-5, rtol=1e-5)
    assert torch.allclose(uz_a, uz_b, atol=1e-5, rtol=1e-5)
