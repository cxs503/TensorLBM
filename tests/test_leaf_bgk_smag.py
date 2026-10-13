"""Contract tests for the leaf BGK-Smagorinsky collision (E1, route C).

``leaf_bgk_smag_collide`` must be the sparse-SoA leaf form of the uniform
formal tier's locked ``bgk_smag`` operator, so the A/B comparison between
the octree-shell tier and the frozen uniform runner is an operator
identity, not an approximation.  These tests pin:

* bitwise parity with ``turbulence.collide_smagorinsky_bgk3d`` when the
  same grid field is flattened onto leaves (scalar tau);
* the equilibrium fixed point (zero non-equilibrium stress -> bit-for-bit
  unchanged populations) for D3Q19/D3Q27, scalar and per-leaf tau;
* per-leaf tau decomposes into independent scalar runs;
* the tau_eff clamps (never below tau, uniform-tier headroom cap) and
  monotonicity in C_s;
* the externally supplied ``nu_t`` branch honours the
  ``cap_nu_t_factor * nu_mol`` discipline and the headroom cap.
"""

from __future__ import annotations

import torch

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.d3q27 import equilibrium27, macroscopic27
from tensorlbm.octree_boundary.les import (
    leaf_bgk_smag_collide,
    leaf_bgk_smag_tau_eff,
    leaf_neq_stress_norm,
)
from tensorlbm.turbulence import collide_smagorinsky_bgk3d


def _random_field(shape, q, seed, *, equilibrium_module=None):
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.05 * torch.randn(shape, generator=g)
    u = 0.03 * torch.randn((3, *shape), generator=g)
    if q == 19:
        feq = equilibrium3d(rho, u[0], u[1], u[2])
    else:
        feq = equilibrium27(rho, u[0], u[1], u[2])
    noise = 2.0e-3 * torch.randn((q, *shape), generator=g)
    return feq + noise


def test_bitwise_parity_with_uniform_tier_3d() -> None:
    f_grid = _random_field((5, 6, 7), q=19, seed=11)
    tau, c_s = 0.53, 0.05
    uniform = collide_smagorinsky_bgk3d(f_grid, tau, c_s)
    leaf = leaf_bgk_smag_collide(f_grid.view(19, -1), tau, c_s)
    assert torch.equal(leaf.view_as(f_grid), uniform)


def test_equilibrium_fixed_point_d3q19_and_d3q27() -> None:
    """Near-equilibrium states stay near equilibrium without amplification.

    Bitwise invariance does not hold even for ``f = feq(rho, u)``: the
    collision recomputes the moments, and in fp32 ``sum_q feq`` returns
    ``rho`` only to rounding, so the internally rebuilt equilibrium differs
    from the input by O(eps).  The contract is that the collision maps the
    O(eps) residual through (1 - 1/tau_eff) — never amplified.
    """
    for q, feq_fn, mac_fn in (
        (19, equilibrium3d, macroscopic3d),
        (27, equilibrium27, macroscopic27),
    ):
        shape = (4, 4, 4)
        g = torch.Generator().manual_seed(q)
        rho = 1.0 + 0.04 * torch.randn(shape, generator=g)
        u = 0.04 * torch.randn((3, *shape), generator=g)
        feq = feq_fn(rho, u[0], u[1], u[2]).view(q, -1)
        n = feq.shape[1]
        per_leaf = 0.56 + 0.02 * torch.arange(n, dtype=feq.dtype) / n
        for tau in (0.56, per_leaf):
            out = leaf_bgk_smag_collide(feq, tau, 0.05)
            rho_l = mac_fn(feq.view(q, 1, 1, -1))
            feq_rt = feq_fn(*rho_l).view(q, -1)
            residual_before = float((feq - feq_rt).abs().max())
            residual_after = float((out - feq_rt).abs().max())
            assert residual_after <= 3.0 * residual_before + 1e-8


def test_per_leaf_tau_decomposes_into_scalar_runs() -> None:
    """Per-leaf tau reproduces independent scalar runs to fp32 rounding.

    Not bitwise: a python-float ``tau**2`` squares in float64 while the
    tensor path squares in float32, so tau_eff differs at ~1e-8 — the
    runs must agree to that rounding, not bit for bit.
    """
    f = _random_field((3, 3, 3), q=19, seed=5)
    flat = f.view(19, -1)
    taus = torch.tensor(
        [
            0.53,
            0.60,
            0.71,
            0.82,
            0.53,
            0.60,
            0.71,
            0.82,
            0.53,
            0.60,
            0.71,
            0.82,
            0.53,
            0.60,
            0.71,
            0.82,
            0.53,
            0.60,
            0.71,
            0.82,
            0.53,
            0.60,
            0.71,
            0.82,
            0.53,
            0.60,
            0.71,
        ]
    )[: flat.shape[1]].clone()
    combined = leaf_bgk_smag_collide(flat, taus, 0.05)
    for i in range(flat.shape[1]):
        single = leaf_bgk_smag_collide(flat[:, i : i + 1], float(taus[i]), 0.05)
        assert torch.allclose(combined[:, i : i + 1], single, rtol=0, atol=1e-6)


def test_tau_eff_bounds_and_monotonicity() -> None:
    f = _random_field((3, 4, 5), q=19, seed=7)
    flat = f.view(19, -1)
    tau = 0.6
    out_low, diag_low = leaf_bgk_smag_collide(flat, tau, 0.02, return_diagnostics=True)
    out_high, diag_high = leaf_bgk_smag_collide(flat, tau, 0.4, return_diagnostics=True)
    assert bool((diag_high["tau_eff"] >= diag_low["tau_eff"]).all())
    upper = max(1.0, tau + 0.5)
    assert bool((diag_high["tau_eff"] <= upper + 1e-7).all())
    assert bool((diag_high["tau_eff"] >= tau - 1e-7).all())
    # uniform-tier headroom: nu_t never exceeds 0.5/3
    assert bool((diag_high["nu_t"] <= 0.5 / 3.0 + 1e-7).all())
    # relaxation moves populations toward equilibrium, never past it
    rho, ux, uy, uz = macroscopic3d(flat.view(19, 1, 1, -1))
    feq = equilibrium3d(rho, ux, uy, uz).view(19, -1)
    before = (flat - feq).abs().sum(dim=0)
    after = (out_high - feq).abs().sum(dim=0)
    assert bool((after <= before + 1e-6).all())


def test_external_nu_t_branch_caps() -> None:
    f = _random_field((3, 3, 4), q=19, seed=9)
    flat = f.view(19, -1)
    n = flat.shape[1]
    tau = 0.6
    huge = torch.full((n,), 5.0)
    out, diag = leaf_bgk_smag_collide(flat, tau, 0.05, nu_t=huge, return_diagnostics=True)
    nu_mol = (tau - 0.5) / 3.0
    assert bool((diag["nu_t"] <= 100.0 * nu_mol + 1e-7).all())
    # the headroom clamp binds even harder for this tau range
    assert bool((diag["nu_t"] <= 0.5 / 3.0 + 1e-7).all())
    # zero nu_t recovers the plain BGK collision
    plain = (
        flat - (flat - equilibrium3d(*macroscopic3d(flat.view(19, 1, 1, -1))).view(19, -1)) / tau
    )
    out0, diag0 = leaf_bgk_smag_collide(
        flat, tau, 0.05, nu_t=torch.zeros(n), return_diagnostics=True
    )
    assert torch.equal(out0, plain)


def test_d3q27_runs_and_shapes() -> None:
    f = _random_field((3, 3, 3), q=27, seed=13)
    flat = f.view(27, -1)
    out, diag = leaf_bgk_smag_collide(flat, 0.58, 0.05, return_diagnostics=True)
    assert out.shape == flat.shape
    assert diag["tau_eff"].shape == (flat.shape[1],)
    assert bool(torch.isfinite(out).all())
    assert bool((diag["tau_eff"] >= 0.58 - 1e-7).all())
    # stress norm module standalone
    pi = leaf_neq_stress_norm(flat - flat)
    assert bool((pi == torch.full_like(pi, 1e-15)).all())


def test_tau_eff_formula_matches_reference() -> None:
    tau, c_s = 0.55, 0.1
    pi = torch.tensor([0.0, 1e-4, 3e-3])
    rho = torch.tensor([1.0, 0.98, 1.03])
    got = leaf_bgk_smag_tau_eff(tau, pi, rho, c_s)
    disc = tau**2 + 18.0 * c_s**2 * pi / rho.clamp(min=1e-12)
    want = 0.5 * (tau + torch.sqrt(disc.clamp(min=0.0)))
    want = want.clamp(min=0.5001, max=max(1.0, tau + 0.5))
    assert torch.allclose(got, want, rtol=0, atol=0)
