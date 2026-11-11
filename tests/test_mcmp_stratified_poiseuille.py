"""Unit/physics tests for the MCMP stratified-Poiseuille reparation (GAP-LBM-MCMP-POISEUILLE).

Covers three things the A-arm verdict depends on:

1. the closed-form two-layer reference has the right limits
   (single-phase parabola, exact shear-stress continuity) -- ``analytic.py``;
2. ``force_scheme="guo"`` is (a) bit-identical to the legacy velocity-shift path
   when switched off, and (b) delivers exactly ``F`` of momentum per collision
   step, i.e. it is a consistent second-order forcing and not a re-parametrised
   shift;
3. the two-phase channel at the *segregating* coupling ``G_12 = -2.5`` stays
   stratified (``phi_contrast > 0.5``) and finite, while the ``G_12 = -1.0``
   parameter set of the pre-repair audit mixes -- the reason its per-component
   flow-rate comparison was not a two-phase measurement at all.

The quantitative acceptance records (errors vs the reference at three
resolutions) live in ``artifacts/mcmp_stratified_poiseuille/records/`` and are
re-derived by ``tests/benchmarks/test_mcmp_stratified_bench.py``.
"""

from __future__ import annotations

import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cases.mcmp_stratified_poiseuille.analytic import (  # noqa: E402
    CS2,
    layer_viscosities,
    layered_poiseuille,
)
from cases.mcmp_stratified_poiseuille.run_sweep import run_case  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import collide_sc_two_component  # noqa: E402

torch.set_grad_enabled(False)


@pytest.mark.unit
def test_analytic_single_phase_limit():
    ny, H, G, nu = 33, 32.0, 5e-5, CS2 * 0.5
    ana = layered_poiseuille(ny, H / 2, G, nu, nu)
    assert math.isclose(max(ana["profile"]), G * H * H / (8 * nu), rel_tol=1e-12)
    assert math.isclose(ana["Q_total"], G * H**3 / (12 * nu), rel_tol=1e-12)
    assert ana["stress_jump"] < 1e-12


@pytest.mark.unit
def test_analytic_stress_continuity_and_ode():
    ny, G = 25, 5e-5
    nu_lo, nu_up = CS2 * 0.5, CS2 * 0.25
    ana = layered_poiseuille(ny, 12.0, G, nu_lo, nu_up)
    assert ana["stress_jump"] < 1e-12
    prof = ana["profile"]
    for j in range(2, ny - 2):
        if abs(j - 12) > 1.5:
            nu_j = nu_lo if j < 12 else nu_up
            d2 = prof[j + 1] - 2 * prof[j] + prof[j - 1]
            assert abs(d2 - (-G / nu_j)) < 2e-5


@pytest.mark.unit
def test_layer_viscosity_rules():
    nu_lo, nu_up = layer_viscosities(1.0, 0.75, 0.7, 0.3, "component")
    assert math.isclose(nu_lo, CS2 * 0.5) and math.isclose(nu_up, CS2 * 0.25)
    nu_lo_m, nu_up_m = layer_viscosities(1.0, 0.75, 0.7, 0.3, "mixture")
    assert math.isclose(nu_lo_m, 0.7 * CS2 * 0.5 + 0.3 * CS2 * 0.25)
    assert math.isclose(nu_up_m, 0.3 * CS2 * 0.5 + 0.7 * CS2 * 0.25)
    assert nu_lo_m < nu_lo and nu_up_m > nu_up


@pytest.mark.unit
def test_guo_scheme_is_opt_in_bit_identical():
    ny, nx = 6, 3
    torch.manual_seed(0)
    rho = torch.rand(ny, nx) + 0.5
    z = torch.zeros(ny, nx)
    f1, f2 = equilibrium(rho, z, z), equilibrium(rho, z, z)
    legacy = collide_sc_two_component(f1, f2, G_12=-2.5, gx=1e-5)
    explicit = collide_sc_two_component(f1, f2, G_12=-2.5, gx=1e-5, force_scheme="shift")
    assert torch.equal(legacy[0], explicit[0]) and torch.equal(legacy[1], explicit[1])


@pytest.mark.unit
@pytest.mark.parametrize("scheme", ["shift", "guo"])
def test_forcing_delivers_exact_momentum(scheme):
    """Zero-velocity uniform state + uniform force F -> M' - M = F exactly."""
    ny, nx = 5, 4
    rho = torch.full((ny, nx), 0.5)
    z = torch.zeros((ny, nx))
    f1, f2 = equilibrium(rho, z, z), equilibrium(rho, z, z)
    fx_body = 1e-3
    out1, _ = collide_sc_two_component(
        f1, f2, G_12=0.0, tau1=1.0, tau2=1.0, gx=fx_body, force_scheme=scheme
    )
    from tensorlbm.d2q9 import C

    cx = C[:, 0].float().view(9, 1, 1)
    dp = ((out1 - f1) * cx).sum(dim=0)
    expected = rho * fx_body
    # float32 round-off of the (rho ~ 0.5, F ~ 1e-3) populations
    assert torch.allclose(dp, expected, rtol=1e-5, atol=1e-8), (
        scheme, float((dp - expected).abs().max()))


@pytest.mark.physics
def test_segregating_coupling_stays_stratified():
    rec = run_case(ny=25, steps=2500, G_12=-2.5, force_scheme="shift")
    assert rec["finite"]
    assert rec["phi_contrast"] > 0.5, rec["phi_contrast"]
    assert 9.0 < rec["interface_crossing"] < 15.0


@pytest.mark.physics
def test_weak_coupling_mixes():
    """G_12 = -1.0 (the pre-repair audit parameter) does not hold two layers."""
    rec = run_case(ny=25, steps=2500, G_12=-1.0, force_scheme="shift")
    assert rec["finite"]
    assert rec["phi_contrast"] < 0.5, rec["phi_contrast"]
    assert rec["stratified"] is False