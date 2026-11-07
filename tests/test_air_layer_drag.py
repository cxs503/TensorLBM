"""Tests for the air-layer / micro-bubble drag-reduction module (campaign IN30-N03).

Fast checks only (the full DP closure lives in
``scripts/air_layer_drag_campaign.py``); every test asserts one of the campaign
gates on a small grid so CI stays cheap.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.air_layer_drag.campaign import (
    ChannelConfig,
    _gas_layer_phi,
    positive_boundary,
    single_phase_flux,
    signed_distance_patch,
)
from tensorlbm.air_layer_drag.core import (
    CS2,
    PhaseFieldChannel,
    PhaseFieldParams,
    SinglePhaseChannel,
    circular_bubble,
    laplace_pressure,
)
from tensorlbm.air_layer_drag.economics import (
    CompressionCase,
    compressor_power,
    net_power_gain,
)
from tensorlbm.air_layer_drag.ledger import GasLedger
from tensorlbm.air_layer_drag.validations import (
    laplace_single,
    mass_conservation_gate,
    single_phase_cf,
)


# ---------------------------------------------------------------------------
# lattice / equilibrium bookkeeping
# ---------------------------------------------------------------------------
def test_equilibrium_moments():
    """sum(f_eq) = p, sum(c f_eq) = rho u, sum(w) = 1."""
    from tensorlbm.air_layer_drag.core import W9

    assert math.isclose(float(np.sum(W9)), 1.0, rel_tol=1e-14)
    ny = nx = 6
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.5, 1.5, (ny, nx))
    ux = rng.uniform(-0.05, 0.05, (ny, nx))
    uy = rng.uniform(-0.05, 0.05, (ny, nx))
    pbar = rng.uniform(0.2, 0.5, (ny, nx))
    feq = PhaseFieldChannel._equilibrium_f(rho, ux, uy, pbar)
    assert np.allclose(feq.sum(axis=0), pbar, atol=1e-13)
    from tensorlbm.air_layer_drag.core import CX, CY

    assert np.allclose(np.tensordot(CX, feq, axes=(0, 0)), rho * ux, atol=1e-13)
    assert np.allclose(np.tensordot(CY, feq, axes=(0, 0)), rho * uy, atol=1e-13)


def test_single_phase_gate_cf():
    """Gate (a): Cf from the momentum-exchange wall shear matches 12/Re within 3%."""
    res = single_phase_cf(nx=8, ny=32, nu=0.05, body_force_x=1e-5, steps=30000)
    assert res["cf_rel_error"] < 0.03
    assert res["momentum_balance_rel_error"] < 0.01
    assert res["profile_rel_l2_error"] < 0.01


def test_single_phase_shear_scales_with_force():
    """Wall shear must be linear in the driving force (Stokes regime)."""
    a = single_phase_cf(nx=8, ny=24, nu=0.05, body_force_x=1e-5, steps=20000)
    b = single_phase_cf(nx=8, ny=24, nu=0.05, body_force_x=2e-5, steps=20000)
    ratio = b["wall_shear_mem"] / a["wall_shear_mem"]
    assert 1.95 < ratio < 2.05


# ---------------------------------------------------------------------------
# phase field
# ---------------------------------------------------------------------------
def test_params_derived_quantities():
    p = PhaseFieldParams()
    assert p.delta == pytest.approx(0.25 * p.interface_width)
    assert p.kappa == pytest.approx(2 * p.beta * p.delta ** 2)
    assert p.sigma == pytest.approx((8.0 / 3.0) * p.beta * p.delta)
    assert p.density_ratio >= 100.0
    assert p.interface_width >= 3.0


def test_laplace_law_scaling():
    """Gate (b): dP * R is radius independent in the well-resolved range.

    R >= 5 W is required: at R = 16 with W = 6 the curvature resolution bias
    reaches ~30% (documented in docs/IN30_N03_AIR_LAYER_DR.md), so the scaling
    check is made where the diffuse interface is actually resolved.
    """
    p = PhaseFieldParams(interface_width=6)
    r1 = laplace_single(p, 22.0, box=76, steps=2000)
    r2 = laplace_single(p, 28.0, box=76, steps=2000)
    s1, s2 = r1["sigma_effective"], r2["sigma_effective"]
    assert s1 > 0.0 and s2 > 0.0
    assert abs(s1 - s2) / s1 < 0.08
    # the discrete surface tension stays within 15% of the continuous formula
    assert abs(s1 / p.sigma - 1.0) < 0.15


def test_mass_conservation_closed_domain():
    """Gate (c): closed periodic domain keeps sum(phi) and the mass to round-off."""
    res = mass_conservation_gate(PhaseFieldParams(interface_width=6), box=48,
                                steps=1200, n_samples=4, with_injection=False)
    assert res["phi_rel_drift"] < 1e-10
    assert res["mass_rel_drift"] < 1e-10


def test_injected_gas_ledger_closes():
    """Gate (c): the volumetric gas source is exactly accounted."""
    res = mass_conservation_gate(PhaseFieldParams(interface_width=6), box=48,
                                steps=1200, n_samples=4, with_injection=True)
    led = res["ledger"]
    assert led["injected_volume"] > 0.0
    assert led["ledger_rel_residual"] < 1e-10


def test_gas_injection_changes_phase_field():
    """Injection must actually add gas and keep the ledger consistent."""
    p = PhaseFieldParams(interface_width=6)
    phi0 = np.ones((48, 48))
    ch = PhaseFieldChannel(48, 48, p, phi0, walls=False)
    mask = np.zeros((48, 48), dtype=bool)
    mask[1:4, :] = True
    v0 = ch.total_gas_volume()
    ch.step(200, inject=(mask, 1e-2))
    assert ch.total_gas_volume() > v0
    gain = ch.total_gas_volume() - v0
    assert abs(gain - ch.injected_volume) / ch.injected_volume < 1e-6


def test_patch_geometry_signed_distance():
    """Signed distance: positive inside the wall patch, negative outside, periodic."""
    d = signed_distance_patch(16, 32, x0=0.0, x1=8.0, y0=-1.0, y1=3.5)
    assert d[1, 4] > 0.0            # inside the patch
    assert d[10, 4] < 0.0           # above the layer
    assert d[1, 20] < 0.0           # downstream of the patch end
    assert abs(d[1, 30] - d[1, 2]) < 6.0   # roughly periodic in x


def test_gas_layer_phi_profile():
    cfg = ChannelConfig(nx=32, ny=24)
    phi = _gas_layer_phi(cfg, coverage=0.5, thickness=6)
    assert phi.min() < -0.98 and phi.max() > 0.98
    # a layer thinner than the interface width cannot reach phi = -1
    thin = _gas_layer_phi(cfg, coverage=0.5, thickness=3)
    assert -1.0 < thin.min() < -0.5
    assert np.all(np.isfinite(phi))
    assert phi[1, :16].mean() < -0.5      # gas at the wall over the covered span
    assert phi[10, :].mean() > 0.5        # liquid far from the wall


# ---------------------------------------------------------------------------
# economics
# ---------------------------------------------------------------------------
def test_compressor_power_monotone_in_efficiency():
    case = CompressionCase(eta_c=0.6)
    p1 = compressor_power(0.1, case)
    case_lo = CompressionCase(eta_c=0.5)
    case_hi = CompressionCase(eta_c=0.7)
    assert compressor_power(0.1, case_lo) > p1 > compressor_power(0.1, case_hi)


def test_net_power_sign_and_domain():
    case = CompressionCase(eta_c=0.6)
    good = net_power_gain(0.5, 1e-3, case)
    bad = net_power_gain(0.01, 0.5, case)
    assert good["dP_net_W"] > 0.0 and good["net_positive"]
    assert bad["dP_net_W"] < 0.0 and not bad["net_positive"]


def test_positive_domain_shrinks_with_lower_efficiency():
    """Gate (f): lower compressor efficiency must shrink the positive-gain domain."""
    rows = {}
    for eta in (0.5, 0.6, 0.7):
        case = CompressionCase(eta_c=eta)
        rows[eta] = positive_boundary([
            net_power_gain(0.4, cq, case) for cq in np.linspace(1e-4, 3e-2, 60)])
    widths = {e: (v["cq_max_positive"] or 0.0) for e, v in rows.items()}
    assert widths[0.5] <= widths[0.6] <= widths[0.7]
    assert widths[0.7] > 0.0


def test_ledger_drift_helpers():
    assert GasLedger.closed_mass_drift([2.0, 2.0, 2.0]) == 0.0
    assert GasLedger.phase_inventory_drift([100.0, 100.0 + 1e-12, 100.0]) < 1e-13
    with pytest.raises(ValueError):
        GasLedger.closed_mass_drift([1.0])


# ---------------------------------------------------------------------------
# channel regression (short runs, no instability)
# ---------------------------------------------------------------------------
def test_air_layer_case_reduces_drag_and_conserves_phase():
    cfg = ChannelConfig(nx=48, ny=32, spinup_steps=1200, measure_steps=400,
                        measure_window=2)
    base = single_phase_flux(cfg)["mass_flux"]
    from tensorlbm.air_layer_drag.campaign import air_layer_case

    res = air_layer_case(cfg, coverage=1.0, thickness=5, baseline_flux=base)
    assert 0.0 < res["drag_reduction"] < 0.9
    assert res["phi_drift_rel"] < 1e-9
    assert abs(res["ledger"]["residual"]) < 1e-8
    assert np.isfinite(res["mass_flux"])


def test_partial_coverage_is_stable():
    cfg = ChannelConfig(nx=48, ny=32, spinup_steps=1000, measure_steps=300,
                        measure_window=2)
    base = single_phase_flux(cfg)["mass_flux"]
    from tensorlbm.air_layer_drag.campaign import air_layer_case

    res = air_layer_case(cfg, coverage=0.5, thickness=5, baseline_flux=base)
    assert np.isfinite(res["drag_reduction"])
    assert res["drag_reduction"] > 0.0
    assert res["phi_drift_rel"] < 1e-9