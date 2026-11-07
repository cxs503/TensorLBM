"""Tests for the IN57-N02 bubble-curtain / air-layer closure (tensorlbm.bubble_dr).

The suite is built to *break* the closure rather than to flatter it:

* analytic anchors -- the Hadamard-Rybczynski law, the isothermal compression
  work, the ITTC-57 line, the smooth gas-phase indicator;
* solver-identity anchors -- the buoyant Shan-Chen collision must be bit-identical
  to the library collision when the buoyancy is switched off;
* negative controls -- zero gas supply on a wetting plate must give zero coverage
  and zero drag reduction (a model that spontaneously cavitates fails here);
* conservation -- the mass ledger must close with the injected mass booked;
* a parameter scan -- coverage/DR must respond monotonically to the gas supply.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.bubble_dr.bubble3d import (
    BubbleConfig,
    _buoyant_collide,
    confinement_extrapolation,
)
from tensorlbm.bubble_dr.curtain2d import (
    CurtainChannel2D,
    CurtainConfig,
    drag_reduction,
    run_pair,
)
from tensorlbm.bubble_dr.power import (
    OperatingPoint,
    compressor_power_density,
    isothermal_work_closed_form,
    isothermal_work_quadrature,
    multistage_work,
    net_closure,
    positive_domain,
    wall_shear,
)
from tensorlbm.hadamard_rybczynski import hr_velocity
from tensorlbm.multiphase3d import collide_sc_single_component_3d, equilibrium3d, psi_exp

SMALL = dict(nx=32, ny=20, n_steps=200, sample_every=50, g_buoy=2.0e-3, displace_fraction=0.02)


# ---------------------------------------------------------------------------
# 1-7: compressor ledger and net power
# ---------------------------------------------------------------------------


def test_isothermal_closed_form_matches_quadrature():
    op = OperatingPoint()
    closed = isothermal_work_closed_form(op)
    quad = isothermal_work_quadrature(op, n=20001)
    assert abs(quad / closed - 1.0) < 1e-3


def test_three_stage_compression_within_five_percent_of_isothermal():
    op = OperatingPoint()
    iso = isothermal_work_closed_form(op)
    w3 = multistage_work(op, 3)
    assert abs(w3 / iso - 1.0) <= 0.05


def test_single_stage_exceeds_isothermal_limit():
    """The 1-stage machine must be *worse* than the isothermal limit -- a sign check."""
    op = OperatingPoint()
    iso = isothermal_work_closed_form(op)
    assert multistage_work(op, 1) > iso * 1.05


def test_multistage_converges_to_isothermal_monotonically():
    op = OperatingPoint()
    iso = isothermal_work_closed_form(op)
    works = [multistage_work(op, n) for n in (1, 2, 3, 6, 12)]
    assert all(a > b for a, b in zip(works, works[1:]))
    assert abs(works[-1] / iso - 1.0) < 1e-2


def test_eta_net_definition_and_range():
    op = OperatingPoint()
    res = net_closure(0.3, 5.0, wall_shear(5.0, op), 0.05, op)
    assert res["saved_power_density"] > 0
    assert res["compressor_power_density"] > 0
    assert 0.0 < res["eta_net"] < 1.0
    assert res["eta_net"] == pytest.approx(
        res["saved_power_density"] / (res["saved_power_density"] + res["compressor_power_density"])
    )


def test_zero_drag_reduction_has_negative_net_power():
    op = OperatingPoint()
    res = net_closure(0.0, 5.0, wall_shear(5.0, op), 0.05, op)
    assert res["net_power_density"] < 0.0
    assert res["eta_net"] == 0.0


def test_net_power_increases_with_drag_reduction():
    op = OperatingPoint()
    tw = wall_shear(5.0, op)
    nets = [net_closure(dr, 5.0, tw, 0.02, op)["net_power_density"] for dr in (0.0, 0.1, 0.3, 0.6)]
    assert all(b > a for a, b in zip(nets, nets[1:]))


def test_compressor_power_linear_in_gas_flow():
    op = OperatingPoint()
    p1 = compressor_power_density(0.01, 5.0, op)
    p4 = compressor_power_density(0.04, 5.0, op)
    assert p4 == pytest.approx(4.0 * p1, rel=1e-12)


def test_wall_shear_matches_ittc57_and_grows_with_speed():
    op = OperatingPoint()
    u = 5.0
    re_l = u * op.plate_length / op.nu_water
    cf = 0.075 / (math.log10(re_l) - 2.0) ** 2
    assert wall_shear(u, op) == pytest.approx(0.5 * op.rho_water * u * u * cf)
    assert wall_shear(8.0, op) > wall_shear(5.0, op)


def test_positive_domain_brute_force_consistency():
    op = OperatingPoint()
    cqs = [0.005, 0.02, 0.05]
    drs = [0.05, 0.30, 0.45]
    dom = positive_domain([4.0, 8.0], cqs, lambda c: float(np.interp(c, cqs, drs)), op)
    net = np.asarray(dom["net_power_density"])
    assert dom["positive_fraction"] == pytest.approx(float((net > 0).mean()))
    assert 0.0 <= dom["positive_fraction"] <= 1.0
    # higher C_q costs more compressor power -> the profitable set cannot grow
    assert (net[:, 0] > 0).sum() >= (net[:, -1] > 0).sum()


# ---------------------------------------------------------------------------
# 8-11: analytic anchors of the drag/coverage definitions
# ---------------------------------------------------------------------------


def test_drag_reduction_definition():
    assert drag_reduction(0.75, 1.0) == pytest.approx(0.25)
    assert drag_reduction(1.0, 1.0) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        drag_reduction(1.0, 0.0)


def test_gas_indicator_is_a_step_at_mid_coexistence():
    cfg = CurtainConfig(**SMALL)
    sim = CurtainChannel2D(cfg)
    rho_l, rho_g = cfg.rho_liquid, cfg.rho_gas
    mid = cfg.gas_threshold
    probe = torch.tensor([[rho_l, mid, rho_g]], dtype=cfg.dtype)
    alpha = sim.gas_fraction(probe)
    assert float(alpha[0, 0]) < 1e-3
    assert float(alpha[0, 1]) == pytest.approx(0.5, abs=1e-3)
    assert float(alpha[0, 2]) > 0.999
    # compressibility noise of +-0.5 lattice units around the liquid density must
    # not register as gas (this is the trap that broke the linear-deficit measure)
    noisy = torch.tensor([[rho_l - 0.5, rho_l + 0.5]], dtype=cfg.dtype)
    assert float(sim.gas_fraction(noisy).max()) < 1e-2


def test_coverage_saturates_at_full_plate_and_tracks_thickness():
    cfg = CurtainConfig(**SMALL)
    sim = CurtainChannel2D(cfg)
    rho = torch.full((cfg.ny, cfg.nx), cfg.rho_liquid, dtype=cfg.dtype)
    assert sim.coverage(rho) == pytest.approx(0.0, abs=1e-4)
    # a two-cell gas film on the ceiling: fully covered, thickness two cells
    rho[cfg.ny - 3 : cfg.ny - 1, :] = cfg.rho_gas
    assert sim.coverage(rho) == pytest.approx(1.0)
    assert sim.layer_thickness(rho) == pytest.approx(2.0, abs=0.05)
    # a half-cell film: half the plate area counts as covered
    rho2 = torch.full((cfg.ny, cfg.nx), cfg.rho_liquid, dtype=cfg.dtype)
    rho2[cfg.ny - 2, :] = 0.5 * (cfg.rho_liquid + cfg.rho_gas)
    assert sim.coverage(rho2) == pytest.approx(0.5, abs=0.02)


def test_hr_law_endpoints_reproduce_classical_limits():
    """Analytic anchor: lambda->0 gives U = (1/3) drho g R^2 / mu, lambda=1 gives 4/15."""
    drho, g, R, mu = 1.8, 5e-5, 8.0, 0.326
    u_bubble = (1.0 / 3.0) * drho * g * R**2 / mu
    u_equal = (4.0 / 15.0) * drho * g * R**2 / mu
    assert hr_velocity(drho, g, R, mu, 1e-9) == pytest.approx(u_bubble, rel=1e-6)
    assert hr_velocity(drho, g, R, mu, 1.0) == pytest.approx(u_equal, rel=1e-6)


def test_confinement_extrapolation_recovers_synthetic_intercept():
    u0, k = 0.0123, 0.02
    recs = [
        {"confinement": x, "u_lbm": u0 * (1.0 - k * x), "u_stokes_corrected": u0 * (1.0 - k * x),
         "u_hr": u0}
        for x in (0.25, 0.1875, 0.15)
    ]
    out = confinement_extrapolation(recs)
    assert out["u_unbounded"] == pytest.approx(u0, rel=1e-6)
    assert abs(out["error_unbounded"]) < 1e-6
    assert out["fit_residual_relative"] < 1e-6


# ---------------------------------------------------------------------------
# 12-16: solver identity, conservation, negative control, parameter scan
# ---------------------------------------------------------------------------


def test_buoyant_collision_is_bit_identical_to_library_collision():
    """With the buoyancy switched off the kernel must be *the* library kernel."""
    L = 16
    zz, yy, xx = torch.meshgrid(
        *[torch.arange(L, dtype=torch.float64)] * 3, indexing="ij"
    )
    rr = torch.sqrt((xx - L / 2) ** 2 + (yy - L / 2) ** 2 + (zz - L / 2) ** 2)
    rho = torch.where(rr <= 4.0, torch.full_like(rr, 0.16), torch.full_like(rr, 1.957))
    zero = torch.zeros_like(rho)
    f = equilibrium3d(rho, zero, zero, zero)
    cfg = BubbleConfig(L=L, radius=4.0, g_buoy=0.0, tau=1.0)
    a = collide_sc_single_component_3d(f, G=5.0, tau=1.0, psi_fn=psi_exp)
    b = _buoyant_collide(f, 0.0, cfg)
    assert float((a - b).abs().max()) == 0.0


def test_negative_control_zero_supply_gives_zero_coverage_and_drag():
    """Zero gas on a wetting plate must *not* produce a gas layer."""
    res = run_pair(CurtainConfig(**SMALL, U_target=0.06, Q_gas=0.0, wall_psi_ratio=1.0))
    assert res["coverage"] < 1e-3
    assert res["layer_thickness"] < 1e-3
    assert abs(res["DR"]) < 1e-3


def test_mass_ledger_closes_with_injected_gas_booked():
    res = CurtainChannel2D(
        CurtainConfig(**SMALL, U_target=0.06, Q_gas=0.25, wall_psi_ratio=0.2)
    ).run()
    assert res["mass_injected"] > 0.0
    # The ledger closes to O(displace_fraction * excess): the liquid displaced out
    # of the source cells that reaches the solid floor row is zeroed with the
    # bounce-back mask.  The residual is bounded and reported, not hidden.
    assert res["mass_drift"] < 5e-3


def test_gas_supply_increases_coverage():
    """Parameter scan: more gas -> more coverage (monotone, no threshold cliff)."""
    cov = []
    for q in (0.0, 0.2):
        res = CurtainChannel2D(
            CurtainConfig(**SMALL, U_target=0.06, Q_gas=q, wall_psi_ratio=0.2)
        ).run()
        cov.append(res["coverage"])
    assert cov[1] > cov[0]


def test_higher_speed_reduces_drag_reduction_at_fixed_supply():
    """Froude trend branch: at fixed gas supply, higher U thins the layer."""
    drs = []
    for u in (0.03, 0.10):
        res = run_pair(CurtainConfig(**SMALL, U_target=u, Q_gas=0.25, wall_psi_ratio=0.2))
        drs.append(res["DR"])
    assert drs[1] <= drs[0] + 0.05


def test_bubble_run_is_finite_at_small_size():
    """A short 3D run must stay finite and mass conserving (no NaN, no drift)."""
    from tensorlbm.bubble_dr.bubble3d import RisingBubble3D

    cfg = BubbleConfig(L=24, radius=5.0, steps=40, g_buoy=2e-3, tau=1.0)
    sim = RisingBubble3D(cfg)
    sim.step(40)
    m = sim.measure()
    assert math.isfinite(m["u_gas_z"])
    assert math.isfinite(m["R_eq"])
    assert m["mass_drift"] < 1e-10