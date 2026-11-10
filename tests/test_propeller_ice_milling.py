"""Tests for :mod:`tensorlbm.propeller_ice_milling` (campaign node p367seed-n08).

The suite verifies the *machinery*, not the campaign verdicts:

* geometry invariants of the milling configuration (blade never overlaps the ice,
  the tip clearance stays open at every blade angle);
* exact mass conservation for a frozen geometry and the audited ledger for the
  moving boundary;
* the viscous-dissipation estimator against the analytic Taylor-Green decay
  (``int eps dt == -dKE``), which is what the energy ledger relies on;
* the momentum-exchange force: correct sign of the drag and mirror symmetry for a
  static obstacle in a uniform stream;
* the post-processing helpers (Richardson order, FFT peak, pulse counting) on
  synthetic data with known answers;
* the ISO 19906 empirical baseline and the dimensional-analysis scaling.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.propeller_ice_milling import (
    BLADE, ICE, MillingConfig, Iso19906Ice, GeometryCache, build_labels,
    equilibrium, macroscopic, lbm_step, run_milling, dissipation_rate,
    torque_coefficient, richardson_order, dominant_frequency, count_pulses,
    smooth_signal, fluctuation_rms, mass_ledger_residual, energy_ledger_residual,
    iso19906_milling_torque, physical_scaling, blade_mask, ice_mask,
)

torch.set_num_threads(1)


def _cfg(**kw):
    base = dict(n=48, revolutions=0.5, idle_revolutions=0.0, tip_ma=0.18,
                reynolds=120.0, blade_thickness_frac=0.12, gap_frac=0.15)
    base.update(kw)
    return MillingConfig(**base)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_blade_never_overlaps_ice_and_gap_stays_open():
    cfg = _cfg(n=64)
    cache = GeometryCache(cfg)
    r, th = cache.r, cache.th
    n_steps = 12
    for k in range(n_steps):
        angle = 2.0 * math.pi * k / n_steps
        lab = cache.labels(angle)
        blade = lab == BLADE
        ice = lab == ICE
        assert not bool((blade & ice).any()), "blade and ice masks overlap"
        # minimum radial distance between blade cells and ice cells
        if ice.any() and blade.any():
            r_ice_min = float(r[ice].min())
            r_blade_max = float(r[blade].max())
            assert r_ice_min - r_blade_max >= 0.5 * cfg.gap, (
                "tip clearance smaller than half the prescribed gap")
    # the blade mask must actually rotate (otherwise nothing is moving)
    a = cache.labels(0.0)
    b = cache.labels(math.pi / 3.0)
    assert not bool((a == BLADE).equal(b == BLADE))


def test_masks_are_thin_plates_with_the_prescribed_thickness():
    cfg = _cfg(n=96)
    cache = GeometryCache(cfg)
    m = blade_mask(cache.r, cache.th, cfg, 0.0, cache.sinth, cache.costh)
    # the plate lies along theta = 0 -> its cell count is ~ R * thickness
    area = float(m.sum())
    expect = (cfg.blade_radius - cfg.hub_radius) * cfg.blade_thickness * cfg.n_blades
    assert 0.5 * expect < area < 1.6 * expect


# ---------------------------------------------------------------------------
# conservation
# ---------------------------------------------------------------------------


def test_frozen_geometry_mass_is_conserved_to_machine_precision():
    """With the rotor *and* the wall velocity frozen there is no source term, so
    the half-way bounce-back must conserve ``sum(f)`` to round-off."""
    cfg = _cfg(n=48)
    cfg.omega = 0.0                      # frozen rotor: no moving-wall term
    cache = GeometryCache(cfg)
    labels = cache.labels(0.0).clone()
    solid = labels > 0
    rho0 = torch.ones((cfg.n, cfg.n))
    u0 = torch.zeros((cfg.n, cfg.n))
    f = equilibrium(rho0, u0, u0)
    f = torch.where(solid.unsqueeze(0), torch.zeros_like(f), f)
    m0 = float(f.sum(dtype=torch.float64).item())
    for _ in range(20):
        f, _, _ = lbm_step(f, labels, cfg, collect_dissipation=False)
    m1 = float(f.sum(dtype=torch.float64).item())
    assert abs(m1 - m0) / m0 <= 1e-10


def test_moving_blade_mass_ledger_books_every_exchange():
    cfg = _cfg(n=48, revolutions=0.35)
    hist = run_milling(cfg, collect_dissipation=False)
    led = mass_ledger_residual(hist)
    # audit residual is the float32 round-off floor of the ledger itself ...
    assert led["relative"] <= 2e-4
    # ... while the moving-wall BC injects a physical mass source that must be
    # booked explicitly (isothermal LBM compresses under the rotor pressure)
    assert led["wall_mass_source_relative"] > 1e-6
    assert led["added"] > 0 and led["removed"] > 0


def test_dissipation_estimator_matches_taylor_green_decay():
    n = 64
    k = 2.0 * math.pi / n
    nu = 0.02
    tau = 0.5 + 3.0 * nu
    u0 = 0.02
    y = torch.arange(n).view(n, 1).float()
    x = torch.arange(n).view(1, n).float()
    ux = u0 * torch.sin(k * y) * torch.cos(k * x)
    uy = -u0 * torch.cos(k * y) * torch.sin(k * x)
    f = equilibrium(torch.ones((n, n)), ux, uy)
    cfg = MillingConfig(n=n, tip_ma=0.18)
    cfg.nu = nu
    cfg.tau = tau
    labels = torch.zeros((n, n), dtype=torch.int8)

    def ke(state):
        r, u, v = macroscopic(state)
        return float((0.5 * r * (u * u + v * v)).sum())

    k0 = ke(f)
    total = 0.0
    for _ in range(200):
        f, _, diss = lbm_step(f, labels, cfg, collect_dissipation=True)
        total += diss
    k1 = ke(f)
    assert k1 < k0
    ratio = total / (k0 - k1)
    assert 0.8 < ratio < 1.2, f"dissipation/kinetic-energy-decay ratio {ratio}"


# ---------------------------------------------------------------------------
# force evaluation
# ---------------------------------------------------------------------------


def test_momentum_exchange_force_sign_and_symmetry():
    """Static square obstacle in a uniform stream: drag along the flow, no lift.

    The instantaneous momentum-exchange force carries the well-known half-step
    oscillation of the Ladd scheme, so the *time mean* is what is asserted.
    """
    n = 64
    cfg = MillingConfig(n=n, tip_ma=0.18, reynolds=200.0)
    labels = torch.zeros((n, n), dtype=torch.int8)
    c = n // 2
    labels[c - 3:c + 3, c - 3:c + 3] = ICE
    u0 = 0.02
    f = equilibrium(torch.ones((n, n)),
                    torch.full((n, n), u0), torch.zeros((n, n)))
    acc = np.zeros(2)
    for it in range(60):
        f, diag, _ = lbm_step(f, labels, cfg, collect_dissipation=False)
        if it >= 20:                     # discard the impulsive start-up
            acc += np.array([float(diag["force"][ICE, 0]),
                             float(diag["force"][ICE, 1])])
    fx, fy = acc / 40.0
    assert fx > 0.0, f"drag must point downstream, got Fx={fx}"
    assert abs(fy) < 0.15 * abs(fx), f"lift must vanish by symmetry, got Fy={fy}"


# ---------------------------------------------------------------------------
# post-processing helpers
# ---------------------------------------------------------------------------


def test_richardson_order_recovers_a_known_exponent():
    hs = [1.0, 0.8, 0.625]
    exact, amp, p_true = 1.5, 2.0, 2.0
    vals = [exact + amp * h ** p_true for h in hs]
    out = richardson_order(hs, vals)
    assert abs(out["p"] - p_true) < 0.05
    assert abs(out["f_extrapolated"] - exact) < 1e-6


def test_dominant_frequency_and_pulse_count_on_a_synthetic_train():
    period = 800
    n = 2400
    t = np.arange(n, dtype=float)
    sig = np.zeros(n)
    duty = 0.3
    for i in range(n):
        ph = (t[i] % period) / period
        if ph <= duty:
            sig[i] = math.sin(math.pi * ph / duty)
    sig += 0.01 * np.random.default_rng(0).standard_normal(n)
    f_bpf = 1.0 / period
    out = dominant_frequency(smoothed := smooth_signal(sig, 16), dt=1.0)
    assert abs(out["f_peak"] - f_bpf) / f_bpf < 0.01
    pk = count_pulses(smoothed, prominence_frac=0.25)
    assert pk["n_peaks"] == 3


def test_fluctuation_rms_and_smoothing():
    x = [1.0, -1.0] * 200
    rms = fluctuation_rms(x)
    assert rms["mean"] == pytest.approx(0.0, abs=1e-12)
    sm = np.asarray(smooth_signal(x, 20))
    assert np.std(sm) < 0.1 * np.std(x)


# ---------------------------------------------------------------------------
# empirical baseline / scaling
# ---------------------------------------------------------------------------


def test_iso19906_baseline_is_physical_and_monotone():
    base = iso19906_milling_torque(Iso19906Ice())
    for key in ("p_G", "contact_area", "contact_force", "torque"):
        assert base[key] > 0.0
    stronger = iso19906_milling_torque(Iso19906Ice(sigma_c=3.0e6))
    assert stronger["torque"] > base["torque"]
    # p_G scales linearly with the ice strength
    assert stronger["p_G"] / base["p_G"] == pytest.approx(1.5, rel=1e-9)


def test_physical_scaling_and_torque_coefficient():
    cfg = _cfg()
    sc = physical_scaling(cfg, rev_per_s=2.0)
    assert sc["u_tip_m_per_s"] == pytest.approx(2.0 * math.pi * 2.0, rel=1e-12)
    ct = torque_coefficient(10.0, cfg)
    assert ct == pytest.approx(10.0 / (cfg.omega ** 2 * cfg.blade_radius ** 4))


def test_energy_ledger_runs_and_reports_all_terms():
    cfg = _cfg(n=48, revolutions=0.25)
    hist = run_milling(cfg, collect_dissipation=True)
    led = energy_ledger_residual(hist, cfg)
    for key in ("int_power_in", "delta_total_energy", "int_dissipation",
                "boundary_energy", "relative"):
        assert key in led
    assert led["int_dissipation"] > 0.0
    assert led["int_power_in"] != 0.0