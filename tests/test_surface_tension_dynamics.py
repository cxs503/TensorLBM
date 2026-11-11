"""Tests for :mod:`tensorlbm.surface_tension_dynamics`.

Two layers of protection:

* **measurement chain** (fast, ``unit``) -- the estimators that turn a raw
  amplitude record into ``(omega, gamma)``, the analytic reference profiles and
  the geometric detectors are tested against synthetic signals with known
  answers, so a silently biased estimator cannot turn into a passing report;
* **solver anchors** (``physics``/``benchmark``, small grids so the PR scope
  stays inside a few seconds) -- the negative control must produce no wave, the
  capillary wave must oscillate with the right order of magnitude, the Marangoni
  channel must recover the analytic interface velocity and the coalescence
  detector must see the neck grow.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.surface_tension_dynamics.capillary_wave import (
    WaveConfig,
    extrema_half_period,
    interface_sharpness,
    standing_wave_phi,
    zero_crossing_period,
)
from tensorlbm.surface_tension_dynamics.coalescence import (
    CoalescenceConfig,
    neck_radius,
    two_drop_phi,
)
from tensorlbm.surface_tension_dynamics.marangoni import (
    MarangoniConfig,
    analytic_couette_profile,
    run_marangoni_channel,
)
from tensorlbm.surface_tension_dynamics.capillary_wave import run_standing_wave


# --------------------------------------------------------------------------- #
# measurement chain
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_finite_depth_theory_reduces_to_deep_water():
    cfg = WaveConfig(nx=64, ny=32, mode=1, sigma=0.05)  # k = 0.0982, H = 16 -> kH = pi/2
    deep = cfg.omega_theory_deep()
    finite = cfg.omega_theory()
    # k*H = pi/2 here -> tanh = 0.917: the finite-depth correction must bite
    assert finite < deep
    assert finite / deep == pytest.approx(math.sqrt(math.tanh(math.pi / 2)), rel=1e-12)
    deep_box = WaveConfig(nx=64, ny=512, mode=1, sigma=0.05)  # kH = 25 -> tanh = 1
    assert deep_box.omega_theory() / deep_box.omega_theory_deep() == pytest.approx(1.0, rel=1e-6)



@pytest.mark.unit
def test_wave_config_rejects_nonlinear_and_unstable_inputs():
    with pytest.raises(ValueError):
        WaveConfig(nx=64, ny=32, mode=4, amplitude=0.9).validate()  # A*k = 0.35
    with pytest.raises(ValueError):
        WaveConfig(nx=64, ny=32, tau=0.5).validate()
    with pytest.raises(ValueError):
        WaveConfig(nx=63, ny=32).validate()
    WaveConfig(nx=64, ny=32).validate()  # the default case must be accepted


@pytest.mark.unit
def test_standing_wave_phi_bounds_and_interface_position():
    cfg = WaveConfig(nx=64, ny=32, mode=2, amplitude=0.4, width=2.0)
    phi = standing_wave_phi(cfg)
    assert float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0
    x = torch.arange(cfg.nx, dtype=torch.float64)
    yi = 0.5 * cfg.ny + cfg.amplitude * torch.cos(cfg.k * x)
    # the phi = 0.5 contour sits on y_i: interpolate the column-wise crossing
    for j in (0, cfg.nx // 4, cfg.nx // 2):
        col = phi[:, j]
        below = (col > 0.5).to(torch.int64).sum().item()
        assert abs(below - yi[j].item()) <= 1.0
    # the central-difference read-out of a tanh interface is 0.5*tanh(2/W), i.e.
    # it underestimates 1/W (0.383 vs 0.5 at W=2) -- a known, documented bias
    assert interface_sharpness(phi) == pytest.approx(0.5 * math.tanh(2.0 / cfg.width), rel=0.02)


@pytest.mark.regression
def test_interface_sharpness_ignores_the_periodic_seam():
    """Regression: the seam at y=0/ny-1 used to pin the read-out at 0.5 for every W.

    A plain central-difference ``|grad phi|`` sees ``phi`` jump from 1 (row 0) to 0
    (row -1) across the periodic seam and returns 0.5 there, i.e. the interface
    width was reported as constant 2 cells whatever ``W`` was.  The trimmed
    diagnostic must instead scale like ``1/W``.
    """
    vals = {
        w: interface_sharpness(standing_wave_phi(WaveConfig(nx=32, ny=32, mode=1, width=w)))
        for w in (1.5, 2.0, 4.0)
    }
    assert vals[1.5] > vals[2.0] > vals[4.0]
    assert vals[4.0] < 0.3  # untrimmed this would be exactly 0.5 for any W
    for w, v in vals.items():
        assert v == pytest.approx(0.5 * math.tanh(2.0 / w), rel=0.02)


@pytest.mark.unit
def test_extrema_half_period_recovers_frequency_under_strong_damping():
    omega, gamma = 0.0374, 0.006  # gamma/omega = 0.16: strongly damped
    t = [i * 2.0 for i in range(1500)]
    a = [math.exp(-gamma * ti) * math.cos(omega * ti) for ti in t]
    est = extrema_half_period(t, a)
    assert est["n_half_periods"] >= 5
    assert est["omega"] / omega == pytest.approx(1.0, rel=0.01)
    assert est["spread"] < 4.0  # spacing is regular to a couple of samples


@pytest.mark.unit
def test_extrema_half_period_flags_a_short_record():
    est = extrema_half_period([0.0, 1.0, 2.0], [1.0, 0.5, 0.0])
    assert est["n_half_periods"] == 0
    assert math.isnan(est["omega"])


@pytest.mark.unit
def test_zero_crossing_period_recovers_frequency():
    omega = 0.05
    t = [float(i) for i in range(1000)]
    a = [math.cos(omega * ti) for ti in t]
    est = zero_crossing_period(t, a)
    assert est["omega"] / omega == pytest.approx(1.0, rel=0.01)


# --------------------------------------------------------------------------- #
# Marangoni reference
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_analytic_couette_profile_is_piecewise_linear_and_symmetric():
    cfg = MarangoniConfig(dsdx=1e-5, Ly=64, Lx=4)
    prof = analytic_couette_profile(cfg)
    ui = cfg.u_interface()
    assert prof[1] < prof[31] and prof[31] == pytest.approx(prof[32], rel=1e-12)
    assert prof[32] > prof[60]
    # peak sits at the interface (rows 31/32 straddle y0 = 32)
    assert max(prof) <= ui * 1.001
    # the two bounce-back wall rows are solid, the first fluid row is half a
    # cell from the no-slip plane at y = 0.5 (measured zero crossing of the fit)
    h1, h2 = cfg.heights()
    assert prof[0] == 0.0 and prof[-1] == 0.0
    assert prof[1] == pytest.approx(ui * 1.0 / h1, rel=1e-12)
    assert prof[-2] == pytest.approx(ui * 1.0 / h2, rel=1e-12)


@pytest.mark.unit
def test_marangoni_interface_velocity_scales_linearly_with_gradient():
    a = MarangoniConfig(dsdx=1.0e-5, Ly=32, Lx=2)
    b = MarangoniConfig(dsdx=3.0e-5, Ly=32, Lx=2)
    assert b.u_interface() / a.u_interface() == pytest.approx(3.0, rel=1e-12)
    # asymmetric layer heights give a lower interface velocity than symmetric ones
    sym = MarangoniConfig(dsdx=1e-5, Ly=32, Lx=2, y0=16.0)
    asym = MarangoniConfig(dsdx=1e-5, Ly=32, Lx=2, y0=8.0)
    assert asym.u_interface() < sym.u_interface()
    mu = (asym.tau - 0.5) / 3.0
    assert asym.u_interface() == pytest.approx(1e-5 * 7.5 * 23.5 / (31.0 * mu), rel=1e-12)


@pytest.mark.physics
@pytest.mark.slow
def test_marangoni_channel_reaches_analytic_interface_velocity():
    cfg = MarangoniConfig(dsdx=2.0e-5, Ly=32, Lx=2, steps=4000, sample=500)
    res = run_marangoni_channel(cfg)
    m = res["metrics"]
    assert m["u_drift_rel"] < 0.05, "the profile must be steady before it is compared"
    assert abs(m["u_interface_rel_err"]) < 0.20
    assert m["profile_rel_l2"] < 0.20


# --------------------------------------------------------------------------- #
# coalescence geometry
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_two_drop_phi_makes_two_disks_and_a_detected_neck():
    cfg = CoalescenceConfig(radius=6.0, gap=0.0, width=1.2)
    phi = two_drop_phi(cfg)
    assert phi.dim() == 2
    assert float(phi.min()) >= 0.0 and float(phi.max()) <= 1.0
    r, n = neck_radius(phi)
    assert n >= 1 and r > 0.0
    # a well separated pair has no neck at all
    far = two_drop_phi(CoalescenceConfig(radius=6.0, gap=8.0, width=1.2))
    r_far, n_far = neck_radius(far)
    assert n_far == 0 and r_far == 0.0


@pytest.mark.unit
def test_neck_radius_recovers_a_known_cylinder():
    ny, nx = 40, 40
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1) + 0.5
    x = torch.arange(nx, dtype=torch.float64).view(1, -1) + 0.5
    # liquid band |y - 20| < 7 -> the neck half height is 7
    phi = 0.5 + 0.5 * torch.tanh(2.0 * (7.0 - torch.abs(y - 20.0)) / 1.0) * torch.ones_like(x)
    r, n = neck_radius(phi)
    assert r == pytest.approx(7.0, abs=0.2)
    assert n == 14


# --------------------------------------------------------------------------- #
# solver anchors (small grids)
# --------------------------------------------------------------------------- #
@pytest.mark.physics
@pytest.mark.slow
def test_negative_control_without_surface_tension_has_no_wave():
    cfg = WaveConfig(nx=32, ny=16, mode=1, amplitude=0.2, sigma=0.0, steps=600, sample=10)
    res = run_standing_wave(cfg)
    nc = res["negative_control"]
    assert nc["amp_max_dev"] < 0.01 * nc["amp_initial"] + 1e-6
    assert nc["umax"] < 1e-2


@pytest.mark.physics
@pytest.mark.slow
def test_capillary_wave_oscillates_with_the_analytic_period():
    """Order-of-magnitude anchor on a tiny grid (the 5% claim is a campaign test).

    ``ny = 32`` gives ``kH = k*16``; with ``mode=1, nx=32`` -> ``k = 0.196`` and
    ``kH = 3.14``, so the finite-depth factor is negligible and the oscillation
    must be visible within one analytic period.
    """
    cfg = WaveConfig(nx=32, ny=32, mode=1, amplitude=0.2, sigma=0.1, width=2.0, tau=0.55,
                     steps=420, sample=2)
    res = run_standing_wave(cfg)
    theory = res["theory"]["omega"]
    ext = res["extrema"]
    assert ext["n_half_periods"] >= 1
    if ext.get("omega") == ext.get("omega"):
        assert ext["omega"] / theory == pytest.approx(1.0, rel=0.30)
    # the interface must stay resolved through the run
    assert res["sharpness_final"] > 0.7 * res["sharpness_initial"]
    assert res["mass_drift_rel"] < 5e-3