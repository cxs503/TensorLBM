"""Tests for the hindered-settling dispersion closure (``settling_dispersion``)."""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.frazil_suspension import hindered_settling_velocity
from tensorlbm.settling_dispersion import (
    _scalar_equilibrium,
    _scalar_stream_nf,
    fit_dispersion_law,
    killer_test_dispersion,
    measure_hindered_dispersion,
    random_disk_array,
    run_settling_column_dispersion,
)

torch.set_num_threads(1)


# ---------------------------------------------------------------------------
# Array geometry
# ---------------------------------------------------------------------------


def test_random_disk_array_matches_target_fraction():
    ny, nx, radius, target = 96, 128, 6.0, 0.2
    disks = random_disk_array(ny, nx, target, radius=radius, seed=0)
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1) + 0.5
    x = torch.arange(nx, dtype=torch.float64).view(1, -1) + 0.5
    mask = torch.zeros((ny, nx), dtype=torch.bool)
    for d in disks:
        mask |= (x - d.center[0]) ** 2 + (y - d.center[1]) ** 2 <= radius**2
    assert abs(float(mask.double().mean()) - target) < 0.02


def test_random_disk_array_is_reproducible_and_separated():
    a = random_disk_array(64, 64, 0.15, radius=5.0, seed=3, min_gap=1.0)
    b = random_disk_array(64, 64, 0.15, radius=5.0, seed=3, min_gap=1.0)
    assert [d.center for d in a] == [d.center for d in b]
    for i, p in enumerate(a):
        for q in a[i + 1:]:
            dx = abs(p.center[0] - q.center[0])
            dx = min(dx, 64 - dx)
            dy = abs(p.center[1] - q.center[1])
            dy = min(dy, 64 - dy)
            assert math.hypot(dx, dy) >= 2 * 5.0


def test_random_disk_array_rejects_impossible_fraction():
    with pytest.raises(ValueError):
        random_disk_array(32, 32, 0.8, radius=6.0)


# ---------------------------------------------------------------------------
# Scalar operator
# ---------------------------------------------------------------------------


def test_scalar_no_flux_walls_conserve_mass():
    ny, nx = 32, 64
    fluid = torch.ones((ny, nx), dtype=torch.bool)
    fluid[10:14, 20:26] = False  # an obstacle
    x = torch.arange(nx, dtype=torch.float64).view(1, -1) + 0.5
    c = torch.exp(-0.5 * ((x - 32.0) / 4.0) ** 2).expand(ny, nx).contiguous()
    g = _scalar_equilibrium(c, torch.zeros_like(c), torch.zeros_like(c))
    m0 = math.fsum((g.sum(dim=0) * fluid).flatten().tolist())
    steps = 300
    for _ in range(steps):
        cc = g.sum(dim=0)
        g = g - (g - _scalar_equilibrium(cc, torch.zeros_like(cc), torch.zeros_like(cc))) / 0.56
        g = _scalar_stream_nf(g, fluid)
    m1 = math.fsum((g.sum(dim=0) * fluid).flatten().tolist())
    drift = abs(m1 - m0) / m0
    # the scheme is algebraically conservative: the only drift is float64
    # round-off, so it must accumulate at ~1e-16 per step
    assert drift < 1e-13, drift
    assert drift / steps < 2e-16, drift / steps


# ---------------------------------------------------------------------------
# Closed-form law
# ---------------------------------------------------------------------------


def test_fit_dispersion_law_recovers_synthetic_parameters():
    d0, alpha = 0.35, 2.6
    cs = [0.1, 0.2, 0.3]
    d = [d0 * (1 - c) ** alpha for c in cs]
    out = fit_dispersion_law(cs, d)
    assert out["d0"] == pytest.approx(d0, rel=1e-10)
    assert out["alpha"] == pytest.approx(alpha, rel=1e-10)
    assert out["max_relative_error"] < 1e-12


def test_fit_dispersion_law_reports_holdout_error():
    d0, alpha = 0.35, 2.6
    cs = [0.1, 0.2, 0.3, 0.4]
    d = [d0 * (1 - c) ** alpha for c in cs]
    out = fit_dispersion_law(cs, d, holdout=0.4)
    assert out["holdout_relative_error"] < 1e-12
    assert out["calibration_concentrations"] == [0.1, 0.2, 0.3]


def test_fit_dispersion_law_needs_two_points():
    with pytest.raises(ValueError):
        fit_dispersion_law([0.2], [0.1])


# ---------------------------------------------------------------------------
# Kynch column
# ---------------------------------------------------------------------------


def _const(value):
    return lambda c: torch.full_like(c, value)


def test_column_kynch_interface_speed():
    """The falling interface must move at the analytic shock speed v_s(c0)."""
    c0 = 0.2
    out = run_settling_column_dispersion(
        c0, d_of_c=_const(0.02), d_m=1e-4, H=80, travel_cells=12, max_steps=20000
    )
    expected = float(hindered_settling_velocity(c0, out["v_s0"], exponent=out["exponent"]))
    assert out["interface_velocity_expected"] == pytest.approx(expected, rel=1e-12)
    assert out["relative_error"] < 0.03, out["relative_error"]


def test_column_mass_is_conserved():
    out = run_settling_column_dispersion(
        0.2, d_of_c=_const(0.02), d_m=1e-4, H=48, travel_cells=2, max_steps=200
    )
    # algebraically conservative; only float64 round-off accumulates
    assert out["mass_drift"] < 1e-13, out["mass_drift"]
    assert out["mass_drift"] / max(1, out["steps"]) < 2e-16


def test_column_rejects_bad_concentration():
    with pytest.raises(ValueError):
        run_settling_column_dispersion(0.7, H=32)


def test_killer_test_detects_the_dispersion_term():
    """Switching D_settle off must change the interface thickness by >20%."""
    out = killer_test_dispersion(
        0.2, d_of_c=_const(0.1), d_m=1e-4, H=80, travel_cells=12, max_steps=20000
    )
    assert out["thickness_with"] > 1.5 * out["thickness_without"]
    assert out["killer_pass"]


# ---------------------------------------------------------------------------
# Resolved dispersion (slow smoke test)
# ---------------------------------------------------------------------------


def test_resolved_dispersion_exceeds_molecular():
    """A measured D_eff above the molecular value on a small resolved array."""
    out = measure_hindered_dispersion(
        0.2,
        d_m=0.02,
        ny=64,
        nx=96,
        radius=5.0,
        flow_steps=900,
        scalar_steps=1200,
        body_force=3e-4,
        check_every=25,
    )
    assert math.isfinite(out["d_eff"])
    assert out["d_eff"] > 1.2 * out["d_m"]
    assert out["mass_drift"] < 1e-10
    assert out["n_disks"] > 3