"""Verification of the pulsatile-channel (Womersley) LBM case.

Anchor: ``CASE-LBM-WOMERSLEY-PULSATILE`` (Womersley 1955, *J. Physiol.*
127:553) — centreline amplitude ratio and phase lag vs the analytic solution at
``α = 5``, whole-period average, tolerance 5 %.

Everything asserted here is derived from an *independent* route, never from the
implementation's own algebra:

* flux identity ``⟨Ŵ⟩ = 1`` by quadrature (the normalisation the gate relies on),
* quasi-steady limit ``(3/2)(1-ŷ²)`` with a measured ``O(α²)`` order,
* a finite-difference solution of the same ODE (no cosh algebra),
* the classical high-``α`` asymptotes (centreline → plug, wall shear leads by 45°),
* steady Poiseuille ``u_c = a H²/(8ν)``, which calibrates the Guo forcing
  bookkeeping and the half-way bounce-back wall position *before* the pulsatile
  case is trusted,
* a rest-start control run (no analytic initial condition) that must return the
  same phasor as the seeded run — otherwise the "agreement" would only be the
  initial condition being re-measured.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.verification.womersley_pulsatile import (
    GATE,
    ChannelPulsatileConfig,
    channel_centreline_phasor,
    channel_phasor,
    evaluate_gate,
    quasi_steady_shape,
    run_pulsatile_channel,
    run_steady_poiseuille,
    unit_bulk_kappa,
)

TRAPZ = getattr(np, "trapezoid", None) or getattr(np, "trapz")
NU = 0.05


# ---------------------------------------------------------------------------
# 1. flux identity: the unit-bulk normalisation the gate depends on
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
def test_unit_bulk_flux_identity(alpha):
    """∫Ŵ dy / H = 1 exactly — walls included, they carry Ŵ = 0."""
    H = 32.0
    omega = NU * (alpha / H) ** 2
    y = np.linspace(-H / 2.0, H / 2.0, 40001)
    assert abs(TRAPZ(channel_phasor(y, H, omega, NU), y) / H - 1.0) < 1e-8


# ---------------------------------------------------------------------------
# 2. quasi-steady limit -> Poiseuille, second order in alpha
# ---------------------------------------------------------------------------
def test_quasi_steady_limit_is_second_order():
    """Ŵ → (3/2)(1-ŷ²) with error O(α²); halving α must quarter the error."""
    H = 32.0
    y = np.linspace(-H / 2.0, H / 2.0, 20001)
    ref = quasi_steady_shape(y, H)
    errs = []
    for alpha in (0.05, 0.025, 0.0125):
        omega = NU * (alpha / H) ** 2
        errs.append(np.abs(channel_phasor(y, H, omega, NU) - ref).max() / 1.5)
    assert errs[1] / errs[0] == pytest.approx(0.25, rel=5e-2)
    assert errs[2] / errs[1] == pytest.approx(0.25, rel=5e-2)


# ---------------------------------------------------------------------------
# 3. finite-difference solution of the same ODE (Bessel/cosh-free route)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("alpha", [1.0, 5.0, 10.0])
def test_phasor_matches_finite_difference_ode_solution(alpha):
    """Solve ν Û'' - iω Û = -a with FD and compare with the closed form."""
    H = 32.0
    omega = NU * (alpha / H) ** 2
    n = 4000
    y = np.linspace(-H / 2.0, H / 2.0, n + 1)
    h = y[1] - y[0]
    # unknowns are the interior nodes 1..n-1; Dirichlet Û = 0 at both walls
    m = n - 1
    main = np.full(m, -2.0 / h**2 - 1j * omega / NU, dtype=complex)
    off = np.full(m - 1, 1.0 / h**2, dtype=complex)
    rhs = np.full(m, -1.0 / NU, dtype=complex)  # a = 1
    # Thomas algorithm (complex tridiagonal)
    a = off.copy()
    b = main.copy()
    c = off.copy()
    d = rhs.copy()
    for i in range(1, m):
        w = a[i - 1] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]
    x = np.empty(m, dtype=complex)
    x[-1] = d[-1] / b[-1]
    for i in range(m - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]
    u_fd = np.zeros(n + 1, dtype=complex)
    u_fd[1:-1] = x
    # closed form with a = 1, normalised to unit bulk
    bulk_fd = TRAPZ(u_fd, y) / H
    w_num = u_fd / bulk_fd
    w_ana = channel_phasor(y, H, omega, NU)
    interior = np.abs(2.0 * y / H) <= 0.9
    rel = np.abs(w_num[interior] - w_ana[interior]) / np.abs(w_ana[interior]).max()
    assert rel.max() < 2e-3, f"alpha={alpha}: FD vs closed form rel={rel.max():.2e}"


# ---------------------------------------------------------------------------
# 4. classical high-alpha asymptotes
# ---------------------------------------------------------------------------
def test_high_alpha_asymptotes():
    """Centreline → plug in phase with the flow rate; wall shear leads by 45°."""
    # centreline amplitude decays towards 1 and the lag towards 0
    alphas = (5.0, 10.0, 20.0, 50.0, 100.0)
    amps = [abs(channel_centreline_phasor(a)) for a in alphas]
    assert all(x > y for x, y in zip(amps, amps[1:])) and amps[-1] > 1.0
    assert amps[-1] < 1.02  # 1.5 -> 1.014 over alpha = 0 -> 100
    lags = [-math.degrees(np.angle(channel_centreline_phasor(a))) for a in (50.0, 100.0)]
    assert abs(lags[-1]) < 1.0
    # wall shear phase: dŴ/dŷ at ŷ = -1 = κ tanh(κ) / (1 - tanh(κ)/κ) -> κ,
    # i.e. arg -> +45° for large |κ| (the classical Womersley asymptote).
    for alpha in (50.0, 100.0):
        kappa = unit_bulk_kappa(alpha)
        shear = kappa * np.tanh(kappa) / (1.0 - np.tanh(kappa) / kappa)
        assert math.degrees(np.angle(shear)) == pytest.approx(45.0, abs=2.0)
    # intermediate alpha: the centreline *lags* the flow rate (sign structure)
    assert -math.degrees(np.angle(channel_centreline_phasor(5.0))) > 3.0


# ---------------------------------------------------------------------------
# 5. steady Poiseuille: Guo forcing bookkeeping + wall position
# ---------------------------------------------------------------------------
def test_steady_poiseuille_calibrates_forcing_and_walls():
    """Constant force → u_c = a H²/(8ν); a wrong a/2 shift or wall shows here."""
    res = run_steady_poiseuille(ny=12, nu=NU, a0=1e-4, steps=12000)
    assert res["centre_rel_pct"] < 2.0, res["centre_rel_pct"]
    assert res["profile_rel_max_pct"] < 3.0, res["profile_rel_max_pct"]
    assert abs(res["density_drift_pct"]) < 1e-6


# ---------------------------------------------------------------------------
# 6. the case anchor: alpha = 5, whole-period phase average
# ---------------------------------------------------------------------------
def test_pulsatile_gate_alpha5_meets_anchor():
    """Centreline amplitude and phase vs the exact solution, < 5 % / < 5°."""
    res = run_pulsatile_channel(
        ChannelPulsatileConfig(ny=16, nu=NU, alpha=5.0, skip_periods=2, measure_periods=2)
    )
    gate = res["gate"]
    assert gate["status"] == "pass", gate
    # much tighter than the anchor's tolerance, so a regression is visible
    assert res["amp_rel_pct"] < 1.0, res["amp_rel_pct"]
    assert abs(res["phase_diff_deg"]) < 1.0, res["phase_diff_deg"]
    assert res["profile_amp_rel_pct_max"] < 3.0, res["profile_amp_rel_pct_max"]
    assert res["profile_wave_l2_pct"] < 1.0, res["profile_wave_l2_pct"]
    assert res["Ma"] < 0.1
    assert abs(res["density_drift_pct"]) < 1e-6


def test_gate_rejects_off_target_alpha():
    """The gate must not silently pass when the effective alpha is not 5."""
    fake = {"alpha_effective": 4.0, "amp_rel_pct": 0.0, "phase_diff_deg": 0.0}
    assert evaluate_gate(fake)["status"] == "not_applicable"


def test_rest_start_converges_to_same_phasor():
    """Control: starting from rest (no analytic seeding) must give the same lag.

    Without this the "agreement" could simply be the analytic initial condition
    being measured back.  Tolerances are on the physics, not on the seeding.
    """
    cfg_kwargs = dict(ny=16, nu=NU, alpha=5.0, measure_periods=2)
    seeded = run_pulsatile_channel(
        ChannelPulsatileConfig(skip_periods=1, init_from_exact=True, **cfg_kwargs)
    )
    rest = run_pulsatile_channel(
        ChannelPulsatileConfig(skip_periods=4, init_from_exact=False, **cfg_kwargs)
    )
    ana = rest["centre_lag_ana_deg"]
    assert abs(rest["centre_lag_num_deg"] - ana) < 0.5
    assert abs(rest["centre_lag_num_deg"] - seeded["centre_lag_num_deg"]) < 0.1
    assert abs(rest["amp_rel_pct"] - seeded["amp_rel_pct"]) < 0.2
    assert GATE["alpha"] == pytest.approx(rest["alpha_effective"], abs=5e-3)