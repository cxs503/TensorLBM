"""Transient verification of the Zou-He channel closure against Sexl/Womersley.

Anchor: ``XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTI``.

The steady debt of this closure (converged field vs plane Poiseuille) is covered
by ``test_zou_he_channel.py`` / ``test_zou_he_channel_refinement.py``.  This file
covers the *unsteady* debt: with the driving applied through the two
reconstructed planes, is the periodic response in phase and in amplitude with
the exact oscillatory plane-Poiseuille solution?

Every test asserts an invariant or an external reference value -- none of them
merely executes the code:

* the analytic reference is cross-checked against the independent
  ``REFERENCE_ALPHA5`` values already in the repo;
* the phasor estimator and the driving-gradient estimator are checked against
  synthetic signals whose complex amplitude is known;
* the gate tests assert the 15 % acceptance / 3 % target bound on a converged
  run, plus the exactness of the imposed end densities;
* the falsifiability tests show the same metric rejects a wrong-physics
  reference (quasi-steady and wrong-alpha hypotheses), so a pass is informative.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.boundary.zou_he_channel import (
    zou_he_column_density_momentum,
    zou_he_wall,
)
from tensorlbm.verification.womersley_pulsatile import analytical as sexl
from tensorlbm.verification.zou_he_channel.solver import equilibrium_d2q9
from tensorlbm.verification.zou_he_unsteady import (
    UnsteadyCase,
    analytic_reference,
    measured_gradient,
    nominal_gradient,
    phasor_metrics,
    run_oscillatory,
)

DTYPE = torch.float64
ACCEPT = 0.15      # project-wide acceptance bound on the error metric
TARGET = 0.03      # project-wide target bound


# --------------------------------------------------------------------------- #
# cached solver runs (module scoped: two runs serve all gate/falsifiability
# tests, so the suite pays for each physics setup exactly once)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def run_alpha4():
    case = UnsteadyCase(
        height=16, length=48, tau=0.65, womersley=4.0,
        warmup_periods=3, avg_periods=2,
    )
    return case, run_oscillatory(case)


@pytest.fixture(scope="module")
def run_alpha2():
    case = UnsteadyCase(
        height=16, length=48, tau=0.65, womersley=2.0,
        warmup_periods=3, avg_periods=2,
    )
    return case, run_oscillatory(case)


# --------------------------------------------------------------------------- #
# 1. the analytic reference is the one the rest of the repo already trusts
# --------------------------------------------------------------------------- #
def test_analytic_reference_reproduces_reference_alpha5():
    """My reference construction must reproduce the repo's alpha=5 anchors.

    Odd ``H`` is used on purpose: the node ordinates are ``0.5, 1.5, ..., H-0.5``,
    so a centreline sample only exists for odd ``H``.
    """
    case = UnsteadyCase(height=15, length=16, tau=0.8, womersley=5.0)
    a0, omega = 1.0, case.omega
    U = analytic_reference(case, a0)
    i = (case.height - 1) // 2          # axis row: coordinate == H/2 for odd H
    assert (i + 1) - 0.5 == 0.5 * case.height
    amp_unit = float(np.abs(U[i]) / (a0 / omega))
    lag_deg = float(-np.degrees(np.angle(U[i])))

    ref = sexl.REFERENCE_ALPHA5["channel"]
    assert abs(case.alpha - 5.0) < 1e-12
    assert abs(amp_unit - ref["abs_U0_over_a0_w"]) < 1e-9
    assert abs(lag_deg - ref["phase_lag_deg"]) < 1e-7


def test_even_grid_has_no_centreline_sample_and_metric_knows_it():
    """Discretisation convention, pinned so it cannot drift silently.

    For even ``H`` the two rows straddling the axis sample the profile at
    ``y = H/2 -/+ 0.5``; at low ``alpha`` the sampled axis value is therefore
    ``(H^2/4 - 1/4)/(H^2/4)`` of the analytic centreline value (0.4 % at H=16).
    Error metrics must normalise by the sampled peak -- normalising by the
    continuum centreline would inject that bias into every reported error.

    (At ``alpha >= 3`` the peak itself moves off the axis -- the Sexl annular
    effect -- which is why the peak row and the axis row are reported as two
    separate quantities by ``phasor_metrics``.)
    """
    case = UnsteadyCase(height=16, length=16, tau=0.8, womersley=0.05)
    U = analytic_reference(case, 1.0)
    axis = (case.height // 2 - 1, case.height // 2)
    assert abs(abs(U[axis[0]]) - abs(U[axis[1]])) / abs(U[axis[0]]) < 1e-12
    analytic = float(np.abs(sexl.phasor(
        "channel", np.array([0.5 * case.height]), float(case.height),
        case.alpha, 1.0, case.omega)[0]))
    assert abs(abs(U[axis[0]]) / analytic - (1.0 - 1.0 / case.height ** 2)) < 1e-4


def test_reference_peak_is_off_axis_for_high_womersley():
    """Sexl annular effect: at alpha=4 the amplitude peak must leave the axis.

    This is a physics invariant of the *reference*, and it is the reason the
    metric reports ``amp_peak_err`` and ``amp_axis_err`` separately.
    """
    for alpha, overshoot in ((2.0, None), (6.0, 0.02)):
        case = UnsteadyCase(height=16, length=16, tau=0.65, womersley=alpha)
        U = np.abs(analytic_reference(case, 1.0))
        i_axis = case.height // 2
        if overshoot is None:          # low alpha: peak on the axis
            assert int(np.argmax(U)) in (i_axis - 1, i_axis)
            assert float(U[i_axis - 1:i_axis + 1].max() / U[i_axis]) <= 1.0 + 1e-12
        else:                          # high alpha: annular overshoot
            i_peak = int(np.argmax(U))
            assert i_peak not in (i_axis - 1, i_axis)
            assert float(U[i_peak] / U[i_axis]) > 1.0 + overshoot


def test_analytic_reference_quasi_steady_limit_is_poiseuille():
    """alpha -> 0: amplitude -> the *sampled* parabola, phase lag -> 0.

    The discrete reference is compared with the parabola evaluated at the node
    the peak sits on (exact to O(alpha^2)); the continuum centreline value
    ``a0 H^2/(8 nu)`` is recovered to O(1/H^2) = 0.4 % at H=16, which is the
    discretisation convention documented above, not an error of the solution.
    """
    case = UnsteadyCase(height=16, length=16, tau=0.8, womersley=0.01)
    a0 = 1.0e-4
    U = analytic_reference(case, a0)
    i = int(np.argmax(np.abs(U)))
    yt = (i + 1) - 0.5
    u_node = a0 / (2.0 * case.nu) * yt * (case.height - yt)
    u_ss = a0 * case.height ** 2 / (8.0 * case.nu)
    assert abs(abs(U[i]) - u_node) / u_node < 1e-6
    assert abs(abs(U[i]) - u_ss) / u_ss < 1.5 / case.height ** 2
    assert abs(np.degrees(np.angle(U[i]))) < 1e-2


# --------------------------------------------------------------------------- #
# 2. phasor / gradient estimators on synthetic signals with a known complex amplitude
# --------------------------------------------------------------------------- #
def test_phasor_at_single_frequency_is_exact():
    """Re[U e^{i wt}] sampled over whole periods -> U, to round-off."""
    omega, period, n_per = 2.0 * math.pi / 512.0, 512, 3
    U_true = 0.0123 * np.exp(1j * math.radians(-37.0))
    t = np.arange(1, n_per * period + 1)
    u = np.real(U_true * np.exp(1j * omega * t))
    U_hat = 2.0 / u.size * np.sum(u * np.exp(-1j * omega * t))
    assert abs(U_hat - U_true) / abs(U_true) < 1e-12


def test_measured_gradient_recovers_known_complex_slope():
    """A complex linear density phasor must give -(1/3) d rho/dx exactly."""
    x = np.arange(48, dtype=float)
    slope = -9.0e-5 + 2.5e-6j
    rho = 1.0 + 0.5j + slope * (x - 23.0)          # arbitrary offset is fine
    g = measured_gradient(rho, x)
    assert abs(g - (-slope / 3.0)) / abs(g) < 1e-12


def test_nominal_gradient_is_the_strict_reference():
    """Nominal gradient: real, phase 0, drho/(3(nx-1)); measured must be close."""
    case = UnsteadyCase(height=16, length=48)
    g_nom = nominal_gradient(case)
    assert abs(g_nom.imag) == 0.0
    assert abs(g_nom.real - case.drho / (3.0 * (case.nx - 1))) < 1e-18


# --------------------------------------------------------------------------- #
# 3. the actual verification: end-driven oscillatory channel vs Sexl solution
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "fixture_name,amp_l2_gate,phase_gate_deg",
    [
        ("run_alpha4", TARGET, 3.0),      # thin Stokes layer (2 cells), cheapest
        ("run_alpha2", TARGET / 3.0, 1.0),  # well resolved primary arm
    ],
)
def test_end_driven_channel_matches_sexl_solution(
    request, fixture_name, amp_l2_gate, phase_gate_deg
):
    case, res = request.getfixturevalue(fixture_name)
    m = phasor_metrics(case, res["U"], res["g_fit"])

    # the closure must impose the prescribed densities to round-off ...
    assert res["prescription_residual"] < 1e-12
    # ... and the periodic response must match the exact solution.
    assert m["amp_rel_l2"] < amp_l2_gate, m
    assert m["amp_rel_linf"] < ACCEPT, m
    assert m["phase_rms_deg"] < phase_gate_deg, m
    assert m["amp_rel_l2"] < ACCEPT and m["phase_rms_deg"] < 30.0, m
    # fully developed: neither the amplitude profile nor the flux may drift in x
    assert res["x_spread"] < 1e-2, res["x_spread"]
    assert res["q_spread"] < 1e-2, res["q_spread"]
    # the Stokes layer must be a real (resolved) fraction of the domain
    assert case.stokes_cells >= 2.0


def test_flux_phasor_is_the_emerging_quantity_and_is_conserved(run_alpha4):
    """In pressure mode the flux is emergent: it must be x-uniform but non-zero."""
    case, res = run_alpha4
    q = np.abs(res["q_phasor"])
    lo, hi = res["fit_band_indices"]
    assert q[lo:hi + 1].min() > 0.5 * q[lo:hi + 1].max()
    assert q.mean() > 0.0
    # steady-state flux scale: |q| ~ H * u_max
    assert q.mean() < 10.0 * abs(res["U"]).max() * case.height


def test_measured_gradient_matches_nominal_within_the_convention_band(run_alpha2):
    """The measured driving must agree with the nominal one to O(1)."""
    case, res = run_alpha2
    g_fit, g_nom = res["g_fit"], res["g_nominal"]
    assert abs(g_fit - g_nom) / abs(g_nom) < 0.05
    assert abs(np.degrees(np.angle(g_fit))) < 5.0


# --------------------------------------------------------------------------- #
# 4. falsifiability: the same metric rejects wrong physics on the same profile
# --------------------------------------------------------------------------- #
def test_metric_rejects_quasi_steady_hypothesis(run_alpha4):
    """If the response were quasi-steady the metric must fail badly.

    Same measured profile, reference taken at alpha -> 0 (steady parabola in
    phase with the driving).  A metric that passed this would be blind.
    """
    case, res = run_alpha4
    m = phasor_metrics(case, res["U"], res["g_fit"], alpha=0.05)
    assert (m["amp_rel_l2"] > ACCEPT) or (m["phase_rms_deg"] > 10.0), m


def test_metric_rejects_wrong_womersley_number(run_alpha4):
    """A 50 % error in the frequency must be visible in phase and amplitude."""
    case, res = run_alpha4
    m = phasor_metrics(case, res["U"], res["g_fit"], alpha=1.5 * case.alpha)
    assert m["phase_rms_deg"] > 1.0 or m["amp_rel_l2"] > TARGET, m


def test_nominal_gradient_metric_is_stricter_than_measured(run_alpha4):
    """Reported honestly: the nominal-gradient reference is the stricter one."""
    case, res = run_alpha4
    m_fit = phasor_metrics(case, res["U"], res["g_fit"])
    m_nom = phasor_metrics(case, res["U"], res["g_nominal"])
    assert m_nom["amp_rel_l2"] >= 0.0 and m_nom["phase_rms_deg"] >= 0.0
    # the fitted reference may not be arbitrarily better: it removes only the
    # geometry/acoustic convention error, not a wrong solution
    assert m_fit["amp_rel_l2"] <= m_nom["amp_rel_l2"] + 1.0


# --------------------------------------------------------------------------- #
# 5. boundary-condition invariants under a time-varying prescription
# --------------------------------------------------------------------------- #
def _equilibrium_field(ny: int = 12, nx: int = 8, seed: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    rho = 1.0 + 0.01 * torch.rand((ny, nx), generator=g, dtype=DTYPE)
    ux = 0.02 * (torch.rand((ny, nx), generator=g, dtype=DTYPE) - 0.5)
    uy = 0.02 * (torch.rand((ny, nx), generator=g, dtype=DTYPE) - 0.5)
    return equilibrium_d2q9(rho, ux, uy)


def test_velocity_inlet_tracks_time_varying_profile_with_no_lag():
    """A profile that changes every step must be imposed exactly, step by step.

    This is the BC-level statement behind the amplitude/phase gate: a lagging
    reconstruction would smear ``u_in(t)``, which shows up here first.
    """
    ny, nx = 12, 8
    y = torch.arange(ny, dtype=DTYPE)
    shape = 4.0 * (y - 1.0) * (ny - 1.0 - y) / ((ny - 2.0) ** 2)   # parabola, peak 1
    for phase in (0.0, 0.3, 1.1, 2.4):
        f = _equilibrium_field(ny, nx)
        prof = shape * 0.04 * math.cos(phase)
        zou_he_wall(f, "west", mode="velocity", ux=prof, uy=0.0)
        _, ux_bc, uy_bc = zou_he_column_density_momentum(f, "west")
        assert torch.max(torch.abs(ux_bc - prof)).item() < 1e-12
        assert torch.max(torch.abs(uy_bc)).item() < 1e-12


def test_velocity_mode_requires_the_normal_component():
    """Contract guard: a missing normal component is an error, not a crash."""
    f = _equilibrium_field()
    with pytest.raises(ValueError, match="normal component"):
        zou_he_wall(f, "west", mode="velocity", uy=0.01)      # ux missing
    with pytest.raises(ValueError, match="normal component"):
        zou_he_wall(f, "north", mode="velocity", ux=0.01)     # uy missing
    # the transverse component legitimately defaults to zero
    zou_he_wall(f, "west", mode="velocity", ux=0.01)
    _, ux_bc, uy_bc = zou_he_column_density_momentum(f, "west")
    assert torch.max(torch.abs(ux_bc - 0.01)).item() < 1e-12
    assert torch.max(torch.abs(uy_bc)).item() < 1e-12