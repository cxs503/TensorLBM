"""Transient (start-up) regression battery for the force-corrected NEBB driver.

Closes the "transient / unsteady channel start-up -- out of scope" hole in
``docs/xval_nebb_bodyforce_anchor.md`` §5 with physics assertions, not smoke
tests:

* the closed-form start-up series is checked against **an independent
  numerical solution of the Navier-Stokes equation** (implicit finite
  differences), against its two analytic limits (rest at ``t=0``, the steady
  parabola at ``t -> inf``) and against the ballistic limit ``u = g t``;
* the LBM start-from-rest run must follow the series inside 0.3 % of the steady
  peak at every sampled time, with mass conserved to machine precision;
* the slowest-mode decay rate must recover the nominal viscosity to 1 % --
  a time-domain measurement of ``nu`` that the steady benchmark cannot make;
* three discriminating (negative-control) assertions: the half-way bounce-back
  arm, the wall *convention* error and a wrong viscosity must all be rejected;
* the transient error must converge at the expected second order.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorlbm.nebb_bodyforce_startup import (
    decay_rate_report,
    fit_decay_rate,
    modal_amplitude,
    plane_poiseuille_startup,
    run_startup,
    startup_amplitude_envelope,
    startup_errors,
    steady_poiseuille,
)

G, NU, H = 2e-5, 0.1, 20.0
Y = np.linspace(0.0, H, 21)


# --------------------------------------------------------------------------- #
# 1. the closed-form referee itself
# --------------------------------------------------------------------------- #
def test_series_vanishes_at_t_zero():
    """At ``t = 0`` the series must cancel the steady parabola (rest state)."""
    u = plane_poiseuille_startup(Y, 0.0, G, NU, H, n_terms=4096)
    scale = steady_poiseuille(Y, G, NU, H).max()
    assert np.abs(u).max() / scale < 1e-5  # truncation of the 1/k^3 tail


def test_series_tends_to_steady_parabola():
    u = plane_poiseuille_startup(Y, 1e7, G, NU, H)
    ref = steady_poiseuille(Y, G, NU, H)
    assert np.abs(u - ref).max() / ref.max() < 1e-12


def test_series_ballistic_limit():
    """Short-time limit: deep in the interior ``u -> g t`` (momentum injected)."""
    u = plane_poiseuille_startup(Y, 3.0, G, NU, H, n_terms=4096)
    mid = u[10]
    assert abs(mid / (G * 3.0) - 1.0) < 5e-3


def test_series_symmetry():
    u = plane_poiseuille_startup(Y, 150.0, G, NU, H)
    # atol is needed: y = 0 and y = H carry an exact 0 vs a -1e-19 roundoff
    assert np.allclose(u, u[::-1], rtol=1e-12, atol=1e-15 * np.abs(u).max())


def test_series_against_independent_fd_solution():
    """Independent referee: backward-Euler FD solution of the same IVP.

    This is the check that makes the series usable as an external anchor: the
    series is *not* trusted because it is quoted in a book but because an
    unrelated discretisation of the PDE reproduces it.
    """
    N, dt = 401, 1.0
    y = np.linspace(0.0, H, N)
    dy = H / (N - 1)
    r = NU * dt / dy**2
    m = N - 2
    A = np.diag((1.0 + 2.0 * r) * np.ones(m)) + np.diag(-r * np.ones(m - 1), 1) + np.diag(
        -r * np.ones(m - 1), -1
    )
    u = np.zeros(N)
    targets = {50: None, 200: None}
    for step in range(1, 201):
        u[1:-1] = np.linalg.solve(A, u[1:-1] + dt * G)
        if step in targets:
            targets[step] = u.copy()
    for step, fd in targets.items():
        exact = plane_poiseuille_startup(y, step, G, NU, H)
        rel = np.abs(fd - exact).max() / exact.max()
        assert rel < 5e-3, (step, rel)


def test_envelope_matches_projected_mode():
    """The leading-mode amplitude envelope equals the sine projection of ``u_s - u``."""
    t = 120.0
    u = plane_poiseuille_startup(Y, t, G, NU, H)
    us = steady_poiseuille(Y, G, NU, H)
    assert abs(modal_amplitude(u, us, H) / startup_amplitude_envelope(t, G, NU, H) - 1.0) < 0.02


def test_fit_decay_rate_recovers_known_rate():
    t = np.arange(200.0, 1200.0, 100.0)
    a = 3e-3 * np.exp(-2.5e-3 * t)
    fit = fit_decay_rate(t, a)
    assert abs(fit["rate"] / 2.5e-3 - 1.0) < 1e-9
    assert fit["r2"] > 1.0 - 1e-12


def test_fit_decay_rate_rejects_thin_samples():
    with pytest.raises(ValueError):
        fit_decay_rate([1.0, 2.0], [1.0, 0.5])


def test_startup_errors_are_scale_normalised():
    """A zero profile at ``t = 0`` must not divide by rounding noise."""
    us = steady_poiseuille(Y, G, NU, H)
    e = startup_errors(np.zeros_like(us), plane_poiseuille_startup(Y, 0.0, G, NU, H), us.max())
    assert np.isfinite(e["max_rel_to_scale"])
    assert 0.0 < e["max_rel_to_scale"] < 1e-2


# --------------------------------------------------------------------------- #
# 2. the LBM against the series
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def reference_run():
    return run_startup(ny=21, nu=0.1)


def test_lbm_follows_startup_series(reference_run):
    res = reference_run
    worst = max(s.errors["max_rel_to_scale"] for s in res.samples)
    assert worst < 3e-3, worst
    assert res.samples[-1].errors["max_rel_to_scale"] < 1e-4
    # the velocity actually grows: first sample above zero, last near the peak
    assert res.samples[1].errors["umax_got"] > 0.0
    assert res.samples[-1].errors["umax_got"] / res.samples[-1].errors["umax_ref"] > 0.99


def test_startup_conserves_mass(reference_run):
    assert reference_run.mass_drift_rel < 1e-10


def test_decay_rate_measures_viscosity(reference_run):
    d = decay_rate_report(reference_run)
    assert d["r2"] > 0.999
    assert d["rate_rel_err"] < 1e-2, d
    assert abs(d["nu_effective"] / 0.1 - 1.0) < 1e-2


def test_half_step_time_level_visible_early(reference_run):
    """Early snapshots sit on the ``t + 1/2`` level (the half-force-step shift).

    The driver reports ``u + g/(2 rho)`` from the pre-collision moment, so at
    early times -- where ``du/dt ~ g`` -- the output is a half step ahead of the
    step counter.  Comparing against ``t + 1/2`` must cut the residual
    substantially; if it did not, the residual would be a real error, not a
    bookkeeping one.
    """
    early = [s for s in reference_run.samples if 0 < s.t <= 0.2 * reference_run.config["diffusion_time"]]
    assert early
    for s in early:
        assert s.offset_scan["+0.5"] < 0.5 * s.offset_scan["+0.0"], s.offset_scan


def test_wall_convention_is_discriminated(reference_run):
    """Using the half-way-bounce-back gap for the reference must be far worse."""
    late = reference_run.samples[-1]
    y_node = np.arange(late.profile.size, dtype=float)
    gx = reference_run.config["gx"]
    scale = steady_poiseuille(y_node, gx, NU, H).max()
    err_node = np.abs(late.profile - plane_poiseuille_startup(y_node, late.t, gx, NU, H)).max() / scale
    y_half = y_node - 0.5
    err_half = (
        np.abs(late.profile - plane_poiseuille_startup(y_half, late.t, gx, NU, H - 1.0)).max()
        / scale
    )
    assert err_half > 100.0 * err_node


def test_second_order_transient_convergence():
    """``ny = 11 -> 21 -> 41`` must shrink the error by ~4x per doubling."""
    errs = []
    for ny in (11, 21, 41):
        res = run_startup(ny=ny, nu=0.1)
        errs.append(max(s.errors["max_rel_to_scale"] for s in res.samples))
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    assert all(r > 3.0 for r in ratios), (errs, ratios)
    order = math.log2(ratios[-1])
    assert 1.8 < order < 2.2, order


def test_decay_rate_tracks_viscosity():
    for nu in (0.05, 0.2):
        res = run_startup(ny=21, nu=nu)
        d = decay_rate_report(res)
        assert d["rate_rel_err"] < 5e-3, (nu, d)


# --------------------------------------------------------------------------- #
# 3. discriminating power (negative controls on the LBM side)
# --------------------------------------------------------------------------- #
def test_halfway_bounce_back_is_rejected():
    """Without the force-corrected wall the transient error is ~2 orders larger."""
    ref = max(s.errors["max_rel_to_scale"] for s in run_startup(ny=21, nu=0.1).samples)
    bb = max(
        s.errors["max_rel_to_scale"]
        for s in run_startup(ny=21, nu=0.1, wall="halfway_bounce_back").samples
    )
    assert bb > 20.0 * ref, (bb, ref)
    assert bb > 0.03  # blows the owner stretch threshold


def test_decay_rate_rejects_wrong_viscosity():
    res = run_startup(ny=21, nu=0.1)
    d = decay_rate_report(res)
    theory = d["rate_theory"]
    assert abs(d["rate_measured"] - theory / 2.0) / (theory / 2.0) > 0.9
    assert abs(d["rate_measured"] - 2.0 * theory) / (2.0 * theory) > 0.4


def test_classic_nebb_rate_is_distinguishable():
    """The shear-reconstruction closure differs from Zou/He mainly in the *rate*.

    The steady profiles agree to well under 1 %, so only the transient reveals
    the closure difference -- which is exactly why this benchmark was needed.
    """
    ref = decay_rate_report(run_startup(ny=21, nu=0.1))
    zou = decay_rate_report(run_startup(ny=21, nu=0.1, wall="nonequilibrium_nebb"))
    assert zou["rate_rel_err"] > 5.0 * ref["rate_rel_err"]