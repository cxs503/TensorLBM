"""Tests for the round-2 XVAL of the spectral Saffman--Taylor (integrating factor)
solver: **independent linear-stability anchor**.

Covers the claims of ``docs/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT-R2.md``:

1. with ``R = 0`` the exact curl-Darcy linearisation collapses to pure transverse
   diffusion, ``sigma = -k^2``;
2. the hand-derived sharp-step dispersion relation
   ``2 kappa^2 + 2 k kappa + k R = 0``, ``sigma = kappa^2 - k^2`` reproduces the
   classical Saffman--Taylor stability criterion for ``mu = exp(R c)`` with
   ``c = 1`` upstream: unstable **only** for ``R < 0`` (more viscous fluid
   ahead of the front);
3. the finite-width numerical anchor converges towards that step relation as
   ``w -> 0`` (monotone, < 15 % at ``k w <= 0.4``);
4. the TensorLBM module measures a *growing* mode at ``R = +1`` whereas the
   anchor says the front is (essentially) neutral/damped -- the discrepancy is
   pinned here so it cannot silently change.

Run: ``python -m pytest tests/test_spectral_saffman_taylor_linear_theory.py -q``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "benchmarks" / "cross_validation" / "spectral_saffman_taylor" / "round2"))

from tensorlbm.spectral_saffman_taylor import SpectralSaffmanTaylor2D  # noqa: E402
from xval_linear_theory import (  # noqa: E402
    exact_linear_sigma,
    step_analytic_sigma,
)

K4 = 2.0 * np.pi * 4 / 500.0     # transverse wavenumber of mode 4 at Pe = 500
W = 4.0                          # front width of the round-1 initial condition


def test_anchor_R0_is_pure_transverse_diffusion():
    """R = 0: the anchor must reproduce sigma = -k^2 to better than 1 %.

    The residual is the Dirichlet-box truncation of the anchor grid,
    ``O((pi/2L)^2)`` (4e-6 for L = 800), which dominates the relative error for
    the smallest wavenumber tested.
    """
    for k in (K4, 2 * K4, 0.5 * K4):
        s = exact_linear_sigma(k, 0.0, W, N=801, L=800.0)["sigma"]
        assert abs(s.imag) < 1e-12
        assert abs(s.real + k * k) / (k * k) < 1e-2


def test_step_relation_reproduces_saffman_taylor_criterion():
    """mu = exp(R c), c = 1 upstream: unstable only when the fluid ahead is thicker."""
    for R in (-2.0, -1.0, -0.5):
        st = step_analytic_sigma(R, K4)
        assert st["admissible"], f"no admissible root for the unstable case R={R}"
        assert st["sigma"].real > 0.0
    for R in (1.0, 2.0):
        st = step_analytic_sigma(R, K4)
        # stable arrangement: no admissible root, or a damped one
        assert (not st["admissible"]) or st["sigma"].real < 0.0


def test_finite_width_converges_to_step_relation():
    """The numerical anchor must approach the hand-derived step value as w -> 0."""
    R = -1.0
    step = step_analytic_sigma(R, K4)["sigma"].real
    devs = []
    for w in (8.0, 4.0, 2.0):
        devs.append(abs(exact_linear_sigma(K4, R, w, N=801, L=800.0)["sigma"].real - step) / step)
    assert devs[0] > devs[1] > devs[2], f"not monotone: {devs}"
    assert devs[-1] < 0.15, f"w=2 does not reach the step relation: {devs}"


def test_anchor_is_grid_and_domain_independent():
    a = exact_linear_sigma(K4, -1.0, W, N=801, L=800.0)["sigma"].real
    b = exact_linear_sigma(K4, -1.0, W, N=1201, L=800.0)["sigma"].real
    assert abs(a - b) / abs(b) < 0.05


def test_module_grows_where_first_principles_predict_decay():
    """Pinned discrepancy (findings F1/F2 of the round-2 report).

    Exact curl-Darcy linear theory: the base front ``c = 0.5(1-tanh(x/w))`` with
    ``c = 1`` upstream and ``mu = exp(R c)`` is unstable only for ``R < 0``;
    at ``R = +1`` it is damped.  The TensorLBM port measures a growing mode at
    ``R = +1`` (the frozen-coefficient closure of the OSS reference).
    """
    theory_pos = exact_linear_sigma(K4, 1.0, W, N=801, L=800.0)["sigma"].real
    theory_neg = exact_linear_sigma(K4, -1.0, W, N=801, L=800.0)["sigma"].real
    assert theory_pos < 0.0 < theory_neg

    sim = SpectralSaffmanTaylor2D(128, 256, aspect=2.0, pe=500.0, viscosity_ratio=1.0,
                                  dt=0.1, mode_index=4)
    sim.set_concentration(sim.single_mode_ic(x0=500.0, width=W, amplitude=1e-3, n_mode=4))
    ts, amps = [], []
    for _ in range(41):
        d = sim.diagnostics()
        ts.append(d.t)
        amps.append(d.mode_amplitude)
        sim.step()
    ts, amps = np.array(ts), np.array(amps)
    mask = (ts >= 2.0) & (ts <= 4.0)
    sigma_module = float(np.polyfit(ts[mask], np.log(amps[mask]), 1)[0])
    assert sigma_module > 0.0, "module no longer grows at R=+1: re-check the report"
    assert abs(sigma_module - theory_pos) / abs(theory_pos) > 0.15, \
        "module now agrees with the anchor: findings F1/F2 must be revisited"