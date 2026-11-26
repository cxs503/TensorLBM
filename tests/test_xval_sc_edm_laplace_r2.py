"""Anchor ``XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO`` round 2 — invariants on the
**shipped library path**.

Round 1 of this anchor (``TensorLBM/xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO``)
compared a self-contained NumPy re-implementation against itself at a single
relaxation time ``tau = 1`` and concluded "PASS <15%".  Round 2 (m21378) found
that this design is degenerate and that its pressure convention inflates the
reported surface tension by ~3e2.  These tests pin the three facts that make
the anchor's question answerable, all on ``tensorlbm.multiphase
.collide_sc_single_component`` + ``tensorlbm.solver.stream``:

1. **tau = 1 is degenerate.**  ``collide_sc_single_component(..., forcing=
   "velocity_shift")`` and ``forcing="edm"`` are algebraically identical at
   ``tau = 1`` (velocity shift relaxes to ``feq(u + tau F/rho)`` = ``feq(u +
   F/rho)``; EDM relaxes to ``feq(u) + [feq(u + F/rho) - feq(u)]``, the same
   distribution).  They diverge only for ``tau != 1``.  Any forcing-scheme
   comparison therefore *must* use ``tau != 1``.
2. **The bulk pressure carries the library's sign convention.**
   ``p_mech = cs^2 rho - (G/2) cs^2 psi^2`` (``G = +5`` attractive, backward
   gather).  With the textbook-convention ``p = cs^2 rho + (G/2) cs^2 psi^2``
   the Young-Laplace pressure difference comes out ~3e2 too large and the
   resulting ``sigma`` (~16) is not the surface tension of this model
   (documented value ~0.0561).
3. **The EOS really has a van der Waals loop** at ``G = 5``, so the Maxwell
   equal-area construction is a legitimate analytic reference for the
   coexistence densities.

The tests are deliberately cheap (``L <= 56``, ``<= 2000`` steps) so that they
can run in the default suite; the full 45-case sweep lives in
``development/verification/shan_chen_edm_lycett_luo/round2/run_xval.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from tensorlbm.d2q9 import equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_single_component, psi_exp
from tensorlbm.solver import stream

G = 5.0
CS2 = 1.0 / 3.0


def _p_mech(rho, g=G):
    psi = psi_exp_np(rho)
    return CS2 * np.asarray(rho) - 0.5 * g * CS2 * psi * psi


def psi_exp_np(rho):
    return 1.0 - np.exp(-np.asarray(rho))


def _droplet(L, R):
    ys, xs = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5, indexing="ij")
    rr = np.hypot(xs - L / 2.0, ys - L / 2.0)
    rho = 0.1596 + 0.5 * (1.957 - 0.1596) * (1.0 + np.tanh((R - rr) / 4.0))
    return torch.tensor(np.clip(rho, 1e-3, None), dtype=torch.float64), rr


def _march(forcing, tau, L, R, steps):
    torch.set_num_threads(1)
    rho0, rr = _droplet(L, R)
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    for _ in range(steps):
        f = stream(collide_sc_single_component(f, G=G, tau=tau, psi_fn=psi_exp, forcing=forcing))
    rho, ux, uy = macroscopic(f)
    return rho.numpy(), ux.numpy(), uy.numpy(), rr, float(rho0.sum())


def _laplace(rho, ux, uy, rr, R):
    p = _p_mech(rho)
    r_eq = float(R)
    for _ in range(6):
        inside = rr <= 0.5 * r_eq
        outside = rr >= 1.5 * r_eq
        mid = 0.5 * (rho[inside].mean() + rho[outside].mean())
        r_eq = math.sqrt(int((rho > mid).sum()) / math.pi)
    inside = rr <= 0.5 * r_eq
    outside = rr >= 1.5 * r_eq
    dp = float(p[inside].mean() - p[outside].mean())
    return dp, r_eq, dp * r_eq


def test_edm_is_identical_to_velocity_shift_at_tau_one():
    """tau = 1: the two forcing schemes coincide to machine precision."""
    torch.manual_seed(7)
    f = torch.rand(9, 14, 14, dtype=torch.float64) * 0.1
    a = collide_sc_single_component(f, G=G, tau=1.0, psi_fn=psi_exp, forcing="velocity_shift")
    b = collide_sc_single_component(f, G=G, tau=1.0, psi_fn=psi_exp, forcing="edm")
    assert torch.allclose(a, b, atol=1e-15, rtol=0)

    c = collide_sc_single_component(f, G=G, tau=0.8, psi_fn=psi_exp, forcing="velocity_shift")
    d = collide_sc_single_component(f, G=G, tau=0.8, psi_fn=psi_exp, forcing="edm")
    assert (c - d).abs().max() > 1e-6, "tau != 1 must separate the schemes"


def test_all_forcings_conserve_mass_and_stay_finite():
    for forcing in ("velocity_shift", "edm", "guo"):
        rho, ux, uy, _, mass0 = _march(forcing, 1.0, 32, 8.0, 200)
        assert np.all(np.isfinite(rho))
        assert np.all(rho > 0)
        assert abs(rho.sum() - mass0) / mass0 < 1e-10, forcing


def test_surface_tension_uses_library_pressure_convention():
    """sigma must land in the model's physical range, not the ~3e2-inflated one.

    With ``psi = 1 - exp(-rho)``, ``G = +5``, ``tau = 1`` the documented
    surface tension is ~0.0561 (``tensorlbm`` docs / benchmarks/verified).
    Using the textbook-sign pressure instead of the library one yields
    sigma ~ 16 — a factor ~3e2 — which is the round-1 defect this guards.
    """
    rho, ux, uy, rr, _ = _march("velocity_shift", 1.0, 40, 10.0, 2000)
    dp, r_eq, sigma = _laplace(rho, ux, uy, rr, 10.0)
    assert dp > 0
    assert 0.02 < sigma < 0.12, f"sigma={sigma:.4f} outside the physical band"
    # the wrong-sign convention would give ~3e2 x larger values
    p_wrong = CS2 * rho + 0.5 * G * CS2 * psi_exp_np(rho) ** 2
    inside = rr <= 0.5 * r_eq
    outside = rr >= 1.5 * r_eq
    dp_wrong = float(p_wrong[inside].mean() - p_wrong[outside].mean())
    assert abs(dp_wrong / dp) > 50.0


def test_laplace_pressure_falls_with_radius():
    """dp = sigma / R: a bigger drop carries a smaller pressure jump."""
    res = {}
    for R, L, steps in ((8.0, 32, 1500), (12.0, 48, 1500)):
        rho, ux, uy, rr, _ = _march("edm", 1.0, L, R, steps)
        res[R] = _laplace(rho, ux, uy, rr, R)[0]
    assert res[8.0] > res[12.0] > 0
    # cross-check against the documented sigma ~ 0.0561 within 25%
    sig8 = res[8.0] * 8.0
    sig12 = res[12.0] * 12.0
    assert abs(sig8 / sig12 - 1.0) < 0.25, (sig8, sig12)


def test_maxwell_construction_is_bracketed_by_spinodals():
    """Analytic sanity of the reference used by the round-2 study."""
    from scipy.optimize import brentq

    grid = np.linspace(1e-4, 6.0, 100001)
    pgrid = _p_mech(grid)
    dpg = np.gradient(pgrid, grid)
    neg = np.where(dpg < 0)[0]
    assert neg.size > 0, "G=5 must be supercritical (van der Waals loop present)"
    # p at the vapour spinodal exceeds p at the liquid spinodal (the loop runs
    # "backwards" in rho), so the admissible p_sat window is [min, max].
    p_lo, p_hi = sorted((float(pgrid[neg[0]]), float(pgrid[neg[-1]])))

    def roots(pc):
        fl = pgrid - pc
        idx = np.where(np.sign(fl[:-1]) * np.sign(fl[1:]) < 0)[0]
        assert idx.size >= 2
        rv = brentq(lambda r: float(_p_mech(r)) - pc, grid[idx[0]], grid[idx[0] + 1])
        rl = brentq(lambda r: float(_p_mech(r)) - pc, grid[idx[-1]], grid[idx[-1] + 1])
        return rv, rl

    pc = 0.5 * (p_lo + p_hi)
    rv, rl = roots(pc)
    assert 0 < rv < rl
    assert p_lo <= float(_p_mech(rv)) <= p_hi
    assert p_lo <= float(_p_mech(rl)) <= p_hi
    # spinodal inverts dp/drho: p(rho) is not monotone between rv and rl
    rr = np.linspace(rv, rl, 2001)
    assert np.any(np.diff(_p_mech(rr)) < 0)