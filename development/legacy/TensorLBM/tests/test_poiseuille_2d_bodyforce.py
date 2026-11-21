"""Regression battery for the 2-D body-force plane Poiseuille benchmark (B14).

Case: ``benchmarks/verified/poiseuille_2d_bodyforce``.  Reference: the exact
steady solution of the incompressible momentum equation for a constant body
force, ``u(y) = g/(2 nu) y (H-y)``.  Pre-declared acceptance: every profile
metric within 2 %.

The last test is a negative control: it re-applies the *pre-fix* Guo source
term (missing the zero-sum part ``9 (c_i.u) c_ix``) and asserts that the mass
sink is detected.  If somebody reintroduces the truncated source term, the
"mass conserved" test above fails and this control explains why.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.ice_hydro.lbm_channel import (
    CX,
    CY,
    WT,
    ChannelConfig,
    D2Q9Channel,
)

G = 1.0e-5
TAU = 0.8
_CACHE: dict[tuple, dict] = {}


def _legacy_collide(self, f):  # negative control: truncated (non-zero-sum) source
    rho = f.sum(axis=0)
    ux = (f * CX[:, None, None]).sum(axis=0) / rho
    uy = (f * CY[:, None, None]).sum(axis=0) / rho
    ux_eq = ux + 0.5 * self.force
    usq = 1.5 * (ux_eq * ux_eq + uy * uy)
    omega = 1.0 / self.cfg.tau
    feq = np.empty_like(f)
    for i in range(9):
        cu = 3.0 * (CX[i] * ux_eq + CY[i] * uy)
        feq[i] = WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
    fnew = f - omega * (f - feq)
    cst = (1.0 - 0.5 * omega) * 3.0 * self.force
    for i in range(9):
        fnew[i] += cst * WT[i] * (CX[i] - ux_eq)
    return fnew


class _Legacy(D2Q9Channel):
    _collide = _legacy_collide


def run_channel(H: int, steps: int, legacy: bool = False) -> dict:
    """Steady-state Poiseuille run on ``H`` fluid rows (nx=4, periodic in x)."""
    key = (H, steps, legacy)
    if key in _CACHE:
        return _CACHE[key]
    ny = H + 4
    cfg = ChannelConfig(nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
                        cover_amplitude=0.0, cover_depth=float(ny - 3),
                        bc="no_slip", forcing="guo")
    ch =(_Legacy if legacy else D2Q9Channel)(cfg)
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size
    nu = (TAU - 0.5) / 3.0
    y = rows - 0.5
    u_ana = G / (2.0 * nu) * y * (n_fluid - y)
    u0 = np.zeros(ny)
    u0[rows] = u_ana
    ch.init_equilibrium(u0)
    acc = np.zeros(n_fluid)
    rho_acc = 0.0
    n_win = max(steps // 4, 1)
    n_acc = 0
    for step in range(1, steps + 1):
        ch.f = ch._stream_from(ch._collide(ch.f))
        if step > steps - n_win:
            rho = ch.f.sum(axis=0)
            ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
            acc += ux[0, rows]
            rho_acc += float(rho[ch.fluid].mean())
            n_acc += 1
    u = acc / n_acc
    rho_mean = rho_acc / n_acc
    res = {
        "u": u, "u_ana": u_ana, "rows": rows,
        "eps_inf": float(np.max(np.abs(u - u_ana)) / u_ana.max()),
        "eps_2": float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana)),
        "mass_drift": float(rho_mean - 1.0),
    }
    _CACHE[key] = res
    return res


def test_poiseuille_reference_case_within_2_percent():
    """Primary acceptance: profile within the pre-declared 2 % gate."""
    r = run_channel(H=16, steps=5000)
    assert r["eps_inf"] <= 0.02, r["eps_inf"]
    assert r["eps_2"] <= 0.02, r["eps_2"]
    # the archived reference case (H=32, 11000 steps) is far tighter; the
    # convergence test below pins the H-dependence instead of re-running it here.


def test_poiseuille_mass_is_conserved():
    """Guo source term must be zero-sum: no systematic density drift."""
    r = run_channel(H=16, steps=4000)
    assert abs(r["mass_drift"]) <= 1e-8, r["mass_drift"]


def test_poiseuille_is_step_invariant():
    """A genuine steady state: the score cannot depend on how long we ran."""
    a = run_channel(H=16, steps=4000)
    b = run_channel(H=16, steps=8000)
    assert abs(a["eps_inf"] - b["eps_inf"]) <= 2e-4, (a["eps_inf"], b["eps_inf"])


def test_poiseuille_second_order_convergence():
    """Half-way bounce-back is second order for a straight wall."""
    e8 = run_channel(H=8, steps=2000)["eps_inf"]
    e16 = run_channel(H=16, steps=5000)["eps_inf"]
    ratio = e8 / e16
    assert 3.0 <= ratio <= 5.0, ratio


def test_truncated_guo_source_is_detected():
    """Negative control: the pre-fix source term sinks mass (regression guard)."""
    r = run_channel(H=16, steps=4000, legacy=True)
    assert r["mass_drift"] < -1e-5, r["mass_drift"]