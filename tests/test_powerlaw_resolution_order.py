"""Regression tests for the power-law channel verification surface.

These tests lock in the findings of ``docs/powerlaw_resolution_order_20261108.md``
and replace the previous *un-asserted* verification story of the power-law channel:

1. The analytic reference must be evaluated at the **cell centres** y = 1..H.
   The helper used by ``tests/test_powerlaw.py`` evaluated it half a cell off
   (``y - 0.5``), which inflated the reported L2 error by 9-20x at H = 14-34 and
   hid the real discretisation error.  ``test_reference_abscissa_is_cell_centred``
   and ``test_half_cell_shifted_reference_inflates_error`` guard both directions.
2. The profile amplitude is anchored to the closed form
   ``U_max = n/(n+1) (fx/K)^(1/n) Hh^((n+1)/n)`` instead of being injected by hand
   (``test_newtonian_channel_matches_closed_form``).
3. The recorded, honest state of the shear-thinning (n = 0.5) channel is a *known
   deficit*: it is not steady at H = 14 even after 90 000 steps and the H = 40 error
   is ~5.7 %, i.e. outside the platform 3 % goal.  ``test_recorded_shear_thinning_deficit``
   freezes that evidence so a future fix shows up as a deliberate change.

Runtime of the live simulation here: ~1 min, single thread, device-independent (CPU).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
ARTIFACT = REPO / "artifacts" / "powerlaw_resolution_order_20261108"

import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic  # noqa: E402
from tensorlbm.powerlaw import (  # noqa: E402
    collide_powerlaw_bgk_forced,
    powerlaw_viscosity,
    strain_rate_shear_rate_2d,
    tau_from_viscosity,
)
from tensorlbm.solver import stream  # noqa: E402

NU_MIN, NU_MAX = 1e-5, 0.3
H_FIXTURE, N_FIXTURE = 14, 1.0
U_TARGET, NU_WALL = 0.03, 0.05


def _params(H: int, n: float):
    Hh = H / 2.0
    gamma_w = (n + 1.0) * U_TARGET / (n * Hh)
    K = NU_WALL * gamma_w ** (1.0 - n)
    fx = NU_WALL * gamma_w / Hh
    return K, fx


def _analytic(H: int, n: float, fx: float, K: float, shift: float = 0.0):
    """Exact profile at the cell centres y = 1..H (walls at 0.5 and H+0.5)."""
    Hh = H / 2.0
    y_c = (H + 1.0) / 2.0
    y = np.arange(1, H + 1, dtype=np.float64)
    pref = n / (n + 1.0) * (fx / K) ** (1.0 / n)
    u = pref * (Hh ** ((n + 1.0) / n) - np.abs(y - shift - y_c) ** ((n + 1.0) / n))
    return u, y, float(pref * Hh ** ((n + 1.0) / n))


def _run(H: int, n: float, K: float, fx: float, n_steps: int, tol: float = 1e-5):
    """Same scheme as tests/test_powerlaw.py (pre-streaming half-way BB walls)."""
    torch.manual_seed(0)
    ny, nx = H + 2, H
    wall = torch.zeros((ny, nx), dtype=torch.bool)
    wall[0, :] = True
    wall[-1, :] = True
    zero = torch.zeros((ny, nx))
    f = equilibrium(torch.ones((ny, nx)), zero, zero)
    tau_eff = torch.full((ny, nx), 3.0 * K + 0.5)
    f_pre = f.clone()
    col = H // 2
    hist: list[float] = []
    for step in range(1, n_steps + 1):
        f = collide_powerlaw_bgk_forced(
            f, fx, K, n, nu_min=NU_MIN, nu_max=NU_MAX, tau_field=tau_eff
        )
        f = torch.where(wall.unsqueeze(0), f_pre[OPPOSITE], f)
        f = stream(f)
        _, ux, uy = macroscopic(f)
        gamma = strain_rate_shear_rate_2d(ux, uy)
        tau_eff = tau_from_viscosity(
            powerlaw_viscosity(gamma, K, n, nu_min=NU_MIN, nu_max=NU_MAX)
        )
        f_pre = f.clone()
        if step % 200 == 0:
            hist.append(float(ux[ny // 2, col]))
            if len(hist) >= 10 and step >= 4000:
                recent = hist[-10:]
                drift = (max(recent) - min(recent)) / max(abs(sum(recent) / len(recent)), 1e-12)
                if drift < tol:
                    break
    return ux[1:-1, col].numpy().astype(np.float64)


@pytest.fixture(scope="module")
def newtonian_profile():
    K, fx = _params(H_FIXTURE, N_FIXTURE)
    # tol 1e-6 / 30k steps: the H=14 Newtonian channel needs ~25k steps to settle
    u = _run(H_FIXTURE, N_FIXTURE, K, fx, n_steps=30000, tol=1e-6)
    u_ana, y, u_max_exact = _analytic(H_FIXTURE, N_FIXTURE, fx, K)
    return u, u_ana, y, u_max_exact


def test_reference_abscissa_is_cell_centred():
    """The exact profile must be evaluated at y = 1..H, not at y - 0.5.

    With the correct abscissa the wall-adjacent fluid cell (y = 1) carries a
    finite velocity; the half-cell shifted reference puts it exactly on the wall
    and returns u = 0 there, which is what inflated the legacy L2 errors.
    """
    K, fx = _params(20, 1.5)
    u_ok, y, _ = _analytic(20, 1.5, fx, K, shift=0.0)
    u_shift, _, _ = _analytic(20, 1.5, fx, K, shift=0.5)
    assert y[0] == 1.0
    assert u_ok[0] > 0.05 * u_ok.max(), "wall-adjacent cell must not sit on the wall"
    assert u_shift[0] == pytest.approx(0.0, abs=1e-12)


def test_newtonian_channel_matches_closed_form(newtonian_profile):
    """n = 1: profile AND amplitude match the closed form to < 2 % on a 16-cell channel."""
    u, u_ana, _, u_max_exact = newtonian_profile
    l2 = float(np.linalg.norm(u - u_ana) / np.linalg.norm(u_ana))
    amp = float(u.max() - u_max_exact) / u_max_exact
    slip = float((u - u_ana).mean()) / u_max_exact
    assert l2 < 0.02, f"L2 error {l2:.4%} exceeds 2% on H={H_FIXTURE}"
    assert abs(amp) < 0.015, f"centre-line amplitude error {amp:.4%} exceeds 1.5%"
    assert abs(slip) < 0.015, f"uniform slip {slip:.4%} exceeds 1.5% of U_max"


def test_half_cell_shifted_reference_inflates_error(newtonian_profile):
    """The legacy (half-cell shifted) reference overstates the error by ~an order of magnitude."""
    u, _, _, _ = newtonian_profile
    K, fx = _params(H_FIXTURE, N_FIXTURE)
    u_ok, _, _ = _analytic(H_FIXTURE, N_FIXTURE, fx, K, shift=0.0)
    u_shift, _, _ = _analytic(H_FIXTURE, N_FIXTURE, fx, K, shift=0.5)
    e_ok = float(np.linalg.norm(u - u_ok) / np.linalg.norm(u_ok))
    e_shift = float(np.linalg.norm(u - u_shift) / np.linalg.norm(u_shift))
    assert e_shift > 8.0 * e_ok, (
        f"shifted reference should be >=8x worse: {e_shift:.4%} vs {e_ok:.4%}"
    )
    assert e_shift > 0.05  # and it is above the 5 % tolerance the legacy test used


@pytest.mark.skipif(not (ARTIFACT / "results.json").exists(), reason="study artifact absent")
def test_recorded_shear_thinning_deficit():
    """Freeze the honest state of the n = 0.5 channel (known deficit, see doc)."""
    with open(ARTIFACT / "results.json") as fh:
        res = json.load(fh)
    h40 = res.get("h40_single_grid_check", {})
    assert h40, "H=40 single-grid check missing from the study artifact"
    e_thin = h40["0.5"]["l2_rel"]
    e_newton = h40["1.0"]["l2_rel"]
    assert e_newton < 0.01, f"Newtonian H=40 error {e_newton:.4%} regressed above 1%"
    assert 0.03 < e_thin < 0.09, (
        f"n=0.5 H=40 error {e_thin:.4%} left the recorded 3-9% band: "
        "if this is a genuine improvement, update docs/powerlaw_resolution_order_20261108.md"
    )
    # the shear-thinning run is not steady: u_max keeps growing between 30k and 90k steps
    step = res.get("step_convergence_n0.5_H14", {})
    if isinstance(step, dict) and "30000" in step and "90000" in step:
        assert step["90000"]["u_max"] > 1.3 * step["30000"]["u_max"], (
            "the recorded non-convergence signature changed - re-examine the deficit"
        )