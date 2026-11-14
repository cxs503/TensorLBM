"""Flat-interface coexistence invariant for the SCMP Shan-Chen solver.

Adversarial-review test (round R1) for
``XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION``.

A planar (flat) liquid-vapour interface has zero Laplace pressure, so the SC
equation of state must return the *same* bulk pressure on both sides:

    p_l - p_v -> 0        (mechanical equilibrium, dp = sigma/R with R -> inf)

and the plateau densities it selects are the *coexistence* densities of the
discretised scheme.  This is the reference state the droplet (curved) protocol
must be compared against; the test checks that the library keeps this invariant
for the two first-order-equivalent forcings (velocity_shift, edm).

The test is deliberately small (128 x 4, a few thousand steps) so it runs in
a couple of seconds; the production sweep lives in
``xval/XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION/verify_r1/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[1]
XV = REPO / "xval" / "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(XV))

import ref_sc as REF  # noqa: E402

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.multiphase import collide_sc_single_component, psi_exp  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402

G_LIB = 5.0  # library convention (attraction for G > 0)
G_EFF = -5.0  # textbook convention used by the EOS reference
TAU = 1.0


def _init_flat(nx: int, ny: int, rho_l: float, rho_v: float, w: float, half: float):
    x = np.arange(nx) + 0.5
    c = nx / 2.0
    prof = rho_v + 0.5 * (rho_l - rho_v) * (
        np.tanh((x - (c - half)) / w) - np.tanh((x - (c + half)) / w)
    )
    rho = np.repeat(prof[None, :], ny, axis=0)
    return torch.tensor(np.clip(rho, 1e-3, None), dtype=torch.float32)


def _run(forcing: str, nx: int = 128, ny: int = 4, half: float = 20.0, steps: int = 3000):
    rho_l0, rho_v0 = 1.957, 0.1596
    rho0 = _init_flat(nx, ny, rho_l0, rho_v0, 4.0, half)
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))
    for _ in range(steps):
        f = stream(
            collide_sc_single_component(f, G=G_LIB, tau=TAU, psi_fn=psi_exp, forcing=forcing)
        )
    from tensorlbm.d2q9 import macroscopic

    rho_t, ux, uy = macroscopic(f)
    rho = rho_t.numpy()
    x = np.arange(nx) + 0.5
    c = nx / 2.0
    liq = np.abs(x - c) <= half - 10.0
    vap = (x < c - half - 30.0) | (x > c + half + 30.0)
    p = REF.eos_pressure(rho, G_EFF)
    return {
        "rho_l": float(rho[:, liq].mean()),
        "rho_v": float(rho[:, vap].mean()),
        "p_l": float(p[:, liq].mean()),
        "p_v": float(p[:, vap].mean()),
        "u_max": float(np.sqrt(ux.numpy() ** 2 + uy.numpy() ** 2).max()),
    }


@pytest.mark.parametrize("forcing", ["velocity_shift", "edm"])
def test_flat_interface_mechanical_equilibrium(forcing):
    """p_l == p_v across a planar SC interface (dp must vanish as R -> inf)."""
    r = _run(forcing)
    rel = abs(r["p_l"] - r["p_v"]) / abs(r["p_v"])
    assert rel < 0.05, (
        f"{forcing}: flat-interface pressure jump {r['p_l'] - r['p_v']:+.3e} "
        f"({100 * rel:.2f}% of p_v) is too large for a planar interface "
        f"(rho_l={r['rho_l']:.4f}, rho_v={r['rho_v']:.4f})"
    )
    # the plateau densities must bracket the analytic Maxwell coexistence state
    rho_v_mx, rho_l_mx, _ = REF.maxwell_coexistence(G_EFF)
    assert abs(r["rho_l"] - rho_l_mx) / rho_l_mx < 0.10
    assert abs(r["rho_v"] - rho_v_mx) / rho_v_mx < 0.20


def test_flat_interface_first_order_equivalence():
    """velocity_shift and edm agree (both are O(F) schemes) on the flat state."""
    a = _run("velocity_shift")
    b = _run("edm")
    assert abs(a["rho_l"] - b["rho_l"]) / a["rho_l"] < 0.01
    assert abs(a["rho_v"] - b["rho_v"]) / a["rho_v"] < 0.05