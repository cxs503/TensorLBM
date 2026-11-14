"""Regression tests for the V4 verifier of the alpha-family linear interpolated BB.

Anchor ``XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME``.  The driver is
``xval/XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME/verify_v4.py``: force-driven
plane Poiseuille, d2Q9, production operator ``interpolated_bounce_back_2d``
(solid-side write contract), float64, analytic reference from Bouzidi et al. (2001).

Pinned here
-----------
1. **Wall position** -- the parabola fit of the numerical profile puts the no-slip
   plane within ``0.08`` cell of the requested ``delta`` for both branches
   (``delta < 1/2`` and ``delta > 1/2``) of the BFL closure.
2. **Container independence** -- an absorbing solid band (populations destroyed
   after streaming) and the periodic container of ``verify_v3_moving_wall.py`` give
   the same flow: the residual is a property of the closure, not of the box.
3. **Solid-slot bookkeeping** -- the reported velocity amplitude follows the mean
   fluid density, which is set by how the *dead storage* inside the solid cells is
   initialised.  ``init="rest"`` (solid filled with the lattice rest equilibrium,
   the convention of the V3 driver) is mass neutral; leaving the solid at zero
   costs ``>3 %`` amplitude.  This is the documented integration trap of the
   contract; the test fails if either side of it changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XV = REPO / "xval" / "XVAL-LBM-LINEAR-INTERPOLATED-BB-ALPHA-SCHEME"
sys.path.insert(0, str(XV))
sys.path.insert(0, str(REPO / "src"))

verify = pytest.importorskip("verify_v4")

STEPS = 600  # the measured residuals are stationary well before this horizon


@pytest.mark.parametrize("delta", [0.25, 0.75])
def test_wall_position_within_one_tenth_cell(delta: float) -> None:
    r = verify.run_case(delta_b=delta, closure="bfl", carrier="bgk",
                        init="rest", steps=STEPS)
    assert abs(r["w_bot_err"]) < 0.08, (r["w_bot_err"], delta)
    assert abs(r["w_top_err"]) < 0.08, (r["w_top_err"], delta)
    assert r["l2_shape"] < 0.06, r["l2_shape"]        # shape residual; measured max 0.046 (delta=0.9)
    assert abs(r["rho_fluid_mean"] - 1.0) < 0.01, r["rho_fluid_mean"]


def test_container_independence() -> None:
    """Same physics in a periodic box (V3) and in an absorbing-solid band."""
    a = verify.run_case(delta_b=0.25, closure="bfl", carrier="bgk",
                        mode="per", n_wall=2, init="rest", steps=STEPS)
    b = verify.run_case(delta_b=0.25, closure="bfl", carrier="bgk",
                        mode="iso", n_wall=3, init="rest", steps=STEPS)
    assert abs(a["l2_rel"] - b["l2_rel"]) < 1e-3, (a["l2_rel"], b["l2_rel"])
    assert abs(a["w_bot_err"] - b["w_bot_err"]) < 1e-3, (a["w_bot_err"], b["w_bot_err"])


def test_solid_slot_initialization_is_documented() -> None:
    """Rest-equilibrium solid fill is mass neutral; a zero fill is not (trap)."""
    rest = verify.run_case(delta_b=0.25, closure="bfl", carrier="bgk",
                           init="rest", steps=STEPS)
    zero = verify.run_case(delta_b=0.25, closure="bfl", carrier="bgk",
                           init="zero", steps=STEPS)
    assert abs(rest["A_over_ref"] - 1.0) < 0.01, rest["A_over_ref"]
    assert abs(zero["A_over_ref"] - 1.0) > 0.03, zero["A_over_ref"]
    # the *shape* is unaffected by the bookkeeping -- only the amplitude is
    assert zero["l2_shape"] < 0.06, zero["l2_shape"]