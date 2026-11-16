"""Round-3 model-matrix tests: the *boundary-condition* axis.

Round 1-2 of the matrix (`tests/test_model_matrix.py`,
`tests/test_model_matrix_plus_coverage.py`) ran only periodic boxes, so no
case could see the boundary operator.  ``wall_poiseuille`` puts the shipped
half-way bounce-back into the matrix and compares the steady channel profile
with the a-priori analytic parabola.  This file asserts the invariants that
make that comparison meaningful, plus three negative controls (a green gate is
worth nothing if it cannot go red).
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.boundary.boundaries3d import bounce_back_cells_3d, free_slip_cells_3d
from tensorlbm.d3q19 import C, W
from tensorlbm.model_matrix.cases import (
    Grid,
    _parabolic_wall_offset,
    run_wall_poiseuille,
    wall_poiseuille_exact,
)
from tensorlbm.model_matrix.registry import build_collider, get_model

TAU = 0.8
G = 1e-5


def _case(model="d3q19/bgk", ny=10, g=G):
    return run_wall_poiseuille(get_model(model), Grid(nx=4, ny=ny, nz=1), TAU, g=g, max_steps=20000)


# --------------------------------------------------------------------------- #
#  The case itself
# --------------------------------------------------------------------------- #


def test_reference_parabola_geometry():
    """The reference must vanish exactly at the half-way wall positions."""
    ny = 18
    y = torch.arange(ny, dtype=torch.float64)
    u = wall_poiseuille_exact(y, ny, G, 0.1)
    # walls sit at Y = 0.5 and Y = ny - 1.5: interpolate the parabola there
    h = 1e-6
    lo = wall_poiseuille_exact(torch.tensor([0.5 + h], dtype=torch.float64), ny, G, 0.1)
    hi = wall_poiseuille_exact(torch.tensor([ny - 1.5 - h], dtype=torch.float64), ny, G, 0.1)
    assert abs(float(lo)) < 1e-9 and abs(float(hi)) < 1e-9
    assert float(u[ny // 2]) > 0.0


def test_wall_poiseuille_gate_passes_and_is_second_order():
    """Two resolutions: the gate passes and the L2 error drops like O(H^-2)."""
    coarse = _case(ny=10)
    fine = _case(ny=18)
    assert coarse["status"] == "PASS", coarse["metrics"]
    assert fine["status"] == "PASS", fine["metrics"]
    l2c, l2f = coarse["metrics"]["rel_l2_pct"], fine["metrics"]["rel_l2_pct"]
    assert l2f < l2c
    ratio = l2c / l2f
    # H doubles 8 -> 16; a second-order wall gives ~4, a first-order one ~2.
    assert ratio > 3.0, f"wall convergence ratio {ratio:.2f} is not second order"
    # mass must be conserved to round-off: the wall reflects, it does not leak
    assert fine["metrics"]["mass_drift_rel"] < 1e-10
    # the channel is symmetric; an asymmetric reflection/stream convention shows here
    assert fine["metrics"]["symmetry_rel"] < 1e-6
    # the wall sits on the half-lattice point: |offset| -> 0 with resolution
    assert abs(fine["metrics"]["wall_offset_lattice"]) < 0.05


def test_wall_offset_shrinks_with_resolution():
    """`offset` is a wall-*position* error: it must shrink like 1/H, not stay O(1)."""
    a = _case(ny=10)["metrics"]["wall_offset_lattice"]
    b = _case(ny=18)["metrics"]["wall_offset_lattice"]
    assert 0.0 < abs(b) < abs(a)
    assert abs(a) / abs(b) > 1.5


def test_composition_matches_validated_forcing_stepper():
    """The collide -> source -> NoDynamics -> BB -> stream order is the shipped one.

    :class:`tensorlbm.rotating_frame.RotatingFrameLBM` is the repository's
    validated Guo-forced stepper (Ekman benchmarks).  At zero rotation it is a
    plain body-force channel, so the two independent compositions must agree on
    the *fluid* rows after both reach steady state.  Solid rows are excluded:
    they are not physical in either implementation.
    """
    from tensorlbm.rotating_frame import RotatingFrameLBM

    ny = 10
    ref = RotatingFrameLBM(
        shape=(1, ny, 4), tau=TAU, body_force=(G, 0.0, 0.0), solid=_solid(ny), dtype=torch.float64
    )
    ref.run_until_steady(tol=1e-9, max_steps=8000, check_every=500)
    _rho, ux, _uy, _uz = ref.fields()
    ref_profile = ux[0, :, 0]
    ours = torch.tensor(_case(ny=ny)["curves"]["u"], dtype=torch.float64)
    fl = slice(1, ny - 1)
    scale = float(ref_profile[fl].abs().max())
    diff = float((ours[fl] - ref_profile[fl]).abs().max()) / scale
    assert diff < 5e-3, f"adapters disagree on the fluid rows: rel diff {diff:.2e}"


def _solid(ny: int, shifted: bool = False) -> torch.Tensor:
    solid = torch.zeros((1, ny, 4), dtype=torch.bool)
    solid[:, 0, :] = True
    solid[:, ny - 2 if shifted else ny - 1, :] = True
    return solid


# --------------------------------------------------------------------------- #
#  Negative controls: the gate must be able to go red
# --------------------------------------------------------------------------- #


def _run_custom(ny: int, wall_fn, steps: int = 3000, g: float = G):
    """Steady channel with a *substituted* wall closure (negative controls only)."""
    from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
    from tensorlbm.solver3d import stream3d

    solid = _solid(ny)
    shape = (1, ny, 4)
    nu = (TAU - 0.5) / 3.0
    y = torch.arange(ny, dtype=torch.float64)
    u0 = wall_poiseuille_exact(y, ny, g, nu).view(1, ny, 1).expand(shape).clone()
    zero = torch.zeros(shape, dtype=torch.float64)
    f = equilibrium3d(torch.ones(shape, dtype=torch.float64), u0, zero, zero)
    coll = build_collider("d3q19/bgk", TAU)
    c = C.to(torch.float64)
    src = (3.0 * W.to(torch.float64)).view(-1, 1, 1, 1) * (c[:, 0].view(-1, 1, 1, 1) * g)
    for _ in range(steps):
        f_pre = f
        f_post = coll(f) + src
        f_post = torch.where(solid.unsqueeze(0), f_pre, f_post)
        f_post = wall_fn(f_post, solid)
        f = stream3d(f_post)
    if not bool(torch.isfinite(f).all()):
        return float("inf"), float("nan")  # a blow-up is a failure, not a pass
    _rho, ux, _uy, _uz = macroscopic3d(f)
    profile = ux[0, :, 0]
    ue = wall_poiseuille_exact(y, ny, g, nu)
    fl = slice(1, ny - 1)
    rel_l2 = 100.0 * math.sqrt(
        float(((profile[fl] - ue[fl]) ** 2).sum()) / float((ue[fl] ** 2).sum())
    )
    y_lo, y_hi = _parabolic_wall_offset(y, profile)
    return rel_l2, 0.5 * ((y_lo - 0.5) + ((ny - 1.5) - y_hi))


def test_negative_control_free_slip_wall_is_rejected():
    """A free-slip (specular) wall must fail the no-slip gate."""
    rel_l2, offset = _run_custom(10, lambda f, s: free_slip_cells_3d(f, s, axis=1) if False else free_slip_cells_3d(f, s))
    assert rel_l2 > 3.0 or abs(offset) > 0.15, (rel_l2, offset)


def test_negative_control_leaking_wall_is_detected():
    """An absorbing (mass-leaking) wall must trip the mass and profile criteria.

    This is the defect the round-1/2 periodic cases could not see at all: a wall
    that swallows populations conserves neither mass nor momentum, yet every
    periodic case stays bit-identical.
    """

    def absorbing(f, solid):
        return torch.where(solid.unsqueeze(0), torch.zeros_like(f), f)

    rel_l2, _offset = _run_custom(10, absorbing, steps=1500)
    assert rel_l2 > 3.0, f"leaking wall not detected (rel_l2 {rel_l2:.2f}%)"


def test_negative_control_wrong_reflection_table_is_detected():
    """A mis-mapped opposite table (wrong reflection pairing) must be caught."""

    def wrong_reflection(f, solid):
        return torch.where(solid.unsqueeze(0), torch.roll(f, 1, dims=0), f)

    rel_l2, _offset = _run_custom(10, wrong_reflection, steps=1500)
    assert rel_l2 > 3.0, f"wrong reflection pairing not detected (rel_l2 {rel_l2:.2f}%)"


def test_negative_control_reversed_force_is_rejected():
    """Reversed body force gives a reversed profile -> the gate must fail."""
    out = _case(ny=10, g=-G)
    assert out["status"] == "FAIL"
    assert out["metrics"]["u_center"] < 0.0
