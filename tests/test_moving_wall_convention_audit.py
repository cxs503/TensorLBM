"""Contract pins for the seven moving-wall bounce-back implementations.

Background: ``benchmarks/gaps.md`` G18 recorded (2026-08-18) that the library
has no moving-wall bounce-back and that runners re-implement the Ladd
correction inline.  The law now lives in seven places with seven contracts, and
until this file nothing pinned them to each other.  The audit in
``artifacts/moving_wall_audit_20261108/`` establishes the reference property
used here:

**Galilean fixed point.**  For a state that translates *with* the wall,
``f = f^eq(rho, u_w)``, the Ladd map ``f_q <- f_opp(q) + 6 w_q rho (c_q . u_w)``
returns ``f^eq_q`` identically (substitute ``f^eq_opp(q) = w_q[1 - 3(c_q.u_w) +
4.5(c_q.u_w)^2 - 1.5 u_w^2]``).  A correct operator is therefore *exactly*
transparent to a uniform Galilean flow: it must not modify the state and must
transfer zero net momentum.  That makes the property a zero-tolerance pin, and
it is dimension-, grid- and viscosity-independent, so the test needs no time
stepping.

Failure modes this catches
--------------------------
* flipped sign of the correction (deviation ~12 w rho c.u_w, i.e. 400% of `rho U`);
* missing density factor (deviation appears at ``rho != 1``, invisible at rho=1);
* wall velocity read on the wrong cell (fluid-side implementations read ``u_w``
  at the *fluid* node, not at the body node);
* mask ignored / hard-coded wall row (the correction must follow the mask the
  caller passes, not a fixed row).

Run:  PYTHONPATH=src python3 -m pytest tests/test_moving_wall_convention_audit.py -q
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.boundary.boundaries import bounce_back_cells  # noqa: F401  (contract doc)
from tensorlbm.cac_moving_body import moving_wall_bounce, moving_wall_bounce_full
from tensorlbm.d2q9 import C as C2
from tensorlbm.d2q9 import OPPOSITE as OPP2
from tensorlbm.d2q9 import W as W2
from tensorlbm.d2q9 import equilibrium as eq2
from tensorlbm.d3q19 import C as C3
from tensorlbm.d3q19 import equilibrium3d as eq3
from tensorlbm.herschel_bulkley_lbm import moving_wall_bounce_back_2d
from tensorlbm.multiphase_water_entry import _moving_wall_bb_2d
from tensorlbm.propeller_benchmark import moving_wall_bounce_back_3d
from tensorlbm.rotating_cylinder import moving_wall_bounce_back
from tensorlbm.yield_stress_channel import _moving_wall_bounce_back as ys_mwbb

DTYPE = torch.float64
U = 0.02
NY = 12
NX = 4

# Tolerance: the property is exact in exact arithmetic, so anything above
# round-off means a convention error.  ``multiphase_water_entry._moving_wall_bb_2d``
# builds its lattice vectors in float32 (``C2.to(device).float()``), which lifts
# the noise floor to ~5e-9; 1e-6 keeps four orders of margin below the smallest
# physical defect (missing rho at rho=1.2 -> 1.7e-2 of rho U).
TOL = 1e-6


# ---------------------------------------------------------------------------
# state construction / adapters (one per shipped contract)
# ---------------------------------------------------------------------------


def _galilean_2d(rho0: float, wall_rows: int = 1):
    solid = torch.zeros((NY, NX), dtype=torch.bool)
    solid[:wall_rows, :] = True
    solid[NY - wall_rows:, :] = True
    moving = torch.zeros_like(solid)
    moving[:wall_rows, :] = True
    f = eq2(torch.full((NY, NX), rho0, dtype=DTYPE),
            torch.full((NY, NX), U, dtype=DTYPE),
            torch.zeros((NY, NX), dtype=DTYPE))
    return f, solid, moving


def _galilean_3d(rho0: float, wall_rows: int = 1, n: int = 4):
    solid = torch.zeros((NY, n), dtype=torch.bool)
    solid[:wall_rows, :] = True
    solid[NY - wall_rows:, :] = True
    moving = torch.zeros_like(solid)
    moving[:wall_rows, :] = True
    z = torch.full((n, NY, n), 0.0, dtype=DTYPE)
    f = eq3(torch.full((n, NY, n), rho0, dtype=DTYPE),
            torch.full((n, NY, n), U, dtype=DTYPE), z, z)
    return f, solid, moving


def _rotating_cylinder(f, mask, solid, u):
    return moving_wall_bounce_back(f, mask, mask.to(f.dtype) * u, torch.zeros_like(mask, dtype=f.dtype))


def _herschel(f, mask, solid, u):
    return moving_wall_bounce_back_2d(f, mask, u, 0.0)


def _water_entry(f, mask, solid, u):
    return _moving_wall_bb_2d(f, f, mask, solid, u, 0.0)


def _propeller(f, mask, solid, u):
    nz = f.shape[3]
    m3 = mask.unsqueeze(0).expand(nz, *mask.shape).contiguous()
    ux = m3.to(f.dtype) * u
    return moving_wall_bounce_back_3d(f, m3, ux, torch.zeros_like(ux), torch.zeros_like(ux))


def _cac_full(f, mask, solid, u):
    nz = f.shape[3]
    return moving_wall_bounce_full(
        f,
        body=mask.unsqueeze(0).expand(nz, *mask.shape).contiguous(),
        fluid=(~solid).unsqueeze(0).expand(nz, *solid.shape).contiguous(),
        u_wall=(u, 0.0, 0.0),
        density=f.sum(dim=0),
    )


def _cac_side(f, mask, solid, u):
    nz = f.shape[3]
    return moving_wall_bounce(
        f, f,
        body=mask.unsqueeze(0).expand(nz, *mask.shape).contiguous(),
        fluid=(~solid).unsqueeze(0).expand(nz, *solid.shape).contiguous(),
        u_wall=(u, 0.0, 0.0),
        density=f.sum(dim=0),
    )


def _ys(f, mask, solid, u):
    """yield_stress_channel runner has the moving wall on the **top** row.

    It also historically hard-codes that row (``top[-1, :]``) instead of using
    the mask it was handed; ``moving_mask`` was added so the operator follows
    the caller's mask.  Both spellings must reproduce the Galilean state.
    """
    if mask[0].any():
        return ys_mwbb(f, mask, (u, 0.0), moving_mask=mask)
    return ys_mwbb(f, mask, (u, 0.0))


UNITS_2D = {
    "rotating_cylinder.moving_wall_bounce_back": _rotating_cylinder,
    "herschel_bulkley_lbm.moving_wall_bounce_back_2d": _herschel,
    "multiphase_water_entry._moving_wall_bb_2d": _water_entry,
    "yield_stress_channel._moving_wall_bounce_back": _ys,
}
UNITS_3D = {
    "propeller_benchmark.moving_wall_bounce_back_3d": _propeller,
    "cac_moving_body.moving_wall_bounce_full": _cac_full,
    "cac_moving_body.moving_wall_bounce": _cac_side,
}


def _dev2d(apply, rho0, wall_rows=1):
    f, solid, moving = _galilean_2d(rho0, wall_rows)
    out = apply(f, moving, solid, U)
    return float((out - f).abs().max()) / (rho0 * U)


def _dev3d(apply, rho0, wall_rows=1):
    f, solid, moving = _galilean_3d(rho0, wall_rows)
    out = apply(f, moving, solid, U)
    return float((out - f).abs().max()) / (rho0 * U)


# ---------------------------------------------------------------------------
# negative control: the pin must be able to fail
# ---------------------------------------------------------------------------


def _flipped_oracle(f, mask, solid, u):
    """Solid-side Ladd with the correction sign reversed (the historical bug)."""
    opp = OPP2.to(f.device)
    c = C2.to(dtype=f.dtype)
    w = W2.to(dtype=f.dtype)
    rho = f.sum(dim=0)
    corr = -6.0 * w.view(9, 1, 1) * rho.view(1, *rho.shape) * (c[:, 0].view(9, 1, 1) * u)
    return torch.where(mask.view(1, *mask.shape), f[opp] + corr, f)


def test_negative_control_sign_flip_is_detected():
    dev = _dev2d(_flipped_oracle, 1.0)
    assert dev > 1.0, f"sign flip should deviate by O(1) of rhoU, got {dev:.3e}"


# ---------------------------------------------------------------------------
# the pins
# ---------------------------------------------------------------------------


def test_all_2d_implementations_are_galilean_transparent():
    bad = {name: _dev2d(fn, 1.0) for name, fn in UNITS_2D.items()}
    assert all(v < TOL for v in bad.values()), {k: f"{v:.3e}" for k, v in bad.items()}


# ``cac_moving_body.moving_wall_bounce_full`` selects its correction cells with
# ``surface = body & links_from(fluid)``, i.e. the links whose source is the
# *fluid* -- the correction lands on the populations streaming INTO the body
# instead of the ones returning from it.  On a 1-cell wall the two sets are
# disjoint, so the wall is only half-corrected: 33.3% of ``rho U`` in the
# Galilean test (a static wall would be ~0).  Reported, not silently accepted:
# see docs/moving_wall_convention_audit_20261108.md §5.
def test_cac_moving_wall_bounce_full_is_galilean_transparent():
    """F1 (audit 2026-11-08) fixed: the Ladd term now lands on the returning
    links (``x_b + c_q`` in fluid), so the operator is Galilean-transparent."""
    assert _dev3d(_cac_full, 1.0) < TOL


def test_all_3d_implementations_are_galilean_transparent():
    bad = {name: _dev3d(fn, 1.0) for name, fn in UNITS_3D.items()}
    assert all(v < TOL for v in bad.values()), {k: f"{v:.3e}" for k, v in bad.items()}


def test_density_factor_is_honoured_at_rho_not_one():
    """A correction written without ``rho`` is exactly right at rho=1 and wrong everywhere else.

    ``yield_stress_channel`` shipped ``6 w (c.u_w)`` (no density) — invisible in
    a Couette check at rho=1, 33% of the correction at rho=1.2.
    """
    bad = {name: _dev2d(fn, 1.2) for name, fn in UNITS_2D.items()}
    assert all(v < TOL for v in bad.values()), {k: f"{v:.3e}" for k, v in bad.items()}
    bad3 = {name: _dev3d(fn, 1.2) for name, fn in UNITS_3D.items()}
    assert all(v < TOL for v in bad3.values()), {k: f"{v:.3e}" for k, v in bad3.items()}


def test_yield_stress_operator_follows_the_passed_mask():
    """The correction must belong to the mask the caller passes, not to row -1.

    With the moving wall on the **bottom** row the historical operator applied
    no correction at all (it hard-coded ``top[-1, :]``), i.e. it silently
    degenerated to a static wall for any geometry whose moving wall is not the
    last row.
    """
    dev = _dev2d(_ys, 1.0)
    assert dev < TOL, f"correction did not follow the mask: dev={dev:.3e}"


def test_static_wall_is_untouched_by_moving_operators():
    """``u_w = 0`` must reproduce plain bounce-back bit for bit.

    Only the solid-side operators are compared: the fluid-side ones write into
    the *fluid* neighbour and are not the same object as a swap on the wall.
    """
    f, solid, moving = _galilean_2d(1.0)
    ref = bounce_back_cells(f, moving)
    for name in ("rotating_cylinder.moving_wall_bounce_back",
                 "herschel_bulkley_lbm.moving_wall_bounce_back_2d",
                 "yield_stress_channel._moving_wall_bounce_back"):
        got = UNITS_2D[name](f, moving, solid, 0.0)
        assert (got - ref).abs().max() < 1e-15, name