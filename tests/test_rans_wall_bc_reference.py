"""Analytic-reference verification for ``tensorlbm.turbulence.rans_wall_bc``.

Verification-debt closure (G21 orphan-module gap fill, m6600).

``rans_wall_bc`` is the RANS (k-ε / k-ω) near-wall boundary-value setter that
``wall_model``'s log-law ``u_tau`` feeds into.  It had **zero direct tests**
(only ``wall_model`` itself was exercised).  This suite anchors the two
implemented wall values to their closed-form OpenFOAM-equivalent references and
to the log-law eddy-viscosity invariant, plus the near-wall mask logic.

Closed-form references (lattice units, C_μ=0.09, κ=0.41):

    k_wall   = u_tau² / √C_μ
    ε_wall   = u_tau³ / (κ · y)

Physical anchor (the reason these two values are the *right* wall BC): in the
log-law region the eddy viscosity is ``ν_t = κ·u_tau·y``.  The k-ε model
defines ``ν_t = C_μ·k²/ε``.  Substituting the two wall values must therefore
reproduce the log-law eddy viscosity *exactly*:

    C_μ · k_wall² / ε_wall  ==  κ · u_tau · y        (invariant I1)

That invariant is the strongest single check: it couples the two wall values to
the log-law and would break if either constant (C_μ or κ) or either formula were
wrong.  A negative control (κ→0.40) proves the invariant has discriminating
power.

All cases are tiny tensors on CPU/float64 — no solver run, no cluster.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm.turbulence import rans_wall_bc as rwb

# Mirror the module's private constants so the tests are independent of a
# silent constant edit (a change to C_MU/KAPPA must fail these, not pass silently).
C_MU = 0.09
KAPPA = 0.41

# A physically meaningful friction velocity / wall distance (lattice units).
U_TAU = 0.05
Y_VAL = 0.5
NU = 5e-4  # kinematic viscosity, chosen so y+ = y*u_tau/nu = 50 (log-law region)


# ── helpers ────────────────────────────────────────────────────────────────
def _fields(nz=4, ny=6, nx=5, u_tau=U_TAU):
    """Build a small 3-D field set with a single near-wall plane (j==1)."""
    k = torch.full((nz, ny, nx), 0.123, dtype=torch.float64)
    eps = torch.full((nz, ny, nx), 0.0456, dtype=torch.float64)
    ut = torch.full((nz, ny, nx), u_tau, dtype=torch.float64)
    near = torch.zeros((nz, ny, nx), dtype=torch.bool)
    near[:, 1, :] = True  # first off-wall cell
    return k, eps, ut, near


def _yplus(u_tau: float, y: float, nu: float) -> float:
    return y * u_tau / nu


# ── I1: log-law eddy-viscosity invariant ───────────────────────────────────
def test_i1_wall_bc_reproduces_loglaw_eddy_viscosity():
    """C_μ·k_wall²/ε_wall must equal κ·u_tau·y (log-law ν_t) to machine precision."""
    k, eps, ut, near = _fields()
    rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
    # read back only the masked (near-wall) cells
    kw = k[:, 1, :]
    ew = eps[:, 1, :]
    nu_t_model = C_MU * kw * kw / ew
    nu_t_loglaw = KAPPA * U_TAU * Y_VAL
    rel = (nu_t_model - nu_t_loglaw).abs().max().item() / nu_t_loglaw
    assert rel < 1e-12, f"I1 violated: rel={rel:.3e}"


def test_i1_negative_control_wrong_kappa_breaks_invariant():
    """If κ were 0.40 (not 0.41), I1 must break by ~2.4% — proves the test bites."""
    k, eps, ut, near = _fields()
    rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
    kw = k[:, 1, :]
    ew = eps[:, 1, :]
    # recompute the invariant but with the *wrong* κ in the log-law side
    nu_t_model = C_MU * kw * kw / ew
    nu_t_wrong = 0.40 * U_TAU * Y_VAL
    rel = (nu_t_model - nu_t_wrong).abs().max().item() / nu_t_wrong
    assert rel > 0.02, f"negative control failed to break: rel={rel:.3e}"


# ── exact closed-form values ────────────────────────────────────────────────
def test_k_wall_closed_form():
    k, eps, ut, near = _fields()
    rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
    expect = U_TAU**2 / math.sqrt(C_MU)
    got = k[:, 1, :].flatten()
    assert torch.allclose(got, torch.full_like(got, expect), rtol=0, atol=1e-15)


def test_epsilon_wall_closed_form():
    k, eps, ut, near = _fields()
    rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
    expect = U_TAU**3 / (KAPPA * Y_VAL)
    got = eps[:, 1, :].flatten()
    assert torch.allclose(got, torch.full_like(got, expect), rtol=0, atol=1e-18)


def test_wall_values_scale_with_utau():
    """k ∝ u_tau², ε ∝ u_tau³ (scaling sanity across a u_tau sweep)."""
    for u in (0.01, 0.05, 0.2):
        k, eps, ut, near = _fields(u_tau=u)
        rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
        k_exp = torch.full((4, 5), u**2 / math.sqrt(C_MU), dtype=torch.float64)
        e_exp = torch.full((4, 5), u**3 / (KAPPA * Y_VAL), dtype=torch.float64)
        assert torch.allclose(k[:, 1, :], k_exp), f"k scaling wrong at u_tau={u}"
        assert torch.allclose(eps[:, 1, :], e_exp), f"eps scaling wrong at u_tau={u}"


# ── mask selectivity ────────────────────────────────────────────────────────
def test_only_near_wall_cells_modified():
    k, eps, ut, near = _fields()
    k0 = k.clone()
    e0 = eps.clone()
    rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
    # interior (non-near-wall) cells must be bit-identical
    interior = ~near
    assert torch.equal(k[interior], k0[interior]), "interior k was modified"
    assert torch.equal(eps[interior], e0[interior]), "interior eps was modified"
    # near-wall cells must actually change
    assert not torch.equal(k[:, 1, :], k0[:, 1, :]), "near-wall k unchanged"


def test_zero_utau_is_noop():
    """u_tau <= 1e-12 must leave the field untouched (mask guard)."""
    k, eps, ut, near = _fields(u_tau=0.0)
    k0 = k.clone()
    e0 = eps.clone()
    rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
    assert torch.equal(k, k0), "zero u_tau modified k"
    assert torch.equal(eps, e0), "zero u_tau modified eps"


def test_tiny_utau_hits_floor():
    """Tiny-but-nonzero u_tau: k clamps to 1e-8, ε clamps to 1e-12."""
    k, eps, ut, near = _fields(u_tau=1e-6)
    rwb.set_rans_wall_bc_k_epsilon(k, eps, ut, near, NU, y_val=Y_VAL)
    assert torch.allclose(k[:, 1, :], torch.full_like(k[:, 1, :], 1e-8))
    assert torch.allclose(eps[:, 1, :], torch.full_like(eps[:, 1, :], 1e-12))


# ── compute_near_wall_from_solid ────────────────────────────────────────────
def test_near_wall_flat_wall():
    """A solid slab at j==0 → near-wall rows are j==1 (physical adjacency) and
    j==5 (periodic wrap of the roll-based stencil).  The roll convention is
    correct for periodic domains; in a non-periodic domain the opposite face is
    over-marked — documented as a caveat in the report §5."""
    solid = torch.zeros(4, 6, 5, dtype=torch.bool)
    solid[:, 0, :] = True
    near = rwb.compute_near_wall_from_solid(solid)
    expect = torch.zeros_like(solid)
    expect[:, 1, :] = True
    expect[:, 5, :] = True  # torch.roll periodic wrap
    assert torch.equal(near, expect)


def test_near_wall_corner_counts_both_neighbors():
    """A fluid cell touching solid on two faces is still a single True cell."""
    solid = torch.zeros(5, 5, 5, dtype=torch.bool)
    solid[0, :, :] = True   # top face
    solid[:, 0, :] = True   # side face
    near = rwb.compute_near_wall_from_solid(solid)
    # fluid cell (1,1,*) touches solid on both -j and -i faces → True (once)
    assert near[1, 1, :].all()
    # a cell two away from any solid is False
    assert not near[3, 3, :].any()
    # solid cells themselves are never near-wall
    assert not near[solid].any()


def test_near_wall_isolated_solid_cell():
    """A single solid voxel marks its 6 fluid neighbors (within the box)."""
    solid = torch.zeros(5, 5, 5, dtype=torch.bool)
    solid[2, 2, 2] = True
    near = rwb.compute_near_wall_from_solid(solid)
    expect = torch.zeros_like(solid)
    for dk, dj, di in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        expect[2 + dk, 2 + dj, 2 + di] = True
    assert torch.equal(near, expect)


# ── validity gate: wall function is only valid in the log-law region ───────
def test_test_config_is_in_loglaw_region():
    """The suite's (u_tau, y, ν) must sit in 30 < y+ < 600, else the wall
    function is being applied outside its validity range and the numbers above
    are meaningless."""
    yp = _yplus(U_TAU, Y_VAL, NU)
    assert 30.0 < yp < 600.0, f"y+={yp:.1f} outside log-law validity 30..600"


# ── documented-but-unimplemented ω gap (finding, not a pass/fail gate) ─────
def test_omega_is_documented_but_not_implemented():
    """The module docstring lists an ω_wall formula for k-ω SST, but the
    function signature has no ``omega`` argument and never writes one.  This
    test pins the *current* contract (no omega parameter) so a future fix is a
    deliberate, reviewed change rather than a silent drift.  See the report §5.
    """
    import inspect
    sig = inspect.signature(rwb.set_rans_wall_bc_k_epsilon)
    assert "omega" not in sig.parameters, (
        "an 'omega' parameter now exists — update this test and the report "
        "when the k-ω SST wall value is actually implemented"
    )
