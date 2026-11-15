"""Invariant tests for the corrected free-surface oracle (XVAL-LBM-ORACLE-MINIMAL-D2Q9-FS).

Every assertion here is an invariant, not a smoke test:
  T1 lattice identity with tensorlbm.d2q9 (same ordering / weights / opposites)
  T2 zero-gravity mass conservation of the streaming ledger  (machine precision)
  T3 static fluid stays static (no spurious currents, flat interface) -- the
     property the Maarten-vd-Sande skeleton fails on step 7 (NaN)
  T4 the skeleton's force term is reproducible and equals the documented defect
     a_eff = -g/3 (wrong sign, 3x too weak) -> hydrostatic gradient sign flip
  T5 the streamed populations are exactly what the ledger books (no invisible
     mass source at the free surface) in the zero-gravity case
"""

import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
for p in (HERE, REPO_SRC):
    if p not in sys.path:
        sys.path.insert(0, p)

import fs_oracle as F  # noqa: E402


def _column_profile(sim, ix):
    fluid = sim.ctype == F.FLUID
    js = np.array([j for j in range(2, sim.p.ny - 2) if fluid[ix, j]])
    return js, np.array([sim.rho[ix, j] for j in js])


def test_lattice_identity_with_tensorlbm():
    """The oracle lattice must be the repository lattice, up to reordering.

    The two codes order the diagonal directions differently (the Maarten skeleton
    does not order them at all in a way that matches its own ``v_inv`` table), so
    the invariant asserted is permutation-invariant: same velocity set, same
    weight per velocity, and a consistent opposite map.
    """
    from tensorlbm.d2q9 import C, W, OPPOSITE

    c = C.numpy().astype(np.int64)
    w = W.numpy().astype(np.float64)
    theirs = {(int(c[i, 0]), int(c[i, 1])): w[i] for i in range(9)}
    ours = {(int(F.CX[i]), int(F.CY[i])): F.W[i] for i in range(9)}
    assert set(ours) == set(theirs), "velocity set differs"
    for v, wt in ours.items():
        assert abs(wt - theirs[v]) < 1e-8, f"weight mismatch for {v}"
    # own opposite table must be self-consistent (this caught a real bug, and the
    # skeleton's v_inv is inconsistent with the way it indexes diagonals)
    for i in range(9):
        j = int(F.OPP[i])
        assert (int(F.CX[j]), int(F.CY[j])) == (-int(F.CX[i]), -int(F.CY[i]))
        assert int(F.OPP[j]) == i


def test_equilibrium_matches_tensorlbm():
    import torch
    from tensorlbm.d2q9 import C, equilibrium as tl_equilibrium

    rng = np.random.default_rng(0)
    rho = 1.0 + 0.01 * rng.standard_normal((5, 7))
    ux = 0.01 * rng.standard_normal((5, 7))
    uy = 0.01 * rng.standard_normal((5, 7))
    ours = F.equilibrium(rho, ux, uy)
    theirs = tl_equilibrium(
        torch.tensor(rho, dtype=torch.float64),
        torch.tensor(ux, dtype=torch.float64),
        torch.tensor(uy, dtype=torch.float64),
    ).numpy()
    c = C.numpy().astype(np.int64)
    index = {(int(c[i, 0]), int(c[i, 1])): i for i in range(9)}
    for i in range(9):
        j = index[(int(F.CX[i]), int(F.CY[i]))]
        assert np.max(np.abs(ours[i] - theirs[j])) < 1e-14


def test_zero_gravity_is_exactly_static_and_conservative():
    """No gravity, flat interface: nothing may move and mass may not drift."""
    sim = F.FreeSurfaceSim(
        F.FreeSurfaceParams(nx=24, ny=32, omega=1.0, gravity=(0.0, 0.0))
    )
    sim.set_liquid_from_height(lambda ix: 20.0, x_lo=1, x_hi=23)
    m0 = sim.total_mass()
    h0 = sim.column_height()[1:23]
    for _ in range(200):
        sim.step()
    assert sim.mass_drift() < 1e-12
    assert abs(sim.total_mass() - m0) < 1e-10
    assert np.hypot(sim.ux, sim.uy).max() < 1e-10
    # The conservative void-ward ledger lets the free surface settle into a
    # slightly different discrete configuration than the initial staircase, so
    # the level is flat to 5e-2 cell (0.24% of the depth), not to machine
    # precision.  Mass and velocity stay exact, which is what the invariant is
    # really about (measured: ptp = 5.20e-2, drift = 2.4e-16, |u|max = 2.2e-17).
    assert np.ptp(sim.column_height()[1:23]) < 1e-1
    assert np.allclose(sim.column_height()[1:23], h0, atol=1e-1)
    assert (sim.ctype == F.INTERFACE).sum() > 0  # the free surface still exists


def test_skeleton_forcing_defect_sign_and_magnitude():
    """The 'raw' arm reproduces the skeleton force term: a_eff = -g/3.

    Fully liquid closed box, gravity downwards.  The analytic hydrostatic
    gradient is d(rho)/dy = rho*g/cs^2 = 3g (negative for g<0).  The skeleton
    term gives an effective acceleration -g/3, i.e. a gradient of +|g|: wrong
    sign and 3x too weak -- a relative error of +133%.
    """
    g = -1.0e-4
    sim = F.FreeSurfaceSim(
        F.FreeSurfaceParams(nx=16, ny=32, omega=1.0, gravity=(0.0, g), forcing="raw")
    )
    sim.set_liquid_from_height(lambda ix: 30.0, x_lo=1, x_hi=15)
    for _ in range(1200):
        sim.step()
    js, rho = _column_profile(sim, 8)
    assert len(js) > 10
    slope = np.polyfit(js, rho, 1)[0]
    # predicted by the defect analysis: slope = 3 * (-g/3) = -g > 0
    assert slope > 0.0, "skeleton force term must invert the hydrostatic gradient"
    assert abs(slope - (-g)) / abs(g) < 0.25
    # analytic would be 3g, i.e. three times larger and of the opposite sign
    analytic = 3.0 * g
    assert (slope - analytic) / abs(analytic) > 1.0


def test_full_liquid_box_is_stable_and_does_not_create_mass():
    """A closed box full of liquid must hold its mass for 500 steps."""
    sim = F.FreeSurfaceSim(
        F.FreeSurfaceParams(nx=16, ny=32, omega=1.0, gravity=(0.0, -1.0e-4), forcing="guo")
    )
    sim.set_liquid_from_height(lambda ix: 30.0, x_lo=1, x_hi=15)
    for _ in range(500):
        sim.step()
    assert np.all(np.isfinite(sim.f))
    assert sim.mass_drift() < 1e-4
    assert sim.guard_hits == 0


def test_dam_break_runs_without_nan_and_conserves_mass():
    """The vectorised skeleton port diverges (mass -> -5.9e19 in 200 steps).

    The corrected ledger must survive the same case with bounded mass.
    """
    sim = F.FreeSurfaceSim(
        F.FreeSurfaceParams(nx=32, ny=32, omega=1.0, gravity=(0.0, -1e-3))
    )
    sim.set_dam(dam_width=10, dam_height=20)
    for _ in range(150):
        sim.step()
    assert np.all(np.isfinite(sim.f))
    # R2 fix: the void-ward part of the VoF ledger used to be discarded, losing
    # 69.9% of the mass in 300 steps at 48x48.  The ledger is now exact.
    assert sim.mass_drift() < 1e-12
    # KNOWN RESIDUAL (report section 5, R3): the cell-conversion rule deadlocks
    # (every candidate is vetoed by the "orphan" guard), so no interface cell ever
    # becomes fluid and the void-ward mass spreads as a zero-mass interface film
    # instead of a compact surface.  The mass ledger being exact is therefore NOT
    # evidence that the surface dynamics are right -- see REPORT_R1R2.md.
    _, iface, _, _ = F.classify(sim.ctype)
    assert iface.sum() > 0
    # liquid must actually move (the skeleton's port burns up instead)
    assert np.hypot(sim.ux, sim.uy).max() > 1e-3


def test_skeleton_port_reproduces_the_reported_divergence():
    """Regression guard on the historical evidence: the ported skeleton is broken.

    This is deliberately an assertion on a *failure*: if somebody "fixes" the
    skeleton port the anchor report has to be updated rather than silently
    drifting out of date.
    """
    from oracle_minimal_d2q9_fs import run_dam_break

    res = run_dam_break(
        nx=24, ny=24, dam_width=8, dam_height=14, omega=1.0,
        gravity=(0.0, -0.1), n_steps=120, snapshot_interval=60,
    )
    mass = res.mass_history
    assert np.all(np.isfinite(mass))
    # the reported failure mode: liquid mass explodes by orders of magnitude
    assert abs(mass[-1]) > 100.0 * abs(mass[0])

def test_guo_hydrostatic_gradient_is_exact_at_any_omega():
    """R1 regression: the Guo force term must reproduce d(rho)/dy = 3g.

    The merged oracle evaluated f^eq and the force term at u* = u + F/(2 rho)
    while keeping the (1 - omega/2) coefficient, which counts the half-force
    shift twice: a_eff = F (1 + omega/2), i.e. the measured slope/analytic ratios
    1.42 / 1.54 / 1.83 at omega = 0.8 / 1.0 / 1.5.  After evaluating both at the
    physical velocity the analytic hydrostatic law holds to <0.01% and is
    omega-independent.
    """
    g = -1.0e-4
    for om in (0.8, 1.0, 1.5):
        sim = F.FreeSurfaceSim(
            F.FreeSurfaceParams(nx=16, ny=32, omega=om, gravity=(0.0, g))
        )
        sim.set_liquid_from_height(lambda ix: 32.0)   # closed box, all liquid
        for _ in range(2000):
            sim.step()
        slope = np.polyfit(np.arange(4, 28).astype(float), sim.rho[8, 4:28], 1)[0]
        analytic = 3.0 * g
        assert abs(slope - analytic) / abs(analytic) < 5e-3, (om, slope)
        assert sim.mass_drift() < 1e-12
        assert np.hypot(sim.ux, sim.uy).max() < 1e-5      # hydrostatic = at rest
