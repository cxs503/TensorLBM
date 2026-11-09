"""Tests for the ice-interface momentum-exchange repair (mission AG27W2-N01).

Covers

* the sub-grid wetted fraction against analytic geometry,
* the four conservation gates C1-C4 (mass / momentum / angular momentum /
  energy) on the analytic bench cases,
* the two defect reproductions: staircase wetted fraction (wall-position
  error) and the static Ladd estimator on a *moving* ice surface (the
  IN37-N05 "ice books only part of the momentum budget" defect class).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorlbm.ice_interface_momentum import (
    CX,
    CY,
    CS2,
    W,
    ConservationGates,
    Disk,
    HalfPlane,
    IceInterfaceChannel,
    InterfaceSet,
    _gates_for,
    run_couette_moving_wall,
    run_disk_momentum_budget,
    run_poiseuille_subgrid,
    subgrid_wetted_fraction,
)


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def test_wetted_fraction_flat_wall_subgrid_position():
    """A flat wall at y = 3.75 must leave fractional coverage, sum = area."""
    iface = InterfaceSet().add(HalfPlane((0.0, 1.0), (0.0, 3.75)))
    geo = subgrid_wetted_fraction(iface, nx=6, ny=8, n_cell=16)
    alpha = geo["alpha"]
    # row j=3 spans y in [3, 4]: 0.75 of it is ice -> alpha = 0.25;
    # row j=2 is fully ice and row j=4 fully water
    assert alpha[2].max() == pytest.approx(0.0, abs=1e-12)
    assert alpha[3].min() == pytest.approx(0.25, abs=4e-2)
    assert alpha[4].min() == pytest.approx(1.0, abs=1e-12)
    # total water area = sum of alpha = 8 rows - 3.75 = 4.25 per column
    assert alpha.sum() / 6 == pytest.approx(8 - 3.75, rel=1e-2)
    # chi of the vertical link through the interface must be fractional
    chi_up = geo["chi"][2]   # direction (0, 1)
    assert 0.0 < chi_up[3, 0] < 1.0
    w = geo["w"]
    assert np.all(w <= 1.0) and np.all(w >= 0.0)


def test_wetted_fraction_disk_area():
    """Sum of alpha must equal the analytic water area of a disk complement."""
    R, cx, cy, nx, ny = 3.3, 10.7, 8.2, 20, 16
    iface = InterfaceSet().add(Disk(cx, cy, R))
    geo = subgrid_wetted_fraction(iface, nx=nx, ny=ny, n_cell=16)
    water = geo["alpha"].sum()
    assert water == pytest.approx(nx * ny - np.pi * R * R, rel=5e-3)


def test_wetted_fraction_staircase_limit():
    """A face-aligned wall gives exactly ``w in {0, 1}`` for the *face* links.

    The diagonal links crossing a face-aligned wall are genuinely fractional
    (they cut the interface at 45 degrees), so only the axis-aligned directions
    are expected to be binary.
    """
    iface = InterfaceSet().add(HalfPlane((0.0, 1.0), (0.0, 3.0)))
    geo = subgrid_wetted_fraction(iface, nx=4, ny=6)
    chi = geo["chi"]
    for k in (2, 4):                      # (0, 1) and (0, -1)
        vals = chi[k]
        iface_frac = (vals > 0) & (vals < 1)
        # a face-aligned wall cuts the straddling link exactly in half
        assert np.allclose(vals[iface_frac], 0.5, atol=1e-12)
    w = geo["w"]
    blocked = chi[2] > 0.5
    assert np.all(w[2][blocked] == pytest.approx(1.0))


# ---------------------------------------------------------------------------
# conservation gates
# ---------------------------------------------------------------------------
def test_poiseuille_subgrid_conservation_gates():
    """Plane Poiseuille between sub-grid walls: all four gates must close."""
    res = run_poiseuille_subgrid(steps=2500, subgrid=True)
    g = res["gates"]
    assert g["c1_mass"]["rel"] < 1e-10
    assert g["c2_momentum"]["rel"] < 1e-9
    assert g["c3_angular"]["rel"] < 1e-9
    assert g["c4_energy"]["rel"] < 1e-2
    assert g["all_passed"] is True
    # the momentum budget must reproduce the body-force impulse
    assert res["mem_over_discrete_budget"] == pytest.approx(1.0, rel=0.10)
    # the analytic streamwise profile (incl. the sub-grid wall positions)
    assert res["u_profile_rel_rms"] < 0.02


def test_subgrid_beats_staircase_wall_position():
    """The sub-grid wetted fraction removes the wall-position error."""
    sub = run_poiseuille_subgrid(steps=2500, subgrid=True)
    sta = run_poiseuille_subgrid(steps=2500, subgrid=False)
    assert sub["u_profile_rel_rms"] < 0.25 * sta["u_profile_rel_rms"]
    assert sub["wall_force_rel_error"] < sta["wall_force_rel_error"]


def test_couette_moving_wall_needs_the_moving_term():
    """A moving ice surface: the static Ladd estimator loses the exchange."""
    mov = run_couette_moving_wall(steps=1500, moving_term=True)
    sta = run_couette_moving_wall(steps=1500, moving_term=False)
    # static estimator books ~nothing at all (the defect signature)
    assert sta["wall_force_measured"] < 1e-6
    assert mov["wall_force_measured"] > 1e-4
    # the moving-interface term is O(3 u_w) in lattice units
    assert sta["wall_force_rel_error"] == pytest.approx(1.0, abs=0.02)
    # mass / momentum / energy ledgers close for the moving interface too
    assert mov["gates"]["c1_mass"]["rel"] < 1e-10
    assert mov["gates"]["c2_momentum"]["rel"] < 1e-9
    assert mov["gates"]["c4_energy"]["rel"] < 1e-2
    # the linear Couette profile is reproduced
    assert mov["u_profile_rel_rms"] < 0.02


def test_corner_links_change_the_force_but_not_the_gate():
    """Falsified sub-hypothesis: a missing corner link is *not* caught by C2.

    The ledger follows whatever closure is actually applied, so dropping the
    diagonal closure links keeps the mass/momentum identities intact (measured
    residual 0.0) while it changes the computed force materially.  Recording
    this keeps the negative result in the audit trail: mask completeness must be
    checked against a reference force, not against a conservation identity.
    """
    iface = InterfaceSet().add(
        HalfPlane((0.0, 1.0), (0.0, 1.37), u_wall=(0.05, 0.0))
    ).add(HalfPlane((0.0, -1.0), (0.0, 14.63)))
    forces = {}
    gates = {}
    for corner in (True, False):
        ch = IceInterfaceChannel(iface, 18, 8, 0.8, corner_links=corner)
        s0 = (ch.mass(), ch.momentum(), ch.angular_momentum(),
              ch.kinetic_energy(), ch.lattice_energy())
        for _ in range(200):
            ch.step()
        forces[corner] = float(ch.force(0)[0])
        gates[corner] = _gates_for(ch, *s0)
    assert gates[True]["c2_momentum"]["rel"] < 1e-9
    assert gates[False]["c2_momentum"]["rel"] < 1e-9   # identity survives ...
    assert abs(forces[True] - forces[False]) > 1e-6     # ... the force does not


def test_disk_floe_momentum_budget():
    """Ice floe in a periodic channel: MEM force vs. the momentum budget."""
    res = run_disk_momentum_budget(steps=1200, subgrid=True)
    assert res["gates"]["c1_mass"]["rel"] < 1e-10
    assert res["gates"]["c2_momentum"]["rel"] < 1e-9
    # the ice force must be a finite fraction of the drive (not zero, not 10x)
    assert 0.2 < res["mem_over_budget"] < 3.0


def test_ledger_books_the_moving_interface_mass_flux():
    """The moving-wall mass injection must appear in the ledger (not vanish)."""
    yb, yt, U = 1.37, 14.63, 0.06
    ny = int(np.ceil(yt)) + 3
    iface = InterfaceSet().add(
        HalfPlane((0.0, 1.0), (0.0, yb), u_wall=(U, 0.0))
    ).add(HalfPlane((0.0, -1.0), (0.0, yt)))
    ch = IceInterfaceChannel(iface, ny, 6, 0.8)
    m0 = ch.mass()
    for _ in range(300):
        ch.step()
    # mass flux per link is 2 w rho (c.u_w)/cs^2; it must be booked exactly
    booked = ch.ledger.mass_in
    assert abs((ch.mass() - m0) - booked) < 1e-10 * max(1.0, abs(m0))
    # and the momentum carried by that flux is of order rho u_w * mass flux
    assert ch.ledger.momentum_conv >= 0.0


def test_gate_thresholds_are_the_contract():
    """The gate thresholds are the contract of the mission criterion."""
    g = ConservationGates()
    assert g.tol_mass == 1e-10
    assert g.tol_momentum == 1e-9
    assert g.tol_angular == 1e-9
    assert g.tol_energy == 1e-2
    ok = g.evaluate(c1_mass=dict(rel=1e-12), c2_momentum=dict(rel=1e-12),
                    c3_angular=dict(rel=1e-12), c4_energy=dict(rel=1e-3))
    assert ok["all_passed"] is True
    bad = g.evaluate(c1_mass=dict(rel=1e-4), c2_momentum=dict(rel=1e-12),
                     c3_angular=dict(rel=1e-12), c4_energy=dict(rel=1e-3))
    assert bad["all_passed"] is False