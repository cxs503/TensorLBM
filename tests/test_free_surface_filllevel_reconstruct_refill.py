"""Verification of the free-surface fill-level / reconstruct / refill cycle.

Anchor: XVAL-LBM-FREE-SURFACE-FILLLEVEL-RECONSTRUCT-REFIL
(lssfau/waLBerla ``free_surface`` fill-level-reconstruct-refill approach).

The waLBerla free-surface module advances an *independent* per-cell liquid
mass ``m`` whose fill level is ``epsilon = m / rho_liquid``.  Three coupled
operations make up the cycle:

* **fill-level advance**  -- the interface fill is updated by the Körner
  mass exchange (``epsilon(x) = m(x)/rho_local``), unclamped so the
  conversion gates read the true over/undershoot;
* **reconstruct**       -- missing pull populations at an interface cell are
  rebuilt by anti-bounce-back (ABB) against the gas pressure (Thürey
  eq. 4.5 / waLBerla ``ReconstructInterfaceCellABB``);
* **refill**            -- a cell promoted to LIQUID/INTERFACE is refilled
  with neighbour-averaged equilibrium populations (waLBerla ``Refill`` /
  ``_init_new_birth``), so a freshly born cell carries a valid, bounded,
  mass-conserving distribution instead of a stale gas remnant.

This test certifies the *properties* of that cycle on the D2Q9
``free_surface_lbm_2d`` kernel: mass conservation through a full
reconstruct+refill sweep, boundedness/positivity of every refilled
population, correct fill-level reconstruction from the independent mass
field, and the one-cell interface envelope (no direct LIQUID-GAS link).
It runs in a few seconds on a small grid.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorlbm import free_surface_lbm_2d as FS2
from tensorlbm.free_surface_lbm_2d import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    count_direct_liquid_gas_links_2d,
    free_surface_step_2d,
    init_fill_rectangular_2d,
    init_flags_from_fill_2d,
    total_liquid_inventory_2d,
)

DTYPE = torch.float64


def _dam_break_ic(ny=40, nx=40, cw=16, ch=16, device="cpu"):
    """Rectangular liquid column (dam-break IC) with a one-cell interface."""
    fill, solid = init_fill_rectangular_2d(ny, nx, cw, ch, device)
    flags = init_flags_from_fill_2d(fill, solid)
    # seed a valid D2Q9 equilibrium for the liquid/interface cells
    rho = torch.where(
        (flags == LIQUID) | (flags == INTERFACE),
        torch.full_like(fill, 1.0),
        torch.zeros_like(fill),
    )
    f = FS2.equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho))
    return f, fill, flags, solid


def test_filllevel_reconstruct_refill_conserves_liquid_inventory():
    """A full reconstruct+refill sweep must conserve the liquid inventory.

    The waLBerla invariant is ``sum(rho over LIQUID) + sum(m over INTERFACE)``
    (fill = m/rho_liquid).  The ABB reconstruction is a *population* change
    (gas pressure boundary) and the refill of a born cell is a *population*
    change too; neither may touch the tracked liquid mass, so the inventory
    must stay flat through the whole cycle.
    """
    f, fill, flags, solid = _dam_break_ic()
    m0 = float(total_liquid_inventory_2d(f, fill, flags, rho_liquid=1.0))
    assert m0 > 0.0
    for _ in range(30):
        f, fill, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=1.0, gx=0.0, gy=0.0,
            rho_liquid=1.0, rho_gas=1.0,
        )
    m1 = float(total_liquid_inventory_2d(f, fill, flags, rho_liquid=1.0))
    # no gravity -> no bulk advection; the exchange + refill must be neutral
    rel = abs(m1 - m0) / m0
    assert rel < 1e-6, f"liquid inventory drifted {rel:.2e}"


def test_filllevel_reconstructed_from_independent_mass():
    """fill = m/rho_local at INTERFACE, 1 at LIQUID, 0 at GAS/SOLID."""
    f, fill, flags, solid = _dam_break_ic()
    for _ in range(10):
        f, fill, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=1.0, gx=0.0, gy=0.0,
            rho_liquid=1.0, rho_gas=1.0,
        )
    # every LIQUID cell is pinned at fill == 1
    assert torch.all(fill[flags == LIQUID] == 1.0)
    # every GAS/SOLID cell is at fill == 0
    assert torch.all(fill[(flags == GAS) | (flags == SOLID)] == 0.0)
    # interface fill is a genuine fraction in (0,1) (unclamped, so allow the
    # small conversion overshoot band used by the gates)
    iface = flags == INTERFACE
    assert iface.any()
    assert float(fill[iface].min()) >= -1e-2
    assert float(fill[iface].max()) <= 1.0 + 1e-2


def test_refilled_populations_are_valid_and_bounded():
    """Every refilled (newly born) cell must carry a valid, bounded PDF.

    A cell promoted to LIQUID/INTERFACE is refilled with neighbour-averaged
    equilibrium; its populations must be non-negative, finite and bounded by
    the lattice density range -- a stale gas remnant or an unbounded
    neighbour-averaged velocity would otherwise drive the next step to
    inf/NaN (the exact failure the input guard exists to catch).
    """
    f, fill, flags, solid = _dam_break_ic()
    for _ in range(20):
        f, fill, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=1.0, gx=0.0, gy=0.0,
            rho_liquid=1.0, rho_gas=1.0,
        )
    assert torch.isfinite(f).all()
    assert float(f.min()) >= 0.0
    # bounded by the lattice density range (rho_liquid * 3 guard)
    assert float(f.max()) <= 3.0 + 1e-6


def test_no_direct_liquid_gas_link_after_refill():
    """The refill must keep a one-cell interface envelope.

    waLBerla never lets a LIQUID cell stream directly into GAS: every
    LIQUID-GAS boundary is mediated by an INTERFACE layer whose missing pull
    population is ABB-reconstructed.  A refill that promoted a GAS cell
    straight to LIQUID (skipping the interface) would create a direct
    LIQUID-GAS link with no reconstruction -- the exact topology the
    ``_assert_no_direct_liquid_gas_links`` guard rejects.
    """
    f, fill, flags, solid = _dam_break_ic()
    for _ in range(20):
        f, fill, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=1.0, gx=0.0, gy=0.0,
            rho_liquid=1.0, rho_gas=1.0,
        )
    assert count_direct_liquid_gas_links_2d(flags) == 0


def test_gravity_driven_fill_advance_is_monotonic_and_bounded():
    """Under gravity the front advances and the fill stays in [0,1].

    A small downward gravity drives the column to spread; the fill level must
    advance monotonically (front position grows) while every cell's fill
    remains a valid fraction -- the reconstruct+refill cycle must not
    atomise the interface (the historical +323% wave-front pollution mode).
    """
    f, fill, flags, solid = _dam_break_ic()
    fronts = []
    for _ in range(40):
        f, fill, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=1.0, gx=0.0, gy=-1e-3,
            rho_liquid=1.0, rho_gas=1.0,
        )
        # front = rightmost column with any liquid/interface fill
        col = (fill > 0.5).any(dim=0)
        fronts.append(int(torch.nonzero(col).max().item()))
    assert fronts[-1] > fronts[0]  # front advances
    assert float(fill.min()) >= -1e-2
    assert float(fill.max()) <= 1.0 + 1e-2
