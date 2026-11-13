"""I/I mass-exchange selector: pair symmetry and conservation invariants.

Anchor XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE.  The INTERFACE/INTERFACE stage of
the Koerner eq. 9 exchange must commit ``+dm`` on one end of a link and ``-dm``
on the other.  The historical repo selector reads the *pull source* liquid
neighbour count against the *receiver* gas neighbour state, which is not
symmetric under ``i <-> j``; ``interface_exchange_rule="pair_symmetric"`` keeps
the intent with a pair-symmetric predicate.  These tests bind the shipped
solver to the audit in
``xval/XVAL-LBM-THUEREY-VOF-MASS-EXCHANGE/run_operator_audit.py``.
"""
from __future__ import annotations

import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.free_surface_lbm_2d import (
    GAS, INTERFACE, LIQUID, SOLID, free_surface_step_2d,
)

RULES = ("repo", "pair_symmetric", "koerner", "oss")


def _slab(ny=24, nx=32):
    """3-cell-thick INTERFACE slab in GAS (the asymmetric-selector configuration)."""
    flags = torch.full((ny, nx), GAS, dtype=torch.int8)
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[0, :] = solid[-1, :] = True
    solid[:, 0] = solid[:, -1] = True
    flags[solid] = SOLID
    flags[8:11, 5:20] = INTERFACE
    f = equilibrium(torch.ones(ny, nx), torch.full((ny, nx), 0.02), torch.zeros(ny, nx))
    mass = torch.full((ny, nx), 0.5)
    mass[flags != INTERFACE] = 0.0
    return f, mass, flags, solid


def _run(rule, steps=10, **kw):
    f, mass, flags, solid = _slab(**kw)
    ledger = {}
    dead = live = 0
    for _ in range(steps):
        ledger.clear()
        f, mass, flags = free_surface_step_2d(
            f, mass.clone(), flags, solid, tau=1.0, mass=mass.clone(),
            return_mass=True, ledger=ledger, interface_exchange_rule=rule)
        dead += ledger["exchange_ii_dead_links"]
        live += ledger["exchange_ii_live_links"]
    return dead, live


def test_pair_symmetric_selector_deadens_fewer_links():
    dead_repo, live_repo = _run("repo")
    dead_sym, live_sym = _run("pair_symmetric")
    dead_k, live_k = _run("koerner")
    assert dead_k == 0, "the Koerner arm has no selector"
    assert dead_sym < dead_repo, (
        f"pair_symmetric must only deaden links dead at both ends "
        f"(repo {dead_repo}, sym {dead_sym})")
    assert live_sym + dead_sym == live_repo + dead_repo, "same I/I link population"


def test_pair_symmetric_is_strictly_antisymmetric():
    """Every live I/I pair contributes +dm/-dm: the net is round-off, not a fraction."""
    f, mass, flags, solid = _slab()
    ledger = {}
    net = absx = 0.0
    for _ in range(10):
        ledger.clear()
        f, mass, flags = free_surface_step_2d(
            f, mass.clone(), flags, solid, tau=1.0, mass=mass.clone(), return_mass=True,
            ledger=ledger, interface_exchange_rule="pair_symmetric")
        net += ledger["exchange_interface_delta"]
        absx += abs(ledger["exchange_interface_delta"])
    scale = float(mass.abs().sum())
    assert abs(net) < 1e-5 * max(absx, 1e-12) + 1e-6 * scale, (
        f"net I/I mass {net!r} is not round-off against sum|dm|={absx!r}")


def test_rule_argument_is_validated():
    f, mass, flags, solid = _slab()
    try:
        free_surface_step_2d(f, mass.clone(), flags, solid, mass=mass.clone(),
                             return_mass=True, interface_exchange_rule="nope")
    except ValueError as exc:
        assert "interface_exchange_rule" in str(exc)
    else:
        raise AssertionError("unknown rule must raise")


def test_default_rule_is_bit_identical_to_repo():
    """Default argument keeps the historical results (no silent physics change)."""
    f, mass, flags, solid = _slab()
    a = free_surface_step_2d(f, mass.clone(), flags, solid, tau=1.0, mass=mass.clone(), return_mass=True)
    b = free_surface_step_2d(f, mass.clone(), flags, solid, tau=1.0, mass=mass.clone(),
                             return_mass=True, interface_exchange_rule="repo")
    for x, y in zip(a, b):
        assert torch.equal(x, y)
