"""Lattice <-> physical unit bootstrap layer and its machine-verifiable gate.

Public surface
--------------
``LatticeBridge``      explicit conversion sites (length/time/velocity/
                       viscosity/density/stress/force) + mutation factors
``bootstrap_bridge``   derive ``dx``/``dt`` from a physical operating point,
                       a grid and the compressibility constraint
``MUTATIONS``/``mutate``
                       single-site coefficient defects used by the battery
``ANCHORS``/``TOLERANCES``
                       the four analytic anchors and their pre-declared
                       tolerances
``run_gate``           full acceptance gate -> JSON report
``channel_poiseuille`` / ``channel_couette`` / ``taylor_green``
                       minimal D2Q9 kernels the anchors run on

Typical use::

    from tensorlbm.units import run_gate
    report = run_gate()
    assert report["overall_passed"]

or on the command line::

    python -m tensorlbm.units.gate
"""

from .anchors import ANCHORS, TOLERANCES, AnchorCase, build_case, evaluate
from .bridge import (
    CS_LATTICE,
    MUTATIONS,
    LatticeBridge,
    bootstrap_bridge,
    mutate,
)
from .gate import GRID_SPREAD_TOL, ROUNDTRIP_TOL, run_gate
from .lattice import channel_couette, channel_poiseuille, taylor_green

__all__ = [
    "ANCHORS",
    "AnchorCase",
    "CS_LATTICE",
    "GRID_SPREAD_TOL",
    "LatticeBridge",
    "MUTATIONS",
    "ROUNDTRIP_TOL",
    "TOLERANCES",
    "bootstrap_bridge",
    "build_case",
    "channel_couette",
    "channel_poiseuille",
    "evaluate",
    "mutate",
    "run_gate",
    "taylor_green",
]