"""Viscoelastic (Oldroyd-B / FENE-P) branch for TensorLBM.

Public surface
--------------
``ViscoelasticChannel2D``
    Coupled lattice-Boltzmann + conformation-tensor solver for plane Poiseuille
    flow with polymer stress feedback and realizability projection.
``analytic_channel``
    Exact steady plane-Poiseuille solution of both models (velocity, ``C``,
    ``N1``) -- the verification anchor.
``energy_budget``
    Pump work = solvent dissipation + polymer dissipation + elastic energy rate.
``virk``
    Virk/Prandtl-Karman asymptotes (Newtonian, MDR, ``Delta B``) -- the
    drag-reduction envelope.
"""

from __future__ import annotations

from . import conformation, energy, solver, virk
from .conformation import (
    FeneP,
    OldroydB,
    analytic_channel,
    min_eigenvalue,
    project_implementable,
    trace3,
)
from .energy import energy_budget
from .solver import ChannelResult, ViscoelasticChannel2D, ViscoelasticDivergenceError

__all__ = [
    "conformation",
    "energy",
    "solver",
    "virk",
    "OldroydB",
    "FeneP",
    "trace3",
    "min_eigenvalue",
    "project_implementable",
    "analytic_channel",
    "energy_budget",
    "ViscoelasticChannel2D",
    "ChannelResult",
    "ViscoelasticDivergenceError",
]