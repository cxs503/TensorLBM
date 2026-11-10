"""Turbulence-validation face of TensorLBM.

This package turns the LBM solver from "laminar-capable" into "turbulence-capable
and turbulence-*validated*" by providing, in one place:

* :mod:`.mrt` - a multi-relaxation-time collision operator built on a generated
  orthonormal moment basis (D2Q9 and D3Q19), with a Smagorinsky sub-grid closure
  driven by the non-equilibrium stress;
* :mod:`.lattice` - the two discrete velocity sets, shared by Python and the C
  kernels;
* :mod:`.native` - compilation/first-class invocation of the two C kernels
  (planar D2Q9 and 3D D3Q19 channel) plus typed readers for their artefacts;
* :mod:`.metrics` - Strouhal number, reattachment length, wall-unit reduction,
  u+ vs DNS comparison, control-volume drag;
* :mod:`.reference` - every experimental/DNS anchor in one declaration with its
  source (channel anchors come from the shipped MKM-1999 tables);
* :mod:`.cases` / :mod:`.campaign` - the three-case x three-Reynolds campaign and
  the paper-grade report generator;
* :mod:`.verification` - definition-level checks of the collision operator
  (analytic shear-wave decay, analytic forced Poiseuille).

The wall treatment is the non-equilibrium extrapolation (NEEM) scheme of Guo et
al. (2002) for open boundaries plus a Reichardt (1951) wall-stress model for the
wall-adjacent cell (see ``mrt_channel3d.c`` and :mod:`.metrics`).
"""

from __future__ import annotations

from .lattice import D2Q9, D3Q19, Lattice, get_lattice
from .mrt import MRTModel, build_moment_matrix, stress_mode_indices

__all__ = [
    "D2Q9",
    "D3Q19",
    "Lattice",
    "MRTModel",
    "build_moment_matrix",
    "get_lattice",
    "stress_mode_indices",
]