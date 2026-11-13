"""Cross-validation of TensorLBM unit conversion against 2b-t/LB-t approach.

This module provides:
- ``LbtStyleConverter``: A converter mirroring the 2b-t/LB-t ``Converter`` API
  (physical_length/equivalent_lbm_length triple) for direct comparison.
- ``CaseSchema``: JSON-based case definition schema inspired by LB-t's
  ``settings.json`` pattern.
- ``bootstrap_from_schema``: Build a ``LatticeBridge`` from a case schema.
- ``run_xval``: Full cross-validation benchmark producing a JSON report.

Reference
---------
2b-t/LB-t (Tobit Flatscher), ``include/lbt/converter.hpp``:
  Converter(physical_length, equivalent_lbm_length,
            physical_velocity, equivalent_lbm_velocity,
            physical_density, equivalent_lbm_density)
  Derives: time, kinematic_viscosity, pressure conversion factors.

J. Latt, "Choice of units in lattice Boltzmann simulations",
  Palabos Documentation (2008).
"""

from .converter import LbtStyleConverter
from .case_schema import CaseSchema, bootstrap_from_schema
from .xval_benchmark import run_xval

__all__ = [
    "LbtStyleConverter",
    "CaseSchema",
    "bootstrap_from_schema",
    "run_xval",
]
