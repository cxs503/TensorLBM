"""Graph-unroll BPTT calibration of an LES parameter (D2Q9).

Cross-validation counterpart of the TensorFlow-1.x ``graph_unroll`` branch in
``loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow``
(``LatFlow/Domain.py:134-140, 192-196, 244-253``): the time loop stays inside
the autograd graph, ``tau`` is a graph function of the local field and the
scalar model constant, so a field mismatch at the end of an unrolled window
trains the LES parameter by plain backpropagation.

Entry points:

* :func:`~tensorlbm.les_graph_unroll.solver2d.unroll` — the differentiable
  D2Q9 collide/stream chain (``graph_unroll``).
* :class:`~tensorlbm.les_graph_unroll.calibrate.CalibProblem` — coarse-grid
  unroll + filtered reference snapshots.
* :func:`~tensorlbm.les_graph_unroll.calibrate.bptt_calibrate` — the BPTT
  calibration itself, plus ``fd_gradient`` / ``golden_minimize`` /
  ``sweep_loss`` as independent controls.
* :func:`~tensorlbm.les_graph_unroll.reference2d.run_reference` — the
  fine-grid reference solver and the analytic initial conditions.

The cross-validation run and its artefacts live in
``scripts/xval_graph_unroll_bptt_les.py`` and
``artifacts/xval_graph_unroll_bptt_les/``.
"""

from __future__ import annotations

from .calibrate import (
    CalibProblem,
    autograd_gradient,
    bptt_calibrate,
    fd_gradient,
    field_loss,
    golden_minimize,
    measure_eddy_viscosity,
    snapshot_errors,
    sweep_loss,
)
from .solver2d import collide_stream, equilibrium, macroscopic, tau_from_cs, unroll

__all__ = [
    "CalibProblem",
    "autograd_gradient",
    "bptt_calibrate",
    "collide_stream",
    "equilibrium",
    "fd_gradient",
    "field_loss",
    "golden_minimize",
    "macroscopic",
    "measure_eddy_viscosity",
    "snapshot_errors",
    "sweep_loss",
    "tau_from_cs",
    "unroll",
]
