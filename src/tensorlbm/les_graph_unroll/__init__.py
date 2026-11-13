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
* :class:`~tensorlbm.les_graph_unroll.selection.SelectionCase` /
  :func:`~tensorlbm.les_graph_unroll.selection.select_and_judge` — the round-2
  objective/closure *selection* layer (train/hold-out split, BPTT + controls).
* :func:`~tensorlbm.les_graph_unroll.tgv_anchor.run_tgv_anchor` — the
  Taylor-Green anchor harness: exact-solution anchor + BPTT selection +
  held-out ``ref``/``got`` benchmark rows in one call.  Re-exported from the
  top-level package as ``tensorlbm.run_tgv_anchor``.

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
from .selection import (
    OBJECTIVES,
    SelectionCase,
    bptt_select,
    enstrophy,
    holdout_metrics,
    kinetic_energy,
    objective_loss,
    select_and_judge,
    sweep_select,
)
from .solver2d import collide_stream, equilibrium, macroscopic, tau_from_cs, unroll
from .tgv_anchor import AnchorConfig, run_tgv_anchor

__all__ = [
    "OBJECTIVES",
    "AnchorConfig",
    "SelectionCase",
    "bptt_select",
    "enstrophy",
    "holdout_metrics",
    "kinetic_energy",
    "objective_loss",
    "run_tgv_anchor",
    "select_and_judge",
    "sweep_select",
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
