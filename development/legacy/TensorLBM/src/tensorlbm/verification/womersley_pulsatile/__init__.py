"""Pulsatile (Womersley) duct-flow verification: analytic reference + solvers.

Case ``CASE-LBM-WOMERSLEY-PULSATILE`` (reference: Womersley 1955, J. Physiol.
127:553).  Acceptance: centreline velocity amplitude ratio and phase lag versus
the analytic Womersley solution at ``alpha = 5`` (whole-period phase averaged)
within 5 %.

Public API
----------
``analytical``  complex-phasor solutions (parallel-plate channel and circular
                pipe), force scaling and the steady Poiseuille limits.
``metrics``     whole-period phase averaging + harmonic (phasor) fitting and
                the amplitude/phase comparison metrics.
``gates``       predeclared acceptance gates (tolerances are fixed a priori).
``channel2d``   D2Q9 solver/driver for the parallel-plate geometry.
``pipe3d``      D3Q19 solver/driver for the circular-pipe geometry with BFL
                interpolated bounce-back on the true circle.
"""

from . import analytical, gates, metrics
from .channel2d import ChannelPulsatileConfig, ChannelPulsatileSolver, run_case as run_channel_case
from .gates import CRITERIA, TOL_PCT, convergence_verdict, evaluate_case
from .metrics import fit_phasor, fold_phase, harmonic_metrics
from .pipe3d import PipePulsatileConfig, PipePulsatileSolver, run_case as run_pipe_case

__all__ = [
    "analytical",
    "metrics",
    "gates",
    "ChannelPulsatileConfig",
    "ChannelPulsatileSolver",
    "PipePulsatileConfig",
    "PipePulsatileSolver",
    "run_channel_case",
    "run_pipe_case",
    "evaluate_case",
    "convergence_verdict",
    "CRITERIA",
    "TOL_PCT",
    "fold_phase",
    "fit_phasor",
    "harmonic_metrics",
]