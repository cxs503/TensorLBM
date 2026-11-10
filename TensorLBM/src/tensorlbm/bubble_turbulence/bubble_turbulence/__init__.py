"""Bubble-turbulence interaction module for air lubrication DNS.

Implements Lagrangian point-particle microbubbles coupled to a D3Q19 MRT
turbulent channel flow solver.  The module targets the physical question:
how do microbubbles (d+ = 5-20 wall units) modulate the near-wall turbulent
structure and reduce friction drag?

Three drag-reduction mechanisms are decomposed:
  1. Effective slip (bubble layer acts as a mobile boundary)
  2. Turbulence modulation (suppression of Reynolds stress / burst frequency)
  3. Viscosity modification (mixture viscosity change)

References:
  - Lu, Tryggvason & Maxey (2005) DNS of bubbly channel flow
  - Ferrante & Elghobashi (2004) On the physical mechanisms of DR
  - Madavan, Deutsch & Merkle (1985) Reduction of turbulent skin friction
    by microbubbles
"""

from tensorlbm.bubble_turbulence.bubble_solver import (
    BubbleConfig,
    BubbleSolver,
    TwoPhaseChannelSolver,
)
from tensorlbm.bubble_turbulence.statistics import (
    ChannelStatistics,
    compute_q_criterion,
)
from tensorlbm.bubble_turbulence.mechanisms import (
    DragReductionDecomposition,
    compute_slip_length,
)
from tensorlbm.bubble_turbulence.campaign import run_bubble_campaign

__all__ = [
    "BubbleConfig",
    "BubbleSolver",
    "TwoPhaseChannelSolver",
    "ChannelStatistics",
    "compute_q_criterion",
    "DragReductionDecomposition",
    "compute_slip_length",
    "run_bubble_campaign",
]
