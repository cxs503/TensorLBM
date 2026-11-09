"""Cross-solver verification bench (node P78-N01).

One physical problem, three independent discretisations, one measurement
operator:

* :mod:`.lbm` -- D2Q9 lattice Boltzmann (numpy), half-way bounce-back walls,
  Guo forcing;
* :mod:`.fvm` -- cell-centred finite volume in vorticity-streamfunction form,
  exact discrete Poisson solve, Woods wall vorticity;
* :mod:`.fem` -- Q2-Q1 Taylor-Hood finite elements with Newton iteration.

The bench checks (i) agreement with the Ghia et al. (1982) lid-driven cavity
reference, (ii) observed order of accuracy against a manufactured solution,
(iii) the steady energy budget ``P = eps`` and the discrete mass budget, and
(iv) three-way code-to-code consistency.

All three solvers are dependency-light reference implementations living in this
package on purpose: the cross-solver arbitration must be reproducible inside a
single checkout (no GPU stack, no external solver binaries), and each one uses a
*methodologically different* discretisation of the same PDE.
"""

from .fvm import FVMResult, solve_psi_omega
from .fem import FEMResult, solve_taylor_hood
from .lbm import LBMResult, run_lbm_cavity, run_lbm_mms
from .ledger import (
    compare_ghia,
    divergence_ledger,
    energy_ledger,
    ghia_reference,
    relative_l2,
)
from .mms import ManufacturedSolution
from .runner import BenchmarkConfig, run_benchmark

__all__ = [
    "FVMResult",
    "FEMResult",
    "LBMResult",
    "BenchmarkConfig",
    "ManufacturedSolution",
    "compare_ghia",
    "divergence_ledger",
    "energy_ledger",
    "ghia_reference",
    "relative_l2",
    "run_benchmark",
    "run_lbm_cavity",
    "run_lbm_mms",
    "solve_psi_omega",
    "solve_taylor_hood",
]
