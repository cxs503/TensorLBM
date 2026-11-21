"""Linear-interpolated bounce-back (alpha/Bouzidi) scheme validation.

This module validates the interpolated bounce-back boundary condition
(also known as the Bouzidi-Lallemand-Luo scheme or Inamuro wet-node method)
for curved boundaries in D2Q9 LBM.

The scheme achieves second-order accuracy by linearly interpolating
distribution functions based on the fractional distance q from the
fluid node to the curved boundary.

References
----------
- Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001). "Momentum transfer
  of a lattice Boltzmann fluid with boundaries." Physics of Fluids, 13(11), 3452.
- Inamuro, T., Yoshino, M., & Ogino, F. (1995). "A non-slip boundary condition
  for lattice Boltzmann simulations." Physics of Fluids, 7(12), 2928.
- Krüger, T., et al. (2017). "The Lattice Boltzmann Method." Springer, Ch. 5.
"""

from .solver import (
    LinearInterpolatedBBAlphaConfig,
    LinearInterpolatedBBAlphaSolver,
    run_convergence_study,
    run_single_case,
)
from .metrics import (
    compute_l2_error,
    compute_linfinity_error,
    compute_flow_rate_error,
    compute_max_velocity_error,
    ConvergenceResult,
)

__all__ = [
    "LinearInterpolatedBBAlphaConfig",
    "LinearInterpolatedBBAlphaSolver",
    "run_convergence_study",
    "run_single_case",
    "compute_l2_error",
    "compute_linfinity_error",
    "compute_flow_rate_error",
    "compute_max_velocity_error",
    "ConvergenceResult",
]
