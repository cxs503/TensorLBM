"""Adjoint / differentiable LBM with out-of-core (checkpointed) gradients.

This is the **importable** home of the differentiable-LBM machinery: the
package ships inside the installable tree (``src/tensorlbm``), so

>>> from tensorlbm.adjoint_ootc import ootc_gradient, bptt_gradient

works without any ``sys.path`` manipulation.  (Before this module moved
here the implementation lived in a nested ``TensorLBM/src/...`` shadow
tree that the installed package could not reach -- see
``docs/ADJOINT_OOTC_INVERSION_CONVERGENCE.md``.)

Provides
--------
- ``lbm2d``: D2Q9 BGK forward model (numpy, float64, deterministic)
- ``bptt``: full backpropagation-through-time reference gradient
- ``ootc``: out-of-core (checkpointed) adjoint gradient, memory O(k*N*Q)
  instead of O(T*N*Q)
- ``inversion``: long-horizon parameter inversion driver (safeguarded
  secant/Newton on the analytic gradient) that converges to the
  objective's stationary point

This package is distinct from :mod:`tensorlbm.adjoint`, which is the
frozen-field *surrogate* (``shape_derivative_only=True``) and must not
be used to drive an optimisation.
"""

from . import bptt, inversion, lbm2d, ootc
from .bptt import bptt_gradient
from .inversion import (
    fd_gradient,
    invert_tau,
    reference_from_tau,
    terminal_objective,
)
from .lbm2d import equilibrium, initial_pulse, macro, step
from .ootc import ootc_gradient

__all__ = [
    "lbm2d",
    "bptt",
    "ootc",
    "inversion",
    "macro",
    "equilibrium",
    "step",
    "initial_pulse",
    "bptt_gradient",
    "ootc_gradient",
    "invert_tau",
    "terminal_objective",
    "reference_from_tau",
    "fd_gradient",
]
