"""Adjoint / differentiable LBM module.

Provides:
- ``lbm2d``: D2Q9 BGK forward model (numpy, float64, deterministic)
- ``bptt``: Full backpropagation-through-time reference gradient
- ``ootc``: Out-of-core (checkpointed) adjoint gradient with O(T/k) memory

The OOTC gradient reproduces the BPTT reference up to floating-point
summation order (relative difference ~1e-15), while using only O(k·N·Q)
peak memory instead of O(T·N·Q).  See Griewank & Walther (2008) ch. 2
for the checkpointing theory.
"""

from .lbm2d import EX, EY, W, macro, equilibrium, collide, stream, step, initial_pulse
from .bptt import anti_stream, b_action, seed_terminal, bptt_gradient
from .ootc import ootc_gradient

__all__ = [
    # Forward model
    "EX", "EY", "W", "macro", "equilibrium", "collide", "stream", "step",
    "initial_pulse",
    # BPTT reference
    "anti_stream", "b_action", "seed_terminal", "bptt_gradient",
    # OOTC (out-of-core)
    "ootc_gradient",
]
