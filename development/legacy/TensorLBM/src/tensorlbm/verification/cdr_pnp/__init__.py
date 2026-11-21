"""CDR (Convection-Diffusion-Reaction) + PNP (Poisson-Nernst-Planck) verification face.

This module provides:
- CDR: operator-splitting LBM transport + RK4 reaction for Da number scanning
- PNP: Poisson solver + Nernst-Planck ion transport + electro-osmotic flow
- Analytical reference solutions for verification
- Machine-checkable gates (C1-C7)

Physics
-------
CDR equation: ∂c/∂t + u·∇c = D∇²c + k·c^n

PNP system:
  Poisson:    ∇²ψ = -ρ_e/ε = -(F/ε) Σ z_i c_i
  Nernst-Planck: ∂c_i/∂t + u·∇c_i = ∇·[D_i(∇c_i + (z_i F/RT) c_i ∇ψ)]
  Navier-Stokes: ρ(∂u/∂t + u·∇u) = -∇p + μ∇²u + ρ_e E

References
----------
- Damköhler number analysis: Levenspiel (1999), Chemical Reaction Engineering
- Gouy-Chapman double layer: Hunter (1981), Zeta Potential in Colloid Science
- Henry function: Henry (1931), Proc. R. Soc. Lond. A 133, 106
- Ohshima (1994), J. Colloid Interface Sci. 168, 285
"""

from __future__ import annotations

from .cdr_solver import (
    CDRConfig,
    rk4_reaction_step,
    analytical_cdr_1d,
    boundary_layer_thickness,
    run_cdr_1d,
)
from .pnp_solver import (
    PNPConfig,
    gouy_chapman_psi,
    debye_length,
    helmholtz_smoluchowski_velocity,
    henry_function,
    solve_poisson_1d,
    nernst_planck_step,
    run_pnp_edl,
    run_electroosmosis,
    run_electrophoresis,
)
from .gates import (
    GATE_THRESHOLDS,
    check_c1_da1_concentration,
    check_c2_debye_length,
    check_c3_electroosmosis,
    check_c4_henry,
    check_c5_conservation,
    check_c6_negative_controls,
)

__all__ = [
    "CDRConfig",
    "PNPConfig",
    "GATE_THRESHOLDS",
    "rk4_reaction_step",
    "analytical_cdr_1d",
    "boundary_layer_thickness",
    "run_cdr_1d",
    "gouy_chapman_psi",
    "debye_length",
    "helmholtz_smoluchowski_velocity",
    "henry_function",
    "solve_poisson_1d",
    "nernst_planck_step",
    "run_pnp_edl",
    "run_electroosmosis",
    "run_electrophoresis",
    "check_c1_da1_concentration",
    "check_c2_debye_length",
    "check_c3_electroosmosis",
    "check_c4_henry",
    "check_c5_conservation",
    "check_c6_negative_controls",
]
