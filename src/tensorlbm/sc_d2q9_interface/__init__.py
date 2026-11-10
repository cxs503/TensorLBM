"""SC/D2Q9 two-phase LBM with measurable interface tension.

This module implements a Shan-Chen pseudopotential two-phase lattice Boltzmann
method on the D2Q9 lattice with the following improvements over the base
:meth:`tensorlbm.multiphase` implementation:

1. **Mei-Luo-Shyy weighted isotropic gradient** (9-point and 13-point stencils)
   that eliminates directional bias in the interface force, reducing spurious
   currents by an order of magnitude compared to 4-point central differences.

2. **Momentum-flux pressure measurement** — the thermodynamic pressure is
   extracted from the diagonal of the momentum flux tensor Π_αα = Σᵢ fᵢ cᵢα cᵢα,
   not from the EOS approximation p = ρcs² + Gψ²/2.  This separates the true
   mechanical pressure from discretisation artefacts.

3. **Parasitic-flow acoustic SNR diagnostic** — the spurious velocity magnitude
   is compared against the capillary pressure Δp = σ/κ to ensure the signal
   (capillary pressure jump) dominates the noise (parasitic flow dynamic
   pressure ρ·cs·max|u_spurious|) by at least a factor of 3.

4. **Contact angle boundary condition** — wall pseudopotential ψ_w controls the
   static contact angle; a circle-fit estimator measures the angle from the
   mid-density level set with uncertainty quantification.

5. **Laplace three-radius self-consistency gate** — static droplets at
   R ∈ {20, 40, 80} lu are simulated; Δp vs 1/R must be linear with R² > 0.95
   and slope = σ within 5% of the input surface tension.

References
----------
Shan & Chen (1993) Phys. Rev. E 47, 1815.
Mei, Luo & Shyy (1997) J. Comput. Phys. 134, 246 (weighted isotropic gradient).
Wagner & Pohl (2004) Europhys. Lett. 65, 145 (pressure tensor in SC LBM).
Benzi et al. (2016) Phys. Rep. 654, 1 (contact angle in pseudopotential LBM).
"""

from __future__ import annotations

from .gradient import (
    gradient_9point_mls,
    gradient_13point_mls,
    laplacian_9point_isotropic,
)
from .pressure import (
    momentum_flux_pressure,
    pressure_jump_droplet,
)
from .sc_core import (
    SCConfig,
    SCState,
    psi_exp,
    collide_sc,
    compute_sc_force_mls,
    init_droplet,
    run_simulation,
)
from .diagnostics import (
    parasitic_flow_snr,
    measure_surface_tension_laplace,
    laplace_three_radius_gate,
    LaplaceResult,
    SNRResult,
)
from .contact_angle import (
    measure_contact_angle,
    ContactAngleResult,
)

__all__ = [
    # Gradient operators
    "gradient_9point_mls",
    "gradient_13point_mls",
    "laplacian_9point_isotropic",
    # Pressure measurement
    "momentum_flux_pressure",
    "pressure_jump_droplet",
    # SC core
    "SCConfig",
    "SCState",
    "psi_exp",
    "collide_sc",
    "compute_sc_force_mls",
    "init_droplet",
    "run_simulation",
    # Diagnostics
    "parasitic_flow_snr",
    "measure_surface_tension_laplace",
    "laplace_three_radius_gate",
    "LaplaceResult",
    "SNRResult",
    # Contact angle
    "measure_contact_angle",
    "ContactAngleResult",
]
