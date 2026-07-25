"""D3Q27 CUMULANT + Smagorinsky LES wrapper (domain-averaged eddy viscosity).

Since CUMULANT D3Q27 does not support per-cell τ, we compute the
mean Smagorinsky eddy viscosity from the non-equilibrium stress
norm and apply a single effective τ_eff to all cells.

This is a moderate-accuracy stabilisation that prevents the
pressure-field divergence seen in bare_hull at long integration
times while preserving the pure-friction drag mechanism of
D3Q27 CUMULANT.
"""

from __future__ import annotations

import torch

from .d3q27 import equilibrium27, macroscopic27
from .turbulence import _neq_stress_norm_27, _smagorinsky_tau


def collide_cumulant_smag_d3q27(
    f: torch.Tensor,
    tau: float,
    C_s: float = 0.05,
    omega_b: float = 1.0,
    omega_odd: float = 1.0,
    omega_even: float = 1.0,
) -> torch.Tensor:
    """D3Q27 CUMULANT collision with domain-averaged Smagorinsky LES.

    Args:
        f: Distribution tensor of shape ``(27, nz, ny, nx)``.
        tau: Molecular relaxation time.
        C_s: Smagorinsky constant (default 0.05).
        omega_b, omega_odd, omega_even: CUMULANT relaxation parameters.

    Returns:
        Updated distribution tensor.
    """
    from .cumulant import collide_cumulant_d3q27

    if C_s <= 0:
        return collide_cumulant_d3q27(f, tau, omega_b, omega_odd, omega_even)

    # Compute domain-averaged effective τ
    rho, ux, uy, uz = macroscopic27(f)
    feq = equilibrium27(rho, ux, uy, uz)
    f_neq = f - feq
    pi_norm = _neq_stress_norm_27(f_neq)  # (nz, ny, nx)
    tau_eff_per_cell = _smagorinsky_tau(tau, pi_norm, rho, C_s)

    # Mean eddy-viscosity τ for entire domain
    tau_eff = float(tau_eff_per_cell.mean().item())
    # Clamp to [tau, tau*10] — prevent runaway
    tau_eff = max(tau, min(tau_eff, tau * 10.0))

    return collide_cumulant_d3q27(f, tau_eff, omega_b, omega_odd, omega_even)
