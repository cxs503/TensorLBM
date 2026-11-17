"""Moment-consistent order-parameter equilibrium for D2Q9 and D3Q19.

The ``lee_lin`` selector used by the public steppers denotes this local
Hermite construction; it does not claim implementation of every term in a
published Lee--Lin hydrodynamic scheme. In lattice units it satisfies
sum(g_eq)=phi, sum(c*g_eq)=phi*u, and
sum(cc*g_eq)=phi*uu + Gamma*mu*I. Thus M=Gamma*(tau_g-1/2).

In particular there is no cs^2*phi contribution to the second moment.
Keeping that contribution (the historical ``swift`` path) adds ordinary
diffusion even when Gamma=0. Neither construction alone supplies the
time-dependent advection correction needed by some high-order CH schemes.
"""

from __future__ import annotations

import math

import torch


def validate_convention(geq_convention: str) -> None:
    if geq_convention not in ("swift", "lee_lin"):
        raise ValueError("geq_convention must be 'swift' or 'lee_lin'")


def ch_equilibrium(
    phi: torch.Tensor,
    velocity: tuple[torch.Tensor, ...],
    mu: torch.Tensor | None = None,
    Gamma: float = 0.5,
) -> torch.Tensor:
    """Return CH equilibrium, with velocity components ordered x, y[, z].

    ``mu=None`` means zero chemical potential (not automatic equilibrium of
    the free-energy functional). Pass the model's discrete chemical potential
    for a well-prepared nonuniform initial state. Weights are constructed at
    the runtime dtype from binary64 constants, including in 3D.
    """
    dimension = phi.ndim
    if dimension not in (2, 3) or len(velocity) != dimension:
        raise ValueError("CH equilibrium requires 2-D or 3-D phi and matching velocity")
    if not phi.is_floating_point():
        raise ValueError("phi must have a floating dtype")
    if not math.isfinite(Gamma) or Gamma < 0:
        raise ValueError("Gamma must be finite and nonnegative")
    for value in (*velocity, *((mu,) if mu is not None else ())):
        if value.shape != phi.shape or value.dtype != phi.dtype or value.device != phi.device:
            raise ValueError("velocity and mu must match phi shape, dtype and device")
    if dimension == 2:
        from ..d2q9 import W_EXACT, C

        weights = torch.tensor(W_EXACT, device=phi.device, dtype=phi.dtype)
    else:
        from ..d3q19 import W_EXACT64, C

        weights = W_EXACT64.to(device=phi.device, dtype=phi.dtype)

    shape = (-1,) + (1,) * dimension
    c = C.to(device=phi.device, dtype=phi.dtype)
    w = weights.view(shape)
    cu = sum(c[:, axis].view(shape) * u for axis, u in enumerate(velocity))
    usq = sum(u * u for u in velocity)
    geq = w * phi * (3.0 * cu + 4.5 * cu**2 - 1.5 * usq)
    geq[0] = geq[0] + phi
    if mu is not None:
        # Isotropy: sum(w*lambda)=0; sum(w*lambda*c_a*c_b)=delta_ab.
        lam = 1.5 * (3.0 * (c * c).sum(1) - dimension)
        geq = geq + w * Gamma * lam.view(shape) * mu
    return geq
