"""CDR (Convection-Diffusion-Reaction) solver via operator splitting.

Implements: ∂c/∂t + u·∇c = D∇²c - k·c^n  (reaction as SINK)

Operator splitting:
  1. Transport step: implicit Euler for advection-diffusion
  2. Reaction step: RK4 for ODE dc/dt = -k·c^n (decay)

Damköhler number: Da = k·L²/D (reaction rate / diffusion rate)
For Da << 1: diffusion-dominated, nearly uniform concentration
For Da >> 1: reaction-dominated, thin boundary layer δ ~ √(D/k)

Analytical solution for n=1, u=0, steady state:
  d²c/dx² - (k/D)·c = 0
  c(x) = c₀·cosh(√(k/D)·(L-x)) / cosh(√(k/D)·L)
  BC: c(0)=c₀, dc/dx(L)=0
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

__all__ = [
    "CDRConfig",
    "rk4_reaction_step",
    "analytical_cdr_1d",
    "boundary_layer_thickness",
    "run_cdr_1d",
]


@dataclass(frozen=True)
class CDRConfig:
    nx: int = 256
    L: float = 1.0
    D: float = 0.01
    u: float = 0.0
    k: float = 1.0          # reaction rate (SINK: -k·c^n)
    n: int = 1
    c0: float = 1.0
    steps: int = 10000
    dt: float = 0.001
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    @property
    def dx(self) -> float:
        return self.L / (self.nx - 1)


def analytical_cdr_1d(
    x: np.ndarray, D: float, k: float,
    c0: float = 1.0, u: float = 0.0, n: int = 1, L: float = 1.0,
) -> np.ndarray:
    """Steady-state analytical solution for 1D diffusion-reaction (n=1, u=0).

    d²c/dx² - (k/D)·c = 0, c(0)=c₀, dc/dx(L)=0
    """
    if n != 1:
        raise NotImplementedError("Analytical solution only for n=1")
    x = np.asarray(x, dtype=np.float64)
    if u != 0.0:
        raise NotImplementedError("Analytical solution only for u=0")
    lam = math.sqrt(k / D) if k > 0 else 0.0
    if lam < 1e-15:
        return np.full_like(x, c0)
    return c0 * np.cosh(lam * (L - x)) / np.cosh(lam * L)


def boundary_layer_thickness(D: float, k: float) -> float:
    """δ = √(D/k) for first-order reaction."""
    if k <= 0:
        return float('inf')
    return math.sqrt(D / k)


def rk4_reaction_step(c: torch.Tensor, k: float, n: int, dt: float) -> torch.Tensor:
    """RK4 for dc/dt = -k·c^n (decay reaction)."""
    def rhs(cv): return -k * cv ** n
    k1 = rhs(c)
    k2 = rhs(c + 0.5 * dt * k1)
    k3 = rhs(c + 0.5 * dt * k2)
    k4 = rhs(c + dt * k3)
    return c + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _build_transport_matrix(nx, u, D, dx, dt, c0):
    n = nx - 2
    r_d = D * dt / (dx * dx)
    r_a = u * dt / (2.0 * dx)
    main = np.full(n, 1.0 + 2.0 * r_d)
    lower = np.full(n - 1, -r_d - r_a)
    upper = np.full(n - 1, -r_d + r_a)
    main[-1] += (-r_d + r_a)  # Neumann BC
    A = diags([lower, main, upper], [-1, 0, 1], shape=(n, n), format='csc')
    rhs_correction = np.zeros(n)
    rhs_correction[0] = (r_d + r_a) * c0
    return A, rhs_correction


def run_cdr_1d(config: CDRConfig, reaction_every: int = 1) -> dict:
    nx = config.nx
    dx = config.dx
    L = config.L
    D = config.D
    k = config.k
    n_order = config.n
    c0 = config.c0
    u_val = config.u
    dt = config.dt
    steps = config.steps

    x = np.linspace(0, L, nx)
    c = c0 * (1.0 - x / L)

    A, rhs_corr = _build_transport_matrix(nx, u_val, D, dx, dt, c0)
    mass_initial = np.sum(c) * dx

    c_snapshots = []
    snapshot_steps = {0, steps // 4, steps // 2, 3 * steps // 4, steps - 1}

    for step in range(steps):
        # Transport step
        rhs = c[1:-1] + rhs_corr
        c_interior = spsolve(A, rhs)
        c[0] = c0
        c[1:-1] = c_interior
        c[-1] = c[-2]

        # Reaction step (decay: dc/dt = -k·c^n)
        if step % reaction_every == 0 and k > 0:
            c_torch = torch.from_numpy(c.copy())
            c_torch = rk4_reaction_step(c_torch, k, n_order, dt)
            c = c_torch.numpy()
            c[0] = c0

        if step in snapshot_steps:
            c_snapshots.append(c.copy())

    mass_final = np.sum(c) * dx
    mass_residual = abs(mass_final - mass_initial) / max(mass_initial, 1e-15)

    return {
        "x": x, "c": c,
        "c_snapshots": c_snapshots,
        "mass_initial": mass_initial, "mass_final": mass_final,
        "mass_residual": mass_residual, "config": config,
    }
