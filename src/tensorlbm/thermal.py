"""Thermal lattice Boltzmann method using the double-distribution-function (DDF) model.

Implements the passive-scalar thermal LBM (He *et al.* 1998, Peng *et al.*
2003) in which a second set of distribution functions *g* evolves on a D2Q5
lattice and carries the temperature field.  The momentum D2Q9 solver is
extended with a buoyancy body force (Boussinesq approximation) coupling the
temperature back to the velocity field.

Overview
--------
The momentum solver (D2Q9) is unchanged — any existing BGK/MRT/TRT collider
can be used.  The thermal step adds:

1. :func:`equilibrium_thermal` — D2Q5 equilibrium for the temperature.
2. :func:`collide_thermal_bgk` — BGK collision for the temperature distribution.
3. :func:`stream_thermal` — streaming step on the D2Q5 lattice.
4. :func:`macroscopic_thermal` — recover temperature T from *g*.
5. :func:`apply_buoyancy_force` — add the Boussinesq buoyancy force to *f*.

The D2Q5 lattice
----------------
Velocity set (cx, cy):

    0: ( 0,  0)   weight 2/6
    1: ( 1,  0)   weight 1/6
    2: ( 0,  1)   weight 1/6
    3: (-1,  0)   weight 1/6
    4: ( 0, -1)   weight 1/6

The D2Q5 equilibrium is:

    g_i^eq = w_i * T * (1 + 3 * (cx_i * ux + cy_i * uy))

(The convection velocity is the local macroscopic velocity from the D2Q9 field.)

References
----------
He, X., Chen, S., & Doolen, G. D. (1998).
    A novel thermal model for the lattice Boltzmann method in incompressible
    limit. *J. Comput. Phys.* 146(1), 282–300.
Peng, Y., Shu, C., & Chew, Y. T. (2003).
    Simplified thermal lattice Boltzmann model for incompressible thermal
    flows. *Phys. Rev. E* 68, 026701.
"""
from __future__ import annotations

import functools
from typing import Any

import torch

__all__ = [
    "C_D2Q5",
    "W_D2Q5",
    "equilibrium_thermal",
    "collide_thermal_bgk",
    "stream_thermal",
    "macroscopic_thermal",
    "apply_buoyancy_force",
]

# ---------------------------------------------------------------------------
# D2Q5 lattice constants
# ---------------------------------------------------------------------------

C_D2Q5 = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
    dtype=torch.int64,
)

W_D2Q5 = torch.tensor(
    [2.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
    dtype=torch.float32,
)

OPPOSITE_D2Q5 = torch.tensor([0, 3, 4, 1, 2], dtype=torch.int64)

# Module-level streaming cache
_stream_thermal_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}


@functools.cache
def _c_thermal(device: torch.device) -> torch.Tensor:
    return C_D2Q5.to(device)


@functools.cache
def _w_thermal(device: torch.device) -> torch.Tensor:
    return W_D2Q5.to(device)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def equilibrium_thermal(
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """Compute the D2Q5 equilibrium temperature distribution.

    The equilibrium function encodes the temperature convected at the
    macroscopic velocity (ux, uy):

        g_i^eq = w_i · T · (1 + 3·(cx_i·ux + cy_i·uy))

    Args:
        T:   Temperature field, shape ``(ny, nx)``.
        ux:  x-velocity, shape ``(ny, nx)`` (from D2Q9 macroscopic).
        uy:  y-velocity, shape ``(ny, nx)`` (from D2Q9 macroscopic).

    Returns:
        Equilibrium distribution *g_eq*, shape ``(5, ny, nx)``.
    """
    device = T.device
    c = _c_thermal(device)
    w = _w_thermal(device).view(5, 1, 1)
    cx = c[:, 0].view(5, 1, 1).float()
    cy = c[:, 1].view(5, 1, 1).float()
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    return w * T.unsqueeze(0) * (1.0 + 3.0 * cu)


def collide_thermal_bgk(
    g: torch.Tensor,
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau_T: float,
) -> torch.Tensor:
    """BGK collision step for the D2Q5 temperature distribution.

    The thermal diffusivity is α = (τ_T − ½) / 3.

    Args:
        g:     Temperature distribution, shape ``(5, ny, nx)``.
        T:     Macroscopic temperature, shape ``(ny, nx)``.
        ux:    x-velocity (from D2Q9), shape ``(ny, nx)``.
        uy:    y-velocity (from D2Q9), shape ``(ny, nx)``.
        tau_T: Thermal relaxation time (τ_T > 0.5).

    Returns:
        Post-collision distribution, shape ``(5, ny, nx)``.
    """
    geq = equilibrium_thermal(T, ux, uy)
    return g - (g - geq) / tau_T


def stream_thermal(g: torch.Tensor) -> torch.Tensor:
    """Streaming step for the D2Q5 temperature distribution (periodic).

    Uses cached index tensors for efficiency.

    Args:
        g: Temperature distribution, shape ``(5, ny, nx)``.

    Returns:
        Streamed distribution of the same shape.
    """
    ny, nx = g.shape[1], g.shape[2]
    device = g.device
    c = _c_thermal(device)

    cache_key = (ny, nx, device.type, device.index)
    if cache_key not in _stream_thermal_cache:
        y_src = (torch.arange(ny, device=device).unsqueeze(0) - c[:, 1].unsqueeze(1)) % ny
        x_src = (torch.arange(nx, device=device).unsqueeze(0) - c[:, 0].unsqueeze(1)) % nx
        q_idx = torch.arange(5, device=device).view(5, 1, 1).expand(5, ny, nx)
        y_idx = y_src.unsqueeze(2).expand(5, ny, nx)
        x_idx = x_src.unsqueeze(1).expand(5, ny, nx)
        _stream_thermal_cache[cache_key] = (q_idx, y_idx, x_idx)

    q_idx, y_idx, x_idx = _stream_thermal_cache[cache_key]
    return g[q_idx, y_idx, x_idx]


def macroscopic_thermal(g: torch.Tensor) -> torch.Tensor:
    """Recover the macroscopic temperature from the D2Q5 distributions.

    T = Σ_i g_i

    Args:
        g: Temperature distribution, shape ``(5, ny, nx)``.

    Returns:
        Temperature field, shape ``(ny, nx)``.
    """
    return g.sum(dim=0)


def apply_buoyancy_force(
    f: torch.Tensor,
    T: torch.Tensor,
    T_ref: float,
    beta: float,
    g_y: float = -1.0,
) -> torch.Tensor:
    """Apply the Boussinesq buoyancy body force to the D2Q9 distribution.

    The Boussinesq approximation gives a body force in the y-direction:

        F_y = ρ · β · (T − T_ref) · g_y

    where β is the thermal expansion coefficient and g_y is the gravitational
    acceleration (negative = downward in lattice coordinates).

    The force is applied using the first-order Guo scheme:

        f_i ← f_i + w_i · 3 · cy_i · F_y

    Args:
        f:      D2Q9 distribution tensor, shape ``(9, ny, nx)``.
        T:      Temperature field, shape ``(ny, nx)``.
        T_ref:  Reference (mean) temperature.
        beta:   Thermal expansion coefficient.
        g_y:    Gravitational acceleration in y (default −1, dimensionless
                lattice units).

    Returns:
        Updated D2Q9 distribution, shape ``(9, ny, nx)``.
    """
    from .d2q9 import C, W, macroscopic

    device = f.device
    c = C.to(device).float()
    w = W.to(device).float()

    rho, _, _ = macroscopic(f)
    F_y = rho * beta * (T - T_ref) * g_y

    cy = c[:, 1].view(9, 1, 1)
    w_view = w.view(9, 1, 1)
    return f + w_view * 3.0 * cy * F_y.unsqueeze(0)
