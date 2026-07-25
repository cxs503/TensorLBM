"""Bouzidi interpolated BC q-field computation for SUBOFF submarine hull.

Computes the fractional wall-distance *q* for every D3Q19 lattice link
crossing the SUBOFF bare-hull surface via bisection ray-marching against
the analytical axisymmetric profile.

Reference
---------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001).
"Momentum transfer of a Boltzmann-lattice fluid with boundaries."
*Physics of Fluids*, 13(11), 3452–3459.

Groves, N.C., Huang, T.T., Chang, M.S. (1989).
"Geometric Characteristics of DARPA SUBOFF Models", DTRC/SHD-1298-01.
"""
from __future__ import annotations

import math

import torch

from .d3q19 import C as C3D
from .suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask


# ---------------------------------------------------------------------------
# PyTorch implementation of the normalised SUBOFF radius profile
# ---------------------------------------------------------------------------

def _suboff_radius_norm_torch(
    xi: torch.Tensor, config: SuboffConfig,
) -> torch.Tensor:
    """Normalised hull radius r(xi)/R_max for xi ∈ [0,1] (PyTorch, autograd-safe).

    Parameters
    ----------
    xi : torch.Tensor
        Normalised axial coordinate ∈ [0, 1].
    config : SuboffConfig
        Geometry configuration.

    Returns
    -------
    torch.Tensor
        Normalised radius, same shape as *xi*.
    """
    alpha = config.bow_fraction
    beta = config.stern_fraction
    n = config.stern_exponent

    bow_mask = (xi >= 0.0) & (xi < alpha)
    mid_mask = (xi >= alpha) & (xi <= 1.0 - beta)
    stern_mask = (xi > 1.0 - beta) & (xi <= 1.0)

    xi_bow = xi / alpha
    bow_r = torch.sqrt(torch.clamp(2.0 * xi_bow - xi_bow**2, 0.0, 1.0))

    eta = (xi - (1.0 - beta)) / beta
    if n == 2.0:
        stern_r = torch.sqrt(torch.clamp(1.0 - eta**2, 0.0, 1.0))
    else:
        stern_r = torch.clamp(1.0 - eta**n, 0.0, 1.0) ** (1.0 / n)

    r = torch.where(
        bow_mask, bow_r,
        torch.where(
            mid_mask, torch.ones_like(xi),
            torch.where(stern_mask, stern_r, torch.zeros_like(xi)),
        ),
    )
    return torch.clamp(r, 0.0, 1.0)


def _inside_hull(
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    x_bow: float,
    cy: float,
    cz: float,
    hull_length: float,
    radius: float,
    config: SuboffConfig,
) -> torch.Tensor:
    """Test whether points (x, y, z) are inside the SUBOFF bare hull.

    All inputs are 1-D float64 tensors of the same length.
    Returns a bool tensor.
    """
    xi = (x - x_bow) / hull_length
    in_axial = (xi >= 0.0) & (xi <= 1.0)
    r_norm = _suboff_radius_norm_torch(xi, config)
    r_max_lu = r_norm * radius
    r_dist = torch.sqrt((y - cy) ** 2 + (z - cz) ** 2)
    return in_axial & (r_dist <= r_max_lu)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_q_suboff(
    nx: int,
    ny: int,
    nz: int,
    cx: float,
    cy: float,
    cz: float,
    hull_length: float,
    hull_type: str = "bare_hull",
    config: SuboffConfig | None = None,
    device: torch.device | str = "cpu",
    n_bisect: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the BFL fractional-distance field *q* for a SUBOFF hull (D3Q19).

    For every fluid node adjacent to the SUBOFF surface, computes the
    fractional distance *q ∈ (0, 1]* along each D3Q19 lattice link to the
    hull surface via bisection ray-marching against the analytical
    axisymmetric radius profile.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions (x = axial/flow, y = transverse, z = vertical).
    cx, cy, cz : float
        Hull axis midpoint (cells).  Same convention as
        :func:`~tensorlbm.suboff_cad.build_suboff_mask`.
    hull_length : float
        Total hull length (lattice units).
    hull_type : str
        SUBOFF variant: ``"bare_hull"``, ``"with_sail"``, or ``"full"``.
    config : SuboffConfig, optional
        Parametric geometry; uses :class:`SuboffConfig` defaults when *None*.
    device : torch.device or str
        PyTorch device.
    n_bisect : int
        Number of bisection iterations (10 → ~1/1024 lu precision).

    Returns
    -------
    fluid_boundary_mask : torch.Tensor of bool, shape (19, nz, ny, nx)
        True where fluid node (k,j,i) has the hull boundary in direction d.
    q_field : torch.Tensor of float32, shape (19, nz, ny, nx)
        Fractional distance q for each (direction, fluid node) pair.
        Non-boundary entries are 0.5.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if config is None:
        config = SuboffConfig()

    radius = config.r_over_l * hull_length
    x_bow = cx - hull_length / 2.0

    c = C3D.to(device)  # (19, 3)

    # ---- Build solid mask once (on device) ----
    hull_type_enum = SuboffHullType(hull_type)
    solid, _stats = build_suboff_mask(
        hull_type_enum,
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=hull_length,
        config=config,
        device=str(device),
    )
    solid = solid.to(device)

    fluid_boundary_mask = torch.zeros(
        (19, nz, ny, nx), dtype=torch.bool, device=device,
    )
    q_field = torch.full(
        (19, nz, ny, nx), 0.5, dtype=torch.float32, device=device,
    )

    for d in range(19):
        dcx = float(c[d, 0].item())
        dcy = float(c[d, 1].item())
        dcz = float(c[d, 2].item())
        if dcx == 0.0 and dcy == 0.0 and dcz == 0.0:
            continue  # rest direction

        # ---- Identify boundary links ----
        sx, sy, sz = int(dcz), int(dcy), int(dcx)
        nb_solid = torch.roll(solid, shifts=(-sz, -sy, -sx), dims=(0, 1, 2))
        boundary = ~solid & nb_solid  # (nz, ny, nx)

        if not boundary.any():
            continue

        # ---- Extract indices of boundary cells (on CPU for indexing) ----
        # Use nonzero to get indices, then move back to device for coordinates
        idx = boundary.nonzero(as_tuple=False)  # (N, 3) → [k, j, i]
        n_cells = idx.shape[0]

        # Fluid cell coordinates (float)
        k_f = idx[:, 0].to(dtype=torch.float64, device=device)
        j_f = idx[:, 1].to(dtype=torch.float64, device=device)
        i_f = idx[:, 2].to(dtype=torch.float64, device=device)

        # ---- Bisection on boundary cells only ----
        t_lo = torch.zeros(n_cells, dtype=torch.float64, device=device)
        t_hi = torch.ones(n_cells, dtype=torch.float64, device=device)

        for _ in range(n_bisect):
            t_mid = (t_lo + t_hi) * 0.5

            x_mid = i_f + t_mid * dcx
            y_mid = j_f + t_mid * dcy
            z_mid = k_f + t_mid * dcz

            inside = _inside_hull(
                x_mid, y_mid, z_mid,
                x_bow, cy, cz, hull_length, radius, config,
            )

            # If inside → surface is closer → lower hi
            # If outside → surface is further → raise lo
            t_lo = torch.where(~inside, t_mid, t_lo)
            t_hi = torch.where(inside, t_mid, t_hi)

        # Final q
        q = ((t_lo + t_hi) * 0.5).clamp(1e-6, 1.0).float()

        # ---- Scatter back to full-size tensor ----
        fluid_boundary_mask[d, k_f.long(), j_f.long(), i_f.long()] = True
        q_field[d, k_f.long(), j_f.long(), i_f.long()] = q

    return fluid_boundary_mask, q_field


__all__ = ["compute_q_suboff"]
