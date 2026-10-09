"""Bouzidi interpolated BC q-field computation for the SUBOFF hull (D3Q27).

D3Q27 counterpart of :func:`tensorlbm.interpolated_bc_suboff.compute_q_suboff`:
computes the fractional wall-distance *q* for every D3Q27 lattice link
crossing the SUBOFF bare-hull surface via bisection ray-marching against the
analytical axisymmetric profile.

The geometry predicates are shared with the D3Q19 module (the SUBOFF radius
profile and the inside-hull test are lattice-independent); only the velocity
set, the number of channels and the output shapes differ (27 instead of 19).

Reference
---------
Bouzidi, M., Firdaouss, M., & Lallemand, P. (2001).
"Momentum transfer of a Boltzmann-lattice fluid with boundaries."
*Physics of Fluids*, 13(11), 3452–3459.

Groves, N.C., Huang, T.T., Chang, M.S. (1989).
"Geometric Characteristics of DARPA SUBOFF Models", DTRC/SHD-1298-01.
"""

from __future__ import annotations

import torch

from .d3q27 import C as C27
from .interpolated_bc_suboff import _inside_hull
from .suboff_cad import SuboffConfig, SuboffHullType, build_suboff_mask

__all__ = ["compute_q_suboff_27"]


def compute_q_suboff_27(
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
    solid_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the BFL fractional-distance field *q* for a SUBOFF hull (D3Q27).

    Identical algorithm to :func:`compute_q_suboff` (D3Q19) — for every fluid
    node adjacent to the SUBOFF surface, bisection ray-marches along each of
    the 27 D3Q27 lattice links against the analytical axisymmetric radius
    profile to find the fractional distance *q ∈ (0, 1]* to the hull surface.
    The 8 corner links (|c| = √3) of D3Q27 are included.

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
        SUBOFF variant: ``\"bare_hull\"``, ``\"with_sail\"``, or ``\"full\"``.
    config : SuboffConfig, optional
        Parametric geometry; uses :class:`SuboffConfig` defaults when *None*.
    device : torch.device or str
        PyTorch device.
    n_bisect : int
        Number of bisection iterations (10 → ~1/1024 lu precision).
    solid_mask : torch.Tensor, optional
        Existing boolean SUBOFF mask with shape ``(nz, ny, nx)``.  Reusing the
        solver's CAD mask avoids a second full-domain geometry construction.

    Returns
    -------
    fluid_boundary_mask : torch.Tensor of bool, shape (27, nz, ny, nx)
        True where fluid node (k,j,i) has the hull boundary in direction d.
    q_field : torch.Tensor of float32, shape (27, nz, ny, nx)
        Fractional distance q for each (direction, fluid node) pair.
        Non-boundary entries are 0.5.
    """
    if isinstance(device, str):
        device = torch.device(device)
    if config is None:
        config = SuboffConfig()

    radius = config.r_over_l * hull_length
    x_bow = cx - hull_length / 2.0

    c = C27.to(device)  # (27, 3)

    # ---- Build solid mask once (on device) ----
    if solid_mask is None:
        hull_type_enum = SuboffHullType(hull_type)
        solid, _stats = build_suboff_mask(
            hull_type_enum,
            nx=nx,
            ny=ny,
            nz=nz,
            cx=cx,
            cy=cy,
            cz=cz,
            length=hull_length,
            config=config,
            device=str(device),
        )
        solid = solid.to(device)
    else:
        if solid_mask.shape != (nz, ny, nx) or solid_mask.dtype != torch.bool:
            raise ValueError(
                "solid_mask must be boolean with shape (nz, ny, nx)",
            )
        solid = solid_mask.to(device=device)

    fluid_boundary_mask = torch.zeros(
        (27, nz, ny, nx),
        dtype=torch.bool,
        device=device,
    )
    q_field = torch.full(
        (27, nz, ny, nx),
        0.5,
        dtype=torch.float32,
        device=device,
    )

    for d in range(27):
        dcx = float(c[d, 0].item())
        dcy = float(c[d, 1].item())
        dcz = float(c[d, 2].item())
        if dcx == 0.0 and dcy == 0.0 and dcz == 0.0:
            continue  # rest direction

        # ---- Identify boundary links ----
        # Tensor storage is (z, y, x), while D3Q27 vectors are (x, y, z).
        # Keep the components in their physical axes and only reorder them
        # when forming the torch-roll tuple (same convention as the D3Q19
        # compute_q_suboff).
        nb_solid = torch.roll(
            solid,
            shifts=(-int(dcz), -int(dcy), -int(dcx)),
            dims=(0, 1, 2),
        )
        boundary = ~solid & nb_solid  # (nz, ny, nx)

        if not boundary.any():
            continue

        # ---- Extract indices of boundary cells (on CPU for indexing) ----
        idx = boundary.nonzero(as_tuple=False)  # (N, 3) → [k, j, i]
        n_cells = idx.shape[0]

        # Fluid cell coordinates (float).  Ten bisections only resolve q to
        # about 1e-3 lattice units, so FP32 coordinates retain ample margin
        # while avoiding very slow consumer-GPU FP64 preprocessing.
        k_f = idx[:, 0].to(dtype=torch.float32, device=device)
        j_f = idx[:, 1].to(dtype=torch.float32, device=device)
        i_f = idx[:, 2].to(dtype=torch.float32, device=device)

        endpoint_in_main_body = _inside_hull(
            i_f + dcx,
            j_f + dcy,
            k_f + dcz,
            x_bow,
            cy,
            cz,
            hull_length,
            radius,
            config,
        )

        # ---- Bisection on boundary cells only ----
        t_lo = torch.zeros(n_cells, dtype=torch.float32, device=device)
        t_hi = torch.ones(n_cells, dtype=torch.float32, device=device)

        for _ in range(n_bisect):
            t_mid = (t_lo + t_hi) * 0.5

            x_mid = i_f + t_mid * dcx
            y_mid = j_f + t_mid * dcy
            z_mid = k_f + t_mid * dcz

            inside = _inside_hull(
                x_mid,
                y_mid,
                z_mid,
                x_bow,
                cy,
                cz,
                hull_length,
                radius,
                config,
            )

            # If inside → surface is closer → lower hi
            # If outside → surface is further → raise lo
            t_lo = torch.where(~inside, t_mid, t_lo)
            t_hi = torch.where(inside, t_mid, t_hi)

        # Final q
        q = ((t_lo + t_hi) * 0.5).clamp(1e-6, 1.0).float()
        # Sail and control surfaces are voxelised rather than described by
        # the axisymmetric profile.  Their solid endpoint is therefore not
        # inside the main-body implicit function; use standard half-way BB
        # on those links instead of a spurious q≈1 result.
        q = torch.where(endpoint_in_main_body, q, torch.full_like(q, 0.5))

        # ---- Scatter back to full-size tensor ----
        fluid_boundary_mask[d, k_f.long(), j_f.long(), i_f.long()] = True
        q_field[d, k_f.long(), j_f.long(), i_f.long()] = q

    return fluid_boundary_mask, q_field
