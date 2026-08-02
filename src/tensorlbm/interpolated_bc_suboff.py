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

from dataclasses import asdict, dataclass

import torch

from .d3q19 import C as C3D
from .suboff_cad import (
    SuboffConfig,
    SuboffHullType,
    build_suboff_mask,
    suboff_appendages_contain_points,
)

SUBOFF_APPENDAGE_LINK_SCHEME = "continuous_parametric_bisection_v1"


@dataclass(frozen=True, slots=True)
class SuboffAppendageLinkDiagnostics:
    scheme: str
    target_links: int
    n_bisect: int
    minimum_q: float | None
    maximum_q: float | None

    def to_dict(self) -> dict[str, str | int | float | None]:
        return asdict(self)

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
    # Keep this device-native implementation algebraically identical to
    # suboff_cad.suboff_radius_profile.  The previous ellipsoid surrogate
    # produced q-values for a different body than build_suboff_mask(), which
    # invalidated BFL resistance runs even though their voxel mask was real.
    length_ft = 14.291667
    bow_end = 3.333333 / length_ft
    mid_end = 10.645833 / length_ft
    stern_end = 13.979167 / length_ft

    bow_mask = (xi >= 0.0) & (xi < bow_end)
    mid_mask = (xi >= bow_end) & (xi <= mid_end)
    stern_mask = (xi > mid_end) & (xi <= stern_end)
    tail_mask = (xi > stern_end) & (xi <= 1.0)

    x_ft = xi * length_ft

    tmp = 0.3 * x_ft - 1.0
    tmp2 = tmp * tmp
    tmp4 = tmp2 * tmp2
    bow_poly = (
        1.126395101 * x_ft * tmp4
        + 0.442874707 * x_ft * x_ft * (tmp2 * tmp)
        + 1.0
        - tmp4 * (1.2 * x_ft + 1.0)
    )
    bow_r = torch.clamp(bow_poly, min=0.0).pow(1.0 / 2.1)

    r1, k0, k1 = 0.1175, 10.0, 44.6244
    ksi = (13.979167 - x_ft) / 3.333333
    ksi2 = ksi * ksi
    ksi3 = ksi2 * ksi
    ksi4 = ksi3 * ksi
    ksi5 = ksi4 * ksi
    ksi6 = ksi5 * ksi
    stern_poly = (
        r1 * r1
        + r1 * k0 * ksi2
        + (20.0 - 20.0 * r1 * r1 - 4.0 * r1 * k0 - k1 / 3.0) * ksi3
        + (-45.0 + 45.0 * r1 * r1 + 6.0 * r1 * k0 + k1) * ksi4
        + (36.0 - 36.0 * r1 * r1 - 4.0 * r1 * k0 - k1) * ksi5
        + (-10.0 + 10.0 * r1 * r1 + r1 * k0 + k1 / 3.0) * ksi6
    )
    stern_r = torch.sqrt(torch.clamp(stern_poly, min=0.0))

    tail_poly = 1.0 - (3.2 * x_ft - 44.733333) ** 2
    tail_r = 0.1175 * torch.sqrt(torch.clamp(tail_poly, min=0.0))

    r = torch.where(
        bow_mask, bow_r,
        torch.where(
            mid_mask, torch.ones_like(xi),
            torch.where(
                stern_mask, stern_r,
                torch.where(tail_mask, tail_r, torch.zeros_like(xi)),
            ),
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
    solid_mask: torch.Tensor | None = None,
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
    solid_mask : torch.Tensor, optional
        Existing boolean SUBOFF mask with shape ``(nz, ny, nx)``.  Reusing the
        solver's CAD mask avoids a second full-domain geometry construction.

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
    if solid_mask is None:
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
    else:
        if solid_mask.shape != (nz, ny, nx) or solid_mask.dtype != torch.bool:
            raise ValueError(
                "solid_mask must be boolean with shape (nz, ny, nx)",
            )
        solid = solid_mask.to(device=device)

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
        # Tensor storage is (z, y, x), while D3Q19 vectors are (x, y, z).
        # Keep the components in their physical axes and only reorder them
        # when forming the torch-roll tuple.  The former double permutation
        # made x-directed BFL masks inspect z-neighbours (and vice versa).
        nb_solid = torch.roll(
            solid,
            shifts=(-int(dcz), -int(dcy), -int(dcx)),
            dims=(0, 1, 2),
        )
        boundary = ~solid & nb_solid  # (nz, ny, nx)

        if not boundary.any():
            continue

        # ---- Extract indices of boundary cells (on CPU for indexing) ----
        # Use nonzero to get indices, then move back to device for coordinates
        idx = boundary.nonzero(as_tuple=False)  # (N, 3) → [k, j, i]
        n_cells = idx.shape[0]

        # Fluid cell coordinates (float)
        # Ten bisections only resolve q to about 1e-3 lattice units, so FP32
        # coordinates retain ample margin while avoiding very slow consumer-
        # GPU FP64 execution during geometry preprocessing.
        k_f = idx[:, 0].to(dtype=torch.float32, device=device)
        j_f = idx[:, 1].to(dtype=torch.float32, device=device)
        i_f = idx[:, 2].to(dtype=torch.float32, device=device)

        endpoint_in_main_body = _inside_hull(
            i_f + dcx, j_f + dcy, k_f + dcz,
            x_bow, cy, cz, hull_length, radius, config,
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
                x_mid, y_mid, z_mid,
                x_bow, cy, cz, hull_length, radius, config,
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


def refine_q_suboff_appendages(
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    solid: torch.Tensor,
    bare_hull: torch.Tensor,
    *,
    center: tuple[float, float, float],
    length: float,
    n_bisect: int = 12,
) -> tuple[torch.Tensor, SuboffAppendageLinkDiagnostics]:
    """Replace AFF-8 halfway links by continuous parametric intersections.

    The endpoint mask and the bisection predicate share the DARPA sail and
    swept-NACA fin equations.  Every selected link starts at a fluid lattice
    node and ends in an appendage-only solid node, so bisection yields the
    first fluid-to-solid fraction without an empirical q correction.
    """
    if (
        fluid_boundary_mask.ndim != 4
        or fluid_boundary_mask.shape[0] != 19
        or fluid_boundary_mask.dtype is not torch.bool
        or q_field.shape != fluid_boundary_mask.shape
        or not q_field.is_floating_point()
        or solid.shape != fluid_boundary_mask.shape[1:]
        or bare_hull.shape != solid.shape
        or solid.dtype is not torch.bool
        or bare_hull.dtype is not torch.bool
        or not (
            fluid_boundary_mask.device
            == q_field.device == solid.device == bare_hull.device
        )
    ):
        raise ValueError("SUBOFF link fields must be matching device tensors")
    if bool((bare_hull & ~solid).any()):
        raise ValueError("bare_hull must be a subset of full solid")
    if n_bisect < 1:
        raise ValueError("n_bisect must be positive")
    if length <= 0.0:
        raise ValueError("length must be positive")

    appendage_only = solid & ~bare_hull
    refined = q_field.clone()
    all_values: list[torch.Tensor] = []
    target_links = 0
    for direction in range(1, 19):
        cx, cy, cz = (int(value) for value in C3D[direction].tolist())
        target_neighbor = torch.roll(
            appendage_only,
            shifts=(-cz, -cy, -cx),
            dims=(0, 1, 2),
        )
        selected = fluid_boundary_mask[direction] & target_neighbor
        indices = selected.nonzero(as_tuple=False)
        if not indices.numel():
            continue
        target_links += int(indices.shape[0])
        z0 = indices[:, 0].to(q_field.dtype)
        y0 = indices[:, 1].to(q_field.dtype)
        x0 = indices[:, 2].to(q_field.dtype)
        lower = torch.zeros_like(x0)
        upper = torch.ones_like(x0)
        for _ in range(n_bisect):
            midpoint = 0.5 * (lower + upper)
            inside = suboff_appendages_contain_points(
                x0 + midpoint * cx,
                y0 + midpoint * cy,
                z0 + midpoint * cz,
                center=center,
                length=length,
            )
            upper = torch.where(inside, midpoint, upper)
            lower = torch.where(inside, lower, midpoint)
        values = (0.5 * (lower + upper)).clamp(1.0e-6, 1.0)
        refined[
            direction, indices[:, 0], indices[:, 1], indices[:, 2]
        ] = values
        all_values.append(values)

    if all_values:
        values = torch.cat(all_values)
        minimum_q = float(values.min())
        maximum_q = float(values.max())
    else:
        minimum_q = maximum_q = None
    return refined, SuboffAppendageLinkDiagnostics(
        scheme=SUBOFF_APPENDAGE_LINK_SCHEME,
        target_links=target_links,
        n_bisect=n_bisect,
        minimum_q=minimum_q,
        maximum_q=maximum_q,
    )


__all__ = [
    "SUBOFF_APPENDAGE_LINK_SCHEME",
    "SuboffAppendageLinkDiagnostics",
    "compute_q_suboff",
    "refine_q_suboff_appendages",
]
