"""Zalesak slotted-cylinder rotation benchmark for free-surface (VOF) transport.

Physics
-------
The Zalesak (1979) slotted-cylinder test is the canonical kinematic benchmark for
interface-capturing schemes.  A slotted cylinder (2-D) or slotted sphere (3-D) is
placed off-centre in a unit box and transported by the **prescribed,
analytically divergence-free** rigid-rotation field

    u(x) = Omega * (-(y - y0), x - x0)                    (2-D)
    u    = Omega * (-(y - y0), x - x0, 0)                 (3-D)

After one full revolution (t = 2*pi/Omega) the exact solution equals the initial
condition, so every deviation is a discretisation error of the interface
transport operator.  Three error families are scored:

* mass conservation  -- liquid volume  integral(phi) dV
* interface sharpness -- 10-90 % interface thickness in grid cells
* shape fidelity     -- overlap (Jaccard) and Hausdorff distance of the
  phi >= 0.5 set against the initial shape

Why the velocity field is prescribed
------------------------------------
Zalesak's benchmark is deliberately *kinematic*: it isolates the interface
transport operator from the coupled hydrodynamics, exactly as Zalesak (1979) and
all subsequent VOF / level-set / FCT studies do.  The free-surface fill-fraction
transport in :mod:`tensorlbm.free_surface_common` and
:mod:`tensorlbm.free_surface_lbm_2d` is likewise an operator that receives its
velocity from a momentum solver; here that solver is replaced by the exact
rigid-rotation field -- the only way to obtain a *known* exact solution against
which the transport operator can be graded.

Transport operator
------------------
``phi`` (liquid volume fraction, 0 = gas, 1 = liquid) is advanced in the
conservative **face-flux form** of the free-surface transport equation

    d(phi)/dt + div(phi * u) = 0

discretised on the lattice cell faces of the D2Q9 (2-D) / D3Q19 (3-D) stencils
(the +-e_d face pairs of the respective velocity set), with

* MUSCL / van-Leer limited second-order face reconstruction (or plain upwind),
* an optional **conservative interface-compression** flux
  q = D_c * (phi_R - phi_L) along the interface normal,
  D_c = c_comp * u_max * n_d_hat**2, bounded so that phi in [0, 1] is never
  violated.

Both the advective and the compressive fluxes are *single-valued per face*, so
the discrete update telescopes and the total volume is conserved to round-off.
The reported mass-conservation metric is therefore a genuine machine-precision
statement, not an artefact of a loose tolerance.

References
----------
Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport
    algorithms for fluids. J. Comput. Phys. 31, 335-362.
Rider, W. J., & Kothe, D. B. (1998). Reconstructing volume tracking.
    J. Comput. Phys. 141, 112-152.
Roache, P. J. (1994). Perspective: a method for uniform reporting of grid
    refinement studies. J. Fluids Eng. 116, 405-413.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

__all__ = [
    "slotted_cylinder_phi_2d",
    "slotted_sphere_phi_3d",
    "slotted_cylinder_extruded_phi_3d",
    "analytic_volume_2d",
    "cross_section_index",
    "cross_section",
    "rigid_rotation_field",
    "discrete_divergence_max",
    "advect_face_flux",
    "interface_compression_flux",
    "zalesak_step",
    "liquid_volume",
    "interface_thickness",
    "shape_overlap",
    "hausdorff_distance",
    "l1_shape_error",
    "convergence_order",
    "gci",
    "run_zalesak_case",
]


# --------------------------------------------------------------------------- #
# Geometry: initial liquid volume fraction                                    #
# --------------------------------------------------------------------------- #


def _cell_centres(n: int, *, device, dtype) -> torch.Tensor:
    """Cell-centre coordinates of a unit box [0,1]^D with n cells per axis."""
    h = 1.0 / n
    return (torch.arange(n, device=device, dtype=dtype) + 0.5) * h


def _axis_components(u: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    """Map a velocity list ordered (u_x, u_y[, u_z]) onto array axes.

    All fields in this module are stored in TensorLBM's standard descending
    axis order -- (..., y, x) in 2-D and (..., z, y, x) in 3-D -- so the
    velocity component that shifts axis ``d`` is ``u[D-1-d]``.
    """
    return list(reversed(list(u)))


def _signed_shape_2d(
    xs: torch.Tensor,
    *,
    cx: float,
    cy: float,
    radius: float,
    slot_width: float,
    slot_depth: float,
) -> torch.Tensor:
    """Signed indicator of the 2-D slotted cylinder: > 0 inside the liquid.

    The slot is the rectangle |x-cx| <= slot_width/2,
    cy+radius-slot_depth <= y <= cy+radius (cut down from the cylinder top),
    as in Zalesak (1979).
    """
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    inside_disk = radius**2 - ((xx - cx) ** 2 + (yy - cy) ** 2)
    y_top = cy + radius
    y_slot = y_top - slot_depth
    in_slot = ((xx - cx).abs() <= 0.5 * slot_width) & (yy <= y_top) & (yy >= y_slot)
    return torch.where(in_slot, torch.full_like(inside_disk, -1.0), inside_disk.sign())


def _signed_shape_3d(
    xs: torch.Tensor,
    *,
    cx: float,
    cy: float,
    cz: float,
    radius: float,
    slot_width: float,
    slot_depth: float,
    cylindrical_slot: bool = False,
    extruded: bool = False,
) -> torch.Tensor:
    """Signed indicator of the 3-D slotted body (> 0 inside the liquid).

    The rotation takes place in the **x-z plane** about the line
    ``(x, z) = (cx, cz0) = (0.5, 0.5)`` (see :func:`rigid_rotation_field`), so
    the sphere sits *off-axis* at ``(cx, cy, cz) = (0.5, 0.5, 0.75)`` -- exactly
    the 2-D configuration promoted to 3-D: in-plane vertical coordinate ``z``
    takes the place of the 2-D ``y``.

    ``cylindrical_slot=False`` (default): the slot is the **planar-faced cut**
    ``|x - cx| <= slot_width/2`` and ``cz + radius - slot_depth <= z <= cz + radius``
    -- the 2-D slot of :func:`slotted_cylinder_phi_2d` promoted to 3-D, bounded
    by the planes ``x = cx +- slot_width/2`` and ``z = cz + radius - slot_depth``
    and open through the sphere in ``y``.  The slice ``y = cy`` is then exactly
    the 2-D Zalesak initial condition, while the other slices are progressively
    smaller circles.

    ``cylindrical_slot=True``: the slot is a circular bore of radius
    ``slot_width/2`` about the line ``(x, y) = (cx, cy)`` (a hole drilled down
    through the same top band), which is a genuinely three-dimensional cut --
    no slice of it equals the 2-D shape.
    """
    zz, yy, xx = torch.meshgrid(xs, xs, xs, indexing="ij")
    if extruded:
        # In-plane disk (x, z), invariant in y: the 2-D Zalesak body extruded.
        inside_sphere = radius**2 - ((xx - cx) ** 2 + (zz - cz) ** 2)
    else:
        inside_sphere = radius**2 - ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2)
    z_top = cz + radius
    z_slot = z_top - slot_depth
    z_ok = (zz <= z_top) & (zz >= z_slot)
    if cylindrical_slot:
        in_slot = (((xx - cx) ** 2 + (yy - cy) ** 2) <= (0.5 * slot_width) ** 2) & z_ok
    else:
        in_slot = ((xx - cx).abs() <= 0.5 * slot_width) & z_ok
    return torch.where(in_slot, torch.full_like(inside_sphere, -1.0), inside_sphere.sign())


def _supersampled_phi(n: int, signed_fn, *, sub: int, device, dtype) -> torch.Tensor:
    """Sub-cell integrated volume fraction of the region signed_fn > 0.

    sub**D midpoint samples per cell give second-order geometric accuracy of the
    cell fractions and an O(sub**-2) accurate liquid volume (verified in
    tests/test_free_surface_zalesak.py).
    """
    h = 1.0 / (n * sub)
    xs = (torch.arange(n * sub, device=device, dtype=dtype) + 0.5) * h
    inside = (signed_fn(xs) > 0).to(dtype)
    d = inside.dim()
    new_shape: list[int] = []
    for _ in range(d):
        new_shape.extend([n, sub])
    reduced = inside.reshape(new_shape)
    sum_axes = tuple(range(1, 2 * d, 2))
    phi = reduced.sum(dim=sum_axes) / float(sub**d)
    return phi.contiguous().clamp(0.0, 1.0)


def slotted_cylinder_phi_2d(
    n: int,
    *,
    cx: float = 0.5,
    cy: float = 0.75,
    radius: float = 0.15,
    slot_width: float = 0.05,
    slot_depth: float = 0.25,
    sub: int = 8,
    device="cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """Initial (= exact final) 2-D liquid fraction field, shape (n, n)."""
    device = torch.device(device)

    def signed(xs):
        return _signed_shape_2d(
            xs, cx=cx, cy=cy, radius=radius,
            slot_width=slot_width, slot_depth=slot_depth,
        )

    return _supersampled_phi(n, signed, sub=sub, device=device, dtype=dtype)


def slotted_sphere_phi_3d(
    n: int,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
    cz: float = 0.75,
    radius: float = 0.15,
    slot_width: float = 0.05,
    slot_depth: float = 0.25,
    sub: int = 4,
    cylindrical_slot: bool = False,
    device="cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """Initial 3-D liquid fraction field, shape (n, n, n) ordered (z, y, x).

    The sphere is centred off-axis at (cx, cy, cz) = (0.5, 0.5, 0.75) and the
    rotation is about the line (x, z) = (0.5, 0.5) in the x-z plane, so the
    slice y = cy is the 2-D Zalesak problem in (x, z).
    """
    device = torch.device(device)

    def signed(xs):
        return _signed_shape_3d(
            xs, cx=cx, cy=cy, cz=cz, radius=radius,
            slot_width=slot_width, slot_depth=slot_depth,
            cylindrical_slot=cylindrical_slot,
        )

    phi = _supersampled_phi(n, signed, sub=sub, device=device, dtype=dtype)
    # The exact shape is symmetric about the plane y = cy, and the prescribed
    # flow is too.  The sub-cell sampling is not (the sub-cell offsets are not
    # mirror-symmetric), so project the discrete field onto the exact symmetry:
    # otherwise the y-gradient of phi is pure sampling noise, and the interface
    # compression turns that noise into a spurious out-of-plane flux that
    # contaminates the y = cy cross-section.
    return 0.5 * (phi + torch.flip(phi, dims=[1])).contiguous()


def slotted_cylinder_extruded_phi_3d(
    n: int,
    *,
    cx: float = 0.5,
    cy: float = 0.5,
    cz: float = 0.75,
    radius: float = 0.15,
    slot_width: float = 0.05,
    slot_depth: float = 0.25,
    sub: int = 4,
    device="cpu",
    dtype=torch.float64,
) -> torch.Tensor:
    """2-D Zalesak body (slotted disk in the x-z plane) extruded through y.

    This is the *kernel-equivalence* configuration: the field is exactly
    invariant in ``y`` and the interface normal has no out-of-plane component,
    so a D3Q19 run must reproduce the D2Q9 solution on the cross-section to
    round-off (the sharpening operator, which is geometry dependent, then acts
    identically in both).
    """
    device = torch.device(device)

    def signed(xs):
        return _signed_shape_3d(
            xs, cx=cx, cy=cy, cz=cz, radius=radius,
            slot_width=slot_width, slot_depth=slot_depth, extruded=True,
        )

    return _supersampled_phi(n, signed, sub=sub, device=device, dtype=dtype)


def cross_section_index(n: int, *, coord: float = 0.5) -> int:
    """Index (along the out-of-plane axis) of the slice containing ``coord``.

    Used for the 3-D run: the slice ``y = cy`` of the slotted sphere is the 2-D
    Zalesak problem.
    """
    return max(0, min(n - 1, int(coord * n)))


def cross_section(field: torch.Tensor, *, coord: float = 0.5) -> torch.Tensor:
    """In-plane slice of a 2-D field (row ``coord``) or a 3-D field (``y = coord``).

    For the 3-D field the out-of-plane axis is ``y`` (axis 1, shape ``(z, y, x)``),
    so the returned plane is ordered ``(z, x)`` -- the in-plane vertical and
    horizontal coordinates of the rotation.
    """
    i = cross_section_index(field.shape[0], coord=coord)
    return field[i] if field.dim() == 2 else field[:, i]


def analytic_volume_2d(
    *,
    radius: float = 0.15,
    slot_width: float = 0.05,
    slot_depth: float = 0.25,
    sub: int = 2000,
) -> float:
    """Reference area of the 2-D slotted cylinder by brute-force quadrature."""
    h = 1.0 / sub
    x = (torch.arange(sub, dtype=torch.float64) + 0.5) * h
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    cx, cy = 0.5, 0.75
    disk = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius**2
    slot = (
        ((xx - cx).abs() <= 0.5 * slot_width)
        & (yy <= cy + radius)
        & (yy >= cy + radius - slot_depth)
    )
    return float((disk & ~slot).double().sum().item() * h * h)

# --------------------------------------------------------------------------- #
# Velocity field                                                              #
# --------------------------------------------------------------------------- #


def rigid_rotation_field(
    n: int,
    *,
    dim: int = 2,
    centre: Sequence[float] | None = None,
    omega: float = 1.0,
    device="cpu",
    dtype=torch.float64,
) -> list[torch.Tensor]:
    """Analytic divergence-free rigid rotation in lattice units (dx = dt = 1).

    2-D: rotation in the x-y plane about ``(x0, y0) = centre``, returned as
    ``[u_x, u_y]`` with arrays of shape ``(n, n)`` ordered ``(y, x)``.

    3-D: rotation in the **x-z plane** about ``(x0, z0) = centre``, returned as
    ``[u_x, u_y, u_z]`` with arrays of shape ``(n, n, n)`` ordered ``(z, y, x)``
    and ``u_y = 0``.  Choosing the x-z plane keeps the sphere of
    :func:`slotted_sphere_phi_3d` off-axis (a rotation about the z axis through
    the sphere centre would be degenerate: pure spin, no transport).

    ``omega`` is in lattice units, i.e. the returned velocities already carry
    the factor ``dt/dx``.
    """
    device = torch.device(device)
    c = _cell_centres(n, device=device, dtype=dtype)
    if centre is None:
        centre = (0.5, 0.5)
    if dim == 2:
        y, x = torch.meshgrid(c, c, indexing="ij")
        return [-omega * (y - centre[1]), omega * (x - centre[0])]
    if dim == 3:
        z, y, x = torch.meshgrid(c, c, c, indexing="ij")
        return [
            -omega * (z - centre[1]),          # u_x  (in-plane horizontal)
            torch.zeros_like(x),               # u_y  (out of plane)
            omega * (x - centre[0]),           # u_z  (in-plane vertical)
        ]
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def discrete_divergence_max(u: Sequence[torch.Tensor]) -> float:
    """max |div u| over the grid, using the face-flux divergence.

    Face velocities u_{i+1/2} = (u_i + u_{i+1}) / 2 (periodic) are exactly the
    quantities the transport operator consumes, so this is the divergence that
    matters for discrete mass conservation.
    """
    comp = _axis_components(u)
    total = torch.zeros_like(u[0])
    for d, ud in enumerate(comp):
        face = 0.5 * (ud + torch.roll(ud, -1, dims=d))
        total = total + (face - torch.roll(face, 1, dims=d))
    return float(total.abs().max().item())


# --------------------------------------------------------------------------- #
# Transport: conservative face-flux advection                                 #
# --------------------------------------------------------------------------- #


def _limited_slope(a: torch.Tensor, b: torch.Tensor, limiter: str = "vanleer") -> torch.Tensor:
    """Slope limited from the backward (a) and forward (b) differences.

    ``limiter="vanleer"``: ``2ab/(a+b)`` (smooth, total-variation diminishing);
    ``limiter="superbee"``: ``sign(a) max(min(2|a|,|b|), min(|a|,2|b|))``
    (least diffusive of the classical TVD limiters);
    ``limiter="minmod"``: the most diffusive TVD choice.
    Both fall back to zero slope at extrema or sign changes.
    """
    if limiter == "vanleer":
        den = a + b
        den = torch.where(den.abs() < 1e-300, torch.full_like(den, 1e-300), den)
        return torch.where((a * b) > 0, 2.0 * a * b / den, torch.zeros_like(a))
    aa, ab = a.abs(), b.abs()
    if limiter == "superbee":
        cand = torch.maximum(torch.minimum(2.0 * aa, ab), torch.minimum(aa, 2.0 * ab))
    elif limiter == "minmod":
        cand = torch.minimum(aa, ab)
    else:
        raise ValueError(f"unknown limiter {limiter!r}")
    return torch.where((a * b) > 0, torch.sign(a) * cand, torch.zeros_like(a))


def advect_face_flux(
    phi: torch.Tensor,
    u: Sequence[torch.Tensor],
    *,
    scheme: str = "muscl",
    limiter: str = "vanleer",
) -> torch.Tensor:
    """One explicit transport sub-step of phi in conservative flux form.

        phi_new_i = phi_i - sum_d ( F^d_{i+1/2} - F^d_{i-1/2} )
        F^d_{i+1/2} = u^d_{i+1/2} * phi_face

    The face value phi_face is upwind (scheme="upwind") or MUSCL / van-Leer
    limited (scheme="muscl").  Every face flux is single-valued, so sum(phi) is
    conserved exactly.

    Args:
        phi: Volume fraction, shape n^D.
        u: Velocity components, same shape/ordering as phi, lattice units.
        scheme: "upwind" (1st order) or "muscl" (2nd order limited).
    """
    if scheme not in ("upwind", "muscl"):
        raise ValueError("scheme must be 'upwind' or 'muscl'")
    comp = _axis_components(u)
    out = phi
    for d, ud in enumerate(comp):
        phi_plus = torch.roll(phi, -1, dims=d)   # phi[i+1]
        phi_minus = torch.roll(phi, 1, dims=d)   # phi[i-1]
        u_face = 0.5 * (ud + torch.roll(ud, -1, dims=d))  # u at face i+1/2
        if scheme == "upwind":
            face_val = torch.where(u_face > 0, phi, phi_plus)
        else:
            slope = _limited_slope(phi - phi_minus, phi_plus - phi, limiter)
            slope_plus = torch.roll(slope, -1, dims=d)
            cfl = u_face.abs().clamp(max=1.0)
            pos = phi + 0.5 * (1.0 - cfl) * slope
            neg = phi_plus - 0.5 * (1.0 - cfl) * slope_plus
            face_val = torch.where(u_face > 0, pos, neg)
        flux = u_face * face_val.clamp(0.0, 1.0)
        out = out - (flux - torch.roll(flux, 1, dims=d))
    return out


def interface_compression_flux(
    phi: torch.Tensor,
    u: Sequence[torch.Tensor],
    *,
    c_comp: float = 0.5,
    u_max: float | None = None,
) -> torch.Tensor:
    """Conservative interface-sharpening step (anti-diffusive normal flux, FCT-limited).

    For each face (i, i+1) in direction d the anti-diffusive flux is

        q          = D_c * (phi_{i+1} - phi_i)
        D_c        = c_comp * u_max * n_d_hat**2 * alpha_face
        alpha_face = min(4 phi_i (1 - phi_i), 4 phi_{i+1} (1 - phi_{i+1}))

    with n_d_hat the face-averaged interface-normal component.  ``q > 0`` moves
    liquid from the low-fraction side to the high-fraction side, which steepens a
    monotone profile (negative diffusion) and vanishes in the bulk through
    ``alpha_face``.

    Anti-diffusion is unconditionally unstable unless it is limited.  The limiter
    used here is the Zalesak (1979) flux-corrected-transport capacity limiter in
    its conservative form: with ``out_i`` / ``in_i`` the total anti-diffusive
    outflow / inflow tentatively assigned to cell ``i`` (summed over all faces),

        alpha_i = min(1, phi_i / out_i),   beta_i = min(1, (1 - phi_i) / in_i)

    and every face flux is scaled by ``min(alpha_donor, beta_acceptor)``.  Each
    cell therefore loses at most what it has and gains at most the room it has,
    so ``phi`` stays in ``[0, 1]`` while the anti-diffusive step remains a
    divergence of single-valued face fluxes and hence **exactly conservative**.

    Args:
        phi: Volume fraction, shape n^D.
        u: Velocity components (used for u_max only).
        c_comp: Compression coefficient (0 disables sharpening).
        u_max: Reference speed; defaults to max_d max|u_d|.
    """
    if c_comp == 0.0:
        return phi
    if u_max is None:
        u_max = max(float(ud.abs().max().item()) for ud in u)
    eps = 1e-30
    dims = range(len(u))
    grads = [0.5 * (torch.roll(phi, -1, dims=d) - torch.roll(phi, 1, dims=d)) for d in dims]
    mag = torch.sqrt(sum(g * g for g in grads) + eps)
    alpha = 4.0 * phi * (1.0 - phi)
    fluxes = []
    out_tot = torch.zeros_like(phi)
    in_tot = torch.zeros_like(phi)
    for d in dims:
        n_d = grads[d] / mag
        n_face = 0.5 * (n_d + torch.roll(n_d, -1, dims=d))
        phi_plus = torch.roll(phi, -1, dims=d)
        alpha_face = torch.minimum(alpha, torch.roll(alpha, -1, dims=d))
        d_c = (c_comp * u_max) * n_face * n_face * alpha_face
        q = d_c * (phi_plus - phi)                   # flux from cell i to i+1
        fwd = q.clamp(min=0.0)                       # i -> i+1
        bwd = (-q).clamp(min=0.0)                    # i+1 -> i
        # Tentatively add this direction's fluxes to the per-cell budgets.
        # Cell i loses fwd on face (i, i+1) and bwd on face (i-1, i); it gains
        # bwd on face (i, i+1) and fwd on face (i-1, i).
        out_tot = out_tot + fwd + torch.roll(bwd, 1, dims=d)
        in_tot = in_tot + bwd + torch.roll(fwd, 1, dims=d)
        fluxes.append(q)
    alpha_c = torch.where(out_tot > eps, (phi / out_tot.clamp(min=eps)).clamp(max=1.0),
                          torch.ones_like(phi))
    beta_c = torch.where(in_tot > eps, ((1.0 - phi) / in_tot.clamp(min=eps)).clamp(max=1.0),
                         torch.ones_like(phi))
    out = phi
    for d, q in zip(dims, fluxes):
        donor_alpha = torch.where(q > 0, alpha_c, torch.roll(alpha_c, -1, dims=d))
        acc_beta = torch.where(q > 0, torch.roll(beta_c, -1, dims=d), beta_c)
        q_lim = q * torch.minimum(donor_alpha, acc_beta)
        out = out - (q_lim - torch.roll(q_lim, 1, dims=d))
    return out


def zalesak_step(
    phi: torch.Tensor,
    u: Sequence[torch.Tensor],
    *,
    c_comp: float = 0.0,
    scheme: str = "muscl",
    limiter: str = "vanleer",
) -> torch.Tensor:
    """One full interface-transport step: advection followed by sharpening."""
    phi = advect_face_flux(phi, u, scheme=scheme, limiter=limiter)
    return interface_compression_flux(phi, u, c_comp=c_comp)


# --------------------------------------------------------------------------- #
# Metrics                                                                     #
# --------------------------------------------------------------------------- #


def liquid_volume(phi: torch.Tensor) -> float:
    """Liquid volume in a unit box: V = mean(phi) (since sum(dx^D n^D) = 1)."""
    return float(phi.mean().item())


def _interface_mask(phi: torch.Tensor) -> torch.Tensor:
    """Cells where the phi = 0.5 iso-surface passes (sign change with a neighbour)."""
    base = phi > 0.5
    on_iface = torch.zeros_like(base)
    for d in range(phi.dim()):
        on_iface = on_iface | (base ^ (torch.roll(phi, -1, dims=d) > 0.5))
    return on_iface


def _grad_mag(phi: torch.Tensor) -> torch.Tensor:
    g2 = torch.zeros_like(phi)
    for d in range(phi.dim()):
        g = 0.5 * (torch.roll(phi, -1, dims=d) - torch.roll(phi, 1, dims=d))
        g2 = g2 + g * g
    return torch.sqrt(g2)


def interface_thickness(phi: torch.Tensor, *, delta: float = 0.8) -> dict:
    """10-90 % interface thickness in grid cells, from the local gradient.

    For a monotone interface profile the 10-90 % band width is
    ``thickness = delta / |grad phi|`` evaluated where the phi = 0.5 iso-surface
    crosses the cell.  ``delta = 0.8`` is the 0.1 -> 0.9 span of a linear
    profile; it is divided by |grad phi| in *per-cell* units, so the result is
    directly in grid cells.

    Returns dict with mean / p95 / max thickness and the number of interface
    cells sampled.
    """
    mask = _interface_mask(phi)
    grad = _grad_mag(phi)
    vals = (delta / grad.clamp(min=1e-12))[mask]
    if vals.numel() == 0:
        return {"mean": float("nan"), "p95": float("nan"), "max": float("nan"), "n_cells": 0}
    vals = vals.sort().values
    p95 = float(vals[min(int(0.95 * (vals.numel() - 1)) , vals.numel() - 1)].item())
    return {
        "mean": float(vals.mean().item()),
        "p95": p95,
        "max": float(vals.max().item()),
        "n_cells": int(mask.sum().item()),
    }


def shape_overlap(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    thr: float = 0.5,
    mode: str = "jaccard",
) -> float:
    """Overlap of the two ``phi >= thr`` sets.

    ``mode="jaccard"`` (default, symmetric): ``|A inter B| / |A union B|`` (IoU).
    ``mode="recall"``: ``|A inter B| / |A|`` -- the fraction of set ``a`` covered
    by set ``b``, i.e. the classical "overlap with the initial shape" measure.
    """
    ma = a >= thr
    mb = b >= thr
    inter = float((ma & mb).sum().item())
    if mode == "jaccard":
        union = float((ma | mb).sum().item())
        return inter / union if union else float("nan")
    if mode == "recall":
        ref = float(ma.sum().item())
        return inter / ref if ref else float("nan")
    raise ValueError("mode must be 'jaccard' or 'recall'")


def _contour_points(phi: torch.Tensor, *, thr: float = 0.5) -> torch.Tensor:
    """Cell-centre coordinates (physical units, unit box) of phi = thr crossings."""
    n = phi.shape[0]
    idx = _interface_mask(phi).nonzero()
    if idx.numel() == 0:
        return torch.zeros((0, phi.dim()), dtype=phi.dtype)
    return (idx.to(phi.dtype) + 0.5) / n


def hausdorff_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    thr: float = 0.5,
    in_cells: bool = True,
) -> float:
    """Symmetric Hausdorff distance between the phi = thr interfaces.

    Both interface point sets are the cell centres that straddle phi = thr; the
    distance is the max of the two directed Hausdorff distances.  Reported in
    grid cells by default (multiplied by n).
    """
    pa = _contour_points(a, thr=thr)
    pb = _contour_points(b, thr=thr)
    if pa.numel() == 0 or pb.numel() == 0:
        return float("nan")
    d = torch.cdist(pa, pb)
    hd = max(float(d.min(dim=1).values.max().item()), float(d.min(dim=0).values.max().item()))
    return hd * (a.shape[0] if in_cells else 1.0)


def l1_shape_error(phi: torch.Tensor, phi_ref: torch.Tensor) -> float:
    """Cell-averaged L1 error |phi - phi_ref| (dimensionless)."""
    return float((phi - phi_ref).abs().mean().item())


def convergence_order(e_coarse: float, e_fine: float, *, r: float = 2.0) -> float:
    """Observed order p = ln(e_coarse / e_fine) / ln(r)."""
    if e_fine <= 0 or e_coarse <= 0:
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(r)


def gci(e_coarse: float, e_fine: float, p: float, *, r: float = 2.0, fs: float = 1.25) -> float:
    """Roache (1994) grid-convergence index of the fine grid, in per cent.

    ``GCI_fine = 100 * Fs * |e_fine| / (r^p - 1)`` where ``e`` is the fine-grid
    discretisation error of the metric of interest (here the L1 shape error),
    ``p`` the observed order and ``Fs = 1.25`` the safety factor.  The result
    bounds the relative numerical error of the fine-grid value.
    """
    denom = r**p - 1.0
    if not math.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return 100.0 * fs * abs(e_fine) / denom


# --------------------------------------------------------------------------- #
# Case driver                                                                 #
# --------------------------------------------------------------------------- #


def run_zalesak_case(
    n: int,
    *,
    dim: int = 2,
    u_max_lat: float = 0.1,
    steps: int | None = None,
    c_comp: float = 0.5,
    scheme: str = "muscl",
    limiter: str = "vanleer",
    cylindrical_slot: bool = False,
    phi_init: torch.Tensor | None = None,
    sample_every: int | None = None,
    dtype=torch.float64,
    device="cpu",
    verbose: bool = False,
) -> dict:
    """Run one Zalesak revolution and return metrics plus a sampled history.

    The number of lattice steps for one revolution is
    ``steps = round(2*pi*R_max / (u_max_lat * dx))`` with ``R_max`` the largest
    distance from the rotation centre to the liquid, and the angular rate is
    fixed to ``omega = 2*pi/steps`` so that the total rotation after ``steps``
    steps is *exactly* 2*pi (no end-of-run phase error).

    Args:
        n: Cells per axis.
        dim: 2 (D2Q9 face set) or 3 (D3Q19 face set).
        u_max_lat: Target peak lattice speed (< 0.2 for explicit stability).
        steps: Override the step count derived from ``u_max_lat``.
        c_comp: Interface-compression coefficient (0 = pure advection control).
        scheme: "upwind" or "muscl".
        cylindrical_slot: 3-D only -- genuinely 3-D slot instead of the
            z-invariant slab (see :func:`slotted_sphere_phi_3d`).
        sample_every: History sampling interval (default ~100 samples total).
    """
    device = torch.device(device)
    dx = 1.0 / n
    if phi_init is None:
        phi0 = (
            slotted_cylinder_phi_2d(n, device=device, dtype=dtype)
            if dim == 2
            else slotted_sphere_phi_3d(n, cylindrical_slot=cylindrical_slot, device=device, dtype=dtype)
        )
    else:
        phi0 = phi_init.to(device=device, dtype=dtype).clone()
    # Largest in-plane distance from the rotation centre to any liquid cell.
    # In-plane axes are (0, dim-1) -- (y, x) in 2-D and (z, x) in 3-D -- because
    # the 3-D rotation is performed in the x-z plane.
    idx = (phi0 > 0.5).nonzero().to(dtype)
    plane = idx[:, [0, dim - 1]] if phi0.dim() == dim else idx
    offsets = (plane + 0.5) * dx - 0.5
    r_max = float(offsets.pow(2).sum(dim=1).sqrt().max().item())
    if steps is None:
        steps = max(8, int(round(2.0 * math.pi * r_max / (u_max_lat * dx))))
    # Lattice angular rate: omega_lat * r gives cells travelled per step, so
    # omega_lat = 2*pi/steps (physical) / dx  ->  peak lattice speed u_max_lat.
    omega = 2.0 * math.pi / (steps * dx)
    u = rigid_rotation_field(n, dim=dim, centre=(0.5, 0.5), omega=omega, device=device, dtype=dtype)
    if sample_every is None:
        sample_every = max(1, steps // 100)

    div_max = discrete_divergence_max(u)
    v0 = liquid_volume(phi0)
    thickness0 = interface_thickness(phi0)

    phi = phi0.clone()
    history: list[dict] = []
    out_of_bounds = 0
    with torch.no_grad():
        for step in range(1, steps + 1):
            phi = zalesak_step(phi, u, c_comp=c_comp, scheme=scheme, limiter=limiter)
            out_of_bounds += int(((phi < -1e-12) | (phi > 1.0 + 1e-12)).sum().item())
            if step % sample_every == 0 or step == steps:
                history.append(
                    {
                        "step": step,
                        "time_frac": step / steps,
                        "volume": liquid_volume(phi),
                        "volume_rel_err": (liquid_volume(phi) - v0) / v0,
                        "thickness_mean": interface_thickness(phi)["mean"],
                        "thickness_p95": interface_thickness(phi)["p95"],
                        "overlap": shape_overlap(phi, phi0, mode="recall"),
                        "l1": l1_shape_error(phi, phi0),
                    }
                )
            if verbose and step % max(1, steps // 10) == 0:
                print(f"  [{n}^{dim}] step {step}/{steps} V={liquid_volume(phi):.12f}")

    thickness = interface_thickness(phi)
    v1 = liquid_volume(phi)
    metrics = {
        "n": n,
        "dim": dim,
        "lattice": "D2Q9" if dim == 2 else "D3Q19",
        "scheme": scheme,
        "limiter": limiter,
        "c_comp": c_comp,
        "cylindrical_slot": bool(cylindrical_slot),
        "steps": steps,
        "omega_lattice": omega,
        "u_max_lattice": omega * r_max,
        "r_max": r_max,
        "div_max": div_max,
        "volume_initial": v0,
        "volume_final": v1,
        "volume_rel_change": (v1 - v0) / v0,
        "thickness_initial_mean": thickness0["mean"],
        "thickness_mean": thickness["mean"],
        "thickness_p95": thickness["p95"],
        "thickness_max": thickness["max"],
        "overlap": shape_overlap(phi, phi0, mode="jaccard"),
        "overlap_recall": shape_overlap(phi, phi0, mode="recall"),
        "overlap_iou": shape_overlap(phi, phi0, mode="jaccard"),
        "hausdorff_cells": hausdorff_distance(phi, phi0),
        "l1_error": l1_shape_error(phi, phi0),
        "boundedness_violations": out_of_bounds,
        "phi_min": float(phi.min().item()),
        "phi_max": float(phi.max().item()),
    }
    return {"metrics": metrics, "phi_final": phi, "phi_initial": phi0, "history": history}
