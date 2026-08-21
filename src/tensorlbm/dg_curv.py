"""Curvilinear prism-band DG-LBM: a body-fitted near-wall layer.

This module builds on :mod:`tensorlbm.dg_advection` and :mod:`tensorlbm.dg_band`
but carries *per-element geometric factors* so the DG band can follow a curved
wall (e.g. a sphere) instead of a staircased Cartesian shell.

Design (geometry-agnostic solver + adapters)
--------------------------------------------
The DG operators are written in *reference* space and only consume three
geometric tensors per element:

* ``contrav[b, a, beta]`` — the contravariant metric ``∂ξ^a/∂x^beta``
  (the inverse Jacobian, transposed).  The physical advective speed along
  reference axis ``a`` is ``ĉ^a_q = Σ_beta contrav[b,a,beta] c_q^beta``.
* ``face_J[a, sgn, b]`` — the physical/reference face-area scaling for the
  ``±ξ^a`` face of element ``b`` (so fluxes are integrated over the true
  curved face).
* ``n_phys[a, sgn, b, :]`` — the outward unit normal of that face (used for
  the specular wall bounce-back ghost).

Because the solver only sees these factors, it is **generic**: a sphere, a
SUBOFF hull or any star-shaped body differs only in the adapter that produces
the factors (here ``make_sphere_prism_topology``).  Curvature enters *only*
through the factors; the operator code is shared.

For an *affine* element (constant Jacobian) — which is the case for prism
layers marched straight out along surface normals — the volume term reduces to
a standard reference DG advection with the speed rotated by ``contrav``, and the
Jacobian cancels from the volume integral exactly.  This makes the curvilinear
path a clean generalisation of the Cartesian one (which is recovered when
``contrav=I``, ``face_J=1``, ``n_phys`` = axis normals).

The element layout for the sphere is a structured grid in local reference
coordinates ``(ξ_stream = x, ξ_azimuth = φ, ξ_radial = r)``; each element is an
affine hex (spherical shell sector).  Neighbour exchange reuses the packed
``index_select`` machinery of :mod:`tensorlbm.dg_band`; only the geometry
factors differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .d3q19 import OPPOSITE as OPP3D
from .dg_advection import _Ops, equilibrium_dg, get_ops, macroscopic_dg
from .dg_band import (
    BandTopology,
    _neighbour_face_value,
    _override_face,
    write_back_exports,
)


@dataclass
class PrismGeometry:
    """Per-element geometric factors for a curvilinear prism band.

    Shapes use ``n_band`` = number of prism elements, ``ndim = 3``.
    """

    contrav: torch.Tensor  # (n_band, 3, 3)  ∂ξ^a/∂x^β
    face_J: torch.Tensor  # (3, 2, n_band)    surface Jacobian weight (= detJ, same for all faces)
    n_phys: torch.Tensor  # (3, 2, n_band, 3) outward unit normal (ax, sgn, elem, xyz)
    specular: torch.Tensor  # (n_band, Q) int  reflected-population index map
    detJ: torch.Tensor  # (n_band,) volume Jacobian weight |det J|

    def to(self, device: torch.device) -> "PrismGeometry":
        return PrismGeometry(
            contrav=self.contrav.to(device),
            face_J=self.face_J.to(device),
            n_phys=self.n_phys.to(device),
            specular=self.specular.to(device),
        )


# ---------------------------------------------------------------------------
# Geometry factor computation for an affine-hex prism band
# ---------------------------------------------------------------------------


def _sphere_point(C, R, s, phi, r):
    """Isoparametric map for a spherical-shell band.

    C: (3,) centre.  R: sphere radius.  s: streamwise offset (x-Cx).
    phi: azimuth in the (y,z) plane.  r: radial distance from centre (>= R).
    Returns the physical point on the band.
    """
    rho = torch.sqrt(torch.clamp(R * R - s * s, min=1e-12))
    x = C[0] + r * (s / R)
    y = C[1] + r * (rho / R) * torch.cos(phi)
    z = C[2] + r * (rho / R) * torch.sin(phi)
    return torch.stack([x, y, z], dim=-1)  # (..., 3)


def _geometry_from_J(J: torch.Tensor, device) -> tuple:
    """Geometric factors from a per-element Jacobian ``J`` (n_band, 3, 3).

    ``J_b`` columns are the three edge vectors mapping reference ``[-1,1]^3`` to
    physical space.  Returns ``(contrav, face_J, n_phys, detJ)``:
    ``contrav = J^{-T}`` (the contravariant metric), ``face_J`` the physical
    face-area scaling (here set to ``detJ`` so volume and surface fluxes carry
    the *same* Jacobian weight and the reference-space divergence theorem stays
    balanced — see ``dg_rhs_band_geo``), ``n_phys`` the outward unit face
    normals, and ``detJ = |det J|`` the per-element volume measure.
    """
    Jinv = torch.linalg.inv(J)  # (n_band, 3, 3)
    contrav = Jinv.transpose(1, 2)  # J^{-T}
    detJ = torch.linalg.det(J).abs()  # (n_band,) volume measure

    # Surface fluxes in reference space must be scaled by the *same* factor as the
    # volume term (the Jacobian determinant).  Using per-face areas instead would
    # break the divergence-theorem balance and inject energy into a uniform field.
    face_J = detJ.view(1, 1, -1).expand(3, 2, detJ.shape[0]).contiguous()

    n_phys = torch.empty(3, 2, J.shape[0], 3, device=device)
    for a in (0, 1, 2):
        # physical outward normal of face a (constant reference coord ξ_a) is
        # J^{-T} e_a == ROW a of Jinv (NOT column a).
        col = Jinv[:, a, :]
        n = col / (torch.norm(col, dim=-1, keepdim=True) + 1e-30)
        n_phys[a, 1] = n  # + face
        n_phys[a, 0] = -n  # - face
    return contrav, face_J, n_phys, detJ


# ---------------------------------------------------------------------------
# Sphere prism-band topology (local orthonormal frame per element)
# ---------------------------------------------------------------------------


def make_sphere_prism_topology(
    solid: torch.Tensor,
    center: tuple[float, float, float],
    R: float,
    n_layers: int = 4,
    first_height: float = 0.5,
    growth: float = 1.1,
    n_az: int = 24,
    n_stream: int = 16,
    polar_cap: float = 0.985,  # exclude |cos(theta)| > polar_cap (front/back caps)
    vel: torch.Tensor | None = None,  # (Q,3) lattice velocities (for specular map)
    wall_bc: str = "bounceback",  # "bounceback" (no-slip, like Cartesian) or "specular"
    dtype=torch.float32,
    device: torch.device | str = "cpu",
) -> tuple:
    """Build a body-fitted prism band hugging a sphere.

    Each band element is an *affine hex* built from a local orthonormal frame
    at its surface cell: radial unit ``n`` (true sphere normal), an azimuthal
    tangent ``t_az`` and a streamwise (polar) tangent ``t_stream``.  Because the
    frame is orthonormal, the element Jacobian is orthogonal and the inner-face
    normal is exactly the radial direction — the band truly follows the curved
    wall.  The band is stratified in ``(stream_bin, az_bin, layer)`` local
    reference coordinates (radial, streamwise/polar, azimuthal).

    Polar caps (|cosθ| > ``polar_cap``) are excluded and left to the exterior
    LBM (their contribution to drag is tiny).  Returns ``(topo, geo, meta)``.
    """
    nz, ny, nx = solid.shape
    torch.tensor(center, dtype=torch.float64, device=solid.device)
    cx, cy, cz = center

    # --- surface fluid cells (adjacent to solid) ---
    fluid = ~solid
    near = torch.zeros_like(solid)
    for ax, sgn in [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]:
        near |= torch.roll(solid, sgn, dims=ax) & fluid
    surf = torch.nonzero(near, as_tuple=False)  # (N, 3) z,y,x
    N = surf.shape[0]
    sx, sy, sz = surf[:, 2].double(), surf[:, 1].double(), surf[:, 0].double()

    # direction (radial unit) and local frame per surface cell
    d = torch.stack([sx - cx, sy - cy, sz - cz], dim=-1)
    dnorm = d.norm(dim=-1, keepdim=True)
    n0 = d / dnorm  # (N,3) outward radial
    ex = torch.tensor([[1.0, 0.0, 0.0]], dtype=n0.dtype, device=n0.device)
    ey = torch.tensor([[0.0, 1.0, 0.0]], dtype=n0.dtype, device=n0.device)
    cross = torch.cross(n0, ex, dim=-1)
    cn = cross.norm(dim=-1)
    t_az = cross / (cn.unsqueeze(-1) + 1e-30)
    pole = cn < 1e-6
    if bool(pole.any()):
        t_az[pole] = torch.cross(n0[pole], ey, dim=-1)
    t_az = t_az / t_az.norm(dim=-1, keepdim=True)
    t_stream = torch.cross(n0, t_az, dim=-1)  # (N,3) orthonormal completion

    # stratify by (stream_bin from d_x, az_bin from azimuth)
    dx = n0[:, 0]  # cos(theta)
    phi = torch.atan2(sz - cz, sy - cy)
    phi = (phi + 2 * np.pi) % (2 * np.pi)
    keep = dx.abs() <= polar_cap
    stream_bin = ((dx + 1.0) / 2.0 * n_stream).long().clamp(0, n_stream - 1)
    az_bin = (phi / (2 * np.pi) * n_az).long() % n_az

    # map (stream_bin, az_bin) -> surface index (only for kept cells)
    surf_idx = torch.full((n_stream, n_az), -1, dtype=torch.long, device=solid.device)
    for i in range(N):
        if not bool(keep[i]):
            continue
        surf_idx[int(stream_bin[i]), int(az_bin[i])] = i

    # layer heights / centres
    h = torch.zeros(n_layers, dtype=torch.float64, device=solid.device)
    h[0] = first_height
    for k in range(1, n_layers):
        h[k] = h[k - 1] * growth
    torch.zeros(n_layers, dtype=torch.float64, device=solid.device)
    rc = torch.zeros(n_layers, dtype=torch.float64, device=solid.device)
    run = 0.0
    for k in range(n_layers):
        run += h[k] / 2.0
        rc[k] = R + run
        run += h[k] / 2.0

    n_elem = int((surf_idx >= 0).sum().item()) * n_layers
    band_coords = torch.zeros(n_elem, 3, dtype=torch.long, device=device)
    elem_of = torch.full((n_stream, n_az, n_layers), -1, dtype=torch.long, device=solid.device)

    # per-element geometry edges (for J) and centres
    center_b = torch.zeros(n_elem, 3, dtype=torch.float64, device=solid.device)
    nrm_b = torch.zeros(n_elem, 3, dtype=torch.float64, device=solid.device)
    taz_b = torch.zeros(n_elem, 3, dtype=torch.float64, device=solid.device)
    tst_b = torch.zeros(n_elem, 3, dtype=torch.float64, device=solid.device)
    hr_b = torch.zeros(n_elem, dtype=torch.float64, device=solid.device)
    haz_b = torch.zeros(n_elem, dtype=torch.float64, device=solid.device)
    hst_b = torch.zeros(n_elem, dtype=torch.float64, device=solid.device)

    dphi = 2.0 * np.pi / n_az
    b = 0
    for sb in range(n_stream):
        for ab in range(n_az):
            si = int(surf_idx[sb, ab])
            if si < 0:
                continue
            sin_th = float(torch.sqrt(torch.clamp(1.0 - dx[si] * dx[si], min=0.0)).item())
            # Clamp the azimuthal edge to the radial/streamwise scale (~1 lattice
            # unit) so elements near the (excluded) polar caps do not become
            # arbitrarily thin and blow the curvilinear CFL via c_hat ≈ 1/haz.
            haz = max(R * sin_th * dphi, 1.0)
            hst = R * (np.pi / n_stream)
            for k in range(n_layers):
                elem_of[sb, ab, k] = b
                cc = torch.tensor(
                    [sx[si].item(), sy[si].item(), sz[si].item()],
                    dtype=torch.float64,
                    device=n0.device,
                )
                cb = cc + rc[k].item() * n0[si]
                center_b[b] = cb
                nrm_b[b] = n0[si]
                taz_b[b] = t_az[si]
                tst_b[b] = t_stream[si]
                hr_b[b] = h[k].item()
                haz_b[b] = haz
                hst_b[b] = hst
                band_coords[b, 0] = int(round(cb[2].item()))
                band_coords[b, 1] = int(round(cb[1].item()))
                band_coords[b, 2] = int(round(cb[0].item()))
                b += 1
    n_band = b

    # --- Jacobian per element: J = [hr*n, haz*t_az, hst*t_stream] ---
    J = torch.stack(
        [nrm_b * hr_b.unsqueeze(-1), taz_b * haz_b.unsqueeze(-1), tst_b * hst_b.unsqueeze(-1)],
        dim=-1,
    ).to(device)
    contrav, face_J, n_phys, detJ = _geometry_from_J(J, device)

    # --- neighbour / exterior / type arrays ---
    nbr_minus = torch.full((3, n_band), -1, dtype=torch.long, device=device)
    nbr_plus = torch.full((3, n_band), -1, dtype=torch.long, device=device)
    ext_minus = torch.full((3, n_band), -1, dtype=torch.long, device=device)
    ext_plus = torch.full((3, n_band), -1, dtype=torch.long, device=device)
    type_minus = torch.ones((3, n_band), dtype=torch.int8, device=device)
    type_plus = torch.ones((3, n_band), dtype=torch.int8, device=device)

    def flat_of(pt):
        gi = torch.round(pt).long()
        gmax = torch.tensor([nz - 1, ny - 1, nx - 1], device=device).long()
        gi = gi.clamp(min=0).clamp(max=gmax)
        return (gi[:, 0] * ny * nx + gi[:, 1] * nx + gi[:, 2]).to(torch.long)

    def ext_pt(b, axis, sgn):
        # point one element-step outside in local direction (axis,sgn)
        e = [nrm_b[b] * hr_b[b], taz_b[b] * haz_b[b], tst_b[b] * hst_b[b]][axis]
        off = (0.5 + 0.5) * e  # half element + one lattice spacing
        return center_b[b].to(device) + sgn * off

    for sb in range(n_stream):
        for ab in range(n_az):
            for k in range(n_layers):
                b = int(elem_of[sb, ab, k])
                if b < 0:
                    continue
                # Always populate exterior flat indices (valid, in-bounds) so the
                # packed gather never sees -1; band-neighbour faces are overridden
                # by the torch.where(is_band) mask downstream.
                ext_minus[0, b] = flat_of(ext_pt(b, 0, -1).unsqueeze(0))[0]
                ext_plus[0, b] = flat_of(ext_pt(b, 0, +1).unsqueeze(0))[0]
                ext_minus[1, b] = flat_of(ext_pt(b, 1, -1).unsqueeze(0))[0]
                ext_plus[1, b] = flat_of(ext_pt(b, 1, +1).unsqueeze(0))[0]
                ext_minus[2, b] = flat_of(ext_pt(b, 2, -1).unsqueeze(0))[0]
                ext_plus[2, b] = flat_of(ext_pt(b, 2, +1).unsqueeze(0))[0]

                # a=0 radial: minus = inner layer (solid if k==0), plus = outer (exterior)
                if k > 0:
                    nbr_minus[0, b] = int(elem_of[sb, ab, k - 1])
                    type_minus[0, b] = 0
                else:
                    type_minus[0, b] = 2
                if k < n_layers - 1:
                    nbr_plus[0, b] = int(elem_of[sb, ab, k + 1])
                    type_plus[0, b] = 0
                else:
                    type_plus[0, b] = 1
                # a=1 azimuth: periodic wrap
                abm = (ab - 1) % n_az
                abp = (ab + 1) % n_az
                if int(elem_of[sb, abm, k]) >= 0:
                    nbr_minus[1, b] = int(elem_of[sb, abm, k])
                    type_minus[1, b] = 0
                else:
                    type_minus[1, b] = 1
                if int(elem_of[sb, abp, k]) >= 0:
                    nbr_plus[1, b] = int(elem_of[sb, abp, k])
                    type_plus[1, b] = 0
                else:
                    type_plus[1, b] = 1
                # a=2 streamwise
                if sb > 0 and int(elem_of[sb - 1, ab, k]) >= 0:
                    nbr_minus[2, b] = int(elem_of[sb - 1, ab, k])
                    type_minus[2, b] = 0
                else:
                    type_minus[2, b] = 1
                if sb < n_stream - 1 and int(elem_of[sb + 1, ab, k]) >= 0:
                    nbr_plus[2, b] = int(elem_of[sb + 1, ab, k])
                    type_plus[2, b] = 0
                else:
                    type_plus[2, b] = 1

    topo = BandTopology(
        ndim=3,
        shape=tuple(solid.shape),
        n_band=n_band,
        band_coords=band_coords,
        nbr_minus=nbr_minus,
        nbr_plus=nbr_plus,
        ext_minus_idx=ext_minus,
        ext_plus_idx=ext_plus,
        nbr_type_minus=type_minus,
        nbr_type_plus=type_plus,
        periodic=False,
    )

    # --- wall reflection map ---
    # No-slip sphere: use half-way bounce-back (OPPOSITE), identical to the
    # Cartesian band's proven-stable wall BC.  The curvature is captured by the
    # *geometry* (the wall face sits at the true radial position), not by the
    # reflection law.  "specular" (free-slip mirror) is kept as an adapter for
    # slip walls.
    if wall_bc == "specular" and vel is not None:
        specular = _specular_map(vel, nrm_b[:n_band].to(device=device, dtype=dtype), device)
    else:
        Q = OPP3D.shape[0]
        specular = OPP3D.to(device=device).to(torch.long).unsqueeze(0).expand(n_band, Q).clone()

    geo = PrismGeometry(
        contrav=contrav.to(dtype),
        face_J=face_J.to(dtype),
        n_phys=n_phys.to(dtype),
        specular=specular,
        detJ=detJ.to(dtype),
    )
    meta = dict(
        n_stream=n_stream, n_az=n_az, n_layers=n_layers, elem_of=elem_of, R=R, center=center
    )
    return topo, geo, meta


def _specular_map(vel, n_wall, device):
    """Per-element specular reflection index map about the wall normal.

    ``n_wall`` is the outward radial unit normal per element ``(n_band, 3)``
    (pointing into the fluid).  For outflow population i the reflected (incoming)
    population is the lattice velocity closest to ``c_i − 2(c_i·n̂)n̂``.
    """
    if vel is None:
        Q = 19
        return (
            torch.arange(Q, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(n_wall.shape[0], Q)
            .clone()
        )
    Q = vel.shape[0]
    nhat = n_wall.to(device).to(torch.float64)  # (n_band, 3)
    c = vel.to(device).to(torch.float64)  # (Q, 3)
    cdotn = c @ nhat.T  # (Q, n_band)
    c_refl = c.unsqueeze(1) - 2.0 * cdotn.unsqueeze(-1) * nhat.unsqueeze(0)  # (Q, n_band, 3)
    diff = c_refl.unsqueeze(2) - c.unsqueeze(0).unsqueeze(0)  # (Q, n_band, Q', 3)
    dist = diff.norm(dim=-1)  # (Q, n_band, Q')
    spec = dist.argmin(dim=2)  # (Q, n_band)
    return spec.T.to(torch.long).to(device)


# ---------------------------------------------------------------------------
# Curvilinear DG band RHS (reference-space, geometry-aware)
# ---------------------------------------------------------------------------


def dg_rhs_band_geo(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    topo: BandTopology,
    geo: PrismGeometry,
    ext_field: torch.Tensor | None = None,
    q_first: int = 0,
) -> torch.Tensor:
    """Packed-band DG advection RHS on a curvilinear prism band.

    Mirrors :func:`tensorlbm.dg_band.dg_rhs_band` but in local reference
    coordinates.  The advective speed is the contravariant ``ĉ = J^{-T} c``;
    face fluxes are scaled by ``face_J``; solid-wall ghosts use the specular
    reflection about the physical wall normal ``n_phys`` (instead of the
    axis-aligned ``OPPOSITE`` map).  With ``contrav=I, face_J=1`` and the
    specular map equal to ``OPPOSITE`` it reproduces the Cartesian operator.
    """
    ndim = topo.ndim
    n_node = ops.n_node
    velocities.shape[0]
    n_band = topo.n_band
    b_debug = False
    contrav = geo.contrav  # (n_band, 3, 3)
    specular = geo.specular  # (n_band, Q)
    detJ = geo.detJ  # (n_band,) volume weight (= face_J weight)
    dev = f_dg.device

    # ĉ[b, a, q] = Σ_c contrav[b,a,c] c[q,c]
    velocities = velocities.to(device=dev, dtype=f_dg.dtype)
    c_hat = torch.einsum("b a c, q c -> b a q", contrav, velocities)  # (n_band, 3, Q)

    # precompute specular ghost for ALL velocities (restricted later)
    spec = specular.to(dev)  # (n_band, Q)
    Q_idx = spec.T.reshape(spec.shape[1], spec.shape[0], 1, 1, 1).expand(
        spec.shape[1], spec.shape[0], *f_dg.shape[2:]
    )  # (Q, n_band, nodes)
    bb_full = f_dg.gather(0, Q_idx)  # (Q, n_band, *nodes)

    rhs = torch.zeros_like(f_dg)
    for a in range(ndim):  # local reference axis 0=radial,1=azimuth,2=streamwise
        node_axis = 2 + a  # f_dg = (Q, n_band, node0, node1, node2)
        c_along = c_hat[:, a, :]  # (n_band, Q)
        # restrict to active velocities along this axis
        c_active = c_along  # (n_band, Q)
        nonzero = c_active.abs().max(dim=0).values > 0.0  # (Q,)
        sub = f_dg[nonzero]
        c_sub = c_active[:, nonzero]  # (n_band, Q_active) -> need (Q_active, n_band)
        c_sub = c_sub.T  # (Q_active, n_band)
        ext_sub = ext_field[nonzero] if ext_field is not None else None

        # volume term: c_hat · Ax · u along node_axis
        ins = [chr(ord("a") + i) for i in range(sub.ndim)]
        outs = list(ins)
        ins[node_axis] = "u"
        outs[node_axis] = "v"
        vol = torch.einsum(f"vu,{''.join(ins)}->{''.join(outs)}", ops.Ax, sub)

        # surface term
        inner_left = sub.select(node_axis, 0)
        inner_right = sub.select(node_axis, n_node - 1)
        nbr_m = topo.nbr_minus[a]
        nbr_p = topo.nbr_plus[a]
        left_ext = _neighbour_face_value(
            sub, nbr_m, topo.ext_minus_idx[a], ext_sub, n_node - 1, node_axis
        )
        right_ext = _neighbour_face_value(sub, nbr_p, topo.ext_plus_idx[a], ext_sub, 0, node_axis)
        # solid wall ghost via specular reflection
        solid_m = topo.nbr_type_minus[a] == 2  # (n_band,)
        solid_p = topo.nbr_type_plus[a] == 2
        if bool(solid_m.any()):
            bb_left = bb_full.select(node_axis, n_node - 1)[nonzero]  # (Q_active, n_band, nodes)
            left_ext = _override_face(left_ext, bb_left, solid_m)
        if bool(solid_p.any()):
            bb_right = bb_full.select(node_axis, 0)[nonzero]
            right_ext = _override_face(right_ext, bb_right, solid_p)

        pos = (c_sub > 0.0).view(list(c_sub.shape) + [1] * (inner_left.ndim - 2))
        if b_debug:
            print(
                f"[dbg a={a}] pos{tuple(pos.shape)} L{tuple(left_ext.shape)} iL{tuple(inner_left.shape)} R{tuple(right_ext.shape)} iR{tuple(inner_right.shape)}"
            )
        uL = torch.where(pos, left_ext, inner_left)
        uR = torch.where(pos, inner_right, right_ext)

        fl_l = ops.face_lift[:, 0]
        fl_r = ops.face_lift[:, 1]
        shape = [1] * sub.ndim
        shape[node_axis] = n_node
        surf_l = fl_l.view(shape) * uL.unsqueeze(node_axis)
        surf_r = fl_r.view(shape) * uR.unsqueeze(node_axis)
        # Scale BOTH the volume and surface terms by the Jacobian determinant so
        # the reference-space divergence theorem stays balanced (a uniform field
        # must give RHS == 0).  face_J carries detJ for every face; detJ_b scales
        # the volume integral.  Using per-face areas here instead would break the
        # balance and inject energy.
        fj = detJ.view(1, n_band, *([1] * (sub.ndim - 2)))
        surf = (surf_l + surf_r) * fj

        c_view = c_sub.view(list(c_sub.shape) + [1] * (sub.ndim - 2))
        rhs_sub = c_view * (vol * fj - surf)
        rhs[nonzero] = rhs[nonzero] + rhs_sub
    return rhs


def dg_lbm_rhs_band_geo(f_dg, velocities, weights, tau, ops, topo, geo, ext_field=None):
    adv = dg_rhs_band_geo(f_dg, velocities, ops, topo, geo, ext_field)
    rho, us = macroscopic_dg(f_dg, velocities, q_first=0)
    feq = equilibrium_dg(rho, us, velocities, weights, q_first=0, ndim_field=f_dg.ndim)
    return adv - (f_dg - feq) / tau


def dg_lbm_step_band_geo(
    f_dg,
    velocities,
    weights,
    tau,
    ops,
    topo,
    geo,
    ext_field,
    dt=1.0,
    n_substeps=6,
    scheme="rk3",
):
    import math

    dev = f_dg.device
    velocities = velocities.to(device=dev, dtype=f_dg.dtype)
    weights = weights.to(device=dev, dtype=f_dg.dtype)

    min_sub = max(1, int(math.ceil(dt / (2.0 * tau))))
    if n_substeps < min_sub:
        n_substeps = min_sub
    dt_sub = dt / n_substeps

    def rhs(f):
        return dg_lbm_rhs_band_geo(f, velocities, weights, tau, ops, topo, geo, ext_field)

    def euler(f):
        return f + dt_sub * rhs(f)

    def rk3(f):
        k1 = f + dt_sub * rhs(f)
        k2 = 0.75 * f + 0.25 * (k1 + dt_sub * rhs(k1))
        return (1.0 / 3.0) * f + (2.0 / 3.0) * (k2 + dt_sub * rhs(k2))

    step = euler if scheme == "euler" else rk3
    f = f_dg
    for _ in range(n_substeps):
        f = step(f)
        f = f.clamp(min=0.0)
    return f


def hybrid_step_geo(
    f_lbm,
    f_dg,
    velocities,
    weights,
    ops,
    topo,
    geo,
    tau_lbm,
    dt=1.0,
    n_substeps=6,
    scheme="rk3",
    stream_fn=None,
    collide_fn=None,
):
    """One hybrid DG-LBM macro-step on a curvilinear prism band."""
    if stream_fn is None:
        from .solver3d import stream3d as stream_fn
    if collide_fn is None:
        from .solver3d import collide_bgk3d as collide_fn
    f_lbm = collide_fn(f_lbm, tau_lbm)
    Q, *shape = f_lbm.shape
    ext_field = f_lbm.reshape(Q, int(torch.tensor(shape).prod().item()))
    tau_dg = tau_lbm - 0.5
    f_dg = dg_lbm_step_band_geo(
        f_dg, velocities, weights, tau_dg, ops, topo, geo, ext_field, dt, n_substeps, scheme
    )
    f_lbm = stream_fn(f_lbm)
    f_lbm = write_back_exports(f_lbm, f_dg, velocities, ops, topo)
    return f_lbm, f_dg


def init_dg_from_lbm(f_lbm, topo, ops, dtype=torch.float32):
    """Seed every band element's nodal DOFs with the P0 (cell-mean) LBM value
    at the element's grid cell (constant-in-element polynomial)."""
    from .dg_advection import nodal_from_mean

    idx = tuple(topo.band_coords[:, k] for k in range(topo.ndim))
    cell_vals = f_lbm[(slice(None),) + idx]  # (Q, n_band)
    return nodal_from_mean(cell_vals, ops, node_axes=tuple(range(2, 2 + topo.ndim)))


def compute_dg_solid_force_geo(f_dg, topo, geo, velocities, ops):
    """Momentum-exchange force on the solid at the band's curved wall faces.

    Uses the specular reflection map (so the reflected population is geometrically
    correct for a curved wall) with the Ladd exchange
    ``F_α = 2 Σ_{wall b} Σ_{q: c_q·n̂>0} c_{q,α} f_dg[specular[q]]``.
    """
    ndim = topo.ndim
    n_dims = f_dg.ndim
    node_axes = tuple(range(2, n_dims))
    cell_mean = f_dg.mean(dim=node_axes)  # (Q, n_band)
    spec = geo.specular  # (n_band, Q)
    force = torch.zeros(ndim, dtype=f_dg.dtype, device=f_dg.device)
    cf = velocities.to(device=f_dg.device, dtype=f_dg.dtype)  # (Q, ndim)
    for a in range(ndim):
        for sgn, type_arr in ((+1, topo.nbr_type_plus[a]), (-1, topo.nbr_type_minus[a])):
            solid = type_arr == 2
            if not bool(solid.any()):
                continue
            nhat = geo.n_phys[a, 0 if sgn < 0 else 1]  # (n_band, 3) outward normal
            # outflow populations for this axis/face: c_q·n̂ > 0
            cdotn = cf @ nhat.T  # (Q, n_band)
            outflow = (cdotn * sgn) > 0  # (Q, n_band)
            b_idx = torch.nonzero(solid, as_tuple=False).squeeze(-1)
            for bb in b_idx.tolist():
                q_out = torch.nonzero(outflow[:, bb], as_tuple=False).squeeze(-1)
                if q_out.numel() == 0:
                    continue
                # reflected population index at this element
                q_refl = spec[bb, q_out]  # (Qo,)
                cm = cell_mean[q_refl, bb]  # (Qo,)
                force += 2.0 * (cf[q_out] * cm.unsqueeze(1)).sum(dim=0)
    return force


# ---------------------------------------------------------------------------
# Self-validation (CPU).  Run: PYTHONPATH=src python -m tensorlbm.dg_curv
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    from .d3q19 import C as C3D

    dev = torch.device("cpu")
    dtype = torch.float32

    # ---- Test 1: affine == Cartesian reproduction ----
    # A flat Cartesian shell: contrav=I, face_J=1, specular=OPPOSITE.
    # Build a small 3D Cartesian band and compare dg_rhs_band vs dg_rhs_band_geo.
    from .d3q19 import OPPOSITE as OPP3D
    from .dg_band import build_band_topology, dg_rhs_band

    nz, ny, nx = 8, 16, 16
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    # a solid block in the middle
    solid[2:6, 6:10, 6:10] = True
    band = torch.zeros_like(solid)
    for ax in (0, 1, 2):
        band |= torch.roll(solid, 1, dims=ax) & ~solid
        band |= torch.roll(solid, -1, dims=ax) & ~solid
    band &= ~solid
    topo = build_band_topology(band, solid_mask=solid, periodic=False)
    ops = get_ops(degree=1, dx=1.0, dtype=dtype)

    # random field
    torch.manual_seed(0)
    f_dg = torch.rand(19, topo.n_band, 2, 2, 2, dtype=dtype)
    ext = torch.rand(19, nz * ny * nx, dtype=dtype)
    ref = dg_rhs_band(f_dg, C3D, ops, topo, ext, opposite=OPP3D)

    # build a "Cartesian" identity geometry.  The topology's local axis g
    # indexes the *grid dimension* g (0=z,1=y,2=x), while the lattice velocity
    # columns are (cx,cy,cz).  The identity contravariant metric is therefore the
    # *reversal* permutation δ[beta, ndim-1-g], not the unit matrix.
    n_b = topo.n_band
    contrav = torch.eye(3).flip(1).expand(n_b, 3, 3).clone()
    face_J = torch.ones(3, 2, n_b)
    n_phys = torch.zeros(3, 2, n_b, 3)
    for a in range(3):
        n_phys[a, 1, :, a] = 1.0
        n_phys[a, 0, :, a] = -1.0
    specular = OPP3D.unsqueeze(0).expand(n_b, 19).clone()
    detJ = torch.ones(n_b)
    geo_cart = PrismGeometry(
        contrav=contrav, face_J=face_J, n_phys=n_phys, specular=specular, detJ=detJ
    )
    got = dg_rhs_band_geo(f_dg, C3D, ops, topo, geo_cart, ext_field=ext)

    err = (got - ref).abs().max().item()
    print(f"[T1 affine==Cartesian] max|Δrhs| = {err:.3e}")
    assert err < 1e-5, "curvilinear operator must reproduce Cartesian for identity geometry"

    # ---- Test 2: advection mass-conservation (RHS Q-sum == 0) ----
    # Use a *fully periodic* band (every cell is a band cell, no exterior
    # neighbour) so that every face is internal and the upwind fluxes cancel
    # exactly.  The DG advection RHS must then sum to zero over all populations
    # and DOFs (the BGK collision also conserves mass by construction).
    from .dg_band import dg_rhs_band

    band_full = torch.ones(nz, ny, nx, dtype=torch.bool)
    topo2 = build_band_topology(band_full, solid_mask=None, periodic=True)
    nb = topo2.n_band
    base = torch.linspace(0.5, 0.7, nb, dtype=dtype).view(1, nb, 1, 1, 1)
    f2 = (base.expand(19, nb, 2, 2, 2) + 0.05).clamp(min=1e-3)
    rhs_adv = dg_rhs_band(f2, C3D, ops, topo2, ext_field=None, opposite=OPP3D)
    total = rhs_adv.sum().item()
    print(f"[T2 advection mass-conservation] Σrhs = {total:.3e}")
    assert abs(total) < 1e-5 * max(1.0, f2.sum().item()), "advection RHS must conserve mass"

    # ---- Test 3: sphere prism-band topology + curvilinear RHS smoke ----

    nz3, ny3, nx3 = 24, 24, 48
    R = 6.0
    cx, cy, cz = nx3 / 2, ny3 / 2, nz3 / 2
    solid3 = torch.zeros(nz3, ny3, nx3, dtype=torch.bool)
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz3), torch.arange(ny3), torch.arange(nx3), indexing="ij"
    )
    solid3 = ((xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2) <= R * R
    topo_s, geo_s, meta_s = make_sphere_prism_topology(
        solid3,
        center=(cx, cy, cz),
        R=R,
        n_layers=3,
        first_height=0.5,
        growth=1.1,
        n_az=16,
        vel=C3D,
        dtype=dtype,
        device=dev,
    )
    print(
        f"[T3 sphere band] n_band={topo_s.n_band}  n_az={meta_s['n_az']}  n_layers={meta_s['n_layers']}"
    )
    assert topo_s.n_band > 0, "sphere band must contain elements"
    # geometry factors finite & specular valid
    assert torch.isfinite(geo_s.contrav).all(), "contrav must be finite"
    assert torch.isfinite(geo_s.face_J).all() and (geo_s.face_J > 0).all(), (
        "face_J must be finite positive"
    )
    assert geo_s.specular.min() >= 0 and geo_s.specular.max() < 19, "specular map out of range"
    # wall normals (inner radial face) should point roughly radially outward
    nw = geo_s.n_phys[0, 0]  # (n_band, 3) inward normal of inner face
    rad = torch.stack(
        [
            topo_s.band_coords[:, 2].double() - cx,
            topo_s.band_coords[:, 1].double() - cy,
            topo_s.band_coords[:, 0].double() - cz,
        ],
        dim=-1,
    )
    rad = rad / (rad.norm(dim=-1, keepdim=True) + 1e-12)
    align = (nw.double() * rad.to(dev)).sum(dim=-1).abs().mean().item()
    print(f"[T3 sphere band] mean|wall-normal·radial| = {align:.3f} (expect ~1)")
    assert align > 0.9, "wall normals must align with sphere radial direction"

    # curvilinear RHS on a smooth field must be finite
    torch.manual_seed(1)
    f_s = torch.rand(19, topo_s.n_band, 2, 2, 2, dtype=dtype) * 0.05 + 0.5
    ext_s = torch.rand(19, nz3 * ny3 * nx3, dtype=dtype) * 0.05 + 0.5
    rhs_s = dg_rhs_band_geo(f_s, C3D, ops, topo_s, geo_s, ext_field=ext_s)
    assert torch.isfinite(rhs_s).all(), "curvilinear sphere RHS must be finite"
    print(f"[T3 sphere band] curvilinear RHS finite, |rhs|max={rhs_s.abs().max():.3e}")
    print("dg_curv self-test: PASSED")
