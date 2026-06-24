"""Hybrid DG-band + LBM-exterior coupling for the DG-LBM.

This module layers the *hybrid* coupling on top of the validated full-grid DG
advection kernel in :mod:`tensorlbm.dg_advection`:

* A subset of grid cells — the **DG band**, typically a near-wall shell around
  an obstacle — carry polynomial (P1) degrees of freedom stored in a *packed*
  layout ``(Q, n_band, *node_axes)`` (memory-feasible at SUBOFF scale, unlike a
  full-grid DOF tensor).
* The remaining **exterior** cells keep the cheap, exact-shift LBM streaming
  (one value per cell).
* At DG↔LBM interface faces the DG upwind flux reads the exterior P0 cell value
  as its ghost state, and the exterior cell ingests the DG face trace — a single,
  non-double-writing exchange at each macro-step.

The packed per-axis DG operator is mathematically identical to the full-grid one
(:func:`tensorlbm.dg_advection.dg_rhs`); only the neighbour fetch changes from a
``torch.roll`` (periodic full grid) to an explicit neighbour-index gather (packed
band, heterogeneous neighbours).  When the band is the whole domain and the
domain is periodic, :func:`dg_rhs_band` reproduces :func:`dg_rhs` element-wise —
the gate test for the packed operator.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .dg_advection import _Ops, equilibrium_dg, get_ops, macroscopic_dg


@dataclass
class BandTopology:
    """Packed-band neighbour structure for hybrid DG-LBM.

    Attributes:
        ndim: 2 or 3.
        shape: spatial grid shape ``(nz, ny, nx)`` / ``(ny, nx)``.
        n_band: number of band cells.
        band_coords: ``(n_band, ndim)`` int grid coordinate of each band cell
            (in ``(z, y, x)`` / ``(y, x)`` order).
        nbr_minus / nbr_plus: per-axis ``(ndim, n_band)`` int tensors. For band
            cell *b*, ``nbr_minus[a, b]`` is the band index of its neighbour in
            the −a direction, or ``-1`` if that neighbour is exterior (LBM /
            wall / domain boundary).
        ext_minus_idx / ext_plus_idx: per-axis ``(ndim, n_band)`` flat grid
            indices of the neighbour cell in the ±a direction (used to gather the
            exterior P0 value when the neighbour is not a band cell; valid for
            all cells, consulted only where ``nbr_* == -1``).
        periodic: whether the domain wraps (periodic) at its outer boundary.
    """

    ndim: int
    shape: tuple[int, ...]
    n_band: int
    band_coords: torch.Tensor
    nbr_minus: torch.Tensor
    nbr_plus: torch.Tensor
    ext_minus_idx: torch.Tensor
    ext_plus_idx: torch.Tensor
    nbr_type_minus: torch.Tensor       # (ndim, n_band) int8: 0=band,1=exterior,2=solid
    nbr_type_plus: torch.Tensor
    periodic: bool = True

    def to(self, device: torch.device) -> "BandTopology":
        return BandTopology(
            ndim=self.ndim,
            shape=self.shape,
            n_band=self.n_band,
            band_coords=self.band_coords.to(device),
            nbr_minus=self.nbr_minus.to(device),
            nbr_plus=self.nbr_plus.to(device),
            ext_minus_idx=self.ext_minus_idx.to(device),
            ext_plus_idx=self.ext_plus_idx.to(device),
            nbr_type_minus=self.nbr_type_minus.to(device),
            nbr_type_plus=self.nbr_type_plus.to(device),
            periodic=self.periodic,
        )


def build_band_topology(
    band_mask: torch.Tensor,
    solid_mask: torch.Tensor | None = None,
    periodic: bool = True,
) -> BandTopology:
    """Build a :class:`BandTopology` from a boolean band mask.

    Args:
        band_mask: ``(nz, ny, nx)`` or ``(ny, nx)`` boolean tensor; *True* = DG
            band cell.
        solid_mask: optional obstacle mask (same shape). A band-cell neighbour
            that lies in the solid is marked type 2 (bounce-back wall) rather
            than exterior.
        periodic: domain periodicity (only affects boundary cells' exterior
            neighbour coordinates).
    """
    shape = tuple(band_mask.shape)
    ndim = len(shape)
    device = band_mask.device

    coords = torch.nonzero(band_mask, as_tuple=False)          # (n_band, ndim)
    n_band = coords.shape[0]
    grid_to_band = torch.full(shape, -1, dtype=torch.long, device=device)
    grid_to_band[tuple(coords.t())] = torch.arange(n_band, device=device)

    # Flat grid index of each cell (for exterior-value gathers).
    flat = torch.arange(band_mask.numel(), device=device).reshape(shape)

    nbr_minus = torch.empty((ndim, n_band), dtype=torch.long, device=device)
    nbr_plus = torch.empty((ndim, n_band), dtype=torch.long, device=device)
    ext_minus = torch.empty((ndim, n_band), dtype=torch.long, device=device)
    ext_plus = torch.empty((ndim, n_band), dtype=torch.long, device=device)
    # Default: every non-band neighbour is exterior (type 1).
    type_minus = torch.ones((ndim, n_band), dtype=torch.int8, device=device)
    type_plus = torch.ones((ndim, n_band), dtype=torch.int8, device=device)

    for a in range(ndim):
        # Shift the band cells by ±1 along axis a; wrap if periodic else clamp.
        c_minus = coords.clone()
        c_plus = coords.clone()
        c_minus[:, a] -= 1
        c_plus[:, a] += 1
        if periodic:
            c_minus[:, a] %= shape[a]
            c_plus[:, a] %= shape[a]
        else:
            c_minus[:, a] = c_minus[:, a].clamp_(0, shape[a] - 1)
            c_plus[:, a] = c_plus[:, a].clamp_(0, shape[a] - 1)

        nbr_minus[a] = grid_to_band[tuple(c_minus.t())]
        nbr_plus[a] = grid_to_band[tuple(c_plus.t())]
        ext_minus[a] = flat[tuple(c_minus.t())]
        ext_plus[a] = flat[tuple(c_plus.t())]
        # Band neighbours are type 0.
        is_band_m = nbr_minus[a] >= 0
        is_band_p = nbr_plus[a] >= 0
        type_minus[a][is_band_m] = 0
        type_plus[a][is_band_p] = 0
        # Solid neighbours are type 2 (overrides exterior default).
        if solid_mask is not None:
            solid_m = solid_mask[tuple(c_minus.t())]
            solid_p = solid_mask[tuple(c_plus.t())]
            type_minus[a][solid_m] = 2
            type_plus[a][solid_p] = 2

    return BandTopology(
        ndim=ndim,
        shape=shape,
        n_band=n_band,
        band_coords=coords,
        nbr_minus=nbr_minus,
        nbr_plus=nbr_plus,
        ext_minus_idx=ext_minus,
        ext_plus_idx=ext_plus,
        nbr_type_minus=type_minus,
        nbr_type_plus=type_plus,
        periodic=periodic,
    )


def _neighbour_face_value(
    f_dg: torch.Tensor,
    nbr_idx: torch.Tensor,        # (n_band,) band index or -1
    ext_flat_idx: torch.Tensor,   # (n_band,) flat grid index for exterior lookup
    ext_field: torch.Tensor | None,  # (Q, Ncell) exterior P0 values or None
    gather_node: int,             # node index along this axis to read from a band nbr
    node_axis: int,               # position of this axis's node dim in f_dg
) -> torch.Tensor:
    """Face-trace value supplied by the neighbour of every band cell.

    Where the neighbour is itself a band cell, read its face node (DOF); where it
    is exterior, read the P0 value from *ext_field* at the neighbour's flat grid
    index.  Returns ``(Q, n_band, *transverse_node_axes)``.
    """
    n_band = f_dg.shape[1]
    is_band = nbr_idx >= 0
    safe_idx = nbr_idx.clamp(min=0)                               # (n_band,)

    # Band-neighbour face value: gather along the cell axis, then pick the
    # neighbour's far-face node (node_axis is an absolute axis index, already
    # past Q and n_band).
    band_val = f_dg.index_select(1, safe_idx)                     # (Q, n_band, *nodes)
    band_val = band_val.select(node_axis, gather_node)            # → (Q, n_band, *other_nodes)

    if ext_field is None:
        # No exterior possible (band == whole domain); every neighbour is band.
        return band_val

    # Exterior P0 value: (Q, Ncell) → gather (Q, n_band) → broadcast over nodes.
    ext_val = ext_field.index_select(1, ext_flat_idx)             # (Q, n_band)
    # Reshape to broadcast against band_val's node axes.
    ext_shape = [ext_val.shape[0], ext_val.shape[1]] + [1] * (band_val.ndim - 2)
    ext_val = ext_val.view(ext_shape).expand_as(band_val)

    mask = is_band.view(1, n_band, *([1] * (band_val.ndim - 2)))
    return torch.where(mask, band_val, ext_val)


def _override_face(face_val: torch.Tensor, bb_val: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Override *face_val* with *bb_val* where the per-band *mask* is True."""
    shape = [1, mask.shape[0]] + [1] * (face_val.ndim - 2)
    m = mask.to(face_val.dtype).view(shape)
    return m * bb_val + (1.0 - m) * face_val


def dg_rhs_band(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    topo: BandTopology,
    ext_field: torch.Tensor | None = None,
    opposite: torch.Tensor | None = None,
    q_first: int = 0,
) -> torch.Tensor:
    """Packed-band DG advection RHS (dimension-by-dimension, upwind flux).

    Structurally identical to :func:`dg_advection.dg_rhs` but operating on the
    packed band layout with heterogeneous neighbours (band / exterior P0 / solid
    wall).  Solid-wall neighbours use a half-way bounce-back ghost: the inflow
    population *i* sees ``f[OPPOSITE[i]]`` at the shared face node (no-slip to
    2nd order), requiring the lattice *opposite* index map.

    Args:
        f_dg: ``(Q, n_band, *node_axes)`` packed nodal DOFs (node axes last, in
            ``(z, y, x)`` order, so the x-node axis is last).
        velocities: ``(Q, ndim)``.
        ops: 1D DG operators.
        topo: band neighbour structure.
        ext_field: ``(Q, Ncell)`` flattened exterior P0 values (only needed when
            some neighbour is exterior; *None* when the band is the whole domain).
        opposite: ``(Q,)`` lattice opposite-direction index map (for solid walls).
    """
    ndim = topo.ndim
    n_node = ops.n_node
    device = f_dg.device
    n_dims = f_dg.ndim
    # Layout: f_dg = (Q, n_band, *nodes) with node axes in (z, y, x) order, so the
    # x-node axis is LAST.  Lattice velocity columns are (cx, cy, cz) = v=0,1,2.
    # Velocity column v maps to grid dim (ndim-1-v) [x=last] and node axis
    # (n_dims-1-v) [px=last].  Grid dim g maps to topo.nbr_minus[g].
    letters = "abcdefghijklmnopqrst"

    rhs = torch.zeros_like(f_dg)
    for v in range(ndim):                                  # v=0:x, 1:y, 2:z
        c_axis = velocities[:, v].to(f_dg.dtype)
        if c_axis.abs().max().item() == 0.0:
            continue
        node_axis = n_dims - 1 - v                          # px (last), py, pz
        g = ndim - 1 - v                                    # grid dim: x, y, z
        nonzero = c_axis.abs() > 0.0
        sub = f_dg[nonzero]
        c_sub = c_axis[nonzero]
        # Match the exterior ghost field to the active velocities on this axis.
        ext_sub = ext_field[nonzero] if ext_field is not None else None

        # --- Volume term: c · Ax · u along this node axis ---
        ins = [letters[i] for i in range(sub.ndim)]
        outs = list(ins)
        ins[node_axis] = "u"
        outs[node_axis] = "v"
        vol = torch.einsum(f"vu,{''.join(ins)}->{''.join(outs)}", ops.Ax, sub)

        # --- Surface term: upwind face traces from neighbours ---
        inner_left = sub.select(node_axis, 0)        # u_e[0]
        inner_right = sub.select(node_axis, n_node - 1)
        nbr_m = topo.nbr_minus[g]
        nbr_p = topo.nbr_plus[g]
        left_ext = _neighbour_face_value(
            sub, nbr_m, topo.ext_minus_idx[g], ext_sub, n_node - 1, node_axis
        )                                              # neighbour's far-face node
        right_ext = _neighbour_face_value(
            sub, nbr_p, topo.ext_plus_idx[g], ext_sub, 0, node_axis
        )
        # --- Solid-wall bounce-back ghost (half-way, no-slip) ---
        # For a solid neighbour, the inflow population i sees the band's own
        # f[OPPOSITE[i]] at the shared face node (the reflected population).
        # OPPOSITE maps over the full Q axis, so compute on the full f_dg then
        # restrict to the active velocities on this axis.
        if opposite is not None:
            opp = opposite.to(f_dg.device)
            solid_m = topo.nbr_type_minus[g] == 2      # (n_band,)
            solid_p = topo.nbr_type_plus[g] == 2
            if bool(solid_m.any()):
                bb_left = f_dg.index_select(0, opp).select(node_axis, n_node - 1)[nonzero]
                left_ext = _override_face(left_ext, bb_left, solid_m)
            if bool(solid_p.any()):
                bb_right = f_dg.index_select(0, opp).select(node_axis, 0)[nonzero]
                right_ext = _override_face(right_ext, bb_right, solid_p)
        pos = c_sub.view([c_sub.shape[0]] + [1] * (inner_left.ndim - 1)) > 0.0
        uL = torch.where(pos, left_ext, inner_left)
        uR = torch.where(pos, inner_right, right_ext)

        fl_l = ops.face_lift[:, 0]
        fl_r = ops.face_lift[:, 1]
        shape = [1] * sub.ndim
        shape[node_axis] = n_node
        surf = fl_l.view(shape) * uL.unsqueeze(node_axis) + fl_r.view(shape) * uR.unsqueeze(node_axis)

        c_view = c_sub.view([c_sub.shape[0]] + [1] * (sub.ndim - 1))
        rhs_sub = c_view * vol - c_view * surf
        rhs[nonzero] = rhs[nonzero] + rhs_sub
    return rhs


def dg_lbm_rhs_band(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    ops: _Ops,
    topo: BandTopology,
    ext_field: torch.Tensor | None = None,
    opposite: torch.Tensor | None = None,
) -> torch.Tensor:
    """Method-of-lines RHS on the packed band: DG advection + BGK collision.

    ``df/dt = (DG advection RHS) − (f − f_eq)/τ``.  Stable (no collide/advect
    splitting instability) and recovers the DVBE viscosity ν = τ/3 in the band,
    so the band uses ``τ_dg = τ_lbm − ½`` to match the exterior LBM viscosity.
    """
    adv = dg_rhs_band(f_dg, velocities, ops, topo, ext_field, opposite)
    rho, us = macroscopic_dg(f_dg, velocities, q_first=0)
    feq = equilibrium_dg(rho, us, velocities, weights, q_first=0, ndim_field=f_dg.ndim)
    return adv - (f_dg - feq) / tau


def dg_lbm_step_band(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    ops: _Ops,
    topo: BandTopology,
    ext_field: torch.Tensor | None,
    dt: float,
    n_substeps: int = 6,
    scheme: str = "rk3",
    opposite: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sub-cycled SSP-RK3 advance of the band (advection + collision), frozen ghosts."""
    dt_sub = dt / n_substeps

    def rhs(f: torch.Tensor) -> torch.Tensor:
        return dg_lbm_rhs_band(f, velocities, weights, tau, ops, topo, ext_field, opposite)

    def euler(f: torch.Tensor) -> torch.Tensor:
        return f + dt_sub * rhs(f)

    def rk3(f: torch.Tensor) -> torch.Tensor:
        k1 = f + dt_sub * rhs(f)
        k2 = 0.75 * f + 0.25 * (k1 + dt_sub * rhs(k1))
        return (1.0 / 3.0) * f + (2.0 / 3.0) * (k2 + dt_sub * rhs(k2))

    step = euler if scheme == "euler" else rk3
    f = f_dg
    for _ in range(n_substeps):
        f = step(f)
    return f


def dg_advect_band(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    topo: BandTopology,
    ext_field: torch.Tensor | None,
    dt: float,
    n_substeps: int = 1,
    scheme: str = "rk3",
    opposite: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sub-cycled DG advection on the packed band with a *frozen* exterior field.

    The exterior P0 values (*ext_field*) are held constant across the sub-steps
    (the exterior is advanced once per macro-step by the LBM stream, not here).
    """
    dt_sub = dt / n_substeps

    def rhs(f: torch.Tensor) -> torch.Tensor:
        return dg_rhs_band(f, velocities, ops, topo, ext_field=ext_field, opposite=opposite)

    def euler(f: torch.Tensor) -> torch.Tensor:
        return f + dt_sub * rhs(f)

    def rk3(f: torch.Tensor) -> torch.Tensor:
        k1 = f + dt_sub * rhs(f)
        k2 = 0.75 * f + 0.25 * (k1 + dt_sub * rhs(k1))
        return (1.0 / 3.0) * f + (2.0 / 3.0) * (k2 + dt_sub * rhs(k2))

    step = euler if scheme == "euler" else rk3
    f = f_dg
    for _ in range(n_substeps):
        f = step(f)
    return f


def write_back_exports(
    f_lbm: torch.Tensor,
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    topo: BandTopology,
) -> torch.Tensor:
    """Inject DG face traces into the exterior LBM cells (conservative coupling).

    For every band cell whose neighbour across a face is an exterior cell, and
    every population that flows *out* of the band through that face, overwrite
    the exterior cell's (post-stream) population with the **face-averaged** DG
    trace.  This is the single, non-double-writing DG→LBM exchange: the exterior
    stream already ran, reading stale band "holes" for these cells, and this
    corrects them with the true upwind DG value.

    The face average uses Lobatto quadrature over the transverse node axes (for
    P1, a plain mean), matching the DG flux so the exchange is conservative.
    """
    ndim = topo.ndim
    n_node = ops.n_node
    n_dims = f_dg.ndim
    Q = f_lbm.shape[0]
    shape = f_lbm.shape[1:]
    N = int(torch.tensor(shape).prod().item())
    flb = f_lbm.reshape(Q, N).clone()                   # work in flat grid indices

    for g in range(ndim):                               # grid dim (0=z .. ndim-1=x)
        v = ndim - 1 - g                                # velocity column (x⇒cx)
        comp = velocities[:, v].to(f_dg.dtype)
        node_axis = n_dims - ndim + g                   # node axis for this grid dim
        for sgn, nbr_arr, ext_arr, type_arr, nidx in (
            (+1, topo.nbr_plus[g], topo.ext_plus_idx[g], topo.nbr_type_plus[g], n_node - 1),
            (-1, topo.nbr_minus[g], topo.ext_minus_idx[g], topo.nbr_type_minus[g], 0),
        ):
            outflow = ((comp * sgn) > 0)                # (Q,) populations leaving the band
            if not outflow.any():
                continue
            ext_face = type_arr == 1                    # exterior (not band, not solid)
            if not bool(ext_face.any()):
                continue
            out_q = torch.nonzero(outflow, as_tuple=False).squeeze(-1)
            b_idx = torch.nonzero(ext_face, as_tuple=False).squeeze(-1)

            # DG face trace at node nidx along node_axis, averaged over transverse nodes.
            sub = f_dg.index_select(0, out_q).index_select(1, b_idx)  # (Qo, Nb, *nodes)
            face_vals = sub.select(node_axis, nidx)     # (Qo, Nb, *transverse_nodes)
            transverse = [ax for ax in range(face_vals.ndim) if ax >= 2]
            trace = face_vals.mean(dim=transverse)      # (Qo, Nb)

            tgt = ext_arr[b_idx]                        # (Nb,) flat exterior indices
            # Scatter flb[out_q, tgt] = trace (advanced-index assignment).
            rows = out_q.view(-1, 1).expand_as(trace)
            cols = tgt.view(1, -1).expand_as(trace)
            flb[rows, cols] = trace
    return flb.reshape(Q, *shape)


def hybrid_advect(
    f_lbm: torch.Tensor,
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    topo: BandTopology,
    dt: float = 1.0,
    n_substeps: int = 6,
    scheme: str = "rk3",
    stream_fn=None,
    opposite: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One hybrid advection macro-step (no collision): DG-band advect + LBM stream.

    Order (avoids the double-write interface hazard):
      1. Freeze exterior P0 values as the band's ghost field.
      2. Sub-cycle DG advection on the band (ghosts frozen).
      3. Stream the exterior (standard periodic LBM shift).
      4. Overwrite band-adjacent exterior populations with DG face traces.

    Args:
        f_lbm: ``(Q, *shape)`` exterior (and band-hole) populations on the full grid.
        f_dg: ``(Q, n_band, *nodes)`` packed band DOFs.
        stream_fn: callable ``(f) -> f`` for the exterior stream (defaults to the
            2D/3D periodic stream from the lattice module).

    Returns the updated ``(f_lbm, f_dg)``.
    """
    if stream_fn is None:
        stream_fn = _default_stream(topo.ndim)
    Q, *shape = f_lbm.shape
    ext_field = f_lbm.reshape(Q, int(torch.tensor(shape).prod().item()))

    f_dg = dg_advect_band(f_dg, velocities, ops, topo, ext_field, dt, n_substeps, scheme, opposite)
    f_lbm = stream_fn(f_lbm)
    f_lbm = write_back_exports(f_lbm, f_dg, velocities, ops, topo)
    return f_lbm, f_dg


def _default_stream(ndim: int):
    if ndim == 2:
        from .solver import stream as _s  # noqa: PLC0415
    else:
        from .solver3d import stream3d as _s  # noqa: PLC0415
    return _s


def _default_collide(ndim: int):
    if ndim == 2:
        from .solver import collide_bgk as _c  # noqa: PLC0415
    else:
        from .solver3d import collide_bgk3d as _c  # noqa: PLC0415
    return _c


def hybrid_step(
    f_lbm: torch.Tensor,
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    ops: _Ops,
    topo: BandTopology,
    tau_lbm: float,
    dt: float = 1.0,
    n_substeps: int = 6,
    scheme: str = "rk3",
    opposite: torch.Tensor | None = None,
    stream_fn=None,
    collide_fn=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One hybrid DG-LBM macro-step with collision.

    Sequence (collide-then-stream on the exterior; method-of-lines on the band):

      1. Collide the exterior LBM field with τ_lbm (standard BGK).
      2. Advance the band by method-of-lines (advection + collision, τ_dg =
         τ_lbm − ½ so the band's DVBE viscosity τ_dg/3 matches the exterior's
         (τ_lbm − ½)/3).  Exterior ghosts are frozen across the sub-steps.
      3. Stream the exterior.
      4. Write DG face traces into band-adjacent exterior cells.

    Band cells in *f_lbm* are "holes" (collided/streamed but never read — the
    band uses *f_dg*); they cost nothing.
    """
    if stream_fn is None:
        stream_fn = _default_stream(topo.ndim)
    if collide_fn is None:
        collide_fn = _default_collide(topo.ndim)

    # 1. Collide exterior (band holes are collided too but ignored).
    f_lbm = collide_fn(f_lbm, tau_lbm)

    # 2. Band method-of-lines step (frozen exterior ghosts).
    Q, *shape = f_lbm.shape
    ext_field = f_lbm.reshape(Q, int(torch.tensor(shape).prod().item()))
    tau_dg = tau_lbm - 0.5
    f_dg = dg_lbm_step_band(
        f_dg, velocities, weights, tau_dg, ops, topo, ext_field,
        dt, n_substeps, scheme, opposite,
    )

    # 3. Stream exterior.  4. Write-back DG traces.
    f_lbm = stream_fn(f_lbm)
    f_lbm = write_back_exports(f_lbm, f_dg, velocities, ops, topo)
    return f_lbm, f_dg

