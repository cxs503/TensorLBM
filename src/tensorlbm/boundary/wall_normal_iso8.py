"""ISO8 wall-normal table for curved/voxelised walls (3-D), after lanl/MF-LBM.

What this module is
-------------------
A geometry-preprocessing *capability*, not a solver: given a voxelised solid
mask it produces, once per geometry, a table of unit wall normals on every fluid
node that touches the solid -- the same object MF-LBM stores as
``fluid_boundary_nodes % snx/sny/snz`` and consumes in ``Phase_gradient.F90`` to
impose a geometric (contact-angle) wetting condition.

Why a table of normals at all
-----------------------------
Curvature-dependent boundary conditions (geometric wetting, contact angle,
surface-tension / Laplace-pressure closures, second-order slip) need the wall
orientation at the *staircase* boundary.  On a staircase the arithmetic mean of
the four axis directions is anisotropic by O(1) at 45 degrees, so the normals
have to come from an isotropic discrete gradient of a smoothed indicator field.
MF-LBM uses an "ISO8" 34-pair (5x5x5) isotropic stencil applied to the solid
fraction smoothed by four D3Q27 convolutions.

Provenance (zero hand-copy)
---------------------------
Every constant in this module was extracted *programmatically* from
``xval/XVAL-LBM-ISO8-WALL-NORMAL-TABLE/oss_src/wall_boundary_process.f90``
(lanl/MF-LBM, BSD-3-Clause, upstream commit 11be324):

* ``ISO8_WEIGHTS``     <- ``ISO8(1:7)`` in that file
* ``_ISO8_DIFF_OFFSETS`` <- the ``A - B`` pairs of the ``nwx = ISO8(1)*(...)+...``
  expression (the offsets stored are the *difference* offsets ``d = A - B``;
  the symmetric half-offset actually used by the gradient is ``d // 2``)
* ``D3Q27_KERNEL``    <- ``w(0:3)`` and ``ex/ey/ez(0:26)`` of the same file
* ``HALF18``          <- ``ex/ey/ez(1:18)``: MF-LBM's 18-direction half set used
  for boundary-node neighbour lists (note: *not* the full 26)
* ``W_EQU_HALF18``    <- ``w_equ(0:18)``: the weighting used for ``la_weight``

``tests/test_wall_normal_iso8.py`` re-parses the vendored source and asserts
that these literals still agree with it, so the transplant cannot silently
drift from upstream.

Boundary-condition convention
-----------------------------
The normal is the normalised gradient of the solid fraction, i.e. it points
*into the solid* (MF-LBM's convention; the wetting code reflects the interface
normal about it).  Sign is therefore NOT the usual outward fluid normal.

Practical notes
---------------
* The OSS preprocessing domain is padded with ``ghost_layers + overlap`` cells;
  boundary nodes are listed for Fortran indices ``1-overlap .. n+overlap``,
  i.e. the table also contains nodes *outside* the physical box (MF-LBM is run
  on decomposed subdomains, where that overlap is what the neighbouring rank
  needs).  This is reproduced exactly so the two tables can be compared
  elementwise; callers that only want physical-domain nodes should filter with
  :func:`in_physical_domain`.
* Everything is float64 / float32-agnostic NumPy; no torch dependency, so the
  preprocessing step can run on the CPU control plane before a simulation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Sequence, Tuple

import numpy as np

__all__ = [
    "ISO8_WEIGHTS",
    "ISO8_PAIRS",
    "iso8_stencil_scale",
    "D3Q27_KERNEL",
    "HALF18",
    "W_EQU_HALF18",
    "smooth_solid_fraction",
    "classify_boundary_nodes",
    "wall_normal_table",
    "iso8_gradient",
    "iso8_divergence",
    "in_physical_domain",
]

# --------------------------------------------------------------------------- #
# stencil tables (extracted from the OSS source -- see module docstring)
# --------------------------------------------------------------------------- #

ISO8_WEIGHTS: Tuple[float, ...] = (
    0.088888888888888892,
    0.047619047619047616,
    0.019047619047619049,
    0.0099206349206349201,
    0.0031746031746031746,
    0.0015873015873015873,
    0.00019841269841269841,
)

#: ``A - B`` offsets of the ``ISO8(n)*(A - B ...)`` blocks, as written upstream.
_ISO8_DIFF_OFFSETS: Tuple[Tuple[int, Tuple[int, int, int]], ...] = (
    (0, (2, 0, 0)),
    (1, (2, 2, 0)), (1, (2, -2, 0)), (1, (2, 0, 2)), (1, (2, 0, -2)),
    (2, (2, 2, 2)), (2, (2, 2, -2)), (2, (2, -2, 2)), (2, (2, -2, -2)),
    (3, (4, 0, 0)),
    (4, (4, 2, 0)), (4, (4, -2, 0)), (4, (4, 0, 2)), (4, (4, 0, -2)),
    (4, (2, 4, 0)), (4, (2, -4, 0)), (4, (2, 0, 4)), (4, (2, 0, -4)),
    (5, (4, 2, 2)), (5, (4, 2, -2)), (5, (4, -2, 2)), (5, (4, -2, -2)),
    (5, (2, 4, 2)), (5, (2, 4, -2)), (5, (2, -4, 2)), (5, (2, -4, -2)),
    (5, (2, 2, 4)), (5, (2, 2, -4)), (5, (2, -2, 4)), (5, (2, -2, -4)),
    (6, (4, 4, 0)), (6, (4, -4, 0)), (6, (4, 0, 4)), (6, (4, 0, -4)),
)

#: ``(weight, half-offset h)`` pairs of the **x** component of the gradient.
#: The three components do *not* share one pair set: the x block carries the xy
#: and xz edge pairs while the y block carries xy and yz, etc.  The y/z sets are
#: the xy/xz axis permutations of this one (asserted against the source by
#: ``tests/test_wall_normal_iso8.py``); using a single set for all components
#: makes the operator anisotropic and tilts the normals by O(10 deg).
_ISO8_PAIRS_X: Tuple[Tuple[float, Tuple[int, int, int]], ...] = tuple(
    (ISO8_WEIGHTS[wi], (dx // 2, dy // 2, dz // 2)) for wi, (dx, dy, dz) in _ISO8_DIFF_OFFSETS
)


def _axis_permute(pairs, a: int, b: int):
    out = []
    for w, h in pairs:
        h2 = list(h)
        h2[a], h2[b] = h[b], h[a]
        out.append((w, tuple(h2)))
    return tuple(out)


#: per-component pair sets of the isotropic gradient, keyed by gradient axis
ISO8_PAIRS: Dict[int, Tuple[Tuple[float, Tuple[int, int, int]], ...]] = {
    0: _ISO8_PAIRS_X,
    1: _axis_permute(_ISO8_PAIRS_X, 0, 1),
    2: _axis_permute(_ISO8_PAIRS_X, 0, 2),
}

#: D3Q27 offsets and their weights by squared speed (upstream ``ex/ey/ez``/``w``).
D3Q27_OFFSETS: Tuple[Tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
    (1, 1, -1), (-1, -1, 1), (1, 1, 1), (-1, -1, -1),
    (1, -1, -1), (-1, 1, 1), (1, -1, 1), (-1, 1, -1),
)

D3Q27_KERNEL: Tuple[float, float, float, float] = (
    0.29629629629629628,   # 8/27  (rest)
    0.07407407407407407,   # 2/27  (|e|^2 = 1)
    0.018518518518518517,  # 1/54  (|e|^2 = 2)
    0.004629629629629629,  # 1/216 (|e|^2 = 3)
)

#: 18-direction half set used for boundary-node neighbour lists (ex/ey/ez(1:18)).
HALF18: Tuple[Tuple[int, int, int], ...] = (
    (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1),
    (1, 1, 0), (-1, 1, 0), (1, -1, 0), (-1, -1, 0),
    (1, 0, 1), (-1, 0, 1), (1, 0, -1), (-1, 0, -1),
    (0, 1, 1), (0, -1, 1), (0, 1, -1), (0, -1, -1),
)

#: ``w_equ(0:18)`` of the OSS source: index 1..18 = weights of HALF18.
W_EQU_HALF18: Tuple[float, ...] = (
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
)

# defaults, read from the OSS source (i_max_iteration / ghost_layers / overlap_phi)
DEFAULT_SMOOTH_ITERATIONS = 4
DEFAULT_GHOST_LAYERS = 6
DEFAULT_OVERLAP = 4


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #

def _shifted(field: np.ndarray, off: Sequence[int]) -> np.ndarray:
    """``out[p] = field[p + off]`` (zero outside), same shape as ``field``."""
    out = np.zeros_like(field)
    src = []
    dst = []
    for ax, o in enumerate(off):
        n = field.shape[ax]
        if o >= 0:
            src.append(slice(o, n))
            dst.append(slice(0, n - o))
        else:
            src.append(slice(0, n + o))
            dst.append(slice(-o, n))
    out[tuple(dst)] = field[tuple(src)]
    return out


def _convolve_d3q27(field: np.ndarray) -> np.ndarray:
    out = np.zeros_like(field)
    for off in D3Q27_OFFSETS:
        out += D3Q27_KERNEL[off[0] ** 2 + off[1] ** 2 + off[2] ** 2] * _shifted(field, off)
    return out


def _interior_slice(n_pad: int, ghost: int) -> Tuple[slice, slice, slice]:
    """Index range the OSS smoothing loops touch: Fortran ``2-ghost .. n+ghost-1``.

    In padded indexing (``p = i + ghost - 1``) that is ``1 .. n_pad-2``: only the
    outermost layer of the padded array keeps its initial (unsmoothed) value.
    """
    return tuple(slice(1, n_pad - 1) for _ in range(3))  # type: ignore[return-value]


def _pad_solid(walls: np.ndarray, ghost: int, periodic=(0, 0, 0),
               domain_walls=(0, 0, 0, 0, 0, 0), mirror_bc: int = 0) -> np.ndarray:
    """Reproduce ``initialization`` of the OSS preprocessor (ghost fill included).

    Fortran index ``i`` in ``1-ghost .. n+ghost`` maps to padded index
    ``p = i + ghost - 1``; the ghost fill order (z, then y, then x) and the
    extent of every fill are copied from upstream, because the x/y fills sweep
    over the already-filled ghost planes.
    """
    walls = np.asarray(walls)
    if walls.ndim != 3:
        raise ValueError("walls must be a 3-D array")
    nx, ny, nz = walls.shape
    n = (nx, ny, nz)
    g = ghost
    W = np.zeros(tuple(ni + 2 * g for ni in n), dtype=np.int32)
    W[g:g + nx, g:g + ny, g:g + nz] = walls.astype(np.int32)

    def fp(k: Sequence[int]) -> Tuple[int, int, int]:
        """Fortran 1-based (i,j,k) -> padded index triple."""
        return tuple(int(ki) + g - 1 for ki in k)

    # domain-boundary treatment (upstream order: x, then y, then z)
    if domain_walls[0]:
        W[fp((1, 1, 1))[0], :, :] = 1
    if domain_walls[1]:
        W[fp((nx, 1, 1))[0], :, :] = 1
    if domain_walls[2]:
        W[fp((2, 1, 1))[0]:fp((nx - 1, 1, 1))[0] + 1, fp((1, 1, 1))[1], :] = 1
    if domain_walls[3]:
        W[fp((2, 1, 1))[0]:fp((nx - 1, 1, 1))[0] + 1, fp((1, ny, 1))[1], :] = 1
    if domain_walls[4]:
        W[fp((2, 2, 1))[0]:fp((nx - 1, 2, 1))[0] + 1,
          fp((2, 2, 1))[1]:fp((2, ny - 1, 1))[1] + 1, fp((1, 1, 1))[2]] = 1
    if domain_walls[5]:
        W[fp((2, 2, 1))[0]:fp((nx - 1, 2, 1))[0] + 1,
          fp((2, 2, 1))[1]:fp((2, ny - 1, 1))[1] + 1, fp((1, 1, nz))[2]] = 1

    # ghost fill: z (over physical i,j), then y (all k), then x (all j,k).
    # The extents matter: upstream never fills the ghost corners in the z sweep,
    # it leaves them to the later y/x sweeps, so a full-slice z fill would
    # propagate the wrong (still-zero) corner values.
    other = {
        2: (slice(g, g + nx), slice(g, g + ny)),
        1: (slice(g, g + nx), slice(None)),
        0: (slice(None), slice(None)),
    }
    for axis in (2, 1, 0):
        ni = n[axis]
        spans = list(other[axis])
        if periodic[axis]:
            for c in range(g):
                _assign_plane(W, axis, fp([1 - g + c if a == axis else 1 for a in range(3)]),
                              fp([ni - g + 1 + c if a == axis else 1 for a in range(3)]), spans)
                _assign_plane(W, axis, fp([ni + 1 + c if a == axis else 1 for a in range(3)]),
                              fp([1 + c if a == axis else 1 for a in range(3)]), spans)
        else:
            for c in range(g):
                _assign_plane(W, axis, fp([1 - g + c if a == axis else 1 for a in range(3)]),
                              fp([1 if a == axis else 1 for a in range(3)]), spans)
                _assign_plane(W, axis, fp([ni + 1 + c if a == axis else 1 for a in range(3)]),
                              fp([ni if a == axis else 1 for a in range(3)]), spans)

    if mirror_bc == 1:
        # walls(i, 1-ghost:0, k) = walls(i, 2:ghost+1, k)
        W[:, 0:g, :] = W[:, g + 1:g + 1 + g, :]
    return W


def _assign_plane(W: np.ndarray, axis: int, dst: Sequence[int], src: Sequence[int],
                  spans: Sequence[slice]) -> None:
    """``W[dst plane] = W[src plane]`` along ``axis`` over the given other-axis spans."""
    d = list(spans)
    s = list(spans)
    d.insert(axis, dst[axis])
    s.insert(axis, src[axis])
    W[tuple(d)] = W[tuple(s)]


def smooth_solid_fraction(indicator: np.ndarray, n_iter: int = DEFAULT_SMOOTH_ITERATIONS,
                          ghost: int = DEFAULT_GHOST_LAYERS) -> np.ndarray:
    """``n_iter`` D3Q27 convolutions of the (padded) solid indicator.

    The kernel ``{8/27, 2/27, 1/54, 1/216}`` sums to one, so the smoothing is
    mass conserving and does not move the 0.5 level set of a flat wall; only
    the outer ghost layer is left untouched (as upstream).
    """
    F = np.asarray(indicator, dtype=np.float64).copy()
    sl = _interior_slice(F.shape[0], ghost)
    for _ in range(n_iter):
        F_new = F.copy()
        F_new[sl] = _convolve_d3q27(F)[sl]
        F = F_new
    return F


def classify_boundary_nodes(markers: np.ndarray, overlap: int = DEFAULT_OVERLAP,
                            ghost: int = DEFAULT_GHOST_LAYERS) -> Dict[str, np.ndarray]:
    """Solid/fluid boundary-node masks, in the OSS listing region.

    ``markers`` is the padded integer wall array (1 = solid, 0 = fluid) *after*
    the ghost fill.  Returns boolean masks with the same padded shape; only the
    Fortran index window ``1-overlap .. n+overlap`` can be True.
    """
    solid = markers == 1
    fluid = markers == 0
    touch_fluid = np.zeros_like(solid)
    touch_solid = np.zeros_like(solid)
    for off in HALF18:
        nb = _shifted(solid.astype(np.int8), off).astype(bool)
        touch_solid |= nb           # a neighbour is solid
    for off in HALF18:
        nb = _shifted(fluid.astype(np.int8), off).astype(bool)
        touch_fluid |= nb            # a neighbour is fluid
    solid_b = solid & touch_fluid
    fluid_b = fluid & touch_solid
    keep = np.zeros_like(solid_b)
    sl = tuple(slice(ghost - overlap, m - ghost + overlap) for m in solid_b.shape)
    keep[sl] = True
    return {"solid_boundary": solid_b & keep, "fluid_boundary": fluid_b & keep}


def iso8_gradient(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isotropic ISO8 gradient of ``field`` (full array; interior values valid).

    One pair set per component, exactly as the upstream expression writes it.
    The operator is not normalised to ``d/dx``: its gain on an analytic linear
    field is the constant :func:`iso8_stencil_scale` (= 0.880952...) in *every*
    direction (measured anisotropy 1.0 to machine precision).  MF-LBM only ever
    uses the direction of the gradient, and so does :func:`wall_normal_table`.
    """
    F = np.asarray(field, dtype=np.float64)
    g = []
    for axis in (0, 1, 2):
        comp = np.zeros_like(F)
        for w, h in ISO8_PAIRS[axis]:
            # upstream writes exactly this: sum_p w_p [f(p+h_p) - f(p-h_p)],
            # with the signed pair offsets h_p; the distance/axis factor is
            # already baked into the fitted weights (no extra h_axis factor!)
            comp += w * (_shifted(F, h) - _shifted(F, tuple(-c for c in h)))
        g.append(comp)
    return g[0], g[1], g[2]


def iso8_stencil_scale() -> float:
    """``2 * sum_pairs w * h_axis^2``: the constant |grad| gain of the stencil."""
    return 2.0 * sum(w * h[0] for w, h in ISO8_PAIRS[0])


def iso8_divergence(fx: np.ndarray, fy: np.ndarray, fz: np.ndarray) -> np.ndarray:
    """Isotropic divergence ``d_x f_x + d_y f_y + d_z f_z`` with the ISO8 stencil."""
    out = np.zeros(np.shape(fx), dtype=np.float64)
    for comp, comp_axis in ((fx, 0), (fy, 1), (fz, 2)):
        g = iso8_gradient(comp)[comp_axis]
        out += g
    return out


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def wall_normal_table(walls: np.ndarray, *, periodic=(0, 0, 0),
                      domain_walls=(0, 0, 0, 0, 0, 0), mirror_bc: int = 0,
                      n_smooth: int | None = None,
                      ghost: int = DEFAULT_GHOST_LAYERS,
                      overlap: int = DEFAULT_OVERLAP) -> Dict[str, np.ndarray]:
    """Build the wall-normal table for a voxelised solid mask.

    Args:
        walls: ``(nx, ny, nz)`` integer array, 1 = solid, 0 = fluid, in the
            *Fortran 1-based* index convention that MF-LBM's preprocessor uses
            (the array index ``[a,b,c]`` corresponds to Fortran ``(a+1,b+1,c+1)``).
        periodic: per-axis periodic flags (upstream ``iper/jper/kper``).
        domain_walls: ``(xmin,xmax,ymin,ymax,zmin,zmax)`` flags, 1 = the domain
            face itself is a wall (upstream ``domain_wall_status_*``).
        mirror_bc: upstream ``mirror_bc`` (0 = off; upstream leaves it unset).
        n_smooth: smoothing iterations (default = upstream ``i_max_iteration``).
        ghost: ghost layers for the smoothing (upstream 6).
        overlap: node-listing overlap (upstream 4).

    Returns:
        dict with

        ``indices``      (N, 3) int   Fortran (1-based) indices of fluid boundary nodes
        ``normals``      (N, 3) float unit normals, pointing INTO the solid
        ``grad_mag``     (N,)   float |grad| before normalisation
        ``solid_indices``(M, 3) int   Fortran indices of solid boundary nodes
        ``solid_la_weight`` (M,) float upstream ``la_weight``
        ``solid_n_fluid``   (M,) int   upstream ``i_fluid_num``
        ``solid_neighbor_list`` (M,18) int upstream ``neighbor_list`` (0 = unused)
        ``n_fluid_boundary``/``n_solid_boundary`` ints (informational duplicates)
    """
    walls = np.asarray(walls)
    nx, ny, nz = walls.shape
    n_smooth = DEFAULT_SMOOTH_ITERATIONS if n_smooth is None else int(n_smooth)
    # upstream pads by `ghost_layers = ghost_layers + overlap_phi` (6 + 4 = 10);
    # the smoothing then runs over the whole padded array except its outer
    # layer, so the padding width is part of the algorithm, not an option.
    pad = ghost + overlap
    markers = _pad_solid(walls, pad, periodic, domain_walls, mirror_bc)
    masks = classify_boundary_nodes(markers, overlap=overlap, ghost=pad)
    indicator = smooth_solid_fraction(markers.astype(np.float64), n_smooth, pad)
    gx, gy, gz = iso8_gradient(indicator)

    # Fortran ordering (i fastest) so the tables can be compared elementwise
    def _order(mask: np.ndarray) -> np.ndarray:
        """Node list in the upstream listing order (Fortran: i fastest)."""
        p = np.argwhere(mask)
        flat = p[:, 0] + mask.shape[0] * (p[:, 1] + mask.shape[1] * p[:, 2])
        return p[np.argsort(flat, kind="stable")]

    fb = _order(masks["fluid_boundary"])
    sb = _order(masks["solid_boundary"])
    fidx = fb - (pad - 1)  # padded index -> Fortran 1-based index
    sidx = sb - (pad - 1)

    g = np.stack([gx[tuple(fb.T)], gy[tuple(fb.T)], gz[tuple(fb.T)]], axis=1) if len(fb) else np.zeros((0, 3))
    mag = np.linalg.norm(g, axis=1)
    normals = np.divide(g, mag[:, None], out=np.zeros_like(g), where=mag[:, None] > 0.0)

    # upstream solid-node extras: neighbor_list over HALF18, la_weight, i_fluid_num
    m = len(sb)
    nb_list = np.zeros((m, 18), dtype=np.int64)
    la = np.zeros(m, dtype=np.float64)
    n_fluid = np.zeros(m, dtype=np.int64)
    if m:
        W = markers.astype(np.int64)
        for j, off in enumerate(HALF18):
            vals = _shifted(W, off)[tuple(sb.T)]        # neighbour marker value
            hit = vals <= 0
            nb_list[hit, n_fluid[hit]] = j + 1          # 1-based direction index
            n_fluid += hit
            la += np.where(hit, W_EQU_HALF18[j + 1], 0.0)

    return {
        "indices": fidx.astype(np.int64),
        "normals": normals,
        "grad_mag": mag,
        "solid_indices": sidx.astype(np.int64),
        "solid_la_weight": la,
        "solid_n_fluid": n_fluid,
        "solid_neighbor_list": nb_list,
        "n_fluid_boundary": int(len(fidx)),
        "n_solid_boundary": int(len(sidx)),
    }


def in_physical_domain(indices: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    """Mask selecting table rows whose Fortran index lies inside ``1..n``."""
    idx = np.asarray(indices)
    ok = np.ones(len(idx), dtype=bool)
    for ax, n in enumerate(shape):
        ok &= (idx[:, ax] >= 1) & (idx[:, ax] <= n)
    return ok