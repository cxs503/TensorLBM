"""Bitflag cell classification + **per-direction neighbour mask** (D2Q9).

Cross-validation target (XVAL-LBM-BITFLAG-NEIGHBOUR-MASK)
--------------------------------------------------------
Reference implementation: ``Maarten-vd-Sande/lbm`` (Python/NumPy LBM with a
free surface), cached read-only at ``/root/oss-cache/Maarten-vd-Sande_lbm``
(commit ``573a01c``).  Everything described as "OSS" below is cited to
file/line of that checkout -- no value is retyped from prose:

* cell classification by **bitflags** -- ``lbm/util.py:31-48``
  (``FLUID=1, INTERFACE=2, GAS=4, OBSTACLE=8, INLET=16, OUTLET=32`` plus the
  aggregate neighbourhood flags ``NO_FLUID_NEIGH=64, NO_EMPTY_NEIGH=128,
  NO_IFACE_NEIGH=256``);
* those aggregate neighbourhood flags are rebuilt every step by a sweep over
  the 8 neighbour directions and cleared again at the start of the next sweep
  -- ``lbm/update_types.py:118-132`` and ``lbm/update_types.py:29-30``;
* streaming is a **pull** sweep that, for every open cell and every one of the
  8 directions, tests the *neighbour's* ``cell_type`` against the open set
  ``FLUID | INTERFACE | INLET | OUTLET`` -- ``lbm/streamcollide.py:56`` and
  ``lbm/streamcollide.py:63``; a closed neighbour triggers half-way bounce-back
  ``fin[n] = fout[v_inv[n]]`` -- ``lbm/streamcollide.py:88-90``;
* inlet closure ``set_inlet`` (Zou-He density from the outgoing populations,
  prescribed velocity) -- ``lbm/streamcollide.py:218-233``; outlet closure
  ``set_outlet`` (zero gradient, copies the upstream node) --
  ``lbm/streamcollide.py:210-217``;
* the domain-border guard of the streaming loop is
  ``if 0 < ix_next >= fin.shape[1]`` -- ``lbm/streamcollide.py:76-78`` (see
  :func:`oss_border_guard_wraps`, a measured caveat of the reference).

What this module adds
---------------------
The OSS neighbourhood information is *aggregate* (one bit per phase per node)
and is consumed only by the free-surface mass bookkeeping -- the streaming loop
still performs 8 neighbour-type tests per open node.  Here the same idea is
taken to per-direction resolution: a single ``uint8`` **neighbour mask** per
node, bit ``i-1`` set iff the link ``x -> x - c_i`` is closed, i.e. iff the OSS
bounce-back branch would be taken.  That mask

* is geometry-only, so it is built once per geometry change instead of being
  rebuilt every step (OSS ``update_types`` rebuilds its flags every step);
* lets a scalar kernel run branch-free straight-line code for interior nodes
  (``mask == 0``) and touch only the set bits for boundary nodes, instead of
  testing 8 neighbour types for every node.

Semantics are unchanged: both kernels are proven bit-identical to an
OSS-faithful scan kernel by ``tests/test_bitflag_neighbour_mask.py``.

Layout is ``(9, nx, ny)`` with the OSS velocity ordering (``lbm/util.py:6-26``)
so the transcription can be read side by side with the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np

__all__ = [
    "CellFlag",
    "OPEN_BITS",
    "D2Q9_C",
    "D2Q9_W",
    "D2Q9_OPP",
    "MaskStats",
    "StreamStats",
    "open_cells",
    "shifted",
    "neighbour_mask",
    "mask_link_count",
    "mask_stats",
    "stream_scan_vector",
    "stream_mask_vector",
    "stream_scan_scalar",
    "stream_mask_scalar",
    "oss_border_guard_wraps",
]


# ---------------------------------------------------------------------------
# cell classification (bitflags)
# ---------------------------------------------------------------------------
class CellFlag(IntFlag):
    """Node flags with the same bit values as the OSS reference (``lbm/util.py:31-48``).

    Only the *cell type* bits are needed by a static-geometry solver; the OSS
    aggregate neighbourhood bits (``NO_*_NEIGH``) are not reproduced because the
    per-direction mask below carries strictly more information.
    """

    FLUID = 1
    INTERFACE = 2
    GAS = 4
    OBSTACLE = 8
    INLET = 16
    OUTLET = 32


#: Cells a population may stream into.  Identical to the OSS test set
#: ``CT.FLUID | CT.INTERFACE | CT.INLET | CT.OUTLET`` (``lbm/streamcollide.py:56``).
OPEN_BITS = int(CellFlag.FLUID | CellFlag.INTERFACE | CellFlag.INLET | CellFlag.OUTLET)

#: D2Q9 velocity set, OSS ordering (``lbm/util.py:6-14``).
D2Q9_C = np.array(
    [[0, 0], [0, -1], [0, 1], [-1, 0], [-1, -1], [-1, 1], [1, 0], [1, -1], [1, 1]],
    dtype=np.int64,
)
#: D2Q9 weights, OSS ordering (``lbm/util.py:17-25``).
D2Q9_W = np.array(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 9, 1 / 36, 1 / 36],
    dtype=np.float64,
)


def _opposites(c: np.ndarray) -> np.ndarray:
    index = {tuple(int(v) for v in row): i for i, row in enumerate(c.tolist())}
    return np.array(
        [index[tuple(-int(v) for v in row)] for row in c.tolist()], dtype=np.int64
    )


D2Q9_OPP = _opposites(D2Q9_C)


# ---------------------------------------------------------------------------
# array helpers
# ---------------------------------------------------------------------------
def shifted(a: np.ndarray, d0: int, d1: int, fill=0):
    """``out[i0, i1] = a[i0 + d0, i1 + d1]`` with ``fill`` outside the array.

    The offsets follow the **array** axis order, not ``(y, x)``: the streaming
    arrays of this module are laid out ``(9, n0, n1)`` with ``n0 = x`` and
    ``n1 = y`` (the OSS layout, ``lbm/init.py:10-19``), so a pull along lattice
    direction ``c_i = (c_x, c_y)`` shifts by ``(-c_x, -c_y)``.
    

    Explicit padding, *not* ``np.roll``: the OSS guard at
    ``lbm/streamcollide.py:76-78`` lets a negative index alias the opposite
    boundary through Python wrap-around indexing (see
    :func:`oss_border_guard_wraps`); here out-of-domain links are closed, so no
    population crosses a boundary it did not physically cross.
    """
    n0, n1 = a.shape
    out = np.full(a.shape, fill, dtype=a.dtype)
    a0, b0 = max(0, -d0), min(n0, n0 - d0)
    a1, b1 = max(0, -d1), min(n1, n1 - d1)
    if b0 > a0 and b1 > a1:
        out[a0:b0, a1:b1] = a[a0 + d0 : b0 + d0, a1 + d1 : b1 + d1]
    return out


def open_cells(cell_type: np.ndarray) -> np.ndarray:
    """Boolean map of cells a population may occupy (OSS open-bit test)."""
    return (np.asarray(cell_type).astype(np.int64) & OPEN_BITS) != 0


def neighbour_mask(cell_type: np.ndarray) -> np.ndarray:
    """Per-direction closed-link mask, ``uint8``, bit ``i-1`` = direction ``i``.

    Bit ``i-1`` of ``mask[x, y]`` is set iff

    * ``x - c_i`` lies outside the domain, **or**
    * the cell at ``x - c_i`` is not in :data:`OPEN_BITS`.

    Under the OSS pull convention that condition is exactly the branch that
    selects half-way bounce-back instead of streaming
    (``lbm/streamcollide.py:63-90``), so ``mask`` is a complete, geometry-only
    summary of the boundary treatment.
    """
    ct = np.asarray(cell_type)
    if ct.ndim != 2:
        raise ValueError(f"cell_type must be 2-D, got shape {ct.shape}")
    op = open_cells(ct)
    mask = np.zeros(ct.shape, dtype=np.uint8)
    for i in range(1, 9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        # the pulled-from neighbour of x in direction i is x - c_i
        nb_open = shifted(op, -cx, -cy, fill=False)
        mask |= np.where(nb_open, np.uint8(0), np.uint8(1 << (i - 1))).astype(np.uint8)
    return mask


_POPCOUNT = np.array([bin(v).count("1") for v in range(256)], dtype=np.int8)


def mask_link_count(mask: np.ndarray) -> int:
    """Total number of closed links = sum of set bits over all nodes."""
    return int(_POPCOUNT[np.asarray(mask)].sum())


@dataclass(frozen=True)
class MaskStats:
    nodes: int
    open_nodes: int
    interior_nodes: int  # mask == 0: no closed link -> branch-free kernel path
    boundary_nodes: int
    closed_links: int
    open_links: int

    def as_dict(self) -> dict[str, float | int]:
        total = self.closed_links + self.open_links
        return {
            "nodes": self.nodes,
            "open_nodes": self.open_nodes,
            "interior_nodes": self.interior_nodes,
            "boundary_nodes": self.boundary_nodes,
            "closed_links": self.closed_links,
            "open_links": self.open_links,
            "closed_link_fraction": (self.closed_links / total) if total else 0.0,
        }


def mask_stats(cell_type: np.ndarray, mask: np.ndarray | None = None) -> MaskStats:
    ct = np.asarray(cell_type)
    m = neighbour_mask(ct) if mask is None else np.asarray(mask)
    op = open_cells(ct)
    closed = mask_link_count(m)
    return MaskStats(
        nodes=int(ct.size),
        open_nodes=int(op.sum()),
        interior_nodes=int(((m == 0) & op).sum()),
        boundary_nodes=int(((m != 0) & op).sum()),
        closed_links=closed,
        open_links=int(8 * op.sum()) - closed,
    )


@dataclass
class StreamStats:
    """Work accounting shared by all four kernels.

    ``neighbour_type_tests``
        neighbour ``cell_type`` lookups + branch decisions.  This is the inner
        loop of the OSS sweep (``lbm/streamcollide.py:63``): 8 per open node.
    ``mask_word_tests``
        neighbour-mask word tests: 1 per node in the mask kernels (an interior
        node then needs no further decision at all).
    ``bit_tests``
        per-link bit tests of the set bits, i.e. one per closed link.
    ``streams`` / ``bounces``
        populations pulled / half-way bounce-back links executed (identical in
        both kernels -- that is the equivalence claim).

    The hardware-independent cost model is therefore
    ``scan = 8 * open_nodes`` vs ``mask = open_nodes + closed_links``.
    """

    neighbour_type_tests: int = 0
    mask_word_tests: int = 0
    bit_tests: int = 0
    bounces: int = 0
    streams: int = 0

    @property
    def decision_count(self) -> int:
        return self.neighbour_type_tests + self.mask_word_tests + self.bit_tests

    def as_dict(self) -> dict[str, int]:
        return {
            "neighbour_type_tests": self.neighbour_type_tests,
            "mask_word_tests": self.mask_word_tests,
            "bit_tests": self.bit_tests,
            "decision_count": self.decision_count,
            "bounces": self.bounces,
            "streams": self.streams,
        }


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
def _bb_value(
    f_post: np.ndarray,
    i: int,
    x: int,
    y: int,
    wall_velocity: np.ndarray | None,
    rho_wall: np.ndarray | None,
) -> float:
    """Half-way bounce-back population: mirror plus optional moving-wall term.

    Static wall: ``f_i <- f_opp(i)`` at the same node (OSS
    ``lbm/streamcollide.py:88-90``).  A moving wall with velocity ``u_w`` adds the
    standard second-order momentum term ``6 w_i rho_w (c_i . u_w)``.
    """
    val = f_post[D2Q9_OPP[i], x, y]
    if wall_velocity is not None:
        uc = wall_velocity[i]
        if uc != 0.0:
            rw = 1.0 if rho_wall is None else rho_wall[x, y]
            val = val + 6.0 * D2Q9_W[i] * rw * uc
    return val


def stream_scan_scalar(
    f_post: np.ndarray,
    cell_type: np.ndarray,
    *,
    stats: StreamStats | None = None,
    wall_velocity: np.ndarray | None = None,
    rho_wall: np.ndarray | None = None,
) -> np.ndarray:
    """OSS-faithful pull sweep: 8 neighbour-type tests per open node.

    Transcribes ``lbm/streamcollide.py:44-92`` for the open/obstacle subset
    (the free-surface INTERFACE/GAS bookkeeping is deliberately not ported; see
    the module docstring).  Out-of-domain links are closed, unlike the OSS guard.
    """
    f_post = np.asarray(f_post)
    ct = np.asarray(cell_type)
    nvec, nx, ny = f_post.shape
    if nvec != 9:
        raise ValueError("scalar kernels expect the D2Q9 layout (9, nx, ny)")
    op = open_cells(ct)
    out = np.empty_like(f_post)
    for x in range(nx):
        for y in range(ny):
            if not op[x, y]:
                out[:, x, y] = f_post[:, x, y]
                continue
            out[0, x, y] = f_post[0, x, y]
            for i in range(1, 9):
                cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
                nx_i, ny_i = x - cx, y - cy
                if stats is not None:
                    stats.neighbour_type_tests += 1
                if 0 <= nx_i < nx and 0 <= ny_i < ny and op[nx_i, ny_i]:
                    out[i, x, y] = f_post[i, nx_i, ny_i]
                    if stats is not None:
                        stats.streams += 1
                else:
                    out[i, x, y] = _bb_value(f_post, i, x, y, wall_velocity, rho_wall)
                    if stats is not None:
                        stats.bounces += 1
    return out


def stream_mask_scalar(
    f_post: np.ndarray,
    cell_type: np.ndarray,
    mask: np.ndarray,
    *,
    stats: StreamStats | None = None,
    wall_velocity: np.ndarray | None = None,
    rho_wall: np.ndarray | None = None,
) -> np.ndarray:
    """Mask-driven pull sweep: one mask lookup per node, closed bits only.

    Interior nodes (``mask == 0``) take a branch-free straight-line path, so the
    per-node decision cost is one 8-bit test instead of 8 neighbour-type tests;
    boundary nodes iterate the set bits and bounce exactly those links.
    """
    f_post = np.asarray(f_post)
    ct = np.asarray(cell_type)
    m = np.asarray(mask)
    nvec, nx, ny = f_post.shape
    if nvec != 9:
        raise ValueError("scalar kernels expect the D2Q9 layout (9, nx, ny)")
    op = open_cells(ct)
    out = np.empty_like(f_post)
    for x in range(nx):
        for y in range(ny):
            if not op[x, y]:
                out[:, x, y] = f_post[:, x, y]
                continue
            mi = int(m[x, y])
            if stats is not None:
                stats.mask_word_tests += 1
            out[0, x, y] = f_post[0, x, y]
            if mi == 0:
                # interior node: branch-free pull, no per-direction decision
                for i in range(1, 9):
                    cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
                    out[i, x, y] = f_post[i, x - cx, y - cy]
                    if stats is not None:
                        stats.streams += 1
                continue
            for i in range(1, 9):
                if stats is not None:
                    stats.bit_tests += 1
                if mi & (1 << (i - 1)):
                    out[i, x, y] = _bb_value(f_post, i, x, y, wall_velocity, rho_wall)
                    if stats is not None:
                        stats.bounces += 1
                else:
                    cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
                    out[i, x, y] = f_post[i, x - cx, y - cy]
                    if stats is not None:
                        stats.streams += 1
    return out


# ---------------------------------------------------------------------------
# vectorised kernels (production path)
# ---------------------------------------------------------------------------
def stream_scan_vector(
    f_post: np.ndarray,
    cell_type: np.ndarray,
    *,
    wall_velocity: np.ndarray | None = None,
    rho_wall: np.ndarray | None = None,
    stats: StreamStats | None = None,
) -> np.ndarray:
    """Vectorised OSS-style sweep: per-direction neighbour-type test every step."""
    f_post = np.asarray(f_post)
    ct = np.asarray(cell_type)
    op = open_cells(ct)
    out = np.empty_like(f_post)
    out[0] = f_post[0]
    if stats is not None:
        stats.neighbour_type_tests += 8 * int(op.sum())
    for i in range(1, 9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        nb_open = shifted(op, -cx, -cy, fill=False)  # x - c_i open?
        pulled = shifted(f_post[i], -cx, -cy, fill=0.0)
        bb = f_post[D2Q9_OPP[i]]
        if wall_velocity is not None:
            uc = wall_velocity[i]
            rw = 1.0 if rho_wall is None else rho_wall
            bb = bb + 6.0 * D2Q9_W[i] * rw * uc
        out[i] = np.where(nb_open, pulled, bb)
        if stats is not None:
            stats.bounces += int((~nb_open & op).sum())
            stats.streams += int((nb_open & op).sum())
    return out


def stream_mask_vector(
    f_post: np.ndarray,
    cell_type: np.ndarray,
    mask: np.ndarray,
    *,
    wall_velocity: np.ndarray | None = None,
    rho_wall: np.ndarray | None = None,
    stats: StreamStats | None = None,
) -> np.ndarray:
    """Vectorised mask-driven sweep: the closed-link mask replaces the type test."""
    f_post = np.asarray(f_post)
    ct = np.asarray(cell_type)
    m = np.asarray(mask)
    op = open_cells(ct)
    out = np.empty_like(f_post)
    out[0] = f_post[0]
    if stats is not None:
        stats.mask_word_tests += int(op.sum())
    for i in range(1, 9):
        cx, cy = int(D2Q9_C[i, 0]), int(D2Q9_C[i, 1])
        closed = (m & np.uint8(1 << (i - 1))) != 0
        inner = int(closed.sum())
        if stats is not None:
            # only closed links need a decision; open links need none at all
            stats.bit_tests += int((closed & op).sum())
            stats.bounces += int((closed & op).sum())
            stats.streams += int((~closed & op).sum())
        if inner == 0:
            out[i] = shifted(f_post[i], -cx, -cy, fill=0.0)
            continue
        pulled = shifted(f_post[i], -cx, -cy, fill=0.0)
        bb = f_post[D2Q9_OPP[i]]
        if wall_velocity is not None:
            uc = wall_velocity[i]
            rw = 1.0 if rho_wall is None else rho_wall
            bb = bb + 6.0 * D2Q9_W[i] * rw * uc
        out[i] = np.where(closed, bb, pulled)
    return out


# ---------------------------------------------------------------------------
# reference-implementation caveat
# ---------------------------------------------------------------------------
def oss_border_guard_wraps(ix_next: int, iy_next: int, shape: tuple[int, int]) -> bool:
    """Reproduce the OSS streaming border guard (``lbm/streamcollide.py:76-78``).

    The guard is ``0 < ix_next >= fin.shape[1]``, a chained comparison that is
    true only for ``ix_next >= nx``.  A *negative* ``ix_next`` therefore does not
    ``continue``; Python's negative indexing then aliases the opposite boundary,
    i.e. a population can be pulled across the whole domain.  Reproduced here so
    that the port can be checked against the reference behaviour instead of
    silently inheriting it (see ``tests/test_bitflag_neighbour_mask.py``).
    """
    nx, ny = shape
    return (0 < ix_next >= nx) or (0 < iy_next >= ny)