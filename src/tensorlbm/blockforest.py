"""Block-structured forest, METIS-style block-graph load balancing and a
multistream (compute / communication overlap) LBM stepper.

Why this module exists
----------------------
TensorLBM's pre-existing distributed capability is a **z-slab** decomposition
(:mod:`tensorlbm.domain_decomposition`) with a **blocking** halo exchange: the
communicator is built from ``(n_devices, nz, ny, nx)`` slabs, the neighbour rank
is ``rank +/- 1`` and every step is ``fill ghosts -> stream``.  Nothing in the
tree builds a block graph, balances *workload* (as opposed to block count) or
hides communication latency.

The OSS reference this module is cross-validated against is waLBerla
(``lssfau/walberla``), whose "blockforest + METIS + multistream overlap" stack
does three things:

1. ``src/blockforest/BlockForest.h`` - the domain is a forest of uniform bricks
   ("blocks"); neighbourhoods are established between blocks sharing a face, an
   edge or a corner (26-neighbourhood in 3D), and a 1-irregular refinement
   constraint is kept.
2. ``src/blockforest/loadbalancing/MetisPartitioning.cpp`` - load balancing is
   *reduced to graph partitioning*: one graph vertex per block, vertex weight =
   block workload, edge weight = size of the shared interface; the graph is
   handed to ``METIS_PartGraphKway``.
3. The CUDA kernels use a multistream scheme to overlap the MPI exchange with
   local work: pack the *post-collision* boundary planes, start the
   asynchronous transfer on a separate stream, update the interior (which does
   not depend on remote data), and only afterwards unpack the ghosts and update
   the boundary layer.

Sourcing note: this work tree has no network access, so the waLBerla sources
could not be fetched here.  The three practices above are cited at
module/function granularity and treated as *documented* practice; the numeric
cross-validation therefore uses references that are reproducible offline (exact
enumeration for the optimum, plus ``networkx``'s Kernighan-Lin bisection as an
independent OSS reference).  See
``docs/XVAL_BLOCKFOREST_METIS_MULTISTREAM.md`` section 5.

Everything below is pure PyTorch / NumPy and single-process: ghost planes are
copied between in-memory blocks by a worker thread, which is the same
communication pattern an MPI build would issue as ``MPI_Isend``/``MPI_Irecv``,
with an optional injected ``latency`` to emulate the network.
"""

from __future__ import annotations

import itertools
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from .d3q19 import C, equilibrium3d, macroscopic3d
from .solver3d import collide_bgk3d, stream3d_roll

__all__ = [
    "BlockForest",
    "BlockGraph",
    "PartitionResult",
    "MultistreamBlockSim",
    "StepTiming",
    "partition_blockforest",
    "slab_partition",
    "weighted_slab_partition",
    "exact_bisection_cut",
    "monolithic_shear_wave_step",
    "NFACE_OFFSETS",
    "N26_OFFSETS",
]

# (dz, dy, dx) - z is the slow axis, matching the (19, nz, ny, nx) layout.
NFACE_OFFSETS: tuple[tuple[int, int, int], ...] = (
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
)

N26_OFFSETS: tuple[tuple[int, int, int], ...] = tuple(
    d for d in itertools.product((-1, 0, 1), repeat=3) if d != (0, 0, 0)
)


# ---------------------------------------------------------------------------
# 1. Block forest
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BlockForest:
    """A forest of uniform bricks (waLBerla ``BlockForest`` counterpart).

    Blocks are indexed ``(i, j, k)`` = (z, y, x) block coordinates and the
    linear index is ``(i * by + j) * bx + k`` - i.e. **contiguous linear ranges
    are z-slabs**, which is exactly the decomposition implemented by
    :class:`tensorlbm.domain_decomposition.DomainDecomposer`.  This makes the
    slab baseline a genuine baseline of the current repository capability
    rather than a straw man.

    ``refinement[idx]`` is the static refinement level of a block (0 = base
    cells); a level-``L`` block holds ``block_cells * 2**L`` cells per axis.
    Neighbouring blocks may differ by at most one level (1-irregular forest,
    as required by the reference implementation).
    """

    nblocks: tuple[int, int, int] = (4, 4, 4)
    block_cells: tuple[int, int, int] = (16, 16, 16)
    periodic: tuple[bool, bool, bool] = (True, True, True)
    refinement: tuple[int, ...] | None = None

    # -- geometry ----------------------------------------------------------
    @property
    def n_blocks(self) -> int:
        bz, by, bx = self.nblocks
        return bz * by * bx

    def index(self, i: int, j: int, k: int) -> int:
        _, by, bx = self.nblocks
        return (i * by + j) * bx + k

    def coords(self, idx: int) -> tuple[int, int, int]:
        _, by, bx = self.nblocks
        i, rem = divmod(idx, by * bx)
        j, k = divmod(rem, bx)
        return i, j, k

    def level(self, idx: int) -> int:
        if self.refinement is None:
            return 0
        return int(self.refinement[idx])

    def shape_at(self, idx: int) -> tuple[int, int, int]:
        f = 2 ** self.level(idx)
        nz, ny, nx = self.block_cells
        return nz * f, ny * f, nx * f

    def workload_at(self, idx: int) -> float:
        """Lattice cells owned by the block (``METIS`` vertex weight)."""
        nz, ny, nx = self.shape_at(idx)
        return float(nz * ny * nx)

    def global_shape(self) -> tuple[int, int, int]:
        bz, by, bx = self.nblocks
        nz, ny, nx = self.block_cells
        return bz * nz, by * ny, bx * nx

    def _neighbor(self, idx: int, d: tuple[int, int, int]) -> int | None:
        i, j, k = self.coords(idx)
        bz, by, bx = self.nblocks
        nco = [i + d[0], j + d[1], k + d[2]]
        for a, (comp, bi) in enumerate(zip(d, (bz, by, bx))):
            if comp == 0:
                continue
            if self.periodic[a]:
                nco[a] %= bi
            elif not (0 <= nco[a] < bi):
                return None
        return self.index(*nco)

    def face_neighbors(self, idx: int) -> list[tuple[int, int, float]]:
        """Face neighbours of ``idx``: ``(neighbor, interface_area, axis)``."""
        ext = self.shape_at(idx)
        out: list[tuple[int, int, float]] = []
        for axis in (0, 1, 2):
            d = [0, 0, 0]
            for sgn in (-1, 1):
                d[axis] = sgn
                nb = self._neighbor(idx, tuple(d))
                if nb is None or nb == idx:
                    continue
                nb_ext = self.shape_at(nb)
                area = 1.0
                for a2 in range(3):
                    if a2 == axis:
                        continue
                    # interface measured on the *finer* side: transferred cells
                    area *= max(ext[a2], nb_ext[a2])
                out.append((nb, float(area), axis))
        return out

    def graph(self, stencil: int = 6) -> "BlockGraph":
        """Build the block adjacency graph used for load balancing.

        ``stencil=6``  graph edges = face neighbours, edge weight = shared
                       interface area (the METIS graph of the reference).
        ``stencil=26`` adds edge/corner neighbours with their much smaller
                       interface measure - useful when the ghost footprint
                       rather than the plane volume is the cost driver.
        """
        if stencil not in (6, 26):
            raise ValueError("stencil must be 6 or 26")
        n = self.n_blocks
        pair_w: dict[tuple[int, int], float] = {}
        offs = NFACE_OFFSETS if stencil == 6 else N26_OFFSETS
        for idx in range(n):
            ext = self.shape_at(idx)
            for d in offs:
                nb = self._neighbor(idx, d)
                if nb is None or nb == idx:
                    continue
                nb_ext = self.shape_at(nb)
                w = 1.0
                for a in range(3):
                    if d[a] != 0:
                        continue
                    w *= max(ext[a], nb_ext[a])  # interface size on the finer side
                key = (min(idx, nb), max(idx, nb))
                pair_w[key] = pair_w.get(key, 0.0) + w

        # each face is accumulated once from each side (and once per periodic
        # image); halve so the edge weight is the geometric interface area.
        nbrs: list[list[tuple[int, float]]] = [[] for _ in range(n)]
        for (u, v), w in sorted(pair_w.items()):
            w *= 0.5
            nbrs[u].append((v, w))
            nbrs[v].append((u, w))

        xadj = [0]
        adjncy: list[int] = []
        adjwgt: list[float] = []
        for lst in nbrs:
            lst.sort()
            for v, w in lst:
                adjncy.append(v)
                adjwgt.append(w)
            xadj.append(len(adjncy))

        vwgt = [self.workload_at(idx) for idx in range(n)]
        coords = np.array([self.coords(idx) for idx in range(n)], dtype=float)
        return BlockGraph(
            n=n,
            xadj=xadj,
            adjncy=adjncy,
            adjwgt=adjwgt,
            vwgt=vwgt,
            coords=coords,
            nblocks=self.nblocks,
            block_cells=self.block_cells,
            refinement=(
                tuple(self.refinement) if self.refinement is not None else tuple([0] * n)
            ),
        )


@dataclass
class BlockGraph:
    """CSR block graph (the input ``METIS_PartGraphKway`` would receive)."""

    n: int
    xadj: list[int]
    adjncy: list[int]
    adjwgt: list[float]
    vwgt: list[float]
    coords: np.ndarray
    nblocks: tuple[int, int, int]
    block_cells: tuple[int, int, int]
    refinement: tuple[int, ...] = ()

    # -- convenience -------------------------------------------------------
    def neighbors(self, idx: int) -> list[int]:
        return self.adjncy[self.xadj[idx] : self.xadj[idx + 1]]

    def total_weight(self) -> float:
        return float(sum(self.vwgt))

    def cut(self, parts: Sequence[int]) -> float:
        """Total interface weight (in cells) between different parts."""
        parts = np.asarray(parts)
        cut = 0.0
        for u in range(self.n):
            for p in range(self.xadj[u], self.xadj[u + 1]):
                v = self.adjncy[p]
                if parts[u] != parts[v]:
                    cut += self.adjwgt[p]
        return 0.5 * cut

    def part_weights(self, parts: Sequence[int], nparts: int) -> np.ndarray:
        parts = np.asarray(parts)
        w = np.zeros(nparts)
        for idx in range(self.n):
            w[parts[idx]] += self.vwgt[idx]
        return w

    def imbalance(self, parts: Sequence[int], nparts: int) -> float:
        w = self.part_weights(parts, nparts)
        mean = w.mean()
        if mean <= 0:
            return float("inf")
        return float(w.max() / mean - 1.0)

    def to_dense(self) -> np.ndarray:
        a = np.zeros((self.n, self.n))
        for u in range(self.n):
            for p in range(self.xadj[u], self.xadj[u + 1]):
                a[u, self.adjncy[p]] = self.adjwgt[p]
        return a


@dataclass
class PartitionResult:
    """Outcome of one partitioning method."""

    parts: np.ndarray
    cut: float
    part_weights: np.ndarray
    imbalance: float
    method: str
    restarts: int = 1
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 2. METIS-style k-way partitioning (recursive bisection + KL refinement)
# ---------------------------------------------------------------------------
def _connectivity(g: BlockGraph, parts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conn0 = np.zeros(g.n)
    conn1 = np.zeros(g.n)
    for u in range(g.n):
        for p in range(g.xadj[u], g.xadj[u + 1]):
            v = g.adjncy[p]
            if parts[v] == 0:
                conn0[u] += g.adjwgt[p]
            else:
                conn1[u] += g.adjwgt[p]
    return conn0, conn1


def _kl_refine(
    g: BlockGraph,
    parts: np.ndarray,
    nodes: np.ndarray,
    w0: float,
    w1: float,
    target0: float,
    slack: float,
    max_passes: int = 30,
) -> tuple[np.ndarray, float, float]:
    """Kernighan-Lin-style cut refinement with a vertex-weight balance cap.

    Single-node moves first (classic KL gain ordering), then pair swaps.  Only a
    move/swap that strictly reduces the cut *and* keeps ``w0`` within ``slack``
    of ``target0`` is accepted, so balance is never traded away for cut quality.
    """
    conn0, conn1 = _connectivity(g, parts)
    for _ in range(max_passes):
        improved = False
        # -- phase 1: single-node moves ------------------------------------
        for node in nodes:
            node = int(node)
            p = parts[node]
            gain = (conn1[node] - conn0[node]) if p == 0 else (conn0[node] - conn1[node])
            if gain <= 1e-12:
                continue
            w = g.vwgt[node]
            new_w0 = w0 - w if p == 0 else w0 + w
            if abs(new_w0 - target0) > slack:
                continue
            parts[node] = 1 - p
            w0, w1 = new_w0, w1 + (w if p == 0 else -w)
            for q in range(g.xadj[node], g.xadj[node + 1]):
                v = g.adjncy[q]
                wv = g.adjwgt[q]
                if p == 0:
                    conn0[v] -= wv
                    conn1[v] += wv
                else:
                    conn0[v] += wv
                    conn1[v] -= wv
            conn0[node], conn1[node] = conn1[node], conn0[node]
            improved = True
        # -- phase 2: pair swaps -------------------------------------------
        if len(nodes) <= 200:   # pair-swap search is O(n^2) per pass
            p0 = [int(u) for u in nodes if parts[u] == 0]
            p1 = [int(u) for u in nodes if parts[u] == 1]
            best = None
            for a in p0:
                for b in p1:
                    wa, wb = g.vwgt[a], g.vwgt[b]
                    new_w0 = w0 - wa + wb
                    if abs(new_w0 - target0) > slack:
                        continue
                    ga = conn1[a] - conn0[a]
                    gb = conn0[b] - conn1[b]
                    w_ab = 0.0
                    for q in range(g.xadj[a], g.xadj[a + 1]):
                        if g.adjncy[q] == b:
                            w_ab = g.adjwgt[q]
                            break
                    delta = ga + gb - 2.0 * w_ab
                    if delta > 1e-12 and (best is None or delta > best[0]):
                        best = (delta, a, b, new_w0)
            if best is not None:
                _, a, b, new_w0 = best
                w1 = w1 - (new_w0 - w0)
                w0 = new_w0
                parts[a] = 1
                parts[b] = 0
                conn0, conn1 = _connectivity(g, parts)
                improved = True
        if not improved:
            break
    return parts, w0, w1


def _initial_bisection(
    g: BlockGraph,
    nodes: np.ndarray,
    target0: float,
    rng: np.random.Generator,
    jitter: float,
) -> np.ndarray:
    """Spatially contiguous initial split along the longest extent."""
    total = float(np.sum([g.vwgt[u] for u in nodes]))
    frac = 0.5 if total <= 0 else target0 / total
    extent = g.coords[nodes].max(axis=0) - g.coords[nodes].min(axis=0)
    axis = int(np.argmax(extent))
    key = g.coords[nodes, axis].astype(float)
    if jitter > 0:
        key = key + rng.normal(0.0, jitter, size=key.shape)
    order = nodes[np.argsort(key, kind="stable")]
    order_w = np.array([g.vwgt[u] for u in order])
    cum = np.cumsum(order_w)
    cut_at = int(np.searchsorted(cum, frac * total))
    cut_at = max(0, min(len(order), cut_at))
    parts = np.zeros(g.n, dtype=int)
    parts[order[cut_at:]] = 1
    return parts


def _recursive_bisect(
    g: BlockGraph,
    nodes: np.ndarray,
    k: int,
    part_ids: np.ndarray,
    next_id: list[int],
    imbalance_tol: float,
    rng: np.random.Generator,
    jitter: float,
) -> None:
    if k == 1:
        part_ids[nodes] = next_id[0]
        next_id[0] += 1
        return
    k1 = k // 2
    k2 = k - k1
    total = float(np.sum([g.vwgt[u] for u in nodes]))
    slack = 0.5 * imbalance_tol * total / k  # leave slack to deeper levels
    target0 = total * k1 / k

    parts = _initial_bisection(g, nodes, target0, rng, jitter)
    w0 = float(np.sum([g.vwgt[u] for u in nodes if parts[u] == 0]))
    w1 = total - w0
    parts, w0, w1 = _kl_refine(g, parts, nodes, w0, w1, target0, slack)
    if abs(w0 - target0) > slack:  # forced repair of an infeasible split
        side_from, side_to = (0, 1) if w0 > target0 else (1, 0)
        cand = sorted(
            (int(u) for u in nodes if parts[u] == side_from),
            key=lambda u: g.vwgt[u],
        )
        for u in cand:
            parts[u] = side_to
            w0 = w0 - g.vwgt[u] if side_from == 0 else w0 + g.vwgt[u]
            if abs(w0 - target0) <= slack:
                break
        parts, w0, w1 = _kl_refine(g, parts, nodes, w0, w1, target0, slack)

    left = np.array([u for u in nodes if parts[u] == 0], dtype=int)
    right = np.array([u for u in nodes if parts[u] == 1], dtype=int)
    if len(left) == 0 or len(right) == 0:
        for m, u in enumerate(nodes):
            part_ids[u] = next_id[0] + (m % k)
        next_id[0] += k
        return
    _recursive_bisect(g, left, k1, part_ids, next_id, imbalance_tol, rng, jitter)
    _recursive_bisect(g, right, k2, part_ids, next_id, imbalance_tol, rng, jitter)


def _kway_refine(
    g: BlockGraph,
    parts: np.ndarray,
    nparts: int,
    imbalance_tol: float,
    max_passes: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """k-way balance repair + cut reduction (multilevel k-way analogue).

    Recursive bisection balances *subtree* weights; rounding of coarse blocks
    can still leave an individual part over the cap.  This pass (a) moves the
    best-gain node out of an overweight part into an adjacent lighter part and
    (b) then performs plain k-way cut-reduction moves that respect the cap.
    """
    parts = parts.copy()
    total = float(sum(g.vwgt))
    target = total / nparts
    cap = target * (1.0 + imbalance_tol)
    w = g.part_weights(parts, nparts)
    for _ in range(max_passes):
        improved = False
        # (a) balance repair -------------------------------------------------
        for _ in range(4 * g.n):
            heavy = int(np.argmax(w))
            if w[heavy] <= cap + 1e-9:
                break
            best = None
            for node in range(g.n):
                if parts[node] != heavy:
                    continue
                vw = g.vwgt[node]
                to = np.zeros(nparts)
                for q in range(g.xadj[node], g.xadj[node + 1]):
                    to[parts[g.adjncy[q]]] += g.adjwgt[q]
                for p in range(nparts):
                    if p == heavy or w[p] + vw > cap + 1e-9:
                        continue
                    gain = to[p] - to[heavy]
                    if best is None or gain > best[0]:
                        best = (gain, node, p)
            if best is None:
                break
            _, node, p = best
            parts[node] = p
            w[heavy] -= g.vwgt[node]
            w[p] += g.vwgt[node]
            improved = True
        # (b) cut reduction respecting the cap -------------------------------
        for node in range(g.n):
            p = int(parts[node])
            vw = g.vwgt[node]
            to = np.zeros(nparts)
            for q in range(g.xadj[node], g.xadj[node + 1]):
                to[parts[g.adjncy[q]]] += g.adjwgt[q]
            tgt = int(np.argmax(to))
            if tgt == p or to[tgt] - to[p] <= 1e-9:
                continue
            if w[tgt] + vw > cap + 1e-9:
                continue
            parts[node] = tgt
            w[p] -= vw
            w[tgt] += vw
            improved = True
        if not improved:
            break
    return parts, w


def _layered_partition(g: BlockGraph, nparts: int, axis: int) -> PartitionResult:
    """Geometric seed: split the blocks into contiguous layers along ``axis``.

    Cheap k-way candidate that is optimal for grid-like forests; it is the
    "geometric" counterpart of METIS' initial partitioning and guarantees that
    the ported partitioner never loses to a balanced slab.
    """
    other = [a for a in (2, 1, 0) if a != axis]
    key = g.coords[:, [other[0], other[1], axis]]
    order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
    w = np.asarray(g.vwgt, dtype=float)[order]
    cum = np.cumsum(w)
    total = cum[-1]
    parts = np.zeros(g.n, dtype=int)
    for p in range(1, nparts):
        idx = int(np.searchsorted(cum, total * p / nparts))
        idx = min(max(idx, p), g.n - (nparts - p))
        parts[order[idx:]] = p
    return PartitionResult(
        parts=parts,
        cut=g.cut(parts),
        part_weights=g.part_weights(parts, nparts),
        imbalance=g.imbalance(parts, nparts),
        method=f"layered(axis={axis})",
    )


def partition_blockforest(
    forest: "BlockForest | BlockGraph",
    nparts: int,
    *,
    imbalance_tol: float = 0.05,
    restarts: int = 8,
    seed: int = 0,
    stencil: int = 6,
) -> PartitionResult:
    """k-way block-graph partitioning (recursive bisection + KL refinement).

    This mirrors the *intent* of ``METIS_PartGraphKway`` (balance the vertex
    weights, minimise the interface weight) but not its algorithm: METIS is a
    multilevel k-way partitioner with graph coarsening, whereas this
    implementation is a multilevel-*lite* recursive bisection with
    Kernighan-Lin refinement and random restarts.  The quality difference is
    quantified in ``docs/XVAL_BLOCKFOREST_METIS_MULTISTREAM.md``.
    """
    g = forest.graph(stencil=stencil) if isinstance(forest, BlockForest) else forest
    if nparts < 1:
        raise ValueError("nparts must be >= 1")
    if nparts > g.n:
        raise ValueError(f"nparts={nparts} exceeds the number of blocks ({g.n})")
    nodes = np.arange(g.n, dtype=int)
    best: PartitionResult | None = None
    # geometric seeds (contiguous layers along each axis)
    for axis in (0, 1, 2):
        cand = _layered_partition(g, nparts, axis)
        if cand.imbalance <= imbalance_tol + 1e-12 or best is None:
            if best is None or (
                (cand.imbalance <= imbalance_tol + 1e-12)
                and (best.imbalance > imbalance_tol + 1e-12 or cand.cut < best.cut)
            ):
                best = cand
    for r in range(max(1, restarts)):
        rng = np.random.default_rng(seed + r)
        jitter = 0.0 if r == 0 else 0.15
        parts = np.zeros(g.n, dtype=int)
        next_id = [0]
        _recursive_bisect(g, nodes, nparts, parts, next_id, imbalance_tol, rng, jitter)
        parts, _ = _kway_refine(g, parts, nparts, imbalance_tol)
        res = PartitionResult(
            parts=parts,
            cut=g.cut(parts),
            part_weights=g.part_weights(parts, nparts),
            imbalance=g.imbalance(parts, nparts),
            method="recursive_bisection+KL",
            restarts=r + 1,
        )
        feasible = res.imbalance <= imbalance_tol + 1e-12
        if best is None:
            best = res
            continue
        best_feasible = best.imbalance <= imbalance_tol + 1e-12
        if feasible and not best_feasible:
            best = res
        elif feasible == best_feasible and res.cut < best.cut - 1e-12:
            best = res
    return best


def slab_partition(forest: "BlockForest | BlockGraph", nparts: int) -> PartitionResult:
    """Baseline: contiguous z-slabs over the linear block index.

    This is the decomposition the repository ships today
    (``DomainDecomposer`` splits along z), reproduced on the block graph so both
    methods can be scored with the same cut / imbalance metrics.
    """
    g = forest.graph(stencil=6) if isinstance(forest, BlockForest) else forest
    if nparts > g.n:
        raise ValueError(f"nparts={nparts} exceeds the number of blocks ({g.n})")
    parts = np.zeros(g.n, dtype=int)
    edges = np.linspace(0, g.n, nparts + 1)
    for p in range(nparts):
        lo, hi = int(round(edges[p])), int(round(edges[p + 1]))
        parts[lo:hi] = p
    return PartitionResult(
        parts=parts,
        cut=g.cut(parts),
        part_weights=g.part_weights(parts, nparts),
        imbalance=g.imbalance(parts, nparts),
        method="z-slab (repo baseline)",
    )


def weighted_slab_partition(
    forest: "BlockForest | BlockGraph", nparts: int
) -> PartitionResult:
    """Balanced 1-D slab baseline: contiguous ranges with equal *workload*.

    The natural middle ground between the shipped z-slab (equal block count) and
    a full 3-D graph partition: it keeps the 1-D communicator but splits at
    cumulative-weight boundaries, so it is a *balanced* competitor.
    """
    g = forest.graph(stencil=6) if isinstance(forest, BlockForest) else forest
    if nparts > g.n:
        raise ValueError(f"nparts={nparts} exceeds the number of blocks ({g.n})")
    w = np.asarray(g.vwgt, dtype=float)
    cum = np.cumsum(w)
    total = cum[-1]
    parts = np.zeros(g.n, dtype=int)
    for p in range(1, nparts):
        target = total * p / nparts
        idx = int(np.searchsorted(cum, target))
        idx = min(max(idx, p), g.n - (nparts - p))
        parts[idx:] = p
    return PartitionResult(
        parts=parts,
        cut=g.cut(parts),
        part_weights=g.part_weights(parts, nparts),
        imbalance=g.imbalance(parts, nparts),
        method="weighted z-slab",
    )


def exact_bisection_cut(
    g: BlockGraph,
    imbalance_tol: float = 0.05,
    max_nodes: int = 22,
) -> PartitionResult | None:
    """Exhaustive search for the *optimal* balanced bisection (small graphs).

    Returns ``None`` when ``g.n > max_nodes``.  Feasibility is
    ``max(w0, w1) / mean(w0, w1) - 1 <= imbalance_tol``.  Because no
    partitioning algorithm (METIS included) can beat this cut, a partitioner
    that lands within ``x``% of it is provably within ``x``% of *any* reference
    implementation on the same instance.
    """
    n = g.n
    if n > max_nodes:
        return None
    adj = g.to_dense()
    wv = np.asarray(g.vwgt, dtype=float)
    total = wv.sum()
    half = n - 1  # node 0 pinned to part 0 (symmetry breaking)
    chunk = 1 << 14
    edges = [(u, v) for u in range(n) for v in range(u + 1, n) if adj[u, v] > 0]
    best_cut = math.inf
    best_mask = None
    for start in range(0, 1 << half, chunk):
        block = np.arange(start, min(start + chunk, 1 << half), dtype=np.int64)
        w1 = np.zeros(block.size)
        bits = {}
        for i in range(1, n):
            bits[i] = ((block >> (i - 1)) & 1).astype(np.int8)
            w1 += bits[i] * wv[i]
        mean = total / 2.0
        imb = np.maximum(w1, total - w1) / mean - 1.0
        feasible = imb <= imbalance_tol + 1e-12
        if not feasible.any():
            continue
        cut = np.zeros(block.size)
        for (u, v) in edges:
            bu = np.zeros(block.size, dtype=np.int8) if u == 0 else bits[u]
            cut += adj[u, v] * (bu != bits[v])
        cut = np.where(feasible, cut, np.inf)
        idx = int(np.argmin(cut))
        if cut[idx] < best_cut:
            best_cut = float(cut[idx])
            best_mask = int(block[idx])
    if best_mask is None:
        return None
    parts = np.zeros(n, dtype=int)
    for i in range(1, n):
        parts[i] = (best_mask >> (i - 1)) & 1
    return PartitionResult(
        parts=parts,
        cut=g.cut(parts),
        part_weights=g.part_weights(parts, 2),
        imbalance=g.imbalance(parts, 2),
        method="exhaustive optimum",
        extra={"n_enumerated": int(1 << half)},
    )

# ---------------------------------------------------------------------------
# 3. Multistream (compute / communication overlap) LBM stepper
# ---------------------------------------------------------------------------
@dataclass
class StepTiming:
    """Per-step wall-clock breakdown (seconds)."""

    mode: str
    latency: float
    t_collide: float
    t_comm_wall: float
    t_interior: float
    t_shell: float
    t_total: float


class MultistreamBlockSim:
    """Single-process block-forest LBM with a selectable communication mode.

    Field storage follows the reference layout: every block owns **one** buffer
    ``pad`` of shape ``(19, nz+2, ny+2, nx+2)`` whose inner ``(19, nz, ny, nx)``
    window is the owned field and whose 1-cell shell is the ghost layer.  The
    post-collision state is written straight into the owned window, so ghost
    exchange writes *only* the shell - no packing copy is needed.

    ``mode="serialized"``  ghost planes are filled *before* any streaming work
                           starts (the blocking halo exchange of
                           :mod:`tensorlbm.domain_decomposition`).
    ``mode="overlapped"``  ghosts are filled by a worker thread while the main
                           thread streams the block interior - the interior of
                           a block only reads owned cells, so it is independent
                           of the remote planes.  The boundary layer is streamed
                           after the join, i.e. the reference implementation's
                           pack / async-send / interior / wait / boundary order.

    Both modes execute identical arithmetic in a different order, therefore
    they agree bitwise; ``tests/test_blockforest.py`` asserts this.
    """

    def __init__(
        self,
        forest: BlockForest,
        *,
        tau: float = 0.8,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if forest.refinement is not None and any(forest.refinement):
            raise NotImplementedError(
                "MultistreamBlockSim supports uniform forests only; refinement "
                "levels are honoured by the graph/partitioner, not by this stepper."
            )
        nz, ny, nx = forest.block_cells
        if min(nz, ny, nx) < 3:
            raise ValueError("block_cells must be >= 3 along every axis")
        self.forest = forest
        self.tau = float(tau)
        self.dtype = dtype
        self.shape = (nz, ny, nx)
        self.plan = self._build_ghost_plan(forest)
        self.outgoing = self._build_outgoing(forest)
        n = forest.n_blocks
        self.pad: dict[int, torch.Tensor] = {
            idx: torch.zeros((19, nz + 2, ny + 2, nx + 2), dtype=dtype) for idx in range(n)
        }

    # -- owned-field views --------------------------------------------------
    def _own(self, idx: int) -> torch.Tensor:
        nz, ny, nx = self.shape
        return self.pad[idx][:, 1 : nz + 1, 1 : ny + 1, 1 : nx + 1]

    @property
    def f(self) -> dict[int, torch.Tensor]:
        """Owned (ghost-free) views, one per block - read-only use."""
        return {idx: self._own(idx) for idx in range(self.forest.n_blocks)}

    # -- setup -------------------------------------------------------------
    @staticmethod
    def _build_ghost_plan(
        forest: BlockForest,
    ) -> list[tuple[int, int, tuple[slice, ...], tuple[slice, ...]]]:
        """``(target_block, source_block, target_slices, source_slices)``.

        Slices are expressed in *padded* coordinates, i.e. owned cell ``i`` of a
        block lives at padded index ``i + 1``.
        """
        nz, ny, nx = forest.block_cells
        bz, by, bx = forest.nblocks
        plan: list[tuple[int, int, tuple[slice, ...], tuple[slice, ...]]] = []
        for idx in range(forest.n_blocks):
            i, j, k = forest.coords(idx)
            for dz, dy, dx in N26_OFFSETS:
                nco = [i + dz, j + dy, k + dx]
                ok = True
                for a, (comp, bi) in enumerate(zip((dz, dy, dx), (bz, by, bx))):
                    if comp == 0:
                        continue
                    if forest.periodic[a]:
                        nco[a] %= bi
                    elif not (0 <= nco[a] < bi):
                        ok = False
                if not ok:
                    continue
                nb = forest.index(*nco)
                tgt: list[slice] = []
                src: list[slice] = []
                for comp, m in ((dz, nz), (dy, ny), (dx, nx)):
                    if comp == -1:      # ghost below  <- neighbour's last plane
                        tgt.append(slice(0, 1))
                        src.append(slice(m, m + 1))
                    elif comp == 1:     # ghost above  <- neighbour's first plane
                        tgt.append(slice(m + 1, m + 2))
                        src.append(slice(1, 2))
                    else:
                        tgt.append(slice(1, m + 1))
                        src.append(slice(1, m + 1))
                plan.append((idx, nb, tuple(tgt), tuple(src)))
        return plan

    @staticmethod
    def _build_outgoing(
        forest: BlockForest,
    ) -> dict[int, list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]]:
        """Push-style (send) view of the ghost plan.

        ``outgoing[i]`` lists every write whose *source* is block ``i``, i.e.
        everything block ``i`` has to send once its post-collision state is
        ready.  This is what makes the exchange asynchronous: block ``i`` can
        ship its planes as soon as it has collided, without waiting for the
        other blocks.  The 26 entries per block touch pairwise disjoint ghost
        regions, so concurrent pushes never race.
        """
        plan = MultistreamBlockSim._build_ghost_plan(forest)
        out: dict[int, list[tuple[int, tuple[slice, ...], tuple[slice, ...]]]] = {
            idx: [] for idx in range(forest.n_blocks)
        }
        for blk, nb, tgt, src in plan:
            out[nb].append((blk, tgt, src))
        return out

    def _global_slices(self, idx: int) -> tuple[slice, slice, slice]:
        nz, ny, nx = self.shape
        i, j, k = self.forest.coords(idx)
        return (
            slice(i * nz, (i + 1) * nz),
            slice(j * ny, (j + 1) * ny),
            slice(k * nx, (k + 1) * nx),
        )

    # -- initial conditions -------------------------------------------------
    def set_shear_wave(
        self, amplitude: float = 0.05, wavenumber: float | None = None, rho0: float = 1.0
    ) -> None:
        """Kolmogorov shear wave ``u_x(y) = A sin(k y)`` (exact NS solution)."""
        nzg, nyg, nxg = self.forest.global_shape()
        k = 2.0 * math.pi / nyg if wavenumber is None else float(wavenumber)
        y_centers = np.arange(nyg) + 0.5
        profile = amplitude * np.sin(k * y_centers)
        nz, ny, nx = self.shape
        for idx in range(self.forest.n_blocks):
            _, ys, _ = self._global_slices(idx)
            ux = np.broadcast_to(profile[ys][None, :, None], (nz, ny, nx)).copy()
            ux_t = torch.from_numpy(ux).to(self.dtype)
            rho = torch.full((nz, ny, nx), rho0, dtype=self.dtype)
            zero = torch.zeros((nz, ny, nx), dtype=self.dtype)
            self.pad[idx].zero_()
            self._own(idx).copy_(equilibrium3d(rho, ux_t, zero, zero))

    # -- observables --------------------------------------------------------
    def velocity_x(self) -> torch.Tensor:
        """Assemble the global ``u_x`` field ``(nzg, nyg, nxg)``."""
        nzg, nyg, nxg = self.forest.global_shape()
        out = torch.zeros((nzg, nyg, nxg), dtype=self.dtype)
        for idx in range(self.forest.n_blocks):
            zs, ys, xs = self._global_slices(idx)
            _, ux, _, _ = macroscopic3d(self._own(idx))
            out[zs, ys, xs] = ux
        return out

    def total_mass(self) -> float:
        return float(sum(float(self._own(idx).sum()) for idx in range(self.forest.n_blocks)))

    # -- communication ------------------------------------------------------
    def _push_block(self, idx: int) -> None:
        """Push block ``idx``'s post-collision planes into its neighbours."""
        src_pad = self.pad[idx]
        for tgt_blk, tgt, src in self.outgoing[idx]:
            self.pad[tgt_blk][(slice(None),) + tgt] = src_pad[(slice(None),) + src]

    def _fill_ghosts(self, latency: float) -> float:
        """Blocking exchange: latency, then every block ships its planes."""
        t0 = time.perf_counter()
        if latency > 0:
            time.sleep(latency)
        for idx in range(self.forest.n_blocks):
            self._push_block(idx)
        return time.perf_counter() - t0

    def _pump(self, work: "queue.Queue[object]", stats: dict[str, float]) -> None:
        """Worker: one-way latency, then drain the send queue."""
        t0 = time.perf_counter()
        if stats.get("latency", 0.0) > 0:
            time.sleep(stats["latency"])
        while True:
            item = work.get()
            if item is None:
                break
            self._push_block(int(item))  # type: ignore[arg-type]
        stats["wall"] = time.perf_counter() - t0

    # -- streaming kernels --------------------------------------------------
    def _stream_core(self, out: torch.Tensor, blk: int) -> None:
        """Stream the block interior: reads owned cells only (no ghosts)."""
        nz, ny, nx = self.shape
        pad = self.pad[blk]
        for q in range(19):
            cz, cy, cx = int(C[q, 2]), int(C[q, 1]), int(C[q, 0])
            out[q, 1 : nz - 1, 1 : ny - 1, 1 : nx - 1] = pad[
                q,
                2 - cz : nz - cz,
                2 - cy : ny - cy,
                2 - cx : nx - cx,
            ]

    def _stream_shell(self, out: torch.Tensor, blk: int) -> None:
        """Stream the 1-cell boundary layer: needs the ghost planes."""
        nz, ny, nx = self.shape
        pad = self.pad[blk]
        for q in range(19):
            cz, cy, cx = int(C[q, 2]), int(C[q, 1]), int(C[q, 0])
            # padded index of the upstream cell is (i + 1 - cz, j + 1 - cy,
            # k + 1 - cx); all six faces use the full upstream window so that
            # edge/corner cells are written consistently by every face.
            out[q, 0, :, :] = pad[q, 1 - cz, 1 - cy : 1 - cy + ny, 1 - cx : 1 - cx + nx]
            out[q, nz - 1, :, :] = pad[q, nz - cz, 1 - cy : 1 - cy + ny, 1 - cx : 1 - cx + nx]
            out[q, :, 0, :] = pad[q, 1 - cz : 1 - cz + nz, 1 - cy, 1 - cx : 1 - cx + nx]
            out[q, :, ny - 1, :] = pad[q, 1 - cz : 1 - cz + nz, ny - cy, 1 - cx : 1 - cx + nx]
            out[q, :, :, 0] = pad[q, 1 - cz : 1 - cz + nz, 1 - cy : 1 - cy + ny, 1 - cx]
            out[q, :, :, nx - 1] = pad[q, 1 - cz : 1 - cz + nz, 1 - cy : 1 - cy + ny, nx - cx]

    # -- one time step ------------------------------------------------------
    def step(self, mode: str = "serialized", latency: float = 0.0) -> StepTiming:
        """Advance one BGK step in the requested communication mode."""
        if mode not in ("serialized", "overlapped"):
            raise ValueError("mode must be 'serialized' or 'overlapped'")
        n = self.forest.n_blocks
        with torch.no_grad():
            own = [self._own(idx) for idx in range(n)]
            out = {idx: torch.empty_like(own[idx]) for idx in range(n)}
            t0 = time.perf_counter()

            if mode == "serialized":
                for idx in range(n):                       # collide (all blocks)
                    own[idx].copy_(collide_bgk3d(own[idx], self.tau))
                t1 = time.perf_counter()
                self._fill_ghosts(latency)                 # blocking exchange
                t2 = time.perf_counter()
                for idx in range(n):                       # interior streaming
                    self._stream_core(out[idx], idx)
                t3 = time.perf_counter()
                for idx in range(n):                       # boundary streaming
                    self._stream_shell(out[idx], idx)
                t4 = time.perf_counter()
                for idx in range(n):
                    own[idx].copy_(out[idx])
                t5 = time.perf_counter()
                return StepTiming(
                    mode=mode,
                    latency=latency,
                    t_collide=t1 - t0,
                    t_comm_wall=t2 - t1,
                    t_interior=t3 - t2,
                    t_shell=t4 - t3,
                    t_total=t5 - t0,
                )

            # ---- overlapped (multistream) schedule ------------------------
            work: "queue.Queue[object]" = queue.Queue()
            stats: dict[str, float] = {"latency": float(latency)}
            th = threading.Thread(
                target=self._pump, args=(work, stats), name="comm-stream", daemon=True
            )
            t_spawn = time.perf_counter()
            th.start()
            for idx in range(n):
                own[idx].copy_(collide_bgk3d(own[idx], self.tau))
                work.put(idx)          # ship as soon as this block has collided
            t1 = time.perf_counter()
            for idx in range(n):                       # overlaps with the sends
                self._stream_core(out[idx], idx)
            t3 = time.perf_counter()
            work.put(None)
            th.join()                                  # wait for the exchange
            t4 = time.perf_counter()
            for idx in range(n):
                self._stream_shell(out[idx], idx)
            t5 = time.perf_counter()
            for idx in range(n):
                own[idx].copy_(out[idx])
            t6 = time.perf_counter()
            return StepTiming(
                mode=mode,
                latency=latency,
                t_collide=t1 - t_spawn,
                t_comm_wall=stats.get("wall", t4 - t3),
                t_interior=t3 - t1,
                t_shell=t5 - t4,
                t_total=t6 - t0,
            )

    # -- convenience --------------------------------------------------------
    def run(
        self, steps: int, mode: str = "serialized", latency: float = 0.0
    ) -> list[StepTiming]:
        return [self.step(mode=mode, latency=latency) for _ in range(steps)]


def monolithic_shear_wave_step(
    f: torch.Tensor, tau: float, coords: str = "roll"
) -> torch.Tensor:
    """Independent single-block reference (existing repo kernels, periodic).

    ``collide_bgk3d`` + ``stream3d_roll`` is the code path used by the
    production solvers; using it as the reference keeps the decomposed /
    overlapped result honest about being cross-checked against a *different*
    implementation and not against itself.
    """
    g = collide_bgk3d(f, tau)
    if coords != "roll":
        raise ValueError("only 'roll' available")
    return stream3d_roll(g)
