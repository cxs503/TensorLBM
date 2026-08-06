"""P1 geometry for the octree boundary layer of the hybrid AMR architecture.

Implements:

* Morton (Z-order) encoding/decoding of shell leaves (uint64, 3 bits per
  level, root bit ``1`` — scheme of the Octree-LBM-solver reference project).
* The body-fitted octree shell: from the L1 block's solid mask and sphere
  geometry we derive the near-wall fluid cell mask (Euclidean distance to the
  sphere surface, ``bl_thickness_cells`` + a one-cell transition band), split
  every masked cell into 8 depth-1 leaves, and refine wall-adjacent depth-1
  leaves (BFL boundary directions non-empty) into depth-2 leaves.  Solid
  sub-leaves (centres inside the sphere) are dropped.
* :class:`OctreeGrid` — the SoA layout consumed by the P2 stepper
  (``leaf_morton``, ``leaf_level``, ``neighbor_table``, ``q_field``,
  ``f_leaf``, ...) together with leaf statistics (volume vs analytic shell,
  cell saving vs an axis-aligned bounding box) and topology checks.

Coordinate conventions (all in L1 *physical* cell units, origin at the
block's physical-domain corner ``(0, 0, 0)``):

* a coarse L1 cell ``(z, y, x)`` occupies world space ``[x, x+1) x [y, y+1) x
  [z, z+1)``;
* a depth-``l`` leaf has ``dx_l = 2^-l`` and Morton lattice coordinates
  ``L = 2^l * cell + b`` (``b`` the per-axis child index);
* leaf centres are ``(L + 0.5) / 2^l`` in world units.

The Morton code is rooted at the whole L1 block: with ``K`` bits per axis
(``K = ceil(log2(max(nx, ny, nz)))``) a depth-``l`` leaf encodes as
``1 << (3*(K+l)) | interleave(coords, K+l)``, so parent/child relations are
plain ``>> 3`` shifts and codes are unique across levels.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from tensorlbm.boundaries3d import sphere_mask  # noqa: F401  (documented reference; the shell builder uses the centre-based solid mask)

# Sentinel values for neighbor_table (see design doc §3.1):
SHELL_OUTSIDE = -1   # neighbour leaves the shell -> L1 block interface
SOLID = -2           # neighbour is inside the body (solid / dropped leaf)
DOMAIN_OUT = -3      # neighbour is outside the L1 block (should not happen)
FANOUT = -4          # coarse -> fine cross-level link, consult interface_fanout

# ---------------------------------------------------------------------------
# Morton (Z-order) codec — uint64, 3 bits per level, root bit 1
# ---------------------------------------------------------------------------


def _axis_bits(shape: tuple[int, int, int]) -> int:
    """Bits per axis needed to index the L1 block (K in the module docstring)."""
    return max(1, max(int(math.ceil(math.log2(max(shape)))), 1))


def morton_encode(level: int, x: int, y: int, z: int, k: int) -> int:
    """Encode one leaf's Morton code.

    Args:
        level: subdivision depth (1 or 2).
        x, y, z: Morton lattice coordinates (``2^level * cell + child`` bits).
        k: axis bit width of the L1 block (see :func:`_axis_bits`).

    Returns:
        int: ``1 << (3*(k+level)) | interleave(coords, k+level)``.
    """
    width = k + level
    m = 0
    for i in range(width):
        m |= ((x >> i) & 1) << (3 * i)
        m |= ((y >> i) & 1) << (3 * i + 1)
        m |= ((z >> i) & 1) << (3 * i + 2)
    return (1 << (3 * width)) | m


def morton_decode(bits: int, k: int) -> tuple[int, int, int, int]:
    """Decode a Morton code back to ``(level, x, y, z)``."""
    if bits <= 0:
        raise ValueError(f"invalid Morton code {bits}")
    width = (bits.bit_length() - 1) // 3
    level = width - k
    if level < 0:
        raise ValueError(
            f"Morton code {bits} has width {width} < block axis width {k}",
        )
    x = y = z = 0
    for i in range(width):
        x |= ((bits >> (3 * i)) & 1) << i
        y |= ((bits >> (3 * i + 1)) & 1) << i
        z |= ((bits >> (3 * i + 2)) & 1) << i
    return level, x, y, z


def morton_encode_batch(
    level: torch.Tensor, coords: torch.Tensor, k: int,
) -> torch.Tensor:
    """Vectorised Morton encode for ``(n, 3)`` int64 coordinates.

    ``level`` is a broadcastable int tensor/array of subdivision depths.
    """
    level = torch.as_tensor(level, dtype=torch.int64)
    coords = torch.as_tensor(coords, dtype=torch.int64)
    n = coords.shape[0]
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    width = int(level.max().item()) + k
    m = torch.zeros(n, dtype=torch.int64, device=coords.device)
    for i in range(width):
        bit = (torch.tensor(1, dtype=torch.int64, device=coords.device) << i)
        m |= ((x >> i) & 1) << (3 * i)
        m |= ((y >> i) & 1) << (3 * i + 1)
        m |= ((z >> i) & 1) << (3 * i + 2)
    m |= torch.tensor(1, dtype=torch.int64, device=coords.device) << (3 * (level + k))
    return m


def morton_decode_batch(bits: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorised Morton decode -> ``(level (n,), coords (n, 3))``."""
    bits = torch.as_tensor(bits, dtype=torch.int64)
    n = bits.shape[0]
    if n == 0:
        return torch.empty(0, dtype=torch.int64), torch.empty((0, 3), dtype=torch.int64)
    bmax = int(bits.max().item())
    wmax = (bmax.bit_length() + 2) // 3  # ceil(bit_length / 3)
    # per-element width: number of 3-bit groups above the root bit
    level = torch.zeros(n, dtype=torch.int64, device=bits.device)
    tmp = bits.clone()
    for _ in range(wmax):
        has = (tmp > 1).to(torch.int64)
        level += has
        tmp = tmp >> 3
    if bool((level < 0).any()):
        raise ValueError("Morton code width smaller than block axis width")
    x = torch.zeros(n, dtype=torch.int64, device=bits.device)
    y = torch.zeros_like(x)
    z = torch.zeros_like(x)
    for i in range(wmax):
        x |= ((bits >> (3 * i)) & 1) << i
        y |= ((bits >> (3 * i + 1)) & 1) << i
        z |= ((bits >> (3 * i + 2)) & 1) << i
    coords = torch.stack([x, y, z], dim=1)
    return level, coords


def morton_parent(bits: int, k: int) -> int:
    """Morton code of the parent cell (drop the lowest 3 bits)."""
    level = (bits.bit_length() - 1) // 3 - k
    if level <= 0:
        raise ValueError("root has no parent")
    return bits >> 3


def morton_child(bits: int, child: int, k: int) -> int:
    """Morton code of child ``child`` (0..7) of ``bits``."""
    if not 0 <= child <= 7:
        raise ValueError(f"child index must be in [0, 7], got {child}")
    return (bits << 3) | child


# ---------------------------------------------------------------------------
# Shell cell mask (near-wall fluid cells of the L1 block)
# ---------------------------------------------------------------------------


def sphere_distance_field(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    device: torch.device,
) -> torch.Tensor:
    """Signed distance ``|p - center| - radius`` of every cell centre.

    The sample points are the cell *centres* ``(i + 0.5)`` in each axis
    (the L1 block convention used everywhere else, e.g. the leaf centres
    ``(coords + 0.5) / 2^level`` and the ghost-plan sampling).  Evaluating
    at the integer cell corners instead would shift the shell band by half
    a cell relative to the sphere centre and break the reflection symmetry
    of the shell/leaf/BFL geometry about the body.
    """
    nz, ny, nx = shape
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float64) + 0.5,
        torch.arange(ny, device=device, dtype=torch.float64) + 0.5,
        torch.arange(nx, device=device, dtype=torch.float64) + 0.5,
        indexing="ij",
    )
    return (
        (xx - center[0]) ** 2
        + (yy - center[1]) ** 2
        + (zz - center[2]) ** 2
    ).sqrt() - radius


def build_shell_cell_mask(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    bl_thickness_cells: float,
    transition: int = 1,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Near-wall fluid cell mask of the L1 block.

    A cell belongs to the shell when its centre is fluid (outside the sphere)
    and its distance to the sphere surface is ``<= bl_thickness_cells +
    transition`` (the transition band guarantees a buffer zone around the
    wall-adjacent leaves, mirroring the 3D-LBM-AMR ``Remap`` buffer).

    Returns:
        ``(solid, shell_mask, delta_mask)`` — the solid cell mask, the bool
        shell mask ``(nz, ny, nx)`` and the effective shell half-thickness
        ``delta_mask`` used for the analytic shell volume.
    """
    nz, ny, nx = shape
    # Solid = cells whose *centre* lies inside the analytic sphere.  This is
    # the same convention as the shell band (``sphere_distance_field``), the
    # leaf split (``_level1_leaves`` centre test) and the neighbour-table
    # SOLID sentinel (``topology._classify_targets`` samples leaf centres).
    # The corner-based voxel mask from ``boundaries3d.sphere_mask`` instead
    # marks cells whose nearest *corner* is inside — for an integer-centred
    # sphere that cell (e.g. index ``c + R``, corner exactly on the surface)
    # has its centre *outside* the wall, so marking it solid silently deletes
    # the surface leaf ring on that side and breaks the reflection symmetry
    # of the shell/BFL geometry about the body (spurious net link force).
    solid = sphere_distance_field(shape, center, radius, device) <= 0.0
    dist = sphere_distance_field(shape, center, radius, device)
    delta_mask = float(bl_thickness_cells) + float(transition)
    shell_mask = (~solid) & (dist <= delta_mask)
    return solid, shell_mask, delta_mask


# ---------------------------------------------------------------------------
# OctreeGrid — SoA layout of the shell leaves
# ---------------------------------------------------------------------------


@dataclass
class OctreeGrid:
    """Structure-of-arrays geometry of the octree boundary shell.

    See design doc §3.1 for the tensor-layout contract.  All tensors are
    compact and contiguous; ``neighbor_table[d, i]`` holds a leaf enum index
    (same-level or cross-level donor), or one of the sentinels
    ``SHELL_OUTSIDE / SOLID / DOMAIN_OUT / FANOUT``.
    """

    n_leaf: int
    d_max: int
    Q: int
    level_start: torch.Tensor          # (3,) int64: [start_l1, start_l2, n_leaf]
    leaf_morton: torch.Tensor          # (n_leaf,) int64
    leaf_level: torch.Tensor           # (n_leaf,) int64  (1 or 2)
    leaf_center: torch.Tensor          # (n_leaf, 3) float32, world units
    leaf_box: torch.Tensor             # (n_leaf, 2, 3) float32
    neighbor_table: torch.Tensor       # (Q, n_leaf) int64
    q_field: torch.Tensor              # (Q, n_leaf) float32
    bfl_mask: torch.Tensor             # (Q, n_leaf) bool
    interface_links: torch.Tensor      # (n_link, 2) int64 (leaf i, direction d)
    interface_fanout: dict             # {(i, d): list[int]} coarse->fine leaves
    cross_level_donor: torch.Tensor    # (Q, n_leaf) int64, -1 = none
    leaf_host_cell: torch.Tensor       # (n_leaf, 3) int64 (z, y, x) in L1 block
    f_leaf: torch.Tensor               # (Q, n_leaf) float32 (SoA populations)
    morton_to_index: dict = field(default_factory=dict)   # int -> leaf enum
    meta: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)
    checks: dict = field(default_factory=dict)
    # internal working sets populated by the builder (not part of the API)
    _l1_coords: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3), dtype=torch.int64))
    _l2_coords: torch.Tensor = field(default_factory=lambda: torch.empty((0, 3), dtype=torch.int64))
    _k: int = 0
    _c_vec: torch.Tensor | None = None
    _opp: torch.Tensor | None = None
    _solid: torch.Tensor | None = None
    _delta_mask: float = 0.0
    _shell_mask: torch.Tensor | None = None
    _fanout_groups: dict = field(default_factory=dict)  # {(i, d): [leaf enums]}

    # -- level helpers ------------------------------------------------------
    def n_leaf_level(self, level: int) -> int:
        if not 1 <= level <= 2:
            raise ValueError(f"shell levels are 1 and 2, got {level}")
        return int(self.level_start[level] - self.level_start[level - 1])

    def leaf_indices(self, level: int) -> torch.Tensor:
        return torch.arange(
            int(self.level_start[level - 1]), int(self.level_start[level]),
        )

    def level_of(self) -> torch.Tensor:
        return self.leaf_level

    def leaf_volume(self) -> torch.Tensor:
        """Per-leaf volume in L1-cell units (``dx_l^3 = 2^{-3 l}``)."""
        return (2.0 ** (-3 * self.leaf_level.to(torch.float64))).to(
            torch.float64,
        )

    def total_volume(self) -> float:
        return float(self.leaf_volume().sum().item())


# ---------------------------------------------------------------------------
# Leaf construction (level-1 split + wall-adjacent depth-2 refinement)
# ---------------------------------------------------------------------------


def _level1_leaves(
    shell_cells: torch.Tensor,
    center: tuple[float, float, float],
    radius: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split every masked cell into 8 depth-1 leaves, dropping solid ones.

    Args:
        shell_cells: ``(N, 3)`` int64 masked cell coordinates ``(z, y, x)``.

    Returns:
        ``(coords (M,3) int64, centers (M,3) float64, inside (M,) bool)``
        with ``coords`` in Morton lattice order ``(x, y, z)`` and ``centers``
        in world units; ``inside`` flags leaves whose centre is still inside
        the sphere.
    """
    n = shell_cells.shape[0]
    child = torch.arange(8, dtype=torch.int64, device=device)
    bx, by, bz = child & 1, (child >> 1) & 1, (child >> 2) & 1
    cells = shell_cells[:, [2, 1, 0]].repeat_interleave(8, dim=0)  # (x,y,z)
    offs = torch.stack([bx, by, bz], dim=1).repeat(n, 1)           # (8N, 3)
    coords = 2 * cells + offs                                      # level-1 coords
    centers = (coords.to(torch.float64) + 0.5) / 2.0               # world units
    dist2 = (
        (centers[:, 0] - center[0]) ** 2
        + (centers[:, 1] - center[1]) ** 2
        + (centers[:, 2] - center[2]) ** 2
    )
    inside = dist2 <= radius ** 2
    return coords, centers, inside


def _level2_leaves(
    parent_coords: torch.Tensor,
    center: tuple[float, float, float],
    radius: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split depth-1 leaves into 8 depth-2 leaves, dropping solid ones."""
    n = parent_coords.shape[0]
    child = torch.arange(8, dtype=torch.int64, device=device)
    bx, by, bz = child & 1, (child >> 1) & 1, (child >> 2) & 1
    parents = parent_coords.repeat_interleave(8, dim=0)
    offs = torch.stack([bx, by, bz], dim=1).repeat(n, 1)
    coords = 2 * parents + offs
    centers = (coords.to(torch.float64) + 0.5) / 4.0
    dist2 = (
        (centers[:, 0] - center[0]) ** 2
        + (centers[:, 1] - center[1]) ** 2
        + (centers[:, 2] - center[2]) ** 2
    )
    inside = dist2 <= radius ** 2
    return coords, centers, inside


# ---------------------------------------------------------------------------
# Statistics (volume error, cell saving)
# ---------------------------------------------------------------------------


def analytic_shell_volume(
    radius: float, delta_mask: float, dx: float = 1.0,
) -> float:
    """Analytic volume of the shell region covered by the leaf set.

    The masked cells are those whose *centres* lie between ``radius`` and
    ``radius + delta_mask`` from the sphere centre; at unit cell density the
    count of such centres converges to the continuous shell
    ``[R, R + delta_mask]`` (boundary saw-tooth errors cancel statistically).
    This is the reference against which the discrete leaf volume is measured.
    """
    r_in = max(0.0, radius)
    r_out = radius + delta_mask
    return 4.0 / 3.0 * math.pi * (r_out ** 3 - r_in ** 3)


def cell_saving_report(
    n_leaf: int, shell_cells: torch.Tensor, d_max: int,
) -> dict:
    """Leaf count vs the same region resolved as a rectangular box.

    The axis-aligned bounding box of the masked shell cells, subdivided to
    the finest shell resolution (``2^d_max`` per axis), is the rectangular
    reference.  ``saving_fraction = 1 - n_leaf / box_cells``.
    """
    cells = torch.as_tensor(shell_cells)
    lo = cells.min(dim=0).values
    hi = cells.max(dim=0).values + 1
    box_vol = int(((hi - lo).prod().item()))
    box_cells = box_vol * (2 ** (3 * d_max))
    saving = 1.0 - n_leaf / box_cells if box_cells > 0 else 0.0
    return {
        "n_leaf": int(n_leaf),
        "box_volume_cells": box_vol,
        "box_cells_at_dmax": box_cells,
        "saving_fraction": float(saving),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _aabb_sphere_intersect(
    centers: torch.Tensor, half: float,
    center: tuple[float, float, float], radius: float,
) -> torch.Tensor:
    """AABB-vs-sphere intersection for leaf boxes ``[c-half, c+half]``.

    Returns a bool mask per leaf.  Used as the depth-2 refinement criterion:
    a depth-1 leaf is refined when its box crosses the sphere surface, so the
    wall is resolved at depth-2 resolution and no depth-1-scale solid sub-leaf
    volume is lost (the solid drop happens at depth-2 granularity).
    """
    c = torch.tensor(center, dtype=torch.float64, device=centers.device)
    d = (centers - c).abs()
    min_dist2 = ((d - half).clamp(min=0.0) ** 2).sum(dim=1)
    max_dist2 = ((d + half) ** 2).sum(dim=1)
    return (min_dist2 <= radius ** 2) & (max_dist2 >= radius ** 2)


def build_octree_shell(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    bl_thickness_cells: float = 4.0,
    d_max: int = 2,
    transition: int = 1,
    lattice: str = "D3Q19",
    device: torch.device = torch.device("cpu"),
) -> OctreeGrid:
    """Build the body-fitted octree boundary shell around a sphere.

    Args:
        shape: L1 block physical shape ``(nz, ny, nx)``.
        center: sphere centre in L1 *physical* cell coordinates.
        radius: sphere radius in L1 cell units.
        bl_thickness_cells: shell thickness in cells (near-wall band).
        d_max: maximum leaf depth, 1 or 2 (P1 supports both; depth-2 leaves
            are only created where a depth-1 leaf is wall-adjacent).
        transition: extra cell band appended to the shell mask.
        lattice: ``"D3Q19"`` (only Q=19 is supported in P1).

    Returns:
        :class:`OctreeGrid` with topology, q-field and statistics filled in.
    """
    if lattice != "D3Q19":
        raise NotImplementedError("P1 octree shell supports D3Q19 only")
    if d_max not in (1, 2):
        raise ValueError(f"d_max must be 1 or 2, got {d_max}")
    nz, ny, nx = shape
    for s in (nz, ny, nx):
        if s <= 2:
            raise ValueError(f"L1 block too small for a shell: {shape}")
    if not (radius > 0.0):
        raise ValueError(f"radius must be positive, got {radius}")

    from tensorlbm.d3q19 import C, OPPOSITE

    Q = 19
    c_vec = C.to(device)
    opp = OPPOSITE.to(device)
    k = _axis_bits(shape)

    solid, shell_mask, delta_mask = build_shell_cell_mask(
        shape, center, radius, bl_thickness_cells, transition, device,
    )
    if not bool(shell_mask.any()):
        raise ValueError(
            "no shell cells: enlarge bl_thickness_cells or the L1 block",
        )
    shell_cells = torch.nonzero(shell_mask, as_tuple=False).to(torch.int64)

    # ---- depth-1 leaves ---------------------------------------------------
    l1_coords, l1_centers, l1_inside = _level1_leaves(
        shell_cells, center, radius, device,
    )
    # drop only leaves *fully* inside the sphere (max corner distance <= R);
    # leaves whose box crosses the surface are kept and refined below, so the
    # solid drop happens at depth-2 granularity and no depth-1-scale volume is
    # lost to the wall discretisation.
    c = torch.tensor(center, dtype=torch.float64, device=device)
    d_abs = (l1_centers - c).abs()
    fully_inside = (((d_abs + 0.25) ** 2).sum(dim=1) <= radius ** 2)
    keep1 = ~fully_inside
    l1_coords, l1_centers = l1_coords[keep1], l1_centers[keep1]

    # depth-2 refinement criterion: the depth-1 leaf box crosses the sphere
    # surface (body-fitted wall at depth-2 resolution)
    from tensorlbm.octree_boundary.qfield import compute_q_sphere_at_points

    l1_dx = torch.full((l1_coords.shape[0],), 0.5, dtype=torch.float64)
    mask1, _ = compute_q_sphere_at_points(
        l1_centers, l1_dx, center, radius, device=device, lattice=lattice,
    )
    refine = mask1.any(dim=0) | _aabb_sphere_intersect(
        l1_centers, 0.25, center, radius,
    )

    # ---- depth-2 leaves (wall-adjacent depth-1 leaves only) ---------------
    l2_coords = l2_centers = None
    if d_max >= 2 and bool(refine.any()):
        l2_coords, l2_centers, l2_inside = _level2_leaves(
            l1_coords[refine], center, radius, device,
        )
        keep2 = ~l2_inside
        l2_coords, l2_centers = l2_coords[keep2], l2_centers[keep2]
        l1_coords, l1_centers = l1_coords[~refine], l1_centers[~refine]

    # ---- assemble, sort by (level, morton) --------------------------------
    parts_morton: list[torch.Tensor] = []
    parts_level: list[torch.Tensor] = []
    parts_coords: list[torch.Tensor] = []
    parts_centers: list[torch.Tensor] = []
    n1 = l1_coords.shape[0]
    if n1:
        parts_morton.append(
            morton_encode_batch(torch.full((n1,), 1), l1_coords, k),
        )
        parts_level.append(torch.full((n1,), 1, dtype=torch.int64))
        parts_coords.append(l1_coords)
        parts_centers.append(l1_centers)
    n2 = 0 if l2_coords is None else l2_coords.shape[0]
    if n2:
        parts_morton.append(
            morton_encode_batch(torch.full((n2,), 2), l2_coords, k),
        )
        parts_level.append(torch.full((n2,), 2, dtype=torch.int64))
        parts_coords.append(l2_coords)
        parts_centers.append(l2_centers)

    morton = torch.cat(parts_morton)
    level = torch.cat(parts_level)
    coords = torch.cat(parts_coords)
    centers64 = torch.cat(parts_centers)
    order = torch.argsort(morton, stable=True)
    morton, level, coords, centers64 = (
        morton[order], level[order], coords[order], centers64[order],
    )
    n_leaf = int(morton.shape[0])
    level_start = torch.tensor(
        [0, n1, n_leaf], dtype=torch.int64,
    )
    centers = centers64.to(torch.float32)
    dx = (2.0 ** (-level.to(torch.float32))).unsqueeze(1)     # (n, 1)
    leaf_box = torch.stack(
        [centers - 0.5 * dx, centers + 0.5 * dx], dim=1,
    ).to(torch.float32)
    # host L1 cell in (z, y, x) block-index order
    host_cell = torch.floor(centers64)[:, [2, 1, 0]].to(torch.int64)
    host_cell[:, 0] = host_cell[:, 0].clamp(0, nz - 1)
    host_cell[:, 1] = host_cell[:, 1].clamp(0, ny - 1)
    host_cell[:, 2] = host_cell[:, 2].clamp(0, nx - 1)

    grid = OctreeGrid(
        n_leaf=n_leaf,
        d_max=d_max,
        Q=Q,
        level_start=level_start,
        leaf_morton=morton,
        leaf_level=level,
        leaf_center=centers,
        leaf_box=leaf_box,
        neighbor_table=torch.full((Q, n_leaf), SHELL_OUTSIDE, dtype=torch.int64),
        q_field=torch.full((Q, n_leaf), 0.5, dtype=torch.float32),
        bfl_mask=torch.zeros((Q, n_leaf), dtype=torch.bool),
        interface_links=torch.empty((0, 2), dtype=torch.int64),
        interface_fanout={},
        cross_level_donor=torch.full((Q, n_leaf), -1, dtype=torch.int64),
        leaf_host_cell=host_cell,
        f_leaf=torch.zeros((Q, n_leaf), dtype=torch.float32),
        morton_to_index={int(m): i for i, m in enumerate(morton.tolist())},
        meta={
            "shape": tuple(shape),
            "center": tuple(center),
            "radius": float(radius),
            "bl_thickness_cells": float(bl_thickness_cells),
            "transition": int(transition),
            "delta_mask": float(delta_mask),
            "d_max": int(d_max),
            "lattice": lattice,
            "axis_bits": int(k),
        },
    )
    # stash working sets for the topology builder (leaf-enum order = the
    # sorted order above, so coords must be re-sorted to match the enums)
    coords_sorted = coords[order]
    grid._l1_coords = coords_sorted[:n1].contiguous()
    grid._l2_coords = coords_sorted[n1:].contiguous()
    grid._k = k
    grid._c_vec = c_vec
    grid._opp = opp
    grid._solid = solid
    grid._delta_mask = delta_mask
    grid._shell_mask = shell_mask

    from tensorlbm.octree_boundary.topology import (
        build_interface_registry,
        build_neighbor_table,
        run_topology_checks,
    )
    from tensorlbm.octree_boundary.qfield import compute_leaf_q_field

    build_neighbor_table(grid)
    build_interface_registry(grid)
    compute_leaf_q_field(grid, center, radius)

    # ---- statistics ---------------------------------------------------------
    vol_leaf = grid.total_volume()
    vol_analytic = analytic_shell_volume(radius, delta_mask)
    vol_err = abs(vol_leaf - vol_analytic) / vol_analytic if vol_analytic else 0.0
    saving = cell_saving_report(n_leaf, shell_cells, d_max)
    grid.stats = {
        "n_leaf": n_leaf,
        "n_leaf_l1": int(n1),
        "n_leaf_l2": int(n2),
        "n_shell_cells": int(shell_cells.shape[0]),
        "leaf_volume": float(vol_leaf),
        "analytic_shell_volume": float(vol_analytic),
        "volume_error": float(vol_err),
        "n_interface_links": int(grid.interface_links.shape[0]),
        "n_cross_level_donor": int(
            (grid.cross_level_donor >= 0).sum().item()
        ),
        "n_fanout_groups": len(grid.interface_fanout),
        **saving,
    }
    grid.checks = run_topology_checks(grid)
    return grid
