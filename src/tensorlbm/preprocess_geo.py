"""Pre-processing geometry utilities for TensorLBM.

Provides tools for converting arbitrary geometries into the boolean solid-mask
and Bouzidi *q*-field arrays required by the LBM solvers.

Functions
---------
- :func:`poly_to_mask_2d`        – rasterise any 2-D polygon (ray-casting).
- :func:`voxelize_stl_3d`        – import an STL file and voxelise it via
  z-ray casting (pure NumPy; uses *trimesh* when installed for faster loading).
- :func:`random_porosity_mask_2d` – Gaussian-correlated 2-D random porous mask.
- :func:`random_porosity_mask_3d` – Gaussian-correlated 3-D random porous mask.
- :func:`compute_q_generic_3d`   – Bouzidi *q*-field for any voxelised 3-D
  solid on the D3Q19 lattice (q = 0.5 halfway convention).
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .d3q19 import C as _C3D

__all__ = [
    "poly_to_mask_2d",
    "voxelize_stl_3d",
    "random_porosity_mask_2d",
    "random_porosity_mask_3d",
    "compute_q_generic_3d",
    "poly_to_mask_and_q_2d",
]

# ---------------------------------------------------------------------------
# Internal STL helpers (pure NumPy)
# ---------------------------------------------------------------------------


def _parse_stl_binary(data: bytes, n_tri: int) -> np.ndarray:
    """Parse binary STL payload, return (n_tri, 3, 3) float32 vertex array."""
    dt = np.dtype(
        [
            ("normal", np.float32, (3,)),
            ("v0", np.float32, (3,)),
            ("v1", np.float32, (3,)),
            ("v2", np.float32, (3,)),
            ("attr", np.uint16),
        ]
    )
    records = np.frombuffer(data, dtype=dt, count=n_tri, offset=84)
    return np.stack([records["v0"], records["v1"], records["v2"]], axis=1).astype(
        np.float32
    )


def _parse_stl_ascii(data: bytes) -> np.ndarray:
    """Parse ASCII STL payload, return (n_tri, 3, 3) float32 vertex array."""
    text = data.decode("utf-8", errors="replace")
    verts: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("vertex"):
            parts = stripped.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    arr = np.array(verts, dtype=np.float32)
    n_tri = len(arr) // 3
    return arr[: n_tri * 3].reshape(n_tri, 3, 3)


def _parse_stl(path: Path) -> np.ndarray:
    """Parse STL (binary or ASCII), return (n_tri, 3, 3) float32 vertex array."""
    data = path.read_bytes()
    # Detect binary by matching the expected file size: 84 + 50 * n_tri
    if len(data) >= 84:
        n_tri_candidate = int(np.frombuffer(data[80:84], dtype=np.uint32)[0])
        if len(data) == 84 + 50 * n_tri_candidate:
            return _parse_stl_binary(data, n_tri_candidate)
    return _parse_stl_ascii(data)


def _voxelize_triangles(  # noqa: PLR0912
    triangles: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    x_min: float,
    y_min: float,
    z_min: float,
    x_max: float,
    y_max: float,
    z_max: float,
) -> np.ndarray:
    """Voxelise a triangle mesh via z-ray casting.

    For every (ix, iy) column a ray is cast in the +z direction. The column
    cells that have an *odd* number of triangle intersections below their
    centre are marked as solid (inside the closed surface).

    Args:
        triangles: Vertex array of shape ``(N, 3, 3)`` — N triangles, each
            row is ``[v0, v1, v2]`` with ``v_i = (x, y, z)``.
        nx, ny, nz: Grid dimensions.
        x_min, y_min, z_min: Lower corner of the bounding box.
        x_max, y_max, z_max: Upper corner of the bounding box.

    Returns:
        Boolean array of shape ``(nz, ny, nx)`` — True where solid.
    """
    dx = (x_max - x_min) / nx
    dy = (y_max - y_min) / ny
    dz = (z_max - z_min) / nz

    # col_z[iy * nx + ix] accumulates z-intersection values for that column
    col_z: list[list[float]] = [[] for _ in range(ny * nx)]

    for tri in triangles:
        v0, v1, v2 = tri[0], tri[1], tri[2]

        edge1 = v1 - v0
        edge2 = v2 - v0
        normal = np.cross(edge1, edge2).astype(np.float64)
        nz_comp = float(normal[2])

        # Skip triangles whose plane is (nearly) parallel to the z-axis
        if abs(nz_comp) < 1e-12:
            continue

        # Bounding box in column-index space
        xs = np.array([v0[0], v1[0], v2[0]], dtype=np.float64)
        ys = np.array([v0[1], v1[1], v2[1]], dtype=np.float64)

        ix_lo = max(0, int(math.floor((xs.min() - x_min) / dx)))
        ix_hi = min(nx - 1, int(math.ceil((xs.max() - x_min) / dx)))
        iy_lo = max(0, int(math.floor((ys.min() - y_min) / dy)))
        iy_hi = min(ny - 1, int(math.ceil((ys.max() - y_min) / dy)))

        if ix_lo > ix_hi or iy_lo > iy_hi:
            continue

        ix_arr = np.arange(ix_lo, ix_hi + 1)
        iy_arr = np.arange(iy_lo, iy_hi + 1)
        IX, IY = np.meshgrid(ix_arr, iy_arr)  # each (iy_count, ix_count)
        OX = x_min + (IX + 0.5) * dx
        OY = y_min + (IY + 0.5) * dy

        # 2-D point-in-triangle test (signed-area method, vectorised)
        p0 = v0[:2].astype(np.float64)
        p1 = v1[:2].astype(np.float64)
        p2 = v2[:2].astype(np.float64)

        d0 = (p1[0] - p0[0]) * (OY - p0[1]) - (p1[1] - p0[1]) * (OX - p0[0])
        d1 = (p2[0] - p1[0]) * (OY - p1[1]) - (p2[1] - p1[1]) * (OX - p1[0])
        d2 = (p0[0] - p2[0]) * (OY - p2[1]) - (p0[1] - p2[1]) * (OX - p2[0])

        has_neg = (d0 < 0) | (d1 < 0) | (d2 < 0)
        has_pos = (d0 > 0) | (d1 > 0) | (d2 > 0)
        inside = ~(has_neg & has_pos)  # True where column centre is inside 2-D projection

        if not inside.any():
            continue

        # z-intersection: plane equation n·(P - v0) = 0 with P = (ox, oy, t)
        z_isect = v0[2] - (
            normal[0] * (OX - float(v0[0])) + normal[1] * (OY - float(v0[1]))
        ) / nz_comp  # (iy_count, ix_count)

        # Accumulate per-column
        valid_iy = IY[inside].astype(int)
        valid_ix = IX[inside].astype(int)
        valid_z = z_isect[inside]

        for k in range(len(valid_ix)):
            col_z[int(valid_iy[k]) * nx + int(valid_ix[k])].append(float(valid_z[k]))

    # Build solid mask from sorted intersection lists
    solid = np.zeros((nz, ny, nx), dtype=bool)
    z_centers = z_min + (np.arange(nz, dtype=np.float64) + 0.5) * dz

    for col_idx, z_list in enumerate(col_z):
        if not z_list:
            continue
        z_sorted = np.sort(np.asarray(z_list, dtype=np.float64))
        iy, ix = divmod(col_idx, nx)
        # For each cell centre count intersections below it (parity → inside/outside)
        counts = np.searchsorted(z_sorted, z_centers)
        solid[:, iy, ix] = counts % 2 == 1

    return solid


def poly_to_mask_and_q_2d(
    vertices: list[tuple[float, float]],
    ny: int,
    nx: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rasterize a polygon and compute D2Q9 Bouzidi ``q`` factors.

    Args:
        vertices: Ordered polygon vertices in lattice units.
        ny: Number of y-cells.
        nx: Number of x-cells.
        device: Target device.

    Returns:
        Tuple ``(mask, q_field)`` with shapes ``(ny, nx)`` and ``(9, ny, nx)``.
    """
    from .d2q9 import C as _C2D

    mask = poly_to_mask_2d(vertices, ny=ny, nx=nx, device=device)
    q_field = np.full((9, ny, nx), 0.5, dtype=np.float32)
    verts = np.asarray(vertices, dtype=np.float64)
    dirs = _C2D.cpu().numpy()
    mask_np = mask.cpu().numpy()

    for iy in range(ny):
        for ix in range(nx):
            if mask_np[iy, ix]:
                continue
            px = ix + 0.5
            py = iy + 0.5
            for q, (cx, cy) in enumerate(dirs):
                if cx == 0 and cy == 0:
                    continue
                nx_nb = ix + int(cx)
                ny_nb = iy + int(cy)
                if not (0 <= nx_nb < nx and 0 <= ny_nb < ny and mask_np[ny_nb, nx_nb]):
                    continue

                t_min = np.inf
                for i in range(len(verts)):
                    x0, y0 = verts[i]
                    x1, y1 = verts[(i + 1) % len(verts)]
                    ex = x1 - x0
                    ey = y1 - y0
                    denom = cx * ey - cy * ex
                    if abs(denom) < 1e-12:
                        continue
                    dx = x0 - px
                    dy = y0 - py
                    t = (dx * ey - dy * ex) / denom
                    u = (dx * cy - dy * cx) / denom
                    if 0.0 < t <= 1.0 and 0.0 <= u <= 1.0:
                        t_min = min(t_min, t)
                if np.isfinite(t_min):
                    q_field[q, iy, ix] = float(np.clip(t_min, 0.0, 1.0))

    return mask, torch.from_numpy(q_field).to(device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def poly_to_mask_2d(
    vertices: list[tuple[float, float]],
    ny: int,
    nx: int,
    device: torch.device,
) -> torch.Tensor:
    """Boolean solid mask for an arbitrary 2-D polygon.

    Uses the *ray-casting* (even–odd rule) algorithm: for each grid-cell
    centre a horizontal ray is cast in the +x direction and the number of
    polygon edges crossed is counted.  An odd count means the cell is inside
    the solid.

    The polygon is interpreted in the same coordinate frame as the grid:
    vertex coordinates are in lattice units, with (0, 0) at the lower-left
    corner of cell (ix=0, iy=0).  Cell centres lie at ``(ix + 0.5, iy + 0.5)``.

    Args:
        vertices: Ordered list of ``(x, y)`` vertex coordinates in lattice
            units.  The polygon is automatically closed (last vertex connects
            to first).
        ny: Grid height (number of cells in y).
        nx: Grid width (number of cells in x).
        device: Target PyTorch device.

    Returns:
        Boolean tensor of shape ``(ny, nx)`` — True where solid.
    """
    verts = np.array(vertices, dtype=np.float64)
    n = len(verts)

    # Cell centres at (ix + 0.5, iy + 0.5)
    iy_idx, ix_idx = np.mgrid[0:ny, 0:nx]
    px = ix_idx.astype(np.float64) + 0.5  # (ny, nx)
    py = iy_idx.astype(np.float64) + 0.5

    inside = np.zeros((ny, nx), dtype=bool)

    for i in range(n):
        x0, y0 = float(verts[i, 0]), float(verts[i, 1])
        x1, y1 = float(verts[(i + 1) % n, 0]), float(verts[(i + 1) % n, 1])
        edge_dy = y1 - y0

        if abs(edge_dy) < 1e-12:
            continue  # horizontal edge — skip

        # Parity condition: edge straddles py, intersection is to the right of px
        cond_y = (np.minimum(y0, y1) <= py) & (py < np.maximum(y0, y1))
        x_isect = x0 + (py - y0) / edge_dy * (x1 - x0)
        inside ^= cond_y & (x_isect > px)

    return torch.from_numpy(inside).to(device)


def voxelize_stl_3d(
    stl_path: str | Path,
    nx: int,
    ny: int,
    nz: int,
    device: torch.device,
    padding: float = 0.05,
) -> torch.Tensor:
    """Import an STL file and voxelise it into a 3-D boolean solid mask.

    A grid of ``nz × ny × nx`` cells is constructed around the mesh's
    axis-aligned bounding box (AABB), extended by *padding* on every side.
    Each cell whose centre is inside the closed triangular surface is marked
    as solid via z-ray casting.

    The function uses a pure-NumPy fallback that works without any additional
    dependencies.  If *trimesh* is installed it is used for more robust STL
    loading (supports ASCII and non-standard binary headers); the actual
    voxelisation is always performed by the pure-NumPy ray caster.

    Args:
        stl_path: Path to the STL file (binary or ASCII).
        nx: Number of voxels along the x-axis.
        ny: Number of voxels along the y-axis.
        nz: Number of voxels along the z-axis.
        device: Target PyTorch device.
        padding: Fractional padding applied to each side of the mesh AABB.
            E.g. ``0.05`` extends the grid by 5 % of the AABB extent on
            every side, ensuring the mesh is fully contained.

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)`` — True where solid.

    Raises:
        FileNotFoundError: If *stl_path* does not exist.
        ValueError: If the STL file contains no valid triangles.
    """
    path = Path(stl_path)
    if not path.exists():
        raise FileNotFoundError(f"STL file not found: {path}")

    # Load triangles (try trimesh first for robustness, fall back to own parser)
    triangles: np.ndarray
    try:
        import trimesh  # type: ignore[import-untyped]

        mesh = trimesh.load(str(path), force="mesh")
        triangles = np.array(mesh.triangles, dtype=np.float32)
    except ImportError:
        triangles = _parse_stl(path)

    if triangles.shape[0] == 0:
        raise ValueError(f"No triangles found in STL file: {path}")

    # Compute AABB with padding
    all_verts = triangles.reshape(-1, 3).astype(np.float64)
    lo = all_verts.min(axis=0)
    hi = all_verts.max(axis=0)
    span = hi - lo
    span = np.where(span < 1e-12, 1.0, span)  # avoid degenerate dimensions

    x_min = lo[0] - padding * span[0]
    y_min = lo[1] - padding * span[1]
    z_min = lo[2] - padding * span[2]
    x_max = hi[0] + padding * span[0]
    y_max = hi[1] + padding * span[1]
    z_max = hi[2] + padding * span[2]

    solid_np = _voxelize_triangles(
        triangles.astype(np.float64), nx, ny, nz, x_min, y_min, z_min, x_max, y_max, z_max
    )
    return torch.from_numpy(solid_np).to(device)


def random_porosity_mask_2d(
    ny: int,
    nx: int,
    porosity: float,
    device: torch.device,
    seed: int = 0,
    sigma: float = 0.0,
) -> torch.Tensor:
    """Random 2-D solid mask with prescribed porosity.

    Generates a Gaussian-correlated random field, then thresholds it so that
    the solid fraction equals ``1 − porosity``.  When *sigma* = 0 the field
    is uncorrelated (i.i.d. Bernoulli with probability ``1 − porosity``).

    Args:
        ny: Grid height.
        nx: Grid width.
        porosity: Void (fluid) fraction, in ``(0, 1)``.  A value of 0.4
            means 40 % of cells are fluid and 60 % are solid.
        device: Target PyTorch device.
        seed: Random seed for reproducibility.
        sigma: Standard deviation of the Gaussian smoothing kernel in cells.
            Larger values produce larger, more connected pore structures.
            Zero means no smoothing (uncorrelated mask).

    Returns:
        Boolean tensor of shape ``(ny, nx)`` — True where solid.
    """
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {porosity}")

    gen = torch.Generator()
    gen.manual_seed(seed)
    field = torch.randn(1, 1, ny, nx, generator=gen)

    if sigma > 0.0:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        k = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (k / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # (2r+1, 2r+1)
        kernel_2d = kernel_2d.view(1, 1, 2 * radius + 1, 2 * radius + 1)
        field = F.conv2d(field, kernel_2d, padding=radius)

    # Threshold: cells above the porosity-th quantile are solid
    threshold = torch.quantile(field.view(-1), porosity)
    return (field.squeeze(0).squeeze(0) > threshold).to(device)


def random_porosity_mask_3d(
    nz: int,
    ny: int,
    nx: int,
    porosity: float,
    device: torch.device,
    seed: int = 0,
    sigma: float = 0.0,
) -> torch.Tensor:
    """Random 3-D solid mask with prescribed porosity.

    3-D analogue of :func:`random_porosity_mask_2d`.  A separable Gaussian
    filter is applied in all three spatial directions.

    Args:
        nz: Grid depth.
        ny: Grid height.
        nx: Grid width.
        porosity: Void (fluid) fraction, in ``(0, 1)``.
        device: Target PyTorch device.
        seed: Random seed for reproducibility.
        sigma: Gaussian correlation length in cells (0 = uncorrelated).

    Returns:
        Boolean tensor of shape ``(nz, ny, nx)`` — True where solid.
    """
    if not 0.0 < porosity < 1.0:
        raise ValueError(f"porosity must be in (0, 1), got {porosity}")

    gen = torch.Generator()
    gen.manual_seed(seed)
    field = torch.randn(1, 1, nz, ny, nx, generator=gen)

    if sigma > 0.0:
        radius = max(1, int(math.ceil(3.0 * sigma)))
        k = torch.arange(-radius, radius + 1, dtype=torch.float32)
        kernel_1d = torch.exp(-0.5 * (k / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        ksize = 2 * radius + 1

        # Apply separable convolution in z, y, x independently
        kz = kernel_1d.view(1, 1, ksize, 1, 1)
        ky = kernel_1d.view(1, 1, 1, ksize, 1)
        kx = kernel_1d.view(1, 1, 1, 1, ksize)

        field = F.conv3d(field, kz, padding=(radius, 0, 0))
        field = F.conv3d(field, ky, padding=(0, radius, 0))
        field = F.conv3d(field, kx, padding=(0, 0, radius))

    threshold = torch.quantile(field.view(-1), porosity)
    return (field.squeeze(0).squeeze(0) > threshold).to(device)


def compute_q_generic_3d(
    obstacle_mask: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bouzidi *q*-field for an arbitrary voxelised 3-D solid (D3Q19).

    Identifies all **fluid** nodes that are direct D3Q19 lattice neighbours of
    a solid node, and returns a *q*-field initialised to 0.5 (halfway
    bounce-back).  This function serves as the generic counterpart to
    :func:`~tensorlbm.interpolated_bc.compute_q_sphere`: it works for any
    Boolean obstacle mask produced by, e.g., :func:`voxelize_stl_3d`.

    The returned ``q_field`` may be refined in-place by the caller if a more
    accurate surface representation is available.

    Args:
        obstacle_mask: Boolean tensor of shape ``(nz, ny, nx)`` — True where
            solid.
        device: Target PyTorch device.

    Returns:
        Tuple ``(fluid_boundary_mask, q_field)`` where

        - ``fluid_boundary_mask`` is a bool tensor of shape ``(19, nz, ny, nx)``
          — True at ``[d, k, j, i]`` when fluid node ``(i, j, k)`` has a
          solid D3Q19 neighbour in direction *d*.
        - ``q_field`` is a float tensor of shape ``(19, nz, ny, nx)`` with
          *q* = 0.5 at every boundary entry and 0.5 elsewhere (matching the
          interface of :func:`~tensorlbm.interpolated_bc.compute_q_sphere`).
    """
    obstacle_mask = obstacle_mask.to(device)
    nz, ny, nx = obstacle_mask.shape
    c = _C3D.to(device)  # (19, 3): integer lattice velocities

    fluid_boundary_mask = torch.zeros((19, nz, ny, nx), dtype=torch.bool, device=device)
    q_field = torch.full((19, nz, ny, nx), 0.5, dtype=torch.float32, device=device)

    fluid_mask = ~obstacle_mask  # True where fluid

    for d in range(19):
        dcx = int(c[d, 0].item())
        dcy = int(c[d, 1].item())
        dcz = int(c[d, 2].item())

        if dcx == 0 and dcy == 0 and dcz == 0:
            continue  # rest direction — no boundary

        # Neighbour of each cell shifted by (dcx, dcy, dcz)
        # torch.roll wraps at the boundary, but for periodic-free domains we
        # only keep neighbours that are still inside the grid range.
        nb_solid = torch.roll(obstacle_mask, shifts=(-dcz, -dcy, -dcx), dims=(0, 1, 2))

        # Fluid node whose neighbour in direction d is solid
        boundary = fluid_mask & nb_solid
        fluid_boundary_mask[d] = boundary

    return fluid_boundary_mask, q_field


# ---------------------------------------------------------------------------
# Deprecation shim – warn if user imports the old private helpers
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> object:
    """Warn on access to removed private helpers."""
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Expose a note about torch.roll wrap-around for users
_ROLL_NOTE = (
    "compute_q_generic_3d uses torch.roll which wraps at domain boundaries. "
    "For non-periodic domains, boundary cells at the domain edges may have "
    "incorrect fluid_boundary_mask values. Mask those columns if needed."
)


def _check_boundary_warning(obstacle_mask: torch.Tensor) -> None:
    """Warn if obstacle touches domain boundary (roll artefact)."""
    nz, ny, nx = obstacle_mask.shape
    edge = (
        obstacle_mask[0].any()
        or obstacle_mask[-1].any()
        or obstacle_mask[:, 0].any()
        or obstacle_mask[:, -1].any()
        or obstacle_mask[:, :, 0].any()
        or obstacle_mask[:, :, -1].any()
    )
    if edge:
        warnings.warn(
            "Obstacle mask touches the domain boundary. "
            "compute_q_generic_3d uses torch.roll; fluid_boundary_mask entries "
            "at the domain edges may be inaccurate. "
            "Add ghost/padding cells around the obstacle if this matters.",
            stacklevel=3,
        )
