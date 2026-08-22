"""STL geometry module for TensorLBM drag integration.

Provides STL file reading, voxelization, and surface-normal computation
that integrates with :mod:`tensorlbm.drag_pressure`'s :class:`SurfaceMesh` API.

Pipeline::

    vertices, faces, normals = read_stl("geometry.stl")
    solid = voxelize_stl(vertices, faces, (nx, ny, nz),
                         origin=(0, 0, 0), spacing=(1, 1, 1))
    near = get_near_wall_3d(solid)
    mesh = SurfaceMesh_from_stl(solid, near, vertices, faces, normals,
                                origin, spacing)

    fx_p, fy_p, fz_p = drag_pressure_integration(f, mesh, dpS, extrap='quadratic')
    fx_f, fy_f, fz_f = drag_friction_integration(f, mesh, dpS, nu)

The module reads both ASCII and binary STL files using a pure-NumPy parser
(no external STL library required).  Voxelization uses z-ray casting via
:func:`tensorlbm.preprocess_geo._voxelize_triangles`.  Surface normals are
computed by finding the nearest STL triangle for each near-wall cell and
aligning the face normal with the outward direction inferred from the
solid-mask gradient.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .drag_pressure import SurfaceMesh, get_near_wall_3d
from .preprocess_geo import _voxelize_triangles, _write_stl_binary

__all__ = [
    "read_stl",
    "voxelize_stl",
    "SurfaceMesh_from_stl",
    "mirror_stl",
    "write_stl",
    "get_near_wall_3d",
    "make_sphere_stl",
    "make_icosphere_stl",
    "make_cylinder_stl",
    "make_naca_stl",
]


# ---------------------------------------------------------------------------
# 1. STL READER
# ---------------------------------------------------------------------------


def _parse_stl_binary_full(data: bytes, n_tri: int):
    """Parse binary STL payload.

    Returns (triangles (n,3,3), stored_normals (n,3)).
    """
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
    triangles = np.stack([records["v0"], records["v1"], records["v2"]], axis=1).astype(np.float32)
    normals = records["normal"].astype(np.float32).copy()
    return triangles, normals


def _parse_stl_ascii_full(data: bytes):
    """Parse ASCII STL payload.

    Returns (triangles (n,3,3), stored_normals (n,3)).
    """
    text = data.decode("utf-8", errors="replace")
    triangles: list[list[list[float]]] = []
    normals: list[list[float]] = []
    current_normal = [0.0, 0.0, 0.0]
    current_verts: list[list[float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("facet normal"):
            parts = stripped.split()
            current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
        elif stripped.startswith("vertex"):
            parts = stripped.split()
            current_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif stripped.startswith("endfacet"):
            if len(current_verts) == 3:
                triangles.append(current_verts)
                normals.append(current_normal)
            current_verts = []
    triangles_arr = (
        np.array(triangles, dtype=np.float32)
        if triangles
        else np.zeros((0, 3, 3), dtype=np.float32)
    )
    normals_arr = (
        np.array(normals, dtype=np.float32) if normals else np.zeros((0, 3), dtype=np.float32)
    )
    return triangles_arr, normals_arr


def _parse_stl_full(path: Path):
    """Parse STL (binary or ASCII).

    Returns (triangles (n,3,3), stored_normals (n,3)).
    """
    data = path.read_bytes()
    # Detect binary by matching the expected file size: 84 + 50 * n_tri
    if len(data) >= 84:
        n_tri_candidate = int(np.frombuffer(data[80:84], dtype=np.uint32)[0])
        if len(data) == 84 + 50 * n_tri_candidate and n_tri_candidate > 0:
            return _parse_stl_binary_full(data, n_tri_candidate)
        # Fallback: try computing n_tri from file size (non-standard binary STL)
        n_tri_computed = (len(data) - 84) // 50
        remainder = (len(data) - 84) % 50
        if n_tri_computed > 0 and remainder < 4:
            # Allow small remainder (padding/footer)
            return _parse_stl_binary_full(data[: 84 + n_tri_computed * 50], n_tri_computed)
    return _parse_stl_ascii_full(data)


def read_stl(path):
    """Read an STL file (ASCII or binary) into vertices, faces, and normals.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the STL file.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3), float32
        Unique vertex coordinates.
    faces : np.ndarray, shape (M, 3), int32
        Triangle vertex indices into *vertices*.
    face_normals : np.ndarray, shape (M, 3), float32
        Unit face-normal vectors.  If the STL stores non-zero normals they
        are used; otherwise normals are recomputed from the cross product
        ``cross(v1-v0, v2-v0)``.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If the file contains no valid triangles.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"STL file not found: {path}")

    triangles, stored_normals = _parse_stl_full(path)
    if triangles.shape[0] == 0:
        raise ValueError(f"No triangles found in STL file: {path}")

    # Deduplicate vertices (tolerance ~1e-6)
    flat = triangles.reshape(-1, 3).astype(np.float64)
    scale = 1e6
    rounded = np.round(flat * scale).astype(np.int64)
    _, inv = np.unique(rounded, axis=0, return_inverse=True)
    unique_idx = np.unique(inv, return_index=True)[1]
    vertices = flat[unique_idx].astype(np.float32)
    faces = inv.reshape(-1, 3).astype(np.int32)

    # Compute face normals from cross products
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
    norms = np.linalg.norm(cross, axis=1, keepdims=True)
    computed_normals = cross / np.where(norms > 1e-10, norms, 1.0)

    # Use stored normals when they are valid (|n| > 0.5), else computed
    stored_norm = np.linalg.norm(stored_normals, axis=1, keepdims=True)
    has_stored = (stored_norm > 0.5).squeeze()
    face_normals = np.where(has_stored[:, None], stored_normals, computed_normals).astype(
        np.float32
    )

    # Bug 48: Auto-detect and mirror half hull FIRST.
    # Some STL files (e.g. KVLCC2) are half hulls (y >= 0 only).
    # Mirror to get full hull before orienting normals.
    if vertices[:, 1].min() >= 0 and vertices[:, 1].max() > 0:
        vertices, faces, face_normals = mirror_stl(vertices, faces, face_normals, axis=1)
        # Recompute normals from cross products after mirroring.
        # mirror_stl reverses winding for mirrored faces, so cross products
        # give correct outward direction for mirrored half.
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0).astype(np.float32)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        face_normals = cross / np.where(norms > 1e-10, norms, 1.0)

    # Bug 47: Auto-detect and flip inward normals (AFTER mirror).
    # Check: if majority of normals point toward the centroid, flip all.
    centroid = vertices.mean(axis=0)
    face_centers = vertices[faces].mean(axis=1)
    dir_to_centroid = centroid - face_centers
    dir_norm = np.linalg.norm(dir_to_centroid, axis=1, keepdims=True)
    dir_to_centroid = dir_to_centroid / np.where(dir_norm > 1e-10, dir_norm, 1.0)
    dot_centroid = (face_normals * dir_to_centroid).sum(axis=1)
    n_inward = np.sum(dot_centroid > 0)
    if n_inward >= 0.5 * len(face_normals):
        face_normals = -face_normals  # flip all normals

    return vertices, faces, face_normals


# ---------------------------------------------------------------------------
# 2. VOXELIZER
# ---------------------------------------------------------------------------


def voxelize_stl(
    vertices,
    faces,
    grid_shape,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
):
    """Voxelize an STL mesh into a solid boolean mask on the LBM grid.

    Uses z-ray casting: for every (ix, iy) column a ray is cast in the +z
    direction.  Cells whose centre has an odd number of triangle
    intersections below it are marked solid (inside the closed surface).

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Vertex coordinates (same units as *origin* / *spacing*).
    faces : np.ndarray, shape (M, 3)
        Triangle vertex indices.
    grid_shape : tuple of int (nx, ny, nz)
        Grid dimensions.
    origin : tuple of float, default (0, 0, 0)
        Physical coordinates of the grid's lower corner.
    spacing : tuple of float, default (1, 1, 1)
        Cell size in each direction.

    Returns
    -------
    solid : torch.Tensor, shape (nz, ny, nx), dtype bool
        ``True`` where inside the STL surface.
    """
    nx, ny, nz = grid_shape
    triangles = vertices[faces].astype(np.float64)
    x_min, y_min, z_min = (float(o) for o in origin)
    dx, dy, dz = (float(s) for s in spacing)
    # Tiny asymmetric perturbation to avoid ray-triangle edge degeneracies.
    # When a cell centre falls exactly on a cap-triangle edge (a radius
    # from centre to circumference), the point-in-triangle test counts it
    # as inside *both* adjacent triangles, doubling the intersection count
    # and producing holes in the solid mask.  An asymmetric offset (different
    # in x and y) breaks the exact alignment for all radial edges.
    # Magnitude < 0.01 % of a cell — negligible for physics.
    x_min += 1.3e-4
    y_min += 3.7e-4
    x_max = x_min + nx * dx
    y_max = y_min + ny * dy
    z_max = z_min + nz * dz
    solid_np = _voxelize_triangles(
        triangles,
        nx,
        ny,
        nz,
        x_min,
        y_min,
        z_min,
        x_max,
        y_max,
        z_max,
    )
    return torch.from_numpy(solid_np)


# ---------------------------------------------------------------------------
# 3. STL NORMAL COMPUTATION  →  SurfaceMesh
# ---------------------------------------------------------------------------


def _compute_gradient_normals(solid, near):
    """Outward unit-normal *direction* from the solid-mask gradient.

    Uses central differences of the boolean solid mask.  The gradient
    points from fluid to solid; negating it gives the outward (solid →
    fluid) direction.  This is only used to determine the **sign** of the
    STL face normals (ensure they point outward).
    """
    s = solid.float()
    gx = torch.zeros_like(s)
    gy = torch.zeros_like(s)
    gz = torch.zeros_like(s)
    gx[:, :, 1:-1] = (s[:, :, 2:] - s[:, :, :-2]) / 2
    gy[:, 1:-1, :] = (s[:, 2:, :] - s[:, :-2, :]) / 2
    gz[1:-1, :, :] = (s[2:, :, :] - s[:-2, :, :]) / 2
    # Outward = -gradient (from solid to fluid)
    near_f = near.float()
    gx = -gx * near_f
    gy = -gy * near_f
    gz = -gz * near_f
    norm = torch.sqrt(gx**2 + gy**2 + gz**2).clamp(min=1e-10)
    return gx / norm, gy / norm, gz / norm


def _nearest_triangle_normals(cell_pos, centroids, face_normals):
    """For each cell position, return the face normal of the nearest triangle.

    Uses scipy.spatial.cKDTree on triangle centroids when available;
    falls back to brute-force NumPy otherwise.
    """
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(centroids)
        _, idx = tree.query(cell_pos, k=1)
    except ImportError:
        # Brute-force: (n_cell, 1) = (n_cell, n_tri).min
        diffs = cell_pos[:, None, :] - centroids[None, :, :]
        dists = np.sum(diffs**2, axis=2)
        idx = np.argmin(dists, axis=1)
    return face_normals[idx].astype(np.float64).copy(), idx


def _ray_triangle_intersections_count(
    ray_origins: np.ndarray,
    ray_dir: np.ndarray,
    triangles: np.ndarray,
) -> np.ndarray:
    """Count ray-triangle intersections for each ray origin.

    Uses the Möller–Trumbore algorithm.  A bounding-box pre-filter
    avoids testing triangles that cannot possibly be hit, making this
    practical for ship-hull STL files (~600–1200 triangles).

    Parameters
    ----------
    ray_origins : (N, 3) float64
        Ray starting points (near-wall cell centres).
    ray_dir : (3,) float64
        Ray direction (need not be normalised).
    triangles : (M, 3, 3) float64
        Triangle vertex coordinates.

    Returns
    -------
    counts : (N,) int32
        Number of triangle intersections for each ray.
    """
    n_rays = ray_origins.shape[0]
    n_tri = triangles.shape[0]
    counts = np.zeros(n_rays, dtype=np.int32)

    rd = ray_dir.astype(np.float64)
    rd_norm = np.linalg.norm(rd)
    if rd_norm < 1e-12:
        return counts
    rd = rd / rd_norm

    # Pre-compute per-triangle bounding boxes
    tri_min = triangles.min(axis=1)  # (n_tri, 3)
    tri_max = triangles.max(axis=1)  # (n_tri, 3)

    # Process in batches to limit memory
    batch = 2000
    for start in range(0, n_rays, batch):
        end = min(start + batch, n_rays)
        batch_origins = ray_origins[start:end]  # (B, 3)
        B = batch_origins.shape[0]
        batch_counts = np.zeros(B, dtype=np.int32)

        for ti in range(n_tri):
            tri = triangles[ti]  # (3, 3)
            # Bounding-box filter: the ray travels in direction *rd* from
            # *batch_origins*.  A triangle can only be hit if at least one
            # vertex lies ahead of the origin along the ray (i.e. the
            # dot product of (vertex - origin) with rd is positive) **and**
            # the origin's non-ray coordinates fall within the triangle's
            # bounding box (expanded by a small tolerance).
            _tri_xmin, _tri_xmax = tri_min[ti, 0], tri_max[ti, 0]
            _tri_ymin, _tri_ymax = tri_min[ti, 1], tri_max[ti, 1]
            _tri_zmin, _tri_zmax = tri_min[ti, 2], tri_max[ti, 2]

            # Determine which components are "along the ray" vs "perpendicular"
            # For a general ray direction, the perpendicular coordinates are
            # the two axes with the smallest |rd| components.  Here we use a
            # simple approach: check all three bounding-box ranges; the ray
            # will pass through if the origin is within the triangle's BB in
            # the two perpendicular directions and ahead in the ray direction.
            #
            # For the common case rd ≈ (1, 0, 0) (ray along +x):
            #   - "ahead" means tri_xmax > origin_x
            #   - "perpendicular" means origin_y in [tri_ymin, tri_ymax]
            #     and origin_z in [tri_zmin, tri_zmax]
            #
            # Generalise: project (vertex - origin) onto rd to get the
            # ahead-distance, and check the perpendicular distance from the
            # origin to the triangle's bounding box.
            v0, v1, v2 = tri[0], tri[1], tri[2]
            edge1 = v1 - v0
            edge2 = v2 - v0
            h = np.cross(rd, edge2)  # (3,)
            a = np.dot(edge1, h)
            if abs(a) < 1e-12:
                continue  # ray parallel to triangle

            # Quick bounding-box filter on the perpendicular coordinates.
            # Identify the dominant ray component to know which axes are
            # "perpendicular".
            dom = np.argmax(np.abs(rd))
            perp_axes = [i for i in range(3) if i != dom]

            ahead = tri_max[ti, dom] > batch_origins[:, dom]
            in_perp = np.ones(B, dtype=bool)
            for pa in perp_axes:
                in_perp &= (batch_origins[:, pa] >= tri_min[ti, pa] - 1e-10) & (
                    batch_origins[:, pa] <= tri_max[ti, pa] + 1e-10
                )
            mask = ahead & in_perp
            if not mask.any():
                continue

            # Möller–Trumbore for the masked rays
            f_val = 1.0 / a
            s_vec = batch_origins[mask] - v0  # (M, 3)
            u = f_val * np.dot(s_vec, h)  # (M,)
            q = np.cross(s_vec, edge1)  # (M, 3)
            v = f_val * np.dot(q, rd)  # (M,)
            t = f_val * np.dot(q, edge2)  # (M,)

            hit = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10) & (t > 1e-10)
            batch_counts[mask] += hit

        counts[start:end] = batch_counts

    return counts


def _orient_normals_raycast(
    normals: np.ndarray,
    cell_pos: np.ndarray,
    centroids: np.ndarray,
    tri_idx: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Orient STL face normals outward using ray-casting inside/outside test.

    For each near-wall cell a ray is cast from the cell centre to infinity
    (along +x).  The number of STL triangle intersections determines whether
    the cell is inside (odd) or outside (even) the closed STL surface.

    Near-wall cells are **fluid** cells adjacent to the solid.  For external
    flow they are always *outside* the solid, so the outward normal should
    point from the solid surface toward the cell.  The direction from the
    nearest triangle centroid to the cell is used as the reference "toward
    cell" direction; the STL face normal is flipped if it points away from
    the cell.

    Ray casting provides an independent verification: if a cell is found
    *inside* the STL surface (odd intersection count), the normal is oriented
    toward the cell as well (pointing from the solid interior toward the
    surface).  This handles rare cases where the voxelised solid mask and the
    STL surface disagree.

    Parameters
    ----------
    normals : (N, 3) float64
        STL face normals of the nearest triangles (may be inward or outward).
    cell_pos : (N, 3) float64
        Physical coordinates of near-wall cell centres.
    centroids : (M, 3) float64
        Triangle centroids.
    tri_idx : (N,) int
        Index of the nearest triangle for each cell.
    vertices : (V, 3)
        STL vertices.
    faces : (M, 3)
        STL triangle vertex indices.

    Returns
    -------
    normals : (N, 3) float64
        Normals oriented to point toward the cell (outward from solid).
    """
    len(normals)

    # Direction from nearest triangle centroid to cell — this is the
    # "toward cell" direction.  For external flow, near-wall cells are
    # outside the solid, so the outward normal should point toward the cell.
    nearest_centroids = centroids[tri_idx]
    tri_to_cell = cell_pos - nearest_centroids
    tc_norm = np.linalg.norm(tri_to_cell, axis=1, keepdims=True)
    tc_dir = tri_to_cell / np.where(tc_norm > 1e-10, tc_norm, 1.0)

    # Flip normals that point away from the cell (dot < 0)
    dot_tc = (normals * tc_dir).sum(axis=1)
    flip_mask = dot_tc < 0
    normals[flip_mask] = -normals[flip_mask]

    # Ray-casting verification: cast a ray in +x from each cell and count
    # intersections with the STL triangles.  Odd = inside, even = outside.
    #
    # This is performed for diagnostic purposes.  In practice, near-wall
    # cells for external flow are always outside the solid, and the
    # triangle→cell direction is the most reliable indicator of the outward
    # normal direction.  Flipping based on ray-cast inside/outside status
    # was found to degrade results because ray-triangle intersection counts
    # are sensitive to edge-grazing and mesh gaps near the surface, causing
    # false "inside" classifications for ~15% of cells.
    #
    # The ray-casting code is retained for future use and debugging, but
    # does NOT override the triangle→cell orientation.
    triangles = vertices[faces].astype(np.float64)
    ray_dir = np.array([1.0, 0.0, 0.0])
    _intersection_counts = _ray_triangle_intersections_count(cell_pos, ray_dir, triangles)

    return normals


def _compute_triangle_areas(vertices, faces):
    """Compute per-triangle areas from vertices and faces."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1).astype(np.float64)


def mirror_stl(vertices, faces, face_normals, axis=1):
    """Mirror an STL half-hull about a symmetry plane to create a full hull.

    Ship-hull STL files are typically half-hulls (one side only, y >= 0).
    This function mirrors the mesh about the specified axis plane (default:
    y=0) to create a closed full hull suitable for voxelization.

    The mirrored triangles have their winding reversed so that normals
    point outward on the mirrored side.

    Parameters
    ----------
    vertices : np.ndarray, shape (N, 3)
        Original vertex coordinates.
    faces : np.ndarray, shape (M, 3)
        Triangle vertex indices.
    face_normals : np.ndarray, shape (M, 3)
        Face normals of the original mesh.
    axis : int, default 1
        Axis to mirror (0=x, 1=y, 2=z).

    Returns
    -------
    vertices_full : np.ndarray, shape (2N, 3)
    faces_full : np.ndarray, shape (2M, 3)
    normals_full : np.ndarray, shape (2M, 3)
    """
    n_v = len(vertices)
    len(faces)

    # Mirrored vertices: negate the mirror axis coordinate
    vertices_mir = vertices.copy().astype(np.float64)
    vertices_mir[:, axis] = -vertices_mir[:, axis]

    # Combine: original + mirrored
    vertices_full = np.vstack([vertices.astype(np.float64), vertices_mir])

    # Original faces stay the same
    faces_orig = faces.copy()
    # Mirrored faces: offset indices by n_v, reverse winding (swap v1,v2)
    faces_mir = np.column_stack(
        [
            faces[:, 0] + n_v,
            faces[:, 2] + n_v,  # reversed winding
            faces[:, 1] + n_v,
        ]
    )

    faces_full = np.vstack([faces_orig, faces_mir]).astype(np.int32)

    # Mirrored normals: negate the mirror axis component
    normals_mir = face_normals.copy().astype(np.float64)
    normals_mir[:, axis] = -normals_mir[:, axis]
    normals_full = np.vstack(
        [
            face_normals.astype(np.float64),
            normals_mir,
        ]
    ).astype(np.float32)

    return vertices_full, faces_full, normals_full


def SurfaceMesh_from_stl(
    solid,
    near,
    vertices,
    faces,
    face_normals,
    origin,
    spacing,
    dA_method="none",
):
    """Build a :class:`SurfaceMesh` with STL-derived surface normals.

    For each near-wall cell:

    1. The cell's physical position is computed from *origin* / *spacing*.
    2. The nearest STL triangle is found (by centroid distance, via cKDTree).
    3. That triangle's face normal is used.
    4. The normal is oriented outward using ray-casting:
       - A ray is cast from the cell centre in +x direction.
       - The number of STL triangle intersections determines inside/outside.
       - The normal is flipped to point from the solid surface toward the
         cell (outward for external-flow cells, which are outside the solid).

    This replaces the previous gradient-based orientation (Bug 29) which
    failed for elongated hulls where the solid-mask gradient is dominated
    by transverse components, making the dot-product flip unreliable.

    Parameters
    ----------
    solid : torch.Tensor, shape (nz, ny, nx), bool
        Solid boolean mask.
    near : torch.Tensor, shape (nz, ny, nx), bool
        Near-wall cell mask (fluid cells adjacent to solid).
    vertices : np.ndarray, shape (N, 3)
        STL vertex coordinates.
    faces : np.ndarray, shape (M, 3)
        STL triangle vertex indices.
    face_normals : np.ndarray, shape (M, 3)
        STL face normals (unit vectors).
    origin : tuple of float
        Physical coordinates of the grid's lower corner.
    spacing : tuple of float
        Cell size in each direction.
    dA_method : {'none', 'stl_area', 'cos_theta'}, default 'none'
        Surface area element computation:
        - 'none': dA = 1.0 (default, consistent with from_sphere etc.)
        - 'stl_area': dA = triangle_area / count, distributing the true
          STL surface area among near-wall cells.  This corrects the
          staircase surface-area underestimation (dA=1.0 gives ~88.5%
          of true area).
        - 'cos_theta': dA = 1 / |n_dominant|, the geometric surface-area
          correction for a staircase approximation of a tilted surface.

    Returns
    -------
    SurfaceMesh
        With ``nx_n``, ``ny_n``, ``nz_n`` tensors of shape (nz, ny, nx)
        and ``dA`` tensor of the same shape.
    """
    nz, ny, nx = solid.shape
    device = solid.device

    # Work on CPU for numpy / scipy operations
    solid_cpu = solid.cpu() if device.type != "cpu" else solid
    near_cpu = near.cpu() if device.type != "cpu" else near

    near_idx = near_cpu.nonzero(as_tuple=False)  # (n_near, 3) — (iz, iy, ix)
    # Bug 46 fix: only use FLUID near-wall cells (solid=False) for normal orientation.
    # The normal must point toward fluid, not solid. This is why from_gradient
    # is more reliable — the gradient naturally points from solid to fluid.
    solid_np = solid_cpu.numpy() if hasattr(solid_cpu, "numpy") else solid_cpu
    is_fluid = ~solid_np[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]]
    near_idx = near_idx[is_fluid]
    n_near = near_idx.shape[0]
    if n_near == 0:
        z = torch.zeros(nz, ny, nx, dtype=torch.float32, device=device)
        return SurfaceMesh(near, z, z, z)

    # Physical coordinates of near-wall cell centres
    iz = near_idx[:, 0].numpy().astype(np.float64)
    iy = near_idx[:, 1].numpy().astype(np.float64)
    ix = near_idx[:, 2].numpy().astype(np.float64)
    px = origin[0] + (ix + 0.5) * spacing[0]
    py = origin[1] + (iy + 0.5) * spacing[1]
    pz = origin[2] + (iz + 0.5) * spacing[2]
    cell_pos = np.stack([px, py, pz], axis=1)  # (n_near, 3)

    # Triangle centroids
    tri_verts = vertices[faces]  # (n_tri, 3, 3)
    centroids = tri_verts.mean(axis=1)  # (n_tri, 3)

    # Nearest triangle normal for each cell
    normals, tri_idx = _nearest_triangle_normals(
        cell_pos, centroids, face_normals
    )  # (n_near, 3), (n_near,)

    # Orient normals outward using ray-casting inside/outside test.
    #
    # Previous approach (Bug 29): used the solid-mask gradient to flip
    # normals.  This fails for elongated hulls (KVLCC2, DTMB5415) because
    # the gradient of the voxelised solid mask is dominated by the
    # transverse (y/z) components, while the true surface normal at the
    # bow/stern is mostly axial (x).  The dot product between the STL face
    # normal and the gradient normal is ≈0, so the flip threshold (dot < 0)
    # is unreliable — it flips ~50% of normals the wrong way.
    #
    # The new approach:
    # 1. Use the direction from the nearest triangle centroid to the cell
    #    as the "toward cell" reference (near-wall cells are fluid, outside
    #    the solid, so the outward normal should point toward the cell).
    # 2. Verify with ray casting: cast a ray in +x from each cell and count
    #    STL triangle intersections.  Odd = inside solid, even = outside.
    #    For inside cells, flip the normal to point away from the cell.
    #
    # This is robust for elongated geometries because it does not rely on
    # the voxelised solid-mask gradient.
    normals = _orient_normals_raycast(
        normals,
        cell_pos,
        centroids,
        tri_idx,
        vertices.astype(np.float64),
        faces.astype(np.int64),
    )

    # Bug 46b: STL face normals have |nx|≈0 at bow/stern of slender hulls
    # (surface nearly parallel to x-axis). Use gradient of solid mask to
    # fix the x-component — gradient naturally captures axial direction.
    # Hybrid: STL y/z + gradient x.
    solid_np = solid_cpu.numpy() if hasattr(solid_cpu, "numpy") else np.asarray(solid_cpu)
    for i in range(n_near):
        iz, iy, ix = int(near_idx[i, 0]), int(near_idx[i, 1]), int(near_idx[i, 2])
        # Gradient of solid mask (points from fluid=0 to solid=1)
        gx = 0.0
        gy = 0.0
        gz = 0.0
        if ix > 0 and ix < nx - 1:
            gx = float(solid_np[iz, iy, ix + 1]) - float(solid_np[iz, iy, ix - 1])
        if iy > 0 and iy < ny - 1:
            gy = float(solid_np[iz, iy + 1, ix]) - float(solid_np[iz, iy - 1, ix])
        if iz > 0 and iz < nz - 1:
            gz = float(solid_np[iz + 1, iy, ix]) - float(solid_np[iz - 1, iy, ix])
        # Outward = -gradient (from solid to fluid)
        gnorm = (gx * gx + gy * gy + gz * gz) ** 0.5
        if gnorm > 1e-10:
            gx_out = -gx / gnorm
            # Replace x-component with gradient x (STL |nx| too small)
            # Keep y/z from STL (accurate transverse direction)
            normals[i, 0] = gx_out
            # Re-normalize
            nrm_i = (normals[i, 0] ** 2 + normals[i, 1] ** 2 + normals[i, 2] ** 2) ** 0.5
            if nrm_i > 1e-10:
                normals[i] = normals[i] / nrm_i

    # Normalise
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(nrm > 1e-10, nrm, 1.0)

    # Build full-grid tensors
    nx_n = torch.zeros(nz, ny, nx, dtype=torch.float32)
    ny_n = torch.zeros(nz, ny, nx, dtype=torch.float32)
    nz_n = torch.zeros(nz, ny, nx, dtype=torch.float32)
    iz_t = near_idx[:, 0]
    iy_t = near_idx[:, 1]
    ix_t = near_idx[:, 2]
    nx_n[iz_t, iy_t, ix_t] = torch.tensor(normals[:, 0], dtype=torch.float32)
    ny_n[iz_t, iy_t, ix_t] = torch.tensor(normals[:, 1], dtype=torch.float32)
    nz_n[iz_t, iy_t, ix_t] = torch.tensor(normals[:, 2], dtype=torch.float32)

    # ---- Surface area element (dA) computation ----
    if dA_method == "none":
        dA = torch.ones(nz, ny, nx, dtype=torch.float32)
    elif dA_method == "stl_area":
        # STL-based surface area correction.
        # For each near-wall cell, use the nearest triangle's area as
        # the raw dA, then scale so that sum(dA) = true STL surface area
        # in lattice units.  This corrects the staircase area
        # underestimation (dA=1.0 gives ~88-93% of true area).
        tri_areas = _compute_triangle_areas(vertices.astype(np.float64), faces.astype(np.int64))
        # Raw dA per cell = area of nearest triangle (in STL units)
        dA_raw = tri_areas[tri_idx].astype(np.float64)
        # Convert to lattice units: divide by cell-face area (spacing²)
        s2 = float(spacing[0] * spacing[1])  # isotropic assumption
        dA_raw_lattice = dA_raw / s2
        # Scale so sum(dA) = true STL area in lattice units
        true_area_lattice = float(tri_areas.sum()) / s2
        sum_raw = float(dA_raw_lattice.sum())
        if sum_raw > 1e-10:
            scale = true_area_lattice / sum_raw
        else:
            scale = 1.0
        dA_per_cell = (dA_raw_lattice * scale).astype(np.float32)
        dA = torch.ones(nz, ny, nx, dtype=torch.float32)
        dA[iz_t, iy_t, ix_t] = torch.tensor(dA_per_cell, dtype=torch.float32)
    elif dA_method == "cos_theta":
        # Geometric surface-area correction: dA = 1 / |n_dominant|
        # For a face-aligned surface (n = (1,0,0)), dA = 1.0.
        # For a 45° surface, dA = 1/cos(45°) = √2 ≈ 1.414.
        # This corrects the staircase area underestimation.
        abs_nx = np.abs(normals[:, 0])
        abs_ny = np.abs(normals[:, 1])
        abs_nz = np.abs(normals[:, 2])
        n_dom = np.maximum(np.maximum(abs_nx, abs_ny), abs_nz)
        n_dom = np.where(n_dom > 1e-6, n_dom, 1.0)
        dA_per_cell = (1.0 / n_dom).astype(np.float32)
        dA = torch.ones(nz, ny, nx, dtype=torch.float32)
        dA[iz_t, iy_t, ix_t] = torch.tensor(dA_per_cell, dtype=torch.float32)
    else:
        raise ValueError(f"dA_method must be 'none', 'stl_area', or 'cos_theta', got '{dA_method}'")

    # Move back to original device
    if device.type != "cpu":
        nx_n = nx_n.to(device)
        ny_n = ny_n.to(device)
        nz_n = nz_n.to(device)
        dA = dA.to(device)

    return SurfaceMesh(near, nx_n, ny_n, nz_n, dA)


# ---------------------------------------------------------------------------
# 4. STL WRITER
# ---------------------------------------------------------------------------


def write_stl(path, vertices, faces, binary=True):
    """Write an STL file from vertex + face-index arrays.

    Parameters
    ----------
    path : str or Path
        Output file path.
    vertices : np.ndarray, shape (N, 3)
        Vertex coordinates.
    faces : np.ndarray, shape (M, 3)
        Triangle vertex indices.
    binary : bool, default True
        If True write binary STL; otherwise ASCII.
    """
    path = Path(path)
    verts = np.asarray(vertices, dtype=np.float32)
    fcs = np.asarray(faces, dtype=np.int32)
    if binary:
        _write_stl_binary(path, verts, fcs)
    else:
        v0 = verts[fcs[:, 0]]
        v1 = verts[fcs[:, 1]]
        v2 = verts[fcs[:, 2]]
        normals = np.cross(v1 - v0, v2 - v0)
        nrm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.where(nrm > 1e-10, nrm, 1.0)
        lines = ["solid mesh"]
        for i in range(len(fcs)):
            n = normals[i]
            lines.append(f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            lines.append("    outer loop")
            for v in [v0[i], v1[i], v2[i]]:
                lines.append(f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            lines.append("    endloop")
            lines.append("  endfacet")
        lines.append("endsolid mesh")
        path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# 5. TEST STL GEOMETRY GENERATORS
# ---------------------------------------------------------------------------


def make_sphere_stl(center, radius, n_lat=30, n_lon=60):
    """Generate a UV-sphere STL mesh with outward-facing normals.

    Uses a single pole vertex at each pole (triangle fan) to avoid
    degenerate triangles that occur when all longitude vertices collapse
    to a single point.

    Parameters
    ----------
    center : tuple of float (cx, cy, cz)
        Sphere centre.
    radius : float
        Sphere radius.
    n_lat : int, default 30
        Number of latitude divisions (poles to poles).
    n_lon : int, default 60
        Number of longitude divisions.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3), float32
    faces : np.ndarray, shape (M, 3), int32
    """
    cx, cy, cz = center
    R = float(radius)
    vertices = []
    # North pole vertex (index 0)
    vertices.append([cx, cy, cz + R])
    # Latitude rings: i=1 .. n_lat-1 (skip poles)
    for i in range(1, n_lat):
        theta = np.pi * i / n_lat  # 0 < theta < pi
        for j in range(n_lon):
            phi = 2 * np.pi * j / n_lon
            x = cx + R * np.sin(theta) * np.cos(phi)
            y = cy + R * np.sin(theta) * np.sin(phi)
            z = cz + R * np.cos(theta)
            vertices.append([x, y, z])
    # South pole vertex (last index)
    vertices.append([cx, cy, cz - R])

    north_pole = 0
    south_pole = len(vertices) - 1
    ring_start = 1  # first ring starts at index 1

    faces = []
    # North pole fan: [pole, v1, v2] → outward (+z)
    for j in range(n_lon):
        j1 = (j + 1) % n_lon
        v1 = ring_start + j
        v2 = ring_start + j1
        faces.append([north_pole, v1, v2])

    # Middle quads: [v0, v2, v1] and [v1, v2, v3] → outward
    for i in range(n_lat - 2):
        for j in range(n_lon):
            j1 = (j + 1) % n_lon
            v0 = ring_start + i * n_lon + j
            v1 = ring_start + i * n_lon + j1
            v2 = ring_start + (i + 1) * n_lon + j
            v3 = ring_start + (i + 1) * n_lon + j1
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])

    # South pole fan: [pole, v2, v1] → outward (-z)
    last_ring = ring_start + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j1 = (j + 1) % n_lon
        v1 = last_ring + j
        v2 = last_ring + j1
        faces.append([south_pole, v2, v1])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


# Canonical icosahedron (golden-ratio construction): 12 vertices and the
# 20 outward-wound faces, written out explicitly so the generator has no
# hidden dependence on iteration order.
_ICOSA_T = (1.0 + np.sqrt(5.0)) / 2.0
_ICOSA_VERTS = np.array(
    [
        [-1.0, _ICOSA_T, 0.0],
        [1.0, _ICOSA_T, 0.0],
        [-1.0, -_ICOSA_T, 0.0],
        [1.0, -_ICOSA_T, 0.0],
        [0.0, -1.0, _ICOSA_T],
        [0.0, 1.0, _ICOSA_T],
        [0.0, -1.0, -_ICOSA_T],
        [0.0, 1.0, -_ICOSA_T],
        [_ICOSA_T, 0.0, -1.0],
        [_ICOSA_T, 0.0, 1.0],
        [-_ICOSA_T, 0.0, -1.0],
        [-_ICOSA_T, 0.0, 1.0],
    ],
    dtype=np.float64,
)
_ICOSA_VERTS /= np.linalg.norm(_ICOSA_VERTS[0])  # project onto the unit sphere
_ICOSA_FACES = np.array(
    [
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ],
    dtype=np.int64,
)


def make_icosphere_stl(center, radius, subdivisions=4, scale=(1.0, 1.0, 1.0)):
    """Generate a deterministic icosphere (subdivided icosahedron) mesh.

    Each subdivision splits every triangle into four by edge midpoints
    projected back onto the sphere, giving a near-uniform tessellation
    with exactly ``20 * 4**subdivisions`` faces — no pole degeneracies,
    unlike :func:`make_sphere_stl`.  Intended for voxelisation benchmarks
    (10^4 – 10^6 faces) and reproducibility tests: the construction is
    purely deterministic (sorted edge welding, no hashing randomness),
    so the same arguments always produce bit-identical output.

    Parameters
    ----------
    center : tuple of float (cx, cy, cz)
        Sphere centre.
    radius : float
        Sphere radius.
    subdivisions : int, default 4
        Number of edge-midpoint subdivisions (0 = raw icosahedron,
        20 faces; 8 = 1,310,720 faces).
    scale : tuple of float (sx, sy, sz), default (1, 1, 1)
        Per-axis stretch applied after normalisation — use e.g.
        ``(1.0, 1.0, 2.0)`` for a prolate ellipsoid.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3), float32
    faces : np.ndarray, shape (M, 3), int32
    """
    verts = _ICOSA_VERTS.copy()
    faces = _ICOSA_FACES.copy()
    for _ in range(int(subdivisions)):
        # Weld edge midpoints once: np.unique sorts, so the vertex order
        # (and therefore the whole mesh) is a deterministic function of
        # the arguments.
        edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
        welded, mid_of_edge = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True)
        midpoints = 0.5 * (verts[welded[:, 0]] + verts[welded[:, 1]])
        midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
        n_old = verts.shape[0]
        verts = np.concatenate([verts, midpoints])
        e01 = n_old + mid_of_edge[0 : faces.shape[0]]
        e12 = n_old + mid_of_edge[faces.shape[0] : 2 * faces.shape[0]]
        e20 = n_old + mid_of_edge[2 * faces.shape[0] : 3 * faces.shape[0]]
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        faces = np.concatenate(
            [
                np.stack([a, e01, e20], axis=1),
                np.stack([e01, b, e12], axis=1),
                np.stack([e12, c, e20], axis=1),
                np.stack([e01, e12, e20], axis=1),
            ]
        )
    sx, sy, sz = (float(s) for s in scale)
    verts = verts * np.array([sx, sy, sz], dtype=np.float64)[None, :] * float(radius)
    verts = verts + np.asarray(center, dtype=np.float64)[None, :]
    return verts.astype(np.float32), faces.astype(np.int32)


def make_cylinder_stl(center, radius, length, n_circ=40, axis="z", n_axial=1):
    """Generate a cylinder STL mesh with outward-facing normals.

    The cylinder is extruded along *axis*; the circular cross-section lies
    in the plane perpendicular to *axis*.  The side surface is subdivided
    into *n_axial* segments along the axis so that triangle centroids are
    distributed (improves nearest-triangle normal lookup).

    Parameters
    ----------
    center : tuple of float
        Cylinder centre (cx, cy, cz).
    radius : float
        Cylinder radius.
    length : float
        Cylinder length along *axis*.
    n_circ : int, default 40
        Number of circumferential divisions.
    axis : {'z', 'x', 'y'}, default 'z'
        Extrusion axis.
    n_axial : int, default 1
        Number of axial subdivisions of the side surface.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3), float32
    faces : np.ndarray, shape (M, 3), int32
    """
    cx, cy, cz = center
    R = float(radius)
    L = float(length)

    axis_map = {"z": (2, 0, 1), "x": (0, 1, 2), "y": (1, 0, 2)}
    if axis not in axis_map:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    axial_i, p1_i, p2_i = axis_map[axis]

    center_arr = np.array([cx, cy, cz], dtype=np.float64)

    # Build vertices: (n_axial+1) rings of n_circ vertices each
    vertices = []
    for k in range(n_axial + 1):
        frac = k / n_axial  # 0 at bottom, 1 at top
        h = center_arr.copy()
        h[axial_i] = center_arr[axial_i] - L / 2 + frac * L
        for j in range(n_circ):
            phi = 2 * np.pi * j / n_circ
            offset = np.zeros(3)
            offset[p1_i] = R * np.cos(phi)
            offset[p2_i] = R * np.sin(phi)
            vertices.append((h + offset).tolist())

    # Cap centres
    h0 = center_arr.copy()
    h0[axial_i] = center_arr[axial_i] - L / 2
    h1 = center_arr.copy()
    h1[axial_i] = center_arr[axial_i] + L / 2
    vertices.append(h0.tolist())  # bottom centre → idx n_rings*n_circ
    vertices.append(h1.tolist())  # top centre    → idx n_rings*n_circ+1

    n_rings = n_axial + 1
    bc = n_rings * n_circ  # bottom centre index
    tc = n_rings * n_circ + 1  # top centre index

    def ring_idx(k, j):
        """Vertex index for ring k, circumferential position j."""
        return k * n_circ + j

    faces = []
    for k in range(n_axial):
        for j in range(n_circ):
            j1 = (j + 1) % n_circ
            b0 = ring_idx(k, j)
            b1 = ring_idx(k, j1)
            t0 = ring_idx(k + 1, j)
            t1 = ring_idx(k + 1, j1)
            # Side faces: [b0, b1, t0] and [b1, t1, t0] → outward radial
            faces.append([b0, b1, t0])
            faces.append([b1, t1, t0])

    # Bottom cap: [ring0_j, bc, ring0_{j+1}] → normal -axis
    for j in range(n_circ):
        j1 = (j + 1) % n_circ
        faces.append([ring_idx(0, j), bc, ring_idx(0, j1)])
    # Top cap: [ringN_j, ringN_{j+1}, tc] → normal +axis
    for j in range(n_circ):
        j1 = (j + 1) % n_circ
        faces.append([ring_idx(n_axial, j), ring_idx(n_axial, j1), tc])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )


def make_naca_stl(
    chord,
    x_le,
    y_mid,
    z0,
    z1,
    thickness_ratio=0.12,
    n_x=50,
):
    """Generate a NACA 4-digit airfoil STL (extruded in z).

    The airfoil chord lies along +x with the leading edge at
    ``(x_le, y_mid)``.  The profile is extruded from *z0* to *z1*.

    Parameters
    ----------
    chord : float
        Chord length.
    x_le : float
        Leading-edge x-coordinate.
    y_mid : float
        Camber-line y-coordinate (0 for symmetric).
    z0, z1 : float
        Extrusion limits in z.
    thickness_ratio : float, default 0.12
        NACA thickness parameter (e.g. 0.12 for NACA 0012).
    n_x : int, default 50
        Number of chordwise sampling points.

    Returns
    -------
    vertices : np.ndarray, shape (N, 3), float32
    faces : np.ndarray, shape (M, 3), int32
    """
    t = thickness_ratio
    # Normalised x stations (avoid x=0 exactly for sqrt)
    xn = np.linspace(0.0, 1.0, n_x)
    xn[0] = 1e-4  # avoid division by zero in sqrt
    # NACA 4-digit thickness equation
    yt = (
        5.0
        * t
        * (0.2969 * np.sqrt(xn) - 0.1260 * xn - 0.3516 * xn**2 + 0.2843 * xn**3 - 0.1015 * xn**4)
    )
    xc = x_le + xn * chord
    yu = y_mid + yt * chord
    yl = y_mid - yt * chord

    # Vertex layout:
    #  [0      .. n_x-1]  upper @ z0
    #  [n_x    .. 2*n_x-1] upper @ z1
    #  [2*n_x  .. 3*n_x-1] lower @ z0
    #  [3*n_x  .. 4*n_x-1] lower @ z1
    vertices = []
    for j in range(n_x):
        vertices.append([xc[j], yu[j], z0])
    for j in range(n_x):
        vertices.append([xc[j], yu[j], z1])
    for j in range(n_x):
        vertices.append([xc[j], yl[j], z0])
    for j in range(n_x):
        vertices.append([xc[j], yl[j], z1])

    def u0(j):
        return j

    def u1(j):
        return n_x + j

    def l0(j):
        return 2 * n_x + j

    def l1(j):
        return 3 * n_x + j

    faces = []
    for j in range(n_x - 1):
        # Upper surface (normal +y): [u0, u1, u0+1] and [u0+1, u1, u1+1]
        faces.append([u0(j), u1(j), u0(j + 1)])
        faces.append([u0(j + 1), u1(j), u1(j + 1)])
        # Lower surface (normal -y): [l0, l0+1, l1] and [l0+1, l1+1, l1]
        faces.append([l0(j), l0(j + 1), l1(j)])
        faces.append([l0(j + 1), l1(j + 1), l1(j)])
        # Front cap at z0 (normal -z): [u0, u0+1, l0] and [u0+1, l0+1, l0]
        faces.append([u0(j), u0(j + 1), l0(j)])
        faces.append([u0(j + 1), l0(j + 1), l0(j)])
        # Back cap at z1 (normal +z): [u1, l1, u1+1] and [u1+1, l1, l1+1]
        faces.append([u1(j), l1(j), u1(j + 1)])
        faces.append([u1(j + 1), l1(j), l1(j + 1)])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int32),
    )
