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

import struct
from pathlib import Path

import numpy as np
import torch

from .preprocess_geo import _voxelize_triangles, _write_stl_binary
from .drag_pressure import SurfaceMesh, get_near_wall_3d

__all__ = [
    "read_stl",
    "voxelize_stl",
    "SurfaceMesh_from_stl",
    "write_stl",
    "get_near_wall_3d",
    "make_sphere_stl",
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
    triangles = np.stack(
        [records["v0"], records["v1"], records["v2"]], axis=1
    ).astype(np.float32)
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
    triangles_arr = np.array(triangles, dtype=np.float32) if triangles else np.zeros((0, 3, 3), dtype=np.float32)
    normals_arr = np.array(normals, dtype=np.float32) if normals else np.zeros((0, 3), dtype=np.float32)
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
    face_normals = np.where(
        has_stored[:, None], stored_normals, computed_normals
    ).astype(np.float32)

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
        triangles, nx, ny, nz,
        x_min, y_min, z_min, x_max, y_max, z_max,
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
    norm = torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2).clamp(min=1e-10)
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
        dists = np.sum(diffs ** 2, axis=2)
        idx = np.argmin(dists, axis=1)
    return face_normals[idx].astype(np.float64).copy(), idx


def SurfaceMesh_from_stl(
    solid,
    near,
    vertices,
    faces,
    face_normals,
    origin,
    spacing,
):
    """Build a :class:`SurfaceMesh` with STL-derived surface normals.

    For each near-wall cell:

    1. The cell's physical position is computed from *origin* / *spacing*.
    2. The nearest STL triangle is found (by centroid distance, via cKDTree).
    3. That triangle's face normal is used.
    4. The normal is flipped to point outward (aligned with the
       gradient-based normal from the solid mask).

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

    Returns
    -------
    SurfaceMesh
        With ``nx_n``, ``ny_n``, ``nz_n`` tensors of shape (nz, ny, nx).
        ``dA = 1.0`` (default, consistent with
        :meth:`SurfaceMesh.from_sphere` / :meth:`SurfaceMesh.from_cylinder`).
    """
    nz, ny, nx = solid.shape
    device = solid.device

    # Work on CPU for numpy / scipy operations
    solid_cpu = solid.cpu() if device.type != "cpu" else solid
    near_cpu = near.cpu() if device.type != "cpu" else near

    near_idx = near_cpu.nonzero(as_tuple=False)  # (n_near, 3) — (iz, iy, ix)
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

    # Outward direction from gradient-based normal
    grad_nx, grad_ny, grad_nz = _compute_gradient_normals(solid_cpu, near_cpu)
    grad_nx_np = grad_nx[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_ny_np = grad_ny[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_nz_np = grad_nz[near_idx[:, 0], near_idx[:, 1], near_idx[:, 2]].numpy()
    grad_normals = np.stack(
        [grad_nx_np, grad_ny_np, grad_nz_np], axis=1
    )  # (n_near, 3)

    # Flip normals to align with gradient direction (outward)
    dot = (normals * grad_normals).sum(axis=1)
    flip_mask = dot < 0
    normals[flip_mask] = -normals[flip_mask]

    # Fallback for cells where gradient is zero (degenerate corners):
    # use direction from nearest triangle centroid to cell
    grad_norm = np.linalg.norm(grad_normals, axis=1)
    zero_grad = grad_norm < 1e-6
    if zero_grad.any():
        fallback_dir = cell_pos[zero_grad] - centroids[tri_idx[zero_grad]]
        fb_norm = np.linalg.norm(fallback_dir, axis=1, keepdims=True)
        fallback_dir = fallback_dir / np.where(fb_norm > 1e-10, fb_norm, 1.0)
        dot_fb = (normals[zero_grad] * fallback_dir).sum(axis=1)
        flip_fb = dot_fb < 0
        normals[zero_grad][flip_fb] = -normals[zero_grad][flip_fb]

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

    # Move back to original device
    if device.type != "cpu":
        nx_n = nx_n.to(device)
        ny_n = ny_n.to(device)
        nz_n = nz_n.to(device)

    return SurfaceMesh(near, nx_n, ny_n, nz_n)


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
            lines.append(
                f"  facet normal {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}"
            )
            lines.append("    outer loop")
            for v in [v0[i], v1[i], v2[i]]:
                lines.append(
                    f"      vertex {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}"
                )
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


def make_cylinder_stl(
    center, radius, length, n_circ=40, axis="z", n_axial=1
):
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
        5.0 * t
        * (
            0.2969 * np.sqrt(xn)
            - 0.1260 * xn
            - 0.3516 * xn ** 2
            + 0.2843 * xn ** 3
            - 0.1015 * xn ** 4
        )
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

    def u0(j): return j
    def u1(j): return n_x + j
    def l0(j): return 2 * n_x + j
    def l1(j): return 3 * n_x + j

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
