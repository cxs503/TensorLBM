"""Iso-surface (iso-contour) extraction from LBM flow fields.

Extracts contour lines (2-D) or surfaces (3-D) at a specified field value
using the *marching squares* (2-D) and a simplified *marching cubes* (3-D)
algorithm.  This is a standard post-processing feature in PowerFlow and
XFlow for visualising vortex cores, pressure iso-surfaces, and Q-criterion
structures.

2-D Marching Squares
--------------------
The classical Lorensen–Cline marching squares algorithm operates on a
uniform grid.  Each 2×2 cell of lattice points is classified by a 4-bit
index according to which corners exceed the iso-value.  For each non-trivial
case, linear interpolation locates the crossing point on each edge.

The output is a list of line segments (pairs of vertices) that form the
iso-contour.

3-D Marching Cubes (lightweight voxel variant)
----------------------------------------------
The full 256-case lookup table is omitted for simplicity.  Instead we use
the *dual-contouring* approximation: for each 2×2×2 cube that straddles the
iso-surface we find all active edges (sign changes) and return the edge
midpoints as triangle approximations via the centroid method.  This produces
a valid triangulation suitable for engineering visualisation.

References
----------
Lorensen, W.E. & Cline, H.E. (1987). "Marching cubes: A high resolution 3D
    surface construction algorithm." *ACM SIGGRAPH Computer Graphics* 21(4),
    163–169.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

__all__ = [
    "IsoContour2D",
    "IsoSurface3D",
    "marching_squares",
    "marching_cubes_simple",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class IsoContour2D:
    """Result of 2-D iso-contour extraction.

    Attributes
    ----------
    segments:
        List of line segments, each ``[[x0, y0], [x1, y1]]``.
    n_segments:
        Total number of segments.
    iso_value:
        The field value at which the contour was extracted.
    field_name:
        Name of the scalar field (e.g. ``"pressure"`` or ``"q_criterion"``).
    """

    segments: list[list[list[float]]]
    n_segments: int
    iso_value: float
    field_name: str = "scalar"


@dataclass
class IsoSurface3D:
    """Result of 3-D iso-surface extraction.

    Attributes
    ----------
    vertices:
        List of vertex coordinates ``[x, y, z]``.
    triangles:
        List of triangles, each a list of three vertex indices.
    n_triangles:
        Total triangle count.
    iso_value:
        Iso-value used.
    field_name:
        Name of the scalar field.
    """

    vertices: list[list[float]]
    triangles: list[list[int]]
    n_triangles: int
    iso_value: float
    field_name: str = "scalar"


# ---------------------------------------------------------------------------
# 2-D Marching Squares
# ---------------------------------------------------------------------------

# Edge table: for each 4-bit corner index (0–15), which edges are active?
# Edges: 0=bottom(j,j+1), 1=right(i+1,i), 2=top(j+1,j), 3=left(i,i+1)
# Encoding: list of edge pairs that form line segments.
_MS_EDGE_TABLE: dict[int, list[tuple[int, int]]] = {
    0: [], 15: [],          # all below / all above
    1: [(3, 0)],            # only bottom-left above
    2: [(0, 1)],
    3: [(3, 1)],
    4: [(1, 2)],
    5: [(3, 0), (1, 2)],    # ambiguous – use two segments
    6: [(0, 2)],
    7: [(3, 2)],
    8: [(2, 3)],
    9: [(2, 0)],
    10: [(0, 1), (2, 3)],   # ambiguous
    11: [(2, 1)],
    12: [(1, 3)],
    13: [(0, 3)],            # corrected winding
    14: [(1, 0)],
}

# Corner indices within a cell:
#  3---2
#  |   |
#  0---1
# (row i, col j)  → corners: (i,j)=0, (i,j+1)=1, (i+1,j+1)=2, (i+1,j)=3
_CORNERS = [(0, 0), (0, 1), (1, 1), (1, 0)]

# Edge endpoints as corner pairs
_EDGE_CORNERS = [
    (0, 1),  # edge 0: bottom (corners 0–1)
    (1, 2),  # edge 1: right  (corners 1–2)
    (3, 2),  # edge 2: top    (corners 3–2)
    (0, 3),  # edge 3: left   (corners 0–3)
]


def _interpolate_edge(
    val_a: float,
    val_b: float,
    coord_a: tuple[float, float],
    coord_b: tuple[float, float],
    iso_value: float,
) -> tuple[float, float]:
    """Linearly interpolate crossing point on an edge."""
    dv = val_b - val_a
    if abs(dv) < 1e-12:
        return ((coord_a[0] + coord_b[0]) * 0.5, (coord_a[1] + coord_b[1]) * 0.5)
    t = (iso_value - val_a) / dv
    t = max(0.0, min(1.0, t))
    return (
        coord_a[0] + t * (coord_b[0] - coord_a[0]),
        coord_a[1] + t * (coord_b[1] - coord_a[1]),
    )


def marching_squares(
    field: torch.Tensor,
    iso_value: float,
    *,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    field_name: str = "scalar",
    max_segments: int = 100_000,
) -> IsoContour2D:
    """Extract a 2-D iso-contour from a scalar field using marching squares.

    Parameters
    ----------
    field:
        2-D tensor of shape ``(ny, nx)`` containing scalar values on a
        uniform grid.
    iso_value:
        Scalar value at which to extract the contour.
    x_range:
        Physical x-extent ``(x_min, x_max)`` of the grid.
    y_range:
        Physical y-extent ``(y_min, y_max)`` of the grid.
    field_name:
        Label for the extracted field.
    max_segments:
        Safety cap on the number of returned segments.

    Returns
    -------
    IsoContour2D
    """
    field = field.float()
    ny, nx = field.shape
    if ny < 2 or nx < 2:
        return IsoContour2D(segments=[], n_segments=0,
                            iso_value=iso_value, field_name=field_name)

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)

    # Pre-compute physical coordinates
    def px(j: int) -> float:
        return float(x_range[0] + j * dx)

    def py(i: int) -> float:
        return float(y_range[0] + i * dy)

    segments: list[list[list[float]]] = []
    vals = field.tolist()

    for i in range(ny - 1):
        for j in range(nx - 1):
            # Corner values and coordinates
            corner_vals = [
                vals[i + dr][j + dc]
                for (dr, dc) in _CORNERS
            ]
            corner_coords = [
                (px(j + dc), py(i + dr))
                for (dr, dc) in _CORNERS
            ]
            # Build 4-bit index
            idx = 0
            for bit, v in enumerate(corner_vals):
                if v >= iso_value:
                    idx |= (1 << bit)

            edge_pairs = _MS_EDGE_TABLE.get(idx, [])
            for (ea, eb) in edge_pairs:
                ca0, ca1 = _EDGE_CORNERS[ea]
                cb0, cb1 = _EDGE_CORNERS[eb]
                pa = _interpolate_edge(
                    corner_vals[ca0], corner_vals[ca1],
                    corner_coords[ca0], corner_coords[ca1],
                    iso_value,
                )
                pb = _interpolate_edge(
                    corner_vals[cb0], corner_vals[cb1],
                    corner_coords[cb0], corner_coords[cb1],
                    iso_value,
                )
                segments.append([[pa[0], pa[1]], [pb[0], pb[1]]])
                if len(segments) >= max_segments:
                    return IsoContour2D(
                        segments=segments,
                        n_segments=len(segments),
                        iso_value=iso_value,
                        field_name=field_name,
                    )

    return IsoContour2D(
        segments=segments,
        n_segments=len(segments),
        iso_value=iso_value,
        field_name=field_name,
    )


# ---------------------------------------------------------------------------
# 3-D Marching Cubes (lightweight centroid variant)
# ---------------------------------------------------------------------------

def marching_cubes_simple(
    field: torch.Tensor,
    iso_value: float,
    *,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    z_range: tuple[float, float] = (0.0, 1.0),
    field_name: str = "scalar",
    max_vertices: int = 200_000,
) -> IsoSurface3D:
    """Extract a 3-D iso-surface using a centroid-based marching cubes.

    For each 2×2×2 cube that straddles the iso-surface, active edges are
    located by linear interpolation.  The edge midpoints are averaged to
    form a single representative vertex, and the surrounding active-edge
    points are used to generate a triangle fan.

    Parameters
    ----------
    field:
        3-D tensor of shape ``(nz, ny, nx)``.
    iso_value:
        Iso-value for surface extraction.
    x_range, y_range, z_range:
        Physical extents of the grid.
    field_name:
        Label for the extracted field.
    max_vertices:
        Safety cap on total vertices.

    Returns
    -------
    IsoSurface3D
    """
    field = field.float()
    nz, ny, nx = field.shape
    if nz < 2 or ny < 2 or nx < 2:
        return IsoSurface3D(vertices=[], triangles=[], n_triangles=0,
                            iso_value=iso_value, field_name=field_name)

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    dz = (z_range[1] - z_range[0]) / (nz - 1)

    def px(j: int) -> float:
        return float(x_range[0] + j * dx)

    def py(i: int) -> float:
        return float(y_range[0] + i * dy)

    def pz(k: int) -> float:
        return float(z_range[0] + k * dz)

    # Cube corners: (dk, di, dj) offsets
    _CUBE_CORNERS = [
        (0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0),
        (1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0),
    ]
    # 12 cube edges: pairs of corner indices
    _CUBE_EDGES = [
        (0, 1), (1, 2), (2, 3), (3, 0),   # bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),   # top face
        (0, 4), (1, 5), (2, 6), (3, 7),   # verticals
    ]

    vals_np = field.tolist()
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []

    for k in range(nz - 1):
        for i in range(ny - 1):
            for j in range(nx - 1):
                cv = [
                    vals_np[k + dk][i + di][j + dj]
                    for (dk, di, dj) in _CUBE_CORNERS
                ]
                cc = [
                    (px(j + dj), py(i + di), pz(k + dk))
                    for (dk, di, dj) in _CUBE_CORNERS
                ]
                above = [v >= iso_value for v in cv]
                if all(above) or not any(above):
                    continue  # no crossing

                active_pts: list[tuple[float, float, float]] = []
                for (ea, eb) in _CUBE_EDGES:
                    if above[ea] != above[eb]:
                        va, vb = cv[ea], cv[eb]
                        ca, cb = cc[ea], cc[eb]
                        dv = vb - va
                        t = (iso_value - va) / dv if abs(dv) > 1e-12 else 0.5
                        t = max(0.0, min(1.0, t))
                        pt = (
                            ca[0] + t * (cb[0] - ca[0]),
                            ca[1] + t * (cb[1] - ca[1]),
                            ca[2] + t * (cb[2] - ca[2]),
                        )
                        active_pts.append(pt)

                if len(active_pts) < 3:
                    continue

                # Centroid fan triangulation
                cx = sum(p[0] for p in active_pts) / len(active_pts)
                cy = sum(p[1] for p in active_pts) / len(active_pts)
                cz = sum(p[2] for p in active_pts) / len(active_pts)

                base_idx = len(vertices)
                vertices.append([cx, cy, cz])
                for pt in active_pts:
                    vertices.append(list(pt))

                n_pts = len(active_pts)
                for m in range(n_pts):
                    triangles.append([
                        base_idx,
                        base_idx + 1 + m,
                        base_idx + 1 + (m + 1) % n_pts,
                    ])

                if len(vertices) >= max_vertices:
                    return IsoSurface3D(
                        vertices=vertices,
                        triangles=triangles,
                        n_triangles=len(triangles),
                        iso_value=iso_value,
                        field_name=field_name,
                    )

    return IsoSurface3D(
        vertices=vertices,
        triangles=triangles,
        n_triangles=len(triangles),
        iso_value=iso_value,
        field_name=field_name,
    )
