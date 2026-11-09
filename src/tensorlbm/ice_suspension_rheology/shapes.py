"""Discretised particle geometry and non-overlapping placement.

Every shape is returned as a *boolean occupancy mask* on the lattice, and the
particle volume is taken from the **node count of that mask**, not from the
analytic formula.  This matters for the calibration: a staircase sphere of
nominal radius ``a`` has an effective volume a few percent different from
``4/3 pi a**3``, and the Einstein slope is sensitive to ``phi`` (the slope is
``[eta] = d eta_r / d phi``).  Using the discrete volume removes the geometry
error from ``phi`` and leaves only the (measured, reported) error in the
stresslet itself.

Shapes
------
``sphere``       exact lattice sphere, radius ``a``  (Einstein calibration)
``faceted``      convex polyhedron inscribed in a sphere of radius ``a`` with
                 ``n_faces`` faces -- a *near-spherical* ice crystal (frazil
                 grains are polyhedral, not ideal spheres)
``spheroid``     oblate/prolate spheroid with aspect ratio ``aspect``

Rigid-body kinematics
---------------------
A particle in a simple shear ``u = gamma_dot (z - z_mid) e_x`` that is force-
and torque-free translates with the local velocity and rotates with the
half-vorticity ``Omega = gamma_dot/2 e_y``.  :func:`rigid_body_velocity`
returns that field on the particle nodes.  For near-spherical shapes the
frozen half-vorticity rotation is the correct leading-order kinematics; for
strongly non-spherical shapes the orientation follows a Jeffery orbit and the
frozen orientation is an **approximation that is reported as such** in the
module documentation and in the campaign limitations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "Particle",
    "box_grid",
    "sphere_mask",
    "faceted_mask",
    "spheroid_mask",
    "make_particle",
    "measure_effective_fraction",
    "place_particles",
    "polydisperse_radii",
    "rigid_body_velocity",
]


@dataclass
class Particle:
    """A rigid particle: shape + centre + the lattice mask it occupies."""

    mask: np.ndarray            # bool (nz, ny, nx)
    centre: tuple[float, float, float]   # (z, y, x) in lattice units
    half_extent: float          # circumradius, lattice units
    shape: str = "sphere"
    volume: int = 0             # node count (discrete volume)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.volume == 0:
            self.volume = int(self.mask.sum())


def box_grid(nx: int, ny: int, nz: int):
    """Node coordinates ``z, y, x`` of shape ``(nz, ny, nx)``."""
    z, y, x = np.meshgrid(
        np.arange(nz, dtype=float),
        np.arange(ny, dtype=float),
        np.arange(nx, dtype=float),
        indexing="ij",
    )
    return z, y, x


def sphere_mask(
    nx: int, ny: int, nz: int, centre, radius: float, periodic_xy: bool = True
) -> np.ndarray:
    """Lattice sphere: nodes with ``|x - c| <= radius``."""
    z, y, x = box_grid(nx, ny, nz)
    cz, cy, cx = centre
    dz = z - cz
    dy = y - cy
    dx = x - cx
    if periodic_xy:
        dy = dy - ny * np.round(dy / ny)
        dx = dx - nx * np.round(dx / nx)
    return (dx * dx + dy * dy + dz * dz) <= radius * radius


def _icosphere_normals(n_faces: int) -> np.ndarray:
    """Unit *vertex* directions of a geodesic polyhedron with ~``n_faces`` faces.

    Start from an icosahedron (20 faces) and subdivide each face ``k`` times;
    the normals of the resulting triangles are the face normals of the
    inscribed polyhedron.  ``n_faces`` is rounded up to the nearest
    ``20 * 4**k`` (20, 80, 320, 1280 ...).
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    k = max(0, int(round(math.log(max(n_faces / 20.0, 1.0)) / math.log(4.0))))
    vlist = [v for v in verts]

    def mid(a: int, b: int, cache: dict) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = vlist[a] + vlist[b]
            vlist.append(m / np.linalg.norm(m))
            cache[key] = len(vlist) - 1
        return cache[key]

    for _ in range(k):
        cache: dict = {}
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b, cache), mid(b, c, cache), mid(c, a, cache)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces
    verts = np.asarray(vlist)
    # unique vertex directions: the polyhedron {x : max_i v_i.x <= radius}
    # then has its vertices exactly at distance `radius` (circumscribed sphere)
    dirs = np.unique(np.round(verts / np.linalg.norm(verts, axis=1, keepdims=True), 9), axis=0)
    return dirs



def _icosphere_faces(n_faces: int):
    """Unit-sphere geodesic polyhedron: (face normals, per-face apothems).

    Both refer to the polyhedron whose *vertices* lie on the unit sphere, so the
    inscribed grain is ``{x : n_j . x <= radius * h_j}`` with ``h_j = n_j . v``
    the distance from the centre to face ``j`` (``h_j <= 1``).
    """
    phi = (1 + math.sqrt(5)) / 2
    verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=float,
    )
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    k = max(0, int(round(math.log(max(n_faces / 20.0, 1.0)) / math.log(4.0))))
    vlist = [v for v in verts]

    def mid(a: int, b: int, cache: dict) -> int:
        key = (min(a, b), max(a, b))
        if key not in cache:
            m = vlist[a] + vlist[b]
            vlist.append(m / np.linalg.norm(m))
            cache[key] = len(vlist) - 1
        return cache[key]

    for _ in range(k):
        cache: dict = {}
        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b, cache), mid(b, c, cache), mid(c, a, cache)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    verts = np.asarray(vlist)
    normals, apothems = [], []
    for a, b, c in faces:
        n = np.cross(verts[b] - verts[a], verts[c] - verts[a])
        nn = np.linalg.norm(n)
        if nn < 1e-12:
            continue
        n = n / nn
        normals.append(n)
        apothems.append(float(n @ verts[a]))
    return np.array(normals), np.array(apothems)


def faceted_mask(
    nx: int,
    ny: int,
    nz: int,
    centre,
    radius: float,
    n_faces: int = 80,
    periodic_xy: bool = True,
) -> np.ndarray:
    """Near-spherical convex polyhedron inscribed in a sphere of ``radius``.

    ``{x : max_i v_i.(x - c) <= radius}`` for the vertex directions ``v_i`` of a
    geodesic polyhedron: the vertices of the grain lie exactly on the sphere of
    radius ``radius`` and its faces are inside it, so the volume is a little
    below ``4/3 pi radius**3`` (the node count is what enters ``phi``).  This is
    the "near-spherical ice crystal" of the applicability map.
    """
    z, y, x = box_grid(nx, ny, nz)
    cz, cy, cx = centre
    dx = x - cx
    dy = y - cy
    dz = z - cz
    if periodic_xy:
        dy = dy - ny * np.round(dy / ny)
        dx = dx - nx * np.round(dx / nx)
    mask = np.ones_like(dx, dtype=bool)
    normals, apothems = _icosphere_faces(n_faces)
    for n, h in zip(normals, apothems):
        # n = (nz, ny, nx); h <= 1 so the grain is inscribed in the sphere
        d = n[0] * dz + n[1] * dy + n[2] * dx
        mask &= d <= radius * h
    return mask


def spheroid_mask(
    nx: int,
    ny: int,
    nz: int,
    centre,
    a_parallel: float,
    a_perp: float,
    axis: str = "y",
    periodic_xy: bool = True,
) -> np.ndarray:
    """Spheroid with semi-axis ``a_parallel`` along ``axis``.

    ``aspect = a_parallel / a_perp``; ``axis='y'`` (the vorticity axis) with
    ``aspect < 1`` is a *platelet* whose rotation is closest to the frozen
    half-vorticity kinematics (the shape is invariant under that rotation).
    """
    z, y, x = box_grid(nx, ny, nz)
    cz, cy, cx = centre
    dx, dy, dz = x - cx, y - cy, z - cz
    if periodic_xy:
        dy = dy - ny * np.round(dy / ny)
        dx = dx - nx * np.round(dx / nx)
    d = {"x": dx, "y": dy, "z": dz}
    q = (
        (d[axis] / a_parallel) ** 2
        + sum((d[k] / a_perp) ** 2 for k in "xyz" if k != axis)
    )
    return q <= 1.0


def make_particle(
    nx: int,
    ny: int,
    nz: int,
    centre,
    radius: float,
    shape: str = "sphere",
    **kwargs,
) -> Particle:
    """Build a :class:`Particle` for the requested shape keyword."""
    cz, cy, cx = centre
    if shape == "sphere":
        m = sphere_mask(nx, ny, nz, centre, radius)
        he = radius
    elif shape == "faceted":
        n_faces = kwargs.pop("n_faces", 80)
        m = faceted_mask(nx, ny, nz, centre, radius, n_faces=n_faces)
        he = radius
    elif shape == "spheroid":
        aspect = kwargs.pop("aspect", 1.0)
        axis = kwargs.pop("axis", "y")
        m = spheroid_mask(nx, ny, nz, centre, radius, radius * aspect, axis=axis)
        he = radius
    else:
        raise ValueError(f"unknown shape {shape!r}")
    return Particle(mask=m, centre=centre, half_extent=he, shape=shape, meta=dict(kwargs))


def measure_effective_fraction(particles) -> float:
    """``phi`` from the *union* of the discrete masks divided by the box volume.

    Overlapping masks are counted once (union), so the returned fraction is the
    volume the fluid actually sees.
    """
    if not particles:
        return 0.0
    union = np.zeros_like(particles[0].mask, dtype=bool)
    for p in particles:
        union |= p.mask
    vol_box = union.size
    return float(union.sum()) / vol_box


def polydisperse_radii(n: int, mean_radius: float, spread: float, seed: int = 0) -> np.ndarray:
    """Log-normal radii with coefficient of variation ``spread``."""
    rng = np.random.default_rng(seed)
    sigma = math.sqrt(math.log(1.0 + spread * spread))
    mu = math.log(mean_radius) - 0.5 * sigma * sigma
    return np.exp(rng.normal(mu, sigma, size=n))


def place_particles(
    nx: int,
    ny: int,
    nz: int,
    phi_target: float,
    radius: float,
    shape: str = "sphere",
    *,
    polydisperse: float = 0.0,
    seed: int = 0,
    wall_margin: float = 2.0,
    min_gap: float = 1.0,
    max_attempts: int = 200000,
    **shape_kwargs,
) -> list[Particle]:
    """Random sequential addition of non-overlapping particles.

    Particles are kept ``wall_margin`` lattice units away from the two
    wall-normal walls (``z = 0`` and ``z = nz-1``) and at least
    ``radius_i + radius_j + min_gap`` apart (periodic in ``x``/``y``).
    Placement stops when the *discrete* volume fraction reaches
    ``phi_target`` or the attempt budget is exhausted; the achieved fraction is
    returned by :func:`measure_effective_fraction` so a caller never has to
    trust ``phi_target``.

    Random sequential addition jams near ``phi ~ 0.35-0.40`` for touching
    spheres; above that the placement stops early and the achieved ``phi`` is
    reported (honest partial coverage, not an interpolation).
    """
    rng = np.random.default_rng(seed)
    radii = (
        polydisperse_radii(64, radius, polydisperse, seed=seed)
        if polydisperse > 0
        else np.full(64, radius)
    )
    particles: list[Particle] = []
    vol_box = nx * ny * nz
    vol_tot = 0
    attempts = 0
    idx = 0
    while vol_tot / vol_box < phi_target and attempts < max_attempts:
        attempts += 1
        r = float(radii[idx % radii.size])
        idx += 1
        cz = rng.uniform(wall_margin + r, nz - 1 - wall_margin - r)
        cy = rng.uniform(0, ny)
        cx = rng.uniform(0, nx)
        ok = True
        for p in particles:
            dz = cz - p.centre[0]
            dy = cy - p.centre[1]
            dx = cx - p.centre[2]
            dy -= ny * round(dy / ny)
            dx -= nx * round(dx / nx)
            if dz * dz + dy * dy + dx * dx < (r + p.half_extent + min_gap) ** 2:
                ok = False
                break
        if not ok:
            continue
        part = make_particle(nx, ny, nz, (cz, cy, cx), r, shape=shape, **shape_kwargs)
        if part.volume == 0:
            continue
        particles.append(part)
        vol_tot += part.volume
    return particles


def rigid_body_velocity(
    mask: np.ndarray,
    centre,
    shear_rate: float,
    z_mid: float,
    Omega_y: float | None = None,
):
    """Imposed rigid-body velocity on the nodes of ``mask``.

    ``u = gamma_dot (z_c - z_mid) e_x + Omega x (x - x_c)`` with
    ``Omega = Omega_y e_y`` (default ``Omega_y = gamma_dot/2``, the
    torque-free half-vorticity rotation of a sphere in simple shear).

    Returns ``(usx, usy, usz)`` arrays of the shape of ``mask`` (zero outside).
    """
    nz, ny, nx = mask.shape
    z, y, x = box_grid(nx, ny, nz)
    cz, cy, cx = centre
    Om = 0.5 * shear_rate if Omega_y is None else Omega_y
    usx = np.zeros_like(z)
    usy = np.zeros_like(z)
    usz = np.zeros_like(z)
    usx[mask] = shear_rate * (cz - z_mid) + Om * (z[mask] - cz)
    usz[mask] = -Om * (x[mask] - cx)
    return usx, usy, usz