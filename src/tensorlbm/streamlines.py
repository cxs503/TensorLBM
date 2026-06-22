"""Streamline, pathline, and streakline tracing for LBM flow fields.

Provides numerical integration of steady and unsteady velocity fields using
Runge–Kutta 4th-order (RK4) integration.  Results are returned as polyline
lists suitable for VTK / JSON export.

Functions
---------
:func:`trace_streamlines_2d`
    Integrate steady 2-D streamlines from seed points using RK4.
:func:`trace_streamlines_3d`
    Integrate steady 3-D streamlines from seed points using RK4.
:func:`seed_points_uniform_2d`
    Generate a uniform grid of seed points in 2-D.
:func:`seed_points_uniform_3d`
    Generate a uniform grid of seed points in 3-D.
:func:`seed_points_line_2d`
    Generate seed points along a line (e.g., an inlet plane) in 2-D.
:func:`seed_points_line_3d`
    Generate seed points along a line/plane in 3-D.
:func:`compute_residence_time_2d`
    Compute approximate residence-time field by seeding from inlet.
:func:`streamlines_to_dict`
    Serialise streamline polylines to a JSON-friendly dict.

References
----------
Stalling, D., & Hege, H.-C. (1995). Fast and resolution independent line
    integral convolution. *Proc. SIGGRAPH*, 249–256.
McLoughlin, T., Laramee, R. S., Peikert, R., Post, F. H., & Chen, M. (2010).
    Over two decades of integration-based, geometric flow visualization.
    *Computer Graphics Forum*, 29(6), 1807–1829.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Streamline:
    """A single traced streamline (or pathline / streakline).

    Attributes:
        points: List of ``(x, y)`` or ``(x, y, z)`` coordinates.
        scalars: Optional list of per-point scalar values (e.g., velocity
                 magnitude, pressure, vorticity).
        length:  Arc-length of the polyline in lattice units.
        steps:   Number of integration steps taken.
    """
    points: list[tuple[float, ...]] = field(default_factory=list)
    scalars: list[float] = field(default_factory=list)
    length: float = 0.0
    steps: int = 0


# ---------------------------------------------------------------------------
# Bilinear / trilinear interpolation helpers
# ---------------------------------------------------------------------------

def _interp_2d(
    field_2d: torch.Tensor,  # shape (ny, nx)
    x: torch.Tensor,         # (N,) float, in [0, nx-1]
    y: torch.Tensor,         # (N,) float, in [0, ny-1]
) -> torch.Tensor:
    """Bilinear interpolation over a 2-D scalar field."""
    ny, nx = field_2d.shape
    # Normalise to [-1, 1] for grid_sample
    xn = 2.0 * x / (nx - 1) - 1.0
    yn = 2.0 * y / (ny - 1) - 1.0
    grid = torch.stack([xn, yn], dim=-1).view(1, 1, -1, 2)
    val = F.grid_sample(
        field_2d.view(1, 1, ny, nx).float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return val.view(-1)


def _interp_3d(
    field_3d: torch.Tensor,  # shape (nz, ny, nx)
    x: torch.Tensor,         # (N,) float, in [0, nx-1]
    y: torch.Tensor,         # (N,) float, in [0, ny-1]
    z: torch.Tensor,         # (N,) float, in [0, nz-1]
) -> torch.Tensor:
    """Trilinear interpolation over a 3-D scalar field."""
    nz, ny, nx = field_3d.shape
    xn = 2.0 * x / (nx - 1) - 1.0
    yn = 2.0 * y / (ny - 1) - 1.0
    zn = 2.0 * z / (nz - 1) - 1.0
    grid = torch.stack([xn, yn, zn], dim=-1).view(1, 1, 1, -1, 3)
    val = F.grid_sample(
        field_3d.view(1, 1, nz, ny, nx).float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return val.view(-1)


# ---------------------------------------------------------------------------
# 2-D Streamline tracing
# ---------------------------------------------------------------------------

def trace_streamlines_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    seeds: list[tuple[float, float]],
    step_size: float = 0.5,
    max_steps: int = 2000,
    mask: torch.Tensor | None = None,
    scalar_field: torch.Tensor | None = None,
    bidirectional: bool = False,
) -> list[Streamline]:
    """Trace 2-D streamlines using 4th-order Runge–Kutta integration.

    The velocity field is interpolated bilinearly at non-integer positions.
    Integration terminates when a particle leaves the domain, enters a solid
    cell (mask), or exceeds *max_steps*.

    Args:
        ux:           x-velocity field, shape ``(ny, nx)``.
        uy:           y-velocity field, shape ``(ny, nx)``.
        seeds:        List of ``(x0, y0)`` seed coordinates in lattice units.
        step_size:    RK4 integration step in lattice units.
        max_steps:    Maximum number of steps per streamline.
        mask:         Boolean solid mask ``(ny, nx)``; ``True`` = solid.
                      Integration stops when a particle enters a solid cell.
        scalar_field: Optional scalar field ``(ny, nx)`` sampled along
                      each streamline (e.g., pressure, vorticity).
        bidirectional: If ``True``, traces in both forward and backward
                      directions and concatenates the results.

    Returns:
        List of :class:`Streamline` objects, one per seed point.
    """
    ny, nx = ux.shape
    device = ux.device
    lines: list[Streamline] = []

    for x0, y0 in seeds:
        sl = _rk4_integrate_2d(
            ux, uy, float(x0), float(y0),
            step_size, max_steps, nx, ny, mask, scalar_field, forward=True,
        )
        if bidirectional:
            sl_back = _rk4_integrate_2d(
                ux, uy, float(x0), float(y0),
                step_size, max_steps, nx, ny, mask, scalar_field, forward=False,
            )
            # Reverse the backward segment and prepend it (excluding seed point)
            pts_back = list(reversed(sl_back.points[1:]))
            sca_back = list(reversed(sl_back.scalars[1:]))
            merged = Streamline(
                points=pts_back + sl.points,
                scalars=sca_back + sl.scalars,
                length=sl.length + sl_back.length,
                steps=sl.steps + sl_back.steps,
            )
            lines.append(merged)
        else:
            lines.append(sl)

    return lines


def _rk4_integrate_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    x0: float,
    y0: float,
    h: float,
    max_steps: int,
    nx: int,
    ny: int,
    mask: torch.Tensor | None,
    scalar_field: torch.Tensor | None,
    forward: bool = True,
) -> Streamline:
    """Single-direction RK4 integration (2-D)."""
    sign = 1.0 if forward else -1.0
    sl = Streamline()
    x, y = x0, y0
    sl.points.append((x, y))

    if scalar_field is not None:
        xT = torch.tensor([x], device=ux.device)
        yT = torch.tensor([y], device=ux.device)
        sl.scalars.append(float(_interp_2d(scalar_field, xT, yT)[0]))

    for _ in range(max_steps):
        # RK4 stages
        k1x, k1y = _vel_2d(ux, uy, x, y, nx, ny)
        k2x, k2y = _vel_2d(ux, uy, x + sign * 0.5 * h * k1x, y + sign * 0.5 * h * k1y, nx, ny)
        k3x, k3y = _vel_2d(ux, uy, x + sign * 0.5 * h * k2x, y + sign * 0.5 * h * k2y, nx, ny)
        k4x, k4y = _vel_2d(ux, uy, x + sign * h * k3x, y + sign * h * k3y, nx, ny)

        dx = sign * h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        dy = sign * h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0

        x_new = x + dx
        y_new = y + dy

        # Domain check
        if not (0.0 <= x_new < nx and 0.0 <= y_new < ny):
            break

        # Solid mask check
        if mask is not None:
            ix, iy = int(x_new + 0.5), int(y_new + 0.5)
            ix = max(0, min(nx - 1, ix))
            iy = max(0, min(ny - 1, iy))
            if mask[iy, ix]:
                break

        # Stagnation check
        speed = (dx * dx + dy * dy) ** 0.5
        if speed < 1e-10:
            break

        sl.length += speed
        x, y = x_new, y_new
        sl.points.append((x, y))
        sl.steps += 1

        if scalar_field is not None:
            xT = torch.tensor([x], device=ux.device)
            yT = torch.tensor([y], device=ux.device)
            sl.scalars.append(float(_interp_2d(scalar_field, xT, yT)[0]))

    return sl


def _vel_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    x: float,
    y: float,
    nx: int,
    ny: int,
) -> tuple[float, float]:
    """Bilinearly interpolated velocity at (x, y)."""
    x_c = max(0.0, min(float(nx - 1), x))
    y_c = max(0.0, min(float(ny - 1), y))
    xT = torch.tensor([x_c], device=ux.device)
    yT = torch.tensor([y_c], device=ux.device)
    vx = float(_interp_2d(ux, xT, yT)[0])
    vy = float(_interp_2d(uy, xT, yT)[0])
    return vx, vy


# ---------------------------------------------------------------------------
# 3-D Streamline tracing
# ---------------------------------------------------------------------------

def trace_streamlines_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    seeds: list[tuple[float, float, float]],
    step_size: float = 0.5,
    max_steps: int = 2000,
    mask: torch.Tensor | None = None,
    scalar_field: torch.Tensor | None = None,
    bidirectional: bool = False,
) -> list[Streamline]:
    """Trace 3-D streamlines using 4th-order Runge–Kutta integration.

    Args:
        ux:           x-velocity field, shape ``(nz, ny, nx)``.
        uy:           y-velocity field, shape ``(nz, ny, nx)``.
        uz:           z-velocity field, shape ``(nz, ny, nx)``.
        seeds:        List of ``(x0, y0, z0)`` seed coordinates.
        step_size:    RK4 step size in lattice units.
        max_steps:    Maximum steps per streamline.
        mask:         Boolean solid mask ``(nz, ny, nx)``; ``True`` = solid.
        scalar_field: Optional scalar ``(nz, ny, nx)`` sampled along line.
        bidirectional: Trace forward and backward and concatenate.

    Returns:
        List of :class:`Streamline` objects, one per seed.
    """
    nz, ny, nx = ux.shape
    lines: list[Streamline] = []

    for x0, y0, z0 in seeds:
        sl = _rk4_integrate_3d(
            ux, uy, uz, float(x0), float(y0), float(z0),
            step_size, max_steps, nx, ny, nz, mask, scalar_field, forward=True,
        )
        if bidirectional:
            sl_back = _rk4_integrate_3d(
                ux, uy, uz, float(x0), float(y0), float(z0),
                step_size, max_steps, nx, ny, nz, mask, scalar_field, forward=False,
            )
            pts_back = list(reversed(sl_back.points[1:]))
            sca_back = list(reversed(sl_back.scalars[1:]))
            merged = Streamline(
                points=pts_back + sl.points,
                scalars=sca_back + sl.scalars,
                length=sl.length + sl_back.length,
                steps=sl.steps + sl_back.steps,
            )
            lines.append(merged)
        else:
            lines.append(sl)

    return lines


def _rk4_integrate_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    x0: float,
    y0: float,
    z0: float,
    h: float,
    max_steps: int,
    nx: int,
    ny: int,
    nz: int,
    mask: torch.Tensor | None,
    scalar_field: torch.Tensor | None,
    forward: bool = True,
) -> Streamline:
    """Single-direction RK4 integration (3-D)."""
    sign = 1.0 if forward else -1.0
    sl = Streamline()
    x, y, z = x0, y0, z0
    sl.points.append((x, y, z))

    if scalar_field is not None:
        xT = torch.tensor([x], device=ux.device)
        yT = torch.tensor([y], device=ux.device)
        zT = torch.tensor([z], device=ux.device)
        sl.scalars.append(float(_interp_3d(scalar_field, xT, yT, zT)[0]))

    for _ in range(max_steps):
        k1x, k1y, k1z = _vel_3d(ux, uy, uz, x, y, z, nx, ny, nz)
        k2x, k2y, k2z = _vel_3d(
            ux, uy, uz,
            x + sign * 0.5 * h * k1x,
            y + sign * 0.5 * h * k1y,
            z + sign * 0.5 * h * k1z,
            nx, ny, nz,
        )
        k3x, k3y, k3z = _vel_3d(
            ux, uy, uz,
            x + sign * 0.5 * h * k2x,
            y + sign * 0.5 * h * k2y,
            z + sign * 0.5 * h * k2z,
            nx, ny, nz,
        )
        k4x, k4y, k4z = _vel_3d(
            ux, uy, uz,
            x + sign * h * k3x,
            y + sign * h * k3y,
            z + sign * h * k3z,
            nx, ny, nz,
        )

        dx = sign * h * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0
        dy = sign * h * (k1y + 2.0 * k2y + 2.0 * k3y + k4y) / 6.0
        dz = sign * h * (k1z + 2.0 * k2z + 2.0 * k3z + k4z) / 6.0

        x_new, y_new, z_new = x + dx, y + dy, z + dz

        if not (0.0 <= x_new < nx and 0.0 <= y_new < ny and 0.0 <= z_new < nz):
            break

        if mask is not None:
            ix = max(0, min(nx - 1, int(x_new + 0.5)))
            iy = max(0, min(ny - 1, int(y_new + 0.5)))
            iz = max(0, min(nz - 1, int(z_new + 0.5)))
            if mask[iz, iy, ix]:
                break

        speed = (dx * dx + dy * dy + dz * dz) ** 0.5
        if speed < 1e-10:
            break

        sl.length += speed
        x, y, z = x_new, y_new, z_new
        sl.points.append((x, y, z))
        sl.steps += 1

        if scalar_field is not None:
            xT = torch.tensor([x], device=ux.device)
            yT = torch.tensor([y], device=ux.device)
            zT = torch.tensor([z], device=ux.device)
            sl.scalars.append(float(_interp_3d(scalar_field, xT, yT, zT)[0]))

    return sl


def _vel_3d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    x: float,
    y: float,
    z: float,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[float, float, float]:
    """Trilinearly interpolated velocity at (x, y, z)."""
    x_c = max(0.0, min(float(nx - 1), x))
    y_c = max(0.0, min(float(ny - 1), y))
    z_c = max(0.0, min(float(nz - 1), z))
    xT = torch.tensor([x_c], device=ux.device)
    yT = torch.tensor([y_c], device=ux.device)
    zT = torch.tensor([z_c], device=ux.device)
    vx = float(_interp_3d(ux, xT, yT, zT)[0])
    vy = float(_interp_3d(uy, xT, yT, zT)[0])
    vz = float(_interp_3d(uz, xT, yT, zT)[0])
    return vx, vy, vz


# ---------------------------------------------------------------------------
# Seed-point generators
# ---------------------------------------------------------------------------

def seed_points_uniform_2d(
    nx: int, ny: int,
    n_x: int = 8, n_y: int = 8,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Generate a uniform grid of seed points in a 2-D domain.

    Args:
        nx, ny:   Domain extents in lattice units.
        n_x, n_y: Number of seeds along x and y.
        x_range:  ``(x_min, x_max)`` in lattice units; defaults to full domain.
        y_range:  ``(y_min, y_max)`` in lattice units; defaults to full domain.

    Returns:
        List of ``(x, y)`` seed coordinates.
    """
    x0, x1 = x_range if x_range is not None else (1.0, nx - 2.0)
    y0, y1 = y_range if y_range is not None else (1.0, ny - 2.0)
    seeds = []
    for iy in range(n_y):
        y = y0 + (y1 - y0) * iy / max(n_y - 1, 1)
        for ix in range(n_x):
            x = x0 + (x1 - x0) * ix / max(n_x - 1, 1)
            seeds.append((x, y))
    return seeds


def seed_points_line_2d(
    x_seed: float,
    ny: int,
    n_seeds: int = 16,
    y_range: tuple[float, float] | None = None,
) -> list[tuple[float, float]]:
    """Generate seed points along a vertical line at *x_seed* (2-D).

    Args:
        x_seed:   x-coordinate of the seeding plane in lattice units.
        ny:       Domain height in lattice units.
        n_seeds:  Number of seed points.
        y_range:  ``(y_min, y_max)``; defaults to interior of domain.

    Returns:
        List of ``(x, y)`` seed coordinates.
    """
    y0, y1 = y_range if y_range is not None else (1.0, ny - 2.0)
    seeds = []
    for i in range(n_seeds):
        y = y0 + (y1 - y0) * i / max(n_seeds - 1, 1)
        seeds.append((x_seed, y))
    return seeds


def seed_points_uniform_3d(
    nx: int, ny: int, nz: int,
    n_x: int = 4, n_y: int = 4, n_z: int = 4,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> list[tuple[float, float, float]]:
    """Generate a uniform 3-D seed grid."""
    x0, x1 = x_range if x_range is not None else (1.0, nx - 2.0)
    y0, y1 = y_range if y_range is not None else (1.0, ny - 2.0)
    z0, z1 = z_range if z_range is not None else (1.0, nz - 2.0)
    seeds = []
    for iz in range(n_z):
        z = z0 + (z1 - z0) * iz / max(n_z - 1, 1)
        for iy in range(n_y):
            y = y0 + (y1 - y0) * iy / max(n_y - 1, 1)
            for ix in range(n_x):
                x = x0 + (x1 - x0) * ix / max(n_x - 1, 1)
                seeds.append((x, y, z))
    return seeds


def seed_points_line_3d(
    x_seed: float,
    ny: int, nz: int,
    n_y: int = 8, n_z: int = 8,
    y_range: tuple[float, float] | None = None,
    z_range: tuple[float, float] | None = None,
) -> list[tuple[float, float, float]]:
    """Generate seed points on an inlet yz-plane at *x_seed* (3-D)."""
    y0, y1 = y_range if y_range is not None else (1.0, ny - 2.0)
    z0, z1 = z_range if z_range is not None else (1.0, nz - 2.0)
    seeds = []
    for iz in range(n_z):
        z = z0 + (z1 - z0) * iz / max(n_z - 1, 1)
        for iy in range(n_y):
            y = y0 + (y1 - y0) * iy / max(n_y - 1, 1)
            seeds.append((x_seed, y, z))
    return seeds


# ---------------------------------------------------------------------------
# Residence-time field (2-D)
# ---------------------------------------------------------------------------

def compute_residence_time_2d(
    ux: torch.Tensor,
    uy: torch.Tensor,
    x_inlet: float,
    n_seeds: int = 32,
    step_size: float = 0.5,
    max_steps: int = 5000,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Approximate residence-time field by seeding from the inlet plane.

    Each streamline is traced forward; the time elapsed at a cell is estimated
    by counting how many streamlines pass through it (weighted by step_size /
    speed).  The result is an approximation of the flow-age / residence time.

    Args:
        ux, uy:   Velocity fields, shape ``(ny, nx)``.
        x_inlet:  x-coordinate of the inlet seeding plane.
        n_seeds:  Number of seeds along the inlet.
        step_size: RK4 step size.
        max_steps: Maximum steps per streamline.
        mask:     Solid mask.

    Returns:
        Residence-time field, shape ``(ny, nx)``.
    """
    ny, nx = ux.shape
    seeds = seed_points_line_2d(x_inlet, ny, n_seeds)
    lines = trace_streamlines_2d(ux, uy, seeds, step_size, max_steps, mask)

    rt = torch.zeros(ny, nx, device=ux.device, dtype=ux.dtype)
    cnt = torch.zeros(ny, nx, device=ux.device, dtype=ux.dtype)

    for sl in lines:
        for step_idx, (px, py) in enumerate(sl.points):
            ix = max(0, min(nx - 1, int(px + 0.5)))
            iy = max(0, min(ny - 1, int(py + 0.5)))
            t = step_idx * step_size
            rt[iy, ix] += t
            cnt[iy, ix] += 1.0

    valid = cnt > 0
    rt[valid] /= cnt[valid]
    return rt


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def streamlines_to_dict(lines: list[Streamline]) -> dict:
    """Convert a list of streamlines to a JSON-serialisable dictionary.

    Returns:
        Dictionary with keys:

        ``n_lines``
            Number of streamlines.
        ``lines``
            List of per-streamline dicts with keys ``points``, ``scalars``,
            ``length``, ``steps``.
    """
    return {
        "n_lines": len(lines),
        "lines": [
            {
                "points": [list(p) for p in sl.points],
                "scalars": sl.scalars,
                "length": sl.length,
                "steps": sl.steps,
            }
            for sl in lines
        ],
    }


__all__ = [
    "Streamline",
    "trace_streamlines_2d",
    "trace_streamlines_3d",
    "seed_points_uniform_2d",
    "seed_points_line_2d",
    "seed_points_uniform_3d",
    "seed_points_line_3d",
    "compute_residence_time_2d",
    "streamlines_to_dict",
]
