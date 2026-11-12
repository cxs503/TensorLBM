"""2D Lid-Driven Cavity benchmark using D2Q9 LBM.

The lid-driven cavity is the single most commonly used benchmark in
computational fluid dynamics.  A square domain is enclosed on all four
sides by no-slip walls; the top wall (the "lid") slides at constant
velocity *u_lid* in the +x direction, driving a large primary recirculation
vortex and (at higher Re) smaller corner eddies.

Boundary conditions
-------------------
* **Top wall** (y = ny−1): moving lid at u_x = u_lid, u_y = 0,
  implemented with the Zou/He (1997) analytical BC for interior cells.
  Corner cells are treated with simple bounce-back.
* **Bottom, left, right walls**: stationary no-slip bounce-back.

Validation reference
--------------------
Ghia, U., Ghia, K. N., & Shin, C. T. (1982). High-Re solutions for
incompressible flow using the Navier-Stokes equations and a multigrid
method.  Journal of Computational Physics, **48**(3), 387-411.

Tabulated velocity profiles at Re = 100, 400, 1000 along the vertical
and horizontal centrelines are stored in :data:`GHIA_RE100`,
:data:`GHIA_RE400`, and :data:`GHIA_RE1000`.

Outputs
-------
* ``run_metadata.json``  – config + diagnostics + Ghia comparison error
* ``ghia_comparison.csv`` – centreline profiles and Ghia reference values
* ``flow_step_XXXXXX.png`` – velocity-magnitude snapshots every
  ``output_interval`` steps
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import torch

from .boundaries import bounce_back_cells
from .config_io import load_config_json, save_config_json
from .cylinder_flow import _maybe_compile
from .d2q9 import equilibrium, macroscopic
from .logging_config import configure_logging, logger
from .solver import collide_bgk, collide_mrt, stream
from .utils import (
    DiagnosticPoint,
    flow_step_image_path,
    get_reproducibility_metadata,
    prepare_run_dir,
    resolve_device,
    write_legacy_snapshot_alias,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ghia et al. (1982) reference data
# ---------------------------------------------------------------------------

# u/u_lid along the vertical centreline x = 0.5 (Re = 100, 129×129 grid)
# Keys: "y" – y/H positions, "u" – u/u_lid values
GHIA_RE100: dict[str, list[float]] = {
    "y": [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ],
    "u": [
        1.00000,
        0.84123,
        0.78871,
        0.73722,
        0.68717,
        0.23151,
        0.00332,
        -0.13641,
        -0.20581,
        -0.21090,
        -0.15662,
        -0.10150,
        -0.06434,
        -0.04775,
        -0.04192,
        -0.03717,
        0.00000,
    ],
    # v/u_lid along the horizontal centreline y = 0.5
    "x": [
        0.0000,
        0.0625,
        0.0703,
        0.0781,
        0.0938,
        0.1563,
        0.2266,
        0.2344,
        0.5000,
        0.8047,
        0.8594,
        0.9063,
        0.9453,
        0.9531,
        0.9609,
        0.9688,
        1.0000,
    ],
    "v": [
        0.00000,
        0.09233,
        0.10091,
        0.10890,
        0.12317,
        0.16077,
        0.17507,
        0.17527,
        0.05454,
        -0.24533,
        -0.22445,
        -0.16914,
        -0.10313,
        -0.08864,
        -0.07391,
        -0.05906,
        0.00000,
    ],
}

# u/u_lid along vertical centreline (Re = 400)
GHIA_RE400: dict[str, list[float]] = {
    "y": [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ],
    "u": [
        1.00000,
        0.75837,
        0.68439,
        0.61756,
        0.55892,
        0.29093,
        0.16256,
        0.02135,
        -0.11477,
        -0.17119,
        -0.32726,
        -0.24299,
        -0.14612,
        -0.10338,
        -0.09266,
        -0.08186,
        0.00000,
    ],
    "x": [
        0.0000,
        0.0625,
        0.0703,
        0.0781,
        0.0938,
        0.1563,
        0.2266,
        0.2344,
        0.5000,
        0.8047,
        0.8594,
        0.9063,
        0.9453,
        0.9531,
        0.9609,
        0.9688,
        1.0000,
    ],
    "v": [
        0.00000,
        0.18360,
        0.19713,
        0.20920,
        0.22980,
        0.28003,
        0.30174,
        0.30203,
        0.05186,
        -0.38598,
        -0.44993,
        -0.38598,
        -0.22847,
        -0.19254,
        -0.15663,
        -0.12146,
        0.00000,
    ],
}

# u/u_lid along vertical centreline (Re = 1000)
GHIA_RE1000: dict[str, list[float]] = {
    "y": [
        1.0000,
        0.9766,
        0.9688,
        0.9609,
        0.9531,
        0.8516,
        0.7344,
        0.6172,
        0.5000,
        0.4531,
        0.2813,
        0.1719,
        0.1016,
        0.0703,
        0.0625,
        0.0547,
        0.0000,
    ],
    "u": [
        1.00000,
        0.65928,
        0.57492,
        0.51117,
        0.46604,
        0.33304,
        0.18719,
        0.05702,
        -0.06080,
        -0.10648,
        -0.27805,
        -0.38289,
        -0.29730,
        -0.22220,
        -0.20196,
        -0.18109,
        0.00000,
    ],
    "x": [
        0.0000,
        0.0625,
        0.0703,
        0.0781,
        0.0938,
        0.1563,
        0.2266,
        0.2344,
        0.5000,
        0.8047,
        0.8594,
        0.9063,
        0.9453,
        0.9531,
        0.9609,
        0.9688,
        1.0000,
    ],
    "v": [
        0.00000,
        0.27485,
        0.29012,
        0.30353,
        0.32627,
        0.37095,
        0.33075,
        0.32235,
        0.02526,
        -0.31966,
        -0.42665,
        -0.51550,
        -0.39188,
        -0.33714,
        -0.27669,
        -0.21388,
        0.00000,
    ],
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LidDrivenCavityConfig:
    """Configuration for the 2D Lid-Driven Cavity benchmark.

    The domain is a square of ``nx × nx`` cells.  The top row slides at
    ``u_lid`` in the x-direction; all other walls are no-slip.

    Args:
        nx: Number of grid cells per side (square domain, so ny = nx).
        u_lid: Lid velocity in lattice units.
        re: Reynolds number Re = u_lid * nx / nu.
        n_steps: Total number of time steps.
        output_interval: Save a snapshot every this many steps.
        output_root: Root directory for output files.
        run_name: Override the auto-generated run folder name.
        seed: Random seed for reproducibility.
        device: Torch device string (``"cpu"`` or ``"cuda"``).
        overwrite: If True, remove an existing run directory before writing.
        use_compile: If True, wrap hot-path kernels with ``torch.compile``.
        ghia_metric_convention: Which index->coordinate map is used for the
            headline ``ghia_errors`` RMSE written to ``run_metadata.json``:
            ``"legacy_w0"`` (historical default: walls on the first/last node,
            keeps previously published numbers comparable),
            ``"symmetric_w+0.5"`` (nodes at cell centres) or
            ``"recipe_planes"`` (the map implied by this solver's boundary
            recipe, see :func:`cavity_wall_planes`).  All three are always
            reported under ``ghia_metric``; this only selects the headline.
    """

    nx: int = 128
    u_lid: float = 0.1
    re: float = 100.0
    n_steps: int = 10000
    output_interval: int = 2000
    output_root: Path = Path("outputs")
    run_name: str | None = None
    seed: int = 0
    device: str = "cpu"
    overwrite: bool = False
    use_compile: bool = False
    ghia_metric_convention: str = "legacy_w0"

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_root", Path(self.output_root))
        object.__setattr__(self, "device", self.device.lower())

    @property
    def wall_planes(self) -> dict[str, float]:
        """Wall-plane geometry implied by the boundary recipe (see :func:`cavity_wall_planes`)."""
        return cavity_wall_planes(self.nx, self.ny)

    @property
    def ny(self) -> int:
        """Height equals width for the square cavity."""
        return self.nx

    @property
    def nu(self) -> float:
        """Kinematic viscosity derived from Re = u_lid * nx / nu."""
        return self.u_lid * self.nx / self.re

    @property
    def tau(self) -> float:
        """BGK relaxation time τ = 3ν + 0.5."""
        return 3.0 * self.nu + 0.5

    def validate(self) -> None:
        if self.nx < 8:
            msg = "nx must be >= 8"
            raise ValueError(msg)
        if self.n_steps < 1:
            msg = "n_steps must be >= 1"
            raise ValueError(msg)
        if self.output_interval < 1:
            msg = "output_interval must be >= 1"
            raise ValueError(msg)
        if self.u_lid <= 0.0 or self.re <= 0.0:
            msg = "u_lid and re must be > 0"
            raise ValueError(msg)
        if self.tau <= 0.5:
            msg = f"Invalid tau={self.tau:.4f}; increase re or reduce u_lid/nx"
            raise ValueError(msg)
        if self.ghia_metric_convention not in GHIA_METRIC_CONVENTIONS:
            msg = (
                f"ghia_metric_convention={self.ghia_metric_convention!r} not in "
                f"{GHIA_METRIC_CONVENTIONS}"
            )
            raise ValueError(msg)

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        re_label = str(int(self.re)) if float(self.re).is_integer() else f"{self.re:g}"
        return f"nx{self.nx}_re{re_label}_uin{self.u_lid:.3f}_steps{self.n_steps}"

    def save(self, path: str | Path) -> Path:
        """Save this config to a JSON file."""
        return save_config_json(self, path)

    @classmethod
    def load(cls, path: str | Path) -> LidDrivenCavityConfig:
        """Load a :class:`LidDrivenCavityConfig` from a JSON file."""
        return load_config_json(cls, path)


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------


def make_cavity_wall_mask(
    ny: int,
    nx: int,
    device: torch.device,
    include_top: bool = True,
) -> torch.Tensor:
    """Create a Boolean mask for cavity walls.

    Args:
        ny: Number of rows.
        nx: Number of columns.
        device: Target device.
        include_top: If False, the top wall (lid) is excluded from the
            mask so that it is handled by Zou/He BC alone.

    Returns:
        Boolean tensor of shape ``(ny, nx)`` with wall cells set to ``True``.
    """
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True  # bottom wall
    if include_top:
        mask[-1, :] = True  # top wall (lid row)
    mask[:, 0] = True  # left wall
    mask[:, -1] = True  # right wall
    return mask


def zou_he_moving_lid(f: torch.Tensor, u_lid: float) -> torch.Tensor:
    """Zou/He moving-wall BC for the top row (y = ny-1).

    Prescribes ``ux = u_lid``, ``uy = 0`` at the interior top-wall cells
    (x = 1 to nx-2).  Corner cells (x = 0 and x = nx-1) are left unchanged
    (they are handled by the bounce-back pass that precedes this call).

    D2Q9 direction convention (this module uses the same as :mod:`d2q9`):
    ``0=(0,0), 1=(E), 2=(N), 3=(W), 4=(S), 5=(NE), 6=(NW), 7=(SW), 8=(SE)``

    At the top wall the unknown populations heading *into* the domain are
    f4 (S), f7 (SW), f8 (SE).  Using the Zou & He (1997) analytical
    solution with ``uy = 0``:

    .. math::

        \\rho &= f_0 + f_1 + 2f_2 + f_3 + 2f_5 + 2f_6 \\\\
        f_4 &= f_2 \\\\
        f_7 &= f_5 + \\tfrac{1}{2}(f_1 - f_3) - \\tfrac{1}{2}\\rho\\,u_{lid} \\\\
        f_8 &= f_6 - \\tfrac{1}{2}(f_1 - f_3) + \\tfrac{1}{2}\\rho\\,u_{lid}

    Args:
        f: Distribution tensor of shape ``(9, ny, nx)``.
        u_lid: Lid velocity (x-direction).

    Returns:
        Updated distribution tensor (same shape).
    """
    # All cells of the top row (including corners)
    f0 = f[0, -1, :]
    f1 = f[1, -1, :]
    f2 = f[2, -1, :]
    f3 = f[3, -1, :]
    f5 = f[5, -1, :]
    f6 = f[6, -1, :]

    rho = f0 + f1 + 2.0 * f2 + f3 + 2.0 * f5 + 2.0 * f6

    f_new = f.clone()
    f_new[4, -1, :] = f2
    f_new[7, -1, :] = f5 + 0.5 * (f1 - f3) - 0.5 * rho * u_lid
    f_new[8, -1, :] = f6 - 0.5 * (f1 - f3) + 0.5 * rho * u_lid
    return f_new


# ---------------------------------------------------------------------------
# Ghia comparison
# ---------------------------------------------------------------------------


def centreline_coordinates(n: int, wall_offset: float = 0.0) -> "list[float]":
    """Normalised node coordinates used to compare against Ghia (1982).

    The Ghia table normalises by the cavity side length ``H``, with
    ``y = 0`` on the bottom wall and ``y = 1`` on the moving lid.  A raw
    LBM field only carries node indices, so a *convention* is needed to
    map index ``j`` to a physical position; :func:`compare_ghia` used to
    hard-wire the "walls sit on the first/last node" convention
    (``wall_offset = 0``).  Half-way bounce-back walls and Zou/He
    boundary nodes do **not** sit on fluid nodes, so that default is only
    one of several defensible maps, and the choice moves the reported
    RMSE by an amount comparable to the acceptance gates (see
    :func:`ghia_coordinate_sensitivity`).

    The mapping implemented here is

    .. math::  y_j = \frac{j + w}{n - 1 + 2 w}, \qquad j = 0 \ldots n-1

    * ``w = 0.0``  – node 0 is ``y = 0`` and node ``n-1`` is ``y = 1``
      (walls on boundary nodes; the historical default).
    * ``w = 0.5``  – nodes sit at the centres of ``n`` equal cells
      (``y_j = (j + 1/2)/n``), i.e. the wall planes lie half a cell
      outside the node lattice — the convention of half-way bounce-back
      walls (cf. ``d3q27_mrt_physical_validation`` which uses the same
      ``wall_offset = 0.5`` idea).
    * ``w = -0.5`` – wall planes lie half a cell *inside* the lattice
      (solid wall nodes occupy rows 0 and ``n-1``).

    Args:
        n: Number of nodes along the axis (``n >= 2``).
        wall_offset: Wall position in cell units relative to the node
            lattice; see above.

    Returns:
        List of ``n`` normalised coordinates, monotonically increasing.

    Raises:
        ValueError: If ``n < 2`` or the mapping is degenerate
            (``n - 1 + 2 * wall_offset <= 0``).
    """
    import numpy as np

    if int(n) < 2:
        msg = f"centreline_coordinates needs n >= 2, got {n}"
        raise ValueError(msg)
    span = float(n - 1) + 2.0 * float(wall_offset)
    if span <= 0.0:
        msg = f"degenerate wall_offset={wall_offset} for n={n}"
        raise ValueError(msg)
    return [float(v) for v in (np.arange(int(n)) + float(wall_offset)) / span]


# ---------------------------------------------------------------------------
# Metric geometry implied by the boundary recipe
# ---------------------------------------------------------------------------

#: Boundary recipe used by :func:`run_lid_driven_cavity`.
#:
#: * bottom / left / right walls: solid nodes (``make_cavity_wall_mask``) relaxed
#:   towards the mirror state by :func:`bounce_back_cells`  ->  the no-slip plane
#:   sits **half a lattice cell outside** the solid node (index ``0.5`` from it);
#: * top wall: :func:`zou_he_moving_lid` imposes ``ux = u_lid`` on the *fluid*
#:   node row ``ny - 1``  ->  the moving wall plane coincides with that node.
#: Coordinate conventions supported by :func:`ghia_metric_report`.
GHIA_METRIC_CONVENTIONS = ("legacy_w0", "symmetric_w+0.5", "recipe_planes")

CAVITY_BOUNDARY_RECIPE = (
    "bounce_back_solid_rows(bottom,left,right)@half_cell_outside"
    "+zou_he_moving_lid(top_fluid_row)"
)


def cavity_wall_planes(nx: int, ny: int | None = None) -> dict[str, float]:
    """Wall-plane geometry (node-index units) implied by the cavity BC recipe.

    The Ghia (1982) reference normalises by a *square* cavity of side ``H``
    with walls at ``0`` and ``1``.  The discretisation used by
    :func:`run_lid_driven_cavity` does **not** place its four wall planes on
    the four node lines, and it does not place them symmetrically:

    ========  =========================================  ==================
    wall      physical plane (node index)                extent
    ========  =========================================  ==================
    bottom    ``0.5``  (half-way bounce-back off row 0)   --
    lid       ``ny - 1`` (Zou/He node row)                --
    left      ``0.5``  (half-way bounce-back off col 0)   --
    right     ``nx - 1.5`` (half-way bounce-back col nx-1)--
    ========  =========================================  ==================

    Hence the solver's effective cavity is ``H_x = nx - 2`` wide and
    ``H_y = ny - 1.5`` tall: *not* square, with an aspect ratio
    ``H_y / H_x = (ny - 1.5) / (nx - 2)`` that is ``1.19 %`` at ``nx = 128``.
    The Reynolds number on the other hand is built from ``Re = u_lid * nx / nu``
    in :attr:`LidDrivenCavityConfig.nu`, i.e. from ``L = nx`` — a length scale
    that matches **none** of the three geometries.  Both mismatches are
    recorded here so that every reported error can be paired with them
    (audited in m16740; see ``docs/CAVITY_METRIC_GEOMETRY_AUDIT.md``).

    Args:
        nx: Number of node columns.
        ny: Number of node rows (defaults to ``nx``, the square-cavity case).

    Returns:
        Dict with the wall planes, the implied extents ``h_x``/``h_y``, the
        aspect ratio and the ratios between the geometric lengths and the
        ``L = nx`` length scale actually used by ``Re``.

    Raises:
        ValueError: If the grid is too small for a fluid interior.
    """
    ny = int(nx) if ny is None else int(ny)
    nx = int(nx)
    if nx < 4 or ny < 4:
        msg = f"cavity_wall_planes needs nx, ny >= 4, got nx={nx}, ny={ny}"
        raise ValueError(msg)
    h_x = float(nx) - 2.0
    h_y = float(ny) - 1.5
    return {
        "x_lower": 0.5,
        "x_upper": float(nx) - 1.5,
        "y_lower": 0.5,
        "y_upper": float(ny) - 1.0,
        "h_x": h_x,
        "h_y": h_y,
        "aspect_ratio_h_y_over_h_x": h_y / h_x,
        "re_length_scale_used": float(nx),
        "implied_re_ratio_x": h_x / float(nx),
        "implied_re_ratio_y": h_y / float(ny),
    }


def centreline_coordinates_between(
    n: int, lower: float, upper: float
) -> "list[float]":
    """Normalised node coordinates for explicit wall planes at ``lower``/``upper``.

    Generalises :func:`centreline_coordinates`: node ``j`` (``j = 0 .. n-1``)
    sits at the node-index position ``j``, and the two wall planes sit at
    ``lower`` and ``upper`` node-index positions, so

    .. math::  q_j = \frac{j - \mathrm{lower}}{\mathrm{upper} - \mathrm{lower}}

    ``centreline_coordinates(n, w)`` is the special case
    ``lower = -w``, ``upper = n - 1 + w`` (pinned by a test).

    Args:
        n: Number of nodes along the axis (``n >= 2``).
        lower: Lower wall plane in node-index units.
        upper: Upper wall plane in node-index units.

    Returns:
        List of ``n`` normalised coordinates, monotonically increasing.

    Raises:
        ValueError: If ``n < 2`` or ``upper <= lower``.
    """
    import numpy as np

    if int(n) < 2:
        msg = f"centreline_coordinates_between needs n >= 2, got {n}"
        raise ValueError(msg)
    span = float(upper) - float(lower)
    if not np.isfinite(span) or span <= 0.0:
        msg = f"degenerate wall planes lower={lower}, upper={upper}"
        raise ValueError(msg)
    return [float(v) for v in (np.arange(int(n)) - float(lower)) / span]


def _master_profile(reference: dict[str, list[float]], axis: str):
    """Monotone interpolant through one Ghia table column (PCHIP if available)."""
    import numpy as np

    x = np.asarray(reference["y" if axis == "u" else "x"], dtype=float)
    y = np.asarray(reference["u" if axis == "u" else "v"], dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    try:
        from scipy.interpolate import PchipInterpolator

        f = PchipInterpolator(x, y)
        return lambda q: f(np.clip(q, x[0], x[-1]))
    except Exception:  # pragma: no cover - SciPy is a hard dependency of the stack
        return lambda q: np.interp(q, x, y)


def _interior_rows(ref_coords: "list[float]") -> "list[bool]":
    """Mask selecting reference rows strictly inside the cavity (drop y/x = 0, 1)."""
    return [bool(0.0 < float(c) < 1.0) for c in ref_coords]


def ghia_metric_floor(
    reference: dict[str, list[float]], nx: int, ny: int | None = None
) -> dict[str, dict[str, float]]:
    """Predicted metric error of a *perfect* solver under each coordinate map.

    The floor is what :func:`compare_ghia` reports for a field that reproduces
    the reference profile exactly *on the nodes the BC recipe actually
    creates*: sampling the truth at the recipe wall planes, then scoring the
    same samples with a different index->coordinate map, gives a non-zero
    RMSE that is pure metric artefact (no solver error in it).

    Maps scored (all normalised by ``u_lid``, same units as
    :func:`compare_ghia`):

    * ``legacy_w0``           – walls on the first/last node (historical default);
    * ``symmetric_w+0.5``     – nodes at cell centres (symmetric half-cell);
    * ``recipe_planes``       – the map actually implied by the BC recipe
      (self-consistent: floor is identically zero, listed as a control).

    Args:
        reference: Ghia table (e.g. ``GHIA_RE1000``).
        nx: Number of node columns.
        ny: Number of node rows (defaults to ``nx``).

    Returns:
        ``{map_name: {"floor_u": ..., "floor_v": ..., "floor_u_interior": ...,
        "floor_v_interior": ...}}``.
    """
    import numpy as np

    ny = int(nx) if ny is None else int(ny)
    planes = cavity_wall_planes(nx, ny)
    truth_u = np.asarray(
        centreline_coordinates_between(ny, planes["y_lower"], planes["y_upper"])
    )
    truth_v = np.asarray(
        centreline_coordinates_between(nx, planes["x_lower"], planes["x_upper"])
    )
    fu = _master_profile(reference, "u")
    fv = _master_profile(reference, "v")
    field_u = fu(truth_u)
    field_v = fv(truth_v)

    ref_y = np.asarray(reference["y"], dtype=float)
    ref_u = np.asarray(reference["u"], dtype=float)
    ref_x = np.asarray(reference["x"], dtype=float)
    ref_v = np.asarray(reference["v"], dtype=float)
    keep_u = np.asarray(_interior_rows(reference["y"]))
    keep_v = np.asarray(_interior_rows(reference["x"]))

    candidates = {
        "legacy_w0": (
            np.asarray(centreline_coordinates(ny, 0.0)),
            np.asarray(centreline_coordinates(nx, 0.0)),
        ),
        "symmetric_w+0.5": (
            np.asarray(centreline_coordinates(ny, 0.5)),
            np.asarray(centreline_coordinates(nx, 0.5)),
        ),
        "recipe_planes": (truth_u, truth_v),
    }
    out: dict[str, dict[str, float]] = {}
    for name, (cu, cv) in candidates.items():
        eu = np.interp(ref_y, cu, field_u) - ref_u
        ev = np.interp(ref_x, cv, field_v) - ref_v
        out[name] = {
            "floor_u": float(np.sqrt(np.mean(eu**2))),
            "floor_v": float(np.sqrt(np.mean(ev**2))),
            "floor_u_interior": float(np.sqrt(np.mean(eu[keep_u] ** 2))),
            "floor_v_interior": float(np.sqrt(np.mean(ev[keep_v] ** 2))),
        }
    return out


def ghia_metric_report(
    ux: torch.Tensor,
    uy: torch.Tensor,
    u_lid: float,
    reference: dict[str, list[float]],
    *,
    include_centreline: bool = True,
) -> dict[str, object]:
    """Score one flow field against Ghia under every coordinate convention.

    This is the metric block persisted by :func:`run_lid_driven_cavity`
    (``run_metadata.json`` -> ``ghia_metric``).  It exists because m16339
    showed the reported RMSE moves by ~0.4-1.1 % of ``u_lid`` with the
    index->coordinate map, i.e. by an amount comparable to the 3 %/5 %
    acceptance gates: a bare RMSE is not reproducible without its
    convention, and the historical runs did not persist one.

    Every variant is reported **full** (all reference rows) and **interior**
    (wall rows ``y,x = 0, 1`` dropped, because ``np.interp`` silently clamps
    query points outside the node span — see m16339 finding F3).

    Args:
        ux: x-velocity field ``(ny, nx)``.
        uy: y-velocity field ``(ny, nx)``.
        u_lid: Lid velocity used for normalisation.
        reference: Ghia table for the run's Reynolds number.
        include_centreline: Also return the full normalised centreline
            profiles so the score can be recomputed later without rerunning
            (the gap flagged by m16339: existing runs only stored the RMSE).

    Returns:
        Dict with ``boundary_recipe``, ``wall_planes``, ``scores`` (four
        conventions), ``analytic_floor`` and, optionally, the centreline
        profiles.
    """
    ny, nx = int(ux.shape[0]), int(ux.shape[1])
    planes = cavity_wall_planes(nx, ny)
    variants: dict[str, dict[str, tuple[float, float] | float]] = {
        "legacy_w0": {"wall_offset": 0.0},
        "symmetric_w+0.5": {"wall_offset": 0.5},
        "recipe_planes": {
            "y_planes": (planes["y_lower"], planes["y_upper"]),
            "x_planes": (planes["x_lower"], planes["x_upper"]),
        },
    }
    scores: dict[str, dict[str, float]] = {}
    for name, kw in variants.items():
        full = compare_ghia(ux, uy, u_lid, reference, **kw)  # type: ignore[arg-type]
        inter = compare_ghia(
            ux, uy, u_lid, reference, exclude_walls=True, **kw  # type: ignore[arg-type]
        )
        scores[name] = {
            "rmse_u": full["rmse_u"],
            "rmse_v": full["rmse_v"],
            "interior_rmse_u": inter["rmse_u"],
            "interior_rmse_v": inter["rmse_v"],
            "n_scored_u": full["n_scored_u"],
            "n_scored_v": full["n_scored_v"],
        }

    report: dict[str, object] = {
        "boundary_recipe": CAVITY_BOUNDARY_RECIPE,
        "wall_planes": planes,
        "scores": scores,
        "analytic_floor": ghia_metric_floor(reference, nx, ny),
        "notes": (
            "rmse_* = all reference rows; interior_rmse_* = wall rows dropped. "
            "Quote the convention (wall_planes) with any RMSE: the map alone "
            "moves the number by O(0.4-1.1 %) of u_lid (m16339)."
        ),
    }
    if include_centreline:
        import numpy as np

        ux_np = np.asarray(ux.detach().cpu().numpy(), dtype=float) / float(u_lid)
        uy_np = np.asarray(uy.detach().cpu().numpy(), dtype=float) / float(u_lid)
        ys = centreline_coordinates_between(ny, planes["y_lower"], planes["y_upper"])
        xs = centreline_coordinates_between(nx, planes["x_lower"], planes["x_upper"])
        report["centreline"] = {
            "y": ys,
            "u": [float(v) for v in ux_np[:, nx // 2]],
            "x": xs,
            "v": [float(v) for v in uy_np[ny // 2, :]],
            "coordinates": "recipe_planes",
        }
    return report


def compare_ghia(
    ux: torch.Tensor,
    uy: torch.Tensor,
    u_lid: float,
    reference: dict[str, list[float]],
    wall_offset: float = 0.0,
    *,
    y_planes: "tuple[float, float] | None" = None,
    x_planes: "tuple[float, float] | None" = None,
    exclude_walls: bool = False,
) -> dict[str, float]:
    """Compare centreline velocity profiles against Ghia tabulated data.

    Interpolates the LBM fields at the Ghia y/x positions and returns the
    root-mean-square error (RMSE) normalised by ``u_lid``.

    The Ghia tables store ``y`` (and ``v``) in *descending* y order and
    ``x`` in ascending order; ``np.interp`` only requires the *abscissa*
    (the node coordinates) to be increasing, so the descending reference
    abscissa is evaluated as-is and both sides stay consistently ordered.
    Do not sort one side only.

    Args:
        ux: x-velocity field of shape ``(ny, nx)``.
        uy: y-velocity field of shape ``(ny, nx)``.
        u_lid: Lid velocity used for normalisation (must be non-zero).
        reference: One of :data:`GHIA_RE100`, :data:`GHIA_RE400`, or
            :data:`GHIA_RE1000`.
        wall_offset: Node-to-physical-coordinate convention, see
            :func:`centreline_coordinates`.  ``0.0`` reproduces the
            historical behaviour; pass the value that matches the
            boundary treatment actually used by the run (``0.5`` for
            walls half a cell outside the node lattice) and quote it
            next to the error.  The reported RMSE is **not** invariant
            under this choice.

    Caveat (audited in m16339): with ``wall_offset != 0`` the table rows that
    sit exactly on a wall (``y = 0`` and ``y = 1``, ``x = 0`` and ``x = 1``)
    fall *outside* the node span, and ``np.interp`` clamps them to the nearest
    node value instead of flagging them.  Those two rows then carry a
    boundary-value error of order ``|du/dy| * h/2`` which is *not* solver
    error; score the interior rows separately (or extrapolate explicitly) when
    quoting a number, and always state the ``wall_offset`` used.
    Returns:
        Dict with keys ``"rmse_u"`` and ``"rmse_v"``.  A non-finite field
        (diverged run) yields ``nan`` for both — never a silent pass.

    Raises:
        ValueError: If the fields are not 2-D, if the reference table is
            malformed, or if ``u_lid`` is zero/non-finite.
    """
    import numpy as np

    if ux.ndim != 2 or uy.ndim != 2:
        msg = f"ux/uy must be 2-D (ny, nx); got {tuple(ux.shape)} and {tuple(uy.shape)}"
        raise ValueError(msg)
    if tuple(ux.shape) != tuple(uy.shape):
        msg = f"ux/uy shape mismatch: {tuple(ux.shape)} vs {tuple(uy.shape)}"
        raise ValueError(msg)
    u_lid = float(u_lid)
    if not np.isfinite(u_lid) or u_lid == 0.0:
        msg = f"u_lid must be finite and non-zero, got {u_lid}"
        raise ValueError(msg)
    for key in ("y", "u", "x", "v"):
        if key not in reference:
            msg = f"Ghia reference table missing key {key!r}"
            raise ValueError(msg)

    ny, nx = ux.shape

    # Normalise by u_lid
    ux_np = ux.detach().cpu().numpy() / u_lid
    uy_np = uy.detach().cpu().numpy() / u_lid

    # Vertical centreline: u at x = nx//2
    x_mid = nx // 2
    u_profile = ux_np[:, x_mid]  # shape (ny,)
    y_pos = np.asarray(
        centreline_coordinates(ny, wall_offset)
        if y_planes is None
        else centreline_coordinates_between(ny, y_planes[0], y_planes[1])
    )
    ref_u = np.array(reference["u"], dtype=float)
    keep_u = (
        np.asarray(_interior_rows(reference["y"]))
        if exclude_walls
        else np.ones(ref_u.shape, dtype=bool)
    )
    u_ghia_interp = np.interp(reference["y"], y_pos, u_profile)
    rmse_u = float(np.sqrt(np.mean((u_ghia_interp[keep_u] - ref_u[keep_u]) ** 2)))

    # Horizontal centreline: v at y = ny//2
    y_mid = ny // 2
    v_profile = uy_np[y_mid, :]  # shape (nx,)
    x_pos = np.asarray(
        centreline_coordinates(nx, wall_offset)
        if x_planes is None
        else centreline_coordinates_between(nx, x_planes[0], x_planes[1])
    )
    ref_v = np.array(reference["v"], dtype=float)
    keep_v = (
        np.asarray(_interior_rows(reference["x"]))
        if exclude_walls
        else np.ones(ref_v.shape, dtype=bool)
    )
    v_ghia_interp = np.interp(reference["x"], x_pos, v_profile)
    rmse_v = float(np.sqrt(np.mean((v_ghia_interp[keep_v] - ref_v[keep_v]) ** 2)))

    return {
        "rmse_u": rmse_u,
        "rmse_v": rmse_v,
        "n_scored_u": int(keep_u.sum()),
        "n_scored_v": int(keep_v.sum()),
    }


def ghia_coordinate_sensitivity(
    reference: dict[str, list[float]],
    n: int,
    wall_offsets: tuple[float, ...] = (0.0, 0.5, -0.5),
) -> dict[str, dict[str, float]]:
    """Quantify how much the Ghia RMSE depends on the coordinate convention.

    A solver that reproduces the Ghia profile *exactly* on its own nodes
    still scores a non-zero RMSE when the metric's index-to-coordinate
    map differs from the run's boundary convention.  This function
    measures that floor analytically: the reference table is turned into
    a monotone master profile, a "perfect" field is sampled at the node
    positions of convention ``w = 0.5`` (cell-centred nodes — what a
    half-way bounce-back run actually produces), and the error is scored
    with each candidate map.

    Args:
        reference: Ghia table (``GHIA_RE100`` / ``GHIA_RE400`` /
            ``GHIA_RE1000``) or any dict with the same keys.
        n: Number of nodes along the axis (grid resolution).
        wall_offsets: Conventions to score, see
            :func:`centreline_coordinates`.

    Returns:
        ``{"+0.5": {"bias_u": ..., "bias_v": ...}, ...}`` keyed by the convention offset with the
        RMSE (normalised by ``u_lid``, in the same units
        :func:`compare_ghia` returns) of a perfect solver under that
        convention.
    """
    import numpy as np

    def _master(coords, values):
        x = np.asarray(coords, dtype=float)
        y = np.asarray(values, dtype=float)
        order = np.argsort(x)
        x, y = x[order], y[order]
        try:  # monotone cubic when SciPy is available (repo default)
            from scipy.interpolate import PchipInterpolator

            f = PchipInterpolator(x, y)
            return lambda q: f(np.clip(q, x[0], x[-1]))
        except Exception:  # pragma: no cover - SciPy is a hard dependency of the stack
            return lambda q: np.interp(q, x, y)

    fu = _master(reference["y"], reference["u"])
    fv = _master(reference["x"], reference["v"])
    truth_u = np.asarray(centreline_coordinates(n, 0.5))
    truth_v = truth_u
    out: dict[str, dict[str, float]] = {}
    for w in wall_offsets:
        cand = np.asarray(centreline_coordinates(n, w))
        bias_u = float(np.sqrt(np.mean((fu(cand) - fu(truth_u)) ** 2)))
        bias_v = float(np.sqrt(np.mean((fv(cand) - fv(truth_v)) ** 2)))
        out[f"{w:+.1f}"] = {"bias_u": bias_u, "bias_v": bias_v}
    return out


# ---------------------------------------------------------------------------
# Snapshot helper
# ---------------------------------------------------------------------------


def _save_cavity_snapshot(
    run_dir: Path,
    step: int,
    speed: torch.Tensor,
    vort: torch.Tensor,
    wall_mask: torch.Tensor,
) -> None:
    speed_np = speed.detach().cpu().numpy()
    vort_np = vort.detach().cpu().numpy()
    obs_np = wall_mask.detach().cpu().float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    im0 = axes[0].imshow(speed_np, origin="lower", cmap="viridis")
    axes[0].contour(obs_np, levels=[0.5], colors="white", linewidths=0.7)
    axes[0].set_title(f"Velocity magnitude (step {step})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(vort_np, origin="lower", cmap="coolwarm")
    axes[1].contour(obs_np, levels=[0.5], colors="black", linewidths=0.7)
    axes[1].set_title(f"Vorticity (step {step})")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    out = flow_step_image_path(run_dir, step)
    fig.savefig(out, dpi=120)
    write_legacy_snapshot_alias(run_dir, step)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_lid_driven_cavity(config: LidDrivenCavityConfig) -> Path:
    """Run the lid-driven cavity benchmark and save results.

    Args:
        config: Simulation configuration.

    Returns:
        Path of the output directory created for this run.
    """
    configure_logging()
    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = resolve_device(config.device)
    run_dir = prepare_run_dir(
        config.output_root,
        "lid_driven_cavity",
        config.resolved_run_name(),
        config.overwrite,
    )

    ny, nx = config.ny, config.nx

    metadata: dict[str, object] = {
        "config": {**asdict(config), "output_root": str(config.output_root)},
        "derived": {"nu": config.nu, "tau": config.tau, "ny": ny},
        "runtime": {"torch_version": torch.__version__, "device": str(device)},
        "reproducibility": get_reproducibility_metadata(),
    }

    # Initialise: quiescent fluid
    rho0 = torch.ones((ny, nx), device=device)
    ux0 = torch.zeros((ny, nx), device=device)
    uy0 = torch.zeros((ny, nx), device=device)
    # Top row already at rest; lid starts moving at step 1
    f = equilibrium(rho0, ux0, uy0, device=device)

    wall_mask = make_cavity_wall_mask(ny, nx, device, include_top=False)

    _collide_base = collide_mrt if config.tau < 0.60 else collide_bgk
    _collide = _maybe_compile(_collide_base, config.use_compile)
    _stream = _maybe_compile(stream, config.use_compile)

    logger.info(
        "Running lid-driven cavity device=%s NX=%s NY=%s tau=%.4f Re=%.1f steps=%s",
        device,
        nx,
        ny,
        config.tau,
        config.re,
        config.n_steps,
    )
    logger.info("Run directory: %s", run_dir)

    diagnostics: list[dict[str, object]] = []

    from .cylinder_flow import compute_vorticity

    for step in range(1, config.n_steps + 1):
        f = _collide(f, tau=config.tau)
        f = _stream(f)
        # Bounce-back on all four walls (including lid row, handles corners)
        f = bounce_back_cells(f, wall_mask)
        # Overwrite interior lid cells with Zou/He moving-wall BC
        f = zou_he_moving_lid(f, config.u_lid)

        if step % config.output_interval == 0 or step == config.n_steps:
            rho, ux, uy = macroscopic(f)
            # Zero velocity inside walls
            ux = ux.masked_fill(wall_mask, 0.0)
            uy = uy.masked_fill(wall_mask, 0.0)
            speed = torch.sqrt(ux * ux + uy * uy)
            vort = compute_vorticity(ux, uy)

            point = DiagnosticPoint(
                step=step,
                mass=float(rho.sum().item()),
                mass_drift=float(rho.sum().item()) - float(rho0.sum().item()),
                max_speed=float(speed.max().item()),
                mean_rho=float(rho.mean().item()),
            )
            diag_entry: dict[str, object] = {**asdict(point)}
            diagnostics.append(diag_entry)
            logger.info(
                "step=%5d mass=%.6f drift=%+.6f mean_rho=%.6f max|u|=%.6f",
                point.step,
                point.mass,
                point.mass_drift,
                point.mean_rho,
                point.max_speed,
            )
            _save_cavity_snapshot(run_dir, step, speed, vort, wall_mask)

    # Final state comparison with Ghia reference data
    rho, ux, uy = macroscopic(f)
    ux = ux.masked_fill(wall_mask, 0.0)
    uy = uy.masked_fill(wall_mask, 0.0)

    re_int = int(config.re)
    if re_int == 100:
        reference = GHIA_RE100
    elif re_int == 400:
        reference = GHIA_RE400
    elif re_int == 1000:
        reference = GHIA_RE1000
    else:
        reference = None

    ghia_errors: dict[str, float] | None = None
    ghia_metric: dict[str, object] | None = None
    if reference is not None:
        ghia_metric = ghia_metric_report(ux, uy, config.u_lid, reference)
        selected = ghia_metric["scores"][config.ghia_metric_convention]
        ghia_errors = {
            "rmse_u": selected["rmse_u"],
            "rmse_v": selected["rmse_v"],
        }
        logger.info(
            "Ghia comparison (Re=%d, convention=%s): RMSE_u=%.5f  RMSE_v=%.5f",
            re_int,
            config.ghia_metric_convention,
            ghia_errors["rmse_u"],
            ghia_errors["rmse_v"],
        )

    # Write Ghia comparison CSV
    import numpy as np

    ny_final, nx_final = ux.shape
    ux_np = ux.detach().cpu().numpy() / config.u_lid
    uy_np = uy.detach().cpu().numpy() / config.u_lid
    x_mid = nx_final // 2
    y_mid = ny_final // 2
    planes = cavity_wall_planes(nx_final, ny_final)
    y_pos = np.asarray(
        centreline_coordinates_between(ny_final, planes["y_lower"], planes["y_upper"])
    )
    x_pos = np.asarray(
        centreline_coordinates_between(nx_final, planes["x_lower"], planes["x_upper"])
    )

    csv_path = run_dir / "ghia_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["y_pos", "u_lbm", "x_pos", "v_lbm"])
        for j in range(ny_final):
            writer.writerow(
                [
                    round(float(y_pos[j]), 6),
                    round(float(ux_np[j, x_mid]), 6),
                    round(float(x_pos[j]), 6),
                    round(float(uy_np[y_mid, j]), 6),
                ]
            )

    metadata["diagnostics"] = diagnostics
    if ghia_errors is not None:
        metadata["ghia_errors"] = ghia_errors
        metadata["ghia_metric"] = ghia_metric
        metadata["ghia_metric_convention"] = config.ghia_metric_convention
        metadata["ghia_note"] = (
            "rmse_u: RMSE of u/u_lid along vertical centreline vs Ghia (1982). "
            "rmse_v: RMSE of v/u_lid along horizontal centreline vs Ghia (1982). "
            "These headline numbers use ghia_metric_convention; ghia_metric "
            "carries the same score under every supported coordinate map plus "
            "the wall-plane geometry and analytic metric floors, so no RMSE is "
            "quoted without its convention (m16740)."
        )

    meta_path = run_dir / "run_metadata.json"
    meta_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved metadata: %s", meta_path)
    return run_dir


__all__ = [
    "GHIA_RE100",
    "GHIA_RE400",
    "GHIA_RE1000",
    "LidDrivenCavityConfig",
    "compare_ghia",
    "make_cavity_wall_mask",
    "run_lid_driven_cavity",
    "zou_he_moving_lid",
]
