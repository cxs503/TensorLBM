"""Ship wind resistance: superstructure aerodynamics database + hybrid engine.

Node IN38-N04.  Companion to :mod:`tensorlbm.wind_load_lng` (which owns the
wall-modelled LBM solver) -- this module owns the *ship-type superstructure
catalogues*, the *engineering empirical models*, the *hybrid empirical/LBM
routing engine* and the *annual wind-rose assessment*.

Scope
-----
Three typical merchant-ship superstructure arrangements, described as
axis-aligned box assemblies above the free deck (the same representation the
LBM solver voxelises):

* container ship -- aft deckhouse + bridge wings + exhaust casing + three
  container-stow blocks forward of the house + masts;
* tanker (VLCC) -- aft deckhouse + casing + deck pipe runs + centre-line
  cargo manifold + forecastle house + masts;
* bulk carrier -- forward deckhouse + casing + four hatch covers + four deck
  cranes + masts.

Everything is written in the *ship frame*: ``+x`` points forward (bow),
``+y`` to port, ``+z`` up from the free deck at ``z = 0``.  ``theta`` is the
wind angle measured from the bow, so ``theta = 0`` is a head wind and
``theta = 90`` a beam wind from starboard.  Force conventions follow the
wind-load literature::

    q    = 0.5 * rho_air * U^2
    C_X  = X_wind / (q A_F)      longitudinal force, A_F = 0 deg projection
    C_Y  = Y_wind / (q A_L)      transverse force,   A_L = 90 deg projection
    C_N  = N_wind / (q A_L B)    yaw moment about z
    C_K  = K_wind / (q A_L H)    heel moment about x

Three layers
------------
1. :func:`component_buildup` -- **a priori** engineering estimate: every box
   is a bluff body with a handbook drag coefficient, projected-area weighted
   over the wind angle, with an explicit shadowing model.  Nothing is fitted.
2. :func:`blendermann_style` / :func:`fit_blendermann_style` -- a closed-form
   empirical correlation of the shape used by Blendermann-type ship wind-load
   correlations (``|cos|``/``|sin|`` driven with a finite beam-on residual),
   parameterised per ship type.  Parameters are fitted to the LBM database and
   cross-validated leave-one-angle-out; the module never claims to reproduce a
   published coefficient table verbatim.
3. :class:`HybridWindEngine` -- routes each query to the empirical correlation
   (simple, deckhouse-dominated arrangements) or to the interpolated LBM
   database (complex arrangements, e.g. separated container stows), with a
   documented blending band and an uncertainty estimate.

Nothing here imports :mod:`tensorlbm.wind_load_lng` at module import time
(the LBM solver needs torch); the database side is pure numpy/maths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "BoxSpec", "ShipSuperstructure", "HullReference", "CD_HANDBOOK",
    "container_ship_superstructure", "tanker_superstructure",
    "bulk_carrier_superstructure", "catalogue", "validate_geometry",
    "export_stl", "projected_areas", "component_buildup", "shadow_factor",
    "blendermann_style", "blendermann_coefficients", "fit_blendermann_style",
    "HybridWindEngine", "complexity_index", "sign_conventions",
    "wind_rose_annual_resistance", "synthetic_wind_rose",
    "evaluate_acceptance",
]

RHO_AIR = 1.225      # kg/m^3, 15 degC, 1013 hPa
NU_AIR = 1.5e-5      # m^2/s


# ---------------------------------------------------------------------------
# Geometry catalogues
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BoxSpec:
    """Axis-aligned above-deck block (metres, ship frame)."""

    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float
    name: str
    #: engineering drag coefficient normal to the largest face
    cd: float = 1.10
    #: ``True`` for porous / lattice work (railings, open framework)
    porous: bool = False

    @property
    def volume(self) -> float:
        return (self.x1 - self.x0) * (self.y1 - self.y0) * (self.z1 - self.z0)

    @property
    def extents(self) -> tuple[float, float, float]:
        return (self.x1 - self.x0, self.y1 - self.y0, self.z1 - self.z0)

    def to_dict(self) -> dict:
        return {"name": self.name, "x": [self.x0, self.x1],
                "y": [self.y0, self.y1], "z": [self.z0, self.z1],
                "cd": self.cd, "porous": self.porous}


@dataclass(frozen=True)
class HullReference:
    """Hull particulars used for geometry sanity checks and the annual study."""

    length_pp_m: float
    breadth_m: float
    draught_m: float
    block_coefficient: float
    design_speed_kn: float
    description: str = ""

    @property
    def displacement_m3(self) -> float:
        return (self.length_pp_m * self.breadth_m * self.draught_m
                * self.block_coefficient)

    def validate(self) -> list[str]:
        """Return a list of failed consistency checks (empty = pass)."""
        bad: list[str] = []
        if not 0.45 <= self.block_coefficient <= 0.90:
            bad.append(f"Cb={self.block_coefficient} outside 0.45..0.90")
        if not 100.0 <= self.length_pp_m <= 500.0:
            bad.append(f"Lpp={self.length_pp_m} outside 100..500 m")
        if not 15.0 <= self.breadth_m <= 70.0:
            bad.append(f"B={self.breadth_m} outside 15..70 m")
        if not 2.0 <= self.breadth_m / self.draught_m <= 8.0:
            bad.append(f"B/T={self.breadth_m / self.draught_m:.2f} outside 2..8")
        if not 4.0 <= self.length_pp_m / self.breadth_m <= 9.0:
            bad.append(f"L/B={self.length_pp_m / self.breadth_m:.2f} outside 4..9")
        disp = self.displacement_m3
        if not 0.30 * self.length_pp_m ** 3 < disp < 0.90 * self.length_pp_m ** 3:
            bad.append("displacement outside slender-body band")
        return bad


@dataclass
class ShipSuperstructure:
    """An above-deck box assembly plus its hull reference particulars."""

    key: str
    name: str
    boxes: list[BoxSpec]
    hull: HullReference
    #: geometric reference quantities filled by :func:`validate_geometry`
    reference: dict = field(default_factory=dict)

    @property
    def extent(self) -> tuple[float, float, float]:
        xs = [b.x0 for b in self.boxes] + [b.x1 for b in self.boxes]
        ys = [b.y0 for b in self.boxes] + [b.y1 for b in self.boxes]
        zs = [b.z0 for b in self.boxes] + [b.z1 for b in self.boxes]
        return (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    @property
    def height(self) -> float:
        return max(b.z1 for b in self.boxes)

    @property
    def breadth(self) -> float:
        return max(b.y1 for b in self.boxes) - min(b.y0 for b in self.boxes)

    @property
    def solid_volume(self) -> float:
        return sum(b.volume for b in self.boxes)

    def span(self) -> tuple[float, float, float, float]:
        """(xmin, xmax, ymin, ymax) of the assembly (domain centring)."""
        xs = [b.x0 for b in self.boxes] + [b.x1 for b in self.boxes]
        ys = [b.y0 for b in self.boxes] + [b.y1 for b in self.boxes]
        return (min(xs), max(xs), min(ys), max(ys))

    def to_lng_boxes(self):
        """Convert to :class:`tensorlbm.wind_load_lng.Box` (lazy import)."""
        from .wind_load_lng import Box

        return [Box(b.x0, b.x1, b.y0, b.y1, b.z0, b.z1, b.name)
                for b in self.boxes]

    def to_dict(self) -> dict:
        return {
            "key": self.key, "name": self.name,
            "hull": {
                "Lpp_m": self.hull.length_pp_m,
                "B_m": self.hull.breadth_m,
                "T_m": self.hull.draught_m,
                "Cb": self.hull.block_coefficient,
                "displacement_m3": self.hull.displacement_m3,
                "design_speed_kn": self.hull.design_speed_kn,
                "description": self.hull.description,
            },
            "boxes": [b.to_dict() for b in self.boxes],
            "extent_m": list(self.extent),
            "superstructure_volume_m3": self.solid_volume,
            "reference": dict(self.reference),
        }


#: handbook drag coefficients used by :func:`component_buildup`; each value is
#: a round bluff-body value quoted for the component's slenderness regime
#: (short prism / slender prism / flat plate / cylinder / lattice / pipe rack).
CD_HANDBOOK = {
    "prism": 1.10,
    "tall_prism": 1.25,
    "plate": 1.18,
    "cylinder": 1.05,
    "lattice": 0.45,
    "pipe_rack": 0.70,
}


def container_ship_superstructure() -> ShipSuperstructure:
    """Ultra-large container vessel: aft house, deck stow forward, tall casing.

    Container ships are the *complex* end of the catalogue: the deck stow in
    front of the accommodation block forms a forward-facing step and the
    casing sits in its separated wake, so empirical correlations built for
    tanker/bulker silhouettes extrapolate poorly.
    """

    b: list[BoxSpec] = [
        BoxSpec(-16.0, -14.0, -15.0, 15.0, 0.0, 3.0, "aft_bulwark",
                CD_HANDBOOK["plate"]),
        BoxSpec(-14.0, 0.0, -16.0, 16.0, 0.0, 24.0, "deck_house",
                CD_HANDBOOK["prism"]),
        BoxSpec(-12.0, -2.0, -11.0, 11.0, 24.0, 27.0, "bridge_deck",
                CD_HANDBOOK["prism"]),
        BoxSpec(-12.0, -2.0, -24.0, -11.0, 24.0, 26.0, "bridge_wing_ps",
                CD_HANDBOOK["plate"]),
        BoxSpec(-12.0, -2.0, 11.0, 24.0, 24.0, 26.0, "bridge_wing_sb",
                CD_HANDBOOK["plate"]),
        BoxSpec(-24.0, -14.0, -6.0, 6.0, 0.0, 32.0, "funnel_casing",
                CD_HANDBOOK["tall_prism"]),
        BoxSpec(-10.0, -9.0, -3.0, 3.0, 27.0, 33.0, "radar_mast",
                CD_HANDBOOK["lattice"], True),
        BoxSpec(2.0, 20.0, -18.0, 18.0, 0.0, 21.0, "stow_1",
                CD_HANDBOOK["prism"]),
        BoxSpec(22.0, 24.0, -18.0, 18.0, 0.0, 3.0, "lashing_platform",
                CD_HANDBOOK["plate"]),
        BoxSpec(24.0, 42.0, -18.0, 18.0, 0.0, 17.0, "stow_2",
                CD_HANDBOOK["prism"]),
        BoxSpec(44.0, 58.0, -18.0, 18.0, 0.0, 9.0, "stow_3",
                CD_HANDBOOK["prism"]),
        BoxSpec(2.0, 58.0, -21.0, -19.0, 0.0, 1.5, "pipe_run_ps",
                CD_HANDBOOK["pipe_rack"]),
        BoxSpec(2.0, 58.0, 19.0, 21.0, 0.0, 1.5, "pipe_run_sb",
                CD_HANDBOOK["pipe_rack"]),
        BoxSpec(64.0, 74.0, -9.0, 9.0, 0.0, 4.0, "forecastle_house",
                CD_HANDBOOK["prism"]),
    ]
    hull = HullReference(
        length_pp_m=399.9, breadth_m=61.5, draught_m=16.5, block_coefficient=0.66,
        design_speed_kn=22.0,
        description="20 000 TEU class, aft accommodation, deck stow forward",
    )
    return ShipSuperstructure("container", "Ultra-large container ship", b, hull)


def tanker_superstructure() -> ShipSuperstructure:
    """VLCC (double hull, aft house): deckhouse + pipe runs + manifold."""

    b: list[BoxSpec] = [
        BoxSpec(-16.0, -6.0, -12.0, 12.0, 0.0, 20.0, "deck_house",
                CD_HANDBOOK["prism"]),
        BoxSpec(-14.0, -4.0, -8.0, 8.0, 20.0, 23.0, "bridge_deck",
                CD_HANDBOOK["prism"]),
        BoxSpec(-14.0, -4.0, -18.0, -8.0, 20.0, 22.0, "bridge_wing_ps",
                CD_HANDBOOK["plate"]),
        BoxSpec(-14.0, -4.0, 8.0, 18.0, 20.0, 22.0, "bridge_wing_sb",
                CD_HANDBOOK["plate"]),
        BoxSpec(-4.0, 4.0, -5.0, 5.0, 0.0, 26.0, "funnel_casing",
                CD_HANDBOOK["tall_prism"]),
        BoxSpec(-2.0, -1.0, -2.0, 2.0, 23.0, 30.0, "mast",
                CD_HANDBOOK["lattice"], True),
        BoxSpec(4.0, 190.0, -21.0, -18.0, 0.0, 1.6, "pipe_run_ps",
                CD_HANDBOOK["pipe_rack"]),
        BoxSpec(4.0, 190.0, 18.0, 21.0, 0.0, 1.6, "pipe_run_sb",
                CD_HANDBOOK["pipe_rack"]),
        BoxSpec(40.0, 58.0, -9.0, 9.0, 0.0, 4.0, "manifold",
                CD_HANDBOOK["pipe_rack"]),
        BoxSpec(38.0, 60.0, -12.0, 12.0, 4.0, 5.0, "manifold_platform",
                CD_HANDBOOK["plate"]),
        BoxSpec(200.0, 210.0, -10.0, 10.0, 0.0, 6.0, "forecastle_house",
                CD_HANDBOOK["prism"]),
    ]
    hull = HullReference(
        length_pp_m=330.0, breadth_m=60.0, draught_m=21.0, block_coefficient=0.825,
        design_speed_kn=15.5,
        description="VLCC class, aft accommodation, centre-line manifold",
    )
    return ShipSuperstructure("tanker", "VLCC tanker", b, hull)


def bulk_carrier_superstructure() -> ShipSuperstructure:
    """Capesize bulk carrier: forward house, hatch covers, four cranes."""

    b: list[BoxSpec] = [
        BoxSpec(-10.0, 0.0, -12.0, 12.0, 0.0, 18.0, "deck_house",
                CD_HANDBOOK["prism"]),
        BoxSpec(-8.0, 2.0, -8.0, 8.0, 18.0, 21.0, "bridge_deck",
                CD_HANDBOOK["prism"]),
        BoxSpec(-8.0, 2.0, -17.0, -8.0, 18.0, 20.0, "bridge_wing_ps",
                CD_HANDBOOK["plate"]),
        BoxSpec(-8.0, 2.0, 8.0, 17.0, 18.0, 20.0, "bridge_wing_sb",
                CD_HANDBOOK["plate"]),
        BoxSpec(2.0, 9.0, -4.5, 4.5, 0.0, 24.0, "funnel_casing",
                CD_HANDBOOK["tall_prism"]),
        BoxSpec(4.0, 5.0, -1.5, 1.5, 21.0, 28.0, "mast",
                CD_HANDBOOK["lattice"], True),
        BoxSpec(14.0, 44.0, -16.0, 16.0, 0.0, 3.0, "hatch_1",
                CD_HANDBOOK["plate"]),
        BoxSpec(48.0, 78.0, -16.0, 16.0, 0.0, 3.0, "hatch_2",
                CD_HANDBOOK["plate"]),
        BoxSpec(82.0, 112.0, -16.0, 16.0, 0.0, 3.0, "hatch_3",
                CD_HANDBOOK["plate"]),
        BoxSpec(116.0, 146.0, -16.0, 16.0, 0.0, 3.0, "hatch_4",
                CD_HANDBOOK["plate"]),
        BoxSpec(44.0, 48.0, -5.0, 5.0, 0.0, 14.0, "crane_1_column",
                CD_HANDBOOK["tall_prism"]),
        BoxSpec(30.0, 62.0, -2.0, 2.0, 14.0, 17.0, "crane_1_jib",
                CD_HANDBOOK["prism"]),
        BoxSpec(78.0, 82.0, -5.0, 5.0, 0.0, 14.0, "crane_2_column",
                CD_HANDBOOK["tall_prism"]),
        BoxSpec(64.0, 96.0, -2.0, 2.0, 14.0, 17.0, "crane_2_jib",
                CD_HANDBOOK["prism"]),
        BoxSpec(112.0, 116.0, -5.0, 5.0, 0.0, 14.0, "crane_3_column",
                CD_HANDBOOK["tall_prism"]),
        BoxSpec(98.0, 130.0, -2.0, 2.0, 14.0, 17.0, "crane_3_jib",
                CD_HANDBOOK["prism"]),
        BoxSpec(146.0, 150.0, -5.0, 5.0, 0.0, 14.0, "crane_4_column",
                CD_HANDBOOK["tall_prism"]),
        BoxSpec(132.0, 160.0, -2.0, 2.0, 14.0, 17.0, "crane_4_jib",
                CD_HANDBOOK["prism"]),
        BoxSpec(156.0, 166.0, -10.0, 10.0, 0.0, 5.0, "forecastle_house",
                CD_HANDBOOK["prism"]),
    ]
    hull = HullReference(
        length_pp_m=290.0, breadth_m=45.0, draught_m=18.2, block_coefficient=0.830,
        design_speed_kn=14.5,
        description="Capesize bulk carrier, forward accommodation, 4 deck cranes",
    )
    return ShipSuperstructure("bulker", "Capesize bulk carrier", b, hull)


def catalogue() -> dict[str, ShipSuperstructure]:
    """The three catalogued ship types, keyed by short name."""
    return {s.key: s for s in (container_ship_superstructure(),
                               tanker_superstructure(),
                               bulk_carrier_superstructure())}


# ---------------------------------------------------------------------------
# Geometry validation (projection areas, overlaps, hull particulars)
# ---------------------------------------------------------------------------

def projected_areas(ship: ShipSuperstructure, *, nx: int = 1200, ny: int = 600,
                    nz: int = 300) -> dict:
    """Analytic (box-union) projected areas of the above-deck assembly.

    ``A_F`` is the projection on the y-z plane (head wind), ``A_L`` on the x-z
    plane (beam wind) and ``A_plan`` on the deck plane.  A fine regular raster
    resolves the union, so overlapping boxes are counted once.
    """

    xmin, xmax, ymin, ymax = ship.span()
    zmax = ship.height
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    zs = np.linspace(0.0, zmax, nz)

    def raster(axis: str) -> float:
        if axis == "x":
            grid = np.zeros((ny, nz), dtype=bool)
            for b in ship.boxes:
                grid |= ((ys >= b.y0) & (ys <= b.y1))[:, None] & \
                        ((zs >= b.z0) & (zs <= b.z1))[None, :]
            dA = (ys[1] - ys[0]) * (zs[1] - zs[0])
        elif axis == "y":
            grid = np.zeros((nx, nz), dtype=bool)
            for b in ship.boxes:
                grid |= ((xs >= b.x0) & (xs <= b.x1))[:, None] & \
                        ((zs >= b.z0) & (zs <= b.z1))[None, :]
            dA = (xs[1] - xs[0]) * (zs[1] - zs[0])
        else:
            grid = np.zeros((nx, ny), dtype=bool)
            for b in ship.boxes:
                grid |= ((xs >= b.x0) & (xs <= b.x1))[:, None] & \
                        ((ys >= b.y0) & (ys <= b.y1))[None, :]
            dA = (xs[1] - xs[0]) * (ys[1] - ys[0])
        return float(grid.sum()) * dA

    A_F = raster("x")
    A_L = raster("y")
    A_plan = raster("z")
    return {
        "area_frontal_m2": A_F,
        "area_lateral_m2": A_L,
        "area_plan_m2": A_plan,
        "breadth_m": ship.breadth,
        "height_m": ship.height,
        "slenderness_AL_over_AF": A_L / A_F if A_F else float("inf"),
        "lateral_aspect_AL_over_H2": A_L / ship.height ** 2,
    }


def validate_geometry(ship: ShipSuperstructure, *,
                      voxel_areas: Mapping[str, float] | None = None,
                      rel_tol: float = 0.02) -> dict:
    """Geometry-first validation (discipline rule: check the geometry first).

    Checks: hull particulars consistency, strictly positive box extents, no
    positive-volume box overlap, the assembly sits on ``z = 0``, port/starboard
    symmetry in ``y`` (the LBM symmetry cross-check relies on it) and, when
    ``voxel_areas`` is supplied, that the voxel projection areas reproduce the
    analytic silhouette within ``rel_tol``.
    """

    problems: dict[str, list[str]] = {"hull": ship.hull.validate(), "boxes": [],
                                      "overlap": [], "placement": [],
                                      "symmetry": [], "voxel": []}
    for b in ship.boxes:
        lx, ly, lz = b.extents
        if min(lx, ly, lz) <= 0.0:
            problems["boxes"].append(f"{b.name}: non-positive extent")

    for i, a in enumerate(ship.boxes):
        for b in ship.boxes[i + 1:]:
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            oz = min(a.z1, b.z1) - max(a.z0, b.z0)
            if ox > 1e-9 and oy > 1e-9 and oz > 1e-9:
                problems["overlap"].append(
                    f"{a.name} x {b.name}: overlap volume {ox * oy * oz:.3g} m3")

    if abs(min(b.z0 for b in ship.boxes)) > 1e-9:
        problems["placement"].append("assembly does not start at z = 0")

    pos = sorted({round(b.y1, 9) for b in ship.boxes})
    neg = sorted({round(-b.y0, 9) for b in ship.boxes})
    if pos != neg:
        problems["symmetry"].append(f"port {pos} != starboard {neg}")

    ref = projected_areas(ship)
    ship.reference = dict(ref)
    if voxel_areas is not None:
        for key in ("area_frontal_m2", "area_lateral_m2"):
            v = float(voxel_areas[key])
            a = float(ref[key])
            if a > 0 and abs(v - a) / a > rel_tol:
                problems["voxel"].append(
                    f"{key}: voxel {v:.1f} vs analytic {a:.1f} "
                    f"({100.0 * (v - a) / a:+.1f}%)")
    ok = all(not v for v in problems.values())
    return {"ship": ship.key, "ok": ok, "problems": problems,
            "reference": ref, "hull_displacement_m3": ship.hull.displacement_m3,
            "hull_checks_ok": not problems["hull"]}


def _tri(p0, p1, p2, n) -> str:
    return ("facet normal {:.6e} {:.6e} {:.6e}\n"
            "  outer loop\n"
            "    vertex {:.6e} {:.6e} {:.6e}\n"
            "    vertex {:.6e} {:.6e} {:.6e}\n"
            "    vertex {:.6e} {:.6e} {:.6e}\n"
            "  endloop\n"
            "endfacet").format(*n, *p0, *p1, *p2)


def export_stl(ship: ShipSuperstructure, path: str, *,
               name_prefix: str | None = None) -> str:
    """Write the box assembly as an ASCII STL (CAD deliverable, 12 tris/box)."""

    prefix = name_prefix or ship.key

    def quad(p0, p1, p2, p3, n, out):
        out.append(_tri(p0, p1, p2, n))
        out.append(_tri(p0, p2, p3, n))

    out: list[str] = [f"solid {prefix}"]
    for b in ship.boxes:
        (x0, x1), (y0, y1), (z0, z1) = (b.x0, b.x1), (b.y0, b.y1), (b.z0, b.z1)
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        quad(c[0], c[3], c[2], c[1], (0, 0, -1), out)
        quad(c[4], c[5], c[6], c[7], (0, 0, 1), out)
        quad(c[0], c[1], c[5], c[4], (0, -1, 0), out)
        quad(c[2], c[3], c[7], c[6], (0, 1, 0), out)
        quad(c[1], c[2], c[6], c[5], (1, 0, 0), out)
        quad(c[3], c[0], c[4], c[7], (-1, 0, 0), out)
    out.append(f"endsolid {prefix}")
    with open(path, "w", encoding="ascii") as fh:
        fh.write("\n".join(out) + "\n")
    return path


# ---------------------------------------------------------------------------
# Layer 1: a-priori component build-up (handbook drag coefficients)
# ---------------------------------------------------------------------------

def _box_projected_shadow(box: BoxSpec, theta: float) -> float:
    """Projected area of a box onto the plane normal to the wind (m^2).

    For a box with plan extents ``(lx, ly)`` and height ``lz`` the silhouette
    width perpendicular to a wind arriving at angle ``theta`` from the bow is
    ``lx |sin th| + ly |cos th|`` (support function of the plan rectangle), so
    ``A(th) = lz (lx |sin th| + ly |cos th|)``.
    """

    lx, ly, lz = box.extents
    c, s = abs(math.cos(theta)), abs(math.sin(theta))
    return lz * (lx * s + ly * c)


def shadow_factor(boxes: Sequence[BoxSpec], theta: float, *,
                  kappa: float = 0.5) -> list[float]:
    """Fraction of each box's drag that survives upstream shadowing.

    Engineering shadowing model: boxes are ordered along the wind direction; a
    downstream box loses ``kappa`` of its drag in proportion to how much of its
    silhouette is covered by the union of upstream silhouettes.  ``kappa =
    0.5`` is an *a priori* choice for a single wake (full coverage halves the
    force of a body immersed in another body's near wake); it is not fitted to
    any data set here.
    """

    c, s = math.cos(theta), math.sin(theta)

    def along(b: BoxSpec) -> float:
        cx, cy = 0.5 * (b.x0 + b.x1), 0.5 * (b.y0 + b.y1)
        return cx * c + cy * s

    order = sorted(range(len(boxes)), key=lambda i: -along(boxes[i]))

    def lat_interval(b: BoxSpec) -> tuple[float, float]:
        pts = [-x * s + y * c for x in (b.x0, b.x1) for y in (b.y0, b.y1)]
        return (min(pts), max(pts))

    lat = [lat_interval(b) for b in boxes]
    lo = min(v[0] for v in lat)
    hi = max(v[1] for v in lat)
    n_lat, n_z = 96, 64
    edges = np.linspace(lo, hi, n_lat + 1)
    zmax = max(b.z1 for b in boxes)
    zedges = np.linspace(0.0, zmax, n_z + 1)

    covered = np.zeros((n_lat, n_z), dtype=bool)
    factors = [1.0] * len(boxes)
    for i in order:
        b = boxes[i]
        ci = max(int(np.searchsorted(edges, lat[i][0], "right") - 1), 0)
        cj = min(int(np.searchsorted(edges, lat[i][1], "left")), n_lat)
        zk = max(int(np.searchsorted(zedges, b.z0, "right") - 1), 0)
        zl = min(int(np.searchsorted(zedges, b.z1, "left")), n_z)
        if cj <= ci or zl <= zk:
            continue
        own = np.zeros_like(covered)
        own[ci:cj, zk:zl] = True
        n_own = int(own.sum())
        if n_own:
            overlap = float((own & covered).sum()) / n_own
            factors[i] = 1.0 - kappa * overlap
        covered |= own
    return factors


def component_buildup(ship: ShipSuperstructure, *,
                      angles: Iterable[float] = (0, 30, 60, 90, 120, 150, 180),
                      kappa: float = 0.5,
                      reference: Mapping[str, float] | None = None) -> dict:
    """A-priori engineering estimate of the wind coefficients (no fitting).

    Each box contributes ``0.5 rho U^2 C_d A_i(theta) f_i(theta)`` along the
    wind direction, with ``f_i`` the shadowing factor.  The resulting force is
    rotated into the ship frame and normalised with the analytic reference
    areas, so the absolute level is a genuine prediction.
    """

    ref = dict(reference) if reference is not None else dict(
        projected_areas(ship))
    A_F, A_L = ref["area_frontal_m2"], ref["area_lateral_m2"]
    B, H = ref["breadth_m"], ref["height_m"]
    rows = []
    for a in angles:
        th = math.radians(a)
        f = shadow_factor(ship.boxes, th, kappa=kappa)
        F = Fx = Fy = K = N = 0.0
        for b, fac in zip(ship.boxes, f):
            A_i = _box_projected_shadow(b, th)
            if A_i <= 0.0:
                continue
            d = 0.5 * b.cd * A_i * fac          # per unit q
            F += d
            Fx += d * math.cos(th)
            Fy += d * math.sin(th)
            K += d * 0.5 * (b.z0 + b.z1)        # heel lever arm about x
            N += d * 0.5 * (b.x0 + b.x1) * math.sin(th)   # yaw lever arm
        rows.append({
            "angle_deg": float(a),
            "C_X": Fx / A_F,
            "C_Y": Fy / A_L,
            "C_K": K / (A_L * H),
            "C_N": N / (A_L * B),
            "C_F": F / A_F,
            "method": "component_buildup",
            "kappa": kappa,
            "n_components": len(ship.boxes),
        })
    return {"ship": ship.key, "method": "component_buildup",
            "reference": ref, "angles_deg": [r["angle_deg"] for r in rows],
            "rows": rows}


# ---------------------------------------------------------------------------
# Layer 2: Blendermann-style closed-form correlation
# ---------------------------------------------------------------------------

def blendermann_style(theta_deg: float, *,
                      cx0: float, cx90: float, cy0: float,
                      p: float = 1.5, q: float = 1.5) -> dict:
    """Closed-form wind-coefficient correlation (Blendermann-type shape).

    The structural form used by Blendermann-type ship wind-load correlations is
    reproduced: a head-wind drag ``cx0`` decaying as ``|cos th|^p`` towards a
    small but finite beam-on value ``cx90`` (the longitudinal force at beam wind
    is *not* zero -- the separated wake on the lee side produces a net
    longitudinal force), and a transverse force rising as ``|sin th|^q`` with
    odd symmetry in ``theta``::

        C_X(theta) = cx90 + (cx0 - cx90) |cos theta|^p
        C_Y(theta) = cy0 * sign(sin theta) * |sin theta|^q

    The parameters are per-ship-type fits (:func:`fit_blendermann_style`); no
    published coefficient table is reproduced verbatim.
    """

    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    cx = cx90 + (cx0 - cx90) * abs(c) ** p
    cy = cy0 * (1.0 if s >= 0 else -1.0) * abs(s) ** q
    return {"C_X": cx, "C_Y": cy}


def fit_blendermann_style(samples: Sequence[Mapping[str, float]]) -> dict:
    """Least-squares fit of :func:`blendermann_style` to (angle, C_X, C_Y) data.

    ``C_X`` is linear in ``(cx0, cx90)`` for fixed ``p`` and ``C_Y`` linear in
    ``cy0`` for fixed ``q``; ``p`` and ``q`` are scanned on a grid.  ``C_K`` and
    ``C_N`` are then regressed on ``C_X`` and ``C_Y`` respectively, so the full
    four-coefficient set stays internally consistent with the fit.
    """

    ang = np.array([s["angle_deg"] for s in samples], dtype=float)
    cx = np.array([s["C_X"] for s in samples], dtype=float)
    cy = np.array([s["C_Y"] for s in samples], dtype=float)
    ck = np.array([s.get("C_K", 0.0) for s in samples], dtype=float)
    cn = np.array([s.get("C_N", 0.0) for s in samples], dtype=float)
    th = np.radians(ang)

    best = None
    for p in np.arange(0.8, 3.01, 0.05):
        ac = np.abs(np.cos(th)) ** p
        A = np.vstack([ac, np.ones_like(ac)]).T
        sol, *_ = np.linalg.lstsq(A, cx, rcond=None)
        resid = float(np.sum((A @ sol - cx) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(p), float(sol[0]), float(sol[1]))
    _, p, cx0, cx90 = best

    bestq = None
    for q in np.arange(0.8, 3.01, 0.05):
        basis = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** q
        denom = float(np.sum(basis ** 2))
        cy0 = float(np.sum(basis * cy) / denom) if denom else 0.0
        resid = float(np.sum((basis * cy0 - cy) ** 2))
        if bestq is None or resid < bestq[0]:
            bestq = (resid, float(q), cy0)
    _, q, cy0 = bestq

    model = np.array([blendermann_style(a, cx0=cx0, cx90=cx90, cy0=cy0, p=p, q=q)
                      for a in ang])
    mcx, mcy = model[:, 0], model[:, 1]
    kfac = float(np.sum(ck * cx) / np.sum(cx * cx)) if np.any(cx) else 0.0
    nfac = float(np.sum(cn * cy) / np.sum(cy * cy)) if np.any(cy) else 0.0
    return {
        "params": {"cx0": cx0, "cx90": cx90, "cy0": cy0, "p": p,
                   "q": q, "k_factor": kfac, "n_factor": nfac},
        "n_samples": int(len(ang)),
        "rmse_C_X": float(np.sqrt(np.mean((mcx - cx) ** 2))),
        "rmse_C_Y": float(np.sqrt(np.mean((mcy - cy) ** 2))),
        "max_rel_err_C_X": float(np.max(np.abs(mcx - cx) /
                                       np.maximum(np.abs(cx), 1e-9))),
        "max_rel_err_C_Y": float(np.max(np.abs(mcy - cy) /
                                       np.maximum(np.abs(cy), 1e-9))),
    }


def blendermann_coefficients(fit: Mapping[str, float], theta_deg: float) -> dict:
    """Evaluate a fitted correlation at one angle (all four coefficients)."""

    p = fit["params"] if "params" in fit else fit
    base = blendermann_style(theta_deg, cx0=p["cx0"], cx90=p["cx90"],
                             cy0=p["cy0"], p=p["p"], q=p["q"])
    return {"angle_deg": float(theta_deg) % 360.0, "C_X": base["C_X"],
            "C_Y": base["C_Y"],
            "C_K": p["k_factor"] * base["C_X"],
            "C_N": p["n_factor"] * base["C_Y"],
            "method": "blendermann_style"}


# ---------------------------------------------------------------------------
# Layer 3: hybrid engine
# ---------------------------------------------------------------------------

def complexity_index(ship: ShipSuperstructure,
                     reference: Mapping[str, float] | None = None) -> dict:
    """Geometry complexity metrics used by the engine's routing rule.

    * ``n_boxes`` -- number of above-deck blocks;
    * ``blockage`` -- superstructure solid volume over its bounding volume (a
      compact deckhouse scores high, a spread-out stow scores low);
    * ``stow_ratio`` -- fraction of the frontal area carried by tall blocks *in
      front of* the accommodation (container stows, hatch covers, cranes):
      frontal elements upstream of the house generate the separated wake that
      empirical correlations do not model.
    """

    ref = dict(reference) if reference is not None else dict(
        projected_areas(ship))
    A_F = ref["area_frontal_m2"]
    lx, ly, lz = ship.extent
    blocks_front = 0.0
    for b in ship.boxes:
        if b.z1 < 6.0 or b.porous:
            continue
        if "house" in b.name or "bridge" in b.name or "casing" in b.name:
            continue
        blocks_front += _box_projected_shadow(b, 0.0)
    return {
        "n_boxes": len(ship.boxes),
        "blockage": ship.solid_volume / max(lx * ly * lz, 1e-9),
        "stow_ratio": blocks_front / A_F if A_F else 0.0,
        "frontal_area_m2": A_F,
        "lateral_over_frontal": (ref["area_lateral_m2"] / A_F) if A_F else 0.0,
    }


def _catmull_rom_periodic(theta_deg: float, table: Sequence[float],
                          angles: Sequence[float]) -> float:
    """Periodic Catmull-Rom interpolation of a 360 deg periodic table."""

    n = len(angles)
    t = (theta_deg % 360.0 + 360.0) % 360.0
    ang = np.asarray(angles, dtype=float)
    i = int(np.searchsorted(ang, t, side="right") - 1)
    if i < 0:
        i = n - 1
    a0 = ang[i]
    a1 = ang[(i + 1) % n] + (360.0 if i + 1 >= n else 0.0)
    h = a1 - a0
    u = (t - a0) / h
    y0, y1 = table[(i - 1) % n], table[i]
    y2, y3 = table[(i + 1) % n], table[(i + 2) % n]
    return float(0.5 * ((2 * y1) + (-y0 + y2) * u
                        + (2 * y0 - 5 * y1 + 4 * y2 - y3) * u ** 2
                        + (-y0 + 3 * y1 - 3 * y2 + y3) * u ** 3))


def _mirror_table(samples: Sequence[Mapping[str, float]]) -> dict:
    """Expand 0..180 deg samples to a 360 deg periodic table.

    Uses the port/starboard mirror antisymmetry of the catalogue geometries
    (checked by :func:`sign_conventions`): ``C_X(360-a) = C_X(a)``,
    ``C_K(360-a) = C_K(a)``, ``C_Y(360-a) = -C_Y(a)``, ``C_N(360-a) = -C_N(a)``.
    """

    by = {}
    for s in samples:
        by[round(float(s["angle_deg"]) % 360.0, 6)] = s
    rows = {}
    for a, s in by.items():
        rows[a] = {k: float(s[k]) for k in ("C_X", "C_Y", "C_K", "C_N")}
        if 0.0 < a < 180.0:
            rows[round(360.0 - a, 6)] = {
                "C_X": float(s["C_X"]), "C_Y": -float(s["C_Y"]),
                "C_K": float(s["C_K"]), "C_N": -float(s["C_N"])}
    angles = sorted(rows)
    return {"angles_deg": angles, "rows": rows}