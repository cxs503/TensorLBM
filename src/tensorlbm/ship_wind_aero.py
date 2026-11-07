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

Everything is written in the *ship frame* of the LBM solver
(:mod:`tensorlbm.wind_load_lng`): ``+x`` points forward (bow), ``+y`` to
starboard, ``+z`` up from the free deck at ``z = 0``.  ``theta`` is the
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
        # Cb-vs-slenderness consistency: fine hulls need slenderness, full
        # hulls are by construction short and beamy (Cb/L/B sanity check)
        lb = self.length_pp_m / self.breadth_m
        if self.block_coefficient < 0.72 and lb < 5.0:
            bad.append(f"fine form Cb={self.block_coefficient:.2f} with L/B={lb:.2f} < 5")
        if self.block_coefficient > 0.80 and lb > 7.6:
            bad.append(f"full form Cb={self.block_coefficient:.2f} with L/B={lb:.2f} > 7.6")
        fn = self.design_speed_ms / math.sqrt(9.81 * self.length_pp_m)
        if not 0.05 <= fn <= 0.30:
            bad.append(f"design Froude number {fn:.3f} outside 0.05..0.30")
        return bad

    @property
    def design_speed_ms(self) -> float:
        return self.design_speed_kn * 0.514444


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
        BoxSpec(-16.0, -14.0, -15.0, -6.5, 0.0, 3.0, "aft_bulwark_ps",
                CD_HANDBOOK["plate"]),
        BoxSpec(-16.0, -14.0, 6.5, 15.0, 0.0, 3.0, "aft_bulwark_sb",
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
        BoxSpec(-2.0, -1.0, -2.0, 2.0, 26.0, 33.0, "mast",
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
        BoxSpec(4.0, 5.0, -1.5, 1.5, 24.0, 31.0, "mast",
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
                      p: float = 1.5, q: float = 1.5,
                      cx180: float | None = None) -> dict:
    """Closed-form wind-coefficient correlation (Blendermann-type shape).

    The structural form used by Blendermann-type ship wind-load correlations is
    reproduced: a head-wind drag ``cx0``, a finite beam-on value ``cx90`` (the
    longitudinal force at beam wind is *not* zero -- the separated wake on the
    lee side produces a net longitudinal force) and an astern value ``cx180``
    (normally *negative*: a following wind propels the ship), interpolated with
    ``|cos theta|^p`` towards the beam value, plus an odd transverse force
    rising as ``|sin theta|^q``::

        c = cos(theta),  s = sin(theta)
        C_X(theta) = cx90 + (cx0 - cx90) max(c,0)^p
                          + (cx180 - cx90) max(-c,0)^p
        C_Y(theta) = cy0 sign(s) |s|^q

    ``cx180`` defaults to ``0.9 * cx0`` with a negative sign (a following wind
    gives thrust of roughly the head-wind magnitude); it is fitted from the
    database whenever an astern sample exists.  The parameters are per
    ship-type fits (:func:`fit_blendermann_style`); no published coefficient
    table is reproduced verbatim.
    """

    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    if cx180 is None:
        cx180 = -0.9 * abs(cx0)
    head = max(c, 0.0) ** p
    astern = max(-c, 0.0) ** p
    cx = cx90 + (cx0 - cx90) * head + (cx180 - cx90) * astern
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
        head = np.maximum(np.cos(th), 0.0) ** p
        astern = np.maximum(-np.cos(th), 0.0) ** p
        A = np.vstack([head, astern, np.ones_like(head)]).T
        sol, *_ = np.linalg.lstsq(A, cx, rcond=None)
        resid = float(np.sum((A @ sol - cx) ** 2))
        if best is None or resid < best[0]:
            best = (resid, float(p), float(sol[0]), float(sol[2]), float(sol[1]))
    _, p, a_head, cx90, a_astern = best
    cx0 = a_head + cx90
    cx180 = a_astern + cx90

    bestq = None
    for q in np.arange(0.8, 3.01, 0.05):
        basis = np.sign(np.sin(th)) * np.abs(np.sin(th)) ** q
        denom = float(np.sum(basis ** 2))
        cy0 = float(np.sum(basis * cy) / denom) if denom else 0.0
        resid = float(np.sum((basis * cy0 - cy) ** 2))
        if bestq is None or resid < bestq[0]:
            bestq = (resid, float(q), cy0)
    _, q, cy0 = bestq

    preds = [blendermann_style(a, cx0=cx0, cx90=cx90, cy0=cy0, p=p, q=q,
                               cx180=cx180) for a in ang]
    mcx = np.array([d["C_X"] for d in preds])
    mcy = np.array([d["C_Y"] for d in preds])
    kfac = float(np.sum(ck * cx) / np.sum(cx * cx)) if np.any(cx) else 0.0
    nfac = float(np.sum(cn * cy) / np.sum(cy * cy)) if np.any(cy) else 0.0
    return {
        "params": {"cx0": cx0, "cx90": cx90, "cx180": cx180, "cy0": cy0,
                   "p": p, "q": q, "k_factor": kfac, "n_factor": nfac},
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
                             cy0=p["cy0"], p=p["p"], q=p["q"],
                             cx180=p.get("cx180"))
    return {"angle_deg": float(theta_deg) % 360.0, "C_X": base["C_X"],
            "C_Y": base["C_Y"],
            "C_K": p["k_factor"] * base["C_X"],
            "C_N": p["n_factor"] * base["C_Y"],
            "method": "blendermann_style"}


# ---------------------------------------------------------------------------
# Layer 3: hybrid engine
# ---------------------------------------------------------------------------

def _union_area(boxes: Sequence[BoxSpec], axis: str, *, nx: int = 800,
                ny: int = 800) -> float:
    """Rasterised area of the union of box silhouettes on a coordinate plane."""

    if axis == "x":                        # frontal silhouette (y, z)
        a_lo = min(b.y0 for b in boxes); a_hi = max(b.y1 for b in boxes)
        c_lo = min(b.z0 for b in boxes); c_hi = max(b.z1 for b in boxes)
        pairs = [(b.y0, b.y1, b.z0, b.z1) for b in boxes]
    else:                                  # lateral silhouette (x, z)
        a_lo = min(b.x0 for b in boxes); a_hi = max(b.x1 for b in boxes)
        c_lo = min(b.z0 for b in boxes); c_hi = max(b.z1 for b in boxes)
        pairs = [(b.x0, b.x1, b.z0, b.z1) for b in boxes]
    av = np.linspace(a_lo, a_hi, nx)
    cv = np.linspace(c_lo, c_hi, ny)
    grid = np.zeros((nx, ny), dtype=bool)
    for a0, a1, c0, c1 in pairs:
        grid |= ((av >= a0) & (av <= a1))[:, None] & \
                ((cv >= c0) & (cv <= c1))[None, :]
    return float(grid.sum()) * (av[1] - av[0]) * (cv[1] - cv[0])


def complexity_index(ship: ShipSuperstructure,
                     reference: Mapping[str, float] | None = None,
                     *, tall_m: float = 6.0) -> dict:
    """Geometry complexity metrics used by the engine's routing rule.

    * ``distributed_blockage`` -- **routing metric**: union frontal silhouette
      area of the blocks that are *not* part of the main accommodation/casing
      cluster and rise more than ``tall_m`` above deck, divided by ``A_F``.
      These are the secondary tall structures (container stows ahead of the
      house, deck crane columns, ...) whose interference wake a few-parameter
      silhouette correlation cannot represent; low deck furniture (pipes,
      hatch covers, manifold) is excluded because the component method handles
      it acceptably.
    * ``n_boxes`` -- number of above-deck blocks;
    * ``blockage`` -- superstructure solid volume over its bounding volume.
    """

    ref = dict(reference) if reference is not None else dict(
        projected_areas(ship))
    A_F = ref["area_frontal_m2"]
    lx, ly, lz = ship.extent
    secondary = [b for b in ship.boxes
                 if (b.z1 - b.z0) >= tall_m
                 and "house" not in b.name and "bridge" not in b.name
                 and "casing" not in b.name]
    if secondary:
        A_sec = _union_area(secondary, "x")
    else:
        A_sec = 0.0
    return {
        "n_boxes": len(ship.boxes),
        "blockage": ship.solid_volume / max(lx * ly * lz, 1e-9),
        "distributed_blockage": A_sec / A_F if A_F else 0.0,
        "secondary_tall_boxes": [b.name for b in secondary],
        "frontal_area_m2": A_F,
        "lateral_over_frontal": (ref["area_lateral_m2"] / A_F) if A_F else 0.0,
    }


def _pchip_periodic(theta_deg: float, table: Sequence[float],
                    angles: Sequence[float]) -> float:
    """Shape-preserving (Fritsch-Carlson limited) periodic interpolation.

    Plain cubic (Catmull-Rom) interpolation overshoots badly across the 60 deg
    gaps that remain in a leave-one-angle-out test of a 7-point polar -- the
    beam-wind minimum of ``C_X`` and the zero crossing of ``C_Y`` are exactly
    where that overshoot is largest.  The monotone limiter of Fritsch & Carlson
    removes the overshoot while keeping smoothness, so the same routine is used
    for the shipped table and for the cross-validation.
    """

    n = len(angles)
    ang = np.asarray(angles, dtype=float)
    y = np.asarray(table, dtype=float)
    # periodic node spacing and secant slopes
    h = np.diff(np.concatenate([ang, [ang[0] + 360.0]]))
    delta = np.diff(np.concatenate([y, [y[0]]])) / h
    # Fritsch-Carlson limiter for the derivative at each node
    m = np.zeros(n)
    for i in range(n):
        d_prev = delta[(i - 1) % n]
        d_next = delta[i % n]
        if d_prev * d_next <= 0.0:
            m[i] = 0.0
        else:
            h_prev = h[(i - 1) % n]
            h_next = h[i % n]
            w1 = 2.0 * h_next + h_prev
            w2 = h_next + 2.0 * h_prev
            m[i] = (w1 + w2) / (w1 / d_prev + w2 / d_next)

    t = (theta_deg % 360.0 + 360.0) % 360.0
    i = int(np.searchsorted(ang, t, side="right") - 1)
    if i < 0:
        i = n - 1
    h_i = h[i % n]
    t_local = t + 360.0 if i == n - 1 else t
    u = (t_local - ang[i]) / h_i
    y0 = y[i]
    y1 = y[(i + 1) % n]
    m0 = m[i]
    m1 = m[(i + 1) % n]
    u2, u3 = u * u, u * u * u
    return float((2 * u3 - 3 * u2 + 1) * y0 + (u3 - 2 * u2 + u) * h_i * m0
                 + (-2 * u3 + 3 * u2) * y1 + (u3 - u2) * h_i * m1)


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


class HybridWindEngine:
    """Hybrid empirical/LBM wind-coefficient engine for one ship type.

    Parameters
    ----------
    ship
        Catalogued superstructure (supplies the complexity metric).
    database
        Per-angle LBM records with keys ``angle_deg``, ``C_X``, ``C_Y``,
        ``C_K``, ``C_N`` and optionally ``C_X_std``/``C_Y_std``.  May be empty,
        in which case the engine is empirical-only.
    correlation
        Output of :func:`fit_blendermann_style` (the empirical branch).  When
        omitted it is fitted on ``database`` itself.
    route
        ``"auto"`` (default), ``"empirical"`` or ``"lbm"``.
    complexity_threshold
        ``distributed_blockage`` above which the complex (LBM) branch is used.
    blend_width
        Width of the linear blending band in ``distributed_blockage`` units,
        centred on ``complexity_threshold``.
    """

    def __init__(self, ship: ShipSuperstructure, database: Sequence[Mapping] = (),
                 correlation: Mapping | None = None, *, route: str = "auto",
                 complexity_threshold: float = 0.30, blend_width: float = 0.10):
        if route not in ("auto", "empirical", "lbm"):
            raise ValueError("route must be one of auto/empirical/lbm")
        self.ship = ship
        self.route = route
        self.threshold = float(complexity_threshold)
        self.blend_width = float(blend_width)
        self.reference = dict(ship.reference) if ship.reference else dict(
            projected_areas(ship))
        self.complexity = complexity_index(ship, self.reference)

        self.database = [dict(r) for r in database]
        self.table = _mirror_table(self.database) if self.database else None
        self.correlation = (dict(correlation) if correlation is not None
                            else (fit_blendermann_style(self.database)
                                  if self.database else None))
        if self.route == "lbm" and self.table is None:
            raise ValueError("route='lbm' needs a non-empty LBM database")

    # -- routing ---------------------------------------------------------
    def lbm_weight(self) -> float:
        """Weight (0..1) on the LBM branch implied by the complexity metric."""

        if self.route == "lbm":
            return 1.0
        if self.route == "empirical":
            return 0.0
        if self.table is None:
            return 0.0
        r = self.complexity["distributed_blockage"]
        lo = self.threshold - 0.5 * self.blend_width
        hi = self.threshold + 0.5 * self.blend_width
        if r <= lo:
            return 0.0
        if r >= hi:
            return 1.0
        return float((r - lo) / (hi - lo))

    def routing_report(self) -> dict:
        w = self.lbm_weight()
        branch = ("lbm" if w >= 0.999 else
                  "empirical" if w <= 0.001 else "blend")
        return {"branch": branch, "lbm_weight": w,
                "threshold": self.threshold, "blend_width": self.blend_width,
                "complexity": dict(self.complexity),
                "has_lbm_table": self.table is not None,
                "n_lbm_angles": 0 if self.table is None else
                len(self.table["angles_deg"])}

    # -- branches --------------------------------------------------------
    def _lbm_eval(self, theta_deg: float) -> dict:
        t = self.table
        out = {}
        for key in ("C_X", "C_Y", "C_K", "C_N"):
            vals = [t["rows"][a][key] for a in t["angles_deg"]]
            out[key] = _pchip_periodic(theta_deg, vals, t["angles_deg"])
        for key in ("C_X_std", "C_Y_std"):
            vals = [float(t["rows"][a].get(key, 0.0)) for a in t["angles_deg"]]
            out[key] = _pchip_periodic(theta_deg, vals, t["angles_deg"])
        out["method"] = "lbm_table"
        return out

    def _emp_eval(self, theta_deg: float) -> dict:
        if self.correlation is None:
            raise RuntimeError("no empirical correlation available")
        return blendermann_coefficients(self.correlation, theta_deg)

    # -- public ----------------------------------------------------------
    def coefficients(self, theta_deg: float) -> dict:
        """Four wind coefficients at ``theta_deg`` with provenance."""

        w = self.lbm_weight()
        emp = self._emp_eval(theta_deg) if w < 1.0 else None
        lbm = self._lbm_eval(theta_deg) if w > 0.0 else None
        out = {"angle_deg": float(theta_deg) % 360.0, "lbm_weight": w}
        if w <= 0.0:
            src = emp
            out["method"] = "empirical"
        elif w >= 1.0:
            src = lbm
            out["method"] = "lbm_table"
        else:
            src = {k: w * lbm[k] + (1.0 - w) * emp[k]
                   for k in ("C_X", "C_Y", "C_K", "C_N")}
            src["C_X_std"] = w * lbm.get("C_X_std", 0.0)
            src["C_Y_std"] = w * lbm.get("C_Y_std", 0.0)
            out["method"] = "blend"
        for k in ("C_X", "C_Y", "C_K", "C_N", "C_X_std", "C_Y_std"):
            if k in src:
                out[k] = float(src[k])
        if "C_X_std" not in out:
            out["C_X_std"] = 0.0
            out["C_Y_std"] = 0.0
        return out

    def polar(self, angles: Iterable[float] | None = None) -> list[dict]:
        ang = list(angles) if angles is not None else list(range(0, 361, 10))
        return [self.coefficients(a) for a in ang]

    def force(self, theta_deg: float, wind_speed_ms: float, *,
              rho_air: float = RHO_AIR) -> dict:
        """Dimensional forces/moments from the coefficients (ship frame)."""

        c = self.coefficients(theta_deg)
        q = 0.5 * rho_air * wind_speed_ms ** 2
        A_F = self.reference["area_frontal_m2"]
        A_L = self.reference["area_lateral_m2"]
        B = self.reference["breadth_m"]
        H = self.reference["height_m"]
        return {"q_Pa": q, "X_N": c["C_X"] * q * A_F,
                "Y_N": c["C_Y"] * q * A_L,
                "K_Nm": c["C_K"] * q * A_L * H,
                "N_Nm": c["C_N"] * q * A_L * B,
                "coefficients": c}

    def to_dict(self) -> dict:
        return {"ship": self.ship.key, "routing": self.routing_report(),
                "correlation": self.correlation,
                "reference": self.reference}

    @classmethod
    def from_records(cls, ship: ShipSuperstructure,
                     records: Sequence[Mapping], **kw) -> "HybridWindEngine":
        """Build the engine from LBM run records (``coefficients`` sub-dict)."""

        samples = []
        for r in records:
            coeff = r.get("coefficients", r)
            samples.append({"angle_deg": float(r["angle_deg"]),
                            "C_X": float(coeff["C_X"]),
                            "C_Y": float(coeff["C_Y"]),
                            "C_K": float(coeff.get("C_K", 0.0)),
                            "C_N": float(coeff.get("C_N", 0.0)),
                            "C_X_std": float(coeff.get("C_X_std", 0.0)),
                            "C_Y_std": float(coeff.get("C_Y_std", 0.0))})
        return cls(ship, samples, **kw)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def sign_conventions(samples: Sequence[Mapping] | None = None) -> dict:
    """Document and (when data is supplied) check the coefficient conventions.

    Definitions (ship frame, wind blowing *from* ``theta`` measured from the
    bow): ``C_X > 0`` is a retarding longitudinal force, ``C_Y`` follows the
    transverse force on ``+y`` (starboard, so a beam wind from starboard gives
    ``C_Y < 0``), ``C_K`` is the heel moment about ``+x`` and ``C_N`` the yaw
    moment about ``+z``.

    With ``samples`` the *port/starboard* symmetry of the catalogue geometry is
    used as a numerical self-check.  The symmetry that holds for every one of
    these ships is the one about the centreline plane, i.e. ``theta`` and
    ``-theta``: the ballistic range 0..180 deg cannot test it directly, but the
    consequence ``C_Y(0) = C_Y(180) = 0`` and ``C_N(0) = C_N(180) = 0`` can.
    Those residuals measure how much port/starboard asymmetry the discretised
    LBM campaign introduces (a fore-aft mirror such as ``C_X(180-a) = C_X(a)``
    is *not* used: real superstructures and the head/astern force difference
    break it, and the data show it).
    """

    out: dict = {
        "frame": {"x": "+forward (bow)", "y": "+starboard",
                  "z": "+up (deck at z=0)"},
        "theta_deg": "wind direction measured from the bow (0 = head wind, "
                     "90 = beam wind from starboard)",
        "C_X": "X_wind / (q A_F), positive = retarding",
        "C_Y": "Y_wind / (q A_L), sign follows the transverse force in +y; a "
               "beam wind from starboard gives C_Y < 0 (force towards port)",
        "C_K": "heel moment about +x / (q A_L H)",
        "C_N": "yaw moment about +z / (q A_L B)",
        "checks": [],
    }
    if not samples:
        return out
    by = {round(float(s["angle_deg"]) % 360.0, 6): s for s in samples}
    scale_y = max((abs(float(s["C_Y"])) for s in samples), default=0.0) or 1.0
    scale_n = max((abs(float(s["C_N"])) for s in samples), default=0.0) or 1.0
    checks = []
    for a in (0.0, 180.0):
        if a not in by:
            continue
        checks.append({
            "angle_deg": a,
            "C_Y": float(by[a]["C_Y"]),
            "C_N": float(by[a]["C_N"]),
            "residual_side_force_rel": abs(float(by[a]["C_Y"])) / scale_y,
            "residual_yaw_moment_rel": abs(float(by[a]["C_N"])) / scale_n,
        })
    out["checks"] = checks
    out["max_residual_side_force_rel"] = max(
        (c["residual_side_force_rel"] for c in checks), default=None)
    out["max_residual_yaw_moment_rel"] = max(
        (c["residual_yaw_moment_rel"] for c in checks), default=None)
    for a in (0.0, 90.0, 180.0):
        if a in by:
            out[f"C_X_at_{int(a)}"] = float(by[a]["C_X"])
            out[f"C_Y_at_{int(a)}"] = float(by[a]["C_Y"])
    return out


def _weibull_pdf(v: float, k: float, c: float) -> float:
    if v <= 0.0:
        return 0.0
    return (k / c) * (v / c) ** (k - 1.0) * math.exp(-(v / c) ** k)


def synthetic_wind_rose(name: str = "north_atlantic") -> dict:
    """A representative open-ocean wind rose (documented, *not* measured data).

    Eight 45 deg sectors, occurrence probability, and a Weibull (k, c) pair per
    sector.  The numbers are representative of North Atlantic route statistics
    used for annual wind-resistance studies: prevailing westerlies (sectors
    around 270 deg) with ``c ~ 10 m/s``, the eastern quadrant calmer, mean wind
    speed over the rose ~7.5 m/s.  They are **documented representative values**
    and must be replaced by measured route statistics for a real ship study.
    """

    if name != "north_atlantic":
        raise ValueError("only the 'north_atlantic' representative rose is built in")
    sectors = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    prob = [0.085, 0.075, 0.080, 0.100, 0.130, 0.150, 0.210, 0.170]
    c = [9.0, 8.0, 7.0, 7.5, 9.0, 9.5, 10.5, 10.0]
    k = [2.0, 2.0, 1.9, 1.9, 2.0, 2.1, 2.2, 2.1]
    s = sum(prob)
    prob = [p / s for p in prob]
    mean = 0.0
    for p, kk, cc in zip(prob, k, c):
        mean += p * cc * math.gamma(1.0 + 1.0 / kk)
    return {"name": name, "sectors_deg_from": sectors, "probability": prob,
            "weibull_shape_k": k, "weibull_scale_c": c,
            "mean_wind_speed_ms": mean,
            "provenance": "representative North Atlantic route statistics "
                          "(documented values, not measured data)"}


def calm_water_resistance_estimate(hull: HullReference, speed_ms: float,
                                   *, rho_water: float = 1025.0,
                                   form_factor: float = 1.15,
                                   wetted_area_factor: float = 0.90) -> dict:
    """Order-of-magnitude calm-water resistance (ITTC-57 friction dominated).

    ``S = wetted_area_factor * Lpp (2T + B)`` (a coarse wetted-area
    approximation, ~10% for conventional hulls, stated as such) and
    ``R = 0.5 rho S V^2 C_F (1+k)`` with the ITTC-57 friction line.  Used only
    to quote the wind resistance as a percentage; it is *not* a resistance
    prediction tool.
    """

    S = wetted_area_factor * hull.length_pp_m * (2.0 * hull.draught_m + hull.breadth_m)
    Re = speed_ms * hull.length_pp_m / 1.19e-6
    cf = 0.075 / (math.log10(Re) - 2.0) ** 2
    r = 0.5 * rho_water * S * speed_ms ** 2 * cf * form_factor
    return {"wetted_area_m2": S, "Re": Re, "C_F": cf,
            "form_factor": form_factor, "R_calm_water_N": r,
            "provenance": "ITTC-57 friction + coarse wetted-area estimate"}


def wind_rose_annual_resistance(engine: HybridWindEngine, rose: Mapping,
                                *, ship_speed_ms: float, heading_deg: float = 0.0,
                                rho_air: float = RHO_AIR,
                                n_speed: int = 48, v_max_factor: float = 3.0,
                                baseline_resistance_N: float | None = None,
                                extra_frontal_area_m2: float = 0.0) -> dict:
    """Annual mean wind added resistance from a wind rose.

    For every rose sector the *apparent* wind is obtained from the true wind
    vector minus the ship velocity vector (no current, no wave-orbit
    correction), the apparent wind angle feeds :meth:`HybridWindEngine.
    coefficients`, and the longitudinal wind force is

        ``dR = 0.5 rho_air A_F C_X(theta_a) V_a^2``

    (positive = retarding, negative = thrust).  The sector Weibull speed
    distribution is integrated with a midpoint rule on ``n_speed`` bins out to
    ``v_max_factor * c``.  Reported: the probability-weighted means of ``dR``
    (net, positive part only), the probability of a propulsive wind, and -- if
    a baseline resistance is supplied -- the ratio.
    """

    A_F = engine.reference["area_frontal_m2"] + extra_frontal_area_m2
    psi = math.radians(heading_deg)
    hx, hy = math.cos(psi), math.sin(psi)
    prob = list(rose["probability"])
    sectors = list(rose["sectors_deg_from"])
    ks = list(rose["weibull_shape_k"])
    cs = list(rose["weibull_scale_c"])
    total = sum(prob)
    prob = [p / total for p in prob]

    mean_net = 0.0
    mean_pos = 0.0
    p_thrust = 0.0
    per_sector = []
    weights = []
    for ang_from, p_s, k, c in zip(sectors, prob, ks, cs):
        # unit vector of the true wind *velocity* (blowing towards ang_from+180)
        wdir = math.radians(ang_from + 180.0)
        wx, wy = math.cos(wdir), math.sin(wdir)
        vmax = v_max_factor * c
        dv = vmax / n_speed
        s_net = s_pos = s_w = s_thrust = 0.0
        for i in range(n_speed):
            v = (i + 0.5) * dv
            w = _weibull_pdf(v, k, c) * dv
            if w <= 0.0:
                continue
            ax = v * wx - ship_speed_ms * hx
            ay = v * wy - ship_speed_ms * hy
            v_a = math.hypot(ax, ay)
            if v_a < 1e-9:
                continue
            # wind blows along (ax, ay); the coefficient angle is measured from
            # the bow for a wind blowing *from* the opposite direction
            theta = math.degrees(math.atan2(-ay, -ax)) % 360.0
            cx = engine.coefficients(theta)["C_X"]
            dR = 0.5 * rho_air * A_F * cx * v_a ** 2
            s_net += w * dR
            s_pos += w * max(dR, 0.0)
            s_thrust += w * (1.0 if dR < 0.0 else 0.0)
            s_w += w
        if s_w > 0:
            s_net /= s_w
            s_pos /= s_w
            s_thrust /= s_w
        mean_net += p_s * s_net
        mean_pos += p_s * s_pos
        p_thrust += p_s * s_thrust
        per_sector.append({"sector_from_deg": ang_from, "probability": p_s,
                           "mean_dR_net_N": s_net, "mean_dR_positive_N": s_pos,
                           "thrust_fraction": s_thrust})
        weights.append({"sector_from_deg": ang_from, "p": p_s})

    out = {
        "ship": engine.ship.key,
        "heading_deg": heading_deg,
        "ship_speed_ms": ship_speed_ms,
        "mean_wind_added_resistance_N": mean_net,
        "mean_wind_added_resistance_positive_N": mean_pos,
        "thrust_probability": p_thrust,
        "frontal_area_used_m2": A_F,
        "extra_frontal_area_m2": extra_frontal_area_m2,
        "equivalent_added_power_kW": mean_net * ship_speed_ms / 1e3,
        "rose": dict(rose),
        "per_sector": per_sector,
        "engine_routing": engine.routing_report(),
    }
    if baseline_resistance_N:
        out["wind_over_calm_water_pct"] = 100.0 * mean_net / baseline_resistance_N
        out["baseline_resistance_N"] = baseline_resistance_N
    return out


# ---------------------------------------------------------------------------
# Acceptance evaluation
# ---------------------------------------------------------------------------

def evaluate_acceptance(criteria: Sequence[Mapping]) -> dict:
    """Assemble the machine-checkable acceptance table.

    ``criteria`` entries are ``{"id", "description", "value", "threshold",
    "operator" ("<=", ">=", "<", ">"), "unit",
    "hard" (bool: gates the node)}``.  Returns the table plus ``all_hard_passed``
    so the report can state pass/FAIL without rhetoric.
    """

    rows = []
    for c in criteria:
        v = c.get("value")
        th = c.get("threshold")
        op = c.get("operator", "<=")
        if v is None or th is None:
            passed = None
            verdict = "NOT_EVALUATED"
        else:
            passed = {"<=": v <= th, "<": v < th, ">=": v >= th, ">": v > th}[op]
            verdict = "PASS" if passed else "FAIL"
        rows.append({"id": c["id"], "description": c["description"],
                     "value": v, "threshold": th, "operator": op,
                     "unit": c.get("unit", ""), "hard": bool(c.get("hard", True)),
                     "verdict": verdict, "passed": passed})
    hard = [r for r in rows if r["hard"] and r["passed"] is not None]
    soft = [r for r in rows if not r["hard"] and r["passed"] is not None]
    return {"criteria": rows,
            "all_hard_passed": all(r["passed"] for r in hard) if hard else False,
            "n_hard": len(hard),
            "n_hard_passed": sum(1 for r in hard if r["passed"]),
            "n_soft": len(soft),
            "n_soft_passed": sum(1 for r in soft if r["passed"]),
            "n_not_evaluated": sum(1 for r in rows if r["passed"] is None)}