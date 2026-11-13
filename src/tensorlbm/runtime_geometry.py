"""Runtime geometry configuration + compile-time case parameters.

This module reimplements the *input convention* of joconnor22/LIFE
(GPL-3.0), cross-validated against its sources — **no LIFE source code is
copied**, only the documented file format and discretisation rules:

===============================================  ===========================
LIFE reference                                   convention adopted here
===============================================  ===========================
``src/Objects.cpp:geometryReadIn`` (l.324-472)   two-pass scan (count, then
                                                 build), per-keyword columns,
                                                 ``pos = start + i*space``,
                                                 ``ID = start_id + i``
``src/Objects.cpp:350-386``                      reserve estimate
                                                 ``n*floor(2 pi r/dx)`` /
                                                 ``n*(floor(L/dx)+1)``
``src/IBMBody.cpp:47-105``                       circle markers
                                                 ``floor(2 pi r/dx)`` evenly
                                                 spaced; filament markers
                                                 ``floor(L/dx)+1`` along the
                                                 centreline, rotated by
                                                 ``[[c,-s],[s,c]]``
``src/IBMBody.cpp:100``, ``inc/Utils.h:258``     angle input in **degrees**,
                                                 converted to radians once
``src/Objects.cpp:removeOverlapMarkers``         markers closer than ``0.5 dx``
                                                 are deduplicated
                                                 (flexible beats rigid; two
                                                 flexible bodies => error)
``inc/params.h``                                 grid / viscosity / BCs /
                                                 step count as *compile-time*
                                                 constants
``input/geometry.config``                        body list as *runtime* text
===============================================  ===========================

Why this split matters (the practice being cross-validated): in LIFE a new
geometry is a text edit, but a new grid or ``omega`` forces
``make clean && make`` because those live in ``params.h``.  The Python
analogue implemented here is

* :class:`CaseParams` — a **frozen** dataclass ("compile-time" block).  It
  cannot be mutated; :meth:`SimulationSetup.apply_params` refuses an edit
  unless ``rebuild=True`` (explicit re-compilation analogue).  Every run
  records :meth:`CaseParams.fingerprint`, so two runs with different
  physics cannot be silently compared.
* :class:`Scene` — the **runtime** geometry.  Bodies can be added/removed
  at any time (LIFE allows this on restart) and doing so never changes
  ``CaseParams.fingerprint()``.

Coordinate convention: ``geometry.config`` carries *physical* coordinates
exactly as LIFE does (Turek–Hron set-up: height 0.41 m, cylinder at
(0.2, 0.2) m with radius 0.05 m).  :func:`Scene.mask` maps them to lattice
indices through ``x_lattice = (x_phys - origin_x)/dx``; ``origin`` is the
physical position of lattice index 0.  For the in-repo Schäfer–Turek 2D-1
case at ``D=20`` (``dx = 0.005 m``, ``origin = (0, 0)``) the LIFE line
``CIRCLE 1 0 0.2 0.2 0.0 0.0 0.05`` reproduces the verified hard-coded
``cylinder_mask`` bit for bit (see ``tests/test_runtime_geometry.py``).

Deliberate differences versus LIFE (documented, not silent):

1. ``#`` comments are accepted on **any** line; LIFE only skips the leading
   comment block, so a trailing comment makes its ``>>`` reader misparse.
2. Wrong token counts / unknown keywords / malformed numbers raise
   ``ValueError``; LIFE reads into uninitialised or stale variables.
3. Circle bodies are always rigid in LIFE (``eRigid``); here the flag is
   explicit in the dataclass and validation is left to the caller.
4. Bodies are also rasterised to a boolean solid mask
   (:meth:`Scene.mask`) so the same scene can drive half-way bounce-back;
   LIFE is marker-only (IBM with epsilon correction).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "CaseParams",
    "CircleBody",
    "FilamentBody",
    "Scene",
    "SimulationSetup",
    "DEFAULT_GEOMETRY_PATH",
    "parse_geometry_config",
    "read_geometry_config",
    "scene_from_life_case",
    "MASK_RULE",
]

#: Path (relative to a case directory) at which LIFE looks for the body list.
DEFAULT_GEOMETRY_PATH = Path("input") / "geometry.config"

#: Marker de-duplication radius in lattice units (LIFE: ``mag < 0.5``).
OVERLAP_TOL = 0.5

#: Rotation matrix handedness, from ``inc/Utils.h:getRotationMatrix``.
_ROT_SIGN = 1.0

MASK_RULE = (
    "circle: (|x-xc|^2+|y-yc|^2) <= r^2 in lattice-index space with "
    "xc=(x_phys-origin_x)/dx (identical to boundaries.cylinder_mask); "
    "filament: 0<=s<=L and |t|<=height/2 in the body frame, s along the "
    "centreline. Physical filament thickness is height (LIFE geometry column "
    "2); LIFE itself spreads only the centreline markers."
)


# ---------------------------------------------------------------------------
# compile-time block
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CaseParams:
    """Discretisation/physics block — the ``params.h`` analogue.

    Frozen on purpose: in LIFE these are ``const int``/``#define`` values, so
    changing one is a recompilation.  Here the same effect is obtained by
    refusing in-place edits (:meth:`SimulationSetup.apply_params`) and by
    carrying :meth:`fingerprint` into the run provenance.
    """

    nx: int
    ny: int
    tau: float
    steps: int
    dx: float = 1.0
    origin: tuple[float, float] = (0.0, 0.0)
    bc_left: str = "velocity"
    bc_right: str = "pressure"
    bc_bottom: str = "wall"
    bc_top: str = "wall"

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError(f"nx/ny must be positive, got {self.nx}x{self.ny}")
        if self.dx <= 0.0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        if self.tau <= 0.5:
            raise ValueError(f"tau must be > 0.5 for a stable lattice, got {self.tau}")
        if self.steps < 0:
            raise ValueError(f"steps must be >= 0, got {self.steps}")

    def as_dict(self) -> dict[str, Any]:
        return {
            "nx": self.nx,
            "ny": self.ny,
            "tau": self.tau,
            "steps": self.steps,
            "dx": self.dx,
            "origin": list(self.origin),
            "bc_left": self.bc_left,
            "bc_right": self.bc_right,
            "bc_bottom": self.bc_bottom,
            "bc_top": self.bc_top,
        }

    def fingerprint(self) -> str:
        """Stable 16-hex digest of the compile-time block."""
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    # convenience: lattice helpers -------------------------------------------------
    def lattice_xy(self, x_phys: float, y_phys: float) -> tuple[float, float]:
        return (
            (x_phys - self.origin[0]) / self.dx,
            (y_phys - self.origin[1]) / self.dx,
        )


# ---------------------------------------------------------------------------
# runtime bodies
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CircleBody:
    """Rigid circular obstacle (LIFE ``CIRCLE`` keyword)."""

    body_id: int
    centre: tuple[float, float]
    radius: float
    keyword: str = "CIRCLE"
    flexible: bool = False  # LIFE builds circles as eRigid

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError(f"circle radius must be positive, got {self.radius}")

    def marker_count(self, dx: float) -> int:
        """LIFE: ``numNodes = floor(2 pi r / Dx)``."""
        n = int(math.floor(2.0 * math.pi * self.radius / dx))
        if n < 3:
            raise ValueError(
                f"circle {self.body_id} too small for dx={dx}: {n} markers "
                "(LIFE would create a degenerate body)"
            )
        return n

    def markers(self, dx: float) -> np.ndarray:
        n = self.marker_count(dx)
        theta = 2.0 * math.pi * np.arange(n, dtype=float) / n
        cx, cy = self.centre
        return np.column_stack((cx + self.radius * np.cos(theta), cy + self.radius * np.sin(theta)))

    def rasterise(self, nx: int, ny: int, dx: float, origin: tuple[float, float]) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
        xc = (self.centre[0] - origin[0]) / dx
        yc = (self.centre[1] - origin[1]) / dx
        r = self.radius / dx
        return (xx - xc) ** 2 + (yy - yc) ** 2 <= r * r

    def bbox(self) -> tuple[float, float, float, float]:
        cx, cy = self.centre
        return (cx - self.radius, cx + self.radius, cy - self.radius, cy + self.radius)


@dataclass(frozen=True)
class FilamentBody:
    """Two-node-per-element elastic/rigid beam (LIFE ``FILAMENT`` keyword)."""

    body_id: int
    start: tuple[float, float]
    length: float
    height: float
    angle_deg: float
    flexible: bool = True
    n_elements: int = 20
    bc: str = "CLAMPED"
    rho: float = 1000.0
    youngs_modulus: float = 1.0e6
    keyword: str = "FILAMENT"

    def __post_init__(self) -> None:
        if self.length <= 0.0:
            raise ValueError(f"filament length must be positive, got {self.length}")
        if self.height <= 0.0:
            raise ValueError(f"filament height must be positive, got {self.height}")
        if self.n_elements < 1:
            raise ValueError(f"filament n_elements must be >= 1, got {self.n_elements}")
        if self.bc not in {"CLAMPED", "SUPPORTED"}:
            raise ValueError(f"filament BC must be CLAMPED or SUPPORTED, got {self.bc!r}")

    @property
    def angle_rad(self) -> float:
        """LIFE converts degrees to radians once, at construction."""
        return self.angle_deg * math.pi / 180.0

    def rotation(self) -> np.ndarray:
        a = _ROT_SIGN * self.angle_rad
        return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def marker_count(self, dx: float) -> int:
        """LIFE: ``numNodes = floor(L / Dx) + 1``."""
        n = int(math.floor(self.length / dx)) + 1
        if n < 2:
            raise ValueError(
                f"filament {self.body_id} shorter than dx={dx} "
                f"(floor(L/dx)+1 = {n} nodes)"
            )
        return n

    def markers(self, dx: float) -> np.ndarray:
        n = self.marker_count(dx)
        s = np.arange(n, dtype=float) * (self.length / (n - 1))
        rot = self.rotation()
        local = np.column_stack((s, np.zeros_like(s)))
        return np.asarray(self.start, dtype=float) + local @ rot.T

    def rasterise(self, nx: int, ny: int, dx: float, origin: tuple[float, float]) -> np.ndarray:
        yy, xx = np.meshgrid(np.arange(ny, dtype=float), np.arange(nx, dtype=float), indexing="ij")
        px = (xx * dx + origin[0]) - self.start[0]
        py = (yy * dx + origin[1]) - self.start[1]
        rot = self.rotation()
        s = rot[0, 0] * px + rot[1, 0] * py
        t = rot[0, 1] * px + rot[1, 1] * py
        return (s >= 0.0) & (s <= self.length) & (np.abs(t) <= 0.5 * self.height)

    def beam_properties(self, dx: float) -> dict[str, float]:
        """Cross-section used by LIFE's FEM body (``A = Dx*h``, ``I = Dx*h^3/12``)."""
        h = self.height
        return {
            "A": dx * h,
            "I": dx * h**3 / 12.0,
            "length": self.length,
            "n_elements": float(self.n_elements),
            "rho": self.rho,
            "E": self.youngs_modulus,
        }

    def bbox(self) -> tuple[float, float, float, float]:
        pts = self.markers_at_endpoints()
        xs, ys = pts[:, 0], pts[:, 1]
        h = 0.5 * self.height
        return (xs.min() - h, xs.max() + h, ys.min() - h, ys.max() + h)

    def markers_at_endpoints(self) -> np.ndarray:
        rot = self.rotation()
        local = np.array([[0.0, 0.0], [self.length, 0.0]])
        return np.asarray(self.start, dtype=float) + local @ rot.T


Body = CircleBody | FilamentBody


# ---------------------------------------------------------------------------
# runtime scene
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    """Runtime body list (the ``geometry.config`` content).

    Editable on purpose: adding/removing bodies is the runtime edit that LIFE
    allows between (restart) runs without touching ``params.h``.
    """

    bodies: list[Body] = field(default_factory=list)

    # -- editing -----------------------------------------------------------
    def add(self, body: Body) -> None:
        if any(b.body_id == body.body_id for b in self.bodies):
            raise ValueError(f"duplicate body id {body.body_id} in scene")
        self.bodies.append(body)

    def remove(self, body_id: int) -> Body:
        for i, b in enumerate(self.bodies):
            if b.body_id == body_id:
                return self.bodies.pop(i)
        raise KeyError(f"body id {body_id} not in scene")

    def ids(self) -> list[int]:
        return [b.body_id for b in self.bodies]

    # -- LIFE first pass: reserve estimates --------------------------------
    def counts(self, dx: float) -> dict[str, int]:
        """Body/marker counts exactly as LIFE's pre-scan computes them."""
        bodies = len(self.bodies)
        nodes = 0
        for b in self.bodies:
            n = b.marker_count(dx)
            nodes += n
        return {"bodies": bodies, "markers": nodes}

    # -- markers -----------------------------------------------------------
    def markers(self, dx: float, *, dedupe: bool = True) -> np.ndarray:
        if not self.bodies:
            return np.empty((0, 2))
        chunks = [b.markers(dx) for b in self.bodies]
        pts = np.vstack(chunks)
        owners = np.concatenate(
            [[i] * len(c) for i, c in enumerate(chunks)]  # body index per marker
        )
        if not dedupe:
            return pts
        keep = _overlap_keep(pts, owners, [b.flexible for b in self.bodies], dx)
        return pts[keep]

    # -- rasterisation -----------------------------------------------------
    def mask(self, nx: int, ny: int, dx: float, origin: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
        mask = np.zeros((ny, nx), dtype=bool)
        for b in self.bodies:
            mask |= b.rasterise(nx, ny, dx, origin)
        return mask

    def validate_domain(self, nx: int, ny: int, dx: float, origin: tuple[float, float] = (0.0, 0.0)) -> None:
        """Reject bodies that leave the lattice (silently clipped by the mask)."""
        lx = nx * dx
        ly = ny * dx
        for b in self.bodies:
            x0, x1, y0, y1 = b.bbox()
            if x0 < origin[0] or y0 < origin[1] or x1 > origin[0] + lx or y1 > origin[1] + ly:
                raise ValueError(
                    f"body {b.body_id} ({b.keyword}) bbox "
                    f"[{x0:.4g},{x1:.4g}]x[{y0:.4g},{y1:.4g}] leaves the domain "
                    f"[{origin[0]:.4g},{origin[0] + lx:.4g}]x[{origin[1]:.4g},{origin[1] + ly:.4g}]"
                )

    def __len__(self) -> int:
        return len(self.bodies)


def _overlap_keep(pts: np.ndarray, owners: np.ndarray, flexible: Sequence[bool], dx: float) -> np.ndarray:
    """Replicate ``ObjectsClass::removeOverlapMarkers`` (LIFE, 0.5 dx rule)."""
    n = len(pts)
    drop = np.zeros(n, dtype=bool)
    for i in range(n):
        if drop[i]:
            continue
        for j in range(i + 1, n):
            if drop[j]:
                continue
            d = math.dist(pts[i], pts[j]) / dx
            if d >= OVERLAP_TOL:
                continue
            fi, fj = flexible[owners[i]], flexible[owners[j]]
            if fi and fj:
                raise ValueError(
                    f"markers {i} and {j} of two flexible bodies are {d:.3f} dx apart "
                    "(LIFE: 'Cannot place two flexible bodies so close')"
                )
            # LIFE keeps the flexible marker; for rigid-vs-rigid it drops the
            # earlier-indexed one (src/Objects.cpp:removeOverlapMarkers).
            if fi and not fj:
                drop[j] = True
                continue
            drop[i] = True
            break
    return ~drop


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
_GENERAL_COLUMNS = 7  # keyword number id x y space_x space_y


def parse_geometry_config(text: str, *, source: str = "<string>") -> Scene:
    """Parse LIFE-format ``geometry.config`` text into a :class:`Scene`."""
    scene = Scene()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        keyword = tokens[0].upper()
        where = f"{source}:{lineno}"

        if len(tokens) < _GENERAL_COLUMNS:
            raise ValueError(
                f"{where}: {keyword} line needs at least {_GENERAL_COLUMNS} columns "
                f"(KEYWORD NUMBER ID X Y SPACING_X SPACING_Y), got {len(tokens)}"
            )
        try:
            number = int(tokens[1])
            start_id = int(tokens[2])
            x = float(tokens[3])
            y = float(tokens[4])
            sx = float(tokens[5])
            sy = float(tokens[6])
        except ValueError as exc:
            raise ValueError(f"{where}: non-numeric general column ({exc})") from exc
        if number < 1:
            raise ValueError(f"{where}: NUMBER must be >= 1, got {number}")

        specific = tokens[_GENERAL_COLUMNS:]
        if keyword == "CIRCLE":
            if len(specific) != 1:
                raise ValueError(
                    f"{where}: CIRCLE expects 1 specific column (RADIUS), got {len(specific)}"
                )
            radius = float(specific[0])
            for i in range(number):
                scene.add(
                    CircleBody(
                        body_id=start_id + i,
                        centre=(x + i * sx, y + i * sy),
                        radius=radius,
                    )
                )
        elif keyword == "FILAMENT":
            if len(specific) != 8:
                raise ValueError(
                    f"{where}: FILAMENT expects 8 specific columns "
                    f"(LENGTH HEIGHT ANGLE FLEX_RIGID N_ELEMENTS BC DENSITY YOUNG_MODULUS), "
                    f"got {len(specific)}"
                )
            length, height, angle_deg = (float(v) for v in specific[:3])
            flex_tok = specific[3].upper()
            if flex_tok not in {"FLEXIBLE", "RIGID"}:
                raise ValueError(f"{where}: FLEX_RIGID must be FLEXIBLE or RIGID, got {flex_tok!r}")
            n_elements = int(specific[4])
            bc = specific[5].upper()
            rho = float(specific[6])
            young = float(specific[7])
            for i in range(number):
                scene.add(
                    FilamentBody(
                        body_id=start_id + i,
                        start=(x + i * sx, y + i * sy),
                        length=length,
                        height=height,
                        angle_deg=angle_deg,
                        flexible=flex_tok == "FLEXIBLE",
                        n_elements=n_elements,
                        bc=bc,
                        rho=rho,
                        youngs_modulus=young,
                    )
                )
        else:
            raise ValueError(f"{where}: unknown body keyword {keyword!r} (CIRCLE|FILAMENT)")

    if not scene.bodies:
        raise ValueError(f"{source}: no bodies found in geometry configuration")
    return scene


def read_geometry_config(path: str | Path) -> Scene:
    """Read ``<case>/input/geometry.config`` (or an explicit file path)."""
    p = Path(path)
    if p.is_dir():
        p = p / DEFAULT_GEOMETRY_PATH
    if not p.is_file():
        raise FileNotFoundError(f"geometry configuration not found: {p}")
    return parse_geometry_config(p.read_text(encoding="utf-8"), source=str(p))


def scene_from_life_case(case_dir: str | Path) -> tuple[Scene, Path]:
    """Convenience alias returning ``(scene, config_path)`` for a LIFE case."""
    p = Path(case_dir) / DEFAULT_GEOMETRY_PATH
    return read_geometry_config(p), p


# ---------------------------------------------------------------------------
# setup object: the compile-time / runtime boundary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SimulationSetup:
    """A compiled solver configuration: params (frozen) + scene (editable)."""

    params: CaseParams
    scene: Scene

    def edit_geometry(self, scene: Scene) -> "SimulationSetup":
        """Runtime geometry change — params fingerprint is unchanged."""
        new = SimulationSetup(params=self.params, scene=scene)
        assert new.param_fingerprint == self.param_fingerprint
        return new

    def add_body(self, body: Body) -> "SimulationSetup":
        scene = Scene(list(self.scene.bodies))
        scene.add(body)
        return self.edit_geometry(scene)

    def remove_body(self, body_id: int) -> "SimulationSetup":
        scene = Scene(list(self.scene.bodies))
        scene.remove(body_id)
        return self.edit_geometry(scene)

    def apply_params(self, **changes: Any) -> "SimulationSetup":
        """Change the compile-time block — refused unless ``rebuild=True``.

        Mirrors LIFE, where every value in ``params.h`` is a header constant:
        changing one requires ``make clean && make``.
        """
        if not changes.pop("rebuild", False):
            raise RuntimeError(
                "CaseParams changes require a rebuild (LIFE: params.h is "
                "compile-time; edit params.h + `make clean && make`) — pass "
                "rebuild=True to acknowledge"
            )
        new = SimulationSetup(params=replace(self.params, **changes), scene=self.scene)
        # A rebuild invalidates the compiled solver: recompute the geometry
        # fingerprint too, so provenance cannot mix the two.
        return new

    @property
    def param_fingerprint(self) -> str:
        return self.params.fingerprint()

    def provenance(self, dx: float | None = None) -> dict[str, Any]:
        dx = self.params.dx if dx is None else dx
        return {
            "param_fingerprint": self.param_fingerprint,
            "params": self.params.as_dict(),
            "geometry": {
                "bodies": [b.body_id for b in self.scene.bodies],
                "keywords": [b.keyword for b in self.scene.bodies],
                "counts": self.scene.counts(dx),
            },
            "mask_rule": MASK_RULE,
        }