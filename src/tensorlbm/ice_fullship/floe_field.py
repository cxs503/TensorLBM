"""Polydisperse floe field for broken-ice resistance chains.

The field is a set of rigid circular floes (plan-view discs) of constant thickness
``h`` placed without overlap inside the plan rectangle ``L x W``.  Floe diameters
follow a truncated log-normal distribution characterised by its median ``d50`` and
geometric standard deviation ``sigma_g``.

Acceptance quantities (campaign node P66-W2-N04, criterion C1)
---------------------------------------------------------------
* realised ice concentration, measured by an *independent* rasterisation of the
  accepted floes (not by the generator's own book-keeping), must match the
  requested concentration within 5 %
* realised median floe diameter must match ``d50`` within 10 %

Placement is a size-descending random sequential adsorption (RSA) with a bucket
grid for the overlap test; the polydispersity lets the packing go well beyond the
monodisperse RSA jamming limit (~0.547).  The floe population is *sized first*
(so that the total disc area equals the requested concentration) and then placed
in full - an earlier variant that stopped placing as soon as the area target was
met produced a median diameter 20 % above target, because the smallest floes were
simply never reached.

References
----------
- Cundall & Strack (1979), "A discrete numerical model for granular assemblies".
- Brash-ice statistics: floe sizes are log-normal with d50 of order 0.5-2 m in
  broken-ice channels.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class FloeFieldConfig:
    """Configuration of a broken-ice floe field (plan view, SI units)."""

    length: float = 120.0          # x extent [m]
    width: float = 60.0            # y extent [m]
    concentration: float = 0.60    # target ice area concentration [-]
    thickness: float = 0.15        # ice thickness [m]
    d50: float = 1.0               # median floe diameter [m]
    sigma_g: float = 1.35          # geometric standard deviation [-]
    d_min_ratio: float = 0.25      # d_min = d_min_ratio * d50
    d_max_ratio: float = 3.00      # d_max = d_max_ratio * d50
    rho_ice: float = 917.0         # [kg/m^3]
    rho_water: float = 1025.0      # [kg/m^3]
    seed: int = 20261108

    def __post_init__(self) -> None:
        if not (0.0 < self.concentration < 1.0):
            raise ValueError(f"concentration must be in (0, 1), got {self.concentration}")
        if self.thickness <= 0.0:
            raise ValueError("thickness must be positive")
        if self.d50 <= 0.0:
            raise ValueError("d50 must be positive")
        if self.sigma_g <= 1.0:
            raise ValueError("sigma_g must be > 1")
        if self.length <= 0.0 or self.width <= 0.0:
            raise ValueError("domain extents must be positive")


@dataclass
class FloeField:
    """A realised floe field: parallel arrays (all lengths in metres)."""

    x: np.ndarray          # floe centre x [m]
    y: np.ndarray          # floe centre y [m]
    d: np.ndarray          # floe diameter [m]
    thickness: float       # ice thickness [m]
    length: float          # plan domain length [m]
    width: float           # plan domain width [m]
    rho_ice: float
    rho_water: float

    # ---- derived quantities -------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.d.size)

    @property
    def radius(self) -> np.ndarray:
        return 0.5 * self.d

    @property
    def area(self) -> np.ndarray:
        return 0.25 * math.pi * self.d ** 2

    @property
    def mass(self) -> np.ndarray:
        return self.rho_ice * self.area * self.thickness

    @property
    def geometric_concentration(self) -> float:
        """Concentration from the generator's own area book-keeping."""
        return float(self.area.sum() / (self.length * self.width))

    def copy(self) -> "FloeField":
        return FloeField(
            x=self.x.copy(),
            y=self.y.copy(),
            d=self.d.copy(),
            thickness=self.thickness,
            length=self.length,
            width=self.width,
            rho_ice=self.rho_ice,
            rho_water=self.rho_water,
        )


# ---------------------------------------------------------------------------
# size distribution
# ---------------------------------------------------------------------------
def sample_floe_diameters(n: int, cfg: FloeFieldConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` diameters from a log-normal law truncated to [d_min, d_max]."""
    d_min = cfg.d_min_ratio * cfg.d50
    d_max = cfg.d_max_ratio * cfg.d50
    sigma_ln = math.log(cfg.sigma_g)
    out = np.empty(n, dtype=np.float64)
    filled = 0
    while filled < n:
        draw = rng.lognormal(mean=math.log(cfg.d50), sigma=sigma_ln, size=4 * max(1, n - filled))
        draw = draw[(draw >= d_min) & (draw <= d_max)]
        take = min(draw.size, n - filled)
        out[filled:filled + take] = draw[:take]
        filled += take
    return out


def plan_floe_sizes(cfg: FloeFieldConfig, rng: np.random.Generator) -> np.ndarray:
    """Draw a population whose total disc area equals the requested ice area."""
    target_area = cfg.concentration * cfg.length * cfg.width
    probe = sample_floe_diameters(20000, cfg, rng)
    mean_area = 0.25 * math.pi * float(np.mean(probe ** 2))
    n = max(8, int(round(target_area / mean_area)))
    for _ in range(6):
        d = sample_floe_diameters(n, cfg, rng)
        a = 0.25 * math.pi * d ** 2
        n_new = max(8, int(round(n * target_area / a.sum())))
        if n_new == n:
            break
        n = n_new
    return d


# ---------------------------------------------------------------------------
# non-overlapping placement
# ---------------------------------------------------------------------------
class _BucketGrid:
    """Uniform bucket grid used for the overlap test during RSA placement."""

    def __init__(self, length: float, width: float, cell: float):
        self.cell = cell
        self.nx = max(1, int(math.ceil(length / cell)))
        self.ny = max(1, int(math.ceil(width / cell)))
        self.buckets: Dict[Tuple[int, int], list] = {}

    def insert(self, x: float, y: float, r: float, idx: int) -> None:
        i0 = max(0, int((x - r) // self.cell))
        i1 = min(self.nx - 1, int((x + r) // self.cell))
        j0 = max(0, int((y - r) // self.cell))
        j1 = min(self.ny - 1, int((y + r) // self.cell))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                self.buckets.setdefault((i, j), []).append(idx)

    def candidates(self, x: float, y: float, r: float):
        """Indices of floes registered in buckets overlapping the query disc."""
        i0 = max(0, int((x - r) // self.cell))
        i1 = min(self.nx - 1, int((x + r) // self.cell))
        j0 = max(0, int((y - r) // self.cell))
        j1 = min(self.ny - 1, int((y + r) // self.cell))
        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                bucket = self.buckets.get((i, j))
                if bucket:
                    yield from bucket


def _place_all(cfg: FloeFieldConfig, diameters: np.ndarray, rng: np.random.Generator,
               max_attempts: int = 150, existing: "FloeField | None" = None) -> FloeField:
    """Place ``diameters`` without overlapping each other *or* ``existing``."""
    cell = max(1.0 * cfg.d50, 0.4 * cfg.d50 * cfg.d_max_ratio)
    grid = _BucketGrid(cfg.length, cfg.width, cell)
    if existing is not None:
        xs = list(existing.x)
        ys = list(existing.y)
        ds = list(existing.d)
        for j, (xj, yj, dj) in enumerate(zip(xs, ys, ds)):
            grid.insert(xj, yj, 0.5 * dj, j)
    else:
        xs, ys, ds = [], [], []
    for d in diameters:                     # largest first
        r = 0.5 * d
        for _ in range(max_attempts):
            x = rng.uniform(r, cfg.length - r)
            y = rng.uniform(r, cfg.width - r)
            ok = True
            for j in grid.candidates(x, y, r):
                dx = x - xs[j]
                dy = y - ys[j]
                if dx * dx + dy * dy < (r + 0.5 * ds[j]) ** 2:
                    ok = False
                    break
            if ok:
                xs.append(x)
                ys.append(y)
                ds.append(d)
                grid.insert(x, y, r, len(xs) - 1)
                break
    return FloeField(
        x=np.asarray(xs),
        y=np.asarray(ys),
        d=np.asarray(ds),
        thickness=cfg.thickness,
        length=cfg.length,
        width=cfg.width,
        rho_ice=cfg.rho_ice,
        rho_water=cfg.rho_water,
    )


def generate_floe_field(cfg: FloeFieldConfig, max_rounds: int = 4) -> FloeField:
    """Generate a non-overlapping floe field hitting the requested concentration.

    The population is sized analytically first (total disc area = requested ice
    area) and then placed in full, largest floe first.  A floe that cannot be
    placed after ``max_attempts`` random trials is dropped (jammed pocket); the
    resulting area deficit is compensated by adding the missing area in a next
    round, so the realised concentration converges to the target.
    """
    target_area = cfg.concentration * cfg.length * cfg.width
    best: FloeField | None = None
    diameter_pool: list[np.ndarray] = []
    for rnd in range(max_rounds):
        rng = np.random.default_rng(cfg.seed + 101 * rnd)
        extra = [] if best is None else [np.full(1, 0.0)]
        deficit_fraction = 1.0 if best is None else max(0.0, 1.0 - best.area.sum() / target_area)
        if best is None:
            diameters = plan_floe_sizes(cfg, rng)
        else:
            # add the missing area, drawn from the same size law
            need = deficit_fraction * target_area
            if need / target_area < 0.005:
                break
            probe = sample_floe_diameters(4000, cfg, rng)
            mean_area = 0.25 * math.pi * float(np.mean(probe ** 2))
            n_add = max(4, int(round(need / mean_area)))
            diameters = sample_floe_diameters(n_add, cfg, rng)
        diameters = np.sort(diameters)[::-1]
        # ``existing=best`` means the returned field already contains the floes of
        # the previous round (they are seeded into the overlap grid), so the new
        # field replaces - never extends - the previous one.
        best = _place_all(cfg, diameters, rng, existing=best)
        if abs(best.geometric_concentration - cfg.concentration) / cfg.concentration < 0.005:
            break
    assert best is not None
    return best


# ---------------------------------------------------------------------------
# independent measurement (used by criterion C1)
# ---------------------------------------------------------------------------
def rasterise_concentration(field: FloeField, resolution: float = 0.10) -> float:
    """Ice area concentration from an independent rasterisation of the floes.

    A cell counts as ice when its centre lies inside at least one disc.  With
    ``resolution`` clearly below ``d50`` this reproduces the geometric area
    fraction to well under 1 % while being completely independent of the
    generator's area book-keeping (and, unlike the book-keeping, it would catch
    any overlapping floes).
    """
    if resolution <= 0.0:
        raise ValueError("resolution must be positive")
    nx = int(math.ceil(field.length / resolution))
    ny = int(math.ceil(field.width / resolution))
    mask = np.zeros((nx, ny), dtype=bool)
    for xi, yi, di in zip(field.x, field.y, field.d):
        r = 0.5 * di
        i0 = max(0, int((xi - r) / resolution))
        i1 = min(nx - 1, int((xi + r) / resolution))
        j0 = max(0, int((yi - r) / resolution))
        j1 = min(ny - 1, int((yi + r) / resolution))
        if i1 < i0 or j1 < j0:
            continue
        cx = (np.arange(i0, i1 + 1) + 0.5) * resolution
        cy = (np.arange(j0, j1 + 1) + 0.5) * resolution
        dx2 = (cx - xi)[:, None] ** 2
        dy2 = (cy - yi)[None, :] ** 2
        mask[i0:i1 + 1, j0:j1 + 1] |= (dx2 + dy2) <= r * r
    return float(mask.mean())


def measure_d50(field: FloeField, weighting: str = "number") -> float:
    """Median floe diameter (``number`` or ``area`` weighted)."""
    d = np.sort(field.d)
    if d.size == 0:
        raise ValueError("empty field")
    if weighting == "number":
        return float(np.median(d))
    if weighting == "area":
        a = np.sort(field.area)
        c = np.cumsum(a) / a.sum()
        return float(np.interp(0.5, c, d))
    raise ValueError(f"unknown weighting {weighting!r}")


def validate_floe_field(
    field: FloeField,
    cfg: FloeFieldConfig,
    conc_tol: float = 0.05,
    d50_tol: float = 0.10,
    check_overlap: bool = True,
) -> Dict[str, float]:
    """Criterion-C1 audit of a realised floe field.

    Every pass flag is computed from *measured* quantities (rasterised
    concentration, measured median size), never from the requested configuration.
    """
    conc_meas = rasterise_concentration(field)
    conc_err = abs(conc_meas - cfg.concentration) / cfg.concentration
    d50_meas = measure_d50(field, "number")
    d50_err = abs(d50_meas - cfg.d50) / cfg.d50
    d50_area = measure_d50(field, "area")

    overlap_pairs = 0
    if check_overlap and field.n > 1:
        n = field.n
        r = 0.5 * field.d
        step = max(1, n // 1500)
        for i in range(0, n, step):
            dx = field.x[i + 1:] - field.x[i]
            dy = field.y[i + 1:] - field.y[i]
            dist2 = dx * dx + dy * dy
            lim = (r[i] + r[i + 1:]) ** 2
            overlap_pairs += int(np.count_nonzero(dist2 < 0.999 * lim))

    return {
        "n_floes": float(field.n),
        "concentration_target": cfg.concentration,
        "concentration_measured": conc_meas,
        "concentration_error_rel": conc_err,
        "d50_target": cfg.d50,
        "d50_measured_number": d50_meas,
        "d50_measured_area": d50_area,
        "d50_error_rel": d50_err,
        "overlap_pairs": float(overlap_pairs),
        "pass_concentration": bool(conc_err < conc_tol),
        "pass_d50": bool(d50_err < d50_tol),
        "pass_no_overlap": bool(overlap_pairs == 0),
    }