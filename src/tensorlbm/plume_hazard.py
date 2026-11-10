"""Buoyant plume self-similarity and flammable gas-cloud (LFL) hazard models.

IN70-N03 — the *far-field* arm of the alternative-fuel leak chain.  The lattice
solver in :mod:`tensorlbm.scalar_transport` resolves the near field (source
region, advection-diffusion + Boussinesq buoyancy); at the resolutions an LBM can
afford the far field of a turbulent plume is *not* resolved, so the plume rise
and the hazard envelope are produced by the standard integral/empirical closures
that the engineering practice uses.  Three pieces, each with its own anchor:

1. :func:`integrate_plume` — Morton-Taylor-Turner **integral plume model**
   (top-hat profiles, entrainment velocity ``u_e = alpha w``)::

       dQ/dz = 2 pi alpha b w          (volume)
       dM/dz = g' A,   g' = B / Q      (momentum, Boussinesq buoyancy)
       Q = A w,  M = A w^2,  A = pi b^2

   Its self-similar solutions are the anchors:

   * pure plume (``B > 0``): ``Q ~ z^{5/3}``, ``b = 6 alpha z / 5``,
     ``w ~ z^{-1/3}``, ``g' ~ z^{-5/3}``;
   * pure momentum jet (``B = 0``, the *negative control*): ``Q ~ z``,
     ``b = 2 alpha z``, ``w ~ z^{-1}``.

   The tests check the fitted exponents (5/3 and 1), the analytic prefactors
   ``Q_plume = (3 pi^2 c_3^4 B / 4)^{1/3} z^{5/3}`` with ``c_3 = 6 alpha/5`` and
   ``Q_jet = 2 alpha sqrt(pi M) z``, and the falsifiable prediction that the
   plume prefactor scales as ``alpha^{4/3}`` while the exponent does not move.

2. :func:`gaussian_plume_concentration` / :func:`lfl_hazard_distance` — the
   **Gaussian plume** with Briggs/Pasquill-Gifford dispersion (the empirical
   model the regulations are written against), ground reflection included, plus
   buoyant plume rise.

3. :func:`lfl_contour_on_grid` — LFL isopleth extraction on a grid with
   bilinear threshold crossings, so the hazard distance is grid convergent
   (acceptance criterion <= 3 %).

Evidence levels: the integral model is a *closure-based* model (REAL for the
model, but it is not resolved turbulence — stated explicitly in the report);
the Gaussian plume is the standard empirical correlation (REAL as a model,
uncertainty factor 2 against field data).

References
----------
Morton, B. R., Taylor, G. I., & Turner, J. S. (1956). Turbulent gravitational
    convection from maintained and instantaneous sources. *Proc. R. Soc. A*
    234, 1-23.
Briggs, G. A. (1973). Diffusion estimation for small emissions. ATDL
    Contribution File No. 79, NOAA.
Turner, D. B. (1994). *Workbook of Atmospheric Dispersion Estimates*, 2nd ed.,
    CRC Press.
NFPA 52 / EN 1160 / IGC Doc 75-14: flammability limits of methane (5.0 vol%),
    hydrogen (4.0 vol%), ammonia (15 vol%).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "IntegralPlumeConfig",
    "integrate_plume",
    "fit_power_law",
    "plume_self_similar",
    "jet_self_similar",
    "GasSpecies",
    "SPECIES",
    "sigma_y_briggs",
    "sigma_z_briggs",
    "plume_rise_briggs",
    "gaussian_plume_concentration",
    "lfl_hazard_distance",
    "lfl_contour_on_grid",
    "hazard_wind_sweep",
]


# ---------------------------------------------------------------------------
# 1. Integral (Morton-Taylor-Turner) plume / jet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IntegralPlumeConfig:
    """Top-hat integral plume configuration (dimensionless or SI, consistent).

    ``volume_flux`` and ``momentum_flux`` are the source-plane values at
    ``z_start``; ``buoyancy_flux`` is ``B = g' Q`` (mass/volume specific
    buoyancy times volume flux), which the model conserves.  Set
    ``buoyancy_flux = 0`` for the momentum-jet negative control.
    """

    entrainment: float = 0.05
    volume_flux: float = 1.0
    momentum_flux: float = 1.0
    buoyancy_flux: float = 0.0
    z_start: float = 1.0
    z_end: float = 1.0e3
    n_steps: int = 4000
    dz: float | None = None


def _rhs(z: float, state: tuple[float, float], cfg: IntegralPlumeConfig) -> tuple[float, float]:
    q, m = state
    area = q * q / m  # A = Q^2/M  (Q = A w, M = A w^2)
    width = math.sqrt(area / math.pi)
    w = m / q
    dq = 2.0 * math.pi * cfg.entrainment * width * w
    dm = (cfg.buoyancy_flux / q) * area if cfg.buoyancy_flux else 0.0
    return dq, dm


def integrate_plume(cfg: IntegralPlumeConfig) -> dict[str, list[float]]:
    """RK4 march of the top-hat integral model from ``z_start`` to ``z_end``.

    Returns lists ``z, Q, M, w, b, gprime, buoyancy_flux`` where
    ``gprime = B/Q`` (the model conserves ``B`` by construction) and ``b`` is the
    top-hat radius.
    """
    n = int(cfg.n_steps)
    z0, z1 = float(cfg.z_start), float(cfg.z_end)
    dz = cfg.dz if cfg.dz is not None else (z1 - z0) / n
    q, m = float(cfg.volume_flux), float(cfg.momentum_flux)
    out: dict[str, list[float]] = {k: [] for k in ("z", "Q", "M", "w", "b", "gprime", "buoyancy_flux")}
    z = z0
    for _ in range(n + 1):
        area = q * q / m
        out["z"].append(z)
        out["Q"].append(q)
        out["M"].append(m)
        out["w"].append(m / q)
        out["b"].append(math.sqrt(area / math.pi))
        out["gprime"].append(cfg.buoyancy_flux / q)
        out["buoyancy_flux"].append((cfg.buoyancy_flux / q) * q)
        k1 = _rhs(z, (q, m), cfg)
        k2 = _rhs(z + 0.5 * dz, (q + 0.5 * dz * k1[0], m + 0.5 * dz * k1[1]), cfg)
        k3 = _rhs(z + 0.5 * dz, (q + 0.5 * dz * k2[0], m + 0.5 * dz * k2[1]), cfg)
        k4 = _rhs(z + dz, (q + dz * k3[0], m + dz * k3[1]), cfg)
        q += dz / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        m += dz / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        z += dz
    return out


def fit_power_law(
    x: list[float], y: list[float], window: tuple[float, float] | None = None
) -> dict[str, float]:
    """Least-squares fit ``y = a x^p`` on ``window`` (log-log), plus ``rms``.

    Returns ``{"exponent": p, "coefficient": a, "rms": r, "n": count}``; ``rms``
    is the residual of ``ln y`` and is the honest measure of how well the data
    follow a pure power law over the window.
    """
    xs, ys = [], []
    for xv, yv in zip(x, y):
        if window is not None and not (window[0] <= xv <= window[1]):
            continue
        if xv <= 0.0 or yv <= 0.0:
            continue
        xs.append(math.log(xv))
        ys.append(math.log(yv))
    n = len(xs)
    if n < 2:
        raise ValueError("not enough points for a power-law fit")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((v - mx) ** 2 for v in xs)
    sxy = sum((v - mx) * (w - my) for v, w in zip(xs, ys))
    p = sxy / sxx
    a = math.exp(my - p * mx)
    rms = math.sqrt(sum((w - (my + p * (v - mx))) ** 2 for v, w in zip(xs, ys)) / n)
    return {"exponent": p, "coefficient": a, "rms": rms, "n": n}


def plume_self_similar(cfg: IntegralPlumeConfig, z: float) -> dict[str, float]:
    """Analytic self-similar (far-field) values of the pure plume at ``z``."""
    a = cfg.entrainment
    b = cfg.buoyancy_flux
    if b <= 0.0:
        raise ValueError("plume_self_similar needs buoyancy_flux > 0")
    c3 = 6.0 * a / 5.0
    c2 = (3.0 * b / (4.0 * math.pi * c3**2)) ** (1.0 / 3.0)
    q = math.pi * c3**2 * c2 * z ** (5.0 / 3.0)
    m = math.pi * c3**2 * c2**2 * z ** (4.0 / 3.0)
    return {
        "Q": q,
        "M": m,
        "w": c2 * z ** (-1.0 / 3.0),
        "b": c3 * z,
        "gprime": b / q,
        "width_slope": c3,
    }


def jet_self_similar(cfg: IntegralPlumeConfig, z: float) -> dict[str, float]:
    """Analytic self-similar values of the pure momentum jet at ``z``."""
    a = cfg.entrainment
    m0 = cfg.momentum_flux
    q = 2.0 * a * math.sqrt(math.pi * m0) * z
    return {
        "Q": q,
        "M": m0,
        "w": math.sqrt(m0 / math.pi) / (2.0 * a * z),
        "b": 2.0 * a * z,
        "gprime": 0.0,
    }


# ---------------------------------------------------------------------------
# 2. Flammable species and the Gaussian plume
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GasSpecies:
    """Flammable species data (mass-based LFL plus a density ratio to air)."""

    name: str
    lfl_volume_fraction: float
    molar_mass: float  # g/mol
    density_ratio_to_air: float  # rho_gas / rho_air at the same T, p

    @property
    def lfl_mass_fraction(self) -> float:
        """LFL as a mass fraction of the mixture (-)."""
        m_air = 28.97
        return self.lfl_volume_fraction * self.molar_mass / (
            self.lfl_volume_fraction * self.molar_mass
            + (1.0 - self.lfl_volume_fraction) * m_air
        )

    def lfl_mass_concentration(self, rho_air: float = 1.2) -> float:
        """LFL as a mass concentration (kg of gas per m^3 of mixture)."""
        return self.lfl_mass_fraction * rho_air


SPECIES: dict[str, GasSpecies] = {
    "CH4": GasSpecies("CH4", 0.050, 16.04, 0.554),
    "H2": GasSpecies("H2", 0.040, 2.016, 0.0696),
    "NH3": GasSpecies("NH3", 0.150, 17.03, 0.596),
}

# Briggs open-country dispersion coefficients (x, sigma in m)
_BRIGGS = {
    "A": (0.22, 0.20),
    "B": (0.16, 0.12),
    "C": (0.11, 0.08),
    "D": (0.08, 0.06),
    "E": (0.06, 0.03),
    "F": (0.04, 0.016),
}


def sigma_y_briggs(x: float, stability: str = "D") -> float:
    """Briggs open-country lateral spread ``sigma_y`` (m) at downwind distance x (m)."""
    a_y, _ = _BRIGGS[stability.upper()]
    return a_y * x * (1.0 + 1e-4 * x) ** -0.5


def sigma_z_briggs(x: float, stability: str = "D") -> float:
    """Briggs open-country vertical spread ``sigma_z`` (m) at x (m)."""
    _, a_z = _BRIGGS[stability.upper()]
    s = stability.upper()
    if s in ("A", "B"):
        return a_z * x
    if s == "C":
        return a_z * x * (1.0 + 2e-4 * x) ** -0.5
    if s == "D":
        return a_z * x * (1.0 + 1.5e-3 * x) ** -0.5
    if s == "E":
        return a_z * x * (1.0 + 3e-4 * x) ** -1.0
    return a_z * x * (1.0 + 3e-4 * x) ** -1.0


def plume_rise_briggs(
    buoyancy_flux: float, wind: float, x: float, stack_height: float = 0.0
) -> float:
    """Briggs buoyant plume rise ``dh = 1.6 F^{1/3} x^{2/3} / U`` (neutral/unstable).

    ``buoyancy_flux F = g dT/T V (m^4/s^3)`` — it is the same conserved quantity
    ``B`` that the integral plume model carries, which is the bridge between the
    two models.
    """
    if buoyancy_flux <= 0.0 or wind <= 0.0:
        return 0.0
    return 1.6 * buoyancy_flux ** (1.0 / 3.0) * x ** (2.0 / 3.0) / wind


def gaussian_plume_concentration(
    x: float,
    y: float,
    z: float,
    source_rate: float,
    wind: float,
    effective_height: float,
    stability: str = "D",
) -> float:
    """Ground-reflecting Gaussian plume concentration (kg/m^3) at a point.

    ``C = Q/(2 pi U sy sz) exp(-y^2/2sy^2) [exp(-(z-H)^2/2sz^2) + exp(-(z+H)^2/2sz^2)]``
    """
    if x <= 0.0:
        return 0.0
    sy = sigma_y_briggs(x, stability)
    sz = sigma_z_briggs(x, stability)
    if sy <= 0.0 or sz <= 0.0:
        return 0.0
    h = float(effective_height)
    vert = math.exp(-((z - h) ** 2) / (2.0 * sz * sz)) + math.exp(
        -((z + h) ** 2) / (2.0 * sz * sz)
    )
    return (
        source_rate
        / (2.0 * math.pi * wind * sy * sz)
        * math.exp(-(y * y) / (2.0 * sy * sy))
        * vert
    )


def _super_threshold_range(
    gas: GasSpecies,
    source_rate: float,
    wind: float,
    stability: str,
    stack_height: float,
    x_max: float,
    n_scan: int = 600,
) -> tuple[float, float, float] | None:
    """Bracket the LFL interval on the ground-level centreline by a log scan."""
    thr = gas.lfl_mass_concentration()
    x_min = 1.0e-2
    xs = [x_min * (x_max / x_min) ** (i / (n_scan - 1)) for i in range(n_scan)]
    cs = [
        gaussian_plume_concentration(x, 0.0, 0.0, source_rate, wind, stack_height, stability)
        for x in xs
    ]
    hot = [i for i, c in enumerate(cs) if c >= thr]
    if not hot:
        return None
    i_last = hot[-1]
    lo = xs[i_last]
    hi = xs[i_last + 1] if i_last + 1 < n_scan else x_max
    return lo, hi, thr


def lfl_hazard_distance(
    species: GasSpecies,
    source_rate: float,
    wind: float,
    stability: str = "D",
    stack_height: float = 1.0,
    ground_level: bool = True,
    x_max: float = 5.0e4,
) -> dict[str, float]:
    """Downwind extent of the LFL ground-level isopleth (m).

    The super-threshold interval on the ground-level centreline (``y = z = 0``)
    is bracketed by a logarithmic scan and its outer edge refined by bisection.
    Returns ``{"distance", "threshold", "max_halfwidth", "x_at_max_halfwidth"}``;
    ``distance = 0`` means no flammable cloud reaches the ground.
    """
    brk = _super_threshold_range(
        species, source_rate, wind, stability, stack_height, x_max
    )
    if brk is None:
        return {
            "distance": 0.0,
            "threshold": species.lfl_mass_concentration(),
            "max_halfwidth": 0.0,
            "x_at_max_halfwidth": 0.0,
        }
    lo, hi, thr = brk

    def c_axis(x: float) -> float:
        return gaussian_plume_concentration(
            x, 0.0, 0.0, source_rate, wind, stack_height, stability
        )

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if c_axis(mid) >= thr:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6 * max(1.0, lo):
            break
    best_w = 0.0
    best_x = 0.0
    n = 400
    x_start = 1.0e-2
    for i in range(1, n):
        xv = x_start * (lo / x_start) ** (i / (n - 1))
        c0 = c_axis(xv)
        if c0 <= thr:
            continue
        sy = sigma_y_briggs(xv, stability)
        w = sy * math.sqrt(2.0 * math.log(c0 / thr))
        if w > best_w:
            best_w, best_x = w, xv
    return {
        "distance": lo,
        "threshold": thr,
        "max_halfwidth": best_w,
        "x_at_max_halfwidth": best_x,
    }


def lfl_contour_on_grid(
    species: GasSpecies,
    source_rate: float,
    wind: float,
    stability: str = "D",
    stack_height: float = 1.0,
    dx: float = 2.0,
    y_half: float | None = None,
    x_max: float | None = None,
) -> dict[str, float]:
    """LFL isopleth metrics from the analytic field sampled on a grid.

    The field is sampled with spacing ``dx`` and the threshold crossing on the
    centreline row is located by linear interpolation, so the returned
    ``distance`` is grid convergent (this is what the <= 3 % grid-refinement
    criterion is checked on).
    """
    thr = species.lfl_mass_concentration()
    ref = lfl_hazard_distance(
        species, source_rate, wind, stability=stability, stack_height=stack_height
    )
    x_hi = x_max if x_max is not None else max(1.5 * ref["distance"], 16.0 * dx)
    y_half = y_half if y_half is not None else max(3.0 * ref["max_halfwidth"], 10.0 * dx)
    nx = max(8, int(round(x_hi / dx)))
    ny = max(8, int(round(2.0 * y_half / dx)))
    dist = 0.0
    width = 0.0
    for j in range(nx):
        xv = (j + 0.5) * dx
        c0 = gaussian_plume_concentration(
            xv, 0.0, 0.0, source_rate, wind, stack_height, stability
        )
        if c0 >= thr:
            if j + 1 < nx:
                c1 = gaussian_plume_concentration(
                    (j + 1.5) * dx, 0.0, 0.0, source_rate, wind, stack_height, stability
                )
                frac = (c0 - thr) / (c0 - c1) if c0 != c1 else 0.0
                dist = (j + 0.5 + frac) * dx
            else:
                dist = (j + 0.5) * dx
        for i in range(ny):
            yv = -y_half + (i + 0.5) * dx
            if gaussian_plume_concentration(
                xv, yv, 0.0, source_rate, wind, stack_height, stability
            ) >= thr:
                width = max(width, 2.0 * abs(yv))
    return {
        "distance": dist,
        "max_width": width,
        "dx": dx,
        "nx": nx,
        "ny": ny,
        "threshold": thr,
    }


@dataclass
class HazardSweep:
    """Wind-speed sweep of the LFL hazard envelope."""

    species: str
    source_rate: float
    winds: list[float]
    stabilities: tuple[str, ...] = ("D", "F")
    stack_height: float = 1.0
    rows: list[dict[str, float]] = field(default_factory=list)

    def run(self, gas: GasSpecies | None = None) -> list[dict[str, float]]:
        gas = gas or SPECIES[self.species]
        self.rows = []
        for u in self.winds:
            entry: dict[str, float] = {"wind": float(u)}
            for stab in self.stabilities:
                res = lfl_hazard_distance(
                    gas,
                    self.source_rate,
                    float(u),
                    stability=stab,
                    stack_height=self.stack_height,
                )
                entry[f"distance_{stab}"] = res["distance"]
                entry[f"halfwidth_{stab}"] = res["max_halfwidth"]
            entry["distance"] = entry[f"distance_{self.stabilities[0]}"]
            self.rows.append(entry)
        return self.rows


def hazard_wind_sweep(
    species: str,
    source_rate: float,
    winds: list[float],
    stabilities: tuple[str, ...] = ("D", "F"),
    stack_height: float = 1.0,
) -> dict[str, object]:
    """LFL hazard distance vs wind speed for one species (monotone check helper)."""
    gas = SPECIES[species]
    rows = HazardSweep(
        species=species,
        source_rate=source_rate,
        winds=list(winds),
        stabilities=stabilities,
        stack_height=stack_height,
    ).run(gas)
    primary = [r[f"distance_{stabilities[0]}"] for r in rows]
    monotone = all(
        primary[i] >= primary[i + 1] - 1e-9 for i in range(len(primary) - 1)
    )
    return {
        "species": species,
        "rows": rows,
        "primary_stability": stabilities[0],
        "monotone_decreasing": monotone,
        "distances": primary,
    }