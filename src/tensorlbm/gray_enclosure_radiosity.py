"""Gray-diffuse enclosure radiosity solver with energy-conservation auditing.

Physics
-------
For an enclosure of ``N`` opaque, gray, diffuse surfaces with areas
:math:`A_i`, emissivities :math:`\\varepsilon_i`, view factors
:math:`F_{ij}` and radiosities :math:`J_i` (W/m2):

.. math::
    J_i = \\varepsilon_i \\sigma T_i^4 + (1-\\varepsilon_i) G_i, \\qquad
    G_i = \\sum_j F_{ij} J_j , \\qquad
    q_i = J_i - G_i = \\varepsilon_i (\\sigma T_i^4 - G_i)

:math:`q_i` is the net radiative flux *leaving* surface ``i``.  The linear
system ``(I - R F) J = e`` with ``R = diag(1 - eps)`` and
``e_i = eps_i sigma T_i^4`` is solved directly (LU), then audited:

* **energy conservation** of the enclosure:
  :math:`|\\sum_i A_i q_i| / \\sum_i A_i |q_i|` (exact zero for a closed
  enclosure whose view factors satisfy the summation rule),
* **algebraic closure** :math:`\\|(I - R F) J - e\\|_\\infty / \\|e\\|_\\infty`,
* **surface closure** :math:`\\max_i |q_i - \\varepsilon_i(\\sigma T_i^4 - G_i)|`
  normalised by the largest emitted flux.

Reradiating (adiabatic) surfaces are the degenerate case
:math:`\\varepsilon_i = 0` with unspecified temperature: their row becomes
:math:`J_i - \\sum_j F_{ij} J_j = 0`, i.e. ``q_i = 0``, which is exactly the
reradiating boundary condition.  No special-casing is needed and no
temperature is invented for them.

Analytic anchors (independent of the solver)
-------------------------------------------
* :func:`analytic_two_surface_flux` — series-resistance solution for a
  two-surface enclosure
  :math:`q_1 = \\sigma(T_1^4-T_2^4)/[1/\\varepsilon_1 + (A_1/A_2)(1/\\varepsilon_2 - 1)]`
  (equal to the parallel-plate result for :math:`A_1 = A_2`, and to the
  small-body-in-large-enclosure result for :math:`A_1/A_2 \\to 0`).
* :func:`analytic_reradiating_network_flux` — three-surface enclosure with one
  reradiating surface, solved through the standard radiation-network
  series/parallel resistance
  :math:`1/(A_1F_{12} + 1/(1/(A_1F_{1r}) + 1/(A_2F_{2r})))`.
* Concentric-sphere / concentric-cylinder view factors come from
  :mod:`tensorlbm.radiative_view_factors` (closed form).

References
----------
Modest, M. F. (2013). *Radiative Heat Transfer*, 3rd ed., ch. 5 and ch. 10.
Incropera, F. P. et al. (2007). *Fundamentals of Heat and Mass Transfer*, 6th
    ed., ch. 13 (radiation exchange between surfaces).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .radiative_view_factors import (
    vf_concentric_spheres,
    vf_parallel_coaxial_disks,
)

__all__ = [
    "STEFAN_BOLTZMANN",
    "GraySurface",
    "GrayEnclosure",
    "EnclosureSolution",
    "analytic_two_surface_flux",
    "analytic_reradiating_network_flux",
    "concentric_spheres_enclosure",
    "cube_enclosure",
]

STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m2 K4), CODATA / SI 2019


@dataclass
class GraySurface:
    """Opaque, gray, diffuse surface of an enclosure.

    Parameters
    ----------
    area : float
        Surface area [m2] (> 0).
    emissivity : float
        Hemispherical total emissivity in [0, 1].
    temperature : float or None
        Surface temperature [K].  ``None`` marks a *reradiating* (adiabatic)
        surface: ``emissivity`` is ignored and forced to 0, ``q = 0``.
    external_flux : float
        Absorbed external irradiation per unit area [W/m2] (e.g. solar
        ``alpha_s * G``).  Bookkeeping only: it never enters the radiative
        exchange balance inside the enclosure, it is what the surface must
        reject by conduction/convection in steady state.
    name : str
        Label used in reports.
    """

    area: float
    emissivity: float
    temperature: float | None = None
    external_flux: float = 0.0
    name: str = ""

    def __post_init__(self) -> None:
        if not math.isfinite(self.area) or self.area <= 0.0:
            raise ValueError(f"{self.name}: area must be finite and > 0, got {self.area!r}")
        if self.temperature is None:
            self.emissivity = 0.0
        else:
            if not math.isfinite(self.temperature) or self.temperature <= 0.0:
                raise ValueError(
                    f"{self.name}: temperature must be finite and > 0, got {self.temperature!r}"
                )
            if not (0.0 <= self.emissivity <= 1.0):
                raise ValueError(
                    f"{self.name}: emissivity must be in [0, 1], got {self.emissivity!r}"
                )
        if not math.isfinite(self.external_flux):
            raise ValueError(f"{self.name}: external_flux must be finite")


@dataclass
class EnclosureSolution:
    """Result container with the audit quantities filled in."""

    radiosity: np.ndarray  # J_i [W/m2]
    irradiation: np.ndarray  # G_i [W/m2]
    net_flux: np.ndarray  # q_i = J_i - G_i [W/m2], positive = leaving
    emitted: np.ndarray  # eps_i sigma T_i^4 [W/m2]
    total_power_leaving: np.ndarray  # A_i q_i [W]
    energy_residual: float  # |sum A q| / sum A|q|
    linear_residual: float  # ||(I-RF)J - e||inf / ||e||inf
    closure_residual: float  # max|q - eps(sigmaT^4 - G)| / max emitted
    summation_residual: float  # max_i |sum_j F_ij - 1|
    reciprocity_residual: float  # max_ij |A_i F_ij - A_j F_ji| / max(A)
    adiabatic_flux: float  # max |q_i| over reradiating surfaces [W/m2]

    def as_dict(self) -> dict:
        return {
            "radiosity_W_m2": self.radiosity.tolist(),
            "irradiation_W_m2": self.irradiation.tolist(),
            "net_flux_W_m2": self.net_flux.tolist(),
            "emitted_W_m2": self.emitted.tolist(),
            "total_power_leaving_W": self.total_power_leaving.tolist(),
            "energy_residual": self.energy_residual,
            "linear_residual": self.linear_residual,
            "closure_residual": self.closure_residual,
            "summation_residual": self.summation_residual,
            "reciprocity_residual": self.reciprocity_residual,
            "adiabatic_flux_W_m2": self.adiabatic_flux,
        }


@dataclass
class GrayEnclosure:
    """Gray-diffuse enclosure: surfaces + view factors.

    ``view_factors[i, j] = F_{ij}`` must satisfy the summation rule for a
    closed enclosure and the reciprocity relation.  Both are *audited*, not
    silently repaired: a violating matrix raises unless ``tol`` is relaxed.
    """

    surfaces: Sequence[GraySurface]
    view_factors: np.ndarray
    summation_tol: float = 1e-4
    reciprocity_tol: float = 1e-6
    _areas: np.ndarray = field(init=False, repr=False)
    _eps: np.ndarray = field(init=False, repr=False)
    _T: np.ndarray = field(init=False, repr=False)
    _adiabatic: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.surfaces = list(self.surfaces)
        n = len(self.surfaces)
        if n < 2:
            raise ValueError("an enclosure needs at least 2 surfaces")
        F = np.asarray(self.view_factors, dtype=np.float64)
        if F.shape != (n, n):
            raise ValueError(f"view_factors shape {F.shape} does not match ({n}, {n})")
        if not np.all(np.isfinite(F)):
            raise ValueError("view_factors contain non-finite entries")
        if np.any(F < -1e-12):
            raise ValueError("view_factors contain negative entries")
        # keep the diagonal: concave surfaces legitimately see themselves
        # (F_ii > 0), and dropping it would break the summation rule.
        self.view_factors = np.clip(F, 0.0, None)

        self._areas = np.array([s.area for s in self.surfaces], dtype=np.float64)
        self._eps = np.array([s.emissivity for s in self.surfaces], dtype=np.float64)
        self._T = np.array(
            [s.temperature if s.temperature is not None else np.nan for s in self.surfaces],
            dtype=np.float64,
        )
        self._adiabatic = np.array([s.temperature is None for s in self.surfaces])

        self.check_view_factors()

    # -- audits ------------------------------------------------------------
    def summation_residual(self) -> float:
        return float(np.max(np.abs(self.view_factors.sum(axis=1) - 1.0)))

    def reciprocity_residual(self) -> float:
        A = self._areas
        lhs = A[:, None] * self.view_factors
        rhs = (A[:, None] * self.view_factors).T
        return float(np.max(np.abs(lhs - rhs)) / np.max(A))

    def check_view_factors(self) -> None:
        s = self.summation_residual()
        r = self.reciprocity_residual()
        if s > self.summation_tol:
            raise ValueError(
                f"view factors violate the summation rule: max|sum_j F_ij - 1| = {s:.3e} "
                f"> summation_tol = {self.summation_tol:.1e}"
            )
        if r > self.reciprocity_tol:
            raise ValueError(
                f"view factors violate reciprocity: max|A_i F_ij - A_j F_ji|/A_max = {r:.3e} "
                f"> reciprocity_tol = {self.reciprocity_tol:.1e}"
            )

    # -- solve -------------------------------------------------------------
    def solve(self) -> EnclosureSolution:
        """Solve the radiosity system and audit energy conservation."""
        F = self.view_factors
        eps = self._eps
        sigma_t4 = np.where(np.isnan(self._T), 0.0, STEFAN_BOLTZMANN * np.nan_to_num(self._T) ** 4)
        e = eps * sigma_t4

        m = np.eye(F.shape[0]) - (1.0 - eps)[:, None] * F
        J = np.linalg.solve(m, e)
        G = F @ J
        q = J - G

        emitted = e
        A = self._areas
        power = A * q
        denom = float(np.sum(A * np.abs(q)))
        energy_residual = (
            abs(float(np.sum(power))) / denom if denom > 1e-300 else abs(float(np.sum(power)))
        )
        lin = float(np.linalg.norm(m @ J - e, ord=np.inf))
        lin = lin / max(float(np.linalg.norm(e, ord=np.inf)), 1e-300)
        emitted_max = max(float(np.max(np.abs(emitted))), 1e-300)
        closure = float(
            np.max(np.abs(q - eps * (np.where(np.isnan(self._T), 0.0, sigma_t4) - G)))
        ) / emitted_max
        adiabatic_flux = (
            float(np.max(np.abs(q[self._adiabatic]))) if np.any(self._adiabatic) else 0.0
        )

        return EnclosureSolution(
            radiosity=J,
            irradiation=G,
            net_flux=q,
            emitted=emitted,
            total_power_leaving=power,
            energy_residual=energy_residual,
            linear_residual=lin,
            closure_residual=closure,
            summation_residual=self.summation_residual(),
            reciprocity_residual=self.reciprocity_residual(),
            adiabatic_flux=adiabatic_flux,
        )


# ---------------------------------------------------------------------------
# Analytic anchors
# ---------------------------------------------------------------------------


def analytic_two_surface_flux(
    T1: float, T2: float, eps1: float, eps2: float, A1: float, A2: float
) -> float:
    r"""Series-resistance solution for a two-surface enclosure.

    .. math::
        q_1 = \frac{\sigma (T_1^4 - T_2^4)}
        {\frac{1}{\varepsilon_1} + \frac{A_1}{A_2}\left(\frac{1}{\varepsilon_2} - 1\right)}

    per unit area of surface 1 [W/m2], positive when surface 1 is hotter.
    Special cases: :math:`A_1 = A_2` gives the parallel-plate result
    :math:`\sigma(T_1^4-T_2^4)/(1/\varepsilon_1+1/\varepsilon_2-1)`;
    :math:`A_1/A_2 \to 0` gives :math:`\varepsilon_1\sigma(T_1^4-T_2^4)`.
    """
    for label, value in (("T1", T1), ("T2", T2)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    for label, value in (("eps1", eps1), ("eps2", eps2)):
        if not (0.0 < value <= 1.0):
            raise ValueError(f"{label} must be in (0, 1], got {value!r}")
    for label, value in (("A1", A1), ("A2", A2)):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be finite and > 0, got {value!r}")
    denom = 1.0 / eps1 + (A1 / A2) * (1.0 / eps2 - 1.0)
    return float(STEFAN_BOLTZMANN * (T1**4 - T2**4) / denom)


def analytic_reradiating_network_flux(
    T1: float,
    T2: float,
    eps1: float,
    eps2: float,
    A1: float,
    A2: float,
    F12: float,
    F1r: float,
    F2r: float,
) -> float:
    r"""Three-surface enclosure, surface ``r`` reradiating (adiabatic).

    Standard radiation-network solution, per unit area of surface 1 [W/m2]:

    .. math::
        q_1 = \frac{\sigma(T_1^4-T_2^4)}{R}, \qquad
        R = \frac{1-\varepsilon_1}{\varepsilon_1 A_1}
          + \frac{1-\varepsilon_2}{\varepsilon_2 A_2}
          + \frac{1}{A_1F_{12} + \frac{1}{\frac{1}{A_1F_{1r}} + \frac{1}{A_2F_{2r}}}}

    multiplied by ``A1`` from the per-area convention
    (:math:`q_1 [W/m^2] = Q/A_1`).
    """
    if not (0.0 < eps1 <= 1.0) or not (0.0 < eps2 <= 1.0):
        raise ValueError("eps1, eps2 must be in (0, 1]")
    if F1r <= 0.0 or F2r <= 0.0 or F12 < 0.0:
        raise ValueError("view factors must be non-negative with F1r, F2r > 0")
    r_surf1 = (1.0 - eps1) / (eps1 * A1)
    r_surf2 = (1.0 - eps2) / (eps2 * A2)
    r_space = 1.0 / (
        A1 * F12 + 1.0 / (1.0 / (A1 * F1r) + 1.0 / (A2 * F2r))
    )
    R_total = r_surf1 + r_surf2 + r_space
    Q = STEFAN_BOLTZMANN * (T1**4 - T2**4) / R_total  # W
    return float(Q / A1)


# ---------------------------------------------------------------------------
# Enclosure builders (closed-form view factors)
# ---------------------------------------------------------------------------


def concentric_spheres_enclosure(
    r1: float, r2: float, eps1: float, eps2: float, T1: float, T2: float
) -> GrayEnclosure:
    """Two-surface concentric-sphere enclosure with closed-form view factors.

    Inner sphere (1) is convex: ``F_11 = 0``, ``F_12 = 1``.  Outer sphere (2):
    ``F_21 = (r1/r2)^2`` (closed form), ``F_22 = 1 - F_21``.
    """
    from .radiative_view_factors import vf_concentric_spheres_inner_to_outer

    A1, A2 = 4.0 * math.pi * r1 * r1, 4.0 * math.pi * r2 * r2
    F12 = vf_concentric_spheres_inner_to_outer(r1, r2)
    F21 = vf_concentric_spheres(r1, r2)
    F = np.array([[0.0, F12], [A1 * F12 / A2, 1.0 - F21]])
    surfaces = [
        GraySurface(area=A1, emissivity=eps1, temperature=T1, name="inner_sphere"),
        GraySurface(area=A2, emissivity=eps2, temperature=T2, name="outer_sphere"),
    ]
    return GrayEnclosure(surfaces, F)


def cube_enclosure(
    side: float,
    eps_opposite: float,
    eps_adjacent: float,
    T_opposite: float,
    T_adjacent: float,
) -> GrayEnclosure:
    """Six-face cube enclosure built from the closed-form family.

    For a cube of side :math:`s`, face-to-face view factors are the
    parallel-rectangle value at ``X = Y = 1`` for the opposite face and the
    perpendicular-rectangle value at ``X = Y = 1`` for the four adjacent faces.
    The exact summation rule ``F_opp + 4 F_adj = 1`` is therefore satisfied to
    the accuracy of the two catalogue numbers (|residual| ~ 2e-5); here the
    adjacent value is taken as the closed-form perpendicular result and the
    opposite value as the consistent complement ``1 - 4 F_adj``, so the
    resulting matrix is *exactly* summation- and reciprocity-consistent.
    """
    from .radiative_view_factors import vf_perpendicular_rectangles_common_edge

    F_adj = float(vf_perpendicular_rectangles_common_edge(1.0, 1.0))
    F_opp = 1.0 - 4.0 * F_adj
    n = 6
    F = np.zeros((n, n))
    pairs_opp = [(0, 1), (2, 3), (4, 5)]
    for i, j in pairs_opp:
        F[i, j] = F_opp
        F[j, i] = F_opp
    for i in range(n):
        for j in range(n):
            if i != j and F[i, j] == 0.0:
                F[i, j] = F_adj
    areas = [side * side] * n
    surfaces = []
    for k in range(n):
        eps = eps_opposite if k in (0, 1) else eps_adjacent
        T = T_opposite if k in (0, 1) else T_adjacent
        surfaces.append(GraySurface(area=areas[k], emissivity=eps, temperature=T, name=f"face{k}"))
    return GrayEnclosure(surfaces, F)