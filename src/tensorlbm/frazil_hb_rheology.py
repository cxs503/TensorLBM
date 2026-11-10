"""Frazil-ice suspended flow: volume fraction -> Herschel-Bulkley rheology.

Campaign node **P519-N04** — first Herschel-Bulkley (H-B) non-Newtonian
rheology verification surface for *frazil-ice suspended flow* in TensorLBM.

Why a new surface
-----------------
The repository already owns three neighbouring pieces, and this module is
deliberately built *on top of them* instead of duplicating them:

* :mod:`tensorlbm.frazil_rheology` — the **W9-1215 physical root**: the
  Krieger-Dougherty relative viscosity ``mu_r(phi)`` and the shear-thinning
  index ``n(phi)`` of a frazil slurry (concentration -> effective-medium
  viscosity window, ``phi_max = 0.60``, ``[eta] = 2.5``).
* :mod:`tensorlbm.herschel_bulkley` — the **analytic** H-B surface (pipe/plane
  Poiseuille profiles, plug relation, Couette regimes) plus the legacy
  ``slush_rheology`` closure (representative, un-fitted, lattice units only,
  not tied to the W9-1215 root).
* :mod:`tensorlbm.herschel_bulkley_lbm` — the **numerical membrane**
  (variable-viscosity BGK with the Papanastasiou-regularised H-B viscosity,
  Guo forcing, half-way bounce-back walls, wall-aware gradients).

What was missing — and is added here — is the *chain* that turns an ice volume
fraction ``phi`` into a **yield stress** and carries it, together with the
W9-1215 ``K(phi)``/``n(phi)`` pair, into a dimensionally self-consistent
lattice-Boltzmann case::

    phi  --(gel/percolation closure)-->  tau_y [Pa]
         --(W9-1215 root)------------->  K [Pa s^n], n [-]
         --(mixture density)---------->  rho [kg m^-3]
         --(kinematic reduction)------>  tau_y/rho [m^2 s^-2], K/rho [m^2 s^-n]
         --(dx, dt scaling)----------->  lattice H-B parameters + a_lat

Physics of the yield stress (new closure)
-----------------------------------------
A frazil suspension has *no* yield stress in the dilute regime: the disc-shaped
crystals rearrange freely.  A continuous crystal network — and therefore a
yield stress — appears only above a **gel (percolation) volume fraction**
``phi_gel``.  Continuum percolation of oblate particles is close to
``phi_c ~ 0.2`` for aspect ratios of ``O(1/10)``; the frazil discs themselves
are thinner (``O(1/100)``) but aggregate into porous flocs, which raises the
effective aspect ratio.  Ice-slurry rheometry (Guilpart et al. 1999;
Aygün & Vincenzini 2000) reports a measurable yield stress from roughly
10-20 % ice, growing steeply towards dense slush.  The closure used here is

    tau_y(phi) = 0,                                                phi <= phi_gel
    tau_y(phi) = tau_y_ref * ((phi-phi_gel)/(phi_ref-phi_gel))**p,  otherwise

with ``phi_gel = 0.20`` (network percolation), ``p = 2.5`` (the 2-3 range quoted
for ice slurries / frazil floes; above the 1.8 rigidity-percolation exponent of
a purely entropic network) and the single anchor ``tau_y_ref = 25 Pa`` at
``phi_ref = 0.50``, inside the ``O(10)-O(10^2) Pa`` range reported for dense
slush.  This is a *representative closure*, not a fit to a particular data set.
The verification is conditioned on ``(n, r_y)`` — so the numerical gates below
do not depend on these closure constants; only the reported physical case does.

Dimensional chain
-----------------
Every quantity is tracked in ``(kg, m, s)`` exponent form
(:func:`unit_exponents`) and both representations are carried side by side:

============================  =========================  ==============================
quantity                      SI                         lattice (``dx``, ``dt``)
============================  =========================  ==============================
dynamic tau_y                 Pa = kg m^-1 s^-2          ``tau_y * dt^2/dx^2``
dynamic K                     Pa s^n = kg m^-1 s^(n-2)   ``K * dt^(2-n) / dx^2``
kinematic tau_y/rho           m^2 s^-2                   ``(tau_y/rho) dt^2/dx^2``
kinematic K/rho               m^2 s^(n-2)                ``(K/rho) dt^(2-n)/dx^2``
velocity u                    m s^-1                     ``* dt/dx``
============================  =========================  ==============================

:func:`scaling_consistency` returns the residuals of these identities evaluated
on the analytic H-B profile — the machine-checkable form of the campaign gate
"量纲一致性检查通过".

Discrete geometry of the channel kernel (half-cell bookkeeping)
---------------------------------------------------------------
The D2Q9 channel of :mod:`tensorlbm.herschel_bulkley_lbm` uses *node-based*
post-streaming bounce-back: the incoming population at a solid node is replaced
by the opposite population of its fluid neighbour, so the no-slip plane of the
kernel is the **centre of the solid row**, not the mid-link.  With ``N`` fluid
rows and one solid row on each side the gap is therefore ``H = N + 1`` cells,
the fluid cell centres sit at a wall distance ``s = 1 .. N`` (``y = N/2`` is the
mid-plane), and the body force ``a`` acts on the ``N`` fluid rows, so the
momentum balance per unit width gives the *derived* wall stress

    tau_w = a N/2          (plane Poiseuille),     beta_eff = a N/(N+1)  (Couette).

Both numbers are fixed by the scheme, not fitted.  Their consequence is a
half-cell residual that is first order in ``1/N``; the grid ladder of the
benchmark quantifies it and the GCI gate measures the remaining uncertainty.
The alternative conventions (``H = N``, or ``tau_w = a H/2``) are reported in
the artifact for transparency.

Similarity of the H-B channel
-----------------------------
For plane H-B Poiseuille flow the analytic solution

    u(y) = (n/(n+1)) K^(-1/n) (H/(2 tau_w)) [ (tau_w - tau_y)^((n+1)/n)
                                              - (2 tau_w |y|/H - tau_y)_+^((n+1)/n) ]

is *homogeneous*: the non-dimensional profile

    utilde(xi) = u K^(1/n) / (H tau_w^(1/n)),   xi = 2y/H

depends only on ``(n, r_y)`` with ``r_y = tau_y/tau_w``.  The lattice run is
therefore driven at the same ``(n, r_y)`` and compared in non-dimensional
variables; the physical profile follows as ``u = utilde H tau_w^(1/n)/K^(1/n)``
in SI.  :func:`lattice_mapping` does that conversion with ``dx`` and ``dt`` chosen
so that the maximum lattice velocity stays in the Mach-controlled range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .frazil_rheology import (
    DEFAULT_INDEX_EXPONENT,
    DEFAULT_INTRINSIC_VISCOSITY,
    DEFAULT_N_PACKED,
    DEFAULT_PHI_MAX,
    frazil_consistency_index,
    frazil_flow_index,
)
from .herschel_bulkley import HBRheology, validate_rheology

__all__ = [
    "PHI_GEL",
    "YIELD_EXPONENT",
    "PHI_YIELD_REF",
    "TAU_Y_REF_PA",
    "RHO_WATER",
    "RHO_ICE",
    "NU_WATER",
    "FrazilHBState",
    "FrazilHBError",
    "mixture_density",
    "frazil_yield_stress",
    "frazil_hb_state",
    "unit_exponents",
    "kinematic_constitutive",
    "dimensional_consistency",
    "physical_flow_case",
    "lattice_mapping",
    "verification_case",
    "scaling_consistency",
    "hb_plane_shear_rate",
    "hb_couette_adverse_shear_rate",
    "yield_fraction",
]

# ---------------------------------------------------------------------------
# Closure constants (all documented in the module docstring)
# ---------------------------------------------------------------------------

PHI_GEL = 0.20  # network percolation threshold of the frazil crystal assembly
YIELD_EXPONENT = 2.5  # tau_y ~ (phi-phi_gel)^p, p in the ice-slurry range 2-3
PHI_YIELD_REF = 0.50  # volume fraction at which the yield stress is anchored
TAU_Y_REF_PA = 25.0  # Pa, anchor value for dense slush (O(10)-O(10^2) Pa)

RHO_WATER = 999.8  # kg m^-3, water at 0 C
RHO_ICE = 917.0  # kg m^-3
NU_WATER = 1.792e-6  # m^2 s^-1, kinematic viscosity of water at 0 C


class FrazilHBError(ValueError):
    """Raised for out-of-range or non-physical frazil H-B input."""


# ---------------------------------------------------------------------------
# Closures
# ---------------------------------------------------------------------------


def _check_phi(phi: float, phi_max: float = DEFAULT_PHI_MAX) -> float:
    if not math.isfinite(phi):
        raise FrazilHBError(f"phi must be finite, got {phi!r}")
    if phi < 0.0 or phi >= phi_max:
        raise FrazilHBError(
            f"phi must satisfy 0 <= phi < phi_max={phi_max}, got {phi!r} "
            "(the closure is undefined past the packed/floc limit)"
        )
    return float(phi)


def mixture_density(
    phi: float,
    *,
    rho_water: float = RHO_WATER,
    rho_ice: float = RHO_ICE,
    phi_max: float = DEFAULT_PHI_MAX,
) -> float:
    """Density of the frazil suspension ``rho = (1-phi) rho_w + phi rho_i``."""
    p = _check_phi(phi, phi_max)
    if rho_water <= 0.0 or rho_ice <= 0.0:
        raise FrazilHBError("densities must be > 0")
    return (1.0 - p) * rho_water + p * rho_ice


def frazil_yield_stress(
    phi: float,
    *,
    phi_gel: float = PHI_GEL,
    phi_ref: float = PHI_YIELD_REF,
    tau_y_ref: float = TAU_Y_REF_PA,
    exponent: float = YIELD_EXPONENT,
    phi_max: float = DEFAULT_PHI_MAX,
) -> float:
    """Yield stress ``tau_y(phi)`` [Pa] of a frazil suspension.

    Zero below the gel threshold ``phi_gel`` (no continuous crystal network),
    then ``tau_y_ref * ((phi-phi_gel)/(phi_ref-phi_gel))**exponent``.

    Args:
        phi: Ice volume fraction, ``0 <= phi < phi_max``.
        phi_gel: Percolation threshold of the crystal network.
        phi_ref: Volume fraction at which the anchor is imposed.
        tau_y_ref: Anchor yield stress [Pa] at ``phi_ref``.
        exponent: Power of the network-strength growth.
        phi_max: Packed / floc limit of the suspension.

    Returns:
        Yield stress in Pa (``0.0`` below the gel point).

    Raises:
        FrazilHBError: for out-of-range arguments.
    """
    p = _check_phi(phi, phi_max)
    if not (0.0 <= phi_gel < phi_max):
        raise FrazilHBError(f"phi_gel must satisfy 0 <= phi_gel < phi_max, got {phi_gel!r}")
    if not (phi_gel < phi_ref < phi_max):
        raise FrazilHBError(f"phi_ref must satisfy phi_gel < phi_ref < phi_max, got {phi_ref!r}")
    if tau_y_ref < 0.0 or not math.isfinite(tau_y_ref):
        raise FrazilHBError(f"tau_y_ref must be finite and >= 0, got {tau_y_ref!r}")
    if exponent <= 0.0:
        raise FrazilHBError(f"exponent must be > 0, got {exponent!r}")
    if p <= phi_gel:
        return 0.0
    s = (p - phi_gel) / (phi_ref - phi_gel)
    return float(tau_y_ref * s**exponent)


@dataclass(frozen=True)
class FrazilHBState:
    """Herschel-Bulkley state of a frazil suspension (SI + kinematic form).

    Attributes:
        phi: Ice volume fraction.
        rho: Suspension density [kg m^-3].
        mu_r: Krieger-Dougherty relative viscosity of the W9-1215 root.
        n: Flow index ``n(phi)`` of the W9-1215 root (``<= 1``).
        tau_y: Yield stress [Pa] (dynamic).
        K: Consistency index [Pa s^n] (dynamic).
        tau_y_kin: Kinematic yield stress ``tau_y/rho`` [m^2 s^-2].
        K_kin: Kinematic consistency ``K/rho`` [m^2 s^-n].
    """

    phi: float
    rho: float
    mu_r: float
    n: float
    tau_y: float
    K: float
    tau_y_kin: float
    K_kin: float

    def as_dict(self) -> dict[str, float]:
        return {
            "phi": self.phi,
            "rho": self.rho,
            "mu_r": self.mu_r,
            "n": self.n,
            "tau_y_Pa": self.tau_y,
            "K_Pa_sn": self.K,
            "tau_y_kin_m2_s2": self.tau_y_kin,
            "K_kin_m2_sn": self.K_kin,
        }

    def yield_fraction(self, tau_w_kin: float) -> float:
        """Plastic ratio ``r_y = tau_y/tau_w`` of a wall-stress-driven flow."""
        if tau_w_kin <= 0.0:
            raise FrazilHBError(f"tau_w_kin must be > 0, got {tau_w_kin!r}")
        return self.tau_y_kin / tau_w_kin

    def with_plastic_ratio(self, r_y: float, tau_w_kin: float) -> HBRheology:
        """Kinematic H-B parameters at plastic ratio ``r_y`` and wall stress."""
        if not (0.0 <= r_y < 1.0):
            raise FrazilHBError(f"r_y must satisfy 0 <= r_y < 1, got {r_y!r}")
        return validate_rheology(r_y * tau_w_kin, self.K_kin, self.n)

    def yield_fraction_of(self, tau_w_kin: float) -> float:
        return self.yield_fraction(tau_w_kin)


def frazil_hb_state(
    phi: float,
    *,
    nu_water: float = NU_WATER,
    rho_water: float = RHO_WATER,
    rho_ice: float = RHO_ICE,
    phi_max: float = DEFAULT_PHI_MAX,
    intrinsic: float = DEFAULT_INTRINSIC_VISCOSITY,
    n_packed: float = DEFAULT_N_PACKED,
    index_exponent: float = DEFAULT_INDEX_EXPONENT,
    phi_gel: float = PHI_GEL,
    phi_ref: float = PHI_YIELD_REF,
    tau_y_ref: float = TAU_Y_REF_PA,
    yield_exponent: float = YIELD_EXPONENT,
) -> FrazilHBState:
    """Full H-B state of a frazil suspension at volume fraction ``phi``.

    ``K`` and ``n`` come from the **W9-1215 physical root**
    (:mod:`tensorlbm.frazil_rheology`): ``K = rho * nu_water * mu_r(phi)`` and
    ``n = n(phi)``.  ``tau_y`` comes from :func:`frazil_yield_stress`.

    Args:
        phi: Ice volume fraction.
        nu_water: Kinematic viscosity of the carrier liquid [m^2 s^-1].
        rho_water, rho_ice: Component densities [kg m^-3].
        phi_max, intrinsic, n_packed, index_exponent: W9-1215 root parameters.
        phi_gel, phi_ref, tau_y_ref, yield_exponent: yield-stress closure.

    Returns:
        :class:`FrazilHBState` with both the dynamic (SI) and the kinematic
        representation populated.

    Raises:
        FrazilHBError: on out-of-range input.
    """
    p = _check_phi(phi, phi_max)
    if nu_water <= 0.0:
        raise FrazilHBError(f"nu_water must be > 0, got {nu_water!r}")
    rho = mixture_density(p, rho_water=rho_water, rho_ice=rho_ice, phi_max=phi_max)
    mu_r = float(
        frazil_consistency_index(
            p,
            mu_water=1.0,
            phi_max=phi_max,
            intrinsic=intrinsic,
        )
    )
    n = float(
        frazil_flow_index(
            p,
            n_water=1.0,
            n_packed=n_packed,
            phi_max=phi_max,
            exponent=index_exponent,
        )
    )
    tau_y = frazil_yield_stress(
        p,
        phi_gel=phi_gel,
        phi_ref=phi_ref,
        tau_y_ref=tau_y_ref,
        exponent=yield_exponent,
        phi_max=phi_max,
    )
    K = rho * nu_water * mu_r
    return FrazilHBState(
        phi=p,
        rho=rho,
        mu_r=mu_r,
        n=n,
        tau_y=tau_y,
        K=K,
        tau_y_kin=tau_y / rho,
        K_kin=K / rho,
    )


# ---------------------------------------------------------------------------
# Dimensional analysis
# ---------------------------------------------------------------------------

#: exponents of the base units (kg, m, s) used by :func:`unit_exponents`
_KG, _M, _S = 0, 1, 2


def unit_exponents(quantity: str, n: float = 1.0) -> tuple[float, float, float]:
    """``(kg, m, s)`` exponents of the rheological quantities used here.

    Args:
        quantity: One of ``rho``, ``tau`` (dynamic stress), ``tau_kin``
            (kinematic stress), ``K`` (dynamic consistency), ``K_kin``
            (kinematic consistency), ``nu`` (kinematic viscosity),
            ``gamma_dot`` (shear rate), ``u`` (velocity), ``H`` (length).
        n: Flow index, entering the time exponent of ``K``.

    Returns:
        Exponent triple ``(kg, m, s)``.

    Raises:
        FrazilHBError: for an unknown quantity.
    """
    table = {
        "rho": (1.0, -3.0, 0.0),
        "tau": (1.0, -1.0, -2.0),
        "tau_kin": (0.0, 2.0, -2.0),
        "K": (1.0, -1.0, -2.0 + n),
        "K_kin": (0.0, 2.0, -2.0 + n),
        "nu": (0.0, 2.0, -1.0),
        "gamma_dot": (0.0, 0.0, -1.0),
        "u": (0.0, 1.0, -1.0),
        "H": (0.0, 1.0, 0.0),
    }
    if quantity not in table:
        raise FrazilHBError(f"unknown quantity {quantity!r}; known: {sorted(table)}")
    return table[quantity]


def _sub(a: tuple[float, float, float], b: tuple[float, float, float]):
    return tuple(x - y for x, y in zip(a, b))


def kinematic_constitutive(
    state: FrazilHBState, gamma_dot: float, *, m: float = 1.0e4
) -> dict[str, float]:
    """Kinematic H-B viscosity/stress at a shear rate, in SI-derived units.

    ``nu = tau_y_kin*(1-exp(-m g))/g + K_kin*g**(n-1)`` (Papanastasiou), so the
    dimensional bookkeeping can be checked numerically as well as symbolically.
    """
    if gamma_dot < 0.0:
        raise FrazilHBError(f"gamma_dot must be >= 0, got {gamma_dot!r}")
    g = float(gamma_dot)
    if g <= 0.0:
        nu = state.tau_y_kin * m + state.K_kin * (0.0 if state.n > 1.0 else 0.0)
        return {"gamma_dot": 0.0, "nu": nu, "tau": 0.0}
    ratio = (1.0 - math.exp(-m * g)) / g if m * g > 1e-12 else m
    nu = state.tau_y_kin * ratio + state.K_kin * g ** (state.n - 1.0)
    return {"gamma_dot": g, "nu": nu, "tau": nu * g}


def dimensional_consistency(state: FrazilHBState, *, tol: float = 1e-10) -> dict:
    """Unit-algebra and round-trip check of the ``phi -> SI -> kinematic`` chain.

    Three independent families of checks:

    1. **Unit algebra** — the ``(kg, m, s)`` exponents of the derived kinematic
       quantities must equal ``tau_y/rho`` and ``K/rho`` computed symbolically,
       and the constitutive product ``nu*gamma_dot`` must be a kinematic stress.
    2. **Numeric round trip** — ``tau_y_kin*rho == tau_y`` and
       ``K_kin*rho == K`` to machine precision.
    3. **Constitutive units** — at a representative shear rate the kinematic
       viscosity has the exponents of ``m^2 s^-1`` and the stress those of
       ``m^2 s^-2`` in *both* the SI and the kinematic representation.

    Returns:
        dict with ``passed`` (bool), ``checks`` (name -> {residual, passed}),
        ``exponents`` and ``worst_residual``.
    """
    checks: dict[str, dict] = {}

    def record(name: str, residual: float, limit: float) -> None:
        checks[name] = {"residual": float(residual), "passed": bool(residual <= limit)}

    # 1. unit algebra
    e_tau_kin = unit_exponents("tau_kin")
    e_K_kin = unit_exponents("K_kin", state.n)
    record(
        "units_tau_y_kin_is_stress_over_density",
        float(
            max(
                abs(a - b)
                for a, b in zip(e_tau_kin, _sub(unit_exponents("tau"), unit_exponents("rho")))
            )
        ),
        1e-12,
    )
    record(
        "units_K_kin_is_K_over_density",
        float(
            max(
                abs(a - b)
                for a, b in zip(e_K_kin, _sub(unit_exponents("K", state.n), unit_exponents("rho")))
            )
        ),
        1e-12,
    )
    e_nu_g = tuple(
        a + b
        for a, b in zip(unit_exponents("nu"), unit_exponents("gamma_dot"))
    )
    record(
        "units_nu_times_gamma_dot_is_tau_kin",
        float(max(abs(a - b) for a, b in zip(e_nu_g, e_tau_kin))),
        1e-12,
    )
    e_K_from_nu = tuple(
        a - b for a, b in zip(unit_exponents("nu"), (0.0, 0.0, -(state.n - 1.0)))
    )
    record(
        "units_K_solves_nu_equals_K_g_n_minus_1",
        float(max(abs(a - b) for a, b in zip(e_K_from_nu, e_K_kin))),
        1e-12,
    )

    # 2. numeric round trips (relative)
    record("roundtrip_tau_y_kin_times_rho", abs(state.tau_y_kin * state.rho - state.tau_y) / max(state.tau_y, 1e-30), tol)
    record("roundtrip_K_kin_times_rho", abs(state.K_kin * state.rho - state.K) / state.K, tol)

    # 3. constitutive units at a representative shear rate
    g = 10.0
    si = kinematic_constitutive(state, g)
    record(
        "constitutive_nu_positive_finite",
        0.0 if (math.isfinite(si["nu"]) and si["nu"] > 0.0) else 1.0,
        1e-12,
    )
    record(
        "constitutive_tau_equals_nu_g",
        abs(si["tau"] - si["nu"] * g) / (si["nu"] * g),
        tol,
    )
    worst = max(v["residual"] for v in checks.values())
    return {
        "passed": all(v["passed"] for v in checks.values()),
        "checks": checks,
        "exponents": {
            "tau_kin": list(e_tau_kin),
            "K_kin": list(e_K_kin),
            "nu": list(unit_exponents("nu")),
        },
        "worst_residual": float(worst),
        "tau_y_Pa": state.tau_y,
        "K_Pa_sn": state.K,
        "n": state.n,
        "phi": state.phi,
    }


# ---------------------------------------------------------------------------
# Analytic H-B shear-rate distributions (new; companions of the velocity
# profiles already in tensorlbm.herschel_bulkley)
# ---------------------------------------------------------------------------


def yield_fraction(tau_y: float, tau_w: float) -> float:
    """Plastic ratio ``r_y = tau_y/tau_w`` (``0`` for a simple fluid)."""
    if tau_w <= 0.0:
        raise FrazilHBError(f"tau_w must be > 0, got {tau_w!r}")
    if tau_y < 0.0:
        raise FrazilHBError(f"tau_y must be >= 0, got {tau_y!r}")
    return min(1.0, tau_y / tau_w)


def hb_plane_shear_rate(
    y: torch.Tensor | float,
    H: float,
    tau_w: float,
    rheo: HBRheology,
) -> torch.Tensor:
    """Exact H-B plane-Poiseuille shear rate ``gamma_dot(y)`` (walls ``+-H/2``).

    Inverts ``tau = tau_y + K gamma_dot**n`` on the momentum-balance stress
    ``sigma(y) = tau_w * 2|y|/H``::

        gamma_dot(y) = ((tau_w*2|y|/H - tau_y)/K)**(1/n)   for |y| > y_c
        gamma_dot(y) = 0                                   inside the plug

    with the plug half-height ``y_c = (H/2)*tau_y/tau_w``.  The velocity profile
    in :func:`tensorlbm.herschel_bulkley.hb_plane_velocity_profile` is the
    integral of this field, which is checked in the test suite.
    """
    if H <= 0.0 or tau_w <= 0.0:
        raise FrazilHBError(f"H and tau_w must be > 0, got H={H!r}, tau_w={tau_w!r}")
    yy = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.float64)
    yy = yy.to(torch.float64)
    sig = tau_w * 2.0 * torch.clamp(torch.abs(yy) / H, min=0.0)
    inner = torch.clamp(sig - rheo.tau_y, min=0.0)
    return torch.pow(inner / rheo.K, 1.0 / rheo.n)


def hb_couette_adverse_shear_rate(
    y: torch.Tensor | float,
    beta: float,
    rheo: HBRheology,
) -> torch.Tensor:
    """Exact shear rate of the adverse-gradient Couette-Poiseuille case.

    Stress ``sigma(y) = beta*y`` (lower plate at rest at ``y = 0``, adverse
    gradient ``beta``), plug on ``0 <= y <= y_yield = tau_y/beta``::

        gamma_dot(y) = ((beta*y - tau_y)/K)**(1/n)  for y > y_yield, else 0.

    Companion of
    :func:`tensorlbm.herschel_bulkley.hb_couette_adverse_profile`.
    """
    if beta <= 0.0:
        raise FrazilHBError(f"beta must be > 0, got {beta!r}")
    yy = y if isinstance(y, torch.Tensor) else torch.as_tensor(y, dtype=torch.float64)
    yy = yy.to(torch.float64)
    inner = torch.clamp(beta * yy - rheo.tau_y, min=0.0)
    return torch.pow(inner / rheo.K, 1.0 / rheo.n)


# ---------------------------------------------------------------------------
# Physical case -> lattice case (dimensional scaling)
# ---------------------------------------------------------------------------


def physical_flow_case(
    phi: float,
    *,
    H: float = 0.05,
    r_y: float | None = 0.40,
    a_si: float | None = None,
    nu_water: float = NU_WATER,
    rho_water: float = RHO_WATER,
    rho_ice: float = RHO_ICE,
    phi_gel: float = PHI_GEL,
    phi_ref: float = PHI_YIELD_REF,
    tau_y_ref: float = TAU_Y_REF_PA,
    yield_exponent: float = YIELD_EXPONENT,
) -> dict:
    """Physical plane-channel case of a frazil suspension.

    A channel of gap ``H`` [m] driven by a uniform body force.  For a
    yield-stress material the meaningful control parameter is the *plastic
    ratio* ``r_y = tau_y/tau_w`` (it fixes the plug fraction and the shape of
    the non-dimensional profile), so by default the wall stress is imposed
    through ``r_y`` and the body force follows from the momentum balance
    ``a = 2 tau_w/(rho H)``.  Alternatively a body force ``a_si`` can be given
    directly; if it is too small to yield the material (``r_y >= 1``) the case is
    flagged ``yielded=False`` and the analytic velocity is zero.

    Args:
        phi: Ice volume fraction.
        H: Channel gap [m].
        r_y: Target plastic ratio ``tau_y/tau_w`` (``None`` together with
            ``a_si`` switches to the body-force parameterisation).
        a_si: Body-force acceleration [m s^-2] (defaults to the value implied
            by ``r_y``, or to ``2.5e-3`` for a simple fluid with ``tau_y = 0``).
        nu_water, rho_water, rho_ice, phi_gel, phi_ref, tau_y_ref, yield_exponent:
            Forwarded to :func:`frazil_hb_state`.

    Returns:
        dict with the :class:`FrazilHBState`, ``H``, ``a_si``, ``tau_w_Pa``,
        ``tau_w_kin``, ``r_y``, ``yielded``, the analytic maximum velocity
        [m s^-1], the flow rate per unit width [m^2 s^-1] and the plug
        half-height [m].
    """
    from .herschel_bulkley import hb_plane_flow_rate, hb_plane_velocity_profile

    if H <= 0.0:
        raise FrazilHBError(f"H must be > 0, got {H!r}")
    state = frazil_hb_state(
        phi,
        nu_water=nu_water,
        rho_water=rho_water,
        rho_ice=rho_ice,
        phi_gel=phi_gel,
        phi_ref=phi_ref,
        tau_y_ref=tau_y_ref,
        yield_exponent=yield_exponent,
    )
    if r_y is not None and not (0.0 <= r_y < 1.0):
        raise FrazilHBError(f"r_y must satisfy 0 <= r_y < 1, got {r_y!r}")

    if r_y is not None and state.tau_y > 0.0:
        if r_y <= 0.0:
            raise FrazilHBError("r_y must be > 0 for a yield-stress material")
        tau_w_pa = state.tau_y / r_y
    else:
        a_for_stress = 2.5e-3 if a_si is None else float(a_si)
        if a_for_stress <= 0.0:
            raise FrazilHBError(f"a_si must be > 0, got {a_for_stress!r}")
        tau_w_pa = state.rho * a_for_stress * H / 2.0
    a_used = 2.0 * tau_w_pa / (state.rho * H)
    tau_w_kin = tau_w_pa / state.rho
    r_y_eff = yield_fraction(state.tau_y_kin, tau_w_kin)
    yielded = bool(r_y_eff < 1.0)
    rheo_kin = validate_rheology(state.tau_y_kin, state.K_kin, state.n)
    if yielded:
        u_max = float(
            hb_plane_velocity_profile(
                torch.tensor([0.0], dtype=torch.float64), H, tau_w_kin, rheo_kin
            )[0]
        )
        q = float(hb_plane_flow_rate(tau_w_kin, H, rheo_kin))
    else:
        u_max = 0.0
        q = 0.0
    return {
        "state": state,
        "H": H,
        "a_si": a_used,
        "dPdx_Pa_per_m": state.rho * a_used,
        "tau_w_Pa": tau_w_pa,
        "tau_w_kin": tau_w_kin,
        "r_y": r_y_eff,
        "yielded": yielded,
        "u_max_si": u_max,
        "q_si": q,
        "y_c_si": (H / 2.0) * r_y_eff,
        "rheo_kin": rheo_kin,
    }


def lattice_mapping(
    case: dict,
    *,
    gap_cells: int,
    u_lat_target: float = 0.015,
) -> dict:
    """``(dx, dt)`` mapping of a physical case onto a lattice (dimensional bookkeeping).

    ``dx = H/gap_cells`` and ``dt`` is fixed by ``u_lat_target``: with the
    mapping of the module docstring the lattice velocity is *linear* in ``dt``
    (the stress mapping carries ``dt^2`` and the consistency mapping
    ``dt^(2-n)`` against the ``K^-1/n`` of the profile), so
    ``dt = dt_ref u_lat_target/u_ref`` is exact and needs no iteration.

    This function is used for the **dimensional consistency check** only.  The
    lattice H-B parameters it produces are the exact images of the physical ones
    and therefore inherit the physical Reynolds number: for the frazil closure
    of :func:`frazil_hb_state` at ``phi >= 0.4`` the implied relaxation time is
    ``tau - 1/2 = 3 nu_lat ~ 1e-4`` at ``dx = H/64``, i.e. the physical case is
    *not resolved* by the lattice-Boltzmann solver at that resolution (reported
    as such).  The numerical verification runs the *similar* non-dimensional
    problem built by :func:`verification_case`.

    Returns:
        dict with ``dx``, ``dt``, ``tau_w_lat``, ``K_lat``, ``rheo`` (mapped
        lattice H-B parameters), ``a_lat``, ``u_lat`` (analytic max lattice
        velocity), ``tau_minus_half`` (``3 nu`` at the characteristic shear
        rate) and the plastic ratio ``r_y``.
    """
    from .herschel_bulkley import hb_plane_velocity_profile

    state: FrazilHBState = case["state"]
    if gap_cells < 8:
        raise FrazilHBError(f"gap_cells too small to resolve the plug: {gap_cells!r}")
    if not (0.0 < u_lat_target < 0.1):
        raise FrazilHBError(f"u_lat_target must be in (0, 0.1), got {u_lat_target!r}")
    H = case["H"]
    H_lat = float(gap_cells) + 1.0  # discrete gap of the channel kernel
    dx = H / H_lat
    r_y = case["r_y"]
    # reference mapping with dt = 1, then rescale onto the target velocity
    tau_ref = case["tau_w_kin"]
    K_ref = state.K_kin
    if tau_ref <= 0.0 or K_ref <= 0.0:
        raise FrazilHBError("degenerate physical case (zero wall stress)")
    rheo_ref = validate_rheology(r_y * tau_ref, K_ref, state.n)
    u_ref = float(
        hb_plane_velocity_profile(
            torch.zeros(1, dtype=torch.float64), H_lat, tau_ref, rheo_ref
        )[0]
    )
    if not math.isfinite(u_ref) or u_ref <= 0.0:
        raise FrazilHBError("degenerate physical case (no flow at the given plastic ratio)")
    dt = u_lat_target / u_ref
    tau_w_lat = case["tau_w_kin"] * dt**2 / dx**2
    K_lat = state.K_kin * dt ** (2.0 - state.n) / dx**2
    rheo = validate_rheology(r_y * tau_w_lat, K_lat, state.n)
    u_lat = float(
        hb_plane_velocity_profile(
            torch.zeros(1, dtype=torch.float64), H_lat, tau_w_lat, rheo
        )[0]
    )
    gamma_char = u_lat / (H_lat / 4.0)
    nu_char = K_lat * gamma_char ** (state.n - 1.0)
    return {
        "dx": dx,
        "dt": dt,
        "rheo": rheo,
        "a_lat": 2.0 * tau_w_lat / gap_cells,
        "tau_w_lat": tau_w_lat,
        "K_lat": K_lat,
        "u_lat": u_lat,
        "r_y": r_y,
        "n": state.n,
        "gamma_lat_char": gamma_char,
        "nu_lat_char": nu_char,
        "tau_minus_half": 3.0 * nu_char,
        "gap_cells": gap_cells,
        "H_lat": H_lat,
        "u_lat_target": u_lat_target,
    }


def verification_case(
    n: float,
    r_y: float,
    *,
    gap_cells: int = 64,
    u_lat_target: float = 0.015,
    nu_char_target: float = 0.03,
    c_regularisation: float = 30.0,
) -> dict:
    """Well-conditioned lattice case at the similarity point ``(n, r_y)``.

    The non-dimensional H-B plane-channel solution depends only on ``(n, r_y)``
    (see the module docstring), so the lattice stress level is free and is used
    here to place the *numerics* in the well-behaved range: the consistency
    index is chosen so that the kinematic viscosity at the characteristic shear
    rate equals ``nu_char_target`` (relaxation time ``3 nu + 1/2 ~ 0.6``), and
    the wall stress follows from the H-B momentum balance.

    Derivation (``C0 = (n/(2(n+1)))(1-r_y)^((n+1)/n)``, gap ``H = N+1``)::

        gamma_char = 4 u_lat/H                          (shear rate at wall scale)
        K_lat      = nu_char_target * gamma_char**(1-n)
        tau_w_lat  = K_lat * (u_lat/(C0 H))**n          (u_max = C0 H tau^(1/n)/K^(1/n))
        tau_y_lat  = r_y * tau_w_lat
        a_lat      = 2 tau_w_lat/N                      (momentum balance over N rows)
        m          = c_regularisation/gamma_char        (Papanastasiou rate)

    With ``c_regularisation = 30`` the regularised yield term is within
    ``exp(-30) ~ 1e-13`` of the exact H-B law everywhere in the yielded layer,
    while the plug viscosity ``tau_y*m`` stays ``O(nu_char_target)`` — the
    conditioning compromise that makes the plug rigid *and* the BGK relaxation
    time moderate (a large ``m`` such as ``1e6`` drives ``tau`` into the
    thousands and stalls the steady state).

    Args:
        n: Flow index of the W9-1215 root.
        r_y: Plastic ratio ``tau_y/tau_w`` (``0 <= r_y < 1``).
        gap_cells: Channel gap in lattice cells.
        u_lat_target: Target maximum lattice velocity (Mach control).
        nu_char_target: Target kinematic viscosity at the characteristic shear
            rate (relaxation-time control).
        c_regularisation: Stiffness ratio ``m*gamma_char`` of the plug.

    Returns:
        dict with ``rheo`` (:class:`HBRheology`, lattice/kinematic), ``a_lat``,
        ``tau_w_lat``, ``u_lat`` (analytic maximum lattice velocity), ``m``,
        ``gamma_char``, ``nu_char``, ``nu_plug``, ``tau_minus_half`` and the
        similarity parameters ``n`` and ``r_y``.
    """
    if not (0.0 <= r_y < 1.0):
        raise FrazilHBError(f"r_y must satisfy 0 <= r_y < 1, got {r_y!r}")
    if not (0.0 < n <= 2.0):
        raise FrazilHBError(f"n must be in (0, 2], got {n!r}")
    if gap_cells < 8:
        raise FrazilHBError(f"gap_cells too small to resolve the plug: {gap_cells!r}")
    if not (0.0 < u_lat_target < 0.1):
        raise FrazilHBError(f"u_lat_target must be in (0, 0.1), got {u_lat_target!r}")
    if nu_char_target <= 0.0:
        raise FrazilHBError(f"nu_char_target must be > 0, got {nu_char_target!r}")
    if c_regularisation <= 1.0:
        raise FrazilHBError(f"c_regularisation must be > 1, got {c_regularisation!r}")

    from .herschel_bulkley import hb_plane_velocity_profile

    H_disc = float(gap_cells) + 1.0  # gap = N fluid rows + 2 solid rows (node bounce-back)
    gamma_char = 4.0 * u_lat_target / H_disc
    K_lat = nu_char_target * gamma_char ** (1.0 - n)
    C0 = (n / (2.0 * (n + 1.0))) * (1.0 - r_y) ** ((n + 1.0) / n)
    tau_w_lat = K_lat * (u_lat_target / (C0 * H_disc)) ** n
    rheo = validate_rheology(r_y * tau_w_lat, K_lat, n)
    u_lat = float(
        hb_plane_velocity_profile(
            torch.zeros(1, dtype=torch.float64), H_disc, tau_w_lat, rheo
        )[0]
    )
    m = c_regularisation / gamma_char
    nu_plug = float(rheo.tau_y * m)
    return {
        "rheo": rheo,
        "a_lat": 2.0 * tau_w_lat / gap_cells,  # momentum balance over the N fluid rows
        "tau_w_lat": tau_w_lat,
        "K_lat": K_lat,
        "H_disc": H_disc,
        "u_lat": u_lat,
        "m": float(m),
        "gamma_char": gamma_char,
        "nu_char": nu_char_target,
        "nu_plug": nu_plug,
        "tau_plug": 3.0 * nu_plug + 0.5,
        "tau_minus_half": 3.0 * nu_char_target,
        "n": float(n),
        "r_y": float(r_y),
        "gap_cells": gap_cells,
        "u_lat_target": u_lat_target,
        "c_regularisation": c_regularisation,
        "y_c_lat": 0.5 * H_disc * r_y,
    }


def scaling_consistency(
    case: dict,
    mapping: dict,
    verification: dict | None = None,
) -> dict:
    """SI <-> lattice consistency: dimensional identities + dimensionless similarity.

    Five dimensional identities are evaluated on the analytic H-B profile at the
    ``(dx, dt)`` of :func:`lattice_mapping`:

    1. ``u_SI(y) = (dx/dt) u_lat(y/dx)`` (velocity mapping);
    2. ``tau_y/tau_w = r_y`` identical in both representations (plastic similarity);
    3. ``K_lat = K_kin dt^(2-n)/dx^2`` (consistency-index mapping);
    4. ``tau_w_lat = tau_w_kin dt^2/dx^2`` (stress mapping);
    5. ``y_c_lat = y_c_SI/dx`` (plug length mapping).

    If ``verification`` (from :func:`verification_case`) is given, a sixth check
    compares the **non-dimensional** profiles of the physical and the
    numerically conditioned cases — the similarity statement that licenses using
    a different stress level for the actual lattice run.

    Returns:
        dict with ``passed``, ``worst_residual``, per-check residuals, and the
        conditioning diagnostics of the mapped case (``tau_minus_half``,
        ``resolved``) which document whether the physical case is resolvable by
        the solver at that ``dx``.
    """
    from .herschel_bulkley import hb_plane_velocity_profile

    H = case["H"]
    state: FrazilHBState = case["state"]
    n_cells = mapping["gap_cells"]
    H_lat = float(mapping.get("H_lat", n_cells + 1.0))
    dx, dt = mapping["dx"], mapping["dt"]
    xi = torch.linspace(-0.999, 0.999, 501, dtype=torch.float64)

    rheo_lat = mapping["rheo"]
    u_si = hb_plane_velocity_profile(xi * H / 2.0, H, case["tau_w_kin"], case["rheo_kin"])
    u_lat = hb_plane_velocity_profile(xi * H_lat / 2.0, H_lat, mapping["tau_w_lat"], rheo_lat)
    vel_res = float(torch.max(torch.abs(u_si - u_lat * dx / dt)) / torch.max(torch.abs(u_si)))

    r_y_si = state.tau_y_kin / case["tau_w_kin"]
    r_y_lat = rheo_lat.tau_y / mapping["tau_w_lat"]
    K_res = abs(rheo_lat.K - state.K_kin * dt ** (2.0 - state.n) / dx**2) / rheo_lat.K
    tau_res = abs(mapping["tau_w_lat"] - case["tau_w_kin"] * dt**2 / dx**2) / mapping["tau_w_lat"]
    y_c_si = (H / 2.0) * r_y_si
    y_c_lat = (H_lat / 2.0) * r_y_lat
    plug_res = abs(y_c_lat - y_c_si / dx) / max(y_c_si / dx, 1e-30)

    checks = {
        "velocity_mapping": vel_res,
        "plastic_ratio_similarity": abs(r_y_si - r_y_lat) / max(r_y_si, 1e-30),
        "consistency_mapping": K_res,
        "stress_mapping": tau_res,
        "plug_length_mapping": plug_res,
    }

    if verification is not None:
        rheo_v = verification["rheo"]
        H_v = float(verification.get("H_disc", n_cells + 1.0))
        u_v = hb_plane_velocity_profile(
            xi * H_v / 2.0,
            H_v,
            verification["tau_w_lat"],
            rheo_v,
        )
        # non-dimensional: utilde = u K^(1/n) / (h tau_w^(1/n))
        def utilde(u, h, tau_w, rheo):
            return u * rheo.K ** (1.0 / rheo.n) / (h * tau_w ** (1.0 / rheo.n))

        ut_si = utilde(u_si, H, case["tau_w_kin"], case["rheo_kin"])
        ut_v = utilde(u_v, H_v, verification["tau_w_lat"], rheo_v)
        sim_res = float(torch.max(torch.abs(ut_si - ut_v)) / torch.max(torch.abs(ut_si)))
        checks["dimensionless_profile_similarity"] = sim_res

    worst = max(checks.values())
    return {
        "passed": bool(worst <= 1e-9),
        "worst_residual": float(worst),
        "checks": {k: float(v) for k, v in checks.items()},
        "dx": dx,
        "dt": dt,
        "u_max_si": float(u_si.max()),
        "u_max_lat": float(u_lat.max()),
        "y_c_si": float(y_c_si),
        "y_c_lat": float(y_c_lat),
        "r_y": float(r_y_si),
        "tau_minus_half_mapped": float(mapping["tau_minus_half"]),
        "resolved_by_solver": bool(0.05 <= mapping["tau_minus_half"] <= 10.0),
    }
