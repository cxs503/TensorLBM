"""Analytical stability criteria for a floating ice cover under a uniform stream.

Two *competing* critical-velocity criteria are implemented, because the campaign node
asks for an adjudication ("互裁") between an LBM-DEM coupled simulation and the
prescribed analytical criterion.

(1) Prescribed criterion (campaign metadata, eq. 2)
        D k^4 + rho_w g = rho_w U_cr^2 k^2
    ==> U_cr^2(k) = D k^2 / rho_w + g / k^2
    ==> U_cr = min_k U_cr(k) = (4 D g / rho_w)^(1/4)          [k_c = (rho_w g / D)^(1/4)]
    This is mathematically identical to Euler buckling of a beam on an elastic
    (Winkler) foundation carrying an *axial compressive line load* N = rho_w U^2
    (i.e. the flow momentum flux acting as a membrane compression):
        D w'''' + N w'' + rho_w g w = p,   N_cr = 2 sqrt(D rho_w g)
    So the prescribed formula is the "momentum-flux compression" (membrane
    buckling) criterion.

(2) First-principles hydroelastic criterion (steady divergence / flexural-gravity
    resonance).  Linear potential flow over a gently deflected cover of water depth h
    gives a wall-pressure perturbation
        p_hat = rho_w U^2 k coth(k h) w_hat
    Balance against bending + buoyancy:
        D k^4 + rho_w g = rho_w U^2 k coth(k h)
    ==> U_B(k) = sqrt( (D k^4 + rho_w g) / (rho_w k coth(k h)) )
               = sqrt( tanh(k h) (D k^4 + rho_w g) / (rho_w k) )
    which is *exactly* the phase speed c(k) of flexural-gravity waves:
        omega^2 = (D k^4 + rho_w g) / ( rho_w / (k tanh(k h)) + m_s )
    Hence onset of the steady divergence coincides with a stationary
    (phase speed = U) flexural-gravity wave, i.e. U_cr = min_k c(k).
    Both statements are derived in docs/p338w3_n03_ice_hydro_breakup.md.

The LBM channel measurement (module :mod:`lbm_channel`) discriminates between the two:
it measures the wall-pressure gain and finds the k^1 * coth(kh) law of route (2).

Units: SI throughout (m, s, kg, N, Pa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------------------
# Physical constants (documented, single source of truth)
# --------------------------------------------------------------------------------------
RHO_W = 1025.0      # kg/m^3  sea water
RHO_I = 917.0       # kg/m^3  ice
G = 9.81            # m/s^2


@dataclass(frozen=True)
class CoverProperties:
    """Canonical 2-D ice-cover / channel case definition."""

    h_i: float = 1.0                 # ice cover thickness [m]
    E: float = 5.0e9                 # effective Young's modulus of ice [Pa]
    nu: float = 0.3                  # Poisson ratio
    sigma_f: float = 7.0e5           # flexural (bending) strength [Pa]
    rho_i: float = RHO_I
    rho_w: float = RHO_W
    g: float = G
    h: float = 30.0                  # water depth below the cover [m]
    L: float = 200.0                 # channel (cover) length [m]
    B: float = 1.0                   # unit width [m]

    @property
    def D(self) -> float:
        return plate_rigidity(self.E, self.h_i, self.nu)

    @property
    def mass_per_area(self) -> float:
        return self.rho_i * self.h_i

    @property
    def M_f(self) -> float:
        return crack_moment_capacity(self.sigma_f, self.h_i)


def plate_rigidity(E: float, h_i: float, nu: float) -> float:
    """Flexural rigidity D = E h^3 / (12 (1 - nu^2))  [N m]."""
    return E * h_i**3 / (12.0 * (1.0 - nu**2))


def characteristic_length(D: float, rho_w: float = RHO_W, g: float = G) -> float:
    """Elastic-foundation characteristic length l_c = (D / (rho_w g))^(1/4)  [m]."""
    return (D / (rho_w * g)) ** 0.25


# --------------------------------------------------------------------------------------
# Route (1): prescribed k^2 criterion  ==  momentum-flux (membrane) buckling
# --------------------------------------------------------------------------------------
def critical_velocity_k2(k: float, D: float, rho_w: float = RHO_W, g: float = G) -> float:
    """Mode-wise critical velocity of the prescribed criterion, U_cr(k).

    D k^4 + rho_w g = rho_w U^2 k^2  ->  U = sqrt((D k^4 + rho_w g)/(rho_w k^2)).
    """
    k = np.asarray(k, dtype=float)
    return np.sqrt((D * k**4 + rho_w * g) / (rho_w * k**2))


def u_cr_k2(D: float, rho_w: float = RHO_W, g: float = G) -> tuple[float, float]:
    """Minimum of the prescribed criterion: (U_cr, k_c) = ((4 D g/rho_w)^(1/4), (rho_w g/D)^(1/4))."""
    k_c = (rho_w * g / D) ** 0.25
    u_cr = (4.0 * D * g / rho_w) ** 0.25
    return float(u_cr), float(k_c)


def momentum_flux_compression(U: float, rho_w: float = RHO_W) -> float:
    """Axial compressive line load implied by the prescribed criterion, N = rho_w U^2 [N/m]."""
    return rho_w * U * U


# --------------------------------------------------------------------------------------
# Route (2): flexural-gravity dispersion / steady divergence
# --------------------------------------------------------------------------------------
def flexural_gravity_omega(
    k: np.ndarray,
    D: float,
    h: float,
    rho_w: float = RHO_W,
    g: float = G,
    mass_per_area: float = 0.0,
) -> np.ndarray:
    """Flexural-gravity angular frequency for a floating plate over water of depth h.

        omega^2 = (D k^4 + rho_w g) / ( rho_w / (k tanh(kh)) + m_s )

    Deep-water limit (tanh -> 1) and shallow-water limit (tanh(kh) -> kh) are both
    reproduced (see module docstring).
    """
    k = np.asarray(k, dtype=float)
    kh = np.clip(k * h, 1e-12, None)
    added_mass = rho_w / (k * np.tanh(kh))
    omega = np.sqrt((D * k**4 + rho_w * g) / (added_mass + mass_per_area))
    return omega


def phase_speed(
    k: np.ndarray,
    D: float,
    h: float,
    rho_w: float = RHO_W,
    g: float = G,
    mass_per_area: float = 0.0,
) -> np.ndarray:
    """Phase speed c(k) = omega/k of flexural-gravity waves [m/s]."""
    k = np.asarray(k, dtype=float)
    return flexural_gravity_omega(k, D, h, rho_w, g, mass_per_area) / k


def steady_divergence_velocity(
    k: np.ndarray, D: float, h: float, rho_w: float = RHO_W, g: float = G
) -> np.ndarray:
    """Steady divergence velocity U_B(k) = sqrt((D k^4 + rho_w g)/(rho_w k coth(kh))).

    Identical to :func:`phase_speed` with zero plate mass (analytically the same
    expression); kept separate so the two derivations stay auditable.
    """
    k = np.asarray(k, dtype=float)
    kh = np.clip(k * h, 1e-12, None)
    return np.sqrt((D * k**4 + rho_w * g) / (rho_w * k / np.tanh(kh)))


def u_cr_flexural_gravity(
    D: float,
    h: float,
    rho_w: float = RHO_W,
    g: float = G,
    k_lo: float = 1e-5,
    k_hi: float = 10.0,
) -> tuple[float, float]:
    """Minimum of U_B(k) over the continuous wavenumber range: (U_cr, k_c).

    Deep-water closed form limit: k_c = (rho_w g/(3 D))^(1/4),
    U_cr = sqrt(D k_c^3/rho_w + g/k_c).
    """
    from scipy.optimize import minimize_scalar

    def f(logk: float) -> float:
        k = np.exp(logk)
        return float(steady_divergence_velocity(np.array([k]), D, h, rho_w, g)[0])

    res = minimize_scalar(
        f, bounds=(np.log(k_lo), np.log(k_hi)), method="bounded",
        options={"xatol": 1e-10},
    )
    k_c = float(np.exp(res.x))
    return float(res.fun), k_c


# --------------------------------------------------------------------------------------
# Finite channel: discrete modes
# --------------------------------------------------------------------------------------
def mode_wavenumbers(n_modes: int, L: float, pinned: bool = True) -> np.ndarray:
    """Discrete wavenumbers of a cover pinned (or clamped) at both channel ends."""
    n = np.arange(1, n_modes + 1, dtype=float)
    return n * np.pi / L


def critical_mode(
    k: np.ndarray, velocity_fn: Callable[[np.ndarray], np.ndarray]
) -> tuple[int, float, float]:
    """Index (1-based mode number), wavenumber and velocity of the most critical mode."""
    u = velocity_fn(k)
    i = int(np.argmin(u))
    return i + 1, float(k[i]), float(u[i])


# --------------------------------------------------------------------------------------
# Fracture quantities
# --------------------------------------------------------------------------------------
def crack_moment_capacity(sigma_f: float, h_i: float) -> float:
    """Elastic bending moment capacity per unit width, M_f = sigma_f h^2 / 6  [N m/m]."""
    return sigma_f * h_i**2 / 6.0


def bending_moment_from_curvature(w: np.ndarray, a: float, D: float) -> np.ndarray:
    """Discrete bond moment M_i = D * curvature_i (central second difference)."""
    w = np.asarray(w, dtype=float)
    M = np.zeros_like(w)
    M[1:-1] = D * (w[2:] - 2.0 * w[1:-1] + w[:-2]) / a**2
    return M


def critical_crack_position(L: float, mode: int = 1) -> float:
    """Predicted crack position for the critical (sinusoidal) mode: antinode of |w''|.

    For a pinned-pinned cover the n=1 mode has |w''| maximal at mid-span, so the
    first bending crack is predicted at x = L/2.
    """
    return L / 2.0 if mode == 1 else L / (2.0 * mode)

def antinode_positions(L: float, mode: int) -> list[float]:
    """Positions of the |w''| maxima (crack sites) of a pinned-pinned mode.

    Mode n has n antinodes at x_j = (2j-1) L / (2n), j = 1..n; for n = 1 this
    reduces to the mid-span prediction.
    """
    return [float((2 * j - 1) * L / (2.0 * mode)) for j in range(1, mode + 1)]
