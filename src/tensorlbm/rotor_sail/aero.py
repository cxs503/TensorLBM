"""Flettner rotor-sail aerodynamics: coefficient handling, ship inflow, thrust.

This module is the *engineering layer* on top of the LBM solver in
:mod:`tensorlbm.rotor_sail.lbm2d`.  It contains

* a small, explicitly-sourced table of measured/computed rotating-cylinder
  force coefficients used for interpolation (``REFERENCE_TABLE``);
* the ship-side inflow model: a deck-level turbulent boundary layer profile and
  a propeller-race augmentation/deficit, both expressed as ``u(z)/V_ship``;
* the spanwise (strip) integration of rotor thrust compared with the standard
  engineering "area-mean velocity" formula -- the difference is the inflow
  non-uniformity (interference) error that a rotor-sail retrofit assessment can
  incur;
* the apparent-wind thrust decomposition used by the voyage model.

All formulas are two-dimensional section aerodynamics: the rotor is treated as
a finite-span circular cylinder whose sectional coefficients vary along the
span through the local inflow speed (no tip-vortex downwash model), which is the
standard assumption of rotor-sail performance codes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

AIR_DENSITY = 1.225  # kg/m^3, 15 degC, 101325 Pa


# --------------------------------------------------------------------------- #
# Reference coefficient table
# --------------------------------------------------------------------------- #
#: Rotating-cylinder lift/drag coefficients used as the engineering
#: interpolation basis.  ``source`` documents the provenance; ``re`` is the
#: Reynolds number of the dataset.  The laminar row set is the M&K benchmark
#: used by the LBM validation (Mittal & Kumar, *J. Fluid Mech.* 476 (2003)
#: 303-334, Re = 200, 2D DNS), transcribed from the paper's coefficient plots
#: (digitisation uncertainty ~2-3 %).  The high-Reynolds row set follows the
#: smooth-cylinder wind-tunnel trend summarised by Seifert (2012, *Prog.
#: Aerosp. Sci.* 52:1-22) and Prandtl (1925, *Naturwissenschaften* 13:93-108).
REFERENCE_TABLE: dict[str, dict] = {
    "re200_mittal_kumar": {
        "re": 200.0,
        "source": "Mittal & Kumar (2003) JFM 476:303, 2D DNS, digitised",
        "alpha": [0.0, 1.0, 2.0, 3.0, 4.0],
        "cl": [0.0, 1.30, 2.29, 3.15, 3.95],
        "cd": [1.31, 0.70, -0.05, -0.92, -1.75],
    },
    "re1e5_experiment": {
        "re": 1.0e5,
        "source": "Prandtl (1925) / Seifert (2012) review, smooth cylinder",
        "alpha": [0.0, 1.0, 2.0, 3.0, 4.0],
        "cl": [0.0, 1.10, 2.00, 2.70, 3.30],
        "cd": [1.20, 0.55, 0.00, -0.60, -1.15],
    },
}


def interpolate_coefficients(
    alpha: float | np.ndarray, table: dict = REFERENCE_TABLE["re200_mittal_kumar"]
) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation (clamped) of ``C_L``, ``C_D`` in the spin ratio."""
    a = np.atleast_1d(np.asarray(alpha, dtype=float))
    cl = np.interp(a, table["alpha"], table["cl"])
    cd = np.interp(a, table["alpha"], table["cd"])
    return cl, cd


def power_coefficient(alpha: float | np.ndarray, cq: float) -> np.ndarray:
    """Rotor drive power coefficient ``C_P = C_Q * alpha`` (``P = C_P 0.5 rho V^3 A``)."""
    return np.asarray(cq, dtype=float) * np.atleast_1d(np.asarray(alpha, dtype=float))


# --------------------------------------------------------------------------- #
# Ship-side inflow models
# --------------------------------------------------------------------------- #
def deck_boundary_layer_profile(
    ny: int,
    delta: float,
    z0: float = 0.0,
    exponent: float = 1.0 / 7.0,
) -> list[float]:
    """Turbulent boundary-layer profile ``u(z)/U = ((z+z0)/delta)^(1/n)``.

    ``z`` is measured from the deck surface (row 0 of the LBM channel) and the
    profile is clipped at 1 for ``z >= delta``.  ``z0`` shifts the origin so a
    non-zero velocity can be imposed at the wall row (the LBM channel has no
    solid deck, it is a free-stream far field).
    """
    z = np.arange(ny, dtype=float)
    ratio = np.clip((z + z0) / max(delta, 1e-9), 0.0, 1.0)
    with np.errstate(invalid="ignore"):
        prof = np.where(ratio > 0.0, ratio**exponent, 0.0)
    return prof.tolist()


def propeller_race_profile(
    ny: int,
    jet_ratio: float,
    race_radius_rows: float,
    centre_row: float,
    decay: float = 0.35,
) -> np.ndarray:
    """Propeller-race augmentation ``u(z)/U = 1 + (V_j/V - 1) exp(-r^2/2 s^2)``.

    ``jet_ratio = V_j / V_ship`` is the axial jet velocity ratio from actuator
    disk theory, ``race_radius_rows`` the Gaussian half-width of the race.
    """
    z = np.arange(ny, dtype=float)
    r = z - centre_row
    return 1.0 + (jet_ratio - 1.0) * (1.0 - decay) * np.exp(-0.5 * (r / max(race_radius_rows, 1e-9)) ** 2)


def combined_wake_profile(
    ny: int,
    delta_rows: float,
    jet_ratio: float = 1.0,
    race_radius_rows: float = 1.0,
    race_centre_row: float | None = None,
    exponent: float = 1.0 / 7.0,
) -> np.ndarray:
    """Deck boundary layer multiplied by the propeller-race augmentation."""
    bl = np.asarray(deck_boundary_layer_profile(ny, delta=delta_rows, z0=0.5, exponent=exponent))
    if jet_ratio == 1.0:
        return bl
    centre = 0.0 if race_centre_row is None else race_centre_row
    race = propeller_race_profile(ny, jet_ratio, race_radius_rows, centre)
    return bl * race


# --------------------------------------------------------------------------- #
# Spanwise integration vs area-mean engineering formula
# --------------------------------------------------------------------------- #
@dataclass
class RotorGeometrySpec:
    """Physical rotor dimensions (SI)."""

    diameter: float = 3.0  # m
    height: float = 18.0  # m
    n_rotors: int = 4
    spin_ratio: float = 2.5
    omega: float = 12.0  # rad/s (fitted to spin_ratio at the design ship speed)


def strip_thrust(
    u_over_vship: np.ndarray,
    v_ship: float,
    geom: RotorGeometrySpec,
    table: dict = REFERENCE_TABLE["re200_mittal_kumar"],
    rho_air: float = AIR_DENSITY,
) -> dict:
    """Quasi-3D thrust by integrating 2D section coefficients over the span.

    At each spanwise station the local inflow ``u_i = u(z_i)`` sets the local
    spin ratio ``alpha_i = omega R / u_i``; the local section force is
    ``0.5 rho u_i^2 C_L(alpha_i) D dz`` (thrust = lift, wind normal to the
    rotor axis and to the ship centre-plane).
    """
    prof = np.asarray(u_over_vship, dtype=float)
    n = prof.size
    dz = geom.height / n
    u = prof * v_ship
    alpha = np.where(u > 1e-6, geom.omega * 0.5 * geom.diameter / np.maximum(u, 1e-6), 0.0)
    cl, cd = interpolate_coefficients(alpha, table)
    q = 0.5 * rho_air * u**2 * geom.diameter * dz
    thrust_strip = float(np.sum(q * cl)) * geom.n_rotors
    side_strip = float(np.sum(q * cd)) * geom.n_rotors
    mean_u = float(np.mean(u))
    alpha_mean = geom.omega * 0.5 * geom.diameter / mean_u
    cl_m, cd_m = interpolate_coefficients(np.array([alpha_mean]), table)
    area = geom.diameter * geom.height
    thrust_mean = float(0.5 * rho_air * mean_u**2 * area * cl_m[0]) * geom.n_rotors
    side_mean = float(0.5 * rho_air * mean_u**2 * area * cd_m[0]) * geom.n_rotors
    return {
        "thrust_strip_N": thrust_strip,
        "side_force_strip_N": side_strip,
        "thrust_mean_velocity_N": thrust_mean,
        "side_force_mean_velocity_N": side_mean,
        "alpha_local_min": float(np.min(alpha)),
        "alpha_local_max": float(np.max(alpha)),
        "alpha_mean": float(alpha_mean),
        "u_mean": mean_u,
        "u_min": float(np.min(u)),
        "u_max": float(np.max(u)),
        "relative_difference": (thrust_strip - thrust_mean) / thrust_mean if thrust_mean else float("nan"),
    }


def uniform_inflow_thrust(
    v_ref: float,
    geom: RotorGeometrySpec,
    table: dict = REFERENCE_TABLE["re200_mittal_kumar"],
    rho_air: float = AIR_DENSITY,
) -> dict:
    """Thrust with a uniform inflow at ``v_ref`` (undisturbed reference case)."""
    alpha = geom.omega * 0.5 * geom.diameter / v_ref
    cl, cd = interpolate_coefficients(np.array([alpha]), table)
    area = geom.diameter * geom.height
    q = 0.5 * rho_air * v_ref**2 * area
    return {
        "thrust_N": float(q * cl[0]) * geom.n_rotors,
        "side_force_N": float(q * cd[0]) * geom.n_rotors,
        "alpha": float(alpha),
        "cl": float(cl[0]),
        "cd": float(cd[0]),
    }


# --------------------------------------------------------------------------- #
# Apparent-wind thrust decomposition
# --------------------------------------------------------------------------- #
def apparent_wind(
    v_ship: float, v_true_wind: float, wind_angle_rel_ship_deg: float
) -> tuple[float, float]:
    """Apparent wind speed (m/s) and angle (deg, 0 = head wind)."""
    beta = math.radians(wind_angle_rel_ship_deg)
    # true wind vector components in the ship frame: x' along the ship heading
    wx = v_true_wind * math.cos(beta)
    wy = v_true_wind * math.sin(beta)
    ax = wx - v_ship  # wind relative to the ship: -v_ship along the heading
    speed = math.hypot(ax, wy)
    angle = math.degrees(math.atan2(abs(wy), -ax)) if speed > 1e-9 else 0.0
    return speed, angle


def rotor_sail_thrust(
    v_app: float,
    beta_deg: float,
    geom: RotorGeometrySpec,
    table: dict = REFERENCE_TABLE["re200_mittal_kumar"],
    rho_air: float = AIR_DENSITY,
    eta_drive: float = 0.92,
) -> dict:
    """Thrust along the ship heading and rotor drive power for apparent wind.

    ``beta_deg`` is the apparent-wind angle measured from the ship heading
    (0 = head wind).  The aerodynamic force is decomposed into lift (normal to
    the apparent wind) and drag (along it); the useful thrust along the ship
    heading is ``T = L sin(beta) - D cos(beta)``.  The rotor is assumed to be
    feathered (spun down) when that quantity is negative.
    """
    if v_app <= 1e-6:
        return {
            "thrust_N": 0.0, "drive_power_W": 0.0, "cl": 0.0, "cd": 0.0,
            "alpha": 0.0, "beta_deg": beta_deg, "feathered": True,
        }
    alpha = geom.omega * 0.5 * geom.diameter / v_app
    cl, cd = interpolate_coefficients(np.array([alpha]), table)
    area = geom.diameter * geom.height
    q = 0.5 * rho_air * v_app**2 * area
    lift = q * cl[0] * geom.n_rotors
    drag = q * cd[0] * geom.n_rotors
    b = math.radians(beta_deg)
    thrust = lift * math.sin(b) - drag * math.cos(b)
    cq = float(cd[0]) * 0.15  # torque coefficient from the 2D LBM fit, see docs
    power = (0.5 * rho_air * v_app**3 * area * cq * alpha * geom.n_rotors) / max(eta_drive, 1e-6)
    feathered = thrust <= 0.0
    if feathered:
        thrust, power = 0.0, 0.0
    return {
        "thrust_N": float(thrust),
        "drive_power_W": float(power),
        "cl": float(cl[0]),
        "cd": float(cd[0]),
        "alpha": float(alpha),
        "beta_deg": float(beta_deg),
        "feathered": bool(feathered),
    }