"""Voyage-level energy and CO2 assessment for a Flettner rotor-sail retrofit.

Chain: Weibull wind statistics on a route -> apparent wind per heading ->
rotor thrust assist (section coefficients from the LBM-validated table) ->
main-engine load reduction -> fuel and CO2 saving, net of the rotor drive power.

Reference ship: a 180 m general cargo / coaster type, 14 kn service speed, four
rotors of D = 3 m, H = 18 m on the aft deck.  All numbers are reported with the
inputs recorded so the assessment is reproducible and auditable.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .aero import (
    AIR_DENSITY,
    REFERENCE_TABLE,
    RotorGeometrySpec,
    apparent_wind,
    interpolate_coefficients,
    rotor_sail_thrust,
)

KNOT = 0.514444  # m/s
HFO_CO2_FACTOR = 3.114  # t CO2 / t fuel (IMO, HFO)
LSFO_SFOC = 180.0  # g/kWh, representative two-stroke main engine


@dataclass
class VoyageInputs:
    """Route and ship inputs of the annual assessment."""

    ship_speed_kn: float = 14.0
    route_length_nm: float = 3000.0
    round_trips_per_year: float = 8.0
    weibull_a: float = 8.0  # m/s, scale
    weibull_k: float = 2.0  # shape
    wind_angle_step_deg: float = 30.0
    n_rotors: int = 4
    rotor_diameter: float = 3.0
    rotor_height: float = 18.0
    rotor_omega: float = 12.0  # rad/s (~115 rpm)
    eta_drive: float = 0.92  # motor + gearbox
    eta_propulsion: float = 0.65  # hull + propeller (thrust -> shaft power)
    sfoc_g_per_kwh: float = LSFO_SFOC
    co2_factor: float = HFO_CO2_FACTOR
    table_key: str = "re200_mittal_kumar"


def weibull_pdf(v: np.ndarray, a: float, k: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return (k / a) * (v / a) ** (k - 1.0) * np.exp(-((v / a) ** k))


def weibull_bins(a: float, k: float, v_max: float = 25.0, n: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Discretised Weibull distribution: bin centres and probabilities."""
    edges = np.linspace(0.0, v_max, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    dv = edges[1] - edges[0]
    p = weibull_pdf(centres, a, k) * dv
    return centres, p / p.sum()


def run_voyage_assessment(coeffs_or_table: dict, inputs: VoyageInputs) -> dict:
    """Annual thrust assist, fuel and CO2 saving for a rotor-sail retrofit.

    For each true-wind speed bin and each wind direction (uniform in 0..360 deg
    relative to the ship heading, i.e. an unbiased wind rose), the apparent wind
    is evaluated and the rotor thrust along the heading is integrated.
    """
    table = REFERENCE_TABLE[inputs.table_key]
    geom = RotorGeometrySpec(
        diameter=inputs.rotor_diameter,
        height=inputs.rotor_height,
        n_rotors=inputs.n_rotors,
        spin_ratio=inputs.rotor_omega * 0.5 * inputs.rotor_diameter / (inputs.ship_speed_kn * KNOT),
        omega=inputs.rotor_omega,
    )
    v_ship = inputs.ship_speed_kn * KNOT
    speeds, probs = weibull_bins(inputs.weibull_a, inputs.weibull_k)
    angles = np.arange(0.0, 360.0, inputs.wind_angle_step_deg)

    rows = []
    thrust_sum = 0.0
    power_sum = 0.0
    for v, p in zip(speeds, probs):
        for beta_true in angles:
            v_app, beta_app = apparent_wind(v_ship, float(v), float(beta_true))
            out = rotor_sail_thrust(
                v_app, beta_app, geom, table=table, rho_air=AIR_DENSITY, eta_drive=inputs.eta_drive
            )
            w = float(p) * (len(angles) ** -1)
            thrust_sum += w * out["thrust_N"]
            power_sum += w * out["drive_power_W"]
            rows.append(
                {
                    "v_true": float(v),
                    "beta_true_deg": float(beta_true),
                    "v_app": v_app,
                    "beta_app_deg": beta_app,
                    "thrust_N": out["thrust_N"],
                    "drive_power_W": out["drive_power_W"],
                    "cl": out["cl"],
                    "cd": out["cd"],
                    "alpha": out["alpha"],
                    "feathered": out["feathered"],
                }
            )

    sea_hours = (
        inputs.round_trips_per_year * 2.0 * inputs.route_length_nm / max(inputs.ship_speed_kn, 1e-6)
    )
    mean_thrust = thrust_sum
    mean_drive = power_sum
    propulsive_saving_kw = mean_thrust * v_ship / max(inputs.eta_propulsion, 1e-6) / 1000.0
    drive_kw = mean_drive / 1000.0
    net_kw = propulsive_saving_kw - drive_kw
    fuel_saved_t = net_kw * sea_hours * inputs.sfoc_g_per_kwh / 1e6
    co2_saved_t = fuel_saved_t * inputs.co2_factor
    gross_fuel_t = propulsive_saving_kw * sea_hours * inputs.sfoc_g_per_kwh / 1e6
    gross_co2_t = gross_fuel_t * inputs.co2_factor

    usable = [r for r in rows if r["thrust_N"] > 0.0]
    summary = {
        "sea_hours_per_year": sea_hours,
        "mean_thrust_N": mean_thrust,
        "mean_drive_power_kW": drive_kw,
        "propulsive_power_saving_kW": propulsive_saving_kw,
        "net_power_saving_kW": net_kw,
        "fuel_saved_t_per_year": fuel_saved_t,
        "co2_saved_t_per_year": co2_saved_t,
        "gross_co2_reduction_t_per_year": gross_co2_t,
        "thrust_available_fraction_of_bins": len(usable) / len(rows),
        "mean_thrust_coefficient_equivalent": mean_thrust / (
            0.5
            * AIR_DENSITY
            * v_ship**2
            * inputs.rotor_diameter
            * inputs.rotor_height
            * inputs.n_rotors
        ),
    }
    return {
        "task": "flettner_voyage_assessment",
        "inputs": asdict(inputs),
        "rotor": asdict(geom),
        "coefficient_table": {
            "key": inputs.table_key,
            "source": table["source"],
            "re": table["re"],
            "alpha": table["alpha"],
            "cl": table["cl"],
            "cd": table["cd"],
        },
        "summary": summary,
        "bins": rows,
        "notes": [
            "Two-dimensional section coefficients; no tip-vortex/end-plate downwash model.",
            "Rotor torque coefficient taken from the LBM momentum-exchange torque fit.",
            "Wind rose assumed direction-uniform relative to the ship heading.",
        ],
    }