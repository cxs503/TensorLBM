"""Pedestrian wind comfort assessment for TensorLBM.

Implements the Lawson LDDC comfort criteria and the NEN 8100 / Davenport
exceedance-probability approach for outdoor pedestrian environments.

The module ingests a statistical characterisation of the wind field
(mean speed + turbulence intensity, or a time-series) and returns:
- Comfort class per sensor point (A–E or Lawson categories)
- Exceedance probability (fraction of time U > threshold)
- Recommended mitigation zones

Reference thresholds follow:
  Lawson LDDC (1990 rev.)  – sitting / standing / walking / running
  NEN 8100:2006            – Dutch standard (A = comfortable, D/E = dangerous)
  CIBSE AM14 (2015)        – Building effects comfort guide
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# Lawson LDDC criteria  (m/s)
LAWSON_THRESHOLDS = {
    "A_sitting":  2.5,
    "B_standing": 4.0,
    "C_walking":  6.0,
    "D_running":  8.0,
}

# NEN 8100 exceedance-probability limits
NEN8100_THRESHOLDS = {
    "A": {"speed": 5.0,  "max_exceed": 0.05},
    "B": {"speed": 5.0,  "max_exceed": 0.10},
    "C": {"speed": 5.0,  "max_exceed": 0.20},
    "D": {"speed": 10.0, "max_exceed": 0.05},
    "E": {"speed": 10.0, "max_exceed": 0.10},
}

ComfortClass = Literal["A", "B", "C", "D", "E", "F_dangerous"]


@dataclass
class WindSensorPoint:
    """Wind statistics at a single sensor/probe location."""
    label: str
    x: float
    y: float
    z: float = 1.5            # pedestrian height (m)
    mean_speed: float = 0.0   # m/s
    turbulence_intensity: float = 0.1   # I_u = σ_u / U_mean
    # Weibull parameters for speed distribution (optional)
    weibull_k: float = 2.0    # shape
    weibull_c: float | None = None  # scale (defaults to 2*mean/sqrt(π))


@dataclass
class WindComfortResult:
    """Wind comfort assessment at a sensor point."""
    label: str
    x: float
    y: float
    z: float
    mean_speed: float
    effective_gust_speed: float    # U_eff = U_mean * (1 + g * I_u), g≈3.5
    exceedance_5ms: float          # P(U > 5 m/s)
    exceedance_10ms: float         # P(U > 10 m/s)
    lawson_category: str
    nen8100_class: ComfortClass
    is_comfortable: bool
    mitigation_suggested: bool
    notes: str = ""


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def _weibull_exceedance(u_threshold: float, k: float, c: float) -> float:
    """P(U > u_threshold) from Weibull(k, c) distribution."""
    if c <= 0 or k <= 0:
        return 0.0
    return math.exp(-((u_threshold / c) ** k))


def _gumbel_exceedance(u_threshold: float, mu: float, sigma: float) -> float:
    """P(U > u_threshold) from Gumbel extreme-value distribution."""
    beta = sigma * math.sqrt(6) / math.pi
    gamma_e = 0.5772  # Euler–Mascheroni
    mode = mu - beta * gamma_e
    z = (u_threshold - mode) / beta
    return 1.0 - math.exp(-math.exp(-z))


def _lawson_category(u_eff: float) -> str:
    if u_eff <= LAWSON_THRESHOLDS["A_sitting"]:
        return "A_sitting"
    if u_eff <= LAWSON_THRESHOLDS["B_standing"]:
        return "B_standing"
    if u_eff <= LAWSON_THRESHOLDS["C_walking"]:
        return "C_walking"
    if u_eff <= LAWSON_THRESHOLDS["D_running"]:
        return "D_running"
    return "E_dangerous"


def _nen8100_class(exceed_5: float, exceed_10: float) -> ComfortClass:
    """Classify according to NEN 8100 using worst-applicable class."""
    if exceed_5 <= 0.05:
        return "A"
    if exceed_5 <= 0.10:
        return "B"
    if exceed_5 <= 0.20:
        return "C"
    if exceed_10 <= 0.05:
        return "D"
    if exceed_10 <= 0.10:
        return "E"
    return "F_dangerous"


# ---------------------------------------------------------------------------
# Core assessment
# ---------------------------------------------------------------------------

def assess_wind_comfort(
    sensors: list[WindSensorPoint],
    gust_factor: float = 3.5,
    comfort_threshold_class: ComfortClass = "C",
    reference_code: Literal["lawson", "nen8100", "both"] = "both",
) -> list[WindComfortResult]:
    """Assess pedestrian wind comfort at a list of sensor points.

    Parameters
    ----------
    sensors : list[WindSensorPoint]
        Probe locations with wind statistics.
    gust_factor : float
        Peak factor g for gust calculation  (U_eff = U_mean*(1 + g*I_u)).
    comfort_threshold_class : str
        NEN 8100 class threshold for is_comfortable flag.
    reference_code : str
        Which standard to apply.

    Returns
    -------
    list[WindComfortResult]
    """
    class_order = ["A", "B", "C", "D", "E", "F_dangerous"]
    threshold_idx = class_order.index(comfort_threshold_class) if comfort_threshold_class in class_order else 2

    results: list[WindComfortResult] = []

    for s in sensors:
        U = s.mean_speed
        Iu = s.turbulence_intensity
        U_eff = U * (1.0 + gust_factor * Iu)

        # Weibull scale default
        c = s.weibull_c if s.weibull_c is not None else (
            U * 2.0 / math.sqrt(math.pi) if U > 0 else 1.0
        )
        k = s.weibull_k

        exceed_5 = _weibull_exceedance(5.0, k, c)
        exceed_10 = _weibull_exceedance(10.0, k, c)

        lawson_cat = _lawson_category(U_eff)
        nen_class = _nen8100_class(exceed_5, exceed_10)

        is_ok = class_order.index(nen_class) <= threshold_idx if nen_class in class_order else False
        suggest_mitigation = not is_ok or lawson_cat in ("D_running", "E_dangerous")

        notes_parts = []
        if s.z < 1.0:
            notes_parts.append("sensor below pedestrian height")
        if U_eff > 15.0:
            notes_parts.append("extreme gust speed – structural risk")
        if lawson_cat == "E_dangerous":
            notes_parts.append("Lawson dangerous: consider windbreaks or canopies")

        results.append(WindComfortResult(
            label=s.label,
            x=s.x,
            y=s.y,
            z=s.z,
            mean_speed=U,
            effective_gust_speed=U_eff,
            exceedance_5ms=exceed_5,
            exceedance_10ms=exceed_10,
            lawson_category=lawson_cat,
            nen8100_class=nen_class,
            is_comfortable=is_ok,
            mitigation_suggested=suggest_mitigation,
            notes="; ".join(notes_parts),
        ))

    return results


# ---------------------------------------------------------------------------
# Aggregate summary
# ---------------------------------------------------------------------------

def wind_comfort_summary(results: list[WindComfortResult]) -> dict:
    """Return a summary dict suitable for JSON response."""
    n = len(results)
    if n == 0:
        return {"n_sensors": 0}

    class_order = ["A", "B", "C", "D", "E", "F_dangerous"]
    class_counts: dict[str, int] = {c: 0 for c in class_order}
    for r in results:
        c = r.nen8100_class
        class_counts[c] = class_counts.get(c, 0) + 1

    worst = max(results, key=lambda r: class_order.index(r.nen8100_class) if r.nen8100_class in class_order else 0)
    n_uncomfortable = sum(1 for r in results if not r.is_comfortable)

    return {
        "n_sensors": n,
        "n_uncomfortable": n_uncomfortable,
        "comfort_fraction": (n - n_uncomfortable) / n,
        "worst_point": worst.label,
        "worst_class": worst.nen8100_class,
        "worst_gust_speed_ms": worst.effective_gust_speed,
        "class_distribution": class_counts,
        "sensors": [
            {
                "label": r.label,
                "x": r.x,
                "y": r.y,
                "mean_speed_ms": r.mean_speed,
                "gust_speed_ms": r.effective_gust_speed,
                "exceed_5ms": r.exceedance_5ms,
                "exceed_10ms": r.exceedance_10ms,
                "lawson": r.lawson_category,
                "nen8100": r.nen8100_class,
                "comfortable": r.is_comfortable,
                "mitigation": r.mitigation_suggested,
                "notes": r.notes,
            }
            for r in results
        ],
    }
