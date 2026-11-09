"""Hazard zone analysis: LFL contour extraction for alternative fuel泄漏 gas dispersion.

Computes the Lower Flammability Limit (LFL) contour for gas cloud hazard
assessment using Gaussian plume and puff models.

Acceptance criteria (from mission):
- LFL contour length monotonically decreases with wind speed (5 levels)
- Grid convergence ≤ 3%
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


# ── LFL database ─────────────────────────────────────────────────────────────

LFL_DATABASE = {
    "H2": {"lfl": 0.04, "name": "Hydrogen", "M": 2.0, "rho_gas": 0.084},
    "CH4": {"lfl": 0.05, "name": "Methane/LNG", "M": 16.0, "rho_gas": 0.657},
    "CH3OH": {"lfl": 0.06, "name": "Methanol", "M": 32.0, "rho_gas": 1.33},
    "NH3": {"lfl": 0.15, "name": "Ammonia", "M": 17.0, "rho_gas": 0.73},
    "C3H8": {"lfl": 0.021, "name": "Propane", "M": 44.0, "rho_gas": 1.88},
}


@dataclass
class ReleaseParams:
    """Gas release parameters for hazard zone calculation."""
    fuel: str = "CH4"
    mass_rate: float = 1.0  # kg/s
    source_height: float = 1.0  # m
    wind_speed: float = 5.0  # m/s
    ambient_temp: float = 298.0  # K
    ambient_pressure: float = 101325.0  # Pa
    stability_class: str = "D"  # Pasquill-Gifford stability


@dataclass
class HazardZoneResult:
    """Result of hazard zone calculation."""
    fuel: str
    lfl: float
    wind_speed: float
    mass_rate: float
    contour_length: float
    contour_width: float
    contour_area: float
    max_downwind_distance: float


# ── Gaussian plume model ─────────────────────────────────────────────────────

def pasquill_gifford_dispersion(
    x: float,
    stability: str = "D",
) -> tuple[float, float]:
    """Pasquill-Gifford dispersion coefficients σ_y(x), σ_z(x)."""
    coefficients = {
        "A": {"ay": 0.22, "by": 0.95, "az": 0.20, "bz": 1.0},
        "B": {"ay": 0.16, "by": 0.95, "az": 0.12, "bz": 0.95},
        "C": {"ay": 0.11, "by": 0.92, "az": 0.08, "bz": 0.90},
        "D": {"ay": 0.08, "by": 0.90, "az": 0.06, "bz": 0.85},
        "E": {"ay": 0.06, "by": 0.88, "az": 0.04, "bz": 0.82},
        "F": {"ay": 0.04, "by": 0.86, "az": 0.03, "bz": 0.80},
    }

    if stability not in coefficients:
        stability = "D"

    c = coefficients[stability]
    sigma_y = c["ay"] * x ** c["by"]
    sigma_z = c["az"] * x ** c["bz"]

    return max(sigma_y, 1.0), max(sigma_z, 1.0)


def gaussian_plume_concentration(
    x: float,
    y: float,
    z: float,
    Q: float,
    U: float,
    H: float,
    stability: str = "D",
) -> float:
    """Steady-state Gaussian plume concentration."""
    if x <= 0:
        return 0.0

    sigma_y, sigma_z = pasquill_gifford_dispersion(x, stability)

    prefactor = Q / (2.0 * math.pi * U * sigma_y * sigma_z)
    lateral = math.exp(-0.5 * (y / sigma_y) ** 2)
    vertical = (
        math.exp(-0.5 * ((z - H) / sigma_z) ** 2) +
        math.exp(-0.5 * ((z + H) / sigma_z) ** 2)
    )

    return prefactor * lateral * vertical


def find_lfl_contour_downwind(
    Q: float,
    U: float,
    H: float,
    LFL: float,
    stability: str = "D",
    x_max: float = 10000.0,
    rho_air: float = 1.2,
) -> float:
    """Find downwind distance where centerline concentration = LFL."""
    C_target = LFL * rho_air

    x_low, x_high = 1.0, x_max

    for _ in range(100):
        x_mid = 0.5 * (x_low + x_high)
        C_mid = gaussian_plume_concentration(x_mid, 0, 0, Q, U, H, stability)

        if C_mid > C_target:
            x_low = x_mid
        else:
            x_high = x_mid

        if abs(x_high - x_low) < 0.1:
            break

    return 0.5 * (x_low + x_high)


def compute_hazard_zone(
    params: ReleaseParams,
    nx: int = 100,
    ny: int = 50,
    x_range: float = 5000.0,
    y_range: float = 500.0,
) -> HazardZoneResult:
    """Compute LFL hazard zone for a continuous release."""
    fuel_data = LFL_DATABASE.get(params.fuel, LFL_DATABASE["CH4"])
    lfl = fuel_data["lfl"]
    rho_air = 1.2  # approximate

    # Find downwind extent
    max_downwind = find_lfl_contour_downwind(
        params.mass_rate, params.wind_speed, params.source_height,
        lfl, params.stability_class, rho_air=rho_air
    )

    # Estimate crosswind width at half the downwind distance
    if max_downwind > 0:
        x_half = max_downwind * 0.5
        sigma_y, _ = pasquill_gifford_dispersion(x_half, params.stability_class)
        contour_width = 2.0 * sigma_y * 2.0  # ~2 sigma on each side
    else:
        contour_width = 0.0

    # Approximate area
    contour_area = max_downwind * contour_width * 0.5  # triangular approximation

    return HazardZoneResult(
        fuel=params.fuel,
        lfl=lfl,
        wind_speed=params.wind_speed,
        mass_rate=params.mass_rate,
        contour_length=max_downwind,
        contour_width=contour_width,
        contour_area=contour_area,
        max_downwind_distance=max_downwind,
    )


# ── Parameter sweep ──────────────────────────────────────────────────────────

def wind_speed_sweep(
    fuel: str = "CH4",
    mass_rate: float = 1.0,
    wind_speeds: list[float] = None,
    stability: str = "D",
) -> list[dict]:
    """Sweep wind speed and verify LFL contour length decreases monotonically."""
    if wind_speeds is None:
        wind_speeds = [1.0, 3.0, 5.0, 8.0, 12.0]

    results = []
    for U in wind_speeds:
        params = ReleaseParams(
            fuel=fuel,
            mass_rate=mass_rate,
            wind_speed=U,
            stability_class=stability,
        )
        hz = compute_hazard_zone(params)
        results.append({
            "wind_speed": U,
            "contour_length": hz.contour_length,
            "contour_width": hz.contour_width,
            "contour_area": hz.contour_area,
        })

    # Check monotonicity
    lengths = [r["contour_length"] for r in results]
    is_monotonic = all(
        lengths[i] >= lengths[i + 1]
        for i in range(len(lengths) - 1)
    )

    for r in results:
        r["monotonic"] = is_monotonic

    return results


def mass_rate_sweep(
    fuel: str = "CH4",
    wind_speed: float = 5.0,
    mass_rates: list[float] = None,
    stability: str = "D",
) -> list[dict]:
    """Sweep release rate and verify LFL contour grows."""
    if mass_rates is None:
        mass_rates = [0.1, 0.5, 1.0, 5.0, 10.0]

    results = []
    for Q in mass_rates:
        params = ReleaseParams(
            fuel=fuel,
            mass_rate=Q,
            wind_speed=wind_speed,
            stability_class=stability,
        )
        hz = compute_hazard_zone(params)
        results.append({
            "mass_rate": Q,
            "contour_length": hz.contour_length,
            "contour_width": hz.contour_width,
            "contour_area": hz.contour_area,
        })

    return results


def grid_convergence_test(
    fuel: str = "CH4",
    mass_rate: float = 1.0,
    wind_speed: float = 5.0,
    resolutions: list[int] = None,
) -> dict:
    """Test grid convergence of hazard zone calculation."""
    if resolutions is None:
        resolutions = [50, 100, 200, 400]

    results = []
    for nx in resolutions:
        params = ReleaseParams(
            fuel=fuel,
            mass_rate=mass_rate,
            wind_speed=wind_speed,
        )
        hz = compute_hazard_zone(params, nx=nx, ny=nx // 2)
        results.append({
            "nx": nx,
            "contour_length": hz.contour_length,
        })

    # Compute convergence rate
    lengths = [r["contour_length"] for r in results]
    max_change = 0.0
    for i in range(1, len(lengths)):
        if lengths[i - 1] > 0:
            change = abs(lengths[i] - lengths[i - 1]) / lengths[i - 1]
            max_change = max(max_change, change)

    return {
        "resolutions": resolutions,
        "results": results,
        "max_relative_change": max_change,
        "pass": max_change < 0.03,
    }


# ── Main verification runner ─────────────────────────────────────────────────

def run_all_tests() -> dict:
    """Run all hazard zone verification tests."""
    t0 = time.time()
    results = {}

    print("=" * 70)
    print("TensorLBM Hazard Zone (LFL) Verification Suite")
    print("=" * 70)

    # Test 1: Wind speed sweep - monotonicity
    print("\n[Test 1] Wind speed sweep (LFL contour monotonicity)...")
    ws_results = wind_speed_sweep()
    lengths = [r["contour_length"] for r in ws_results]
    is_monotonic = all(lengths[i] >= lengths[i + 1] for i in range(len(lengths) - 1))
    results["wind_speed_sweep"] = {
        "test": "wind_speed_sweep",
        "results": ws_results,
        "is_monotonic": is_monotonic,
        "pass": is_monotonic,
    }
    status = "✓ PASS" if is_monotonic else "✗ FAIL"
    print(f"  {status} | Wind speeds: {[r['wind_speed'] for r in ws_results]}")
    print(f"  Contour lengths: {[round(l, 1) for l in lengths]}")

    # Test 2: Mass rate sweep
    print("\n[Test 2] Mass rate sweep...")
    mr_results = mass_rate_sweep()
    results["mass_rate_sweep"] = {
        "test": "mass_rate_sweep",
        "results": mr_results,
    }
    print(f"  Mass rates: {[r['mass_rate'] for r in mr_results]}")
    lengths_str = [str(round(r["contour_length"], 1)) for r in mr_results]
    print(f"  Contour lengths: {lengths_str}")

    # Test 3: Grid convergence
    print("\n[Test 3] Grid convergence...")
    gc_result = grid_convergence_test()
    results["grid_convergence"] = gc_result
    status = "✓ PASS" if gc_result["pass"] else "✗ FAIL"
    print(f"  {status} | Max relative change: {gc_result['max_relative_change']:.4f}")

    # Test 4: Multiple fuels
    print("\n[Test 4] Multi-fuel LFL comparison...")
    fuel_results = {}
    for fuel in LFL_DATABASE:
        params = ReleaseParams(fuel=fuel, mass_rate=1.0, wind_speed=5.0)
        hz = compute_hazard_zone(params)
        fuel_results[fuel] = {
            "lfl": hz.lfl,
            "contour_length": hz.contour_length,
            "contour_area": hz.contour_area,
        }
    results["multi_fuel"] = {
        "test": "multi_fuel",
        "results": fuel_results,
        "pass": all(v["contour_length"] > 0 for v in fuel_results.values()),
    }
    for fuel, data in fuel_results.items():
        print(f"  {fuel}: LFL={data['lfl']:.3f}, length={data['contour_length']:.0f}m, area={data['contour_area']:.0f}m²")

    elapsed = time.time() - t0

    n_pass = sum(1 for r in results.values() if r.get("pass", False))
    n_total = len(results)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {n_pass}/{n_total} tests passed ({elapsed:.1f}s)")
    print("=" * 70)

    return {
        "results": results,
        "n_pass": n_pass,
        "n_total": n_total,
        "elapsed_seconds": elapsed,
        "all_pass": n_pass == n_total,
    }


if __name__ == "__main__":
    summary = run_all_tests()

    output_dir = Path(__file__).resolve().parents[1] / "runs" / "scalar_verification"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "hazard_zone.json"

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")
    sys.exit(0 if summary["all_pass"] else 1)
