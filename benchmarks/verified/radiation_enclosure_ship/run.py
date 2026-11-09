"""Ship cabin radiation enclosure benchmark.

Three-surface enclosure: deck (floor), bulkhead (wall), overhead (ceiling).
Solar radiation load on deck surface.

Physical setup:
- Deck: 4m × 6m, ε=0.9, T=300K, solar absorptance=0.6
- Bulkhead: 4m × 3m, ε=0.8, T=295K
- Overhead: 6m × 3m, ε=0.85, T=298K
- Solar irradiance: 800 W/m² from above (45° elevation)

View factors computed analytically and normalized.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from tensorlbm.view_factor import ViewFactorEnclosure
from tensorlbm.thermal_radiation import (
    SurfaceRadiationProps,
    SolarSettings,
    SkyRadiationSettings,
    RadiationEnclosureConfig,
    compute_net_radiation_flux,
    planck_band_radiance,
    STEFAN_BOLTZMANN,
)


def run_benchmark():
    """Run the ship cabin radiation benchmark."""
    
    # Geometry
    deck_area = 4.0 * 6.0  # 24 m²
    bulkhead_area = 4.0 * 3.0  # 12 m²
    overhead_area = 6.0 * 3.0  # 18 m²
    
    # Surface properties
    deck = SurfaceRadiationProps(
        emissivity=0.9,
        solar_absorptance=0.6,
        temperature=300.0,
        area=deck_area,
    )
    bulkhead = SurfaceRadiationProps(
        emissivity=0.8,
        solar_absorptance=0.5,
        temperature=295.0,
        area=bulkhead_area,
    )
    overhead = SurfaceRadiationProps(
        emissivity=0.85,
        solar_absorptance=0.4,
        temperature=298.0,
        area=overhead_area,
    )
    
    surfaces = [deck, bulkhead, overhead]
    areas = [deck_area, bulkhead_area, overhead_area]
    
    # View factors (approximate for rectangular enclosure)
    # For a 3-surface enclosure, we need F_ij such that Σ_j F_ij = 1
    # and A_i F_ij = A_j F_ji
    # Using approximate values for a rectangular room
    F_init = np.array([
        [0.0, 0.4, 0.6],  # deck sees bulkhead and overhead
        [0.5, 0.0, 0.5],  # bulkhead sees deck and overhead
        [0.4, 0.4, 0.0],  # overhead sees deck and bulkhead
    ])
    
    # Normalize
    enc = ViewFactorEnclosure(areas, F_init, normalize=True)
    vf_list = enc.to_list()
    
    print("View factor matrix (normalized):")
    print(f"  F_deck→bulkhead = {vf_list[0][1]:.4f}")
    print(f"  F_deck→overhead = {vf_list[0][2]:.4f}")
    print(f"  F_bulkhead→deck = {vf_list[1][0]:.4f}")
    print(f"  F_bulkhead→overhead = {vf_list[1][2]:.4f}")
    print(f"  F_overhead→deck = {vf_list[2][0]:.4f}")
    print(f"  F_overhead→bulkhead = {vf_list[2][1]:.4f}")
    print(f"  Summation check: {enc.check_summation():.2e}")
    print(f"  Reciprocity check: {enc.check_reciprocity():.2e}")
    
    # Solar settings
    solar = SolarSettings(
        enabled=True,
        irradiance=800.0,  # W/m²
        direction=(0.0, -0.707, -0.707),  # 45° from above
    )
    
    # Sky settings
    sky = SkyRadiationSettings(
        enabled=True,
        T_ambient=295.0,
        T_dewpoint=285.0,
    )
    
    # Surface normals (outward)
    normals = torch.tensor([
        [0.0, -1.0, 0.0],  # deck (facing up, normal is down for enclosure)
        [1.0, 0.0, 0.0],   # bulkhead (facing inward)
        [0.0, 1.0, 0.0],   # overhead (facing down)
    ], dtype=torch.float64)
    
    # Configuration
    cfg = RadiationEnclosureConfig(
        surfaces=surfaces,
        solar=solar,
        sky=sky,
        view_factors=vf_list,
        tol=1e-12,
        max_iter=500,
    )
    
    # Solve
    result = compute_net_radiation_flux(cfg, surface_normals=normals)
    
    print("\n=== Results ===")
    print(f"Solver: {result.solver_method}")
    print(f"Energy conservation residual: {result.energy_conservation_residual:.2e}")
    print()
    
    print("Surface temperatures [K]:")
    for i, name in enumerate(["Deck", "Bulkhead", "Overhead"]):
        print(f"  {name}: {result.surface_temperatures[i]:.1f}")
    print()
    
    print("Net radiation flux [W/m²]:")
    for i, name in enumerate(["Deck", "Bulkhead", "Overhead"]):
        print(f"  {name}: {result.net_flux[i]:.2f}")
    print()
    
    print("Solar flux absorbed [W/m²]:")
    for i, name in enumerate(["Deck", "Bulkhead", "Overhead"]):
        print(f"  {name}: {result.solar_flux[i]:.2f}")
    print()
    
    print("Sky radiation flux [W/m²]:")
    for i, name in enumerate(["Deck", "Bulkhead", "Overhead"]):
        print(f"  {name}: {result.sky_flux[i]:.2f}")
    print()
    
    print("8-12 μm band radiance [W/(m²·sr)]:")
    for i, name in enumerate(["Deck", "Bulkhead", "Overhead"]):
        print(f"  {name}: {result.band_radiance_8_12[i]:.4f}")
    print()
    
    print(f"Total emitted power: {result.total_emitted_power:.2f} W")
    print(f"Total absorbed power: {result.total_absorbed_power:.2f} W")
    
    # Save results
    output = {
        "benchmark": "radiation_enclosure_ship",
        "geometry": {
            "deck_area_m2": deck_area,
            "bulkhead_area_m2": bulkhead_area,
            "overhead_area_m2": overhead_area,
        },
        "view_factors": vf_list,
        "view_factor_checks": {
            "summation_residual": enc.check_summation(),
            "reciprocity_residual": enc.check_reciprocity(),
        },
        "solar_irradiance_W_m2": 800.0,
        "results": {
            "net_flux_W_m2": result.net_flux,
            "solar_flux_W_m2": result.solar_flux,
            "sky_flux_W_m2": result.sky_flux,
            "radiosity_W_m2": result.radiosity,
            "irradiation_W_m2": result.irradiation,
            "band_radiance_8_12_W_m2_sr": result.band_radiance_8_12,
            "surface_temperatures_K": result.surface_temperatures,
            "total_emitted_power_W": result.total_emitted_power,
            "total_absorbed_power_W": result.total_absorbed_power,
            "energy_conservation_residual": result.energy_conservation_residual,
            "solver_method": result.solver_method,
        },
    }
    
    output_path = Path(__file__).parent / "result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    return output


if __name__ == "__main__":
    run_benchmark()
