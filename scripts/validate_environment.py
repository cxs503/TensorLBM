"""Validation script for environment field interface.

Generates validation metrics and figures for the paper-quality report.
"""

import json
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorlbm.environment import (
    LogLawWindProfile,
    UniformCurrentProfile,
    ConservativeTemperatureField,
)
from tensorlbm.environment.wind_profile import WindProfileConfig
from tensorlbm.environment.current_profile import CurrentProfileConfig
from tensorlbm.environment.temperature_field import TemperatureFieldConfig


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def validate_wind_profile():
    """Validate wind profile log law and generate figure."""
    config = WindProfileConfig(
        reference_height=10.0,
        reference_speed=10.0,
        roughness_length=0.0002,  # Open water
    )
    profile = LogLawWindProfile(config)
    
    # Generate profile
    heights = np.linspace(0.5, 50.0, 100)
    speeds = profile.get_profile(heights)
    
    # Analytical log law for comparison
    u_star = 0.41 * config.reference_speed / np.log(config.reference_height / config.roughness_length)
    analytical = (u_star / 0.41) * np.log(heights / config.roughness_length)
    
    # Error metric
    error = np.max(np.abs(speeds - analytical)) / config.reference_speed
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Wind profile
    axes[0].plot(speeds, heights, 'b-', linewidth=2, label='Log Law (implemented)')
    axes[0].plot(analytical, heights, 'r--', linewidth=2, label='Analytical')
    axes[0].axhline(y=config.reference_height, color='g', linestyle=':', label=f'Ref height ({config.reference_height} m)')
    axes[0].set_xlabel('Wind Speed (m/s)', fontsize=12)
    axes[0].set_ylabel('Height (m)', fontsize=12)
    axes[0].set_title('Wind Profile - Logarithmic Law', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim(0, 50)
    
    # Right: Error analysis
    rel_error = np.abs(speeds - analytical) / config.reference_speed * 100
    axes[1].plot(heights, rel_error, 'k-', linewidth=2)
    axes[1].axhline(y=5.0, color='r', linestyle='--', label='5% tolerance')
    axes[1].set_xlabel('Height (m)', fontsize=12)
    axes[1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1].set_title('Log Law Validation Error', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_png = fig_to_base64(fig)
    plt.close(fig)
    
    return {
        "max_relative_error": float(error),
        "is_valid": error < 0.05,
        "figure_base64": fig_png,
    }


def validate_current_profile():
    """Validate current profile uniformity and generate figure."""
    config = CurrentProfileConfig(
        reference_depth=10.0,
        reference_speed=1.0,
    )
    profile = UniformCurrentProfile(config)
    
    # Generate profile
    depths = np.linspace(0, 100, 100)
    speeds = profile.get_profile(depths)
    
    # Error metric
    max_error = np.max(np.abs(speeds - config.reference_speed)) / config.reference_speed
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Current profile
    axes[0].plot(speeds, depths, 'b-', linewidth=2, label='Uniform (implemented)')
    axes[0].axhline(y=config.reference_depth, color='g', linestyle=':', label=f'Ref depth ({config.reference_depth} m)')
    axes[0].set_xlabel('Current Speed (m/s)', fontsize=12)
    axes[0].set_ylabel('Depth (m)', fontsize=12)
    axes[0].set_title('Current Profile - Uniform', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].invert_yaxis()  # Depth increases downward
    
    # Right: Error analysis
    rel_error = np.abs(speeds - config.reference_speed) / config.reference_speed * 100
    axes[1].plot(depths, rel_error, 'k-', linewidth=2)
    axes[1].axhline(y=1.0, color='r', linestyle='--', label='1% tolerance')
    axes[1].set_xlabel('Depth (m)', fontsize=12)
    axes[1].set_ylabel('Relative Error (%)', fontsize=12)
    axes[1].set_title('Uniform Profile Validation Error', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_png = fig_to_base64(fig)
    plt.close(fig)
    
    return {
        "max_relative_error": float(max_error),
        "is_valid": max_error < 0.01,
        "figure_base64": fig_png,
    }


def validate_temperature_field():
    """Validate temperature field conservation and generate figure."""
    config = TemperatureFieldConfig(
        reference_temperature=273.15,
        conservative=True,
    )
    
    # Linear gradient initial condition
    def initial_T(x, y, z):
        return 273.15 + 0.5 * x  # 0.5 K/m gradient
    
    field = ConservativeTemperatureField(
        config=config,
        initial_temperature=initial_T,
    )
    
    # Create test domain
    x = np.linspace(0, 20, 21)
    y = np.linspace(0, 20, 21)
    z = np.array([5.0])
    
    # Get temperature field
    X, Y = np.meshgrid(x, y)
    T = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            T[j, i] = field.get_temperature(x[i], y[j], 5.0, t=0.0)
    
    # Check conservation
    cons = field.check_conservation(x, y, np.array([5.0]), t=0.0)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Temperature field
    im = axes[0].contourf(X, Y, T, levels=20, cmap='coolwarm')
    plt.colorbar(im, ax=axes[0], label='Temperature (K)')
    axes[0].set_xlabel('X (m)', fontsize=12)
    axes[0].set_ylabel('Y (m)', fontsize=12)
    axes[0].set_title('Temperature Field (Conservative)', fontsize=14)
    axes[0].set_aspect('equal')
    
    # Right: Heat flux vectors
    qx_arr = np.zeros_like(X)
    qy_arr = np.zeros_like(X)
    for i in range(len(x)):
        for j in range(len(y)):
            qx, qy, qz = field.get_heat_flux(x[i], y[j], 5.0, t=0.0)
            qx_arr[j, i] = qx
            qy_arr[j, i] = qy
    
    axes[1].quiver(X, Y, qx_arr, qy_arr, scale=0.5)
    axes[1].set_xlabel('X (m)', fontsize=12)
    axes[1].set_ylabel('Y (m)', fontsize=12)
    axes[1].set_title('Heat Flux Vectors', fontsize=14)
    axes[1].set_aspect('equal')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig_png = fig_to_base64(fig)
    plt.close(fig)
    
    return {
        "is_conservative": bool(cons["is_conservative"]),
        "relative_energy_change": float(cons["relative_change"]),
        "is_valid": bool(cons["is_conservative"]),
        "figure_base64": fig_png,
    }


def main():
    """Run all validations and generate report JSON."""
    print("=" * 60)
    print("Environment Field Interface Validation")
    print("=" * 60)
    
    # Validate wind profile
    print("\n[1/3] Validating Wind Profile (Log Law)...")
    wind_result = validate_wind_profile()
    print(f"  Max relative error: {wind_result['max_relative_error']:.6f}")
    print(f"  Valid (< 0.05): {wind_result['is_valid']}")
    
    # Validate current profile
    print("\n[2/3] Validating Current Profile (Uniform)...")
    current_result = validate_current_profile()
    print(f"  Max relative error: {current_result['max_relative_error']:.6f}")
    print(f"  Valid (< 0.01): {current_result['is_valid']}")
    
    # Validate temperature field
    print("\n[3/3] Validating Temperature Field (Conservative)...")
    temp_result = validate_temperature_field()
    print(f"  Is conservative: {temp_result['is_conservative']}")
    print(f"  Relative energy change: {temp_result['relative_energy_change']:.2e}")
    print(f"  Valid: {temp_result['is_valid']}")
    
    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    all_pass = (
        wind_result['is_valid'] and
        current_result['is_valid'] and
        temp_result['is_valid']
    )
    
    print(f"wind_profile_log_law_error: {wind_result['max_relative_error']:.6f} < 0.05 → {'PASS' if wind_result['is_valid'] else 'FAIL'}")
    print(f"current_profile_uniform: {current_result['max_relative_error']:.6f} < 0.01 → {'PASS' if current_result['is_valid'] else 'FAIL'}")
    print(f"temperature_field_conservative: {temp_result['is_conservative']} → {'PASS' if temp_result['is_valid'] else 'FAIL'}")
    print(f"\nOverall: {'ALL PASS' if all_pass else 'SOME FAILED'}")
    
    # Generate report JSON
    report = {
        "mission_id": 673,
        "wind_profile_log_law_error": wind_result['max_relative_error'],
        "wind_profile_valid": wind_result['is_valid'],
        "current_profile_uniform_error": current_result['max_relative_error'],
        "current_profile_valid": current_result['is_valid'],
        "temperature_field_conservative": temp_result['is_conservative'],
        "temperature_field_valid": temp_result['is_valid'],
        "all_criteria_pass": all_pass,
        "figures": [
            {
                "name": "wind_profile_validation",
                "caption": "Wind profile log law validation: implemented vs analytical solution with error analysis",
                "png_base64": wind_result['figure_base64'],
            },
            {
                "name": "current_profile_validation",
                "caption": "Current profile uniform validation: speed profile and error analysis",
                "png_base64": current_result['figure_base64'],
            },
            {
                "name": "temperature_field_validation",
                "caption": "Temperature field conservative validation: field distribution and heat flux vectors",
                "png_base64": temp_result['figure_base64'],
            },
        ],
    }
    
    # Write report JSON
    report_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'm673_validation_report.json')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport saved to: {report_path}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
