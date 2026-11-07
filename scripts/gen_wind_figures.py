"""Generate figures for the wind-resistance campaign report."""
import base64
import io
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm.wind_resistance import (
    CONTAINER_SHIP, TANKER, BULK_CARRIER, TEST_SHIP,
    blendermann_CX, blendermann_CY, blendermann_CN,
    WindResistanceEngine, voyage_assessment, default_wind_rose,
    lbm_aerodynamic_coefficients,
)


def plot_coefficient_comparison():
    """Plot C_X, C_Y, C_N vs wind angle for all ship types."""
    angles = np.arange(0, 181, 5)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ship_types = [
        ("container", "Container Ship", CONTAINER_SHIP),
        ("tanker", "Tanker", TANKER),
        ("bulk_carrier", "Bulk Carrier", BULK_CARRIER),
    ]

    for ax, (st, label, geo) in zip(axes, ship_types):
        cx = [blendermann_CX(a, st) for a in angles]
        cy = [blendermann_CY(a, st) for a in angles]
        cn = [blendermann_CN(a, st) for a in angles]

        ax.plot(angles, cx, 'b-', label='C_X', linewidth=2)
        ax.plot(angles, cy, 'r-', label='C_Y', linewidth=2)
        ax.plot(angles, cn, 'g-', label='C_N', linewidth=2)
        ax.set_xlabel('Wind Angle ψ (°)')
        ax.set_ylabel('Coefficient')
        ax.set_title(f'{label}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linewidth=0.5)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_lbm_vs_empirical():
    """Plot LBM vs empirical coefficients for test ship."""
    angles = [0, 30, 60, 90, 120, 150, 180]
    emp_cx = [blendermann_CX(a, "container") for a in angles]

    # Run LBM at low resolution for comparison
    lbm_cx = []
    for a in angles:
        result = lbm_aerodynamic_coefficients(
            TEST_SHIP, a, wind_speed=10.0,
            resolution=12, max_steps=50,
        )
        lbm_cx.append(result["C_X"])

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(angles))
    width = 0.35

    ax.bar(x - width/2, emp_cx, width, label='Empirical (Blendermann)', color='steelblue')
    ax.bar(x + width/2, lbm_cx, width, label='LBM D3Q19/MRT (res=12)', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{a}°' for a in angles])
    ax.set_ylabel('C_X')
    ax.set_xlabel('Wind Angle')
    ax.set_title('Longitudinal Wind Coefficient: Empirical vs LBM')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def plot_voyage_assessment():
    """Plot voyage assessment results."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Wind rose
    rose = default_wind_rose()
    angles = [b.angle_centre for b in rose]
    probs = [b.probability for b in rose]

    ax1 = axes[0]
    ax1.bar(angles, probs, width=25, color='steelblue', alpha=0.7)
    ax1.set_xlabel('Wind Direction (°)')
    ax1.set_ylabel('Probability')
    ax1.set_title('North Atlantic Wind Rose')
    ax1.set_xticks(angles)
    ax1.set_xticklabels([f'{a}°' for a in angles], rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Voyage results for all ship types
    ax2 = axes[1]
    ship_types = ["container", "tanker", "bulk_carrier"]
    labels = ["Container", "Tanker", "Bulk Carrier"]
    drag_fracs = []
    thrust_fracs = []

    for st in ship_types:
        engine = WindResistanceEngine(ship_type=st, complexity_threshold=2.0)
        result = voyage_assessment(st, voyage_hours=8760.0, engine=engine)
        drag_fracs.append(result["drag_fraction"])
        thrust_fracs.append(result["thrust_fraction"])

    x = np.arange(len(labels))
    width = 0.35
    ax2.bar(x - width/2, drag_fracs, width, label='Drag', color='coral')
    ax2.bar(x + width/2, thrust_fracs, width, label='Thrust', color='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel('Fraction of Voyage Time')
    ax2.set_title('Annual Wind Assistance (Empirical)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


if __name__ == "__main__":
    print("Generating coefficient comparison plot...")
    fig1 = plot_coefficient_comparison()
    print(f"  Size: {len(fig1)} bytes")

    print("Generating LBM vs empirical plot...")
    fig2 = plot_lbm_vs_empirical()
    print(f"  Size: {len(fig2)} bytes")

    print("Generating voyage assessment plot...")
    fig3 = plot_voyage_assessment()
    print(f"  Size: {len(fig3)} bytes")

    # Save to files
    import os
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/coeff_comparison.png", "wb") as f:
        f.write(base64.b64decode(fig1))
    with open("artifacts/lbm_vs_empirical.png", "wb") as f:
        f.write(base64.b64decode(fig2))
    with open("artifacts/voyage_assessment.png", "wb") as f:
        f.write(base64.b64decode(fig3))

    print("Figures saved to artifacts/")
