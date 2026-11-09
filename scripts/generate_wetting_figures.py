"""Generate wetting-dynamics figures for the campaign report."""

from __future__ import annotations

import base64
import io
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.contact_line.wetting_dynamics import (
    DEG, RAD, GAMMA_WATER, G_SI, MU_WATER, RHO_WATER,
    cassie_baxter_angle, wenzel_angle, jurin_height,
    tanner_law_radius, mkt_contact_line_velocity, mkt_dynamic_angle,
    MKTParams,
)

FIGURES = {}


def _encode(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def fig1_cassie_baxter_wenzel():
    """Figure 1: Cassie-Baxter and Wenzel apparent contact angles."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Cassie-Baxter
    ax = axes[0]
    theta_Y_vals = np.linspace(10, 170, 100)
    for f_s, ls, label in [(0.3, "--", "f_s=0.3"), (0.5, "-", "f_s=0.5"), (0.7, "-.", "f_s=0.7")]:
        theta_cb = [cassie_baxter_angle(t * DEG, f_s) * RAD for t in theta_Y_vals]
        ax.plot(theta_Y_vals, theta_cb, ls, label=label)
    ax.plot(theta_Y_vals, theta_Y_vals, "k:", alpha=0.5, label="Young's angle")
    ax.set_xlabel("Young's angle θ_Y (°)")
    ax.set_ylabel("Apparent angle θ_CB (°)")
    ax.set_title("Cassie-Baxter Apparent Contact Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Wenzel
    ax = axes[1]
    for r, ls, label in [(1.2, "--", "r=1.2"), (1.5, "-", "r=1.5"), (2.0, "-.", "r=2.0")]:
        theta_w = []
        for t in theta_Y_vals:
            cos_w = r * math.cos(t * DEG)
            cos_w = max(-1.0, min(1.0, cos_w))
            theta_w.append(math.acos(cos_w) * RAD)
        ax.plot(theta_Y_vals, theta_w, ls, label=label)
    ax.plot(theta_Y_vals, theta_Y_vals, "k:", alpha=0.5, label="Young's angle")
    ax.set_xlabel("Young's angle θ_Y (°)")
    ax.set_ylabel("Apparent angle θ_W (°)")
    ax.set_title("Wenzel Apparent Contact Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def fig2_capillary_rise():
    """Figure 2: Capillary rise vs tube radius (Jurin's law)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    R_vals = np.logspace(-4, -2, 100)  # 0.1 mm to 10 mm
    for theta_deg, label in [(0, "θ=0°"), (30, "θ=30°"), (60, "θ=60°"), (90, "θ=90°")]:
        h_vals = [jurin_height(GAMMA_WATER, t * DEG, RHO_WATER, G_SI, R) for t in [theta_deg] for R in R_vals]
        ax.loglog(R_vals * 1000, [h * 1000 for h in h_vals], label=label)

    ax.set_xlabel("Tube radius R (mm)")
    ax.set_ylabel("Capillary rise h (mm)")
    ax.set_title("Jurin's Law: Capillary Rise vs Tube Radius")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def fig3_droplet_spreading():
    """Figure 3: Droplet spreading radius vs time (Tanner's law)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    R_0 = 1.0e-3  # 1 mm
    theta_0 = 150 * DEG
    theta_e = 30 * DEG
    t_vals = np.logspace(-2, 2, 100)  # 0.01 s to 100 s

    for alpha, label in [(0.1, "α=0.1 (Tanner)"), (0.15, "α=0.15"), (0.05, "α=0.05")]:
        R_vals = [tanner_law_radius(R_0, theta_0, theta_e, GAMMA_WATER, MU_WATER, t, alpha) for t in t_vals]
        ax.loglog(t_vals, [R * 1000 for R in R_vals], label=label)

    ax.axhline(y=R_0 * 1000, color="k", linestyle=":", alpha=0.5, label="R₀")
    ax.set_xlabel("Time t (s)")
    ax.set_ylabel("Spreading radius R (mm)")
    ax.set_title("Tanner's Law: Droplet Spreading Radius vs Time")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def fig4_cox_voinov():
    """Figure 4: Cox-Voinov dynamic contact angle vs capillary number."""
    fig, ax = plt.subplots(figsize=(8, 5))

    Ca_vals = np.logspace(-4, -0.5, 100)
    theta_e = 60 * DEG
    L = 1e-3
    lm = 1e-9
    K = 9 * math.log(L / lm)

    theta_d_vals = []
    for Ca in Ca_vals:
        theta_d3 = theta_e**3 + K * Ca
        theta_d = theta_d3 ** (1.0 / 3.0)
        theta_d_vals.append(theta_d * RAD)

    ax.loglog(Ca_vals, theta_d_vals, "b-", label="Cox-Voinov")
    ax.axhline(y=60, color="k", linestyle="--", alpha=0.5, label="θ_e = 60°")
    ax.set_xlabel("Capillary number Ca")
    ax.set_ylabel("Dynamic angle θ_d (°)")
    ax.set_title("Cox-Voinov Law: Dynamic Contact Angle vs Ca")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    return fig


def fig5_mkt():
    """Figure 5: MKT contact-line velocity vs dynamic angle."""
    fig, ax = plt.subplots(figsize=(8, 5))

    params = MKTParams(theta_0=60 * DEG)
    theta_d_vals = np.linspace(10, 170, 200)
    v_vals = []
    for t in theta_d_vals:
        try:
            v = mkt_contact_line_velocity(t * DEG, params)
            v_vals.append(abs(v))
        except (ValueError, OverflowError):
            v_vals.append(float("nan"))

    ax.semilogy(theta_d_vals, v_vals, "b-")
    ax.axvline(x=60, color="k", linestyle="--", alpha=0.5, label="θ_e = 60°")
    ax.set_xlabel("Dynamic angle θ_d (°)")
    ax.set_ylabel("|U_cl| (m/s)")
    ax.set_title("MKT: Contact-Line Velocity vs Dynamic Angle")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def generate_all():
    """Generate all figures and return as dict of {name: base64_png}."""
    generators = {
        "cassie_baxter_wenzel": fig1_cassie_baxter_wenzel,
        "capillary_rise": fig2_capillary_rise,
        "droplet_spreading": fig3_droplet_spreading,
        "cox_voinov": fig4_cox_voinov,
        "mkt_velocity": fig5_mkt,
    }
    figures = {}
    for name, gen in generators.items():
        fig = gen()
        figures[name] = _encode(fig)
        plt.close(fig)
    return figures


if __name__ == "__main__":
    figures = generate_all()
    print(f"Generated {len(figures)} figures")
    for name in figures:
        print(f"  {name}: {len(figures[name])} bytes (base64)")
