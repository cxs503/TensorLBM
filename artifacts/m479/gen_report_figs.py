#!/usr/bin/env python3
"""m479: Generate report figures from existing evidence data."""

import json
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
CASE = REPO / "benchmarks" / "verified" / "aircraft_icing_naca0012_rime"
OUT = REPO / "runs" / "m479" / "report_figs"
OUT.mkdir(parents=True, exist_ok=True)


def load_evidence():
    ev = {}
    for f in sorted((CASE / "evidence").iterdir()):
        with open(f) as fh:
            ev[f.name] = json.load(fh)
    return ev


def fig_beta_curve(ev):
    """β distribution along airfoil surface."""
    b1 = ev["b1_prod"]
    beta_data = b1["beta_e_curve"]
    s = np.array(beta_data["s_over_c"])
    beta = np.array(beta_data["beta"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(s, beta, "b-", linewidth=1.5, label="β(s/c)")
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5, linestyle="--", label="Stagnation")
    ax.set_xlabel("s/c (normalized arc length)")
    ax.set_ylabel("Collection efficiency β")
    ax.set_title("NACA 0012 Rime Icing — β Distribution (b1_prod)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.1, 0.65)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_convergence(ev):
    """Grid convergence ladder + Richardson extrapolation."""
    ladder = {}
    for name in ["ladder_L1", "ladder_L2", "ladder_L3", "ladder_L4"]:
        ladder[name] = ev[name]["t_max_mm"]

    grids = [320, 640, 890, 1280]
    t_max = [ladder[f"ladder_L{i+1}"] for i in range(4)]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grids, t_max, "bo-", linewidth=2, markersize=8, label="t_max (mm)")
    ax.axhline(y=10.508, color="r", linestyle="--", linewidth=1, label="Richardson f_ext = 10.508 mm")
    ax.set_xlabel("Grid resolution (nx)")
    ax.set_ylabel("Maximum ice thickness (mm)")
    ax.set_title("NACA 0012 Rime — Grid Convergence Ladder")
    ax.legend()
    ax.grid(True, alpha=0.3)

    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def fig_lwc_linearity(ev):
    """Deposited mass vs LWC linearity."""
    runs = ["m20_lwc025", "b1_prod", "m20_lwc10"]
    lwc_vals = [0.25, 0.50, 1.00]
    dep_vals = [ev[r]["euler_deposited_kg"] for r in runs]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.loglog(lwc_vals, dep_vals, "ro-", linewidth=2, markersize=10, label="Deposited mass")
    # Reference line with slope 1
    ref_dep = dep_vals[1]
    ref_lwc = lwc_vals[1]
    lwc_ref = np.linspace(0.1, 2.0, 100)
    dep_ref = ref_dep * (lwc_ref / ref_lwc)
    ax.loglog(lwc_ref, dep_ref, "k--", linewidth=1, alpha=0.5, label="Slope = 1.0 (linear)")
    ax.set_xlabel("LWC (g/m³)")
    ax.set_ylabel("Deposited mass (kg)")
    ax.set_title("NACA 0012 Rime — LWC Linearity (G9)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    buf = io.BytesIO()
    fig.savefig(buf, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def main():
    ev = load_evidence()

    # Generate figures
    figs = {
        "beta_curve.png": fig_beta_curve(ev),
        "convergence.png": fig_convergence(ev),
        "g9_lwc_linearity.png": fig_lwc_linearity(ev),
    }

    for name, data in figs.items():
        path = OUT / name
        with open(path, "wb") as f:
            f.write(data)
        b64 = base64.b64encode(data).decode()
        print(f"{name}: {len(data)} bytes, base64 {len(b64)} chars")

    print(f"\nFigures saved to {OUT}")
    return figs


if __name__ == "__main__":
    main()
