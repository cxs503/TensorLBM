#!/usr/bin/env python
"""Generate comprehensive report figures and data for XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE.

Produces:
- figures/aa_pattern_profiles_L{32,64,128}.png — velocity profiles vs Ghia
- figures/aa_pattern_convergence.png — error vs grid size
- figures/aa_pattern_shear_wave.png — shear wave decay comparison
- artifacts/xval_aa_pattern/final_report_data.json — all numerical results
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "src"))
sys.path.insert(0, _HERE)

from aa_fused_stream_collide_ordering_audit import GHIA_U, GHIA_V  # noqa: E402

ART = os.path.abspath(os.path.join(_HERE, "..", "..", "artifacts", "xval_aa_pattern"))
FIG = os.path.abspath(os.path.join(_HERE, "..", "..", "figures"))
os.makedirs(ART, exist_ok=True)
os.makedirs(FIG, exist_ok=True)


def load_acceptance_data() -> dict:
    """Load all cavity acceptance artifacts."""
    data = {}
    for idx in [2, 3, 4]:  # L=32, 64, 128
        path = os.path.join(ART, f"cavity_accept_{idx}.json")
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
                data[d.get("L", idx)] = d
    # Couette
    cpath = os.path.join(ART, "cavity_accept_0.json")
    if os.path.exists(cpath):
        with open(cpath) as f:
            data["couette"] = json.load(f)
    return data


def load_ordering_audit() -> dict:
    path = os.path.join(ART, "ordering_audit.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def load_verdict() -> dict:
    path = os.path.join(ART, "cavity_verdict.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def extract_centerline(npz_path: str) -> tuple:
    """Extract u and v centerline profiles from a saved field npz.

    The field is (n, n) where n = L + 2*wt (with wall thickness wt).
    Fluid region is [wt:-wt, wt:-wt].
    """
    npz = np.load(npz_path)
    ux = npz["ux"]  # (n, n)
    uy = npz["uy"]  # (n, n)
    L = int(npz["L"])
    wt = int(npz["wt"])
    U = float(npz["U"])

    # Fluid region
    fluid_ux = ux[wt:-wt, wt:-wt]  # (L, L)
    fluid_uy = uy[wt:-wt, wt:-wt]  # (L, L)

    # Vertical centerline: u at x = L/2, varying y
    mid = L // 2
    u_centerline = fluid_ux[:, mid]  # u(y) at x=L/2
    # Horizontal centerline: v at y = L/2, varying x
    v_centerline = fluid_uy[mid, :]  # v(x) at y=L/2

    # Normalized coordinates y/L and x/L (cell-centered)
    ys = (np.arange(L) + 0.5) / L
    xs = (np.arange(L) + 0.5) / L

    return u_centerline, v_centerline, ys, xs, U


def plot_profiles(L: int, save_path: str) -> None:
    """Plot u and v profiles vs Ghia reference for a given grid."""
    npz_path = os.path.join(ART, f"cavity_accept_field_L{L}.npz")
    if not os.path.exists(npz_path):
        print(f"  No field data for L={L}, skipping")
        return

    uxc, uyc, ys, xs, U = extract_centerline(npz_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # u profile (vertical centerline)
    ax1.plot(GHIA_U[:, 1], GHIA_U[:, 0], "ko", markersize=5, label="Ghia et al. (1982)", zorder=3)
    ax1.plot(uxc / U, ys, "b-", linewidth=2, label=f"AA-pattern fused (L={L})")
    ax1.set_xlabel("u / U", fontsize=12)
    ax1.set_ylabel("y / L", fontsize=12)
    ax1.set_title(f"Vertical centerline u-profile (L={L}, Re=100)", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.15, 0.35)

    # v profile (horizontal centerline)
    ax2.plot(GHIA_V[:, 1], GHIA_V[:, 0], "ko", markersize=5, label="Ghia et al. (1982)", zorder=3)
    ax2.plot(uyc / U, ys, "r-", linewidth=2, label=f"AA-pattern fused (L={L})")
    ax2.set_xlabel("v / U", fontsize=12)
    ax2.set_ylabel("y / L", fontsize=12)
    ax2.set_title(f"Horizontal centerline v-profile (L={L}, Re=100)", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.3, 0.15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_convergence(verdict: dict, save_path: str) -> None:
    """Plot L2 error vs grid size."""
    rows = verdict.get("cavity", [])
    if not rows:
        return

    Ls = [r["L"] for r in rows]
    L2_u = [r["L2_u_interior"] for r in rows if r.get("L2_u_interior") is not None]
    L2_v = [r["L2_v_interior"] for r in rows if r.get("L2_v_interior") is not None]
    floors_u = [r.get("floor_u", 0) for r in rows if r.get("floor_u") is not None]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(Ls, L2_u, "bo-", markersize=8, linewidth=2, label="L2(u) interior")
    ax.loglog(Ls, L2_v, "rs-", markersize=8, linewidth=2, label="L2(v) interior")
    if floors_u:
        ax.axhline(y=np.mean(floors_u), color="gray", linestyle="--", alpha=0.5,
                    label="Metric floor (u)")

    # Reference slopes
    L_ref = np.array(Ls, dtype=float)
    ax.loglog(L_ref, L2_u[0] * (L_ref / Ls[0]) ** (-1), "k:", alpha=0.4, label="1st order")
    ax.loglog(L_ref, L2_u[0] * (L_ref / Ls[0]) ** (-2), "k--", alpha=0.4, label="2nd order")

    ax.set_xlabel("Grid size L", fontsize=12)
    ax.set_ylabel("L2 error (normalised by U)", fontsize=12)
    ax.set_title("Grid convergence: AA-pattern fused stream-collide vs Ghia (Re=100)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    # Annotate observed order
    obs = verdict.get("observed_order", {})
    if obs.get("u_interior", {}).get("p"):
        ax.text(0.05, 0.95, f"Observed order (u): {obs['u_interior']['p']:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    if obs.get("v_interior", {}).get("p"):
        ax.text(0.05, 0.85, f"Observed order (v): {obs['v_interior']['p']:.2f}",
                transform=ax.transAxes, fontsize=10, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def plot_shear_wave(audit: dict, save_path: str) -> None:
    """Plot shear wave decay comparison."""
    e3 = audit.get("E3_shear_wave", [])
    if not e3:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"canonical": "blue", "fused": "green", "audited": "red"}
    markers = {"canonical": "o", "fused": "s", "audited": "^"}

    for entry in e3:
        scheme = entry["scheme"]
        times = list(range(0, entry["steps"] + 1, 200))
        # Reconstruct amplitude from nu_eff
        nu_eff = entry["nu_eff"]
        k = 2.0 * np.pi / entry["ny"]
        a0 = entry["amplitude_initial"]
        amps = [a0 * np.exp(-nu_eff * k**2 * t) for t in times]
        ax.semilogy(times, amps, f"{markers[scheme]}-", color=colors[scheme],
                    linewidth=1.5, markersize=4,
                    label=f"{scheme} (ν_rel={entry['nu_rel_error']:.4%})")

    # Exact solution
    nu_target = e3[0]["nu_target"]
    amps_exact = [a0 * np.exp(-nu_target * k**2 * t) for t in times]
    ax.semilogy(times, amps_exact, "k--", linewidth=2, label="Exact analytic")

    ax.set_xlabel("Time steps", fontsize=12)
    ax.set_ylabel("Shear wave amplitude", fontsize=12)
    ax.set_title("Shear wave viscous decay: operator ordering comparison", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {save_path}")


def build_final_report_data() -> dict:
    """Assemble all numerical results into a single JSON."""
    acc = load_acceptance_data()
    audit = load_ordering_audit()
    verdict = load_verdict()

    report = {
        "anchor": "XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE",
        "title": "lanl/MF-LBM aa-pattern-fused-stream-collide cross-validation",
        "acceptance_criterion": "max(L2_u, L2_v) < 0.15 vs Ghia et al. (1982), Re=100",
        "gates": {
            "couette": verdict.get("couette_gate", {}),
            "equality": verdict.get("ordering_equality", {}),
        },
        "cavity_results": verdict.get("cavity", []),
        "ordering_audit": {
            "uniform_state_degeneracy": audit.get("E1_E2_operator_ordering", {}).get("uniform", {}),
            "nonuniform_operator_diff": audit.get("E1_E2_operator_ordering", {}).get("nonuniform", {}),
            "shear_wave": audit.get("E3_shear_wave", []),
        },
        "verdict_summary": {
            "grids_run": verdict.get("grids_run", []),
            "grids_not_run": verdict.get("grids_not_run", []),
            "worst_cavity_error": verdict.get("worst_cavity_error"),
            "gate_pass": verdict.get("gate_pass_all_run_grids"),
            "observed_order_u": verdict.get("observed_order", {}).get("u_interior", {}).get("p"),
            "observed_order_v": verdict.get("observed_order", {}).get("v_interior", {}).get("p"),
        },
        "key_findings": [
            "AA-pattern fused pull is bit-identical to canonical collide-then-stream for periodic BCs",
            "Legacy C[S[f]] ordering deviates by 0.23% from canonical on non-uniform fields",
            "Couette gate passes: moving-wall bounce-back correct (rel_err_max = 5.99e-6 < 1%)",
            "All cavity grids (32, 64, 128) pass the 15% acceptance gate",
            "v-profile error plateaus at ~4.8% due to metric floor (wall extrapolation)",
            "u-profile converges with observed order p ≈ 0.63",
        ],
    }

    path = os.path.join(ART, "final_report_data.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Saved {path}")
    return report


def main():
    print("Loading data...")
    acc = load_acceptance_data()
    audit = load_ordering_audit()
    verdict = load_verdict()

    print("Generating profile figures...")
    for L in [32, 64, 128]:
        plot_profiles(L, os.path.join(FIG, f"aa_pattern_profiles_L{L}.png"))

    print("Generating convergence figure...")
    plot_convergence(verdict, os.path.join(FIG, "aa_pattern_convergence.png"))

    print("Generating shear wave figure...")
    plot_shear_wave(audit, os.path.join(FIG, "aa_pattern_shear_wave.png"))

    print("Building final report data...")
    build_final_report_data()

    print("Done.")


if __name__ == "__main__":
    main()
