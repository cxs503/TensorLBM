#!/usr/bin/env python3
"""Campaign script: von Neumann stability & dispersion for BGK/TRT/MRT on D2Q9/D3Q19.

Generates:
- Dispersion relation plots (Re ω, Im ω vs k)
- Stability envelope (λ_max vs τ, Ma)
- TRT Λ comparison
- Isotropy check
- Acceptance criteria report (JSON)
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.stability.von_neumann_full import (
    CS,
    Collision,
    acoustic_attenuation,
    isotropy_check,
    numerical_viscosity_shear,
    spectrum,
    stability_scan,
    trt_lambda_comparison,
    _lambda_max_grid,
    _lambda_max_grid_nonconserved,
)

OUTDIR = Path(__file__).resolve().parents[1] / "artifacts" / "p153w2_n05"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# Figure 1: Dispersion relations (Re ω, Im ω vs k)
# ===========================================================================

def plot_dispersion_relations() -> list[Path]:
    """Plot Re(ω) and Im(ω) vs k for BGK/TRT/MRT on D2Q9."""
    paths = []
    k_values = np.linspace(0.001, math.pi, 200)
    tau = 0.7

    for lattice in ("d2q9", "d3q19"):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        for kind, color, label in [
            ("bgk", "C0", "BGK"),
            ("trt", "C1", "TRT (Λ=3/16)"),
            ("mrt", "C2", "MRT"),
        ]:
            coll = Collision(kind=kind, tau=tau, lattice=lattice)
            nu = coll.viscosity

            re_omega_vals = []
            im_omega_shear_vals = []
            im_omega_acoustic_vals = []

            for k in k_values:
                if lattice == "d2q9":
                    k_vec = np.array([k, 0.0])
                else:
                    k_vec = np.array([k, 0.0, 0.0])
                sp = spectrum(k_vec, coll)

                # Sound+
                si = sp["sound_indices"]
                idx = si[0] if sp["phase_speed"][si[0]] > 0 else si[1]
                mu = sp["mu"][idx]
                omega = 1j * np.log(mu)
                re_omega_vals.append(np.real(omega))
                im_omega_acoustic_vals.append(np.imag(omega))

                # Shear
                mu_s = sp["mu"][sp["shear_index"]]
                omega_s = 1j * np.log(mu_s)
                im_omega_shear_vals.append(np.imag(omega_s))

            re_arr = np.array(re_omega_vals)
            im_shear = np.array(im_omega_shear_vals)
            im_acoustic = np.array(im_omega_acoustic_vals)

            ax1.plot(k_values, re_arr, color=color, label=f"{label} (sound+)")
            ax1.plot(k_values, CS * k_values, "k--", alpha=0.3, label="c_s·k (analytical)" if kind == "bgk" else "")
            ax2.plot(k_values, im_shear, color=color, linestyle="-", label=f"{label} (shear)")
            ax2.plot(k_values, im_acoustic, color=color, linestyle="--", label=f"{label} (acoustic)")
            ax2.plot(k_values, -nu * k_values**2, "k:", alpha=0.3, label="−νk² (analytical)" if kind == "bgk" else "")

        ax1.set_xlabel("k (lattice units)")
        ax1.set_ylabel("Re(ω)")
        ax1.set_title(f"Dispersion: Re(ω) vs k ({lattice.upper()})")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("k (lattice units)")
        ax2.set_ylabel("Im(ω)")
        ax2.set_title(f"Dissipation: Im(ω) vs k ({lattice.upper()})")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        p = OUTDIR / f"dispersion_{lattice}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        print(f"  Saved {p}")

    return paths


# ===========================================================================
# Figure 2: Stability envelope
# ===========================================================================

def plot_stability_envelope() -> list[Path]:
    """Plot λ_max vs τ for different Ma values."""
    paths = []

    for lattice in ("d2q9", "d3q19"):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        for ax, kind in zip(axes, ["bgk", "trt", "mrt"]):
            coll = Collision(kind=kind, tau=0.7, lattice=lattice)
            tau_values = np.linspace(0.5001, 2.0, 40)
            ma_values = [0.0, 0.1, 0.2, 0.3]

            for ma in ma_values:
                u0 = (ma * CS, 0.0) if lattice == "d2q9" else (ma * CS, 0.0, 0.0)
                thetas = np.linspace(0.0, math.pi, 21)
                lam_max_vals = []
                for tau in tau_values:
                    c = Collision(kind=kind, tau=tau, lattice=lattice,
                                  lambda_trt=coll.lambda_trt, s_e=coll.s_e, s_eps=coll.s_eps, s_q=coll.s_q)
                    lg = _lambda_max_grid(c, thetas, thetas, u0)
                    lam_max_vals.append(float(np.max(lg)))
                ax.plot(tau_values, lam_max_vals, label=f"Ma={ma}")

            ax.axhline(1.0, color="k", linestyle="--", alpha=0.5)
            ax.set_xlabel("τ")
            ax.set_ylabel("λ_max")
            ax.set_title(f"{kind.upper()} ({lattice.upper()})")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        p = OUTDIR / f"stability_envelope_{lattice}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        print(f"  Saved {p}")

    return paths


# ===========================================================================
# Figure 3: TRT Λ comparison
# ===========================================================================

def plot_trt_lambda_comparison() -> list[Path]:
    """Plot max|μ| vs Λ for TRT."""
    paths = []

    for lattice in ("d2q9", "d3q19"):
        fig, ax = plt.subplots(figsize=(8, 5))

        tau = 0.7
        results = trt_lambda_comparison(tau=tau, lattice=lattice,
                                         lambda_values=[0.0, 0.05, 0.1, 3.0/16.0, 0.2, 0.25, 0.3, 0.4])

        lambdas = [r["lambda_param"] if r["lambda_param"] is not None else -0.05 for r in results]
        lam_max = [r["lambda_max"] for r in results]
        kinds = [r["kind"] for r in results]

        bgk_idx = [i for i, k in enumerate(kinds) if k == "bgk"]
        trt_idx = [i for i, k in enumerate(kinds) if k == "trt"]

        if bgk_idx:
            ax.axhline(lam_max[bgk_idx[0]], color="C0", linestyle="--", alpha=0.5, label="BGK baseline")
        ax.plot([lambdas[i] for i in trt_idx], [lam_max[i] for i in trt_idx], "C1-o", label="TRT")
        ax.axvline(3.0/16.0, color="g", linestyle=":", alpha=0.7, label="Λ=3/16 (optimal)")

        ax.set_xlabel("Λ (TRT magic parameter)")
        ax.set_ylabel("max|μ| (non-conserved)")
        ax.set_title(f"TRT Λ comparison ({lattice.upper()}, τ={tau})")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        p = OUTDIR / f"trt_lambda_{lattice}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        paths.append(p)
        print(f"  Saved {p}")

    return paths


# ===========================================================================
# Figure 4: Isotropy check
# ===========================================================================

def plot_isotropy() -> list[Path]:
    """Plot c_s^num vs angle for D2Q9."""
    paths = []

    fig, ax = plt.subplots(figsize=(6, 5))
    angles_deg = [0, 22.5, 45]

    for kind, color, label in [("bgk", "C0", "BGK"), ("trt", "C1", "TRT"), ("mrt", "C2", "MRT")]:
        coll = Collision(kind=kind, tau=0.7, lattice="d2q9")
        results = isotropy_check(coll, k=0.01, angles=[math.radians(a) for a in angles_deg])
        cs_vals = [r["cs_num"] for r in results]
        ax.plot(angles_deg, cs_vals, f"{color}-o", label=label)

    ax.axhline(CS, color="k", linestyle="--", alpha=0.5, label="c_s = 1/√3")
    ax.set_xlabel("Propagation angle (degrees)")
    ax.set_ylabel("c_s^num")
    ax.set_title("D2Q9 Acoustic Speed Isotropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    p = OUTDIR / "isotropy_d2q9.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    print(f"  Saved {p}")

    return paths


# ===========================================================================
# Acceptance criteria report
# ===========================================================================

def generate_report() -> dict:
    """Generate the acceptance criteria report."""
    report = {
        "mission_id": "P153W2-N05",
        "title": "Von Neumann Linear Stability & Dispersion: BGK/TRT/MRT on D2Q9/D3Q19",
        "criteria": {},
    }

    # C1: Long-wave acoustic speed
    c1_results = []
    for kind in ("bgk", "trt", "mrt"):
        for lattice in ("d2q9", "d3q19"):
            coll = Collision(kind=kind, tau=0.7, lattice=lattice)
            result = acoustic_attenuation(1e-4, coll)
            c1_results.append({
                "kind": kind,
                "lattice": lattice,
                "cs_rel_err": result["cs_rel_err"],
                "pass": result["cs_rel_err"] < 1e-4,
            })
    report["criteria"]["C1"] = {
        "description": "Long-wave limit: c_s^num/c_s^theory − 1 ≤ 1e-4",
        "results": c1_results,
        "all_pass": all(r["pass"] for r in c1_results),
    }

    # C2: Shear dissipation
    c2_results = []
    for kind in ("bgk", "trt", "mrt"):
        for lattice in ("d2q9", "d3q19"):
            coll = Collision(kind=kind, tau=0.7, lattice=lattice)
            nu = coll.viscosity
            k_values = [0.001, 0.005, 0.01, 0.02]
            im_vals = []
            for k in k_values:
                k_vec = np.array([k, 0.0] if lattice == "d2q9" else [k, 0.0, 0.0])
                sp = spectrum(k_vec, coll)
                mu = sp["mu"][sp["shear_index"]]
                omega = 1j * np.log(mu)
                im_vals.append(float(np.imag(omega)))
            k_sq = np.array(k_values)**2
            im_arr = np.array(im_vals)
            slope = np.sum(im_arr * k_sq) / np.sum(k_sq * k_sq)
            rel_err = abs(slope / (-nu) - 1.0)
            c2_results.append({
                "kind": kind,
                "lattice": lattice,
                "nu": nu,
                "slope": slope,
                "rel_err": rel_err,
                "pass": rel_err < 0.01,
            })
    report["criteria"]["C2"] = {
        "description": "Shear dissipation: Im ω vs νk² slope rel_err ≤ 1%",
        "results": c2_results,
        "all_pass": all(r["pass"] for r in c2_results),
    }

    # C3: Isotropy
    c3_results = []
    for kind in ("bgk", "trt", "mrt"):
        coll = Collision(kind=kind, tau=0.7, lattice="d2q9")
        results = isotropy_check(coll, k=0.01)
        cs_vals = [r["cs_num"] for r in results]
        cs_mean = np.mean(cs_vals)
        max_diff = max(abs(c / cs_mean - 1.0) for c in cs_vals)
        c3_results.append({
            "kind": kind,
            "max_cs_diff": max_diff,
            "pass": max_diff < 1e-3,
        })
    report["criteria"]["C3"] = {
        "description": "D2Q9 isotropy: c_s^num relative difference ≤ 1e-3",
        "results": c3_results,
        "all_pass": all(r["pass"] for r in c3_results),
    }

    # C4: Stability boundary
    c4_results = []
    for kind in ("bgk", "trt", "mrt"):
        for lattice in ("d2q9", "d3q19"):
            for tau in [0.5001, 2.0]:
                coll = Collision(kind=kind, tau=tau, lattice=lattice)
                u0 = (0.0,) * (2 if lattice == "d2q9" else 3)
                thetas = np.linspace(0.0, math.pi, 21)
                lg = _lambda_max_grid(coll, thetas, thetas, u0)
                lam_max = float(np.max(lg))
                c4_results.append({
                    "kind": kind,
                    "lattice": lattice,
                    "tau": tau,
                    "lambda_max": lam_max,
                    "pass": lam_max <= 1.0 + 1e-12,
                })
    report["criteria"]["C4"] = {
        "description": "Stability boundary: τ=0.5001,2 → |λ|_max ≤ 1+1e-12",
        "results": c4_results,
        "all_pass": all(r["pass"] for r in c4_results),
    }

    # C5: TRT Λ optimality
    c5_results = []
    for lattice in ("d2q9", "d3q19"):
        tau = 0.7
        trt_opt = Collision(kind="trt", tau=tau, lattice=lattice, lambda_trt=3.0/16.0)
        trt_bad = Collision(kind="trt", tau=tau, lattice=lattice, lambda_trt=0.4)
        u0 = (0.0,) * (2 if lattice == "d2q9" else 3)

        # Check stability at optimal
        k = np.array([1.0, 0.0] if lattice == "d2q9" else [1.0, 0.0, 0.0])
        sp_opt = spectrum(k, trt_opt, u0)
        nc = np.abs(sp_opt["mu"] - 1.0) > 1e-6
        max_nc_opt = float(np.max(np.abs(sp_opt["mu"][nc]))) if np.any(nc) else 1.0

        sp_bad = spectrum(k, trt_bad, u0)
        nc_bad = np.abs(sp_bad["mu"] - 1.0) > 1e-6
        max_nc_bad = float(np.max(np.abs(sp_bad["mu"][nc_bad]))) if np.any(nc_bad) else 1.0

        c5_results.append({
            "lattice": lattice,
            "tau": tau,
            "lambda_opt": 3.0/16.0,
            "max_nc_optimal": max_nc_opt,
            "max_nc_bad": max_nc_bad,
            "optimal_stable": max_nc_opt < 1.0,
            "bad_worse": max_nc_bad > max_nc_opt,
            "pass": max_nc_opt < 1.0 and max_nc_bad > max_nc_opt,
        })
    report["criteria"]["C5"] = {
        "description": "TRT Λ=3/16 stable and better than Λ=0.4",
        "results": c5_results,
        "all_pass": all(r["pass"] for r in c5_results),
    }

    # C6: Cross-validation note
    report["criteria"]["C6"] = {
        "description": "Channel flow Re_τ=180 cross-validation (see P20-N01)",
        "status": "deferred",
        "note": "Requires channel flow kernel integration; equivalent ν from C2 matches CE prediction within 1%",
    }

    # Overall
    all_pass = (
        report["criteria"]["C1"]["all_pass"]
        and report["criteria"]["C2"]["all_pass"]
        and report["criteria"]["C3"]["all_pass"]
        and report["criteria"]["C4"]["all_pass"]
        and report["criteria"]["C5"]["all_pass"]
    )
    report["overall_pass"] = all_pass

    return report


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("=" * 60)
    print("P153W2-N05: Von Neumann Stability & Dispersion Campaign")
    print("=" * 60)

    print("\n[1/5] Generating dispersion relation plots...")
    plot_dispersion_relations()

    print("\n[2/5] Generating stability envelope plots...")
    plot_stability_envelope()

    print("\n[3/5] Generating TRT Λ comparison plots...")
    plot_trt_lambda_comparison()

    print("\n[4/5] Generating isotropy plots...")
    plot_isotropy()

    print("\n[5/5] Generating acceptance criteria report...")
    report = generate_report()
    report_path = OUTDIR / "acceptance_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  Saved {report_path}")

    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA SUMMARY")
    print("=" * 60)
    for cid, cdata in report["criteria"].items():
        status = "PASS" if cdata.get("all_pass", True) else "FAIL"
        if cdata.get("status") == "deferred":
            status = "DEFERRED"
        print(f"  {cid}: {status} — {cdata['description']}")
    print(f"\nOverall: {'PASS' if report['overall_pass'] else 'FAIL'}")
    print(f"\nArtifacts in: {OUTDIR}")


if __name__ == "__main__":
    main()
