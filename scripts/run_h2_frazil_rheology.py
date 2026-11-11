#!/usr/bin/env python3
"""H2-FRAZIL-RHEOLOGY verification: frazil power-law suspension rheology.

Runs the D3Q19 shear cell with particles at various volume fractions,
measures effective viscosity, and compares against:
  1. Einstein dilute limit (phi=0.05): eta_r = 1 + 2.5*phi
  2. Krieger-Dougherty (phi=0.20): eta_r = (1 - phi/0.6)^(-1.5)
  3. Shear rate sweep curve

Produces:
  - artifacts/h2_frazil_rheology/campaign.json
  - artifacts/h2_frazil_rheology/eta_r_vs_phi.png
  - artifacts/h2_frazil_rheology/shear_sweep.png
  - docs/H2_FRAZIL_RHEOLOGY.md
"""

from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import torch
torch.set_num_threads(2)

from tensorlbm.frazil_powerlaw_shear import (
    pure_case, run_shear_case, fit_power_law,
    place_particles_random, build_obstacle_mask,
)
from tensorlbm.frazil_powerlaw_mature import (
    einstein_relative_viscosity,
    einstein_batchelor_green,
    krieger_dougherty,
    hydrodynamic_volume_fraction,
)

# ---------------------------------------------------------------------------
# Configuration (small domain for fast verification)
# ---------------------------------------------------------------------------
NU = 0.2
RADIUS = 2.5  # particle radius in lattice units

# Small domain for fast turnaround
NX, NY, NZ = 20, 20, 24

# Shear rate ladder
GAMMA_RATES = [1e-3, 2e-3, 4e-3]

# Volume fractions
PHI_VALUES = [0.0, 0.05, 0.10, 0.20]

# Fast simulation parameters
N_STEPS = 800
N_WARMUP = 400
SAMPLE_EVERY = 10

ARTIFACT_DIR = os.path.join(ROOT, "artifacts", "h2_frazil_rheology")


def gamma_to_uwall(gamma_dot: float) -> float:
    H = NZ - 2
    return gamma_dot * H / 2.0


def run_pure_cases() -> list[dict]:
    """Run pure fluid at each shear rate."""
    results = []
    for gi, gamma in enumerate(GAMMA_RATES):
        u_wall = gamma_to_uwall(gamma)
        rec = pure_case(
            nx=NX, ny=NY, nz=NZ, nu=NU, u_wall=u_wall,
            n_steps=N_STEPS, n_warmup=N_WARMUP, sample_every=SAMPLE_EVERY,
        )
        rec["gamma_target"] = gamma
        rec["rate_index"] = gi
        rec["family"] = "pure"
        rec["phi"] = 0.0
        results.append(rec)
        print(f"  pure gamma={gamma:.2e}: gamma_meas={rec['gamma_dot']:.2e} "
              f"R2={rec['profile_r2']:.4f}")
    return results


def run_suspension_cases(phi: float, seed: int = 42) -> list[dict]:
    """Run suspension at given volume fraction."""
    results = []
    for gi, gamma in enumerate(GAMMA_RATES):
        u_wall = gamma_to_uwall(gamma)
        rec = run_shear_case(
            phi=phi, u_wall=u_wall,
            nx=NX, ny=NY, nz=NZ, radius=RADIUS, nu=NU,
            n_primary=1, seed=seed + gi,
            n_steps=N_STEPS, n_warmup=N_WARMUP, sample_every=SAMPLE_EVERY,
            wall_margin=3, placement_attempts=10000,
        )
        rec["gamma_target"] = gamma
        rec["rate_index"] = gi
        rec["family"] = "dispersed"
        rec["phi_declared"] = phi
        results.append(rec)
        print(f"  phi={phi:.2f} gamma={gamma:.2e}: gamma_meas={rec['gamma_dot']:.2e} "
              f"phi_lat={rec['phi_lattice']:.4f} n_part={rec['n_particles_placed']}")
    return results


def compute_eta_r(pure_results: list[dict], susp_results: list[dict]) -> list[dict]:
    """Compute relative viscosity using shear rate ratio method."""
    out = []
    for r in susp_results:
        gamma_target = r.get("gamma_target", 0)
        gamma_meas = r["gamma_dot"]
        
        # Find matching pure case
        pure_match = None
        for p in pure_results:
            if abs(p.get("gamma_target", 0) - gamma_target) < 1e-10:
                pure_match = p
                break
        
        if pure_match and pure_match["gamma_dot"] > 0 and gamma_meas > 0:
            # eta_r = gamma_pure / gamma_susp at same wall velocity
            eta_r = pure_match["gamma_dot"] / gamma_meas
        else:
            eta_r = float("nan")
        
        out.append({
            "phi_declared": r["phi_declared"],
            "phi_lattice": r["phi_lattice"],
            "gamma_target": gamma_target,
            "gamma_meas": gamma_meas,
            "eta_r": eta_r,
            "n_particles": r.get("n_particles_placed", 0),
            "profile_r2": r.get("profile_r2", 0),
        })
    
    return out


def evaluate_criteria(eta_r_data: list[dict]) -> dict:
    """Evaluate acceptance criteria."""
    criteria = {}
    
    by_phi = {}
    for d in eta_r_data:
        by_phi.setdefault(d["phi_declared"], []).append(d)
    
    # Criterion 1: Einstein dilute limit (phi=0.05)
    if 0.05 in by_phi:
        eta_r_vals = [d["eta_r"] for d in by_phi[0.05] if np.isfinite(d["eta_r"])]
        if eta_r_vals:
            eta_r_mean = np.mean(eta_r_vals)
            eta_r_einstein = einstein_relative_viscosity(0.05)
            rel_err = abs(eta_r_mean - eta_r_einstein) / eta_r_einstein
            criteria["c1_einstein_dilute"] = {
                "phi": 0.05,
                "eta_r_meas": float(eta_r_mean),
                "eta_r_einstein": float(eta_r_einstein),
                "rel_err": float(rel_err),
                "threshold": 0.10,
                "passed": bool(rel_err < 0.10),
            }
    
    # Criterion 2: Krieger-Dougherty (phi=0.20)
    if 0.20 in by_phi:
        eta_r_vals = [d["eta_r"] for d in by_phi[0.20] if np.isfinite(d["eta_r"])]
        if eta_r_vals:
            eta_r_mean = np.mean(eta_r_vals)
            eta_r_kd = krieger_dougherty(0.20)
            rel_err = abs(eta_r_mean - eta_r_kd) / eta_r_kd
            criteria["c2_krieger_dougherty"] = {
                "phi": 0.20,
                "eta_r_meas": float(eta_r_mean),
                "eta_r_kd": float(eta_r_kd),
                "rel_err": float(rel_err),
                "threshold": 0.10,
                "passed": bool(rel_err < 0.10),
            }
    
    n_pass = sum(1 for c in criteria.values() if c.get("passed", False))
    n_defined = len(criteria)
    
    return {
        "criteria": criteria,
        "n_pass": n_pass,
        "n_defined": n_defined,
        "all_passed": bool(n_pass == n_defined and n_defined > 0),
    }


def generate_figures(eta_r_data: list[dict], pure_results: list[dict],
                     criteria: dict) -> list[str]:
    """Generate matplotlib figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping figures")
        return []
    
    figures = []
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    # Figure 1: eta_r vs phi
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    phi_vals = sorted(set(d["phi_declared"] for d in eta_r_data if d["phi_declared"] > 0))
    eta_r_means = []
    eta_r_stds = []
    
    for phi in phi_vals:
        vals = [d["eta_r"] for d in eta_r_data
                if d["phi_declared"] == phi and np.isfinite(d["eta_r"])]
        if vals:
            eta_r_means.append(np.mean(vals))
            eta_r_stds.append(np.std(vals))
        else:
            eta_r_means.append(float("nan"))
            eta_r_stds.append(0)
    
    ax.errorbar(phi_vals, eta_r_means, yerr=eta_r_stds, fmt='o-',
                label='LBM shear cell', capsize=3, markersize=8)
    
    phi_fine = np.linspace(0, 0.35, 100)
    eta_einstein = [einstein_relative_viscosity(p) for p in phi_fine]
    eta_ebg = [einstein_batchelor_green(p) for p in phi_fine]
    eta_kd = [krieger_dougherty(p) for p in phi_fine]
    
    ax.plot(phi_fine, eta_einstein, '--', label='Einstein (1906)', alpha=0.7)
    ax.plot(phi_fine, eta_ebg, '-.', label='Einstein-Batchelor-Green', alpha=0.7)
    ax.plot(phi_fine, eta_kd, '--', label='Krieger-Dougherty', alpha=0.7)
    
    ax.set_xlabel('Volume fraction $\\phi$', fontsize=12)
    ax.set_ylabel('Relative viscosity $\\eta_r$', fontsize=12)
    ax.set_title('H2-FRAZIL-RHEOLOGY: Relative Viscosity vs Volume Fraction', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.01, 0.35)
    ax.set_ylim(0.8, max(3.5, max([x for x in eta_r_means if np.isfinite(x)] + [3.0]) * 1.2))
    
    for key, crit in criteria.get("criteria", {}).items():
        phi = crit.get("phi", 0)
        eta_meas = crit.get("eta_r_meas", 0)
        passed = crit.get("passed", False)
        marker = 'PASS' if passed else 'FAIL'
        color = 'green' if passed else 'red'
        ax.annotate(f'{marker}\nerr={crit.get("rel_err", 0):.1%}',
                   xy=(phi, eta_meas),
                   xytext=(phi + 0.04, eta_meas + 0.15),
                   fontsize=8, color=color, fontweight='bold')
    
    fig.tight_layout()
    path1 = os.path.join(ARTIFACT_DIR, "eta_r_vs_phi.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    figures.append(path1)
    print(f"  Figure saved: {path1}")
    
    # Figure 2: Shear rate sweep
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for phi in sorted(set(d["phi_declared"] for d in eta_r_data)):
        pts = [d for d in eta_r_data if d["phi_declared"] == phi]
        gammas = [d["gamma_meas"] for d in pts if d["gamma_meas"] > 0]
        taus = []
        for d in pts:
            if np.isfinite(d["eta_r"]) and d["gamma_meas"] > 0:
                taus.append(d["eta_r"] * NU * d["gamma_meas"])
            else:
                taus.append(float("nan"))
        
        label = f'$\\phi$={phi:.2f}' if phi > 0 else 'pure'
        valid = [(g, t) for g, t in zip(gammas, taus) if np.isfinite(t) and t > 0]
        if valid:
            g, t = zip(*valid)
            ax.loglog(g, t, 'o-', label=label, markersize=6)
    
    ax.set_xlabel('Shear rate $\\dot{\\gamma}$ [lattice units]', fontsize=12)
    ax.set_ylabel('Shear stress $\\tau$ [lattice units]', fontsize=12)
    ax.set_title('Flow Curves: Frazil Suspension Shear Sweep', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    
    fig.tight_layout()
    path2 = os.path.join(ARTIFACT_DIR, "shear_sweep.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    figures.append(path2)
    print(f"  Figure saved: {path2}")
    
    return figures


def generate_report(
    pure_results: list[dict],
    eta_r_data: list[dict],
    criteria: dict,
    figures: list[str],
    elapsed: float,
) -> str:
    """Generate markdown report."""
    os.makedirs(os.path.join(ROOT, "docs"), exist_ok=True)
    
    lines = []
    lines.append("# H2-FRAZIL-RHEOLOGY: Frazil Power-Law Suspension Rheology")
    lines.append("")
    lines.append("## 1. Case Introduction")
    lines.append("")
    lines.append("**Physical background**: Frazil ice is a suspension of small ice crystals")
    lines.append("in supercooled water. The effective viscosity of the slurry depends on the")
    lines.append("crystal volume fraction through well-known rheological closures.")
    lines.append("")
    lines.append("**Objective**: Verify that the D3Q19 LBM shear cell correctly captures the")
    lines.append("concentration-dependent viscosity increase, comparing against:")
    lines.append("")
    lines.append("- **Einstein (1906)**: $\\eta_r = 1 + 2.5\\phi$ (dilute, $\\phi \\lesssim 0.05$)")
    lines.append("- **Krieger-Dougherty (1959)**: $\\eta_r = (1 - \\phi/\\phi_{max})^{-[\\eta]\\phi_{max}}$")
    lines.append("")
    lines.append("### Parameters")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|---|---|")
    lines.append(f"| Kinematic viscosity $\\nu$ | {NU} |")
    lines.append(f"| Domain | {NX}×{NY}×{NZ} |")
    lines.append(f"| Particle radius | {RADIUS} cells |")
    lines.append(f"| Shear rates | {GAMMA_RATES} |")
    lines.append(f"| Volume fractions | {PHI_VALUES} |")
    lines.append(f"| Warmup / Sample steps | {N_WARMUP} / {N_STEPS} |")
    lines.append("")
    
    lines.append("## 2. Method")
    lines.append("")
    lines.append("1. **Pure fluid calibration**: Couette flow without particles at each shear rate.")
    lines.append("2. **Particle placement**: Random sequential placement of rigid spheres.")
    lines.append("3. **Suspension simulation**: Shear cell with particles, measure shear rate.")
    lines.append("4. **Relative viscosity**: $\\eta_r = \\dot{\\gamma}_{pure} / \\dot{\\gamma}_{susp}$")
    lines.append("   at the same wall velocity (same applied stress).")
    lines.append("5. **Comparison**: Measured $\\eta_r(\\phi)$ vs analytical closures.")
    lines.append("")
    
    lines.append("## 3. Results")
    lines.append("")
    lines.append("### Pure Fluid Calibration")
    lines.append("")
    lines.append("| $\\dot{\\gamma}_{target}$ | $\\dot{\\gamma}_{meas}$ | R² |")
    lines.append("|---|---|---|")
    for r in pure_results:
        lines.append(f"| {r['gamma_target']:.2e} | {r['gamma_dot']:.2e} | {r['profile_r2']:.4f} |")
    lines.append("")
    
    lines.append("### Relative Viscosity vs Volume Fraction")
    lines.append("")
    lines.append("| $\\phi$ | $\\eta_r$ (mean±std) | N |")
    lines.append("|---|---|---|")
    
    by_phi = {}
    for d in eta_r_data:
        by_phi.setdefault(d["phi_declared"], []).append(d)
    
    for phi in sorted(by_phi.keys()):
        vals = [d["eta_r"] for d in by_phi[phi] if np.isfinite(d["eta_r"])]
        if vals:
            lines.append(f"| {phi:.2f} | {np.mean(vals):.4f} ± {np.std(vals):.4f} | {len(vals)} |")
    lines.append("")
    
    lines.append("### Acceptance Criteria")
    lines.append("")
    lines.append("| Criterion | $\\eta_r$ meas | $\\eta_r$ ref | Rel. err | Threshold | Status |")
    lines.append("|---|---|---|---|---|---|")
    for key, crit in criteria.get("criteria", {}).items():
        meas = crit.get("eta_r_meas", float("nan"))
        ref = crit.get("eta_r_einstein", crit.get("eta_r_kd", float("nan")))
        err = crit.get("rel_err", float("nan"))
        thresh = crit.get("threshold", 0.10)
        passed = "✓ PASS" if crit.get("passed") else "✗ FAIL"
        lines.append(f"| {key} | {meas:.4f} | {ref:.4f} | {err:.2%} | {thresh:.0%} | {passed} |")
    lines.append("")
    
    n_pass = criteria.get("n_pass", 0)
    n_total = criteria.get("n_defined", 0)
    lines.append(f"**Overall**: {n_pass}/{n_total} criteria passed")
    lines.append("")
    
    lines.append("## 4. Figures")
    lines.append("")
    for fig_path in figures:
        fig_name = os.path.basename(fig_path)
        lines.append(f"![{fig_name}]({os.path.relpath(fig_path, os.path.join(ROOT, 'docs'))})")
        lines.append("")
    
    lines.append("## 5. Notes and Limitations")
    lines.append("")
    lines.append("- **Finite domain**: 20×20×24 cells is small; wall confinement affects results.")
    lines.append("- **Particle resolution**: Radius 2.5 cells gives ~50 voxels per sphere.")
    lines.append("- **Stress measurement**: Shear rate ratio method; direct momentum exchange")
    lines.append("  would be more accurate but requires careful calibration.")
    lines.append("- **Statistical sampling**: Only 3 shear rates, 1 seed per condition.")
    lines.append("")
    
    lines.append("## 6. Error Assessment")
    lines.append("")
    for key, crit in criteria.get("criteria", {}).items():
        rel_err = crit.get("rel_err", float("nan"))
        threshold = crit.get("threshold", 0.10)
        if isinstance(rel_err, (int, float)):
            ok = "YES ✓" if rel_err < threshold else "NO ✗"
            lines.append(f"- **{key}**: rel. error = {rel_err:.2%}, threshold = {threshold:.0%} → <{threshold:.0%}? **{ok}**")
    lines.append("")
    
    lines.append("## References")
    lines.append("")
    lines.append("- Einstein, A. (1906) *Ann. Phys.* 19, 289.")
    lines.append("- Krieger, I. M. & Dougherty, T. J. (1959) *Trans. Soc. Rheol.* 3, 137.")
    lines.append("")
    lines.append(f"*Elapsed: {elapsed:.1f} s*")
    
    report = "\n".join(lines)
    report_path = os.path.join(ROOT, "docs", "H2_FRAZIL_RHEOLOGY.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    return report_path


def main():
    t0 = time.time()
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("H2-FRAZIL-RHEOLOGY: Frazil Power-Law Suspension Rheology")
    print("=" * 60)
    
    # Step 1: Pure fluid
    print("\n[1/4] Running pure fluid cases...")
    pure_results = run_pure_cases()
    
    # Step 2: Suspension cases
    print("\n[2/4] Running suspension cases...")
    all_susp_results = []
    for phi in PHI_VALUES:
        if phi == 0.0:
            continue
        print(f"\n  phi = {phi:.2f}:")
        results = run_suspension_cases(phi, seed=42)
        all_susp_results.extend(results)
    
    # Step 3: Compute eta_r
    print("\n[3/4] Computing relative viscosity...")
    eta_r_data = compute_eta_r(pure_results, all_susp_results)
    for d in eta_r_data:
        print(f"  phi={d['phi_declared']:.2f}: eta_r={d['eta_r']:.4f} "
              f"(phi_lat={d['phi_lattice']:.4f})")
    
    # Step 4: Evaluate criteria
    print("\n[4/4] Evaluating acceptance criteria...")
    criteria = evaluate_criteria(eta_r_data)
    for key, crit in criteria["criteria"].items():
        status = "PASS" if crit.get("passed") else "FAIL"
        print(f"  {key}: {status} (rel_err={crit.get('rel_err', 'N/A')})")
    
    # Figures
    print("\nGenerating figures...")
    figures = generate_figures(eta_r_data, pure_results, criteria)
    
    # Report
    elapsed = time.time() - t0
    print("\nGenerating report...")
    report_path = generate_report(pure_results, eta_r_data, criteria, figures, elapsed)
    
    # Artifact
    artifact = {
        "schema": "h2_frazil_rheology/v1",
        "mission_id": 14882,
        "anchor": "H2-FRAZIL-RHEOLOGY",
        "parameters": {
            "nu": NU, "domain": [NX, NY, NZ], "radius": RADIUS,
            "gamma_rates": GAMMA_RATES, "phi_values": PHI_VALUES,
            "n_steps": N_STEPS, "n_warmup": N_WARMUP,
        },
        "pure_results": [{k: v for k, v in r.items() if k != "ux_profile"}
                        for r in pure_results],
        "eta_r_data": eta_r_data,
        "criteria": criteria,
        "elapsed_s": elapsed,
        "external_anchor": {
            "einstein_1906": "Einstein, A. (1906) Ann. Phys. 19, 289",
            "krieger_dougherty_1959": "Krieger & Dougherty (1959) Trans. Soc. Rheol. 3, 137",
        },
        "repro_cmd": f"cd {ROOT} && PYTHONPATH=src python3 scripts/run_h2_frazil_rheology.py",
    }
    
    artifact_path = os.path.join(ARTIFACT_DIR, "campaign.json")
    with open(artifact_path, "w") as f:
        json.dump(artifact, f, indent=2, default=float)
    print(f"\nArtifact saved: {artifact_path}")
    
    print(f"\n{'=' * 60}")
    print(f"Total elapsed: {elapsed:.1f} s")
    print(f"Criteria: {criteria['n_pass']}/{criteria['n_defined']} passed")
    print(f"{'=' * 60}")
    
    return 0 if criteria.get("all_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
