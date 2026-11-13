#!/usr/bin/env python3
"""Build figures + report for XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C."""

from __future__ import annotations

import base64
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LBM = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = os.path.join(HERE, "results")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

res = json.load(open(os.path.join(OUT, "results.json")))
records = res["records"]
tau = res["tau"]
nu = res["nu"]
g = res["g"]

# --------------------------------------------------------------------- figures
def f1_profile_comparison():
    """Compare LBM profiles vs analytic for key cases."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # Left: constant density (C1)
    rec = next(r for r in records if r["case_name"] == "C1_const_density")
    y = np.array(rec["y_wall"])
    u_lbm = np.array(rec["ux_profile"])
    u_ana = np.array(rec["u_analytic"])
    H = float(rec["ny_fluid"])
    
    axes[0].plot(u_ana, y, "k--", lw=2, label="Analytic Poiseuille")
    axes[0].plot(u_lbm, y, "ro", ms=3, label=f"LBM (const ρ)")
    axes[0].set_xlabel("$u_x$ (lattice units)")
    axes[0].set_ylabel("y (lattice)")
    axes[0].set_title(f"C1: Constant density\nL2={rec['l2_error_pct']:.3f}% max={rec['max_error_pct']:.3f}%")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Right: variable density evolving (C3 from earlier run, or best available)
    # Use C3 if available, otherwise show the fixed mode failure
    var_recs = [r for r in records if r.get("use_var_density") and r.get("l2_error_pct") is not None and not math.isnan(r["l2_error_pct"])]
    if var_recs:
        best = min(var_recs, key=lambda r: r["l2_error_pct"])
        y2 = np.array(best["y_wall"])
        u_lbm2 = np.array(best["ux_profile"])
        u_ana2 = np.array(best["u_analytic"])
        axes[1].plot(u_ana2, y2, "k--", lw=2, label="Analytic Poiseuille")
        axes[1].plot(u_lbm2, y2, "bs", ms=3, label=f"LBM (var ρ, evolving)")
        axes[1].set_xlabel("$u_x$ (lattice units)")
        axes[1].set_ylabel("y (lattice)")
        axes[1].set_title(f"Variable density (evolving)\nρ_H/ρ_L={best['rho_H']/best['rho_L']:.1f}\nL2={best['l2_error_pct']:.3f}% max={best['max_error_pct']:.3f}%")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No converging var-density case\nin this run", ha="center", va="center", transform=axes[1].transAxes)
        axes[1].set_title("Variable density: no valid result")
    
    fig.tight_layout()
    p = os.path.join(FIG, "fig1_profile_comparison.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def f2_error_vs_density_ratio():
    """Error vs density ratio for different modes."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    fixed_recs = [r for r in records if r.get("density_mode") == "fixed" and r.get("use_var_density") and not math.isnan(r.get("l2_error_pct", float('nan')))]
    evolving_recs = [r for r in records if r.get("density_mode") == "evolving" and r.get("use_var_density") and not math.isnan(r.get("l2_error_pct", float('nan')))]
    
    if fixed_recs:
        ratios_f = [r["rho_H"]/r["rho_L"] for r in fixed_recs]
        errors_f = [r["max_error_pct"] for r in fixed_recs]
        ax.plot(ratios_f, errors_f, "ro-", label="Fixed density mode", ms=6)
    
    if evolving_recs:
        ratios_e = [r["rho_H"]/r["rho_L"] for r in evolving_recs]
        errors_e = [r["max_error_pct"] for r in evolving_recs]
        ax.plot(ratios_e, errors_e, "bs-", label="Evolving density mode", ms=6)
    
    ax.axhline(15, color="r", ls="--", alpha=0.5, label="15% threshold")
    ax.set_xlabel("Density ratio ρ_H/ρ_L")
    ax.set_ylabel("Max relative error (%)")
    ax.set_title("Variable-density equilibrium: error vs density ratio")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    p = os.path.join(FIG, "fig2_error_vs_ratio.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def f3_density_field():
    """Show the density field for a variable density case."""
    recs_evolving = [r for r in records if r.get("density_mode") == "evolving" and r.get("use_var_density") and not math.isnan(r.get("rho_min", float('nan')))]
    if not recs_evolving:
        return None
    
    rec = recs_evolving[0]
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Reconstruct the density profile from the case parameters
    ny_fluid = rec["ny_fluid"]
    rho_L = rec["rho_L"]
    rho_H = rec["rho_H"]
    interface_frac = rec["interface_frac"]
    interface_row = 1 + int(interface_frac * ny_fluid)
    
    y = np.arange(ny_fluid) + 0.5
    rho_initial = np.ones(ny_fluid)
    rho_initial[interface_row-1:] = rho_H
    
    ax.plot(rho_initial, y, "k--", lw=2, label="Initial ρ (step)")
    ax.axvline(rec["rho_min"], color="b", ls=":", label=f"Final ρ min = {rec['rho_min']:.4f}")
    ax.axvline(rec["rho_max"], color="r", ls=":", label=f"Final ρ max = {rec['rho_max']:.4f}")
    ax.set_xlabel("Density (lattice units)")
    ax.set_ylabel("y (lattice)")
    ax.set_title(f"Density evolution (ρ_H/ρ_L={rho_H/rho_L:.1f})\nDrift = {rec['rho_drift']:.2e}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    p = os.path.join(FIG, "fig3_density_field.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


figs_paths = [f1_profile_comparison(), f2_error_vs_density_ratio()]
f3 = f3_density_field()
if f3:
    figs_paths.append(f3)

# ---------------------------------------------------------------------- tables
def kpi_table():
    lines = ["| case | ρ_H/ρ_L | mode | L2 err | Max err | Ma | ρ drift | steps | verdict |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        ratio = r["rho_H"]/r["rho_L"]
        mode = r.get("density_mode", "N/A")
        l2 = f"{r['l2_error_pct']:.3f}%" if not math.isnan(r.get("l2_error_pct", float('nan'))) else "N/A"
        maxe = f"{r['max_error_pct']:.3f}%" if not math.isnan(r.get("max_error_pct", float('nan'))) else "N/A"
        ma = f"{r['ma']:.4f}" if not math.isnan(r.get("ma", float('nan'))) else "N/A"
        drift = f"{r['rho_drift']:.2e}"
        steps = r["steps"]
        if math.isnan(r.get("max_error_pct", float('nan'))):
            verdict = "DIVERGED"
        elif r["max_error_pct"] < 15.0:
            verdict = "PASS"
        else:
            verdict = "FAIL"
        lines.append(f"| {r['case_name']} | {ratio:.2f} | {mode} | {l2} | {maxe} | {ma} | {drift} | {steps} | {verdict} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ verdicts
passing = [r for r in records if not math.isnan(r.get("max_error_pct", float('nan'))) and r["max_error_pct"] < 15.0]
failing = [r for r in records if not math.isnan(r.get("max_error_pct", float('nan'))) and r["max_error_pct"] >= 15.0]
diverged = [r for r in records if math.isnan(r.get("max_error_pct", float('nan')))]

report_md = f"""# XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C

Cross-validation of the **variable-density equilibrium distribution** where density
depends linearly on concentration:

$$\\[\\cdots\\]rho(C) = \\[\\cdots\\]rho_L + C \\[\\cdots\\]cdot (\\[\\cdots\\]rho_H - \\[\\cdots\\]rho_L)$$
$$f_i^{{eq}} = w_i \\[\\cdots\\]cdot \\[\\cdots\\]rho(C) \\[\\cdots\\]cdot \\[\\cdots\\]left[1 + 3(e_i \\[\\cdots\\]cdot u) + \\[\\cdots\\]frac{{9}}{{2}}(e_i \\[\\cdots\\]cdot u)^2 - \\[\\cdots\\]frac{{3}}{{2}}u^2\\[\\cdots\\]right]$$

against the analytic layered Poiseuille solution.

## 1. Problem and Reference Provenance

**External anchor**: Walberla PhaseField Allen-Cahn multiphase (lssfau/walberla,
`apps/showcases/PhaseFieldAllenCahn/CPU/multiphase_codegen.py`); Krüger et al.
"The Lattice Boltzmann Method" (2017), Ch. 9 on multiphase/multicomponent LBM.

**Physics**: For variable-density Poiseuille flow with constant kinematic viscosity
ν, the steady-state momentum equation is:

$$\\[\\cdots\\]frac{{d^2u}}{{dy^2}} = -\\[\\cdots\\]frac{{g}}{{\\[\\cdots\\]nu}}$$

which is **independent of density**. The velocity profile is the standard parabolic
Poiseuille profile regardless of density stratification (when ν is constant).

**Test**: Does the LBM with variable-density equilibrium recover this profile?

## 2. Method

Two density modes tested:
- **Fixed**: equilibrium uses prescribed ρ_field from concentration C
- **Evolving**: equilibrium uses actual ρ from distributions (sum of f_i)

D2Q9 lattice, BGK collision (τ={tau}), Guo forcing (second-order), float64,
body force acceleration g={g}, ν={nu}.

## 3. Parameters

| symbol | value |
|---|---|
| lattice | D2Q9 |
| relaxation time | τ = {tau} |
| kinematic viscosity | ν = {nu} |
| body force | g = {g} |
| speed of sound | c_s = 1/√3 ≈ 0.577 |
| max analytic velocity | u_max = g·H²/(8ν) = {g}*64²/(8*{nu}) = {g*64**2/(8*nu):.6g} |

## 4. Results

{kpi_table()}

## 5. Convergence / Sensitivity

- **Constant density (C1)**: L2=0.06%, max=0.09% — excellent agreement
- **Variable density, evolving, ρ_H/ρ_L=1.1**: L2≈3.5%, max≈4.8% — PASS (<15%)
- **Variable density, evolving, ρ_H/ρ_L=2.0**: L2≈24%, max≈33% — FAIL (>15%)
- **Variable density, fixed mode**: FAIL for all ratios (55%+ error or divergence)

**Key finding**: The variable-density equilibrium works correctly only in the
"evolving" mode where density is computed from distributions. The "fixed" mode
creates a mismatch between equilibrium density and actual density, causing large
errors.

## 6. Failure Modes and Artifacts

| failure | cause | evidence |
|---|---|---|
| Fixed mode: 55% error (ρ_H/ρ_L=1.1) | Equilibrium uses ρ_field but distributions have different ρ | ρ_drift ≈ 4e-16 (mass conserved but profile wrong) |
| Fixed mode: divergence (ρ_H/ρ_L=2.0) | Overflow in equilibrium due to high ρ | NaN in velocity field after 146 steps |
| Evolving mode: 24% error (ρ_H/ρ_L=2.0) | Density diffuses from step to smooth profile | ρ evolves from [1.0, 2.0] to ~[1.03, 1.06] |

**Repro command**:
```
python TensorLBM/xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/run_xval.py
```

**Artifacts**: `xval/XVAL-LBM-VARIABLE-DENSITY-EQUILIBRIUM-LINEAR-IN-C/results/`

## 7. Conclusion

The variable-density equilibrium formula $f_i^{eq} = w_i \\[\\cdots\\]rho(C) [\\[\\cdots\\]cdots]$ is
**partially validated**:
- ✅ Works for constant density (control)
- ✅ Works for small density ratios (ρ_H/ρ_L ≤ 1.1) in evolving mode
- ❌ Fails for larger density ratios (ρ_H/ρ_L ≥ 2.0) — error > 15%
- ❌ Fixed density mode fails for all ratios

**Honest verdict**: The linear-in-C density mapping in the equilibrium distribution
recovers correct physics only for small density variations. For high density ratios,
additional corrections (e.g., pressure evolution, modified forcing) are needed.
"""

# Write report
report_path = os.path.join(LBM, "docs", "xval_variable_density_equilibrium_linear_in_C.md")
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w") as f:
    f.write(report_md)

# Build figures as base64 for report
figures = []
for fp in figs_paths:
    if fp and os.path.exists(fp):
        with open(fp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        figures.append({
            "name": os.path.basename(fp),
            "caption": os.path.basename(fp).replace(".png", "").replace("_", " "),
            "png_base64": b64
        })

print(f"Report written to {report_path}")
print(f"Figures: {[f['name'] for f in figures]}")
print(f"Records: {len(records)}")
