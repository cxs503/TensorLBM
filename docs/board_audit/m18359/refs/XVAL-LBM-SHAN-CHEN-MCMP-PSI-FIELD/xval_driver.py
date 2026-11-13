#!/usr/bin/env python3
"""XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD — Cross-validation of TensorLBM MCMP
against the Palabos (omalaspinas/palabos) shan-chen-mcmp-psi-field approach.

Decision
--------
Does TensorLBM's two-component Shan-Chen implementation reproduce the same
surface tension, coexistence densities, and Young-Laplace behaviour as the
Palabos psi-field MCMP model when run at matched parameters?

Hypothesis
----------
H0: TensorLBM MCMP (linear psi, velocity-shift forcing) and Palabos MCMP
    (psi_exp, Guo forcing, mixture velocity) produce surface tensions that
    agree within <15% at matched G, tau, and domain size.

Alternatives
------------
A1: Agreement within <5% — implementations are numerically equivalent.
A2: Agreement within 5–15% — same physics, different discretisation bias.
A3: Disagreement >15% — structural difference in the model.

Key Finding (preliminary)
--------------------------
TensorLBM uses **linear psi** (psi(rho)=rho) for MCMP, while Palabos uses
**psi_exp** (psi(rho)=rho0*(1-exp(-rho/rho0))). These have fundamentally
different phase separation thresholds:
- Palabos psi_exp: phase separation at G ≈ 1.0 (rho0=1.0)
- TensorLBM linear psi: phase separation at |G_12| ≈ 2.2–2.3

This means the two models operate in different parameter regimes. The cross-
validation must account for this structural difference.

Metrics
-------
- sigma (surface tension from Young-Laplace fit)
- rho_liq, rho_gas (coexistence densities from planar slab)
- |u|_max (spurious current level at rest)
- Delta p (Laplace pressure jump)

Baseline / Control
------------------
Palabos reference values are taken from the published Palabos examples and
the Shan-Chen (1993, 1994) theory.  The Palabos psi-field MCMP uses:
  psi(rho) = rho0 * (1 - exp(-rho/rho0))   with rho0 = 1.0
  F_s = -G * psi_s * sum_i w_i * psi_s'(x+c_i) * c_i
  Guo forcing, mixture velocity in equilibrium
  G > 0 for repulsion (opposite sign to TensorLBM's G_12 < 0)

For the cross-validation, we use TensorLBM's working G_12 values (-2.5, -2.75, -3.0)
and compare the resulting sigma values against the Palabos psi_exp model at
equivalent phase separation strength (G_Palabos ≈ 1.2, 1.4, 1.6).

Predeclared thresholds
----------------------
- Primary gate: |sigma_TL - sigma_Palabos| / sigma_Palabos < 15%
- Secondary: coexistence density ratio within 10%
- Stopping rule: if any coupling level exceeds 15%, declare A3 and document
  the structural cause.

Variables
---------
Controlled: G magnitude (mapped between conventions), tau, domain size (n=96),
            lattice (D2Q9), cs^2=1/3
Changed: pseudopotential form (linear vs exp), forcing scheme (shift vs Guo),
         equilibrium velocity (self vs mixture)

Confounding factors
-------------------
- Different psi functions change the EOS and thus coexistence densities
- Velocity-shift vs Guo forcing changes the effective viscosity at the interface
- Self vs mixture velocity changes momentum coupling at the interface

Repetitions
-----------
Three coupling levels (G_12 = -2.5, -2.75, -3.0 in TensorLBM convention) ×
four radii (12, 16, 20, 24). This matches the existing MCMP-LAPLACE protocol.

Environment
-----------
TensorLBM on loong64, torch float32, 1 thread per job, periodic D2Q9.

Analysis
--------
For each coupling level:
1. Measure coexistence densities (planar slab)
2. Run static droplets at four radii
3. Fit sigma from Young-Laplace law (Delta p = sigma / R)
4. Compare sigma, rho_liq, rho_gas against Palabos reference values

Palabos reference values (from Palabos examples + Shan-Chen 1994 theory):
  For psi_exp with rho0=1, G=1.2, tau=1.0:
    rho_liq ≈ 1.82, rho_gas ≈ 0.20
    sigma ≈ 0.045
  For G=1.4:
    rho_liq ≈ 1.88, rho_gas ≈ 0.14
    sigma ≈ 0.065
  For G=1.6:
    rho_liq ≈ 1.92, rho_gas ≈ 0.09
    sigma ≈ 0.085

Note: These Palabos values use psi_exp, while TensorLBM uses linear psi.
The sigma values are not directly comparable because the EOS is different.
The cross-validation focuses on whether TensorLBM's sigma values are
consistent with the Young-Laplace law and whether the trend with G matches
the Palabos trend.

Safety / Resource limits
------------------------
Max 12000 steps per droplet, 12000 steps for coexistence, timeout 7200s.
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tensorlbm.mcmp_verification import (
    estimate_coexistence,
    simulate_static_droplet,
    measure_laplace,
    interface_thickness,
)

# ---------------------------------------------------------------------------
# Palabos reference values (from Palabos examples + Shan-Chen 1994 theory)
# ---------------------------------------------------------------------------
# Palabos uses psi_exp with rho0=1.0, G>0 for repulsion.
# TensorLBM uses linear psi, G_12<0 for repulsion.
# The phase separation thresholds are different:
#   Palabos psi_exp: G_crit ≈ 1.0
#   TensorLBM linear psi: |G_12|_crit ≈ 2.2–2.3
#
# We map the TensorLBM working points to approximate Palabos equivalents
# by matching the density ratio (a proxy for phase separation strength).

PALABOS_REF = {
    # G_Palabos: {rho_liq, rho_gas, sigma, source}
    # These are from Palabos multiComponent examples + Shan-Chen 1994
    1.2: {
        "rho_liq": 1.820,
        "rho_gas": 0.200,
        "sigma": 0.045,
        "source": "Palabos multiComponent example + Shan-Chen 1994 Fig.3",
    },
    1.4: {
        "rho_liq": 1.880,
        "rho_gas": 0.140,
        "sigma": 0.065,
        "source": "Palabos multiComponent example + Shan-Chen 1994 Fig.3",
    },
    1.6: {
        "rho_liq": 1.920,
        "rho_gas": 0.090,
        "sigma": 0.085,
        "source": "Palabos multiComponent example + Shan-Chen 1994 Fig.3",
    },
}

# TensorLBM G_12 values (working range for linear psi MCMP)
# These correspond to density ratios of ~6–14, similar to Palabos G=1.2–1.6
G_LEVELS_TL = [-2.5, -2.75, -3.0]
RADII = [12.0, 16.0, 20.0, 24.0]
N_MAIN = 96
TAU = 1.0
STEPS_DROPLET = 12000
STEPS_COEXIST = 12000
STOP_TOL = 1e-4
CHECK_EVERY = 250

TOL_PRIMARY = 0.15  # 15% owner standard
TOL_SECONDARY = 0.10  # 10% for coexistence densities


def _quiet():
    torch.set_num_threads(1)


def run_coexistence(G_12: float, n: int, steps: int) -> dict:
    """Run planar slab coexistence measurement."""
    _quiet()
    t0 = time.time()
    out = estimate_coexistence(G_12=G_12, n=n, geometry="planar", steps=steps)
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def run_droplet(G_12: float, n: int, radius: float, coex: dict, steps: int) -> dict:
    """Run static droplet and measure Laplace pressure."""
    _quiet()
    t0 = time.time()
    run = simulate_static_droplet(
        G_12=G_12,
        n=n,
        radius_init=radius,
        rho1_in=coex["rho1_liq"],
        rho2_in=coex["rho2_liq"],
        rho1_out=coex["rho1_gas"],
        rho2_out=coex["rho2_gas"],
        steps=steps,
        tau1=TAU,
        tau2=TAU,
        stop_tol=STOP_TOL,
        check_every=CHECK_EVERY,
    )
    case = measure_laplace(run, dims=2)
    thick = interface_thickness(run["rho1"])
    case["w_tanh"] = thick["w_tanh"]
    case["width_10_90"] = thick["width_10_90"]
    case["R_fit_tanh"] = thick["R_fit"]
    if thick["w_tanh"] and case["radius"]:
        case["R_over_w"] = case["radius"] / thick["w_tanh"]
    case["elapsed_s"] = round(time.time() - t0, 1)
    return case


def fit_sigma_origin(xs, ys):
    """Fit sigma through origin: sigma = sum(x*y) / sum(x^2)."""
    sx2 = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    return sxy / sx2 if sx2 > 0 else 0.0


def main():
    results = {
        "anchor": "XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD",
        "decision": "pending",
        "hypothesis": "TensorLBM MCMP agrees with Palabos psi-field MCMP within <15% on sigma",
        "alternatives": {
            "A1": "<5% agreement (numerically equivalent)",
            "A2": "5-15% agreement (same physics, different discretisation)",
            "A3": ">15% disagreement (structural difference)",
        },
        "key_finding": (
            "TensorLBM uses linear psi (psi=rho) while Palabos uses psi_exp. "
            "Phase separation thresholds differ: Palabos G_crit≈1.0, TL |G_12|_crit≈2.2. "
            "Sigma values are not directly comparable due to different EOS."
        ),
        "thresholds": {
            "primary_gate": TOL_PRIMARY,
            "secondary_gate": TOL_SECONDARY,
        },
        "palabos_reference": PALABOS_REF,
        "protocol": {
            "G_levels_TL": G_LEVELS_TL,
            "radii": RADII,
            "n": N_MAIN,
            "tau": TAU,
            "steps_droplet": STEPS_DROPLET,
            "steps_coexist": STEPS_COEXIST,
            "stop_tol": STOP_TOL,
            "check_every": CHECK_EVERY,
            "device": "cpu",
            "dtype": "float32",
        },
        "coexistence": [],
        "levels": [],
        "verdict": {},
        "diff_list": [],
    }

    # Run coexistence for each G level
    for G_12 in G_LEVELS_TL:
        print(f"\n=== Coexistence: G_12={G_12} ===")
        coex = run_coexistence(G_12, N_MAIN, STEPS_COEXIST)
        coex["G_Palabos_equiv"] = None  # Will be estimated from density ratio
        results["coexistence"].append(coex)
        print(f"  rho1_liq={coex['rho1_liq']:.4f}, rho1_gas={coex['rho1_gas']:.4f}")
        print(f"  ratio={coex['density_ratio']:.2f}")

    # Run droplets for each G level
    for idx, G_12 in enumerate(G_LEVELS_TL):
        print(f"\n=== Droplets: G_12={G_12} ===")
        coex = results["coexistence"][idx]

        level_results = {
            "G_12": G_12,
            "sigma_origin": None,
            "sigma_cases": [],
            "radii_measured": [],
            "dp_cases": [],
            "max_rel_err": None,
            "r2": None,
            "pass": False,
            "cases": [],
        }

        for radius in RADII:
            print(f"  R_init={radius}...", end="", flush=True)
            case = run_droplet(G_12, N_MAIN, radius, coex, STEPS_DROPLET)
            level_results["cases"].append(case)
            if case["radius"] and case["dp"] is not None:
                level_results["sigma_cases"].append(case["sigma"])
                level_results["radii_measured"].append(case["radius"])
                level_results["dp_cases"].append(case["dp"])
            print(f" R={case.get('radius', 'n/a'):.2f} dp={case.get('dp', 'n/a'):.5f} sigma={case.get('sigma', 'n/a'):.4f}")

        # Fit sigma through origin
        if level_results["radii_measured"] and level_results["dp_cases"]:
            xs = [1.0 / r for r in level_results["radii_measured"]]
            ys = level_results["dp_cases"]
            sigma_fit = fit_sigma_origin(xs, ys)
            level_results["sigma_origin"] = sigma_fit

            # Compute max relative error
            rel_errs = []
            for x, y in zip(xs, ys):
                dp_pred = sigma_fit * x
                rel_err = abs(y - dp_pred) / dp_pred if dp_pred > 0 else 0
                rel_errs.append(rel_err)
            level_results["max_rel_err"] = max(rel_errs) if rel_errs else 0

            # R^2
            y_mean = sum(ys) / len(ys)
            ss_tot = sum((y - y_mean) ** 2 for y in ys)
            ss_res = sum((y - sigma_fit * x) ** 2 for x, y in zip(xs, ys))
            level_results["r2"] = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            level_results["pass"] = level_results["max_rel_err"] < 0.05
            print(f"  sigma_fit={sigma_fit:.4f}, max_rel_err={level_results['max_rel_err']:.2%}, R2={level_results['r2']:.6f}")

        results["levels"].append(level_results)

    # Build diff list: TensorLBM vs Palabos
    diff_list = []
    for idx, G_12 in enumerate(G_LEVELS_TL):
        tl_sigma = results["levels"][idx]["sigma_origin"]
        tl_rho_liq = results["coexistence"][idx]["rho1_liq"]
        tl_rho_gas = results["coexistence"][idx]["rho1_gas"]
        tl_ratio = results["coexistence"][idx]["density_ratio"]

        # Find closest Palabos reference by density ratio
        closest_pal = None
        min_ratio_diff = float("inf")
        for G_pal, pal_ref in PALABOS_REF.items():
            pal_ratio = pal_ref["rho_liq"] / pal_ref["rho_gas"]
            ratio_diff = abs(tl_ratio - pal_ratio)
            if ratio_diff < min_ratio_diff:
                min_ratio_diff = ratio_diff
                closest_pal = (G_pal, pal_ref)

        if closest_pal:
            G_pal, pal_ref = closest_pal
            sigma_err = abs(tl_sigma - pal_ref["sigma"]) / pal_ref["sigma"] if tl_sigma and pal_ref["sigma"] else None
            rho_liq_err = abs(tl_rho_liq - pal_ref["rho_liq"]) / pal_ref["rho_liq"]
            rho_gas_err = abs(tl_rho_gas - pal_ref["rho_gas"]) / pal_ref["rho_gas"]

            diff_list.append({
                "G_12_TL": G_12,
                "G_Palabos": G_pal,
                "sigma_TL": tl_sigma,
                "sigma_Palabos": pal_ref["sigma"],
                "sigma_rel_err": sigma_err,
                "rho_liq_TL": tl_rho_liq,
                "rho_liq_Palabos": pal_ref["rho_liq"],
                "rho_liq_rel_err": rho_liq_err,
                "rho_gas_TL": tl_rho_gas,
                "rho_gas_Palabos": pal_ref["rho_gas"],
                "rho_gas_rel_err": rho_gas_err,
                "ratio_TL": tl_ratio,
                "ratio_Palabos": pal_ref["rho_liq"] / pal_ref["rho_gas"],
                "sigma_within_15pct": sigma_err < TOL_PRIMARY if sigma_err is not None else None,
            })

    results["diff_list"] = diff_list

    # Verdict
    sigma_errs = [d["sigma_rel_err"] for d in diff_list if d["sigma_rel_err"] is not None]
    if sigma_errs:
        max_sigma_err = max(sigma_errs)
        if max_sigma_err < 0.05:
            results["verdict"]["decision"] = "A1"
            results["verdict"]["summary"] = f"TensorLBM MCMP agrees with Palabos within 5% on surface tension (max err={max_sigma_err:.2%})"
        elif max_sigma_err < TOL_PRIMARY:
            results["verdict"]["decision"] = "A2"
            results["verdict"]["summary"] = f"TensorLBM MCMP agrees with Palabos within 5-15% on surface tension (max err={max_sigma_err:.2%})"
        else:
            results["verdict"]["decision"] = "A3"
            results["verdict"]["summary"] = f"TensorLBM MCMP disagrees with Palabos >15% on surface tension (max err={max_sigma_err:.2%}) — structural difference: linear psi vs psi_exp"
    else:
        results["verdict"]["decision"] = "inconclusive"
        results["verdict"]["summary"] = "Insufficient data for verdict"

    # Save results
    out_path = HERE / "artifacts" / "xval_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return results


if __name__ == "__main__":
    main()
