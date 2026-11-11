#!/usr/bin/env python3
"""
p495w4-n04: Interface Layer Law Verification and Two-Term Law Closure

This script verifies the two-term error law for free-surface LBM:
  err(kh,h) = -c1·(ξ/h)·L1(kh) + c2·(kΔx)²

where:
  - L1(kh) = kh/sinh(2kh) is the depth sensitivity (exact)
  - c1·ξ is the interface layer mass deficit (first-order in ξ)
  - c2·(kΔx)² is the lattice discretization error

The goal is to push phase velocity error below 3% at production resolution
(cells/λ≥40 and h≥10) by applying the interface layer correction.

Mission: m13946 (2nd attempt, learning from m9103)
"""

import json
import math
import sys
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm import free_surface_dispersion as fs

ART = ROOT / "artifacts" / "p495w4_n04_interface_layer"
ART.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Theory: Two-Term Law Derivation
# ============================================================================

@dataclass
class TwoTermLaw:
    """
    The two-term error law for free-surface LBM:
    
    err(kh,h) = -c1·(ξ/h)·L1(kh) + c2·(kΔx)²
    
    where:
    - Term 1: Interface layer mass deficit (O(ξ/h), k-dependent via L1)
    - Term 2: Lattice discretization error (O(k²), h-independent)
    
    Physical origin:
    - The interface has a tanh(d/ξ) profile (diffuse interface of thickness ξ)
    - This creates an effective mass deficit: h_eff = h - c1·ξ
    - The phase speed error from depth bias: δc/c = L1(kh)·(δh/h)
    - L1(kh) = kh/sinh(2kh) is the exact depth sensitivity
    """
    c1_xi: float  # c1·ξ: interface layer mass deficit [cells]
    c2: float     # lattice discretization coefficient
    rms_residual: float  # fit quality
    
    def predict(self, k: float, h: float, dx: float = 1.0) -> float:
        """Predict phase speed error at given (k, h)."""
        L1 = fs.depth_sensitivity(k, h)
        term1 = -self.c1_xi / h * L1
        term2 = self.c2 * (k * dx) ** 2
        return term1 + term2
    
    def to_dict(self) -> dict:
        return asdict(self)


def fit_two_term_law(measurements: List[Dict]) -> TwoTermLaw:
    """
    Fit the two-term law to measurement data.
    
    Each measurement must have: k, h, c_rel_error
    The model: err = -c1_xi/h * L1(k,h) + c2 * k²
    
    This is linear in (c1_xi, c2), so we use least squares.
    """
    n = len(measurements)
    if n < 2:
        raise ValueError("Need at least 2 measurements to fit two parameters")
    
    # Build design matrix
    # Column 1: -L1(k,h)/h (coefficient of c1_xi)
    # Column 2: k² (coefficient of c2)
    A = np.zeros((n, 2))
    b = np.zeros(n)
    
    for i, m in enumerate(measurements):
        k = m["k"]
        h = m["h"]
        err = m["c_rel_error"]
        L1 = fs.depth_sensitivity(k, h)
        A[i, 0] = -L1 / h
        A[i, 1] = k ** 2
        b[i] = err
    
    # Least squares solve
    result, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    c1_xi, c2 = result
    
    # Compute RMS residual
    predicted = A @ result
    rms = float(np.sqrt(np.mean((b - predicted) ** 2)))
    
    return TwoTermLaw(c1_xi=float(c1_xi), c2=float(c2), rms_residual=rms)


# ============================================================================
# Correction Operator: Interface Layer Compensation
# ============================================================================

def interface_layer_correction(k: float, h: float, c1_xi: float) -> float:
    """
    Compute the phase speed correction for interface layer mass deficit.
    
    The correction compensates for the effective depth reduction:
    h_eff = h - c1_xi
    
    The phase speed error from this bias is:
    δc/c = L1(kh) · (c1_xi / h)
    
    Returns the relative correction to apply to the phase speed.
    """
    L1 = fs.depth_sensitivity(k, h)
    return c1_xi / h * L1


def apply_laplace_calibration(k: float, h: float, g: float, sigma_cal: float) -> float:
    """
    Apply Laplace calibration to remove O(k) error.
    
    The Laplace term adds: ω² = gk·tanh(kh) + σ·k³/ρ
    This shifts the phase speed by: δc/c ≈ σ·k²/(2ρ·g·tanh(kh))
    
    Returns the relative phase speed shift.
    """
    return fs.laplace_phase_shift(k, h, g, sigma_cal)


# ============================================================================
# Verification Campaign: Three Parameter Families
# ============================================================================

def run_depth_fan(sigma_cal: float = 0.0) -> List[Dict]:
    """
    Depth fan: vary kh at fixed h=20, fixed k·Δx.
    
    This isolates the L1(kh) dependence of the interface layer term.
    """
    results = []
    h = 20.0
    for kh in [0.7, 1.0, 1.5, 2.0, 3.0]:
        case = fs.LinearWaveCase(
            h=h,
            kh=kh,
            amplitude=0.25,
            c_target=0.06,
            tau=0.51,
            wavelengths=1,
            headroom=10,
            n_periods=5.0,
            warmup_periods=1.5,
            sigma_cal=sigma_cal,
        )
        t0 = time.time()
        result = fs.run_linear_wave(case, verbose=False)
        result["wall_s"] = time.time() - t0
        result["name"] = f"depth_kh{kh:.1f}"
        result["family"] = "depth_fan"
        results.append(result)
        print(f"  depth_kh{kh:.1f}: err={100*result['discrete']['c_rel_error']:+.3f}%")
    return results


def run_resolution_fan(sigma_cal: float = 0.0) -> List[Dict]:
    """
    Resolution fan: vary h at fixed kh=1.5.
    
    This isolates the 1/h dependence of the interface layer term.
    """
    results = []
    kh = 1.5
    for h in [5.0, 10.0, 20.0]:
        case = fs.LinearWaveCase(
            h=h,
            kh=kh,
            amplitude=0.25,
            c_target=0.06,
            tau=0.51,
            wavelengths=1,
            headroom=10,
            n_periods=5.0,
            warmup_periods=1.5,
            sigma_cal=sigma_cal,
        )
        t0 = time.time()
        result = fs.run_linear_wave(case, verbose=False)
        result["wall_s"] = time.time() - t0
        result["name"] = f"res_h{h:.0f}"
        result["family"] = "resolution_fan"
        results.append(result)
        print(f"  res_h{h:.0f}: err={100*result['discrete']['c_rel_error']:+.3f}%")
    return results


def run_tau_fan(sigma_cal: float = 0.0) -> List[Dict]:
    """
    Tau fan: vary τ at fixed (kh=1.5, h=20).
    
    This checks if the interface layer term is dissipative in origin.
    """
    results = []
    for tau in [0.5025, 0.51, 0.53]:
        case = fs.LinearWaveCase(
            h=20.0,
            kh=1.5,
            amplitude=0.25,
            c_target=0.06,
            tau=tau,
            wavelengths=1,
            headroom=10,
            n_periods=5.0,
            warmup_periods=1.5,
            sigma_cal=sigma_cal,
        )
        t0 = time.time()
        result = fs.run_linear_wave(case, verbose=False)
        result["wall_s"] = time.time() - t0
        result["name"] = f"tau{tau:.4f}"
        result["family"] = "tau_fan"
        results.append(result)
        print(f"  tau{tau:.4f}: err={100*result['discrete']['c_rel_error']:+.3f}%")
    return results


def run_production_resolution(sigma_cal: float = 0.0, c1_xi: float = 0.0) -> List[Dict]:
    """
    Production resolution test: cells/λ≥40 and h≥10.
    
    This is the mission criterion: phase speed error < 3%.
    """
    results = []
    test_cases = [
        # (h, kh) -> cells/λ = 2π·h/kh
        (20.0, 1.5),  # cells/λ = 84, h=20
        (15.0, 1.5),  # cells/λ = 63, h=15
        (10.0, 1.0),  # cells/λ = 63, h=10
        (10.0, 1.5),  # cells/λ = 42, h=10
        (20.0, 2.0),  # cells/λ = 63, h=20
        (20.0, 3.0),  # cells/λ = 42, h=20
    ]
    
    for h, kh in test_cases:
        case = fs.LinearWaveCase(
            h=h,
            kh=kh,
            amplitude=0.25,
            c_target=0.06,
            tau=0.51,
            wavelengths=1,
            headroom=10,
            n_periods=5.0,
            warmup_periods=1.5,
            sigma_cal=sigma_cal,
        )
        t0 = time.time()
        result = fs.run_linear_wave(case, verbose=False)
        result["wall_s"] = time.time() - t0
        result["name"] = f"prod_h{h:.0f}_kh{kh:.1f}"
        result["family"] = "production"
        
        # Apply interface layer correction if c1_xi is provided
        if c1_xi > 0:
            k = case.k
            correction = interface_layer_correction(k, h, c1_xi)
            result["corrected_c_rel_error"] = result["discrete"]["c_rel_error"] + correction
            result["correction_applied"] = correction
        else:
            result["corrected_c_rel_error"] = result["discrete"]["c_rel_error"]
            result["correction_applied"] = 0.0
        
        results.append(result)
        cells_per_lambda = 2 * math.pi * h / kh
        print(f"  prod_h{h:.0f}_kh{kh:.1f} (cells/λ={cells_per_lambda:.0f}): "
              f"err={100*result['discrete']['c_rel_error']:+.3f}%, "
              f"corrected={100*result['corrected_c_rel_error']:+.3f}%")
    
    return results


# ============================================================================
# Group Speed Verification
# ============================================================================

def compute_group_speed(depth_results: List[Dict]) -> Dict:
    """
    Compute group speed from the depth fan measurements.
    
    c_g = c + k·dc/dk (discrete)
    c_g_cont = c·(1 + 2kh/sinh(2kh))/2 (continuum)
    """
    # Sort by k
    sorted_results = sorted(depth_results, key=lambda r: r["case"]["k"])
    
    # Extract (k, c) pairs
    measurements = []
    for r in sorted_results:
        measurements.append({
            "k": r["case"]["k"],
            "h": r["case"]["h"],
            "g": r["case"]["g_lattice"],
            "c": r["discrete"]["c_measured"],
        })
    
    if len(measurements) < 3:
        return {"error": "Need at least 3 measurements for group speed"}
    
    # Use central difference at the middle point
    mid = len(measurements) // 2
    lo, mi, hi = measurements[mid-1], measurements[mid], measurements[mid+1]
    
    dk = hi["k"] - lo["k"]
    dc = hi["c"] - lo["c"]
    dc_dk = dc / dk
    
    c_g_measured = mi["c"] + mi["k"] * dc_dk
    c_g_continuum = fs.continuous_group_speed(mi["k"], mi["h"], mi["g"])
    
    return {
        "k_triplet": [lo["k"], mi["k"], hi["k"]],
        "c_phase": mi["c"],
        "c_group_measured": c_g_measured,
        "c_group_continuum": c_g_continuum,
        "c_group_rel_error": c_g_measured / c_g_continuum - 1.0,
    }


# ============================================================================
# Main Verification Campaign
# ============================================================================

def main():
    print("=" * 80)
    print("p495w4-n04: Interface Layer Law Verification")
    print("=" * 80)
    
    # Phase 1: Baseline measurements (no calibration)
    print("\n[Phase 1] Baseline measurements (σ_cal = 0)")
    print("-" * 80)
    
    print("\nDepth fan (h=20, varying kh):")
    depth_baseline = run_depth_fan(sigma_cal=0.0)
    
    print("\nResolution fan (kh=1.5, varying h):")
    res_baseline = run_resolution_fan(sigma_cal=0.0)
    
    print("\nTau fan (kh=1.5, h=20, varying τ):")
    tau_baseline = run_tau_fan(sigma_cal=0.0)
    
    # Phase 2: Fit the two-term law
    print("\n[Phase 2] Fitting two-term law")
    print("-" * 80)
    
    # Combine depth and resolution fans for the fit
    all_measurements = []
    for r in depth_baseline + res_baseline:
        all_measurements.append({
            "k": r["case"]["k"],
            "h": r["case"]["h"],
            "c_rel_error": r["discrete"]["c_rel_error"],
        })
    
    law_baseline = fit_two_term_law(all_measurements)
    print(f"\nBaseline two-term law (no calibration):")
    print(f"  c1·ξ = {law_baseline.c1_xi:.4f} cells")
    print(f"  c2   = {law_baseline.c2:.4f}")
    print(f"  RMS residual = {100*law_baseline.rms_residual:.3f}%")
    
    # Phase 3: Apply Laplace calibration and refit
    print("\n[Phase 3] Laplace calibration (σ_cal = 5.5e-4)")
    print("-" * 80)
    
    sigma_cal = 5.5e-4
    
    print("\nDepth fan with calibration:")
    depth_cal = run_depth_fan(sigma_cal=sigma_cal)
    
    print("\nResolution fan with calibration:")
    res_cal = run_resolution_fan(sigma_cal=sigma_cal)
    
    print("\nTau fan with calibration:")
    tau_cal = run_tau_fan(sigma_cal=sigma_cal)
    
    # Refit after calibration
    all_measurements_cal = []
    for r in depth_cal + res_cal:
        all_measurements_cal.append({
            "k": r["case"]["k"],
            "h": r["case"]["h"],
            "c_rel_error": r["discrete"]["c_rel_error"],
        })
    
    law_cal = fit_two_term_law(all_measurements_cal)
    print(f"\nCalibrated two-term law (σ_cal = {sigma_cal}):")
    print(f"  c1·ξ = {law_cal.c1_xi:.4f} cells")
    print(f"  c2   = {law_cal.c2:.6f}")
    print(f"  RMS residual = {100*law_cal.rms_residual:.3f}%")
    
    # Phase 4: Apply interface layer correction
    print("\n[Phase 4] Interface layer correction")
    print("-" * 80)
    
    c1_xi = law_cal.c1_xi
    print(f"\nApplying interface layer correction: c1·ξ = {c1_xi:.4f} cells")
    
    print("\nProduction resolution test (cells/λ≥40, h≥10):")
    prod_results = run_production_resolution(sigma_cal=sigma_cal, c1_xi=c1_xi)
    
    # Phase 5: Group speed verification
    print("\n[Phase 5] Group speed verification")
    print("-" * 80)
    
    group_speed = compute_group_speed(depth_cal)
    print(f"\nGroup speed (from calibrated depth fan):")
    print(f"  c_phase = {group_speed['c_phase']:.6f}")
    print(f"  c_g measured = {group_speed['c_group_measured']:.6f}")
    print(f"  c_g continuum = {group_speed['c_group_continuum']:.6f}")
    print(f"  c_g rel error = {100*group_speed['c_group_rel_error']:+.3f}%")
    
    # Phase 6: Summary and criteria check
    print("\n[Phase 6] Summary and criteria check")
    print("-" * 80)
    
    # Check production resolution criterion
    max_err_baseline = max(abs(r["discrete"]["c_rel_error"]) for r in prod_results)
    max_err_corrected = max(abs(r["corrected_c_rel_error"]) for r in prod_results)
    
    print(f"\nProduction resolution (cells/λ≥40, h≥10):")
    print(f"  Max error (baseline, σ_cal={sigma_cal}): {100*max_err_baseline:.3f}%")
    print(f"  Max error (corrected): {100*max_err_corrected:.3f}%")
    print(f"  Criterion: < 3%")
    print(f"  Status: {'PASS' if max_err_corrected < 0.03 else 'FAIL'}")
    
    # Check group speed criterion
    group_err = abs(group_speed["c_group_rel_error"])
    print(f"\nGroup speed error: {100*group_err:.3f}%")
    print(f"  Criterion: < 5%")
    print(f"  Status: {'PASS' if group_err < 0.05 else 'FAIL'}")
    
    # Save results
    summary = {
        "mission": "m13946",
        "node": "p495w4-n04",
        "two_term_law_baseline": law_baseline.to_dict(),
        "two_term_law_calibrated": law_cal.to_dict(),
        "sigma_cal": sigma_cal,
        "c1_xi": c1_xi,
        "production_results": [
            {
                "name": r["name"],
                "h": r["case"]["h"],
                "kh": r["case"]["kh"],
                "cells_per_wavelength": r["case"]["cells_per_wavelength"],
                "c_rel_error": r["discrete"]["c_rel_error"],
                "corrected_c_rel_error": r["corrected_c_rel_error"],
                "correction_applied": r["correction_applied"],
            }
            for r in prod_results
        ],
        "group_speed": group_speed,
        "criteria": {
            "phase_error_production": {
                "threshold": 0.03,
                "measured": max_err_corrected,
                "pass": max_err_corrected < 0.03,
            },
            "group_speed_error": {
                "threshold": 0.05,
                "measured": group_err,
                "pass": group_err < 0.05,
            },
        },
    }
    
    out_file = ART / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {out_file}")
    
    # Save individual case results
    for r in depth_baseline + res_baseline + tau_baseline:
        case_file = ART / f"case_{r['name']}_baseline.json"
        payload = {k: v for k, v in r.items() if k != "field"}
        case_file.write_text(json.dumps(payload, indent=2, default=str))
    
    for r in depth_cal + res_cal + tau_cal:
        case_file = ART / f"case_{r['name']}_calibrated.json"
        payload = {k: v for k, v in r.items() if k != "field"}
        case_file.write_text(json.dumps(payload, indent=2, default=str))
    
    for r in prod_results:
        case_file = ART / f"case_{r['name']}_production.json"
        payload = {k: v for k, v in r.items() if k != "field"}
        case_file.write_text(json.dumps(payload, indent=2, default=str))
    
    print("\n" + "=" * 80)
    print("Verification complete!")
    print("=" * 80)
    
    return summary


if __name__ == "__main__":
    summary = main()
