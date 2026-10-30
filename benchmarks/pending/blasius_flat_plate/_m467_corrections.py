#!/usr/bin/env python3
"""m467: Test combined correction methods for Blasius C_f.

Key findings:
1. Edge velocity effect: u_e/U ≈ 1.08, causes +21% error
2. Near-wall resolution: LBM overpredicts first cell velocity by ~27%
3. Combined: Standard FD gives +27% error

Correction strategies:
1. Edge velocity correction: C_f_corr = C_f_sim * (U/u_e)^2.5
2. Profile fitting: Fit u_sim(eta) = A * u_blasius(eta), use A to correct gradient
3. Combined: Apply both corrections

This script tests these methods on the existing results.
"""

import json
import math
import numpy as np
from pathlib import Path

CF_COEFF = 2.0 * 0.3320573362151963

results_dir = Path("results")
result_files = [
    "R6p_plate400_y1600_trt.json",  # Best case
]

print("=" * 80)
print("m467: Combined Correction Methods for Blasius C_f")
print("=" * 80)

for fn in result_files:
    with open(results_dir / fn) as f:
        r = json.load(f)
    
    print(f"\nCase: {fn}")
    print(f"Grid: {r['grid']}, Collision: {r['collision']}")
    print(f"Re_x: {r['Rex']:.1f}")
    
    # Get probe station
    x_eff = r['x_eff']
    station = next(s for s in r['stations'] if s['x_eff'] == x_eff)
    
    # Reference C_f
    Cf_ref = CF_COEFF / math.sqrt(r['Rex'])
    Cf_sim = station['Cf_fd']
    Cf_err = station['Cf_fd_err_pct']
    
    print(f"\nBaseline (standard FD):")
    print(f"  C_f ref:  {Cf_ref:.6f}")
    print(f"  C_f sim:  {Cf_sim:.6f}")
    print(f"  C_f err:  {Cf_err:+.2f}%")
    
    # Method 1: Edge velocity correction
    u_e_over_U = station['u_edge_local_over_U']
    # Theoretical correction: C_f ~ u_e^(-0.5) for Blasius with edge velocity
    # But more accurately: C_f = 2 * f''(0) * u_e / sqrt(Re_x) * (u_e/U)^(-2)
    # Simplified: C_f_corr = C_f_sim * (U/u_e)^2
    Cf_edge = Cf_sim * (1.0 / u_e_over_U) ** 2
    Cf_edge_err = (Cf_edge - Cf_ref) / Cf_ref * 100
    
    print(f"\nMethod 1: Edge velocity correction (u_e/U = {u_e_over_U:.5f})")
    print(f"  C_f corrected: {Cf_edge:.6f}")
    print(f"  C_f err: {Cf_edge_err:+.2f}%")
    print(f"  Error reduction: {Cf_err - Cf_edge_err:.2f} pp")
    
    # Method 2: Profile fitting
    prof = r['profile_subsampled']
    eta = np.array([p['eta'] for p in prof])
    u_sim = np.array([p['u_sim'] for p in prof])
    u_ref = np.array([p['u_blasius'] for p in prof])
    
    # Fit in the near-wall region (eta < 4)
    m_fit = (eta > 0.05) & (eta < 4.0) & (u_ref > 0.001) & (u_ref < 0.99 * r['U'])
    if m_fit.sum() > 10:
        u_fit = u_sim[m_fit]
        uref_fit = u_ref[m_fit]
        A_fit = np.dot(u_fit, uref_fit) / np.dot(uref_fit, uref_fit)
        
        # The gradient scales with A, so C_f_corrected = C_f_sim / A_fit
        Cf_fit = Cf_sim / A_fit
        Cf_fit_err = (Cf_fit - Cf_ref) / Cf_ref * 100
        
        print(f"\nMethod 2: Profile fitting (A = {A_fit:.5f})")
        print(f"  C_f corrected: {Cf_fit:.6f}")
        print(f"  C_f err: {Cf_fit_err:+.2f}%")
        print(f"  Error reduction: {Cf_err - Cf_fit_err:.2f} pp")
    
    # Method 3: Combined (edge + profile fitting)
    if m_fit.sum() > 10:
        Cf_combined = Cf_fit * (1.0 / u_e_over_U) ** 2
        Cf_combined_err = (Cf_combined - Cf_ref) / Cf_ref * 100
        
        print(f"\nMethod 3: Combined (edge + profile fitting)")
        print(f"  C_f corrected: {Cf_combined:.6f}")
        print(f"  C_f err: {Cf_combined_err:+.2f}%")
        print(f"  Error reduction: {Cf_err - Cf_combined_err:.2f} pp")
    
    # Method 4: Use the normalized C_f from the station data
    Cf_norm_ue = station['Cf_fd_norm_ue_err_pct']
    print(f"\nMethod 4: Station data normalized C_f")
    print(f"  C_f err (normalized): {Cf_norm_ue:+.2f}%")

print("\n" + "=" * 80)
print("Conclusion:")
print("=" * 80)
print("The standard FD method gives +27.34% error for the best case.")
print("Edge velocity correction reduces this to ~+9%.")
print("Profile fitting can further reduce the error.")
print("However, these are post-hoc corrections and require knowledge of the")
print("edge velocity or the Blasius profile, which may not be available in")
print("general simulations.")
print("\nThe fundamental issue is that LBM with standard half-way bounce-back")
print("has first-order accuracy near the wall, leading to ~27% overprediction")
print("of the near-wall velocity gradient.")
print("\nTo achieve <3% error, one would need:")
print("  1. Higher-order boundary conditions (interpolated bounce-back)")
print("  2. Wall models or wall functions")
print("  3. Much finer grid resolution near the wall (grid stretching)")
print("  4. Post-hoc corrections based on edge velocity or profile fitting")
