#!/usr/bin/env python3
"""LBM hold-out test for cluster execution."""

import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

def run_lbm_holdout(kh, eps, protocol):
    """Run LBM hold-out for given parameters."""
    from tensorlbm.wave_tank.tank import WaveTankConfig, run_case
    from tensorlbm.wave_probe_inversion.inversion import MultiProbeLeastSquares
    from tensorlbm.wave_tank.gauges import fit_harmonic
    from tensorlbm.wave_tank.stokes2 import dispersion_solve
    import math
    
    DEPTH = {0.7: 12.0, 1.5: 24.0, 3.0: 40.0}
    h = DEPTH.get(kh, 24.0)
    g = 6.5e-3
    nu = 0.008
    
    # Compute wave parameters
    k = kh / h
    omega = math.sqrt(g * 0.9 * k * math.tanh(k * h))
    wavelength = 2 * math.pi / k
    c_theory = omega / k
    
    # Set up probes
    s = protocol["s_over_lambda"] * wavelength
    L = protocol["L_over_lambda"] * wavelength
    n_probes = protocol.get("n_probes", 5)
    
    lam_mid = protocol["L_over_lambda"] + 0.5
    probe_frac = tuple([0.1 + i * protocol["s_over_lambda"] for i in range(n_probes)])
    probe_frac = tuple([p / lam_mid for p in probe_frac])
    
    cfg = WaveTankConfig(
        kh=kh, eps=eps, h=h, g=g, nu=nu,
        absorb=True, lam_mid=lam_mid, probe_frac=probe_frac,
        warm_periods=6.0, meas_periods=6.0, sample_every=2,
    )
    
    result = run_case(cfg, quiet=True)
    
    # Extract and fit
    probe_eta = result.series["probe_eta"]
    t = result.series["t"]
    omega_fit = cfg.wave.omega
    
    c_probes = []
    for i in range(n_probes):
        c = fit_harmonic(t, probe_eta[i], omega_fit)
        c_probes.append(c)
    c_probes = np.array(c_probes)
    
    # Inversion
    n_in = cfg.geometry()["n_in"]
    probe_cols = result.series["probe_cols"]
    x_probes = np.array([n_in + col for col in probe_cols])
    
    inv = MultiProbeLeastSquares(omega_fit, k, 0.0)
    inv_result = inv.invert(x_probes, c_probes)
    
    c_measured = inv_result.c_hat
    R_measured = inv_result.R_hat
    c_error = abs(c_measured - c_theory) / c_theory
    
    return {
        "kh": kh,
        "c_theory": c_theory,
        "c_measured": c_measured,
        "c_error": c_error,
        "R_measured": R_measured,
        "passed": c_error < 0.03,
    }

def main():
    """Run all LBM hold-out tests."""
    protocol = {
        "dx_over_lambda": 1/20,
        "s_over_lambda": 0.45,
        "L_over_lambda": 2.0,
        "n_probes": 5,
    }
    
    results = []
    for kh in [0.7, 1.5, 3.0]:
        print(f"Running kh={kh}...")
        try:
            res = run_lbm_holdout(kh, 0.08, protocol)
            results.append(res)
            print(f"  c_error = {res['c_error']:.4f} ({res['c_error']*100:.2f}%)")
        except Exception as e:
            print(f"  Failed: {e}")
            results.append({"kh": kh, "error": str(e)})
    
    # Save results
    with open('artifacts/m7467/lbm_holdout_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nResults saved to artifacts/m7467/lbm_holdout_results.json")

if __name__ == '__main__':
    main()
