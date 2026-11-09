#!/usr/bin/env python3
"""MIZ parameter sweep (IN101-N04 rewrite).

Runs a systematic parameter sweep over:
- Ice concentration C_floe ∈ [0.1, 0.9]
- Wave period Tp ∈ [6, 12] s
- Wave height Hs ∈ [0.5, 3.0] m

Outputs (per run):
- Wave attenuation coefficient α
- Characteristic floe size a_floe and ratio a_floe/λ_peak
- Resistance components (R_ow, R_wave, R_ice, R_total)
- Roll RAO reduction vs open water
- Energy budget residual

Usage:
    python tools/run_miz_sweep.py [--output runs/miz_sweep/]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.miz_coupling import (
    ShipParams,
    WaveSpectrum,
    characteristic_floe_size,
    run_miz_simulation,
    verify_wave_attenuation_concentration_correlation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="MIZ parameter sweep")
    parser.add_argument("--output", type=str, default="runs/miz_sweep",
                        help="Output directory")
    parser.add_argument("--ship-L", type=float, default=100.0,
                        help="Ship length [m]")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    ship = ShipParams(L=args.ship_L)

    C_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    Tp_values = [6.0, 8.0, 10.0, 12.0]
    Hs_values = [0.5, 1.5, 3.0]

    results = []
    t0 = time.time()

    for Hs in Hs_values:
        for Tp in Tp_values:
            for C in C_values:
                t_start = time.time()
                res = run_miz_simulation(
                    Hs=Hs, Tp=Tp, C_floe=C, ship=ship,
                )
                wave = WaveSpectrum(Hs=Hs, Tp=Tp)
                kp = wave.peak_wavenumber()
                lam = 2 * math.pi / kp if kp > 0 else float('inf')
                af = characteristic_floe_size(0.5e6, kp, Hs=Hs)
                results.append({
                    'Hs': Hs,
                    'Tp': Tp,
                    'C_floe': C,
                    'alpha': res.alpha,
                    'a_floe': res.a_floe,
                    'a_floe_lambda_ratio': res.a_floe / lam if lam < float('inf') else 0.0,
                    'R_open_water': res.R_open_water,
                    'R_wave': res.R_wave,
                    'R_ice': res.R_ice,
                    'R_total': res.R_total,
                    'R_ice_pct_of_total': 100.0 * res.R_ice / res.R_total if res.R_total > 0 else 0.0,
                    'roll_reduction_pct': res.roll_reduction_pct,
                    'energy_residual_pct': res.energy_residual_pct,
                    'runtime_s': time.time() - t_start,
                })

    # Save results
    out_path = out_dir / "sweep_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    r, _, _ = verify_wave_attenuation_concentration_correlation()
    summary = {
        'n_runs': len(results),
        'wave_attenuation_r': r,
        'total_runtime_s': time.time() - t0,
        'R_ice_pct_range': [
            min(x['R_ice_pct_of_total'] for x in results),
            max(x['R_ice_pct_of_total'] for x in results),
        ],
        'roll_reduction_range': [
            min(x['roll_reduction_pct'] for x in results),
            max(x['roll_reduction_pct'] for x in results),
        ],
        'energy_residual_max': max(x['energy_residual_pct'] for x in results),
    }
    with open(out_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Sweep complete: {len(results)} runs in {time.time()-t0:.1f}s")
    print(f"Results: {out_path}")
    print(f"Summary: {out_dir / 'summary.json'}")
    print(f"  wave attenuation r = {r:.4f}")
    print(f"  R_ice % of total: [{summary['R_ice_pct_range'][0]:.1f}, {summary['R_ice_pct_range'][1]:.1f}]")
    print(f"  roll reduction: [{summary['roll_reduction_range'][0]:.1f}, {summary['roll_reduction_range'][1]:.1f}]%")
    print(f"  energy residual max: {summary['energy_residual_max']:.4f}%")


if __name__ == "__main__":
    main()
