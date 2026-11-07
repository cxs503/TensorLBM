#!/usr/bin/env python3
"""Cluster-ready wave tank benchmark runner.

Runs a single wave tank case and outputs results as JSON.
Designed for cluster execution (loong64).
"""

import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.wave_tank_2d import (
    StokesWaveParams,
    WaveTankConfig,
    WaveTank2D,
)


def run_case(kh, ka, nx, ny, depth, n_periods, enable_absorption, label):
    """Run a single wave tank case."""
    k = kh / depth
    amplitude = ka / k
    wp = StokesWaveParams(amplitude=amplitude, wavenumber=k, gravity=1e-4, depth=depth)
    period_steps = wp.period
    n_steps = int(n_periods * period_steps) + 1
    record_interval = max(1, int(period_steps / 20))

    cfg = WaveTankConfig(
        nx=nx, ny=ny, water_depth=depth,
        wave_amplitude=amplitude, wave_kh=kh, gravity=1e-4,
        tau=0.6, n_steps=n_steps, record_interval=record_interval,
        enable_absorption=enable_absorption,
    )

    print(f"[{label}] kh={kh}, ka={ka}, nx={nx}, ny={ny}, "
          f"steps={n_steps}, abs={enable_absorption}")

    tank = WaveTank2D(cfg)
    t0 = time.time()
    tank.run()
    elapsed = time.time() - t0

    results = tank.get_results()
    results["label"] = label
    results["elapsed_s"] = round(elapsed, 1)
    results["n_steps"] = n_steps
    results["enable_absorption"] = enable_absorption

    # Save probe data for plotting
    results["probe1_data"] = tank.probe_history_1
    results["probe2_data"] = tank.probe_history_2
    results["step_list"] = tank.step_list
    results["mass_history"] = tank.mass_history

    print(f"  -> elapsed={elapsed:.1f}s, "
          f"disp_err={results['dispersion_error']:.4%}, "
          f"Kr={results['reflection']['Kr']:.4f}, "
          f"mass_drift={results['mass_drift_relative']:.4%}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kh", type=float, default=1.5)
    parser.add_argument("--ka", type=float, default=0.02)
    parser.add_argument("--nx", type=int, default=400)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--depth", type=float, default=50.0)
    parser.add_argument("--periods", type=float, default=5.0)
    parser.add_argument("--no-abs", action="store_true")
    parser.add_argument("--label", default="case")
    parser.add_argument("--output", default="/tmp/wave_tank_result.json")
    args = parser.parse_args()

    results = run_case(
        kh=args.kh, ka=args.ka, nx=args.nx, ny=args.ny,
        depth=args.depth, n_periods=args.periods,
        enable_absorption=not args.no_abs, label=args.label,
    )

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
