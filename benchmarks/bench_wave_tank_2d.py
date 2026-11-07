#!/usr/bin/env python3
"""Benchmark driver for 2D numerical wave tank with Stokes second-order waves.

Runs validation cases for:
- C1: Dispersion relation error < 1% for kh ∈ {0.7, 1.5, 3.0}
- C2: Harmonic amplitude errors (1st < 3%, 2nd < 10%) for ka ∈ {0.02, 0.08}
- C3: Reflection coefficient < 5% with absorption, > 30% without
- C4: Mass conservation drift < 0.5%/period × 5 periods
- C5: Grid convergence (Δx halving → amplitude change < 3%)

Usage:
    python benchmarks/bench_wave_tank_2d.py [--kh KH] [--ka KA] [--nx NX]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorlbm.wave_tank_2d import (
    StokesWaveParams,
    WaveTankConfig,
    WaveTank2D,
    harmonic_analysis,
    two_point_reflection_coefficient,
)


def run_wave_tank_case(
    kh: float,
    ka: float,
    nx: int = 400,
    ny: int = 100,
    water_depth: float = 50.0,
    gravity: float = 1e-4,
    tau: float = 0.6,
    n_periods: float = 5.0,
    enable_absorption: bool = True,
    device: str = "cpu",
) -> dict:
    """Run a single wave tank case and return results."""
    # Compute wave parameters
    k = kh / water_depth
    amplitude = ka / k
    wave_params = StokesWaveParams(
        amplitude=amplitude,
        wavenumber=k,
        gravity=gravity,
        depth=water_depth,
    )

    # Compute number of steps for n_periods
    period_steps = wave_params.period
    n_steps = int(n_periods * period_steps) + 1
    record_interval = max(1, int(period_steps / 20))  # 20 samples per period

    # Configure tank
    cfg = WaveTankConfig(
        nx=nx,
        ny=ny,
        water_depth=water_depth,
        wave_amplitude=amplitude,
        wave_kh=kh,
        gravity=gravity,
        tau=tau,
        n_steps=n_steps,
        device=device,
        record_interval=record_interval,
        enable_absorption=enable_absorption,
    )

    print(f"  Wave: kh={kh:.2f}, ka={ka:.4f}, λ={wave_params.wavelength:.1f}, "
          f"T={wave_params.period:.0f} steps")
    print(f"  Domain: {nx}×{ny}, depth={water_depth}, τ={tau}")
    print(f"  Gen zone: {cfg.gen_zone_cells} cells, "
          f"Abs zone: {cfg.abs_zone_cells} cells")
    print(f"  Steps: {n_steps} ({n_periods:.1f} periods), "
          f"record every {record_interval} steps")

    # Run
    tank = WaveTank2D(cfg)
    t0 = time.time()
    tank.run()
    elapsed = time.time() - t0

    # Get results
    results = tank.get_results()
    results["elapsed_seconds"] = elapsed
    results["n_steps"] = n_steps
    results["n_periods"] = n_periods
    results["enable_absorption"] = enable_absorption

    # Additional diagnostics
    results["probe1_samples"] = len(tank.probe_history_1)
    results["probe2_samples"] = len(tank.probe_history_2)

    return results


def check_acceptance_criteria(results: dict, case_label: str) -> dict:
    """Check acceptance criteria for a single case."""
    checks = {}

    # C1: Dispersion relation error < 1%
    disp_error = results.get("dispersion_error", 1.0)
    checks["C1_dispersion"] = {
        "value": disp_error,
        "threshold": 0.01,
        "pass": disp_error < 0.01,
    }

    # C2: Harmonic amplitude errors
    harmonics = results.get("harmonics", {})
    wp = results.get("wave_params", {})
    a1_measured = harmonics.get("A1", 0.0)
    a2_measured = harmonics.get("A2", 0.0)
    a1_expected = wp.get("amplitude", 1.0)
    a2_expected = wp.get("second_order_amplitude", 0.0)

    if a1_expected > 0:
        a1_error = abs(a1_measured - a1_expected) / a1_expected
    else:
        a1_error = 0.0

    if a2_expected > 0:
        a2_error = abs(a2_measured - a2_expected) / a2_expected
    else:
        a2_error = 0.0 if a2_measured < 0.1 else 1.0

    checks["C2_first_harmonic"] = {
        "value": a1_error,
        "threshold": 0.03,
        "pass": a1_error < 0.03,
    }
    checks["C2_second_harmonic"] = {
        "value": a2_error,
        "threshold": 0.10,
        "pass": a2_error < 0.10,
    }

    # C3: Reflection coefficient
    refl = results.get("reflection", {})
    kr = refl.get("Kr", 1.0)
    checks["C3_reflection"] = {
        "value": kr,
        "threshold": 0.05,
        "pass": kr < 0.05,
    }

    # C4: Mass conservation
    mass_drift = results.get("mass_drift_relative", 1.0)
    n_periods = results.get("n_periods", 1.0)
    drift_per_period = mass_drift / max(n_periods, 1.0)
    checks["C4_mass_conservation"] = {
        "value": drift_per_period,
        "threshold": 0.005,
        "pass": drift_per_period < 0.005,
    }

    return checks


def run_dispersion_scan(device: str = "cpu") -> dict:
    """C1: Scan kh values for dispersion relation verification."""
    print("\n" + "=" * 60)
    print("C1: Dispersion Relation Verification")
    print("=" * 60)

    kh_values = [0.7, 1.5, 3.0]
    results = {}

    for kh in kh_values:
        print(f"\n--- kh = {kh} ---")
        res = run_wave_tank_case(
            kh=kh, ka=0.02, nx=400, ny=100,
            water_depth=50.0, n_periods=5.0,
            device=device,
        )
        checks = check_acceptance_criteria(res, f"kh={kh}")
        results[f"kh_{kh}"] = {
            "results": res,
            "checks": checks,
        }

        c1 = checks["C1_dispersion"]
        print(f"  Dispersion error: {c1['value']:.4%} "
              f"({'PASS' if c1['pass'] else 'FAIL'}, threshold < 1%)")

    return results


def run_harmonic_scan(device: str = "cpu") -> dict:
    """C2: Scan ka values for harmonic amplitude verification."""
    print("\n" + "=" * 60)
    print("C2: Harmonic Amplitude Verification")
    print("=" * 60)

    ka_values = [0.02, 0.08]
    results = {}

    for ka in ka_values:
        print(f"\n--- ka = {ka} ---")
        res = run_wave_tank_case(
            kh=1.5, ka=ka, nx=400, ny=100,
            water_depth=50.0, n_periods=5.0,
            device=device,
        )
        checks = check_acceptance_criteria(res, f"ka={ka}")
        results[f"ka_{ka}"] = {
            "results": res,
            "checks": checks,
        }

        c2a = checks["C2_first_harmonic"]
        c2b = checks["C2_second_harmonic"]
        print(f"  1st harmonic error: {c2a['value']:.4%} "
              f"({'PASS' if c2a['pass'] else 'FAIL'}, threshold < 3%)")
        print(f"  2nd harmonic error: {c2b['value']:.4%} "
              f"({'PASS' if c2b['pass'] else 'FAIL'}, threshold < 10%)")

    return results


def run_reflection_test(device: str = "cpu") -> dict:
    """C3: Test reflection coefficient with and without absorption."""
    print("\n" + "=" * 60)
    print("C3: Reflection Coefficient Test")
    print("=" * 60)

    results = {}

    # With absorption
    print("\n--- With absorption ---")
    res_abs = run_wave_tank_case(
        kh=1.5, ka=0.02, nx=400, ny=100,
        water_depth=50.0, n_periods=5.0,
        enable_absorption=True, device=device,
    )
    checks_abs = check_acceptance_criteria(res_abs, "with_absorption")
    results["with_absorption"] = {
        "results": res_abs,
        "checks": checks_abs,
    }
    kr_abs = checks_abs["C3_reflection"]["value"]
    print(f"  Kr = {kr_abs:.4f} "
          f"({'PASS' if kr_abs < 0.05 else 'FAIL'}, threshold < 5%)")

    # Without absorption (negative control)
    print("\n--- Without absorption (negative control) ---")
    res_noabs = run_wave_tank_case(
        kh=1.5, ka=0.02, nx=400, ny=100,
        water_depth=50.0, n_periods=5.0,
        enable_absorption=False, device=device,
    )
    checks_noabs = check_acceptance_criteria(res_noabs, "without_absorption")
    results["without_absorption"] = {
        "results": res_noabs,
        "checks": checks_noabs,
    }
    kr_noabs = checks_noabs["C3_reflection"]["value"]
    print(f"  Kr = {kr_noabs:.4f} "
          f"({'PASS' if kr_noabs > 0.30 else 'FAIL'}, threshold > 30%)")

    return results


def run_mass_conservation_test(device: str = "cpu") -> dict:
    """C4: Mass conservation over 5 periods."""
    print("\n" + "=" * 60)
    print("C4: Mass Conservation Test")
    print("=" * 60)

    res = run_wave_tank_case(
        kh=1.5, ka=0.02, nx=400, ny=100,
        water_depth=50.0, n_periods=5.0,
        device=device,
    )
    checks = check_acceptance_criteria(res, "mass_conservation")
    c4 = checks["C4_mass_conservation"]
    print(f"  Mass drift per period: {c4['value']:.4%} "
          f"({'PASS' if c4['pass'] else 'FAIL'}, threshold < 0.5%)")

    return {"results": res, "checks": checks}


def run_grid_convergence(device: str = "cpu") -> dict:
    """C5: Grid convergence study (Δx halving)."""
    print("\n" + "=" * 60)
    print("C5: Grid Convergence Study")
    print("=" * 60)

    # Three resolutions: coarse, medium, fine
    nx_values = [200, 400, 800]
    ny_values = [50, 100, 200]
    results = {}

    amplitudes = []
    for nx, ny in zip(nx_values, ny_values):
        print(f"\n--- nx={nx}, ny={ny} ---")
        res = run_wave_tank_case(
            kh=1.5, ka=0.02, nx=nx, ny=ny,
            water_depth=50.0, n_periods=3.0,
            device=device,
        )
        a1 = res.get("harmonics", {}).get("A1", 0.0)
        amplitudes.append(a1)
        results[f"nx_{nx}"] = {
            "results": res,
            "amplitude_A1": a1,
        }
        print(f"  A1 = {a1:.6f}")

    # Check convergence: |A1_fine - A1_medium| / A1_fine < 3%
    if len(amplitudes) >= 3 and amplitudes[2] > 0:
        change = abs(amplitudes[2] - amplitudes[1]) / amplitudes[2]
    else:
        change = 1.0

    print(f"\n  Grid convergence: ΔA1/A1 = {change:.4%} "
          f"({'PASS' if change < 0.03 else 'FAIL'}, threshold < 3%)")

    results["convergence_change"] = change
    results["pass"] = change < 0.03

    return results


def generate_report(all_results: dict, output_path: str):
    """Generate JSON report with all results."""
    report = {
        "benchmark": "wave_tank_2d",
        "description": "2D numerical wave tank with Stokes second-order waves",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": all_results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Wave tank 2D benchmark")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output", default="runs/wave_tank_2d_results.json",
                        help="Output JSON path")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test (smaller domains)")
    args = parser.parse_args()

    print("=" * 60)
    print("TensorLBM 2D Numerical Wave Tank Benchmark")
    print("Stokes Second-Order Wave Generation + Absorption")
    print("=" * 60)

    all_results = {}

    # Run all acceptance criteria tests
    all_results["C1_dispersion"] = run_dispersion_scan(args.device)
    all_results["C2_harmonics"] = run_harmonic_scan(args.device)
    all_results["C3_reflection"] = run_reflection_test(args.device)
    all_results["C4_mass"] = run_mass_conservation_test(args.device)
    all_results["C5_convergence"] = run_grid_convergence(args.device)

    # Generate report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    generate_report(all_results, args.output)

    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
