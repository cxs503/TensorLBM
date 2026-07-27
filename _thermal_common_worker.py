#!/usr/bin/env python3
"""Thermal/conjugate heat transfer common-module worker.

Runs ONE benchmark on ONE SDAA card.  Launched 4× in parallel by the
launcher script ``_thermal_common_launcher.py``.

Benchmarks (SDAA cards 28-31):
  28: thermal_cavity   — de Vahl Davis, Ra=1e4, Pr=0.71, Nu_ref≈2.0
  29: heated_cylinder  — Re=200, Pr=0.71, Nu_ref≈6.5
  30: conjugate_ht     — channel + heated block, flux continuity <10%
  31: rayleigh_benard  — Ra=1e4 > Ra_c=1708, detect convection

Usage:
  python _thermal_common_worker.py <benchmark> <device_id> [output_path]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch_sdaa  # noqa: F401

from tensorlbm.thermal_common import (
    run_thermal_cavity_common,
    run_heated_cylinder_common,
    run_conjugate_ht_common,
    run_rayleigh_benard_common,
)

BENCHMARKS = {
    "thermal_cavity": {
        "fn": run_thermal_cavity_common,
        "kwargs": {"nx": 100, "ny": 100, "nz": 4, "Ra": 1e4, "Pr": 0.71, "n_steps": 8000},
        "ref_key": "nusselt_ref",
        "val_key": "nusselt",
        "tol": 0.20,
        "desc": "de Vahl Davis thermal cavity, Ra=1e4, Pr=0.71",
    },
    "heated_cylinder": {
        "fn": run_heated_cylinder_common,
        "kwargs": {"D": 48.0, "Re": 200.0, "Pr": 0.71, "n_steps": 6000},
        "ref_key": "nusselt_ref",
        "val_key": "nusselt",
        "tol": 0.20,
        "desc": "Heated cylinder, Re=200, Pr=0.71",
    },
    "conjugate_ht": {
        "fn": run_conjugate_ht_common,
        "kwargs": {"nx": 200, "ny": 80, "nz": 4, "Pr": 0.71, "n_steps": 6000},
        "ref_key": None,
        "val_key": "flux_continuity_error",
        "tol": 0.10,
        "desc": "Conjugate HT, channel + heated block",
    },
    "rayleigh_benard": {
        "fn": run_rayleigh_benard_common,
        "kwargs": {"nx": 100, "ny": 50, "nz": 4, "Ra": 1e4, "Pr": 0.71, "n_steps": 8000},
        "ref_key": None,
        "val_key": "convection_detected",
        "tol": None,
        "desc": "Rayleigh-Benard, Ra=1e4 > Ra_c=1708",
    },
}


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <benchmark> <device_id> [output_path]")
        print(f"Available: {list(BENCHMARKS.keys())}")
        sys.exit(1)

    benchmark_name = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if benchmark_name not in BENCHMARKS:
        print(f"Unknown benchmark: {benchmark_name}")
        print(f"Available: {list(BENCHMARKS.keys())}")
        sys.exit(1)

    cfg = BENCHMARKS[benchmark_name]
    device = f"sdaa:{device_id}"
    torch.sdaa.set_device(device_id)
    tag = f"[{benchmark_name} SDAA:{device_id}]"

    print(f"{tag} Starting: {cfg['desc']}", flush=True)
    print(f"{tag} device={device}, kwargs={cfg['kwargs']}", flush=True)
    t0 = time.time()

    result = cfg["fn"](device=device, **cfg["kwargs"])
    elapsed = time.time() - t0

    # Extract T_field for separate saving
    T_field = result.pop("T_field", None)

    # Evaluate pass/fail
    val = result.get(cfg["val_key"], None)
    ref = result.get(cfg["ref_key"], None) if cfg["ref_key"] else None
    tol = cfg["tol"]

    if cfg["val_key"] == "convection_detected":
        passed = bool(val)
        metric_str = f"convection_detected={val}"
    elif cfg["val_key"] == "flux_continuity_error":
        passed = val is not None and val < tol
        metric_str = f"flux_continuity_error={val:.4f} (target < {tol})"
    else:
        if ref is not None and val is not None:
            rel_err = abs(val - ref) / abs(ref) if abs(ref) > 1e-12 else abs(val - ref)
            passed = rel_err < tol
            metric_str = f"Nu={val:.4f}, ref={ref:.4f}, rel_err={rel_err:.2%} (tol={tol:.0%})"
        else:
            passed = False
            metric_str = "missing reference"

    print(f"{tag} DONE in {elapsed:.1f}s — {metric_str}", flush=True)
    print(f"{tag} PASS={passed}", flush=True)

    # Build output
    output = {
        "benchmark": benchmark_name,
        "device": device,
        "elapsed_s": round(elapsed, 1),
        "passed": passed,
        "metric": metric_str,
        "result": {k: v for k, v in result.items() if k != "T_field"},
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"{tag} Results saved to {out_p}", flush=True)

        # Save T_field as npy
        if T_field is not None:
            import numpy as np
            npy_path = out_p.with_suffix(".npy")
            np.save(npy_path, T_field.numpy())
            print(f"{tag} T_field saved to {npy_path}", flush=True)

    print(f"{tag} Summary: {json.dumps(output, default=str)}", flush=True)


if __name__ == "__main__":
    main()
