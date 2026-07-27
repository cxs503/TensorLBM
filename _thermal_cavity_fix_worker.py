#!/usr/bin/env python3
"""Thermal cavity fix — SDAA:6.

Fixes:
  1. Nusselt sign bug fixed in thermal_common.py (remove extra minus)
  2. Use 5000 steps (sufficient for Ra=1e4 steady state)
  3. Ra=1e4 for stability

Target: Nu > 1.5 (ref=2.0, <25%)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch_sdaa  # noqa: F401

from tensorlbm.thermal_common import run_thermal_cavity_common


def main():
    device_id = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    tag = f"[ThermalCavity-fix SDAA:{device_id}]"

    torch.sdaa.set_device(device_id)
    device = f"sdaa:{device_id}"

    print(f"{tag} Starting: de Vahl Davis thermal cavity, Ra=1e4, Pr=0.71", flush=True)
    t0 = time.time()

    result = run_thermal_cavity_common(
        device=device,
        nx=100, ny=100, nz=4,
        Ra=1e4, Pr=0.71,
        n_steps=5000,
    )

    elapsed = time.time() - t0
    nu_val = result["nusselt"]
    nu_ref = result["nusselt_ref"]
    rel_err = abs(nu_val - nu_ref) / abs(nu_ref) if abs(nu_ref) > 1e-12 else abs(nu_val - nu_ref)
    passed = rel_err < 0.25

    print(f"{tag} DONE in {elapsed:.1f}s — Nu={nu_val:.4f}, ref={nu_ref:.4f}, "
          f"rel_err={rel_err:.2%} (tol=25%)", flush=True)
    print(f"{tag} PASS={passed}", flush=True)

    output = {
        "benchmark": "thermal_cavity_fixed",
        "device": device,
        "elapsed_s": round(elapsed, 1),
        "passed": passed,
        "metric": f"Nu={nu_val:.4f}, ref={nu_ref:.4f}, rel_err={rel_err:.2%}",
        "result": {k: v for k, v in result.items() if k != "T_field"},
    }

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w") as f:
            json.dump(output, f, indent=2, default=str)
        print(f"{tag} Results saved to {out_p}", flush=True)

    print(f"{tag} Summary: {json.dumps(output, default=str)}", flush=True)


if __name__ == "__main__":
    main()
