#!/usr/bin/env python
"""Production runner: 7 collision families × 2 lattices × wall_function.

Launches all 11 combinations in parallel on separate SDAA cards.
Grid sizes are tuned per collision family to fit within 16GB SDAA memory
(the SDAA runtime reserves ~3GB beyond actual allocations).

  D3Q27 (sdaa:0-6): Cumulant, BGK, MRT, TRT, CM, KBC, RLBM
  D3Q19 (sdaa:7-10): Cumulant, BGK, MRT, RLBM

Each run: Re=2e6, u_in=0.06, y_val=0.5, wall_law="log", 1000 steps.

Usage:
    PYTHONPATH=src python examples/dg_suboff_wallfn_fullgrid.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Combination specifications: (lattice, collision, device, nx, ny, nz, hull)
# Grid sizes tuned per collision family for 16GB SDAA memory constraint.
# --------------------------------------------------------------------------- #

SPECS: list[dict] = [
    # D3Q27 (7 cards: sdaa:0-6)
    {"lattice": "D3Q27", "collision": "CUMULANT", "device": "sdaa:0",
     "nx": 384, "ny": 192, "nz": 192, "hull_length": 192.0},
    {"lattice": "D3Q27", "collision": "BGK", "device": "sdaa:1",
     "nx": 384, "ny": 192, "nz": 192, "hull_length": 192.0},
    {"lattice": "D3Q27", "collision": "MRT", "device": "sdaa:2",
     "nx": 320, "ny": 160, "nz": 160, "hull_length": 160.0},
    {"lattice": "D3Q27", "collision": "TRT", "device": "sdaa:3",
     "nx": 320, "ny": 160, "nz": 160, "hull_length": 160.0},
    {"lattice": "D3Q27", "collision": "CM", "device": "sdaa:4",
     "nx": 256, "ny": 128, "nz": 128, "hull_length": 128.0},
    {"lattice": "D3Q27", "collision": "KBC", "device": "sdaa:5",
     "nx": 256, "ny": 128, "nz": 128, "hull_length": 128.0},
    {"lattice": "D3Q27", "collision": "RLBM", "device": "sdaa:6",
     "nx": 384, "ny": 192, "nz": 192, "hull_length": 192.0},
    # D3Q19 (4 cards: sdaa:7-10)
    {"lattice": "D3Q19", "collision": "CUMULANT", "device": "sdaa:7",
     "nx": 384, "ny": 192, "nz": 192, "hull_length": 192.0},
    {"lattice": "D3Q19", "collision": "BGK", "device": "sdaa:8",
     "nx": 384, "ny": 192, "nz": 192, "hull_length": 192.0},
    {"lattice": "D3Q19", "collision": "MRT", "device": "sdaa:9",
     "nx": 320, "ny": 160, "nz": 160, "hull_length": 160.0},
    {"lattice": "D3Q19", "collision": "RLBM", "device": "sdaa:10",
     "nx": 384, "ny": 192, "nz": 192, "hull_length": 192.0},
]


def _run_single(spec: dict) -> dict:
    """Run a single combination and return the artifact dict."""
    from tensorlbm.suboff_wallfn_fullgrid_runner import (
        SuboffWallFnFullGridConfig,
        run_suboff_wallfn_fullgrid,
    )
    cfg = SuboffWallFnFullGridConfig(
        re=2_000_000.0,
        lattice=spec["lattice"],
        collision=spec["collision"],
        nx=spec["nx"],
        ny=spec["ny"],
        nz=spec["nz"],
        n_steps=1000,
        u_in=0.06,
        hull_length=spec["hull_length"],
        device=spec["device"],
        y_val=0.5,
        wall_law="log",
    )
    t0 = time.time()
    artifact = run_suboff_wallfn_fullgrid(cfg)
    artifact["runtime_seconds"] = time.time() - t0
    artifact["device_id"] = spec["device"]
    return artifact


def main() -> None:
    out_dir = Path("artifacts/suboff_wallfn_fullgrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Launching {len(SPECS)} combinations in parallel...")
    for s in SPECS:
        cells = s["nx"] * s["ny"] * s["nz"] / 1e6
        print(f"  {s['lattice']:6s} {s['collision']:8s} → {s['device']}  "
              f"grid={s['nx']}×{s['ny']}×{s['nz']} ({cells:.1f}M) "
              f"hull={s['hull_length']:.0f}")

    # Run all combinations in parallel using multiprocessing
    # Each process pins to its own SDAA card via the device string
    ctx = mp.get_context("spawn")  # spawn is required for SDAA init
    with ctx.Pool(processes=len(SPECS)) as pool:
        results = pool.map(_run_single, SPECS)

    # Write individual artifacts
    for spec, artifact in zip(SPECS, results):
        name = f"{spec['lattice']}_{spec['collision']}".lower()
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(artifact, sort_keys=True, indent=2))
        ct = artifact["Ct_total"]
        finite = artifact["finite"]
        steps = artifact["steps_completed"]
        runtime = artifact.get("runtime_seconds", 0)
        print(f"  {name:25s}: Ct_total={ct:.6f}  finite={finite}  "
              f"steps={steps}  time={runtime:.1f}s")

    # Write combined summary artifact
    summary = {
        "schema": "tensorlbm.suboff-wallfn-fullgrid-summary/v1",
        "status": "diagnostic_only",
        "physical_validation": False,
        "Re": 2_000_000.0,
        "u_in": 0.06,
        "wall_function": "log-law (κ=0.41, B=5.0, y_val=0.5)",
        "reference_Ct": 0.00405,
        "reference_source": "ITTC-1957",
        "n_combinations": len(SPECS),
        "combinations": [],
    }
    for spec, artifact in zip(SPECS, results):
        summary["combinations"].append({
            "lattice": spec["lattice"],
            "collision": spec["collision"],
            "device": spec["device"],
            "grid": {"nx": spec["nx"], "ny": spec["ny"], "nz": spec["nz"]},
            "hull_length": spec["hull_length"],
            "Ct_fric": artifact["Ct_fric"],
            "Ct_pres": artifact["Ct_pres"],
            "Ct_total": artifact["Ct_total"],
            "finite": artifact["finite"],
            "steps_completed": artifact["steps_completed"],
            "runtime_seconds": artifact.get("runtime_seconds", 0),
        })
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2))
    print(f"\nSummary written to {summary_path}")
    print(f"Individual artifacts in {out_dir}/")


if __name__ == "__main__":
    main()
