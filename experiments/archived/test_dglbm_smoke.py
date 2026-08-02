#!/usr/bin/env python3
"""Quick smoke test v2: DG-LBM on CPU, find stable parameters."""
from __future__ import annotations
import json, os, sys, time, traceback
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import shutil

import torch
from tensorlbm.dg_lbm import DGLBMSuboffConfig, _run_suboff_real_dg

U_IN = 0.06
OUT = "/tmp/dglbm_smoke2"

def run(nx, ny, nz, hl, re, nstep, dev, dg_band=3, dg_sub=8, oi=10):
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    print(f"\nDG-LBM: {nx}³, Re={re}, dev={dev}, dg_sub={dg_sub}, steps={nstep}")
    try:
        cfg = DGLBMSuboffConfig(
            nx=nx, ny=ny, nz=nz, hull_length=hl, u_in=U_IN, re=re,
            hull_type="bare_hull", dg_band=dg_band, dg_substeps=dg_sub,
            use_real_dg=True, n_steps=nstep, output_interval=oi,
            device=dev, output_root=Path(OUT), overwrite=True,
            run_name="smoke",
        )
        tau_dg = cfg.tau - 0.5
        print(f"  tau_lbm={cfg.tau:.4f}, tau_dg={tau_dg:.4f}, nu={cfg.nu:.4e}")
        t0 = time.time()
        rd = _run_suboff_real_dg(cfg)
        elapsed = time.time() - t0

        meta = json.loads((Path(rd)/"run_metadata.json").read_text())
        diags = meta.get("diagnostics", [])
        has_nan = any("nan" in str(d.get("mass","")) for d in diags)
        drag = meta.get("drag_force_lu")

        status = "NaN" if has_nan else ("OK" if drag is not None else "??")
        print(f"  Done {elapsed:.0f}s [{status}] drag={drag}")

        for d in diags:
            print(f"    step={d['step']:4d} mass={d.get('mass','?')} max|u|={d.get('max_speed','?')}")
        return not has_nan, elapsed
    except Exception as e:
        traceback.print_exc()
        return False, 0

if __name__ == "__main__":
    # Test 1: 32³, Re=50 (tau_dg=0.0576), 4 substeps
    run(32, 32, 32, 16.0, 50.0, 20, "cpu", dg_band=3, dg_sub=4, oi=10)

    # Test 2: 32³, Re=100 (tau_dg=0.0288), 8 substeps  
    run(32, 32, 32, 16.0, 100.0, 20, "cpu", dg_band=3, dg_sub=8, oi=10)

    # Test 3: 32³, Re=200 (tau_dg=0.0144), 16 substeps
    run(32, 32, 32, 16.0, 200.0, 20, "cpu", dg_band=3, dg_sub=16, oi=10)
