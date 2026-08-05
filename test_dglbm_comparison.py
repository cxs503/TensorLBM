#!/usr/bin/env python3
"""DG-LBM hybrid vs wall function comparison on SUBOFF 160³.

Runs:
  1. DG-LBM hybrid (use_real_dg=True) on sdaa:4
  2. Wall function baseline on sdaa:5

Outputs /tmp/dglbm_results.json with Ct at steps 200, 500, 1000.
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from tensorlbm.dg_lbm import (
    DGLBMSuboffConfig,
    _run_suboff_real_dg,
    _run_suboff_wall_function,
    build_suboff_mask,
)
from tensorlbm.suboff_resistance import _voxel_wetted_area


U_IN = 0.06
OUTPUT_ROOT = "/tmp/dglbm_test_outputs"


def compute_ct(drag_lu, u_in, S, rho=1.0):
    return drag_lu / (0.5 * rho * u_in * u_in * S)


def get_wetted_area(nx, ny, nz, hull_length):
    cx, cy, cz = nx * 0.35, ny * 0.5, nz * 0.5
    obstacle, _ = build_suboff_mask(
        "bare_hull", nx, ny, nz, cx, cy, cz, hull_length, device="cpu",
    )
    return _voxel_wetted_area(obstacle, 1.0)


def parse_drag_from_log(run_dir):
    log = Path(run_dir) / "run.log"
    if not log.exists():
        return {}
    out = {}
    for line in log.read_text().splitlines():
        if "drag=" in line and "step=" in line:
            try:
                parts = line.strip().split()
                sp = [p for p in parts if p.startswith("step=")]
                dp = [p for p in parts if p.startswith("drag=")]
                if sp and dp:
                    out[int(sp[0].split("=")[1])] = float(dp[0].split("=")[1])
            except Exception:
                pass
    return out


def parse_ct_from_log(run_dir):
    log = Path(run_dir) / "run.log"
    if not log.exists():
        return {}
    out = {}
    for line in log.read_text().splitlines():
        if "Ct_tot=" in line and "step=" in line:
            try:
                parts = line.strip().split()
                sp = [p for p in parts if p.startswith("step=")]
                if not sp:
                    continue
                step = int(sp[0].split("=")[1])
                ct = {"Ct_fric": None, "Ct_pres": None, "Ct_tot": None}
                for p in parts:
                    if p.startswith("Ct_fric="):
                        ct["Ct_fric"] = float(p.split("=")[1])
                    elif p.startswith("Ct_pres="):
                        ct["Ct_pres"] = float(p.split("=")[1])
                    elif p.startswith("Ct_tot="):
                        ct["Ct_tot"] = float(p.split("=")[1])
                out[step] = ct
            except Exception:
                pass
    return out


def closest(data, target):
    cand = {k: v for k, v in data.items() if k <= target}
    return data[max(cand)] if cand else None


def run_dglbm(nx, ny, nz, hull_length, re, n_steps, device, dg_band=3, dg_substeps=32, name=None):
    print(f"\n--- DG-LBM: {nx}×{ny}×{nz}, Re={re}, dev={device} ---")
    try:
        cfg = DGLBMSuboffConfig(
            nx=nx, ny=ny, nz=nz, hull_length=hull_length,
            u_in=U_IN, re=re, hull_type="bare_hull",
            dg_band=dg_band, dg_substeps=dg_substeps,
            use_real_dg=True, n_steps=n_steps,
            output_interval=100, device=device,
            output_root=Path(OUTPUT_ROOT),
            overwrite=True,
            run_name=name or f"dglbm_nx{nx}_re{int(re)}",
        )
        print(f"  τ={cfg.tau:.4f}, τ_dg={cfg.tau-0.5:.4f}, ν={cfg.nu:.3e}")
        t0 = time.time()
        run_dir = _run_suboff_real_dg(cfg)
        elapsed = time.time() - t0
        print(f"  ✅ Done in {elapsed:.0f}s")

        meta = json.loads((Path(run_dir) / "run_metadata.json").read_text()) if (Path(run_dir) / "run_metadata.json").exists() else {}
        S = get_wetted_area(nx, ny, nz, hull_length)
        drag_final = meta.get("drag_force_lu")
        drag_by_step = parse_drag_from_log(run_dir)

        return {
            "status": "ok", "run_dir": str(run_dir), "elapsed_s": elapsed,
            "re": re, "tau": cfg.tau, "tau_dg": cfg.tau - 0.5, "nu": cfg.nu,
            "wetted_area": S, "drag_lu_final": drag_final,
            "Ct_final": compute_ct(drag_final, U_IN, S) if drag_final else None,
            "drag_by_step": {str(k): v for k, v in sorted(drag_by_step.items())},
        }
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def run_wallfn(nx, ny, nz, hull_length, re, n_steps, device, cs=0.05, name=None):
    print(f"\n--- WallFn: {nx}×{ny}×{nz}, Re={re:.0e}, dev={device} ---")
    try:
        cfg = DGLBMSuboffConfig(
            nx=nx, ny=ny, nz=nz, hull_length=hull_length,
            u_in=U_IN, re=re, hull_type="bare_hull",
            use_wall_function=True, smagorinsky_cs=cs,
            n_steps=n_steps, output_interval=100,
            device=device, output_root=Path(OUTPUT_ROOT),
            overwrite=True,
            run_name=name or f"wallfn_nx{nx}_re{int(re):.0e}",
        )
        print(f"  τ={cfg.tau:.4f}, ν={cfg.nu:.3e}")
        t0 = time.time()
        run_dir = _run_suboff_wall_function(cfg)
        elapsed = time.time() - t0
        print(f"  ✅ Done in {elapsed:.0f}s")

        meta = json.loads((Path(run_dir) / "run_metadata.json").read_text()) if (Path(run_dir) / "run_metadata.json").exists() else {}
        ct_by_step = parse_ct_from_log(run_dir)

        return {
            "status": "ok", "run_dir": str(run_dir), "elapsed_s": elapsed,
            "re": re, "tau": cfg.tau, "nu": cfg.nu, "Cs": cs,
            "Ct_total_final": meta.get("Ct_total"),
            "ct_by_step": {str(k): v for k, v in sorted(ct_by_step.items())},
        }
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


def main():
    import shutil
    # Clean any previous runs
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT)
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    NX, NY, NZ = 160, 160, 160
    HL = 64.0
    N_STEPS = 1000
    STEPS = [200, 500, 1000]

    S_REF = get_wetted_area(NX, NY, NZ, HL)
    print(f"Wetted area: {S_REF:.1f} lu²")

    # ---- PHASE 1: DG-LBM stability quick test (200 steps) ----
    print("=" * 60)
    print("PHASE 1: DG-LBM quick stabilization test (200 steps)")
    print("=" * 60)

    # Try Re=200 first (τ=0.5576, τ_dg=0.0576 — stable)
    dg_result = run_dglbm(NX, NY, NZ, HL, 200.0, 200, "sdaa:4", dg_band=3, dg_substeps=32,
                          name="dglbm_stab_re200")

    if dg_result["status"] != "ok":
        # Try CPU
        print("SDAA failed, trying CPU...")
        dg_result = run_dglbm(64, 64, 64, 32.0, 200.0, 200, "cpu", dg_band=3, dg_substeps=32,
                              name="dglbm_cpu_re200")
        if dg_result["status"] != "ok":
            print("⚠️  DG-LBM failed on both SDAA and CPU")
            # Write failed result and exit
            out = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "grid": f"{NX}×{NY}×{NZ}",
                "dg_lbm": dg_result,
                "wall_function": {"status": "not_run"},
            }
            Path("/tmp/dglbm_results.json").write_text(json.dumps(out, indent=2, default=str))
            return

    best_re = dg_result["re"]
    print(f"✅ DG-LBM stable at Re={best_re}")

    # ---- PHASE 2: Full DG-LBM run ----
    print("\n" + "=" * 60)
    print(f"PHASE 2: Full DG-LBM run at Re={best_re}, {N_STEPS} steps")
    print("=" * 60)
    dg_full = run_dglbm(NX, NY, NZ, HL, float(best_re), N_STEPS, "sdaa:4",
                        dg_band=3, dg_substeps=32, name=f"dglbm_full_re{int(best_re)}")

    # ---- PHASE 3: Wall function baseline ----
    print("\n" + "=" * 60)
    print(f"PHASE 3: Wall function baseline, Re=2e6, {N_STEPS} steps")
    print("=" * 60)
    wf_result = run_wallfn(NX, NY, NZ, HL, 2e6, N_STEPS, "sdaa:5", cs=0.05,
                           name="wallfn_baseline")

    # ---- Compile results ----
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "grid": f"{NX}×{NY}×{NZ}",
        "hull": "SUBOFF bare_hull",
        "hull_length_lu": HL,
        "u_in": U_IN,
        "wetted_area_lu2": S_REF,
        "n_steps": N_STEPS,
        "compare_steps": STEPS,
    }

    # DG-LBM
    if dg_full.get("status") == "ok":
        ds = dg_full
        drag_map = {int(k): v for k, v in ds.get("drag_by_step", {}).items()}
        dg_section = {
            "status": "ok",
            "re": ds["re"], "tau": ds["tau"], "tau_dg": ds["tau_dg"], "nu": ds["nu"],
            "elapsed_s": ds["elapsed_s"], "run_dir": ds["run_dir"],
            "drag_lu_final": ds["drag_lu_final"], "Ct_final": ds["Ct_final"],
        }
        for s in STEPS:
            d = closest(drag_map, s)
            dg_section[f"Ct_step_{s}"] = compute_ct(d, U_IN, S_REF) if d else None
            dg_section[f"drag_lu_step_{s}"] = d
        out["dg_lbm"] = dg_section
    else:
        out["dg_lbm"] = dg_full or {"status": "failed"}

    # Wall function
    if wf_result.get("status") == "ok":
        ws = wf_result
        ct_map = {int(k): v for k, v in ws.get("ct_by_step", {}).items()}
        wf_section = {
            "status": "ok",
            "re": ws["re"], "tau": ws["tau"], "nu": ws["nu"], "Cs": ws["Cs"],
            "elapsed_s": ws["elapsed_s"], "run_dir": ws["run_dir"],
            "Ct_total_final": ws["Ct_total_final"],
        }
        for s in STEPS:
            c = closest(ct_map, s)
            wf_section[f"Ct_step_{s}"] = c["Ct_tot"] if c else None
            wf_section[f"Ct_fric_step_{s}"] = c["Ct_fric"] if c else None
            wf_section[f"Ct_pres_step_{s}"] = c["Ct_pres"] if c else None
        out["wall_function"] = wf_section
    else:
        out["wall_function"] = wf_result or {"status": "failed"}

    out_path = Path("/tmp/dglbm_results.json")
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✅ Results → {out_path}")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"Grid: {NX}×{NY}×{NZ}, SUBOFF bare_hull, L={HL} lu, S_wet={S_REF:.1f} lu²")

    dg = out.get("dg_lbm", {})
    wf = out.get("wall_function", {})

    if dg.get("status") == "ok":
        print(f"\nDG-LBM (Re={dg['re']}, τ={dg['tau']:.3f}, τ_dg={dg['tau_dg']:.3f}):")
        for s in STEPS:
            ct = dg.get(f"Ct_step_{s}")
            dr = dg.get(f"drag_lu_step_{s}")
            print(f"  Step {s:4d}: Ct={ct if ct else 'N/A':>10s}, drag_lu={dr if dr else 'N/A':>10s}")
        print(f"  Final: Ct={dg.get('Ct_final', 'N/A')}")

    if wf.get("status") == "ok":
        print(f"\nWall Function (Re={wf['re']:.0e}, τ={wf['tau']:.3f}, Cs={wf['Cs']}):")
        for s in STEPS:
            ct = wf.get(f"Ct_step_{s}")
            print(f"  Step {s:4d}: Ct={ct:.6f}" if ct else f"  Step {s:4d}: Ct=N/A")
        print(f"  Final: Ct={wf.get('Ct_total_final', 'N/A')}")

    print(f"\n{json.dumps(out, indent=2, default=str)}")


if __name__ == "__main__":
    main()
