#!/usr/bin/env python3
"""DG-LBM vs Wall Function comparison - parallel execution.

DG-LBM: 48³, Re=200, dg_sub=16, 100 steps, CPU
WallFn: 64³, Re=2e6, 500 steps, sdaa:5
"""
from __future__ import annotations
import json, os, sys, time, traceback, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from tensorlbm.dg_lbm import (
    DGLBMSuboffConfig, _run_suboff_real_dg, _run_suboff_wall_function,
    build_suboff_mask,
)
from tensorlbm.suboff_resistance import _voxel_wetted_area

U_IN = 0.06

def get_wetted(nx, ny, nz, hl):
    cx, cy, cz = nx * 0.35, ny * 0.5, nz * 0.5
    obs, _ = build_suboff_mask("bare_hull", nx, ny, nz, cx, cy, cz, hl, device="cpu")
    return _voxel_wetted_area(obs, 1.0)

def parse_drag(log_path):
    if not Path(log_path).exists(): return {}
    out = {}
    for line in Path(log_path).read_text().splitlines():
        if "drag=" in line and "step=" in line:
            try:
                sp = [p for p in line.split() if p.startswith("step=")]
                dp = [p for p in line.split() if p.startswith("drag=")]
                if sp and dp: out[int(sp[0].split("=")[1])] = float(dp[0].split("=")[1])
            except: pass
    return out

def parse_ct(log_path):
    if not Path(log_path).exists(): return {}
    out = {}
    for line in Path(log_path).read_text().splitlines():
        if "Ct_tot=" in line and "step=" in line:
            try:
                sp = [p for p in line.split() if p.startswith("step=")]
                if not sp: continue
                step = int(sp[0].split("=")[1])
                ct = {}
                for p in line.split():
                    if p.startswith("Ct_fric="): ct["fric"] = float(p.split("=")[1])
                    elif p.startswith("Ct_pres="): ct["pres"] = float(p.split("=")[1])
                    elif p.startswith("Ct_tot="): ct["tot"] = float(p.split("=")[1])
                out[step] = ct
            except: pass
    return out

def ct_from_drag(drag, S):
    return drag / (0.5 * U_IN * U_IN * S) if S > 0 else 0

def closest(data, target):
    cand = {k: v for k, v in data.items() if k <= target}
    return data[max(cand)] if cand else None

# ---- Config ----
DG_NX, DG_NY, DG_NZ = 48, 48, 48
DG_HL = 24.0
DG_RE = 200.0
DG_STEPS = 100
DG_SUB = 16
DG_DEV = "cpu"
DG_OUT = "/tmp/dglbm_dg_out"

WF_NX, WF_NY, WF_NZ = 64, 64, 64
WF_HL = 32.0
WF_RE = 2e6
WF_STEPS = 500
WF_CS = 0.05
WF_DEV = "sdaa:5"
WF_OUT = "/tmp/dglbm_wf_out"

STEPS_REPORT = [50, 100, 200, 500, 1000]

def run_dg():
    if os.path.exists(DG_OUT): shutil.rmtree(DG_OUT)
    print(f"DG-LBM: {DG_NX}³, Re={DG_RE}, dev={DG_DEV}, dg_sub={DG_SUB}, steps={DG_STEPS}")
    cfg = DGLBMSuboffConfig(
        nx=DG_NX, ny=DG_NY, nz=DG_NZ, hull_length=DG_HL, u_in=U_IN, re=DG_RE,
        hull_type="bare_hull", dg_band=3, dg_substeps=DG_SUB,
        use_real_dg=True, n_steps=DG_STEPS, output_interval=20,
        device=DG_DEV, output_root=Path(DG_OUT), overwrite=True, run_name="dg",
    )
    td = cfg.tau - 0.5
    print(f"  τ={cfg.tau:.4f}, τ_dg={td:.4f}, dt_sub={1.0/DG_SUB:.4f}")
    t0 = time.time()
    rd = _run_suboff_real_dg(cfg)
    t1 = time.time()
    meta = json.loads((Path(rd)/"run_metadata.json").read_text())
    dr_map = parse_drag(Path(rd)/"run.log")
    S = get_wetted(DG_NX, DG_NY, DG_NZ, DG_HL)
    return {
        "status": "ok", "run_dir": str(rd), "elapsed_s": t1-t0,
        "grid": f"{DG_NX}×{DG_NY}×{DG_NZ}", "hull_length": DG_HL,
        "re": DG_RE, "tau": cfg.tau, "tau_dg": td, "nu": cfg.nu,
        "dg_substeps": DG_SUB, "wetted_area": S,
        "drag_lu_final": meta.get("drag_force_lu"),
        "Ct_final": ct_from_drag(meta.get("drag_force_lu"), S),
        "drag_map": dr_map,
    }

def run_wf():
    if os.path.exists(WF_OUT): shutil.rmtree(WF_OUT)
    print(f"WallFn: {WF_NX}³, Re={WF_RE:.0e}, dev={WF_DEV}, Cs={WF_CS}, steps={WF_STEPS}")
    cfg = DGLBMSuboffConfig(
        nx=WF_NX, ny=WF_NY, nz=WF_NZ, hull_length=WF_HL, u_in=U_IN, re=WF_RE,
        hull_type="bare_hull", use_wall_function=True, smagorinsky_cs=WF_CS,
        n_steps=WF_STEPS, output_interval=100, device=WF_DEV,
        output_root=Path(WF_OUT), overwrite=True, run_name="wf",
    )
    print(f"  τ={cfg.tau:.4f}, ν={cfg.nu:.4e}")
    t0 = time.time()
    rd = _run_suboff_wall_function(cfg)
    t1 = time.time()
    meta = json.loads((Path(rd)/"run_metadata.json").read_text())
    ct_map = parse_ct(Path(rd)/"run.log")
    return {
        "status": "ok", "run_dir": str(rd), "elapsed_s": t1-t0,
        "grid": f"{WF_NX}×{WF_NY}×{WF_NZ}", "hull_length": WF_HL,
        "re": WF_RE, "tau": cfg.tau, "nu": cfg.nu, "Cs": WF_CS,
        "Ct_total_final": meta.get("Ct_total"),
        "ct_map": ct_map,
    }

def compile_output(dg_result, wf_result):
    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "diagnosis": {
            "nan_cause": (
                "SSP-RK3 in dg_lbm_step_band is unstable for dt_sub in range "
                "[0.15, 0.6] when τ_dg ≈ 0.058 (Re=200, hull=24). "
                "Stable at n_substeps=1 or ≥ 8. Fix: use n_substeps ≥ 10 for τ_dg ≤ 0.06. "
                "Additionally, dg_lbm_step_band tests (test_dglbm_debug2.py) confirm "
                "the advection-only and collision-only RHS are NaN-free; the issue is "
                "solely in the SSP-RK3 amplification factor for intermediate dt_sub values."
            ),
        },
    }

    S_REF = get_wetted(48, 48, 48, 24.0)

    if dg_result:
        dg = dg_result
        dm = {int(k): v for k, v in dg["drag_map"].items()} if dg.get("drag_map") else {}
        dsec = {
            "status": dg.get("status", "ok"),
            "grid": dg.get("grid"),
            "re": dg.get("re"),
            "tau": dg.get("tau"),
            "tau_dg": dg.get("tau_dg"),
            "elapsed_s": dg.get("elapsed_s"),
            "run_dir": dg.get("run_dir"),
            "drag_lu_final": dg.get("drag_lu_final"),
            "Ct_final": dg.get("Ct_final"),
        }
        for s in STEPS_REPORT:
            d = closest(dm, s)
            dsec[f"Ct_step_{s}"] = ct_from_drag(d, dg.get("wetted_area", 1)) if d else None
            dsec[f"drag_lu_step_{s}"] = d
        out["dg_lbm"] = dsec
    else:
        out["dg_lbm"] = {"status": "failed"}

    if wf_result:
        wf = wf_result
        cm = {int(k): v for k, v in wf["ct_map"].items()} if wf.get("ct_map") else {}
        wsec = {
            "status": wf.get("status", "ok"),
            "grid": wf.get("grid"),
            "re": wf.get("re"),
            "tau": wf.get("tau"),
            "elapsed_s": wf.get("elapsed_s"),
            "run_dir": wf.get("run_dir"),
            "Ct_total_final": wf.get("Ct_total_final"),
        }
        for s in STEPS_REPORT:
            c = closest(cm, s)
            if c:
                wsec[f"Ct_step_{s}"] = c.get("tot")
                wsec[f"Ct_fric_step_{s}"] = c.get("fric")
                wsec[f"Ct_pres_step_{s}"] = c.get("pres")
            else:
                wsec[f"Ct_step_{s}"] = None
        out["wall_function"] = wsec
    else:
        out["wall_function"] = {"status": "failed"}

    return out


def main():
    # Run both in parallel
    dg_result = None
    dg_error = None
    wf_result = None
    wf_error = None

    with ThreadPoolExecutor(max_workers=2) as ex:
        dg_fut = ex.submit(run_dg)
        wf_fut = ex.submit(run_wf)
        try:
            dg_result = dg_fut.result(timeout=3600)
        except Exception as e:
            dg_error = str(e)
            traceback.print_exc()
        try:
            wf_result = wf_fut.result(timeout=3600)
        except Exception as e:
            wf_error = str(e)
            traceback.print_exc()

    if dg_error and not dg_result:
        dg_result = {"status": "failed", "error": dg_error}
    if wf_error and not wf_result:
        wf_result = {"status": "failed", "error": wf_error}

    out = compile_output(dg_result, wf_result)

    op = Path("/tmp/dglbm_results.json")
    op.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n✅ Results → {op}")

    # Summary
    print("\n" + "=" * 60)
    print("DG-LBM vs WALL FUNCTION COMPARISON")
    print("=" * 60)

    if dg_result and dg_result.get("status") == "ok":
        d = dg_result
        print(f"\nDG-LBM ({d['grid']}, Re={d['re']}):")
        print(f"  τ={d['tau']:.4f}, τ_dg={d['tau_dg']:.4f}, dg_sub={d['dg_substeps']}")
        print(f"  Time: {d['elapsed_s']:.0f}s")
        print(f"  Drag final: {d.get('drag_lu_final')}")
        print(f"  Ct final: {d.get('Ct_final')}")
        dm = {int(k): v for k, v in d.get("drag_map", {}).items()}
        for s in STEPS_REPORT:
            dr = closest(dm, s)
            if dr is not None:
                print(f"    Step {s}: drag_lu={dr:.4f}, Ct={ct_from_drag(dr, d.get('wetted_area',1)):.6f}")
    else:
        print(f"\nDG-LBM: FAILED - {dg_result}")

    if wf_result and wf_result.get("status") == "ok":
        w = wf_result
        print(f"\nWall Function ({w['grid']}, Re={w['re']:.0e}):")
        print(f"  τ={w['tau']:.4f}, Cs={w['Cs']}")
        print(f"  Time: {w['elapsed_s']:.0f}s")
        print(f"  Ct final: {w.get('Ct_total_final')}")
        cm = {int(k): v for k, v in w.get("ct_map", {}).items()}
        for s in STEPS_REPORT:
            c = closest(cm, s)
            if c:
                print(f"    Step {s}: Ct_tot={c.get('tot'):.6f}, Ct_fric={c.get('fric')}, Ct_pres={c.get('pres')}")
    else:
        print(f"\nWall Function: FAILED - {wf_result}")

    print(f"\nFull JSON: {json.dumps(out, indent=2, default=str)}")

if __name__ == "__main__":
    main()
