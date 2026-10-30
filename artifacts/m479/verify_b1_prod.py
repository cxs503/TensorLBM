#!/usr/bin/env python3
"""m479 验证脚本：复现 aircraft_icing_naca0012_rime 生产点 b1_prod 并比对证据。

用法: CUDA_VISIBLE_DEVICES=0 python artifacts/m479/verify_b1_prod.py

预期: 在 RTX 5090 上 ~29s 完成 b1_prod 单案，与证据文件逐位比对。
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent.parent  # repo root
_SRC = _HERE / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
import torch

from tensorlbm.aircraft_icing import IcingConfig, run_rime_icing

# Reference evidence (from benchmarks/verified/aircraft_icing_naca0012_rime/evidence/b1_prod)
REF = {
    "beta_e_pk": 0.9118858575820923,
    "euler_deposited_kg": 0.0017577295005006153,
    "t_max_mm": 8.334375,
    "upper_horn_mm": 4.1671875,
    "lower_horn_mm": 8.334375,
    "t_stag_mm": 12.5015625,
    "n_ice_cells": 15,
    "ice_area_pct_chord2": 0.091552734375,
    "closure_lagr": 0.0,
    "closure_euler": 1.9798993102412833e-08,
    "cd0": 0.058272411115467534,
    "cl0": 0.1543960766866803,
    "cd_end": 0.09527911897748707,
    "cl_end": 0.1327119534835219,
}

DET_ARRAYS = (
    "airfoil", "solid", "m_w", "impact_mass", "s_grid", "beta_grid",
    "stag", "hist_step", "hist_t", "hist_cd", "hist_cl", "hist_ice",
    "alpha_e", "impact_mass_e", "beta_e_grid",
)

# Expected sha256 first 8 chars from evidence
REF_SHA256 = {
    "airfoil": "d6d5f9fd", "solid": "2542f8be", "m_w": "3304d4eb",
    "impact_mass": "8eafc7bd", "s_grid": "05db2727", "beta_grid": "8eafc7bd",
    "stag": "983eb4c4", "hist_step": "7aaee97c", "hist_t": "fa585134",
    "hist_cd": "7a77c25d", "hist_cl": "fe263d15", "hist_ice": "ff83c199",
    "alpha_e": "63b79f0c", "impact_mass_e": "7d2aaea1", "beta_e_grid": "739b888c",
}


def npz_array_hashes(npz_path) -> dict:
    z = np.load(npz_path)
    out = {}
    for k in DET_ARRAYS:
        a = z[k]
        h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:8]
        out[k] = h
    return out


def main():
    assert torch.cuda.is_available(), "Need CUDA for verification"

    cfg = IcingConfig(
        nx=320, ny=160, chord_frac=0.4, u_in=0.05, aoa_deg=4.0,
        steps=3000, warmup_steps=6000, chord_phys=0.5334, v_inf=67.0,
        lwc=5.0e-4, mvd=20.0e-6, t_static_c=-10.0, t_exposure=360.0,
        rime_density_mode="macklin", droplet_phase="eulerian",
        eulerian_scheme="donor2", collision="cumulant", c_s=0.1,
        re_lu_target=2.5e6, beta_window_mode="clean", drag_law="stokes",
        device="cuda", seed=0, log_every=250,
    )

    print(f"[m479] Starting b1_prod verification on {torch.cuda.get_device_name(0)}")
    print(f"  Grid: {cfg.nx}x{cfg.ny}, steps={cfg.steps}, warmup={cfg.warmup_steps}")
    print(f"  chord_lu={cfg.chord_lu}, dx_phys={cfg.dx_phys*1e3:.4f} mm, Stokes={cfg.stokes:.6f}")

    t0 = time.time()
    r = run_rime_icing(cfg, log=print)
    wall = time.time() - t0
    print(f"\n[m479] Run completed in {wall:.1f}s")

    e = r["eulerian"]
    m = r["metrics"]
    aud = r["audit"]

    # Compute derived quantities
    beta_pk = float(e["beta"]["beta"].max())
    dep = float(e["audit"]["deposited"])
    ice = r["ice_only"]
    x_le = int(m["x_le"])
    t_stag = float(ice[:, x_le].sum()) * cfg.dx_phys * 1e3
    t_max = 1e3 * max(m.get("upper_horn_m", 0.0), m.get("lower_horn_m", 0.0))

    # Compare with reference
    print("\n[m479] === Verification Results ===")
    all_pass = True
    checks = [
        ("beta_e_pk", beta_pk, REF["beta_e_pk"], 1e-10),
        ("euler_deposited_kg", dep, REF["euler_deposited_kg"], 1e-12),
        ("t_max_mm", t_max, REF["t_max_mm"], 1e-6),
        ("upper_horn_mm", 1e3 * m.get("upper_horn_m", 0), REF["upper_horn_mm"], 1e-6),
        ("lower_horn_mm", 1e3 * m.get("lower_horn_m", 0), REF["lower_horn_mm"], 1e-6),
        ("t_stag_mm", t_stag, REF["t_stag_mm"], 1e-6),
        ("n_ice_cells", m["n_ice_cells"], REF["n_ice_cells"], 0),
        ("ice_area_pct", m["ice_area_pct_chord2"], REF["ice_area_pct_chord2"], 1e-10),
        ("closure_lagr", aud["closure_error"], REF["closure_lagr"], 1e-15),
        ("closure_euler", e["audit"]["closure_error"], REF["closure_euler"], 1e-10),
        ("cd0", r["cd0"], REF["cd0"], 1e-10),
        ("cl0", r["cl0"], REF["cl0"], 1e-10),
        ("cd_end", r["cd_end"], REF["cd_end"], 1e-10),
        ("cl_end", r["cl_end"], REF["cl_end"], 1e-10),
    ]

    for name, got, ref, tol in checks:
        diff = abs(got - ref)
        ok = diff <= tol
        status = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"  {status} {name}: got={got}, ref={ref}, diff={diff:.2e}, tol={tol:.2e}")

    # Determinism check (save npz and compare hashes)
    out_dir = Path("runs/m479/b1_prod_verify")
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "result.npz"
    np.savez(npz_path, **{
        "airfoil": r["airfoil"], "solid": r["solid"],
        "m_w": r["m_w"], "impact_mass": r["impact_mass"],
        "s_grid": r["s_grid"], "beta_grid": r["beta_grid"],
        "stag": r["stag"], "hist_step": r["hist_step"],
        "hist_t": r["hist_t"], "hist_cd": r["hist_cd"],
        "hist_cl": r["hist_cl"], "hist_ice": r["hist_ice"],
        "alpha_e": r["alpha_e"], "impact_mass_e": r["impact_mass_e"],
        "beta_e_grid": r["beta_e_grid"],
    })

    hashes = npz_array_hashes(npz_path)
    print("\n[m479] === Determinism Check (sha256) ===")
    for key in DET_ARRAYS:
        got_h = hashes.get(key, "MISSING")
        ref_h = REF_SHA256.get(key, "N/A")
        ok = got_h == ref_h
        status = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"  {status} {key}: {got_h} (ref: {ref_h})")

    print(f"\n[m479] === OVERALL: {'PASS ✓' if all_pass else 'FAIL ✗'} ===")
    print(f"  Wall time: {wall:.1f}s")

    # Save verification result
    result = {
        "mission": "m479",
        "case": "aircraft_icing_naca0012_rime",
        "run": "b1_prod_verify",
        "wall_time_s": wall,
        "all_pass": all_pass,
        "checks": {n: {"got": g, "ref": r, "diff": abs(g - r), "tol": t, "pass": abs(g - r) <= t}
                   for n, g, r, t in checks},
        "sha256_match": {k: hashes.get(k) == REF_SHA256.get(k) for k in DET_ARRAYS},
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }
    with open(out_dir / "verify_result.json", "w") as f:
        json.dump(result, f, indent=2, default=float)
    print(f"\n[m479] Result saved to {out_dir / 'verify_result.json'}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
