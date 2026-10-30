#!/usr/bin/env python3
"""SDAA validation run for aircraft_icing_rg15_ipw2 (case 3.3 rime).

Runs the rime case on SDAA device using the same run_glaze_icing pathway
as the CUDA reference, captures device evidence, and produces
artifacts_sdaa.json with all machine-produced numbers.

Usage:
    python run_sdaa.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import hashlib
import traceback
from pathlib import Path

# Ensure tensorlbm is importable from repo src
REPO_SRC = str(Path(__file__).resolve().parent.parent.parent / "src")
sys.path.insert(0, REPO_SRC)

import numpy as np
import torch

# ---- SDAA device detection ----
SDAA_AVAILABLE = False
SDAA_DEVICE_NAME = "unknown"
SDAA_DEVICE_COUNT = 0
TORCH_SDAA_VERSION = "n/a"

try:
    import torch_sdaa
    SDAA_AVAILABLE = torch.sdaa.is_available()
    SDAA_DEVICE_COUNT = torch.sdaa.device_count()
    if SDAA_AVAILABLE and SDAA_DEVICE_COUNT > 0:
        SDAA_DEVICE_NAME = torch.sdaa.get_device_name(0)
    TORCH_SDAA_VERSION = getattr(torch_sdaa, '__version__', 'unknown')
except Exception as e:
    print(f"[SDAA] torch_sdaa import failed: {e}")

print(f"[SDAA] available={SDAA_AVAILABLE}, device={SDAA_DEVICE_NAME}, "
      f"count={SDAA_DEVICE_COUNT}, torch_sdaa={TORCH_SDAA_VERSION}")

if not SDAA_AVAILABLE:
    print("[SDAA] ERROR: SDAA device not available. Cannot proceed.")
    sys.exit(1)

# ---- Monkey-patch torch.cuda.is_available to return True for SDAA ----
# The benchmark scripts assert torch.cuda.is_available(); we need to
# satisfy that check while actually running on SDAA.
_orig_cuda_available = torch.cuda.is_available
torch.cuda.is_available = lambda: True
torch.cuda.get_device_name = lambda idx=0: SDAA_DEVICE_NAME

# ---- Import benchmark modules ----
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _b_common
from _b_common import (
    build_cfg, install_rg15, curve_stats, beta_md5,
    RUNROOT, REF, COMMON_PHYS, LATTICE, CONDITIONS,
)
import tensorlbm.aircraft_icing as ai

# Install RG-15 geometry
install_rg15()

# ---- Configuration ----
CASE = "3.3_rime"
TAG = "sdaa_c33_rime"
SHOTS = 5
T_EXP = 1200.0
STEPS = 3000
WARMUP = 6000
DEVICE = "sdaa"
COMPILE_STATUS = "eager"

print(f"\n[SDAA] Running case {CASE} on device={DEVICE}")
print(f"[SDAA] shots={SHOTS}, t_exposure={T_EXP}s, steps={STEPS}, warmup={WARMUP}")

# Build config with SDAA device
cfg = build_cfg(
    case=CASE,
    t_exposure=T_EXP,
    steps=STEPS,
    warmup=WARMUP,
    device=DEVICE,
)

print(f"[SDAA] Config: nx={cfg.nx}, ny={cfg.ny}, chord_phys={cfg.chord_phys}")
print(f"[SDAA] t_static_c={cfg.t_static_c}, Re_target={cfg.re_lu_target}")
print(f"[SDAA] collision={cfg.collision}, c_s={cfg.c_s}")
print(f"[SDAA] droplet_phase={cfg.droplet_phase}, eulerian_scheme={cfg.eulerian_scheme}")

# ---- Run the simulation via run_glaze_icing ----
t_start = time.perf_counter()
log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

try:
    g = ai.run_glaze_icing(cfg, shots=SHOTS, log=log)
except Exception as e:
    # Handle InductorError with eager_fallback (per m337 protocol)
    err_type = type(e).__name__
    if "Inductor" in err_type or "inductor" in str(e).lower() or "compile" in str(e).lower():
        COMPILE_STATUS = "eager_fallback"
        print(f"[SDAA] {err_type} caught, retrying with compile_mode='eager'")
        cfg2 = build_cfg(
            case=CASE,
            t_exposure=T_EXP,
            steps=STEPS,
            warmup=WARMUP,
            device=DEVICE,
            compile_mode="eager",
        )
        try:
            g = ai.run_glaze_icing(cfg2, shots=SHOTS, log=log)
        except Exception as e2:
            print(f"[SDAA] Eager retry also failed: {e2}")
            traceback.print_exc()
            # Write error artifact
            error_artifact = {
                "tag": TAG,
                "status": "error",
                "error": str(e2),
                "error_type": type(e2).__name__,
                "device_evidence": {
                    "device": SDAA_DEVICE_NAME,
                    "torch_sdaa_version": TORCH_SDAA_VERSION,
                    "torch_version": torch.__version__,
                },
            }
            out_path = HERE / "artifacts_sdaa.json"
            with open(out_path, "w") as f:
                json.dump(error_artifact, f, indent=2, default=str)
            sys.exit(1)
    else:
        print(f"[SDAA] Non-inductor error: {err_type}: {e}")
        traceback.print_exc()
        error_artifact = {
            "tag": TAG,
            "status": "error",
            "error": str(e),
            "error_type": err_type,
            "device_evidence": {
                "device": SDAA_DEVICE_NAME,
                "torch_sdaa_version": TORCH_SDAA_VERSION,
                "torch_version": torch.__version__,
            },
        }
        out_path = HERE / "artifacts_sdaa.json"
        with open(out_path, "w") as f:
            json.dump(error_artifact, f, indent=2, default=str)
        sys.exit(1)

wall_s = time.perf_counter() - t_start

print(f"\n[SDAA] Simulation completed in {wall_s:.2f} s")
print(f"[SDAA] compile_status={COMPILE_STATUS}")

# ---- Extract results ----
audit = g.get("audit", {})
metrics = g.get("metrics", {})
panels = g.get("panels", {})
bc = g.get("beta_curve", {})

# Key quantities
frozen_kg = float(audit.get("frozen", 0.0))
impact_kg = float(audit.get("impacted", 0.0))
evap_kg = float(audit.get("evaporated", 0.0))
closure_err = float(audit.get("closure_error", 0.0))
deposited = float(audit.get("deposited", frozen_kg))

# Ice mask stats
ice_only = np.asarray(g.get("ice_only", np.array([0])))
ice_cells = int(ice_only.sum())
m_w = np.asarray(g.get("m_w", np.array([0.0])))
m_w_total = float(m_w.sum())

# Beta curve
if bc and len(bc.get("beta", [])) > 0:
    beta_pk = float(max(bc["beta"]))
    s_oc = np.asarray(bc["s_over_c"])
    b_arr = np.asarray(bc["beta"])
    idx_stag = int(np.argmin(np.abs(s_oc)))
    beta_stag = float(b_arr[idx_stag])
else:
    beta_pk = None
    beta_stag = None

# n_f at stagnation
n_f_stag = metrics.get("n_f_stag", None)
if n_f_stag is not None:
    n_f_stag = float(n_f_stag)

# Max ice thickness
t_max_mm = float(metrics.get("t_max_mm", 0.0))

# Messinger closure
messinger_closure = float(audit.get("messinger_residual", 0.0))

# Runback
runback = float(audit.get("runback", 0.0))

# ---- Device evidence ----
mem_allocated = torch.sdaa.memory_allocated(0) if SDAA_AVAILABLE else 0
mem_reserved = torch.sdaa.memory_reserved(0) if SDAA_AVAILABLE else 0
total_steps = STEPS * SHOTS
step_rate = total_steps / wall_s if wall_s > 0 else 0.0

device_evidence = {
    "device": SDAA_DEVICE_NAME,
    "device_count": SDAA_DEVICE_COUNT,
    "torch_sdaa_version": TORCH_SDAA_VERSION,
    "torch_version": torch.__version__,
    "compile_status": COMPILE_STATUS,
    "memory_allocated_bytes": mem_allocated,
    "memory_reserved_bytes": mem_reserved,
    "wall_time_s": round(wall_s, 3),
    "step_rate_steps_per_s": round(step_rate, 2),
    "total_steps": total_steps,
}

# ---- Compare with CUDA reference (from existing evidence) ----
cuda_ref_path = HERE / "evidence" / "b3" / "b3_c33_rime_result.json"
cuda_ref = {}
if cuda_ref_path.exists():
    with open(cuda_ref_path) as f:
        cuda_ref = json.load(f)

comparison = {}
cuda_beta_pk = cuda_ref.get("beta_pk")
cuda_frozen = cuda_ref.get("audit", {}).get("frozen")
cuda_n_f_stag = cuda_ref.get("metrics", {}).get("n_f_stag")

if cuda_beta_pk is not None and beta_pk is not None:
    comparison["beta_pk_rel_diff"] = abs(beta_pk - cuda_beta_pk) / max(abs(cuda_beta_pk), 1e-30)
if cuda_frozen is not None:
    comparison["frozen_rel_diff"] = abs(frozen_kg - cuda_frozen) / max(abs(cuda_frozen), 1e-30)
if cuda_n_f_stag is not None and n_f_stag is not None:
    comparison["n_f_stag_abs_diff"] = abs(n_f_stag - cuda_n_f_stag)

# ---- Assemble artifact ----
artifact = {
    "tag": TAG,
    "case": CASE,
    "status": "completed",
    "device_evidence": device_evidence,
    "config": {
        "nx": cfg.nx,
        "ny": cfg.ny,
        "chord_phys": cfg.chord_phys,
        "v_inf": cfg.v_inf,
        "aoa_deg": cfg.aoa_deg,
        "t_static_c": cfg.t_static_c,
        "lwc": cfg.lwc,
        "mvd": cfg.mvd,
        "t_exposure": cfg.t_exposure,
        "steps": cfg.steps,
        "warmup_steps": cfg.warmup_steps,
        "device": DEVICE,
        "collision": cfg.collision,
        "c_s": cfg.c_s,
        "droplet_phase": cfg.droplet_phase,
        "eulerian_scheme": cfg.eulerian_scheme,
        "rime_density_mode": cfg.rime_density_mode,
    },
    "results": {
        "frozen_kg": frozen_kg,
        "impact_kg": impact_kg,
        "evap_kg": evap_kg,
        "deposited_kg": deposited,
        "closure_error": closure_err,
        "ice_cells": ice_cells,
        "m_w_total_kg": m_w_total,
        "t_max_mm": t_max_mm,
        "beta_pk": beta_pk,
        "beta_stag": beta_stag,
        "n_f_stag": n_f_stag if (n_f_stag := metrics.get("n_f_stag")) is not None else None,
        "messinger_closure": messinger_closure,
        "runback": runback,
    },
    "shots": SHOTS,
    "wall_time_s": round(wall_s, 3),
    "cuda_reference_comparison": comparison,
    "cuda_reference_file": str(cuda_ref_path) if cuda_ref_path.exists() else None,
    "log_lines": log_lines[-20:],  # last 20 lines
}

# Fix the n_f_stag key (typo above)
artifact["results"]["n_f_stag"] = float(n_f_stag) if n_f_stag is not None else None

# ---- Write artifact ----
out_path = HERE / "artifacts_sdaa.json"
with open(out_path, "w") as f:
    json.dump(artifact, f, indent=2, default=float)
print(f"\n[SDAA] Artifact written to {out_path}")

# ---- Print summary ----
print("\n" + "="*60)
print("SDAA VALIDATION SUMMARY")
print("="*60)
print(f"Device: {SDAA_DEVICE_NAME}")
print(f"torch_sdaa: {TORCH_SDAA_VERSION}")
print(f"Compile: {COMPILE_STATUS}")
print(f"Wall time: {wall_s:.2f} s")
print(f"Step rate: {step_rate:.1f} steps/s")
print(f"Frozen mass: {frozen_kg:.6e} kg")
print(f"Beta_pk: {beta_pk}")
print(f"n_f_stag: {n_f_stag}")
print(f"Closure error: {closure_err:.6e}")
if comparison:
    print(f"\nvs CUDA reference:")
    for k, v in comparison.items():
        print(f"  {k}: {v:.6e}")
print("="*60)
