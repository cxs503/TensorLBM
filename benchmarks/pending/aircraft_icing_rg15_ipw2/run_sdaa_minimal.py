#!/usr/bin/env python3
"""SDAA minimal validation: single shot, minimal steps.

Proves SDAA pathway works and captures device evidence.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO_SRC = str(Path(__file__).resolve().parent.parent.parent / "src")
sys.path.insert(0, REPO_SRC)

import numpy as np
import torch

# ---- SDAA device detection ----
SDAA_AVAILABLE = False
SDAA_DEVICE_NAME = "unknown"
TORCH_SDAA_VERSION = "n/a"

try:
    import torch_sdaa
    SDAA_AVAILABLE = torch.sdaa.is_available()
    if SDAA_AVAILABLE:
        SDAA_DEVICE_NAME = torch.sdaa.get_device_name(0)
    TORCH_SDAA_VERSION = getattr(torch_sdaa, '__version__', 'unknown')
except Exception as e:
    print(f"[SDAA] torch_sdaa import failed: {e}")

print(f"[SDAA] available={SDAA_AVAILABLE}, device={SDAA_DEVICE_NAME}, torch_sdaa={TORCH_SDAA_VERSION}")

if not SDAA_AVAILABLE:
    print("[SDAA] ERROR: SDAA not available")
    sys.exit(1)

# Monkey-patch for benchmark compatibility
torch.cuda.is_available = lambda: True
torch.cuda.get_device_name = lambda idx=0: SDAA_DEVICE_NAME

# ---- Import benchmark modules ----
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _b_common
from _b_common import build_cfg, install_rg15
import tensorlbm.aircraft_icing as ai

install_rg15()

# ---- Minimal run config ----
CASE = "3.3_rime"
TAG = "sdaa_minimal_c33_rime"
SHOTS = 1
T_EXP = 240.0
STEPS = 200
WARMUP = 500
DEVICE = "sdaa"
COMPILE_STATUS = "eager"

print(f"\n[SDAA] Minimal validation: case={CASE}, shots={SHOTS}, steps={STEPS}, warmup={WARMUP}")

cfg = build_cfg(
    case=CASE,
    t_exposure=T_EXP,
    steps=STEPS,
    warmup=WARMUP,
    device=DEVICE,
)

print(f"[SDAA] Config: nx={cfg.nx}, ny={cfg.ny}, device={cfg.device}")

# ---- Run ----
t_start = time.perf_counter()
log_lines = []

def log(msg):
    print(msg, flush=True)
    log_lines.append(str(msg))

try:
    print("[SDAA] Starting run_glaze_icing...")
    g = ai.run_glaze_icing(cfg, shots=SHOTS, log=log)
    print("[SDAA] SUCCESS: run_glaze_icing completed")
except Exception as e:
    err_type = type(e).__name__
    print(f"[SDAA] Error: {err_type}: {e}")
    traceback.print_exc()
    
    artifact = {
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
    with open(HERE / "artifacts_sdaa.json", "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    sys.exit(1)

wall_s = time.perf_counter() - t_start

# ---- Extract results ----
audit = g.get("audit", {})
metrics = g.get("metrics", {})
bc = g.get("beta_curve", {})

frozen_kg = float(audit.get("frozen", 0.0))
closure_err = float(audit.get("closure_error", 0.0))
ice_cells = int(np.asarray(g.get("ice_only", [0])).sum())

beta_pk = float(max(bc["beta"])) if bc and len(bc.get("beta", [])) > 0 else None
n_f_stag = metrics.get("n_f_stag")
if n_f_stag is not None:
    n_f_stag = float(n_f_stag)

# Device evidence
mem_alloc = torch.sdaa.memory_allocated(0) if SDAA_AVAILABLE else 0
step_rate = (STEPS * SHOTS) / wall_s if wall_s > 0 else 0.0

device_evidence = {
    "device": SDAA_DEVICE_NAME,
    "torch_sdaa_version": TORCH_SDAA_VERSION,
    "torch_version": torch.__version__,
    "compile_status": COMPILE_STATUS,
    "wall_time_s": round(wall_s, 3),
    "step_rate_steps_per_s": round(step_rate, 2),
    "memory_allocated_bytes": mem_alloc,
}

artifact = {
    "tag": TAG,
    "case": CASE,
    "status": "completed",
    "device_evidence": device_evidence,
    "config": {
        "nx": cfg.nx,
        "ny": cfg.ny,
        "steps": STEPS,
        "warmup": WARMUP,
        "shots": SHOTS,
        "device": DEVICE,
        "t_exposure": T_EXP,
    },
    "results": {
        "frozen_kg": frozen_kg,
        "closure_error": closure_err,
        "ice_cells": ice_cells,
        "beta_pk": beta_pk,
        "n_f_stag": n_f_stag,
    },
    "wall_time_s": round(wall_s, 3),
}

with open(HERE / "artifacts_sdaa.json", "w") as f:
    json.dump(artifact, f, indent=2, default=float)

print(f"\n{'='*60}")
print(f"SDAA MINIMAL VALIDATION - SUCCESS")
print(f"{'='*60}")
print(f"Device: {SDAA_DEVICE_NAME}")
print(f"torch_sdaa: {TORCH_SDAA_VERSION}")
print(f"Wall time: {wall_s:.2f} s")
print(f"Step rate: {step_rate:.1f} steps/s")
print(f"Frozen mass: {frozen_kg:.6e} kg")
print(f"Ice cells: {ice_cells}")
print(f"Beta_pk: {beta_pk}")
print(f"n_f_stag: {n_f_stag}")
print(f"Closure error: {closure_err:.6e}")
print(f"{'='*60}")
print(f"Artifact: {HERE / 'artifacts_sdaa.json'}")
