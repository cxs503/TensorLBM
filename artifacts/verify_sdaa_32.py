#!/usr/bin/env python3
"""SDAA verification - 32^3 with 3000 steps (fits in 120s timeout)"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/tp-agent/work/m216/src")

from tensorlbm.boundaries3d import zou_he_moving_lid_3d
from tensorlbm.d3q19 import OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_trt3d, stream3d

REF_U_MIN_3D_RE1000 = -0.2751

def stationary_pre_bounce3d(f_pre, f, wall):
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)

print("=== SDAA Verification (32^3, 3000 steps) ===")
device = torch.device("sdaa:0")
device_name = torch.sdaa.get_device_name(0)
dev_nodes = sorted([str(p) for p in Path("/dev").glob("tcaicard*")])
print(f"Device: {device_name}")
print(f"Device count: {torch.sdaa.device_count()}")
print(f"/dev/tcaicard* nodes: {len(dev_nodes)} found")

nx = 32
nz = ny = nx
re = 1000
u_lid = 0.06
steps = 3000

tau = 3.0 * u_lid * nx / re + 0.5
rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
u0 = torch.zeros((nz, ny, nx), dtype=torch.float32, device=device)
f = equilibrium3d(rho0, u0, u0, u0)

wall = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
wall[:, :, 0] = True
wall[:, :, -1] = True
wall[:, 0, :] = True
wall[0, :, :] = True
wall[-1, :, :] = True

t0 = time.time()
for step in range(1, steps + 1):
    f_pre = f
    f = collide_trt3d(f, tau)
    f = stationary_pre_bounce3d(f_pre, f, wall)
    f = stream3d(f)
    f = zou_he_moving_lid_3d(f, u_lid)
    if step % 1000 == 0:
        elapsed = time.time() - t0
        print(f"step {step}/{steps} t={elapsed:.1f}s", flush=True)

elapsed = time.time() - t0
rho, ux, uy, uz = macroscopic3d(f)
ux_np = ux.detach().cpu().numpy() / u_lid

z0 = nz // 2
x_mid = nx // 2
u_cl_mid = ux_np[z0, :, x_mid]
u_min_3d = float(u_cl_mid.min())
u_min_err_pct = 100.0 * abs(u_min_3d - REF_U_MIN_3D_RE1000) / abs(REF_U_MIN_3D_RE1000)

finite = bool(torch.isfinite(f).all().item())
print(f"\n=== Result ===")
print(f"u_min = {u_min_3d:.5f} (ref {REF_U_MIN_3D_RE1000})")
print(f"err = {u_min_err_pct:.3f}%")
print(f"elapsed = {elapsed:.1f}s")
print(f"finite = {finite}")

result = {
    "purpose": "SDAA execution verification (32^3 coarse grid)",
    "device": {
        "torch.sdaa.get_device_name(0)": device_name,
        "torch.sdaa.device_count()": torch.sdaa.device_count(),
        "/dev/tcaicard*_count": len(dev_nodes),
        "/dev/tcaicard*_nodes": dev_nodes,
    },
    "config": {
        "nx": nx,
        "steps": steps,
        "re": re,
        "u_lid": u_lid,
        "tau": round(tau, 6),
    },
    "result": {
        "u_min_3d": round(u_min_3d, 5),
        "u_min_err_pct": round(u_min_err_pct, 3),
        "elapsed_s": round(elapsed, 1),
        "finite": finite,
    },
    "note": "32^3 grid too coarse for <3% accuracy; demonstrates SDAA execution; authoritative results from 64^3/96^3 in repo",
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}

Path("/root/tp-agent/work/m216/artifacts/sdaa_verification_32.json").write_text(json.dumps(result, indent=2))
print(f"\nSaved to artifacts/sdaa_verification_32.json")
