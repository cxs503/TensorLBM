#!/usr/bin/env python3
"""Custom runner for backward_step on SDAA — captures device evidence and runs short validation."""
import sys, time, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch

# ── SDAA Device Evidence ──
sdaa_evidence = {}
try:
    import torch_sdaa
    sdaa_evidence["torch_sdaa_version"] = getattr(torch_sdaa, "__version__", "unknown")
    sdaa_evidence["torch_sdaa_imported"] = True
except Exception as e:
    sdaa_evidence["torch_sdaa_imported"] = False
    sdaa_evidence["torch_sdaa_error"] = str(e)

sdaa_evidence["torch_version"] = torch.__version__
sdaa_evidence["sdaa_available"] = bool(getattr(torch, "sdaa", None) and torch.sdaa.is_available())
if sdaa_evidence["sdaa_available"]:
    sdaa_evidence["sdaa_device_count"] = torch.sdaa.device_count()
    props = torch.sdaa.get_device_properties(0)
    sdaa_evidence["sdaa_device_name"] = props.name
    sdaa_evidence["sdaa_total_memory"] = props.total_memory
    sdaa_evidence["sdaa_current_device"] = torch.sdaa.current_device()

print("=== SDAA Device Evidence ===")
for k, v in sdaa_evidence.items():
    print(f"  {k}: {v}")

# ── Run short BFS simulation on SDAA ──
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.solver import collide_mrt, stream
from tensorlbm.boundaries import zou_he_inlet_velocity, zou_he_outlet_pressure, make_sponge_strength

device = torch.device("sdaa:0")

# Parameters: Gartling 1990 BFS, Re=800, ER=2
m = 16
re = 800.0
u_in = 0.06
L_over_H = 10.0  # reduced from 20 for speed; REFERENCE_AUDIT says L/H>7 is sufficient
x_step = 8
ny = 2 * m + 1
H = ny - 1  # = 32
L = int(round(L_over_H * H))
nx = x_step + L
nu = u_in * H / re
tau = 0.5 + 3.0 * nu

print(f"\n=== BFS Simulation (short validation) ===")
print(f"  m={m} H={H} nx={nx} nu={nu:.6f} tau={tau:.6f} Re={re}")

# Build solid mask
solid = torch.zeros((ny, nx), dtype=torch.bool, device=device)
solid[-1, :] = True
solid[0, x_step:] = True
solid[0 : m + 1, :x_step] = True

# Inlet profile
a = float(m)
y_arr = np.arange(ny, dtype=np.float64)
s = y_arr - (m + 0.5)
prof_np = np.zeros(ny, dtype=np.float64)
rows = np.arange(m + 1, ny - 1)
ss = s[rows]
raw = 6.0 * ss * (a - ss) / (a * a)
raw_mean = raw.mean()
prof_np[rows] = raw / raw_mean * u_in
u_prof = torch.tensor(prof_np, dtype=torch.float32, device=device)

# Sponge layer
sponge_peak = 0.3
sponge_width_H = 3.0
sponge_w = int(round(sponge_width_H * H))
sponge_x0 = nx - sponge_w
sigma = sponge_peak * make_sponge_strength(ny, nx, sponge_x0, sponge_w, power=2.0, device=device)

# Initial condition
rho0 = torch.ones((ny, nx), device=device)
ux0 = torch.zeros((ny, nx), device=device)
ux0[:, :x_step] = u_prof.view(ny, 1)
ux0[:, x_step:] = u_in / 2.0
ux0[solid] = 0.0
f = equilibrium(rho0, ux0, torch.zeros_like(ux0))

opp = OPPOSITE.to(device)
solid3 = solid.unsqueeze(0)

REF_X1_H = 6.10

# Measurement setup
K = min(6, m // 2)
lo_rows = np.arange(1, K + 1)
yw = np.arange(K, dtype=np.float64)
ym = yw.mean()
w = yw - ym
w_lo = w / float((w * w).sum())

# Run simulation
steps = 4000
avg_stride = 50
from collections import deque
n_win = 40  # last 2000 steps
win = deque(maxlen=n_win)

t0 = time.time()
sim_data = {"steps": steps, "m": m, "H": H, "nx": nx, "tau": tau, "nu": nu, "u_in": u_in, "re": re, "L_over_H": L_over_H}

for step in range(1, steps + 1):
    fp = f.clone()
    fc = collide_mrt(f, tau=tau)
    fc = torch.where(solid3, fp[opp], fc)
    f = stream(fc)
    f = zou_he_inlet_velocity(f, u_prof, 0.0)
    f = zou_he_outlet_pressure(f, 1.0)
    rho, ux, uy = macroscopic(f)
    feq = equilibrium(rho, ux, uy)
    f = f - sigma.unsqueeze(0) * (f - feq)

    if step % avg_stride == 0:
        uxm = ux.masked_fill(solid, 0.0)
        lo = uxm[lo_rows].detach().cpu().numpy().astype(np.float64)
        s_lo = (w_lo[:, None] * lo).sum(axis=0)
        win.append(s_lo.copy())

    if step % 1000 == 0:
        elapsed = time.time() - t0
        rho_min = rho.min().item()
        rho_max = rho.max().item()
        max_u = ux.abs().max().item()
        mass = rho.sum().item()
        # Measure X1 from window
        x1_h = float("nan")
        if len(win) >= 10:
            avg_lo = np.mean(win, axis=0)
            for c in range(x_step, len(avg_lo) - 1):
                a_v, b_v = avg_lo[c], avg_lo[c + 1]
                if a_v <= 0.0 < b_v:
                    xc = c + (0.0 - a_v) / (b_v - a_v)
                    x1_cells = xc - (x_step - 0.5)
                    x1_h = x1_cells / H
                    break
        err = (x1_h - REF_X1_H) / REF_X1_H * 100 if x1_h == x1_h else float("nan")
        print(f"  step={step:5d} X1/H={x1_h:.4f} err={err:.1f}% rho=[{rho_min:.4f},{rho_max:.4f}] max|u|={max_u:.5f} t={elapsed:.0f}s")
        sim_data[f"step_{step}"] = {"X1_H": round(x1_h, 4) if x1_h == x1_h else None, "err_pct": round(err, 2) if err == err else None, "rho_min": round(rho_min, 4), "rho_max": round(rho_max, 4), "max_u": round(max_u, 5), "elapsed_s": round(elapsed, 1)}

elapsed = time.time() - t0
sim_data["total_elapsed_s"] = round(elapsed, 1)
sim_data["device"] = "sdaa:0"
print(f"\nTotal simulation time: {elapsed:.1f}s")

# Final measurement
if len(win) >= 10:
    avg_lo = np.mean(win, axis=0)
    x1_h = float("nan")
    for c in range(x_step, len(avg_lo) - 1):
        a_v, b_v = avg_lo[c], avg_lo[c + 1]
        if a_v <= 0.0 < b_v:
            xc = c + (0.0 - a_v) / (b_v - a_v)
            x1_cells = xc - (x_step - 0.5)
            x1_h = x1_cells / H
            break
    err = (x1_h - REF_X1_H) / REF_X1_H * 100 if x1_h == x1_h else float("nan")
    sim_data["final_X1_H"] = round(x1_h, 4) if x1_h == x1_h else None
    sim_data["final_err_pct"] = round(err, 2) if err == err else None
    print(f"Final X1/H = {x1_h:.4f}, err = {err:.1f}% (note: short run, not converged)")

# ── Save evidence.json ──
evidence = {
    "case": "backward_step_gartling_re800_er2",
    "sdaa_evidence": sdaa_evidence,
    "simulation": sim_data,
    "note": "Short validation run on SDAA device (4000 steps, m=16). Full convergence requires ~60000+ steps. See result.json for comprehensive pre-existing results from multi-grid ladder runs.",
    "reference": {
        "source": "Gartling 1990 IJNMF 11(7):953-967",
        "X1_over_H": REF_X1_H,
        "X1_over_h_step": REF_X1_H * 2.0,
        "X2_over_H": 4.85,
        "X3_over_H": 10.48,
        "err_tol_pct": 3.0,
    },
}

out_path = Path(__file__).parent / "evidence.json"
with open(out_path, "w") as fp:
    json.dump(evidence, fp, indent=2)
print(f"\nEvidence saved to {out_path}")
