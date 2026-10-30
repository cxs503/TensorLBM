#!/usr/bin/env python3
"""Cavity Re=100 verification: run simulation + generate comparison plots."""
import sys, json, time
from pathlib import Path
import numpy as np

sys.path.insert(0, "/root/tp-agent/work/m349/src")
sys.path.insert(0, "/root/tp-agent/work/m349/benchmarks")

import torch
from tensorlbm.d2q9 import OPPOSITE, equilibrium, macroscopic
from tensorlbm.lid_driven_cavity import GHIA_RE100, zou_he_moving_lid
from tensorlbm.solver import collide_mrt, stream

def stationary_pre_bounce(f_pre, f, wall):
    opp = OPPOSITE.to(f.device)
    return torch.where(wall.unsqueeze(0), f_pre[opp], f)

def run_cavity(nx, re=100.0, u_lid=0.06, steps=80000):
    ny = nx
    tau = 3.0 * u_lid * nx / re + 0.5
    device = torch.device("cpu")
    rho0 = torch.ones((ny, nx), device=device)
    u0 = torch.zeros((ny, nx), device=device)
    f = equilibrium(rho0, u0, u0)

    wall_mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall_mask[0, :] = True
    wall_mask[:, 0] = True
    wall_mask[:, -1] = True
    interior = ~wall_mask
    resid_mask = interior.clone()
    resid_mask[-1, :] = False

    t0 = time.time()
    for step in range(1, steps + 1):
        f_pre = f
        f = collide_mrt(f, tau=tau)
        f = stationary_pre_bounce(f_pre, f, wall_mask)
        f = stream(f)
        f = zou_he_moving_lid(f, u_lid)
    elapsed = time.time() - t0

    rho, ux, uy = macroscopic(f)
    ux_w = ux.masked_fill(wall_mask, 0.0)
    uy_w = uy.masked_fill(wall_mask, 0.0)
    ux_np = ux_w.detach().cpu().numpy() / u_lid
    uy_np = uy_w.detach().cpu().numpy() / u_lid

    x_mid, y_mid = nx // 2, ny // 2
    y_pos = np.linspace(0.0, 1.0, ny)
    x_pos = np.linspace(0.0, 1.0, nx)
    u_cl = ux_np[:, x_mid]
    v_cl = uy_np[y_mid, :]
    u_gi = np.interp(GHIA_RE100["y"], y_pos, u_cl)
    v_gi = np.interp(GHIA_RE100["x"], x_pos, v_cl)
    rmse_u = float(np.sqrt(np.mean((u_gi - np.array(GHIA_RE100["u"])) ** 2)))
    rmse_v = float(np.sqrt(np.mean((v_gi - np.array(GHIA_RE100["v"])) ** 2)))
    dev = np.concatenate(
        [np.abs(u_gi - np.array(GHIA_RE100["u"])), np.abs(v_gi - np.array(GHIA_RE100["v"]))]
    )
    max_abs_dev_pct = 100.0 * float(dev.max())

    u_mid = float(np.interp(0.5, y_pos, u_cl))
    v_mid = float(np.interp(0.5, x_pos, v_cl))

    return {
        "nx": nx, "tau": round(tau, 4), "steps": steps, "elapsed_s": round(elapsed, 1),
        "u_cl": u_cl.tolist(), "v_cl": v_cl.tolist(), "y_pos": y_pos.tolist(), "x_pos": x_pos.tolist(),
        "u_gi": u_gi.tolist(), "v_gi": v_gi.tolist(),
        "rmse_u": rmse_u, "rmse_v": rmse_v, "max_abs_dev_pct": max_abs_dev_pct,
        "u_mid": u_mid, "v_mid": v_mid,
        "ux_field": ux_np.tolist(), "uy_field": uy_np.tolist(),
    }

# Run at nx=64
print("Running cavity Re=100 at nx=64, 80k steps...")
result = run_cavity(64, steps=80000)
print(f"  RMSE_u={result['rmse_u']:.4f}, RMSE_v={result['rmse_v']:.4f}")
print(f"  Max abs dev = {result['max_abs_dev_pct']:.2f}%")
print(f"  u(0.5,0.5) = {result['u_mid']:.4f} (Ghia: -0.2058)")
print(f"  v(0.5,0.5) = {result['v_mid']:.4f} (Ghia: +0.1608)")

# Save results
out_path = Path("/root/tp-agent/work/m349/artifacts/cavity_re100_verification.json")
out_path.parent.mkdir(parents=True, exist_ok=True)
json.dump({k: v for k, v in result.items()}, out_path.open("w"), indent=2)
print(f"Saved {out_path}")

# Generate plots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Plot 1: u-velocity along vertical midline
ax = axes[0]
ax.plot(result["u_gi"], GHIA_RE100["y"], "ro-", markersize=4, label="TensorLBM nx=64")
ax.plot(GHIA_RE100["u"], GHIA_RE100["y"], "b--", label="Ghia 1982")
ax.set_xlabel("u (midline, normalized)")
ax.set_ylabel("y")
ax.set_title("Vertical midline u-velocity")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 2: v-velocity along horizontal midline
ax = axes[1]
ax.plot(result["v_gi"], GHIA_RE100["x"], "ro-", markersize=4, label="TensorLBM nx=64")
ax.plot(GHIA_RE100["v"], GHIA_RE100["x"], "b--", label="Ghia 1982")
ax.set_xlabel("v (midline, normalized)")
ax.set_ylabel("x")
ax.set_title("Horizontal midline v-velocity")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Plot 3: Speed field
ax = axes[2]
ux_f = np.array(result["ux_field"])
uy_f = np.array(result["uy_field"])
speed = np.sqrt(ux_f**2 + uy_f**2)
im = ax.imshow(speed.T, origin="lower", cmap="viridis", extent=[0,1,0,1])
ax.set_title(f"Speed field (max dev={result['max_abs_dev_pct']:.2f}%)")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
fig_path = Path("/root/tp-agent/work/m349/artifacts/cavity_re100_comparison.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"Saved {fig_path}")

# Load existing verified results for comparison
existing = json.loads(Path("/root/tp-agent/work/m349/benchmarks/verified/cavity/re100/result.json").read_text())
print("\n=== Existing verified results (from result.json) ===")
for grid_key in ["128", "192"]:
    g = existing["grids"][grid_key]
    print(f"  nx={g['nx']}: max_dev={g['max_abs_dev_pct']:.2f}%, rmse_u={g['rmse_u']:.4f}, rmse_v={g['rmse_v']:.4f}")
print(f"  Verified: {existing['verified']}")
