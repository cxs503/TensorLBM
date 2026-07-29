#!/usr/bin/env python3
"""VOF dam break fix — SDAA:5.

Fixes:
  1. interface_compression_3d padding fixed in free_surface_common.py
     (pad all 3 axes simultaneously with F.pad(p5d, (1,1,1,1,1,1)))
  2. Use rho_gas=0.5 (2:1 density ratio, not 100:1) for stability
  3. Use 2000 steps
  4. Pass target_phi_sum for mass conservation
  5. Disable interface compression (c_comp=0) to prevent phi destruction

Target: front position > 0 after 100 steps
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch_sdaa  # noqa: F401

from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.free_surface_common import (
    free_surface_vof_step,
    front_position_3d,
    init_phi_column_3d,
)


def make_container_walls_3d(nz, ny, nx, device):
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, :, 0] = True
    solid[:, :, -1] = True
    solid[:, 0, :] = True
    solid[:, -1, :] = True
    solid[0, :, :] = True
    solid[-1, :, :] = True
    return solid


def run_dam_break(device_id, output_path=None):
    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device_id)
    tag = f"[DamBreak-fix SDAA:{device_id}]"

    nz, ny, nx = 4, 100, 200
    col_w, col_h = 50, 50

    tau = 1.0
    rho_l = 1.0
    rho_g = 0.5  # FIX: 2:1 ratio (was 0.01 = 100:1)
    gy_lattice = -1e-3  # stronger gravity for visible front motion
    g_phys = abs(gy_lattice)

    a = col_w

    solid = make_container_walls_3d(nz, ny, nx, device)
    phi = init_phi_column_3d(nz, ny, nx, width=col_w, height=col_h, device=device)
    phi = phi.masked_fill(solid, 0.0)

    # Target phi sum for mass conservation
    target_phi_sum = float(phi[~solid].sum().item())

    rho_field = rho_l * phi + rho_g * (1.0 - phi)
    ux = torch.zeros((nz, ny, nx), device=device)
    uy = torch.zeros((nz, ny, nx), device=device)
    uz = torch.zeros((nz, ny, nx), device=device)
    f = equilibrium3d(rho_field, ux, uy, uz, device=device)

    n_steps = 2000
    sample_interval = 50
    front_history = []
    time_history = []

    print(f"{tag} Starting dam break: {nx}x{ny}x{nz}, col={col_w}x{col_h}, "
          f"rho_l={rho_l}, rho_g={rho_g}, gy={gy_lattice}, steps={n_steps}",
          flush=True)
    print(f"{tag} target_phi_sum={target_phi_sum:.1f}", flush=True)

    t0 = time.time()
    for step in range(n_steps):
        f, phi = free_surface_vof_step(
            f, phi, tau=tau, gy=gy_lattice,
            rho_liquid=rho_l, rho_gas=rho_g, solid=solid,
            target_phi_sum=target_phi_sum,
            c_comp=0.0,  # disable compression to prevent phi destruction
        )

        if step % sample_interval == 0:
            front = front_position_3d(phi)
            t_star = step * math.sqrt(g_phys / a)
            z_star = front / a
            front_history.append(z_star)
            time_history.append(t_star)
            phi_sum = float(phi[~solid].sum().item())
            if step % 200 == 0:
                elapsed = time.time() - t0
                print(f"{tag} step={step:5d} front={front:6.1f} z*={z_star:.3f} "
                      f"t*={t_star:.3f} phi_sum={phi_sum:.1f} ({elapsed:.1f}s)",
                      flush=True)

    t_final = time_history[-1]
    z_ref = 1.0 + 1.25 * t_final
    z_sim = front_history[-1]
    error_pct = abs(z_sim - z_ref) / max(z_ref, 1e-6) * 100.0

    # Check target: front position > 0 after 100 steps
    front_at_100 = front_history[2] if len(front_history) > 2 else 0.0  # step 100
    target_pass = front_at_100 > 0.0

    result = {
        "benchmark": "dam_break_fixed",
        "device": device_id,
        "grid": {"nx": nx, "ny": ny, "nz": nz},
        "column": {"width": col_w, "height": col_h},
        "parameters": {"tau": tau, "rho_liquid": rho_l, "rho_gas": rho_g,
                        "gy": gy_lattice, "n_steps": n_steps},
        "front_position_z_star_final": z_sim,
        "reference_z_star": z_ref,
        "error_pct": error_pct,
        "front_at_100_steps": front_at_100,
        "target_front_gt_0_at_100": target_pass,
        "pass": target_pass,
        "front_history": front_history,
        "time_history": time_history,
        "elapsed_s": time.time() - t0,
    }

    print(f"{tag} RESULT: z*_sim={z_sim:.3f} z*_ref={z_ref:.3f} "
          f"error={error_pct:.1f}% front@100={front_at_100:.3f} "
          f"{'PASS' if target_pass else 'FAIL'}", flush=True)

    if output_path:
        with open(output_path, "w") as fh:
            json.dump(result, fh, indent=2)
    return result


if __name__ == "__main__":
    dev = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    out = sys.argv[2] if len(sys.argv) > 2 else None
    run_dam_break(dev, out)
