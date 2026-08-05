"""CUMULANT D3Q27 bluff body worker — cylinder & sphere drag benchmark.

Uses wall_function_d3q27 + collide_cumulant_smag_d3q27 + far_field_bc_27.
Tests the hypothesis that non-dissipative CUMULANT preserves vortex shedding
better than D3Q19 MRT+Smagorinsky.

Usage:
    python cumulant_bluff_worker.py <device_id> <case> <re> <n_steps> <warmup> <output_path>

    case: "cylinder" (200×80×4, D=24) or "sphere" (120×60×60, D=24)
"""
from __future__ import annotations
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.d3q27 import equilibrium27, macroscopic27, correct_mass27, stream27
from tensorlbm.boundaries_d3q27 import far_field_bc_27
from tensorlbm.wall_model import wall_function_d3q27
from tensorlbm.cumulant_smag import collide_cumulant_smag_d3q27


_REF_CD = {
    "cylinder": {200: 1.30, 500: 1.20},
    "sphere": {1000: 0.47, 10000: 0.40},
}


def build_cylinder_mask(nx, ny, nz, cx, cy, radius, device):
    """Boolean solid mask for a cylinder extruded along z-axis."""
    yy, xx = torch.meshgrid(
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    solid = circle.unsqueeze(0).expand(nz, ny, nx).clone()
    return solid


def build_sphere_mask(nx, ny, nz, cx, cy, cz, radius, device):
    """Boolean solid mask for a 3D sphere."""
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nx, device=device, dtype=torch.float32),
        indexing="ij",
    )
    sphere = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2
    return sphere


def main():
    device_id = int(sys.argv[1])
    case = sys.argv[2]         # "cylinder" or "sphere"
    re = float(sys.argv[3])
    n_steps = int(sys.argv[4])
    warmup = int(sys.argv[5])
    output_path = sys.argv[6]

    # Fixed geometry
    diameter = 24.0
    radius = diameter / 2.0
    u_in = 0.08
    cs_smag = 0.05

    if case == "cylinder":
        nx, ny, nz = 200, 80, 4
        cx_geom = nx * 0.25
        cy_geom = ny * 0.5
        cz_geom = nz * 0.5
    elif case == "sphere":
        nx, ny, nz = 120, 60, 60
        cx_geom = nx * 0.25
        cy_geom = ny * 0.5
        cz_geom = nz * 0.5
    else:
        raise ValueError(f"Unknown case: {case}")

    nu = u_in * diameter / re
    tau = 3.0 * nu + 0.5

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    tag = f"[SDAA:{device_id} D3Q27-CUMULANT+Smag {case} Re={int(re)}]"
    print(f"{tag} nx={nx} ny={ny} nz={nz} D={diameter} u_in={u_in} nu={nu:.6e} tau={tau:.6f} Cs={cs_smag}",
          flush=True)

    t0 = time.time()

    # Build geometry mask
    if case == "cylinder":
        solid = build_cylinder_mask(nx, ny, nz, cx_geom, cy_geom, radius, device)
    else:
        solid = build_sphere_mask(nx, ny, nz, cx_geom, cy_geom, cz_geom, radius, device)

    # Projected frontal area for Cd
    if case == "cylinder":
        A_frontal = diameter * nz   # D × span
    else:
        A_frontal = math.pi * radius ** 2  # πR²

    dyn_p = 0.5 * 1.0 * u_in ** 2 * A_frontal

    # Initialize
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium27(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))
    initial_mass = float(torch.ones_like(rho0).sum().item())

    print(f"{tag} init done ({time.time() - t0:.1f}s), solid_cells={solid.sum().item()}", flush=True)

    cd_hist = []

    for step in range(1, n_steps + 1):
        # 1. Collision: D3Q27 CUMULANT + Smagorinsky (domain-averaged tau)
        f = collide_cumulant_smag_d3q27(f, tau=tau, C_s=cs_smag)

        # 2. Stream
        f = stream27(f)

        # 3. Wall function (log-law body force + drag computation)
        f, drag_fric, drag_pres = wall_function_d3q27(f, solid, nu)

        # 4. Far-field BC
        f = far_field_bc_27(f, u_in=u_in)

        # 5. Mass correction every 100 steps
        if step % 100 == 0:
            f = correct_mass27(f, initial_mass)

        # Compute Cd
        cd_fric = drag_fric / dyn_p if dyn_p > 0 else 0.0
        cd_pres = drag_pres / dyn_p if dyn_p > 0 else 0.0
        cd_total = cd_fric + cd_pres

        if step > warmup and math.isfinite(cd_total):
            cd_hist.append(cd_total)

        # Check for divergence
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            break

        if step % 200 == 0:
            n_samples = max(len(cd_hist), 1)
            cd_avg = sum(cd_hist) / n_samples if cd_hist else float("nan")
            elapsed = time.time() - t0
            print(f"{tag} step={step} Cd_f={cd_fric:.4f} Cd_p={cd_pres:.4f} "
                  f"Cd={cd_total:.4f} Cd_avg={cd_avg:.4f} n={n_samples} ({elapsed:.0f}s)",
                  flush=True)

    elapsed = time.time() - t0

    # Final statistics
    if cd_hist:
        cd_mean = sum(cd_hist) / len(cd_hist)
        n = len(cd_hist)
        cd_std = math.sqrt(sum((c - cd_mean) ** 2 for c in cd_hist) / max(n - 1, 1)) if n > 1 else 0.0
        cd_min = min(cd_hist) if cd_hist else cd_mean
        cd_max = max(cd_hist) if cd_hist else cd_mean
    else:
        cd_mean = float("nan")
        cd_std = 0.0
        cd_min = float("nan")
        cd_max = float("nan")

    ref_cd = _REF_CD.get(case, {}).get(int(re), float("nan"))
    err_pct = abs(cd_mean - ref_cd) / ref_cd * 100 if ref_cd and ref_cd > 0 and math.isfinite(cd_mean) else float("nan")

    # Vortex shedding assessment
    shedding_detected = cd_std > 0.005 if math.isfinite(cd_std) else False

    result = {
        "case": f"{case}_Re{int(re)}",
        "device": f"sdaa:{device_id}",
        "lattice": "D3Q27",
        "collision": f"CUMULANT+Smag(Cs={cs_smag})",
        "boundary": "wall_function_d3q27+far_field_bc_27",
        "grid": f"{nx}x{ny}x{nz}",
        "diameter": diameter,
        "u_in": u_in,
        "Re": re,
        "nu": nu,
        "tau": tau,
        "n_steps": n_steps,
        "warmup": warmup,
        "Cd_mean": cd_mean,
        "Cd_std": cd_std,
        "Cd_min": cd_min,
        "Cd_max": cd_max,
        "Cd_ref": ref_cd,
        "error_pct": err_pct,
        "vortex_shedding": shedding_detected,
        "cd_samples": len(cd_hist),
        "finite": bool(torch.isfinite(f).all().item()),
        "elapsed_s": elapsed,
    }

    print(f"{tag} DONE Cd={cd_mean:.4f}±{cd_std:.4f} (ref={ref_cd}) err={err_pct:.1f}% "
          f"shedding={'YES' if shedding_detected else 'NO'} time={elapsed:.0f}s",
          flush=True)

    Path(output_path).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
