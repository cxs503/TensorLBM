#!/usr/bin/env python3
"""SUBOFF best-combination + RANS k-epsilon benchmarks.

Benchmark 1: SUBOFF 4L domain + Cumulant at Re=1000
  - L=80, nx=320, ny=120, nz=120 (4L domain, 0.48% blockage)
  - u_in=0.06, Re=1000, tau=0.5144
  - Cumulant collision (collide_cumulant_d3q19), pure (no Smag)
  - from_suboff analytical normal
  - 5000 steps
  - Reference: Cf=0.042 (Blasius)

Benchmark 2: SUBOFF Re=1e5 with RANS k-epsilon
  - L=80, nx=200, ny=80, nz=80
  - u_in=0.06, Re=1e5, tau=0.500144
  - RANS k-epsilon (collide_rans_ke → collide_rans_mrt3d)
  - from_gradient normal
  - 5000 steps
  - Reference: Cf=0.00833 (ITTC)

Benchmark 3: SUBOFF Re=1e4 with RANS k-epsilon
  - Same grid, Re=1e4, tau=0.50144
  - RANS k-epsilon
  - from_gradient normal
  - 5000 steps
  - Reference: Cf=0.01875 (ITTC)

Usage:
  PYTHONPATH=src python suboff_best_rans_worker.py <bench> <device_id> <output_path>
  bench: bench1_cumulant | bench2_rans_re1e5 | bench3_rans_re1e4
"""
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d, bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.cumulant import collide_cumulant_d3q19
from tensorlbm.drag_pressure import (
    get_near_wall_3d,
    SurfaceMesh,
    drag_pressure_integration,
    drag_friction_integration,
)
from tensorlbm.suboff_cad import build_suboff_mask, SuboffConfig


# ---------------------------------------------------------------------------
# Benchmark definitions
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "bench1_cumulant": {
        "label": "4L+Cumulant Re=1000",
        "L": 80.0,
        "nx": 320, "ny": 120, "nz": 120,
        "u_in": 0.06,
        "Re": 1000,
        "tau": 0.5144,
        "collision": "cumulant",
        "normal_method": "from_suboff",
        "n_steps": 5000,
        "win": 500,
        "ref_cf": 1.328 / math.sqrt(1000),  # Blasius ≈ 0.042
        "ref_name": "Blasius Cf=1.328/sqrt(Re)",
    },
    "bench2_rans_re1e5": {
        "label": "RANS k-epsilon Re=1e5",
        "L": 80.0,
        "nx": 200, "ny": 80, "nz": 80,
        "u_in": 0.06,
        "Re": 100000,
        "tau": 0.500144,
        "collision": "rans_ke",
        "normal_method": "from_gradient",
        "n_steps": 5000,
        "win": 500,
        "ref_cf": 0.075 / (math.log10(100000) - 2.0) ** 2,  # ITTC ≈ 0.00833
        "ref_name": "ITTC-1957 Cf=0.075/(log10(Re)-2)^2",
    },
    "bench3_rans_re1e4": {
        "label": "RANS k-epsilon Re=1e4",
        "L": 80.0,
        "nx": 200, "ny": 80, "nz": 80,
        "u_in": 0.06,
        "Re": 10000,
        "tau": 0.50144,
        "collision": "rans_ke",
        "normal_method": "from_gradient",
        "n_steps": 5000,
        "win": 500,
        "ref_cf": 0.075 / (math.log10(10000) - 2.0) ** 2,  # ITTC ≈ 0.01875
        "ref_name": "ITTC-1957 Cf=0.075/(log10(Re)-2)^2",
    },
}


def run_benchmark(bench_key, device_id, output_path=None):
    """Run one benchmark and return results dict."""
    cfg = BENCHMARKS[bench_key]
    tag = f"[sdaa:{device_id} {cfg['label']}]"

    device = torch.device(f"sdaa:{device_id}")
    torch.sdaa.set_device(device)

    L = cfg["L"]
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    u_in = cfg["u_in"]
    Re = cfg["Re"]
    tau = cfg["tau"]
    n_steps = cfg["n_steps"]
    win = cfg["win"]
    ref_cf = cfg["ref_cf"]
    ref_name = cfg["ref_name"]
    collision = cfg["collision"]
    normal_method = cfg["normal_method"]

    nu = u_in * L / Re

    # Build SUBOFF bare hull mask
    config = SuboffConfig()
    radius = config.r_over_l * L
    D = 2.0 * radius
    cx = nx * 0.30
    cy = ny * 0.5
    cz = nz * 0.5

    # Wetted-area dynamic pressure scale
    dpS = 0.5 * u_in ** 2 * math.pi * D * L

    print(
        f"{tag} nx={nx} ny={ny} nz={nz} L={L} R={radius:.4f} D={D:.4f} "
        f"u_in={u_in} nu={nu:.6e} tau={tau:.6f} "
        f"collision={collision} normal={normal_method} "
        f"dpS={dpS:.6e} Cf_ref={ref_cf:.6f} ({ref_name})",
        flush=True,
    )

    t0 = time.time()

    solid, stats = build_suboff_mask(
        hull_type="bare_hull",
        nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz,
        length=L, radius=radius,
        config=config,
        device=device,
    )
    n_solid = int(solid.sum().item())
    print(f"{tag} solid cells: {n_solid}  L/D={stats['L_D_ratio']}", flush=True)

    # Near-wall mask
    near = get_near_wall_3d(solid)
    n_near = int(near.sum().item())
    print(f"{tag} near-wall cells: {n_near}", flush=True)

    # Build surface mesh with chosen normal method
    if normal_method == "from_suboff":
        mesh = SurfaceMesh.from_suboff(
            solid, near, cx, cy, cz, L, radius, config)
    elif normal_method == "from_gradient":
        mesh = SurfaceMesh.from_gradient(solid, near)
    else:
        raise ValueError(f"Unknown normal_method: {normal_method}")

    # Solid mask for NoDynamics (19, nz, ny, nx)
    sm = solid.unsqueeze(0).expand(19, nz, ny, nx)

    # Initialise flow field
    rho0 = torch.ones((nz, ny, nx), device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(
        rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device
    )
    im = float(rho0.sum().item())
    print(f"{tag} init done ({time.time() - t0:.1f}s), initial_mass={im}", flush=True)

    # Initialize RANS k-epsilon solver if needed
    ke_solver = None
    if collision == "rans_ke":
        from tensorlbm.rans_ke import KESolver, collide_rans_ke
        _, ux_init, uy_init, uz_init = macroscopic3d(f)
        ke_solver = KESolver(nu=nu, dx=1.0)
        ke_solver.initialize(ux_init, uy_init, uz_init)
        print(f"{tag} RANS k-epsilon solver initialized", flush=True)

    # History
    cd_p_hist, cd_f_hist, cd_tot_hist = [], [], []
    diverged = False
    last_step = 0

    for step in range(1, n_steps + 1):
        # 1. Save pre-collision state
        f_pre = f.clone()

        # 2. Collision (model-specific)
        if collision == "cumulant":
            f = collide_cumulant_d3q19(f, tau=tau)
        elif collision == "rans_ke":
            f = collide_rans_ke(
                f, tau=tau, ke_solver=ke_solver, mask=solid,
                lattice="D3Q19", collision="MRT",
            )
        else:
            raise ValueError(f"Unknown collision: {collision}")

        # 3. NoDynamics: restore solid cells to pre-collision values
        for q in range(19):
            f[q] = torch.where(sm[q], f_pre[q], f[q])

        # 4. Half-way bounce-back (before streaming)
        f = bounce_back_cells_3d(f, solid)

        # 5. Streaming
        f = stream3d(f)

        # 6. Far-field BC
        f = far_field_bc_3d(f, u_in)

        # 7. Mass correction every 200 steps
        if step % 200 == 0:
            f = correct_mass3d(f, im)

        # 8. Drag computation
        fx_p, _, _ = drag_pressure_integration(f, mesh, dpS)
        fx_f, _, _ = drag_friction_integration(f, mesh, dpS, nu)
        cd_p_hist.append(fx_p)
        cd_f_hist.append(fx_f)
        cd_tot_hist.append(fx_p + fx_f)
        last_step = step

        # Divergence guard
        if not torch.isfinite(f).all():
            print(f"{tag} DIVERGED at step {step}", flush=True)
            diverged = True
            break

        # Progress report
        if step % 500 == 0:
            n_avg = min(500, len(cd_tot_hist))
            cd_p_avg = sum(cd_p_hist[-n_avg:]) / n_avg
            cd_f_avg = sum(cd_f_hist[-n_avg:]) / n_avg
            cd_tot_avg = sum(cd_tot_hist[-n_avg:]) / n_avg
            elapsed = time.time() - t0
            print(
                f"{tag} step={step} Cd_p={cd_p_avg:.6f} Cd_f={cd_f_avg:.6f} "
                f"Cd_tot={cd_tot_avg:.6f} Cf_ref={ref_cf:.6f} ({elapsed:.0f}s)",
                flush=True,
            )

    elapsed = time.time() - t0

    # Final averages (last win steps or all if fewer)
    n_final = min(win, len(cd_tot_hist))
    if n_final == 0:
        cd_p_final = cd_f_final = cd_tot_final = float("nan")
    else:
        cd_p_final = sum(cd_p_hist[-n_final:]) / n_final
        cd_f_final = sum(cd_f_hist[-n_final:]) / n_final
        cd_tot_final = sum(cd_tot_hist[-n_final:]) / n_final

    # Cf = Cd_total (since dpS is wetted-area based)
    Cf_num = cd_tot_final
    err_pct = (
        abs(Cf_num - ref_cf) / ref_cf * 100
        if ref_cf > 0 and not diverged
        else float("inf")
    )

    result = {
        "benchmark": bench_key,
        "label": cfg["label"],
        "device": f"sdaa:{device_id}",
        "Re": Re,
        "L": L,
        "D": D,
        "radius": radius,
        "grid": f"{nx}x{ny}x{nz}",
        "u_in": u_in,
        "nu": nu,
        "tau": tau,
        "collision": collision,
        "normal_method": normal_method,
        "n_steps": n_steps,
        "steps_completed": last_step,
        "n_solid": n_solid,
        "n_near": n_near,
        "dpS": dpS,
        "dpS_formula": "0.5*u_in^2*pi*D*L (wetted area)",
        "Cd_pressure": cd_p_final,
        "Cd_friction": cd_f_final,
        "Cd_total": cd_tot_final,
        "Cf_numerical": Cf_num,
        "Cf_ref": ref_cf,
        "ref_name": ref_name,
        "error_pct": err_pct,
        "finite": bool(torch.isfinite(f).all().item()) if not diverged else False,
        "diverged": diverged,
        "elapsed_s": elapsed,
        "avg_window": win,
    }

    status = "DIVERGED" if diverged else "OK"
    print(
        f"{tag} DONE [{status}] Cd_p={cd_p_final:.6f} Cd_f={cd_f_final:.6f} "
        f"Cd_tot={cd_tot_final:.6f} Cf={Cf_num:.6f} "
        f"(ref Cf={ref_cf:.6f}) err={err_pct:.1f}% time={elapsed:.0f}s",
        flush=True,
    )

    if output_path:
        Path(output_path).write_text(json.dumps(result, indent=2))
        print(f"{tag} Results written to {output_path}", flush=True)

    return result


def main():
    if len(sys.argv) < 3:
        print("Usage: python suboff_best_rans_worker.py <bench> <device_id> [output_path]")
        print(f"  bench: {list(BENCHMARKS.keys())}")
        sys.exit(1)

    bench_key = sys.argv[1]
    device_id = int(sys.argv[2])
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if bench_key not in BENCHMARKS:
        print(f"Unknown benchmark: {bench_key}. Choose from {list(BENCHMARKS.keys())}")
        sys.exit(1)

    run_benchmark(bench_key, device_id, output_path)


if __name__ == "__main__":
    main()
