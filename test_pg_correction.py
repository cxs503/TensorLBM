"""Test pressure-gradient-corrected wall_function_3d on bare_hull SUBOFF.

Compares base vs pressure-gradient-corrected wall_function_3d.
D3Q19 + MRT+Smag Cs=0.05, 160x80x80 grid, SDAA:0.

Usage: PYTHONPATH=src python test_pg_correction.py 0
"""
import json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import torch
from tensorlbm.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import correct_mass3d, stream3d
from tensorlbm.suboff_cad import SuboffHullType, build_suboff_mask
from tensorlbm.suboff_resistance import _voxel_wetted_area
from tensorlbm.turbulence import collide_smagorinsky_mrt3d
from tensorlbm.wall_model import wall_function_3d


def run_one(label, solid, nu, tau, u_in, n_steps, dp_dx_corr, alpha, device, log_fh):
    """Run one sim, return ct_history list."""
    nz, ny, nx = solid.shape
    t0 = time.time()

    S = _voxel_wetted_area(solid, 1.0)
    dpS = 0.5 * 1.0 * u_in ** 2 * S

    rho0 = torch.ones(nz, ny, nx, device=device)
    ux0 = torch.full((nz, ny, nx), u_in, device=device)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0), device=device)
    im = float(rho0.sum().item())

    ct_hist = []
    warmup = 200

    msg = f"[{label}] Start: corr={dp_dx_corr} alpha={alpha}"
    print(msg, flush=True)
    log_fh.write(msg + "\n"); log_fh.flush()

    for step in range(1, n_steps + 1):
        f = collide_smagorinsky_mrt3d(f, tau=tau, C_s=0.05)
        f = stream3d(f)
        f, drag_fric, drag_pres = wall_function_3d(
            f, solid, nu, y_val=0.5,
            dp_dx_correction=dp_dx_corr, alpha_pg=alpha,
        )
        f = far_field_bc_3d(f, u_in=u_in)
        if step % 100 == 0:
            f = correct_mass3d(f, im)

        ct_fric = drag_fric / dpS if dpS > 0 else 0.0
        ct_pres = drag_pres / dpS if dpS > 0 else 0.0
        ct_hist.append({"step": step, "ct_fric": ct_fric, "ct_pres": ct_pres, "ct_total": ct_fric + ct_pres})

        if not torch.isfinite(f).all():
            msg = f"[{label}] DIV at step {step}"
            print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
            break

        if step % 500 == 0:
            recent = [x["ct_pres"] for x in ct_hist[warmup:]] if step > warmup else [ct_pres]
            pvar = float(torch.tensor(recent).var().item()) if len(recent) > 1 else 0.0
            msg = (f"[{label}] step={step:4d} Ct_fric={ct_fric:.6f} Ct_pres={ct_pres:.6f} "
                   f"Ct_tot={ct_fric+ct_pres:.6f} Ct_pres_var={pvar:.6e} ({time.time()-t0:.1f}s)")
            print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()

    elapsed = time.time() - t0
    msg = f"[{label}] DONE in {elapsed:.1f}s"
    print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
    return ct_hist, torch.isfinite(f).all().item()


def summarize(results, log_fh):
    """Print summary table."""
    header = f"\n{'Config':<16} {'Ct_pres_mean':>12} {'Ct_pres_var':>14} {'Ct_pres_std':>14} {'Reduction':>10} {'Ct_tot@2000':>12}"
    sep = "-" * len(header)
    for line in ["=" * 70, "SUMMARY", "=" * 70, header, sep]:
        print(line, flush=True); log_fh.write(line + "\n")
    log_fh.flush()

    base_var = results[0].get("ct_pres_var", float('nan'))
    for r in results:
        var = r.get("ct_pres_var", float('nan'))
        mean = r.get("ct_pres_mean", float('nan'))
        std = r.get("ct_pres_std", float('nan'))
        ct2k = r.get("step_2000", {}).get("ct_total", float('nan'))
        if not math.isnan(base_var) and not math.isnan(var) and base_var > 0:
            red = f"{(1.0 - var / base_var) * 100:+5.1f}%"
        else:
            red = "N/A"
        line = f"{r['name']:<16} {mean:12.6f} {var:14.6e} {std:14.6e} {red:>10} {ct2k:12.6f}"
        print(line, flush=True); log_fh.write(line + "\n")
    log_fh.flush()


if __name__ == "__main__":
    did = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = torch.device(f"sdaa:{did}")
    torch.sdaa.set_device(device)

    nx, ny, nz = 160, 80, 80
    hl, u_in, re = 120.0, 0.06, 2.0e6
    nu = u_in * hl / re
    tau = 3.0 * nu + 0.5
    n_steps = 2000
    warmup = 200

    log_path = Path(f"/tmp/pg_test_{did}.log")
    log_fh = open(str(log_path), "w")

    msg = f"Grid={nx}x{ny}x{nz} tau={tau:.6f} nu={nu:.6e} steps={n_steps}"
    print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()

    # Build geometry once
    cx, cy, cz = nx * 0.35, ny / 2.0, nz / 2.0
    solid_cpu, _ = build_suboff_mask(
        SuboffHullType.BARE_HULL, nx=nx, ny=ny, nz=nz,
        cx=cx, cy=cy, cz=cz, length=hl, device="cpu")
    solid = solid_cpu.to(device)
    msg = f"Solid cells: {solid.sum().item()}"
    print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()

    # Configs to test: base + 4 alphas
    configs = [
        ("base",   False, None),
        ("pg_a0.3", True, 0.3),
        ("pg_a0.5", True, 0.5),
        ("pg_a0.7", True, 0.7),
        ("pg_a1.0", True, 1.0),
    ]

    all_raw = []
    results = []

    for name, dp_dx_corr, alpha in configs:
        msg = f"\n{'='*60}\nRUNNING: {name}\n{'='*60}"
        print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()

        ct_hist, finite = run_one(name, solid, nu, tau, u_in, n_steps, dp_dx_corr, alpha or 0.5, device, log_fh)
        all_raw.append({"name": name, "dp_dx_correction": dp_dx_corr, "alpha": alpha, "history": ct_hist, "finite": finite})

        # Build summary row
        row = {"name": name, "alpha": alpha, "dp_dx_correction": dp_dx_corr, "finite": finite}
        for ts in [500, 1000, 1500, 2000]:
            hits = [x for x in ct_hist if x["step"] == ts]
            if hits:
                row[f"step_{ts}"] = {"ct_fric": hits[0]["ct_fric"], "ct_pres": hits[0]["ct_pres"], "ct_total": hits[0]["ct_total"]}
        pres_arr = [x["ct_pres"] for x in ct_hist[warmup:]]
        if len(pres_arr) > 1:
            row["ct_pres_mean"] = float(torch.tensor(pres_arr).mean().item())
            row["ct_pres_var"] = float(torch.tensor(pres_arr).var().item())
            row["ct_pres_std"] = float(torch.tensor(pres_arr).std().item())
        results.append(row)

    # Summary
    summarize(results, log_fh)

    # Save JSON
    json_path = Path(f"/tmp/pg_correction_test_{did}.json")
    json_path.write_text(json.dumps({"grid": f"{nx}x{ny}x{nz}", "tau": tau, "nu": nu, "results": results}, indent=2))
    msg = f"\nJSON saved to {json_path}"
    print(msg, flush=True); log_fh.write(msg + "\n"); log_fh.flush()
    log_fh.close()
