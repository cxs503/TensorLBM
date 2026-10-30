#!/usr/bin/env python3
"""bstep_3d SDAA resolution ladder: H=10,16,24 (+32 if feasible).

Diagnoses X_r/h convergence on SDAA for the backward-facing step benchmark.
Re=100, ER≈1.94, spanwise-periodic 3D (D3Q19).
"""
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

import numpy as np
import torch

# Use local TensorLBM source
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import tensorlbm.boundaries3d as _b3d
import tensorlbm.d3q19 as _d3q19
from tensorlbm.boundaries3d import bounce_back_cells_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d_roll

# ---------- constants ----------
_INLET_DIRS = [1, 7, 9, 11, 13]
_INLET_OPP  = [2, 8, 10, 12, 14]
_OUTLET_DIRS = [2, 8, 10, 12, 14]
_OUTLET_OPP  = [1, 7, 9, 11, 13]
X_R_REF = 3.0
ER_REF  = 1.94

def _patch(device):
    _d3q19.C = _d3q19.C.to(device=device, dtype=torch.float32)
    _d3q19.OPPOSITE = _d3q19.OPPOSITE.to(device)
    _b3d.OPPOSITE = _d3q19.OPPOSITE

# ---------- geometry ----------
def grid_dims(h):
    ny = int(round(2.05 * h))
    nx = 10 * h
    nz = 2 * h
    x_step = 2 * h
    return nx, ny, nz, x_step

def make_solid(nz, ny, nx, step_h, x_step, device):
    solid = torch.zeros((nz, ny, nx), dtype=torch.bool, device=device)
    solid[:, -1, :] = True          # top wall
    solid[:, 0, x_step:] = True     # bottom wall after step
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=device),
        torch.arange(ny, device=device),
        torch.arange(nx, device=device),
        indexing="ij")
    solid |= (xx < x_step) & (yy < step_h)  # step block
    return solid

def parabolic_profile(ny, step_h, u_max):
    H = float(ny - 1 - step_h)
    yp = np.arange(ny, dtype=np.float64) - (step_h - 0.5)
    u = np.zeros(ny, dtype=np.float64)
    inside = (yp >= 0.0) & (yp <= H)
    u[inside] = 4.0 * u_max * yp[inside] * (H - yp[inside]) / (H * H)
    return u

# ---------- BCs ----------
def inlet_bc(f, ux_field):
    """Zou/He velocity inlet at x=0. ux_field shape: (nz, ny)."""
    # f shape: (19, nz, ny, nx)
    # Extract known populations at x=0
    s = f[0, :, :, 0].clone()
    for d in [3, 4, 5, 6, 15, 16, 17, 18]:
        s = s + f[d, :, :, 0]
    sn = f[2, :, :, 0].clone()
    for d in [8, 10, 12, 14]:
        sn = sn + f[d, :, :, 0]
    rho = (s + 2.0 * sn) / (1.0 - ux_field)  # (nz, ny)
    # Build 3D velocity fields for equilibrium (nz, ny, 1)
    rho_3d = rho.unsqueeze(-1)       # (nz, ny, 1)
    ux_3d = ux_field.unsqueeze(-1)   # (nz, ny, 1)
    uy_3d = torch.zeros_like(ux_3d)
    uz_3d = torch.zeros_like(ux_3d)
    feq = equilibrium3d(rho_3d, ux_3d, uy_3d, uz_3d)  # (19, nz, ny, 1)
    for d, od in zip(_INLET_DIRS, _INLET_OPP):
        f[d, :, :, 0] = feq[d, :, :, 0] - feq[od, :, :, 0] + f[od, :, :, 0]
    return f

def outlet_bc(f, rho_out=1.0):
    """Zou/He pressure outlet at x=nx-1."""
    s = f[0, :, :, -1].clone()
    for d in [3, 4, 5, 6, 15, 16, 17, 18]:
        s = s + f[d, :, :, -1]
    sp = f[1, :, :, -1].clone()
    for d in [7, 9, 11, 13]:
        sp = sp + f[d, :, :, -1]
    ux_out = -1.0 + (s + 2.0 * sp) / rho_out  # (nz, ny)
    rho_3d = torch.full_like(ux_out.unsqueeze(-1), rho_out)
    ux_3d = ux_out.unsqueeze(-1)
    uy_3d = torch.zeros_like(ux_3d)
    uz_3d = torch.zeros_like(ux_3d)
    feq = equilibrium3d(rho_3d, ux_3d, uy_3d, uz_3d)  # (19, nz, ny, 1)
    for d, od in zip(_OUTLET_DIRS, _OUTLET_OPP):
        f[d, :, :, -1] = feq[d, :, :, 0] - feq[od, :, :, 0] + f[od, :, :, -1]
    return f

# ---------- measurement ----------
def _zero_crossing(row, x_step):
    """Find first sign change from negative to positive in row."""
    pos = int(np.argmax(row > 0.0))
    if pos == 0 or row[pos] <= 0.0:
        return None
    a, b = row[pos-1], row[pos]
    if b - a == 0.0:
        return None
    return x_step + (pos - 1) + (0.0 - a) / (b - a)

def measure_xr(ux, x_step, step_h):
    """Measure reattachment length X_r/h."""
    nz = ux.shape[0]
    ux_np = ux.detach().cpu().numpy()
    xr_planes = []
    for z in range(nz):
        zc = _zero_crossing(ux_np[z, 1, x_step:], x_step)
        xr_planes.append(0.0 if zc is None else float((zc - (x_step - 0.5)) / step_h))
    xr_arr = np.array(xr_planes)
    row_mean = ux_np[:, 1, x_step:].mean(axis=0)
    zc_mean = _zero_crossing(row_mean, x_step)
    xr_main = 0.0 if zc_mean is None else float((zc_mean - (x_step - 0.5)) / step_h)
    return {
        "xr_h": xr_main,
        "xr_planes_mean": float(xr_arr.mean()),
        "xr_planes_std": float(xr_arr.std()),
    }

# ---------- ER effective ----------
def compute_er_effective(ny, step_h):
    return (ny - 2) / (ny - 1 - step_h)

# ---------- run single case ----------
def run_single(h, steps, device, u_max=0.05, re=100.0, out_interval=5000):
    nx, ny, nz, x_step = grid_dims(h)
    er_nominal = compute_er_effective(ny, h)
    nu = u_max * h / re
    tau = 3.0 * nu + 0.5

    _patch(device)
    solid = make_solid(nz, ny, nx, h, x_step, device)

    prof = parabolic_profile(ny, h, u_max)
    ux0 = torch.tensor(prof, dtype=torch.float32, device=device)
    ux0 = ux0.view(1, ny, 1).expand(nz, ny, nx).contiguous()
    rho0 = torch.ones((nz, ny, nx), dtype=torch.float32, device=device)
    uy0 = torch.zeros_like(rho0)
    uz0 = torch.zeros_like(rho0)
    ux0 = ux0.masked_fill(solid, 0.0)
    f = equilibrium3d(rho0, ux0, uy0, uz0, device=device)

    u_in_field = torch.tensor(
        np.broadcast_to(prof.reshape(1, ny), (nz, ny)).astype(np.float32),
        device=device)

    # Time loop
    t0 = time.time()
    xr_series = []
    step_list = []
    for n in range(1, steps + 1):
        f = collide_bgk3d(f, tau)
        f = stream3d_roll(f)
        f = bounce_back_cells_3d(f, solid)
        f = inlet_bc(f, u_in_field)
        f = outlet_bc(f, 1.0)

        if n % out_interval == 0 or n == steps:
            macro = macroscopic3d(f)
            ux_field = macro[1]  # (rho, ux, uy, uz)
            xr = measure_xr(ux_field, x_step, h)
            xr_series.append(xr["xr_h"])
            step_list.append(n)
            if n % (out_interval * 2) == 0 or n == steps:
                print(f"    step {n}: X_r/h = {xr['xr_h']:.4f}")

    wall_time = time.time() - t0

    # Final measurement
    macro = macroscopic3d(f)
    ux_final = macro[1]
    xr_final = measure_xr(ux_final, x_step, h)

    # Inlet check
    target = parabolic_profile(ny, h, u_max)
    actual = ux_final[:, :, 0].mean(dim=0).detach().cpu().numpy()
    fluid_idx = np.arange(h, ny - 1)
    dev = float(np.abs(actual[fluid_idx] - target[fluid_idx]).max() / u_max)

    return {
        "h": h,
        "nx": nx, "ny": ny, "nz": nz,
        "x_step": x_step,
        "er_nominal": er_nominal,
        "er_dev_pct": abs(er_nominal - ER_REF) / ER_REF * 100,
        "tau": tau,
        "nu": nu,
        "xr_h": xr_final["xr_h"],
        "xr_h_err_pct": (xr_final["xr_h"] - X_R_REF) / X_R_REF * 100,
        "xr_planes_mean": xr_final["xr_planes_mean"],
        "xr_planes_std": xr_final["xr_planes_std"],
        "xr_series": xr_series,
        "steps": step_list,
        "wall_time_s": wall_time,
        "inlet_max_dev": dev,
        "n_steps": steps,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h-values", default="10,16,24")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--device", default="sdaa:0")
    parser.add_argument("--out-dir", default="/root/tp-agent/work/m411/artifacts/bstep3d_ladder")
    args = parser.parse_args()

    device = torch.device(args.device)
    h_values = [int(x) for x in args.h_values.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    evidence = {
        "device": str(device),
        "torch_version": torch.__version__,
        "sdaa_version": str(torch.version.sdaa) if hasattr(torch.version, 'sdaa') else "N/A",
        "sdaa_device_name": torch.sdaa.get_device_name(0) if torch.sdaa.is_available() else "N/A",
        "sdaa_device_count": torch.sdaa.device_count() if torch.sdaa.is_available() else 0,
        "runs": [],
    }

    for h in h_values:
        nx, ny, nz, x_step = grid_dims(h)
        er = compute_er_effective(ny, h)
        print(f"\n{'='*60}")
        print(f"  Running H={h}, steps={args.steps}, device={device}")
        print(f"  Grid: ({nx},{ny},{nz}), ER={er:.6f}")
        print(f"{'='*60}")
        t_start = time.time()
        try:
            res = run_single(h, args.steps, device)
            res["status"] = "OK"
        except Exception as e:
            res = {"h": h, "status": "FAIL", "error": str(e), "traceback": traceback.format_exc()}
            print(f"  FAILED: {e}")
            traceback.print_exc()
        t_wall = time.time() - t_start
        res["wall_time_s"] = t_wall
        results.append(res)

        evidence["runs"].append({
            "h": h,
            "device": str(device),
            "steps": args.steps,
            "wall_time_s": t_wall,
            "xr_h": res.get("xr_h"),
            "status": res["status"],
        })

        if res["status"] == "OK":
            print(f"  H={h}: X_r/h = {res['xr_h']:.4f}, "
                  f"err = {res['xr_h_err_pct']:.2f}%, "
                  f"ER_eff = {res['er_nominal']:.6f}, "
                  f"wall = {t_wall:.1f}s")
        else:
            print(f"  H={h}: FAIL")

    # Save results
    ladder_data = {
        "case": "bstep_3d_sdaa_resolution梯子",
        "reference": {"xr_h_ref": X_R_REF, "er_ref": ER_REF},
        "device": str(device),
        "results": results,
    }
    with open(out_dir / "ladder_results.json", "w") as f:
        json.dump(ladder_data, f, indent=2)

    evidence["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_dir / "evidence.json", "w") as f:
        json.dump(evidence, f, indent=2)

    # Summary table
    print(f"\n{'='*60}")
    print("  RESOLUTION LADDER SUMMARY")
    print(f"{'='*60}")
    print(f"{'H':>4} {'nx':>5} {'ny':>4} {'nz':>4} {'ER_eff':>8} {'X_r/h':>8} {'err%':>8} {'wall_s':>8}")
    for r in results:
        if r["status"] == "OK":
            print(f"{r['h']:>4} {r['nx']:>5} {r['ny']:>4} {r['nz']:>4} "
                  f"{r['er_nominal']:>8.5f} {r['xr_h']:>8.4f} {r['xr_h_err_pct']:>8.2f} {r['wall_time_s']:>8.1f}")
        else:
            print(f"{r['h']:>4} {'FAIL':>5}")

    return results


if __name__ == "__main__":
    main()
