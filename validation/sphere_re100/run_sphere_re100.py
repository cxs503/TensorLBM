"""VAL-SUITE: 3-D sphere flow at Re=100 — grid convergence & drag validation.

Uses TensorLBM D3Q19-BGK with far-field boundary conditions for accurate
drag prediction.  Runs three grid resolutions (D=8, 10, 12 lattice diameters)
and compares time-averaged C_D against the Schiller-Naumann correlation.

Machine criteria:
  - C_D (blockage-corrected) within [1.03, 1.10] band
  - Grid convergence trend monotonic (error decreases with resolution)

Usage
-----
    PYTHONPATH=src python validation/sphere_re100/run_sphere_re100.py
    PYTHONPATH=src python validation/sphere_re100/run_sphere_re100.py --device cuda

Output
------
    runs/sphere_re100/
    ├── convergence_summary.json   Per-resolution C_D data
    ├── verdict.json               PASS/FAIL verdict
    ├── symmetry_velocity_contour.png   Symmetry-plane |u| field
    └── grid_convergence.png       C_D and error vs resolution

Reference
---------
Schiller, L. & Naumann, A. (1933). C_D = 24/Re * (1 + 0.15 * Re^0.687)
At Re=100: C_D = 1.0917
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tensorlbm.boundaries3d import sphere_mask
from tensorlbm.boundary.boundaries3d import far_field_bc_3d
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d
from tensorlbm.obstacles import compute_obstacle_forces_3d
from tensorlbm.solver3d import collide_bgk3d, stream3d


def cd_schiller_naumann(re: float) -> float:
    if re <= 0.0:
        return float("inf")
    return 24.0 / re * (1.0 + 0.15 * re ** 0.687)


def blockage_correction(cd_meas: float, beta: float) -> float:
    return cd_meas / (1.0 + 2.5 * beta)


def run_single(re, D, steps, u_in=0.1, device="cpu", save_fields=False):
    """Run one D3Q19-BGK sphere flow with far-field BC."""
    dev = torch.device(device)
    R = D / 2.0
    nu = u_in * D / re
    tau = 3.0 * nu + 0.5

    # Domain: sphere at 25% x, centred in y-z cross-section
    ny = nz = max(int(math.ceil(R * math.sqrt(math.pi / 0.10))), D + 4)
    nx = max(int(10.0 * D), 4 * D)
    cx, cy, cz = nx * 0.25, ny * 0.5, nz * 0.5

    solid = sphere_mask(nx, ny, nz, cx, cy, cz, R, device=dev)

    rho0 = torch.ones(nz, ny, nx, device=dev)
    ux0 = torch.full((nz, ny, nx), u_in, device=dev)
    ux0[solid] = 0.0
    f = equilibrium3d(rho0, ux0, torch.zeros_like(ux0), torch.zeros_like(ux0))

    area = math.pi * R * R
    dyn = 0.5 * u_in * u_in * area
    beta = math.pi * R * R / (ny * ny)

    cd_hist = []
    t0 = time.time()

    for step in range(1, steps + 1):
        f = collide_bgk3d(f, tau=tau)
        f = stream3d(f)
        fx, _, _ = compute_obstacle_forces_3d(f, solid)
        f = far_field_bc_3d(f, u_in=u_in, obstacle_mask=solid)
        cd = abs(float(fx.item())) / dyn
        if step > steps // 2:
            cd_hist.append(cd)
        if step % max(1, steps // 5) == 0:
            avg = sum(cd_hist) / max(len(cd_hist), 1) if cd_hist else float("nan")
            print(
                f"  step {step:5d}/{steps}  Cd(avg)={avg:.4f}  t={time.time()-t0:.1f}s",
                flush=True,
            )

    cd_mean = sum(cd_hist) / len(cd_hist)
    cd_corr = blockage_correction(cd_mean, beta)
    ref_cd = cd_schiller_naumann(re)

    result = {
        "D": D, "nx": nx, "ny": ny, "nz": nz,
        "re": re, "u_in": float(u_in), "tau": float(tau), "nu": float(nu),
        "blockage_beta": float(beta), "steps": steps,
        "cd_raw": float(cd_mean), "cd_corr": float(cd_corr),
        "ref_cd_sn": float(ref_cd),
        "err_raw_pct": float(abs(cd_mean - ref_cd) / ref_cd * 100),
        "err_corr_pct": float(abs(cd_corr - ref_cd) / ref_cd * 100),
        "wall_time_s": float(time.time() - t0),
    }

    if save_fields:
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(solid, 0.0)
        uy = uy.masked_fill(solid, 0.0)
        uz = uz.masked_fill(solid, 0.0)
        speed = torch.sqrt(ux * ux + uy * uy + uz * uz)
        mid_z = nz // 2
        result["_speed_sym"] = speed[mid_z].detach().cpu().numpy()
        result["_obs_sym"] = solid[mid_z].detach().cpu().float().numpy()

    return result


def main():
    parser = argparse.ArgumentParser(description="VAL-SUITE sphere Re=100")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-root", default="runs/sphere_re100")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA unavailable, falling back to CPU")
        device = "cpu"
    torch.set_num_threads(4)

    re = 100.0
    ref_cd = cd_schiller_naumann(re)
    print(f"=== VAL-SUITE: Sphere Re={re}  Schiller-Naumann C_D = {ref_cd:.4f} ===")
    print(f"   Acceptance band: 1.03 ≤ C_D ≤ 1.10")
    print(f"   Boundary: far-field (free-stream equilibrium on lateral faces)")
    print()

    resolutions = [
        {"D": 8,  "steps": 1500, "u_in": 0.1},
        {"D": 10, "steps": 2000, "u_in": 0.1},
        {"D": 12, "steps": 1500, "u_in": 0.1},
    ]

    os.makedirs(args.output_root, exist_ok=True)
    all_results = []
    finest_fields = None

    for i, res in enumerate(resolutions):
        D = res["D"]; steps = res["steps"]; u_in = res["u_in"]
        is_finest = i == len(resolutions) - 1

        print(f"--- Resolution D={D}  steps={steps} ---", flush=True)
        result = run_single(re=re, D=D, steps=steps, u_in=u_in,
                           device=device, save_fields=is_finest)
        if "_speed_sym" in result:
            finest_fields = {
                "speed": result.pop("_speed_sym"),
                "obs": result.pop("_obs_sym"),
                "nx": result["nx"], "ny": result["ny"],
            }
        all_results.append(result)
        print(f"  C_D(raw)={result['cd_raw']:.4f}  C_D(corr)={result['cd_corr']:.4f}  "
              f"ref={ref_cd:.4f}  err_corr={result['err_corr_pct']:.2f}%")
        print()

    # Save summary
    with open(os.path.join(args.output_root, "convergence_summary.json"), "w") as fp:
        json.dump(all_results, fp, indent=2)

    # ── Figures ─────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if finest_fields is not None:
        fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
        im = ax.imshow(finest_fields["speed"], origin="lower", cmap="viridis",
                       extent=[0, finest_fields["nx"], 0, finest_fields["ny"]],
                       aspect="equal")
        obs = finest_fields["obs"]
        ax.contour(np.arange(0.5, obs.shape[1]+0.5),
                   np.arange(0.5, obs.shape[0]+0.5),
                   obs, levels=[0.5], colors="white", linewidths=1.0)
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
        ax.set_title(f"Symmetry-plane velocity magnitude |u| — Re={re}, D={all_results[-1]['D']}")
        plt.colorbar(im, ax=ax, label="|u| (lattice units)")
        fig.savefig(os.path.join(args.output_root, "symmetry_velocity_contour.png"), dpi=200)
        plt.close(fig)

    Ds = [r["D"] for r in all_results]
    cds_raw = [r["cd_raw"] for r in all_results]
    cds_corr = [r["cd_corr"] for r in all_results]
    errs = [r["err_corr_pct"] for r in all_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    ax1.plot(Ds, cds_raw, "o-", label="C_D (raw)", color="steelblue", markersize=8)
    ax1.plot(Ds, cds_corr, "s-", label="C_D (blockage-corrected)", color="darkorange", markersize=8)
    ax1.axhline(y=ref_cd, color="green", linestyle="--", linewidth=1.5,
                label=f"Schiller-Naumann = {ref_cd:.4f}")
    ax1.axhspan(1.03, 1.10, alpha=0.15, color="green", label="Acceptance band [1.03, 1.10]")
    ax1.set_xlabel("Sphere diameter D (lattice units)")
    ax1.set_ylabel("C_D")
    ax1.set_title("Grid convergence — C_D vs resolution")
    ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)

    ax2.plot(Ds, errs, "D-", color="crimson", markersize=8)
    ax2.axhline(y=3.0, color="gray", linestyle=":", linewidth=1, label="3% threshold")
    ax2.set_xlabel("Sphere diameter D (lattice units)")
    ax2.set_ylabel("Error vs S-N (%)")
    ax2.set_title("Blockage-corrected error vs resolution")
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.savefig(os.path.join(args.output_root, "grid_convergence.png"), dpi=200)
    plt.close(fig)

    # ── Verdict ─────────────────────────────────────────────────────────
    finest = all_results[-1]
    finest_cd = finest["cd_corr"]
    in_band = 1.03 <= finest_cd <= 1.10
    monotonic = all(errs[i] >= errs[i+1] for i in range(len(errs)-1))

    print("=" * 60)
    print("VERDICT")
    print("=" * 60)
    print(f"  Finest grid C_D(corr) = {finest_cd:.4f}")
    print(f"  Schiller-Naumann      = {ref_cd:.4f}")
    print(f"  In band [1.03, 1.10]?  {'PASS' if in_band else 'FAIL'}")
    print(f"  Monotonic convergence? {'PASS' if monotonic else 'FAIL'}")
    print(f"  Overall:               {'PASS' if (in_band and monotonic) else 'FAIL'}")
    print("=" * 60)

    verdict = {
        "re": re, "ref_cd_schiller_naumann": ref_cd,
        "finest_cd_corrected": finest_cd,
        "in_band": in_band, "monotonic_convergence": monotonic,
        "overall_pass": in_band and monotonic,
        "resolutions": all_results,
    }
    with open(os.path.join(args.output_root, "verdict.json"), "w") as fp:
        json.dump(verdict, fp, indent=2)


if __name__ == "__main__":
    main()
