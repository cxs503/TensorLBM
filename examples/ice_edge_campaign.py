#!/usr/bin/env python3
"""Ice-edge zone LBM campaign: full parameter study and analysis.

Runs baseline, parameter scans, and generates publication-quality figures.
"""

from __future__ import annotations

import json
import hashlib
import time
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.d2q9 import C, W, OPPOSITE


def run_simulation(nx, ny, u0, re, ice_edge_frac, ls_ice, n_steps, output_dir, label=""):
    """Run a single ice-edge simulation."""
    device = torch.device("cpu")
    nu = u0 * (ny - 2) / re
    tau = 3.0 * nu + 0.5
    omega = 1.0 / tau
    x_edge = int(nx * ice_edge_frac)
    beta_ice = ls_ice / (1.0 + ls_ice)

    c = C.to(device).float()
    w = W.to(device).float()
    opp = OPPOSITE.to(device)
    mirror_y = [0, 1, 4, 3, 2, 8, 7, 6, 5]

    # Initialize
    rho = torch.ones(ny, nx, device=device)
    ux = torch.full((ny, nx), u0, device=device)
    uy = torch.zeros(ny, nx, device=device)
    u_sq = ux**2 + uy**2
    cu = c[:, 0].view(9,1,1) * ux.unsqueeze(0) + c[:, 1].view(9,1,1) * uy.unsqueeze(0)
    f = w.view(9,1,1) * rho.unsqueeze(0) * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq.unsqueeze(0))

    # Time averaging
    u_sum = torch.zeros_like(ux)
    n_avg = 0

    t0 = time.time()
    for step in range(1, n_steps + 1):
        # Macroscopic
        rho = f.sum(dim=0)
        jx = (f * c[:, 0].view(9,1,1)).sum(dim=0)
        jy = (f * c[:, 1].view(9,1,1)).sum(dim=0)
        ux = jx / rho
        uy = jy / rho

        # Collision
        u_sq = ux**2 + uy**2
        cu = c[:, 0].view(9,1,1) * ux.unsqueeze(0) + c[:, 1].view(9,1,1) * uy.unsqueeze(0)
        feq = w.view(9,1,1) * rho.unsqueeze(0) * (1.0 + 3.0*cu + 4.5*cu**2 - 1.5*u_sq.unsqueeze(0))
        f = f - omega * (f - feq)

        # Streaming
        f_new = torch.empty_like(f)
        for q in range(9):
            cx, cy = int(C[q, 0].item()), int(C[q, 1].item())
            src = f[q]
            if cy > 0:
                tmp = torch.zeros_like(src)
                tmp[cy:, :] = src[:ny-cy, :]
                src = tmp
            elif cy < 0:
                tmp = torch.zeros_like(src)
                tmp[:ny+cy, :] = src[-cy:, :]
                src = tmp
            if cx != 0:
                src = torch.roll(src, shifts=cx, dims=1)
            f_new[q] = src
        f = f_new

        # BCs
        for q in range(9):
            oq = int(opp[q].item())
            if q != oq:
                f[q, 0, :] = f[oq, 0, :]

        for q in range(9):
            oq = int(opp[q].item())
            mq = mirror_y[q]
            if C[q, 1].item() < 0:
                f[q, ny-1, :x_edge] = (1-beta_ice)*f[oq, ny-1, :x_edge] + beta_ice*f[mq, ny-1, :x_edge]
                f[q, ny-1, x_edge:] = f[mq, ny-1, x_edge:]

        # Time averaging
        if step > n_steps // 2 and step % 2 == 0:
            rho = f.sum(dim=0)
            jx = (f * c[:, 0].view(9,1,1)).sum(dim=0)
            u_sum += jx / rho
            n_avg += 1

    elapsed = time.time() - t0

    # Final fields
    if n_avg > 0:
        u_mean = u_sum / n_avg
    else:
        rho = f.sum(dim=0)
        jx = (f * c[:, 0].view(9,1,1)).sum(dim=0)
        u_mean = jx / rho

    # Wall shear
    tau_bottom = nu * (u_mean[1, :] - u_mean[0, :])
    tau_top = nu * (u_mean[-1, :] - u_mean[-2, :])

    # Drag
    F_bottom = tau_bottom.abs().sum().item()
    F_ice = tau_top[:x_edge].abs().sum().item()
    F_open = tau_top[x_edge:].abs().sum().item()
    C_D = (F_bottom + F_ice + F_open) / (0.5 * u0**2 * (ny-2))

    print(f"  [{label}] C_D={C_D:.4f} (bottom={F_bottom:.4f}, ice={F_ice:.4f}, open={F_open:.4f}) [{elapsed:.1f}s]")

    return {
        "u_mean": u_mean.cpu().numpy(),
        "tau_bottom": tau_bottom.cpu().numpy(),
        "tau_top": tau_top.cpu().numpy(),
        "C_D": C_D,
        "C_D_ice": F_ice / (0.5 * u0**2 * (ny-2)),
        "C_D_open": F_open / (0.5 * u0**2 * (ny-2)),
        "elapsed_s": elapsed,
    }


def main():
    output_root = Path("runs/ice_edge_zone/campaign")
    output_root.mkdir(parents=True, exist_ok=True)
    fig_dir = output_root / "figures"
    fig_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("ICE-EDGE ZONE LBM CAMPAIGN")
    print("=" * 60)

    # 1. Baseline
    print("\n[1/4] Baseline simulation...")
    baseline = run_simulation(
        nx=200, ny=40, u0=0.05, re=100.0,
        ice_edge_frac=0.5, ls_ice=2.0, n_steps=3000,
        output_dir=output_root, label="baseline"
    )

    # 2. Mesh independence
    print("\n[2/4] Mesh independence study...")
    mesh_results = []
    for nx, ny, label in [(100, 20, "coarse"), (200, 40, "medium"), (400, 80, "fine")]:
        result = run_simulation(
            nx=nx, ny=ny, u0=0.05, re=100.0,
            ice_edge_frac=0.5, ls_ice=2.0, n_steps=2000,
            output_dir=output_root, label=f"mesh_{label}"
        )
        mesh_results.append({"label": label, "nx": nx, "ny": ny, "C_D": result["C_D"]})

    # 3. Re scan
    print("\n[3/4] Reynolds number scan...")
    re_results = []
    for re in [50, 100, 200, 400]:
        result = run_simulation(
            nx=150, ny=30, u0=0.05, re=re,
            ice_edge_frac=0.5, ls_ice=2.0, n_steps=2000,
            output_dir=output_root, label=f"Re_{re}"
        )
        re_results.append({"Re": re, "C_D": result["C_D"]})

    # 4. Slip length scan
    print("\n[4/4] Slip length scan...")
    slip_results = []
    for ls in [0.0, 0.5, 1.0, 2.0, 5.0]:
        result = run_simulation(
            nx=150, ny=30, u0=0.05, re=100.0,
            ice_edge_frac=0.5, ls_ice=ls, n_steps=2000,
            output_dir=output_root, label=f"Ls_{ls}"
        )
        slip_results.append({"Ls": ls, "C_D": result["C_D"]})

    # Generate figures
    print("\nGenerating figures...")
    figures = []

    # Fig 1: Velocity field
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(baseline["u_mean"], origin="lower", cmap="viridis", aspect="auto")
    plt.colorbar(im, ax=ax, label="u / U_0")
    x_edge = int(200 * 0.5)
    ax.axvline(x=x_edge, color="red", linestyle="--", linewidth=2, label="Ice edge")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title("Streamwise velocity field (baseline)")
    ax.legend()
    path = fig_dir / "fig1_velocity.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "velocity_field", "path": str(path)})

    # Fig 2: Wall shear
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(baseline["tau_bottom"], "b-", linewidth=1.5)
    axes[0].axvline(x=x_edge, color="red", linestyle="--")
    axes[0].set_ylabel("τ_w (bottom)")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(baseline["tau_top"], "r-", linewidth=1.5)
    axes[1].axvline(x=x_edge, color="red", linestyle="--", label="Ice edge")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("τ_w (top)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    plt.suptitle("Wall shear stress")
    plt.tight_layout()
    path = fig_dir / "fig2_shear.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "wall_shear", "path": str(path)})

    # Fig 3: Mesh independence
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [r["label"] for r in mesh_results]
    cds = [r["C_D"] for r in mesh_results]
    ax.bar(labels, cds, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylabel("C_D")
    ax.set_title("Mesh independence")
    for i, (l, cd) in enumerate(zip(labels, cds)):
        ax.text(i, cd + 0.01, f"{cd:.3f}", ha="center")
    ax.grid(True, alpha=0.3, axis="y")
    path = fig_dir / "fig3_mesh.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "mesh_independence", "path": str(path)})

    # Fig 4: C_D vs Re
    fig, ax = plt.subplots(figsize=(8, 5))
    res = [r["Re"] for r in re_results]
    cds_re = [r["C_D"] for r in re_results]
    ax.loglog(res, cds_re, "o-", linewidth=2, markersize=8)
    ax.set_xlabel("Reynolds number Re")
    ax.set_ylabel("C_D")
    ax.set_title("Drag coefficient vs Reynolds number")
    ax.grid(True, alpha=0.3)
    path = fig_dir / "fig4_re.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "cd_vs_re", "path": str(path)})

    # Fig 5: C_D vs slip length
    fig, ax = plt.subplots(figsize=(8, 5))
    lss = [r["Ls"] for r in slip_results]
    cds_ls = [r["C_D"] for r in slip_results]
    ax.semilogx(lss[1:], cds_ls[1:], "^-", linewidth=2, markersize=8)
    ax.plot([lss[0]], [cds_ls[0]], "^", markersize=8)
    ax.set_xlabel("Slip length Ls")
    ax.set_ylabel("C_D")
    ax.set_title("Drag coefficient vs slip length")
    ax.grid(True, alpha=0.3)
    path = fig_dir / "fig5_slip.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    figures.append({"name": "cd_vs_slip", "path": str(path)})

    # Compute sha256
    h = hashlib.sha256()
    for fig_info in figures:
        with open(fig_info["path"], "rb") as f:
            h.update(f.read())
    sha256 = h.hexdigest()

    # Save report
    report = {
        "title": "Ice-Edge Zone Flow: D2Q9 LBM Simulation",
        "mission_id": "p272w2-N08",
        "baseline": {"C_D": baseline["C_D"], "C_D_ice": baseline["C_D_ice"]},
        "mesh_independence": mesh_results,
        "re_scan": re_results,
        "slip_scan": slip_results,
        "figures": [{"name": f["name"]} for f in figures],
        "sha256": sha256,
        "acceptance_criteria": {
            "C1_mesh": "pass" if max(abs(cds[i]-cds[i-1])/cds[i-1] for i in range(1, len(cds))) < 0.1 else "fail",
            "C2_visualization": len(figures) >= 5,
            "C3_parameter_study": len(re_results) >= 3,
            "C6_sha256": sha256[:16] + "...",
        },
    }

    with open(output_root / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print("CAMPAIGN COMPLETE")
    print(f"{'='*60}")
    print(f"Baseline C_D = {baseline['C_D']:.4f}")
    print(f"Figures: {len(figures)}")
    print(f"Report: {output_root / 'report.json'}")
    print(f"SHA256: {sha256[:16]}...")


if __name__ == "__main__":
    main()
