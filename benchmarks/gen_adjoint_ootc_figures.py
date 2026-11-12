#!/usr/bin/env python3
"""Generate figures for the adjoint OOTC benchmark report."""

from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "adjoint_ootc_bench")
FIGURE_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")


def load_bench(profile: str) -> dict:
    """Load benchmark results for a given profile."""
    path = os.path.join(ARTIFACT_DIR, f"adjoint_ootc_bench_{profile}.json")
    if not os.path.exists(path):
        path = os.path.join(ARTIFACT_DIR, "adjoint_ootc_bench.json")
    with open(path) as f:
        return json.load(f)


def gen_memory_scaling():
    """Figure 1: Memory scaling OOTC vs BPTT across resolutions."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Data from quick + medium benchmarks
    resolutions = [32, 64]
    bptt_mem = [0.015, 0.553]  # GiB
    ootc_k20_mem = [0.002, 0.017]  # GiB
    ootc_k100_mem = [0.008, 0.031]  # GiB

    x = np.arange(len(resolutions))
    width = 0.25

    ax.bar(x - width, bptt_mem, width, label="BPTT (full storage)", color="#e74c3c")
    ax.bar(x, ootc_k20_mem, width, label="OOTC k=T/10", color="#3498db")
    ax.bar(x + width, ootc_k100_mem, width, label="OOTC k=T/20", color="#2ecc71")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}²" for n in resolutions])
    ax.set_ylabel("Peak Memory (GiB)")
    ax.set_title("Memory Scaling: OOTC vs BPTT")
    ax.legend()
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    for i, (b, o1, o2) in enumerate(zip(bptt_mem, ootc_k20_mem, ootc_k100_mem)):
        ax.annotate(f"{b:.3f}", (i - width, b), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{o1:.3f}", (i, o1), ha="center", va="bottom", fontsize=8)
        ax.annotate(f"{o2:.3f}", (i + width, o2), ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "adjoint_ootc_memory_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def gen_convergence_plot():
    """Figure 2: Gradient accuracy vs checkpoint interval k."""
    conv_path = os.path.join(ARTIFACT_DIR, "convergence_study.json")
    with open(conv_path) as f:
        data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    k_vals = [r["k"] for r in data["k_sweep"]]
    rel_errs = [r["rel_err"] for r in data["k_sweep"]]
    mem_vals = [r["peak_mem_gib"] for r in data["k_sweep"]]

    # Left: relative error
    ax1.semilogx(k_vals, [max(e, 1e-16) for e in rel_errs], "o-", color="#2980b9", linewidth=2)
    ax1.axhline(1e-5, color="#e74c3c", linestyle="--", label="Gate: 1e-5")
    ax1.set_xlabel("Checkpoint interval k")
    ax1.set_ylabel("Relative gradient error")
    ax1.set_title("Gradient Accuracy vs Checkpoint Interval")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: memory
    ax2.semilogx(k_vals, mem_vals, "s-", color="#27ae60", linewidth=2)
    ax2.axhline(8.0, color="#e74c3c", linestyle="--", label="Gate: 8 GiB")
    ax2.set_xlabel("Checkpoint interval k")
    ax2.set_ylabel("Peak Memory (GiB)")
    ax2.set_title("Memory vs Checkpoint Interval")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "adjoint_ootc_convergence.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def gen_velocity_field():
    """Figure 3: Velocity field at terminal state (N=64)."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "TensorLBM", "src"))
    from tensorlbm.adjoint.lbm2d import initial_pulse, step, macro

    N, T, tau = 64, 2000, 1.0
    f0 = initial_pulse(N, N, amp=0.05, sigma=float(N) / 8.0)
    f = f0.copy()
    for _ in range(T):
        f = step(f, tau)
    _, ux, uy = macro(f)
    speed = np.sqrt(ux**2 + uy**2)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5))

    im1 = ax1.imshow(ux, cmap="RdBu_r", aspect="equal")
    ax1.set_title("u_x (terminal)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    fig.colorbar(im1, ax=ax1, fraction=0.046)

    im2 = ax2.imshow(uy, cmap="RdBu_r", aspect="equal")
    ax2.set_title("u_y (terminal)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(im2, ax=ax2, fraction=0.046)

    im3 = ax3.imshow(speed, cmap="viridis", aspect="equal")
    ax3.set_title("|u| (terminal)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    fig.colorbar(im3, ax=ax3, fraction=0.046)

    plt.suptitle(f"D2Q9 BGK LBM: Terminal Velocity Field (N={N}, T={T}, τ={tau})")
    plt.tight_layout()
    path = os.path.join(FIGURE_DIR, "adjoint_ootc_velocity_field.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")
    return path


def main():
    os.makedirs(FIGURE_DIR, exist_ok=True)
    gen_memory_scaling()
    gen_convergence_plot()
    gen_velocity_field()
    print("All figures generated.")


if __name__ == "__main__":
    main()
