#!/usr/bin/env python3
"""Run the XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED campaign.

Usage:  PYTHONPATH=src python scripts/xval_mpi_ghost_fused.py [--out DIR]

Runs the predeclared matrix (see
``xval/XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED/predeclared.json``), writes
``results.json`` / ``table.csv``, and renders the report figures.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

from tensorlbm.xval_mpi_ghost_fused.campaign import ARMS, run_config, write_artifacts
from tensorlbm.xval_mpi_ghost_fused.case import TGV

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "xval", "XVAL-LBM-MPI-GHOST-EXCHANGE-KERNEL-FUSED")

MATRIX = [
    {"tag": "n64_r2", "n": 64, "ranks": 2, "steps": 400, "dtype": "float32", "u0": 0.04, "tau": 0.65,
     "checkpoints": [100, 200, 300, 400]},
    {"tag": "n64_r4", "n": 64, "ranks": 4, "steps": 400, "dtype": "float32", "u0": 0.04, "tau": 0.65,
     "checkpoints": [400]},
    {"tag": "n32_r2", "n": 32, "ranks": 2, "steps": 200, "dtype": "float32", "u0": 0.04, "tau": 0.65,
     "checkpoints": [200]},
    {"tag": "n32_r2_f64", "n": 32, "ranks": 2, "steps": 100, "dtype": "float64", "u0": 0.04, "tau": 0.65,
     "checkpoints": [100]},
]


def make_figures(results, out_dir: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    paths = []
    by_tag = {r["tag"]: r for r in results}

    # 1 -- TGV decay: numeric vs analytic mode
    result = by_tag["n64_r2"]
    case = TGV(n=result["cfg"]["n"], u0=result["cfg"]["u0"], tau=result["cfg"]["tau"])
    steps = sorted(int(k) for k in result["serial"]["series"])
    analytic = [case.decay(t) * case.u0 * math.sqrt(3.0 / 8.0) for t in steps]
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(steps, [result["serial"]["series"][str(t)] for t in steps], "o-", label="monolithic")
    ax.plot(steps, [result["arms"]["selective_persistent"]["series"][str(t)] for t in steps],
            "x--", label="decomposed, selective exchange")
    ax.plot(steps, analytic, "k:", label=r"analytic $U_0\sqrt{3/8}\,e^{-2\nu k^2 t}$")
    ax.set_xlabel("lattice step $t$")
    ax.set_ylabel(r"$\|u\|_2 / n^{3/2}$")
    ax.set_title("TGV decay, n=64, 2 ranks (float32)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig1_tgv_decay.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 2 -- physics error per arm against the owner band
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [name for name, _, _ in ARMS]
    for offset, tag in enumerate(["n64_r2", "n64_r4", "n32_r2", "n32_r2_f64"]):
        res = by_tag[tag]
        values = [100.0 * res["arms"][name]["tgv_rel_l2"] for name in labels]
        positions = [i + 0.18 * (offset - 1.5) for i in range(len(labels))]
        ax.bar(positions, values, width=0.17, label=tag)
    for level, style in ((15.0, "r-"), (5.0, "orange"), (3.0, "g--")):
        ax.axhline(level, color=style[0], linestyle=style[1:], linewidth=1.2,
                   label=f"{level:g}% ({'owner band' if level == 15.0 else 'target'})")
    ax.set_yscale("log")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("TGV velocity rel-L2 error [%]")
    ax.set_title("physics error vs analytic mode (all arms, all runs)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig2_error_vs_analytic.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 3 -- transport cost: bytes and exchange time
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, key, title, scale in (
        (axes[0], "bytes_per_step", "exchanged bytes per step (2 faces)", 1.0 / 1024.0),
        (axes[1], "exchange_ms", "exchange wall time [ms/step]", 1.0),
    ):
        for offset, tag in enumerate(["n64_r2", "n64_r4"]):
            res = by_tag[tag]
            values = [res["arms"][name][key] * scale for name in labels]
            positions = [i + 0.19 * (offset - 0.5) for i in range(len(labels))]
            axis.bar(positions, values, width=0.18, label=f"{tag}")
        axis.set_xticks(range(len(labels)))
        axis.set_xticklabels(labels, rotation=15)
        axis.set_title(title)
        axis.set_yscale("log")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig3_transport_cost.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # 4 -- field slice and the exact-zero difference panel
    fields = torch.load(os.path.join(out_dir, "fields_n64_r2.pt"), weights_only=False)
    from tensorlbm.xval_mpi_ghost_fused.case import macroscopics

    dtype = torch.float32
    _, ux_serial, _, _ = macroscopics(fields["serial_field"], dtype)
    _, ux_sel, _, _ = macroscopics(fields["field_selective_persistent"], dtype)
    diff = (ux_sel - ux_serial).abs()
    mid = ux_serial.shape[0] // 2
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    im = axes[0].imshow(ux_serial[mid].numpy(), origin="lower", cmap="RdBu_r")
    axes[0].set_title(r"$u_x$ at mid-$z$, monolithic (t=400)")
    fig.colorbar(im, ax=axes[0], shrink=0.85)
    im2 = axes[1].imshow(diff[mid].numpy(), origin="lower", cmap="magma", vmin=0.0, vmax=1e-8)
    axes[1].set_title(f"|decomposed - monolithic|, max={float(diff.max()):.1e}")
    fig.colorbar(im2, ax=axes[1], shrink=0.85)
    fig.tight_layout()
    path = os.path.join(fig_dir, "fig4_field_identity.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    results = []
    for cfg in MATRIX:
        if args.only and cfg["tag"] not in args.only:
            continue
        print(f"[campaign] {cfg['tag']}: n={cfg['n']} ranks={cfg['ranks']} steps={cfg['steps']} {cfg['dtype']}",
              flush=True)
        results.append(run_config(cfg, args.out))
    write_artifacts(results, args.out)
    figures = make_figures(results, args.out)
    with open(os.path.join(args.out, "figures.json"), "w") as handle:
        json.dump(figures, handle, indent=2)

    ok = True
    for result in results:
        for name, _, _ in ARMS:
            arm = result["arms"][name]
            identity = arm["identity_max_abs_vs_serial"]
            cross = arm["identity_max_abs_vs_selective"]
            good = identity == 0.0 and cross == 0.0
            ok &= good
            print(
                f"[{result['tag']:12s}] {name:20s} identity_vs_serial={identity:.1e} "
                f"cross={cross:.1e} tgv_err={100 * arm['tgv_rel_l2']:.3f}% "
                f"bytes/step={arm['bytes_per_step']} ex_ms={arm['exchange_ms']:.3f} "
                f"step_ms={arm['step_ms']:.2f} {'OK' if good else 'MISMATCH'}"
            )
        print(f"[{result['tag']:12s}] serial tgv_err={100 * result['serial']['tgv_rel_l2']:.3f}%")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())