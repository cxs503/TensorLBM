"""Rebuild the m15432 cylinder-Re100 figures from committed JSON artifacts.

Reads ``artifacts/cylinder_re100_D*/cylinder_re100_unconfined.json`` (emitted by
``tests/validation/cylinder_re100_unconfined.py``) and produces:

  * ``fig_vorticity.png``   — vorticity + speed contour of the final field
  * ``fig_cd_history.png``  — Cd time history vs reference (both resolutions)
  * ``fig_centerline.png``  — centerline ux profile vs free-stream

Usage:
    python -m tests.validation.make_cylinder_re100_figures --artifacts-dir artifacts --out-dir figures
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REF = {"cd": 1.3796, "cl": 0.0005, "st": 0.164}


def _load_all(art_dir: str) -> list[dict]:
    """Return a list of (run_dict, sample_dict) across all emitted artifacts."""
    out = []
    for p in sorted(glob.glob(os.path.join(art_dir, "cylinder_re100_D*", "cylinder_re100_unconfined.json"))):
        with open(p) as fh:
            d = json.load(fh)
        runs = d.get("runs", [])
        samps = d.get("samples", [])
        for r, s in zip(runs, samps):
            out.append((r, s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--out-dir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    data = _load_all(args.artifacts_dir)
    if not data:
        print("no artifacts found under", args.artifacts_dir)
        return

    # ---- vorticity + speed contour (finest run with field data) ----
    field = None
    for r, s in sorted(data, key=lambda rs: -rs[0]["config"].get("diameter", 0)):
        if s.get("wz"):
            field = (r, s)
            break
    if field is not None:
        r, s = field
        name = r["config"]["name"]
        ny, nx = len(s["wz"]), len(s["wz"][0])
        wz = np.array(s["wz"])
        speed = np.sqrt(np.array(s["ux"]) ** 2 + np.array(s["uy"]) ** 2)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        im0 = axes[0].contourf(np.arange(nx), np.arange(ny), wz, levels=24, cmap="RdBu_r")
        axes[0].set_title(f"vorticity wz  ({name})")
        axes[0].set_xlabel("x / lattice"); axes[0].set_ylabel("y / lattice")
        axes[0].set_aspect("auto")
        fig.colorbar(im0, ax=axes[0], label="wz")
        im1 = axes[1].contourf(np.arange(nx), np.arange(ny), speed, levels=24, cmap="viridis")
        axes[1].set_title("speed |u|")
        axes[1].set_xlabel("x / lattice"); axes[1].set_ylabel("y / lattice")
        axes[1].set_aspect("auto")
        fig.colorbar(im1, ax=axes[1], label="|u|")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "fig_vorticity.png"), dpi=140)
        plt.close(fig)
        print("wrote fig_vorticity.png from", name)

    # ---- Cd history vs reference ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r, s in data:
        cd = s.get("cd_tail", [])
        if not cd:
            continue
        nm = r["config"]["name"]
        ax.plot(cd, lw=0.8, label=f"{nm} (mean {np.mean(cd):.4f})")
    ax.axhline(REF["cd"], color="k", ls="--", lw=1.2,
                 label=f"ref Cd = {REF['cd']} (Schäfer & Turek 1996)")
    ax.axhspan(REF["cd"] * 0.95, REF["cd"] * 1.05, color="g", alpha=0.12, label="±5% band")
    ax.set_xlabel("sample index (post-warmup)")
    ax.set_ylabel("Cd")
    ax.set_title("Drag coefficient history, unconfined 2-D cylinder Re=100")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_cd_history.png"), dpi=140)
    plt.close(fig)
    print("wrote fig_cd_history.png")

    # ---- centerline velocity profile ----
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for r, s in data:
        prof = s.get("centerline_ux")
        if not prof:
            continue
        u_in = r["config"]["u_in"]
        ax.plot(np.arange(len(prof)), np.array(prof) / u_in, lw=1.2, label=r["config"]["name"])
    ax.axhline(1.0, color="k", ls=":", lw=1.0, label="free-stream u/u_in = 1")
    ax.set_xlabel("x / lattice")
    ax.set_ylabel("u / u_in (centerline)")
    ax.set_title("Centerline velocity recovery behind the cylinder")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "fig_centerline.png"), dpi=140)
    plt.close(fig)
    print("wrote fig_centerline.png")


if __name__ == "__main__":
    main()
