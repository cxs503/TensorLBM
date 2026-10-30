#!/usr/bin/env python3
"""m318 diagnosis artifacts for benchmarks/verified/suboff_re1000.

Builds the figures for the diagnosis report from ALREADY-PRODUCED run output
(no simulation here):

  fig1  velocity-magnitude cloud of the converged field (mid-plane through the
        hull axis + mid-hull cross-section)            <- <run>/final_field.pt
  fig2  Cd_tot convergence history vs the Blasius reference band
                                                        <- <run>/forces_history.json
  fig3  geometry gate (voxel vs analytic SUBOFF form) vs cells-per-length L
                                                        <- artifacts/*geometry_table.json

Usage:
  python scripts/suboff_re1000_m318_report.py --run runs/m318/suboff_L48_mix50_s12000 \
      --geometry artifacts/suboff_re1000_m318_geometry_table.json \
      --out artifacts/figs_m318 [--report-json /tmp/report_figs.json]
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))
from tensorlbm.d3q19 import macroscopic3d  # noqa: E402

CF_REF = 1.328 / math.sqrt(1000.0)
GATE = 0.03


def fig_field(run_dir: Path, out_png: Path) -> dict:
    f = torch.load(run_dir / "final_field.pt", map_location="cpu", weights_only=True)
    nz, ny, nx = f.shape[1:]
    rho, ux, uy, uz = macroscopic3d(f.to(torch.float32))
    umag = torch.sqrt(ux**2 + uy**2 + uz**2)
    # hull centre line as placed by the engine: (nx*0.25, ny*0.5, nz*0.5)
    iz, iy = int(nz * 0.5), int(ny * 0.5)
    i_mid = int(nx * 0.25)

    plane_xz = umag[iz, :, :].numpy()  # (ny, nx)
    sect = umag[:, :, i_mid].numpy()  # (nz, ny)
    rho_plane = rho[iz, :, :].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    ax = axes[0]
    im = ax.imshow(plane_xz, origin="lower", cmap="viridis", aspect="auto")
    ax.axvspan(nx * 0.25 - nx / 12, nx * 0.25 + nx / 12, color="w", alpha=0.0)
    ax.set_title("(a) |u| mid-plane y=y_axis  [x: streamwise]")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")
    fig.colorbar(im, ax=ax, fraction=0.035, label="|u| (lu)")

    ax = axes[1]
    im = ax.imshow(sect, origin="lower", cmap="viridis", aspect="equal")
    ax.set_title(f"(b) |u| cross-section x = {i_mid} (mid hull)")
    ax.set_xlabel("y (cells)")
    ax.set_ylabel("z (cells)")
    fig.colorbar(im, ax=ax, fraction=0.035, label="|u| (lu)")

    ax = axes[2]
    xs = np.arange(nx)
    ax.plot(xs, plane_xz.mean(axis=0), lw=1.2, label="⟨|u|⟩ over z")
    ax.plot(xs, rho_plane.mean(axis=0) - 1.0, lw=1.0, ls="--", label="⟨rho-1⟩")
    ax.axhline(0.05, color="k", ls=":", lw=1, label="u_in = 0.05")
    ax.set_title("(c) streamwise profiles")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("lu")
    ax.legend(fontsize=8)
    fig.suptitle(
        f"B6 SUBOFF Re=1000 — converged field, L={nx // 6} cells/len, grid {nx}x{ny}x{nz}",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return {
        "grid": [nx, ny, nz],
        "umag_max": float(umag.max()),
        "umag_mean": float(umag.mean()),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
    }


def fig_convergence(run_dir: Path, out_png: Path) -> dict:
    h = json.loads((run_dir / "forces_history.json").read_text())
    s = h["samples"]
    rescale = float(h["rescale_frontal_to_wetted"])
    step = np.array([e.get("step", i * 10) for i, e in enumerate(s)], dtype=float)
    tot = np.array([e["cd_total"] for e in s]) * rescale
    cp = np.array([e["cd_pressure"] for e in s]) * rescale
    cf = np.array([e["cd_friction"] for e in s]) * rescale

    n_win = min(50, max(5, len(s) // 8))
    kern = np.ones(n_win) / n_win
    tot_s = np.convolve(tot, kern, mode="valid")
    step_s = step[n_win - 1 :]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4))
    ax = axes[0]
    ax.plot(step, tot, lw=0.5, color="0.75", label="raw samples")
    ax.plot(step_s, tot_s, lw=1.8, color="C3", label=f"moving mean ({n_win} samples)")
    ax.axhline(CF_REF, color="k", lw=1.3, label=f"Blasius ref {CF_REF:.5f}")
    ax.axhspan(CF_REF * (1 - GATE), CF_REF * (1 + GATE), color="g", alpha=0.15, label="±3% band")
    ax.set_xlabel("step")
    ax.set_ylabel("Cd_tot (wetted area πDL)")
    ax.set_title("(a) Cd_tot convergence")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(step, cp, lw=1.2, label="Cd_pressure")
    ax.plot(step, cf, lw=1.2, label="Cd_friction (mix50)")
    ax.plot(step, tot, lw=1.0, color="0.6", ls=":", label="Cd_total")
    ax.set_xlabel("step")
    ax.set_ylabel("Cd")
    ax.set_title("(b) pressure / friction split")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    tail = tot[-min(len(tot), 1000) :]
    err = (tail.mean() - CF_REF) / CF_REF * 100.0
    w = max(1, int(len(tot) * 0.2))
    drift = (tot[-w:].mean() - tot[-2 * w : -w].mean()) / tot[-2 * w : -w].mean() * 100.0
    fig.suptitle(
        f"B6 SUBOFF Re=1000 — L={h['L_cells']}, {len(s)} force samples "
        f"({h.get('force_sample_interval')} steps apart), tail mean err {err:+.2f}%",
        y=1.02,
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return {
        "n_samples": len(s),
        "tail_mean_cd_tot": float(tail.mean()),
        "tail_err_pct": float(err),
        "drift_pct_last20_vs_prev20": float(drift),
        "cd_first": float(tot[0]),
        "cd_last": float(tot[-1]),
    }


def fig_geometry(geom_json: Path, out_png: Path) -> dict:
    rows = json.loads(geom_json.read_text())
    L = [r["L_cells"] for r in rows]
    dR = [r["r_max_err_pct"] for r in rows]
    dCp = [r["Cp_err_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(L, dR, "o-", label="R_max voxel vs analytic")
    ax.plot(L, dCp, "s-", label="Cp (Cb) voxel vs analytic")
    ax.axhspan(-20, 15, color="g", alpha=0.08)
    ax.set_xlabel("L (cells per hull length)")
    ax.set_ylabel("error (%)")
    ax.set_title("Geometry gate: voxel mask vs analytic SUBOFF form\n(gate: |ΔR|<20%, |ΔCp|<15%; drag trusted only if pass)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    for x, y in zip(L, dR):
        ax.annotate(f"{y:+.1f}%", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=125, bbox_inches="tight")
    plt.close(fig)
    return {"resolutions": L, "dR_pct": dR, "dCp_pct": dCp}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--geometry", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report-json", type=Path, default=None)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = {}
    p1 = args.out / "fig1_field.png"
    meta["fig1"] = fig_field(args.run, p1)
    p2 = args.out / "fig2_convergence.png"
    meta["fig2"] = fig_convergence(args.run, p2)
    p3 = args.out / "fig3_geometry.png"
    meta["fig3"] = fig_geometry(args.geometry, p3)

    figs = []
    for path, name, cap in (
        (p1, "fig1_field", "Converged flow field: |u| mid-plane, mid-hull cross-section, streamwise profiles"),
        (p2, "fig2_convergence", "Cd_tot convergence vs Blasius ±3% band; pressure/friction split"),
        (p3, "fig3_geometry", "Geometry gate: voxel mask vs analytic SUBOFF form across L"),
    ):
        figs.append(
            {
                "name": name,
                "caption": cap,
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            }
        )
    out = {"figures": figs, "meta": meta}
    print(json.dumps(meta, indent=2))
    if args.report_json:
        args.report_json.write_text(json.dumps(out))
        print(f"figures payload -> {args.report_json}")


if __name__ == "__main__":
    main()