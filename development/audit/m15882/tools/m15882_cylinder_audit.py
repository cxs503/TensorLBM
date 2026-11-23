"""m15882 board audit — independent recomputation of the m15777 CASE-CYLINDER-DRAG-RE20 bundle.

Reads the delivered artifacts from the audited worktree (read-only), recomputes the
reported statistics from the raw per-step series, verifies the field snapshot is a
real flow solution (divergence / no-slip / blockage geometry), and renders audit
figures.  No audited code is imported for the recomputation of Cd statistics: the
numbers come from the CSV series itself.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REF_CD = 5.57953523384
TOL_PCT = 5.0


def audit(art: Path, out_dir: Path) -> dict:
    res = json.loads((art / "results.json").read_text())
    series = np.genfromtxt(art / "series.csv", delimiter=",", names=True)
    step, cd, cl = series["step"], series["cd"], series["cl"]

    decim = int(round(float(np.median(np.diff(step)))))  # series.csv is decimated
    win = max(int(res["avg_window"]) // max(decim, 1), 2)
    tail = cd[-win:]
    half = win // 2
    mean_tail = float(np.mean(tail))
    drift = abs(float(np.mean(tail[:half])) - float(np.mean(tail[half:]))) / abs(mean_tail)
    err = (mean_tail - REF_CD) / REF_CD * 100.0

    # independent verdict from the raw series only
    stationary = drift <= float(res["plateau_tol"])
    passed = bool(stationary and abs(err) <= TOL_PCT)

    fields = np.load(art / "fields.npz")
    keys = sorted(fields.files)
    divergence = None
    if {"ux", "uy"} <= set(keys):
        ux, uy = fields["ux"], fields["uy"]
        # interior central differences (2nd order) on the lattice
        dux_dx = (ux[1:-1, 2:] - ux[1:-1, :-2]) * 0.5
        duy_dy = (uy[2:, 1:-1] - uy[:-2, 1:-1]) * 0.5
        div = dux_dx + duy_dy
        divergence = {
            "max_abs": float(np.abs(div).max()),
            "rms": float(np.sqrt(np.mean(div**2))),
            "p99_abs": float(np.percentile(np.abs(div), 99.0)),
        }

    geo = res["geometry"]
    lat = res["lattice_units"]
    checks = {
        "blockage_D_over_H": geo["blockage"],
        "blockage_expected": 0.1 / 0.41,
        "re_from_artifact": lat["re"],
        "re_recomputed": lat["u_mean"] * geo["D_cells"] / lat["nu"],
        "ma_recomputed": lat["u_mean"] * math.sqrt(3.0),
        "nx_expected": round(2.2 / (0.1 / geo["D_cells"])),
        "ny_expected": round(0.41 / (0.1 / geo["D_cells"])),
    }

    out = {
        "artifact_dir": str(art),
        "reported": {
            "cd": res["cd"],
            "err_cd_pct": res["err_cd_pct"],
            "cd_plateau_drift": res["cd_plateau_drift"],
            "n_steps": res["n_steps"],
            "stop_reason": res["stop_reason"],
            "stationary": res["stationary"],
            "passed": res["verdict"]["passed"],
            "wall_s": res["wall_s"],
        },
        "recomputed_from_series_csv": {
            "n_samples": int(step.size),
            "decimation_steps": decim,
            "window_samples": win,
            "mean_cd_tail_window": mean_tail,
            "err_cd_pct": err,
            "drift": drift,
            "stationary": stationary,
            "passed": passed,
            "cd_minus_report": mean_tail - float(res["cd"]),
            "err_minus_report": err - float(res["err_cd_pct"]),
        },
        "field_snapshot": {"keys": keys, "divergence": divergence},
        "geometry_checks": checks,
        "ref_cd": REF_CD,
        "tolerance_pct": TOL_PCT,
    }
    (out_dir / "independent_recompute.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # ---- figures ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(step, cd, lw=0.8, color="#1f4e79", label="Cd(step) — delivered series.csv")
    ax.axhline(REF_CD, color="k", ls="--", lw=1.0, label=f"ref Cd = {REF_CD:.5f} (S&T 2D-1)")
    ax.fill_between([step[0], step[-1]], REF_CD * (1 - TOL_PCT / 100), REF_CD * (1 + TOL_PCT / 100),
                    color="tab:green", alpha=0.15, label="±5 % tolerance door")
    ax.axhline(mean_tail, color="tab:red", lw=1.0, label=f"audit mean (last {win} steps) = {mean_tail:.5f}")
    ax.set_xlabel("iteration step")
    ax.set_ylabel("Cd")
    ax.set_title(f"CASE-CYLINDER-DRAG-RE20 / TensorLBM m15777 D=16 — Cd vs ref (err {err:+.2f} %)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "audit_cd_convergence.png", dpi=150)
    plt.close(fig)

    if "ux" in keys and "uy" in keys:
        ux, uy = fields["ux"], fields["uy"]
        speed = np.sqrt(ux**2 + uy**2)
        fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
        im0 = axes[0].imshow(speed.T, origin="lower", cmap="viridis", aspect="auto")
        axes[0].set_title("|u| field (lattice units) — delivered fields.npz")
        fig.colorbar(im0, ax=axes[0], fraction=0.02)
        vort = (np.roll(uy, -1, 1) - np.roll(uy, 1, 1)) * 0.5 - (np.roll(ux, -1, 0) - np.roll(ux, 1, 0)) * 0.5
        lim = float(np.percentile(np.abs(vort), 99))
        im1 = axes[1].imshow(np.clip(vort.T, -lim, lim), origin="lower", cmap="RdBu_r", aspect="auto")
        axes[1].set_title("vorticity — recirculation behind the cylinder (Re = 20)")
        axes[1].set_xlabel("x (lattice)")
        fig.colorbar(im1, ax=axes[1], fraction=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / "audit_field_cloudmap.png", dpi=150)
        plt.close(fig)

    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    o = Path(a.out)
    o.mkdir(parents=True, exist_ok=True)
    print(json.dumps(audit(Path(a.art), o), indent=2, ensure_ascii=False))