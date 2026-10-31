#!/usr/bin/env python
"""Post-process existing dam-break run and produce validation report vs Martin & Moyce (1952).

Reads front_position.csv from an existing CG-LBM dam-break run, compares with
Martin & Moyce experimental data, generates comparison figures, and writes
validation_report.json.

Usage:
    PYTHONPATH=src python validation/dam_break/postprocess.py \
        --run-dir validation/dam_break/dam_break/mm_val_cg_1600 \
        --out-dir validation/dam_break
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Martin & Moyce (1952) reference data
# ---------------------------------------------------------------------------
_MM_FRONT: list[tuple[float, float]] = [
    (0.0, 1.0), (0.5, 1.1), (1.0, 1.4), (1.5, 1.8),
    (2.0, 2.2), (2.5, 2.7), (3.0, 3.1), (3.5, 3.5),
    (4.0, 3.8), (4.5, 4.1), (5.0, 4.3), (5.5, 4.5), (6.0, 4.7),
]
_EVAL_POINTS: list[float] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def _mm_to_xstar(z: float) -> float:
    return (z + 1.0) / 2.0


def _interp_mm(t: float) -> float | None:
    if t < _MM_FRONT[0][0] or t > _MM_FRONT[-1][0]:
        return None
    for (t0, z0), (t1, z1) in zip(_MM_FRONT[:-1], _MM_FRONT[1:]):
        if t0 <= t <= t1:
            frac = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
            return _mm_to_xstar(z0 + frac * (z1 - z0))
    return _mm_to_xstar(_MM_FRONT[-1][1])


def read_front(csv_path: Path) -> list[tuple[int, float, float]]:
    rows = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["step"]), float(r["t_star"]), float(r["X_star"])))
    return rows


def main(run_dir: str, out_dir: str) -> None:
    rd = Path(run_dir); od = Path(out_dir); od.mkdir(parents=True, exist_ok=True)

    front = read_front(rd / "front_position.csv")
    if not front:
        print("No front data found"); return

    # Read config
    meta = json.loads((rd / "run_metadata.json").read_text())
    cfg = meta["config"]

    # 6-point comparison
    comp = []
    for te in _EVAL_POINTS:
        xr = _interp_mm(te)
        if xr is None: continue
        xs = None
        for i in range(len(front)-1):
            if front[i][1] <= te <= front[i+1][1]:
                f = (te - front[i][1]) / (front[i+1][1] - front[i][1])
                xs = front[i][2] + f*(front[i+1][2] - front[i][2])
                break
        if xs is None: xs = front[-1][2]
        ep = abs(xs - xr) / xr * 100
        comp.append({"T": te, "X_star_sim": round(xs,6), "X_star_ref": round(xr,6), "err_pct": round(ep,4)})

    errs = [r["err_pct"] for r in comp]
    mean_err = sum(errs)/len(errs) if errs else float("nan")
    passed = mean_err <= 6.0

    # Write comparison CSV
    with (od / "mm_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["T","X_star_sim","X_star_ref","err_pct"])
        w.writeheader(); w.writerows(comp)

    # Front comparison figure
    fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    t_mm = [t for t,_ in _MM_FRONT]; z_mm = [z for _,z in _MM_FRONT]
    x_mm = [_mm_to_xstar(z) for z in z_mm]
    ax.plot(t_mm, x_mm, "ko", ms=7, mfc="white", mew=1.8,
            label="Martin & Moyce (1952) experiment", zorder=5)
    t_sim = [s[1] for s in front]; x_sim = [s[2] for s in front]
    ax.plot(t_sim, x_sim, "b-", lw=2, label="CG-LBM simulation", zorder=3)
    for r in comp:
        ax.plot(r["T"], r["X_star_sim"], "rs", ms=8, zorder=6)
        ax.plot(r["T"], r["X_star_ref"], "g^", ms=8, zorder=6)
    ax.set_xlabel("Dimensionless time  $t^* = t\\,\\sqrt{2g/L}$", fontsize=12)
    ax.set_ylabel("Dimensionless front position  $X^* = x_f / L$", fontsize=12)
    ax.set_title("Dam-break surge front: CG-LBM vs Martin & Moyce (1952)", fontsize=13)
    ax.legend(fontsize=11, loc="upper left"); ax.grid(alpha=0.3)
    p = od / "front_comparison.png"
    fig.savefig(p, dpi=150); plt.close(fig)
    print(f"Saved: {p}")

    # Phase sequence figure from existing snapshots
    snaps = sorted(rd.glob("flow_step_*.png"))
    if snaps:
        n = 6; idxs = [int(i*(len(snaps)-1)/(n-1)) for i in range(n)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        axes = axes.flatten()
        for i, idx in enumerate(idxs):
            import matplotlib.image as mpimg
            img = mpimg.imread(snaps[idx])
            axes[i].imshow(img)
            step = int(snaps[idx].stem.split("_")[-1])
            ts = front[i][1] if i < len(front) else step * math.sqrt(2*cfg["g"]/cfg["dam_width"])
            axes[i].set_title(f"t* = {ts:.2f}  (step {step})", fontsize=10)
            axes[i].set_xticks([]); axes[i].set_yticks([])
        fig.suptitle("Dam-break phase-fraction evolution (Color-Gradient LBM)", fontsize=13, y=1.01)
        p2 = od / "phase_sequence.png"
        fig.savefig(p2, dpi=150); plt.close(fig)
        print(f"Saved: {p2}")

    # Validation report
    report = {
        "benchmark": "dam_break_martin_moyce_1952",
        "config": cfg,
        "reference": "Martin & Moyce (1952) Phil. Trans. R. Soc. A 244, 312",
        "criterion": "6-point mean |X*_sim - X*_ref| / X*_ref ≤ 6%",
        "n_points": len(comp),
        "comparison_table": comp,
        "mean_error_pct": round(mean_err, 4),
        "pass": bool(passed),
        "figures": ["front_comparison.png", "phase_sequence.png"],
        "run_dir": str(rd),
    }
    (od / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")

    print(f"\n{'='*60}")
    print(f"Dam-break validation vs Martin & Moyce (1952)")
    print(f"{'='*60}")
    print(f"{'T':>6} {'X*_sim':>10} {'X*_ref':>10} {'err%':>8}")
    print(f"{'-'*36}")
    for r in comp:
        print(f"{r['T']:>6.1f} {r['X_star_sim']:>10.4f} {r['X_star_ref']:>10.4f} {r['err_pct']:>7.2f}%")
    print(f"{'-'*36}")
    print(f"Mean error: {mean_err:.2f}%")
    print(f"Criterion (≤6%): {'PASS' if passed else 'FAIL'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    pa = argparse.ArgumentParser()
    pa.add_argument("--run-dir", default="validation/dam_break/dam_break/mm_val_cg_1600")
    pa.add_argument("--out-dir", default="validation/dam_break")
    args = pa.parse_args()
    main(args.run_dir, args.out_dir)
