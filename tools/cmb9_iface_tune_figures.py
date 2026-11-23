"""DE-X-CMB9-IFACETUNE: report figures (depth-time vs experiment, mid-plane
fill contour at the registered peak time, cavity geometry series).  All data
read from runs/cmb9_*/result.json, runs/cmb9_*_p/planes.npz and
runs/cmb9_*/cav_series.json; nothing typed by hand."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_PEAK_D = 1.857434
EXP_PEAK_PC_MS = 10.39446
DEPTH_GATE = 1.8017
TIME_GATE = 11.4334

ARMS = [
    ("cmb9_a_z27", "A: rr_iface+SMAG.075+HYDRO27 (c8)", "tab:blue"),
    ("cmb9_b_cs0_z245", "B: rr_iface+HYDRO24.5 Cs=0 (c8)", "tab:orange"),
    ("cmb9_c12", "C: rr_iface+SMAG.075+HYDRO24.5 (c12)", "tab:green"),
    ("cmb9_d_cs05_z245", "D: rr_iface+SMAG.05+HYDRO24.5 (c8)", "tab:purple"),
    ("cmb9_base8", "base: rr_iface (c8)", "tab:red"),
    ("cmb9_c12_ctrl", "ctrl: BGK (c12)", "tab:gray"),
]


def load(tree: Path, arm: str):
    p = tree / "runs" / arm / "result.json"
    return json.loads(p.read_text()) if p.exists() else None


def pc_series(res: dict):
    h = res["history"]
    contact = res.get("contact_step")
    dt = h[1]["time_ms"] - h[0]["time_ms"]
    t = [r["time_ms"] - contact * dt for r in h]
    d = [r["depth_over_D"] for r in h]
    return t, d


def fig_depth_time(tree: Path, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    exp = list(csv.DictReader(open(tree / "benchmarks/water_entry/single_drop/fig5_experiment.csv")))
    ax.plot(
        [float(r["time_ms"]) for r in exp],
        [float(r["depth_over_D"]) for r in exp],
        "k.",
        ms=6,
        label="experiment (Santini 2021)",
        zorder=5,
    )
    for arm, label, color in ARMS:
        res = load(tree, arm)
        if res is None:
            continue
        t, d = pc_series(res)
        ax.plot(t, d, color=color, lw=1.4, label=f"{label} ({res['stop_reason'][:6]})")
    ax.axhline(DEPTH_GATE, color="k", ls=":", lw=1)
    ax.axvline(TIME_GATE, color="k", ls=":", lw=1)
    ax.text(0.2, DEPTH_GATE + 0.02, f"depth gate {DEPTH_GATE}D", fontsize=8)
    ax.text(TIME_GATE + 0.1, 0.05, f"time gate {TIME_GATE}ms", fontsize=8, rotation=90)
    ax.set_xlabel("post-contact time (ms)")
    ax.set_ylabel("open depth (D)")
    ax.set_title("CMB9: rr_iface directed tuning (z / time-tax / resolution / Cs)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_contour(tree: Path, out: Path, arm: str = "cmb9_d") -> None:
    p = tree / "runs" / ("cmb9_d_p" if arm == "cmb9_d_cs05_z245" else f"{arm}_p") / "planes.npz"
    if not p.exists():
        print(f"skip contour: {p} missing")
        return
    z = np.load(p)
    # pick the registered time closest to the experimental peak
    times = z["time_ms"]
    idx = int(np.argmin(np.abs(times - EXP_PEAK_PC_MS)))
    fill = z["fill"][idx]
    n = fill.shape[0]
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=140)
    cs = ax.contourf(fill, levels=20, cmap="viridis")
    ax.contour(fill, levels=[0.5], colors="w", linewidths=1.2)
    fig.colorbar(cs, ax=ax, label="fill (0=gas, 1=liquid)")
    ax.set_xlabel("x (cells)")
    ax.set_ylabel("z (cells)")
    ax.set_title(f"{arm}: mid-plane fill @ t={times[idx]:.3f} ms pc (exp peak {EXP_PEAK_PC_MS} ms)")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_cavity(tree: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=140)
    for arm, label, color in ARMS:
        p = tree / "runs" / arm / "cav_series.json"
        if not p.exists():
            continue
        rows = json.loads(p.read_text())["rows"]
        t = [r["time_ms"] for r in rows]
        axes[0].plot(t, [r["cav_depth_cells"] for r in rows], color=color, lw=1.2, label=label)
        axes[1].plot(t, [r["cav_area_cells"] for r in rows], color=color, lw=1.2, label=label)
    axes[0].set_xlabel("time (ms)")
    axes[0].set_ylabel("cavity depth (cells)")
    axes[1].set_xlabel("time (ms)")
    axes[1].set_ylabel("cavity area (cells)")
    axes[0].set_title("mid-plane cavity depth")
    axes[1].set_title("mid-plane cavity area")
    axes[0].legend(fontsize=6)
    axes[1].legend(fontsize=6)
    axes[0].grid(alpha=0.3)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


if __name__ == "__main__":
    import sys

    tree = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    outdir = tree / "runs" / "cmb9"
    outdir.mkdir(parents=True, exist_ok=True)
    fig_depth_time(tree, outdir / "cmb9_depth_time.png")
    fig_contour(tree, outdir / "cmb9_contour.png")
    fig_cavity(tree, outdir / "cmb9_cavity.png")
    print("figures written to", outdir)
