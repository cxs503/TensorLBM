"""DE-X-CMB8-IFACECOMBO: report figures (depth-time vs experiment, mid-plane
fill contour at the registered peak time, cavity geometry series).  All data
read from runs/cmb8_*/result.json, runs/cmb8_a_cs075_z22_p/planes.npz and
runs/cmb8_*/cav_series.json; nothing typed by hand."""
from __future__ import annotations

import base64
import io
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
    ("cmb8_a_cs075_z22", "A: rr_iface+SMAG.075+HYDRO22", "tab:blue"),
    ("cmb8_b_cs075_z245", "B: rr_iface+SMAG.075+HYDRO24.5", "tab:orange"),
    ("cmb8_c_cs01_z22", "C: rr_iface+SMAG.1+HYDRO22", "tab:green"),
    ("cmb8_base", "base: rr_iface", "tab:red"),
    ("cmb8_ctrl", "ctrl: BGK", "tab:gray"),
]


def load(tree: Path, arm: str) -> dict | None:
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
    # experiment reference
    exp = list(
        __import__("csv").DictReader(
            open(tree / "benchmarks/water_entry/single_drop/fig5_experiment.csv")
        )
    )
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
    ax.text(
        0.2, DEPTH_GATE + 0.02, f"depth gate {DEPTH_GATE}D", fontsize=8
    )
    ax.text(
        TIME_GATE + 0.1, 0.05, f"time gate {TIME_GATE}ms", fontsize=8, rotation=90
    )
    ax.set_xlabel("post-contact time (ms)")
    ax.set_ylabel("open depth (D)")
    ax.set_title("CMB8: rr_iface x SMAG x HYDRO depth-time vs experiment")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_contour(tree: Path, out: Path) -> None:
    p = tree / "runs" / "cmb8_a_cs075_z22_p" / "planes.npz"
    if not p.exists():
        return
    z = np.load(p)
    fill = z["fill"]  # (n_planes, nz-2, nx)
    t_ms = z["time_ms"]
    # pick the plane closest to the experimental peak pc time
    i = int(np.argmin(np.abs(t_ms - EXP_PEAK_PC_MS)))
    f = fill[i]
    nz, nx = f.shape
    level = 24.0  # pool_depth 3 * cells 8
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=140)
    X, Z = np.meshgrid(np.arange(nx), np.arange(nz))
    cs = ax.contourf(
        X, Z - level, f, levels=20, cmap="Blues", extend="both"
    )
    ax.contour(X, Z - level, f, levels=[0.5], colors="k", lw=1.2)
    fig.colorbar(cs, ax=ax, label="fill (0=gas, 1=liquid)")
    ax.set_xlabel("x (cells, mid-plane)")
    ax.set_ylabel("z - z_surface (cells)")
    ax.set_title(
        f"Arm A mid-plane fill @ t={t_ms[i]:.2f} ms pc "
        f"(exp peak {EXP_PEAK_PC_MS} ms)"
    )
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig_cavity(tree: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), dpi=140)
    for ax, key, ylabel in (
        (axes[0], "cav_depth_cells", "cavity depth (cells)"),
        (axes[1], "cav_area_cells", "cavity area (cells)"),
    ):
        for arm, label, color in ARMS[:4]:
            p = tree / "runs" / arm / "cav_series.json"
            if not p.exists():
                continue
            rows = json.loads(p.read_text())["rows"]
            t = [r["time_ms"] for r in rows]
            v = [r[key] for r in rows]
            ax.plot(t, v, color=color, lw=1.3, label=label.split(":")[0])
        ax.set_xlabel("time (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    axes[0].set_title("mid-plane cavity depth")
    axes[1].set_title("mid-plane cavity area")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    tree = Path(__file__).resolve().parents[1]
    outdir = tree / "runs" / "cmb8"
    outdir.mkdir(parents=True, exist_ok=True)
    fig_depth_time(tree, outdir / "cmb8_depth_time.png")
    fig_contour(tree, outdir / "cmb8_contour.png")
    fig_cavity(tree, outdir / "cmb8_cavity.png")
    print("figures written to", outdir)
    for f in sorted(outdir.glob("*.png")):
        b64 = base64.b64encode(f.read_bytes()).decode()
        (outdir / f.name.replace(".png", ".b64")).write_text(b64)
        print(f.name, f.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
