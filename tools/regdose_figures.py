#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DE-X-COL2D-REGDOSE: dose-response figure + mid-plane field maps.

Panel (a): measured peak penetration depth vs REG dose (log2 axis) with the
            cited ctrl (BGK) and 1x (COL2 rr_base) anchors, the experimental
            1.857434 D reference and the pre-registered 2x prediction band.
Panel (b): penetration depth vs post-contact time for every arm that carried
            the mid-plane dump (plus the cited 1x peak marker).

Second figure: mid-plane fill-fraction cloud maps (from the SDAA planes dump)
at the sampled time closest to each arm's peak.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXP_PEAK_D = 1.857434
EXP_PEAK_MS_PC = 10.39446
TAU_1X = 0.5003877

DOSE_ARMS = [("reg_d050", 0.5), ("reg_d200", 2.0), ("reg_d400", 4.0)]
PLANE_ARMS = [("ctrl_pl", "ctrl BGK (0x, cited anchor)"),
              ("reg_d200_pl", "REG 2x"),
              ("reg_d400_pl", "REG 4x"),
              ("reg_d050_pl", "REG 0.5x")]


def load(path: Path):
    return json.loads(path.read_text())


def metrics(res: dict) -> dict:
    hist = res.get("history") or []
    with_depth = [r for r in hist if r.get("depth_over_D") is not None]
    peak = max(with_depth, key=lambda r: float(r["depth_over_D"])) if with_depth else hist[-1]
    contact = res.get("contact_step")
    dt_ms = 1000.0 * res["scales"]["dt_s"]
    return {
        "stop": res.get("stop_reason"),
        "peak_D": float(peak.get("depth_over_D") or 0.0),
        "peak_abs_ms": float(peak.get("time_ms")),
        "peak_pc_ms": (float(peak.get("time_ms")) - contact * dt_ms) if contact is not None else None,
        "contact": contact,
        "dt_ms": dt_ms,
        "hist": hist,
    }


def response_figure(runs: Path, cited: dict, out: Path) -> None:
    one_x = next(a for a in cited["arms"] if a["arm"] == "rr_base")
    ctrl = next(a for a in cited["arms"] if a["arm"] == "ctrl_bgk")

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    # ---- panel (a): dose-response -----------------------------------------
    ax.axhline(EXP_PEAK_D, color="k", ls="--", lw=1.2, label=f"experiment {EXP_PEAK_D} D")
    ax.axhspan(1.60, 1.75, color="tab:green", alpha=0.15,
               label="pre-registered 2x band [1.60,1.75] D")
    ax.plot([1.0], [one_x["peak"]["depth_over_D"]], "o", color="tab:red", ms=9,
            label="1x REG (cited, COL2 rr_base)")
    ax.plot([0.0], [ctrl["peak"]["depth_over_D"]], "s", color="tab:blue", ms=8,
            label="ctrl BGK (cited)")
    for name, dose in DOSE_ARMS:
        p = runs / name / "result.json"
        if not p.exists():
            continue
        m = metrics(load(p))
        if m["stop"] == "window_complete":
            ax.plot([dose], [m["peak_D"]], "o", color="tab:purple", ms=9, zorder=3)
            ax.annotate(f"{m['peak_D']:.4f} D", (dose, m["peak_D"]),
                        textcoords="offset points", xytext=(6, -12), fontsize=8)
        else:
            ax.plot([dose], [m["peak_D"]], "x", color="tab:red", ms=11, mew=2, zorder=3)
            ax.annotate(f"early death\n{m['stop']}", (dose, m["peak_D"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=8, color="tab:red")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_xticks([0.0, 0.5, 1.0, 2.0, 4.0])
    ax.set_xticklabels(["0x\n(BGK ctrl)", "0.5x", "1x", "2x", "4x"])
    ax.set_xlabel("REG dose  (rr_tau / 0.5003877)")
    ax.set_ylabel("peak penetration depth d/D")
    ax.set_title("(a) REG dose-response (x = died before contact)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")

    # ---- panel (b): depth vs time -----------------------------------------
    for arm, label in PLANE_ARMS:
        p = runs / arm / "result.json"
        if not p.exists():
            continue
        res = load(p)
        m = metrics(res)
        if m["contact"] is None:
            continue
        ts = [r["time_ms"] - m["contact"] * m["dt_ms"] for r in m["hist"] if r.get("depth_over_D") is not None]
        ds = [float(r["depth_over_D"]) for r in m["hist"] if r.get("depth_over_D") is not None]
        ax2.plot(ts, ds, lw=1.6, label=label)
    ax2.plot([EXP_PEAK_MS_PC], [EXP_PEAK_D], "k*", ms=14,
             label=f"exp {EXP_PEAK_D} D @{EXP_PEAK_MS_PC} ms")
    ax2.plot([one_x["peak_time_ms_pc"]], [one_x["peak"]["depth_over_D"]], "o",
             color="tab:red", ms=9, label="1x REG peak (cited)")
    ax2.set_xlabel("time after contact (ms)")
    ax2.set_ylabel("penetration depth d/D")
    ax2.set_title("(b) depth history, 24.005 ms window")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure -> {out}")


def field_figure(runs: Path, out: Path) -> None:
    panels = []
    for arm, label in PLANE_ARMS:
        rj = runs / arm / "result.json"
        npz = runs / arm / "planes.npz"
        if not rj.exists() or not npz.exists():
            continue
        res = load(rj)
        m = metrics(res)
        data = np.load(npz, allow_pickle=True)
        fills = data["fill"]
        times = data["time_ms"]
        uz = data["uz_m_s"]
        if not len(times):
            continue
        # the plane dump is registered on *absolute* simulation times
        if m["peak_abs_ms"] is not None:
            k = int(np.argmin(np.abs(np.asarray(times) - m["peak_abs_ms"])))
        else:
            k = 0
        panels.append((label, arm, m, float(times[k]), np.asarray(fills[k]), np.asarray(uz[k]),
                       np.asarray(data["x_over_D"]), np.asarray(data["z_over_D"])))
    if not panels:
        print("no plane dumps available -- field figure skipped")
        return
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.0), squeeze=False)
    for ax, (label, arm, m, t, fill, uz, xs, zs) in zip(axes[0], panels):
        im = ax.pcolormesh(xs, zs, fill, cmap="viridis", vmin=0.0, vmax=1.0, shading="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, label="fill fraction")
        pc = f", peak pc {m['peak_pc_ms']:.3f} ms" if m["peak_pc_ms"] is not None else ""
        ax.set_title(f"{label}\n{arm}: t={t:.3f} ms (abs){pc}\npeak {m['peak_D']:.4f} D", fontsize=9)
        ax.set_xlabel("x / D")
        ax.set_ylabel("z / D")
    fig.suptitle("DE-X-COL2D-REGDOSE: mid-plane fill field (SDAA planes dump)", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"figure -> {out} ({n} panels)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--cited", default="artifacts/COL2D-REGDOSE/col2_summary_cited.json")
    ap.add_argument("--out-dir", default="artifacts/COL2D-REGDOSE")
    a = ap.parse_args()
    runs = Path(a.runs_dir)
    out_dir = Path(a.out_dir)
    response_figure(runs, load(Path(a.cited)), out_dir / "regdose_response.png")
    field_figure(runs, out_dir / "regdose_field.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
