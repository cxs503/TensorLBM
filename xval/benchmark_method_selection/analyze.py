#!/usr/bin/env python3
"""Aggregate the XVAL-LBM-BENCHMARK-METHOD-SELECTION arms into the report tables.

Reads ``runs/<arm>_D<d>/result.json`` (programmatic provenance: every number in the
report tables comes from these files) and writes ``summary.json`` + ``table.csv`` +
the comparison figures.

Rules are the predeclared ones (see predeclared.json):
  * acceptance vs the recorded baseline A1 at the same resolution: |Delta| < 15 %
  * convergence: |KPI(D=12) - KPI(D=10)| / |KPI(D=12)| < 10 %
  * St cross-check: |St_fft - St_zc| / St_fft < 5 %
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KPIS = ("cd_mean", "cd_mean_last50", "cl_amp", "st_fft", "st_zc")
ACCEPT_PCT = 15.0
CONV_PCT = 10.0
ST_XCHECK_PCT = 5.0


def load_runs(root: Path = HERE / "runs") -> dict[str, dict]:
    out = {}
    for p in sorted(root.glob("*/result.json")):
        d = json.loads(p.read_text())
        key = f"{d['arm']}_D{d['D']}"
        if abs(d.get("u_in", 0.05) - 0.05) > 1e-9:
            key += f"_U{d['u_in']:.3f}"
        out[key] = d
    for p in sorted(root.glob("*/fail.json")):
        d = json.loads(p.read_text())
        out[f"{d['arm']}_D{d['D']}_FAIL"] = d
    return out


def pct(a: float, b: float) -> float:
    """Signed relative difference of ``a`` w.r.t. ``b`` in percent."""
    if b == 0 or not np.isfinite(b):
        return float("nan")
    return (a - b) / b * 100.0


def build_summary(runs: dict[str, dict]) -> dict:
    rows = []
    for key, d in runs.items():
        if d.get("status") == "blowup":
            rows.append({"arm": d["arm"], "D": d["D"], "status": "blowup", "step": d["step"]})
            continue
        rows.append(
            {
                "arm": d["arm"],
                "D": d["D"],
                "collision": d["collision"],
                "sponge": d["sponge"],
                "tau": d["tau"],
                "nu": d["nu"],
                "ma": d["ma"],
                "re": d["re"],
                "u_in": d["u_in"],
                "blockage": d["blockage"],
                "steps": d["total_steps"],
                "wall_s": d["wall_s"],
                "status": "ok" if d["stability_ok"] else "unstable",
                "cl_max_abs": d["cl_max_abs"],
                **{k: d[k] for k in KPIS},
            }
        )

    table = {}
    for row in rows:
        if row["status"] != "ok":
            continue
        base_key = f"A1_D{row['D']}"
        base = table.get(base_key) or next(
            (r for r in rows if r["arm"] == "A1" and r["D"] == row["D"] and r["status"] == "ok"), None
        )
        entry = dict(row)
        if base is not None and row["arm"] != "A1":
            entry["delta_vs_A1_pct"] = {
                k: pct(row[k], base[k]) for k in ("cd_mean", "cl_amp", "st_fft")
            }
            entry["within_15pct"] = all(
                abs(v) < ACCEPT_PCT for v in entry["delta_vs_A1_pct"].values()
            )
        sp = (
            abs(pct(row["st_fft"], row["st_zc"])) if np.isfinite(row["st_zc"]) else float("nan")
        )
        entry["st_crosscheck_pct"] = sp
        entry["st_crosscheck_ok"] = bool(np.isfinite(sp) and sp < ST_XCHECK_PCT)
        table[f"{row['arm']}_D{row['D']}"] = entry

    convergence = {}
    for arm in ("A1", "A2", "A3", "A4", "A5"):
        lo, hi = table.get(f"{arm}_D10"), table.get(f"{arm}_D12")
        if lo and hi:
            convergence[arm] = {
                k: {
                    "D10": lo[k],
                    "D12": hi[k],
                    "delta_pct": pct(hi[k], lo[k]),
                    "converged_10pct": bool(abs(pct(hi[k], lo[k])) < CONV_PCT),
                }
                for k in ("cd_mean", "cl_amp", "st_fft")
            }
    return {
        "rows": rows,
        "table": table,
        "convergence": convergence,
        "thresholds": {"accept_pct": ACCEPT_PCT, "conv_pct": CONV_PCT, "st_xcheck_pct": ST_XCHECK_PCT},
    }


def write_csv(summary: dict, path: Path) -> None:
    cols = ["arm", "D", "collision", "sponge", "tau", "ma", "blockage", "steps",
            "cd_mean", "cd_mean_last50", "cl_amp", "st_fft", "st_zc", "st_crosscheck_pct",
            "wall_s", "status"]
    lines = [",".join(cols)]
    for r in summary["rows"]:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def make_figures(summary: dict, runs: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figs = []

    # 1) Cl time traces of every arm at D=10 (last 6 shedding periods)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    parsed = HERE / "runs"
    for key in ("A1_D10", "A2_D10", "A4_D10", "A5_D10"):
        npz = parsed / key / "trace.npz"
        if not npz.exists():
            continue
        z = np.load(npz)
        d = runs[key]
        cl = 2.0 * z["fy"] / (d["u_in"] ** 2 * d["D"])
        n = cl.size
        n_show = min(n, 6 * int(d["D"] / (0.15 * d["u_in"])))
        ax.plot(np.arange(n - n_show, n), cl[-n_show:], lw=0.9, label=key)
    ax.set_xlabel("lattice step")
    ax.set_ylabel("$C_l$")
    ax.set_title("Square cylinder Re=200, blockage 1/20, D=10 — lift coefficient")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "cl_traces_D10.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(p)

    # 2) Cd / St / Cl_amp per arm at D=10 with the 15 % band around A1
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    keys = [k for k in ("A1_D10", "A2_D10", "A3_D10", "A4_D10", "A5_D10") if k in summary["table"]]
    labels = [summary["table"][k]["collision"] + ("+sp" if summary["table"][k]["sponge"] else "") for k in keys]
    for ax, kpi, title in zip(
        axes,
        ("cd_mean", "cl_amp", "st_fft"),
        ("mean $C_d$", "$C_l$ amplitude", "Strouhal number"),
    ):
        vals = [summary["table"][k][kpi] for k in keys]
        ax.bar(range(len(keys)), vals, color="#4C78A8")
        base = summary["table"]["A1_D10"][kpi]
        ax.axhline(base, color="k", lw=1, ls="--", label="A1 (recorded)")
        ax.fill_between(
            [-0.5, len(keys) - 0.5],
            base * (1 - ACCEPT_PCT / 100.0),
            base * (1 + ACCEPT_PCT / 100.0),
            color="k",
            alpha=0.08,
            label="15 % band",
        )
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, rotation=30, fontsize=7)
        ax.set_title(f"{title} @ D=10")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "kpi_bars_D10.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(p)

    # 3) resolution ladder A1/A2: D=10 vs D=12
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, kpi, title in zip(axes, ("cd_mean", "st_fft"), ("mean $C_d$", "Strouhal number")):
        for arm, color in (("A1", "#4C78A8"), ("A2", "#F58518"), ("A5", "#54A24B")):
            lo, hi = summary["table"].get(f"{arm}_D10"), summary["table"].get(f"{arm}_D12")
            xs, ys = [], []
            if lo:
                xs.append(10), ys.append(lo[kpi])
            if hi:
                xs.append(12), ys.append(hi[kpi])
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=arm)
        ax.set_xlabel("D (lattice cells)")
        ax.set_xticks([10, 12])
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "resolution_ladder.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(p)
    return figs


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs found — run run_arm.py first")
    summary = build_summary(runs)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(summary, HERE / "table.csv")
    figs = make_figures(summary, runs, HERE / "figures")
    print(json.dumps({"runs": len(runs), "figures": [str(f) for f in figs]}, indent=2))
    for k, v in summary["table"].items():
        print(
            f"{k:9s} Cd={v['cd_mean']:.4f} Clamp={v['cl_amp']:.4f} "
            f"St={v['st_fft']:.4f} (zc {v['st_zc']:.4f}) d_vs_A1={v.get('delta_vs_A1_pct')}"
        )


if __name__ == "__main__":
    main()