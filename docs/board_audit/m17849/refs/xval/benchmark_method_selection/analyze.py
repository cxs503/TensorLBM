#!/usr/bin/env python3
"""Aggregate the XVAL-LBM-BENCHMARK-METHOD-SELECTION matrix into report tables.

Inputs (raw provenance; nothing is hand-typed into the report):
  runs/<arm>_D<D>/result.json    KPI record of a completed arm
  runs/<arm>_D<D>/fail.json      stability failure of an arm (excluded from KPIs)
  oss_crosscheck.json            executed-OSS-source kernel comparison (K0/K1/K2)
  predeclared_v2.json            thresholds, bands and rules fixed before the runs

Outputs:
  summary.json                   every table the report renders
  table.csv                      flat arm x KPI table
  figures/*.png                  lift traces, KPI bars vs the 15 % band,
                                 resolution ladder, vorticity snapshots

Rules are the predeclared ones:
  * acceptance vs the recorded baseline A1 at the same resolution: |Delta| < 15 %
  * convergence: |KPI(D=14) - KPI(D=10)| / |KPI(D=14)| < 10 %
  * St cross-check: |St_fft - St_zc| / St_fft < 5 %
  * a failing arm is reported, never tuned and never silently dropped
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KPIS = ("cd_mean", "cd_mean_last50", "cl_amp", "st_fft", "st_zc")
OSS_STYLE_ARMS = ("A2", "A3", "A5")  # arms whose selection is the OSS praxis
CONVERGENCE_ARMS = ("A1", "A2", "A4")
RES = (10, 14)


def pct(a: float, b: float) -> float:
    """Signed % difference of *a* from *b*."""
    if b == 0 or not np.isfinite(b) or not np.isfinite(a):
        return float("nan")
    return 100.0 * (a - b) / b


def load_runs(root: Path = HERE / "runs") -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(root.glob("*/result.json")):
        d = json.loads(p.read_text())
        out[f"{d['arm']}_D{d['D']}"] = {**d, "artifact": str(p.relative_to(HERE))}
    for p in sorted(root.glob("*/fail.json")):
        d = json.loads(p.read_text())
        out[f"{d['arm']}_D{d['D']}_FAIL"] = {**d, "artifact": str(p.relative_to(HERE))}
    return out


def thresholds() -> dict:
    pre = json.loads((HERE / "predeclared_v2.json").read_text())
    t = pre["predeclared_thresholds"]
    band = t.get("external_caliber_band", {})
    return {
        "accept_pct": t["owner_acceptance_pct"],
        "conv_pct": 10.0,
        "st_xcheck_pct": t["st_crosscheck_tolerance_pct"],
        "external_band": band,
    }


def build_summary(runs: dict[str, dict]) -> dict:
    th = thresholds()
    acc, conv_pct, xp = th["accept_pct"], th["conv_pct"], th["st_xcheck_pct"]

    table: dict[str, dict] = {}
    failures: list[dict] = []
    for key, row in runs.items():
        if key.endswith("_FAIL"):
            failures.append({k: row.get(k) for k in
                             ("arm", "D", "collision", "sponge", "status", "fail_stage",
                              "step", "total_steps_planned", "cl_max_abs", "tau", "tau_minus",
                              "artifact")})
            continue
        entry = dict(row)
        sp = pct(entry["st_zc"], entry["st_fft"])
        entry["st_crosscheck_pct"] = sp
        entry["st_crosscheck_ok"] = bool(np.isfinite(sp) and abs(sp) < xp)
        table[key] = entry

    # deltas vs the recorded selection A1 at the same resolution
    for key, entry in table.items():
        base = table.get(f"A1_D{entry['D']}")
        if base is None or entry["arm"] == "A1":
            continue
        deltas = {k: pct(entry[k], base[k]) for k in ("cd_mean", "cd_mean_last50", "cl_amp", "st_fft")}
        entry["delta_vs_A1_pct"] = deltas
        entry["within_15pct"] = bool(all(np.isfinite(v) and abs(v) < acc for v in deltas.values()))

    # resolution convergence per arm
    convergence: dict[str, dict] = {}
    for arm in CONVERGENCE_ARMS:
        lo, hi = table.get(f"{arm}_D10"), table.get(f"{arm}_D14")
        if not (lo and hi):
            continue
        convergence[arm] = {
            kpi: {
                "D10": lo[kpi],
                "D14": hi[kpi],
                "delta_pct": pct(hi[kpi], lo[kpi]),
                "converged_10pct": bool(abs(pct(hi[kpi], lo[kpi])) < conv_pct),
            }
            for kpi in ("cd_mean", "cl_amp", "st_fft")
        }

    # H1 / H0 verdict over the OSS-style arms at D=10 (the resolution all arms share)
    judged = [f"{arm}_D10" for arm in OSS_STYLE_ARMS if f"{arm}_D10" in table]
    verdicts = {k: table[k].get("within_15pct") for k in judged}
    h1 = bool(verdicts) and all(bool(v) for v in verdicts.values())
    worst = max(
        ((abs(v), k) for k in judged for v in table[k].get("delta_vs_A1_pct", {}).values()
         if np.isfinite(v)),
        default=(float("nan"), None),
    )

    # external caliber (non-adjudicating)
    band = th.get("external_band", {})
    external = {}
    a1 = table.get("A1_D14") or table.get("A1_D10")
    if band and a1:
        for kpi, (lo_b, hi_b) in ((k, band[k]) for k in ("cd_mean", "st_fft") if k in band):
            v = a1[kpi]
            external[kpi] = {
                "value": v,
                "resolution": a1["D"],
                "band": [lo_b, hi_b],
                "in_band": bool(lo_b <= v <= hi_b),
                "delta_from_band_centre_pct": pct(v, 0.5 * (lo_b + hi_b)),
                "adjudicating": False,
            }

    # executed-OSS-source kernel checks (ref = OSS code, got = TensorLBM kernel)
    oss_path = HERE / "oss_crosscheck.json"
    oss = json.loads(oss_path.read_text()) if oss_path.exists() else {}
    crosscheck = {
        "k0_direction_map": oss.get("k0_direction_map", {}),
        "k1_collision": oss.get("k1_collision", {}),
        "k2_streaming": oss.get("k2_streaming", {}),
        "throughput": oss.get("throughput", {}),
        "oss_commit": oss.get("oss_commit"),
        "oss_source_sha256": oss.get("oss_source_sha256_unmodified", {}),
    }

    return {
        "schema": "xval-method-selection/v2",
        "arms": {
            "A1": "MRT + sponge (recorded TensorLBM selection)",
            "A2": "BGK, no sponge (OSS single-relaxation selection)",
            "A3": "TRT magic Lambda=3/16, no sponge (literal OSS transplant)",
            "A4": "MRT, no sponge (sponge-isolated control)",
            "A5": "TRT with Lambda re-derived, tau_minus=2.0, no sponge (A3 remediation)",
        },
        "oss_style_arms": list(OSS_STYLE_ARMS),
        "rows": [dict(r) for r in table.values()] + failures,
        "table": table,
        "failures": failures,
        "convergence": convergence,
        "verdict": {
            "H1_all_oss_arms_within_15pct": h1,
            "per_arm_within_15pct": verdicts,
            "largest_abs_delta_pct": worst[0],
            "largest_abs_delta_kpi_of": worst[1],
            "judged_at_resolution": 10,
        },
        "external_caliber": external,
        "crosscheck": crosscheck,
        "thresholds": th,
    }


def write_csv(summary: dict, path: Path) -> None:
    cols = ["arm", "D", "collision", "sponge", "tau", "ma", "u_in", "re", "blockage", "total_steps",
            "cd_mean", "cd_mean_last50", "cl_mean", "cl_amp", "st_fft", "st_zc",
            "st_crosscheck_pct", "wall_s", "status"]
    lines = [",".join(cols)]
    for r in sorted(summary["table"].values(), key=lambda x: (x["arm"], x["D"])):
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines) + "\n")


def make_figures(summary: dict, out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figs: list[Path] = []
    table, acc = summary["table"], summary["thresholds"]["accept_pct"]
    runs_dir = HERE / "runs"

    def trace(key: str):
        npz = runs_dir / key / "trace.npz"
        if not npz.exists():
            return None
        return np.load(npz)

    # 1) lift coefficient traces, D=10, last ~6 shedding periods, vertically offset
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    shown = [k for k in ("A1_D10", "A2_D10", "A4_D10", "A5_D10") if k in table]
    for i, key in enumerate(shown):
        z = trace(key)
        if z is None:
            continue
        d = table[key]
        cl = 2.0 * z["fy"] / (d["u_in"] ** 2 * d["D"])
        n_show = min(cl.size, int(6 * d["D"] / (0.165 * d["u_in"])))
        ax.plot(np.arange(n_show), cl[-n_show:] + 1.2 * i, lw=0.9,
                label=f"{key} ({d['collision']}{'+sponge' if d['sponge'] else ''})")
    ax.set_xlabel("lattice step (relative, last 6 shedding periods)")
    ax.set_ylabel("$C_l$ (offset +1.2 per arm)")
    ax.set_title("Square cylinder Re=200, blockage 1/20, D=10: lift histories")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out_dir / "fig1_cl_traces.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p)

    # 2) KPI bars vs A1 and the predeclared 15 % band
    keys = [k for k in ("A1_D10", "A2_D10", "A3_D10", "A4_D10", "A5_D10") if k in table]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.5))
    labels = [f"{table[k]['arm']}\n{table[k]['collision'][:4]}" for k in keys]
    for ax, kpi, title in zip(axes, ("cd_mean", "cl_amp", "st_fft"),
                              ("mean $C_d$", "$C_l$ amplitude", "Strouhal number")):
        vals = [table[k][kpi] for k in keys]
        ax.bar(range(len(keys)), vals, color="#4C78A8")
        base = table["A1_D10"][kpi]
        ax.axhline(base, color="k", lw=1.1, ls="--", label="A1 (recorded)")
        ax.fill_between([-0.6, len(keys) - 0.4], base * (1 - acc / 100), base * (1 + acc / 100),
                        color="k", alpha=0.09, label="15 % band")
        for i, v in enumerate(vals):
            dv = pct(v, base)
            ax.text(i, v, f"{dv:+.1f}%", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_title(f"{title} @ D=10")
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "fig2_kpi_vs_A1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p)

    # 3) resolution ladder
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.3))
    for ax, kpi, title in zip(axes, ("cd_mean", "st_fft"), ("mean $C_d$", "Strouhal number")):
        for arm, color in (("A1", "#4C78A8"), ("A2", "#F58518"), ("A4", "#54A24B")):
            xs, ys = [], []
            for D in RES:
                e = table.get(f"{arm}_D{D}")
                if e:
                    xs.append(D), ys.append(e[kpi])
            if xs:
                ax.plot(xs, ys, "o-", color=color, label=arm)
        ax.set_xlabel("D (lattice cells)")
        ax.set_xticks(list(RES))
        ax.set_title(f"{title}: resolution ladder")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    p = out_dir / "fig3_resolution_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p)

    # 4) vorticity snapshots (final state) — sponge effect is visible downstream
    panels = [k for k in ("A1_D10", "A4_D10", "A2_D10", "A5_D10") if k in table and
              (runs_dir / k / "trace.npz").exists()]
    if panels:
        fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 1.75 * len(panels)))
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, panels):
            z = trace(key)
            vort = z["vort"]
            vmax = np.percentile(np.abs(vort), 99.5) or 1.0
            im = ax.imshow(vort, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                           extent=[0, vort.shape[1], 0, vort.shape[0]], aspect="auto")
            ax.contour(z["solid"].astype(float), levels=[0.5], colors="k", linewidths=0.6,
                       extent=[0, vort.shape[1], 0, vort.shape[0]])
            ax.set_title(f"{key} ({table[key]['collision']}"
                         f"{'+sponge' if table[key]['sponge'] else ''}) — final vorticity",
                         fontsize=8)
            ax.set_ylabel("y (cells)", fontsize=7)
            ax.tick_params(labelsize=6)
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        axes[-1].set_xlabel("x (cells)", fontsize=8)
        fig.tight_layout()
        p = out_dir / "fig4_vorticity_fields.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append(p)
    return figs


def main() -> None:
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs found — run xval/benchmark_method_selection/run_matrix.sh first")
    summary = build_summary(runs)
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(summary, HERE / "table.csv")
    figs = make_figures(summary, HERE / "figures")
    print(json.dumps({"runs": len(summary["table"]), "failures": len(summary["failures"]),
                      "figures": [str(f.relative_to(HERE)) for f in figs]}, indent=2))
    for key in sorted(summary["table"], key=lambda k: (summary["table"][k]["arm"], summary["table"][k]["D"])):
        e = summary["table"][key]
        dv = e.get("delta_vs_A1_pct", {})
        print(f"{key:9s} Cd={e['cd_mean']:.4f} Clamp={e['cl_amp']:.4f} St={e['st_fft']:.4f} "
              f"(zc {e['st_zc']:.4f}) dCd={dv.get('cd_mean', float('nan')):+.2f}% "
              f"dSt={dv.get('st_fft', float('nan')):+.2f}% ok={e.get('within_15pct')}")
    for f in summary["failures"]:
        print(f"FAIL {f['arm']}_D{f['D']} stage={f['fail_stage']} step={f['step']}/{f['total_steps_planned']}")


if __name__ == "__main__":
    main()