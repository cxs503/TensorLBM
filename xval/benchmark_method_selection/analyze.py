#!/usr/bin/env python3
"""Aggregate the XVAL-LBM-METHOD-SELECTION matrix into report tables.

Inputs (raw provenance; nothing is hand-typed into the report):
  runs/<arm>_D<D>/result.json    KPI record of a completed arm
  runs/<arm>_D<D>/fail.json      stability failure of an arm (excluded from KPIs)
  oss_crosscheck.json            executed-OSS-source kernel comparison (K0/K1/K2)
  predeclared_v2.json            thresholds, bands and rules fixed before the round-2 runs
  predeclared_v3.json            resolution-ladder decision/thresholds fixed before ANY
                                 D=14 or D=20 KPI existed (round 3)

Outputs:
  summary.json                   every table the report renders
  table.csv                      flat arm x KPI table
  figures/*.png                  lift traces, KPI bars vs the 15 % band,
                                 resolution ladder, vorticity snapshots,
                                 arm-vs-A1 gap across the ladder

Rules are the predeclared ones:
  * acceptance vs the recorded baseline A1 at the same resolution: |Delta| < 15 %
  * convergence: successive-rung |Delta| < 10 % (conv_pct, carried over from round 2)
  * St cross-check: |St_fft - St_zc| / St_fft < 5 %
  * a failing arm is reported, never tuned and never silently dropped
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
KPIS = ("cd_mean", "cd_mean_last50", "cl_amp", "st_fft", "st_zc")
OSS_STYLE_ARMS = ("A2", "A3", "A5")  # arms whose selection is the OSS praxis
CONVERGENCE_ARMS = ("A1", "A2", "A4")
RES = (10, 14, 20)


def pct(a: float, b: float) -> float:
    """Signed % difference of *a* from *b*."""
    if b == 0 or not np.isfinite(b) or not np.isfinite(a):
        return float("nan")
    return 100.0 * (a - b) / b


def sha256_of(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


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


def rungs_of(table: dict, arm: str) -> list[tuple[int, dict]]:
    return [(D, table[f"{arm}_D{D}"]) for D in RES if f"{arm}_D{D}" in table]


def build_convergence(table: dict, conv_pct: float) -> dict:
    """Successive-rung convergence per arm per KPI (predeclared 10 % rule)."""
    out: dict[str, dict] = {}
    for arm in CONVERGENCE_ARMS:
        rungs = rungs_of(table, arm)
        if len(rungs) < 2:
            continue
        entry: dict = {"rungs": [D for D, _ in rungs], "kpis": {}}
        for kpi in ("cd_mean", "cd_mean_last50", "cl_amp", "st_fft"):
            successive = []
            for (da, ea), (db, eb) in zip(rungs, rungs[1:]):
                d = pct(eb[kpi], ea[kpi])
                successive.append({"from": da, "to": db, "delta_pct": d,
                                   "converged_10pct": bool(np.isfinite(d) and abs(d) < conv_pct)})
            overall = pct(rungs[-1][1][kpi], rungs[0][1][kpi])
            entry["kpis"][kpi] = {
                "values": {str(D): e[kpi] for D, e in rungs},
                "successive": successive,
                "overall_pct": overall,
                "converged": bool(successive and all(s["converged_10pct"] for s in successive)),
            }
        entry["converged"] = bool(entry["kpis"]) and all(k["converged"] for k in entry["kpis"].values())
        out[arm] = entry
    return out


def build_ladder_fit(table: dict, band: dict) -> dict:
    """1/D linear fit over the available rungs — CONTEXT ONLY, never a PASS/FAIL number."""
    out: dict[str, dict] = {}
    for arm in CONVERGENCE_ARMS:
        rungs = rungs_of(table, arm)
        if len(rungs) < 3:
            continue
        xs = np.array([1.0 / D for D, _ in rungs])
        entry = {}
        for kpi, (lo_b, hi_b) in ((k, band[k]) for k in ("cd_mean", "st_fft") if k in band):
            ys = np.array([e[kpi] for _, e in rungs])
            b1, b0 = np.polyfit(xs, ys, 1)
            resid = ys - (b0 + b1 * xs)
            entry[kpi] = {
                "intercept_1_over_D_to_zero": float(b0),
                "slope": float(b1),
                "max_residual_pct_of_value": float(100.0 * np.abs(resid).max() / abs(b0)),
                "band": [lo_b, hi_b],
                "intercept_in_band": bool(lo_b <= b0 <= hi_b),
                "adjudicating": False,
            }
        out[arm] = entry
    return out


def build_gap_ladder(table: dict, acc: float) -> dict:
    """A1-vs-OSS-arm KPI gap recomputed at every rung (does the gap survive refinement?)."""
    out: dict[str, dict] = {}
    for arm in OSS_STYLE_ARMS:
        rows = []
        for D in RES:
            e, b = table.get(f"{arm}_D{D}"), table.get(f"A1_D{D}")
            if not (e and b):
                continue
            rows.append({
                "D": D,
                "delta_pct": {k: pct(e[k], b[k]) for k in ("cd_mean", "cd_mean_last50", "cl_amp", "st_fft")},
                "within_15pct": all(abs(v) < acc for v in
                                    (pct(e[k], b[k]) for k in ("cd_mean", "cd_mean_last50", "cl_amp", "st_fft"))
                                    if np.isfinite(v)),
            })
        if rows:
            out[arm] = {
                "rungs": rows,
                "worst_abs_delta_pct": max(abs(v) for r in rows for v in r["delta_pct"].values()
                                           if np.isfinite(v)),
                "out_of_band_at_any_rung": any(not r["within_15pct"] for r in rows),
            }
    return out


def build_failure_probes(failures: list[dict]) -> dict:
    by: dict[str, list[dict]] = {}
    for f in failures:
        by.setdefault(f["arm"], []).append({
            "D": f["D"], "step": f["step"], "total_steps_planned": f["total_steps_planned"],
            "fail_stage": f["fail_stage"], "tau": f["tau"], "tau_minus": f["tau_minus"],
            "artifact": f["artifact"],
        })
    for rows in by.values():
        rows.sort(key=lambda r: r["D"])
    return by


def build_provenance(table: dict) -> dict:
    """Freeze the rung that existed before the ladder (D=10) by content hash."""
    out = {}
    for key in sorted(table):
        if not key.endswith("_D10"):
            continue
        files = {}
        for name in ("result.json", "trace.npz"):
            p = HERE / "runs" / key / name
            if p.exists():
                files[name] = sha256_of(p)
        out[key] = files
    return out


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
                              "u_in", "artifact")})
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

    # resolution convergence per arm (successive rungs)
    convergence = build_convergence(table, conv_pct)
    ladder_fit = build_ladder_fit(table, th.get("external_band", {}))
    gap_ladder = build_gap_ladder(table, acc)
    failure_probes = build_failure_probes(failures)
    provenance = build_provenance(table)

    # did the A3/A5 blow-up recur at a second resolution? (predeclared H_fail_persists)
    fail_persist = {arm: {"resolutions": [r["D"] for r in rows],
                          "all_failed": len(rows) == len([d for d in (10, 14) if True]),
                          "persists": len(rows) >= 2}
                    for arm, rows in failure_probes.items()}

    # H1 / H0 verdict over the OSS-style arms at every rung that has data
    judged = {f"{arm}_D{D}": table[f"{arm}_D{D}"].get("within_15pct")
              for arm in OSS_STYLE_ARMS for D in RES if f"{arm}_D{D}" in table}
    h1 = bool(judged) and all(bool(v) for v in judged.values())
    worst = max(((abs(v), k, kpi) for k in judged
                 for kpi, v in table[k].get("delta_vs_A1_pct", {}).items() if np.isfinite(v)),
                default=(float("nan"), None, None))

    # external caliber (non-adjudicating), highest available resolution first
    band = th.get("external_band", {})
    external = {}
    a1 = next((table[f"A1_D{D}"] for D in sorted(RES, reverse=True) if f"A1_D{D}" in table), None)
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
        "schema": "xval-method-selection/v3",
        "arms": {
            "A1": "MRT + sponge (recorded TensorLBM selection)",
            "A2": "BGK, no sponge (OSS single-relaxation selection)",
            "A3": "TRT magic Lambda=3/16, no sponge (literal OSS transplant)",
            "A4": "MRT, no sponge (sponge-isolated control)",
            "A5": "TRT with Lambda re-derived, tau_minus=2.0, no sponge (A3 remediation)",
        },
        "oss_style_arms": list(OSS_STYLE_ARMS),
        "rungs": list(RES),
        "rows": [dict(r) for r in table.values()] + failures,
        "table": table,
        "failures": failures,
        "failure_probes": failure_probes,
        "fail_persists": fail_persist,
        "convergence": convergence,
        "ladder_fit": ladder_fit,
        "gap_ladder": gap_ladder,
        "provenance_d10": provenance,
        "verdict": {
            "H1_all_oss_arms_within_15pct": h1,
            "per_arm_within_15pct": judged,
            "largest_abs_delta_pct": worst[0],
            "largest_abs_delta_kpi_of": worst[1],
            "largest_abs_delta_kpi": worst[2],
            "judged_at_resolutions": sorted({int(k.split("_D")[1]) for k in judged}),
            "H_conv_all_arms_converged": bool(convergence) and all(
                v["converged"] for v in convergence.values()),
            "H_gap_survives_refinement": bool(gap_ladder) and all(
                v["out_of_band_at_any_rung"] for v in gap_ladder.values()),
            "H_fail_persists_at_higher_D": bool(
                [v for v in fail_persist.values() if len(v["resolutions"]) >= 2]) and all(
                v["persists"] for v in fail_persist.values() if len(v["resolutions"]) >= 2),
            "fail_persists_per_arm": {arm: {"resolutions": v["resolutions"], "persists": v["persists"]}
                                      for arm, v in fail_persist.items()},
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

    # 1) lift coefficient traces, last ~6 shedding periods, vertically offset
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

    # 2) KPI bars vs A1 and the predeclared 15 % band; one panel group per rung with data
    res = [D for D in RES if any(f"{a}_D{D}" in table for a in ("A1", "A2", "A4"))]
    fig, axes = plt.subplots(len(res), 3, figsize=(11.5, 3.2 * len(res)), squeeze=False)
    for jr, D in enumerate(res):
        keys = [f"{a}_D{D}" for a in ("A1", "A2", "A3", "A4", "A5") if f"{a}_D{D}" in table]
        labels = [f"{table[k]['arm']}\n{table[k]['collision'][:4]}" for k in keys]
        for ax, kpi, title in zip(axes[jr], ("cd_mean", "cl_amp", "st_fft"),
                                  ("mean $C_d$", "$C_l$ amplitude", "Strouhal number")):
            vals = [table[k][kpi] for k in keys]
            ax.bar(range(len(keys)), vals, color="#4C78A8")
            base = table[f"A1_D{D}"][kpi]
            ax.axhline(base, color="k", lw=1.1, ls="--", label="A1 (recorded)")
            ax.fill_between([-0.6, len(keys) - 0.4], base * (1 - acc / 100), base * (1 + acc / 100),
                            color="k", alpha=0.09, label="15 % band")
            for i, v in enumerate(vals):
                dv = pct(v, base)
                ax.text(i, v, f"{dv:+.1f}%", ha="center", va="bottom", fontsize=7)
            ax.set_xticks(range(len(keys)))
            ax.set_xticklabels(labels, fontsize=7)
            ax.set_title(f"{title} @ D={D}")
            ax.grid(alpha=0.3, axis="y")
        axes[jr][0].legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "fig2_kpi_vs_A1.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p)

    # 3) resolution ladder: does every KPI keep moving?
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.3))
    for ax, kpi, title in zip(axes, ("cd_mean", "st_fft", "cl_amp"),
                              ("mean $C_d$", "Strouhal number", "$C_l$ amplitude")):
        for arm, color in (("A1", "#4C78A8"), ("A2", "#F58518"), ("A4", "#54A24B")):
            rungs = rungs_of(table, arm)
            if rungs:
                ax.plot([D for D, _ in rungs], [e[kpi] for _, e in rungs], "o-", color=color, label=arm)
        if kpi in summary.get("external_caliber", {}):
            e = summary["external_caliber"][kpi]
            ax.axhspan(*e["band"], color="0.6", alpha=0.25, label="trend band (non-adjudicating)")
        ax.set_xlabel("D (lattice cells, D = cylinder side)")
        ax.set_xticks(list(RES))
        ax.set_title(f"{title}: resolution ladder")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    p = out_dir / "fig3_resolution_ladder.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append(p)

    # 4) vorticity snapshots (final state) — sponge effect is visible downstream
    panels = [k for k in ("A1_D10", "A4_D10", "A2_D10", "A1_D20") if k in table and
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

    # 5) arm-vs-A1 gap across the ladder: does the selection difference survive refinement?
    gaps = summary.get("gap_ladder", {})
    if gaps:
        fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
        for arm, color in (("A2", "#F58518"), ("A3", "#E45756"), ("A5", "#72B7B2")):
            rows = gaps.get(arm, {}).get("rungs", [])
            if not rows:
                continue
            axes[0].plot([r["D"] for r in rows], [table[f"{arm}_D{r['D']}"]["cd_mean"] for r in rows],
                         "o-", color=color, label=arm)
            axes[1].plot([r["D"] for r in rows], [r["delta_pct"]["cd_mean"] for r in rows],
                         "o-", color=color, label=arm)
        a1r = rungs_of(table, "A1")
        if a1r:
            axes[0].plot([D for D, _ in a1r], [e["cd_mean"] for _, e in a1r], "s--", color="#4C78A8",
                         label="A1 (recorded)")
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].axhspan(-acc, acc, color="k", alpha=0.09, label="15 % band")
        axes[0].set_title("mean $C_d$ vs resolution")
        axes[1].set_title("$\\Delta C_d$ vs A1 (same D)")
        for ax in axes:
            ax.set_xlabel("D (lattice cells)")
            ax.set_xticks(list(RES))
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7)
        fig.tight_layout()
        p = out_dir / "fig5_gap_ladder.png"
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
                      "ladder_arms": list(summary["convergence"]),
                      "figures": [str(f.relative_to(HERE)) for f in figs]}, indent=2))
    for key in sorted(summary["table"], key=lambda k: (summary["table"][k]["arm"], summary["table"][k]["D"])):
        e = summary["table"][key]
        dv = e.get("delta_vs_A1_pct", {})
        print(f"{key:9s} Cd={e['cd_mean']:.4f} Clamp={e['cl_amp']:.4f} St={e['st_fft']:.4f} "
              f"(zc {e['st_zc']:.4f}) dCd={dv.get('cd_mean', float('nan')):+.2f}% "
              f"dSt={dv.get('st_fft', float('nan')):+.2f}% ok={e.get('within_15pct')}")
    for f in summary["failures"]:
        print(f"FAIL {f['arm']}_D{f['D']} stage={f['fail_stage']} step={f['step']}/{f['total_steps_planned']}")
    print(json.dumps(summary["verdict"], indent=1))


if __name__ == "__main__":
    main()