"""Round-2 aggregation for XVAL-LBM-ROUND2-METHOD-SELECTION.

Reads the per-arm artifacts written by ``run_arm.py``
(``runs/<arm>_D<D>/{result.json|fail.json}``), builds

  * ``table_round2.csv``        one row per (arm, D): KPIs + delta vs A1 at the same D
  * ``summary_round2.json``     gate verdicts, resolution ladder, 2x2 factorial effects
  * ``figures/fig_r2_*.png``    delta-vs-D, convergence ladder, factorial, Cl traces

and deliberately does **not** look at thresholds defined anywhere but
``predeclared_round2.json`` -- the numbers are gated, not the narrative.

Usage
-----
  python xval/benchmark_method_selection/analyze_round2.py
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
PREDECL = json.loads((ROOT / "predeclared_round2.json").read_text())

# 2x2 factorial {MRT, BGK} x {sponge on, off}; A1 is the reference cell.
FACTORIAL = {"MRT_on": "A1", "MRT_off": "A4", "BGK_on": "A6", "BGK_off": "A2"}
GATE_KPIS = PREDECL["predeclared_thresholds"]["gate_kpis"]          # cd_mean, st_fft, cl_amp
OSS_STYLE_ARMS = ["A2", "A6", "A4"]                                  # arms vs A1 in the gate
ARM_LABEL = {
    "A1": "MRT + sponge (recorded TensorLBM selection)",
    "A2": "BGK, no sponge (OSS single-relaxation selection)",
    "A3": "TRT magic Lambda=3/16, no sponge (literal OSS transplant)",
    "A4": "MRT, no sponge (sponge isolated)",
    "A5": "TRT with Lambda re-derived, no sponge",
    "A6": "BGK + sponge (factorial completion)",
}
RESOLUTIONS = [10, 14, 20]


def load_runs() -> tuple[dict, dict]:
    ok, fail = {}, {}
    if not RUNS.exists():
        return ok, fail
    for d in sorted(RUNS.iterdir()):
        if not d.is_dir() or "_D" not in d.name:
            continue
        res = d / "result.json"
        bad = d / "fail.json"
        # key on the directory name, not on rec["arm"]: the determinism replica
        # runs the same arm (A2) into runs/A2rep_D<D>/ and must not overwrite it.
        tag = d.name.split("_D")[0]
        if res.exists():
            rec = json.loads(res.read_text())
            rec["dir_tag"] = tag
            ok[(tag, int(rec["D"]))] = rec
        elif bad.exists():
            rec = json.loads(bad.read_text())
            rec["dir_tag"] = tag
            fail[(tag, int(rec["D"]))] = rec
    return ok, fail


def pct(got: float, ref: float) -> float:
    if ref is None or got is None or not math.isfinite(got) or not math.isfinite(ref) or ref == 0:
        return float("nan")
    return 100.0 * (got - ref) / abs(ref)


def rows_with_delta(ok: dict) -> list[dict]:
    rows = []
    for (arm, D), rec in sorted(ok.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ref = ok.get(("A1", D))
        row = {
            "arm": arm,
            "role": "determinism_replica" if arm.endswith("rep") else "matrix",
            "D": D,
            "collision": rec["collision"],
            "sponge": rec["sponge"],
            "nx": rec["nx"],
            "ny": rec["ny"],
            "tau": round(rec["tau"], 6),
            "nu": rec["nu"],
            "ma": round(rec["ma"], 5),
            "total_steps": rec["total_steps"],
            "wall_s": rec["wall_s"],
        }
        for k in ("cd_mean", "cd_mean_last50", "cl_amp", "cl_rms", "st_fft", "st_zc"):
            row[k] = rec.get(k)
            row[f"d_{k}_pct"] = pct(rec.get(k), ref.get(k)) if ref else float("nan")
        row["st_crosscheck_pct"] = (
            100.0 * (rec.get("st_fft", float("nan")) - rec.get("st_zc", float("nan")))
            / rec.get("st_fft", float("nan"))
        )
        row["gate_pass"] = all(
            abs(row[f"d_{k}_pct"]) < PREDECL["predeclared_thresholds"]["owner_acceptance_pct"]
            for k in GATE_KPIS
        ) if ref and arm != "A1" else None
        rows.append(row)
    return rows


def factorial(ok: dict, D: int, kpi: str) -> dict | None:
    cells = {k: ok.get((a, D)) for k, a in FACTORIAL.items()}
    if any(v is None for v in cells.values()):
        return None
    v = {k: cells[k][kpi] for k in cells}
    eff_c = 0.5 * ((v["BGK_on"] - v["MRT_on"]) + (v["BGK_off"] - v["MRT_off"]))
    eff_s = 0.5 * ((v["MRT_off"] - v["MRT_on"]) + (v["BGK_off"] - v["BGK_on"]))
    inter = (v["BGK_off"] - v["MRT_off"]) - (v["BGK_on"] - v["MRT_on"])
    base = v["MRT_on"]
    return {
        "kpi": kpi,
        "D": D,
        "cells": {k: round(v[k], 6) for k in v},
        "effect_collision_bgk_minus_mrt": round(eff_c, 6),
        "effect_sponge_off_minus_on": round(eff_s, 6),
        "interaction": round(inter, 6),
        "effect_collision_pct_of_A1": round(pct(base + eff_c, base), 3),
        "effect_sponge_pct_of_A1": round(pct(base + eff_s, base), 3),
        "interaction_pct_of_A1": round(pct(base + inter, base), 3),
        "separable": abs(inter) <= max(abs(eff_c), abs(eff_s)),
    }


def ladder(rows: list[dict], kpis=("cd_mean", "st_fft", "cl_amp", "cl_rms")) -> dict:
    out = {}
    for arm in sorted({r["arm"] for r in rows if not r["arm"].endswith("rep")}):
        per = {}
        for k in kpis:
            got = {r["D"]: r[k] for r in rows if r["arm"] == arm}
            conv = None
            if all(d in got for d in (10, 14, 20)):
                c1 = abs(got[20] - got[14]) / abs(got[20]) if got[20] else float("inf")
                c0 = abs(got[14] - got[10]) / abs(got[14]) if got[14] else float("inf")
                conv = bool(c1 < 0.10 and c0 < 0.10)
                per[f"{k}_rung_pct"] = {
                    "D10_to_D14": round(100 * (got[14] - got[10]) / abs(got[14]), 3),
                    "D14_to_D20": round(100 * (got[20] - got[14]) / abs(got[20]), 3),
                }
            per[k] = {str(d): (None if d not in got else got[d]) for d in RESOLUTIONS}
            per[f"{k}_converged"] = conv
        out[arm] = per
    return out


def closure(rows: list[dict]) -> dict:
    """H1: every OSS-style arm inside 15 % on gate KPIs at D=20 AND closer than at D=10."""
    out = {}
    by = {(r["arm"], r["D"]): r for r in rows if not r["arm"].endswith("rep")}
    for arm in OSS_STYLE_ARMS:
        per = {}
        for k in GATE_KPIS:
            d10 = by.get((arm, 10), {}).get(f"d_{k}_pct")
            d20 = by.get((arm, 20), {}).get(f"d_{k}_pct")
            inside = None if d20 is None else bool(abs(d20) < 15.0)
            closer = None if (d10 is None or d20 is None) else bool(abs(d20) < abs(d10))
            per[k] = {"delta_D10_pct": d10, "delta_D20_pct": d20,
                      "inside_15pct_at_D20": inside, "closer_at_D20": closer}
        vals = [v for k in per for v in (per[k]["inside_15pct_at_D20"], per[k]["closer_at_D20"])]
        per["H1_alt_A_resolution_artifact"] = (
            None if any(v is None for v in vals) else bool(all(vals))
        )
        out[arm] = per
    return out


def determinism(ok: dict, rows: list[dict]) -> dict | None:
    rep = ok.get(("A2rep", 10))
    base = ok.get(("A2", 10))
    if not rep or not base:
        return None
    return {
        "pair": "A2_D10 vs A2rep_D10 (same code, same thread count, independent run)",
        "cd_mean_rel": abs(rep["cd_mean"] - base["cd_mean"]) / abs(base["cd_mean"]),
        "st_fft_rel": abs(rep["st_fft"] - base["st_fft"]) / abs(base["st_fft"]),
        "cl_amp_rel": abs(rep["cl_amp"] - base["cl_amp"]) / abs(base["cl_amp"]),
        "threshold": 1e-9,
    }


def write_csv(rows: list[dict]) -> Path:
    cols = ["arm", "D", "collision", "sponge", "nx", "ny", "tau", "nu", "ma",
            "total_steps", "wall_s", "cd_mean", "cd_mean_last50", "cl_amp", "cl_rms",
            "st_fft", "st_zc", "st_crosscheck_pct",
            "d_cd_mean_pct", "d_st_fft_pct", "d_cl_amp_pct", "d_cl_rms_pct", "gate_pass"]
    p = ROOT / "table_round2.csv"
    with p.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in cols})
    return p


def figures(rows: list[dict], ok: dict, figdir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figdir.mkdir(exist_ok=True)
    out = []
    colors = {"A1": "#1f77b4", "A2": "#d62728", "A4": "#2ca02c", "A6": "#ff7f0e"}
    style = {"A1": "o-", "A2": "s--", "A4": "^:", "A6": "D-."}

    # fig 1: |delta| vs D for the gate KPIs against the 15 % band
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k in zip(axes, GATE_KPIS):
        for arm in ["A2", "A6", "A4"]:
            pts = sorted((r["D"], abs(r[f"d_{k}_pct"])) for r in rows
                         if r["arm"] == arm and math.isfinite(r.get(f"d_{k}_pct", float("nan"))))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style[arm],
                        color=colors[arm], label=arm, ms=6)
        ax.axhspan(0, 15, color="0.85", zorder=0, label="owner 15 % band")
        ax.set_xlabel("D (lattice cells per body width)")
        ax.set_ylabel(f"|delta {k}| vs A1 [%]")
        ax.set_yscale("log")
        ax.set_title(f"{k} vs resolution")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Round 2: selection gap vs resolution (log scale, band = 15 %)")
    f1 = figdir / "fig_r2_1_delta_vs_D.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=130)
    plt.close(fig)
    out.append(str(f1.relative_to(ROOT)))

    # fig 2: convergence ladder per KPI
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, k in zip(axes, ("cd_mean", "st_fft", "cl_amp")):
        for arm in ["A1", "A2", "A4", "A6"]:
            pts = sorted((r["D"], r[k]) for r in rows
                         if r["arm"] == arm and not r["arm"].endswith("rep"))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style[arm],
                        color=colors[arm], label=arm, ms=6)
        ax.set_xlabel("D")
        ax.set_ylabel(k)
        ax.set_title(f"{k} resolution ladder")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Round 2: KPI convergence ladder D = 10 / 14 / 20")
    f2 = figdir / "fig_r2_2_ladder.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=130)
    plt.close(fig)
    out.append(str(f2.relative_to(ROOT)))

    # fig 3: 2x2 factorial interaction map
    Ds = [d for d in RESOLUTIONS if all((a, d) in ok for a in FACTORIAL.values())]
    if Ds:
        fig, axes = plt.subplots(1, min(3, len(Ds)), figsize=(4.2 * min(3, len(Ds)), 4))
        axes = np.atleast_1d(axes)
        for ax, D in zip(axes, Ds):
            for k in ("cd_mean", "st_fft", "cl_amp"):
                cells = {kk: ok[(a, D)][k] for kk, a in FACTORIAL.items()}
                base = cells["MRT_on"]
                xs = {"MRT_on": 0, "MRT_off": 1}
                ax.plot([0, 1], [cells["MRT_on"], cells["MRT_off"]], "o-", color="C0",
                        label="MRT" if k == "cd_mean" else None)
                ax.plot([0, 1], [cells["BGK_on"], cells["BGK_off"]], "s--", color="C3",
                        label="BGK" if k == "cd_mean" else None)
                ax.annotate(k, (1, cells["MRT_off"]), fontsize=7)
                ax.annotate(k, (1, cells["BGK_off"]), fontsize=7)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["sponge ON", "sponge OFF"])
            ax.set_title(f"D={D}: 2x2 cells (A1/A4/A6/A2)")
            ax.grid(alpha=0.3)
        axes[0].legend(fontsize=8)
        fig.suptitle("Round 2: collision x sponge interaction (parallel lines = separable)")
        f3 = figdir / "fig_r2_3_factorial.png"
        fig.tight_layout()
        fig.savefig(f3, dpi=130)
        plt.close(fig)
        out.append(str(f3.relative_to(ROOT)))

    # fig 4: Cl histories at the finest resolution, last 4 shedding periods
    Dmax = max((d for (_, d) in ok), default=None)
    if Dmax:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        for arm in ["A1", "A2", "A4", "A6"]:
            tp = RUNS / f"{arm}_D{Dmax}" / "trace.npz"
            if not tp.exists():
                continue
            z = np.load(tp)
            fy, meta = z["fy"], json.loads(str(z["meta"]))
            u_in, D = meta["u_in"], meta["D"]
            cl = 2.0 * fy / (u_in * u_in * D)
            n = cl.size
            seg = cl[int(0.8 * n):]
            ax.plot(np.arange(seg.size), seg, color=colors[arm], label=arm, lw=1.0)
        ax.set_xlabel("step (last 20 % of the record)")
        ax.set_ylabel("Cl")
        ax.set_title(f"Cl histories at D={Dmax}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        f4 = figdir / "fig_r2_4_cl_traces.png"
        fig.tight_layout()
        fig.savefig(f4, dpi=130)
        plt.close(fig)
        out.append(str(f4.relative_to(ROOT)))

    # fig 5: vorticity fields at Dmax (physics sanity: shedding still present)
    if Dmax and (RUNS / f"A1_D{Dmax}" / "trace.npz").exists():
        fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
        for ax, arm in zip(axes, ["A1", "A2", "A4", "A6"]):
            tp = RUNS / f"{arm}_D{Dmax}" / "trace.npz"
            if not tp.exists():
                ax.set_axis_off()
                continue
            z = np.load(tp)
            v = np.clip(z["vort"], -0.5, 0.5)
            ax.imshow(v, cmap="RdBu_r", origin="lower", vmin=-0.5, vmax=0.5)
            ax.set_title(f"{arm} D={Dmax}")
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle("Round 2: final vorticity (Omega normalized), black square = bounce-back body")
        f5 = figdir / "fig_r2_5_fields.png"
        fig.tight_layout()
        fig.savefig(f5, dpi=130)
        plt.close(fig)
        out.append(str(f5.relative_to(ROOT)))
    return out


def main() -> int:
    ok, fail = load_runs()
    rows = rows_with_delta(ok)
    if not rows:
        print(json.dumps({"status": "no_runs"}))
        return 2
    rowmap = {(r["arm"], r["D"]): r for r in rows}
    summary = {
        "schema": "xval-method-selection/round2",
        "anchor": PREDECL["anchor"],
        "predeclared": "predeclared_round2.json",
        "arms": ARM_LABEL,
        "factorial_cells": FACTORIAL,
        "runs_present": {f"{a}_D{d}": "ok" for (a, d) in sorted(ok)},
        "failures": {
            f"{a}_D{d}": {"fail_stage": r.get("fail_stage"), "step": r.get("step"),
                          "total_steps_planned": r.get("total_steps_planned"),
                          "cl_max_abs": r.get("cl_max_abs")}
            for (a, d), r in sorted(fail.items())
        },
        "rows": rows,
        "gate_at_each_D": {
            str(D): {a: rowmap.get((a, D), {}).get("gate_pass") for a in OSS_STYLE_ARMS}
            for D in RESOLUTIONS
        },
        "ladder": ladder(rows),
        "factorial": [f for D in RESOLUTIONS for k in ("cd_mean", "st_fft", "cl_amp", "cl_rms")
                      if (f := factorial(ok, D, k)) is not None],
        "closure_H1": closure(rows),
        "determinism": determinism(ok, rows),
        "figures": figures(rows, ok, ROOT / "figures"),
    }
    (ROOT / "summary_round2.json").write_text(json.dumps(summary, indent=2) + "\n")
    csv_path = write_csv(rows)
    print(json.dumps({"rows": len(rows), "csv": str(csv_path),
                      "figures": summary["figures"],
                      "gate": summary["gate_at_each_D"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())