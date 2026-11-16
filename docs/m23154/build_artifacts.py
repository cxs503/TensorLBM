#!/usr/bin/env python
"""Aggregate the m23154 Taylor-Green sweep into tables, figures and the report payload.

Every number in ``RESULTS.md`` and in the platform report is injected here from
the per-run JSON records under ``docs/m23154/runs/`` -- nothing is transcribed
by hand.  Figure and report generation are idempotent; missing runs simply
produce a shorter table.

    python docs/m23154/build_artifacts.py            # tables + figures + report payload
    python docs/m23154/build_artifacts.py --report   # also write /tmp/m23154/report.json
"""

from __future__ import annotations

import argparse
import base64
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(HERE, "runs")
FIG = os.path.join(HERE, "figures")

KERNEL_ORDER = [
    "bgk_d3q19_ctl",
    "mrt_d3q27_ctl",
    "trt_d3q27_ctl",
    "rlbm_d3q27_ctl",
    "cascaded_d3q27_ctl",
    "kbc_d3q19",
    "kbc_d3q27",
    "cumulant_d3q27",
    "cumulant_geier_d3q27",
]


def load_runs() -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(RUNS, "*.json"))):
        try:
            with open(path, encoding="utf-8") as fh:
                rec = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print("skip", path, exc, file=sys.stderr)
            continue
        rec["source"] = os.path.basename(path)
        out.append(rec)
    return out


def sort_key(rec: dict):
    k = KERNEL_ORDER.index(rec["kernel"]) if rec["kernel"] in KERNEL_ORDER else 99
    return (rec.get("tier", "Z"), k, rec["n"], rec["tau"], rec["dtype"])


def table(runs: list[dict]) -> str:
    lines = [
        "| kernel | family | N | tau | dtype | nu_sim/nu_th | err_E % | R^2 | mass drift | G1 | G2 | G3 | G4 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(runs, key=sort_key):
        g = r.get("gates", {})
        lines.append(
            "| `{kernel}` | {family} | {n} | {tau:.2f} | {dtype} | {ratio:.5f} | {err:+.4f} | {r2:.6f} | "
            "{mass:.1e} | {g1} | {g2} | {g3} | {g4} |".format(
                kernel=r["kernel"],
                family=r["family"],
                n=r["n"],
                tau=r["tau"],
                dtype=r["dtype"],
                ratio=r["nu_sim"] / r["nu_theory"],
                err=r["err_pct"],
                r2=r["r2"],
                mass=r["mass_drift_rel"],
                g1="PASS" if g.get("G1_rate_le_3pct") else "FAIL",
                g2="PASS" if g.get("G2_r2_ge_0.999") else "FAIL",
                g3="PASS" if g.get("G3_mass_le_1e-12") else "FAIL",
                g4="PASS" if g.get("G4_positivity") else "FAIL",
            )
        )
    return "\n".join(lines)


def verdicts(runs: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in runs:
        d = out.setdefault(r["kernel"], dict(n_runs=0, fails=[], max_abs_err=0.0, min_r2=1.0, dtype=set()))
        d["n_runs"] += 1
        d["dtype"].add(r["dtype"])
        d["max_abs_err"] = max(d["max_abs_err"], abs(r["err_pct"]))
        d["min_r2"] = min(d["min_r2"], r["r2"])
        if not r.get("gates", {}).get("all_pass", False):
            d["fails"].append(f"{r['kernel']} N={r['n']} tau={r['tau']} err={r['err_pct']:+.3f}%")
    for d in out.values():
        d["dtype"] = sorted(d["dtype"])
        d["verdict"] = "REAL" if not d["fails"] else "WITHHELD"
    return out


def fig_decay(runs: list[dict]) -> str | None:
    sel = [r for r in runs if r["dtype"] == "float64" and r.get("tier") == "A" and abs(r["tau"] - 0.8) < 1e-9]
    if not sel:
        return None
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for r in sorted(sel, key=sort_key):
        t = np.asarray(r["series"]["t"], dtype=float)
        e = np.asarray(r["series"]["e"], dtype=float)
        good = e > 0
        ax.semilogy(t[good], e[good], lw=1.2, label=f"{r['kernel']} (err {r['err_pct']:+.2f}%)")
        ax.semilogy(t[good], r["e0"] * np.exp(-r["gamma_theory"] * t[good]), "k--", lw=0.8,
                    label="analytic gamma_E = 6 nu k^2" if r is sel[0] else None)
    ax.set_xlabel("step")
    ax.set_ylabel("kinetic energy E")
    ax.set_title(f"3-D Taylor-Green decay, N={sel[0]['n']}, tau={sel[0]['tau']:.2f}, float64")
    ax.legend(fontsize=6)
    path = os.path.join(FIG, "fig1_energy_decay.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_viscosity(runs: list[dict]) -> str | None:
    sel = [r for r in runs if r["dtype"] == "float64" and r.get("tier") == "A"]
    if not sel:
        return None
    os.makedirs(FIG, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for kern in KERNEL_ORDER:
        rows = sorted([r for r in sel if r["kernel"] == kern], key=lambda r: r["tau"])
        if not rows:
            continue
        tau = [r["tau"] for r in rows]
        nu = [r["nu_sim"] for r in rows]
        ax.plot(tau, nu, "o-", ms=4, lw=1.1, label=kern)
    taus = np.linspace(0.55, 1.55, 50)
    ax.plot(taus, (taus - 0.5) / 3.0, "k--", lw=1.0, label="theory nu = (tau-1/2)/3")
    ax.set_xlabel("tau")
    ax.set_ylabel("measured nu = gamma_E /(6 k^2)")
    ax.set_title("measured viscosity vs tau (3 % band)")
    ax.legend(fontsize=6)
    path = os.path.join(FIG, "fig2_viscosity_vs_tau.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    runs = load_runs()
    if not runs:
        print("no runs found under", RUNS)
        return 1
    verd = verdicts(runs)
    fig1, fig2 = fig_decay(runs), fig_viscosity(runs)
    payload = dict(n_runs=len(runs), verdicts=verd, figures=[p for p in (fig1, fig2) if p])
    with open(os.path.join(HERE, "results_table.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(summary=verd, runs=[{k: v for k, v in r.items() if k != "series"} for r in runs]),
                  fh, indent=1, sort_keys=True)
    md = ["# m23154 Taylor-Green audit — generated tables", "",
          f"runs aggregated: **{len(runs)}** (source: `docs/m23154/runs/*.json`)", "",
          "## Per-run gates", "", table(runs), "", "## Per-kernel verdict", "",
          "| kernel | runs | dtypes | max abs err_E % | min R^2 | verdict |", "|---|---|---|---|---|---|"]
    for kern, d in verd.items():
        md.append(f"| `{kern}` | {d['n_runs']} | {','.join(d['dtype'])} | {d['max_abs_err']:.4f} | "
                  f"{d['min_r2']:.6f} | **{d['verdict']}** |")
    if fig1:
        md += ["", f"![decay]({os.path.relpath(fig1, HERE)})"]
    if fig2:
        md += ["", f"![viscosity]({os.path.relpath(fig2, HERE)})"]
    with open(os.path.join(HERE, "RESULTS.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")
    print(json.dumps({k: v["verdict"] for k, v in verd.items()}, indent=1))

    if args.report:
        figs = []
        for p, cap in ((fig1, "3-D Taylor-Green kinetic-energy decay vs the analytic gamma_E = 6 nu k^2 (float64)."),
                       (fig2, "Measured viscosity nu = gamma_E/(6k^2) vs tau; dashed line is nu = (tau-1/2)/3.")):
            if p:
                figs.append(dict(name=os.path.basename(p), caption=cap,
                                 png_base64=base64.b64encode(open(p, "rb").read()).decode()))
        md_report = [f"## ③ 计算结果（程序化注入，共 {len(runs)} 个 run）", "", table(runs), "", "## 各核判据", ""]
        for kern, d in verd.items():
            md_report.append(f"* `{kern}`: **{d['verdict']}** — max |err_E| = {d['max_abs_err']:.4f}%, min R² = {d['min_r2']:.6f}")
        os.makedirs("/tmp/m23154", exist_ok=True)
        with open("/tmp/m23154/report_payload.json", "w", encoding="utf-8") as fh:
            json.dump(dict(figures=figs, results_md="\n".join(md_report), verdicts=verd), fh)
        print("wrote /tmp/m23154/report_payload.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
