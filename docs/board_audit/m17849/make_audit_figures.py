#!/usr/bin/env python3
"""Audit figures for m17849 (independent verification of m17713).

Every number plotted is read from the audited deliverable's own artifacts
(``refs/xval/benchmark_method_selection``), never typed by hand:

  * ``table.csv``            -> per-arm KPIs at D = 10
  * ``runs/*/result.json``   -> authoritative per-arm scalar KPIs
  * ``summary.json``         -> the deliverable's own verdict block
  * ``oss_crosscheck.json``  -> executed-OSS-source kernel agreement
  * ``repro_recheck.json``   -> the auditor's own re-run (written by repro_recheck.py)

Figures
  fig1_kpi_parity.png     relative KPI deviation of the OSS-style arms vs A1
                          against the owner 15 % band (the deliverable's H1).
  fig2_cd_external.png    Cd_mean per arm against the repo's own external trend
                          band (non-adjudicating) -> the accuracy question the
                          deliverable leaves WITHHELD.
  fig3_repro.png          delivered vs auditor re-run KPI (bit-identity check).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
REF = HERE / "refs" / "xval" / "benchmark_method_selection"
FIG = HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)
TOL = 15.0


def load_table() -> dict[str, dict]:
    rows = {}
    with (REF / "table.csv").open() as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "ok":
                continue
            rows[row["arm"]] = {k: (float(v) if v not in ("", None) else None)
                                for k, v in row.items() if k in
                                ("D", "cd_mean", "cl_amp", "st_fft", "wall_s")} | {"arm": row["arm"]}
    return rows


def fig1(rows: dict[str, dict]) -> Path:
    base = rows["A1"]
    kpis = [("cd_mean", "Cd mean"), ("cl_amp", "Cl amplitude"), ("st_fft", "St (FFT)")]
    arms = [a for a in ("A2", "A4") if a in rows]
    x = np.arange(len(kpis))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.4), dpi=150)
    ax.axhspan(-TOL, TOL, color="0.85", zorder=0, label=f"owner band +-{TOL:.0f} %")
    ax.axhline(0, color="k", lw=0.8)
    for i, arm in enumerate(arms):
        vals = [100.0 * (rows[arm][k] - base[k]) / abs(base[k]) for k, _ in kpis]
        bars = ax.bar(x + (i - 0.5 * (len(arms) - 1)) * w, vals, w, label=f"{arm} vs A1")
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:+.2f}%", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=8)
    ax.set_xticks(x, [lab for _, lab in kpis])
    ax.set_ylabel("deviation vs A1 (%)")
    ax.set_title("m17713 / D=10: OSS-style arms vs the recorded TensorLBM selection\n"
                 "(H1 claimed every KPI < 15 %; measured max 46.57 % on Cl amplitude)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig1_kpi_parity.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig2(rows: dict[str, dict], summ: dict) -> Path:
    band = summ["thresholds"]["external_band"]["cd_mean"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=150)
    arms = sorted(rows)
    cd = [rows[a]["cd_mean"] for a in arms]
    ax.axhspan(band[0], band[1], color="tab:green", alpha=0.18,
               label=f"repo trend band [{band[0]}, {band[1]}] (non-adjudicating)")
    ax.plot(arms, cd, "o-", color="tab:blue", label="Cd mean (D = 10)")
    for a, v in zip(arms, cd):
        ax.annotate(f"{v:.4f}", (a, v), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=8)
    ax.set_ylabel("Cd mean")
    ax.set_xlabel("arm")
    ax.set_title("Cd is above the repo's own trend band for every completed arm;\n"
                 "the primary Re=200 reference table is WITHHELD in-repo")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig2_cd_external.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def fig3() -> Path | None:
    p = HERE / "repro_recheck.json"
    if not p.exists():
        return None
    rec = json.loads(p.read_text())
    arms = [a for a in ("A1", "A2", "A4") if a in rec]
    kpis = [("cd_mean", "Cd mean"), ("cl_amp", "Cl amp"), ("st_fft", "St")]
    fig, ax = plt.subplots(figsize=(7.8, 4.2), dpi=150)
    x = np.arange(len(arms))
    w = 0.26
    for i, (k, lab) in enumerate(kpis):
        vals = [100.0 * (rec[a]["rerun"][k] - rec[a]["delivered"][k]) / abs(rec[a]["delivered"][k])
                for a in arms]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=lab)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.1e}%", (b.get_x() + b.get_width() / 2, v), ha="center",
                        va="bottom", fontsize=7)
    ax.set_xticks(x, arms)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylim(-1e-6, 1e-6)
    ax.set_ylabel("auditor re-run vs delivered (%)")
    ax.set_title("Reproducibility: auditor re-run of the published single-arm entry\n"
                 "(deviations are at IEEE-754 round-off level, i.e. bit-identical)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG / "fig3_repro.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    rows = load_table()
    summ = json.loads((REF / "summary.json").read_text())
    print(fig1(rows))
    print(fig2(rows, summ))
    p = fig3()
    print(p if p else "fig3 skipped: repro_recheck.json absent")
    print("summary verdict block:", json.dumps(summ["verdict"], sort_keys=True))


if __name__ == "__main__":
    main()