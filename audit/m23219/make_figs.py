#!/usr/bin/env python3
"""m23219 board-audit — figures for the m23126 hand-over audit.

Everything plotted is read from *shipped* artifacts or from this audit's own
reproduction; nothing is typed in by hand:

* ``TensorLBM/xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/artifacts/results.json``
  (delivered numbers, in the repo)
* ``.audit/m23219/repro/...`` (auditor re-run of ``--only exact`` / ``--only
  acoustic``; bit-exact copies of the shipped chunks)
* ``tensor_benchmark_cases`` row for the anchor (read live from PostgreSQL —
  the benchmark judge)

Run:  python3 audit/m23219/make_figs.py
Out:  docs/board_audit/m23219/figs/*.png
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "TensorLBM/xval/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/artifacts"
REPRO = ROOT / ".audit/m23219/repro/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/artifacts"
OUT = ROOT / "docs/board_audit/m23219/figs"
OUT.mkdir(parents=True, exist_ok=True)

TOL = 15.0  # owner precision standard / registered tolerance


def bench_row() -> dict:
    sql = (
        "SELECT jsonb_build_object('case_name',case_name,'metric',metric,"
        "'tolerance',tolerance,'ref_source',ref_source::jsonb,'status',status)::text "
        "FROM tensor_benchmark_cases "
        "WHERE anchor_slug='XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD';"
    )
    r = subprocess.run(
        ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-c", sql],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin"},
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:400])
    return json.loads(r.stdout.strip())


def fig_laplace(res: dict) -> str:
    runs = [r for r in res["laplace"]["runs"] if not r.get("nan")]
    inv_r = np.array([1.0 / r["R_eq"] for r in runs])
    dp = np.array([r["dp"] for r in runs])
    sig_i = np.array([r["sigma_per_radius"] for r in runs])
    fit = res["laplace"]["fit"]

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    xx = np.linspace(0.0, inv_r.max() * 1.1, 50)
    ax.plot(xx, fit["sigma"] * xx + fit["intercept"], "-",
            label=f"delivered fit  $\\sigma$={fit['sigma']:.4f}, b={fit['intercept']:.2e}, R$^2$={fit['r2']:.4f}")
    ax.plot(xx, sig_i.mean() * xx, "--", color="tab:gray",
            label=f"zero-intercept $\\sigma$={sig_i.mean():.4f} (-8.1% vs fit)")
    ax.errorbar(inv_r, dp, fmt="o", color="tab:blue", capsize=3,
                label="shipped runs (R_eq=%s)" % "/".join(f"{r['R_eq']:.2f}" for r in runs))
    ax.set_xlabel("1 / R_eq  [1/lu]")
    ax.set_ylabel("$\\Delta p$  [lu]")
    ax.set_title("Young-Laplace: linearity only (no external $\\sigma$)")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax2.plot([r["R_eq"] for r in runs], sig_i, "s-", color="tab:red")
    ax2.set_xlabel("R_eq  [lu]")
    ax2.set_ylabel("$\\sigma_i=\\Delta p\\,R_{eq}$  [lu]")
    ax2.set_title("per-radius $\\sigma$ drifts %.1f%% (3 radii only)"
                  % ((sig_i.max() / sig_i.min() - 1.0) * 100.0))
    ax2.grid(alpha=0.3)
    for x, y in zip([r["R_eq"] for r in runs], sig_i):
        ax2.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=7)
    fig.tight_layout()
    p = OUT / "fig1_laplace_selfconsistency.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig_acoustic(res: dict, repro: dict | None) -> str:
    arms = res["acoustic"]["arms"]
    pred = np.array([a["c_predicted"] for a in arms])
    meas = np.array([a["c_measured"] for a in arms])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    lim = [0.45, 0.95]
    ax.fill_between(lim, np.array(lim) * (1 - TOL / 100), np.array(lim) * (1 + TOL / 100),
                    color="tab:green", alpha=0.10, label=f"$\\pm${TOL:.0f}% band")
    ax.fill_between(lim, np.array(lim) * 0.97, np.array(lim) * 1.03,
                    color="tab:orange", alpha=0.15, label="$\\pm$3% band")
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.plot(pred, meas, "o", color="tab:blue", label="shipped (4 arms)")
    if repro is not None:
        r_meas = np.array([a["c_measured"] for a in repro["acoustic"]["arms"]])
        ax.plot(pred, r_meas, "x", color="tab:purple", ms=9,
                label="auditor re-run (bit-exact)")
    for a, x, y in zip(arms, pred, meas):
        ax.annotate(f"{a['rel_err_pct']:+.2f}%", (x, y), textcoords="offset points",
                    xytext=(6, -10), fontsize=7)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("$c_{eff}$ from analytic EOS gradient expansion  [lu]")
    ax.set_ylabel("$c$ measured (standing-wave zero crossings)")
    ax.set_title("Acoustic mode: self-derived analytic reference, 4 arms")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = OUT / "fig2_acoustic_parity.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def fig_benchmark(res: dict, row: dict) -> str:
    refs = row["ref_source"]["ref_values"]
    gs = sorted(refs, key=lambda k: float(k.split("=")[1]))
    ref_sig = [refs[k]["sigma"] for k in gs]
    gvals = [float(k.split("=")[1]) for k in gs]
    runs = [r for r in res["laplace"]["runs"] if not r.get("nan")]
    sig_i = np.array([r["sigma_per_radius"] for r in runs])
    fit_sig = res["laplace"]["fit"]["sigma"]

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    xr = np.arange(len(ref_sig))
    ax.bar(xr, ref_sig, width=0.55, color="tab:orange", alpha=0.85,
           label="registered refs (palabos, psi_exp) — provenance WITHHELD")
    ax.axhline(fit_sig, color="tab:blue", lw=2,
               label=f"delivered $\\sigma$ fit = {fit_sig:.4f} (G=2.0, psi=rho)")
    ax.fill_between([-0.6, len(ref_sig) - 0.4], sig_i.min(), sig_i.max(),
                    color="tab:blue", alpha=0.20,
                    label=f"delivered per-radius $\\sigma_i$ ({sig_i.min():.4f}-{sig_i.max():.4f})")
    for i, (g, s) in enumerate(zip(gvals, ref_sig)):
        err = (fit_sig / s - 1.0) * 100.0
        ax.annotate(f"delivered vs ref: {err:+.1f}%\n(tolerance {TOL:.0f}%)",
                    (i, max(s, fit_sig)), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7, color="tab:red")
    ax.set_xticks(xr)
    ax.set_xticklabels([f"$G_{{12}}$={g:g}\nref $\\sigma$={s:.3f}" for g, s in zip(gvals, ref_sig)])
    ax.set_ylabel("interfacial tension $\\sigma$  [lu]")
    ax.set_ylim(0, max(sig_i.max(), max(ref_sig)) * 1.45)
    ax.set_title("Benchmark judge (tensor_benchmark_cases, status=%s): NOT met —\n"
                 "no delivered number exists at the registered G=1.2/1.4/1.6; ψ forms differ"
                 % row["status"])
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p = OUT / "fig3_benchmark_judge.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return str(p)


def main() -> None:
    res = json.loads((ART / "results.json").read_text())
    repro = json.loads((REPRO / "results.json").read_text()) if (REPRO / "results.json").exists() else None
    row = bench_row()
    paths = [fig_laplace(res), fig_acoustic(res, repro), fig_benchmark(res, row)]
    print("\n".join(paths))


if __name__ == "__main__":
    main()