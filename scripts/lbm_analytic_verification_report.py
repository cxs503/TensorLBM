#!/usr/bin/env python3
"""Run the four analytic D3Q19 verification cases and dump JSON + figure.

Usage::

    PYTHONPATH=src python scripts/lbm_analytic_verification_report.py \
        --json artifacts/lbm_analytic_verification_r1.json \
        --figure docs/assets/lbm_analytic_verification_r1.png

The numbers written here are the data source of
``docs/lbm_analytic_verification_r1.md`` (no hand-copied values).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_lbm_verification import CASES  # noqa: E402


def _plot(metrics: dict, figure_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    # (a) shear-wave decay: LBM samples vs analytic exp(-nu k^2 t)
    a = metrics["A:shear_wave"]
    steps = np.array([s for s, _ in a["decay_samples"]], dtype=float)
    amps = np.array([abs(v) for _, v in a["decay_samples"]], dtype=float)
    k = 2 * np.pi / 64
    t_axis = np.linspace(0, steps.max(), 200)
    axes[0].semilogy(steps, amps, "o", ms=3.5, label="LBM amplitude")
    axes[0].semilogy(
        t_axis,
        abs(a["A_init"]) * np.exp(-a["nu_expected"] * k**2 * t_axis),
        "-",
        lw=1.2,
        label=r"analytic $A_0e^{-\nu k^2t}$, $\nu=1/6$",
    )
    axes[0].set_xlabel("step")
    axes[0].set_ylabel(r"$|u_y(x=0)|$")
    axes[0].set_title(f"(a) shear wave: $\\nu$ err {a['err_pct']:.2f}%")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    # (b) Couette + Poiseuille velocity profiles vs analytic
    b, c = metrics["B:couette"], metrics["C:poiseuille"]
    axes[1].plot(b["u_num"], b["y"], "o", ms=4, label="Couette LBM")
    axes[1].plot(b["u_exact"], b["y"], "-", lw=1.2, label="Couette analytic")
    axes[1].plot(
        np.array(c["u_num"]) * (b["u_top"] / max(c["u_num"])),
        c["y"],
        "s",
        ms=4,
        label="Poiseuille LBM (scaled)",
    )
    axes[1].plot(
        np.array(c["u_exact"]) * (b["u_top"] / max(c["u_exact"])),
        c["y"],
        "--",
        lw=1.2,
        label="Poiseuille analytic (scaled)",
    )
    axes[1].set_xlabel(r"$u_x$  (Poiseuille rescaled to $u_{top}$)")
    axes[1].set_ylabel("wall-normal index")
    axes[1].set_title(f"(b) profiles: Couette {b['err_pct']:.2f}%, Poiseuille {c['err_pct']:.2f}%")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    # (c) per-case error vs the 1% acceptance threshold
    labels = ["A: nu", "B: Couette", "C: Poiseuille", "D: Cf"]
    errs = [a["err_pct"], b["err_pct"], c["err_pct"], metrics["D:friction"]["err_pct"]]
    colours = ["#2a9d8f" if e < 1.0 else "#e76f51" for e in errs]
    bars = axes[2].bar(labels, errs, color=colours)
    axes[2].axhline(1.0, color="k", ls="--", lw=1.0, label="1% acceptance")
    axes[2].set_ylabel("max relative error [%]")
    axes[2].set_yscale("log")
    axes[2].set_title("(c) errors vs analytic benchmark")
    for bar, e in zip(bars, errs):
        axes[2].annotate(
            f"{e:.2e}%",
            (bar.get_x() + bar.get_width() / 2, e),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=110)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="artifacts/lbm_analytic_verification_r1.json")
    ap.add_argument("--figure", default="docs/assets/lbm_analytic_verification_r1.png")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    metrics = {}
    t0 = time.time()
    for name, runner in CASES:
        t_case = time.time()
        m = runner(args.device)
        m["wall_s"] = round(time.time() - t_case, 1)
        metrics[m["case"]] = m
        print(
            f"{m['case']:<16} err={m['err_pct']:.4e}%  passed={m['passed']}  {m['wall_s']}s",
            flush=True,
        )

    payload = {
        "suite": "tests/test_lbm_verification.py",
        "device": args.device,
        "threshold_pct": 1.0,
        "all_passed": all(m["passed"] for m in metrics.values()),
        "wall_s_total": round(time.time() - t0, 1),
        "cases": metrics,
    }
    out = ROOT / args.json
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {out}")

    if not args.no_figure:
        _plot(metrics, ROOT / args.figure)
        print(f"wrote {ROOT / args.figure}")
    return 0 if payload["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
