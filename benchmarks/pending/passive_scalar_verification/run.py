#!/usr/bin/env python
"""Passive-scalar transport verification surface — campaign node P73W2B-N05.

Four gates, one artifact, four figures:

``G1`` Taylor–Aris shear dispersion analytic anchor
      ``D_eff/D = 1 + Pe^2/210`` (plane slit, ``Pe = U H / D``) and
      ``D_eff/D = 1 + Pe^2/48`` (circular pipe, ``Pe = U a / D``) over
      ``Pe in [0.1, 100]``, checked by (a) the discrete Aris auxiliary problem
      on the lattice node set, (b) a time-marched D2Q9 scalar LBM.
``G2`` ``C = 0.5 erfc((x-x0)/sqrt(4Dt))`` L2 convergence order ``>= 1.9``
      on three grids refined with ``dt ~ dx^2``.
``G3`` ``Sc = nu/D in [0.7, 10]``: ``D_eff/D`` at fixed ``Pe`` must not depend
      on ``Sc`` (max spread ``< 2%``).
``G4`` discrete mass conservation ``|dM|/M < 1e-8`` for every run.
``G5`` ``artifacts/passive_scalar_verification.json``.

Usage
-----
    python benchmarks/pending/passive_scalar_verification/run.py            # formal
    python benchmarks/pending/passive_scalar_verification/run.py --smoke    # seconds
    python benchmarks/pending/passive_scalar_verification/run.py --no-figs

The formal run takes a few minutes on a CPU-only box (the dispersion runs
dominate: ``4 h^2/D`` steps each).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tensorlbm.passive_scalar_verification import (  # noqa: E402
    VerificationSpec,
    quick_spec,
    run_verification,
    taylor_aris_ratio,
)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _summary(report: dict) -> str:
    lines: list[str] = []
    crit = report["criteria"]
    lines.append("=" * 78)
    lines.append(f"passive-scalar verification surface  [{report['label']}]")
    lines.append(f"verdict: {report['verdict']}   runtime: {report['runtime_seconds']:.1f} s")
    lines.append("=" * 78)

    g1 = crit["G1"]
    lines.append(
        f"G1 Taylor-Aris  max|err| = {_fmt(g1['max_err_pct'], 4)} %  "
        f"(gate < {g1['threshold_pct']:g} %)  -> {'PASS' if g1['pass'] else 'FAIL'}"
    )
    for row in g1["anchor_rows"]:
        lines.append(
            f"    {row['geometry']:4s} Pe={row['pe_target']:7.2f} (discrete Aris, n={row['n_nodes']})"
            f"  D_eff/D={row['ratio_lattice']:.6f}  closed={row['peak_ratio']:.6f}"
            f"  err={100 * row['rel_err_vs_peak_ratio']:+.4f} %"
        )
    for row in g1["lbm_rows"]:
        lines.append(
            f"    slit Pe={row['pe_target']:7.2f} (LBM, ny={row['ny']}, steps={row['steps']})"
            f"  D_eff/D={row['ratio_sim']:.6f}  closed={row['ratio_closed_form']:.6f}"
            f"  err={100 * row['rel_err_vs_closed']:+.4f} %"
        )
    for row in g1.get("lbm_probe_rows", []):
        lines.append(
            f"    [probe, out of gate] slit Pe={row['pe_target']:g} u={row['u_mean']:.4f}"
            f"  D_eff/D={row['ratio_sim']:.6f} closed={row['ratio_closed_form']:.6f}"
            f"  err={100 * row['rel_err_vs_closed']:+.3f} %"
        )

    g2 = crit["G2"]
    lines.append(
        f"G2 erf diffusion  orders={['%.3f' % o for o in g2['orders']]}"
        f"  min={_fmt(g2['min_order'])} (gate >= {g2['threshold_order']:g})"
        f"  -> {'PASS' if g2['pass'] else 'FAIL'}"
    )
    for case, err in zip(g2["cases"], g2["errors"]):
        lines.append(
            f"    level {case['level']} nx={case['nx']:4d} dx={case['dx']:.4f}"
            f" steps={case['steps']:5d}  L2={err:.4e}"
        )

    g3 = crit["G3"]
    lines.append(
        f"G3 Schmidt independence  spread={_fmt(g3['ratio_spread_pct'], 3)} %"
        f"  max|dev vs closed|={_fmt(g3['max_dev_vs_closed_pct'], 3)} %"
        f"  (gate < {g3['threshold_pct']:g} %) -> {'PASS' if g3['pass'] else 'FAIL'}"
    )
    for row in g3["rows"]:
        lines.append(
            f"    Sc={row['schmidt']:5.2f} D={row['diffusivity']:.4f} u={row['u_mean']:+.4f}"
            f"  D_eff/D={row['ratio_sim']:.6f}  err={100 * row['rel_err_vs_closed']:+.3f} %"
        )
    ctrl = g3.get("resolution_control")
    if ctrl:
        ref = next(
            (
                r
                for r in g3["rows"]
                if r["schmidt"] == ctrl["schmidt"] and r["ny"] == ctrl["ny_reference"]
            ),
            None,
        )
        ref_txt = f"  (ny={ref['ny']} was {100 * ref['rel_err_vs_closed']:+.3f} %)" if ref else ""
        lines.append(
            f"    [resolution control] Sc={ctrl['schmidt']:g} ny={ctrl['ny']}"
            f"  D_eff/D={ctrl['ratio_sim']:.6f} err={100 * ctrl['rel_err_vs_closed']:+.3f} %"
            f"{ref_txt}"
        )

    g4 = crit["G4"]
    lines.append(
        f"G4 mass conservation  max drift = {g4['max_drift']:.3e}"
        f"  (gate < {g4['threshold']:g}) -> {'PASS' if g4['pass'] else 'FAIL'}"
    )
    lines.append(f"G5 artifact: {crit['G5']['path']}")
    lines.append("=" * 78)
    return "\n".join(lines)


def _make_figures(report: dict, fig_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    crit = report["criteria"]

    # ---- figure 1: erf diffusion front + convergence -----------------------
    data = report.get("_figure_data", {})
    if data.get("erf"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        colours = ["C0", "C1", "C2", "C3"]
        for i, case in enumerate(data["erf"]):
            x = np.asarray(case["x"])
            c = np.asarray(case["c_numeric"])
            exact = np.asarray(case["c_exact"])
            mask = np.asarray(case["mask"], dtype=bool)
            ax.plot(x[mask], c[mask], ".", ms=3, color=colours[i % 4], label=f"LBM nx={case['nx']}")
        ref = data["erf"][-1]
        x = np.asarray(ref["x"])
        exact = np.asarray(ref["c_exact"])
        mask = np.asarray(ref["mask"], dtype=bool)
        ax.plot(
            x[mask],
            exact[mask],
            "-",
            color="k",
            lw=1.2,
            label=r"exact $0.5\,\mathrm{erfc}(x/\sqrt{4Dt})$",
        )
        ax.set_xlabel("x")
        ax.set_ylabel("C")
        ax.set_title("erf diffusion front (D=0.1, t=200)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        ax = axes[1]
        dx = np.asarray(data["erf_dx"])
        err = np.asarray(data["erf_errors"])
        ax.loglog(dx, err, "o-", color="C3", label="LBM L2 error")
        # reference second-order line through the finest point
        ax.loglog(dx, err[-1] * (dx / dx[-1]) ** 2, "k--", lw=1.0, label=r"slope 2 reference")
        for k, p in enumerate(crit["G2"]["orders"]):
            ax.annotate(f"p={p:.2f}", (dx[k] * 1.15, err[k] * 1.05), fontsize=8)
        ax.set_xlabel(r"$\Delta x$")
        ax.set_ylabel(r"$L_2$ error")
        ax.set_title(f"convergence: min order {crit['G2']['min_order']:.2f} (gate 1.9)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")
        fig.tight_layout()
        path = fig_dir / "fig1_erf_diffusion_convergence.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # ---- figure 2: Taylor–Aris over Pe ------------------------------------
    g1 = crit["G1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    pe_curve = np.logspace(-1, 2, 200)
    ax.loglog(
        pe_curve,
        [taylor_aris_ratio(p, "slit") for p in pe_curve],
        "-",
        color="C0",
        lw=1.4,
        label=r"slit closed form $1+Pe^2/210$",
    )
    ax.loglog(
        pe_curve,
        [taylor_aris_ratio(p, "pipe") for p in pe_curve],
        "-",
        color="C2",
        lw=1.4,
        label=r"pipe closed form $1+Pe^2/48$",
    )
    for geometry, colour, marker in (("slit", "C0", "o"), ("pipe", "C2", "s")):
        rows = [r for r in g1["anchor_rows"] if r["geometry"] == geometry]
        if rows:
            ax.loglog(
                [r["pe_target"] for r in rows],
                [r["ratio_lattice"] for r in rows],
                marker,
                ms=5,
                mfc="none",
                color=colour,
                label=f"{geometry}: discrete Aris (n={rows[0]['n_nodes']})",
            )
    for row in g1["lbm_rows"]:
        ax.loglog(
            row["pe_target"],
            row["ratio_sim"],
            "k*",
            ms=11,
            label="LBM time march (slit)" if row is g1["lbm_rows"][0] else None,
        )
    for row in g1.get("lbm_probe_rows", []):
        ax.loglog(
            row["pe_target"],
            row["ratio_sim"],
            "x",
            color="C3",
            ms=9,
            label="LBM probe (lattice limit, out of gate)",
        )
    ax.set_xlabel("Pe")
    ax.set_ylabel(r"$D_\mathrm{eff}/D$")
    ax.set_title("Taylor–Aris dispersion: analytic anchor vs lattice solvers")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    for geometry, colour, marker in (("slit", "C0", "o"), ("pipe", "C2", "s")):
        rows = [r for r in g1["anchor_rows"] if r["geometry"] == geometry]
        if rows:
            ax.semilogx(
                [r["pe_target"] for r in rows],
                [100 * r["rel_err_vs_peak_ratio"] for r in rows],
                marker,
                ms=5,
                mfc="none",
                color=colour,
                label=f"{geometry}: discrete Aris",
            )
    if g1["lbm_rows"]:
        ax.semilogx(
            [r["pe_target"] for r in g1["lbm_rows"]],
            [100 * r["rel_err_vs_closed"] for r in g1["lbm_rows"]],
            "k*",
            ms=11,
            label="LBM time march (slit)",
        )
    for row in g1.get("lbm_probe_rows", []):
        ax.semilogx(
            row["pe_target"],
            100 * row["rel_err_vs_closed"],
            "x",
            color="C3",
            ms=9,
            label="LBM probe (out of gate)",
        )
    ax.axhspan(
        -g1["threshold_pct"],
        g1["threshold_pct"],
        color="0.85",
        zorder=0,
        label=f"±{g1['threshold_pct']:g}% gate",
    )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("Pe")
    ax.set_ylabel("relative error of $D_\\mathrm{eff}/D$  [%]")
    ax.set_title(f"error vs analytic anchor (max {g1['max_err_pct']:.3f} %)")
    ax.set_ylim(-12, 12)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = fig_dir / "fig2_taylor_aris_anchor.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)

    # ---- figure 3: Schmidt independence + mass conservation ---------------
    g3 = crit["G3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    sc = [r["schmidt"] for r in g3["rows"]]
    ratio = [r["ratio_sim"] for r in g3["rows"]]
    ax.semilogx(sc, ratio, "o-", color="C0", label="LBM $D_\\mathrm{eff}/D$")
    ax.axhline(
        g3["ratio_closed_form"],
        color="k",
        ls="--",
        lw=1.0,
        label=f"closed form {g3['ratio_closed_form']:.5f}",
    )
    ax.fill_between(
        [min(sc) * 0.8, max(sc) * 1.25],
        g3["ratio_closed_form"] * (1 - g3["threshold_pct"] / 100),
        g3["ratio_closed_form"] * (1 + g3["threshold_pct"] / 100),
        color="0.85",
        label=f"±{g3['threshold_pct']:g}% band",
    )
    ctrl = g3.get("resolution_control")
    if ctrl:
        ax.semilogx(
            [ctrl["schmidt"]],
            [ctrl["ratio_sim"]],
            "s",
            ms=8,
            mfc="none",
            color="C3",
            label=f"resolution control (ny={ctrl['ny']})",
        )
    ax.set_xlabel(f"Sc = nu / D   (Pe fixed at {g3['rows'][0]['pe_target']:g})")
    ax.set_ylabel(r"$D_\mathrm{eff}/D$")
    ax.set_title(f"Schmidt independence: spread {g3['ratio_spread_pct']:.2f} %")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    g4 = crit["G4"]
    labels = list(g4["drifts"].keys())
    values = [max(g4["drifts"][k], 1e-18) for k in labels]
    ax.bar(labels, values, color="C0")
    ax.axhline(g4["threshold"], color="C3", ls="--", label=f"gate {g4['threshold']:g}")
    ax.set_yscale("log")
    ax.set_ylabel(r"max $|\Delta M|/M$")
    ax.set_title(f"discrete mass conservation (max {g4['max_drift']:.2e})")
    ax.tick_params(axis="x", labelrotation=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = fig_dir / "fig3_schmidt_and_mass.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    paths.append(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="under-resolved smoke run (gates are not expected to pass)",
    )
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "artifacts" / "passive_scalar_verification.json")
    )
    parser.add_argument(
        "--fig-dir", default=str(REPO_ROOT / "artifacts" / "passive_scalar_verification_figs")
    )
    parser.add_argument("--no-figs", action="store_true")
    parser.add_argument(
        "--threads", type=int, default=0, help="torch CPU threads (0 = leave the default)"
    )
    parser.add_argument("--json-only", action="store_true", help="print the artifact path only")
    parser.add_argument(
        "--payload",
        default="",
        help="also write the full report (with raw profiles) to this path; "
        "use it to render figures later with --figs-from",
    )
    parser.add_argument(
        "--figs-from", default="", help="render figures only, from a payload written by --payload"
    )
    args = parser.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)

    if args.figs_from:
        payload = json.loads(Path(args.figs_from).read_text())
        figures = _make_figures(payload, Path(args.fig_dir))
        for fig in figures:
            print(f"figure: {fig}")
        return 0

    spec: VerificationSpec = quick_spec() if args.smoke else VerificationSpec()
    started = time.time()
    report = run_verification(spec, out_path=args.out, verbose=not args.json_only)
    if args.payload:
        payload_path = Path(args.payload)
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
        if not args.json_only:
            print(f"payload: {payload_path}")
    figures = [] if args.no_figs else _make_figures(report, Path(args.fig_dir))

    if not args.json_only:
        print(_summary(report))
        for fig in figures:
            print(f"figure: {fig}")
        print(f"artifact: {args.out}   (wall clock {time.time() - started:.1f} s)")

    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
