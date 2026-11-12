#!/usr/bin/env python3
"""Entry point for the 3-D three-way coupling prototype (GAP-LBM-T5-ICE-COUPLING).

Writes, under ``runs/three_way_coupled3d/`` (and a tracked copy under
``artifacts/three_way_coupled3d/``):

* ``results.json``   — every number of the run (fluid / structure / ice / gates /
  benchmark / histories), no hand transcription anywhere;
* ``fig_*.png``      — convergence, load histories and the benchmark got-vs-ref chart;
* ``benchmark.md``   — the benchmark table rendered straight from ``results.json``.

Two modes::

    # full staged three-way run (fluid -> interface -> structure -> DEM march)
    OMP_NUM_THREADS=8 PYTHONPATH=src python scripts/three_way_coupled3d_run.py \
        --n 64 --diameter 12 --steps 3000 --fsi 600

    # fluid-leg only, used for the periodic-box confinement study (D/N scan)
    OMP_NUM_THREADS=8 PYTHONPATH=src python scripts/three_way_coupled3d_run.py \
        --fluid-only --n 48 --diameter 8 --steps 2000 --tag n48_d8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.three_way.coupled3d import (  # noqa: E402
    Coupled3DConfig,
    Fluid3DIBM,
    clift_cd,
    run_coupled3d,
    schiller_naumann_cd,
)

OUT = ROOT / "runs" / "three_way_coupled3d"
MIRROR = ROOT / "artifacts" / "three_way_coupled3d"


def hasimoto_factor(radius: float, box: float) -> float:
    """Periodic-array drag enhancement ``1/(1 - 2.837 a/L + 4.19 (a/L)^3)``.

    Hasimoto (1959), *On the periodic fundamental solutions of the Stokes equations*,
    J. Fluid Mech. 5:317 — leading correction for a simple-cubic array of spheres.
    Declared as a Stokes-limit correction; used only to explain the residual, never to
    replace the raw measurement.
    """
    x = radius / box
    return 1.0 / max(1.0 - 2.837 * x + 4.19 * x ** 3, 1e-6)


def fluid_only(args) -> dict:
    n = (args.n, args.n, args.n)
    centre = (0.25 * args.n, 0.5 * args.n, 0.5 * args.n)
    flow = Fluid3DIBM(
        n=n, nu=args.nu, centre=centre, diameter=args.diameter,
        n_theta=args.n_theta, n_phi=args.n_phi, u_init=args.u,
    )
    volume = float(args.n ** 3)
    drive = 1.0e-5
    series = []
    drag_hist = []
    windows = []
    for step in range(args.steps):
        flow.step(drive)
        drag = float(flow.body_force()[0])
        u = flow.mean_velocity_u()
        drive = max(drag, 0.0) / volume * (args.u / max(u, 1e-12)) ** 0.5
        if step % 10 == 0:
            series.append([float(step), drag, u,
                           2.0 * drag / (max(u, 1e-30) ** 2 * flow.area)])
        if step >= args.steps - args.measure:
            drag_hist.append(drag)
    drag_mean = sum(drag_hist) / len(drag_hist)
    drag_std = float(torch.tensor(drag_hist, dtype=torch.float64).std())
    u_mean = series[-1][2]
    re = u_mean * args.diameter / args.nu
    cd = 2.0 * drag_mean / (u_mean ** 2 * flow.area)
    corr = hasimoto_factor(0.5 * args.diameter, float(args.n))
    return {
        "tag": args.tag or f"n{args.n}_d{args.diameter}",
        "n": args.n,
        "diameter": args.diameter,
        "d_over_n": args.diameter / args.n,
        "nu": args.nu,
        "u": u_mean,
        "re": re,
        "steps": args.steps,
        "drag_mean": drag_mean,
        "drag_std": drag_std,
        "cd_got": cd,
        "cd_ref_clift": clift_cd(re),
        "cd_ref_schiller_naumann": schiller_naumann_cd(re),
        "rel_err_pct": 100.0 * (cd - clift_cd(re)) / clift_cd(re),
        "hasimoto_factor": corr,
        "cd_confinement_corrected": cd / corr,
        "rel_err_corrected_pct": 100.0 * (cd / corr - clift_cd(re)) / clift_cd(re),
        "series": series,
    }


def make_figures(res: dict, out: Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    f = res["fluid"]
    conv = f.get("cd_convergence") or []
    if conv:
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        steps = [c[0] for c in conv]
        ax[0].plot(steps, [c[3] for c in conv], lw=1.2, label="LBM C_D")
        ax[0].axhline(f["cd_clift"], color="k", ls="--", label="Clift-Grace-Weber")
        ax[0].set_xlabel("lattice step")
        ax[0].set_ylabel("C_D")
        ax[0].set_title(f"sphere drag convergence (Re={f['re_measured']:.2f})")
        ax[0].legend()
        ax[0].grid(alpha=0.3)
        ax[1].plot(steps, [c[1] for c in conv], lw=1.2, color="tab:red")
        ax[1].set_xlabel("lattice step")
        ax[1].set_ylabel("drag (lattice units)")
        ax[1].set_title("momentum-exchange drag")
        ax[1].grid(alpha=0.3)
        p = out / "fig_fluid_convergence.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        figs.append(p.name)

    h = res.get("history") or {}
    if h.get("t"):
        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(h["t"], h["tip"], lw=1.2, label="boom tip deflection")
        ax[0].set_xlabel("coupling step")
        ax[0].set_ylabel("tip deflection (lattice)")
        ax[0].set_title("structure response (two-way)")
        ax[0].grid(alpha=0.3)
        ax2 = ax[0].twinx()
        ax2.plot(h["t"], h["f_fluid"], color="tab:orange", lw=0.9, alpha=0.7,
                 label="fluid load")
        ax2.plot(h["t"], h["f_ice"], color="tab:green", lw=0.9, alpha=0.7,
                 label="ice contact load")
        ax2.set_ylabel("load (lattice)")
        ax[0].legend(loc="upper left")
        ax2.legend(loc="lower right")
        ax[1].plot(h["t"], h["ice_x_min"], lw=1.2, label="min particle x")
        ax[1].axhline(res["ice"]["face_x"], color="k", ls="--", label="boom face x")
        ax[1].set_xlabel("coupling step")
        ax[1].set_ylabel("x (lattice)")
        ax[1].set_title("ice approach / contact")
        ax[1].legend()
        ax[1].grid(alpha=0.3)
        p = out / "fig_three_way_histories.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        figs.append(p.name)

    rows = res.get("benchmark", {}).get("rows", [])
    if rows:
        labels = [r["quantity"] for r in rows]
        errs = [r["rel_err_pct"] for r in rows]
        fig, ax = plt.subplots(figsize=(8, 0.7 * len(rows) + 1.6))
        colors = ["tab:green" if r["verdict"] == "PASS" else "tab:red" for r in rows]
        ax.barh(range(len(rows)), errs, color=colors)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("relative error vs reference (%)")
        ax.set_title("benchmark: got vs reference (log scale)")
        ax.set_xscale("symlog", linthresh=1e-6)
        ax.axvline(0.0, color="k", lw=0.8)
        for i, r in enumerate(rows):
            ax.text(errs[i], i, f"  tol {r['tol_pct']}%", va="center", fontsize=7)
        p = out / "fig_benchmark.png"
        fig.tight_layout()
        fig.savefig(p, dpi=140)
        plt.close(fig)
        figs.append(p.name)
    return figs


def emit_benchmark_md(res: dict, out: Path) -> Path:
    lines = [
        "<!-- GENERATED by scripts/three_way_coupled3d_run.py — do not edit -->",
        "",
        "# GAP-LBM-T5-ICE-COUPLING — round-2 benchmark (ref vs got)",
        "",
        f"* fluid grid `{res['config']['n']}`, D = {res['config']['diameter']}, "
        f"nu = {res['config']['nu']}, measured Re = {res['fluid']['re_measured']:.3f}",
        "",
        "| quantity | reference | ref | got | rel err % | tol % | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in res["benchmark"]["rows"]:
        lines.append(
            f"| {r['quantity']} | {r['ref_source']} | {r['ref']:.6g} | {r['got']:.6g} | "
            f"{r['rel_err_pct']:+.4g} | {r['tol_pct']:g} | {r['verdict']} |"
        )
    lines += ["", "## gates", "", "| gate | pass | criterion |", "|---|---|---|"]
    for k, g in res["gates"].items():
        lines.append(f"| {k} | {g['pass']} | {g['criterion']} |")
    path = out / "benchmark.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--diameter", type=float, default=12.0)
    ap.add_argument("--nu", type=float, default=0.03)
    ap.add_argument("--u", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=0, help="fluid-stage steps (default: from phases)")
    ap.add_argument("--measure", type=int, default=1000)
    ap.add_argument("--fsi", type=int, default=400)
    ap.add_argument("--n-ice", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--n-theta", type=int, default=6)
    ap.add_argument("--n-phi", type=int, default=12)
    ap.add_argument("--fluid-only", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)

    if args.fluid_only:
        out = Path(args.out) if args.out else (OUT / "fluid_scan")
        out.mkdir(parents=True, exist_ok=True)
        payload = fluid_only(args)
        path = out / f"fluid_{payload['tag']}.json"
        path.write_text(json.dumps(payload, indent=1))
        print(f"[fluid-only] {payload['tag']}: CD={payload['cd_got']:.4f} "
              f"ref={payload['cd_ref_clift']:.4f} err={payload['rel_err_pct']:+.2f}% "
              f"corrected={payload['rel_err_corrected_pct']:+.2f}% -> {path}")
        return 0

    cfg = Coupled3DConfig(
        n=(args.n, args.n, args.n),
        diameter=args.diameter,
        nu=args.nu,
        u_target=args.u,
        centre=(0.25 * args.n, 0.5 * args.n, 0.5 * args.n),
        hex_origin=(0.0, 0.0, 0.25 * args.n),
        hex_extent=(0.5 * args.n, 34.0, 0.5 * args.n),
        hex_cells=(max(4, args.n // 8), 16, max(4, args.n // 8)),
        n_ice=args.n_ice,
        n_fsi=args.fsi,
        seed=args.seed,
        n_measure=args.measure,
    )
    if args.steps:
        cfg.n_warm = max(1, args.steps - args.measure)
        cfg.n_measure = args.measure
        cfg.n_drive_phases = 1
    res = run_coupled3d(cfg)
    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1))
    figs = make_figures(res, out)
    emit_benchmark_md(res, out)
    MIRROR.mkdir(parents=True, exist_ok=True)
    for p in [out / "results.json", out / "benchmark.md"] + [out / f for f in figs]:
        shutil.copy2(p, MIRROR / p.name)
    verdicts = {k: g["pass"] for k, g in res["gates"].items()}
    print(f"[run] gates: {verdicts}")
    print(f"[run] artefacts: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
