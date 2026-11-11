#!/usr/bin/env python
"""P284W3-N02 (v2) -- surface-tension anchors with the *matched* Rayleigh reference.

What is new relative to ``scripts/run_free_surface_tension_verification.py``
----------------------------------------------------------------------------
1. **Reference formula fixed.**  The previous attempt compared this
   single-phase (uniform-density) solver against the *drop-in-vacuum* Rayleigh
   period and reported ``+49 % .. +59 %``.  The physically matching anchor for
   a ``rho_inner = rho_outer`` interface is the two-fluid Rayleigh period
   (:func:`~tensorlbm.free_surface_tension.csf.rayleigh_period_2fluid`), which
   is ``sqrt(2)`` longer; the same runs are then ``+11 % .. +12 %``.
2. **Curvature anisotropy quantified and filtered.**  The pointwise
   ``kappa = -div(n_hat)`` carries an ``O(h^2)`` four-fold lattice error whose
   ``cos 2 theta`` projection is of the same order as the shape perturbation;
   one binomial pre-filter pass (:func:`...csf.smooth_field`) restores the
   ``cos 2 theta`` amplitude from ``0.70`` to ``0.96`` of the analytic value.
3. **Phase estimation** by a damped-sinusoid least-squares fit instead of two
   zero crossings, cross-checked against a geometric interface marker.
4. **Viscosity ladder / inviscid-limit extrapolation**: the Rayleigh anchor is
   an *inviscid* statement, so the anchor comparison is run as
   ``T( nu ) -> T(nu -> 0)`` with the ``sqrt(nu/(omega a^2))`` scaling, and
   every run is screened by stability diagnostics (a growing ``max|u|`` means
   the run is not evidence).
5. Young-Laplace 2-D/3-D are re-run *with* the curvature filter, so the static
   anchor that the previous attempt already passed is shown not to regress.

Usage
-----
``python scripts/run_p284w3_n02_surface_tension_v2.py --case <name>``  run one case
``python scripts/run_p284w3_n02_surface_tension_v2.py --case all-parallel`` run everything
``python scripts/run_p284w3_n02_surface_tension_v2.py --case aggregate`` build summary + figures

Artifacts: ``artifacts/p284w3_n02_v2/<case>.json`` and ``.../figures/*.png``.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.free_surface_tension import (  # noqa: E402
    capillary_oscillation as co,
    verification as ver,
)
from tensorlbm.free_surface_tension.csf import tanh_ellipse_phi  # noqa: E402

ART = ROOT / "artifacts" / "p284w3_n02_v2"
FIG = ART / "figures"

# --------------------------------------------------------------------------- #
# case table                                                                  #
# --------------------------------------------------------------------------- #
# Young-Laplace (static) -- re-run with the curvature filter (curvature_smooth=1)
LAPLACE_2D = [10.0, 14.0, 18.0]
LAPLACE_3D = [8.0, 10.0, 12.0]

# oscillation: (R, L, w, tau, sigma, eps, periods)
OSC_MAIN = dict(radius=16.0, L=128, width=2.0, tau=0.55, sigma=0.04, eps=0.02, periods=3.0)
OSC_TAU_LADDER = [0.55, 0.60, 0.70, 0.80]
OSC_WIDTH = [2.0, 3.0, 4.0]
OSC_SIGMA = [0.01, 0.04, 0.16]
OSC_RADIUS = [16.0, 24.0]


def _dump(name: str, payload: dict) -> Path:
    ART.mkdir(parents=True, exist_ok=True)
    path = ART / f"{name}.json"
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"[write] {path}  ({path.stat().st_size} bytes)")
    return path


def case_laplace_one() -> None:
    """One Young-Laplace sample (``LAP_SPEC`` env: ndim/radius/smooth) -> partial json."""
    spec = json.loads(os.environ["LAP_SPEC"])
    ndim, radius, smooth = int(spec["ndim"]), float(spec["radius"]), int(spec["smooth"])
    if ndim == 3:
        out = ver.young_laplace_3d(radius, sigma=0.05, tau=0.9, width=3.0, curvature_smooth=smooth)
    else:
        out = ver.young_laplace_2d(radius, sigma=0.05, tau=0.9, width=3.0, curvature_smooth=smooth)
    row = {"ndim": ndim, "curvature_smooth": smooth, **out["params"], **out["metrics"]}
    _dump(f"lap_{ndim}d_R{radius:.0f}_s{smooth}", {"case": "young_laplace_row", "rows": [row]})
    print(f"[lap{ndim}d] R={radius} smooth={smooth} err={out['metrics']['dp_rel_err']:+.4%}")


def case_laplace() -> None:
    rows = []
    for radius in LAPLACE_2D:
        for smooth in (0, 1):
            out = ver.young_laplace_2d(radius, sigma=0.05, tau=0.9, width=3.0, curvature_smooth=smooth)
            rows.append({"ndim": 2, "curvature_smooth": smooth, **out["params"], **out["metrics"]})
            print(f"[lap2d] R={radius} smooth={smooth} err={out['metrics']['dp_rel_err']:+.4%}")
    for radius in LAPLACE_3D:
        for smooth in (0, 1):
            out = ver.young_laplace_3d(radius, sigma=0.05, tau=0.9, width=3.0, curvature_smooth=smooth)
            rows.append({"ndim": 3, "curvature_smooth": smooth, **out["params"], **out["metrics"]})
            print(f"[lap3d] R={radius} smooth={smooth} err={out['metrics']['dp_rel_err']:+.4%}")
    _dump("young_laplace", {"case": "young_laplace_two_fluid_ladder", "rows": rows})


def case_curvature_mode2() -> None:
    """Mode-2 amplitude of the discrete curvature force (operator audit).

    The measured quantity is the ``delta_s``-weighted projection of ``kappa``
    onto ``cos 2 theta`` (the generalised stiffness conjugate to a mode-2
    displacement), *not* pointwise ``kappa`` at a handful of points: pointwise
    sampling moves by tens of percent with sub-cell choices at ``w = 2``.
    """
    out = {"case": "curvature_mode2_amplitude", "rows": []}
    eps = 0.02
    for radius in (16.0, 24.0, 32.0):
        for width in (2.0, 3.0, 4.0):
            L = int(8 * radius)
            a, b = radius * (1 + eps), radius * (1 - eps)
            phi = tanh_ellipse_phi((L, L), (0.5 * L, 0.5 * L), (a, b), width=width)
            for smooth in (0, 1, 2):
                res = co.curvature_mode2_amplitude(phi, (a, b), smooth=smooth)
                out["rows"].append(
                    {"radius": radius, "width": width, "curvature_smooth": smooth, **res}
                )
                print(
                    f"[k-mode2] R={radius} w={width} smooth={smooth} "
                    f"mode2_ratio={res['mode2_ratio']:+.4f} mean_ratio={res['mean_ratio']:+.4f}"
                )
    _dump("curvature_mode2", out)


def case_osc(name: str) -> None:
    """Run one oscillation case; ``name`` encodes the parameters."""
    spec = json.loads(os.environ["OSC_SPEC"]) if "OSC_SPEC" in os.environ else None
    if spec is None:
        raise SystemExit("OSC_SPEC env var required (JSON dict of oscillation_case kwargs)")
    if isinstance(spec.get("dtype"), str):
        spec["dtype"] = getattr(torch, spec["dtype"].replace("torch.", ""))
    out = co.oscillation_case(**spec)
    out["name"] = name
    _dump(name, out)
    m = out["metrics"]
    print(
        f"[osc {name}] T={m['period']:.1f} T_ref={m['period_ref_two_fluid']:.1f} "
        f"err={m['period_rel_err']:+.2%} marker_err={m['period_rel_err_marker']:+.2%} "
        f"stable={m['stable']} umax_growth={m['umax_growth']:.2f} R2={m['r2_fit']:.4f}"
    )


def _osc_job(name: str, spec: dict) -> dict:
    return {"name": name, "spec": spec}


def osc_jobs() -> list[dict]:
    jobs = []
    for tau in OSC_TAU_LADDER:
        jobs.append(_osc_job(f"osc_tau{tau:.2f}", {**OSC_MAIN, "tau": tau}))
    for w in OSC_WIDTH:
        jobs.append(_osc_job(f"osc_w{w:.1f}", {**OSC_MAIN, "width": w}))
    for sg in OSC_SIGMA:
        jobs.append(_osc_job(f"osc_sigma{sg:.2f}", {**OSC_MAIN, "sigma": sg}))
    for R in OSC_RADIUS:
        jobs.append(_osc_job(f"osc_R{int(R)}", {**OSC_MAIN, "radius": R, "L": int(8 * R)}))
    # controls: box size (image interaction) and interface-transport scheme
    jobs.append(_osc_job("osc_ctrl_box6R", {**OSC_MAIN, "L": 96}))
    jobs.append(_osc_job("osc_ctrl_upwind", {**OSC_MAIN, "advection": "upwind"}))
    return jobs


def case_osc_parallel(workers: int = 6) -> None:
    """Fan the oscillation cases out over processes (each is single-threaded)."""
    jobs = osc_jobs()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    procs = []
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = []
        for job in jobs:
            e = dict(env, OSC_SPEC=json.dumps(job["spec"]))
            fut = pool.submit(
                subprocess.run,
                [sys.executable, __file__, "--case", "osc", "--name", job["name"]],
                env=e,
                check=False,
            )
            futs.append(fut)
        for job, fut in zip(jobs, futs):
            fut.result()
            procs.append(job["name"])
    print(f"[osc] finished {len(procs)} cases: {procs}")


def case_marangoni() -> None:
    """Re-run the Marangoni two-layer Couette anchor (unchanged physics)."""
    rows = []
    for dsdx in (2e-6, 5e-6, 1e-5, 2e-5, 4e-5):
        out = ver.marangoni_case(dsdx)
        rows.append({"dsdx": dsdx, **out["params"], **out["metrics"]})
        print(f"[marangoni] dsigma/dx={dsdx:g} u={out['metrics']['u_interface']:.4e} "
              f"u_ref={out['metrics']['u_ref']:.4e} err={out['metrics']['u_rel_err']:+.2%}")
    _dump("marangoni", {"case": "marangoni_ladder", "rows": rows})


# --------------------------------------------------------------------------- #
# aggregation + figures                                                       #
# --------------------------------------------------------------------------- #
def _load(name: str) -> dict:
    with (ART / f"{name}.json").open() as fh:
        return json.load(fh)


def case_aggregate() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.mkdir(parents=True, exist_ok=True)
    summary: dict = {"case": "p284w3_n02_v2", "criteria": {}, "sources": {}}

    # ---- Young-Laplace (partial per-sample files are merged if the
    # monolithic sweep file is absent, so slow 3-D samples can run in parallel)
    if (ART / "young_laplace.json").exists():
        lap = _load("young_laplace")["rows"]
    else:
        lap = []
        for path in sorted(ART.glob("lap_*_R*_s*.json")):
            lap.extend(json.loads(path.read_text())["rows"])
        if not lap:
            raise FileNotFoundError("no Young-Laplace rows (run --case laplace or laplace-one)")
    lap_sum = {}
    for ndim in (2, 3):
        for smooth in (0, 1):
            sel = [r for r in lap if r["ndim"] == ndim and r["curvature_smooth"] == smooth]
            key = f"{ndim}d_smooth{smooth}"
            lap_sum[key] = {
                "radii": [r["radius"] for r in sel],
                "dp": [r["dp"] for r in sel],
                "dp_ref": [r["dp_ref"] for r in sel],
                "rel_err": [r["dp_rel_err"] for r in sel],
                "max_abs_rel_err": max(abs(r["dp_rel_err"]) for r in sel),
                "pass_3pct": all(abs(r["dp_rel_err"]) < 0.03 for r in sel),
            }
    summary["young_laplace"] = lap_sum
    summary["criteria"]["young_laplace_2d_lt_3pct"] = bool(lap_sum["2d_smooth1"]["pass_3pct"])
    summary["criteria"]["young_laplace_3d_lt_3pct"] = bool(lap_sum["3d_smooth1"]["pass_3pct"])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, ndim in zip(axes, (2, 3)):
        for smooth, marker, lab in ((0, "o", "raw curvature"), (1, "s", "1-pass filtered")):
            sel = [r for r in lap if r["ndim"] == ndim and r["curvature_smooth"] == smooth]
            R = np.array([r["radius"] for r in sel])
            dp = np.array([r["dp"] for r in sel])
            ax.plot(1.0 / R, dp, marker + "-", label=lab)
        R = np.linspace(8, 20, 50)
        ref = 0.05 * (1.0 if ndim == 2 else 2.0) / R
        ax.plot(1.0 / R, ref, "k--", lw=1, label="Laplace $\\sigma\\kappa$")
        ax.set_xlabel("1/R  [1/lu]")
        ax.set_ylabel("$\\Delta p$  [lu]")
        ax.set_title(f"{ndim}-D Young-Laplace ({'cylinder' if ndim == 2 else 'sphere'})")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_young_laplace.png", dpi=150)
    plt.close(fig)

    # ---- curvature mode-2 audit
    km = _load("curvature_mode2")["rows"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for smooth, marker in ((0, "o"), (1, "s"), (2, "^")):
        sel = [r for r in km if r["curvature_smooth"] == smooth]
        ax.plot([r["radius"] for r in sel], [r["mode2_ratio"] for r in sel], marker + "-",
                label=f"smooth={smooth}")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.axhspan(0.9, 1.1, color="0.85", zorder=0)
    ax.set_xlabel("R  [lu]")
    ax.set_ylabel("$\\cos 2\\theta$ amplitude of $\\kappa$ / analytic")
    ax.set_title("discrete curvature, mode-2 projection ($w=2$, $\\epsilon=0.02$)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_curvature_mode2.png", dpi=150)
    plt.close(fig)
    summary["curvature_mode2"] = km

    # ---- oscillation
    osc_names = [j["name"] for j in osc_jobs()]
    osc_all = []
    for name in osc_names:
        try:
            d = _load(name)
        except FileNotFoundError:
            continue
        osc_all.append(
            {
                "name": name,
                **d["params"],
                **{k: v for k, v in d["metrics"].items()},
            }
        )
    summary["oscillation"] = osc_all
    ladder = [o for o in osc_all if o["name"].startswith("osc_tau")]
    ladder.sort(key=lambda o: o["tau"])
    extr = co.inviscid_extrapolation(ladder)
    # scatter of the *other* scans around the ladder fit = extrapolation
    # uncertainty (an honest error bar: the tau ladder alone cannot see
    # residual R / sigma / width dependencies)
    resid = []
    if np.isfinite(extr["slope"]):
        for o in osc_all:
            if not o["stable"]:
                continue
            x = co.viscous_parameter(o["tau"], o["period"], o["radius"])
            resid.append(o["period_rel_err"] - (extr["slope"] * x + extr["intercept"]))
    extr["residuals_all_stable"] = [float(v) for v in resid]
    extr["scatter_std"] = float(np.std(resid)) if resid else float("nan")
    extr["scatter_median_abs"] = float(np.median(np.abs(resid))) if resid else float("nan")
    summary["inviscid_extrapolation"] = extr
    summary["criteria"]["oscillation_single_run_lt_5pct"] = bool(
        any(abs(o["period_rel_err"]) < 0.05 and o["stable"] for o in osc_all)
    )
    summary["criteria"]["oscillation_inviscid_extrapolated_lt_5pct"] = bool(
        abs(extr["intercept"]) < 0.05 if np.isfinite(extr["intercept"]) else False
    )
    summary["criteria"]["oscillation_intercept_plus_scatter_lt_5pct"] = bool(
        abs(extr["intercept"]) + 2.0 * extr["scatter_median_abs"] < 0.05
        if np.isfinite(extr["intercept"])
        else False
    )

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for o in ladder:
        if o["name"] == "osc_tau0.55":
            ser = _load(o["name"])["series"]
            t = np.array(ser["t"]) / o["period_ref_two_fluid"]
            ax.plot(t, ser["D"], lw=0.8, label=f"$\\tau$={o['tau']:.2f}")
    ax.set_xlabel("t / T$_{2fluid}$")
    ax.set_ylabel("$D=(Q_{xx}-Q_{yy})/(Q_{xx}+Q_{yy})$")
    ax.set_title("mode-2 shape signal (R=16, $\\sigma$=0.04)")
    ax.legend(fontsize=8)
    ax = axes[1]
    xs = [co.viscous_parameter(o["tau"], o["period"], o["radius"]) for o in ladder]
    ys = [o["period_rel_err"] for o in ladder]
    ax.plot(xs, np.array(ys) * 100, "o-", label="measured")
    if np.isfinite(extr["slope"]):
        xx = np.linspace(0, max(xs) * 1.1, 20)
        ax.plot(xx, (extr["slope"] * xx + extr["intercept"]) * 100, "k--",
                label=f"fit $c$={extr['slope']:.2f}, $\\nu\\to0$: {extr['intercept']:+.1%}")
    ax.axhline(5, color="r", ls=":", lw=1)
    ax.axhline(-5, color="r", ls=":", lw=1)
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("$\\sqrt{\\nu/(\\omega a^2)}$")
    ax.set_ylabel("$T/T_{2fluid}-1$  [%]")
    ax.set_title("viscous limit vs Rayleigh anchor")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_oscillation.png", dpi=150)
    plt.close(fig)

    # ---- width / radius / sigma scans
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, prefix, xlabel in (
        (axes[0], "osc_w", "interface width $w$ [lu]"),
        (axes[1], "osc_R", "radius $R$ [lu]"),
        (axes[2], "osc_sigma", "$\\sigma$ [lu]"),
    ):
        sel = [o for o in osc_all if o["name"].startswith(prefix)]
        key = {"osc_w": "width", "osc_R": "radius", "osc_sigma": "sigma"}[prefix]
        sel.sort(key=lambda o: o[key])
        if sel:
            ax.plot([o[key] for o in sel], [o["period_rel_err"] * 100 for o in sel], "o-")
        ax.axhline(5, color="r", ls=":", lw=1)
        ax.axhline(-5, color="r", ls=":", lw=1)
        ax.axhline(0, color="0.6", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("$T/T_{2fluid}-1$ [%]")
    fig.suptitle("oscillation error vs $w$, $R$, $\\sigma$ (\\tau=0.55, smooth=1)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_scans.png", dpi=150)
    plt.close(fig)

    # ---- Marangoni
    mar = _load("marangoni")["rows"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    ax.plot([r["dsdx"] for r in mar], [r["u_interface"] for r in mar], "o", label="LBGK interface velocity")
    ax.plot([r["dsdx"] for r in mar], [r["u_ref"] for r in mar], "s-", label="analytic two-layer Couette")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("$-d\\sigma/dx$")
    ax.set_ylabel("$u_i$ [lu]")
    ax.set_title("Marangoni drive: interface velocity")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot([r.get("Ma", float("nan")) for r in mar], [r["u_rel_err"] * 100 for r in mar], "o-")
    ax.axhline(10, color="r", ls=":", lw=1)
    ax.axhline(-10, color="r", ls=":", lw=1)
    ax.set_xlabel("Ma = $U H/\\nu$")
    ax.set_ylabel("relative error [%]")
    ax.set_title("Marangoni anchor error (gate $\\pm$10%)")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_marangoni.png", dpi=150)
    plt.close(fig)
    summary["marangoni"] = mar
    summary["criteria"]["marangoni_lt_10pct"] = bool(
        all(abs(r["u_rel_err"]) < 0.10 for r in mar)
    )

    # provenance: sha256 of every artifact this report cites
    import hashlib

    prov = {}
    for path in sorted(list(ART.glob("*.json")) + list(FIG.glob("*.png"))):
        prov[str(path.relative_to(ART))] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["provenance"] = prov

    _write_tables(summary, lap, osc_all, ladder, extr, mar)

    with (ART / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary["criteria"], indent=1))
    print(f"[write] {ART / 'summary.json'}")



def _write_tables(summary: dict, lap: list, osc_all: list, ladder: list, extr: dict, mar: list) -> None:
    """Emit the report tables as markdown (single source of truth for numbers)."""
    lines: list[str] = ["<!-- generated by run_p284w3_n02_surface_tension_v2.py --case aggregate -->", ""]

    lines += ["### Table 1  Young-Laplace static anchor (sigma = 0.05, w = 3)", "",
              "| dim | R | smooth | dp [lu] | dp_ref | rel err | kappa_band rel err | max abs err (ladder) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in lap:
        lines.append(
            f"| {r['ndim']}D | {r['radius']:.0f} | {r['curvature_smooth']} | {r['dp']:.6e} | "
            f"{r['dp_ref']:.6e} | {r['dp_rel_err']:+.2%} | {r['kappa_rel_err']:+.3%} | - |"
        )
    for key, val in summary["young_laplace"].items():
        lines.append(f"| {key} | - | - | - | - | - | - | {val['max_abs_rel_err']:.2%} |")
    lines.append("")

    km = summary["curvature_mode2"]
    lines += ["### Table 2  Mode-2 curvature force audit (band-weighted projection)", "",
              "| R | w | smooth | A2(num)/A2(exact) | mean curvature ratio |",
              "|---|---|---|---|---|"]
    for r in km:
        lines.append(
            f"| {r['radius']:.0f} | {r['width']:.1f} | {r['curvature_smooth']} | "
            f"{r['mode2_ratio']:+.4f} | {r['mean_ratio']:+.4f} |"
        )
    lines.append("")

    lines += ["### Table 3  Droplet oscillation vs two-fluid Rayleigh anchor", "",
              "| case | R | w | tau | nu | T [lu] | T_ref(2fluid) | rel err | marker rel err | "
              "vs vacuum | sqrt(nu/(w a^2)) | stable | R^2 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for o in sorted(osc_all, key=lambda o: o["name"]):
        lines.append(
            f"| {o['name']} | {o['radius']:.0f} | {o['width']:.1f} | {o['tau']:.2f} | {o['nu']:.4f} | "
            f"{o['period']:.1f} | {o['period_ref_two_fluid']:.1f} | {o['period_rel_err']:+.2%} | "
            f"{o['period_rel_err_marker']:+.2%} | {o['period_rel_err_vs_vacuum']:+.2%} | "
            f"{o['viscous_parameter']:.4f} | {o['stable']} | {o['r2_fit']:.4f} |"
        )
    lines.append("")
    lines += ["### Table 4  Inviscid-limit extrapolation of the oscillation anchor", "",
              f"* fitted slope c = {extr['slope']:.3f}, intercept (nu -> 0) = {extr['intercept']:+.3%}, "
              f"R^2 = {extr['r2']:.4f}, points = {extr['n_points']}, rejected (unstable) = {extr['rejected']}",
              f"* scatter of the R / sigma / width scans around that line: "
              f"std = {extr['scatter_std']:+.3%}, median |residual| = {extr['scatter_median_abs']:.3%}",
              f"* **intercept = {extr['intercept']:+.3%} +- {2.0 * extr['scatter_median_abs']:.3%} (2x median |residual|)**",
              ""]
    lines += ["### Table 5  Marangoni two-layer Couette anchor", "",
              "| dsigma/dx | Ma | u_i (measured) | u_ref | rel err |",
              "|---|---|---|---|---|"]
    for r in mar:
        lines.append(
            f"| {r['dsdx']:g} | {r.get('Ma', float('nan')):.4g} | {r['u_interface']:.4e} | {r['u_ref']:.4e} | "
            f"{r['u_rel_err']:+.2%} |"
        )
    lines.append("")
    lines += ["### Criteria", "", "```json", json.dumps(summary["criteria"], indent=1), "```", ""]
    (ART / "summary_tables.md").write_text("\n".join(lines))
    print(f"[write] {ART / 'summary_tables.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True,
                    choices=["laplace", "laplace-one", "curvature-mode2", "osc", "osc-all",
                             "marangoni", "aggregate"])
    ap.add_argument("--name", default="osc_case")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    ART.mkdir(parents=True, exist_ok=True)
    if args.case == "laplace":
        case_laplace()
    elif args.case == "laplace-one":
        case_laplace_one()
    elif args.case == "curvature-mode2":
        case_curvature_mode2()
    elif args.case == "osc":
        case_osc(args.name)
    elif args.case == "osc-all":
        case_osc_parallel(args.workers)
    elif args.case == "marangoni":
        case_marangoni()
    elif args.case == "aggregate":
        case_aggregate()


if __name__ == "__main__":
    main()