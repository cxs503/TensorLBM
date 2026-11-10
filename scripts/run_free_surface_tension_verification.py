#!/usr/bin/env python
"""Acceptance driver: free-surface CSF surface tension (TensorLBM).

Usage
-----
Run one shard of one case (this is what the cluster jobs do)::

    python scripts/run_free_surface_tension_verification.py \\
        --case oscillation --index 0 --total 3

Merge everything, judge the criteria and draw the figures::

    python scripts/run_free_surface_tension_verification.py --case aggregate

Every shard writes ``artifacts/free_surface_tension/<case>__<index>.json`` so
that the acceptance numbers in the report are *generated*, never typed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.free_surface_tension import verification as V  # noqa: E402

OUT = Path("artifacts/free_surface_tension")

# --------------------------------------------------------------------------- #
# case tables (single source of truth for the report parameters)              #
# --------------------------------------------------------------------------- #
CASES = {
    "laplace2d": {
        "fn": lambda ip: V.young_laplace_2d(radius=(10, 14, 18)[ip], sigma=0.05, tau=0.9, width=3.0),
        "n": 3,
    },
    "laplace3d": {
        "fn": lambda ip: V.young_laplace_3d(radius=(8, 10, 12)[ip], sigma=0.05, tau=0.9, width=3.0),
        "n": 3,
    },
    "oscillation": {
        "fn": lambda ip: V.droplet_oscillation(
            sigma=(0.04, 0.08, 0.16)[ip],
            radius=12.0,
            L=64,
            eps=0.02,
            tau=0.9,
            width=2.0,
            periods=2.5,
            sharpen=False,
            compression=0.0,
            sample=2,
        ),
        "n": 3,
    },
    # Interface-thickness convergence of the measured period: the diffuse
    # interface softens the mode-2 restoring force by O((k w)^2), so the period
    # is extrapolated to w -> 0 (see the report).
    "oscillation_width": {
        "fn": lambda ip: V.droplet_oscillation(
            sigma=0.08,
            radius=12.0,
            L=64,
            eps=0.02,
            tau=0.9,
            width=(1.5, 2.0, 3.0, 4.5)[ip],
            periods=2.5,
            sharpen=False,
            compression=0.0,
            sample=2,
        ),
        "n": 4,
    },
    "marangoni": {
        "fn": lambda ip: V.marangoni_case(
            dsdx=(2e-6, 5e-6, 1e-5, 2e-5, 4e-5)[ip], Ly=32, Lx=4, steps=6000, sample=200
        ),
        "n": 5,
    },
    # Grid refinement at fixed *dimensionless* interface thickness w/R = 1/3 --
    # the only meaningful refinement for a diffuse-interface model (see report
    # section on the spurious-current scaling).
    "spurious": {
        "fn": lambda ip: V.spurious_case(
            radius=(12, 18, 26, 36)[ip], sigma=0.05, tau=0.9, steps=1500,
            width=(12, 18, 26, 36)[ip] / 3.0,
        ),
        "n": 4,
    },
    # Parametric sweep at fixed interface thickness (3 cells).  Kept because it
    # shows that the parasitic current is set by h/w, not by h/R.
    "spurious_fixedwidth": {
        "fn": lambda ip: V.spurious_case(radius=(8, 12, 18, 26)[ip], sigma=0.05, tau=0.9, steps=4000),
        "n": 4,
    },
    "mass": {
        # conservative flux-form transport (the conservative_clamp restores every
        # clipped overshoot); the semi-Lagrangian path used for the oscillation
        # case is *not* conservative and its drift is reported with that case.
        "fn": lambda ip: V.mass_conservation(steps=1000, sharpen=False),
        "n": 1,
    },
}

CRITERIA = {
    "laplace2d": ("dp_rel_err", 0.03, "2-D droplet  Delta p = sigma/R"),
    "laplace3d": ("dp_rel_err", 0.05, "3-D droplet  Delta p = 2 sigma/R"),
    "oscillation": ("period_rel_err", 0.05, "Rayleigh n=2 period"),
    "spurious": ("slope", 1.8, "|u_spurious| ~ h^2 exponent (w/R fixed)"),
    "mass": ("mass_loss", 0.001, "sum(phi) loss over 1000 steps"),
    "marangoni": ("r2", 0.95, "u_i vs Ma linear fit R^2"),
}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return float(obj)
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def run_shard(case: str, index: int, out: Path) -> Path:
    torch.set_num_threads(int(os.environ.get("TLBM_THREADS", "4")))
    spec = CASES[case]
    index = index % spec["n"]  # tolerate 0-based and 1-based shard ids
    t0 = time.time()
    res = spec["fn"](index)
    res["group"] = case  # shard group (several groups share one physical case)
    res["wall_time_s"] = time.time() - t0
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{case}__{index}.json"
    path.write_text(json.dumps(_jsonable(res), indent=2))
    print(f"[{case}:{index}] -> {path}  ({res['wall_time_s']:.1f}s)")
    print("   metrics:", json.dumps(_jsonable(res["metrics"]), indent=None))
    return path


# --------------------------------------------------------------------------- #
# aggregation                                                                 #
# --------------------------------------------------------------------------- #
def aggregate(out: Path, figdir: Path) -> dict:
    case2group = {
        "young_laplace_2d": "laplace2d",
        "young_laplace_3d": "laplace3d",
        "droplet_oscillation": "oscillation",
        "marangoni": "marangoni",
        "spurious": "spurious",
        "mass_conservation": "mass",
    }
    files = sorted(out.glob("*__*.json")) + sorted((out / "fixedwidth").glob("*__*.json"))
    data: dict[str, list[dict]] = {}
    for f in files:
        rec = json.loads(f.read_text())
        group = rec.get("group") or case2group.get(rec["case"], rec["case"])
        if f.parent.name == "fixedwidth":
            group = "spurious_fixedwidth"
        data.setdefault(group, []).append({"idx": int(f.stem.split("__")[1]), **rec})
    for k in data:
        data[k].sort(key=lambda r: r["idx"])

    summary: dict = {"cases": {}, "criteria": {}, "files": [str(f) for f in files]}

    # --- Laplace 2-D / 3-D
    for case in ("laplace2d", "laplace3d"):
        rows = data.get(case, [])
        if not rows:
            continue
        summary["cases"][case] = [
            {
                "R": r["params"]["radius"],
                "L": r["params"]["L"],
                "dp": r["metrics"]["dp"],
                "dp_ref": r["metrics"]["dp_ref"],
                "dp_rel_err": r["metrics"]["dp_rel_err"],
                "kappa_band": r["metrics"]["kappa_band"],
                "kappa_ref": r["metrics"]["kappa_ref"],
                "umax": r["metrics"]["umax"],
            }
            for r in rows
        ]
    # --- oscillation
    rows = data.get("oscillation", [])
    if rows:
        summary["cases"]["oscillation"] = [
            {
                "sigma": r["params"]["sigma"],
                "period": r["metrics"]["period"],
                "period_ref": r["metrics"]["period_ref"],
                "period_ref_2d": r["metrics"]["period_ref_2d"],
                "period_rel_err": r["metrics"]["period_rel_err"],
                "period_std": r["metrics"]["period_std"],
                "mass_loss": r["metrics"]["mass_loss"],
                "umax": r["metrics"]["umax"],
            }
            for r in rows
        ]
    # --- oscillation vs interface thickness (extrapolate w -> 0)
    rows = data.get("oscillation_width", [])
    if rows:
        ok = [r for r in rows if r["metrics"].get("period") is not None]
        w = [r["params"]["width"] for r in ok]
        T = [r["metrics"]["period"] for r in ok]
        # T(w) = T0 + c w^2  (diffuse-interface dispersion error is O((k w)^2))
        a, b, r2 = V.linear_fit([wi * wi for wi in w], T)
        summary["cases"]["oscillation_width"] = {
            "w": w,
            "period": T,
            "period_ref": rows[0]["metrics"]["period_ref"],
            "period_ref_2d": rows[0]["metrics"]["period_ref_2d"],
            "n_zero_crossings": [r["metrics"]["n_zero_crossings"] for r in rows],
            "t0_extrapolated": b,
            "slope_w2": a,
            "r2": r2,
            "rel_err_w0_vs_3d": b / rows[0]["metrics"]["period_ref"] - 1.0,
            "rel_err_w0_vs_2d": b / rows[0]["metrics"]["period_ref_2d"] - 1.0,
        }

    # --- Marangoni (linearity)
    rows = data.get("marangoni", [])
    if rows:
        ma = [r["params"]["Ma"] for r in rows]
        ui = [r["metrics"]["u_interface"] for r in rows]
        slope, intercept, r2 = V.linear_fit(ma, ui)
        summary["cases"]["marangoni"] = {
            "Ma": ma,
            "u_interface": ui,
            "u_ref": [r["metrics"]["u_ref"] for r in rows],
            "u_rel_err": [r["metrics"]["u_rel_err"] for r in rows],
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
        }
    # --- spurious convergence
    for case_key, block_key in (
        ("spurious", "spurious"),
        ("spurious_fixedwidth", "spurious_fixedwidth"),
    ):
        rows = data.get(case_key, [])
        if not rows:
            continue
        h = [r["metrics"]["h"] for r in rows]
        u = [r["metrics"]["umax"] for r in rows]
        slope, intercept, r2 = V.power_law_slope(h, u)
        summary["cases"][block_key] = {
            "h": h,
            "R": [r["params"]["radius"] for r in rows],
            "umax": u,
            "umax_early": [r["metrics"]["umax_early"] for r in rows],
            "dp_rel_err": [r["metrics"]["dp_rel_err"] for r in rows],
            "slope": slope,
            "intercept": intercept,
            "r2": r2,
        }
    # --- mass
    rows = data.get("mass", [])
    if rows:
        summary["cases"]["mass"] = {
            "mass0": rows[0]["metrics"]["mass0"],
            "mass1": rows[0]["metrics"]["mass1"],
            "mass_loss": rows[0]["metrics"]["mass_loss"],
            "steps": rows[0]["params"]["steps"],
        }

    # --- criteria verdicts
    verdicts = {}
    for key, case_name in (
        ("laplace2d", "laplace2d"),
        ("laplace3d", "laplace3d"),
        ("oscillation", "oscillation"),
        ("marangoni", "marangoni"),
        ("spurious", "spurious"),
        ("mass", "mass"),
    ):
        metric, thr, desc = CRITERIA[key]
        block = summary["cases"].get(case_name)
        if block is None:
            verdicts[key] = {"status": "MISSING", "criterion": desc}
            continue
        if key in ("laplace2d", "laplace3d", "oscillation"):
            vals = [abs(r[metric]) for r in block if r.get(metric) is not None]
            worst = max(vals)
            verdicts[key] = {
                "criterion": desc,
                "threshold": thr,
                "worst_abs": worst,
                "per_case": vals,
                "pass": worst < thr,
            }
        elif key == "marangoni":
            verdicts[key] = {
                "criterion": desc,
                "threshold": thr,
                "r2": block["r2"],
                "pass": block["r2"] > thr,
            }
        elif key == "spurious":
            verdicts[key] = {
                "criterion": desc,
                "threshold": thr,
                "slope": block["slope"],
                "r2": block["r2"],
                "pass": block["slope"] >= thr,
            }
        else:
            verdicts[key] = {
                "criterion": desc,
                "threshold": thr,
                "mass_loss": block["mass_loss"],
                "pass": block["mass_loss"] < thr,
            }
    summary["criteria"] = verdicts
    summary["all_pass"] = all(v.get("pass", False) for v in verdicts.values())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(verdicts, indent=2))
    print("ALL PASS:", summary["all_pass"])
    make_figures(data, summary, figdir)
    return summary


# --------------------------------------------------------------------------- #
# figures                                                                     #
# --------------------------------------------------------------------------- #
def make_figures(data: dict, summary: dict, figdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "font.size": 9})

    # 1 -- Laplace: dp history + dp vs 1/R
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for case, ax, title in (
        ("laplace2d", axes[0], "(a) 2-D cylinder: dp history"),
        ("laplace3d", axes[1], "(b) 3-D sphere: dp history"),
    ):
        rows = data.get(case, [])
        for r in rows:
            ser = r["series"]
            t = [s["step"] for s in ser]
            y = [s["dp"] / r["metrics"]["dp_ref"] for s in ser]
            ax.plot(t, y, "o-", ms=3, label=f"R={r['params']['radius']}")
        ax.axhline(1.0, color="k", ls="--", lw=1)
        ax.set_xlabel("lattice steps")
        ax.set_ylabel(r"$\Delta p\,/\,(\sigma\kappa_{ref})$")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    ax = axes[2]
    for case, lab, mk in (("laplace2d", "2-D cylinder", "o"), ("laplace3d", "3-D sphere", "s")):
        rows = data.get(case, [])
        if not rows:
            continue
        inv_r = [1.0 / r["params"]["radius"] for r in rows]
        dp = [r["metrics"]["dp"] for r in rows]
        ax.plot(inv_r, dp, mk, label=f"{lab} (measured)")
        if case == "laplace2d":
            ref = [0.05 * x for x in inv_r]
        else:
            ref = [2 * 0.05 * x for x in inv_r]
        ax.plot(inv_r, ref, "k--", lw=1, label=f"{lab} (Laplace)")
    ax.set_xlabel(r"$1/R$")
    ax.set_ylabel(r"$\Delta p$")
    ax.set_title("(c) Laplace law")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(figdir / "fig1_young_laplace.png")
    plt.close(fig)

    # 2 -- oscillation: signals + period vs sigma
    rows = data.get("oscillation", [])
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for r in rows:
            ser = r["series"]
            axes[0].plot(ser["t"], ser["D"], lw=1, label=f"sigma={r['params']['sigma']} (T={r['metrics']['period']:.0f})")
        axes[0].set_xlabel("lattice steps")
        axes[0].set_ylabel(r"$D=(Q_{xx}-Q_{yy})/(Q_{xx}+Q_{yy})$")
        axes[0].set_title("(a) mode-2 deformation history")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3)
        sig = [r["params"]["sigma"] for r in rows]
        axes[1].plot(sig, [r["metrics"]["period"] for r in rows], "o", label="measured")
        axes[1].plot(sig, [r["metrics"]["period_ref"] for r in rows], "k--", label="Rayleigh n=2")
        axes[1].set_xlabel(r"$\sigma$")
        axes[1].set_ylabel("period $T$")
        axes[1].set_title("(b) period vs surface tension")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig2_oscillation.png")
        plt.close(fig)

    block = summary["cases"].get("oscillation_width")
    if block:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        ax.plot([v * v for v in block["w"]], block["period"], "o", label="measured")
        ww = [0.0, max(block["w"]) ** 2 * 1.05]
        ax.plot(ww, [block["t0_extrapolated"] + block["slope_w2"] * v for v in ww], "b--", lw=1,
                label=f"fit $T_0$={block['t0_extrapolated']:.0f}")
        ax.axhline(block["period_ref"], color="k", ls=":", label="Rayleigh n=2 (3-D)")
        ax.axhline(block["period_ref_2d"], color="g", ls=":", label="2-D cylinder $6\\sigma$")
        ax.set_xlabel(r"$w^2$ (interface thickness$^2$)")
        ax.set_ylabel("period $T$")
        ax.set_title("period vs interface thickness (extrapolation)")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig2b_oscillation_width.png")
        plt.close(fig)

    # 3 -- Marangoni: profiles + linearity
    rows = data.get("marangoni", [])
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        cmap = plt.get_cmap("viridis")
        for i, r in enumerate(rows):
            axes[0].plot(
                r["series"]["ux_profile"],
                r["series"]["y"],
                color=cmap(i / max(1, len(rows) - 1)),
                lw=1.2,
                label=f"Ma={r['params']['Ma']:.3f}",
            )
        axes[0].set_xlabel(r"$u_x$")
        axes[0].set_ylabel("y")
        axes[0].set_title("(a) Marangoni-driven profile (steady)")
        axes[0].legend(fontsize=6)
        axes[0].grid(alpha=0.3)
        block = summary["cases"]["marangoni"]
        axes[1].plot(block["Ma"], block["u_interface"], "o", label="measured")
        axes[1].plot(block["Ma"], block["u_ref"], "k--", lw=1, label="analytic")
        axes[1].set_xlabel("Ma")
        axes[1].set_ylabel(r"$u_i$")
        axes[1].set_title(f"(b) linearity $R^2={block['r2']:.4f}$")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig3_marangoni.png")
        plt.close(fig)

    # 4 -- spurious currents
    for key, tag, title in (
        ("spurious", "fig4_spurious.png", "(w/R = 1/3 refinement)"),
        ("spurious_fixedwidth", "fig4b_spurious_fixedwidth.png", "(w = 3 cells sweep)"),
    ):
        block = summary["cases"].get(key)
        if not block:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for i, R in enumerate(block["R"]):
            rows = [r for r in data.get(key.replace("_fixedwidth", ""), []) if r["params"]["radius"] == R]
            if rows:
                ser = rows[0]["series"]
                axes[0].semilogy(ser["t"], ser["umax"], lw=1, label=f"R={R}")
        axes[0].set_xlabel("lattice steps")
        axes[0].set_ylabel(r"$\max|u|$")
        axes[0].set_title(f"(a) parasitic current history {title}")
        axes[0].legend(fontsize=7)
        axes[0].grid(alpha=0.3, which="both")
        h = block["h"]
        axes[1].loglog(h, block["umax"], "o", label="measured")
        u0 = block["umax"][-1]
        axes[1].loglog(h, [u0 * (x / h[-1]) ** 2 for x in h], "k--", lw=1, label=r"$\propto h^{2}$")
        axes[1].set_xlabel(r"$h = 1/R$")
        axes[1].set_ylabel(r"$\max|u|$")
        axes[1].set_title(f"(b) slope={block['slope']:.2f}")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3, which="both")
        fig.tight_layout()
        fig.savefig(figdir / tag)
        plt.close(fig)

    # 5 -- interface profile / field snapshot of the 2-D droplet
    rows = data.get("laplace2d", [])
    if rows:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
        import numpy as np

        ser = rows[0]["series"]
        axes[0].plot([s["step"] for s in ser], [s["mass"] for s in ser], "o-", ms=3)
        axes[0].set_xlabel("lattice steps")
        axes[0].set_ylabel(r"$\sum\phi$")
        axes[0].set_title("(a) interface mass, 2-D droplet")
        axes[0].grid(alpha=0.3)
        # pressure field reconstruction: radial pressure profile is not stored
        # in the JSON (only scalars), so plot kappa/dp consistency instead
        radii = [r["params"]["radius"] for r in rows]
        kerr = [r["metrics"]["kappa_band"] / r["metrics"]["kappa_ref"] - 1 for r in rows]
        dp_err = [r["metrics"]["dp_rel_err"] for r in rows]
        idx = np.arange(len(radii))
        axes[1].bar(idx - 0.2, [100 * v for v in kerr], width=0.4, label=r"$\kappa_{num}$ vs $1/R$")
        axes[1].bar(idx + 0.2, [100 * v for v in dp_err], width=0.4, label=r"$\Delta p$ vs $\sigma/R$")
        axes[1].set_xticks(idx)
        axes[1].set_xticklabels([f"R={r}" for r in radii])
        axes[1].axhline(0, color="k", lw=0.8)
        axes[1].set_ylabel("relative error [%]")
        axes[1].set_title("(b) curvature vs pressure-jump error")
        axes[1].legend(fontsize=7)
        axes[1].grid(alpha=0.3, axis="y")
        fig.tight_layout()
        fig.savefig(figdir / "fig5_laplace_consistency.png")
        plt.close(fig)

    # 6 -- mass conservation
    rows = data.get("mass", [])
    if rows:
        ser = rows[0]["series"]
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.plot(ser["t"], [v / ser["mass"][0] - 1 for v in ser["mass"]], "o-", ms=3)
        ax.set_xlabel("lattice steps")
        ax.set_ylabel(r"$\sum\phi/\sum\phi_0 - 1$")
        ax.set_title("mass conservation of the interface transport")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "fig6_mass_conservation.png")
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True, choices=[*CASES, "aggregate"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--total", type=int, default=1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--figdir", type=Path, default=Path("artifacts/free_surface_tension/figures"))
    ap.add_argument("--all-shards", action="store_true", help="run every shard of the case")
    args = ap.parse_args()

    if args.case == "aggregate":
        aggregate(args.out, args.figdir)
        return 0
    if args.all_shards:
        for i in range(CASES[args.case]["n"]):
            run_shard(args.case, i, args.out)
        return 0
    run_shard(args.case, args.index, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())