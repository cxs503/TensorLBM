#!/usr/bin/env python
"""Finite-``kH`` debt of the periodic-shear Taylor-Aris meter (kH -> 0 extrapolation).

Why this script exists
----------------------
``src/tensorlbm/taylor_dispersion.py`` (m7568) measures ``D_eff`` from the decay
rate of *one* axial Fourier mode: ``lambda = k^2 D_eff``.  That reading is only
``k^2 D_eff(k -> 0)`` in the long-wave limit.  Everything shipped by m7568 and
the entry ``benchmarks/verified/taylor_aris_dispersion`` (periodic-shear arm) was
measured at a single axial aspect ratio ``L/H = 8``, i.e. ``kH = 2*pi/8 = 0.785``
-- a value at which the finite-wave-number bias is *not* obviously negligible
(m12613, ``benchmarks/pending/scalar_dispersion_mach``, measured up to +7 % of
meter bias at ``kH = 0.84`` for the wall-bounded slit).

This script re-adjudicates the periodic-shear claim with the meter protocol
fixed: the same shipped D2Q5 kernel and the same modal meter, but the axial
aspect ratio is swept over ``L/H in {8, 16, 32, 64}`` (``kH = 0.785 ... 0.098``)
at fixed transverse resolution, and ``D_eff`` is extrapolated to ``kH -> 0``.
A second sweep refines the transverse resolution ``H in {16, 20, 26, 32}`` at
fixed ``Pe`` to measure what the ``kH -> 0`` residual actually is.

No handwritten physics kernel: every lattice operator comes from
``tensorlbm.passive_scalar`` / ``tensorlbm.taylor_dispersion``.

Usage
-----
    python run_kh.py --list
    python run_kh.py --case 3 --out-dir artifacts
    python run_kh.py --all --out-dir artifacts
    python run_kh.py --assemble --out-dir artifacts
    python run_kh.py --figs --out-dir artifacts
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3]
sys.path.insert(0, str(_REPO / "src"))

import torch  # noqa: E402

from tensorlbm import taylor_dispersion as td  # noqa: E402

D_LATTICE = 0.1  # molecular diffusivity (tau_D = 0.8 exactly)
KH_LEVELS = (8, 16, 32, 64)  # L/H aspect ratios -> kH = 2*pi/(L/H)
PE_MAIN = (2.0, 5.0, 8.0, 12.0)
H_REFINE = (16, 20, 26, 32)
H_FIXED = 16
GROWTH = 2.0  # target total modal decay factor exp(-GROWTH)
MIN_STEPS = 6000
MAX_STEPS = 12000  # keep the fit window >= 3 transverse relaxation times


def _steps(rate_theory: float) -> int:
    return int(min(MAX_STEPS, max(MIN_STEPS, math.ceil(GROWTH / max(rate_theory, 1e-9)))))


def _case_specs() -> list[dict]:
    specs: list[dict] = []
    for pe in PE_MAIN:
        for lh in KH_LEVELS:
            specs.append(
                {
                    "tag": f"pe{pe:g}_LH{lh}",
                    "kind": "kH_sweep",
                    "pe": pe,
                    "H": H_FIXED,
                    "nx": H_FIXED * lh,
                    "LH": lh,
                }
            )
    for h in H_REFINE:
        if h == H_FIXED:
            continue  # already in the kH sweep (pe8_LH32)
        specs.append(
            {
                "tag": f"pe8_LH32_H{h}",
                "kind": "H_refine",
                "pe": 8.0,
                "H": h,
                "nx": h * 32,
                "LH": 32,
            }
        )
    # Ma separation at fixed Pe, fixed H, fixed kH: scale D and U together
    # (Pe = U H / D invariant, Ma = U/c_s scales) -> breaks the Ma<->H collinearity.
    for s_d in (0.5, 2.0):
        specs.append(
            {
                "tag": f"pe8_LH32_Dscale{s_d:g}",
                "kind": "ma_separation",
                "pe": 8.0,
                "H": H_FIXED,
                "nx": H_FIXED * 32,
                "LH": 32,
                "d_scale": s_d,
            }
        )
    return specs


def _run(spec: dict) -> dict:
    d_lat = D_LATTICE * spec.get("d_scale", 1.0)
    cfg = td.ShearDispersionConfig(
        ny=spec["H"],
        nx=spec["nx"],
        D=d_lat,
        pe=spec["pe"],
        n_steps=0,  # set below from the closed-form rate
    )
    steps = _steps(cfg.k**2 * cfg.d_eff_theory)
    cfg = td.ShearDispersionConfig(
        ny=spec["H"],
        nx=spec["nx"],
        D=d_lat,
        pe=spec["pe"],
        n_steps=steps,
    )
    t0 = time.time()
    res = td.run_periodic_shear_dispersion(cfg)
    wall = time.time() - t0
    kH = cfg.k * spec["H"]
    return {
        **spec,
        "k": cfg.k,
        "kH": kH,
        "kH2": kH**2,
        "u_amp": cfg.u_amp,
        "ma": cfg.u_amp / math.sqrt(1.0 / 3.0),
        "ma2": (cfg.u_amp**2) / (1.0 / 3.0),
        "d_lattice": d_lat,
        "n_steps": steps,
        "n_cells": spec["H"] * spec["nx"],
        "d_eff_numeric": res.d_eff_numeric,
        "d_eff_theory": res.d_eff_theory,
        "ratio": res.d_eff_numeric / res.d_eff_theory,
        "rel_error": res.rel_error,
        "rate_fit": res.rate_fit,
        "rate_theory": cfg.k**2 * cfg.d_eff_theory,
        "transverse_rate": res.transverse_rate,
        "wall_seconds": wall,
    }


def _fit_quad(x: np.ndarray, y: np.ndarray) -> dict:
    """Least-squares quadratic through the origin-agnostic data, return intercept."""
    deg = min(2, len(x) - 1)
    coef = np.polyfit(x, y, deg)
    pred = np.polyval(coef, x)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "degree": deg,
        "coef": [float(c) for c in coef],
        "intercept": float(np.polyval(coef, 0.0)),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
        "max_abs_resid": float(np.abs(y - pred).max()),
    }


def assemble(out_dir: Path) -> dict:
    rows = []
    for f in sorted(out_dir.glob("case_*.json")):
        rows.append(json.loads(f.read_text()))

    kh = {}
    for row in rows:
        if row["kind"] != "kH_sweep":
            continue
        kh.setdefault(row["pe"], []).append(row)

    extrap = {}
    for pe, group in sorted(kh.items()):
        group.sort(key=lambda r: -r["kH"])  # kH descending -> kH^2 descending
        x = np.array([r["kH2"] for r in group])
        y = np.array([r["ratio"] for r in group])
        fit = _fit_quad(x, y)
        shipped = group[0]  # largest kH = the shipped L/H = 8 protocol
        extrap[f"{pe:g}"] = {
            "pe": pe,
            "fit": fit,
            "ratio_kH0": fit["intercept"],
            "rel_error_kH0_pct": 100.0 * (fit["intercept"] - 1.0),
            "rel_error_at_LH8_pct": 100.0 * shipped["rel_error"],
            "rel_error_at_finest_kH_pct": 100.0 * group[-1]["rel_error"],
            "cancellation_pp": abs(
                100.0 * (shipped["rel_error"] - (fit["intercept"] - 1.0))
            ),
            "points": [
                {
                    "LH": r["LH"],
                    "kH": r["kH"],
                    "kH2": r["kH2"],
                    "ratio": r["ratio"],
                    "rel_error_pct": 100.0 * r["rel_error"],
                    "n_steps": r["n_steps"],
                    "wall_seconds": r["wall_seconds"],
                }
                for r in group
            ],
        }

    # --- transverse refinement at fixed Pe, kH -> 0 (L/H = 32, kH = 0.196) -----
    refine = [r for r in rows if r["kind"] == "H_refine"]
    base = [r for r in rows if r["kind"] == "kH_sweep" and r["pe"] == 8.0 and r["LH"] == 32]
    refine = base + refine
    refine.sort(key=lambda r: r["H"])
    h_fit = None
    if len(refine) >= 3:
        h = np.array([1.0 / r["H"] ** 2 for r in refine])
        y = np.array([r["ratio"] for r in refine])
        coef = np.polyfit(h, y, 1)
        pred = np.polyval(coef, h)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        h_fit = {
            "x": "1/H^2",
            "slope": float(coef[0]),
            "intercept": float(coef[1]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "rel_error_extrapolated_pct": 100.0 * (float(coef[1]) - 1.0),
        }

    # --- lattice Mach law at fixed H and fixed kH (Pe sweep, L/H = 32) --------
    mach_rows = [
        r for r in rows if r["kind"] == "kH_sweep" and r["LH"] == 32
    ]
    mach_rows.sort(key=lambda r: r["ma2"])
    mach_fit = None
    if len(mach_rows) >= 3:
        x = np.array([r["ma2"] for r in mach_rows])
        y = np.array([r["ratio"] for r in mach_rows])
        coef = np.polyfit(x, y, 1)
        pred = np.polyval(coef, x)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        mach_fit = {
            "x": "Ma^2",
            "slope": float(coef[0]),
            "intercept": float(coef[1]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "rel_error_at_ma0_pct": 100.0 * (float(coef[1]) - 1.0),
            "points": [
                {"pe": r["pe"], "ma2": r["ma2"], "rel_error_pct": 100.0 * r["rel_error"]}
                for r in mach_rows
            ],
        }

    sep = [r for r in rows if r["kind"] == "ma_separation"]
    sep += [r for r in rows if r["kind"] == "kH_sweep" and r["pe"] == 8.0 and r["LH"] == 32]
    sep.sort(key=lambda r: r["ma2"])
    sep_fit = None
    if len(sep) >= 3:
        x = np.array([r["ma2"] for r in sep])
        y = np.array([r["ratio"] for r in sep])
        coef = np.polyfit(x, y, 1)
        pred = np.polyval(coef, x)
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        sep_fit = {
            "x": "Ma^2 at fixed Pe, H, kH (D and U scaled together)",
            "slope": float(coef[0]),
            "intercept": float(coef[1]),
            "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0,
            "rel_error_at_ma0_pct": 100.0 * (float(coef[1]) - 1.0),
            "points": [
                {"d_lattice": r.get("d_lattice", 0.1 * r.get("d_scale", 1.0)), "ma2": r["ma2"],
                 "rel_error_pct": 100.0 * r["rel_error"]} for r in sep
            ],
        }

    out = {
        "ma_separation_pe8_LH32": sep_fit,
        "protocol": {
            "module": "tensorlbm.taylor_dispersion.run_periodic_shear_dispersion",
            "D": D_LATTICE,
            "tau_d": 3 * D_LATTICE + 0.5,
            "dtype": "float64",
            "kh_levels_L_over_H": list(KH_LEVELS),
            "pe_main": list(PE_MAIN),
            "H_fixed": H_FIXED,
            "H_refine": list(H_REFINE),
            "steps_rule": "n_steps = max(6000, ceil(2/rate_theory))",
            "meter": "single axial Fourier mode, cross-section averaged amplitude, "
            "log-slope over second half of the record -> D_eff = rate/k^2",
            "closed_form": "D_eff/D = 1 + Pe^2/(8 pi^2),  Pe = U H / D (single-harmonic "
            "transverse shear u_x = U sin(2 pi y/H))",
        },
        "cases": rows,
        "kH_extrapolation": extrap,
        "transverse_refinement_pe8_LH32": {"fit": h_fit, "points": refine},
        "mach_law_pe_sweep_LH32": mach_fit,
    }
    (out_dir / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def capture_field(out_dir: Path) -> dict:
    """Capture one concentration field with the shipped operators (figure only).

    This is the *same* code path as the meter (library ``equilibrium_scalar`` /
    ``collide_scalar_bgk`` / ``stream_scalar``); it only keeps the final field so
    the shear-augmented transverse structure can be shown.  No physics is added.
    """
    from tensorlbm.passive_scalar import (  # noqa: PLC0415
        collide_scalar_bgk,
        equilibrium_scalar,
        macroscopic_scalar,
        stream_scalar,
    )

    H, LH, pe = 16, 8, 8.0
    cfg = td.ShearDispersionConfig(ny=H, nx=H * LH, D=D_LATTICE, pe=pe, n_steps=MIN_STEPS)
    ny, nx = cfg.ny, cfg.nx
    dtype = torch.float64
    k = cfg.k
    y = torch.arange(ny, dtype=dtype)
    u_profile = cfg.u_amp * torch.sin(2.0 * math.pi * y / ny)
    ux = u_profile.view(ny, 1).repeat(1, nx).contiguous()
    uy = torch.zeros_like(ux)
    x = torch.arange(nx, dtype=dtype)
    phi0 = 1.0 + cfg.amplitude * torch.cos(k * x).view(1, nx).repeat(ny, 1)
    g = equilibrium_scalar(phi0, ux, uy)
    for _ in range(MIN_STEPS):
        phi = macroscopic_scalar(g)
        g = collide_scalar_bgk(g, phi, ux, uy, cfg.tau_d)
        g = stream_scalar(g)
    phi = macroscopic_scalar(g)
    payload = {
        "H": H,
        "nx": nx,
        "pe": pe,
        "u_amp": cfg.u_amp,
        "phi": phi.tolist(),
        "u_profile": u_profile.tolist(),
        "mean_phi": float(phi.mean()),
    }
    (out_dir / "field_capture.json").write_text(json.dumps(payload))
    return payload


def figures(out_dir: Path, fig_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads((out_dir / "results.json").read_text())
    fig_dir.mkdir(parents=True, exist_ok=True)
    made: list[str] = []

    # ---- panel A: D_eff/D vs (kH)^2, extrapolated to kH -> 0 ------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(data["kH_extrapolation"])))
    for (pe, blk), c in zip(data["kH_extrapolation"].items(), colors):
        pts = blk["points"]
        x = [p["kH2"] for p in pts]
        y = [p["ratio"] for p in pts]
        ax.plot(x, y, "o", color=c, ms=5, label=f"Pe={pe}")
        xs = np.linspace(0.0, max(x) * 1.05, 50)
        ax.plot(xs, np.polyval(blk["fit"]["coef"], xs), "-", color=c, lw=1.0, alpha=0.85)
        ax.plot([0.0], [blk["ratio_kH0"]], "*", color=c, ms=11)
        ax.axhline(1.0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"$(kH)^2$,  $kH = 2\pi\,H/L$")
    ax.set_ylabel(r"$D_\mathrm{eff}^\mathrm{LBM}/D_\mathrm{eff}^\mathrm{closed}$")
    ax.set_title("kH -> 0 extrapolation (stars), H=16 fixed")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # ---- panel B: error vs aspect ratio, shipped protocol vs extrapolated ----
    ax = axes[1]
    pes = list(data["kH_extrapolation"].keys())
    xpos = np.arange(len(pes))
    ship = [data["kH_extrapolation"][p]["rel_error_at_LH8_pct"] for p in pes]
    ext = [data["kH_extrapolation"][p]["rel_error_kH0_pct"] for p in pes]
    ax.bar(xpos - 0.19, ship, 0.36, label="shipped protocol (L/H=8, kH=0.785)", color="#4c72b0")
    ax.bar(xpos + 0.19, ext, 0.36, label="kH->0 extrapolated", color="#dd8452")
    ax.axhspan(-3, 3, color="0.85", zorder=0, label="+-3% gate")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"Pe={p}" for p in pes])
    ax.set_ylabel("relative error of $D_\\mathrm{eff}$  [%]")
    ax.set_title("what the single-aspect-ratio meter reported vs its own kH->0 limit")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    p1 = fig_dir / "kh_extrapolation.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    made.append(str(p1))

    # ---- panel C: transverse refinement + Mach law ---------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    ax = axes[0]
    pts = data["transverse_refinement_pe8_LH32"]["points"]
    fit = data["transverse_refinement_pe8_LH32"]["fit"]
    hs = np.array([1.0 / p["H"] ** 2 for p in pts])
    ys = np.array([100.0 * p["rel_error"] for p in pts])
    ax.plot(hs, ys, "o", color="#4c72b0", label="Pe=8, L/H=32")
    xs = np.linspace(0, hs.max() * 1.1, 50)
    ax.plot(xs, 100.0 * (fit["intercept"] + fit["slope"] * xs - 1.0), "-", color="#4c72b0",
            lw=1.0, label=f"fit  R²={fit['r2']:.4f}")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"$1/H^2$  (transverse resolution)")
    ax.set_ylabel("error of $D_\\mathrm{eff}$ [%]")
    ax.set_title("transverse-discretisation bias (kH=0.196 fixed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    mf = data["mach_law_pe_sweep_LH32"]
    mp = mf["points"]
    ax.plot([p["ma2"] for p in mp], [p["rel_error_pct"] for p in mp], "o-", color="#c44e52")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel(r"$Ma^2 = U^2/c_s^2$")
    ax.set_ylabel("error of $D_\\mathrm{eff}$ [%]")
    ax.set_title(f"Mach dependence at fixed H, kH (R²={mf['r2']:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = fig_dir / "kh_bias_mechanism.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    made.append(str(p2))

    # ---- panel D: concentration field of the shipped-protocol case ----------
    fp = out_dir / "field_capture.json"
    if fp.exists():
        cap = json.loads(fp.read_text())
        phi = np.array(cap["phi"])
        u = np.array(cap["u_profile"])
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.6),
                                 gridspec_kw={"width_ratios": [3, 1]})
        im = axes[0].imshow(phi - cap["mean_phi"], aspect="auto", cmap="RdBu_r",
                            extent=[0, cap["nx"], 0, cap["H"]], origin="lower")
        axes[0].set_xlabel("x  [lattice units]")
        axes[0].set_ylabel("y  (transverse period H)")
        axes[0].set_title(f"scalar field, Pe={cap['pe']:g}, H={cap['H']}, L/H=8")
        fig.colorbar(im, ax=axes[0], label=r"$\phi - \langle\phi\rangle$")
        axes[1].plot(u, np.arange(cap["H"]) + 0.5, "o-", color="#55a868", ms=3)
        axes[1].set_xlabel(r"$u_x(y)=U\sin(2\pi y/H)$")
        axes[1].set_ylabel("y")
        axes[1].set_title("imposed shear")
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        p3 = fig_dir / "shear_field.png"
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        made.append(str(p3))

    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case", type=int, default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--figs", action="store_true")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out-dir", default=str(_HERE.parent / "artifacts"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)

    specs = _case_specs()
    if args.list:
        for i, s in enumerate(specs):
            print(i, s)
        return
    if args.case is not None:
        spec = specs[args.case]
        row = _run(spec)
        (out_dir / f"case_{args.case:03d}.json").write_text(json.dumps(row, indent=2))
        print(json.dumps(row, indent=2))
        return
    if args.all:
        for i, spec in enumerate(specs):
            target = out_dir / f"case_{i:03d}.json"
            if target.exists():
                continue
            t0 = time.time()
            row = _run(spec)
            target.write_text(json.dumps(row, indent=2))
            print(f"[{i+1}/{len(specs)}] {spec['tag']}: rel_error={100*row['rel_error']:+.3f}% "
                  f"({time.time()-t0:.1f}s)", flush=True)
    if args.capture:
        capture_field(out_dir)
        print("field captured")
    if args.assemble:
        out = assemble(out_dir)
        print(json.dumps({k: out[k] for k in ("kH_extrapolation",)}, indent=2, ensure_ascii=False)[:4000])
        print(json.dumps(out["transverse_refinement_pe8_LH32"]["fit"], indent=2))
        print(json.dumps(out["mach_law_pe_sweep_LH32"], indent=2))
    if args.figs:
        for p in figures(out_dir, _HERE.parent / "figures"):
            print("figure:", p)


if __name__ == "__main__":
    main()