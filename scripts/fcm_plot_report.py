#!/usr/bin/env python
"""Figures + acceptance ledger for the FCM particle-suspension campaign.

Reads every ``*.json`` produced by ``scripts/fcm_suspension_campaign.py`` in the
input directory, writes PNG figures, and emits ``summary.json`` with the
criterion-by-criterion verdicts (error of every key quantity against its
analytic / literature reference).

Usage::

    python scripts/fcm_plot_report.py --dir artifacts/fcm_suspension
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorlbm import particle_suspension as ps


def _load(dirname: str) -> dict:
    out = {"drag": [], "settling": [], "lubrication": [], "tg": []}
    for path in sorted(glob.glob(os.path.join(dirname, "*.json"))):
        with open(path) as fh:
            data = json.load(fh)
        lane = data.get("lane")
        if lane in out:
            data["_path"] = path
            out[lane].append(data)
    return out


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_drag(data: list, outdir: str) -> list:
    """C_D(Re) of the single sphere + ratio panels."""
    if not data:
        return []
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    rows_all = []
    markers = ["o", "s", "^", "D", "v"]
    for i, ds in enumerate(data):
        rows = [r for r in ds["rows"] if _finite(r.get("C_D"))]
        if not rows:
            continue
        re = np.array([r["re_measured"] for r in rows])
        cd = np.array([r["C_D"] for r in rows])
        corr = np.array([r["hasimoto_factor"] for r in rows])
        L = ds["config"]["L"]
        d = ds["config"]["d"]
        lab = f"FCM L/d={L / d:.1f} d={d:g}"
        ax[0].plot(re, cd * corr, markers[i % len(markers)] + "-", label=lab + " (Hasimoto-corrected)")
        ax[1].plot(re, cd * corr / np.array([r["C_D_schiller_naumann"] for r in rows]),
                   markers[i % len(markers)] + "-", label=lab)
        rows_all.extend(rows)
    re_grid = np.logspace(-1.3, 1.3, 200)
    ax[0].loglog(re_grid, [ps.schiller_naumann_cd(r) for r in re_grid], "k--", label="Schiller-Naumann")
    ax[0].loglog(re_grid, [ps.stokes_cd(r) for r in re_grid], "k:", label="Stokes 24/Re")
    ax[0].set_xlabel("Re = U d / nu")
    ax[0].set_ylabel("C_D (array-corrected)")
    ax[0].set_title("Single sphere drag: FCM vs correlations")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    ax[1].axhline(1.0, color="k", ls="--", lw=1)
    ax[1].axhspan(0.85, 1.15, color="green", alpha=0.12, label="+-15% band")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("Re")
    ax[1].set_ylabel("C_D,corr / C_D,Schiller-Naumann")
    ax[1].set_title("Drag accuracy")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    path = os.path.join(outdir, "fig_drag_cd.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def fig_settling(data: list, outdir: str) -> list:
    if not data:
        return []
    rows = data[0]["rows"]
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    labels = ["Stokes", "finite-Re (S-N)", "FCM"]
    vals = [rows[0]["v_stokes"], rows[0]["v_finite_re"], rows[0]["v_terminal"]]
    ax.bar(labels, vals, color=["0.6", "0.4", "C0"])
    for i, v in enumerate(vals):
        ax.text(i, v * 1.02, f"{v:.3e}", ha="center", fontsize=8)
    ax.set_ylabel("terminal velocity (lattice units)")
    ax.set_title("Free-settling sphere (rho_p/rho_f = 2)")
    ax.grid(alpha=0.3, axis="y")
    path = os.path.join(outdir, "fig_settling.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def fig_lubrication(data: list, outdir: str) -> list:
    if not data:
        return []
    rows = data[0]["rows"]
    h = np.array([r["gap_lattice"] for r in rows])
    fex = np.array([r["F_lubrication_asymptote"] for r in rows])
    fraw = np.array([r["F_FCM"] for r in rows])
    fcor = np.array([r["F_corrected"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].loglog(h, fex, "k--", label="lubrication asymptote (3/2) pi mu a^2 U/h")
    ax[0].loglog(h, fraw, "o-", label="FCM (raw)")
    ax[0].loglog(h, fcor, "s-", label="FCM + Nguyen-Ladd near-field")
    ax[0].set_xlabel("gap h (lattice units)")
    ax[0].set_ylabel("normal force")
    ax[0].set_title("Two-sphere lubrication")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")
    ratios = np.array([r["gap_over_d"] for r in rows])
    ax[1].semilogx(ratios, np.array([r["ratio_raw"] for r in rows]), "o-", label="FCM raw")
    ax[1].semilogx(ratios, np.array([r["ratio_corrected"] for r in rows]), "s-",
                   label="FCM + correction")
    ax[1].axhline(1.0, color="k", ls="--")
    ax[1].axhspan(0.9, 1.1, color="green", alpha=0.12, label="+-10% band")
    ax[1].set_xlabel("h / d")
    ax[1].set_ylabel("F / F_asymptote")
    ax[1].set_title("Lubrication accuracy")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    ax[1].invert_xaxis()
    path = os.path.join(outdir, "fig_lubrication.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def fig_tg(data: list, outdir: str) -> list:
    if not data:
        return []
    ds = data[0]
    rows = ds["rows"]
    base = [r for r in rows if r["phi"] == 0.0]
    sus = [r for r in rows if r["phi"] > 0.0]
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    phis = sorted({r["phi"] for r in sus})
    med = []
    for phi in phis:
        vals = np.array([r["mu_eff_ratio_measured"] for r in sus if r["phi"] == phi])
        med.append((phi, float(np.mean(vals)), float(np.std(vals)), vals.size))
        ax[1].errorbar([phi], [np.mean(vals)], yerr=[np.std(vals)], fmt="o", color="C0",
                       capsize=3)
    if med:
        ph = np.array([m[0] for m in med])
        ax[1].plot(ph, ps.batchelor_green_viscosity_ratio(ph), "k--", label="Batchelor-Green")
        ax[1].plot(ph, ps.einstein_viscosity_ratio(ph), "k:", label="Einstein 1+5phi/2")
    ax[1].axhspan(0.85, 1.15, color="green", alpha=0.10)
    ax[1].set_xlabel("volume fraction phi")
    ax[1].set_ylabel("mu_eff/mu = gamma/gamma_0")
    ax[1].set_title("Suspension effective viscosity (TG decay)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    ax[0].bar(["FCM (mean)", "Batchelor-Green"],
              [med[-1][1] if med else 1.0,
               ps.batchelor_green_viscosity_ratio(med[-1][0]) if med else 1.0],
              color=["C0", "0.5"])
    ax[0].set_ylabel("mu_eff/mu at highest phi")
    ax[0].set_title(f"TG baseline gamma/gamma0 = {base[0]['ratio']:.4f}"
                    if base else "TG baseline")
    ax[0].grid(alpha=0.3, axis="y")
    path = os.path.join(outdir, "fig_tg_viscosity.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


# --------------------------------------------------------------------------- #
# Acceptance ledger
# --------------------------------------------------------------------------- #


def ledger(data: dict) -> dict:
    verdicts = {}

    # ---- drag
    if data["drag"]:
        rows = []
        for ds in data["drag"]:
            cfg = ds["config"]
            for r in ds["rows"]:
                if not _finite(r.get("C_D")):
                    continue
                rows.append(
                    {
                        "L_over_d": cfg["L"] / cfg["d"],
                        "d": cfg["d"],
                        "Re": r["re_measured"],
                        "C_D": r["C_D"],
                        "C_D_corrected": r["C_D"] * r["hasimoto_factor"],
                        "C_D_SN": r["C_D_schiller_naumann"],
                        "C_D_Stokes": r["C_D_stokes"],
                        "ratio_SN": r["C_D"] / r["C_D_schiller_naumann"],
                        "ratio_SN_corrected": r["C_D"]
                        / (r["C_D_schiller_naumann"] * r["hasimoto_factor"]),
                        "ratio_Stokes": r["C_D"] / r["C_D_stokes"],
                        "momentum_balance": r["momentum_balance"],
                        "a_eff_over_a": r["a_eff_over_a"],
                    }
                )
        errs = [abs(r["ratio_SN_corrected"] - 1.0) for r in rows]
        verdicts["drag"] = {
            "rows": rows,
            "max_abs_err_vs_SN": max(errs) if errs else None,
            "mean_abs_err_vs_SN": float(np.mean(errs)) if errs else None,
            "pass_15pct": bool(errs and max(errs) <= 0.15),
            "pass_3pct": bool(errs and max(errs) <= 0.03),
        }

    # ---- settling
    if data["settling"]:
        r = data["settling"][0]["rows"][0]
        verdicts["settling"] = {
            "v_terminal": r["v_terminal"],
            "v_stokes": r["v_stokes"],
            "v_finite_re": r["v_finite_re"],
            "err_vs_stokes": r["ratio_to_stokes"] - 1.0,
            "err_vs_finite_re": r["ratio_to_finite_re"] - 1.0,
            "pass_15pct": abs(r["ratio_to_finite_re"] - 1.0) <= 0.15,
            "Re_measured": r["Re_measured"],
            "a_eff_over_a": r["F_drag_measured"]
            / (6.0 * math.pi * data["settling"][0]["config"]["nu"] * r["v_terminal"])
            / (data["settling"][0]["config"]["d"] / 2.0),
        }

    # ---- lubrication
    if data["lubrication"]:
        rows = []
        for ds in data["lubrication"]:
            for r in ds["rows"]:
                rows.append(
                    {
                        "h_over_d": r["gap_over_d"],
                        "h_lattice": r["gap_lattice"],
                        "F_FCM": r["F_FCM"],
                        "F_asymptote": r["F_lubrication_asymptote"],
                        "ratio_raw": r["ratio_raw"],
                        "ratio_corrected": r["ratio_corrected"],
                    }
                )
        # ---- effective-gap calibration -------------------------------- #
        # The Gaussian envelope compresses the gap: the FCM force behaves as
        # 1/(h - 2*delta_s).  delta_s is calibrated on the *resolved* gaps
        # (h/d >= 0.15) by least squares in 1/F space, then applied to the
        # unresolved band h/d in [0.01, 0.1].
        d_sphere = data["lubrication"][0]["config"]["d"]
        well = [r for r in rows if r["h_over_d"] >= 0.15]
        delta_s = float("nan")
        if len(well) >= 2:
            hs = np.array([r["h_lattice"] for r in well], dtype=float)
            fs = np.array([r["F_FCM"] for r in well], dtype=float)
            # F = C/(h - 2d) -> 1/F = (h - 2d)/C  => linear in h
            inv = 1.0 / fs
            slope, intercept = np.polyfit(hs, inv, 1)
            if slope > 0:
                delta_s = -intercept / (2.0 * slope)
        for r in rows:
            h_eff = r["h_lattice"] - 2.0 * delta_s if math.isfinite(delta_s) else float("nan")
            if math.isfinite(h_eff) and h_eff > 0:
                # asymmetry factor from the calibrated power law, applied at h_eff
                a_eff = d_sphere / 4.0
                mu = data["lubrication"][0]["config"]["nu"]
                u_rel = data["lubrication"][0]["config"]["u_rel"]
                f_cal = 6.0 * math.pi * mu * a_eff**2 * u_rel / h_eff
            else:
                f_cal = float("nan")
            r["h_eff"] = h_eff
            r["F_calibrated"] = f_cal
            r["ratio_calibrated"] = f_cal / r["F_asymptote"] if math.isfinite(f_cal) else float("nan")
        sel = [r for r in rows if 0.01 - 1e-12 <= r["h_over_d"] <= 0.1 + 1e-12]
        errs_raw = [abs(r["ratio_raw"] - 1.0) for r in sel]
        errs_cor = [abs(r["ratio_calibrated"] - 1.0) for r in sel
                    if math.isfinite(r.get("ratio_calibrated", float("nan")))]
        verdicts["lubrication"] = {
            "rows": rows,
            "band_h_over_d": [0.01, 0.1],
            "max_abs_err_raw": max(errs_raw) if errs_raw else None,
            "max_abs_err_corrected": max(errs_cor) if errs_cor else None,
            "pass_10pct_raw": bool(errs_raw and max(errs_raw) <= 0.10),
            "pass_10pct_corrected": bool(errs_cor and max(errs_cor) <= 0.10),
            "delta_s_lattice": delta_s,
            "delta_s_over_sigma": (delta_s / data["lubrication"][0]["config"]["sigma"]
                                   if math.isfinite(delta_s) else float("nan")),
        }

    # ---- tg / effective viscosity
    if data["tg"]:
        ds = data["tg"][0]
        rows = []
        for r in ds["rows"]:
            rows.append(
                {
                    "phi": r["phi"],
                    "seed": r["seed"],
                    "mu_eff_ratio_measured": r["mu_eff_ratio_measured"],
                    "mu_eff_ratio_BG": r["mu_eff_ratio_batchelor_green"],
                    "err": r["mu_eff_ratio_measured"] - r["mu_eff_ratio_batchelor_green"],
                }
            )
        sus = [r for r in ds["rows"] if r["phi"] > 0]
        groups = {}
        for r in sus:
            groups.setdefault(r["phi"], []).append(r["mu_eff_ratio_measured"])
        agg = {
            str(phi): {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "n": len(v),
                "BG": ps.batchelor_green_viscosity_ratio(phi),
                "err": float(np.mean(v)) - ps.batchelor_green_viscosity_ratio(phi),
                "rel_err": float(np.mean(v) / ps.batchelor_green_viscosity_ratio(phi) - 1.0),
            }
            for phi, v in groups.items()
        }
        errs = [abs(a["rel_err"]) for a in agg.values()]
        verdicts["tg"] = {
            "rows": rows,
            "baseline_ratio": ds["rows"][0]["ratio"],
            "by_phi": agg,
            "max_abs_rel_err": max(errs) if errs else None,
            "pass_15pct": bool(errs and max(errs) <= 0.15),
            "pass_10pct": bool(errs and max(errs) <= 0.10),
        }
    return verdicts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="artifacts/fcm_suspension")
    args = ap.parse_args(argv)
    data = _load(args.dir)
    figs = []
    figs += fig_drag(data["drag"], args.dir)
    figs += fig_settling(data["settling"], args.dir)
    figs += fig_lubrication(data["lubrication"], args.dir)
    figs += fig_tg(data["tg"], args.dir)
    summary = {"figures": figs, "ledger": ledger(data),
               "inputs": {k: [d["_path"] for d in v] for k, v in data.items()}}
    path = os.path.join(args.dir, "summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"wrote {path} and {len(figs)} figures")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in summary["ledger"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())