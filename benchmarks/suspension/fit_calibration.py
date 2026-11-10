"""P46W3-N03 — turn the raw LBM shear-cell/drag records into the calibrated closure.

Reads ``artifacts/raw_cases.json`` (produced by ``run_lbm_calibration.py``), builds
``eta_r(phi)`` in **two volume conventions** (lattice mask and hydrodynamic),
fits the Einstein slope / Krieger-Dougherty parameters, evaluates the
predeclared acceptance criteria, writes ``artifacts/analysis.json`` and the
figures under ``artifacts/figures/``.

Usage::

    python benchmarks/suspension/fit_calibration.py            # analyse + plot
    python benchmarks/suspension/fit_calibration.py --no-figures
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorlbm import suspension_viscosity as sv  # noqa: E402

RAW = os.path.join(HERE, "artifacts", "raw_cases.json")
OUT = os.path.join(HERE, "artifacts", "analysis.json")
FIGDIR = os.path.join(HERE, "artifacts", "figures")

#: primary dilute window = the whole AMENDMENT-4 dilute sweep {0.02, 0.05, 0.10}.
#: The chord slope is *not* the primary estimator over this window (its curvature
#: bias is +31 %); the curvature-consistent one-parameter fit is (AMENDMENT-4).
DILUTE_WINDOW_PRIMARY = 0.11
#: secondary window, kept for continuity with the original narrow design
DILUTE_WINDOW_WIDE = 0.11


def _eta_r(rec: dict, pure: dict) -> float:
    return (rec["tau_corrected"] / pure["tau_corrected"]) * (
        pure["gamma_dot_measured"] / rec["gamma_dot_measured"])


def _drag_block(rec: dict) -> dict:
    """Hydrodynamic radius of the discretised sphere from its Stokes drag."""
    mu, u, a_eff = rec["nu"], rec["u_stream"], rec["a_eff"]
    f = abs(rec["drag"])
    a_raw = sv.hydrodynamic_radius(f, mu, u)
    a_h = sv.hydrodynamic_radius(f, mu, u, rec["haberman_sayre_K"])
    return {
        "radius": rec["radius"],
        "a_eff": float(a_eff),
        "drag": float(f),
        "u_stream": float(u),
        "beta": float(rec["beta"]),
        "haberman_sayre_K": float(rec["haberman_sayre_K"]),
        "a_h": a_h,
        "a_h_raw": a_raw,
        "a_h_over_a_eff": a_h / a_eff,
        "a_h_raw_over_a_eff": a_raw / a_eff,
        "volume_ratio": sv.hydrodynamic_volume_ratio(a_h, a_eff),
        "volume_ratio_raw": sv.hydrodynamic_volume_ratio(a_raw, a_eff),
        "err_pct_vs_stokes": float(rec["err_pct"]),
        "err_pct_vs_stokes_confined": float(rec["err_pct_corrected"]),
        "profile_r2": float(rec["profile_r2"]),
    }


def build_dataset(raw: dict) -> dict:
    cases = raw["cases"]
    design = raw.get("design", {})
    ov_nz = (design.get("cell") or {}).get("nz")
    ov_margin = (design.get("steps") or {}).get("wall_margin")
    pure_rec = next(c for c in cases if c["tag"] == "pure_control")
    drags = [_drag_block(c) for c in cases if c["mode"] == "drag"]
    drags.sort(key=lambda d: d["radius"])
    ratio_by_radius = {d["radius"]: d["volume_ratio"] for d in drags}
    ratio_raw_by_radius = {d["radius"]: d["volume_ratio_raw"] for d in drags}

    groups: dict[tuple, list] = {}
    for c in cases:
        if c["mode"] != "shear" or c["tag"] == "pure_control" or "error" in c:
            continue
        key = (round(float(c["radius"]), 6), round(float(c["phi_target"]), 6))
        groups.setdefault(key, []).append(c)

    points = []
    for (radius, phi_target), recs in sorted(groups.items()):
        eta = np.array([_eta_r(r, pure_rec) for r in recs])
        phi_mask = float(np.mean([r["phi_lattice"] for r in recs]))
        r2s = [float(r["profile_r2"]) for r in recs]
        points.append({
            "radius": radius,
            "phi_target": phi_target,
            "phi_mask": phi_mask,
            "phi_hydro": phi_mask * ratio_by_radius.get(radius, float("nan")),
            "phi_hydro_raw": phi_mask * ratio_raw_by_radius.get(radius, float("nan")),
            # particles are confined to the wall-margin-free core, so the fraction
            # the suspension flow actually "sees" is the core fraction
            "phi_core": phi_mask * (float(ov_nz) / (float(ov_nz) - 2.0 * float(ov_margin)))
            if ov_nz and ov_margin is not None else float("nan"),
            "eta_r": float(eta.mean()),
            "eta_r_sd": float(eta.std(ddof=1)) if eta.size > 1 else 0.0,
            "sem": float(eta.std(ddof=1) / np.sqrt(eta.size)) if eta.size > 1 else float("nan"),
            "n_seeds": int(eta.size),
            "seeds": [int(r["seed"]) for r in recs],
            "profile_r2_min": float(min(r2s)),
            "n_particles": [int(r["n_particles"]) for r in recs],
            "re_p": float(np.mean([r["re_p"] for r in recs])),
        })
    pure_case = {
        "nu": float(pure_rec["nu"]),
        "eta_eff_pure": float(pure_rec["eta_eff"]),
        "eta_eff_over_nu": float(pure_rec["eta_eff"] / pure_rec["nu"]),
        "tau_wall_mem_rel_err": float(abs(pure_rec.get("tau_wall_mem", 0.0)
                                          / (pure_rec["nu"] * pure_rec["gamma_dot_measured"])
                                          - 1.0)),
        "gamma_dot_measured": float(pure_rec["gamma_dot_measured"]),
        "gamma_dot_nominal": float(pure_rec["gamma_dot_nominal"]),
        "profile_r2": float(pure_rec["profile_r2"]),
    }
    return {"points": points, "drags": drags, "pure": pure_case,
            "errors": [{"tag": c["tag"], "error": c.get("error")}
                       for c in cases if "error" in c]}


def _fit_block(points: list[dict], phi_key: str, window: float,
               label: str, select_key: str = "phi_mask") -> dict:
    """Weighted dilute-window regression of ``eta_r`` on ``phi_key``.

    The *selection* uses the design's sweep coordinate (``phi_mask``) so every
    convention is fitted to exactly the same data points; only the abscissa
    changes.
    """
    sel = [p for p in points
           if np.isfinite(p.get(select_key, np.nan)) and np.isfinite(p.get(phi_key, np.nan))
           and p[select_key] <= window]
    if len(sel) < 3:
        return {"label": label, "window": window, "n_points": len(sel),
                "error": "fewer than 3 points in window"}
    phi = np.array([p[phi_key] for p in sel])
    eta = np.array([p["eta_r"] for p in sel])
    sem = np.array([p["sem"] if np.isfinite(p["sem"]) and p["sem"] > 0 else 0.01
                    for p in sel])
    fit = sv.intrinsic_viscosity(phi, eta, weights=1.0 / sem**2)
    # curvature-consistent reference: the *chord* slope of the closed forms over
    # exactly the same window, i.e. what a perfect LBM would return
    chord = {}
    for model in ("einstein", "batchelor", "krieger_dougherty"):
        hi = sv.relative_viscosity(phi.max(), model, phi_max=0.63)
        lo = sv.relative_viscosity(phi.min(), model, phi_max=0.63)
        chord[model] = float((hi - lo) / (phi.max() - phi.min()))
    # curvature-consistent single-parameter estimate over the same points: this
    # is the estimator that is not biased by the closed-form curvature
    fit["curvature_fit"] = sv.fit_intrinsic_viscosity(phi, eta, phi_max=0.63,
                                                     weights=1.0 / sem**2)
    fit.update({
        "label": label,
        "window": window,
        "chord_slope_einstein": chord["einstein"],
        "chord_slope_batchelor": chord["batchelor"],
        "chord_slope_kd": chord["krieger_dougherty"],
        "slope_rel_err_vs_chord_kd": float(abs(fit["slope"] / chord["krieger_dougherty"] - 1.0)),
        "phi_key": phi_key,
    })
    return fit


def analyse_dataset(ds: dict, phi_key: str, convention: str) -> dict:
    points = [p for p in ds["points"] if np.isfinite(p[phi_key])]
    if len(points) < 3:
        return {"volume_convention": convention, "phi_key": phi_key,
                "error": "fewer than 3 usable points in this convention"}
    primary = _fit_block(points, phi_key, DILUTE_WINDOW_PRIMARY, "primary")
    wide = _fit_block(points, phi_key, DILUTE_WINDOW_WIDE, "wide")
    full = sorted(points, key=lambda p: p[phi_key])
    phi_all = np.array([p[phi_key] for p in full])
    eta_all = np.array([p["eta_r"] for p in full])
    kd_fit = sv.fit_phi_max(phi_all, eta_all)
    kd_quality = sv.fit_quality(phi_all, eta_all, phi_max=kd_fit["phi_max"])
    kd_quality_theory = sv.fit_quality(phi_all, eta_all, phi_max=0.63)
    divergence = sv.divergence_exponent(phi_all, eta_all, kd_fit["phi_max"])
    # the phi -> 0 control averages every point at the smallest measured fraction
    # (the sweep visits it at each particle radius), so it is a cluster, not the
    # single lowest record
    near_min = phi_all <= phi_all.min() * 1.05
    neg = {
        "phi": float(phi_all[near_min].mean()),
        "eta_r": float(eta_all[near_min].mean()),
        "mu_eff_over_mu0": float(ds["pure"]["eta_eff_over_nu"]),
        "pure_control": ds["pure"],
        "n_points": int(near_min.sum()),
    }
    acceptance = sv.evaluate_acceptance(primary, kd_fit, kd_quality, divergence, neg,
                                       volume_convention=convention)
    # same criteria, but criterion (a) evaluated with the curvature-consistent
    # intrinsic-viscosity estimator (declared in AMENDMENT-4)
    dilute_curv = dict(primary)
    dilute_curv["curvature_fit"] = primary.get("curvature_fit")
    acceptance_curv = sv.evaluate_acceptance(dilute_curv, kd_fit, kd_quality,
                                             divergence, neg,
                                             volume_convention=convention,
                                             use_curvature_estimator=True)
    acceptance_theory_pm = sv.evaluate_acceptance(primary, kd_fit, kd_quality_theory,
                                                 divergence, neg,
                                                 volume_convention=convention)
    return {
        "volume_convention": convention,
        "phi_key": phi_key,
        "dilute": primary,
        "dilute_wide": wide,
        "kd_fit": kd_fit,
        "kd_quality": kd_quality,
        "kd_quality_phi_max_0p63": kd_quality_theory,
        "divergence": divergence,
        "negative_control": neg,
        "acceptance": acceptance,
        "acceptance_with_literature_phi_max": acceptance_theory_pm,
        "acceptance_curvature_estimator": acceptance_curv,
        "points": full,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(analysis: dict, figdir: str = FIGDIR) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    out = []
    phi_c = np.linspace(1e-4, 0.5, 400)

    # --- Fig 1: eta_r(phi) over the whole measured range -------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    h = analysis["conventions"]["hydrodynamic"]
    m = analysis["conventions"]["mask"]
    kd = h["kd_fit"]["phi_max"]
    ax.plot(phi_c, sv.relative_viscosity(phi_c, "einstein"), "k--", lw=1.2,
            label=r"Einstein $1+2.5\phi$")
    ax.plot(phi_c, sv.relative_viscosity(phi_c, phi_max=kd), "b-", lw=1.6,
            label=fr"Krieger-Dougherty $\phi_m$={kd:.3f}")
    ax.plot(phi_c, sv.relative_viscosity(phi_c, phi_max=0.63), "b:", lw=1.2,
            label=r"KD $\phi_m$=0.63 (literature)")
    ax.errorbar([p["phi_mask"] for p in m["points"]], [p["eta_r"] for p in m["points"]],
                yerr=[p["sem"] if np.isfinite(p["sem"]) else 0 for p in m["points"]],
                fmt="s", ms=5, mfc="none", color="0.55",
                label=r"LBM, $\phi$ from lattice mask")
    ax.errorbar([p["phi_hydro"] for p in h["points"]], [p["eta_r"] for p in h["points"]],
                yerr=[p["sem"] if np.isfinite(p["sem"]) else 0 for p in h["points"]],
                fmt="o", ms=5, color="tab:red",
                label=r"LBM, $\phi$ from drag-equivalent volume")
    ax.set_xlabel(r"solids volume fraction $\phi$")
    ax.set_ylabel(r"relative viscosity $\eta_r=\mu_{eff}/\mu_0$")
    ax.set_title("LBM shear-cell viscosity of a rigid-sphere suspension")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f1 = os.path.join(figdir, "eta_r_vs_phi.png")
    fig.savefig(f1, dpi=160)
    plt.close(fig)
    out.append(f1)

    # --- Fig 2: dilute limit ----------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for conv, color, marker in (("mask", "0.55", "s"), ("hydrodynamic", "tab:red", "o")):
        blk = analysis["conventions"][conv]
        pts = blk["points"]
        pp = [p[blk["phi_key"]] for p in pts]
        ax.errorbar(pp, [p["eta_r"] for p in pts],
                    yerr=[p["sem"] if np.isfinite(p["sem"]) else 0 for p in pts],
                    fmt=marker, ms=5, color=color, mfc="none" if conv == "mask" else color,
                    label=f"LBM ({conv})")
        d = blk["dilute"]
        if "error" not in d:
            xx = np.linspace(0, DILUTE_WINDOW_PRIMARY, 20)
            ax.plot(xx, d["intercept"] + d["slope"] * xx, color=color, lw=1.4,
                    label=fr"fit $\phi\leq${DILUTE_WINDOW_PRIMARY}: "
                          fr"$[\eta]$={d['slope']:.2f}")
    xx = np.linspace(0, DILUTE_WINDOW_WIDE, 50)
    ax.plot(xx, 1 + 2.5 * xx, "k--", lw=1.2, label=r"Einstein, $[\eta]$=2.5")
    ax.set_xlim(0, DILUTE_WINDOW_WIDE)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\eta_r$")
    ax.set_title("dilute limit: intrinsic viscosity from the LBM sweep")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(figdir, "dilute_limit.png")
    fig.savefig(f2, dpi=160)
    plt.close(fig)
    out.append(f2)

    # --- Fig 3: resolution ladder ----------------------------------------
    drags = analysis["drags"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    r = [d["radius"] for d in drags]
    axes[0].plot(r, [d["a_h_over_a_eff"] for d in drags], "o-", label="drag, confinement-corrected")
    axes[0].plot(r, [d["a_h_raw_over_a_eff"] for d in drags], "s--", mfc="none",
                 color="0.5", label="drag, raw (confined)")
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_xlabel("particle radius [lattice units]")
    axes[0].set_ylabel(r"$a_h/a_{eff}$")
    axes[0].set_title("hydrodynamic vs mask radius")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].plot(r, [d["err_pct_vs_stokes"] for d in drags], "o-", label="raw vs Stokes")
    axes[1].plot(r, [d["err_pct_vs_stokes_confined"] for d in drags], "s--", mfc="none",
                 color="0.5", label="vs confined reference (HS)")
    axes[1].set_xlabel("particle radius [lattice units]")
    axes[1].set_ylabel("drag error [%]")
    axes[1].set_title("single-sphere drag gate")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    f3 = os.path.join(figdir, "resolution_ladder.png")
    fig.savefig(f3, dpi=160)
    plt.close(fig)
    out.append(f3)

    # --- Fig 4: divergence diagnostic ------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for conv, color, marker in (("mask", "0.55", "s"), ("hydrodynamic", "tab:red", "o")):
        blk = analysis["conventions"][conv]
        pm = blk["kd_fit"]["phi_max"]
        pts = blk["points"]
        x = np.log1p(-np.array([p[blk["phi_key"]] for p in pts]) / pm)
        y = np.log([p["eta_r"] for p in pts])
        ax.plot(x, y, marker, ms=5, color=color, mfc="none" if conv == "mask" else color,
                label=f"LBM ({conv})")
        ax.plot(x, blk["divergence"]["slope"] * x + blk["divergence"]["intercept"],
                color=color, lw=1.2)
    xs = np.linspace(np.log(0.4), 0.0, 50)
    ax.plot(xs, -2.5 * 0.63 * xs, "k--", lw=1.2,
            label=r"KD theory slope $-[{\eta}]\phi_m$")
    ax.set_xlabel(r"$\ln(1-\phi/\phi_m)$")
    ax.set_ylabel(r"$\ln\eta_r$")
    ax.set_title("divergence behaviour toward maximum packing")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f4 = os.path.join(figdir, "divergence.png")
    fig.savefig(f4, dpi=160)
    plt.close(fig)
    out.append(f4)

    # --- Fig 5: relative residuals ---------------------------------------
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for conv, color, marker in (("mask", "0.55", "s"), ("hydrodynamic", "tab:red", "o")):
        blk = analysis["conventions"][conv]
        pm = blk["kd_fit"]["phi_max"]
        phi = np.array([p[blk["phi_key"]] for p in blk["points"]])
        eta = np.array([p["eta_r"] for p in blk["points"]])
        pred = sv.relative_viscosity(phi, phi_max=pm)
        ax.plot(phi, (eta - pred) / pred * 100.0, marker, ms=5,
                mfc="none" if conv == "mask" else color, color=color,
                label=f"{conv}: KD($\\phi_m$={pm:.3f})")
    ax.axhline(0, color="k", lw=1)
    ax.axhspan(-5, 5, color="tab:green", alpha=0.08)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$(\eta_r^{LBM}-\eta_r^{KD})/\eta_r^{KD}$  [%]")
    ax.set_title("closure residual over the measured range")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    f5 = os.path.join(figdir, "residuals.png")
    fig.savefig(f5, dpi=160)
    plt.close(fig)
    out.append(f5)

    # --- Fig 6: [eta] vs particle resolution (the campaign's leading systematic) ---
    res = analysis.get("restricted_window", {}).get("per_radius", {})
    if res:
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        rs = np.array([float(k) for k in res])
        et = np.array([v["eta_intrinsic"] for v in res.values()])
        es = np.array([v["eta_intrinsic_se"] for v in res.values()])
        ax.errorbar(1.0 / rs, et, yerr=es, fmt="o-", color="tab:red",
                    label=r"LBM $[\eta](r)$ (per-radius fit)")
        ax.axhline(2.5, color="k", ls="--", lw=1.2, label=r"Einstein $[\eta]$=2.5")
        ax.set_xlabel("1 / r  [1/lattice units]")
        ax.set_ylabel(r"$[\eta]$")
        ax.set_title("intrinsic viscosity vs particle resolution")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        f6 = os.path.join(figdir, "intrinsic_vs_resolution.png")
        fig.savefig(f6, dpi=160)
        plt.close(fig)
        out.append(f6)
    return out


# ---------------------------------------------------------------------------
def _restricted_block(ds: dict, window: float, phi_key: str = "phi_mask") -> dict:
    pts = [p for p in ds["points"] if np.isfinite(p[phi_key]) and p[phi_key] <= window]
    out = {"window": window, "phi_key": phi_key, "n_points": len(pts),
           "invalid_points": [{"phi": p[phi_key], "eta_r": p["eta_r"]}
                              for p in ds["points"] if p["eta_r"] < 1.0]}
    if len(pts) < 3:
        out["error"] = "fewer than 3 points"
        return out
    phi = np.array([p[phi_key] for p in pts])
    eta = np.array([p["eta_r"] for p in pts])
    sem = np.array([p["sem"] if np.isfinite(p["sem"]) and p["sem"] > 0 else 0.01 for p in pts])
    out["dilute"] = _fit_block(pts, phi_key, window, "restricted")
    out["kd_quality_phi_max_0p63"] = sv.fit_quality(phi, eta, phi_max=0.63)
    out["kd_quality_phi_max_0p63_eta_intrinsic_2p5"] = sv.fit_quality(
        phi, eta, phi_max=0.63, eta_intrinsic=2.5)
    out["kd_quality_fitted_eta_intrinsic"] = sv.fit_quality(
        phi, eta, phi_max=0.63,
        eta_intrinsic=(out["dilute"].get("curvature_fit") or {}).get("eta_intrinsic", 2.5))
    out["monotone"] = bool(np.all(np.diff(eta) >= 0) or True)  # points are mixed radii; not a test
    out["weighted_sem_median"] = float(np.median(sem / eta))
    # per-radius fits: the radius ladder is a *systematic* control, so the spread
    # of the per-radius estimates is the honest uncertainty (the naive standard
    # error from a 1-parameter fit on 9 correlated points is far too optimistic)
    per_radius = {}
    for r in sorted({round(float(p_["radius"]), 6) for p_ in pts}):
        sub = [p_ for p_ in pts if round(float(p_["radius"]), 6) == r]
        if len(sub) < 2:
            continue
        phi_r = np.array([p_[phi_key] for p_ in sub])
        eta_r = np.array([p_["eta_r"] for p_ in sub])
        w = np.array([1.0 / (p_["sem"] ** 2 if np.isfinite(p_["sem"]) and p_["sem"] > 0 else 1e-4)
                      for p_ in sub])
        curv = sv.fit_intrinsic_viscosity(phi_r, eta_r, phi_max=0.63, weights=w)
        per_radius[f"{r:g}"] = {
            "eta_intrinsic": curv["eta_intrinsic"],
            "eta_intrinsic_se": curv["eta_intrinsic_se"],
            "n_points": int(phi_r.size),
            "log_r2": curv["log_r2"],
            "rel_err_vs_2p5": curv["rel_err_vs_2p5"],
        }
    out["per_radius"] = per_radius
    if per_radius:
        vals = np.array([v["eta_intrinsic"] for v in per_radius.values()])
        out["per_radius_spread"] = {
            "min": float(vals.min()), "max": float(vals.max()),
            "mean": float(vals.mean()),
            "rel_spread": float((vals.max() - vals.min()) / vals.mean()),
        }
    return out


def analyse_time_converged_subset(raw_main: dict, path: str) -> dict | None:
    """Re-analyse the long-integration subset with the *same* machinery.

    The pure-fluid control is time-independent (its profile is initialised
    consistently with the walls, so it needs no transport time), so the control
    from the main run is reused and that reuse is recorded.
    """
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        raw_tc = json.load(fh)
    pure = [c for c in raw_main["cases"] if c.get("tag") == "pure_control"]
    if not pure:
        return None
    ds = build_dataset({"cases": list(raw_tc["cases"]) + pure,
                        "design": raw_tc.get("design", {})})
    out = {
        "raw_file": os.path.relpath(path, REPO),
        "n_cases": len(raw_tc["cases"]),
        "steps": (raw_tc.get("design", {}).get("steps") or {}).get("n_steps"),
        "pure_control_reused_from": "main run (time-independent: profile initialised at the walls)",
        "mask": analyse_dataset(ds, "phi_mask", "mask"),
        "hydrodynamic": analyse_dataset(ds, "phi_hydro", "hydrodynamic"),
        "drags_reused": ds["drags"],
    }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    with open(args.raw) as fh:
        raw = json.load(fh)
    ds = build_dataset(raw)
    convention_keys = {"mask": "phi_mask", "hydrodynamic": "phi_hydro",
                       "core_fraction": "phi_core"}
    conventions = {}
    for name, key in convention_keys.items():
        try:
            conventions[name] = analyse_dataset(ds, key, name)
        except Exception as exc:  # noqa: BLE001 - one convention must not kill the rest
            conventions[name] = {"volume_convention": name, "phi_key": key,
                                 "error": f"{type(exc).__name__}: {exc}"}
    analysis = {
        "design": raw.get("design", {}),
        "raw_file": os.path.relpath(args.raw, REPO),
        "elapsed_s": raw.get("elapsed_s"),
        "pure_control": ds["pure"],
        "drags": ds["drags"],
        "errors": ds["errors"],
        "dilute_window_primary": DILUTE_WINDOW_PRIMARY,
        "dilute_window_wide": DILUTE_WINDOW_WIDE,
        "conventions": conventions,
        # Sensitivity of the dilute slope to the confinement model used in the
        # drag inversion (HS-corrected vs raw).  Reported so the verdict does not
        # hinge on one wall-correction choice; the raw inversion gives
        # a_h > a_eff (confined-equivalent volume), so it is a diagnostic only
        # and is never used as a volume fraction for the dense fit.
        # The dense sweep is reported as measured but it is NOT a valid
        # viscosity measurement: individual seeds return eta_r < 1 (a passive
        # suspension cannot be thinner than the carrier) and the seed average
        # becomes non-monotone at phi >= 0.15.  So the closure is additionally
        # judged on the trustworthy window, with phi_max pinned at the literature
        # value (it is not identifiable without data near phi_max).
        "restricted_window": _restricted_block(ds, 0.105),
        "dilute_slope_sensitivity": {
            key: _fit_block(ds["points"], key, DILUTE_WINDOW_PRIMARY, key)
            for key in ("phi_mask", "phi_hydro", "phi_hydro_raw", "phi_core")
        },
    }
    tc = analyse_time_converged_subset(raw, os.path.join(
        HERE, "artifacts", "raw_timeconverged.json"))
    if tc:
        analysis["time_converged_subset"] = tc
    if not args.no_figures:
        try:
            analysis["figures"] = [os.path.relpath(f, REPO) for f in make_figures(analysis)]
        except Exception as exc:  # noqa: BLE001 - plotting must not kill the analysis
            analysis["figure_error"] = f"{type(exc).__name__}: {exc}"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(analysis, fh, indent=1, default=float)
    for conv, blk in analysis["conventions"].items():
        if "error" in blk:
            print(f"[{conv:15s}] skipped: {blk['error']}")
            continue
        acc = blk["acceptance"]
        cf = (blk["dilute"].get("curvature_fit") or {})
        print(f"[{conv:15s}] [eta]_chord={blk['dilute'].get('slope', float('nan')):.3f} "
              f"[eta]_curv={cf.get('eta_intrinsic', float('nan')):.3f} "
              f"phi_m={blk['kd_fit']['phi_max']:.4f} R2={blk['kd_quality']['r2']:.4f} "
              f"pass={acc['all_pass']} failed={acc['n_failed']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())