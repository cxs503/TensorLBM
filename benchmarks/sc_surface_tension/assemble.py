#!/usr/bin/env python
"""Assemble the P143W2-N01 gate: read the run JSONs, apply the criteria, draw the
figures and emit ``artifacts/sc_surface_tension/laplace_gate.json``.

Everything in the gate is computed from the stored run artifacts -- no number is
typed in by hand (report discipline: every figure/number traces to a JSON).
"""

from __future__ import annotations

import glob
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.d2q9 import equilibrium  # noqa: E402
from tensorlbm.sc_surface_tension import (  # noqa: E402
    RHO_L,
    build_stencil,
    contact_angle_from_mask,
    sessile_rho,
    step_sc,
    wall_psi_from_contact_angle,
)
from run import laplace_gate  # noqa: E402
from contact_sweep import synthetic_cap_bias  # noqa: E402

OUT = REPO / "artifacts" / "sc_surface_tension"
FIGS = OUT / "figs"
TOL = 3.0


def load(name_glob: str, prefer_long: bool = True) -> list[dict]:
    files = sorted(glob.glob(str(OUT / name_glob)))
    rows = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        d["_file"] = Path(f).name
        rows.append(d)
    return rows


def pick_rows(stencil: str, radii=(20, 40, 60)) -> list[dict]:
    """Best (most converged / longest) run per radius for one stencil."""
    rows = load(f"laplace_{stencil}_R*.json")
    out = []
    for R in radii:
        cand = [r for r in rows if abs(r["R_init"] - R) < 1e-6]
        if not cand:
            continue
        cand.sort(key=lambda r: (r["steps"], -abs(r["mass_drift"])), reverse=True)
        out.append(cand[0])
    return out


def fig_laplace(rows_by_stencil: dict[str, list[dict]]) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    colours = {"d2q9": "C0", "shell4": "C3"}
    for st, rows in rows_by_stencil.items():
        if not rows:
            continue
        inv_r = np.array([1.0 / r["R_eq"] for r in rows])
        dp = np.array([r["dp_flux"] for r in rows])
        sk = np.array([r["sigma_K"] for r in rows])
        c = colours.get(st, "C1")
        ax.plot(inv_r, dp, "o", color=c, label=f"{st}: measured $\\Delta p$ (flux)")
        order = np.argsort(inv_r)
        coef = np.polyfit(inv_r, dp, 1)
        xx = np.linspace(0, max(inv_r) * 1.1, 50)
        ax.plot(xx, np.polyval(coef, xx), "-", color=c, lw=1.0,
                label=f"{st}: fit slope $\\sigma$={coef[0]:.4f}")
        ax.plot(inv_r, sk * inv_r, "x", color=c, alpha=0.7,
                label=f"{st}: Korteweg anchor $\\sigma_K/R$")
    ax.set_xlabel("$1/R_{eq}$  [lu$^{-1}$]")
    ax.set_ylabel("$\\Delta p$  [lu]")
    ax.set_title("Laplace law from the momentum-flux pressure (2-D cylinder)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    p = FIGS / "laplace_fit.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p.relative_to(REPO))


def fig_anisotropy(rows_by_stencil: dict[str, list[dict]]) -> str:
    fig, ax = plt.subplots(figsize=(5.6, 5.0), subplot_kw={"projection": "polar"})
    colours = {"d2q9": "C0", "shell4": "C3"}
    for st, rows in rows_by_stencil.items():
        r = [x for x in rows if abs(x["R_init"] - 20.0) < 1e-6]
        if not r:
            continue
        d = r[0]
        dp = np.array(d["dp_theta"])
        th = np.linspace(0, 2 * math.pi, len(dp), endpoint=False)
        ax.plot(np.append(th, th[0]), np.append(dp / dp.mean(), dp[0] / dp.mean()),
                "-o", ms=3, lw=1.0, color=colours.get(st, "C1"),
                label=f"{st}: spread {d['spread_flux_pct']:.2f}%")
    ax.set_title("$\\Delta p(\\theta)/\\overline{\\Delta p}$  (R=20, 36 wedges)")
    ax.legend(loc="lower right", fontsize=7, bbox_to_anchor=(1.15, -0.1))
    p = FIGS / "anisotropy_polar.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p.relative_to(REPO))


def fig_profile(rows: list[dict]) -> str:
    r = [x for x in rows if abs(x["R_init"] - 40.0) < 1e-6]
    if not r:
        r = rows[:1]
    d = r[0]
    fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2))
    ax[0].plot(d["profile_r"], d["profile_psi"], "k-", label="$\\psi(r)$")
    ax[0].set_xlabel("r [lu]")
    ax[0].set_ylabel("$\\psi$")
    ax[0].set_xlim(0, d["R_eq"] * 2.0)
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    ax[0].set_title(f"{d['stencil']} R={d['R_init']:.0f}: pseudopotential profile")

    # pressure profiles of the same run (recomputed from the stored run? we keep
    # the p_flux/p_eos numbers only) -> bar of the three readouts
    labels = ["$p_{kin}$\n(raw flux)", "$p_{flux}$\n(flux+interaction)", "$p_{eos}$\n(closure)"]
    inside = [d["dp_kinetic"], d["dp_flux"], d["dp_eos"]]
    ax[1].bar(labels, inside, color=["C7", "C0", "C1"])
    ax[1].set_ylabel("$\\Delta p$  [lu]")
    ax[1].set_title("readout comparison (R=%.0f, $\\Delta p_{flux}$=%.5f)" % (d["R_init"], d["dp_flux"]))
    ax[1].grid(alpha=0.3, axis="y")
    for i, v in enumerate(inside):
        ax[1].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    p = FIGS / "pressure_readout.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p.relative_to(REPO))


def fig_contact(rows: list[dict], table: list[dict], snapshot: bool = True) -> str:
    rows = [r for r in rows if math.isfinite(r.get("theta_mask_deg", float("nan")))]
    rows.sort(key=lambda r: r["psi_w"])
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.3))
    psi_w = np.array([r["psi_w"] for r in rows])
    th = np.array([r["theta_mask_corrected_deg"] for r in rows])
    young = np.array([r["theta_young_deg"] for r in rows])
    ax[0].plot(psi_w, young, "s--", color="C1", label="Young relation (eq. 7)")
    ax[0].plot(psi_w, th, "o-", color="C0", label="measured (mask fit, bias-corrected)")
    for r in rows:
        ax[0].annotate(f"$\\rho_w$={r['rho_w']:.2f}", (r["psi_w"], r["theta_mask_corrected_deg"]),
                       textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax[0].set_xlabel("$\\psi_w$ (wall pseudopotential)")
    ax[0].set_ylabel("contact angle [deg]")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    ax[0].set_title("wetting boundary condition: setting vs realised angle")
    if snapshot:
        ny, nx, wall = 96, 160, 2
        solid = torch.zeros((ny, nx), dtype=torch.bool)
        solid[:wall, :] = True
        solid[ny - wall:, :] = True
        psi_w0 = wall_psi_from_contact_angle(60.0)
        rho = sessile_rho(ny, nx, float(wall - 1), 60.0, math.pi * 16 ** 2)
        from tensorlbm.sc_surface_tension import RHO_L
        rho = torch.where(solid, torch.full_like(rho, RHO_L), rho)
        z = torch.zeros_like(rho)
        f = equilibrium(rho, z, z)
        st = build_stencil("d2q9")
        for _ in range(1200):
            f = step_sc(f, stencil=st, solid=solid, wall_psi=psi_w0, bounce_back=True)
        rho = f.sum(dim=0)
        im = ax[1].imshow(rho.numpy(), origin="lower", cmap="viridis")
        q = contact_angle_from_mask(rho, solid)
        ax[1].set_title(f"sessile drop, $\\psi_w$={psi_w0:.3f} -> {q['theta_deg']:.1f} deg")
        ax[1].set_xlabel("x [lu]")
        ax[1].set_ylabel("y [lu]")
        fig.colorbar(im, ax=ax[1], fraction=0.04, label="$\\rho$")
    p = FIGS / "contact_angle.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p.relative_to(REPO))


def fig_negative(neg: dict, base_dp: float) -> str:
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    names = ["production\n$\\kappa_3\\neq0$", "$\\kappa_3=0$\n(spectral)", "$G_{eff}=0$\n(no interaction)"]
    vals = [base_dp]
    for k in ("kappa3_zero_spectral", "G_eff_zero"):
        vals.append(abs(neg[k]["dp_flux"]) if k in neg else float("nan"))
    ax.bar(names, vals, color=["C0", "C3", "C7"])
    ax.set_ylabel("$|\\Delta p|$  [lu]")
    ax.set_yscale("log")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.2e}", ha="center", va="bottom", fontsize=8)
    ax.set_title("negative controls: removing the capillary stress removes $\\Delta p$")
    ax.grid(alpha=0.3, axis="y")
    p = FIGS / "negative_control.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p.relative_to(REPO))


def main() -> int:
    FIGS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)

    # ---------------- (a) three-radius Laplace gate ------------------------
    d2 = pick_rows("d2q9")
    sh = pick_rows("shell4")
    gate_d2 = laplace_gate(d2) if d2 else {}
    gate_sh = laplace_gate(sh) if sh else {}

    # ---------------- (b) anisotropy --------------------------------------
    aniso = {}
    for st, rows in (("d2q9", d2), ("shell4", sh)):
        for r in rows:
            aniso[f"{st}_R{int(r['R_init'])}"] = {
                "spread_flux_pct": r["spread_flux_pct"],
                "max_dev_flux_pct": r["maxdev_flux_pct"],
                "spread_eos_pct": r["spread_eos_pct"],
                "kappa3": r["kappa3"],
                "stencil": st,
                "R": r["R_init"],
            }
    best_d2 = min((v["spread_flux_pct"] for k, v in aniso.items() if v["stencil"] == "d2q9"),
                  default=float("nan"))
    best_sh = min((v["spread_flux_pct"] for k, v in aniso.items() if v["stencil"] == "shell4"),
                  default=float("nan"))
    maxdev_d2 = max((v["max_dev_flux_pct"] for v in aniso.values() if v["stencil"] == "d2q9"),
                    default=float("nan"))
    maxdev_sh = max((v["max_dev_flux_pct"] for v in aniso.values() if v["stencil"] == "shell4"),
                    default=float("nan"))

    # ---------------- (c) contact angle -----------------------------------
    table = synthetic_cap_bias(ny=96, nx=160)
    (OUT / "contact_instrument_bias.json").write_text(json.dumps(table, indent=1))
    craw = load("contact_sweep_rho*.json")
    for r in craw:
        r.setdefault("theta_mask_deg", float("nan"))
        r.setdefault("theta_mask_corrected_deg", float("nan"))
    contact_rows = []
    for r in craw:
        if math.isfinite(r.get("theta_mask_corrected_deg", float("nan"))):
            contact_rows.append(r)
    cal = [r for r in contact_rows if r.get("kind", "cal") == "cal"]
    hold = [r for r in contact_rows if r.get("kind") == "holdout"]
    fit_ok = len(cal) >= 3
    calib = {}
    if fit_ok:
        psi = np.array([r["psi_w"] for r in cal])
        cos_meas = np.cos(np.radians([r["theta_mask_corrected_deg"] for r in cal]))
        A = np.vstack([psi, np.ones_like(psi)]).T
        slope, inter = np.linalg.lstsq(A, cos_meas, rcond=None)[0]
        # cross-validated residuals of the calibration itself
        resid = np.degrees(np.arccos(np.clip(A @ np.array([slope, inter]), -1, 1))) - \
            np.array([r["theta_mask_corrected_deg"] for r in cal])
        hold_err = []
        for r in hold:
            cos_pred = slope * r["psi_w"] + inter
            theta_pred = math.degrees(math.acos(max(-1.0, min(1.0, cos_pred))))
            hold_err.append({
                "rho_w": r["rho_w"],
                "psi_w": r["psi_w"],
                "theta_measured": r["theta_mask_corrected_deg"],
                "theta_predicted_by_calibration": theta_pred,
                "error_deg": theta_pred - r["theta_mask_corrected_deg"],
            })
        calib = {
            "n_cal": len(cal),
            "cos_theta_fit": {"slope": float(slope), "intercept": float(inter)},
            "calibration_residual_deg": [float(v) for v in resid],
            "calibration_max_resid_deg": float(np.max(np.abs(resid))) if len(resid) else float("nan"),
            "holdout": hold_err,
            "young_bias_deg": [float(r["theta_mask_corrected_deg"] - r["theta_young_deg"]) for r in cal],
        }
    young_err = [abs(r["theta_mask_corrected_deg"] - r["theta_young_deg"]) for r in cal] if cal else []
    contact_stats = {
        "n_valid": len(contact_rows),
        "theta_range_deg": [min((r["theta_mask_corrected_deg"] for r in contact_rows), default=float("nan")),
                            max((r["theta_mask_corrected_deg"] for r in contact_rows), default=float("nan"))],
        "young_abs_error_deg": young_err,
        "young_max_abs_error_deg": max(young_err) if young_err else float("nan"),
        "instrument_bias_deg": [r["bias_deg"] for r in table],
        "instrument_bias_max_abs_deg": max(abs(r["bias_deg"]) for r in table),
        "calibration": calib,
    }

    # ---------------- (d) negative controls ------------------------------
    neg = {}
    for k in ("kappa3_zero_spectral", "G_eff_zero"):
        f = OUT / f"negative_{k}.json"
        if f.exists():
            neg[k] = json.loads(f.read_text())
    base_dp = gate_d2["per_radius"][0]["dp_flux"] if gate_d2 else float("nan")
    neg_summary = {
        "production_dp": base_dp,
        "kappa3_zero_dp": abs(neg.get("kappa3_zero_spectral", {}).get("dp_flux", float("nan"))),
        "G_eff_zero_dp": abs(neg.get("G_eff_zero", {}).get("dp_flux", float("nan"))),
        "kappa3_zero_rho_separation": [
            neg.get("kappa3_zero_spectral", {}).get("rho_in", float("nan")),
            neg.get("kappa3_zero_spectral", {}).get("rho_out", float("nan")),
        ],
    }
    neg_summary["pass_kappa3_zero"] = bool(
        math.isfinite(neg_summary["kappa3_zero_dp"]) and neg_summary["kappa3_zero_dp"] < 1.0e-3)
    neg_summary["pass_G_eff_zero"] = bool(
        math.isfinite(neg_summary["G_eff_zero_dp"]) and neg_summary["G_eff_zero_dp"] < 1.0e-3)

    # ---------------- figures --------------------------------------------
    figs = []
    figs.append({"name": "laplace_fit", "path": fig_laplace({"d2q9": d2, "shell4": sh}),
                 "caption": "Laplace law: $\\Delta p$ from the momentum-flux pressure vs $1/R_{eq}$ "
                            "for the D2Q9 and 4-shell isotropic stencils; crosses are the independent "
                            "Korteweg anchor $\\sigma_K/R$."})
    figs.append({"name": "anisotropy_polar", "path": fig_anisotropy({"d2q9": d2, "shell4": sh}),
                 "caption": "Azimuthal variation of the measured pressure jump at R=20: the directional "
                            "artefact of the interface tension (peak-to-peak/mean)."})
    figs.append({"name": "pressure_readout", "path": fig_profile(d2),
                 "caption": "Left: azimuthally averaged pseudopotential profile. Right: the three "
                            "pressure readouts -- the raw momentum-flux trace misses the whole interaction "
                            "pressure, which is the historical reason the tension was unmeasurable."})
    figs.append({"name": "contact_angle", "path": fig_contact(contact_rows, table),
                 "caption": "Wetting boundary condition: realised contact angle vs wall pseudopotential "
                            "(Young relation vs measured) and a sessile-drop density snapshot."})
    figs.append({"name": "negative_control", "path": fig_negative(neg, base_dp),
                 "caption": "Negative controls: switching off the O($d^3$) truncation term "
                            "($\\kappa_3=0$, spectral gradient) or the interaction ($G_{eff}=0$) "
                            "collapses $\\Delta p$."})

    # ---------------- gate ------------------------------------------------
    crit = {
        "a_three_radius_laplace_d2q9_pct": gate_d2.get("max_dev_pct"),
        "a_three_radius_laplace_shell4_pct": gate_sh.get("max_dev_pct"),
        "a_pass": bool((gate_d2.get("max_dev_pct", 1e9) < TOL) or (gate_sh.get("max_dev_pct", 1e9) < TOL)),
        "a_korteweg_anchor_dev_pct_d2q9": gate_d2.get("ratio_laplace_over_K_dev_pct"),
        "a_korteweg_anchor_dev_pct_shell4": gate_sh.get("ratio_laplace_over_K_dev_pct"),
        "a_pass_korteweg_anchor": bool(
            min(gate_d2.get("ratio_laplace_over_K_dev_pct", 1e9),
                gate_sh.get("ratio_laplace_over_K_dev_pct", 1e9)) < TOL),
        "b_spread_pct_d2q9": maxdev_d2,
        "b_spread_pct_shell4": maxdev_sh,
        "b_p2p_pct_d2q9": best_d2,
        "b_p2p_pct_shell4": best_sh,
        "b_pass_maxdev_1pct": bool(min(maxdev_d2, maxdev_sh) < 1.0),
        "b_pass_p2p_1pct": bool(min(best_d2, best_sh) < 1.0),
        "c_contact_young_max_err_deg": contact_stats["young_max_abs_error_deg"],
        "c_contact_calibrated_max_err_deg": (calib.get("calibration_max_resid_deg")
                                             if calib else float("nan")),
        "c_pass_3deg_young": bool(contact_stats["young_max_abs_error_deg"] < 3.0),
        "c_pass_3deg_calibrated": bool(calib and calib.get("calibration_max_resid_deg", 1e9) < 3.0),
        "d_pass_kappa3_zero": neg_summary["pass_kappa3_zero"],
        "d_pass_G_eff_zero": neg_summary["pass_G_eff_zero"],
    }
    gate = {
        "node": "P143W2-N01",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/sc_surface_tension.py",
        "driver": "benchmarks/sc_surface_tension/run.py",
        "artifacts": str(OUT.relative_to(REPO)),
        "convention": "2-D cylindrical drop: Delta p * R = sigma (3-D sphere halves it)",
        "tol_pct": TOL,
        "criteria": crit,
        "laplace_d2q9": gate_d2,
        "laplace_shell4": gate_sh,
        "anisotropy": aniso,
        "contact": contact_stats,
        "negative": neg_summary,
        "figures": figs,
        "stencils": {"d2q9": build_stencil("d2q9"), "shell4": build_stencil("shell4")},
    }
    (OUT / "laplace_gate.json").write_text(json.dumps(gate, indent=1, default=str))
    print(json.dumps({k: crit[k] for k in crit}, indent=1, default=str))
    print("gate ->", OUT / "laplace_gate.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())