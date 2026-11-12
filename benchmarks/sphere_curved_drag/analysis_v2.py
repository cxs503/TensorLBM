"""Analysis for the m16085 root-cause follow-up (GAP-LBM-SPHERE v2).

Inputs (all produced by scripts in this directory, nothing hand-copied):
  out/stokes_Re0.1_L36.json          m16085 production record (tau = 18.5)
  out/smoke_Re10_L36.json            m16085 smoke record   (tau = 18.5, mislabelled)
  out/diag_visc_k*.json              nu_eff(k, tau) tables
  out/fix_Re0.1_u0.0025_L36.json     corrected-protocol run (tau = 1.4)
  out/fix_Re0.1_u0.0025_L36.log      its progress record -> CD(t)

Outputs: out/results_v2.json, figures/*.png
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorlbm.array_drag_reference import array_drag_factor_from_lambda  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


def K_hasimoto(lam: float) -> float:
    """Hasimoto (1959) SC array correction (canonical series).

    Was ``- 27.4 * lam**5`` until the 2026-11-12 audit; the term is
    ``-27.3594 lam**6`` (``docs/ARRAY_DRAG_REFERENCE_AUDIT.md``).
    """
    return array_drag_factor_from_lambda(lam)


def load(name: str) -> dict:
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else {}


def parse_log(path: Path):
    """Progress lines of the driver: step=, U=, g=, F_bfl=."""
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if "step=" not in line:
            continue
        d = {}
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    d[k] = float(v.split("/")[0])
                except ValueError:
                    pass
        if "step" in d and "U" in d and "F_bfl" in d:
            rows.append(d)
    return rows


def main() -> int:
    m16085 = load("stokes_Re0.1_L36.json")
    smoke = load("smoke_Re10_L36.json")
    fixed = load("fix_Re0.1_u0.0025_L36.json")
    visc = []
    for n in ("diag_visc_k.json", "diag_visc_k_extra.json", "diag_visc_k_tau14.json"):
        p = OUT / n
        if p.exists():
            visc.extend(json.loads(p.read_text()))

    R = float(m16085.get("R", 6.0))
    L = int(m16085.get("L", 36))
    lam_h = R / L
    K = K_hasimoto(lam_h)

    # ---- 1. the m16085 Re reporting bug (mu = nu/3) -------------------
    u = float(m16085["u_measured"])
    nu = float(m16085["nu"])
    re_reported = float(m16085["Re_measured"])
    re_true = u * 2.0 * R / nu

    # ---- 2. the viscosity-controlled prediction ----------------------
    nu_eff = {}
    for r in visc:
        nu_eff[(round(r["tau"], 3), round(r["k"], 5))] = r["nu_eff"]
    k_off = round(2 * math.pi / L, 5)          # 0.1745 for L = 36
    nve = nu_eff.get((18.5, k_off))
    re_eff = u * 2.0 * R / nve if nve else float("nan")
    cd_pred = 24.0 / re_eff * K if nve else float("nan")
    cd_meas = float(m16085["cd_mb_fluid"])
    cd_analytic = float(m16085["cd_stokes_confined"])
    f_pred = 6.0 * math.pi * nve * R * u * K if nve else float("nan")
    f_meas = float(m16085["F_momentum_balance"])

    # ---- 3. the corrected-protocol run --------------------------------
    corr = {}
    if fixed:
        uf = float(fixed["u_measured"])
        corr = {
            "tag": fixed["tag"],
            "tau": fixed["tau"],
            "nu": fixed["nu"],
            "u_measured": uf,
            "u_target": fixed["u_target"],
            "Re_measured": float(fixed["Re_measured"]),
            "F_momentum_balance": float(fixed["F_momentum_balance"]),
            "F_bfl": float(fixed["F_bfl"]),
            "bfl_rel_diff_pct": float(fixed["bfl_rel_diff_pct"]),
            "cd_mb_fluid": float(fixed["cd_mb_fluid"]),
            "cd_stokes_confined": float(fixed["cd_stokes_confined"]),
            "cd_analytic_at_measured_Re": 24.0 / (uf * 2.0 * R / fixed["nu"])
            * K_hasimoto(R / L) if uf > 0 else float("nan"),
            "cd_err_pct_vs_measured_Re": 100.0
            * (float(fixed["cd_mb_fluid"])
               - 24.0 / (uf * 2.0 * R / fixed["nu"]) * K_hasimoto(R / L))
            / (24.0 / (uf * 2.0 * R / fixed["nu"]) * K_hasimoto(R / L)) if uf > 0 else float("nan"),
            "box_diffusion_steps": (L * L) / float(fixed["nu"]),
            "cd_err_pct": 100.0 * (float(fixed["cd_mb_fluid"]) - float(fixed["cd_stokes_confined"]))
            / float(fixed["cd_stokes_confined"]),
            "steady": bool(fixed["steady"]),
            "mfp_over_R": float(fixed.get("mfp_over_R", float("nan"))),
            "hydrodynamic_regime": bool(fixed.get("hydrodynamic_regime", True)),
        }

    # ---- figures ------------------------------------------------------
    # fig 1: nu_eff / nu_theory vs k, one curve per tau
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    taus = sorted({r["tau"] for r in visc})
    for tau in taus:
        pts = sorted((r["k"], r["nu_eff"] / r["nu_theory"]) for r in visc if r["tau"] == tau)
        if len(pts) < 2:
            continue
        ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=f"$\\tau$={tau:g}")
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.axvline(1.0 / R, color="C3", lw=0.9, ls="--")
    ax.annotate("k = 1/R", xy=(1.0 / R, 0.06), xytext=(1.0 / R * 1.15, 0.1),
                color="C3", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("wavenumber k (lattice units)")
    ax.set_ylabel(r"$\nu_{\rm eff}(k)/\nu_{\rm theory}$,  $\nu_{\rm theory}=(\tau-1/2)/3$")
    ax.set_title("Steady single-mode forcing: BGK viscosity is only hydrodynamic for $c_s\\tau \\ll R$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "nu_eff_vs_k.png", dpi=140)
    plt.close(fig)

    # fig 2: m16085 drag: measured vs analytic vs viscosity-corrected prediction
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["Hasimoto\n(analytic, $\\nu$=6)", "m16085 measured\n($\\tau$=18.5)",
              "prediction with\n$\\nu_{\\rm eff}(k{=}2\\pi/L)$"]
    vals = [cd_analytic, cd_meas, cd_pred]
    cols = ["C0", "C3", "C2"]
    bars = ax.bar(labels, vals, color=cols, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylabel("$C_D$ (Re = 0.1 target, $\\lambda=R/L=1/6$)")
    ax.set_title("The 10x $C_D$ deficit is reproduced by the kinetic viscosity, not by the wall")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "cd_deficit_explained.png", dpi=140)
    plt.close(fig)

    # fig 3: corrected-protocol convergence (from the driver progress record)
    log = parse_log(OUT / "fix_Re0.1_u0.0025_L36.log")
    if log:
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ts = [r["step"] for r in log]
        cds = [2.0 * r["F_bfl"] / (r["U"] ** 2 * math.pi * R * R) for r in log]
        ax.plot(ts, cds, "o-", label="$C_D$ from the BFL wall ledger")
        ax.axhline(cd_analytic, color="C0", ls="--", label="Hasimoto analytic (confined Stokes)")
        ax.set_xlabel("step")
        ax.set_ylabel("$C_D$")
        ax.set_title("Corrected protocol: Re=0.1 at $\\tau$=1.4 (low $u$ instead of large $\\nu$)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG / "corrected_protocol_convergence.png", dpi=140)
        plt.close(fig)

    res = {
        "m16085_case": {
            "tau": float(m16085["tau"]),
            "nu": nu,
            "u_measured": u,
            "Re_measured_reported": re_reported,
            "Re_measured_correct": re_true,
            "Re_report_bias_factor": re_reported / re_true,
            "mfp_over_R": float(m16085.get("mfp_over_R", (float(m16085["tau"]) / math.sqrt(3)) / R)),
            "cd_measured": cd_meas,
            "cd_stokes_confined": cd_analytic,
            "cd_ratio_meas_over_analytic": cd_meas / cd_analytic,
            "F_measured": f_meas,
            "steady": bool(m16085.get("steady", False)),
        },
        "viscosity_root_cause": {
            "k_used": k_off,
            "nu_eff_at_k": nve,
            "nu_eff_over_nu": (nve / nu) if nve else float("nan"),
            "Re_eff": re_eff,
            "cd_predicted": cd_pred,
            "F_predicted": f_pred,
            "cd_pred_over_meas": cd_pred / cd_meas,
            "F_pred_over_meas": f_pred / f_meas,
        },
        "corrected_protocol_run": corr,
        "viscosity_table": visc,
        "smoke_case_note": {
            "tag": smoke.get("tag"),
            "declared_Re_target": smoke.get("Re_target"),
            "actual_tau": smoke.get("tau"),
            "Re_actual": smoke.get("Re_measured", float("nan")) / 3.0 if smoke else float("nan"),
        },
    }
    (OUT / "results_v2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("m16085_case", "viscosity_root_cause",
                                          "corrected_protocol_run")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())