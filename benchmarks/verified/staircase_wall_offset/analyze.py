#!/usr/bin/env python3
"""Post-process the staircase-wall ladder: offset decomposition + hold-out closure.

Machine-judged criteria (declared before the numbers were read, see docs report):
  C1 hold-out closure : delta_BB fitted on the two coarsest grids predicts the
                        finest grid's flux to <=3% (target) -- a blind prediction
  C2 offset constant   : |delta_BB(grid) - mean| <= 0.02 cell on every grid
  C3 raw flux vs H     : Q within 3% of the nominal-wall analytic -> reported,
                        expected FAIL (staircase geometry aliasing)
  C4 raw profile       : theta_nom <= 3% on >=2 grids -> reported, expected FAIL
"""

import json
import math
import statistics
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
rows = json.loads((HERE / "artifacts/ladder.json").read_text())["rows"]
fw = [r for r in rows if r["wall"] == "halfway"]
fw.sort(key=lambda r: r["H"])
if len(fw) < 3:
    raise SystemExit(f"need >=3 halfway grids, got {len(fw)}")

H = [r["H"] for r in fw]
d_eff = [r["delta_eff"] for r in fw]
d_geo = [r["delta_geom"] for r in fw]
d_bb = [r["delta_hydro"] for r in fw]
q_rel = [100.0 * (r["Q_over_Qana"] - 1.0) for r in fw]
theta = [r["theta_nom_pct"] for r in fw]

# --- C1: fit delta_BB on the coarsest two grids, predict the finest ---------
fit = statistics.fmean(d_bb[:2])
nu = fw[0]["nu"]
pred, meas = [], []
for r in fw:
    g = r["g"]
    h_pred = r["H_vol"] + fit
    q_pred = 2.0 * g * h_pred**3 / (3.0 * nu)
    pred.append(q_pred)
    meas.append(r["q_unit"])
err_holdout = 100.0 * abs(pred[-1] - meas[-1]) / meas[-1]
err_all = [100.0 * abs(p - m) / m for p, m in zip(pred, meas)]

mean_bb = statistics.fmean(d_bb)
c1 = err_holdout <= 3.0
c2 = max(abs(x - mean_bb) for x in d_bb) <= 0.02
c3 = all(abs(x) <= 3.0 for x in q_rel)
c4 = sum(1 for x in theta if x <= 3.0) >= 2

# --- interp diagnostics (if present) ---------------------------------------
ip = [r for r in rows if r["wall"] == "interp"]
interp_diag = []
for r in ip:
    rep = r.get("bfl_report") or {}
    n_tot = sum(rep.get(k, 0) for k in ("cut_links", "interpolated_links", "fallback_links"))
    interp_diag.append(
        {
            "H": r["H"],
            "theta_nom_pct": r["theta_nom_pct"],
            "delta_eff": r["delta_eff"],
            "delta_hydro": r["delta_hydro"],
            "Q_over_Qana": r["Q_over_Qana"],
            "fallback_frac": None if not n_tot else rep["fallback_links"] / n_tot,
            "q_mean": rep.get("q_mean"),
        }
    )

summary = {
    "case": "staircase_wall_offset_ladder",
    "anchor": "GAP-LBM-STAIRCASE-WALL-OFFSET",
    "grids": [
        {"L": r["L"], "H": r["H"], "steps": r["steps"], "q_drift_rel": r["q_drift_rel"]} for r in fw
    ],
    "rows": [
        {
            k: r[k]
            for k in (
                "L",
                "H",
                "delta_geom",
                "delta_eff",
                "delta_hydro",
                "Q_over_Qana",
                "Q_over_Qana_vol",
                "theta_nom_pct",
                "theta_vol_pct",
                "u_max_rel_err_pct",
                "q_drift_rel",
                "H_vol",
                "H_eff",
                "n_fluid",
            )
        }
        for r in fw
    ],
    "delta_BB_fit_from_coarse": fit,
    "delta_BB_mean_all": mean_bb,
    "q_pred": pred,
    "q_meas": meas,
    "closure_err_pct_all_grids": err_all,
    "holdout_err_pct_finest": err_holdout,
    "criteria": {
        "C1_holdout_closure_le_3pct": {"pass": c1, "value_pct": err_holdout, "target_pct": 3.0},
        "C2_offset_constant_le_0p02_cell": {
            "pass": c2,
            "value_cell": max(abs(x - mean_bb) for x in d_bb),
            "target_cell": 0.02,
        },
        "C3_raw_flux_le_3pct": {
            "pass": c3,
            "value_pct_max": max(abs(x) for x in q_rel),
            "target_pct": 3.0,
        },
        "C4_raw_profile_le_3pct_two_grids": {
            "pass": c4,
            "value_pct_min": min(theta),
            "n_grids_pass": sum(1 for x in theta if x <= 3.0),
            "target_pct": 3.0,
        },
    },
    "interp_diagnostics": interp_diag,
    "all_primary_pass": bool(c1 and c2),
}
(HERE / "artifacts/summary.json").write_text(json.dumps(summary, indent=1))

# --- figures ---------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.9))
r0 = fw[2]
p = r0["profile"]
ax[0].plot([b["s"] for b in p], [b["u_num"] for b in p], "o-", ms=3, label="LBM (halfway BB)")
ax[0].plot(
    [b["s"] for b in p], [b["u_ana"] for b in p], "k--", lw=1.2, label="analytic, nominal wall H"
)
ax[0].set_xlabel("s [lattice units]")
ax[0].set_ylabel("u")
ax[0].set_title(f"45-deg staircase channel, H={r0['H']:g} (theta={r0['theta_nom_pct']:.1f}%)")
ax[0].legend(fontsize=7)

ax[1].plot(H, d_eff, "o-", label=r"$\delta_{eff}$ (flux)")
ax[1].plot(H, d_geo, "s--", label=r"$\delta_{geom}$ (mask volume)")
ax[1].plot(H, [g + mean_bb for g in d_geo], "^:", label=r"$\delta_{geom}+0.19$")
ax[1].set_xlabel("nominal half-width H")
ax[1].set_ylabel("offset [cells]")
ax[1].set_title("offset decomposition (halfway bounce-back)")
ax[1].legend(fontsize=7)

ax[2].plot(H, q_rel, "o-", label="measured Q vs nominal H")
ax[2].plot(
    H,
    [100.0 * (pr / m - 1.0) for pr, m in zip(pred, meas)],
    "^-",
    label=r"a-priori Q$(H_{vol}+0.19)$",
)
ax[2].axhline(0, color="k", lw=0.8, ls=":")
ax[2].axhline(3, color="r", lw=0.8, ls=":")
ax[2].axhline(-3, color="r", lw=0.8, ls=":")
ax[2].set_xlabel("nominal half-width H")
ax[2].set_ylabel("flux error [%]")
ax[2].set_title(f"flux: hold-out closure error {err_holdout:.2f}%")
ax[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig(HERE / "artifacts/staircase_ladder.png", dpi=140)
print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "grids")}, indent=1))
