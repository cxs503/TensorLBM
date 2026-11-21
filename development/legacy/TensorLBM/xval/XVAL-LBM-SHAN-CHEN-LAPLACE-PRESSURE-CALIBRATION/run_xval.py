"""XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION.

Cross-validation of the *pressure calibration* practice of
``lbm-principles-practice/code`` (``chapter9/shanchen.cpp``: ``computePressure``
:126-139, ``calculateDeltaPressure`` :326-377) against

  (a) the Maxwell construction of the Shan-Chen equation of state (analytic),
  (b) the coexistence constants documented by TensorLBM
      (``RHO_L_DEFAULT=1.957``, ``RHO_V_DEFAULT=0.1596``, ``SIGMA_DEFAULT=0.05611``),
  (c) the coexistence densities measured from the reference C++ run
      (1.6707 / 0.18473 at g=-4.7, OSS inventory table §9),
  (d) the Laplace law across a radius sweep (radius independence of dp*R).

Outputs: artifacts/*.json + artifacts/fig*.png
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorlbm.d2q9 import equilibrium
from tensorlbm.multiphase import (RHO_L_DEFAULT, RHO_V_DEFAULT, SIGMA_DEFAULT,
                                  psi_exp, step_sc)
from tensorlbm.multiphase.laplace import (laplace_pressure_calibration,
                                          fit_laplace, sc_maxwell_coexistence,
                                          sc_pressure)

ART = Path(__file__).resolve().parent / "artifacts"
ART.mkdir(parents=True, exist_ok=True)
torch.set_default_dtype(torch.float64)

REF_DENSITIES = {"rho_l": 1.6707, "rho_v": 0.18473, "g": -4.7}   # OSS inventory §9


def droplet(rho_l, rho_v, R, n, width):
    yy, xx = torch.meshgrid(torch.arange(n, dtype=torch.float64),
                            torch.arange(n, dtype=torch.float64), indexing="ij")
    d = torch.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2) - R
    rho = rho_v + (rho_l - rho_v) * 0.5 * (1.0 - torch.tanh(d / width))
    return equilibrium(rho, torch.zeros_like(rho), torch.zeros_like(rho)), xx, yy


def relax(R, n=48, width=3.0, G=5.0, tau=1.0, steps=2500, ramp=0, probe_steps=None):
    """Relax a droplet with the library step_sc; optional linear ramp of G."""
    f, xx, yy = droplet(RHO_L_DEFAULT, RHO_V_DEFAULT, R, n, width)
    t0 = time.time()
    for s in range(1, steps + 1):
        g_eff = G * (min(1.0, s / ramp) if ramp else 1.0)
        f = step_sc(f, G=g_eff, tau=tau)
        if not torch.isfinite(f.sum(0)).all():
            return {"status": "diverged", "step": s, "radius_init": R,
                    "wall_s": time.time() - t0}
        if probe_steps and s == probe_steps:
            break
    return {"status": "ok", "f": f, "xx": xx, "yy": yy, "wall_s": time.time() - t0,
            "radius_init": R, "steps": steps, "width": width, "ramp": ramp}


def main():
    out = {"reference": {"repo": "lbm-principles-practice/code",
                         "file": "chapter9/shanchen.cpp",
                         "procedure": "4x4 box bulk pressures + rho_av crossing radius + gamma=dp*r",
                         "oss_inventory_densities": REF_DENSITIES},
           "library_constants": {"rho_l": RHO_L_DEFAULT, "rho_v": RHO_V_DEFAULT,
                                 "sigma": SIGMA_DEFAULT},
           "analytics": {}, "sweep": [], "defect_probes": []}

    # ---- (a) Maxwell construction: analytic coexistence of the SC EOS -------
    for G in (5.0, 4.7):
        out["analytics"][f"maxwell_G{G}"] = sc_maxwell_coexistence(G)
    mw5 = out["analytics"]["maxwell_G5.0"]
    out["analytics"]["maxwell_G5.0"]["err_rho_l_vs_library_pct"] = \
        abs(mw5["rho_l"] - RHO_L_DEFAULT) / RHO_L_DEFAULT * 100
    out["analytics"]["maxwell_G5.0"]["err_rho_v_vs_library_pct"] = \
        abs(mw5["rho_v"] - RHO_V_DEFAULT) / RHO_V_DEFAULT * 100
    mw47 = out["analytics"]["maxwell_G4.7"]
    out["analytics"]["maxwell_G4.7"]["err_rho_l_vs_reference_pct"] = \
        abs(mw47["rho_l"] - REF_DENSITIES["rho_l"]) / REF_DENSITIES["rho_l"] * 100
    out["analytics"]["maxwell_G4.7"]["err_rho_v_vs_reference_pct"] = \
        abs(mw47["rho_v"] - REF_DENSITIES["rho_v"]) / REF_DENSITIES["rho_v"] * 100

    # ---- (b) sign falsification: +G/6 psi^2 (as in wetting.measure_young_laplace)
    for G in (5.0,):
        pv = float(sc_pressure(torch.tensor(RHO_V_DEFAULT), G))
        pl = float(sc_pressure(torch.tensor(RHO_L_DEFAULT), G))
        sign_wrong = (RHO_L_DEFAULT / 3 + G / 6 * float(psi_exp(torch.tensor(RHO_L_DEFAULT))) ** 2) - \
                     (RHO_V_DEFAULT / 3 + G / 6 * float(psi_exp(torch.tensor(RHO_V_DEFAULT))) ** 2)
        out["analytics"]["pressure_sign_check"] = {
            "p_l_minus_p_v_correct_sign": pl - pv,
            "p_l_minus_p_v_plus_sign": sign_wrong,
            "ratio_wrong_over_correct": sign_wrong / (pl - pv),
        }

    # ---- (d) radius sweep with the library solver ---------------------------
    radii = [10.0, 12.0, 14.0]
    fields = {}
    for R in radii:
        res = relax(R)
        if res["status"] != "ok":
            out["sweep"].append({"radius_init": R, "status": res["status"],
                                 "step": res.get("step")})
            print(f"  R={R}: diverged at step {res.get('step')}")
            continue
        rho = res["f"].sum(0)
        try:
            cal = laplace_pressure_calibration(rho, G=5.0, box=4)
        except Exception as exc:                                  # guarded estimator
            out["sweep"].append({"radius_init": R, "status": f"calibration_failed: {exc}"})
            print(f"  R={R}: calibration failed: {exc}")
            continue
        cal.update(status="ok", radius_init=R, wall_s=res["wall_s"])
        out["sweep"].append(cal)
        fields[R] = (rho, res["xx"], res["yy"])
        print(f"  R={R}: rho_l={cal['rho_l']:.4f} rho_v={cal['rho_v']:.4f} "
              f"dp={cal['dp']:.5f} r={cal['radius']:.3f} gamma={cal['gamma']:.5f}")

    ok = [r for r in out["sweep"] if r.get("status") == "ok"]
    if len(ok) >= 2:
        fit = fit_laplace([r["dp"] for r in ok], [r["radius"] for r in ok])
        out["laplace_fit"] = fit
        out["laplace_fit"]["err_vs_SIGMA_DEFAULT_pct"] = \
            abs(fit["sigma"] - SIGMA_DEFAULT) / SIGMA_DEFAULT * 100
        gs = [r["gamma"] for r in ok]
        out["laplace_fit"]["gamma_spread_pct"] = (max(gs) - min(gs)) / abs(np.mean(gs)) * 100
        print(f"  fit: sigma={fit['sigma']:.5f} r2={fit['r2']:.4f} "
              f"err_vs_SIGMA_DEFAULT={out['laplace_fit']['err_vs_SIGMA_DEFAULT_pct']:.1f}%")

    # ---- library default start-up (width=1.5, no ramp): defect probe --------
    for width, ramp, tag in ((1.5, 0, "library_default_width1.5_noramp"),
                             (3.0, 0, "width3_noramp")):
        res = relax(14.0, width=width, ramp=ramp, steps=600)
        out["defect_probes"].append({"arm": tag, "status": res["status"],
                                     "step": res.get("step"), "radius_init": 14.0,
                                     "width": width, "ramp": ramp})
        print(f"  defect probe {tag}: {res['status']} step={res.get('step')}")

    # ---- figures ------------------------------------------------------------
    R_big = max(fields) if fields else None
    if R_big is not None:
        rho, xx, yy = fields[R_big]
        p = sc_pressure(rho, 5.0).numpy()
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        ax[0].imshow(rho.numpy(), origin="lower", cmap="viridis")
        ax[0].set_title(f"density rho (R_init={R_big:.0f})"); ax[0].set_xlabel("x"); ax[0].set_ylabel("y")
        im = ax[1].imshow(p, origin="lower", cmap="coolwarm")
        ax[1].set_title("lattice pressure p = rho/3 - G psi^2/6"); fig.colorbar(im, ax=ax[1], shrink=.8)
        rr = np.hypot(xx.numpy() - rho.shape[1] / 2, yy.numpy() - rho.shape[0] / 2)
        prof = rho.numpy()[rho.shape[0] // 2]
        ax[2].plot(np.arange(len(prof)) - rho.shape[1] / 2, prof, "o-", ms=3, label="rho mid row")
        for r in ok:
            ax[2].axvline(r["radius"], ls="--", lw=.8)
        ax[2].set_xlabel("x - x_c"); ax[2].set_ylabel("rho"); ax[2].legend(fontsize=7)
        ax[2].set_title("mid-row profile + measured radii"); ax[2].grid(alpha=.3)
        fig.tight_layout(); fig.savefig(ART / "fig1_fields_profiles.png", dpi=150); plt.close(fig)

    if len(ok) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.4))
        inv_r = np.array([1 / r["radius"] for r in ok]); dp = np.array([r["dp"] for r in ok])
        ax.plot(inv_r, dp, "o", ms=8, label="SCMP droplets (library step_sc)")
        xx_ = np.linspace(0, inv_r.max() * 1.1, 50)
        ax.plot(xx_, out["laplace_fit"]["sigma"] * xx_, "-",
                label=f"fit sigma={out['laplace_fit']['sigma']:.4f} (R2={out['laplace_fit']['r2']:.3f})")
        ax.plot(xx_, SIGMA_DEFAULT * xx_, "--", label=f"SIGMA_DEFAULT={SIGMA_DEFAULT:.4f}")
        ax.set_xlabel("1/R"); ax.set_ylabel("Delta p"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_title("Laplace law calibration (reference procedure)")
        fig.tight_layout(); fig.savefig(ART / "fig2_laplace_fit.png", dpi=150); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 4))
        labels = ["library\nconstants", "Maxwell\nG=5", "Maxwell\nG=4.7", "reference\nC++ run"]
        rl = [RHO_L_DEFAULT, mw5["rho_l"], mw47["rho_l"], REF_DENSITIES["rho_l"]]
        rv = [RHO_V_DEFAULT, mw5["rho_v"], mw47["rho_v"], REF_DENSITIES["rho_v"]]
        xs = np.arange(4)
        ax.bar(xs - .2, rl, .4, label="rho_l"); ax.bar(xs + .2, rv, .4, label="rho_v")
        ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=8); ax.legend(fontsize=8)
        ax.set_title("Coexistence: Maxwell (analytic) vs documented/measured"); ax.grid(alpha=.3, axis="y")
        for i, (a, b) in enumerate(zip(rl, rv)):
            ax.text(i - .2, a + .03, f"{a:.3f}", ha="center", fontsize=7)
            ax.text(i + .2, b + .03, f"{b:.3f}", ha="center", fontsize=7)
        fig.tight_layout(); fig.savefig(ART / "fig3_coexistence.png", dpi=150); plt.close(fig)

    with open(ART / "xval_results.json", "w") as fp:
        json.dump(out, fp, indent=2, default=float)
    print("artifacts ->", ART)


if __name__ == "__main__":
    main()
