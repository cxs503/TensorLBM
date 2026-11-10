#!/usr/bin/env python3
"""Assemble the P40-N02 verification artefacts (numbers + figures).

Reads the per-case JSON files produced by ``scripts/run_p40n02_amr_case.py``
and writes

* ``artifacts/p40n02/verification_results.json`` — every criterion with its
  number, pass flag and the artefact path (numbers are injected
  programmatically, nothing is hand-copied),
* ``artifacts/p40n02/fig1_tg_energy.png`` … ``fig4_cavity_profiles.png``.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from tensorlbm.amr_flux_verification import (  # noqa: E402
    CavityConfig,
    CylinderConfig,
    cavity_compare,
    cylinder_compare,
)

ART = os.path.join(ROOT, "artifacts", "p40n02")


def load(name: str):
    path = os.path.join(ART, name)
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def fig_tg(tg: dict) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    t = np.asarray(tg["times_fine"], dtype=float)
    ax[0].semilogy(t, tg["energy_uniform_fine"], "k-", lw=2, label="uniform 128$^2$ (reference)")
    ax[0].semilogy(t, tg["energy_amr"], "r--", lw=2, label="AMR 64$^2$ + 64$^2$ patch")
    ax[0].semilogy(t, tg["energy_theory"], "b:", lw=1.5, label="analytic $E_0e^{-4\\nu k^2 t}$")
    ax[0].set_xlabel("time (fine lattice units)")
    ax[0].set_ylabel("kinetic energy")
    ax[0].set_title("Taylor-Green vortex decay, Re=100")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    e_amr = np.asarray(tg["energy_amr"])
    e_ref = np.asarray(tg["energy_uniform_fine"])
    ax[1].plot(t, 100 * np.abs(e_amr - e_ref) / e_ref, "r-", lw=1.8)
    ax[1].axhline(2.0, color="k", ls="--", lw=1, label="2 % criterion")
    ax[1].set_xlabel("time (fine lattice units)")
    ax[1].set_ylabel("$|E_{AMR}-E_{uniform}|/E_{uniform}$  [%]")
    ax[1].set_title(f"L2 error = {100*tg['l2_rel_vs_uniform']:.2f} %  (≤2 %)")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ART, "fig1_tg_energy.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_reflux(rc: dict) -> str:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    on = np.asarray(rc["reflux_on"]["mass_history"], dtype=float)
    off = np.asarray(rc["reflux_off"]["mass_history"], dtype=float)
    on = np.abs(on - on[0]) / on[0]
    off = np.abs(off - off[0]) / off[0]
    ax[0].semilogy(np.arange(len(on)), np.maximum(on, 1e-17), "b-", lw=2,
                   label=f"reflux ON  (final {on[-1]:.1e})")
    ax[0].semilogy(np.arange(len(off)), np.maximum(off, 1e-17), "r-", lw=2,
                   label=f"reflux OFF (final {off[-1]:.1e})")
    ax[0].axhline(1e-10, color="k", ls="--", lw=1, label="1e-10 criterion")
    ax[0].set_xlabel("coarse step")
    ax[0].set_ylabel("relative composite mass drift")
    ax[0].set_title("Reflux conservation (periodic 2-level patch)")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3, which="both")

    names = ["mass drift\n(reflux on)", "net interface\nflux residual", "mass drift\n(reflux off)"]
    vals = [rc["reflux_on"]["mass_drift_rel"],
            rc["reflux_on"]["interface_net_flux_residual_total"],
            rc["reflux_off"]["mass_drift_rel"]]
    ax[1].bar(range(3), np.maximum(vals, 1e-17), color=["tab:blue", "tab:green", "tab:red"])
    ax[1].set_yscale("log")
    ax[1].set_xticks(range(3))
    ax[1].set_xticklabels(names, fontsize=8)
    ax[1].axhline(1e-10, color="k", ls="--", lw=1)
    ax[1].axhline(1e-8, color="gray", ls=":", lw=1)
    for i, v in enumerate(vals):
        ax[1].annotate(f"{v:.1e}", (i, max(v, 1e-17)), ha="center", va="bottom", fontsize=8)
    ax[1].set_title("conservation metrics (criteria 1e-10 / 1e-8)")
    ax[1].grid(alpha=0.3, axis="y", which="both")
    fig.tight_layout()
    out = os.path.join(ART, "fig2_reflux_conservation.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_cavity(cmp: dict) -> str:
    pa, pu = cmp["amr"]["profiles"], cmp["uniform"]["profiles"]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 4.4))
    ax[0].plot(pa["u"], pa["y"], "r-", lw=2, label="AMR (64$^2$ + wall strips)")
    ax[0].plot(pu["u"], pu["y"], "k--", lw=1.6, label="uniform 128$^2$")
    ax[0].set_xlabel("u / u_lid"); ax[0].set_ylabel("y / L")
    ax[0].set_title("vertical centre line")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(pa["x"], pa["v"], "r-", lw=2, label="AMR")
    ax[1].plot(pu["x"], pu["v"], "k--", lw=1.6, label="uniform 128$^2$")
    ax[1].set_xlabel("x / L"); ax[1].set_ylabel("v / u_lid")
    ax[1].set_title(f"horizontal centre line (max err {100*cmp['error_max_vs_uniform_normalised_by_u_lid']:.2f} % u_lid)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(ART, "fig4_cavity_profiles.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def fig_cylinder(cmp: dict) -> str:
    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.2))
    labels = ["$C_D$", "St"]
    amr = [cmp["amr"]["c_d"], cmp["amr"]["st"]]
    uni = [cmp["uniform"]["c_d"], cmp["uniform"]["st"]]
    x = np.arange(2)
    ax.bar(x - 0.18, uni, 0.36, color="k", alpha=0.75, label="uniform 2x")
    ax.bar(x + 0.18, amr, 0.36, color="tab:red", alpha=0.85, label="AMR")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    for i in range(2):
        ax.annotate(f"{100*abs(amr[i]-uni[i])/uni[i]:.2f} %", (i, max(amr[i], uni[i]) * 1.02),
                    ha="center", fontsize=9)
    ax.set_title("Cylinder Re=100: AMR vs uniform (error ≤2 %)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out = os.path.join(ART, "fig5_cylinder_cd_st.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> int:
    tg = load("tg.json")
    rc = load("reflux.json")
    cav_amr = load("cavity_amr.json")
    cav_uni = load("cavity_uniform.json")
    cyl_amr = load("cylinder_amr.json")
    cyl_uni = load("cylinder_uniform.json")

    results = {
        "node": "P40-N02",
        "repo": "TensorLBM",
        "module": "src/tensorlbm/amr_composite2d.py",
        "criteria": {},
    }
    figs = {}
    cfg_c = CavityConfig()
    cfg_y = CylinderConfig()

    if tg:
        results["taylor_green"] = {k: v for k, v in tg.items() if k not in ("energy_amr", "energy_uniform_fine", "energy_theory", "times_fine")}
        results["criteria"]["b_taylor_green_l2"] = {
            "value": tg["l2_rel_vs_uniform"], "threshold": 0.02,
            "pass": bool(tg["l2_rel_vs_uniform"] <= 0.02),
            "max_rel": tg["max_rel_vs_uniform"],
            "artefact": "artifacts/p40n02/tg.json",
        }
        figs["fig1"] = fig_tg(tg)
    if rc:
        results["reflux"] = {k: v for k, v in rc.items() if k != "reflux_on" or True}
        results["reflux_clean"] = {
            "reflux_on": {k: v for k, v in rc["reflux_on"].items() if k != "mass_history"},
            "reflux_off": {k: v for k, v in rc["reflux_off"].items() if k != "mass_history"},
            "config": rc["config"],
        }
        results["criteria"]["a_reflux_mass_drift"] = {
            "value": rc["reflux_on"]["mass_drift_rel"], "threshold": 1e-10,
            "pass": bool(rc["reflux_on"]["mass_drift_rel"] <= 1e-10),
            "artefact": "artifacts/p40n02/reflux.json",
        }
        results["criteria"]["e_interface_net_flux"] = {
            "value": rc["reflux_on"]["interface_net_flux_residual_total"],
            "threshold": 1e-8,
            "pass": bool(rc["reflux_on"]["interface_net_flux_residual_total"] <= 1e-8),
            "artefact": "artifacts/p40n02/reflux.json",
        }
        results["criteria"]["control_no_reflux_drift"] = {
            "value": rc["reflux_off"]["mass_drift_rel"], "threshold": "> 1e-8 (control)",
            "pass": bool(rc["reflux_off"]["mass_drift_rel"] > 1e-8),
        }
        figs["fig2"] = fig_reflux(rc)
    if cav_amr and cav_uni:
        cmp = cavity_compare(cav_amr, cav_uni, cfg_c)
        results["cavity"] = {k: v for k, v in cmp.items() if k not in ("amr", "uniform")}
        results["cavity"]["amr_steps"] = cav_amr["steps"]
        results["cavity"]["uniform_steps"] = cav_uni["steps"]
        results["cavity"]["amr_mass_drift_rel"] = cav_amr["mass_drift_rel"]
        results["cavity"]["uniform_mass_drift_rel"] = cav_uni["mass_drift_rel"]
        results["criteria"]["c_cavity_centreline"] = {
            "value": cmp["error_max_vs_uniform_normalised_by_u_lid"],
            "threshold": 0.01,
            "pass": bool(cmp["error_max_vs_uniform_normalised_by_u_lid"] <= 0.01),
            "normalised_by_peak": cmp["error_max_vs_uniform_normalised_by_peak"],
            "ghia_u_abs_err": cmp["ghia_u_max_abs_error"],
            "ghia_v_abs_err": cmp["ghia_v_max_abs_error"],
            "artefact": "artifacts/p40n02/cavity_amr.json",
        }
        figs["fig4"] = fig_cavity(cmp)
    if cyl_amr and cyl_uni:
        cmp = cylinder_compare(cyl_amr, cyl_uni, cfg_y)
        results["cylinder"] = {k: v for k, v in cmp.items() if k not in ("amr", "uniform")}
        results["cylinder"]["amr"] = cyl_amr
        results["cylinder"]["uniform"] = cyl_uni
        results["criteria"]["d_cylinder_cd_st"] = {
            "value_cd_rel": cmp["error_c_d_rel"],
            "value_st_rel": cmp["error_st_rel"],
            "threshold": 0.02,
            "pass": bool(cmp["error_c_d_rel"] <= 0.02 and cmp["error_st_rel"] <= 0.02),
            "artefact": "artifacts/p40n02/cylinder_amr.json",
        }
        figs["fig5"] = fig_cylinder(cmp)

    results["criteria"]["f_pytest"] = {
        "value": "tests/test_amr_composite2d.py",
        "threshold": ">=15 tests, all green",
    }
    results["figures"] = figs
    out = os.path.join(ART, "verification_results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(json.dumps({k: v for k, v in results["criteria"].items()}, indent=2, default=float))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())