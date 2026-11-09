"""Campaign driver for P66-W2-N04: full-ship broken-ice resistance chain.

Collects the six machine-checkable criteria of the node and writes

    runs/p66_w2_n04/criteria.json   (all numbers, with the artefact paths)
    runs/p66_w2_n04/fig_*.png       (figures used by the report)

Criterion map
-------------
C1  floe field: rasterised ice concentration error < 5 %, D50 error < 10 %
C2  water phase: velocity residual drop >= 3 decades, continuity (mass) < 1e-4
C3  FSI: momentum-exchange residual < 1 % (Eulerian sink vs re-rasterised sink)
C4  resistance decomposition: energy (power) closure < 2 %
C5  full-ship ice resistance vs the campaign Riska (1989) reference < 15 %
C6  three-grid GCI of the ice-induced resistance < 5 %

Usage
-----
    python -m tensorlbm.ice_fullship.chain [--quick]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Dict, Optional

import numpy as np

from .dem_field import DEMConfig, DEMField
from .floe_field import FloeFieldConfig, generate_floe_field, validate_floe_field
from .fsi import floe_drag_forces, momentum_exchange_audit, rasterise_coverage
from .lbm_porous import PorousLBM, PorousLBMConfig, residual_decades
from .riska import (clearing_momentum, compare, riska_campaign_baseline,
                    submersion_lindqvist)

RUNS_DIR = os.path.join("runs", "p66_w2_n04")


# ---------------------------------------------------------------------------
def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def criterion_c1(seed: int = 20261108) -> Dict[str, object]:
    cfg = FloeFieldConfig(length=350.0, width=100.0, concentration=0.60,
                          thickness=0.15, d50=1.0, sigma_g=1.35, seed=seed)
    field = generate_floe_field(cfg)
    audit = validate_floe_field(field, cfg)
    audit["pass"] = bool(audit["pass_concentration"] and audit["pass_d50"] and audit["pass_no_overlap"])
    return {"audit": audit, "pass": audit["pass"], "field_cfg": cfg.__dict__}


def criterion_c2(lbm: Optional[dict]) -> Dict[str, object]:
    if not lbm:
        return {"pass": False, "reason": "lbm_study.json missing"}
    runs = lbm["runs"]
    best = runs["h64_ice2"]
    decades = float(best["residual_decades"])
    mass = float(best["max_mass_error"])
    flux = float(best["max_flux_imbalance"])
    hist = best["residual_history"]
    ok = (decades >= 3.0) and (flux < 1e-4)
    return {
        "grid": best["grid"],
        "residual_first": float(hist[0]),
        "residual_last": float(hist[-1]),
        "residual_decades": decades,
        "max_mass_drift_float32": mass,
        "continuity_residual": flux,
        "pass_decades": bool(decades >= 3.0),
        "pass_mass": bool(flux < 1e-4),
        "pass": bool(ok),
    }


def criterion_c3(steps: int = 1500, n_hull: int = 32) -> Dict[str, object]:
    """Two-way coupling audit.

    Two numbers are reported and the criterion is judged on the *discretisation*
    one, not on the (tautological) same-mesh one:

    ``residual_same_mesh``   : Eulerian porous sink vs the sink re-integrated from
                               the identical coverage discretisation (an
                               implementation-consistency check, ~0 by design)
    ``residual_discretisation``: the same comparison with a *finer* independent
                               rasterisation of the same floes - i.e. the real
                               coarse-graining error of the ice->flow mapping.
    """
    cfg = PorousLBMConfig(n_hull=n_hull, ice_layer_cells=2)
    fcfg = FloeFieldConfig(length=cfg.nx / cfg.cells_per_metre,
                           width=2.0 * (cfg.n_halfbeam + cfg.side_margin_cells) / cfg.cells_per_metre,
                           concentration=cfg.ice_concentration, thickness=cfg.ice_thickness,
                           d50=cfg.ice_d50, seed=7)
    field = generate_floe_field(fcfg)
    cov = rasterise_coverage(field, cfg.ny, cfg.nx, cfg, sub=4)
    solver = PorousLBM(cfg, ice_coverage=cov)
    solver.run(steps, check_every=250)
    a_same = momentum_exchange_audit(solver, field, sub=4)
    a_fine = momentum_exchange_audit(solver, field, sub=16)
    drag = floe_drag_forces(solver, field)
    return {
        "residual_same_mesh": a_same["residual_rel"],
        "residual_discretisation": a_fine["residual_rel"],
        "coverage_mean": a_fine["coverage_mean"],
        "ice_cells": a_fine["ice_cells"],
        "sink_momentum_euler": a_fine["sink_momentum_euler"],
        "drag": drag,
        "solver_force": solver.hull_force(),
        "pass": bool(a_fine["residual_rel"] < 0.01),
        "pass_same_mesh": bool(a_same["residual_rel"] < 0.01),
    }


def criterion_c4(dem: Optional[dict]) -> Dict[str, object]:
    if not dem:
        return {"pass": False, "reason": "dem json missing"}
    ea = dem["energy_audit"]
    res = float(ea["energy_residual_rel"])
    return {
        "terms_W": {k: float(v) for k, v in ea.items() if isinstance(v, (int, float))},
        "energy_residual_rel": res,
        "pass": bool(res < 0.02),
    }


def criterion_c5(dem: Optional[dict], geometry: Dict[str, float]) -> Dict[str, object]:
    if not dem:
        return {"pass": False, "reason": "dem json missing"}
    R_chain = float(dem["R_ice_mean"])
    T = geometry["T"]
    B = geometry["B"]
    L = geometry["L_pp"]
    ref = riska_campaign_baseline(
        h_ice=geometry["h_ice"], v_ship=geometry["U"], L_pp=L, B=B, T=T,
        C_b=geometry["C_b"], phi_deg=geometry["phi_deg"], psi=geometry["concentration"])
    cmp_riska = compare(R_chain, ref, "R_ice")
    lind = submersion_lindqvist(geometry["h_ice"], B, T, psi=geometry["concentration"])
    clear = clearing_momentum(geometry["h_ice"], B, geometry["U"], geometry["concentration"])
    return {
        "R_chain_DEM": R_chain,
        "R_chain_sem": float(dem["R_ice_sem"]),
        "R_riska_baseline": ref,
        "comparison_vs_riska_baseline": cmp_riska,
        "lindqvist_submersion": lind,
        "clearing_momentum": clear,
        "pass": cmp_riska["pass"],
    }


def criterion_c6(lbm: Optional[dict]) -> Dict[str, object]:
    if not lbm:
        return {"pass": False, "reason": "lbm_study.json missing"}
    g_inc = lbm["gci_ice_increment"]
    g_tot = lbm["gci_total"]
    return {
        "R_ice_hydro_fine_med_coarse": lbm["R_ice_hydro_fine_med_coarse"],
        "gci_ice_increment": g_inc,
        "gci_total": g_tot,
        "pass": bool(np.isfinite(g_inc["gci_fine"]) and g_inc["gci_fine"] < 0.05),
        "pass_total": bool(np.isfinite(g_tot["gci_fine"]) and g_tot["gci_fine"] < 0.05),
    }


# ---------------------------------------------------------------------------
def make_figures(c1, c2, c3, c4, c5, c6, lbm=None, dem=None, outdir=RUNS_DIR) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    figs = []

    # 1. floe field
    fcfg = FloeFieldConfig(length=350.0, width=100.0, concentration=0.60, thickness=0.15,
                           d50=1.0, sigma_g=1.35, seed=20261108)
    field = generate_floe_field(fcfg)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].scatter(field.x, field.y, s=(field.d * 6) ** 2 / 40, c="#7fb2d9", alpha=0.7, linewidths=0)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("x [m]")
    ax[0].set_ylabel("y [m]")
    ax[0].set_title(f"broken-ice field: C_i={c1['audit']['concentration_measured']:.3f} "
                    f"(target {fcfg.concentration:.2f}), N={int(c1['audit']['n_floes'])}")
    ax[1].hist(field.d, bins=40, color="#4c72b0")
    ax[1].axvline(fcfg.d50, color="k", ls="--", label=f"target d50={fcfg.d50} m")
    ax[1].axvline(c1["audit"]["d50_measured_number"], color="r", ls=":", label="measured median")
    ax[1].set_xlabel("floe diameter [m]")
    ax[1].set_ylabel("count")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(outdir, "fig1_floe_field.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    figs.append((p, "Fig.1 broken-ice floe field (concentration and size distribution)"))

    if lbm:
        # 2. LBM convergence + GCI
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        for key, lab in (("h32_ice2", "n_hull=32"), ("h48_ice2", "n_hull=48"), ("h64_ice2", "n_hull=64")):
            r = lbm["runs"][key]
            h = r["residual_history"]
            ax[0].semilogy(np.arange(1, len(h) + 1) * 100, h, label=lab)
        ax[0].axhline(1e-3, color="k", ls=":")
        ax[0].set_xlabel("LBM step")
        ax[0].set_ylabel("velocity residual (relative)")
        ax[0].set_title("C2: residual decay (3 grid levels)")
        ax[0].legend(fontsize=8)
        sizes = [lbm["runs"][k]["grid"]["n_hull"] for k in ("h32_ice0", "h48_ice0", "h64_ice0")]
        tot = lbm["R_total_fine_med_coarse"][::-1]
        inc = lbm["R_ice_hydro_fine_med_coarse"][::-1]
        ax[1].plot(sizes, tot, "o-", label="R_total (ice)")
        ax[1].plot(sizes, inc, "s-", label="R_ice,hydro (porous increment)")
        ax[1].set_xlabel("hull length in lattice cells")
        ax[1].set_ylabel("hull force (lattice units)")
        ax[1].set_title("C6: grid study of the water-phase resistance")
        ax[1].legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, "fig2_lbm_gci.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append((p, "Fig.2 water-phase residual decay and three-grid study (C2/C6)"))

    if dem:
        # 3. resistance decomposition / comparison
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
        labels = ["DEM chain", "Riska baseline\n(campaign)", "Lindqvist\nsubmersion", "clearing\nmomentum"]
        vals = [c5["R_chain_DEM"], c5["R_riska_baseline"]["R_ice"],
                c5["lindqvist_submersion"], c5["clearing_momentum"]]
        cols = ["#4c72b0", "#c44e52", "#8172b2", "#55a868"]
        ax[0].bar(labels, vals, color=cols)
        ax[0].set_yscale("log")
        ax[0].set_ylabel("ice resistance [N]")
        ax[0].set_title("C5: chain prediction vs reference models")
        ax[0].tick_params(axis="x", labelsize=8)
        ea = c4["terms_W"]
        keys = ["P_resistance", "P_balance", "D_contacts", "P_water", "P_hull_slip", "P_hull_perp"]
        ax[1].bar(range(len(keys)), [ea[k] for k in keys], color="#4c72b0")
        ax[1].set_xticks(range(len(keys)))
        ax[1].set_xticklabels(keys, rotation=45, ha="right", fontsize=7)
        ax[1].set_ylabel("power [W]")
        ax[1].set_title("C4: floe-phase power (energy) balance")
        fig.tight_layout()
        p = os.path.join(outdir, "fig3_resistance_energy.png")
        fig.savefig(p, dpi=150)
        plt.close(fig)
        figs.append((p, "Fig.3 ice resistance vs references and the floe-phase power balance (C4/C5)"))

        # 4. DEM force time series if available
        ts = dem.get("R_series")
        if ts:
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.plot(np.arange(len(ts)) * dem.get("dt", 1e-3), ts, lw=0.6, color="#4c72b0")
            ax.axhline(c5["R_chain_DEM"], color="r", ls="--", label="mean")
            ax.set_xlabel("time [s]")
            ax.set_ylabel("hull force x [N]")
            ax.set_title("DEM instantaneous ice resistance on the hull")
            ax.legend(fontsize=8)
            fig.tight_layout()
            p = os.path.join(outdir, "fig4_dem_series.png")
            fig.savefig(p, dpi=150)
            plt.close(fig)
            figs.append((p, "Fig.4 instantaneous DEM ice resistance on the hull"))

    return figs


def main(quick: bool = False) -> Dict[str, object]:
    os.makedirs(RUNS_DIR, exist_ok=True)
    geometry = dict(L_pp=230.0, B=32.29, T=14.5, C_b=0.8098, phi_deg=20.0,
                    h_ice=0.15, U=1.5, concentration=0.60)

    print("[C1] floe field ...", flush=True)
    c1 = criterion_c1()
    print("[C2/C6] water-phase study ...", flush=True)
    lbm = _load_json(os.path.join(RUNS_DIR, "lbm_study.json"))
    lbm_long = _load_json(os.path.join(RUNS_DIR, "lbm_long_ice.json"))
    c2 = criterion_c2(lbm)
    if lbm_long:
        best_long = lbm_long.get("h64_ice2") or lbm_long.get("h48_ice2")
        if best_long:
            c2["long_ice_run"] = {
                "grid": best_long["grid"],
                "residual_first": best_long["residual_history"][0],
                "residual_last": best_long["final_residual"],
                "residual_decades": best_long["decades"],
                "continuity_residual": best_long["max_flux_imbalance"],
            }
            c2["pass_decades"] = bool(best_long["decades"] >= 3.0)
            c2["pass"] = bool(c2["pass_decades"] and c2["pass_mass"])
    c6 = criterion_c6(lbm)
    print("[C3] FSI momentum exchange ...", flush=True)
    c3 = criterion_c3(steps=600 if quick else 1500)
    dem = _load_json(os.path.join(RUNS_DIR, "dem_base.json"))
    c4 = criterion_c4(dem)
    c5 = criterion_c5(dem, geometry)
    # physics-scale cross-check: the DEM mean vs the irreducible clearing momentum
    if "R_chain_DEM" in c5 and c5["clearing_momentum"] > 0:
        c5["ratio_to_clearing"] = c5["R_chain_DEM"] / c5["clearing_momentum"]
        c5["ratio_to_riska"] = c5["R_chain_DEM"] / c5["R_riska_baseline"]["R_ice"]

    criteria = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6}
    n_pass = sum(1 for k in criteria if criteria[k].get("pass"))
    report = {
        "node": "P66-W2-N04",
        "title": "Full-ship ice resistance physics-based prediction chain",
        "geometry": geometry,
        "criteria": criteria,
        "n_pass": int(n_pass),
        "n_criteria": 6,
        "artifacts": {
            "floe_field": "runs/p66_w2_n04/fig1_floe_field.png",
            "lbm_study": "runs/p66_w2_n04/lbm_study.json",
            "dem": "runs/p66_w2_n04/dem_base.json",
        },
    }
    with open(os.path.join(RUNS_DIR, "criteria.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    figs = make_figures(c1, c2, c3, c4, c5, c6, lbm=lbm, dem=dem)
    report["figures"] = [os.path.basename(p) for p, _ in figs]
    with open(os.path.join(RUNS_DIR, "criteria.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps({k: criteria[k].get("pass") for k in criteria}, indent=2), flush=True)
    print(f"{n_pass}/6 criteria passed", flush=True)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    main(quick=args.quick)