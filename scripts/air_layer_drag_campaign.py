#!/usr/bin/env python3
"""Campaign IN30-N03 runner: air-layer / micro-bubble drag-reduction closure.

Produces ``artifacts/air_layer_drag/results.json`` (all machine-checkable
numbers), figures under ``artifacts/air_layer_drag/`` and a markdown note under
``docs/``.

Usage::

    python scripts/air_layer_drag_campaign.py            # full campaign
    python scripts/air_layer_drag_campaign.py --quick    # cheap sanity run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.air_layer_drag.campaign import (  # noqa: E402
    ChannelConfig,
    air_layer_case,
    positive_boundary,
    single_phase_flux,
)
from tensorlbm.air_layer_drag.core import PhaseFieldParams  # noqa: E402
from tensorlbm.air_layer_drag.economics import (  # noqa: E402
    CompressionCase,
    compressor_power,
    net_power_gain,
)
from tensorlbm.air_layer_drag.validations import (  # noqa: E402
    laplace_gate,
    mass_conservation_gate,
    single_phase_cf,
    surface_tension,
)

OUT = ROOT / "artifacts" / "air_layer_drag"
DOCS = ROOT / "docs"


# ---------------------------------------------------------------------------
# workers
# ---------------------------------------------------------------------------
def _theta_worker(args):
    cfg_kw, coverage, thickness, baseline = args
    cfg = ChannelConfig(**cfg_kw)
    return air_layer_case(cfg, coverage=coverage, thickness=thickness,
                          baseline_flux=baseline)


def _qg_worker(args):
    cfg_kw, coverage, thickness, injection, baseline = args
    cfg = ChannelConfig(**cfg_kw)
    return air_layer_case(cfg, coverage=coverage, thickness=thickness,
                          injection=injection, vent=injection,
                          baseline_flux=baseline)


def _run_pool(fn, jobs, procs):
    if procs <= 1:
        return [fn(j) for j in jobs]
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(fn, jobs)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(res: dict) -> list[tuple[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = []
    lam = res["gates"]["b_laplace"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    rr = np.array([r["radius"] for r in lam["runs"]])
    dd = np.array([r["dp_measured"] for r in lam["runs"]])
    ax.plot(1.0 / rr, dd, "o", label="LBM static bubble")
    xs = np.linspace(0, 1.2 / rr.min(), 50)
    ax.plot(xs, lam["sigma_fit_from_slope"] * xs, "-",
            label=f"slope = sigma_eff = {lam['sigma_fit_from_slope']:.4g}")
    ax.plot(xs, lam["sigma_nominal"] * xs, "--",
            label=f"nominal sigma = {lam['sigma_nominal']:.4g}")
    ax.set_xlabel("1/R  [1/lattice]")
    ax.set_ylabel("pressure jump  dP [lattice]")
    ax.set_title("Gate (b): Laplace law, density ratio 100")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = OUT / "fig1_laplace_law.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(("fig1_laplace_law.png",
                 "Static-bubble pressure jump versus 1/R; the fitted slope is the "
                 "effective surface tension of the discrete model (density ratio 100)."))

    th = res["theta_sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    cov = [r["coverage_measured"] for r in th]
    dr = [r["drag_reduction"] for r in th]
    ax.plot(cov, dr, "o-", label="LBM 2-D channel")
    ax.set_xlabel("wall coverage theta [-]")
    ax.set_ylabel("drag reduction DR [-]")
    ax.set_title(f"DR vs coverage (h_g = {res['config']['thickness']} nodes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = OUT / "fig2_dr_vs_theta.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(("fig2_dr_vs_theta.png",
                 "Drag reduction versus wall coverage theta at fixed gas-layer thickness."))

    qg = res["qg_sweep"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    cq = [r["cq_inj"] for r in qg]
    dr2 = [r["drag_reduction"] for r in qg]
    ax.plot(cq, dr2, "s-", label="LBM (injection source + downstream vent)")
    ax.set_xlabel("C_q = Q_g /(U_b L) [-]")
    ax.set_ylabel("drag reduction DR [-]")
    ax.set_title("DR versus dimensionless gas injection rate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = OUT / "fig3_dr_vs_cq.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(("fig3_dr_vs_cq.png",
                 "Drag reduction versus dimensionless gas injection rate C_q; "
                 "saturation marks the break-even against compressor work."))

    eco = res["economics"]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for eta, rows in eco["rows"].items():
        ax.plot([r["gas_flow_ratio"] for r in rows],
                [r["dP_net_W"] / 1e3 for r in rows], "o-", label=f"eta_c = {eta}")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("C_q [-]")
    ax.set_ylabel("net power gain dP_net [kW]")
    ax.set_title("Net power ledger (100 m x 20 m plate, V = 10 m/s, h = 5 m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    p = OUT / "fig4_net_power.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    figs.append(("fig4_net_power.png",
                 "Net power gain versus gas flow for compressor efficiencies 0.5/0.6/0.7; "
                 "the zero crossing is the positive-gain boundary."))
    return figs


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--procs", type=int, default=min(8, os.cpu_count() or 1))
    ap.add_argument("--reuse", action="store_true",
                    help="reuse artifacts/air_layer_drag/{gates,sweeps}.json if present")
    args = ap.parse_args()

    t0 = time.time()
    nx, ny = (48, 32) if args.quick else (96, 40)
    spin, meas = (2000, 800) if args.quick else (6000, 2000)
    thickness = 6 if args.quick else 8
    cfg_kw = dict(nx=nx, ny=ny, spinup_steps=spin, measure_steps=meas)
    cfg = ChannelConfig(**cfg_kw)

    reuse_gates = OUT / "gates.json"
    reuse_sweeps = OUT / "sweeps.json"
    if args.reuse and reuse_gates.exists() and reuse_sweeps.exists():
        g = json.loads(reuse_gates.read_text())
        gate_a = g["gate_a"]
        gate_b0 = g["gate_b0"]
        gate_b = g["gate_b"]
        gate_c, gate_c_closed = g["gate_c"], g["gate_c_closed"]
        sw = json.loads(reuse_sweeps.read_text())
        base, theta_rows, qg_rows = sw["baseline"], sw["theta_sweep"], sw["qg_sweep"]
        # the artefact records the grid/thickness actually used by the sweeps
        cfg_kw = sw["config"]
        cfg = ChannelConfig(**cfg_kw)
        nx, ny = cfg_kw["nx"], cfg_kw["ny"]
        spin, meas = cfg_kw["spinup_steps"], cfg_kw["measure_steps"]
        thickness = sw["thickness"]
        print("[reuse] gates + sweeps loaded from artefacts", flush=True)
        return _finish(args, t0, nx, ny, spin, meas, thickness, cfg, gate_a, gate_b0,
                       gate_b, gate_c, gate_c_closed, base, theta_rows, qg_rows)

    print("[1/6] gate (a) single-phase friction", flush=True)
    gate_a = single_phase_cf(nx=8, ny=48, nu=cfg.nu_l, body_force_x=cfg.body_force_x,
                             steps=20000 if args.quick else 50000)

    print("[2/6] gate (b) surface tension + Laplace", flush=True)
    params = cfg.params
    gate_b0 = surface_tension(params=params, ny=64, nx=32, steps=2000 if args.quick else 4000)
    radii = (16.0, 24.0, 32.0)
    gate_b = laplace_gate(params, radii=radii, box=96,
                          steps=1500 if args.quick else 5000)
    # calibrate the discrete surface tension on R = 16 only, then predict R = 24, 32
    calib = gate_b["runs"][0]["sigma_effective"]
    calib_errs = [abs(r["dp_measured"] - calib / r["radius"]) / (calib / r["radius"])
                  for r in gate_b["runs"][1:]]
    gate_b["sigma_calibrated_at_R16"] = calib
    gate_b["calibration_ratio"] = calib / params.sigma
    gate_b["heldout_rel_errors"] = calib_errs
    gate_b["pass_5pct_calibrated"] = bool(len(calib_errs) > 0 and max(calib_errs) <= 0.05)

    print("[3/6] gate (c) mass conservation", flush=True)
    gate_c = mass_conservation_gate(params, box=64, steps=1500 if args.quick else 3000,
                                    n_samples=6, with_injection=True)
    gate_c_closed = mass_conservation_gate(params, box=64, steps=1500 if args.quick else 3000,
                                          n_samples=6, with_injection=False)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gates.json").write_text(json.dumps({
        "gate_a": gate_a, "gate_b0": gate_b0, "gate_b": gate_b,
        "gate_c": gate_c, "gate_c_closed": gate_c_closed}, indent=1))

    print("[4/6] baseline + theta sweep", flush=True)
    base = single_phase_flux(cfg)
    covs = (0.2, 0.4, 0.6, 0.8, 1.0)
    jobs = [(cfg_kw, c, thickness, base["mass_flux"]) for c in covs]
    theta_rows = _run_pool(_theta_worker, jobs, args.procs)

    return _finish(args, t0, nx, ny, spin, meas, thickness, cfg, gate_a, gate_b0,
                   gate_b, gate_c, gate_c_closed, base, theta_rows, None)


def _finish(args, t0, nx, ny, spin, meas, thickness, cfg, gate_a, gate_b0, gate_b,
            gate_c, gate_c_closed, base, theta_rows, qg_rows):
    print("[5/6] gas-injection sweep", flush=True)
    cfg_kw = dict(nx=nx, ny=ny, spinup_steps=spin, measure_steps=meas)
    if qg_rows is None:
        injs = (0.0, 2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3) if not args.quick else (0.0, 5.0e-4, 2.0e-3)
        jobs = [(cfg_kw, 1.0, thickness, q, base["mass_flux"]) for q in injs]
        qg_rows = _run_pool(_qg_worker, jobs, args.procs)

    print("[6/6] economics", flush=True)
    # dimensionless mapping: C_q from the simulation, DR from the same case
    # persist the expensive part before the (cheap) post-processing
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sweeps.json").write_text(json.dumps({
        "baseline": base, "theta_sweep": theta_rows, "qg_sweep": qg_rows,
        "config": cfg_kw, "thickness": thickness}, indent=1))
    curve = [(r["cq_inj"], r["drag_reduction"]) for r in qg_rows
             if r["drag_reduction"] is not None and r["cq_inj"] > 0]
    curve += [(r["cq"], r["drag_reduction"]) for r in theta_rows
              if r["drag_reduction"] is not None]
    curve = sorted((cq, min(max(dr, 0.0), 1.0)) for cq, dr in curve
                   if math.isfinite(cq) and math.isfinite(dr))
    case_kwargs = dict(speed=10.0, length=100.0, width=20.0, cf_1ph=3.0e-3,
                       submergence=5.0)
    econ = {"etas": [0.5, 0.6, 0.7], "rows": {}, "boundaries": {}, "curve": curve}
    for eta in (0.5, 0.6, 0.7):
        case = CompressionCase(eta_c=eta, **case_kwargs)
        rows = [net_power_gain(max(min(dr, 1.0), 0.0), cq, case) for cq, dr in curve]
        econ["rows"][str(eta)] = rows
        econ["boundaries"][str(eta)] = positive_boundary(rows)
    case06 = CompressionCase(eta_c=0.6, **case_kwargs)
    econ["reference_case"] = {
        "resistance_1ph_N": case06.resistance_1ph,
        "delivered_power_1ph_W": case06.delivered_power_1ph,
        "injection_pressure_Pa": case06.injection_pressure,
        "wetted_area_m2": case06.wetted_area,
        "eta_c": case06.eta_c,
    }
    # eta sensitivity of the positive domain width
    # analytic break-even gas flow: P_saved(C_q*) = P_comp(C_q*)
    dr_level = float(np.median([dr for _, dr in curve])) if curve else 0.0
    ln_r = math.log(case06.injection_pressure / case06.p_atm)
    q_area = case06.speed * case06.wetted_area
    econ["breakeven"] = {
        "dr_level": dr_level,
        "cq_star_by_eta": {
            eta: (dr_level * case06.cf_1ph * 0.5 * case06.rho_water * case06.speed ** 2
                * eta / (case06.p_atm * ln_r))
            for eta in (0.5, 0.6, 0.7)},
        "simulated_cq_range": [min(cq for cq, _ in curve), max(cq for cq, _ in curve)]
        if curve else None,
        "note": ("C_q is normalised by the wetted area Q_g/(V A); the 2-D channel "
                 "resolves h_g/H ~ 0.19, so its C_q is 1-2 orders above the values a "
                 "real ship needs (h_g/delta ~ 1e-3). The analytic break-even C_q* is "
                 "therefore below the simulated range and the simulated net power is "
                 "negative for all eta_c: reported honestly, not scaled."),
    }
    widths = {e: (b["cq_max_positive"] or 0.0) for e, b in econ["boundaries"].items()}
    econ["positive_domain_monotone_in_eta"] = bool(
        widths["0.5"] <= widths["0.6"] <= widths["0.7"])

    # monotonicity check for gate (d)
    dr_theta = [r["drag_reduction"] for r in theta_rows]
    theta_monotone = all(b > a for a, b in zip(dr_theta, dr_theta[1:]))
    dr_qg = [r["drag_reduction"] for r in qg_rows]
    cq_qg = [r["cq_inj"] for r in qg_rows]
    incr = all(b >= a - 1e-12 for a, b in zip(dr_qg, dr_qg[1:]))
    sat = (len(dr_qg) >= 3 and dr_qg[-1] - dr_qg[-2] < 0.5 * (dr_qg[-2] - dr_qg[0])
           and dr_qg[-1] < 0.95)

    literature_path = ROOT / "references" / "air_lubrication_reference_points.json"
    lit = None
    if literature_path.exists():
        raw = literature_path.read_bytes()
        lit = {"path": str(literature_path.relative_to(ROOT)),
               "sha256": hashlib.sha256(raw).hexdigest(),
               "payload": json.loads(raw.decode())}

    result = {
        "campaign": "IN30-N03",
        "module": "tensorlbm.air_layer_drag",
        "generated_unix": int(time.time()),
        "wall_clock_s": None,
        "config": {
            "grid": [nx, ny],
            "spinup_steps": spin,
            "measure_steps": meas,
            "thickness": thickness,
            "body_force_x": cfg.body_force_x,
            "nu_l": cfg.nu_l,
            "nu_g": cfg.nu_g,
            "rho_l": cfg.rho_l,
            "rho_g": cfg.rho_g,
            "density_ratio": cfg.params.density_ratio,
            "mu_ratio": cfg.params.mu_g_over_mu_l,
            "sigma": cfg.sigma,
            "interface_width": cfg.interface_width,
            "mobility": cfg.mobility,
            "contact_angle": "90 deg (neutral wetting, no-flux phase BC)",
        },
        "gates": {
            "a_single_phase_cf": gate_a,
            "b0_surface_tension": gate_b0,
            "b_laplace": gate_b,
            "c_mass_injection": gate_c,
            "c_mass_closed": gate_c_closed,
            "d_monotone_theta": theta_monotone,
            "d_monotone_qg": incr,
            "d_saturation": bool(incr and sat),
            "e_literature": lit,
            "f_economics": {
                "eta_c": [0.5, 0.6, 0.7],
                "positive_domain_monotone_in_eta": econ["positive_domain_monotone_in_eta"],
                "positive_boundaries": econ["boundaries"],
            },
        },
        "baseline": base,
        "theta_sweep": theta_rows,
        "qg_sweep": qg_rows,
        "economics": econ,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    figs = make_figures(result)
    result["figures"] = [{"file": n, "caption": c} for n, c in figs]
    result["wall_clock_s"] = round(time.time() - t0, 1)
    (OUT / "results.json").write_text(json.dumps(result, indent=1))
    print("wrote", OUT / "results.json")
    print("wall clock", result["wall_clock_s"], "s")
    print("gate a rel err  :", round(gate_a["cf_rel_error"], 5))
    print("gate b laplace  :", round(gate_b["max_rel_error_vs_reference"], 4),
          "heldout(cal R16):", [round(e, 4) for e in gate_b["heldout_rel_errors"]])
    print("gate c drift    :", gate_c["mass_rel_drift"], gate_c["ledger"]["ledger_rel_residual"])
    print("DR(theta)       :", [round(x, 4) for x in dr_theta])
    print("DR(Cq_inj)      :", [round(x, 4) for x in dr_qg], "Cq:", [round(c, 4) for c in cq_qg])
    print("econ boundaries :", econ["boundaries"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())