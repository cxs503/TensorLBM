"""Campaign driver: microbubble drag reduction closure (IN62-N04).

Runs the three solver levels, grades them against their anchors, closes the
net-power ledger and writes everything to ``artifacts/microbubble_in62_n04``:

``result.json``    every number quoted in the report (programmatic, no hand copy)
``profiles.npz``   void-fraction / velocity profiles
``figures.json``   base64 PNGs with captions
``criteria.json``  machine-readable pass/fail of the acceptance criteria
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .boundary_layer import BLParams, drag_reduction_re_theta, log_law_check
from .lbm import ChannelConfig, MixtureChannelLBM, two_layer_drag_reduction
from .loglaw import B_INTERCEPT, KAPPA, drag_reduction_layer
from .power import CompressorSpec, ledger_scan, profitability_threshold
from .transport import (
    TransportConfig,
    TransportFV,
    TransportLBM,
    alpha_profile_analytic,
    analytic_supply,
    mass_ledger,
)

NU_WATER = 1.5e-5          # kinematic viscosity of sea water at 15 C [m^2/s]
RE_THETA_GRID = (1000.0, 2000.0, 3000.0, 5000.0)
ALPHA_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def _transport_case(u_tau: float, delta: float, alpha_w: float,
                    ny: int = 64) -> dict:
    """Void transport verification in lattice units: closed form vs FV vs LBM.

    The *physical* transport problem has ``D ~ 1e-4 m^2/s`` on a layer of
    ``1 mm``, i.e. a lattice diffusivity ``cs^2(tau-1/2) ~ 1e-4`` -- an LBM run
    at that relaxation time would need ~1e8 steps.  The verification is
    therefore done at the same Peclet number (``Pe = 2 c_e Sc_t / kappa``, which
    is scale free) with ``D = 0.2`` in lattice units.
    """
    cfg = TransportConfig(ny=ny, H=32.0, D=0.2, alpha_w=alpha_w,
                          v_sweep=0.00275)
    y = (np.arange(cfg.ny) + 0.5) * cfg.dy
    ana = alpha_profile_analytic(cfg, y)
    fv = TransportFV(cfg)
    fv.run(20000)
    lbm = TransportLBM(cfg)
    lbm.run(120000, tol=1e-13)
    p_fv = fv.profile()
    p_lb = lbm.profile()
    scale = max(cfg.alpha_w, 1e-12)
    led = mass_ledger(fv)
    return {
        "Pe": cfg.peclet,
        "physical_Pe": (2.0 * 0.1 * 0.9 / KAPPA),
        "H_over_delta": cfg.H / delta,
        "D": cfg.D,
        "v_sweep": cfg.v_sweep,
        "alpha_w": alpha_w,
        "y": y.tolist(),
        "alpha_analytic": ana.tolist(),
        "alpha_fv": p_fv.tolist(),
        "alpha_lbm": p_lb.tolist(),
        "err_fv_max": float(np.max(np.abs(p_fv - ana)) / scale),
        "err_lbm_max": float(np.max(np.abs(p_lb - ana)) / scale),
        "err_lbm_max_interior": float(np.max(np.abs(p_lb[2:-2] - ana[2:-2])) / scale),
        "ledger_fv": led,
        "supply": fv.supply(),
        "supply_ratio": fv.supply()["supply"] / u_tau,
    }


def _laminar_lbm_validation() -> dict:
    """Two-phase mixture LBM channel vs the exact two-layer solution."""
    cases = []
    base = None
    for cfg in (ChannelConfig(ny=32),
                ChannelConfig(ny=32, alpha_layer=0.5, h_layer=5.0),
                ChannelConfig(ny=32, alpha_layer=0.2, h_layer=10.0)):
        lb = MixtureChannelLBM(cfg)
        lb.run(25000)
        ex = two_layer_drag_reduction(cfg.H, cfg.U_target, cfg.nu_liquid,
                                      cfg.alpha_layer, cfg.h_layer,
                                      cfg.nu_gas_ratio)
        if base is None:
            base = lb.G
        dr_lbm = 1.0 - lb.G / base
        cases.append({
            "alpha_layer": cfg.alpha_layer,
            "h_layer": cfg.h_layer,
            "h_over_H": cfg.h_layer / cfg.H,
            "U_bulk": lb.bulk_velocity(),
            "G": lb.G,
            "DR_lbm": dr_lbm,
            "DR_exact": ex["DR"],
            "rel_error_DR": (dr_lbm - ex["DR"]) / ex["DR"] if ex["DR"] else 0.0,
            "tau_w_ladd": lb.wall_shear_ladd(),
            "tau_w_exact": ex["bubbly"]["tau_w_bottom"] if ex["DR"] else
            ex["clean"]["tau_w_bottom"],
            "y": (lb.y / cfg.H).tolist(),
            "u_over_U": (lb.profile() / cfg.U_target).tolist(),
            "u_exact_over_U": (ex["bubbly"]["u"] / cfg.U_target).tolist(),
        })
    return {"cases": cases}


def _turbulent_route(re_theta: float, alpha_w: float,
                     params: BLParams) -> dict:
    """Clean layer -> void profile -> BL solve + profile-aware log law."""
    clean = drag_reduction_re_theta(re_theta, alpha_w=0.0, p=params)
    bl = clean["clean"]
    delta = float(bl["y"][-1])
    u_tau = float(bl["u_tau"])
    nu = NU_WATER
    if alpha_w <= 0.0:
        return {
            "re_theta": clean["re_theta_clean"],
            "delta": delta,
            "u_tau": u_tau,
            "cf_clean": clean["cf_clean"],
            "DR_bl": 0.0,
            "DR_ana": 0.0,
            "DR_mixing_length": 0.0,
            "DR_loglaw": 0.0,
            "alpha_buffer": 0.0,
            "supply_ratio": 0.0,
        }
    cfg = TransportConfig(alpha_w=alpha_w).with_flow(u_tau, delta)

    def alpha_of_y(yy):
        return alpha_profile_analytic(cfg, yy)

    bub = drag_reduction_re_theta(re_theta, alpha_w=alpha_w,
                                  alpha_of_y=alpha_of_y, p=params)
    ana = drag_reduction_layer(alpha_w, alpha_of_y, u_tau, nu, delta)
    supply = analytic_supply(cfg)
    a_buf = float(np.atleast_1d(alpha_of_y(min(30.0 * nu / u_tau, cfg.H)))[0])
    return {
        "re_theta": bub["re_theta_clean"],
        "delta": delta,
        "u_tau": u_tau,
        "cf_clean": bub["cf_clean"],
        "cf_bubbly": bub["cf_bubbly"],
        "DR_bl": bub["DR_total"],
        "DR_mixing_length": bub["DR_mixing_length"],
        "DR_ana": ana["DR"],
        "DR_loglaw": ana["DR_loglaw"],
        "kappa_eff_layer": ana["kappa_eff"],
        "alpha_buffer": a_buf,
        "supply_ratio": supply["supply"] / u_tau,
        "Pe": cfg.peclet,
    }


def run_campaign(out_dir: str = "artifacts/microbubble_in62_n04",
                 re_theta_grid=RE_THETA_GRID, alpha_grid=ALPHA_GRID,
                 params: BLParams | None = None,
                 ship_speed: float = 10.0, plate_length: float = 100.0,
                 make_figures: bool = True) -> dict:
    """Run the full campaign and write the artifacts."""
    params = params or BLParams(n=600)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # -- level 1: laminar two-phase LBM vs exact ---------------------------
    lam = _laminar_lbm_validation()

    # -- level 2: void-fraction transport ---------------------------------
    # reference flow: U = 1 m/s, Re_theta = 2000 (u_tau and delta from the
    # boundary-layer solve, not assumed)
    ref = drag_reduction_re_theta(2000.0, alpha_w=0.0, p=params)["clean"]
    u_tau_ref = float(ref["u_tau"])
    delta_ref = float(ref["y"][-1])
    transport = _transport_case(u_tau_ref, delta_ref, 0.2)

    # -- level 3: turbulent DR grid ---------------------------------------
    table = []
    by_alpha: dict[float, dict] = {}
    for rt in re_theta_grid:
        for aw in alpha_grid:
            row = _turbulent_route(float(rt), float(aw), params)
            row["alpha_w"] = float(aw)
            table.append(row)
            by_alpha.setdefault(float(aw), row)

    # -- anchors and criteria ---------------------------------------------
    dr_pairs = [(r["DR_bl"], r["DR_ana"]) for r in table if r["alpha_w"] > 0]
    gap = [abs(a - b) / b if b > 0 else 0.0 for a, b in dr_pairs]
    gap_max = float(max(gap)) if gap else 0.0

    dr_lam_err = max(abs(c["rel_error_DR"]) for c in lam["cases"][1:])
    supply_of = {aw: row["supply_ratio"] for aw, row in by_alpha.items()}

    dr_of = {aw: by_alpha[aw]["DR_bl"] for aw in by_alpha}
    dr_ana_of = {aw: by_alpha[aw]["DR_ana"] for aw in by_alpha}

    a_keys = sorted(k for k in dr_of if k > 0)

    def _lookup(table_: dict, a: float) -> float:
        key = min(table_, key=lambda kk: abs(kk - a))
        return table_[key]

    ledger = ledger_scan(
        U=ship_speed, length=plate_length,
        alpha_grid=a_keys,
        dr_of=lambda a: _lookup(dr_of, a),
        supply_of=lambda a: _lookup(supply_of, a),
        op=CompressorSpec(),
    )

    near = min(dr_of, key=lambda k: abs(k - 0.2))
    thr = profitability_threshold(ship_speed, ledger["tau_w0"], dr_of[near],
                                  supply_of[near], CompressorSpec())
    thr_recirc = {a: profitability_threshold(ship_speed, ledger["tau_w0"],
                                             dr_of[a], supply_of[a],
                                             CompressorSpec())
                  for a in a_keys}
    ledger_pos_recirc = ledger_scan(
        U=ship_speed, length=plate_length, alpha_grid=a_keys,
        dr_of=lambda a: _lookup(dr_of, a),
        supply_of=lambda a: _lookup(supply_of, a), op=CompressorSpec(),
        recirculation=thr["recirculation_min"],
    )

    criteria = {
        "C1_laminar_lbm_vs_exact": {
            "description": "two-phase mixture LBM DR vs exact two-layer solution",
            "max_rel_error": dr_lam_err,
            "threshold": 0.05,
            "pass": bool(dr_lam_err < 0.05),
        },
        "C2_transport_lbm_vs_llosed_form": {
            "description": "void profile: FV vs closed form (primary route)",
            "fv_max_rel_error": transport["err_fv_max"],
            "threshold": 0.03,
            "pass": bool(transport["err_fv_max"] < 0.03),
        },
        "C2b_transport_lbm": {
            "description": "void profile: D2Q9 advection-diffusion LBM vs closed form",
            "lbm_max_rel_error": transport["err_lbm_max"],
            "threshold": 0.03,
            "pass": bool(transport["err_lbm_max"] < 0.03),
        },
        "C3_mass_ledger": {
            "description": "|injected - captured - outlet| / injected",
            "closure_error": abs(transport["ledger_fv"]["closure_error"]),
            "threshold": 0.02,
            "pass": bool(abs(transport["ledger_fv"]["closure_error"]) < 0.02),
        },
        "C4_two_route_dr_agreement": {
            "description": "DR: mixing-length BL route vs profile-aware log law",
            "max_rel_gap": gap_max,
            "threshold": 0.20,
            "pass": bool(gap_max < 0.20),
        },
        "C5_net_power_positive": {
            "description": "P_net > 0 for eta_c > 0.5 at some alpha_w in 0-0.3",
            "positive_fraction": ledger["positive_fraction"],
            "best_alpha_w": ledger["best_alpha_w"],
            "best_net_power_density": ledger["best_net_power_density"],
            "pass": bool(ledger["positive_fraction"] > 0.0),
        },
        "C5b_profitability_design_limit": {
            "description": ("gas recovery needed to make the layer pay: "
                            "recirculation_min and C_q_max at alpha_w = 0.2"),
            "alpha_w": near,
            "C_q": thr["C_q"],
            "C_q_max": thr["C_q_max"],
            "recirculation_min": thr["recirculation_min"],
            "flow_ratio": thr["flow_ratio"],
        },
    }

    results = {
        "node": "IN62-N04",
        "bl_log_law_check": log_law_check(2000.0, params),
        "bl_log_law_check_5000": log_law_check(5000.0, params),
        "module": "tensorlbm.microbubble",
        "closure": {"kappa": KAPPA, "B": B_INTERCEPT,
                    "kappa_eff_law": "kappa*sqrt(1-alpha)"},
        "sigma_water_nu": NU_WATER,
        "laminar_lbm": lam,
        "transport": transport,
        "turbulent": {
            "re_theta_grid": list(re_theta_grid),
            "alpha_grid": list(alpha_grid),
            "rows": table,
        },
        "power": ledger,
        "power_thresholds": thr_recirc,
        "power_near_alpha02": thr,
        "power_with_recirculation": ledger_pos_recirc,
        "criteria": criteria,
        "ship": {"speed": ship_speed, "plate_length": plate_length},
    }
    (out / "result.json").write_text(json.dumps(results, indent=2))
    (out / "criteria.json").write_text(json.dumps(criteria, indent=2))
    np.savez(
        out / "profiles.npz",
        y_transport=np.asarray(transport["y"]),
        alpha_analytic=np.asarray(transport["alpha_analytic"]),
        alpha_fv=np.asarray(transport["alpha_fv"]),
        alpha_lbm=np.asarray(transport["alpha_lbm"]),
        y_laminar=np.asarray(lam["cases"][1]["y"]),
        u_laminar=np.asarray(lam["cases"][1]["u_over_U"]),
        u_laminar_exact=np.asarray(lam["cases"][1]["u_exact_over_U"]),
    )

    if make_figures:
        from .figures import make_all
        figs = make_all(results, out)
        (out / "figures.json").write_text(json.dumps(figs, indent=2))
        results["figures"] = [{"name": f["name"], "caption": f["caption"],
                               "path": f.get("path")} for f in figs]
        (out / "result.json").write_text(json.dumps(results, indent=2))

    return results


if __name__ == "__main__":   # pragma: no cover
    res = run_campaign()
    print(json.dumps(res["criteria"], indent=2))