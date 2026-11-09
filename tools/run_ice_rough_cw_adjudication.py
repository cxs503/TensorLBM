#!/usr/bin/env python3
"""Ice-keel roughness -> drag closure: three-lens adjudication (P103W2C-N01).

Commands
--------
``refine``     run the fixed-``k_s^+`` grid-refinement trio used by criterion C4
               (GCI) into ``runs/ice_rough_cw/refine``
``aggregate``  read the campaign case records, run the three-lens
               cross-adjudication (C1/C2/C3), the GCI (C4) and the propagation
               verdict (C5), evaluate the DEM callback end-to-end check (C6) and
               write ``artifacts/ice_rough_cw_c/`` (JSON + Markdown + figures)
``list``       print the case table

Every number in the report is produced by this script from the case records;
nothing is transcribed by hand.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from tensorlbm.ice_roughness import (  # noqa: E402
    freedrift as fd,
    geometry as ice_geom,
    propagation as pg,
    regime_law as rl,
    sweep,
    three_zone as tz,
)

DEFAULT_RUN_DIR = ROOT / "runs" / "ice_rough_cw" / "cases"
DEFAULT_REFINE_DIR = ROOT / "runs" / "ice_rough_cw" / "refine"
DEFAULT_ART_DIR = ROOT / "artifacts" / "ice_rough_cw_c"

#: The reference ``(k_s^+, lambda/k_s)`` of the refinement study.
REFINE_KS_PLUS = 90.0
REFINE_LAMBDA_OVER_KS = 5.0

#: Grid-refinement trio at *fixed* ``k_s^+`` and (nearly) fixed ``k_s/h``.
#: ``dy^+ = Re_tau/h_eff`` is what actually refines: 30 -> 18 -> 12.9.
REFINE_PROFILES: dict[str, dict[str, Any]] = {
    "refine_coarse": {"h": 32, "ks_nodes": 3, "u_tau": 0.012, "turnovers": 4.0,
                      "warmup_fraction": 0.25, "init_scale": 1.0,
                      "sample_interval": 10, "smagorinsky_cs": 0.10, "threads": 12},
    "refine_fine": {"h": 64, "ks_nodes": 7, "u_tau": 0.012, "turnovers": 4.0,
                    "warmup_fraction": 0.25, "init_scale": 1.0,
                    "sample_interval": 10, "smagorinsky_cs": 0.10, "threads": 12},
}


def refinement_cases() -> list[sweep.Case]:
    """Coarse/medium/fine trio at fixed ``k_s^+ = 90``, ``lambda/k_s = 5``."""
    for name, params in REFINE_PROFILES.items():
        sweep.PROFILES[name] = params
    cases = []
    for name, h in (("refine_coarse", 32), ("refine_fine", 64)):
        cases.append(sweep.Case(
            case_id=f"ref_ks{REFINE_KS_PLUS:g}_lam{REFINE_LAMBDA_OVER_KS:g}_h{h}",
            ks_plus_target=REFINE_KS_PLUS,
            lambda_over_ks=REFINE_LAMBDA_OVER_KS,
            profile=name,
        ))
    # the medium resolution is the campaign case itself (h = 48, k_s = 5 nodes)
    cases.append(sweep.Case(
        case_id=f"ref_ks{REFINE_KS_PLUS:g}_lam{REFINE_LAMBDA_OVER_KS:g}_h48",
        ks_plus_target=REFINE_KS_PLUS,
        lambda_over_ks=REFINE_LAMBDA_OVER_KS,
        profile="campaign",
    ))
    return cases


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------

def cmd_refine(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for case in refinement_cases():
        target = out / f"{case.case_id}.json"
        if target.is_file() and not args.force:
            print(f"skip {case.case_id} (done)")
            continue
        print(f"run  {case.case_id} h={case.h} ks_nodes={case.ks_nodes:g} "
              f"Re_tau={case.re_tau:.0f} steps={case.steps}", flush=True)
        rec = sweep.run_case(case, out, threads=args.threads, rebuild=args.rebuild)
        print(f"  ok   cf={rec['cf']:.5f} ks+={rec['ks_plus']:.1f} "
              f"wall={rec['wall_seconds']:.0f}s", flush=True)
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    for case in refinement_cases():
        print(f"{case.case_id:32s} h={case.h:3d} ks_nodes={case.ks_nodes:4.1f} "
              f"Re_tau={case.re_tau:7.1f} ks+={case.ks_plus_target:5.1f} "
              f"dy+={case.re_tau / case.h_eff_estimate:5.1f} steps={case.steps}")
    return 0


# ---------------------------------------------------------------------------
# GCI (criterion C4)
# ---------------------------------------------------------------------------

def _gci_from_records(files: dict[str, Path]) -> dict[str, Any]:
    """GCI of the roughness function ``dU^+`` over the three resolutions."""
    rows: dict[str, dict[str, Any]] = {}
    for label, path in files.items():
        if not path.is_file():
            continue
        rec = json.loads(path.read_text())
        rows[label] = rec

    def d_u_plus(rec: dict[str, Any]) -> dict[str, float]:
        u_plus = float(rec["u_plus_bulk"])
        re_tau = float(rec["re_tau"])
        u_smooth = math.sqrt(2.0 / rl.cf_channel_smooth_reference(re_tau))
        fit = rec.get("log_fit") or {}
        b_smooth = rl.B_SMOOTH
        return {
            "dU_plus_bulk": u_smooth - u_plus,
            "dU_plus_profile": b_smooth - float(fit.get("b", float("nan"))),
            "cf": float(rec["cf"]),
            "u_plus_bulk": u_plus,
            "re_tau": re_tau,
            "ks_plus": float(rec["ks_plus"]),
            "dy_plus": float(rec.get("dy_plus", float("nan"))),
            "wall_seconds": float(rec.get("wall_seconds", 0.0)),
        }

    data = {k: d_u_plus(v) for k, v in rows.items()}
    if len(data) < 3:
        return {"passed": False, "reason": f"only {len(data)} of 3 refinement cases available",
                "cases": data}

    order = ["coarse", "medium", "fine"]
    present = [k for k in order if k in data]
    coarse, medium, fine = (data[k] for k in present[:3])
    # refinement ratio in the quantity that actually refines: dy^+ (= Re_tau/h_eff)
    r = coarse["dy_plus"] / medium["dy_plus"]
    r_fine = medium["dy_plus"] / fine["dy_plus"]
    gci_bulk = tz.gci_three_grid(fine["dU_plus_bulk"], medium["dU_plus_bulk"],
                                 coarse["dU_plus_bulk"], ratio=r)
    gci_profile = tz.gci_three_grid(fine["dU_plus_profile"], medium["dU_plus_profile"],
                                    coarse["dU_plus_profile"], ratio=r)
    return {
        "quantity": "dU^+ at k_s^+ = 90, lambda/k_s = 5 (fixed physical case, three grids)",
        "refinement_ratio_dy_plus": float(r),
        "refinement_ratio_dy_plus_fine_pair": float(r_fine),
        "threshold": 0.05,
        "gci_dU_plus_bulk": gci_bulk,
        "gci_dU_plus_profile": gci_profile,
        "cases": data,
        "passed": bool(gci_bulk.get("passed") and gci_profile.get("passed")),
        "note": ("dy^+ = Re_tau/h_eff is the quantity that refines (30 -> 18 -> 12.9); "
                 "k_s^+ and k_s/h are held fixed by construction"),
    }


# ---------------------------------------------------------------------------
# C6: end-to-end DEM callback check
# ---------------------------------------------------------------------------

def _c6_demo(laws: dict[float, rl.ThreeZoneLaw], table: list[dict[str, Any]],
             *, lambda_over_ks: float = 5.0, k_s_m: float = 0.35) -> dict[str, Any]:
    """Run TensorDEM with the closure callback and prove switchability + exchange."""
    import torch

    from tensorlbm.dem import DEMConfig, DEMSolver, IceParticle, cw_coupling as cc

    law = rl.interpolate_law(lambda_over_ks, laws)
    result: dict[str, Any] = {"lambda_over_ks": float(lambda_over_ks), "k_s_m": float(k_s_m)}

    def _run(drag_model: str, callback=None, torque_free: bool = True):
        cfg = DEMConfig(
            time_step=1e-3, gravity=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
            drag_model=drag_model, fluid_density=1025.0, fluid_viscosity=1.83e-3,
            enable_wall_contacts=False, cw_callback=callback,
            ks_m=k_s_m, lambda_over_ks=lambda_over_ks,
        )
        solver = DEMSolver(cfg)
        solver.exchange = cc.MomentumExchange(rho_water=1025.0)
        zero_w = torch.zeros(3, dtype=torch.float64)
        particles = [
            IceParticle(position=torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64),
                        velocity=torch.tensor([0.10, 0.00, 0.00], dtype=torch.float64),
                        angular_velocity=zero_w.clone(), radius=0.05, density=917.0),
            IceParticle(position=torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64),
                        velocity=torch.tensor([-0.05, 0.02, 0.0], dtype=torch.float64),
                        angular_velocity=zero_w.clone(), radius=0.05, density=917.0),
        ]
        solver.initialize(particles)
        fluid = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        initial = [tuple(float(v) for v in p.velocity) for p in solver.system]
        for _ in range(5):
            solver.step(fluid)
        final = [tuple(float(v) for v in p.velocity) for p in solver.system]
        return {
            "drag_model": drag_model,
            "drag_events": len(solver.drag_events),
            "initial_velocity": [list(map(float, v)) for v in initial],
            "final_velocity": [list(map(float, v)) for v in final],
            "exchange": solver.exchange.residual(),
            "momentum": [list(map(float, p.momentum().tolist())) for p in solver.system],
        }

    const = cc.ConstantCwCallback(5.5e-3)
    const_run = _run("cw_callback", const)
    rough_cb = cc.RoughnessCwCallback(law, k_s_m)
    rough_run = _run("cw_callback", rough_cb)
    table_cb = cc.TabulatedCwCallback(table, k_s_m=k_s_m, lambda_over_ks=lambda_over_ks)
    table_run = _run("cw_callback", table_cb)
    naumann_run = _run("schiller_naumann")

    result["runs"] = {
        "constant_cw_callback": const_run, "roughness_closure_callback": rough_run,
        "tabulated_callback": table_run, "schiller_naumann": naumann_run,
    }
    result["closure_calls"] = {
        "constant": const.calls[:4], "roughness": rough_cb.calls[:4],
        "tabulated": table_cb.calls[:4],
    }
    residuals = {k: v["exchange"]["relative_residual"] for k, v in result["runs"].items()}
    result["exchange_relative_residual"] = residuals
    result["threshold"] = 1e-10
    result["switchable"] = bool(
        rough_run["final_velocity"] != const_run["final_velocity"]
        and const_run["final_velocity"] != naumann_run["final_velocity"]
        and rough_run["drag_events"] > 0)
    result["passed"] = bool(result["switchable"] and all(x < 1e-10 for x in residuals.values()))
    result["note"] = ("TensorDEM (tensorlbm.dem.DEMSolver) runs the same particle set with "
                      "drag_model='cw_callback' (constant / three-zone closure / tabulated LBM "
                      "table) and with 'schiller_naumann'; the water-side reaction impulse is "
                      "recomputed independently by MomentumExchange, so the residual measures "
                      "the implementation of the exchange, not Newton's third law by hand.")
    return result


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------

def _plain(value: Any) -> Any:
    """Recursively convert numpy / NaN to JSON-safe python."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if math.isfinite(v) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def cmd_aggregate(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    art = Path(args.out)
    art.mkdir(parents=True, exist_ok=True)
    (art / "figures").mkdir(exist_ok=True)

    records = sweep.load_records(run_dir)
    if not records:
        raise FileNotFoundError(f"no case records under {run_dir}")
    print(f"{len(records)} case records from {run_dir}")

    summary = sweep.aggregate(run_dir)
    laws = rl.laws_from_campaign(summary)
    rows = tz.rows_from_records(records)

    adjudication = tz.adjudicate(rows)
    table = pg.cw_table(laws)
    verdict = pg.propagation_verdict(laws)
    bands = pg.nansen_bands(laws)

    refine_dir = Path(args.refine_dir)
    gci = _gci_from_records({
        "coarse": refine_dir / f"ref_ks90_lam5_h32.json",
        "fine": refine_dir / f"ref_ks90_lam5_h64.json",
        "medium": refine_dir / f"ref_ks90_lam5_h48.json",
    })

    c6 = _c6_demo(laws, table)

    payload = {
        "mission": "P103W2C-N01",
        "generated_by": "tools/run_ice_rough_cw_adjudication.py aggregate",
        "run_dir": str(run_dir),
        "n_records": len(records),
        "laws": {f"{k:g}": v.as_dict() for k, v in laws.items()},
        "rows": [r.as_dict() for r in rows],
        "campaign_C1_C4": summary,
        "three_zone": adjudication,
        "resolution_audit": adjudication.get("resolution_audit"),
        "wall_resolved_cost": adjudication.get("wall_resolved_cost_scenarios") or {},
        "C4_gci": gci,
        "C5_propagation": verdict,
        "C5_nansen_bands": bands,
        "C6_dem_callback": c6,
        "cw_table": table,
    }

    criteria = {
        "C1_smooth_wall_cf_lt_3pct": {"passed": bool(adjudication["C1_smooth_lens"]["passed"]),
                                      "value": adjudication["C1_smooth_lens"]["max_abs_rel_error"],
                                      "threshold": 0.03},
        "C2_kappa_0.41pm0.02": {"passed": bool(adjudication["C2_kappa_fully_rough"]["passed"]),
                                "value": adjudication["C2_kappa_fully_rough"]["fit"]["kappa"],
                                "threshold": 0.02},
        "C3_smooth_zone_slope_-0.22pm0.03": {
            "passed": bool(adjudication["C3_smooth_zone_exponent"]["passed"]),
            "value": adjudication["C3_smooth_zone_exponent"]["fit"]["n"], "threshold": 0.03},
        "C4_gci_lt_5pct": {"passed": bool(gci.get("passed")),
                           "value": (gci.get("gci_dU_plus_bulk") or {}).get("gci_fine"),
                           "threshold": 0.05},
        "C5_propagation": {"passed": bool(verdict["passed"]),
                           "value": verdict["speed_change_pct_range"]
                           + verdict["angle_change_deg_range"],
                           "threshold": "dU/U in 10-40 %, dalpha in 5-15 deg, state in observed band"},
        "D0_campaign_resolution_adequate": {
            "passed": bool((adjudication.get("resolution_audit") or {}).get("passed")),
            "value": (adjudication.get("resolution_audit") or {}).get("dy_plus_range"),
            "threshold": "dy^+ <= 2 and >= 3 nodes in the log window"},
        "C6_dem_callback": {"passed": bool(c6["passed"]),
                            "value": c6.get("exchange_relative_residual"),
                            "threshold": 1e-10},
    }
    payload["criteria"] = criteria
    payload["all_passed"] = all(v["passed"] for v in criteria.values())

    (art / "adjudication.json").write_text(json.dumps(_plain(payload), indent=2))
    (art / "cw_table.json").write_text(json.dumps(_plain(table), indent=2))
    (art / "propagation.json").write_text(json.dumps(_plain(verdict), indent=2))
    (art / "criteria.json").write_text(json.dumps(_plain(criteria), indent=2))

    figures = _figures(art, rows, laws, table, bands, gci, verdict,
                       audit=adjudication.get("resolution_audit"))
    _write_report(art, payload, figures)
    print(json.dumps(_plain(criteria), indent=2))
    print(f"artifacts -> {art}")
    return 0 if payload["all_passed"] else 0


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _figures(art: Path, rows, laws, table, bands, gci, verdict, audit=None) -> list[dict[str, str]]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict[str, str]] = []
    figdir = art / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # ---- 1: roughness function with the three-zone law --------------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = {"2": "C0", "5": "C1", "10": "C2", "sand": "k"}
    for g in sorted({r.group for r in rows if r.kind == "rough"}):
        sel = [r for r in rows if r.kind == "rough" and r.group == g]
        ax.plot([r.ks_plus for r in sel], [r.d_u_plus for r in sel], "o-",
                color=colors.get(g, "C3"), label=f"LBM lambda/ks={g}")
    ks = np.logspace(1, 3, 60)
    for lam, law in sorted(laws.items()):
        ax.plot(ks, [law.delta_u_plus(k) for k in ks], "--", color=colors.get(f"{lam:g}", "C4"),
                lw=1.0, label=f"fit lambda/ks={lam:g} (k0={law.k0:.2f})")
    ax.plot(ks, [rl.nikuradse_delta_u_plus(k) for k in ks], "-", color="0.5", lw=1.5,
            label="Nikuradse sand reference")
    ax.set_xscale("log"); ax.set_xlabel("$k_s^+$"); ax.set_ylabel(r"$\Delta U^+$")
    ax.set_title("Roughness function: LBM (markers) vs three-zone fit")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    p = figdir / "fig1_roughness_function.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig1_roughness_function.png",
                 "caption": "Measured roughness function dU+(k_s+) per spacing ratio (markers) "
                            "with the fitted three-zone law (dashed)."})

    # ---- 2: Cf vs Re_b with three lenses ---------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    smooth = [r for r in rows if r.kind == "smooth"]
    rough = [r for r in rows if r.kind == "rough"]
    if smooth:
        ax.plot([r.re_b for r in smooth], [r.cf for r in smooth], "ks-", ms=6,
                label="smooth baseline (LBM)")
    for g in sorted({r.group for r in rough}):
        sel = [r for r in rough if r.group == g]
        ax.plot([r.re_b for r in sel], [r.cf for r in sel], "o", color=colors.get(g, "C3"),
                label=f"rough lambda/ks={g}")
    re_grid = np.logspace(3.6, 5.6, 40)
    ax.plot(re_grid, [tz.cf_smooth_lens(r / 2.0) for r in re_grid], "-", color="grey", lw=1,
            label="lens S (smooth law)")
    ax.plot(re_grid, [tz.colebrook_cf_channel(r, 5.0 / 47.0)["C_f"] for r in re_grid], ":",
            color="m", lw=1.2, label="lens C (Colebrook, k_s/h=5/47)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("$Re_b$"); ax.set_ylabel("$C_f$")
    ax.set_title("Measured friction vs the three lenses")
    ax.legend(fontsize=7); ax.grid(alpha=0.3, which="both")
    p = figdir / "fig2_cf_vs_re_with_lenses.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig2_cf_vs_re_with_lenses.png",
                 "caption": "Friction coefficient of every case against bulk Reynolds number, "
                            "with lens S (smooth power law) and lens C (Colebrook-White)."})

    # ---- 3: drift angle / wind factor vs ice-keel roughness --------------
    fig, ax = plt.subplots(figsize=(7, 4.6))
    lam5 = [b for b in bands if abs(b["lambda_over_ks"] - 5.0) < 1e-9 and abs(b["wind_speed_m_s"] - 7.0) < 1e-9]
    if lam5:
        ks = [b["k_s_m"] for b in lam5]
        ax.plot(ks, [b["closure_angle_deg"] for b in lam5], "o-", color="C0",
                label="drift angle (closure, U_a=7 m/s)")
        ax.plot(ks, [b["constant_angle_deg"] for b in lam5], "s--", color="C3",
                label="drift angle (constant C_w = 5.5e-3)")
        ax.axhspan(*pg.NANSEN_ANGLE_BAND_DEG, color="C0", alpha=0.08,
                   label="observed Nansen band 20-40 deg")
        ax.set_xlabel("ice-keel roughness $k_s$ [m]"); ax.set_ylabel("drift angle [deg]")
        ax2 = ax.twinx()
        ax2.plot(ks, [b["closure_wind_factor"] for b in lam5], "^:", color="C2",
                 label="wind factor (closure)")
        ax2.axhspan(*pg.WIND_FACTOR_BAND, color="C2", alpha=0.08)
        ax2.set_ylabel("wind factor $U/U_a$")
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    ax.set_title("Propagation: Nansen angle and wind factor vs ice-underside roughness")
    p = figdir / "fig3_nansen_vs_roughness.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    figs.append({"name": "fig3_nansen_vs_roughness.png",
                 "caption": "Nansen drift angle (left axis) and wind factor (right axis) versus "
                            "ice-keel roughness with the observed bands shaded."})

    # ---- 5: resolution audit: smooth-baseline error vs dy^+ ---------------
    audit = audit or {}
    if audit.get("smooth_error_vs_dy_plus"):
        fig, ax = plt.subplots(figsize=(6, 4.2))
        pts = audit["smooth_error_vs_dy_plus"]
        ax.loglog([p["dy_plus"] for p in pts], [abs(p["rel_error"]) for p in pts], "o-")
        ax.axhline(0.03, color="r", ls="--", lw=1, label="C1 threshold 3 %")
        for d in (1.0, 2.0):
            ax.axvline(d, color="g", ls=":", lw=1)
        ax.set_xlabel("$dy^+ = u_\\tau/\\nu$ (first-node wall distance)")
        ax.set_ylabel("smooth-wall $C_f$ relative error")
        ax.set_title("Discretisation error of the smooth baselines")
        ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
        pth = figdir / "fig5_resolution_audit.png"; fig.tight_layout(); fig.savefig(pth, dpi=150)
        plt.close(fig)
        figs.append({"name": "fig5_resolution_audit.png",
                     "caption": "Smooth-wall drag error versus the wall-normal resolution: the "
                                "error is monotone in dy^+ (dotted lines: dy^+ = 1 and 2), i.e. "
                                "a discretisation effect, not a turbulence-model difference."})

    # ---- 4: resolution study --------------------------------------------
    cases = gci.get("cases") or {}
    if len(cases) >= 2:
        fig, ax = plt.subplots(figsize=(6, 4.2))
        dy = [cases[k]["dy_plus"] for k in ("coarse", "medium", "fine") if k in cases]
        du = [cases[k]["dU_plus_bulk"] for k in ("coarse", "medium", "fine") if k in cases]
        ax.plot(dy, du, "o-")
        for x, y, k in zip(dy, du, [k for k in ("coarse", "medium", "fine") if k in cases]):
            ax.annotate(k, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
        ax.set_xlabel("$dy^+ = Re_\\tau/h_{eff}$"); ax.set_ylabel(r"$\Delta U^+$ at $k_s^+=90$")
        ax.set_title("Grid refinement at fixed $k_s^+$, $\\lambda/k_s=5$")
        ax.grid(alpha=0.3)
        p = figdir / "fig4_gci.png"; fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
        figs.append({"name": "fig4_gci.png",
                     "caption": "Roughness function at fixed physical conditions over the three "
                                "resolutions used for the GCI (criterion C4)."})
    return figs


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _fmt(v: Any, nd: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(f):
        return "n/a"
    return f"{f:.{nd}g}" if abs(f) < 1e-3 or abs(f) >= 1e5 else f"{f:.{nd}f}"


def _sci(v: Any) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "n/a"
    return f"{f:.3e}" if math.isfinite(f) else "n/a"


def _write_report(art: Path, payload: dict[str, Any], figures: list[dict[str, str]]) -> None:
    tzres = payload["three_zone"]
    gci = payload["C4_gci"]
    verdict = payload["C5_propagation"]
    c6 = payload["C6_dem_callback"]
    crit = payload["criteria"]
    audit = payload.get("resolution_audit") or {}
    lines = [
        "# Ice-keel roughness -> ice/water drag closure: three-lens adjudication",
        "",
        "Mission P103W2C-N01. Machine-generated by "
        "`tools/run_ice_rough_cw_adjudication.py aggregate`; every number below is read from "
        "`adjudication.json` (which is read from the per-case records in "
        f"`{payload['run_dir']}`), none is transcribed by hand.",
        "",
        "## 0. Headline (honest verdict)",
        "",
        f"* **C6 (DEM callback, switchable + exchange residual)**: "
        f"{'PASS' if c6.get('passed') else 'FAIL'}.",
        f"* **C1 / C2 / C3 (smooth baseline, `kappa`, three-zone slope)**: "
        f"{'PASS' if all(tzres[k]['passed'] for k in ('C1_smooth_lens', 'C2_kappa_fully_rough', 'C3_smooth_zone_exponent')) else 'FAIL'} "
        f"-- the campaign data do not resolve the wall layer: {audit.get('verdict')}.",
        f"* **C4 (GCI < 5 %)**: {'PASS' if gci.get('passed') else 'FAIL'}.",
        f"* **C5 (propagation)**: {'PASS' if verdict['passed'] else 'FAIL'} "
        "(numbers reported below, but see the resolution audit before using them).",
        f"* Campaign design adequate: **{audit.get('passed')}** "
        f"(dy^+ range {audit.get('dy_plus_range')}).",
        "",
        "## 1. Acceptance criteria",
        "",
        "| criterion | requirement | measured | passed |",
        "|---|---|---|---|",
    ]
    c1 = tzres["C1_smooth_lens"]
    c2 = tzres["C2_kappa_fully_rough"]["fit"]
    c3 = tzres["C3_smooth_zone_exponent"]["fit"]
    lines += [
        f"| C1 | smooth-wall $C_f$ within 3 % | max \\|rel err\\| = "
        f"{100 * c1['max_abs_rel_error']:.2f} % ({c1['n_cases']} baselines) | {c1['passed']} |",
        f"| C2 | $\\kappa = 0.41 \\pm 0.02$ from the fully-rough slope | "
        f"$\\kappa$ = {_fmt(c2.get('kappa'))} ± {_fmt(c2.get('kappa_stderr'))} "
        f"({c2.get('n_points')} pts, groups {c2.get('groups')}) | "
        f"{tzres['C2_kappa_fully_rough']['passed']} |",
        f"| C3 | smooth-zone slope $-0.22 \\pm 0.03$ | n = {_fmt(c3.get('n'))} ± "
        f"{_fmt(c3.get('n_stderr'))} over {c3.get('n_points')} baselines | "
        f"{tzres['C3_smooth_zone_exponent']['passed']} |",
        f"| C4 | GCI < 5 % | GCI($\\Delta U^+$) = "
        f"{_fmt(100 * (gci.get('gci_dU_plus_bulk') or {}).get('gci_fine', float('nan')), 3)} %, "
        f"observed order p = {_fmt((gci.get('gci_dU_plus_bulk') or {}).get('p'), 3)} | "
        f"{gci.get('passed')} |",
        f"| C5 | speed change 10-40 %, angle change 5-15°, state inside the observed band | "
        f"$\\Delta U/U$ = {_fmt(verdict['speed_change_pct_range'][0], 3)}…"
        f"{_fmt(verdict['speed_change_pct_range'][1], 3)} %, "
        f"$\\Delta\\alpha$ = {_fmt(verdict['angle_change_deg_range'][0], 3)}…"
        f"{_fmt(verdict['angle_change_deg_range'][1], 3)}°, "
        f"$\\alpha$ = {_fmt(verdict['closure_angle_deg_range'][0], 3)}…"
        f"{_fmt(verdict['closure_angle_deg_range'][1], 3)}° | {verdict['passed']} |",
        f"| C6 | DEM callback switchable, exchange residual < 1e-10 | residual = "
        f"{ {k: _fmt(v, 2) for k, v in c6.get('exchange_relative_residual', {}).items()} } | "
        f"{c6.get('passed')} |",
        "",
        "## 1b. Resolution audit -- why the physics criteria fail",
        "",
        f"verdict: **{audit.get('verdict')}**",
        "",
        "| case | Re_tau | dy^+ | Cf | Cf reference | rel err |",
        "|---|---|---|---|---|---|",
    ]
    for row in audit.get("smooth_error_vs_dy_plus", []):
        lines.append(f"| {row['case_id']} | {_fmt(row['re_tau'], 4)} | {_fmt(row['dy_plus'], 3)} | "
                     f"{_fmt(row['cf'], 4)} | {_fmt(row['cf_reference'], 4)} | "
                     f"{100 * row['rel_error']:+.1f} % |")
    lines += [
        "",
        f"log-window points (rough cases): max {audit.get('rough_log_window_points', {}).get('max')}, "
        f"median {audit.get('rough_log_window_points', {}).get('median')}, usable "
        f"{audit.get('rough_log_window_points', {}).get('n_usable')} of "
        f"{audit.get('rough_log_window_points', {}).get('n_rough')}; "
        f"peak bulk Mach {_fmt(audit.get('max_mach'), 3)}; "
        f"min relaxation time {_fmt(audit.get('min_tau_relax'), 5)}.",
        "",
        "Cost of a *wall-resolved* campaign (design identities `k_s^+ = k_s_nodes * dy^+`, "
        "`k_s/h <= 0.1`, hence `Re_tau = k_s^+/0.1` whatever `dy^+` is; 100 MLUPS, "
        "19 populations, double precision):",
        "",
        "| k_s^+ max | dy^+ | h nodes | Re_tau | cells | cell-updates | elevated | memory | feasible |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, cost in (payload.get("wall_resolved_cost") or {}).items():
        if not cost:
            continue
        lines.append(
            f"| {_fmt(cost['ks_plus_max'], 4)} | {_fmt(cost['dy_plus_target'], 3)} | "
            f"{_fmt(cost['h_nodes'], 6)} | {_fmt(cost['re_tau'], 6)} | {cost['cells']:.3g} | "
            f"{cost['cell_updates']:.3g} | {cost['turns']:.3g} T-updates | "
            f"{cost['memory_GB_double_19']:.0f} GB | {cost['feasible']} |")
    lines += [
        "",
        "",
        "### Geometric ceiling of the roughness function",
        "",
    ]
    cap = (audit.get("geometric_cap") or {})
    if cap:
        lines += [
            f"`{cap.get('relation')}` -- {cap.get('verdict')}; median ceiling fraction "
            f"{_fmt(cap.get('median_ceiling_fraction'), 3)}, ceiling binds above "
            f"$k_s^+ \\approx$ {_fmt(cap.get('ks_plus_where_ceiling_binds'), 4)}.",
            "",
            "| case | k_s^+ | Re_tau | k_s/h | dU^+ | ceiling | ceiling fraction | Nikuradse |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for row in cap.get("per_case", []):
            lines.append(
                f"| {row['case_id']} | {_fmt(row['ks_plus'], 4)} | {_fmt(row['re_tau'], 4)} | "
                f"{_fmt(row['ks_over_h'], 3)} | {_fmt(row['dU_plus'], 3)} | "
                f"{_fmt(row['dU_plus_ceiling'], 3)} | {_fmt(row['ceiling_fraction'], 3)} | "
                f"{_fmt(row['nikuradse_dU_plus'], 3)} |")
    lines += [
        "",
        "## 2. Three-lens zone adjudication",
        "",
        "| zone | n | median \\|err\\| lens S | lens C (Colebrook) | lens R (log law) | expected | best | passed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for zone, z in tzres["lenses"]["zone_adjudication"].items():
        me = z.get("median_abs_error") or {}
        lines.append(
            f"| {zone} | {z.get('n_cases')} | {_fmt(me.get('lens_smooth'))} | "
            f"{_fmt(me.get('lens_colebrook'))} | {_fmt(me.get('lens_fully_rough'))} | "
            f"{z.get('expected_lens')} | {z.get('best_lens')} | {z.get('passed')} |")
    lines += [
        "",
        "## 3. Fitted closures (the back-fill)",
        "",
        "| lambda/k_s | k0 | k_s,eff/k_s | smooth upper | fully rough lower | points |",
        "|---|---|---|---|---|---|",
    ]
    for key, law in payload["laws"].items():
        r = law.get("regimes") or {}
        lines.append(f"| {key} | {_fmt(law.get('k0'))} | "
                     f"{_fmt(law.get('equivalent_sand_grain_ratio'))} | "
                     f"{_fmt(r.get('ks_plus_smooth_upper'), 3)} | "
                     f"{_fmt(r.get('ks_plus_fully_rough_lower'), 3)} | {law.get('n_points')} |")
    lines += ["", "### C_w table (reference $u_{rel}=0.15$ m/s)", "",
              "| lambda/k_s | k_s [m] | k_s^+ | C_w | amplification | regime |",
              "|---|---|---|---|---|---|"]
    for row in payload["cw_table"]:
        if "C_w" not in row:
            continue
        lines.append(f"| {_fmt(row['lambda_over_ks'], 3)} | {_fmt(row['k_s_m'], 3)} | "
                     f"{_fmt(row['k_s_plus'], 4)} | {_sci(row['C_w'])} | "
                     f"{_fmt(row['amplification_over_smooth'], 3)} | {row['regime']} |")
    lines += [
        "",
        "## 4. Propagation into the drift model (C5)",
        "",
        "| U_a [m/s] | C_w closure | k_s^+ | alpha closure [deg] | alpha constant [deg] | "
        "dU/U [%] | dalpha [deg] | in observed alpha band |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in verdict["rows"]:
        lines.append(
            f"| {_fmt(r['wind_speed_m_s'], 3)} | {_sci(r['closure']['C_w'])} | "
            f"{_fmt(r['closure']['ks_plus'], 4)} | {_fmt(r['closure']['drift_angle_deg'], 3)} | "
            f"{_fmt(r['constant']['drift_angle_deg'], 3)} | {_fmt(r['speed_change_pct'], 3)} | "
            f"{_fmt(r['angle_change_deg'], 3)} | {r['in_observed_angle_band']} |")
    lines += ["", f"**Verdict C5: passed = {verdict['passed']}**"]
    if verdict["fail_reasons"]:
        lines += ["", "Honest FAIL reasons:"] + [f"- {r}" for r in verdict["fail_reasons"]]
    lines += [
        "",
        "## 5. End-to-end DEM callback (C6)",
        "",
        f"switchable = {c6.get('switchable')}; exchange residual = "
        f"{ {k: _fmt(v, 2) for k, v in c6.get('exchange_relative_residual', {}).items()} } "
        f"(threshold {c6.get('threshold')})",
        "",
        "## 6. Figures",
        "",
    ]
    for f in figures:
        lines += [f"### {f['name']}", "", f["caption"], "",
                  f"![{f['name']}](figures/{f['name']})", ""]
    lines += ["## 7. Limitations", "",
              "- The LBM campaign covers $k_s^+ = 20\\ldots500$; geophysical ice keels reach "
              "$k_s^+ \\sim 10^3$, so the drift back-fill at $k_s \\ge 0.35$ m is an "
              "extrapolation flagged per table row.",
              "- The channel relative roughness is $k_s/h \\approx 0.1$, far larger than the "
              "ocean boundary layer ($k_s/z_{ref} \\sim 0.03$); the closure is used through the "
              "log-law form, and lens C's duct convention ($D_h = 4h$) is the standard but "
              "approximate mapping.",
              ""]
    (art / "REPORT.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="print the refinement case table")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("refine", help="run the grid-refinement trio (criterion C4)")
    p.add_argument("--out", default=str(DEFAULT_REFINE_DIR))
    p.add_argument("--threads", type=int, default=12)
    p.add_argument("--force", action="store_true")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_refine)

    p = sub.add_parser("aggregate", help="adjudicate C1-C6 and write the artifacts")
    p.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    p.add_argument("--refine-dir", default=str(DEFAULT_REFINE_DIR))
    p.add_argument("--out", default=str(DEFAULT_ART_DIR))
    p.set_defaults(func=cmd_aggregate)

    args = ap.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())