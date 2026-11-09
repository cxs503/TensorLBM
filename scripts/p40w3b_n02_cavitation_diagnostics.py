"""P40W3B-N02 diagnostics: cavity-dynamics validation face, numerically audited.

Writes every number of the report to ``artifacts/p40w3b_n02/`` (no hand
transcription).  Three blocks:

``closure.json``   -- phase-change closure time scales (analytic) and the
                      integrated closure-vs-Rayleigh-Plesset verification;
``stability.json`` -- stability envelope of the homogeneous-mixture engine
                      (where the run goes non-finite) and the anchored vs
                      un-anchored isobaric cavity metric on the same field;
``sheet.json``     -- L_cav/c and delta_cav/c for NACA0015 at sigma = 1.0/0.8/0.6
                      against Brennen (1994/95) and Merkle et al. (1998).
``panel.json``     -- panel-method flow-engine verification (analytic cylinder)
                      and the status of the constant-pressure cavity branch.

Usage::

    python3 scripts/p40w3b_n02_cavitation_diagnostics.py --steps 900 --cpc 50
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(2)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.cavitation_airfoil import (  # noqa: E402
    CavitationFlowConfig,
    MixtureCavitationSolver,
    cavity_metrics,
    reference_data,
)
from tensorlbm.cavitation_numerics import (  # noqa: E402
    brinkman_eta_for_leakage,
    brinkman_leakage,
    cavity_length_isobaric,
    integrate_closure_growth,
    pressure_anchor,
    rayleigh_plesset_time,
    schnerr_sauer_growth_time,
    stability_budget,
)

ART = ROOT / "artifacts" / "p40w3b_n02"


def _cfg(sigma: float, cpc: int, cfl: float, smag: float) -> CavitationFlowConfig:
    return CavitationFlowConfig(
        airfoil="NACA0015",
        alpha_deg=5.0,
        sigma=sigma,
        cells_per_chord=cpc,
        domain_x_chords=6.0,
        domain_y_chords=3.0,
        le_x_chords=1.5,
        le_y_chords=1.5,
        cfl=cfl,
        smagorinsky=smag,
        c_dest=1.0,
        c_prod=1.0,
        spinup_steps=300,
        ramp_steps=200,
        sample_interval=10,
        seed=0,
    )


def run_case(spec: dict) -> dict:
    """One mixture-engine run; records the divergence step and both metrics."""
    cfg = _cfg(spec["sigma"], spec["cpc"], spec["cfl"], spec["smag"])
    cfg.n_steps = spec["steps"]
    solver = MixtureCavitationSolver(cfg)
    solver.u = solver.u.masked_fill(solver.damp_u > 1.5, 0.0)
    solver.v = solver.v.masked_fill(solver.damp_v > 1.5, 0.0)
    t0 = time.time()
    diverged_at = None
    max_u_hist: list[tuple[int, float]] = []
    for k in range(cfg.n_steps):
        solver.step()
        if k % 50 == 0:
            mu = float(solver.u.abs().max())
            max_u_hist.append((k, mu))
        if not torch.isfinite(solver.u).all() or not torch.isfinite(solver.p).all():
            diverged_at = k
            break
    wall = time.time() - t0
    out = {
        "case": spec,
        "wall_seconds": round(wall, 1),
        "diverged_at_step": diverged_at,
        "steps_completed": solver._step,
        "max_u_history": max_u_hist,
        "stable": diverged_at is None,
    }
    if diverged_at is None:
        metrics = {
            "anchored": cavity_length_isobaric(solver.p, cfg, solver.geometry),
            "un_anchored": cavity_length_isobaric(
                solver.p, cfg, solver.geometry, anchor_offset=0.0
            ),
            "transported_alpha": cavity_metrics(
                solver.alpha, cfg, threshold=cfg.cavity_threshold, geometry=solver.geometry
            ),
        }
        offset = pressure_anchor(solver.p)
        out["metrics"] = metrics
        out["p_offset"] = offset
        out["p_min_physical"] = float((solver.p + offset).min())
        out["alpha_max"] = float(solver.alpha.max())
        out["cl"] = solver.cl()
        out["cd"] = solver.cd()
        out["cl_history"] = solver.cl_history
        out["time_history"] = solver.time_history
        out["lcav_history"] = solver.lcav_history
    return out


def closure_block() -> dict:
    cfg = CavitationFlowConfig(cells_per_chord=50, cfl=0.2, sigma=0.8)
    rows = []
    for a in (1e-6, 1e-4, 1e-2, 0.3, 0.7):
        r = min(
            max((a / (1.0 - a) * 3.0 / (4.0 * math.pi * cfg.nuclei_density)) ** (1.0 / 3.0),
                cfg.cavity_threshold * 0.0 + 1e-12),
            1.0,
        )
        for floor in (cfg.cavity_threshold * 0.02, 1.0e-4):
            r_b = max(r, floor)
            ss = schnerr_sauer_growth_time(
                a, 0.5 * cfg.sigma, r_b_floor=floor
            )
            rp = rayleigh_plesset_time(0.5 * cfg.sigma, r_b)
            rows.append(
                {
                    "alpha": a,
                    "r_b": r_b,
                    "r_b_floor": floor,
                    "ss_growth_time": ss,
                    "rp_time": rp,
                    "ratio": ss / rp,
                }
            )
    growth = integrate_closure_growth(
        0.5 * cfg.sigma, n_steps=20000, alpha0=1e-4, r_b_floor=1e-4
    )
    growth.pop("time")
    growth.pop("alpha")
    return {
        "sigma": cfg.sigma,
        "dp": 0.5 * cfg.sigma,
        "rows": rows,
        "integrated": growth,
        "budget": stability_budget(cfg),
        "brinkman": {
            "eta_current": 1.0e-3,
            "slip_current": brinkman_leakage(1.0e-3, cfg.cfl * cfg.h),
            "eta_for_1pct": brinkman_eta_for_leakage(cfg.cfl * cfg.h, 0.01),
            "eta_for_0p1pct": brinkman_eta_for_leakage(cfg.cfl * cfg.h, 1.0e-3),
        },
    }


def panel_block() -> dict:
    import numpy as np

    from tensorlbm.cavitation_free_streamline import flow_solve, panel_geometry

    n = 160
    a = 0.5
    th = np.linspace(0.0, 2.0 * math.pi, n + 1)
    nodes = np.stack([a * np.cos(th), a * np.sin(th)], axis=1)
    geom = panel_geometry(nodes)
    st = flow_solve(geom, 0.0, kutta="none")
    thm = th[:-1] + np.diff(th) / 2.0
    analytic = -2.0 * np.sin(thm)
    err = float(np.abs(st.qt - analytic).max() / np.abs(analytic).max())
    # lifting branch: symmetric section with a sharp closed TE
    th2 = np.linspace(0.0, math.pi, 81)
    x = 0.5 * (1.0 - np.cos(th2))
    yt = 5.0 * 0.15 * (
        0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1036 * x**4
    )
    up = np.stack([x, yt], axis=1)
    lo = np.stack([x[::-1], -yt[::-1]], axis=1)
    foil = np.vstack([lo, up[1:]])
    gf = panel_geometry(foil)
    lift = flow_solve(gf, 5.0)
    thin = 2.0 * math.pi * math.radians(5.0)
    return {
        "cylinder": {
            "n_panels": n,
            "radius": a,
            "max_rel_err_qt": err,
            "cp_min": float(st.cp.min()),
            "cl": st.cl,
            "cd": st.cd,
        },
        "lifting_branch": {
            "airfoil": "NACA0015 (closed TE, 160 panels)",
            "alpha_deg": 5.0,
            "cl_computed": lift.cl,
            "cl_thin_aerofoil": thin,
            "ratio": lift.cl / thin,
            "status": "WITHHELD",
            "reason": "computed circulation is ~0.5x thin-aerofoil theory: the "
            "Kutta closure is not validated, so cavity results from this branch "
            "are not reported",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--cpc", type=int, default=50)
    ap.add_argument("--wave", default="all")
    args = ap.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    if args.wave in {"all", "closure"}:
        (ART / "closure.json").write_text(json.dumps(closure_block(), indent=2))
        (ART / "panel.json").write_text(json.dumps(panel_block(), indent=2))
        print("closure.json / panel.json written")
    if args.wave in {"all", "stability"}:
        spec = {
            "sigma": 0.8,
            "cpc": args.cpc,
            "cfl": 0.2,
            "smag": 0.1,
            "steps": args.steps,
        }
        out = run_case(spec)
        (ART / "stability.json").write_text(json.dumps(out, indent=2))
        print("stability.json:", out["stable"], out["steps_completed"])
    if args.wave in {"all", "sheet"}:
        ref = reference_data()["sheet"]
        rows = []
        for sigma in ref["sigma"]:
            spec = {
                "sigma": sigma,
                "cpc": args.cpc,
                "cfl": 0.2,
                "smag": 0.1,
                "steps": args.steps,
            }
            out = run_case(spec)
            row = {"sigma": sigma, "stable": out["stable"], "steps": out["steps_completed"]}
            if out["stable"]:
                m = out["metrics"]["anchored"]
                mu = out["metrics"]["un_anchored"]
                row.update(
                    {
                        "l_cav": m["l_cav"],
                        "delta_cav": m["delta_cav"],
                        "l_cav_un_anchored": mu["l_cav"],
                        "delta_cav_un_anchored": mu["delta_cav"],
                        "l_cav_alpha": out["metrics"]["transported_alpha"]["l_cav"],
                        "p_offset": out["p_offset"],
                        "alpha_max": out["alpha_max"],
                        "cl": out["cl"],
                        "cd": out["cd"],
                        "wall_seconds": out["wall_seconds"],
                    }
                )
            rows.append(row)
        payload = {"cpc": args.cpc, "steps": args.steps, "reference": ref, "rows": rows}
        (ART / "sheet.json").write_text(json.dumps(payload, indent=2))
        print("sheet.json written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())