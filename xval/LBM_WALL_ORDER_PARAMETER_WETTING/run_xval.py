#!/usr/bin/env python3
"""XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER harness.

Arms
----
``port``   TensorLBM implementation of the wall-order-parameter wetting model
           (``tensorlbm.wetting.wall_order_parameter_2d``): static equilibration
           by mass-conserving Allen-Cahn descent on the *same* free energy,
           stencils and wall ghost as the faithful LBM port.  This arm carries
           the quantitative contact-angle law.
``lbm``    the faithful LBM port (``step``) of the reference's two-distribution
           scheme, run at the same geometry: used to *measure* the accuracy
           limit of the reference's order-parameter closure (spurious
           ``psi``-diffusion) rather than to produce the wetting law.
``oss``    ``oss_runner.py`` -- the vendored reference code itself.

Every run writes ``runs/<arm>/result.json`` (+ the raw field ``psi.npz``); the
report tables and the figures are built from those files only (no hand-copied
numbers).  ``summarize`` applies the predeclared gates of ``predeclared.json``.

Usage
-----
    python3 run_xval.py port  --psi-wall 0.35 --xi 3
    python3 run_xval.py lbm   --psi-wall 0.0 --mobility 0.1
    python3 run_xval.py summarize
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from tensorlbm.wetting.wall_order_parameter_2d import (  # noqa: E402
    WallOrderWettingConfig,
    analytic_contact_angle,
    contact_angle_from_field,
    relax_to_equilibrium,
    run_to_equilibrium,
    step,
    initialise_drop_on_wall,
    macroscopics,
)

PREDECLARED = json.loads((HERE / "predeclared.json").read_text())


def _write(run_dir: Path, payload: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "result.json").write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# arm: port (static equilibration)
# ---------------------------------------------------------------------------
def arm_port(psi_wall: float, xi: float, ny: int, nx: int, radius: float,
             surface_tension: float, omega: float, max_iter: int, tag: str = "") -> dict:
    cfg = WallOrderWettingConfig(
        ny=ny, nx=nx, psi_wall=psi_wall, psi_wall_top=-1.0,
        interface_width=xi, surface_tension=surface_tension,
    )
    t0 = time.time()
    out = relax_to_equilibrium(cfg, radius=radius, centre_y=radius, max_iter=max_iter,
                               tol=PREDECLARED["predeclared_thresholds"]["port_tolerance"], omega=omega)
    wall = time.time() - t0
    meas = contact_angle_from_field(out["psi"], psi_wall=psi_wall, interface_width=xi)
    theta_analytic = math.degrees(analytic_contact_angle(psi_wall))
    theta_ref = PREDECLARED.get("reference_contact_angle", {}).get(f"{psi_wall:+.2f}")
    payload = {
        "arm": "port",
        "tag": tag,
        "psi_wall": psi_wall,
        "interface_width": xi,
        "ny": ny,
        "nx": nx,
        "radius": radius,
        "surface_tension": surface_tension,
        "omega": omega,
        "max_iter": max_iter,
        "iterations": out["iterations"],
        "converged": out["converged"],
        "drift": out["drift"],
        "mass_error": out["mass_error"],
        "measurement": meas,
        "theta_deg": meas["theta_cap_deg"],
        "theta_circle_deg": meas["theta_circle_deg"],
        "theta_analytic_deg": theta_analytic,
        "theta_reference_deg": theta_ref,
        "err_vs_analytic_pct": (
            None if not math.isfinite(meas["theta_cap_deg"])
            else (meas["theta_cap_deg"] - theta_analytic) / theta_analytic * 100.0
        ),
        "err_vs_reference_pct": (
            None if theta_ref is None or not math.isfinite(meas["theta_cap_deg"])
            else (meas["theta_cap_deg"] - theta_ref) / theta_ref * 100.0
        ),
        "wall_seconds": wall,
        "parameters": {
            "a": cfg.a, "kappa": cfg.kappa,
            "nu": (cfg.tau_f - 0.5) / 3.0,
            "xi_over_dx": xi,
            "Laplace_pressure_units": 1.0,
        },
    }
    run_dir = HERE / "runs" / (tag or f"port_psi{psi_wall:+.2f}_xi{xi:.1f}")
    _write(run_dir, payload)
    np.savez(run_dir / "psi.npz", psi=out["psi"].numpy())
    return payload


# ---------------------------------------------------------------------------
# arm: lbm (faithful port of the reference two-distribution scheme)
# ---------------------------------------------------------------------------
def arm_lbm(psi_wall: float, xi: float, ny: int, nx: int, radius: float,
            surface_tension: float, mobility: float, tau_g: float, steps: int,
            closure: str, tag: str = "") -> dict:
    cfg = WallOrderWettingConfig(
        ny=ny, nx=nx, psi_wall=psi_wall, psi_wall_top=-1.0, interface_width=xi,
        surface_tension=surface_tension, mobility=mobility, tau_g=tau_g, closure=closure,
    )
    state = initialise_drop_on_wall(cfg, radius=radius, centre_y=radius)
    psi0 = state["g"].sum(dim=0)
    mass0 = float(psi0.sum())
    t0 = time.time()
    series = []
    finite = True
    used = 0
    for k in range(1, steps + 1):
        state = step(state, cfg)
        if k % max(1, steps // 20) == 0 or k == 1:
            psi = state["g"].sum(dim=0)
            finite = bool(torch.isfinite(psi).all())
            series.append({
                "step": k,
                "psi_min": float(psi.min()) if finite else None,
                "psi_max": float(psi.max()) if finite else None,
                "mass_error": abs(float(psi.sum()) - mass0) / abs(mass0),
            })
            used = k
            if not finite:
                break
    wall = time.time() - t0
    payload = {
        "arm": "lbm",
        "tag": tag,
        "psi_wall": psi_wall,
        "interface_width": xi,
        "ny": ny,
        "nx": nx,
        "radius": radius,
        "surface_tension": surface_tension,
        "mobility": mobility,
        "tau_g": tau_g,
        "closure": closure,
        "steps_requested": steps,
        "steps_completed": used,
        "finite": finite,
        "series": series,
        "psi_max_initial": float(psi0.max()),
        "psi_max_final": series[-1]["psi_max"] if series else None,
        "dissociation_ratio": (
            None if not series or series[-1]["psi_max"] is None
            else series[-1]["psi_max"] / float(psi0.max())
        ),
        "wall_seconds": wall,
        "parameters": {"a": cfg.a, "kappa": cfg.kappa, "mobility": mobility,
                       "diffusive_ratio_spurious_over_mobility": 1.0 / mobility},
    }
    run_dir = HERE / "runs" / (tag or f"lbm_psi{psi_wall:+.2f}_m{mobility:g}_{closure}")
    _write(run_dir, payload)
    return payload


# ---------------------------------------------------------------------------
# summarize: tables + gates
# ---------------------------------------------------------------------------
def summarize() -> dict:
    runs = HERE / "runs"
    port = []
    lbm = []
    oss = []
    for path in sorted(runs.glob("*/result.json")):
        d = json.loads(path.read_text())
        arm = d.get("arm")
        d["_dir"] = path.parent.name
        (port if arm == "port" else lbm if arm == "lbm" else oss).append(d)
    tol_accept = PREDECLARED["accept_pct"]
    rows = []
    for d in sorted(port, key=lambda x: (x["interface_width"], x["psi_wall"])):
        th, th_a = d.get("theta_deg"), d.get("theta_analytic_deg")
        if th is not None and th_a:
            d["err_vs_analytic_pct"] = (th - th_a) / th_a * 100.0
        rows.append({
            "run": d["_dir"],
            "psi_wall": d["psi_wall"],
            "xi": d["interface_width"],
            "theta_deg": d["theta_deg"],
            "theta_circle_deg": d["theta_circle_deg"],
            "theta_analytic_deg": d["theta_analytic_deg"],
            "err_vs_analytic_pct": d["err_vs_analytic_pct"],
            "converged": d["converged"],
            "iterations": d["iterations"],
            "pass_vs_analytic": (
                None if d["err_vs_analytic_pct"] is None
                else abs(d["err_vs_analytic_pct"]) <= tol_accept
            ),
        })
    summary = {
        "generated_from": str(runs),
        "accept_pct": tol_accept,
        "n_port": len(port),
        "n_lbm": len(lbm),
        "n_oss": len(oss),
        "port_rows": rows,
        "lbm_rows": [
            {
                "run": d["_dir"], "closure": d["closure"], "mobility": d["mobility"],
                "steps_completed": d["steps_completed"], "finite": d["finite"],
                "psi_max_final": d["psi_max_final"], "dissociation_ratio": d["dissociation_ratio"],
            }
            for d in sorted(lbm, key=lambda x: x["_dir"])
        ],
        "oss_rows": [
            {
                "run": d["_dir"], "psi_wall": d["psi_wall"], "finite": d["finite"],
                "steps": d["steps"], "theta_deg": d.get("measurement", {}).get("theta_cap_deg"),
                "psi_min": d["psi_min"], "psi_max": d["psi_max"],
            }
            for d in sorted(oss, key=lambda x: x["_dir"])
        ],
    }
    base = [r for r in rows if r["xi"] == sorted({r["xi"] for r in rows})[0]]
    summary["gates"] = {
        "n_rows": len(rows),
        "max_abs_err_vs_analytic_pct": max(
            [abs(r["err_vs_analytic_pct"]) for r in rows if r["err_vs_analytic_pct"] is not None],
            default=None,
        ),
        "median_abs_err_pct": (
            float(np.median([abs(r["err_vs_analytic_pct"]) for r in rows
                             if r["err_vs_analytic_pct"] is not None]))
            if any(r["err_vs_analytic_pct"] is not None for r in rows) else None
        ),
        "all_converged": all(r["converged"] for r in rows) if rows else None,
        "all_within_accept": all(
            (r["pass_vs_analytic"] is True) for r in rows if r["pass_vs_analytic"] is not None
        ) if rows else None,
        "baseline_xi": base[0]["xi"] if base else None,
    }
    # convergence in xi: spread of theta at fixed psi_wall
    conv = {}
    for d in port:
        conv.setdefault(d["psi_wall"], {})[d["interface_width"]] = d["theta_deg"]
    summary["convergence_in_xi"] = {
        f"{k:+.2f}": v for k, v in sorted(conv.items()) if len(v) > 1
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["port", "lbm", "summarize"])
    ap.add_argument("--psi-wall", type=float, default=0.0)
    ap.add_argument("--xi", type=float, default=3.0)
    ap.add_argument("--ny", type=int, default=56)
    ap.add_argument("--nx", type=int, default=80)
    ap.add_argument("--radius", type=float, default=16.0)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--omega", type=float, default=0.4)
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--mobility", type=float, default=0.1)
    ap.add_argument("--tau-g", type=float, default=0.7)
    ap.add_argument("--closure", default="isotropic")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv[1:])
    if a.command == "port":
        p = arm_port(a.psi_wall, a.xi, a.ny, a.nx, a.radius, a.sigma, a.omega, a.max_iter, a.tag)
        print(json.dumps({k: p[k] for k in ("theta_deg", "theta_analytic_deg",
                                            "err_vs_analytic_pct", "converged", "iterations")}))
    elif a.command == "lbm":
        p = arm_lbm(a.psi_wall, a.xi, a.ny, a.nx, a.radius, a.sigma, a.mobility,
                    a.tau_g, a.steps, a.closure, a.tag)
        print(json.dumps({k: p[k] for k in ("finite", "steps_completed", "psi_max_final",
                                            "dissociation_ratio")}))
    else:
        print(json.dumps(summarize()["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))