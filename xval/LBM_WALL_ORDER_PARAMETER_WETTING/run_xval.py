#!/usr/bin/env python3
"""XVAL-LBM-DIFFUSE-INTERFACE-WETTING-VIA-WALL-ORDER-PARAMETER — harness.

Anchor: cross-validation of diffuse-interface wetting imposed through a *wall
order parameter* (``psi_w``) against the open-source implementation
``ebinan92/Fingering_dynamics`` (``lattice_boltzmann/validation.py``, wettability
block: a ghost layer of the order parameter stacked above/below the fluid box,
read by the isotropic 9-point gradient/Laplacian; the contact angle is emergent).

Arms
----
``port``  TensorLBM diffuse-interface module
          (:mod:`tensorlbm.wetting.wall_order_parameter_2d`): mass-conserving
          Allen-Cahn descent on the same free energy / stencils / wall ghost as
          the faithful LBM port.  This arm carries the quantitative law.
``cac``   *integration* arm: the production phase-field solver
          (:class:`tensorlbm.cac_lbm2d.CAC2D`, ``wall_phi_policy="wall_order"``)
          driven through :func:`tensorlbm.wetting.ladder.contact_wall_order`.
          Different solver, different order parameter (conserved ``phi`` vs
          ``psi``), angle measured by an independent metrics toolbox.

Every run writes ``runs/<tag>/result.json`` (fields saved as ``.npz`` for the
port arm); ``summarize`` rebuilds ``summary.json`` and applies the predeclared
gates.  No number in the report is typed by hand.

Reproduce (fully):

    python3 xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py sweep
    python3 xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py summarize

One arm:

    python3 xval/LBM_WALL_ORDER_PARAMETER_WETTING/run_xval.py port --psi-wall 0.35 --xi 3
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
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.wetting.wall_order_parameter_2d import (  # noqa: E402
    WallOrderWettingConfig,
    analytic_contact_angle,
    contact_angle_from_field,
    relax_to_equilibrium,
)

PREDECLARED = json.loads((HERE / "predeclared.json").read_text())
RUNS = HERE / "runs"


def _write(tag: str, payload: dict) -> None:
    d = RUNS / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "result.json").write_text(json.dumps(payload, indent=2, default=str))


# ---------------------------------------------------------------------------
# arm: port (diffuse-interface module, static equilibration)
# ---------------------------------------------------------------------------
def arm_port(psi_wall: float, xi: float, ny: int = 56, nx: int = 80, radius: float = 16.0,
             surface_tension: float = 0.05, omega: float = 0.4, max_iter: int = 30000,
             tag: str | None = None) -> dict:
    cfg = WallOrderWettingConfig(ny=ny, nx=nx, psi_wall=psi_wall, psi_wall_top=-1.0,
                                 interface_width=xi, surface_tension=surface_tension)
    t0 = time.time()
    out = relax_to_equilibrium(cfg, radius=radius, centre_y=radius, max_iter=max_iter, omega=omega)
    wall = time.time() - t0
    meas = contact_angle_from_field(out["psi"], psi_wall=psi_wall, interface_width=xi)
    theta_analytic = math.degrees(analytic_contact_angle(psi_wall))
    theta = meas["theta_cap_deg"]
    payload = {
        "arm": "port",
        "tag": tag or f"port_psi{psi_wall:+.2f}_xi{xi:.1f}",
        "psi_wall": psi_wall,
        "interface_width": xi,
        "ny": ny, "nx": nx, "radius": radius,
        "iterations": out["iterations"],
        "converged": out["converged"],
        "drift": out["drift"],
        "mass_error": out["mass_error"],
        "theta_deg": theta,
        "theta_analytic_deg": theta_analytic,
        "err_pct": (theta - theta_analytic) / theta_analytic * 100.0 if math.isfinite(theta) else None,
        "pass": bool(math.isfinite(theta)
                     and abs(theta - theta_analytic) / theta_analytic * 100.0
                     < PREDECLARED["predeclared_thresholds"]["accept_pct"]),
        "measurement": meas,
        "wall_seconds": wall,
        "parameters": {"a": cfg.a, "kappa": cfg.kappa, "xi": xi, "sigma": surface_tension},
    }
    _write(payload["tag"], payload)
    np.savez(RUNS / payload["tag"] / "psi.npz", psi=out["psi"].numpy())
    return payload


# ---------------------------------------------------------------------------
# arm: cac (production phase-field solver, wall_order policy)
# ---------------------------------------------------------------------------
def arm_cac(psi_wall: float, steps: int = 2500, shape=(112, 112), tag: str | None = None) -> dict:
    from tensorlbm.wetting import ladder as L

    t0 = time.time()
    try:
        res = L.contact_wall_order(psi_wall, steps=steps, shape=shape)
    except Exception as exc:  # measurement or solver failure: recorded, never tuned
        res = {"theta_measured_deg": float("nan"), "err_pct": float("nan"),
               "meas": {"error": f"{type(exc).__name__}: {exc}"}, "psi_wall": psi_wall}
    theta = float(res.get("theta_measured_deg", float("nan")))
    theta_analytic = math.degrees(analytic_contact_angle(psi_wall))
    payload = {
        "arm": "cac",
        "tag": tag or f"cac_psi{psi_wall:+.2f}_steps{steps}",
        "psi_wall": psi_wall,
        "steps": steps,
        "shape": list(shape),
        "wall_phi_policy": "wall_order",
        "theta_deg": theta,
        "theta_analytic_deg": theta_analytic,
        "err_pct": (theta - theta_analytic) / theta_analytic * 100.0 if math.isfinite(theta) else None,
        "pass": bool(math.isfinite(theta)
                     and abs(theta - theta_analytic) / theta_analytic * 100.0
                     < PREDECLARED["predeclared_thresholds"]["accept_pct"]),
        "detail": res,
        "wall_seconds": time.time() - t0,
    }
    _write(payload["tag"], payload)
    return payload


# ---------------------------------------------------------------------------
# sweep + summary
# ---------------------------------------------------------------------------
def sweep(only: str | None = None, quick: bool = False) -> list[dict]:
    rows = []
    if only in (None, "port"):
        for psi_wall, xi, ny, nx in PREDECLARED["port_ladder"]:
            rows.append(arm_port(psi_wall, xi, ny=ny, nx=nx,
                                 max_iter=4000 if quick else 30000))
            print(json.dumps({k: rows[-1][k] for k in ("tag", "theta_deg", "theta_analytic_deg",
                                                       "err_pct", "pass")}), flush=True)
    if only in (None, "cac"):
        for psi_wall in PREDECLARED["cac_psi_ladder"]:
            rows.append(arm_cac(psi_wall, steps=1000 if quick else 2500))
            print(json.dumps({k: rows[-1][k] for k in ("tag", "theta_deg", "theta_analytic_deg",
                                                       "err_pct", "pass")}), flush=True)
    return rows


def summarize() -> dict:
    accept = PREDECLARED["predeclared_thresholds"]["accept_pct"]
    rows = []
    for f in sorted(RUNS.glob("*/result.json")):
        rows.append(json.loads(f.read_text()))
    port = [r for r in rows if r.get("arm") == "port"]
    cac = [r for r in rows if r.get("arm") == "cac"]
    errs = [abs(r["err_pct"]) for r in port if r.get("err_pct") is not None]
    window = PREDECLARED["validated_window"]["abs_psi_max"]
    in_win = [r for r in port if abs(r["psi_wall"]) <= window and r.get("err_pct") is not None]
    out_win = [r for r in port if abs(r["psi_wall"]) > window and r.get("err_pct") is not None]
    summary = {
        "anchor": PREDECLARED["spec_id"],
        "accept_pct": accept,
        "n_port": len(port),
        "n_cac": len(cac),
        "port_rows": [{k: r.get(k) for k in ("tag", "psi_wall", "interface_width", "theta_deg",
                                             "theta_analytic_deg", "err_pct", "pass", "converged",
                                             "iterations")}
                      | {"layer_fraction": r.get("measurement", {}).get("layer_fraction"),
                         "cap_measurable": r.get("measurement", {}).get("cap_measurable")}
                      for r in sorted(port, key=lambda r: (r["interface_width"], r["psi_wall"]))],
        "cac_rows": [{k: r.get(k) for k in ("tag", "psi_wall", "steps", "theta_deg",
                                            "theta_analytic_deg", "err_pct", "pass")} for r in cac],
        "gates": {
            "n_in_window": len(in_win),
            "n_out_of_window": len(out_win),
            "max_abs_err_pct_in_window": max((abs(r["err_pct"]) for r in in_win), default=None),
            "median_abs_err_pct_in_window": float(np.median([abs(r["err_pct"]) for r in in_win]))
            if in_win else None,
            "all_in_window_within_accept": bool(in_win) and all(abs(r["err_pct"]) < accept for r in in_win),
            "out_of_window_max_abs_err_pct": max((abs(r["err_pct"]) for r in out_win), default=None),
            "cac_all_within_accept": bool(cac) and all(r.get("pass") for r in cac),
            "all_converged": all(r.get("converged") for r in port),
        },
        "convergence_in_xi": {},
        "layer_capacity": {},
    }
    for r in port:
        summary["convergence_in_xi"].setdefault(f"{r['psi_wall']:+.2f}", {})[
            f"{r['interface_width']:.1f}"] = r["theta_deg"]
        summary["layer_capacity"][f"{r['psi_wall']:+.2f}@{r['interface_width']:.1f}"] = \
            r.get("measurement", {}).get("layer_fraction")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["port", "cac", "sweep", "summarize"])
    ap.add_argument("--psi-wall", type=float, default=None)
    ap.add_argument("--xi", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.cmd == "port":
        r = arm_port(args.psi_wall if args.psi_wall is not None else 0.35, args.xi,
                     max_iter=4000 if args.quick else 30000)
        print(json.dumps({k: r[k] for k in ("tag", "theta_deg", "theta_analytic_deg", "err_pct")}))
    elif args.cmd == "cac":
        r = arm_cac(args.psi_wall if args.psi_wall is not None else 0.35, steps=args.steps)
        print(json.dumps({k: r[k] for k in ("tag", "theta_deg", "theta_analytic_deg", "err_pct")}))
    elif args.cmd == "sweep":
        sweep(quick=args.quick)
        print(json.dumps(summarize()["gates"], indent=2))
    else:
        s = summarize()
        print(json.dumps(s["gates"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())