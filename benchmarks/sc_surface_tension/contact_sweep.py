#!/usr/bin/env python
"""Contact-angle sweep for the wetting boundary condition (node P143W2-N01).

Two biases are separated on purpose:

1. **Instrument bias** -- the mid-density contour of a *discretised* circular cap
   does not reproduce the exact cap geometry, so the circle-fit estimator carries
   a small systematic offset.  It is measured once against analytic caps
   (:func:`synthetic_cap_bias`) and used as a calibration curve.
2. **Model bias** -- Young's relation ``cos(theta) = (psi_w-psi_g)/(psi_l-psi_g)``
   (eq. 7 of the module) is a *leading-order* mapping for the pseudopotential
   wall; the realised angle differs.  A five-point sweep measures the realised
   mapping and the held-out targets quantify how well it can be inverted.

Usage::

    python benchmarks/sc_surface_tension/contact_sweep.py --task sweep
    python benchmarks/sc_surface_tension/contact_sweep.py --task assemble
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch  # noqa: E402

from tensorlbm.sc_surface_tension import (  # noqa: E402
    MID_RHO,
    contact_angle_from_mask,
    psi_exp,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run import _jsonable, run_contact_angle  # noqa: E402


def psi_of_rho(rho_w: float) -> float:
    return float(psi_exp(torch.tensor(float(rho_w))))


def synthetic_cap_bias(ny: int = 96, nx: int = 160, wall: int = 2,
                       thetas=(45, 60, 75, 90, 120, 150), R: float = 30.0) -> list[dict]:
    """Estimator bias of :func:`contact_angle_from_rho` on analytic caps."""
    solid = torch.zeros((ny, nx), dtype=torch.bool)
    solid[:wall, :] = True
    solid[ny - wall:, :] = True
    yy = torch.arange(ny, dtype=torch.float32).view(ny, 1).expand(ny, nx)
    xx = torch.arange(nx, dtype=torch.float32).view(1, nx).expand(ny, nx)
    rows = []
    for th in thetas:
        zc = -R * math.cos(math.radians(th))
        mask = ((xx - 0.5 * (nx - 1)) ** 2 + (yy - (wall - 1 + 0.5 + zc)) ** 2) < R * R
        rho = torch.where(mask, torch.tensor(1.9), torch.tensor(0.15))
        q = contact_angle_from_mask(rho, solid)
        rows.append({"theta_true": float(th), "theta_measured": q["theta_deg"],
                     "bias_deg": q["theta_deg"] - float(th), "R_fit": q["R_fit"]})
    return rows


def correct(theta_meas: float, table: list[dict]) -> float:
    """Subtract the interpolated analytic-cap bias from a measured angle."""
    pts = sorted(((r["theta_measured"], r["bias_deg"]) for r in table))
    xs = [p[0] for p in pts]
    if not xs:
        return theta_meas
    if theta_meas <= xs[0]:
        return theta_meas - pts[0][1]
    if theta_meas >= xs[-1]:
        return theta_meas - pts[-1][1]
    for i in range(1, len(pts)):
        if theta_meas <= xs[i]:
            t = (theta_meas - xs[i - 1]) / (xs[i] - xs[i - 1])
            b = pts[i - 1][1] + t * (pts[i][1] - pts[i - 1][1])
            return theta_meas - b
    return theta_meas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="sweep", choices=["sweep", "assemble"])
    ap.add_argument("--rho-w", default="0.6,0.9,1.2,1.5,1.8")
    ap.add_argument("--holdout", default="1.05,1.35")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--ny", type=int, default=72)
    ap.add_argument("--nx", type=int, default=112)
    ap.add_argument("--R0", type=float, default=18.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "artifacts" / "sc_surface_tension"))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    table = synthetic_cap_bias(ny=args.ny, nx=args.nx)
    (out / "contact_instrument_bias.json").write_text(json.dumps(_jsonable(table), indent=1))

    runs = []
    specs = [("cal", float(v)) for v in args.rho_w.split(",")]
    specs += [("holdout", float(v)) for v in args.holdout.split(",")]
    for kind, rho_w in specs:
        psi_w = psi_of_rho(rho_w)
        # target angle from Young's relation with this psi_w (eq. 7)
        pl, pg = psi_of_rho(1.957), psi_of_rho(0.1596)
        cos_t = max(-1.0, min(1.0, (psi_w - pg) / (pl - pg)))
        theta_young = math.degrees(math.acos(cos_t))
        res = run_contact_angle(theta_young, R0=args.R0, ny=args.ny, nx=args.nx,
                                steps=args.steps, sample=2000, verbose=True,
                                wall_override=psi_w)
        theta_cal = correct(res["theta_measured"], table)
        row = {
            "kind": kind,
            "rho_w": rho_w,
            "psi_w": psi_w,
            "theta_young_deg": theta_young,
            "theta_measured_deg": res["theta_measured"],
            "theta_corrected_deg": theta_cal,
            "bias_vs_young_deg": res["theta_measured"] - theta_young,
            "bias_corrected_vs_young_deg": theta_cal - theta_young,
            "R_fit": res["R_fit"],
            "z_c": res["z_c"],
            "n_contour": res["n_contour"],
            "mass_drift": res["mass_drift"],
            "steps": res["steps"],
            "status": res.get("status", "ok"),
        }
        runs.append(row)
        (out / f"contact_sweep_rho{rho_w:.2f}.json").write_text(json.dumps(_jsonable(row), indent=1))
        print(f"[sweep] rho_w={rho_w:.2f} psi_w={psi_w:.3f} young={theta_young:.1f} "
              f"meas={res['theta_measured']:.1f} corr={theta_cal:.1f}", flush=True)
    (out / "contact_sweep_rows.json").write_text(json.dumps(_jsonable(runs), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())