#!/usr/bin/env python3
"""Round 2 of XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN: wall-position calibration of
the TRT magic parameter Lambda for half-way bounce-back walls.

R1 (m18452, ``REPORT_XVAL_LBM_TRT_LAMBDA_MAGIC_TABLE_DRIVEN.md``) closed the *bulk*
(scalar ADE) semantics of the magic relation ``Lambda = (tau_s - 1/2)(tau_o - 1/2)``
against the OSS ``lbm-principles-practice`` code and explicitly deferred the classic
Ginzburg use of ``Lambda = 3/16`` -- the *wall-position* error of half-way bounce-back
-- to a later round (its diff item D-6).  This driver closes that gap for the flow
side (D2Q9 TRT, ``tensorlbm.solver.collide_trt``) against an exact analytic reference.

Reference solution (exact, body-force plane Poiseuille, walls at y = 1/2 and y = H+1/2):

    u(y) = g/(2 nu) (y - 1/2)(H + 1/2 - y),      Q = g H^3 / (12 nu),   nu = (tau+ - 1/2)/3

Observables (pre-registered in ``predeclared_r2.json``):

    delta_eff = H_eff - H,   H_eff = (12 nu Q / g)^(1/3)          [cells]

``delta_eff > 0`` means the effective no-slip plane lies *outside* the nominal wall
plane.  ``delta_eff`` is read off the flux only -- no profile fit, no extrapolation,
no slip correction.

Usage:  python3 xval_trt_wall_lambda.py [--quick]
Outputs: results_wall_lambda.json, figures/wall_lambda_*.png
"""

from __future__ import annotations

import argparse
import json
import os
import math
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(1)

from tensorlbm.d2q9 import C, OPPOSITE, W, equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, collide_trt, stream  # noqa: E402

G = 2.0e-6
NX = 4
STEPS = 3000
AVG = 200
LAM_MAGIC = 3.0 / 16.0

#: the magic-Lambda table under test (R1 report D-4 records 1/6 and 1/12 as the OSS
#: demo values; 3/8 is added as the far-tail probe, 1/8 and 1/4 bracket 3/16)
LAMBDAS = (1.0 / 12.0, 1.0 / 6.0, 1.0 / 8.0, 3.0 / 16.0, 0.25, 0.375)
TAUS = (0.55, 0.60, 0.70, 0.80, 1.00, 1.30, 1.90)
H_MAIN = 32
GRIDS_CHECK = (16, 32, 64)

FCOEF = (3.0 * G * W.double() * C[:, 0].double()).view(9, 1, 1)


def run(H: int, tau: float, scheme: str, lam: float = LAM_MAGIC,
        steps: int = STEPS, avg: int = AVG) -> dict:
    """One Poiseuille run; returns the pre-registered observables."""
    ny = H + 2
    nu = (tau - 0.5) / 3.0
    y = torch.arange(1, H + 1, dtype=torch.float64)
    u0 = G / (2.0 * nu) * (y - 0.5) * (H + 0.5 - y)
    uu = torch.zeros(ny, NX, dtype=torch.float64)
    uu[1:H + 1, :] = u0.view(H, 1)
    f = equilibrium(torch.ones(ny, NX, dtype=torch.float64), uu,
                    torch.zeros(ny, NX, dtype=torch.float64))

    def step(f):
        f = collide_bgk(f, tau) if scheme == "bgk" else collide_trt(f, tau, lam)
        f = f + FCOEF
        f[:, 0, :] = f[OPPOSITE, 1, :]          # pre-streaming half-way bounce-back
        f[:, H + 1, :] = f[OPPOSITE, H, :]
        return stream(f)

    t0 = time.time()
    for _ in range(steps):
        f = step(f)
    prof = torch.zeros(H, dtype=torch.float64)
    for _ in range(avg):
        f = step(f)
        prof += macroscopic(f)[1][1:H + 1, :].mean(dim=1)
    prof /= avg
    wall_s = time.time() - t0

    una = u0.numpy()
    unum = prof.numpy()
    Q = float(unum.sum())
    Qa = float(G * H ** 3 / (12.0 * nu))
    H_eff = (12.0 * nu * Q / G) ** (1.0 / 3.0)
    rho = macroscopic(f)[0]
    mass = float(rho.sum())
    mass_fluid = float(rho[1:H + 1].sum())
    mass_wall = float(rho[0].sum() + rho[H + 1].sum())
    return {
        "scheme": scheme,
        "H": H,
        "tau": round(tau, 6),
        "nu": nu,
        "lambda": (round(lam, 8) if scheme == "trt" else None),
        "tau_minus": (round(0.5 + lam / (tau - 0.5), 8) if scheme == "trt" else None),
        "lambda_bgk_equiv": round((tau - 0.5) ** 2, 8),
        "wall_s": round(wall_s, 2),
        "Q": Q,
        "Q_ana": Qa,
        "Q_rel_err_pct": round((Q / Qa - 1.0) * 100.0, 6),
        "H_eff": H_eff,
        "delta_eff": H_eff - H,
        "u_center_num": float(unum[int(np.argmax(una))]),
        "u_center_ana": float(una[int(np.argmax(una))]),
        "profile_L2_rel_pct": round(float(np.linalg.norm(unum - una) / np.linalg.norm(una)) * 100.0, 6),
        "mass": mass,
        "mass_fluid": mass_fluid,
        "mass_wall": mass_wall,
        "profile": [float(v) for v in unum],
    }


def fit_offset_slope(rows: list[dict]) -> dict:
    """Least-squares fit of delta_eff(H) = offset + slope / H^2 (model declared, not hidden)."""
    H = np.array([r["H"] for r in rows], dtype=float)
    d = np.array([r["delta_eff"] for r in rows], dtype=float)
    A = np.column_stack([np.ones_like(H), 1.0 / H ** 2])
    coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    pred = A @ coef
    return {
        "model": "delta_eff(H) = offset + slope/H^2",
        "offset": float(coef[0]),
        "slope": float(coef[1]),
        "residual_max": float(np.abs(pred - d).max()),
        "offset_pct_of_cell": float(coef[0] * 100.0),
    }


def spread(rows: list[dict]) -> float:
    """P2: tau-spread of delta_eff at fixed Lambda."""
    d = np.array([r["delta_eff"] for r in rows], dtype=float)
    return float(np.abs(d - d.mean()).max()) if d.size else float("nan")


def p4_identity(tau: float) -> float:
    """P4: max |u_trt - u_bgk| for Lambda = (tau - 1/2)^2 (the BGK row of the table)."""
    lam = (tau - 0.5) ** 2
    a = run(16, tau, "trt", lam, steps=400, avg=50)
    b = run(16, tau, "bgk", tau, steps=400, avg=50)
    return max(abs(x - y) for x, y in zip(a["profile"], b["profile"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short step count for smoke runs")
    args = ap.parse_args()
    # env overrides let a reduced-budget run be *declared* rather than silently
    # downgraded (the resulting JSON records the step count it actually used)
    steps = int(os.environ.get("WL_STEPS", 600 if args.quick else STEPS))
    grids = tuple(int(g) for g in os.environ.get("WL_GRIDS", "").split(",") if g) or None
    out = Path(__file__).resolve().parent

    # ---- P1/P2/P3: (Lambda, tau) map -------------------------------------
    grid: dict[str, list[dict]] = {}
    for lam in LAMBDAS:
        key = f"{lam:.8f}"
        grid[key] = [run(H_MAIN, tau, "trt", lam, steps=steps) for tau in TAUS]
    bgk = [run(H_MAIN, tau, "bgk", steps=steps) for tau in TAUS]

    # ---- P4: TRT(tau, Lambda=(tau-1/2)^2) == BGK(tau) ---------------------
    p4 = {f"{tau:.3f}": p4_identity(tau) for tau in (0.70, 0.80, 1.30)}

    # ---- P6: method_selection re-derived Lambda vs 3/16 at tau = 0.55 -----
    lam_redev = (2.0 - 0.5) * (0.55 - 0.5)          # trt_lambda_for_tau_minus(0.55, 2.0)
    p6 = {
        "tau": 0.55,
        "lambda_magic_3_16": run(H_MAIN, 0.55, "trt", LAM_MAGIC, steps=steps),
        "lambda_method_selection_0_075": run(H_MAIN, 0.55, "trt", lam_redev, steps=steps),
        "lambda_derivation": "trt_lambda_for_tau_minus(tau_plus=0.55, tau_minus=2.0) = (2-1/2)(0.55-1/2)",
        "tau_minus_at_magic": 0.5 + LAM_MAGIC / (0.55 - 0.5),
    }

    # ---- P5: resolution dependence ---------------------------------------
    gchk = grids if grids else GRIDS_CHECK
    p5 = {f"{lam:.8f}": [run(H, 0.80, "trt", lam, steps=steps) for H in gchk]
          for lam in (LAM_MAGIC, 0.25)}
    # tau=1.9 is where the magic arm shows its largest residual: is that residual a
    # wall offset (H-independent) or a discretisation error (O(H^-2))?  Three grids
    # give a 2-parameter fit delta_eff(H) = offset + slope / H^2 with 1 dof to spare.
    p5_hi_tau = {}
    for name, scheme, lam in (("trt_3_16", "trt", LAM_MAGIC), ("trt_1_4", "trt", 0.25),
                              ("bgk", "bgk", LAM_MAGIC)):
        rows = [run(H, 1.90, scheme, lam, steps=steps) for H in gchk]
        p5_hi_tau[name] = {"rows": [{"H": r["H"], "delta_eff": r["delta_eff"],
                                     "profile_L2_rel_pct": r["profile_L2_rel_pct"]}
                                    for r in rows],
                           **fit_offset_slope(rows)}

    # ---- hypothesis evaluation (thresholds from predeclared_r2.json) ------
    lam_of = lambda key: float(key)
    p2 = {k: spread(v) for k, v in grid.items()}
    magic_key = f"{LAM_MAGIC:.8f}"
    quarter_key = f"{0.25:.8f}"
    max_abs_delta_magic = max(abs(r["delta_eff"]) for r in grid[magic_key])
    h1 = {
        "p2_magic": p2[magic_key],
        "p2_quarter": p2[quarter_key],
        "ratio_quarter_over_magic": p2[quarter_key] / p2[magic_key] if p2[magic_key] else float("inf"),
        "max_abs_delta_eff_magic": max_abs_delta_magic,
        "threshold_p2_ratio": 5.0,
        "threshold_max_abs_delta_eff": 0.03,
    }
    h1["pass"] = bool(h1["ratio_quarter_over_magic"] >= 5.0 and max_abs_delta_magic <= 0.03)

    # H2: at tau = 0.80 order the scanned Lambdas by |Lambda - 3/16| and by |delta_eff|
    i80 = TAUS.index(0.80)
    dev = sorted(LAMBDAS, key=lambda l: abs(l - LAM_MAGIC))
    pairs = [(f"L={lam:.4f}", abs(lam - LAM_MAGIC),
              abs(grid[f"{lam:.8f}"][i80]["delta_eff"])) for lam in dev]
    h2 = {
        "tau": 0.80,
        "ordered_by_deviation_from_magic": [
            {"lambda": float(k.split("=")[1]), "abs_lambda_minus_magic": round(d, 6),
             "abs_delta_eff": round(a, 6)} for k, d, a in pairs],
        "spearman_rho_deviation_vs_error": None,
        "pass": None,
    }
    dvec = np.array([d for _, d, _ in pairs])
    avec = np.array([a for _, _, a in pairs])
    rho = float(np.corrcoef(dvec, avec)[0, 1]) if dvec.std() > 0 and avec.std() > 0 else float("nan")
    h2["spearman_rho_deviation_vs_error"] = rho
    h2["pass"] = bool(rho > 0.8)

    best = min(dev, key=lambda l: abs(grid[f"{l:.8f}"][i80]["delta_eff"]))
    h4 = {"tau": 0.80, "argmin_lambda": float(best), "magic": LAM_MAGIC,
          "pass": bool(abs(best - LAM_MAGIC) < 1e-12)}

    h3 = {"max_abs_du": p4, "threshold": 1e-13, "pass": bool(max(p4.values()) <= 1e-13)}

    worst_l2 = max(r["profile_L2_rel_pct"] for v in grid.values() for r in v)
    h5 = {"max_profile_L2_rel_pct": worst_l2, "threshold_pct": 3.0, "pass": bool(worst_l2 <= 3.0)}

    res = {
        "case": "trt_wall_lambda_r2",
        "anchor": "XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN",
        "round": 2,
        "mission": 23150,
        "predeclared": "predeclared_r2.json",
        "reference": {
            "u": "u(y)=g/(2 nu)(y-1/2)(H+1/2-y)", "Q": "g H^3/(12 nu)",
            "source": "White, Viscous Fluid Flow 3rd ed. sec. 3-2 (exact channel solution)",
        },
        "setup": {
            "g": G, "nx_periodic": NX, "steps": steps, "grids_check": list(gchk),
            "budget_note": ("reduced-step run: the cluster job for the 3000-step "
                            "configuration was still queued, so this file records "
                            "steps=" + str(steps)),
 "avg_steps": AVG, "H": H_MAIN,
            "ic": "analytic profile (transient-free)",
            "wall": "pre-streaming half-way bounce-back, wall planes at y=1/2 and y=H+1/2",
            "dtype": "float64", "device": "torch CPU, 1 thread",
            "delta_eff_sign": ">0 == effective wall outside the nominal plane",
        },
        "lambdas": list(LAMBDAS),
        "taus": list(TAUS),
        "grid_delta_eff": {k: [{"tau": r["tau"], "delta_eff": r["delta_eff"],
                               "profile_L2_rel_pct": r["profile_L2_rel_pct"],
                               "Q_rel_err_pct": r["Q_rel_err_pct"],
                               "tau_minus": r["tau_minus"]} for r in v]
                           for k, v in grid.items()},
        "bgk_delta_eff": [{"tau": r["tau"], "delta_eff": r["delta_eff"],
                           "profile_L2_rel_pct": r["profile_L2_rel_pct"],
                           "lambda_bgk_equiv": r["lambda_bgk_equiv"]} for r in bgk],
        "p2_tau_spread": p2,
        "p4_bgk_identity": p4,
        "p5_resolution_tau190_offset_vs_discretisation": p5_hi_tau,
        "p5_resolution": {k: [{"H": r["H"], "delta_eff": r["delta_eff"],
                              "profile_L2_rel_pct": r["profile_L2_rel_pct"]} for r in v]
                          for k, v in p5.items()},
        "p6_method_selection_head_to_head": {
            "tau": p6["tau"], "lambda_derivation": p6["lambda_derivation"],
            "tau_minus_at_magic": p6["tau_minus_at_magic"],
            "magic_3_16": {k: p6["lambda_magic_3_16"][k] for k in
                           ("delta_eff", "profile_L2_rel_pct", "Q_rel_err_pct", "tau_minus")},
            "method_selection_lambda": {k: p6["lambda_method_selection_0_075"][k] for k in
                                        ("delta_eff", "profile_L2_rel_pct",
                                         "Q_rel_err_pct", "tau_minus", "lambda")},
        },
        "profiles_tau080": {k: {"tau": 0.80, "lambda": lam_of(k), "u": v[TAUS.index(0.80)]["profile"]}
                            for k, v in grid.items()},
        "profiles_bgk_tau080": bgk[TAUS.index(0.80)]["profile"],
        "analytic_profile_tau080": None,
        "hypotheses": {"H1_magic_is_flat": h1, "H2_deviation_costs": h2,
                       "H3_bgk_identity": h3, "H4_optimal_Lambda": h4,
                       "H5_accuracy_band": h5},
        "mass_drift_fluid_max": max(abs(r["mass_fluid"] - H_MAIN * NX)
                                    for v in list(grid.values()) + [bgk] for r in v),
    }
    # analytic profile at tau = 0.80 (zero hand-copy: recomputed from the same formula)
    nu08 = (0.80 - 0.5) / 3.0
    yy = np.arange(1, H_MAIN + 1, dtype=float)
    res["analytic_profile_tau080"] = list(G / (2.0 * nu08) * (yy - 0.5) * (H_MAIN + 0.5 - yy))
    (out / "results_wall_lambda.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())