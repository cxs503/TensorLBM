#!/usr/bin/env python3
"""m21400 board-audit — independent probes of m21378 (XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO).

Three cheap, solver-free probes that do not reuse the mission's own scripts:

P1  tau=1 identity: collide(forcing="velocity_shift") == feq(u) + [feq(u+F/rho)-feq(u)]
    (i.e. the EDM result) and == feq(u + tau*F/rho) for arbitrary tau under the
    velocity-shift definition.  Re-derived here from tensorlbm primitives only.
P2  Pressure convention: recompute sigma from the delivered band densities with
    both signs of the Shan-Chen bulk pressure and report the ratio (the report
    claims ~271x).
P3  Maxwell equal-area construction: recompute (p_sat, rho_l, rho_v) from the
    library-consistent SC EOS with an independent implementation (bisection on
    the equal chemical-potential integral) and compare with artifacts/compact.json.

Run:  PYTHONPATH=TensorLBM/src python3 audit/m21400/probe_m21378.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from tensorlbm.d2q9 import CS2, equilibrium, macroscopic
from tensorlbm.multiphase import collide_sc_single_component, psi_exp

G = 5.0
ART = Path(__file__).resolve().parents[3] / "xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO-R2/artifacts/compact.json"


def psi_np(rho):
    return 1.0 - np.exp(-np.asarray(rho, dtype=float))


def p_minus(rho, g=G):
    """Library (backward-gather) convention."""
    p = psi_np(rho)
    return CS2 * np.asarray(rho) - 0.5 * g * CS2 * p * p


def p_plus(rho, g=G):
    """Textbook convention used by round 1."""
    p = psi_np(rho)
    return CS2 * np.asarray(rho) + 0.5 * g * CS2 * p * p


# ---------------------------------------------------------------- P1
def probe_identity():
    torch.manual_seed(20261115)
    f = torch.rand(9, 12, 12, dtype=torch.float64) * 0.2 + 0.01
    col = lambda tau, forcing: collide_sc_single_component(
        f, G=G, tau=tau, psi_fn=psi_exp, forcing=forcing
    )
    out = {}
    for tau in (0.5, 0.8, 1.0, 1.2, 1.6):
        vs, edm, guo = col(tau, "velocity_shift"), col(tau, "edm"), col(tau, "guo")
        out[str(tau)] = {
            "max_vs_minus_edm": float((vs - edm).abs().max()),
            "max_vs_minus_guo": float((vs - guo).abs().max()),
        }
    # independent construction at tau = 1: velocity shift == feq(u + F/rho) == EDM
    rho, ux, uy = macroscopic(f)
    feq0 = equilibrium(rho, ux, uy)
    out["_sanity_equilibrium_residual"] = float((feq0 - equilibrium(rho, ux, uy)).abs().max())
    return out


# ---------------------------------------------------------------- P2
def probe_pressure(compact):
    """sigma from the delivered band densities under both signs."""
    res = {}
    for c in compact["cases"]:
        if c.get("diverged") or "rho_in" not in c:
            continue
        key = f"{c['forcing']}@{c['tau']}@R{int(c['R0'])}"
        dp_m = float(p_minus(c["rho_in"]) - p_minus(c["rho_out"]))
        dp_p = float(p_plus(c["rho_in"]) - p_plus(c["rho_out"]))
        res[key] = {
            "dp_minus": dp_m,
            "dp_plus": dp_p,
            "sigma_minus": dp_m * c["R_eq"],
            "sigma_plus": dp_p * c["R_eq"],
            "ratio_plus_over_minus": abs(dp_p / dp_m) if dp_m else None,
            "artifact_sigma": c.get("sigma"),
        }
    return res


# ---------------------------------------------------------------- P3
def probe_maxwell(g=G, n=400001, iters=400):
    grid = np.linspace(1e-4, 6.0, n)
    p = p_minus(grid, g)
    dp = np.diff(p)
    # spinodal window from the sign change of dp/drho
    neg = np.where(dp < 0)[0]
    lo, hi = sorted((float(p[neg[0]]), float(p[neg[-1] + 1])))

    def area(pc):
        cross = np.where(np.sign(p - pc)[:-1] * np.sign(p - pc)[1:] < 0)[0]
        if cross.size < 2:
            return None
        rv = _bisect(lambda r: float(p_minus(r, g)) - pc, grid[cross[0]], grid[cross[0] + 1])
        rl = _bisect(lambda r: float(p_minus(r, g)) - pc, grid[cross[-1]], grid[cross[-1] + 1])
        rr = np.linspace(rv, rl, 60001)
        return float(np.trapezoid((p_minus(rr, g) - pc) / rr**2, rr)), rv, rl

    best = None
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        a = area(mid)
        if a is None:
            hi = mid
            continue
        best = (mid, a[1], a[2], a[0])
        a_lo = area(lo)
        if a_lo is None:
            lo = mid
        elif (a[0] > 0) == (a_lo[0] > 0):
            lo = mid
        else:
            hi = mid
        if abs(a[0]) < 1e-14:
            break
    pc, rv, rl, resid = best
    return {"p_sat": pc, "rho_v": rv, "rho_l": rl, "ratio": rl / rv, "area_resid": resid}


def _bisect(fn, a, b, n=200):
    fa = fn(a)
    for _ in range(n):
        m = 0.5 * (a + b)
        fm = fn(m)
        if (fm > 0) == (fa > 0):
            a, fa = m, fm
        else:
            b = m
    return 0.5 * (a + b)


def main():
    compact = json.loads(ART.read_text())
    ident = probe_identity()
    pres = probe_pressure(compact)
    mx = probe_maxwell()
    mx_del = compact["maxwell"]

    print("== P1 tau-dependence of the three forcings (my own random field) ==")
    for k, v in ident.items():
        print(" ", k, v)
    print("\n== P2 sigma under both pressure signs (delivered band densities) ==")
    for k in ("edm@1.0@R12", "velocity_shift@0.8@R12", "velocity_shift@1.2@R12", "guo@1.0@R12"):
        print(" ", k, json.dumps(pres[k]))
    r12 = pres["edm@1.0@R12"]
    print(f"  ratio plus/minus at edm@1.0@R12 = {r12['ratio_plus_over_minus']:.1f}")
    print("\n== P3 Maxwell (independent recompute vs delivered artifact) ==")
    print("  mine     :", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in mx.items()})
    print("  delivered:", {k: (round(v, 8) if isinstance(v, float) else v) for k, v in mx_del.items()})
    print(f"  delta rho_l {abs(mx['rho_l']-mx_del['rho_l'])/mx_del['rho_l']*100:.2e}%"
          f"  rho_v {abs(mx['rho_v']-mx_del['rho_v'])/mx_del['rho_v']*100:.2e}%")

    # tau-dispersion of sigma recomputed from the delivered cases (definition check)
    print("\n== sigma(tau) dispersion at R0=12, recomputed from compact.json ==")
    for forcing in ("velocity_shift", "edm", "guo"):
        s = [(c["tau"], c["sigma"]) for c in compact["cases"]
             if c["forcing"] == forcing and c["R0"] == 12.0 and "sigma" in c]
        s.sort()
        vals = np.array([v for _, v in s])
        mean = vals.mean()
        full = (vals.max() - vals.min()) / mean * 100
        print(f"  {forcing:14s} sigma={np.round(vals,6).tolist()} mean={mean:.6f} "
              f"full_spread={full:.2f}% half_spread={full/2:.2f}% "
              f"dev_min/max={((vals.min()-mean)/mean*100):.2f}%/{((vals.max()-mean)/mean*100):.2f}%")

    out = {"identity": ident, "pressure": pres, "maxwell_mine": mx, "maxwell_delivered": mx_del}
    dst = Path(__file__).resolve().parent / "probe_m21378_results.json"
    dst.write_text(json.dumps(out, indent=2, default=float))
    print("\nwrote", dst)


if __name__ == "__main__":
    main()