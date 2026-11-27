#!/usr/bin/env python3
"""XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO (round 2) — forcing-scheme discriminance on
the **library path**.

Round 1 (``TensorLBM/xval/XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO``, m18323) compared a
self-contained NumPy re-implementation against itself and reached "PASS (<15%)".
Round 2 re-asks the same anchor question with three corrections:

* **Object**: every number here is produced by ``tensorlbm.multiphase
  .collide_sc_single_component`` + ``tensorlbm.solver.stream`` — the shipped
  library path, not a re-implementation.
* **Pressure convention**: the SC bulk pressure used to evaluate Young-Laplace is
  the one implied by the library's backward-gather force convention,
  ``p_mech(rho) = cs^2 rho - (G/2) cs^2 psi^2``  (``G = +5`` attractive),
  which is algebraically identical to the textbook
  ``p = cs^2 rho + (G_eff cs^2/2) psi^2`` with ``G_eff = -G``.
* **Non-degenerate design**: at ``tau = 1`` the EDM and velocity-shift collide
  operators are *algebraically identical* (see ``LEG_IDENTITY``); round 1 swept
  only ``tau = 1``, so its "vs vs edm" arms could not differ by construction.
  This sweep therefore runs ``tau in {0.8, 1.0, 1.2}``.

Legs
----
LEG_IDENTITY  algebraic identity tau=1: ``|f_vs - f_edm|`` must be O(1e-16);
              at tau != 1 it must be O(1e-4) per step.  Proves the round-1
              design flaw numerically.
LEG_LAPLACE   Young-Laplace ``dp = sigma / R_eq`` over 5 radii per
              (forcing, tau): slope through the origin = sigma, linearity R^2,
              and spread of the per-radius sigma (Laplace-law consistency).
LEG_MAXWELL   Maxwell equal-area construction on the SC EOS (analytic, no
              lattice solver) -> ``rho_l, rho_v, p_sat``; compared with the
              measured band densities (thermodynamic acceptance, <15%).
LEG_SCHEME    Pairwise discriminance of the three forcings at matched (tau, R)
              in sigma, spurious current and coexistence density.

Usage
-----
    python run_xval.py --quick                  # smoke (1 radius, 1 tau)
    python run_xval.py --shard 0 --nshards 9    # one slice of the full study
    python run_xval.py --merge                  # merge shard JSONs -> result.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

FORCINGS = ("velocity_shift", "edm", "guo")
TAUS = (0.8, 1.0, 1.2)
RADII = (10.0, 12.0, 14.0, 16.0, 18.0)

CFG = {
    "model": "SCMP single-component Shan-Chen (Shan & Chen 1993/1994), D2Q9",
    "psi": "psi(rho) = 1 - exp(-rho)  (rho0 = 1)",
    "G_lib": 5.0,
    "G_eff": -5.0,
    "lattice": "D2Q9",
    "rho_l_init": 1.957,
    "rho_v_init": 0.1596,
    "w_int": 4.0,
    "domain_rule": "L = 4 R (same rule as benchmarks/verified/laplace_droplet)",
    "bands": (0.5, 1.5),
    "sample_every": 250,
    "max_steps": 4000,
    "conv_rel_dp": 2e-3,
    "sigma_locked_doc": 0.0561,  # tensorlbm docs, velocity_shift, tau=1
    "criteria_pct": 15.0,
}


# ---------------------------------------------------------------------------
# Reference (analytic / independent) pieces — no lattice solver involved
# ---------------------------------------------------------------------------
CS2 = 1.0 / 3.0


def psi_exp_np(rho):
    return 1.0 - np.exp(-np.asarray(rho))


def p_mech(rho, G=5.0):
    """Library-convention mechanical pressure  p = cs^2 rho - (G/2) cs^2 psi^2."""
    psi = psi_exp_np(rho)
    return CS2 * np.asarray(rho) - 0.5 * G * CS2 * psi * psi


def maxwell_coexistence(G=5.0, tol=1e-13):
    """Maxwell equal-area construction on the SC EOS (analytic reference).

    p(rho) = cs^2 rho - (G/2) cs^2 psi(rho)^2 ; find p_sat such that the two
    outer roots of p(rho) = p_sat satisfy
        int_{rho_v}^{rho_l} (p(rho) - p_sat)/rho^2 drho = 0
    (equal chemical potential).  Bisection on p_sat between the two spinodal
    pressures; returns rho_v < rho_l and p_sat.
    """
    from scipy.optimize import brentq

    grid = np.linspace(1e-4, 6.0, 200001)
    pgrid = p_mech(grid, G)
    dpgrid = np.gradient(pgrid, grid)
    neg = np.where(dpgrid < 0)[0]
    if neg.size == 0:
        raise RuntimeError("no spinodal: G below critical")
    # p at the vapour spinodal is HIGHER than at the liquid spinodal, so the
    # admissible p_sat window is [min, max] of the two spinodal pressures.
    lo, hi = sorted((float(pgrid[neg[0]]), float(pgrid[neg[-1]])))

    def area(pc):
        fl = pgrid - pc
        idx = np.where(np.sign(fl[:-1]) * np.sign(fl[1:]) < 0)[0]
        if idx.size < 2:
            return None
        rv = brentq(lambda r: float(p_mech(r, G)) - pc, grid[idx[0]], grid[idx[0] + 1])
        rl = brentq(lambda r: float(p_mech(r, G)) - pc, grid[idx[-1]], grid[idx[-1] + 1])
        rr = np.linspace(rv, rl, 40001)
        return float(np.trapezoid((p_mech(rr, G) - pc) / rr**2, rr)), rv, rl

    a_lo = area(lo)
    best = None
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        am = area(mid)
        if am is None:
            hi = mid
            continue
        best = (mid, am[1], am[2])
        a_lo = area(lo)
        if a_lo is not None:
            if (am[0] > 0) == (a_lo[0] > 0):
                lo = mid
            else:
                hi = mid
        else:
            lo = mid
        if abs(am[0]) < tol or (hi - lo) < 1e-12:
            break
    pc, rv, rl = best
    return {"p_sat": pc, "rho_v": rv, "rho_l": rl, "ratio": rl / rv}


# ---------------------------------------------------------------------------
# Library-path solver
# ---------------------------------------------------------------------------
class LibSolver:
    """Thin wrapper over the shipped tensorlbm SCMP collide + stream."""

    def __init__(self, forcing: str, tau: float, dtype=None, threads: int = 1):
        import torch

        torch.set_num_threads(threads)
        self.t = torch
        self.dtype = dtype or torch.float32
        from tensorlbm.d2q9 import equilibrium, macroscopic
        from tensorlbm.multiphase import collide_sc_single_component, psi_exp
        from tensorlbm.solver import stream

        self._eq, self._mac = equilibrium, macroscopic
        self._collide, self._psi, self._stream = collide_sc_single_component, psi_exp, stream
        self.forcing, self.tau = forcing, tau

    def init(self, rho):
        r = self.t.tensor(np.ascontiguousarray(rho), dtype=self.dtype)
        z = self.t.zeros_like(r)
        return self._eq(r, z, z)

    def step(self, f):
        return self._stream(
            self._collide(f, G=CFG["G_lib"], tau=self.tau, psi_fn=self._psi,
                          forcing=self.forcing)
        )

    def fields(self, f):
        rho, ux, uy = self._mac(f)
        return rho.numpy().astype(np.float64), ux.numpy().astype(np.float64), uy.numpy().astype(np.float64)


def init_droplet(L, R):
    ys, xs = np.meshgrid(np.arange(L) + 0.5, np.arange(L) + 0.5, indexing="ij")
    rr = np.hypot(xs - L / 2.0, ys - L / 2.0)
    rho = CFG["rho_v_init"] + 0.5 * (CFG["rho_l_init"] - CFG["rho_v_init"]) * (
        1.0 + np.tanh((R - rr) / CFG["w_int"])
    )
    return np.clip(rho, 1e-3, None), rr


def measure(rho, ux, uy, rr, R_guess):
    """Young-Laplace measurement with a self-consistent equivalent radius."""
    p = p_mech(rho, CFG["G_lib"])
    umag = np.hypot(ux, uy)
    r_eq = float(R_guess)
    for _ in range(8):
        inside = rr <= 0.5 * r_eq
        outside = rr >= 1.5 * r_eq
        mid = 0.5 * (rho[inside].mean() + rho[outside].mean())
        n_liq = int((rho > mid).sum())
        r_new = math.sqrt(n_liq / math.pi)
        if abs(r_new - r_eq) < 1e-6:
            r_eq = r_new
            break
        r_eq = r_new
    a, b = CFG["bands"]
    inside = rr <= a * r_eq
    outside = rr >= b * r_eq
    out = {
        "R_eq": r_eq,
        "p_in": float(p[inside].mean()),
        "p_out": float(p[outside].mean()),
        "rho_in": float(rho[inside].mean()),
        "rho_out": float(rho[outside].mean()),
        "u_max": float(umag.max()),
        "u_mean": float(umag.mean()),
        "mass": float(rho.sum()),
        "fs": float(np.abs(np.isfinite(rho) - 1).sum()),
    }
    out["dp"] = out["p_in"] - out["p_out"]
    out["rho_ratio"] = out["rho_in"] / max(out["rho_out"], 1e-12)
    out["sigma"] = out["dp"] * out["R_eq"]
    return out


def run_case(forcing, tau, R, max_steps=None, threads=1):
    L = int(round(4 * R))
    rho0, rr = init_droplet(L, R)
    sol = LibSolver(forcing, tau, threads=threads)
    f = sol.init(rho0)
    max_steps = max_steps or CFG["max_steps"]
    hist, prev = [], None
    t0 = time.time()
    diverged = False
    for n in range(1, max_steps + 1):
        f = sol.step(f)
        if n % CFG["sample_every"] == 0:
            rho, ux, uy = sol.fields(f)
            if not np.all(np.isfinite(rho)):
                diverged = True
                break
            m = measure(rho, ux, uy, rr, R)
            hist.append({"step": n, **m})
            if prev is not None and prev["dp"] != 0:
                if abs(m["dp"] - prev["dp"]) <= CFG["conv_rel_dp"] * abs(prev["dp"]):
                    prev = m
                    break
            prev = m
    res = {
        "forcing": forcing, "tau": tau, "R0": R, "L": L,
        "steps": hist[-1]["step"] if hist else 0,
        "wall_s": round(time.time() - t0, 2),
        "diverged": diverged,
        "history": [{"step": h["step"], "dp": h["dp"], "u_max": h["u_max"]} for h in hist[-6:]],
    }
    if hist and not diverged:
        final = hist[-1]
        res.update({k: v for k, v in final.items() if k != "step"})
    return res


def leg_identity():
    """tau=1: EDM and velocity-shift must coincide to machine precision."""
    import torch

    torch.set_num_threads(1)
    sys.path.insert(0, str(REPO / "src"))
    from tensorlbm.multiphase import collide_sc_single_component, psi_exp

    g = torch.Generator().manual_seed(20261115)
    f = torch.rand(9, 16, 16, generator=g, dtype=torch.float64) * 0.1
    out = {}
    for tau in (1.0, 0.8, 1.2):
        fa = collide_sc_single_component(f, G=5.0, tau=tau, psi_fn=psi_exp, forcing="velocity_shift")
        fb = collide_sc_single_component(f, G=5.0, tau=tau, psi_fn=psi_exp, forcing="edm")
        fc = collide_sc_single_component(f, G=5.0, tau=tau, psi_fn=psi_exp, forcing="guo")
        out[str(tau)] = {
            "max_abs_vs_minus_edm": float((fa - fb).abs().max()),
            "max_abs_vs_minus_guo": float((fa - fc).abs().max()),
        }
    return out


def leg_laplace(cases):
    """dp = sigma / R through the origin, per (forcing, tau)."""
    groups = {}
    for c in cases:
        if c.get("diverged") or "dp" not in c:
            continue
        groups.setdefault((c["forcing"], c["tau"]), []).append(c)
    out = {}
    for (forcing, tau), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["R_eq"])
        if len(rows) < 3:
            out[f"{forcing}@{tau}"] = {"n": len(rows), "note": "insufficient radii"}
            continue
        x = np.array([1.0 / r["R_eq"] for r in rows])
        y = np.array([r["dp"] for r in rows])
        sig = float((x * y).sum() / (x * x).sum())           # slope through origin
        pred = sig * x
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        sigs = np.array([r["sigma"] for r in rows])
        out[f"{forcing}@{tau}"] = {
            "n": len(rows),
            "sigma_fit": sig,
            "r2_origin": r2,
            "radii": [round(r["R_eq"], 3) for r in rows],
            "sigma_per_radius": [round(float(s), 6) for s in sigs],
            "sigma_spread_pct": float((sigs.max() - sigs.min()) / abs(sigs.mean()) * 100.0),
            "dp_per_radius": [round(float(v), 6) for v in y],
            "resid_pct": [round(float(abs(yi - pi) / abs(yi)) * 100.0, 3) for yi, pi in zip(y, pred)],
        }
    return out


def leg_maxwell(cases, mx):
    out = {}
    for c in cases:
        if c.get("diverged") or "rho_in" not in c:
            continue
        key = f"{c['forcing']}@{c['tau']}@R{c['R0']:.0f}"
        e_l = abs(c["rho_in"] - mx["rho_l"]) / mx["rho_l"] * 100.0
        e_v = abs(c["rho_out"] - mx["rho_v"]) / mx["rho_v"] * 100.0
        out[key] = {"rho_in": c["rho_in"], "rho_out": c["rho_out"],
                    "err_rho_l_pct": e_l, "err_rho_v_pct": e_v}
    return out


def leg_scheme(cases):
    """Pairwise discriminance at matched (tau, R)."""
    idx = {(c["forcing"], c["tau"], c["R0"]): c for c in cases if not c.get("diverged")}
    out = {}
    for tau in sorted({c["tau"] for c in cases}):
        for R in sorted({c["R0"] for c in cases}):
            vs, ed, gu = (idx.get((f, tau, R)) for f in FORCINGS)
            if not all((vs, ed, gu)):
                continue
            out[f"tau{tau}_R{R:.0f}"] = {
                "sigma": {f: (idx[(f, tau, R)].get("sigma") if (f, tau, R) in idx else None) for f in FORCINGS},
                "u_max": {f: idx[(f, tau, R)].get("u_max") for f in FORCINGS},
                "d_sigma_pct_edm_vs_vs": (ed.get("sigma", np.nan) - vs.get("sigma", np.nan)) / vs.get("sigma", np.nan) * 100.0,
                "d_sigma_pct_guo_vs_vs": (gu.get("sigma", np.nan) - vs.get("sigma", np.nan)) / vs.get("sigma", np.nan) * 100.0,
                "d_u_pct_edm_vs_vs": (ed["u_max"] - vs["u_max"]) / vs["u_max"] * 100.0,
                "d_u_pct_guo_vs_vs": (gu["u_max"] - vs["u_max"]) / vs["u_max"] * 100.0,
            }
    return out


def build_tasks(quick=False):
    if quick:
        return [(f, 1.0, 12.0) for f in FORCINGS] + [("edm", 0.8, 12.0), ("velocity_shift", 0.8, 12.0)]
    return [(f, tau, R) for tau in TAUS for f in FORCINGS for R in RADII]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--shard", type=int, default=1)   # 1-based
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    args = ap.parse_args()

    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)

    if args.merge:
        cases, ident = [], None
        for p in sorted(art.glob("shard_*.json")):
            d = json.loads(p.read_text())
            cases.extend(d["cases"])
            ident = ident or d.get("identity")
        mx = maxwell_coexistence(CFG["G_lib"])
        res = {
            "anchor": "XVAL-LBM-SHAN-CHEN-EDM-LYCETT-LUO",
            "round": 2,
            "cfg": CFG,
            "maxwell": mx,
            "identity": ident,
            "cases": cases,
            "legs": {
                "laplace": leg_laplace(cases),
                "maxwell": leg_maxwell(cases, mx),
                "scheme": leg_scheme(cases),
            },
        }
        (art / "result.json").write_text(json.dumps(res, indent=2, default=float))
        print(f"merged {len(cases)} cases -> {art / 'result.json'}")
        return

    tasks = build_tasks(args.quick)
    sid = (args.shard - 1) % args.nshards
    mine = [t for k, t in enumerate(tasks) if k % args.nshards == sid]
    print(f"shard {sid + 1}/{args.nshards}: {len(mine)} cases of {len(tasks)}", flush=True)
    cases = []
    for forcing, tau, R in mine:
        r = run_case(forcing, tau, R, args.max_steps, args.threads)
        print(f"  {forcing:14s} tau={tau:<4} R={R:<5} steps={r['steps']:<5} "
              f"dp={r.get('dp', float('nan')):.5f} sigma={r.get('sigma', float('nan')):.5f} "
              f"u_max={r.get('u_max', float('nan')):.3e} div={r['diverged']} ({r['wall_s']}s)",
              flush=True)
        cases.append(r)
    payload = {"cases": cases, "identity": leg_identity()}
    (art / f"shard_{sid:02d}.json").write_text(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()