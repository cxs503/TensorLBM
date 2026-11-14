"""R1 adversarial verification of XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION.

Motivation
----------
The first-pass unit (mission m20921) reported a converged Shapiro-Chen droplet
calibration sigma = 0.0555 (<3% vs the library-locked 0.0561), but left one
fact unexplained and one leg untested:

  (a) the forcing-scheme sensitivity was **56.8%** (velocity_shift dp=0.003421
      vs guo dp=0.001908 at R=16, L=6R) with coexistence densities moving in
      opposite directions (rho_v 0.1605 -> 0.1248).  The report recorded the
      fact and attributed it to "a converged model difference" without test.
  (b) sigma was only measured at tau = 1.0 (nu = 1/6); the tau dependence was
      listed as not done.

This script tests (a) and (b) with an independent probe that does not use the
droplet protocol at all: a **flat (planar) interface**.  A planar SC interface
has dp = 0 by construction, so the coexistence densities it selects are the
scheme's own coexistence state, and the SC equation of state must return the
same bulk pressure on both sides (p_l == p_v) for mechanical equilibrium.

  * If a scheme's flat-interface state matches its droplet bulk densities, the
    56.8% spread is a genuine lattice-EOS (scheme-dependent coexistence) shift
    and the *droplet* sigma values are not transferable across schemes.
  * If it does not, the droplet run of that scheme is not at equilibrium and
    the reported sigma for that scheme is invalid (a defect, not physics).

Legs
----
V1  flat interface, 3 forcing schemes -> (rho_l, rho_v, p_l, p_v) vs the
    analytic Maxwell coexistence of the SC EOS (ref_sc.maxwell_coexistence).
V2  droplet R in {16, 20}, L = 4R, 3 forcing schemes -> dp, sigma = dp*R_eq,
    and dp measured against the scheme's **own** flat coexistence pressure
    (removes the finite-box vapour reference bias).
V3  tau in {0.7, 1.0, 1.5} at R = 20 -> sigma(tau) drift (leg (b) above).

All numbers are written to artifacts/verify_r1.json; nothing is hand-copied.
Run:  python verify_r1.py           (add --quick for a 3-minute smoke run)
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
XV_DIR = HERE.parent
REPO = XV_DIR.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(XV_DIR))

import ref_sc as REF  # noqa: E402
import run_xval as XV  # noqa: E402  (reused CFG / TorchSolver / init_droplet)

CFG = XV.CFG
SCHEMES = ("velocity_shift", "edm", "guo")


# ---------------------------------------------------------------------------
# V1 - flat (planar) interface: the scheme's own coexistence state
# ---------------------------------------------------------------------------
def init_flat(nx: int, ny: int, rho_l: float, rho_v: float, w: float, half: float):
    """Liquid slab in the middle of a periodic box -> two planar interfaces."""
    x = np.arange(nx) + 0.5
    c = nx / 2.0
    prof = rho_v + 0.5 * (rho_l - rho_v) * (
        np.tanh((x - (c - half)) / w) - np.tanh((x - (c + half)) / w)
    )
    rho = np.repeat(prof[None, :], ny, axis=0)
    return np.clip(rho, 1e-3, None)


def flat_bands(nx: int, half: float, skip: float = 10.0):
    x = np.arange(nx) + 0.5
    c = nx / 2.0
    liq = np.abs(x - c) <= (half - skip)
    vap = (x < c - half - 3 * skip) | (x > c + half + 3 * skip)
    return liq, vap


def run_flat(solver, nx, ny, rho_l0, rho_v0, w, half, max_steps, sample=500,
             tol=2e-6):
    rho0 = init_flat(nx, ny, rho_l0, rho_v0, w, half)
    f = solver.init(rho0)
    liq, vap = flat_bands(nx, half)
    mass0 = float(rho0.sum())
    hist = []
    t0 = time.time()
    prev = None
    for step in range(1, max_steps + 1):
        f = solver.step(f)
        if step % sample == 0 or step == max_steps:
            rho, ux, uy = solver.fields(f)
            r_l = float(rho[:, liq].mean())
            r_v = float(rho[:, vap].mean())
            p = REF.eos_pressure(rho, CFG["G_eff"])
            rec = {
                "step": step,
                "rho_l": r_l,
                "rho_v": r_v,
                "p_l": float(p[:, liq].mean()),
                "p_v": float(p[:, vap].mean()),
                "u_max": float(np.sqrt(ux**2 + uy**2).max()),
                "mass": float(rho.sum()),
                "mass_drift_rel": abs(float(rho.sum()) - mass0) / mass0,
            }
            hist.append(rec)
            if prev is not None:
                if (abs(r_l - prev["rho_l"]) < tol and abs(r_v - prev["rho_v"]) < tol
                        and abs(rec["p_l"] - prev["p_l"]) < tol * 1e-2):
                    break
            prev = rec
    last = hist[-1]
    last["wall_s"] = round(time.time() - t0, 2)
    last["converged"] = step < max_steps
    last["p_l_minus_p_v"] = last["p_l"] - last["p_v"]
    last["p_l_minus_p_v_pct"] = 100.0 * last["p_l_minus_p_v"] / last["p_v"]
    last["history_tail"] = hist[-3:]
    return last


# ---------------------------------------------------------------------------
# V2 - droplet, three forcing schemes
# ---------------------------------------------------------------------------
def run_droplet(forcing, R, L, max_steps):
    solver = XV.TorchSolver(forcing=forcing, threads=XV_THREADS)
    rho0, rr = XV.init_droplet(L, R)
    f = solver.init(rho0)
    mass0 = float(rho0.sum())
    prev = None
    for step in range(1, max_steps + 1):
        f = solver.step(f)
        if step % 500 == 0:
            fld = solver.fields(f)
            m = XV.measure(fld, rr, R)
            if prev is not None and abs(m["dp"] - prev["dp"]) < 2e-5 and \
                    abs(m["R_eq"] - prev["R_eq"]) < 2e-3 and step >= 4000:
                break
            prev = m
    fld = solver.fields(f)
    m = XV.measure(fld, rr, R)
    m["steps"] = step
    m["mass_drift_rel"] = abs(m["mass"] - mass0) / mass0
    m["forcing"] = forcing
    m["R_init"] = R
    m["L"] = L
    m["sigma_from_dp"] = m["dp"] * m["R_eq"]
    m["field"] = fld
    return m


# ---------------------------------------------------------------------------
# V3 - tau (viscosity) dependence of sigma
# ---------------------------------------------------------------------------
def run_tau(tau, R, L, max_steps):
    import run_xval as _x

    old = CFG["tau"]
    CFG["tau"] = tau
    try:
        m = run_droplet("velocity_shift", R, L, max_steps)
    finally:
        CFG["tau"] = old
    m["tau"] = tau
    return m


XV_THREADS = 4


def _sanitize(obj, _seen=None):
    """Recursively make ``obj`` JSON-safe and break reference cycles."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, dict):
        if id(obj) in _seen:
            return "<cycle>"
        _seen.add(id(obj))
        return {str(k): _sanitize(v, _seen) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        if id(obj) in _seen:
            return "<cycle>"
        _seen.add(id(obj))
        return [_sanitize(v, _seen) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": list(obj.shape)}
    return obj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="3-minute smoke run")
    ap.add_argument("--out", default=str(HERE / "artifacts" / "verify_r1.json"))
    args = ap.parse_args()

    max_flat = 6000 if args.quick else 30000
    max_drop = 4000 if args.quick else 12000
    radii = [16.0] if args.quick else [16.0, 20.0]
    taus = [1.0] if args.quick else [0.7, 1.0, 1.5]

    res: dict = {"anchor": "XVAL-LBM-SHAN-CHEN-LAPLACE-PRESSURE-CALIBRATION",
                 "round": "R1-adversarial", "config": dict(CFG), "legs": {}}

    # ---- analytic reference -----------------------------------------------
    rho_v_mx, rho_l_mx, p_c_mx = REF.maxwell_coexistence(CFG["G_eff"])
    res["maxwell"] = {"rho_v": rho_v_mx, "rho_l": rho_l_mx, "p_c": p_c_mx}

    # ---- V1: flat interface ----------------------------------------------
    print("== V1 flat interface (scheme-specific coexistence) ==")
    nx, ny, half = 256, 8, 40.0
    v1 = {}
    for fs in SCHEMES:
        solver = XV.TorchSolver(forcing=fs, threads=XV_THREADS)
        r = run_flat(solver, nx, ny, CFG["rho_l_init"], CFG["rho_v_init"],
                     CFG["w_int"], half, max_flat)
        r["dev_rho_l_vs_maxwell_pct"] = 100.0 * (r["rho_l"] - rho_l_mx) / rho_l_mx
        r["dev_rho_v_vs_maxwell_pct"] = 100.0 * (r["rho_v"] - rho_v_mx) / rho_v_mx
        r["dev_p_vs_maxwell_pct"] = 100.0 * (r["p_v"] - p_c_mx) / p_c_mx
        v1[fs] = r
        print(f"  {fs:15s} rho_l={r['rho_l']:.5f} rho_v={r['rho_v']:.5f} "
              f"p_l={r['p_l']:.6f} p_v={r['p_v']:.6f} "
              f"p_l-p_v={r['p_l_minus_p_v']:+.2e} "
              f"({r['p_l_minus_p_v_pct']:+.3f}%) steps={r['step']}")
    res["legs"]["V1_flat"] = v1

    # ---- V2: droplet Laplace per forcing ---------------------------------
    print("== V2 droplet Laplace per forcing scheme ==")
    v2 = {}
    for fs in SCHEMES:
        rows = []
        for R in radii:
            L = int(round(4 * R))
            t0 = time.time()
            m = run_droplet(fs, R, L, max_drop)
            rho_flat_l = v1[fs]["rho_l"]
            rho_flat_v = v1[fs]["rho_v"]
            p_flat_l = v1[fs]["p_l"]
            p_flat_v = v1[fs]["p_v"]
            m["dp_vs_own_flat_vapour"] = m["p_in"] - p_flat_v
            m["sigma_own_flat_vapour"] = m["dp_vs_own_flat_vapour"] * m["R_eq"]
            m["rho_in_minus_flat_l_pct"] = 100.0 * (m["rho_in"] - rho_flat_l) / rho_flat_l
            m["rho_out_minus_flat_v_pct"] = 100.0 * (m["rho_out"] - rho_flat_v) / rho_flat_v
            m["p_out_minus_flat_v_pct"] = 100.0 * (
                m["p_out"] - p_flat_v) / max(abs(p_flat_v), 1e-12)
            m["p_in_minus_flat_l_pct"] = 100.0 * (
                m["p_in"] - p_flat_l) / max(abs(p_flat_l), 1e-12)
            m["wall_s"] = round(time.time() - t0, 1)
            m.pop("field", None)
            rows.append(m)
            print(f"  {fs:15s} R={R:4.0f} R_eq={m['R_eq']:.3f} dp={m['dp']:.6f} "
                  f"sigma={m['sigma_from_dp']:.5f} "
                  f"rho_in={m['rho_in']:.4f} rho_out={m['rho_out']:.4f} "
                  f"steps={m['steps']}")
        v2[fs] = rows
    res["legs"]["V2_droplet"] = v2

    # ---- V3: tau dependence ---------------------------------------------
    print("== V3 tau dependence (velocity_shift, R=20) ==")
    v3 = []
    for tau in taus:
        m = run_tau(tau, 20.0, 80, max_drop)
        m.pop("field", None)
        v3.append(m)
        print(f"  tau={tau:.2f} R_eq={m['R_eq']:.3f} dp={m['dp']:.6f} "
              f"sigma={m['sigma_from_dp']:.5f} u_max={m['u_max']:.3f}")
    res["legs"]["V3_tau"] = v3

    # ---- adjudication ----------------------------------------------------
    adj = {}
    # (a) does each scheme's droplet carry its own flat coexistence state?
    for fs in SCHEMES:
        rs = v2[fs]
        adj[fs] = {
            "flat_rho_v": v1[fs]["rho_v"],
            "droplet_rho_out": [r["rho_out"] for r in rs],
            "droplet_rho_out_vs_flat_pct": [r["rho_out_minus_flat_v_pct"] for r in rs],
            "flat_p_l_minus_p_v_pct": v1[fs]["p_l_minus_p_v_pct"],
        }
    sig = {fs: [r["sigma_from_dp"] for r in v2[fs]] for fs in SCHEMES}
    means = {fs: float(np.mean(sig[fs])) for fs in SCHEMES}
    adj["sigma_mean_by_scheme"] = means
    adj["sigma_spread_across_schemes_pct"] = 100.0 * (
        max(means.values()) - min(means.values())) / min(means.values())
    if v3:
        s3 = [m["sigma_from_dp"] for m in v3]
        adj["sigma_tau_spread_pct"] = 100.0 * (max(s3) - min(s3)) / min(s3)
        adj["sigma_vs_locked_doc_pct"] = [
            100.0 * (s - CFG["sigma_locked_doc"]) / CFG["sigma_locked_doc"] for s in s3]
    res["adjudication"] = adj

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_sanitize(res), indent=2, default=float))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())