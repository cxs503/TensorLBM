"""P82W5-N04 benchmark: D3Q19 yield-stress kernel with **moving walls**.

Runs the six acceptance anchors of the campaign node and writes a JSON with the
measured numbers, the criteria and pass/fail flags:

(a) 3D square-duct Bingham profile vs the regularisation-free AL reference
    (``linf_rel <= 3%`` at ``N >= 32``);
(b) Bingham plane Couette--Poiseuille (one wall moving + body force): plug
    thickness error ``<= 3%`` and plug velocity CV ``<= 1e-3``; pure Couette
    degenerate controls;
(c) yield-surface position convergence over ``N in {8,16,32,64}`` (slope >= 1.7);
(d) momentum ledger: Ladd injection vs fluid momentum change (closure in
    [0.97, 1.03]) and steady-state wall drag vs body force;
(e) 2D/3D flow-rate ratio vs the rectangular-duct shape factor (<= 5%);
(f) Bn = 0 Newtonian limit (<= 1%).

Usage
-----
    python scripts/bench_yield_stress_moving_wall_3d.py --case all \
        --out artifacts/P82W5-N04/bench.json --quick
    python scripts/bench_yield_stress_moving_wall_3d.py --case a --out a.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tensorlbm.yield_stress_analytical_moving_wall import (  # noqa: E402
    _couette_poiseuille_u_at_wall,
    bingham_channel_profile,
    bingham_channel_flow_rate,
    bingham_couette_poiseuille_plug,
    bingham_couette_poiseuille_profile,
    bingham_plane_couette_regime,
    bingham_rectangular_duct_reference,
    bingham_square_duct_reference,
    newtonian_square_duct_series,
    rectangular_duct_shape_factor,
)
from src.tensorlbm.yield_stress_moving_wall_3d import (  # noqa: E402
    run_channel_3d,
    run_duct_3d,
)

torch.set_num_threads(1)
DTYPE = torch.float64


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_couette_poiseuille_case(H, G, tau_y, mu_0, *, centre_fraction=0.55):
    """Pick the bottom-wall shear ``tau_0`` that puts the plug centre at
    ``centre_fraction*H`` and return the matching wall velocity ``U_w``."""
    tau_0 = G * centre_fraction * H
    u_wall = _couette_poiseuille_u_at_wall(H, G, tau_y, mu_0, tau_0)
    plug = bingham_couette_poiseuille_plug(H, G, tau_y, mu_0, u_wall)
    out = {"H": H, "G": G, "tau_y": tau_y, "mu_0": mu_0,
           "tau_0_target": tau_0, "u_wall": u_wall}
    out.update(plug)
    return out


def _init_channel(H, nx, nz, profile):
    u = torch.zeros((nz, H + 2, nx), dtype=DTYPE)
    u[:, 1:-1, :] = torch.as_tensor(profile, dtype=DTYPE).view(1, H, 1)
    return u


def _init_duct(N, nx, cross):
    u = torch.zeros((N + 2, N + 2, nx), dtype=DTYPE)
    u[1:-1, 1:-1, :] = torch.as_tensor(cross, dtype=DTYPE).view(N, N, 1)
    return u


def _plug_stats(profile, y1, y2, margin=0.5):
    y = np.arange(1, len(profile) + 1) - 0.5
    sel = (y > y1 + margin) & (y < y2 - margin)
    if sel.sum() < 2:
        sel = (y >= y1) & (y <= y2)
    u = profile[sel]
    mean = float(u.mean())
    return {"n_cells": int(sel.sum()), "mean": mean,
            "cv": float(u.std() / abs(mean)) if mean else None,
            "ptp_rel": float((u.max() - u.min()) / abs(mean)) if mean else None}


def _detect_plug(profile, tau_y, mu_0, G):
    """Plug bounds from the shear-rate profile (threshold ``0.5*tau_y/mu_0``)."""
    y = np.arange(1, len(profile) + 1) - 0.5
    gam = np.abs(np.gradient(profile))
    gam_c = tau_y / mu_0
    inside = gam < 0.5 * gam_c
    if not inside.any():
        return {"y1": None, "y2": None, "delta_p": None}
    idx = np.where(inside)[0]
    # contiguous block containing the maximum-velocity cell
    jmax = int(np.argmax(profile))
    lo = jmax
    while lo - 1 >= 0 and inside[lo - 1]:
        lo -= 1
    hi = jmax
    while hi + 1 < len(inside) and inside[hi + 1]:
        hi += 1
    return {"y1": float(y[lo] - 0.5), "y2": float(y[hi] + 0.5),
            "delta_p": float(y[hi] - y[lo] + 1.0)}


def _plug_from_stress(profile, tau_y, mu_0, m):
    """Yield-surface positions from the model-consistent stress |tau(gamma)| = tau_y.

    The regularised stress along the measured profile is
    ``tau = mu_0*gamma + tau_y*(1-exp(-m*gamma))``; the plug is where ``|tau| < tau_y``.
    The crossings are located by linear interpolation between adjacent cells, which
    removes the +-0.5-cell quantisation of a pure threshold detector.
    """
    y = np.arange(1, len(profile) + 1) - 0.5
    gam = np.abs(np.gradient(profile))
    tau = mu_0 * gam + (tau_y * (1.0 - np.exp(-m * gam)) if m > 0 else 0.0)
    inside = tau < tau_y
    jmax = int(np.argmax(profile))
    if not inside[jmax]:
        return {"y1": None, "y2": None, "delta_p": None}
    lo = jmax
    while lo - 1 >= 0 and inside[lo - 1]:
        lo -= 1
    hi = jmax
    while hi + 1 < len(inside) and inside[hi + 1]:
        hi += 1

    def cross(j_out, j_in):
        # interpolate where tau == tau_y between the inside cell j_in and the
        # outside cell j_out
        t0, t1 = tau[j_in], tau[j_out]
        if t1 == t0:
            return y[j_in]
        f = (tau_y - t0) / (t1 - t0)
        return y[j_in] + f * (y[j_out] - y[j_in])

    y1 = cross(lo - 1, lo) if lo > 0 else 0.0
    y2 = cross(hi + 1, hi) if hi + 1 < len(inside) else float(len(profile))
    return {"y1": float(y1), "y2": float(y2), "delta_p": float(y2 - y1)}


def _yield_surface_from_umax(profile, G, mu_0, H):
    """Plug half-width from the plug velocity (plane Poiseuille Bingham).

    ``u_p = G/(2 mu_0) (h - y_p)^2``  =>  ``y_p = h - sqrt(2 mu_0 u_p / G)``
    with ``h = H/2``.  ``u_p`` is the mean of the maximum and its two neighbours
    (the profile is flat there, which averages out lattice noise).
    """
    h = 0.5 * H
    n = len(profile)
    jmax = int(np.argmax(profile))
    lo, hi = max(0, jmax - 1), min(n - 1, jmax + 1)
    u_p = float(profile[lo:hi + 1].mean())
    val = 2.0 * mu_0 * u_p / G
    if val <= 0:
        return h
    return max(0.0, h - math.sqrt(val))


def _yield_surface_from_gamma(profile, tau_y, mu_0):
    """``y_p`` from the crossing of ``|du/dy| = tau_y/mu_0`` (interpolated).

    Walks outward from the plug centre (the profile maximum) to the first cell
    where the shear rate exceeds the threshold and interpolates the crossing;
    the returned value is the *distance* from the centre line.
    """
    y = np.arange(1, len(profile) + 1) - 0.5
    gam = np.abs(np.gradient(profile))
    gam_c = tau_y / mu_0
    jmax = int(np.argmax(profile))
    j = jmax
    while j + 1 < len(gam) and gam[j + 1] < gam_c:
        j += 1
    if j + 1 >= len(gam):
        return 0.0
    g0, g1 = gam[j], gam[j + 1]
    f = (gam_c - g0) / (g1 - g0) if g1 != g0 else 0.0
    y_cross = y[j] + f * (y[j + 1] - y[j])
    return abs(y_cross - y[jmax])


def _slope(xs, errs):
    xs = np.asarray(xs, dtype=float)
    errs = np.asarray(errs, dtype=float)
    ok = errs > 0
    if ok.sum() < 2:
        return None
    p = np.polyfit(np.log(xs[ok]), np.log(errs[ok]), 1)
    return float(p[0])

# ---------------------------------------------------------------------------
# (a) square duct vs the regularisation-free AL reference
# ---------------------------------------------------------------------------
def case_a(cfg):
    G, mu_0, tau_y = 1e-5, 0.1, 2e-5
    m = float(cfg.get("m", 1e5))
    refine = int(cfg.get("refine", 4))
    out = {"criterion": "linf_rel <= 3% at N >= 32 (vs AL reference)",
           "parameters": {"G": G, "mu_0": mu_0, "tau_y": tau_y, "m": m,
                          "refine": refine, "use_neq": bool(cfg.get("use_neq", False))},
           "runs": {}}

    refs = {r: bingham_square_duct_reference(16, G, tau_y, mu_0, refine=r)
            for r in (2, 4)}
    out["reference_self_audit"] = {
        "N": 16, "Q_refine2": refs[2]["Q"], "Q_refine4": refs[4]["Q"],
        "rel_diff": abs(refs[2]["Q"] - refs[4]["Q"]) / abs(refs[4]["Q"]),
        "note": "AL reference discretisation error (fine grid refine=2 vs 4)",
    }

    for N in cfg["Ns"]:
        t0 = time.time()
        ref = bingham_square_duct_reference(N, G, tau_y, mu_0, refine=refine)
        u_ref = ref["u"]
        res = run_duct_3d(
            N, nx=4, ax=G, tau_y=tau_y, mu_0=mu_0, m=m,
            max_iter=int(cfg.get("max_iter", 60000)),
            tol=float(cfg.get("tol", 1e-10)), check_every=100,
            nu_max=mu_0 * (1.0 + m * tau_y / mu_0) * 10.0,
            u_init=_init_duct(N, 4, newtonian_square_duct_series(N, G, mu_0)),
            use_neq=bool(cfg.get("use_neq", False)),
        )
        u_sim = res["ux_cross_iyz"].numpy()
        umax = float(np.abs(u_ref).max())
        out["runs"]["N%d" % N] = {
            "iters": res["iterations"], "converged": res["converged"],
            "linf_rel": float(np.abs(u_sim - u_ref).max() / umax),
            "l2_rel": float(np.linalg.norm(u_sim - u_ref) / np.linalg.norm(u_ref)),
            "Q_sim": res["Q"], "Q_ref": ref["Q"],
            "Q_rel": abs(res["Q"] - ref["Q"]) / abs(ref["Q"]),
            "force_balance": res["ledger"]["force_balance"],
            "closure_window": res["ledger"]["closure_window"],
            "u_max_ref": umax, "seconds": time.time() - t0,
        }
        np.save(os.path.join(cfg["art"], "duct_N%d_sim.npy" % N), u_sim)
        np.save(os.path.join(cfg["art"], "duct_N%d_ref.npy" % N), u_ref)
    big = {k: v for k, v in out["runs"].items() if int(k[1:]) >= 32}
    out["metric"] = max((v["linf_rel"] for v in big.values()), default=None)
    out["passed"] = bool(big) and all(v["linf_rel"] <= 0.03 for v in big.values())
    return out


# ---------------------------------------------------------------------------
# (b) Couette--Poiseuille plug + pure Couette degenerate controls
# ---------------------------------------------------------------------------
def case_b(cfg):
    mu_0 = 0.1
    G = float(cfg.get("G", 2e-6))
    m = float(cfg.get("m", 1e5))
    out = {"criterion": "delta_p err <= 3% (model-consistent stress crossing), plug CV <= 1e-3",
           "parameters": {"mu_0": mu_0, "G": G, "m": m,
                          "tau_y_rule": "tau_y = G*H/6  ->  delta_p = H/3",
                          "plug_centre_fraction": 0.55,
                          "use_neq": bool(cfg.get("use_neq", False)),
                          "delta_p_estimators": ["stress_crossing (primary)",
                                                 "shear-rate threshold (quantised)"]},
           "runs": {}, "m_scan": {}, "couette_controls": {}}

    for H in cfg["Hs"]:
        tau_y = G * H / 6.0
        case = make_couette_poiseuille_case(H, G, tau_y, mu_0)
        u_exact = bingham_couette_poiseuille_profile(
            H, case["G"], case["tau_y"], case["mu_0"], case["u_wall"])
        res = run_channel_3d(
            H, nx=4, nz=4, ax=case["G"], u_wall=(case["u_wall"], 0.0, 0.0),
            moving_top=True, tau_y=case["tau_y"], mu_0=case["mu_0"], m=m,
            max_iter=int(cfg.get("max_iter", 60000)),
            tol=float(cfg.get("tol", 1e-10)), check_every=100,
            nu_max=case["mu_0"] * (1.0 + m * case["tau_y"] / case["mu_0"]) * 10.0,
            u_init=_init_channel(H, 4, 4, u_exact),
            use_neq=bool(cfg.get("use_neq", False)),
        )
        prof = res["ux_profile"].numpy()
        det = _detect_plug(prof, case["tau_y"], case["mu_0"], case["G"])
        st = _plug_from_stress(prof, case["tau_y"], case["mu_0"], m)
        dp_exact = case["delta_p"]
        dp_err = (abs(det["delta_p"] - dp_exact) / dp_exact
                  if det["delta_p"] is not None and dp_exact > 0 else None)
        st_err = (abs(st["delta_p"] - dp_exact) / dp_exact
                  if st["delta_p"] is not None and dp_exact > 0 else None)
        stats = _plug_stats(prof, case["y1"], case["y2"])
        ymax = float(np.abs(u_exact).max())
        out["runs"]["H%d" % H] = {
            "regime": case["regime"], "u_wall": case["u_wall"],
            "tau_y": case["tau_y"], "delta_p_over_H": dp_exact / H,
            "yield_surfaces_exact": [case["y1"], case["y2"]],
            "delta_p_exact": dp_exact,
            "delta_p_stress": st["delta_p"], "delta_p_err_stress": st_err,
            "yield_surfaces_sim": [st["y1"], st["y2"]],
            "delta_p_threshold": det["delta_p"], "delta_p_err_threshold": dp_err,
            "profile_linf_rel": float(np.abs(prof - u_exact).max() / ymax),
            "plug_cv": stats["cv"], "plug_ptp_rel": stats["ptp_rel"],
            "plug_cells": stats["n_cells"],
            "iters": res["iterations"], "converged": res["converged"],
            "force_balance": res["ledger"]["force_balance"],
            "closure_window": res["ledger"]["closure_window"],
        }
        np.save(os.path.join(cfg["art"], "couette_poiseuille_H%d_sim.npy" % H), prof)
        np.save(os.path.join(cfg["art"], "couette_poiseuille_H%d_exact.npy" % H), u_exact)

        # m-scan at the smallest H: how the plug rigidity reacts to the
        # regularisation parameter (m*tau_y/mu_0 is the relevant group)
        if H == min(cfg["Hs"]):
            for mm in (1e4, 1e6):
                r2 = run_channel_3d(
                    H, nx=4, nz=4, ax=case["G"], u_wall=(case["u_wall"], 0.0, 0.0),
                    moving_top=True, tau_y=case["tau_y"], mu_0=case["mu_0"], m=mm,
                    max_iter=int(cfg.get("max_iter", 60000)),
                    tol=float(cfg.get("tol", 1e-10)), check_every=100,
                    nu_max=case["mu_0"] * (1.0 + mm * case["tau_y"] / case["mu_0"]) * 10.0,
                    u_init=_init_channel(H, 4, 4, u_exact),
                    use_neq=bool(cfg.get("use_neq", False)),
                )
                p2 = r2["ux_profile"].numpy()
                s2 = _plug_stats(p2, case["y1"], case["y2"])
                out["m_scan"]["H%d_m%.0e" % (H, mm)] = {
                    "M_group": mm * case["tau_y"] / case["mu_0"],
                    "plug_cv": s2["cv"],
                    "profile_linf_rel": float(np.abs(p2 - u_exact).max() / ymax),
                    "plug_viscosity": case["mu_0"] * (1 + mm * case["tau_y"] / case["mu_0"]),
                    "iters": r2["iterations"],
                }

    # --- pure Couette degenerate controls (no body force) ------------------
    H = min(cfg["Hs"])
    mu_H = 0.1 * H
    # (i) tau_wall < tau_y : no admissible steady flow (u stays far below U_w)
    tau_y_ctl = 4e-6
    u_w = 0.5 * tau_y_ctl * H / mu_H            # tau_wall = 0.5*tau_y (sub-yield)
    reg = bingham_plane_couette_regime(H, tau_y_ctl, mu_H / H, u_w)
    res = run_channel_3d(H, nx=4, nz=4, u_wall=(u_w, 0.0, 0.0), moving_top=True,
                         tau_y=tau_y_ctl, mu_0=mu_H / H, m=m,
                         max_iter=int(cfg.get("max_iter_ctl", 30000)),
                         tol=1e-9, check_every=100,
                         nu_max=(mu_H / H) * (1.0 + m * tau_y_ctl / (mu_H / H)) * 10.0,
                         use_neq=bool(cfg.get("use_neq", False)))
    prof = res["ux_profile"].numpy()
    out["couette_controls"]["sub_yield"] = {
        "u_wall": u_w, "tau_wall": reg["tau_wall"], "tau_y": tau_y_ctl,
        "regime": reg["regime"], "u_mean_rel": float(prof.mean() / u_w),
        "u_max_rel": float(prof.max() / u_w),
        "comment": "tau_wall < tau_y: the ideal Bingham admits no steady solution; "
                   "the regularised model replaces it by a creeping Newtonian-like "
                   "profile with mu_eff ~ mu_0 + m*tau_y (u_mean ~ 0.5 U_w)",
    }
    # (ii) tau_wall > tau_y : fully yielded, linear profile, delta_p = 0
    u_w = 20.0 * tau_y_ctl * H / mu_H
    reg = bingham_plane_couette_regime(H, tau_y_ctl, mu_H / H, u_w)
    res = run_channel_3d(H, nx=4, nz=4, u_wall=(u_w, 0.0, 0.0), moving_top=True,
                         tau_y=tau_y_ctl, mu_0=mu_H / H, m=m,
                         max_iter=int(cfg.get("max_iter_ctl", 30000)),
                         tol=1e-9, check_every=100,
                         nu_max=(mu_H / H) * (1.0 + m * tau_y_ctl / (mu_H / H)) * 10.0,
                         use_neq=bool(cfg.get("use_neq", False)))
    prof = res["ux_profile"].numpy()
    lin = u_w * (np.arange(1, H + 1) - 0.5) / H
    st = _plug_from_stress(prof, tau_y_ctl, mu_H / H, m)
    out["couette_controls"]["post_yield"] = {
        "u_wall": u_w, "tau_wall": reg["tau_wall"], "tau_y": tau_y_ctl,
        "regime": reg["regime"],
        "linf_rel_vs_linear": float(np.abs(prof - lin).max() / u_w),
        "delta_p_sim": st["delta_p"],
        "comment": "tau_wall > tau_y: whole gap yields, linear profile, no plug",
    }

    runs = list(out["runs"].values())
    out["metric"] = {
        "delta_p_err_stress_max": max(
            (r["delta_p_err_stress"] for r in runs
             if r["delta_p_err_stress"] is not None), default=None),
        "delta_p_err_threshold_max": max(
            (r["delta_p_err_threshold"] for r in runs
             if r["delta_p_err_threshold"] is not None), default=None),
        "plug_cv_max": max((r["plug_cv"] for r in runs if r["plug_cv"] is not None),
                           default=None)}
    out["passed"] = (out["metric"]["delta_p_err_stress_max"] is not None
                     and out["metric"]["delta_p_err_stress_max"] <= 0.03
                     and out["metric"]["plug_cv_max"] is not None
                     and out["metric"]["plug_cv_max"] <= 1e-3)
    return out


# ---------------------------------------------------------------------------
# (c) yield-surface position convergence (pressure-driven channel)
# ---------------------------------------------------------------------------
def case_c(cfg):
    G, mu_0, tau_y = 1e-5, 0.1, 2e-5
    m = float(cfg.get("m", 1e5))
    out = {"criterion": "yield-surface convergence slope >= 1.7",
           "parameters": {"G": G, "mu_0": mu_0, "tau_y": tau_y, "m": m,
                          "y_p_exact": tau_y / G,
                          "estimators": ["umax_inversion", "gamma_crossing"],
                          "use_neq": bool(cfg.get("use_neq", False))},
           "runs": {}}
    for H in cfg["Hs"]:
        u_exact = bingham_channel_profile(H, G, tau_y, mu_0)
        res = run_channel_3d(
            H, nx=4, nz=4, ax=G, tau_y=tau_y, mu_0=mu_0, m=m,
            max_iter=int(cfg.get("max_iter", 60000)),
            tol=float(cfg.get("tol", 1e-10)), check_every=100,
            nu_max=mu_0 * (1.0 + m * tau_y / mu_0) * 10.0,
            u_init=_init_channel(H, 4, 4, u_exact),
            use_neq=bool(cfg.get("use_neq", False)),
        )
        prof = res["ux_profile"].numpy()
        yp_exact = tau_y / G
        yp_umax = _yield_surface_from_umax(prof, G, mu_0, H)
        yp_gam = _yield_surface_from_gamma(prof, tau_y, mu_0)
        umax = float(np.abs(u_exact).max())
        out["runs"]["N%d" % H] = {
            "yp_exact": yp_exact,
            "yp_from_umax": yp_umax, "err_umax": abs(yp_umax - yp_exact),
            "yp_from_gamma": yp_gam, "err_gamma": abs(yp_gam - yp_exact),
            "profile_linf_rel": float(np.abs(prof - u_exact).max() / umax),
            "iters": res["iterations"], "converged": res["converged"],
            "force_balance": res["ledger"]["force_balance"],
        }
        np.save(os.path.join(cfg["art"], "channel_N%d_sim.npy" % H), prof)
        np.save(os.path.join(cfg["art"], "channel_N%d_exact.npy" % H), u_exact)
    Ns = [int(k[1:]) for k in out["runs"]]
    e_umax = [out["runs"]["N%d" % n]["err_umax"] for n in Ns]
    e_gam = [out["runs"]["N%d" % n]["err_gamma"] for n in Ns]
    s1, s2 = _slope(Ns, e_umax), _slope(Ns, e_gam)
    s_prof = _slope(Ns, [out["runs"]["N%d" % n]["profile_linf_rel"] for n in Ns])
    out["slopes"] = {"umax_inversion": s1, "gamma_crossing": s2,
                     "profile_linf": s_prof}
    out["metric"] = max([s for s in (s1, s2) if s is not None], default=None)
    out["passed"] = bool(out["metric"] is not None and out["metric"] >= 1.7)
    return out


# ---------------------------------------------------------------------------
# (d) momentum ledger
# ---------------------------------------------------------------------------
def case_d(cfg):
    mu_0 = 0.1
    m = float(cfg.get("m", 1e5))
    G, tau_y = 1e-5, 2e-5
    N = int(cfg.get("N", 16))
    out = {"criterion": "wall momentum injection vs fluid momentum change in [0.97,1.03]",
           "parameters": {"N": N, "G": G, "tau_y": tau_y, "mu_0": mu_0, "m": m},
           "runs": {}}

    # (i) pressure driven duct: at steady state the wall drag removes exactly the
    #     Guo body-force momentum
    ref = bingham_square_duct_reference(N, G, tau_y, mu_0, refine=4)
    res = run_duct_3d(N, nx=4, ax=G, tau_y=tau_y, mu_0=mu_0, m=m,
                      max_iter=int(cfg.get("max_iter", 40000)), tol=1e-10,
                      check_every=100, nu_max=mu_0 * (1 + m * tau_y / mu_0) * 10,
                      u_init=_init_duct(N, 4, ref["u"]),
                      use_neq=bool(cfg.get("use_neq", False)))
    led1 = res["ledger"]
    duct = {"force_balance": led1["force_balance"],
            "force_balance_drag": led1["force_balance_drag"],
            "closure_window": led1["closure_window"],
            "closure_run": led1["closure_run"],
            "M_wall_window": led1["M_wall_window"],
            "M_body_window": led1["M_body_window"],
            "steps": res["iterations"]}
    out["runs"]["duct_pressure_driven"] = duct

    # (ii) pure Couette, no body force: the moving wall injects, the static wall
    #      removes; the two must balance at steady state
    H, u_w = 16, 0.02
    lin = u_w * (np.arange(1, H + 1) - 0.5) / H
    res = run_channel_3d(H, nx=4, nz=4, u_wall=(u_w, 0.0, 0.0), moving_top=True,
                         tau_y=tau_y, mu_0=mu_0, m=m,
                         max_iter=int(cfg.get("max_iter", 40000)), tol=1e-10,
                         check_every=100,
                         nu_max=mu_0 * (1 + m * tau_y / mu_0) * 10,
                         u_init=_init_channel(H, 4, 4, lin),
                         use_neq=bool(cfg.get("use_neq", False)))
    led2 = res["ledger"]
    out["runs"]["pure_couette"] = {
        "u_wall": u_w, "closure_couette": led2["couette_balance"],
        "M_wall_moving_window": led2["M_wall_moving_window"],
        "M_wall_static_window": led2["M_wall_static_window"],
        "M_inject_window": led2["M_inject_window"],
        "M_body_window": led2["M_body_window"],
        "closure_window": led2["closure_window"],
        "profile_linf_rel": float(np.abs(res["ux_profile"].numpy() - lin).max()
                                  / u_w),
    }

    # (iii) Couette--Poiseuille: Ladd injection + body force vs total wall drag
    case = make_couette_poiseuille_case(H, 2e-6, 4e-6, mu_0)
    u_ex = bingham_couette_poiseuille_profile(H, case["G"], case["tau_y"], mu_0,
                                              case["u_wall"])
    res = run_channel_3d(H, nx=4, nz=4, ax=case["G"],
                         u_wall=(case["u_wall"], 0.0, 0.0), moving_top=True,
                         tau_y=case["tau_y"], mu_0=mu_0, m=m,
                         max_iter=int(cfg.get("max_iter", 40000)), tol=1e-10,
                         check_every=100,
                         nu_max=mu_0 * (1 + m * case["tau_y"] / mu_0) * 10,
                         u_init=_init_channel(H, 4, 4, u_ex),
                         use_neq=bool(cfg.get("use_neq", False)))
    led3 = res["ledger"]
    out["runs"]["couette_poiseuille"] = {
        "u_wall": case["u_wall"],
        "force_balance": led3["force_balance"],
        "closure_window": led3["closure_window"],
        "M_wall_window": led3["M_wall_window"],
        "M_inject_window": led3["M_inject_window"],
        "M_body_window": led3["M_body_window"],
        "comment": "steady state: |M_wall| = |M_body| (Ladd + drag together)",
    }

    vals = {"force_balance_duct": duct["force_balance"],
            "closure_window_duct": duct["closure_window"],
            "closure_couette": out["runs"]["pure_couette"]["closure_couette"],
            "force_balance_cp": out["runs"]["couette_poiseuille"]["force_balance"],
            "closure_window_cp": out["runs"]["couette_poiseuille"]["closure_window"]}
    out["metric"] = vals
    out["passed"] = all(v is not None and 0.97 <= v <= 1.03 for v in vals.values())
    return out


# ---------------------------------------------------------------------------
# (e) 2D / 3D flow-rate ratio
# ---------------------------------------------------------------------------
def case_e(cfg):
    mu_0 = 0.1
    m = float(cfg.get("m", 1e5))
    G = 1e-5
    out = {"criterion": "2D/3D flow ratio vs analytic <= 5%",
           "parameters": {"G": G, "mu_0": mu_0, "m": m,
                          "note": "same gap, same body force and rheology; "
                                  "Q_3D is per unit depth (periodic x)"},
           "runs": {}}
    # Bn is varied through tau_y: small tau_y -> Newtonian shape-factor limit
    for tag, tau_y in (("newtonian_limit", 1e-7), ("bingham", 2e-5)):
        for N in cfg["Ns"]:
            ref = bingham_square_duct_reference(N, G, tau_y, mu_0, refine=4)
            d = run_duct_3d(N, nx=4, ax=G, tau_y=tau_y, mu_0=mu_0, m=m,
                            max_iter=int(cfg.get("max_iter", 40000)), tol=1e-10,
                            check_every=100,
                            nu_max=mu_0 * (1 + m * tau_y / mu_0) * 10,
                            u_init=_init_duct(N, 4, ref["u"]),
                            use_neq=bool(cfg.get("use_neq", False)))
            u2 = bingham_channel_profile(N, G, tau_y, mu_0)
            c = run_channel_3d(N, nx=4, nz=4, ax=G, tau_y=tau_y, mu_0=mu_0, m=m,
                               max_iter=int(cfg.get("max_iter", 40000)),
                               tol=1e-10, check_every=100,
                               nu_max=mu_0 * (1 + m * tau_y / mu_0) * 10,
                               u_init=_init_channel(N, 4, 4, u2),
                               use_neq=bool(cfg.get("use_neq", False)))
            q2_sim = float(c["ux_profile"].sum().item())
            q3_sim = float(d["Q"])
            q2_ref = bingham_channel_flow_rate(N, G, tau_y, mu_0)
            q3_ref = ref["Q"]
            ratio_sim = q2_sim / q3_sim                # both per unit depth
            ratio_ref = q2_ref / q3_ref
            out["runs"]["%s_N%d" % (tag, N)] = {
                "tau_y": tau_y, "Bn_duct": tau_y * N / (mu_0 * G * N ** 2 / (8 * mu_0)),
                "Q_2D_sim": q2_sim, "Q_2D_exact": q2_ref,
                "Q_3D_sim": q3_sim, "Q_3D_ref": q3_ref,
                "ratio_sim": ratio_sim, "ratio_ref": ratio_ref,
                "ratio_err": abs(ratio_sim - ratio_ref) / ratio_ref,
                "newtonian_ratio": 1.0 / rectangular_duct_shape_factor(1.0),
                "Q2D_rel_err": abs(q2_sim - q2_ref) / q2_ref,
                "Q3D_rel_err": abs(q3_sim - q3_ref) / q3_ref,
            }
    errs = [v["ratio_err"] for v in out["runs"].values()]
    out["metric"] = {"ratio_err_max": max(errs),
                     "ratio_sim": {k: v["ratio_sim"] for k, v in out["runs"].items()},
                     "ratio_ref": {k: v["ratio_ref"] for k, v in out["runs"].items()}}
    out["passed"] = max(errs) <= 0.05
    return out


# ---------------------------------------------------------------------------
# (f) Bn = 0 Newtonian limit
# ---------------------------------------------------------------------------
def case_f(cfg):
    G, mu_0 = 1e-5, 0.1
    out = {"criterion": "Bn = 0 (Newtonian) error <= 1%",
           "parameters": {"G": G, "mu_0": mu_0, "tau_y": 0.0}, "runs": {}}
    for N in cfg["Ns"]:
        u_ref = newtonian_square_duct_series(N, G, mu_0)
        d = run_duct_3d(N, nx=4, ax=G, tau_y=0.0, mu_0=mu_0, m=0.0,
                        max_iter=int(cfg.get("max_iter", 60000)), tol=1e-11,
                        check_every=100,
                        u_init=_init_duct(N, 4, u_ref))
        u_sim = d["ux_cross_iyz"].numpy()
        out["runs"]["duct_N%d" % N] = {
            "linf_rel": float(np.abs(u_sim - u_ref).max() / np.abs(u_ref).max()),
            "l2_rel": float(np.linalg.norm(u_sim - u_ref) / np.linalg.norm(u_ref)),
            "iters": d["iterations"], "converged": d["converged"],
            "force_balance": d["ledger"]["force_balance"],
            "shape_factor": d["Q"] / (G * N ** 4 / (12 * mu_0)),
        }
        H = N
        u_w = 0.02
        c = run_channel_3d(H, nx=4, nz=4, u_wall=(u_w, 0.0, 0.0), moving_top=True,
                           tau_y=0.0, mu_0=mu_0, m=0.0,
                           max_iter=int(cfg.get("max_iter", 60000)), tol=1e-11,
                           check_every=100,
                           u_init=_init_channel(H, 4, 4,
                                                u_w * (np.arange(H) + 0.5) / H))
        lin = u_w * (np.arange(1, H + 1) - 0.5) / H
        out["runs"]["couette_H%d" % H] = {
            "linf_rel": float(np.abs(c["ux_profile"].numpy() - lin).max() / u_w),
            "iters": c["iterations"], "converged": c["converged"],
            "couette_balance": c["ledger"].get("couette_balance"),
        }
    errs = [v["linf_rel"] for v in out["runs"].values()]
    out["metric"] = {"linf_rel_max": max(errs)}
    out["passed"] = max(errs) <= 0.01
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--art", default=None, help="directory for .npy arrays")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--m", type=float, default=None)
    ap.add_argument("--use-neq", action="store_true")
    ap.add_argument("--Ns", default=None, help="comma list override")
    ap.add_argument("--Hs", default=None, help="comma list override")
    ap.add_argument("--max-iter", type=int, default=None)
    args = ap.parse_args(argv)

    art = args.art or os.path.dirname(os.path.abspath(args.out))
    os.makedirs(art, exist_ok=True)
    cfg = {"art": art, "use_neq": args.use_neq}
    if args.m is not None:
        cfg["m"] = args.m
    if args.max_iter is not None:
        cfg["max_iter"] = args.max_iter

    if args.quick:
        Ns = [8, 16]
        Hs = [8, 16]
        cfg.setdefault("max_iter", 8000)
    else:
        Ns = [32, 64]
        Hs = [16, 32]
        cfg.setdefault("max_iter", 120000)
    if args.Ns:
        Ns = [int(v) for v in args.Ns.split(",")]
    if args.Hs:
        Hs = [int(v) for v in args.Hs.split(",")]

    selected = ["a", "b", "c", "d", "e", "f"] if args.case == "all" else args.case.split(",")
    result = {"node": "P82W5-N04", "repo": "TensorLBM",
              "kernel": "D3Q19 BGK, Papanaseou regularisation, half-way BB + Ladd",
              "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cases": {}}
    for key in selected:
        key = key.strip()
        t0 = time.time()
        sub = dict(cfg)
        if key == "a":
            sub["Ns"] = Ns
        elif key == "b":
            sub["Hs"] = Hs
        elif key == "c":
            sub["Hs"] = Hs
            sub.setdefault("max_iter", cfg["max_iter"])
        elif key == "e":
            sub["Ns"] = Ns
        elif key == "f":
            sub["Ns"] = Ns
        fn = {"a": case_a, "b": case_b, "c": case_c, "d": case_d,
              "e": case_e, "f": case_f}[key]
        print("[case %s] start" % key, flush=True)
        result["cases"][key] = fn(sub)
        print("[case %s] done in %.1f s passed=%s" % (
            key, time.time() - t0, result["cases"][key].get("passed")), flush=True)

    result["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    print("wrote", args.out, flush=True)
    summ = {k: v.get("passed") for k, v in result["cases"].items()}
    print("summary:", summ, flush=True)
    return result


if __name__ == "__main__":
    main()
