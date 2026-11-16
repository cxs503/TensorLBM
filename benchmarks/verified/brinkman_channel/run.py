#!/usr/bin/env python3
"""Verified case: Brinkman (Darcy-drag) porous channel vs the exact solution.

Capability under test
---------------------
``tensorlbm.brinkman_channel`` adds the volumetric porous drag closure

    a(y) = g - (nu/K) u(y)

to a D2Q9 BGK channel with Guo forcing, i.e. the Brinkman equation

    nu u'' - (nu/K) u + g = 0 ,  u(+-h) = 0 .

The pre-declared decision (``prereg.json``, written *before* any run):

* the steady lattice solution reproduces the closed-form Brinkman profile
  ``u = (gK/nu)[1 - cosh(y/sqrt K)/cosh(beta)]`` to **< 3 %** (``eps_inf``)
  and **< 1 %** in the median case, for every Darcy number of the sweep;
* the bulk flow rate matches the exact shape function
  ``R(beta) = 1 - tanh(beta)/beta`` to **< 1 %**;
* the error decreases at **>= 1.5 observed order** under a self-similar
  refinement (``Da`` and ``Re`` fixed, ``K ~ H^2``);
* mass is conserved (``<= 1e-9``) and the run reaches a declared steady state
  (``<= 1e-9`` relative change of ``u_bar`` over the sampling chunk);
* negative control: the *truncated* Guo source (missing the zero-sum partner
  ``9 (c_i.u) c_ix``) must show a mass sink -- it is the defect this module
  avoids, and it must be detectable, not invisible.

Usage
-----
    python benchmarks/verified/brinkman_channel/run.py swing   # Da sweep
    python benchmarks/verified/brinkman_channel/run.py ladder  # grid refinement
    python benchmarks/verified/brinkman_channel/run.py control # limits + negative control
    python benchmarks/verified/brinkman_channel/run.py merge   # verdict + figures
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from tensorlbm.brinkman_channel import (  # noqa: E402
    BrinkmanChannelConfig,
    _stream,
    exact_mean_velocity,
    exact_shape_function,
    guo_source_2d,
    macroscopic,
    equilibrium,
    run_brinkman_channel,
    verify_exact_solution,
)

TAU = 0.8
ACCEL = 1.0e-5
REF_HEIGHT = 64.0
DA_SWEEP = (1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1)
LADDER = ((34, ), (66, ), (130, ))
LADDER_DA = 1.0e-2

THRESHOLDS = {
    "eps_inf": 0.03,          # ecosystem acceptance for a verified case
    "eps_inf_target": 0.01,   # this case's target (median case must beat it)
    "shape_rel": 0.01,
    "amp_err": 0.01,
    "mass_drift": 1.0e-9,
    "steady_rel_change": 1.0e-9,
    "min_order": 1.5,
}

PREREG = {
    "case": "brinkman_channel",
    "capability": "volume-averaged Darcy/Brinkman drag closure in a bounded channel",
    "reference": "closed-form Brinkman-Poiseuille profile (Brinkman 1947; Vafai & Kim 1990)",
    "protocol": {
        "solver": "D2Q9 BGK + zero-sum Guo (2002) forcing, half-way bounce-back walls",
        "tape": "steady state gate on u_bar (relative change per sampling chunk)",
        "sweep": {"Da": list(DA_SWEEP), "ny": 66, "g": ACCEL},
        "ladder": {
            "Da": LADDER_DA,
            "ny": [n[0] for n in LADDER],
            "g_scaling": "g ~ (64/H)^2 so that Re and Ma stay fixed (self-similar)",
        },
        "controls": [
            "drag off (K -> inf) must reproduce the Poiseuille parabola",
            "truncated Guo source (negative control) must show a mass sink",
        ],
    },
    "thresholds": THRESHOLDS,
    "declared_before_runs": True,
}


def _cfg(ny: int, permeability: float, accel: float, drag: bool = True,
         n_steps: int = 400000) -> BrinkmanChannelConfig:
    return BrinkmanChannelConfig(
        ny=ny, nx=2, tau=TAU, accel=accel, permeability=permeability,
        n_steps=n_steps, check_every=500, steady_tol=1.0e-11,
    )


def _record(r: dict, wall_s: float) -> dict:
    out = {
        k: r[k]
        for k in (
            "beta", "nu", "drag_rate", "eps_inf", "eps_2", "amp_err",
            "u_bar_num", "u_bar_ex", "u_darcy", "shape_num", "shape_ex",
            "u_centre_num", "u_centre_ex", "mass_drift", "steady_rel_change",
            "steps_run", "reached_steady", "max_mach", "x_nonuniformity",
        )
    }
    out["shape_rel"] = abs(out["shape_num"] / out["shape_ex"] - 1.0)
    out["centre_rel"] = abs(out["u_centre_num"] / out["u_centre_ex"] - 1.0)
    out["ny"] = r["config"].ny
    out["H"] = r["config"].height
    out["accel"] = r["config"].accel
    out["permeability"] = r["config"].permeability
    out["da"] = r["config"].permeability / r["config"].height**2
    out["wall_s"] = round(wall_s, 2)
    out["y"] = r["y"]
    out["u_num"] = r["u_num"]
    out["u_exact"] = r["u_exact_profile"]
    return out


def swing() -> dict:
    rows = {}
    for da in DA_SWEEP:
        H = REF_HEIGHT
        cfg = _cfg(int(H + 2), da * H * H, ACCEL)
        t0 = time.time()
        r = run_brinkman_channel(cfg)
        rows[f"da_{da:.0e}"] = _record(r, time.time() - t0)
        print(f"[swing] Da={da:.0e} beta={rows[f'da_{da:.0e}']['beta']:.2f} "
              f"eps_inf={r['eps_inf']:.3%} shape_rel={rows[f'da_{da:.0e}']['shape_rel']:.3%} "
              f"steps={r['steps_run']}")
    return rows


def ladder() -> dict:
    rows = {}
    prev = None
    for (ny,) in LADDER:
        H = float(ny - 2)
        accel = ACCEL * (REF_HEIGHT / H) ** 2   # Re, Ma held fixed
        cfg = _cfg(ny, LADDER_DA * H * H, accel)
        t0 = time.time()
        r = run_brinkman_channel(cfg)
        rec = _record(r, time.time() - t0)
        if prev is not None:
            rec["order_eps_inf"] = math.log(prev["eps_inf"] / rec["eps_inf"]) / math.log(
                rec["H"] / prev["H"])
            rec["order_amp"] = (
                math.log(abs(prev["amp_err"]) / abs(rec["amp_err"])) / math.log(
                    rec["H"] / prev["H"])
                if rec["amp_err"] != 0.0 and prev["amp_err"] != 0.0 else None
            )
        rows[f"ny_{ny}"] = rec
        prev = rec
        print(f"[ladder] ny={ny} H={H:.0f} eps_inf={r['eps_inf']:.3%} "
              f"amp_err={r['amp_err']:+.2e} order={rec.get('order_eps_inf')}")
    return rows


def _legacy_step(f: np.ndarray, cfg: BrinkmanChannelConfig, lam: float) -> np.ndarray:
    """Negative control: Gupta-style truncated source (mass sink), kept verbatim."""
    omega = 1.0 / cfg.tau
    ny = cfg.ny
    rho, ux, uy = macroscopic(f)
    ax = cfg.accel - lam * ux
    ay = -lam * uy
    ax[0, :] = ax[-1, :] = 0.0
    ay[0, :] = ay[-1, :] = 0.0
    ux_h = ux + 0.5 * ax
    uy_h = uy + 0.5 * ay
    feq = equilibrium(rho, ux_h, uy_h)
    # truncated: 9 (c_i.u) c_i . a is missing -> sum_i S_i = -3 rho a_x != 0
    cu = np.empty((9,) + ux.shape)
    from tensorlbm.brinkman_channel import CX, CY, W

    a_dot_c = CX[:, None, None] * ax[None] + CY[:, None, None] * ay[None]
    u_dot_a = ux_h * ax + uy_h * ay
    factor = (1.0 - 0.5 / cfg.tau) * W[:, None, None] * rho[None]
    # only the 9 (c_i.u)(c_i.a) partner is dropped -> sum_i S_i = -3 rho (u.a)
    S_trunc = factor * 3.0 * (a_dot_c - u_dot_a[None])
    fc = f - omega * (f - feq) + S_trunc
    f = _stream(fc)
    from tensorlbm.brinkman_channel import _BOTTOM_IN, _BOTTOM_OUT, _TOP_IN, _TOP_OUT

    f[_BOTTOM_IN, 1] = fc[_BOTTOM_OUT, 1]
    f[_TOP_IN, ny - 2] = fc[_TOP_OUT, ny - 2]
    return f


def control() -> dict:
    out: dict = {}
    # (a) drag off: K -> infinity must collapse onto the Poiseuille parabola
    cfg = _cfg(66, 1.0e12, ACCEL)
    t0 = time.time()
    r = run_brinkman_channel(cfg, drag=False)
    out["drag_off_poiseuille"] = _record(r, time.time() - t0)
    print(f"[control] drag off eps_inf={r['eps_inf']:.3%} amp={r['amp_err']:+.2e}")

    # (b) negative control: truncated Guo source must leak mass
    cfg2 = _cfg(66, 1.0e-2 * 64.0**2, ACCEL, n_steps=4000)
    ny, nx = cfg2.ny, cfg2.nx
    f = equilibrium(np.ones((ny, nx)), np.zeros((ny, nx)), np.zeros((ny, nx)))
    m0 = float(f.sum())
    lam = cfg2.drag_rate
    for _ in range(4000):
        f = _legacy_step(f, cfg2, lam)
    out["negative_control_truncated_guo"] = {
        "steps": 4000,
        "mass_drift": abs(float(f.sum()) - m0) / m0,
        "detected": bool(abs(float(f.sum()) - m0) / m0 > 1.0e-6),
    }
    print(f"[control] truncated-Guo mass drift="
          f"{out['negative_control_truncated_guo']['mass_drift']:.2e}")

    # (c) the analytic reference is re-derived in-repo, not hand-copied
    out["reference_selfcheck"] = verify_exact_solution(
        accel=ACCEL, nu=cfg.nu, permeability=1.0e-2 * 64.0**2, half_width=32.0)
    print("[control] reference selfcheck", out["reference_selfcheck"])
    return out


def verdict(results: dict) -> dict:
    swing_rows = results.get("swing", {})
    ladder_rows = results.get("ladder", {})
    ctrl = results.get("control", {})
    checks: dict[str, dict] = {}

    eps = {k: v["eps_inf"] for k, v in swing_rows.items()}
    worst = max(eps.values()) if eps else float("inf")
    checks["eps_inf_worst_le_3pct"] = {"value": worst, "threshold": THRESHOLDS["eps_inf"],
                                      "pass": bool(worst <= THRESHOLDS["eps_inf"])}
    med = float(np.median(list(eps.values()))) if eps else float("inf")
    checks["eps_inf_median_le_1pct"] = {"value": med,
                                       "threshold": THRESHOLDS["eps_inf_target"],
                                       "pass": bool(med <= THRESHOLDS["eps_inf_target"])}
    shape_worst = max((v["shape_rel"] for v in swing_rows.values()), default=float("inf"))
    checks["shape_rel_le_1pct"] = {"value": shape_worst,
                                   "threshold": THRESHOLDS["shape_rel"],
                                   "pass": bool(shape_worst <= THRESHOLDS["shape_rel"])}
    amp_worst = max((abs(v["amp_err"]) for v in swing_rows.values()), default=float("inf"))
    checks["amp_err_le_1pct"] = {"value": amp_worst,
                                 "threshold": THRESHOLDS["amp_err"],
                                 "pass": bool(amp_worst <= THRESHOLDS["amp_err"])}
    mass_worst = max([abs(v["mass_drift"]) for v in swing_rows.values()]
                     + [abs(v["mass_drift"]) for v in ladder_rows.values()], default=1.0)
    checks["mass_drift"] = {"value": mass_worst,
                            "threshold": THRESHOLDS["mass_drift"],
                            "pass": bool(mass_worst <= THRESHOLDS["mass_drift"])}
    orders = [v["order_eps_inf"] for v in ladder_rows.values() if v.get("order_eps_inf")]
    chk_order = {"value": orders, "threshold": THRESHOLDS["min_order"]}
    chk_order["pass"] = bool(orders and min(orders) >= THRESHOLDS["min_order"])
    checks["observed_order"] = chk_order
    steady_worst = max((v["steady_rel_change"] for v in swing_rows.values()), default=1.0)
    checks["steady"] = {"value": steady_worst,
                        "threshold": THRESHOLDS["steady_rel_change"],
                        "pass": bool(steady_worst <= THRESHOLDS["steady_rel_change"])}
    poise = ctrl.get("drag_off_poiseuille", {}).get("eps_inf", 1.0)
    checks["drag_off_poiseuille_eps_inf"] = {"value": poise,
                                             "threshold": THRESHOLDS["eps_inf"],
                                             "pass": bool(poise <= THRESHOLDS["eps_inf"])}
    neg = ctrl.get("negative_control_truncated_guo", {})
    checks["negative_control_detected"] = {"value": neg.get("mass_drift"),
                                           "pass": bool(neg.get("detected"))}
    passed = all(c["pass"] for c in checks.values())
    return {"checks": checks, "pass": passed}


def figures(results: dict) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs = HERE / "figs"
    figs.mkdir(exist_ok=True)
    made = []

    rows = results.get("swing", {})
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, row in rows.items():
        u = np.array(row["u_num"]) / row["u_darcy"]
        ue = np.array(row["u_exact"]) / row["u_darcy"]
        ax[0].plot(u, row["y"], "-", lw=1.0, label=f"Da={row['da']:.0e} LBM")
        ax[0].plot(ue, row["y"], "--", lw=0.8)
    ax[0].set_xlabel(r"$u/u_D$"); ax[0].set_ylabel(r"$y-y_c$")
    ax[0].set_title("Brinkman channel: LBM (solid) vs exact (dashed)")
    ax[0].legend(fontsize=6, loc="lower right"); ax[0].grid(alpha=0.3)
    das = [rows[k]["da"] for k in rows]
    ax[1].loglog(das, [rows[k]["eps_inf"] for k in rows], "o-", label=r"$\varepsilon_\infty$")
    ax[1].loglog(das, [rows[k]["shape_rel"] for k in rows], "s-", label="shape fn")
    ax[1].axhline(THRESHOLDS["eps_inf"], color="k", ls=":", label="3% acceptance")
    ax[1].set_xlabel("Da"); ax[1].set_ylabel("relative error")
    ax[1].set_title("error vs Darcy number (H=64)"); ax[1].legend(fontsize=7); ax[1].grid(alpha=0.3)
    fig.tight_layout(); p = figs / "swing.png"; fig.savefig(p, dpi=140); plt.close(fig)
    made.append(str(p.relative_to(HERE)))

    lad = results.get("ladder", {})
    if lad:
        Hs = [lad[k]["H"] for k in lad]
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].loglog(Hs, [lad[k]["eps_inf"] for k in lad], "o-", label=r"$\varepsilon_\infty$")
        ax[0].loglog(Hs, [abs(lad[k]["amp_err"]) + 1e-12 for k in lad], "s-", label="|amp err|")
        ref = [lad[f"ny_{LADDER[0][0]}"]["eps_inf"] * (LADDER[0][0] - 2) ** 2 / (h**2) for h in Hs]
        ax[0].loglog(Hs, ref, "k:", label=r"$\propto H^{-2}$")
        ax[0].set_xlabel("H (channel height in lattice units)"); ax[0].set_ylabel("relative error")
        ax[0].set_title(f"self-similar refinement at Da={LADDER_DA:.0e}")
        ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3, which="both")
        key = list(lad)[0]
        row = lad[key]
        ax[1].plot(np.array(row["u_num"]) / row["u_darcy"], row["y"], "o-", ms=3, label="LBM")
        ax[1].plot(np.array(row["u_exact"]) / row["u_darcy"], row["y"], "--", label="exact")
        ax[1].set_xlabel(r"$u/u_D$"); ax[1].set_ylabel(r"$y-y_c$")
        ax[1].set_title(f"coarse grid profile (ny={row['ny']})"); ax[1].legend(fontsize=7)
        ax[1].grid(alpha=0.3)
        fig.tight_layout(); p = figs / "ladder.png"; fig.savefig(p, dpi=140); plt.close(fig)
        made.append(str(p.relative_to(HERE)))
    return made


def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "merge"
    out = HERE / "result.json"
    results = json.loads(out.read_text()) if out.exists() else {}
    if mode == "prereg":
        (HERE / "prereg.json").write_text(json.dumps(PREREG, indent=2) + "\n")
        print("wrote prereg.json")
        return
    if mode in {"swing", "ladder", "control"}:
        results[mode] = {"swing": swing, "ladder": ladder, "control": control}[mode]()
        out.write_text(json.dumps(results, indent=2) + "\n")
        print(f"updated {out}")
        return
    results["verdict"] = verdict(results)
    results["figures"] = figures(results)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["verdict"], indent=2))


if __name__ == "__main__":
    main()