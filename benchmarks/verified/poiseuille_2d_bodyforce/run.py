#!/usr/bin/env python3
"""B14: 2-D body-force plane Poiseuille flow (D2Q9 BGK) — analytic validation.

Physics
-------
Fully developed laminar flow between parallel plates, driven by a constant body
force per unit mass ``g`` (equivalent to a constant streamwise pressure gradient
``dp/dx = -rho g``).  Streamwise periodic, wall-normal bounded:

    nu u''(y) + g = 0,   u(0) = u(H) = 0
    u(y) = g/(2 nu) y (H - y),      u_max = g H^2 / (8 nu),   u_bar = g H^2/(12 nu)

In lattice units ``nu = (tau - 1/2)/3``, ``c_s^2 = 1/3``,
``Re = u_bar H / nu``, ``Ma = u_max / c_s``.  The no-slip walls are half-way
bounce-back surfaces placed at ``y = 0.5`` and ``y = H + 0.5`` (row centres on
integers, ``H = ny - 4`` fluid rows), which is the wall position used by the
reference parabola (no re-calibration, no extrapolation).

Pre-declared protocol (fixed *before* the runs, see ``result.json: protocol``)
----------------------------------------------------------------------------
Decision under test: can the internal D2Q9 BGK channel solver of TensorLBM
(``tensorlbm.ice_hydro.lbm_channel``) reproduce the analytic Poiseuille profile
to **better than 2 %** at a fixed reference resolution, and does the error
decrease at the expected second order under grid refinement?

* unit of analysis: one steady wall-normal profile per (H, tau, forcing, steps).
* metrics (all relative, no tuning):
  - ``eps_inf``   = max_j |u_num - u_ana| / max_j u_ana          (primary, <= 2 %)
  - ``eps_2``     = ||u_num - u_ana||_2 / ||u_ana||_2            (<= 2 %)
  - ``amp_err``   = u_num,max / u_ana,max - 1                    (<= 2 %)
  - ``curv_res``  = max_bulk |nu u'' + g| / g                    (<= 2 %)
  - ``mass_drift``= (rho_end - rho_0)/rho_0 over the averaging window (<= 1e-6)
  - ``steady_rel_change`` = |u_max(t_end) - u_max(t_end-chunk)| / u_max (<= 1e-6)
* pass rule: every metric within threshold at the reference case H=32, tau=0.8,
  forcing=guo; a run that is not steady is reported FAIL (not averaged away).
* negative control: the pre-fix Guo source term (missing the zero-sum part
  ``9 (c_i.u) c_ix``) is reproduced verbatim as ``legacy_guo``; it must exhibit a
  mass sink and a step-count-dependent amplitude (this is what the fix removes).
* repetitions: the solver is deterministic (no RNG, no threading) so repetition
  is not used for noise; instead each configuration is run to machine-steady
  state and the stationarity itself is a reported metric.  Step counts
  {10k,20k,40k} at fixed H are the reproducibility check for the primary metric.

Usage
-----
    PYTHONPATH=TensorLBM/src python benchmarks/verified/poiseuille_2d_bodyforce/run.py res
    PYTHONPATH=TensorLBM/src python benchmarks/verified/poiseuille_2d_bodyforce/run.py sens
    PYTHONPATH=TensorLBM/src python benchmarks/verified/poiseuille_2d_bodyforce/run.py merge
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
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "TensorLBM" / "src"))

from tensorlbm.ice_hydro.lbm_channel import (  # noqa: E402
    CX,
    CY,
    WT,
    ChannelConfig,
    D2Q9Channel,
)

CS2 = 1.0 / 3.0
THRESHOLDS = {
    "eps_inf": 0.02,
    "eps_2": 0.02,
    "amp_err": 0.02,
    "curv_res": 0.02,
    "mass_drift": 1.0e-6,
    "steady_rel_change": 1.0e-6,
}


class LegacyGuoSinkChannel(D2Q9Channel):
    """Pre-fix solver, kept verbatim as the negative control.

    The Guo source term is truncated to its first part ``3 (c_ix - u_x)``; the
    zero-sum partner ``9 (c_i.u) c_ix`` is missing, so ``sum_i S_i = -3 u_x g``:
    a spurious mass sink of exactly that rate per step.
    """

    def _collide(self, f: np.ndarray) -> np.ndarray:
        rho = f.sum(axis=0)
        ux = (f * CX[:, None, None]).sum(axis=0) / rho
        uy = (f * CY[:, None, None]).sum(axis=0) / rho
        ux_eq = ux + 0.5 * self.force
        usq = 1.5 * (ux_eq * ux_eq + uy * uy)
        omega = 1.0 / self.cfg.tau
        feq = np.empty_like(f)
        for i in range(9):
            cu = 3.0 * (CX[i] * ux_eq + CY[i] * uy)
            feq[i] = WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
        fnew = f - omega * (f - feq)
        cst = (1.0 - 0.5 * omega) * 3.0 * self.force
        for i in range(9):
            fnew[i] += cst * WT[i] * (CX[i] - ux_eq)
        return fnew


def verify_reference(nu: float = 0.1, force: float = 1.0e-5, H: float = 32.0) -> dict:
    """Re-derive / re-check the reference inside the repo (no hand-copied numbers).

    Checks the closed form ``u(y) = g/(2 nu) y (H-y)`` against (i) the ODE
    ``nu u'' + g = 0`` sampled on a fine grid, (ii) both no-slip boundary
    conditions, (iii) the textbook invariants ``u_max = gH^2/(8 nu)`` and
    ``u_bar = gH^2/(12 nu)`` (Hagen 1839 / Poiseuille 1841; standard form).
    """
    y = np.linspace(0.0, H, 4001)
    u = analytic(force, nu, y, H)
    d2u = np.gradient(np.gradient(u, y), y)
    # a parabola has an exact central second difference; only the two one-sided
    # end stencils carry truncation error, so both numbers are reported
    interior = nu * d2u[2:-2] + force
    analytic_d2 = -force / nu                      # u'' of the closed form
    return {
        "ode_residual_interior_max": float(np.max(np.abs(interior))),
        "ode_residual_end_stencil_max": float(np.max(np.abs(nu * d2u + force))),
        "ode_residual_analytic_identity": float(abs(nu * analytic_d2 + force)),
        "bc_bottom": float(analytic(force, nu, np.array([0.0]), H)[0]),
        "bc_top": float(analytic(force, nu, np.array([H]), H)[0]),
        "u_max_closed_form": float(force * H ** 2 / (8.0 * nu)),
        "u_max_profile": float(u.max()),
        "u_bar_closed_form": float(force * H ** 2 / (12.0 * nu)),
        "u_bar_profile": float(np.trapezoid(u, y) / H),
    }


def analytic(force: float, nu: float, y: np.ndarray, H: float) -> np.ndarray:
    return force / (2.0 * nu) * y * (H - y)


def simulate(
    H: int,
    tau: float = 0.8,
    force: float = 1.0e-5,
    forcing: str = "guo",
    max_steps: int = 200_000,
    chunk: int = 1000,
    tol: float = 1.0e-7,
    legacy: bool = False,
    fixed_steps: int | None = None,
) -> dict:
    """Run one channel case to (machine) steady state and score the profile."""
    ny = H + 4
    cfg = ChannelConfig(
        nx=4, ny=ny, tau=tau, force=force, steps=0, skip=1,
        cover_amplitude=0.0, cover_depth=float(ny - 3), bc="no_slip", forcing=forcing,
    )
    cls = LegacyGuoSinkChannel if legacy else D2Q9Channel
    ch = cls(cfg)
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size
    nu = (tau - 0.5) / 3.0
    y = rows - 0.5
    u_ana = analytic(force, nu, y, float(n_fluid))
    # start from the analytic parabola: removes the slow diffusive fill-up, the
    # remaining transient is the (physical) wall-layer adjustment.
    u_init = np.zeros(ny)
    u_init[rows] = u_ana
    ch.init_equilibrium(u_init)

    t0 = time.time()
    rho0 = 1.0
    prof_acc = np.zeros(n_fluid)
    n_acc = 0
    rho_acc = 0.0
    umax_prev = None
    steady = False
    step = 0
    hist: list[dict] = []
    while step < max_steps:
        for _ in range(chunk):
            ch.f = ch._stream_from(ch._collide(ch.f))
        step += chunk
        rho = ch.f.sum(axis=0)
        ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
        prof = ux[0, rows]
        umax = float(prof.max())
        rho_mean = float(rho[ch.fluid].mean())
        rel = 0.0 if umax_prev is None else abs(umax - umax_prev) / umax
        hist.append({"step": step, "umax": umax, "rho_mean": rho_mean,
                     "steady_rel_change": rel})
        umax_prev = umax
        if fixed_steps is not None:
            # accumulate the last quarter of a prescribed-length run
            if step >= fixed_steps - chunk:
                prof_acc += prof
                rho_acc += rho_mean
                n_acc += 1
            if step >= fixed_steps:
                break
        else:
            if rel <= tol:
                steady = True
                prof_acc += prof
                rho_acc += rho_mean
                n_acc += 1
                if n_acc >= 2:      # average over two consecutive steady chunks
                    break
            else:
                prof_acc *= 0.0
                rho_acc = 0.0
                n_acc = 0
    elapsed = time.time() - t0
    prof_mean = prof_acc / max(n_acc, 1)
    rho_mean = rho_acc / max(n_acc, 1)
    umax_ana = float(u_ana.max())
    curv = np.diff(prof_mean, 2)
    curv_res = float(np.max(np.abs(nu * curv + force)) / force)
    out = {
        "H": int(n_fluid),
        "ny": ny,
        "nx": int(cfg.nx),
        "tau": tau,
        "nu": nu,
        "force": force,
        "forcing": forcing,
        "legacy_guo": bool(legacy),
        "steps": step,
        "steady": bool(steady) if fixed_steps is None else None,
        "steady_rel_change": float(hist[-1]["steady_rel_change"]),
        "elapsed_s": round(elapsed, 2),
        "u_mean_ana": float(force / (12.0 * nu) * n_fluid ** 2),
        "u_max_ana": umax_ana,
        "u_max_num": float(prof_mean.max()),
        "amp_err": float(prof_mean.max() / umax_ana - 1.0),
        "eps_inf": float(np.max(np.abs(prof_mean - u_ana)) / umax_ana),
        "eps_2": float(np.linalg.norm(prof_mean - u_ana) / np.linalg.norm(u_ana)),
        "curv_res": curv_res,
        "rho_mean": rho_mean,
        "mass_drift": float((rho_mean - rho0) / rho0),
        "Re": float(force / (12.0 * nu) * n_fluid ** 2 * n_fluid / nu),
        "Ma": float(umax_ana / math.sqrt(CS2)),
        "y": y.tolist(),
        "u_numerical": prof_mean.tolist(),
        "u_analytic": u_ana.tolist(),
        "history": hist,
    }
    out["pass"] = bool(
        out["eps_inf"] <= THRESHOLDS["eps_inf"]
        and out["eps_2"] <= THRESHOLDS["eps_2"]
        and abs(out["amp_err"]) <= THRESHOLDS["amp_err"]
        and out["curv_res"] <= THRESHOLDS["curv_res"]
        and abs(out["mass_drift"]) <= THRESHOLDS["mass_drift"]
    )
    return out


def _dump(obj: dict, name: str) -> None:
    (HERE / name).write_text(json.dumps(obj, indent=1))


def run_res(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--max-steps", type=int, default=200_000)
    a = ap.parse_args(argv)
    cases = []
    for H in a.H:
        r = simulate(H, max_steps=a.max_steps)
        cases.append(r)
        print("H=%3d steps=%6d eps_inf=%.5f eps_2=%.5f amp=%+.5f curv=%.4f "
              "drift=%+.2e pass=%s (%.1fs)"
              % (r["H"], r["steps"], r["eps_inf"], r["eps_2"], r["amp_err"],
                 r["curv_res"], r["mass_drift"], r["pass"], r["elapsed_s"]), flush=True)
    _dump({"cases": cases}, "res_raw.json")


def run_sens(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--only", default="tau,forcing,steps,legacy",
                    help="comma separated subset of tau,forcing,steps,legacy")
    a = ap.parse_args(argv)
    H = a.H
    only = {k.strip() for k in a.only.split(",") if k.strip()}
    out: dict[str, list] = {"tau": [], "forcing": [], "steps": [], "legacy": []}
    for tau in ((0.6, 0.8, 1.0, 1.2) if "tau" in only else ()):
        r = simulate(H, tau=tau)
        out["tau"].append(r)
        print("tau=%.2f eps_inf=%.5f amp=%+.5f drift=%+.2e" % (tau, r["eps_inf"], r["amp_err"], r["mass_drift"]), flush=True)
    for forcing in (("guo", "simple") if "forcing" in only else ()):
        r = simulate(H, forcing=forcing)
        out["forcing"].append(r)
        print("forcing=%-6s eps_inf=%.5f amp=%+.5f drift=%+.2e" % (forcing, r["eps_inf"], r["amp_err"], r["mass_drift"]), flush=True)
    for steps in ((10_000, 20_000, 40_000) if "steps" in only else ()):
        r = simulate(H, fixed_steps=steps)
        out["steps"].append(r)
        print("steps=%6d eps_inf=%.5f amp=%+.5f drift=%+.2e" % (steps, r["eps_inf"], r["amp_err"], r["mass_drift"]), flush=True)
    for steps in ((10_000, 20_000, 40_000, 60_000) if "legacy" in only else ()):
        r = simulate(H, fixed_steps=steps, legacy=True)
        out["legacy"].append(r)
        print("legacy steps=%6d eps_inf=%.5f amp=%+.5f drift=%+.2e" % (steps, r["eps_inf"], r["amp_err"], r["mass_drift"]), flush=True)
    _dump(out, "sens_raw_%s.json" % "_".join(sorted(only)))


def _order(cases: list[dict]) -> dict:
    hs = np.array([c["H"] for c in cases], float)
    e = np.array([c["eps_2"] for c in cases], float)
    slope, intercept = np.polyfit(np.log(hs), np.log(e), 1)
    return {"fitted_order": float(slope), "fit_intercept": float(intercept)}


def merge(argv: list[str]) -> None:
    res = json.loads((HERE / "res_raw.json").read_text())["cases"]
    sens: dict[str, list] = {"tau": [], "forcing": [], "steps": [], "legacy": []}
    for f in sorted(HERE.glob("sens_raw_*.json")):
        part = json.loads(f.read_text())
        for k, v in part.items():
            sens.setdefault(k, []).extend(v)
    ref = next(c for c in res if c["H"] == 32)
    payload = {
        "case": "B14_poiseuille_2d_bodyforce",
        "lattice": "D2Q9",
        "collision": "BGK",
        "forcing": "guo (zero-sum source, Guo 2002)",
        "driving": "constant body force g (per unit mass), streamwise periodic",
        "boundary": "half-way bounce-back, walls at y=0.5 and y=H+0.5",
        "analytic": "u(y)=g/(2 nu) y (H-y), u_max=gH^2/(8 nu), u_bar=gH^2/(12 nu)",
        "reference_resolution": {"H": ref["H"], "ny": ref["ny"], "tau": ref["tau"],
                                 "force": ref["force"], "Re": ref["Re"], "Ma": ref["Ma"]},
        "protocol": {"thresholds": THRESHOLDS,
                     "pass_rule": "all metrics within threshold at H=32,tau=0.8,force=1e-5,guo",
                     "metric_definitions": {
                         "eps_inf": "max_j |u_num-u_ana| / max_j u_ana",
                         "eps_2": "||u_num-u_ana||_2 / ||u_ana||_2",
                         "amp_err": "u_num,max/u_ana,max - 1",
                         "curv_res": "max_bulk |nu u'' + g| / g",
                         "mass_drift": "(rho_end-rho_0)/rho_0",
                         "steady_rel_change": "|u_max(t)-u_max(t-chunk)|/u_max"}},
        "reference_case": ref,
        "reference_check": verify_reference(),
        "resolution_study": res,
        "convergence": _order([c for c in res if c["H"] >= 8]),
        "sensitivity": {k: [{kk: vv for kk, vv in c.items()
                             if kk not in ("y", "u_numerical", "u_analytic", "history")}
                            for c in v] for k, v in sens.items()},
        "legacy_vs_fixed_at_H64_30k_steps": (
            json.loads((HERE / "legacy_h64.json").read_text())
            if (HERE / "legacy_h64.json").exists() else None),
        "verdict": {
            "all_pass": bool(all(c["pass"] for c in res)),
            "eps_inf_ref": ref["eps_inf"],
            "eps_2_ref": ref["eps_2"],
            "criteria_2pct_met": bool(ref["eps_inf"] <= 0.02 and ref["eps_2"] <= 0.02),
        },
    }
    (HERE / "result.json").write_text(json.dumps(payload, indent=1))
    print("wrote", HERE / "result.json")
    print("ref H=%d eps_inf=%.5f eps_2=%.5f amp=%+.5f curv=%.2e drift=%+.1e"
          % (ref["H"], ref["eps_inf"], ref["eps_2"], ref["amp_err"], ref["curv_res"], ref["mass_drift"]))
    print("fitted convergence order (eps_2 vs H):", payload["convergence"]["fitted_order"])
    for c in res:
        print("  H=%3d steps=%6d eps_inf=%.5f eps_2=%.5f pass=%s" % (c["H"], c["steps"], c["eps_inf"], c["eps_2"], c["pass"]))


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "merge"
    {"res": run_res, "sens": run_sens, "merge": merge}[cmd](sys.argv[2:])


if __name__ == "__main__":
    main()