#!/usr/bin/env python3
"""CASE-LBM-RB-ONSET-RA1708 — independent re-derivation of the control-arm reference.

Why this file exists
--------------------
``run_case.py`` grades the lattice growth-rate measurement (Route A) *relative to*
a continuum control arm (Route C = ``tensorlbm.rb_continuum_stability``).  If the
control arm were itself wrong, "control passed" would validate nothing: Route C
solves the same two-field streamfunction/temperature formulation the lattice arm
approximates, so a shared modelling mistake (sign of the buoyancy term, wrong
marginal operator, wrong wall conditions) would stay invisible.

This script re-derives the marginal Rayleigh number with an **independent
solution strategy**: the linearised Navier-Stokes-Boussinesq equations for one
horizontal Fourier mode are integrated **in time** (RK4) with second-order finite
differences in the wall-normal direction, and the sign of the fitted growth rate
``sigma(Ra)`` is bisected.  No eigenvalue solver, no ARPACK, no code shared with
Route C (which assembles a generalised eigenproblem).

Equations (same non-dimensionalisation as the case — lengths by the gap ``H``,
time by ``H^2/nu``, temperature by ``dT`` — perturbations of the conductive state)::

    d/dt [(D^2 - k^2) psi] = (D^2 - k^2)^2 psi - k (Ra/Pr) theta
    d/dt theta             = (1/Pr) (D^2 - k^2) theta - k psi

Wall models (perturbation of the quiescent state):

* **free-slip** ``psi = D^2 psi = 0``, ``theta = 0`` -> analytic ``Ra_c = 27 pi^4/4``
* **rigid**     ``psi = D psi = 0``,   ``theta = 0`` -> ``Ra_c = 1707.7597`` (Chandrasekhar 1961)

The free-slip arm is the **calibration**: it has a closed-form answer, so
reproducing ``657.511`` at ``a_c = pi/sqrt(2)`` verifies the implementation (sign
conventions, ``Pr`` handling, wall treatment, growth-rate fit) against theory, not
against the repository's own code.  Only then is the rigid arm's number
admissible as an independent control check.

Independence limitations are listed in the JSON (``limitations``) and repeated in
the verification note: this arm uses the same variable choice
``omega = (D^2-k^2) psi`` as Route C, so a modelling error *in the equations*
would survive; what it rules out is an error in Route C's *discretisation,
eigen-solve or marginal criterion*.

Every number is written to JSON; the report injects it programmatically.

Usage
-----
    PYTHONPATH=src python3 benchmarks/case_pool/rayleigh_benard_onset/independent_check_control.py \
        --out runs/case_rayleigh_benard_onset/independent_control.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as sla

RA_FREEFREE = 27.0 * np.pi ** 4 / 4.0      # 657.51124...
A_FREEFREE = np.pi / np.sqrt(2.0)          # 2.22144...
RA_RIGID = 1707.7597                        # Chandrasekhar 1961, rigid-rigid isothermal
A_RIGID = 3.11632


class ModeOperator:
    """One-mode linear RB operator on interior nodes (walls at y=0,1)."""

    def __init__(self, ny: int, k: float, ra: float, pr: float, walls: str):
        self.ny, self.k, self.ra, self.pr, self.walls = ny, k, ra, pr, walls
        h = 1.0 / (ny + 1)
        self.h = h
        main = -2.0 / h ** 2 * np.ones(ny)
        off = 1.0 / h ** 2 * np.ones(ny - 1)
        A = np.diag(main) + np.diag(off, 1) + np.diag(off, -1)   # D^2, psi_wall = 0
        self.A = A
        self.L = A - k * k * np.eye(ny)                          # D^2 - k^2
        self.lu = sla.lu_factor(self.L)

    def rhs(self, psi: np.ndarray, theta: np.ndarray):
        """RHS of the two-field system, with the mass matrix inverted."""
        omega = self.L @ psi                       # = (D^2 - k^2) psi on interior
        # (D^2 - k^2) omega on the interior, walls carrying the ghost values
        d2_omega = self.A @ omega
        if self.walls == "rigid":
            d2_omega[0] += (2.0 / self.h ** 4) * psi[0]
            d2_omega[-1] += (2.0 / self.h ** 4) * psi[-1]
        l2_psi = d2_omega - self.k ** 2 * omega
        psi_t = sla.lu_solve(self.lu, l2_psi + self.k * (self.ra / self.pr) * theta)
        # theta = 0 at both walls -> plain Dirichlet Laplacian
        theta_t = (self.L @ theta) / self.pr - self.k * psi
        return psi_t, theta_t


def growth_rate(ra: float, k: float, ny: int = 40, pr: float = 0.71, walls: str = "rigid",
                steps: int = 8000, dt: float = 2e-4, seed: int = 0) -> dict:
    """Growth rate of the least-stable mode at one (Ra, k), by RK4 time marching."""
    op = ModeOperator(ny, k, ra, pr, walls)
    rng = np.random.default_rng(seed)
    y = np.linspace(0.0, 1.0, ny + 2)[1:-1]
    psi = np.sin(np.pi * y) + 0.05 * rng.standard_normal(ny)
    theta = np.sin(np.pi * y) + 0.05 * rng.standard_normal(ny)
    amps, ts = [], []
    for n in range(steps):
        k1p, k1t = op.rhs(psi, theta)
        k2p, k2t = op.rhs(psi + 0.5 * dt * k1p, theta + 0.5 * dt * k1t)
        k3p, k3t = op.rhs(psi + 0.5 * dt * k2p, theta + 0.5 * dt * k2t)
        k4p, k4t = op.rhs(psi + dt * k3p, theta + dt * k3t)
        psi = psi + dt / 6.0 * (k1p + 2 * k2p + 2 * k3p + k4p)
        theta = theta + dt / 6.0 * (k1t + 2 * k2t + 2 * k3t + k4t)
        if n % 25 == 0:
            amps.append(float(np.sqrt(np.dot(psi, psi) + np.dot(theta, theta))))
            ts.append(n * dt)
    amps = np.asarray(amps)
    ts = np.asarray(ts)
    lo = int(0.35 * amps.size)
    lp = np.log(amps[lo:])
    p = np.polyfit(ts[lo:], lp, 1)
    pred = np.polyval(p, ts[lo:])
    ss_res = float(np.sum((lp - pred) ** 2))
    ss_tot = float(np.sum((lp - lp.mean()) ** 2))
    return {
        "ra": ra, "k": k, "walls": walls,
        "sigma": float(p[0]),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0,
        "amp_initial": float(amps[1]), "amp_final": float(amps[-1]),
        "steps": steps, "dt": dt, "t_end": steps * dt,
    }


def bisect(k: float, walls: str, ra_lo: float, ra_hi: float, ny: int = 40,
           tol_rel: float = 2e-3, max_iter: int = 8, steps: int = 8000,
           dt: float = 2e-4) -> dict:
    lo, hi = ra_lo, ra_hi
    hist = []
    s_lo = growth_rate(lo, k, ny=ny, walls=walls, steps=steps, dt=dt)
    s_hi = growth_rate(hi, k, ny=ny, walls=walls, steps=steps, dt=dt)
    hist += [s_lo, s_hi]
    if not (s_lo["sigma"] < 0.0 < s_hi["sigma"]):
        return {"valid": False, "sigma_lo": s_lo["sigma"], "sigma_hi": s_hi["sigma"],
                "history": hist, "reason": "bracket does not straddle sigma = 0"}
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        rec = growth_rate(mid, k, ny=ny, walls=walls, steps=steps, dt=dt)
        hist.append(rec)
        if rec["sigma"] > 0.0:
            hi = mid
        else:
            lo = mid
        if (hi - lo) / mid < tol_rel:
            break
    return {"valid": True, "Ra_bracket": [lo, hi], "Ra_c": 0.5 * (lo + hi),
            "sigma_bracket": [s_lo["sigma"], s_hi["sigma"]], "history": hist,
            "n_arms": len(hist), "bracket_width_rel": (hi - lo) / (0.5 * (lo + hi))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="independent RB marginal-stability control")
    ap.add_argument("--out", type=Path,
                    default=Path("runs/case_rayleigh_benard_onset/independent_control.json"))
    ap.add_argument("--ny", type=int, default=40)
    ap.add_argument("--steps", type=int, default=8000)
    args = ap.parse_args(argv)

    t0 = time.time()
    out: dict = {
        "what": "independent re-derivation of the continuum marginal Ra (time-domain FD)",
        "independent_of": "tensorlbm.rb_continuum_stability (FD state-space eigenproblem + ARPACK)",
        "method": "one-mode linearised NS-Boussinesq, RK4 march, sigma(Ra) bisection, no eigen-solve",
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "grid": {"ny": args.ny, "steps": args.steps, "dt": 2e-4, "pr": 0.71},
        "limitations": [
            "uses the same variable choice omega=(D^2-k^2)psi as Route C, so an error in the "
            "equations themselves would not be caught; what it independently checks is Route C's "
            "discretisation, eigen-solve and marginal criterion",
            "second-order FD in y at ny=40: O(h^2) error, so agreement is expected at the "
            "0.1-1 % level, not the 1e-4 of Route C",
            "one Fourier mode at a time at the reference wavenumber a=3.1163: the wavenumber "
            "minimum is not re-minimised, so no independent a_c claim is made",
            "free-slip calibration is run only at a=pi/sqrt(2) (its analytic optimum)",
        ],
    }

    # --- 1. calibration against the closed-form free-slip result -------------
    ff = bisect(A_FREEFREE, "free", RA_FREEFREE * 0.75, RA_FREEFREE * 1.25,
                ny=args.ny, steps=args.steps)
    ff["Ra_analytic"] = RA_FREEFREE
    ff["a_analytic"] = A_FREEFREE
    ff["err_pct"] = 100.0 * (ff["Ra_c"] - RA_FREEFREE) / RA_FREEFREE if ff["valid"] else None
    out["calibration_free_slip"] = ff

    # --- 2. the case's physical problem: rigid-rigid isothermal --------------
    rr = bisect(A_RIGID, "rigid", RA_RIGID * 0.85, RA_RIGID * 1.15,
                ny=args.ny, steps=args.steps)
    rr["Ra_ref"] = RA_RIGID
    rr["err_pct"] = 100.0 * (rr["Ra_c"] - RA_RIGID) / RA_RIGID if rr["valid"] else None
    out["rigid_rigid"] = rr

    # --- 3. grid sensitivity of the independent arm --------------------------
    sens = []
    for ny in (20, 30, 40):
        r = bisect(A_RIGID, "rigid", RA_RIGID * 0.85, RA_RIGID * 1.15,
                   ny=ny, steps=args.steps, max_iter=6)
        if r["valid"]:
            sens.append({"ny": ny, "Ra_c": r["Ra_c"],
                         "err_pct": 100.0 * (r["Ra_c"] - RA_RIGID) / RA_RIGID})
    out["grid_sensitivity"] = sens

    ok_ff = ff["valid"] and abs(ff["err_pct"]) < 1.0
    ok_rr = rr["valid"] and abs(rr["err_pct"]) < 1.5
    out["checks"] = {
        "free_slip_calibration_1pct": bool(ok_ff),
        "rigid_rigid_1.5pct": bool(ok_rr),
        "fits_are_real": bool(all(h["r2"] > 0.99 for h in rr["history"])),
        "grid_sensitivity_reported": bool(len(sens) == 3),
    }
    out["all_checks_pass"] = all(out["checks"].values())
    out["wall_seconds"] = round(time.time() - t0, 2)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({
        "free_slip_Ra_c": ff["Ra_c"], "free_slip_err_pct": ff["err_pct"],
        "rigid_Ra_c": rr["Ra_c"], "rigid_err_pct": rr["err_pct"],
        "checks": out["checks"], "pass": out["all_checks_pass"],
        "wall_seconds": out["wall_seconds"],
    }, indent=2, default=float))
    return 0 if out["all_checks_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())