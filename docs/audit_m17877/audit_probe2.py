#!/usr/bin/env python3
"""m17877 audit experiment 2: fidelity + convergence-certificate probes.

P1  OSS pressure definition: oss fingering.py:130  p = rho/3 + psi*mu
    port (m17724)                                   p = rho/3 + psi*mu/3
    -> re-run the primary case with the OSS definition (it also feeds feq) and
       compare measured Laplace error, to size the impact of the deviation.
P2  convergence certificate: the delivered artifact samples dp only at
    steps 0/500/1000/1500/1999; error sequence 98.7/19.0/3.98/3.50/0.07 % is
    non-monotone and u_max rebounds 3.07e-6 -> 9.64e-6.  Re-sample every 25
    steps from 1200..2000 to expose the oscillation band of dp around sigma/R.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

SRC = "/root/tp-agent/work/m17724/xval/xval_fingering_dynamics.py"
OUT = Path(__file__).resolve().parent / "audit_probe2.json"

spec = importlib.util.spec_from_file_location("xval_m17724", SRC)
mod = importlib.util.module_from_spec(spec)
sys.modules["xval_m17724"] = mod
spec.loader.exec_module(mod)


def run_oss_pressure(steps: int = 2000) -> dict:
    """Same integrator, but pressure = rho/3 + psi*mu as in the OSS source."""
    cfg = mod.XVALConfig()
    sim = mod.SOYFreeEnergyLBM(cfg.ny, cfg.nx, sigma=cfg.sigma, xi=cfg.xi,
                               tau_f=cfg.tau_f, tau_g=cfg.tau_g,
                               gamma=(cfg.tau_g - 0.5) / 3.0, dtype=torch.float64)
    sim._pressure = lambda rho, psi, mu: rho / 3.0 + psi * mu   # OSS fingering.py:130
    sim.init_droplet(cfg.R)
    hist = []
    for s in range(steps):
        sim.step()
        if s % 500 == 0 or s == steps - 1:
            dp, umax, pmin, pmax = sim.measure()
            hist.append([s, dp, umax])
    dp = hist[-1][1]
    an = cfg.sigma / cfg.R
    return {"dp": dp, "err_pct": 100 * abs(dp - an) / an, "history": hist,
            "u_max": hist[-1][2]}


def convergence_band() -> dict:
    cfg = mod.XVALConfig()
    sim = mod.SOYFreeEnergyLBM(cfg.ny, cfg.nx, sigma=cfg.sigma, xi=cfg.xi,
                               tau_f=cfg.tau_f, tau_g=cfg.tau_g,
                               gamma=(cfg.tau_g - 0.5) / 3.0, dtype=torch.float64)
    sim.init_droplet(cfg.R)
    an = cfg.sigma / cfg.R
    rows = []
    for s in range(2000):
        sim.step()
        if s >= 1200 and s % 25 == 0:
            dp, umax, _, _ = sim.measure()
            rows.append([s, dp, 100 * abs(dp - an) / an, umax])
    errs = [r[2] for r in rows]
    return {
        "analytic_dp": an,
        "window": [1200, 1999],
        "samples": rows,
        "err_min_pct": min(errs), "err_max_pct": max(errs),
        "err_mean_pct": sum(errs) / len(errs),
        "err_at_1999_pct": rows[-1][2],
        "umax_min": min(r[3] for r in rows), "umax_max": max(r[3] for r in rows),
    }


if __name__ == "__main__":
    res = {"P1_oss_pressure_definition": run_oss_pressure(),
           "P2_convergence_band": convergence_band()}
    OUT.write_text(json.dumps(res, indent=2, default=float))
    p1 = res["P1_oss_pressure_definition"]
    p2 = res["P2_convergence_band"]
    print("P1 oss-p: dp=%.9e err=%.4f%% u_max=%.3e" % (p1["dp"], p1["err_pct"], p1["u_max"]))
    print("P2 band: err %.3f..%.3f %% (mean %.3f), err@1999=%.3f%%, u_max %.2e..%.2e"
          % (p2["err_min_pct"], p2["err_max_pct"], p2["err_mean_pct"],
             p2["err_at_1999_pct"], p2["umax_min"], p2["umax_max"]))