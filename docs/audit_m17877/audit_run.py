#!/usr/bin/env python3
"""m17877 board-audit independent recompute of m17724's XVAL claim.

Reads m17724's delivered module READ-ONLY, re-runs the primary case and the
R=10 case from scratch, and adds an independent surface-tension route
(capillary excess free energy) so the Laplace check is not scored only through
the artifact's own metric.  No file of the audited worktree is modified.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = "/root/tp-agent/work/m17724/xval/xval_fingering_dynamics.py"
OUT = Path(__file__).resolve().parent / "audit_rerun.json"

spec = importlib.util.spec_from_file_location("xval_m17724", SRC)
mod = importlib.util.module_from_spec(spec)
sys.modules['xval_m17724'] = mod
spec.loader.exec_module(mod)


def sigma_from_capillary_energy(sim) -> float:
    """σ = (1/(2πR)) ∮ κ|∇ψ|²/2 dA -- independent route (free-energy functional)."""
    gx, gy, _ = sim._compute_gradients(sim.psi)
    dens = 0.5 * sim.kappa * (gx ** 2 + gy ** 2)
    area = float(sim.nx * sim.ny)
    # periodic 60x60 domain: only the droplet interface contributes
    return float(dens.sum()) / (2.0 * np.pi * sim._R)


def run_case(R: float, steps: int, sample_every: int) -> dict:
    cfg = mod.XVALConfig()
    gk = mod.SOYFreeEnergyLBM(
        cfg.ny, cfg.nx, sigma=cfg.sigma, xi=cfg.xi, tau_f=cfg.tau_f,
        tau_g=cfg.tau_g, gamma=(cfg.tau_g - 0.5) / 3.0, dtype=torch.float64,
    )
    gk.init_droplet(R)
    t0 = time.time()
    hist = []
    for s in range(steps):
        gk.step()
        if s % sample_every == 0 or s == steps - 1:
            dp, umax, pmin, pmax = gk.measure()
            hist.append([s, dp, umax, pmin, pmax])
    el = time.time() - t0
    dp, umax, pmin, pmax = hist[-1][1:]
    an = cfg.sigma / R
    return {
        "R": R, "steps": steps, "wall_s": round(el, 2),
        "dp_got": dp, "dp_analytic": an,
        "err_pct": 100.0 * abs(dp - an) / an,
        "u_max": umax, "psi_min": pmin, "psi_max": pmax,
        "sigma_eff_from_dp": dp * R,
        "sigma_from_capillary_energy": sigma_from_capillary_energy(gk),
        "history": hist,
    }


def main() -> None:
    res = {
        "source_module": SRC,
        "config": mod.XVALConfig().__dict__,
        "kappa_sigma_relation": "kappa = 0.75*sigma*xi  =>  model sigma = (4/3)kappa/xi = sigma_input "
                                "(reference is model-derived, not independent)",
        "primary_R15_T2000": run_case(R=15.0, steps=2000, sample_every=500),
        "claim_R10_T800": run_case(R=10.0, steps=800, sample_every=200),
    }
    OUT.write_text(json.dumps(res, indent=2, default=float))
    print(json.dumps({k: v for k, v in res.items() if k.startswith(("primary", "claim"))},
                     indent=2, default=float))


if __name__ == "__main__":
    sys.exit(main())