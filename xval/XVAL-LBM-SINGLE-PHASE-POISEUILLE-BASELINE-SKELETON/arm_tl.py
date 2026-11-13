"""TensorLBM arm of ``XVAL-LBM-SINGLE-PHASE-POISEUILLE-BASELINE-SKELETON``.

Runs the in-repo D2Q9 BGK + Guo (2002) forcing + half-way bounce-back channel
(``tensorlbm.verification.layered_poiseuille.solver`` with a single layer) at the
lattice numbers of one arm from ``cases.py`` and returns the cross-section
profile plus the march bookkeeping.  All quantities in lattice units.
"""

from __future__ import annotations

import time
from typing import Dict

import numpy as np

from tensorlbm.verification.layered_poiseuille.solver import ChannelConfig, LayeredChannel
from tensorlbm.verification.xval.poiseuille_skeleton import (
    analytic_profile,
    fit_effective_height,
    profile_kpis,
    wall_normal_coordinates,
)


def run(spec: dict, steady_tol: float = 1e-6, window: int = 200,
        wall_cap_s: float = 900.0) -> Dict:
    from cases import dimensionless  # noqa: WPS433  (local import: keeps arm importable alone)

    res = int(spec["resolution"])
    nu = float(spec["viscosity_lu"])
    g = float(spec["acceleration_lu"])
    H = float(res - 2)
    y, _ = wall_normal_coordinates(res)

    init_mode = "analytic" if spec.get("init") == "analytic" else "uniform"
    unit_profile = (lambda yy: analytic_profile(yy, H, 1.0, nu)) if init_mode == "analytic" else None

    cfg = ChannelConfig(
        height=res - 2,
        nx=res,
        nus=(nu,),
        interface_fracs=(),
        force=g,
        x_boundary="periodic",
        init_mode=init_mode,
        steps=1,                    # unused: this driver marches manually
        sample_every=window,
        tol_steady=steady_tol,
        label=spec["label"],
    )
    t0 = time.time()
    chan = LayeredChannel(cfg, unit_profile)

    prev = chan.profile()
    steps = 0
    converged = False
    truncated = False
    change = float("inf")
    while steps < int(spec["max_steps"]):
        for _ in range(window):
            chan.step()
        steps += window
        prof = chan.profile()
        change = float(np.max(np.abs(prof - prev)) / max(np.max(np.abs(prof)), 1e-300))
        prev = prof.copy()
        if change < steady_tol:
            converged = True
            break
        if time.time() - t0 > wall_cap_s:
            truncated = True
            break
    wall = time.time() - t0

    rho, ux, uy = chan.velocity()
    prof = chan.profile()
    u_ex = analytic_profile(y, H, g, nu)
    kpis = profile_kpis(y, prof, u_ex)
    h_eff = fit_effective_height(y, prof, g, nu)
    interior_rho = rho[1:-1, :]
    interior_uy = uy[1:-1, :]
    return {
        "code": "tensorlbm",
        "label": spec["label"],
        "family": spec["family"],
        "resolution": res,
        "steps": steps,
        "converged": bool(converged),
        "wall_cap_hit": bool(truncated),
        "wall_s": round(wall, 2),
        "steady_change": change,
        "y": y.tolist(),
        "u_profile": prof.tolist(),
        "u_field": ux.tolist(),
        "rho_min": float(interior_rho.min()),
        "rho_max": float(interior_rho.max()),
        "uy_max_abs": float(np.max(np.abs(interior_uy))),
        "u_centerline": float(prof[len(prof) // 2]),
        "h_eff_mean": float(np.mean(h_eff)),
        "h_eff_max_dev": float(np.max(np.abs(h_eff - H))),
        "kpis": kpis,
        "dimensionless": dimensionless(spec),
        "params": dict(spec),
        "q_continuum": float(g * H ** 3 / (12.0 * nu)),
    }