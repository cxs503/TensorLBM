"""Shared helpers for the pseudopotential force-format upgrade benchmark.

Physical convention (see ``tensorlbm.multiphase_edm``)::

    F = -G_eff cs^2 psi grad(psi),   G_eff < 0  -> attraction
    p_mech = rho*cs^2 + (G_eff/2)*cs^2*psi^2

The library (``tensorlbm.multiphase``) uses ``G_lib = -G_eff``; every number
reported here is in the *physical* convention, with ``G_lib`` given alongside.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.multiphase import psi_exp  # noqa: E402
from tensorlbm.multiphase_edm import (  # noqa: E402
    CS2,
    collide_sc_force,
    pressure_mech,
    sc_interaction_force,
    stream_step,
)

# Discrete coexistence densities of psi_exp with G_eff = -5 (tau-independent):
# measured in benchmarks/verified/laplace_droplet (L=100, R=25).
RHO_L = 1.957
RHO_V = 0.1596

G_EFF = -5.0
G_LIB = 5.0

#: force formats under test: label -> (stencil, form, forcing scheme)
FORMATS = {
    # library-historical default: classic force + velocity-shift forcing
    "F0_sc_classic_vshift": ("d2q9", "product", "velocity_shift"),
    # classic force + exact-difference (Kupershtokh) forcing
    "F1_sc_classic_edmforce": ("d2q9", "product", "kupershtokh"),
    # mission item (1): exact-difference *force* on the D2Q9 stencil
    "F2_edm_d2q9": ("d2q9", "difference", "kupershtokh"),
    # mission items (1)+(2): exact-difference force + 6th-order isotropic stencil
    "F3_edm_iso6": ("iso6", "difference", "kupershtokh"),
    # psi^2 divergence (pressure-tensor) form on the iso6 stencil
    "F4_pt_iso6": ("iso6", "psi2", "kupershtokh"),
}


@dataclass
class DropletSetup:
    L: int
    R: float
    tau: float = 1.0
    width: float = 2.0
    rho_l: float = RHO_L
    rho_v: float = RHO_V
    G_eff: float = G_EFF


def droplet_init(setup: DropletSetup, device="cpu") -> torch.Tensor:
    """D2Q9 equilibrium distribution of a tanh liquid droplet (rest fluid)."""
    L = setup.L
    y, x = torch.meshgrid(
        torch.arange(L, dtype=torch.float32), torch.arange(L, dtype=torch.float32),
        indexing="ij",
    )
    c = 0.5 * (L - 1)
    r = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
    phi = 0.5 * (1.0 - torch.tanh((r - setup.R) / (setup.width / 2.0)))
    rho = setup.rho_v + (setup.rho_l - setup.rho_v) * phi
    zero = torch.zeros_like(rho)
    return equilibrium(rho.to(device), zero, zero)


def run_droplet(
    setup: DropletSetup,
    fmt: str = "F3_edm_iso6",
    forcing: str | None = None,
    max_steps: int = 20000,
    check_every: int = 200,
    steady_tol: float = 2e-4,
    steady_window: int = 3,
    thread_count: int = 8,
) -> dict:
    """Time-march a static droplet to steady state and measure the Laplace data.

    Steady state: relative drift of ``dp`` and ``R_eq`` below ``steady_tol`` for
    ``steady_window`` consecutive checks.  Measurements are averages over the
    last ``steady_window`` samples.
    """
    torch.set_num_threads(thread_count)
    stencil, form, default_forcing = FORMATS[fmt]
    forcing = default_forcing if forcing is None else forcing
    f = droplet_init(setup)
    L = setup.L
    c = 0.5 * (L - 1)
    y, x = torch.meshgrid(
        torch.arange(L, dtype=torch.float32), torch.arange(L, dtype=torch.float32),
        indexing="ij",
    )
    r = torch.sqrt((x - c) ** 2 + (y - c) ** 2)
    rho_mid = 0.5 * (setup.rho_l + setup.rho_v)

    mass0 = float(f.sum()) / L / L
    hist = []
    samples = []
    prev_dp = prev_req = None
    n_steady = 0
    step = 0
    converged = False
    while step < max_steps:
        for _ in range(min(check_every, max_steps - step)):
            rho, _, _ = macroscopic(f)
            psi = psi_exp(rho)
            Fx, Fy = sc_interaction_force(psi, setup.G_eff, stencil, form, periodic=True)
            f = collide_sc_force(f, Fx, Fy, setup.tau, forcing)
            f = stream_step(f)
        step += check_every

        rho, ux, uy = macroscopic(f)
        psi = psi_exp(rho)
        p = pressure_mech(rho, psi, setup.G_eff)
        core = r < 0.5 * setup.R
        far = r > 1.5 * setup.R
        dp = float(p[core].mean() - p[far].mean())
        area = float((rho > rho_mid).sum()) * (L * L) / (L * L)  # cell count
        req = math.sqrt(area / math.pi)
        umax = float(torch.sqrt(ux * ux + uy * uy).max())
        mass = float(f.sum()) / L / L
        hist.append({"step": step, "dp": dp, "R_eq": req, "u_max": umax,
                     "mass": mass, "rho_min": float(rho.min()), "rho_max": float(rho.max())})
        if prev_dp is not None:
            drift = max(abs(dp - prev_dp) / max(abs(dp), 1e-30),
                        abs(req - prev_req) / max(req, 1e-30))
            n_steady = n_steady + 1 if drift < steady_tol else 0
        prev_dp, prev_req = dp, req
        samples.append({"step": step, "dp": dp, "R_eq": req, "u_max": umax,
                        "u_max_iface": float(
                            (torch.sqrt(ux * ux + uy * uy)[(r - req).abs() < 2 * setup.width]).max()
                        ) if bool(((r - req).abs() < 2 * setup.width).any()) else float("nan"),
                        "mass": mass})
        if not math.isfinite(dp) or not math.isfinite(req):
            break
        if n_steady >= steady_window:
            converged = True
            break

    tail = samples[-steady_window:]
    dp_m = float(np.mean([s["dp"] for s in tail]))
    req_m = float(np.mean([s["R_eq"] for s in tail]))
    res = {
        "format": fmt,
        "stencil": stencil,
        "form": form,
        "forcing": forcing,
        "setup": asdict(setup),
        "G_lib": -setup.G_eff,
        "steps": step,
        "converged": converged,
        "dp": dp_m,
        "R_eq": req_m,
        "sigma": dp_m * req_m,            # 2D Laplace: dp = sigma / R
        "u_max": float(np.mean([s["u_max"] for s in tail])),
        "u_max_iface": float(np.nanmean([s["u_max_iface"] for s in tail])),
        "mass_drift_rel": abs(float(f.sum()) / setup.L / setup.L - mass0) / mass0,
        "rho_min": min(h["rho_min"] for h in hist),
        "rho_max": max(h["rho_max"] for h in hist),
        "finite": bool(all(math.isfinite(h["dp"]) for h in hist)),
        "history": hist,
    }
    return res


def laplace_fit(results: list[dict]) -> dict:
    """Fit ``dp = sigma/R_eq`` (free intercept) and report per-radius consistency."""
    inv = np.array([1.0 / r["R_eq"] for r in results])
    dp = np.array([r["dp"] for r in results])
    A = np.vstack([inv, np.ones_like(inv)]).T
    coef, *_ = np.linalg.lstsq(A, dp, rcond=None)
    slope, intercept = float(coef[0]), float(coef[1])
    pred = A @ coef
    resid = dp - pred
    ss_tot = float(((dp - dp.mean()) ** 2).sum())
    r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    sig_fit = slope
    per = []
    for r in results:
        sig_i = r["dp"] * r["R_eq"]
        per.append({
            "R_init": r["setup"]["R"],
            "R_eq": r["R_eq"],
            "dp": r["dp"],
            "sigma_i": sig_i,
            "dev_pct": 100.0 * (sig_i - sig_fit) / sig_fit,
        })
    devs = [p["dev_pct"] for p in per]
    devs = [d for d in devs if np.isfinite(d)]
    max_dev = float(np.max(np.abs(devs))) if devs else float("nan")
    return {
        "sigma_fit": sig_fit,
        "intercept": intercept,
        "r2": r2,
        "resid_rms_pct": 100.0 * float(np.sqrt((resid**2).mean())) / float(np.abs(dp).max()),
        "max_sigma_dev_pct": max_dev,
        "intercept_over_slope_over_Rmax": float(
            abs(intercept) / (abs(slope) * inv.mean())
        ),
        "per_radius": per,
    }


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)