"""Rayleigh–Bénard Nusselt-number case (P97-N05 shard worker).

Buoyancy-driven convection between two rigid, isothermal plates, periodic in
``x``, D2Q9 momentum + D2Q5 temperature (double distribution function, Guo
forcing, Boussinesq).  Everything below is a thin driver on top of the
library's own primitives (:mod:`tensorlbm.thermal`), so the operator being
measured is the one shipped in the package.

Diagnostics
-----------
* ``Nu`` from the wall heat flux (second-order one-sided difference — the
  node-based convention that matches the anti-bounce-back Dirichlet walls used
  here; the benchmark README of ``benchmarks/pending/rayleigh_benard`` records
  why the half-way口径 degenerates for this wall type).
* ``Nu_mid`` from the *interior* energy-flux identity
  ``Nu = (⟨u_y T⟩ - α ∂⟨T⟩/∂y) H / (α ΔT)`` at mid-height.  Bottom-wall, top-wall
  and mid-plane Nusselt numbers must agree in a statistically steady state:
  that agreement is a machine-checkable energy-conservation gate which does not
  depend on any reference value.
* ``ke`` (kinetic energy) and ``u_max`` for the flow-regime classification
  (conduction → steady rolls → time-dependent convection).

Usage::

    python scripts/p97_n05_rb_case.py --ra 1e5 --nx 100 --ny 50 --steps 120000 \
        --out runs/p97_n05/rb_ra1e5.json --fields-out runs/p97_n05/rb_ra1e5.npz
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import stream  # noqa: E402
from tensorlbm.thermal import (  # noqa: E402
    W5,
    buoyancy_force,
    collide_bgk_force,
    pre_streaming_bounce_back,
    temperature_collision,
    temperature_equilibrium,
    temperature_stream,
)

RA_CRIT_CONTINUOUS = 1707.76


def rb_params(ny: int, ra: float, pr: float, tau: float) -> dict[str, float]:
    """Lattice parameters of the RB cell.

    The anti-bounce-back Dirichlet walls pin the temperature *on* the first and
    last rows, so the physical gap of the thermal problem is ``H = ny - 1``
    cells and ``Ra`` must be formed with that gap.  (The momentum half-way
    bounce-back walls sit at ``y = -1/2`` and ``ny - 1/2``, i.e. one cell wider:
    that half-cell mismatch between the thermal and momentum walls is a
    documented limitation of the existing operator, kept here so that the
    numbers describe the operator that is actually shipped.  Using ``H = ny``
    instead biases the realised ``Ra`` by ``-(1/ny)^3``, i.e. 2 % at ny = 50 and
    6 % at ny = 16.)
    """
    nu = (tau - 0.5) / 3.0
    alpha = nu / pr
    H = float(ny - 1)
    return {
        "nu": nu,
        "alpha": alpha,
        "tau_T": 3.0 * alpha + 0.5,
        "H": H,
        "g_beta": ra * nu * alpha / H**3,
    }


def temperature_walls_rb(g: torch.Tensor, t_hot: float, t_cold: float) -> torch.Tensor:
    """Anti-bounce-back Dirichlet walls: bottom hot (y = -1/2), top cold."""
    g = g.clone()
    w = W5.to(g.device)
    g[3, 0, :] = -g[4, 0, :] + 2.0 * w[3] * t_hot
    g[4, -1, :] = -g[3, -1, :] + 2.0 * w[4] * t_cold
    return g


def nusselt_walls(T: torch.Tensor, H: float, dT: float, t_hot: float, t_cold: float):
    """Bottom/top wall Nusselt numbers, second-order one-sided differences."""
    nu_bottom = (1.5 * T[0, :] - 2.0 * T[1, :] + 0.5 * T[2, :]) * H / dT
    nu_top = (2.0 * T[-2, :] - 1.5 * T[-1, :] - 0.5 * T[-3, :]) * H / dT
    return float(nu_bottom.mean()), float(nu_top.mean())


def nusselt_midplane(
    uy: torch.Tensor, T: torch.Tensor, alpha: float, H: float, dT: float
) -> float:
    """Interior Nusselt number from the energy-flux identity at mid-height."""
    j = uy.shape[0] // 2
    t_bar = T.mean(dim=1)
    flux = (uy * T).mean(dim=1)
    dTdy = (t_bar[j + 1] - t_bar[j - 1]) / 2.0
    return float((flux[j] - alpha * dTdy) * H / (alpha * dT))


def simulate_rb(
    ra: float,
    nx: int,
    ny: int,
    pr: float = 0.71,
    tau: float = 0.6,
    steps: int = 120000,
    sample_every: int = 200,
    spinup_frac: float = 0.5,
    avg_frac: float = 0.3,
    amp: float = 1e-3,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    """Time-march the RB cell and return Nusselt-number statistics."""
    torch.manual_seed(seed)
    device = torch.device(device)
    p = rb_params(ny, ra, pr, tau)
    nu, alpha, tau_T, H, g_beta = p["nu"], p["alpha"], p["tau_T"], p["H"], p["g_beta"]
    t_hot, t_cold, dT = 1.0, 0.0, 1.0

    yy = torch.arange(ny, dtype=torch.float64).view(ny, 1)
    xx = torch.arange(nx, dtype=torch.float64).view(1, nx)
    T0 = (t_hot - dT * yy / H) + amp * torch.sin(2.0 * math.pi * xx / nx) * torch.sin(
        math.pi * yy / H
    )
    rho = torch.ones((ny, nx), device=device)
    zero = torch.zeros_like(rho)
    f = equilibrium(rho, zero, zero)
    g = temperature_equilibrium(T0.to(torch.float32).expand(ny, nx).contiguous(), zero, zero)

    wall = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    wall[0, :] = True
    wall[-1, :] = True

    hist = []
    t0 = time.time()
    for step in range(1, steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)
        F = buoyancy_force(rho, T, g_beta, t_ref=t_cold)
        g = temperature_collision(g, tau_T, ux, uy)
        g = temperature_stream(g)
        g = temperature_walls_rb(g, t_hot, t_cold)
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)
        if step % sample_every == 0:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nb, nt = nusselt_walls(T, H, dT, t_hot, t_cold)
            nm = nusselt_midplane(uy, T, alpha, H, dT)
            ke = 0.5 * float((rho * (ux**2 + uy**2)).mean())
            hist.append(
                {
                    "step": step,
                    "nu_bottom": nb,
                    "nu_top": nt,
                    "nu_mid": nm,
                    "ke": ke,
                    "u_max": float(torch.sqrt(ux**2 + uy**2).max()),
                }
            )
    elapsed = time.time() - t0

    steps_arr = np.array([h["step"] for h in hist], dtype=float)
    avg_mask = steps_arr >= steps * (1.0 - avg_frac)
    spin_mask = steps_arr >= steps * spinup_frac

    def _avg(key, mask):
        vals = np.array([h[key] for h in hist], dtype=float)[mask]
        return float(vals.mean()) if vals.size else float("nan")

    def _std(key, mask):
        vals = np.array([h[key] for h in hist], dtype=float)[mask]
        return float(vals.std()) if vals.size else float("nan")

    nu_wall_avg = 0.5 * (_avg("nu_bottom", avg_mask) + _avg("nu_top", avg_mask))
    nu_mid_avg = _avg("nu_mid", avg_mask)
    flux_gap = abs(nu_wall_avg - nu_mid_avg) / abs(nu_wall_avg) if nu_wall_avg else float("nan")
    ke_series = np.array([h["ke"] for h in hist], dtype=float)
    late = ke_series[spin_mask]
    unsteady = float(late.std() / late.mean()) if late.size and late.mean() > 0 else float("nan")

    return {
        "ra": ra,
        "pr": pr,
        "nx": nx,
        "ny": ny,
        "tau": tau,
        "tau_T": tau_T,
        "nu_lattice": nu,
        "alpha": alpha,
        "H": H,
        "g_beta": g_beta,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "nu_wall": nu_wall_avg,
        "nu_bottom": _avg("nu_bottom", avg_mask),
        "nu_top": _avg("nu_top", avg_mask),
        "nu_mid": nu_mid_avg,
        "nu_bottom_std": _std("nu_bottom", avg_mask),
        "nu_wall_gap": flux_gap,
        "ke_mean": _avg("ke", avg_mask),
        "u_max_mean": _avg("u_max", avg_mask),
        "ke_fluctuation": unsteady,
        "regime": "conduction"
        if nu_wall_avg < 1.01
        else ("steady" if (unsteady == unsteady and unsteady < 0.02) else "unsteady"),
        "ra_over_ra_c": ra / RA_CRIT_CONTINUOUS,
        "history": hist[:: max(1, len(hist) // 200)],
        "T_field": T.detach().cpu().numpy().astype(np.float32),
        "u_field": torch.sqrt(ux**2 + uy**2).detach().cpu().numpy().astype(np.float32),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--nx", type=int, default=100)
    ap.add_argument("--ny", type=int, default=50)
    ap.add_argument("--pr", type=float, default=0.71)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=120000)
    ap.add_argument("--sample-every", type=int, default=200)
    ap.add_argument("--spinup-frac", type=float, default=0.5)
    ap.add_argument("--avg-frac", type=float, default=0.3)
    ap.add_argument("--amp", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fields-out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    res = simulate_rb(
        ra=args.ra,
        nx=args.nx,
        ny=args.ny,
        pr=args.pr,
        tau=args.tau,
        steps=args.steps,
        sample_every=args.sample_every,
        spinup_frac=args.spinup_frac,
        avg_frac=args.avg_frac,
        amp=args.amp,
    )
    T_field = res.pop("T_field")
    u_field = res.pop("u_field")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    if args.fields_out:
        Path(args.fields_out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.fields_out, T=T_field, umag=u_field)
    print(
        f"Ra={args.ra:.3g} Nu={res['nu_wall']:.4f} (mid {res['nu_mid']:.4f}, "
        f"gap {100 * res['nu_wall_gap']:.2f}%) regime={res['regime']} "
        f"ke_fluc={res['ke_fluctuation']:.3f} [{res['elapsed_s']}s]"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())