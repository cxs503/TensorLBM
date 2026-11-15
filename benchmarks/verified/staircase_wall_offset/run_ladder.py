#!/usr/bin/env python3
"""Staircase (45-degree inclined) wall: grid ladder for the effective wall position.

Physics
-------
Plane Poiseuille flow in a doubly periodic box whose channel walls are the
lattice staircases of the lines ``s = +-H`` with ``s = wrap(x - y)/sqrt(2)``.
The lattice steps stay 1 cell wide while the channel is widened, so a ladder in
``H`` is a *geometric* refinement of the same physical channel: the sawtooth
amplitude relative to the channel width is ``1/H``.

Analytic reference (body force ``g`` along ``(1,1)/sqrt(2)``, no slip at
``|s| = H``, kinematic viscosity ``nu``; White, *Viscous Fluid Flow*, fully
developed laminar channel)::

    u(s) = g (H^2 - s^2) / (2 nu),      Q = 2 g H^3 / (3 nu)   [per unit length]

Two wall formats are compared on the *same* solid mask:

``halfway``  production ``bounce_back_cells`` (no slip on the staircase face)
``interp``   2-point Bouzidi closure with ``q`` from the signed-distance field

Observables (all computed, none tuned)
--------------------------------------
``H_vol``    volume-equivalent half-width from the fluid *mask*:
             ``H_vol = N_fluid / (2 * L * sqrt(2))`` (the map ``(x,y)->(s,t)`` is
             area preserving, so this is exact for the discrete geometry).
``delta_geom = H_vol - H``  how much wider the staircase keeps the channel.
``delta_eff`` from the single integral observable ``Q`` (no profile fitting)::

             H_eff = (3 nu Q / (2 g))^(1/3),   delta_eff = H_eff - H

``theta``    profile error ``max_bin |u_num - u_ana| / u_max^ana`` over core
             bins, evaluated against both the nominal wall ``H`` (``theta_nom``)
             and the volume-equivalent wall ``H_vol`` (``theta_vol``).

Usage::

    PYTHONPATH=src python3 benchmarks/verified/staircase_wall_offset/run_ladder.py \
        --out benchmarks/verified/staircase_wall_offset/artifacts
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.boundary.interpolated_bounce_back import (  # noqa: E402
    interpolated_bounce_back_2d,
)
from tensorlbm.d2q9 import C as C9  # noqa: E402
from tensorlbm.d2q9 import W as W9  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

#: (L, H) ladder.  ``H`` is chosen so the staircase keeps a sane fluid mask
#: (``H < L / (2 sqrt 2)``) and the channel widens in ~1.9x steps.
DEFAULT_GRIDS: tuple[tuple[int, float], ...] = (
    (40, 8.0),
    (56, 11.0),
    (76, 15.0),
    (90, 18.0),
    (112, 22.0),
)
TAU = 0.8
U_MAX_TARGET = 0.02


def force_source_2d(fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """Guo momentum source ``s_i = 3 w_i (c_i . F)``: ``sum_i s_i c_i = F`` exactly."""
    out = torch.zeros((9,) + fx.shape)
    for i in range(9):
        out[i] = 3.0 * W9[i] * (C9[i, 0] * fx + C9[i, 1] * fy)
    return out


def setup_inclined(L: int, H: float):
    """Fluid = {|s| <= H}, ``s = wrap(x - y)/sqrt(2)``; returns (sdf, fluid, s)."""
    x = torch.arange(L).view(1, L).expand(L, L).float()
    y = torch.arange(L).view(L, 1).expand(L, L).float()
    d = torch.remainder(x - y + L // 2, L) - L // 2
    s = d / math.sqrt(2.0)
    fluid = s.abs() <= H
    sdf = H - s.abs()
    sdf = torch.where(fluid, sdf.clamp(min=1e-6), -sdf.abs().clamp(min=1e-6))
    return sdf, fluid, s


def _profile_metrics(u: torch.Tensor, s: torch.Tensor, fluid: torch.Tensor, ref_H: float,
                     g: float, nu: float, u_max_ana: float, nb: int) -> tuple[float, list]:
    """Binned profile error against a channel of half-width ``ref_H``."""
    edges = torch.linspace(-ref_H, ref_H, nb + 1)
    sb = s[fluid]
    ub = u[fluid]
    errs, bins = [], []
    for k in range(nb):
        lo, hi = float(edges[k]), float(edges[k + 1])
        m = (sb >= lo) & (sb < hi)
        if int(m.sum()) == 0:
            continue
        sc = 0.5 * (lo + hi)
        u_ana = g * (ref_H ** 2 - sc ** 2) / (2.0 * nu)
        if abs(u_ana) < 0.2 * u_max_ana:
            continue  # near-wall bins are dominated by the wall offset itself
        um = float(ub[m].mean().item())
        errs.append(abs(um - u_ana) / u_max_ana)
        bins.append({"s": sc, "u_num": um, "u_ana": u_ana})
    theta = 100.0 * max(errs) if errs else float("nan")
    return theta, bins


def run_case(L: int, H: float, wall: str, steps: int | None = None,
             acc_steps: int = 400) -> dict:
    nu = (TAU - 0.5) / 3.0
    g = 2.0 * nu * U_MAX_TARGET / H ** 2
    sdf, fluid, s = setup_inclined(L, H)
    solid = ~fluid
    n_steps = steps if steps is not None else int(round(25.0 * H * H))
    acc_steps = acc_steps if H <= 11.0 else max(acc_steps, 600)

    ex = torch.zeros_like(s)
    ey = torch.zeros_like(s)
    ex[fluid] = g / math.sqrt(2.0)
    ey[fluid] = g / math.sqrt(2.0)
    src = force_source_2d(ex, ey)

    rho0 = torch.ones((L, L))
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

    report = None

    def step_once(f):
        nonlocal report
        f_pre = collide_bgk(f, TAU) + src
        f_st = stream(f_pre)
        if wall == "halfway":
            return bounce_back_cells(f_st, solid)
        out, rep = interpolated_bounce_back_2d(f_st, f_pre, sdf, solid, return_report=True)
        report = rep
        return out

    period_len = L * math.sqrt(2.0)
    checkpoints = {int(0.5 * n_steps), int(0.75 * n_steps)}
    q_hist = []
    t0 = time.time()
    for n in range(1, n_steps + 1):
        f = step_once(f)
        if n in checkpoints:
            ux, uy = macroscopic(f)[1], macroscopic(f)[2]
            u_ax = (ux + uy) / math.sqrt(2.0)
            q_hist.append({"step": n, "q_unit": float(u_ax[fluid].sum().item()) / period_len})

    acc = torch.zeros((L, L))
    for _ in range(acc_steps):
        _, ux, uy = macroscopic(f)
        acc += (ux + uy) / math.sqrt(2.0)
        f = step_once(f)
    u = acc / acc_steps
    wall_time = time.time() - t0

    q_unit = float(u[fluid].sum().item()) / period_len
    q_hist.append({"step": n_steps + acc_steps, "q_unit": q_unit})
    drift = abs(q_hist[-1]["q_unit"] - q_hist[0]["q_unit"]) / abs(q_hist[-1]["q_unit"])

    H_eff = (3.0 * nu * q_unit / (2.0 * g)) ** (1.0 / 3.0)
    n_fluid = int(fluid.sum().item())
    H_vol = n_fluid / (2.0 * L * math.sqrt(2.0))
    u_max = float(u[fluid].max().item())
    u_max_ana = g * H ** 2 / (2.0 * nu)
    nb = max(int(H), 6)

    theta_nom, bins_nom = _profile_metrics(u, s, fluid, H, g, nu, u_max_ana, nb)
    theta_vol, bins_vol = _profile_metrics(u, s, fluid, H_vol, g, nu, u_max_ana, nb)

    row = {
        "case": "inclined45",
        "wall": wall,
        "L": L,
        "H": H,
        "nu": nu,
        "tau": TAU,
        "g": g,
        "steps": n_steps,
        "acc_steps": acc_steps,
        "n_fluid": n_fluid,
        "H_vol": H_vol,
        "delta_geom": H_vol - H,
        "q_unit": q_unit,
        "Q_over_Qana": q_unit / (2.0 * g * H ** 3 / (3.0 * nu)),
        "Q_over_Qana_vol": q_unit / (2.0 * g * H_vol ** 3 / (3.0 * nu)),
        "H_eff": H_eff,
        "delta_eff": H_eff - H,
        "delta_hydro": (H_eff - H) - (H_vol - H),
        "theta_nom_pct": theta_nom,
        "theta_vol_pct": theta_vol,
        "u_max": u_max,
        "u_max_ana": u_max_ana,
        "u_max_rel_err_pct": 100.0 * (u_max - u_max_ana) / u_max_ana,
        "q_hist": q_hist,
        "q_drift_rel": drift,
        "profile": bins_nom,
        "profile_vol": bins_vol,
        "wall_time_s": wall_time,
        "bfl_report": None if report is None else {
            "cut_links": report.cut_links,
            "interpolated_links": report.interpolated_links,
            "fallback_links": report.fallback_links,
            "q_min": report.q_min,
            "q_max": report.q_max,
            "q_mean": report.q_mean,
        },
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "artifacts"))
    ap.add_argument("--walls", default="halfway,interp")
    ap.add_argument("--grids", default=",".join(f"{L}:{H:g}" for L, H in DEFAULT_GRIDS))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    grids = []
    for tok in args.grids.split(","):
        a, b = tok.split(":")
        grids.append((int(a), float(b)))
    walls = args.walls.split(",")

    rows = []
    for L, H in grids:
        for w in walls:
            r = run_case(L, H, w)
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items()
                              if k not in ("profile", "profile_vol", "q_hist")}), flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "ladder.json").write_text(json.dumps({"rows": rows}, indent=1))
    print(f"wrote {out / 'ladder.json'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()