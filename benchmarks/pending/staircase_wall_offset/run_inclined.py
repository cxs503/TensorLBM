#!/usr/bin/env python3
"""Staircase/inclined wall effective wall offset: half-way vs interpolated bounce-back.

Cases (all analytic, body-force driven, fully developed, no free parameters that
could be tuned against the reference):

``inclined``  45-degree plane-Poiseuille channel in a doubly periodic box.  The
              wall function ``s = (x - y)/sqrt(2)`` is invariant under the
              lattice period, so ``|s| <= H`` is an exact periodic channel whose
              walls are a staircase.  Analytic solution (half-width ``H``,
              body force ``g`` per unit mass along ``(1,1)/sqrt(2)``, ``nu``)::

                  u(s) = g (H^2 - s^2) / (2 nu),      Q = 2 g H^3 / (3 nu)

``pipe3d``    circular pipe, staircase cylinder wall, periodic in x.  Analytic::

                  u(r) = g (R^2 - r^2) / (4 nu),      Q = pi g R^4 / (8 nu)

Two boundary formats are compared on the *same* solid mask:

``halfway``   the repository production operator (``bounce_back_cells`` /
              ``bounce_back_cells_3d``): no-slip plane on the staircase face.

``interp``    :func:`tensorlbm.boundary.interpolated_bounce_back.interpolated_bounce_back_2d`
              / ``_3d``: 2-point Bouzidi closure with ``q`` taken from the
              analytic signed distance function, i.e. the no-slip plane is the
              physical wall, not the staircase face.

Reported metric is the *effective wall offset* ``delta_eff``, obtained from a
single integral observable (the volume flux ``Q``) and no profile fitting::

    H_eff = (3 nu Q / (2 g))^(1/3)          delta_eff = H_eff - H      (inclined)
    R_eff = (8 nu Q / pi g)^(1/4)            delta_eff = R_eff - R      (pipe3d)

together with the profile error against the *nominal* wall position, binned over
the wall-normal coordinate:

    theta = max_bin |u_num - u_ana(H_nominal)| / u_max

Usage::

    run_inclined.py --case inclined --all --out artifacts/
    run_inclined.py --case pipe3d   --all --out artifacts/
    run_inclined.py --case inclined --L 64 --wall halfway
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tensorlbm.boundary.interpolated_bounce_back import (  # noqa: E402
    interpolated_bounce_back_2d,
    interpolated_bounce_back_3d,
)
from tensorlbm.boundaries import bounce_back_cells  # noqa: E402
from tensorlbm.boundaries3d import bounce_back_cells_3d  # noqa: E402
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

CS2 = 1.0 / 3.0


def _force_source_2d(fx: torch.Tensor, fy: torch.Tensor) -> torch.Tensor:
    """Momentum source s_i = 3 w_i (c_i . F); sum_i s_i c_i = F exactly, sum_i s_i = 0."""
    from tensorlbm.d2q9 import C as C9
    from tensorlbm.d2q9 import W as W9

    out = torch.zeros((9,) + fx.shape)
    for i in range(9):
        out[i] = 3.0 * W9[i] * (C9[i, 0] * fx + C9[i, 1] * fy)
    return out


def _force_source_3d(fx: torch.Tensor) -> torch.Tensor:
    from tensorlbm.d3q19 import C as C19
    from tensorlbm.d3q19 import W as W19

    out = torch.zeros((19,) + fx.shape)
    for i in range(19):
        out[i] = 3.0 * W19[i] * (C19[i, 0] * fx)
    return out


def _stat(title: str, rows: list[dict]) -> str:
    lines = [f"### {title}", "", "| case | H/R | nu | delta_eff (cells) | theta (%) | Q/Q_ana |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['case']} | {r['H']} | {r['nu']:.4g} | {r['delta_eff']:+.4f} | "
            f"{r['theta_pct']:.3f} | {r['Q_over_Qana']:.5f} |"
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# 45-degree inclined channel (D2Q9, doubly periodic)
# --------------------------------------------------------------------------- #
def setup_inclined(L: int, H: float):
    """Fluid = {|s| <= H}, s = (x - y)/sqrt(2); returns (sdf, fluid, s)."""
    x = torch.arange(L).view(1, L).expand(L, L).float()   # rows = y
    y = torch.arange(L).view(L, 1).expand(L, L).float()
    # ``x - y`` must be wrapped: the doubly periodic box makes the wall-normal
    # coordinate periodic with period L, so the channel is |wrap(x - y)| <= H
    # (using the raw difference would select a finite corner strip, not a channel).
    d = torch.remainder(x - y + L // 2, L) - L // 2
    s = d / math.sqrt(2.0)
    fluid = s.abs() <= H
    sdf = H - s.abs()
    sdf = torch.where(fluid, sdf.clamp(min=1e-6), -sdf.abs().clamp(min=1e-6))
    return sdf, fluid, s


def run_inclined(
    L: int,
    H: float,
    tau: float,
    u_max_target: float,
    wall: str,
    steps: int,
    acc_steps: int = 400,
) -> dict:
    nu = (tau - 0.5) / 3.0
    g = 2.0 * nu * u_max_target / H**2
    sdf, fluid, s = setup_inclined(L, H)
    solid = ~fluid
    ex = torch.zeros_like(s)
    ey = torch.zeros_like(s)
    ex[fluid] = g / math.sqrt(2.0)
    ey[fluid] = g / math.sqrt(2.0)

    rho0 = torch.ones((L, L))
    f = equilibrium(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0))

    def step_once(f):
        f_pre = collide_bgk(f, tau) + src
        f_st = stream(f_pre)
        if wall == "halfway":
            return bounce_back_cells(f_st, solid)
        return interpolated_bounce_back_2d(f_st, f_pre, sdf, solid)

    src = _force_source_2d(ex, ey)
    t0 = time.time()
    for _ in range(steps):
        f = step_once(f)

    # time-averaged field over acc_steps to remove residual unsteadiness
    acc = torch.zeros((L, L))
    for _ in range(acc_steps):
        _, ux, uy = macroscopic(f)
        acc += (ux + uy) / math.sqrt(2.0)   # axial velocity along (1,1)/sqrt(2)
        f = step_once(f)
    u = acc / acc_steps

    # --- integral observable: flux per unit channel length ------------------
    q_total = float(u[fluid].sum().item())
    period_len = L * math.sqrt(2.0)          # one period along the axis
    q_unit = q_total / period_len
    H_eff = (3.0 * nu * q_unit / (2.0 * g)) ** (1.0 / 3.0)
    u_max = float(u[fluid].max().item())
    u_max_ana = g * H**2 / (2.0 * nu)

    # --- binned profile error against the nominal wall ----------------------
    nb = max(int(H), 6)
    edges = torch.linspace(-H, H, nb + 1)
    sb = s[fluid]
    ub = u[fluid]
    errs = []
    bins = []
    for k in range(nb):
        lo, hi = float(edges[k]), float(edges[k + 1])
        m = (sb >= lo) & (sb < hi)
        if int(m.sum()) == 0:
            continue
        sc = 0.5 * (lo + hi)
        u_ana = g * (H**2 - sc**2) / (2.0 * nu)
        if abs(u_ana) < 0.2 * u_max_ana:
            continue          # near-wall bins are dominated by the offset itself
        um = float(ub[m].mean().item())
        errs.append(abs(um - u_ana) / u_max_ana)
        bins.append({"s": sc, "u_num": um, "u_ana": u_ana})
    theta = 100.0 * max(errs) if errs else float("nan")
    return {
        "case": "inclined45",
        "wall": wall,
        "L": L,
        "H": H,
        "nu": nu,
        "tau": tau,
        "g": g,
        "steps": steps,
        "q_unit": q_unit,
        "Q_over_Qana": q_unit / (2.0 * g * H**3 / (3.0 * nu)),
        "delta_eff": H_eff - H,
        "theta_pct": theta,
        "u_max": u_max,
        "u_max_ana": u_max_ana,
        "profile": bins,
        "wall_time_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# 3D staircase pipe (D3Q19, periodic in x, body force along x)
# --------------------------------------------------------------------------- #
def setup_pipe(nx: int, R: float, pad: int = 3):
    n = int(math.ceil(2 * R)) + 2 * pad
    z = torch.arange(n).view(n, 1).expand(n, n).float()
    y = torch.arange(n).view(1, n).expand(n, n).float()
    yc = (n - 1) / 2.0
    r = torch.sqrt((y - yc) ** 2 + (z - yc) ** 2)
    fluid2d = r <= R
    sdf2d = R - r
    sdf2d = torch.where(fluid2d, sdf2d.clamp(min=1e-6), -sdf2d.abs().clamp(min=1e-6))
    fluid = fluid2d.unsqueeze(-1).expand(n, n, nx).contiguous()
    sdf = sdf2d.unsqueeze(-1).expand(n, n, nx).contiguous()
    solid = ~fluid
    return sdf, fluid, solid, r


def run_pipe(
    nx: int,
    R: float,
    tau: float,
    u_max_target: float,
    wall: str,
    steps: int,
    acc_steps: int = 200,
) -> dict:
    nu = (tau - 0.5) / 3.0
    g = 4.0 * nu * u_max_target / R**2
    sdf, fluid, solid, _r = setup_pipe(nx, R)
    nz, ny = sdf.shape[0], sdf.shape[1]
    fx = torch.zeros((nz, ny, nx))
    fx[fluid] = g

    rho0 = torch.ones((nz, ny, nx))
    f = equilibrium3d(rho0, torch.zeros_like(rho0), torch.zeros_like(rho0), torch.zeros_like(rho0))

    src = _force_source_3d(fx)

    def step_once(f):
        f_pre = collide_bgk3d(f, tau) + src
        f_st = stream3d(f_pre)
        if wall == "halfway":
            return bounce_back_cells_3d(f_st, solid)
        return interpolated_bounce_back_3d(f_st, f_pre, sdf, solid)

    t0 = time.time()
    for _ in range(steps):
        f = step_once(f)
    acc = torch.zeros((nz, ny, nx))
    for _ in range(acc_steps):
        _, ux, _, _ = macroscopic3d(f)
        acc += ux
        f = step_once(f)
    u = acc / acc_steps

    q_total = float(u[fluid].sum().item())
    q_unit = q_total / nx                  # per unit pipe length
    R_eff = (8.0 * nu * q_unit / (math.pi * g)) ** 0.25
    u_max_ana = g * R**2 / (4.0 * nu)
    u_max = float(u[fluid].max().item())
    # radial binned error inside the core (r < 0.8 R)
    rr = _r
    errs = []
    bins = []
    nb = max(int(R // 2), 4)
    edges = torch.linspace(0.0, 0.8 * R, nb + 1)
    for k in range(nb):
        lo, hi = float(edges[k]), float(edges[k + 1])
        m = (rr >= lo) & (rr < hi) & fluid[:, :, nx // 2]
        if int(m.sum()) == 0:
            continue
        rc = 0.5 * (lo + hi)
        u_ana = g * (R**2 - rc**2) / (4.0 * nu)
        um = float(u[:, :, nx // 2][m].mean().item())
        errs.append(abs(um - u_ana) / u_max_ana)
        bins.append({"r": rc, "u_num": um, "u_ana": u_ana})
    theta = 100.0 * max(errs) if errs else float("nan")
    return {
        "case": "pipe3d",
        "wall": wall,
        "nx": nx,
        "R": R,
        "nu": nu,
        "tau": tau,
        "g": g,
        "steps": steps,
        "q_unit": q_unit,
        "Q_over_Qana": q_unit / (math.pi * g * R**4 / (8.0 * nu)),
        "delta_eff": R_eff - R,
        "theta_pct": theta,
        "u_max": u_max,
        "u_max_ana": u_max_ana,
        "profile": bins,
        "wall_time_s": time.time() - t0,
    }


def make_figures(rows: list[dict], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    inc = [r for r in rows if r["case"] == "inclined45"]
    if inc:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        for r in inc:
            if r["L"] == inc[0]["L"]:
                p = r["profile"]
                s = [b["s"] for b in p]
                ax.plot(s, [b["u_num"] for b in p], "o-", ms=3,
                        label=f"{r['wall']} (theta={r['theta_pct']:.2f}%)")
        p = inc[0]["profile"]
        ax.plot([b["s"] for b in p], [b["u_ana"] for b in p], "k--", lw=1.2, label="analytic")
        ax.set_xlabel("s = (x-y)/sqrt2  [lattice units]")
        ax.set_ylabel("axial velocity u")
        ax.set_title(f"45deg inclined channel, L={inc[0]['L']}, nominal H={inc[0]['H']}")
        ax.legend(fontsize=7)
        ax2 = axes[1]
        for w, mk in (("halfway", "s"), ("interp", "o")):
            pts = sorted([(r["H"], r["delta_eff"]) for r in inc if r["wall"] == w])
            if pts:
                ax2.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", label=w)
        ax2.axhline(0.0, color="k", lw=0.8, ls=":")
        ax2.set_xlabel("nominal half-width H")
        ax2.set_ylabel("delta_eff  [cells]")
        ax2.set_title("effective wall offset from flux")
        ax2.legend(fontsize=8)
        fig.tight_layout()
        name = out / "staircase_offset_inclined.png"
        fig.savefig(name, dpi=140)
        made.append(str(name))
    pipe = [r for r in rows if r["case"] == "pipe3d"]
    if pipe:
        fig, ax = plt.subplots(figsize=(5.4, 4.2))
        for w, mk in (("halfway", "s"), ("interp", "o")):
            pts = sorted([(r["R"], r["delta_eff"]) for r in pipe if r["wall"] == w])
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], mk + "-", label=w)
        ax.axhline(0.0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("nominal radius R")
        ax.set_ylabel("delta_eff  [cells]")
        ax.set_title("staircase pipe: effective wall offset")
        ax.legend(fontsize=8)
        fig.tight_layout()
        name = out / "staircase_offset_pipe3d.png"
        fig.savefig(name, dpi=140)
        made.append(str(name))
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["inclined", "pipe3d", "both"], default="both")
    ap.add_argument("--wall", choices=["halfway", "interp", "both"], default="both")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--H", type=float, default=12.0)
    ap.add_argument("--R", type=float, default=10.0)
    ap.add_argument("--nx", type=int, default=16)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--umax", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--all", action="store_true", help="run the two-resolution sweep")
    ap.add_argument("--out", default=str(Path(__file__).parent / "artifacts"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    walls = ["halfway", "interp"] if args.wall == "both" else [args.wall]
    rows: list[dict] = []

    if args.case in ("inclined", "both"):
        grids = [(64, 12.0), (96, 18.0)] if args.all else [(args.L, args.H)]
        for L, H in grids:
            for w in walls:
                r = run_inclined(L, H, args.tau, args.umax, w, args.steps)
                rows.append(r)
                print(json.dumps({k: v for k, v in r.items() if k != "profile"}))
    if args.case in ("pipe3d", "both"):
        radii = [(16, 10.0), (20, 14.0)] if args.all else [(args.nx, args.R)]
        for nx, R in radii:
            for w in walls:
                r = run_pipe(nx, R, args.tau, args.umax, w, args.steps, acc_steps=200)
                rows.append(r)
                print(json.dumps({k: v for k, v in r.items() if k != "profile"}))

    figs = make_figures(rows, out)
    payload = {
        "rows": rows,
        "figures": figs,
        "acceptance": {
            "theta_le_3pct_interp": all(r["theta_pct"] <= 3.0 for r in rows if r["wall"] == "interp"),
            "delta_eff_le_0p05_interp": all(abs(r["delta_eff"]) <= 0.05 for r in rows if r["wall"] == "interp"),
            "theta_le_3pct_halfway": all(r["theta_pct"] <= 3.0 for r in rows if r["wall"] == "halfway"),
        },
    }
    (out / "result.json").write_text(json.dumps(payload, indent=2))
    print(_stat("summary", rows))
    print("acceptance:", payload["acceptance"])


if __name__ == "__main__":
    main()