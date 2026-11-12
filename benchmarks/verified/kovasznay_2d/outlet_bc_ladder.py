#!/usr/bin/env python3
"""Kovasznay (1948) Re=40 -- outlet-BC ladder: zero-gradient vs non-reflecting (NSCBC).

Why this file exists
--------------------
* `docs/CASE-LBM-KOVASZNAY-EXACT_independent_audit.md` (m16861) measured that
  **85.2 % of the residual `u` error of the production case comes from the outlet
  plane** (zero-gradient Neumann), and that the `v` error does not converge.
* `tensorlbm.boundaries.nscbc_outlet_2d` (Poinsot-Lele / Thompson characteristic
  outflow) was re-derived and validated on a 1-D acoustic pulse
  (`benchmarks/verified/nscbc_outlet_reflection`, 2026-11-12), but it had **never
  been applied to an outflow that carries vorticity**, i.e. the 2-D steady
  analytic Kovasznay solution used by `CASE-LBM-KOVASZNAY-EXACT`.

This script closes that gap: the same steady analytic solution, the same
library kernels, the same domain, with only the outlet closure varied.

Ladder entries
--------------
``zerograd``  production closure, ``zero_gradient_outlet_2d`` (first order)
``nscbc*``    ``nscbc_outlet_2d`` at several pressure-relaxation rates ``sigma``
``nxextrap``  ``nonequilibrium_extrapolation_outlet_2d`` -- reproduced here only
              to keep the documented honest-FAIL (diverges, see the warning in
              ``src/tensorlbm/boundary/boundaries.py``) machine-checkable

Usage
-----
    PYTHONPATH=src python3 benchmarks/verified/kovasznay_2d/outlet_bc_ladder.py \
        --grids 32 64 --steps 10000 --out artifacts/m16974/outlet_bc_ladder.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorlbm.boundaries import (  # noqa: E402
    nonequilibrium_extrapolation_outlet_2d,
    nscbc_outlet_2d,
    zero_gradient_outlet_2d,
    zou_he_inlet_velocity,
)
from tensorlbm.d2q9 import equilibrium, macroscopic  # noqa: E402
from tensorlbm.solver import collide_bgk, stream  # noqa: E402

CS2 = 1.0 / 3.0
TWOPI = 2.0 * math.pi


def kov_lambda(re: float) -> float:
    return re / 2.0 - math.sqrt(re * re / 4.0 + 4.0 * math.pi * math.pi)


def analytic(nx: int, ny: int, u0: float, lam: float, xmax: float):
    y = torch.arange(ny, dtype=torch.float64).view(-1, 1)
    x = torch.arange(nx, dtype=torch.float64).view(1, -1)
    xp = x / nx * xmax
    yp = y / ny
    e = torch.exp(lam * xp)
    u = u0 * (1.0 - e * torch.cos(TWOPI * yp)) * torch.ones((ny, nx), dtype=torch.float64)
    v = u0 * (lam / TWOPI) * e * torch.sin(TWOPI * yp) * torch.ones((ny, nx), dtype=torch.float64)
    return u.contiguous(), v.contiguous()


def make_outlet(name: str, sigma: float, length: int):
    if name == "zerograd":
        return lambda f: zero_gradient_outlet_2d(f)
    if name == "nxextrap":
        return lambda f: nonequilibrium_extrapolation_outlet_2d(f)
    if name == "nscbc":
        return lambda f: nscbc_outlet_2d(f, rho_target=1.0, sigma=sigma, length=float(length))
    raise SystemExit(f"unknown outlet {name!r}")


def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum() / max((b**2).sum(), 1e-300)))


def masked_max_rel(a: np.ndarray, b: np.ndarray, thr: float) -> float:
    m = np.abs(b) > thr
    if not m.any():
        return float("nan")
    return float(np.max(np.abs(a[m] - b[m])) / np.max(np.abs(b[m])))


def run_one(ny, steps, outlet, sigma, re, u0, xmax, dtype=torch.float64):
    nx = int(round(ny * xmax))
    lam = kov_lambda(re)
    nu = u0 * ny / re
    tau = 0.5 + 3.0 * nu
    u_ana, v_ana = analytic(nx, ny, u0, lam, xmax)
    u_ana_np, v_ana_np = u_ana.numpy(), v_ana.numpy()
    f = equilibrium(torch.ones((ny, nx), dtype=dtype), u_ana, v_ana)
    m0 = float(f.sum())
    u_in, v_in = u_ana[:, 0].contiguous(), v_ana[:, 0].contiguous()
    wall = make_outlet(outlet, sigma, nx)

    def step(f):
        f = collide_bgk(f, tau)
        f = stream(f)
        f = wall(f)
        return zou_he_inlet_velocity(f, u_in, v_in)

    hist = []
    t0 = time.time()
    for k in range(1, steps + 1):
        f = step(f)
        if k % max(1, steps // 4) == 0 or k == steps:
            _, ux, uy = macroscopic(f)
            ux_np, uy_np = ux.numpy(), uy.numpy()
            hist.append({
                "step": k,
                "u_l2": rel_l2(ux_np, u_ana_np),
                "v_l2": rel_l2(uy_np, v_ana_np),
                "mass_drift_pct": 100.0 * (float(f.sum()) - m0) / m0,
                "finite": bool(np.isfinite(ux_np).all() and np.isfinite(uy_np).all()),
            })
    _, ux, uy = macroscopic(f)
    ux_np, uy_np = ux.numpy(), uy.numpy()
    err_u = np.abs(ux_np - u_ana_np)
    n_out = max(1, int(round(0.1 * nx)))
    u0n = u0
    return {
        "ny": ny, "nx": nx, "steps": steps, "outlet": outlet, "sigma": sigma,
        "tau": tau, "nu": nu,
        "u_l2": rel_l2(ux_np, u_ana_np),
        "v_l2": rel_l2(uy_np, v_ana_np),
        "u_max_rel_pct": 100.0 * masked_max_rel(ux_np, u_ana_np, 0.1 * u0n),
        "v_max_rel_pct": 100.0 * masked_max_rel(uy_np, v_ana_np, 0.1 * np.max(np.abs(v_ana_np))),
        # share of the u error living in the last 10 % of columns (outlet region)
        "u_err_share_last10pct": float(err_u[:, -n_out:].sum() / max(err_u.sum(), 1e-300)),
        "u_err_profile": (err_u.max(axis=0) / u0n).tolist(),
        "wall_time_s": time.time() - t0,
        "history": hist,
    }


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[32, 64])
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--steps-nxextrap", type=int, default=3000)
    ap.add_argument("--re", type=float, default=40.0)
    ap.add_argument("--u0", type=float, default=0.03)
    ap.add_argument("--xmax", type=float, default=3.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=str(REPO / "artifacts/m16974/outlet_bc_ladder.json"))
    ap.add_argument("--fig", default=str(REPO / "figures/kovasznay_outlet_bc_ladder.png"))
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records = []
    plan = [("zerograd", None)]
    for sigma in (0.05, 0.25, 0.5):
        plan.append(("nscbc", sigma))

    for ny in args.grids:
        for name, sigma in plan:
            r = run_one(ny, args.steps, name, sigma, args.re, args.u0, args.xmax)
            records.append(r)
            print(f"[{name}{'' if sigma is None else f' s={sigma}'}] ny={ny} "
                  f"u_l2={r['u_l2']:.5e} v_l2={r['v_l2']:.5e} "
                  f"u_mr={r['u_max_rel_pct']:.3f}% v_mr={r['v_max_rel_pct']:.3f}% "
                  f"mass={r['history'][-1]['mass_drift_pct']:+.3f}% "
                  f"outlet_share={100*r['u_err_share_last10pct']:.1f}% "
                  f"({r['wall_time_s']:.0f}s)", flush=True)
            out.write_text(json.dumps({"records": records}, indent=1))
        # honest-FAIL reproduction of the documented unstable closure (short arm)
        r = run_one(ny, args.steps_nxextrap, "nxextrap", None, args.re, args.u0, args.xmax)
        r["diverged"] = bool((not np.isfinite(r["u_l2"])) or r["u_l2"] > 1.0)
        records.append(r)
        print(f"[nxextrap] ny={ny} u_l2={r['u_l2']:.5e} diverged={r['diverged']} "
              f"({r['wall_time_s']:.0f}s)", flush=True)
        out.write_text(json.dumps({"records": records}, indent=1))

    fig = Path(args.fig)
    fig.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig_, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        ax = axes[0]
        for name, sigma in plan:
            sel = [r for r in records if r["outlet"] == name and r["sigma"] == sigma]
            if not sel:
                continue
            lab = name if sigma is None else f"nscbc σ={sigma}"
            ax.semilogy([r["ny"] for r in sel], [r["u_l2"] for r in sel], "o-", label=f"{lab} u")
            ax.semilogy([r["ny"] for r in sel], [r["v_l2"] for r in sel], "s--", label=f"{lab} v")
        ax.set_xlabel("ny (nx = 3 ny)"); ax.set_ylabel("L2 relative error")
        ax.set_title("Kovasznay Re=40: outlet closure vs grid"); ax.grid(alpha=.3)
        ax.legend(fontsize=7, ncol=2)
        ax = axes[1]
        for r in records:
            if r["ny"] != args.grids[-1]:
                continue
            lab = r["outlet"] if r["sigma"] is None else f"nscbc σ={r['sigma']}"
            ax.plot(np.arange(r["nx"]) / r["nx"] * args.xmax, r["u_err_profile"], label=lab)
        ax.set_xlabel("x' (domain lengths)"); ax.set_ylabel("max_y |u_num - u_ana| / U0")
        ax.set_title(f"u error profile along x (ny={args.grids[-1]})")
        ax.grid(alpha=.3); ax.legend(fontsize=7)
        fig_.tight_layout(); fig_.savefig(fig, dpi=150)
        print(f"figure -> {fig}")
    except Exception as exc:  # pragma: no cover - figure is optional
        print(f"figure skipped: {exc}")

    print(f"json -> {out}")
    return {"records": records}


if __name__ == "__main__":
    main()