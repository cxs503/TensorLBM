#!/usr/bin/env python3
"""Integrated arm: dam-break column collapse (Martin & Moyce 1952) per I/I rule.

    python run_dambreak_xval.py --rule pair_symmetric --cells 32 --out artifacts/db_pair_a32.json

Z(T) = x_front/a with T = t sqrt(g/h0); error against the 14 digitised M&M
points (piecewise-linear, no extrapolation).  Every number is computed here.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))

MM = json.loads((REPO / "src" / "artifacts" / "martin_moyce_1952_dam_break.json").read_text())
MM_T = [d["t_star"] for d in MM["data"]]
MM_Z = [d["X_star"] for d in MM["data"]]


def mm_interp(t):
    if t < MM_T[0] or t > MM_T[-1]:
        return None
    import bisect
    i = bisect.bisect_right(MM_T, t) - 1
    i = min(max(i, 0), len(MM_T) - 2)
    w = (t - MM_T[i]) / (MM_T[i + 1] - MM_T[i])
    return MM_Z[i] + w * (MM_Z[i + 1] - MM_Z[i])


def run(rule, a, tau=1.0, g=5e-5, t_max=2.96, sample_dt=0.25, enabled=True):
    torch.set_num_threads(1)
    from tensorlbm.d2q9 import equilibrium
    from tensorlbm.free_surface_lbm_2d import (
        INTERFACE, LIQUID, free_surface_step_2d, init_fill_rectangular_2d,
        init_flags_from_fill_2d, total_liquid_inventory_2d,
    )
    h0, nx = int(2 * a), int(5 * a)
    ny = h0 + a
    fill, solid = init_fill_rectangular_2d(ny, nx, float(a + 1), float(h0 + 1), "cpu")
    flags = init_flags_from_fill_2d(fill, solid)
    f = equilibrium(torch.ones_like(fill), torch.zeros_like(fill), torch.zeros_like(fill))
    mass = fill.clone()
    inv0 = total_liquid_inventory_2d(f, mass, flags)
    dtT = math.sqrt(g / h0)
    n_steps = int(math.ceil(t_max / dtT))
    every = max(1, int(round(sample_dt / dtT)))
    series, ledger = [], {}
    net_ii = 0.0
    abs_ii = 0.0
    dead = 0
    live = 0
    t0 = time.time()
    for step in range(n_steps + 1):
        if step % every == 0:
            fluid = (flags == LIQUID) | (flags == INTERFACE)
            cols = torch.nonzero(fluid.any(dim=0)).flatten()
            front = int(cols.max()) if cols.numel() else 0
            series.append({"step": step, "T": step * dtT, "Z": front / a,
                           "Z_mm": mm_interp(step * dtT)})
        if step == n_steps:
            break
        fill = torch.where(flags == LIQUID, torch.ones_like(fill),
                           torch.where(flags == INTERFACE, mass.clamp(0.0, 1.0), torch.zeros_like(fill)))
        ledger.clear()
        f, mass, flags = free_surface_step_2d(
            f, fill, flags, solid, tau=tau, gy=-g, mass=mass, return_mass=True,
            ledger=ledger, interface_exchange_rule=rule)
        if "exchange_interface_delta" in ledger:
            net_ii += ledger["exchange_interface_delta"]
            abs_ii += abs(ledger["exchange_interface_delta"])
            dead += ledger["exchange_ii_dead_links"]
            live += ledger["exchange_ii_live_links"]
        if not torch.isfinite(f).all():
            return {"rule": rule, "a": a, "status": "nan", "nan_at_step": step}
    inv1 = total_liquid_inventory_2d(f, mass, flags)
    errs = []
    for s in series:
        if s["Z_mm"] is not None:
            s["err_pct"] = 100.0 * (s["Z"] - s["Z_mm"]) / s["Z_mm"]
            errs.append(abs(s["err_pct"]))
    return {
        "rule": rule, "a": a, "tau": tau, "g": g, "h0": h0, "nx": nx, "ny": ny,
        "nu": (tau - 0.5) / 3.0, "steps": n_steps, "status": "ok",
        "Z_err_max_pct": max(errs) if errs else None,
        "Z_err_mean_pct": sum(errs) / len(errs) if errs else None,
        "inventory_drift_pct": 100.0 * (inv1 - inv0) / inv0,
        "ii_net_total": net_ii, "ii_abs_total": abs_ii,
        "ii_dead_links": dead, "ii_live_links": live,
        "wall_seconds": round(time.time() - t0, 1),
        "series": series,
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rule", default="repo")
    ap.add_argument("--cells", type=int, default=32)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--no-exchange", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run(args.rule, args.cells, tau=args.tau, enabled=not args.no_exchange)
    out = Path(args.out) if args.out else HERE / "artifacts" / f"db_{args.rule}_a{args.cells}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"{res['rule']:15s} a={res['a']} Z_err_max={res.get('Z_err_max_pct')} "
          f"drift={res.get('inventory_drift_pct')} ii_net={res.get('ii_net_total')} "
          f"dead={res.get('ii_dead_links')} wall={res.get('wall_seconds')}s -> {out}")


if __name__ == "__main__":
    main()
