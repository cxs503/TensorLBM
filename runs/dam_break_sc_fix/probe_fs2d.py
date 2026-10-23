#!/usr/bin/env python3
"""Focused probe for the 2D free-surface (Körner) module on the dam-break_sc geometry.

Measures several front definitions so the M&M comparison is unambiguous, plus
mass-ledger / inventory conservation and the g=0 creep test.

Usage: PYTHONPATH=src python runs/dam_break_sc_fix/probe_fs2d.py --a 40 --g 2e-4
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("/root/TensorLBM_feat2/src")))

import torch

from tensorlbm.d2q9 import equilibrium
from tensorlbm.free_surface_lbm_2d import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    init_fill_rectangular_2d,
    init_flags_from_fill_2d,
    init_mass_from_fill_2d,
    koerner_step_2d,
    total_liquid_inventory_2d,
)

torch.set_num_threads(12)

# Classic M&M (1952) digitisation, the repository's corrected table
# (benchmarks/bench_fs_2d.py REF_T/REF_Z and dam_break_3d_mm).
MM = {1.0: 1.1, 2.0: 1.8, 3.0: 2.7}


def build(a, device="cpu"):
    nx = int(round(6.4 * a))
    ny = int(round(3.2 * a))
    fill, solid = init_fill_rectangular_2d(ny, nx, a, 2.0 * a, device)
    flags = init_flags_from_fill_2d(fill, solid)
    mass = init_mass_from_fill_2d(fill, flags, 1.0)
    active = (flags == LIQUID) | (flags == INTERFACE)
    rho0 = torch.where(active, torch.ones_like(fill), torch.zeros_like(fill))
    zero = torch.zeros_like(fill)
    f = equilibrium(rho0, zero, zero)
    f = torch.where(active.unsqueeze(0), f, torch.zeros_like(f))
    return f, mass, flags, solid, fill


def measure(f, mass, flags, a):
    rho = f.sum(dim=0)
    liq = flags == LIQUID
    iface = flags == INTERFACE
    fill = mass  # rho_liquid = 1.0
    wet_any = liq | iface
    wet50 = liq | (iface & (fill >= 0.5))

    def rightmost(mask):
        m = mask[1:-1, 1:-1]
        cols = m.any(dim=0).nonzero(as_tuple=True)[0]
        return float(cols.max().item()) + 1.0 if cols.numel() else 0.0

    def toe(mask):
        lo = 3
        hi = max(lo + 3, mask.shape[0] // 8)
        m = mask[lo:hi, 1:-1]
        cols = m.any(dim=0).nonzero(as_tuple=True)[0]
        return float(cols.max().item()) + 1.0 if cols.numel() else 0.0

    xs_any = rightmost(wet_any)
    xs50 = rightmost(wet50)
    xs_toe = toe(wet_any)
    left = wet50[1:-1, 1:5].any(dim=1).nonzero(as_tuple=True)[0]
    h_left = float(left.max().item()) + 1.0 if left.numel() else 0.0
    return {
        "X_any": xs_any / a,
        "X_fill50": xs50 / a,
        "X_toe": xs_toe / a,
        "H": h_left / (2.0 * a),
        "rho_min": float(rho.min().item()),
        "rho_max": float(rho.max().item()),
        "iface": int(iface.sum().item()),
        "liq": int(liq.sum().item()),
    }


def interp(recs, key, Tq):
    Ts = [r["T"] for r in recs]
    Vs = [r[key] for r in recs]
    if Tq <= Ts[0]:
        return Vs[0]
    if Tq >= Ts[-1]:
        return Vs[-1]
    for i in range(1, len(Ts)):
        if Ts[i] >= Tq:
            t0, t1, v0, v1 = Ts[i - 1], Ts[i], Vs[i - 1], Vs[i]
            return v0 + (v1 - v0) * (Tq - t0) / (t1 - t0)
    return Vs[-1]


def run(a, g, steps, rho_gas, tau, sample, device, creep=False):
    f, mass, flags, solid, fill = build(a, device)
    mass0 = float(mass.sum().item())
    inv0 = total_liquid_inventory_2d(f, mass, flags, 1.0)
    sq = math.sqrt(g / a) if g > 0 else 0.0
    recs = []
    t0 = time.perf_counter()
    worst_md = worst_id = 0.0
    for step in range(1, steps + 1):
        f, mass, flags = koerner_step_2d(
            f, mass, flags, solid, tau=tau, gy=-g,
            rho_liquid=1.0, rho_gas=rho_gas,
        )
        if step % sample == 0:
            md = abs(float(mass.sum().item()) - mass0) / mass0
            inv = total_liquid_inventory_2d(f, mass, flags, 1.0)
            idv = abs(inv - inv0) / inv0
            worst_md = max(worst_md, md)
            worst_id = max(worst_id, idv)
            m = measure(f, mass, flags, a)
            rec = {"step": step, "T": step * sq, "mass_drift": md, "inv_drift": idv, **m}
            recs.append(rec)
            print(
                f"  a={a:4.0f} st={step:6d} T={rec['T']:6.3f} "
                f"X_any={m['X_any']:.3f} X_50={m['X_fill50']:.3f} X_toe={m['X_toe']:.3f} "
                f"H={m['H']:.3f} if={m['iface']:5d} md={md:.1e} invd={idv:.1e}",
                flush=True,
            )
        if not math.isfinite(float(f.sum().item())):
            print(f"  a={a:4.0f} NaN at step {step}")
            break
    dt = time.perf_counter() - t0
    print(f"  a={a:4.0f} DONE {dt:.0f}s worst_md={worst_md:.2e} worst_invd={worst_id:.2e}")
    return recs, worst_md, worst_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", type=float, default=40.0)
    ap.add_argument("--g", type=float, default=2e-4)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--rho-gas", type=float, default=0.01)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    a = args.a
    sq = math.sqrt(args.g / a) if args.g > 0 else 0.0
    steps = args.steps or (int(math.ceil(3.2 / sq)) if sq > 0 else 400)
    out = Path("/root/TensorLBM_feat2/runs/dam_break_sc_fix")
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== FS2D dam a={a:.0f} g={args.g} rho_gas={args.rho_gas} tau={args.tau} steps={steps}")
    recs, wmd, wid = run(a, args.g, steps, args.rho_gas, args.tau, args.sample, args.device)
    res = {"a": a, "g": args.g, "rho_gas": args.rho_gas, "tau": args.tau,
           "steps": steps, "worst_mass_drift": wmd, "worst_inv_drift": wid,
           "series": recs}
    for Tq, Xr in MM.items():
        for key in ("X_any", "X_fill50", "X_toe"):
            xs = interp(recs, key, Tq)
            print(f"  CHECK T={Tq} {key}: X={xs:.4f} ref={Xr} err={100*(xs-Xr)/Xr:+.2f}%")
            res[f"T{Tq}_{key}"] = round(xs, 5)
            res[f"T{Tq}_{key}_err_pct"] = round(100 * (xs - Xr) / Xr, 3)
    (out / f"fs2d_a{int(a)}_rg{args.rho_gas}_tau{args.tau}.json").write_text(
        json.dumps(res, indent=2))


if __name__ == "__main__":
    main()