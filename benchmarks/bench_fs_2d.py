#!/usr/bin/env python3
"""2D Free-Surface Dam-Break Benchmark -- Martin and Moyce (1952) comparison.

Domain: ny=200, nx=600; liquid column width=120, height=198.
500 timesteps, tau=0.8, gy=-5e-5.

Time convention (FIXED 2026-09-29):  T = t * sqrt(2 g / a),  Z = front_x / a.
The sqrt(2 g / a) form is the Martin & Moyce (1952) definition and is confirmed
verbatim by Lethe's post-processing (examples/multiphysics/dam-break/
dam-break-2d.py:  time_list = [x * ((2*g/L1)**0.5) for x in time_list]);
the repository kernel already uses it (dam_break.py:299).  This script
previously used T = t * sqrt(g/a) -- a factor sqrt(2) too small -- and an
inconsistent coarse table.  Both are fixed here.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorlbm.d2q9 import equilibrium
from tensorlbm.free_surface_lbm_2d import (
    GAS,
    INTERFACE,
    LIQUID,
    SOLID,
    free_surface_step_2d,
    init_fill_rectangular_2d,
    init_flags_from_fill_2d,
)

# ---------- Martin and Moyce (1952) reference ----------
# CORRECT digitisation (Lethe x_exp/y_exp; identical to the K&O/Trixi-family
# digitisation), valid for T = t*sqrt(2 g / a):
REF_T = [0.00, 0.41, 0.84, 1.19, 1.43, 1.63, 1.82, 1.97, 2.20,
         2.32, 2.50, 2.64, 2.82, 2.96]
REF_Z = [1.00, 1.11, 1.23, 1.44, 1.67, 1.89, 2.11, 2.33, 2.56,
         2.78, 3.00, 3.22, 3.44, 3.67]
# Historical (WRONG) table, kept for reference/comparison only.  It was a coarse
# digitisation that is only meaningful on the old t*sqrt(g/a) axis; it mapped
# M&M's T=3 value (2.7) onto T=2 on that axis:
#   REF_T_old = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
#   REF_Z_old = [1.0, 1.1, 1.4, 1.8, 2.2, 2.7, 3.1, 3.5, 3.8, 4.1]

# ---------- Simulation parameters ----------
NY, NX = 200, 600
COL_WIDTH = 120.0
COL_HEIGHT = float(NY - 2)  # domain height minus top/bottom walls
TAU = 0.8
GY = -5e-5
RHO_LIQ = 1.0
RHO_GAS = 0.01
N_STEPS = 500
DEVICE = "sdaa" if torch.sdaa.is_available() else "cpu"

A = COL_WIDTH
G_ABS = abs(GY)
# M&M convention: T = t*sqrt(2 g / a)  (the factor sqrt2 is the 2026-09-29 fix)
SQRT_GA = math.sqrt(2.0 * G_ABS / A)

# Checkpoints on the CORRECT T = t*sqrt(2g/a) axis (same as REF_T).
T_TARGETS = list(REF_T)


def find_front_x(flags, _fill):
    """Rightmost column (0-based) with liquid or interface."""
    active = (flags[1:-1, :] == LIQUID) | (flags[1:-1, :] == INTERFACE)
    cols = active.any(dim=0).nonzero(as_tuple=False)
    return 0 if cols.numel() == 0 else int(cols[-1].item())


def linear_interp(t_target, t_vals, z_vals):
    if t_target <= t_vals[0]:
        return z_vals[0]
    if t_target >= t_vals[-1]:
        return z_vals[-1]
    for i in range(len(t_vals) - 1):
        if t_vals[i] <= t_target <= t_vals[i + 1]:
            frac = (t_target - t_vals[i]) / (t_vals[i + 1] - t_vals[i])
            return z_vals[i] + frac * (z_vals[i + 1] - z_vals[i])
    return z_vals[-1]


def main():
    print("=" * 70)
    print("  2D FREE-SURFACE DAM-BREAK BENCHMARK")
    print("  Martin and Moyce (1952) comparison")
    print("=" * 70)
    print(f"  Device    : {DEVICE}")
    print(f"  Domain    : {NX} x {NY}")
    print(f"  Column    : w={COL_WIDTH:.0f}  h={COL_HEIGHT:.0f}")
    print(f"  Physics   : tau={TAU}  gy={GY:.1e}  rho_L={RHO_LIQ}  rho_G={RHO_GAS}")
    print(f"  Steps     : {N_STEPS}")
    print(f"  sqrt(2g/a): {SQRT_GA:.6e}   (M&M axis T = t*sqrt(2g/a))")
    print(f"  T_max     : {N_STEPS * SQRT_GA:.4f}")
    print()

    # ---- Init ----
    fill, solid = init_fill_rectangular_2d(NY, NX, COL_WIDTH, COL_HEIGHT, DEVICE)
    flags = init_flags_from_fill_2d(fill, solid)

    rho0 = torch.where(
        flags == LIQUID, torch.full_like(fill, RHO_LIQ), torch.full_like(fill, RHO_GAS)
    )
    f = equilibrium(rho0, torch.zeros_like(fill), torch.zeros_like(fill))

    print(
        f"  Init cells: L={int((flags == LIQUID).sum())}  "
        f"I={int((flags == INTERFACE).sum())}  "
        f"G={int((flags == GAS).sum())}  S={int((flags == SOLID).sum())}"
    )
    print(f"  Init front : x={find_front_x(flags, fill)}")
    print()

    # ---- Run ----
    T_vals, Z_vals = [0.0], [find_front_x(flags, fill) / A]
    report_every = 50

    for step in range(1, N_STEPS + 1):
        f, fill, flags = free_surface_step_2d(
            f,
            fill,
            flags,
            solid,
            tau=TAU,
            gy=GY,
            rho_liquid=RHO_LIQ,
            rho_gas=RHO_GAS,
        )
        if torch.isnan(f).any():
            print(f"  STOP: NaN at step {step}")
            break

        T = step * SQRT_GA
        Z = find_front_x(flags, fill) / A
        T_vals.append(T)
        Z_vals.append(Z)

        if step % report_every == 0:
            print(f"  step={step:4d}  T={T:.4f}  Z={Z:.4f}  front_x={int(Z * A)}")

    # ---- Report at target T ----
    print()
    print("=" * 70)
    print("  COMPARISON  (T_max = {:.4f}); values beyond T_max extrapolated".format(T_vals[-1]))
    print("=" * 70)
    hdr = f"  {'T':>6s}  {'Z_sim':>8s}  {'Z_ref':>8s}  {'|err|':>8s}  {'note':s}"
    print(hdr)
    print("  " + "-" * 55)

    errors = []
    for Tt, Zr in zip(T_TARGETS, REF_Z):
        Zs = linear_interp(Tt, T_vals, Z_vals)
        err = abs(Zs - Zr)
        note = "" if Tt <= T_vals[-1] else "(extrapolated)"
        errors.append(err)
        print(f"  {Tt:6.2f}  {Zs:8.4f}  {Zr:8.4f}  {err:8.4f}  {note}")

    valid = [e for e in errors if not math.isnan(e)]
    mae = sum(valid) / len(valid)
    rmse = math.sqrt(sum(e * e for e in valid) / len(valid))
    print()
    print(f"  MAE  = {mae:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  Max  = {max(valid):.4f}")
    print()

    # Front evolution summary
    print("=" * 70)
    print("  FRONT EVOLUTION (every 50 steps, within simulated range)")
    print("=" * 70)
    print(f"  {'step':>5s}  {'T':>10s}  {'Z':>10s}  {'front_x':>8s}")
    print("  " + "-" * 45)
    for i in range(0, len(T_vals), 50):
        print(f"  {i:5d}  {T_vals[i]:10.4f}  {Z_vals[i]:10.4f}  {int(Z_vals[i] * A):8d}")
    # Always print final
    print(
        f"  {len(T_vals) - 1:5d}  {T_vals[-1]:10.4f}  {Z_vals[-1]:10.4f}  {int(Z_vals[-1] * A):8d}"
    )

    print()
    print(f"  Note: on the M&M axis T = t*sqrt(2g/a) this {N_STEPS}-step run only")
    print(f"        reaches T_max = {N_STEPS * SQRT_GA:.3f}; checkpoints beyond that are")
    print("        (extrapolated) and are indicative only.  The full deep-time")
    print("        comparison (T = 1, 2, 2.96) lives in")
    print("        benchmarks/pending/dam_break_sc/run.py.")
    print("=" * 70)


if __name__ == "__main__":
    main()
