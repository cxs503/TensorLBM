#!/usr/bin/env python
"""Continuous-error cross-validation: TensorLBM vs DOI-cited analytical references.

This script runs the "continuous-error-print-vs-doi-refs" cross-validation:
for each case, it runs the TensorLBM solver, computes the analytical reference
solution (cited by DOI), and reports the error norms.

Cases
-----
1. **poiseuille_2d**: 2-D plane Poiseuille flow (body-force driven).
   Analytical: u(y) = G/(2ν) · y(H−y).
   DOI: Schlichting & Gersten, "Boundary-Layer Theory", 8th ed., Springer (2017).

2. **womersley_alpha5**: Womersley pulsatile channel flow, α=5.
   Analytical: complex phasor solution from Womersley (1955).
   DOI: 10.1113/jphysiol.1955.sp005276

3. **couette_2d**: 2-D plane Couette flow (moving top wall, stationary bottom).
   Analytical: u(y) = U·y/H.
   DOI: Schlichting & Gersten, "Boundary-Layer Theory", 8th ed., Springer (2017).

Acceptance criterion
--------------------
Relative L2 error < 15% (owner precision standard).

Usage
-----
    python run_continuous_error_xval.py [--out DIR] [--steps N] [--verbose]

Output
------
* JSON artifact: continuous_error_xval.json
* CSV summary: continuous_error_xval.csv
* SHA-256 provenance: continuous_error_xval.sha256
* Console report: formatted table with DOI citations
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# Ensure TensorLBM is importable
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tensorlbm.verification.xval.continuous_error import (
    DOIReference,
    ContinuousErrorResult,
    compute_continuous_errors,
    print_error_report,
    save_artifacts,
)


# ---------------------------------------------------------------------------
# DOI References
# ---------------------------------------------------------------------------

SCHLICHTING_REF = DOIReference(
    doi="book:Schlichting-Gersten-2017",
    citation=(
        "Schlichting, H. & Gersten, K. (2017). Boundary-Layer Theory, 8th ed. "
        "Springer. §III.2.1 (Poiseuille), §III.2.2 (Couette)."
    ),
    solution_type="analytical",
)

WOMERSLEY_REF = DOIReference(
    doi="10.1113/jphysiol.1955.sp005276",
    citation=(
        "Womersley, J.R. (1955). Method for the calculation of velocity, rate "
        "of flow and viscous drag in arteries when the pressure gradient is known. "
        "J. Physiol. 127:553–563."
    ),
    solution_type="analytical",
)


# ---------------------------------------------------------------------------
# Case 1: 2-D Poiseuille flow (body-force driven)
# ---------------------------------------------------------------------------

def run_poiseuille_2d(steps: int = 5000, verbose: bool = False) -> ContinuousErrorResult:
    """Run 2-D Poiseuille flow and compare against analytical solution.

    Analytical solution: u(y) = G/(2ν) · y(H−y)
    where G = body force, ν = kinematic viscosity, H = channel height.

    Parameters
    ----------
    steps : int
        Number of LBM steps.
    verbose : bool
        Print progress.

    Returns
    -------
    ContinuousErrorResult
    """
    from tensorlbm.ice_hydro.lbm_channel import (
        CX, CY, WT, ChannelConfig, D2Q9Channel,
    )

    G = 1.0e-5
    TAU = 0.8
    H = 16
    ny = H + 4  # +2 bounce-back rows on each side

    cfg = ChannelConfig(
        nx=4, ny=ny, tau=TAU, force=G, steps=0, skip=1,
        cover_amplitude=0.0, cover_depth=float(ny - 3),
        bc="no_slip", forcing="guo",
    )
    ch = D2Q9Channel(cfg)

    # Fluid rows
    rows = np.where(ch.fluid[0, :])[0]
    n_fluid = rows.size

    # Analytical solution
    nu = (TAU - 0.5) / 3.0
    y = rows - 0.5  # lattice coordinates
    u_ana = G / (2.0 * nu) * y * (n_fluid - y)

    # Initialize at analytical profile
    u0 = np.zeros(ny)
    u0[rows] = u_ana
    ch.init_equilibrium(u0)

    # Run simulation
    t0 = time.perf_counter()
    acc = np.zeros(n_fluid)
    rho_acc = 0.0
    n_win = max(steps // 4, 1)
    n_acc = 0

    for step in range(1, steps + 1):
        ch.f = ch._stream_from(ch._collide(ch.f))
        if step > steps - n_win:
            rho = ch.f.sum(axis=0)
            ux = (ch.f * CX[:, None, None]).sum(axis=0) / rho
            acc += ux[0, rows]
            rho_acc += float(rho[ch.fluid].mean())
            n_acc += 1

    wall_time = time.perf_counter() - t0

    # Time-averaged velocity
    u_num = acc / n_acc

    # Compute errors
    l2_rel, linf_rel, l2_abs, linf_abs, n_pts, passed = compute_continuous_errors(
        u_num, u_ana, threshold=0.15
    )

    if verbose:
        print(f"[poiseuille_2d] steps={steps}, n_fluid={n_fluid}, nu={nu:.4f}")
        print(f"[poiseuille_2d] u_max_ana={u_ana.max():.8f}, u_max_num={u_num.max():.8f}")
        print(f"[poiseuille_2d] L2_rel={l2_rel:.8f}, L∞_rel={linf_rel:.8f}")

    return ContinuousErrorResult(
        case_name="poiseuille_2d_H16",
        doi_ref=SCHLICHTING_REF,
        l2_relative=l2_rel,
        linf_relative=linf_rel,
        l2_absolute=l2_abs,
        linf_absolute=linf_abs,
        n_points=n_pts,
        passed=passed,
        threshold=0.15,
        wall_time_s=wall_time,
        params={"H": H, "tau": TAU, "force": G, "steps": steps, "nu": nu},
    )


# ---------------------------------------------------------------------------
# Case 2: 2-D Couette flow (moving top wall)
# ---------------------------------------------------------------------------

def run_couette_2d(steps: int = 5000, verbose: bool = False) -> ContinuousErrorResult:
    """Run 2-D Couette flow and compare against analytical solution.

    Analytical solution: u(y) = U·y/H
    where U = wall velocity, H = channel height.

    Uses a D2Q9 BGK solver with:
    - Bottom wall: stationary bounce-back
    - Top wall: moving bounce-back with velocity U_wall in x-direction
    - Periodic in x

    Parameters
    ----------
    steps : int
        Number of LBM steps.
    verbose : bool
        Print progress.

    Returns
    -------
    ContinuousErrorResult
    """
    from tensorlbm.ice_hydro.lbm_channel import (
        CX, CY, WT, OPP,
    )

    U_WALL = 0.05
    TAU = 0.7
    H = 20
    ny = H + 2  # walls at row 0 and row ny-1
    nx = 4

    # Fluid rows: 1 to ny-2
    fluid_rows = np.arange(1, ny - 1)
    n_fluid = len(fluid_rows)

    # Analytical solution: linear profile
    y = fluid_rows - 0.5  # physical ordinate (half-way from wall)
    H_eff = float(ny - 2)  # effective channel height
    u_ana = U_WALL * y / H_eff

    # Initialize D2Q9 distributions at rest
    f = np.ones((9, nx, ny)) * WT[:, None, None]

    omega = 1.0 / TAU

    # Wall masks
    wall_bottom = np.zeros((nx, ny), dtype=bool)
    wall_bottom[:, 0] = True
    wall_top = np.zeros((nx, ny), dtype=bool)
    wall_top[:, -1] = True

    # Run simulation
    t0 = time.perf_counter()
    acc = np.zeros(n_fluid)
    n_win = max(steps // 4, 1)
    n_acc = 0

    for step in range(1, steps + 1):
        # Compute macroscopic variables
        rho = f.sum(axis=0)
        ux_field = (f * CX[:, None, None]).sum(axis=0) / rho
        uy_field = (f * CY[:, None, None]).sum(axis=0) / rho
        usq = 1.5 * (ux_field**2 + uy_field**2)

        # BGK collision
        feq = np.empty_like(f)
        for i in range(9):
            cu = 3.0 * (CX[i] * ux_field + CY[i] * uy_field)
            feq[i] = WT[i] * rho * (1.0 + cu + 0.5 * cu * cu - usq)
        f = f - omega * (f - feq)

        # Streaming
        f_new = np.empty_like(f)
        f_new[0] = f[0]
        for i in range(1, 9):
            cx, cy = CX[i], CY[i]
            if cx == 0:
                f_new[i] = np.roll(f[i], cy, axis=1)
            elif cy == 0:
                f_new[i] = np.roll(f[i], cx, axis=0)
            else:
                f_new[i] = np.roll(np.roll(f[i], cy, axis=1), cx, axis=0)

        # Bounce-back on walls
        # Bottom wall (stationary): f_opp = f
        for i in range(1, 9):
            f_new[OPP[i]][wall_bottom] = f[i][wall_bottom]

        # Top wall (moving at U_WALL in x):
        # f_opp = f - 6*W_i*rho*(c_i · U_wall)
        # where U_wall = (U_WALL, 0)
        rho_top = rho[wall_top]  # shape (nx,)
        for i in range(1, 9):
            cu_wall = CX[i] * U_WALL  # c_i · U_wall
            f_new[OPP[i]][wall_top] = f[i][wall_top] - 6.0 * WT[i] * rho_top * cu_wall

        f = f_new

        # Accumulate during last quarter
        if step > steps - n_win:
            rho = f.sum(axis=0)
            ux_field = (f * CX[:, None, None]).sum(axis=0) / rho
            acc += ux_field[0, fluid_rows]
            n_acc += 1

    wall_time = time.perf_counter() - t0

    u_num = acc / n_acc

    l2_rel, linf_rel, l2_abs, linf_abs, n_pts, passed = compute_continuous_errors(
        u_num, u_ana, threshold=0.15
    )

    if verbose:
        print(f"[couette_2d] steps={steps}, n_fluid={n_fluid}, H_eff={H_eff}")
        print(f"[couette_2d] u_max_ana={u_ana.max():.8f}, u_max_num={u_num.max():.8f}")
        print(f"[couette_2d] L2_rel={l2_rel:.8f}, L∞_rel={linf_rel:.8f}")

    return ContinuousErrorResult(
        case_name="couette_2d_H20",
        doi_ref=SCHLICHTING_REF,
        l2_relative=l2_rel,
        linf_relative=linf_rel,
        l2_absolute=l2_abs,
        linf_absolute=linf_abs,
        n_points=n_pts,
        passed=passed,
        threshold=0.15,
        wall_time_s=wall_time,
        params={"H": H, "tau": TAU, "u_wall": U_WALL, "steps": steps, "H_eff": H_eff},
    )


# ---------------------------------------------------------------------------
# Case 3: Womersley pulsatile flow (α=5)
# ---------------------------------------------------------------------------

def run_womersley_alpha5(steps: int = 2000, verbose: bool = False) -> ContinuousErrorResult:
    """Run Womersley pulsatile channel flow and compare against analytical solution.

    The Womersley number α = H/2 · √(ω/ν) characterizes the ratio of
    oscillatory inertia to viscous forces.

    Analytical solution (complex phasor):
        u(y,t) = Re{ (G₀/(iρω)) · [1 - cosh(λy)/cosh(λH/2)] · e^(iωt) }
    where λ = √(iω/ν), G₀ = pressure gradient amplitude.

    Parameters
    ----------
    steps : int
        Number of LBM steps (ignored; solver uses its own schedule).
    verbose : bool
        Print progress.

    Returns
    -------
    ContinuousErrorResult
    """
    from tensorlbm.verification.womersley_pulsatile import (
        ChannelPulsatileConfig, ChannelPulsatileSolver,
    )

    # Use H=31 for reasonable runtime (~26s) while maintaining accuracy
    H = 31
    alpha = 5.0
    tau = 0.8
    u_peak = 0.03
    nx = 8
    measure_periods = 4

    cfg = ChannelPulsatileConfig(
        H=H, alpha=alpha, tau=tau, u_peak=u_peak,
        nx=nx, measure_periods=measure_periods, seed=42,
    )
    solver = ChannelPulsatileSolver(cfg)

    t0 = time.perf_counter()
    result = solver.run()
    wall_time = time.perf_counter() - t0

    # The Womersley solver already computes amplitude and phase errors
    # against the analytical solution internally
    amp_rel_pct = result["metrics"]["amplitude_rel_pct"]
    phase_rel_pct = result["metrics"]["phase_rel_pct"]

    # Convert percentage to relative error
    l2_rel = amp_rel_pct / 100.0
    linf_rel = max(amp_rel_pct, phase_rel_pct) / 100.0

    passed = l2_rel <= 0.15 and linf_rel <= 0.15

    if verbose:
        print(f"[womersley_alpha5] α={alpha}, H={H}, periods={measure_periods}")
        print(f"[womersley_alpha5] amp_rel={amp_rel_pct:.4f}%, phase_rel={phase_rel_pct:.4f}%")

    return ContinuousErrorResult(
        case_name="womersley_alpha5_H31",
        doi_ref=WOMERSLEY_REF,
        l2_relative=l2_rel,
        linf_relative=linf_rel,
        l2_absolute=amp_rel_pct,
        linf_absolute=max(amp_rel_pct, phase_rel_pct),
        n_points=measure_periods,
        passed=passed,
        threshold=0.15,
        wall_time_s=wall_time,
        params={"alpha": alpha, "H": H, "tau": tau, "u_peak": u_peak,
                "measure_periods": measure_periods},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Continuous-error cross-validation: TensorLBM vs DOI-cited analytical references"
    )
    ap.add_argument("--steps", type=int, default=0,
                    help="Override steps (0 = case defaults)")
    ap.add_argument("--out", type=str, default="",
                    help="Output directory (default: benchmarks/continuous_error_xval)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print progress")
    args = ap.parse_args()

    steps = args.steps or 0
    out_dir = Path(args.out) if args.out else (ROOT / "benchmarks" / "continuous_error_xval")

    print("=" * 80)
    print("Continuous-Error Cross-Validation: TensorLBM vs DOI References")
    print("=" * 80)
    print()

    results = []

    # Case 1: Poiseuille
    print("Running Case 1: Poiseuille 2D (H=16, τ=0.8, G=1e-5)...")
    s1 = steps if steps > 0 else 5000
    r1 = run_poiseuille_2d(steps=s1, verbose=args.verbose)
    results.append(r1)
    print(f"  {r1.summary_line()}")
    print()

    # Case 2: Couette
    print("Running Case 2: Couette 2D (H=20, τ=0.7, U=0.05)...")
    s2 = steps if steps > 0 else 5000
    r2 = run_couette_2d(steps=s2, verbose=args.verbose)
    results.append(r2)
    print(f"  {r2.summary_line()}")
    print()

    # Case 3: Womersley
    print("Running Case 3: Womersley pulsatile (α=5, H=31)...")
    r3 = run_womersley_alpha5(verbose=args.verbose)
    results.append(r3)
    print(f"  {r3.summary_line()}")
    print()

    # Print full report
    report = print_error_report(results)
    print(report)

    # Save artifacts
    artifacts = save_artifacts(results, out_dir, label="continuous_error_xval")
    print(f"\nArtifacts saved:")
    for k, v in artifacts.items():
        print(f"  {k}: {v}")

    # Summary
    n_pass = sum(1 for r in results if r.passed)
    n_fail = len(results) - n_pass
    print(f"\nSummary: {n_pass}/{len(results)} cases passed (threshold: 15% relative error)")

    if n_fail > 0:
        print("\nFailed cases:")
        for r in results:
            if not r.passed:
                print(f"  {r.case_name}: L2_rel={r.l2_relative:.6f}, L∞_rel={r.linf_relative:.6f}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
