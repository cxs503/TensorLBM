#!/usr/bin/env python
"""Acoustic scattering benchmark: plane wave off a rigid cylinder (Mie scattering).

A plane acoustic wave impinges on a rigid cylinder (bounce-back obstacle).
The scattered field is measured at several monitor points and compared with
the analytical Mie scattering solution for a rigid (Neumann) cylinder.

Method
------
1. Run simulation WITH cylinder    → total field
2. Run simulation WITHOUT cylinder → incident field
3. Scattered = total − incident
4. Compare scattered pressure amplitude with analytical solution

Analytical solution (2-D, rigid / Neumann cylinder)
---------------------------------------------------
  p_s(r,θ) = δρ · c_s² · |Σ_n ε_n · (i^n · J_n'(ka) / H_n^(1)'(ka)) · H_n^(1)(kr) · cos(nθ))|
where  a = cylinder radius,  k = ω/c_s,  J_n = Bessel J,  H_n^(1) = Hankel first kind,
ε_n = 1 (n=0) or 2 (n≥1) is the Neumann factor, prime denotes derivative w.r.t. argument.

Setup
-----
  * Grid  400 × 200 × 1   (2-D D3Q19 BGK, nz = 1)
  * Cylinder  centre (nx/4, ny/2),  radius R = 10,  bounce-back (rigid)
  * Plane-wave source at x = 0:
        ρ  = 1 + δρ · sin(ωt)
        ux = cs · δρ · sin(ωt)        (impedance relation for right-travelling wave)
  * Sponge layer  sin² damping,  width = 40,  at all x/y boundaries
    (characteristic-based: damps outgoing waves only, preserves incident wave)
  * Monitors  angles 0°, 45°, 90°, 135°, 180°  at  r = 30, 40, 50

Run
---
    PYTHONPATH=src python examples/benchmark_acoustic_scattering.py --device cpu --steps 1500
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
from scipy.special import hankel1, jvp

# --------------------------------------------------------------------------- #
# Make tensorlbm importable when running from the repo root.
# --------------------------------------------------------------------------- #
_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.boundaries import cylinder_mask  # noqa: E402
from tensorlbm.boundaries3d import bounce_back_cells_3d  # noqa: E402
from tensorlbm.d3q19 import equilibrium3d, macroscopic3d  # noqa: E402
from tensorlbm.solver3d import collide_bgk3d, stream3d  # noqa: E402

# =========================================================================== #
# Constants
# =========================================================================== #
CS2 = 1.0 / 3.0          # lattice sound speed squared
CS = math.sqrt(CS2)      # c_s = 1/√3 ≈ 0.5774
INV_CS = 1.0 / CS        # 1/c_s = √3 ≈ 1.7321


# =========================================================================== #
# Analytical solution
# =========================================================================== #

def hankel1_prime(n: int, z: float) -> complex:
    """First derivative of H_n^(1)(z) via the recurrence relation.

    H_n'(z) = ½ · [ H_{n-1}^(1)(z) − H_{n+1}^(1)(z) ]

    Valid for all integer n ≥ 0 (scipy handles negative orders correctly:
    H_{-n}^(1) = (−1)^n · H_n^(1)).
    """
    return 0.5 * (hankel1(n - 1, z) - hankel1(n + 1, z))


def analytical_scattered_pressure(
    r: float,
    theta: float,
    a: float,
    k: float,
    delta_rho: float,
    cs2: float,
    n_max: int = 10,
) -> float:
    """Analytical scattered pressure amplitude for a plane wave off a rigid cylinder.

    p_s(r,θ) = δρ · c_s² · |Σ_{n=0}^{N} ε_n · (i^n · J_n'(ka) / H_n^(1)'(ka))
                                  · H_n^(1)(kr) · cos(nθ))|

    where ε_n = 1 for n = 0 and ε_n = 2 for n ≥ 1 (Neumann factor).

    Returns the real, non-negative amplitude of the scattered pressure.
    """
    ka = k * a
    kr = k * r
    total = 0.0 + 0.0j
    for n in range(n_max + 1):
        eps_n = 1.0 if n == 0 else 2.0          # Neumann factor
        jn_prime = jvp(n, ka)                    # J_n'(ka)
        hn_prime = hankel1_prime(n, ka)          # H_n^(1)'(ka)
        hn_kr = hankel1(n, kr)                   # H_n^(1)(kr)
        coeff = eps_n * (1j ** n) * jn_prime / hn_prime * hn_kr * math.cos(n * theta)
        total += coeff
    return delta_rho * cs2 * abs(total)


# =========================================================================== #
# LBM simulation
# =========================================================================== #

def run_simulation(
    nx: int,
    ny: int,
    nz: int,
    tau: float,
    cx: float,
    cy: float,
    R: float,
    delta_rho: float,
    omega: float,
    steps: int,
    device: str,
    with_cylinder: bool,
    monitor_coords: list[tuple[int, int]],
    sponge_width: int = 40,
    log_every: int = 500,
) -> list[list[float]]:
    """Run a single LBM simulation and return rho history at each monitor point.

    Parameters
    ----------
    with_cylinder : bool
        If True, a rigid cylinder (bounce-back) is placed at (cx, cy) with
        radius R.  If False, no obstacle is present (incident-field run).
    monitor_coords : list of (mx, my)
        Grid coordinates of monitor points.
    """
    dev = torch.device(device)

    # --- Initial condition: uniform fluid at rest ---
    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    # --- Cylinder mask (expand 2-D mask to 3-D for D3Q19) ---
    if with_cylinder:
        cyl_2d = cylinder_mask(nx, ny, cx, cy, R, dev)          # (ny, nx)
        cyl_mask = cyl_2d.unsqueeze(0).expand(nz, ny, nx).contiguous()
    else:
        cyl_mask = None

    # --- Sponge layer: characteristic-based damping ---
    # Damps only the *outgoing* characteristic at each boundary, preserving
    # the incident (right-travelling) wave at the left boundary.
    zz, yy, xx = torch.meshgrid(
        torch.arange(nz, device=dev),
        torch.arange(ny, device=dev),
        torch.arange(nx, device=dev),
        indexing="ij",
    )
    dist_left = xx.float()
    dist_right = (nx - 1 - xx).float()
    dist_y = torch.minimum(yy.float(), (ny - 1 - yy).float())

    def _sponge_profile(d):
        return torch.where(
            d < sponge_width,
            torch.sin(math.pi * d / (2.0 * sponge_width)) ** 2,
            torch.ones_like(d),
        )

    damping_left = _sponge_profile(dist_left)     # 0 at x=0 → 1 interior
    damping_right = _sponge_profile(dist_right)   # 0 at x=nx-1 → 1 interior
    damping_y = _sponge_profile(dist_y)           # 0 at y=0/ny-1 → 1 interior
    # Overall damping: exclude left boundary (plane-wave source at x=0)
    # to avoid absorbing the source signal. Only sponge right/top/bottom.
    damping_neq = torch.minimum(damping_right, damping_y)

    # --- Pre-compute target equilibrium for sponge (rho=1, u=0) ---
    feq_bnd = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)

    # --- Pre-allocate zero column for source equilibrium ---
    zero_col = torch.zeros((nz, ny, 1), device=dev, dtype=f.dtype)

    # --- Monitor recording ---
    n_mon = len(monitor_coords)
    rho_history: list[list[float]] = [[] for _ in range(n_mon)]

    with torch.no_grad():
        for step in range(1, steps + 1):
            # === Collision ===
            f = collide_bgk3d(f, tau)

            # === Streaming (periodic via torch.roll) ===
            f = stream3d(f)

            # === Bounce-back on cylinder (rigid / Neumann) ===
            if with_cylinder and cyl_mask is not None:
                f = bounce_back_cells_3d(f, cyl_mask)

            # === Sponge layer: blend toward TARGET equilibrium (absorbs acoustic wave) ===
            f = feq_bnd + (f - feq_bnd) * damping_neq

            # === Plane-wave source at x = 0 ===
            # Density perturbation + matching velocity for a right-travelling
            # wave (impedance relation  u = c_s · δρ  for a plane wave):
            #   ρ  = 1 + δρ · sin(ωt)
            #   ux = cs · δρ · sin(ωt)
            rho_src = 1.0 + delta_rho * math.sin(omega * step)
            ux_src = CS * delta_rho * math.sin(omega * step)
            rho_col = torch.full((nz, ny, 1), rho_src, device=dev, dtype=f.dtype)
            ux_col = torch.full((nz, ny, 1), ux_src, device=dev, dtype=f.dtype)
            feq_src = equilibrium3d(rho_col, ux_col, zero_col, zero_col.clone(),
                                    device=dev)
            f[:, :, :, 0:1] = feq_src

            # === Record rho at monitors ===
            rho, ux, uy, uz = macroscopic3d(f)
            for i, (mx, my) in enumerate(monitor_coords):
                rho_history[i].append(float(rho[0, my, mx].item()))

            # === Logging ===
            if step % log_every == 0 or step == steps:
                drho_max = float((rho - rho0).abs().max())
                print(f"    step {step:5d}/{steps}  |Δρ|_max = {drho_max:.6e}",
                      flush=True)

    return rho_history


# =========================================================================== #
# Main
# =========================================================================== #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acoustic scattering benchmark: plane wave off rigid cylinder"
    )
    parser.add_argument("--nx", type=int, default=400, help="Grid size in x (default 400)")
    parser.add_argument("--ny", type=int, default=200, help="Grid size in y (default 200)")
    parser.add_argument("--nz", type=int, default=1, help="Grid size in z (default 1)")
    parser.add_argument("--tau", type=float, default=0.55, help="BGK relaxation time (default 0.55)")
    parser.add_argument("--R", type=float, default=20.0, help="Cylinder radius (default 20)")
    parser.add_argument("--delta-rho", type=float, default=0.001,
                        help="Density perturbation amplitude (default 0.001)")
    parser.add_argument("--omega", type=float, default=0.1,
                        help="Angular frequency (default 0.1)")
    parser.add_argument("--steps", type=int, default=1500, help="Number of time steps (default 1500)")
    parser.add_argument("--device", default="cpu", help="Device: 'cpu' or 'cuda' (default cpu)")
    parser.add_argument("--log-every", type=int, default=500, help="Print interval (default 500)")
    parser.add_argument("--sponge-width", type=int, default=40,
                        help="Sponge layer width (default 40)")
    args = parser.parse_args()

    nx, ny, nz = args.nx, args.ny, args.nz
    tau = args.tau
    R = args.R
    delta_rho = args.delta_rho
    omega = args.omega
    steps = args.steps
    device = args.device
    sponge_width = args.sponge_width

    cx, cy = nx // 4, ny // 2   # cylinder centre
    k = omega / CS
    lam = 2.0 * math.pi / k

    # --- Monitor points: angles 0°, 45°, 90°, 135°, 180° at r = 30, 40, 50 ---
    angles_deg = [0, 45, 90, 135, 180]
    distances = [50, 60]
    monitor_info: list[tuple[int, int, float, float]] = []   # (mx, my, r, θ_deg)
    monitor_coords: list[tuple[int, int]] = []
    for r in distances:
        for ang_deg in angles_deg:
            ang = math.radians(ang_deg)
            mx = int(round(cx + r * math.cos(ang)))
            my = int(round(cy + r * math.sin(ang)))
            assert 0 <= mx < nx and 0 <= my < ny, \
                f"Monitor point ({mx}, {my}) out of bounds"
            monitor_info.append((mx, my, float(r), float(ang_deg)))
            monitor_coords.append((mx, my))

    # --- Header ---
    print("=" * 72, flush=True)
    print("  声学散射基准测试: 平面波入射刚性圆柱 (Mie 散射)", flush=True)
    print("=" * 72, flush=True)
    print(f"  网格           : {nx} × {ny} × {nz}", flush=True)
    print(f"  圆柱中心       : ({cx}, {cy}),  半径 R = {R}", flush=True)
    print(f"  密度扰动       : δρ = {delta_rho},  ω = {omega}", flush=True)
    print(f"  声速 cs        : {CS:.6f}", flush=True)
    print(f"  波长 λ         : {lam:.2f}", flush=True)
    print(f"  波数 k         : {k:.6f}", flush=True)
    print(f"  ka             : {k * R:.4f}", flush=True)
    print(f"  τ              : {tau}", flush=True)
    print(f"  吸收层宽度     : {sponge_width}", flush=True)
    print(f"  步数           : {steps}", flush=True)
    print(f"  监测点数       : {len(monitor_coords)}", flush=True)
    print(f"  设备           : {device}", flush=True)
    print("=" * 72, flush=True)

    # --- Run 1: WITH cylinder (total field) ---
    print(flush=True)
    print("  [1/2] 运行带圆柱仿真 (总场) …", flush=True)
    rho_total = run_simulation(
        nx, ny, nz, tau, cx, cy, R, delta_rho, omega, steps, device,
        with_cylinder=True, monitor_coords=monitor_coords,
        sponge_width=sponge_width, log_every=args.log_every,
    )

    # --- Run 2: WITHOUT cylinder (incident field) ---
    print(flush=True)
    print("  [2/2] 运行无圆柱仿真 (入射场) …", flush=True)
    rho_incident = run_simulation(
        nx, ny, nz, tau, cx, cy, R, delta_rho, omega, steps, device,
        with_cylinder=False, monitor_coords=monitor_coords,
        sponge_width=sponge_width, log_every=args.log_every,
    )

    # --- Analysis: scattered field vs analytical ---
    print(flush=True)
    print("=" * 72, flush=True)
    print("  散射场幅度对比 (LBM vs 解析解)", flush=True)
    print("=" * 72, flush=True)
    header = (
        f"  {'r':>4s} {'θ°':>5s}  {'LBM 散射幅度':>16s}  {'解析解':>16s}  "
        f"{'误差%':>8s}  {'结果':>6s}"
    )
    print(header, flush=True)
    print("  " + "-" * 68, flush=True)

    all_pass = True
    errors: list[float] = []
    for i, (mx, my, r, ang_deg) in enumerate(monitor_info):
        # Scattered pressure:  p_s = cs² · (ρ_total − ρ_incident)
        rho_t = np.array(rho_total[i], dtype=np.float64)
        rho_i = np.array(rho_incident[i], dtype=np.float64)
        p_scattered = CS2 * (rho_t - rho_i)

        # Amplitude from the second half (steady state, peak-to-peak / 2)
        half = len(p_scattered) // 2
        p_amp_lbm = float(
            np.max(p_scattered[half:]) - np.min(p_scattered[half:])
        ) / 2.0

        # Analytical scattered pressure amplitude
        theta = math.radians(ang_deg)
        p_amp_ana = analytical_scattered_pressure(r, theta, R, k, delta_rho, CS2)

        if p_amp_ana > 1e-15:
            err_pct = abs(p_amp_lbm - p_amp_ana) / p_amp_ana * 100.0
        else:
            err_pct = 0.0 if p_amp_lbm < 1e-15 else 100.0

        errors.append(err_pct)
        passed = err_pct < 15.0
        if not passed:
            all_pass = False

        status = "PASS" if passed else "FAIL"
        print(
            f"  {r:4.0f} {ang_deg:5.0f}  {p_amp_lbm:16.8e}  {p_amp_ana:16.8e}  "
            f"{err_pct:8.2f}  {status:>6s}",
            flush=True,
        )

    print("  " + "-" * 68, flush=True)
    mean_err = sum(errors) / len(errors) if errors else 0.0
    max_err = max(errors) if errors else 0.0
    print(f"  平均误差: {mean_err:.2f}%   最大误差: {max_err:.2f}%", flush=True)
    print(flush=True)
    if all_pass:
        print("  ✓ PASS — 声学散射基准测试通过 (所有监测点误差 < 15%)", flush=True)
    else:
        print("  ✗ FAIL — 声学散射基准测试未通过 (部分监测点误差 ≥ 15%)", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
