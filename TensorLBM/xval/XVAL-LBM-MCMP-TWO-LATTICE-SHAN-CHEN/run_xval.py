#!/usr/bin/env python3
"""XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN — TensorLBM × je-santos/MPLBM-UT cross-validation.

Cross-validates TensorLBM's two-component Shan-Chen (MCMP) implementation
against the reference algorithm from je-santos/MPLBM-UT.  The MPLBM-UT
convention is re-implemented from its published source (not imported) so that
the two solvers share *no code path* — the comparison is between independent
implementations of the same physics.

MPLBM-UT MCMP conventions (from source inspection):
  - D2Q9 lattice, two distribution functions (two-lattice approach)
  - Linear pseudopotential: psi(rho) = rho
  - Forward-gather force: F_sigma = -G_12 * psi_sigma * sum_i w_i psi_sigma'(x+c_i) c_i
  - G_12 > 0 means repulsion (phase separation) — OPPOSITE sign to TensorLBM
  - Velocity-shift forcing: u_eq = u + tau * F / rho
  - BGK collision with per-component tau
  - Periodic streaming

TensorLBM conventions:
  - Same D2Q9 lattice, two distribution functions
  - Linear pseudopotential: psi(rho) = rho (default for MCMP)
  - Backward-gather force: F_sigma = -G_12 * psi_sigma * sum_i w_i psi_sigma'(x-c_i) c_i
  - G_12 < 0 means repulsion (phase separation)
  - Velocity-shift forcing: u_eq = u + tau * F / rho
  - BGK collision with per-component tau
  - Periodic streaming

The backward-gather with G_12 < 0 is mathematically equivalent to the
forward-gather with G_12 > 0 (the sign flip and direction flip cancel), so
both should produce identical results when G_TL = -G_MPLBM.

Validation protocol:
  1. Static droplet at multiple radii → Laplace law Δp = σ/R
  2. Coexistence densities from flat interface relaxation
  3. Spurious current levels
  4. Surface tension from Laplace fit vs EOS-based prediction

Acceptance criterion: <15% (owner precision standard).

Usage:
    python run_xval.py              # full protocol
    python run_xval.py --quick      # quick smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# D2Q9 lattice constants (shared by both solvers)
# ---------------------------------------------------------------------------

CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float64)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float64)
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=int)
CS2 = 1.0 / 3.0

# Torch versions
CX_T = torch.tensor(CX, dtype=torch.float64)
CY_T = torch.tensor(CY, dtype=torch.float64)
W_T = torch.tensor(W, dtype=torch.float64)

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Reference solver: MPLBM-UT convention (forward gather, G > 0 = repulsive)
# ---------------------------------------------------------------------------

def _eq_ref(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 equilibrium (float64)."""
    cu = CX_T[:, None, None] * ux[None] + CY_T[:, None, None] * uy[None]
    usq = ux * ux + uy * uy
    return W_T[:, None, None] * rho[None] * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq[None])


def _macro_ref(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Macroscopic quantities from distribution."""
    rho = f.sum(dim=0)
    rho_safe = torch.clamp(rho, min=1e-12)
    ux = (f * CX_T[:, None, None]).sum(dim=0) / rho_safe
    uy = (f * CY_T[:, None, None]).sum(dim=0) / rho_safe
    return rho, ux, uy


def _force_mplbm(rho1: torch.Tensor, rho2: torch.Tensor, G12: float) -> tuple:
    """MPLBM-UT force: forward gather, G12 > 0 = repulsive.

    F_1 = -G12 * rho1 * sum_i w_i * rho2(x + c_i) * c_i
    F_2 = -G12 * rho2 * sum_i w_i * rho1(x + c_i) * c_i

    With G12 > 0, the force pushes component 1 away from component 2
    (repulsion → phase separation).
    """
    # Forward gather: sum_i w_i * rho(x + c_i) * c_i
    # Using torch.roll with negative shift to implement x + c_i
    grad_x2 = torch.zeros_like(rho2)
    grad_y2 = torch.zeros_like(rho2)
    grad_x1 = torch.zeros_like(rho1)
    grad_y1 = torch.zeros_like(rho1)

    for i in range(9):
        if CX[i] == 0 and CY[i] == 0:
            continue
        # Forward: rho(x + c_i) = roll by -c_i in index space
        # roll(shifts=(dy, dx)): positive shift moves data down/right
        # To get rho(x + cx, y + cy), we roll by (-cy, -cx)
        rho2_shifted = torch.roll(rho2, shifts=(-int(CY[i]), -int(CX[i])), dims=(0, 1))
        rho1_shifted = torch.roll(rho1, shifts=(-int(CY[i]), -int(CX[i])), dims=(0, 1))
        grad_x2 += W[i] * CX[i] * rho2_shifted
        grad_y2 += W[i] * CY[i] * rho2_shifted
        grad_x1 += W[i] * CX[i] * rho1_shifted
        grad_y1 += W[i] * CY[i] * rho1_shifted

    Fx1 = -G12 * rho1 * grad_x2
    Fy1 = -G12 * rho1 * grad_y2
    Fx2 = -G12 * rho2 * grad_x1
    Fy2 = -G12 * rho2 * grad_y1

    return Fx1, Fy1, Fx2, Fy2


def _collide_mplbm(f1, f2, G12, tau1, tau2):
    """MPLBM-UT collision: velocity-shift forcing, BGK."""
    rho1, ux1, uy1 = _macro_ref(f1)
    rho2, ux2, uy2 = _macro_ref(f2)

    Fx1, Fy1, Fx2, Fy2 = _force_mplbm(rho1, rho2, G12)

    # Velocity shift: u_eq = u + tau * F / rho
    rho1_safe = torch.clamp(rho1, min=1e-12)
    rho2_safe = torch.clamp(rho2, min=1e-12)
    ux1_eq = ux1 + tau1 * Fx1 / rho1_safe
    uy1_eq = uy1 + tau1 * Fy1 / rho1_safe
    ux2_eq = ux2 + tau2 * Fx2 / rho2_safe
    uy2_eq = uy2 + tau2 * Fy2 / rho2_safe

    feq1 = _eq_ref(rho1, ux1_eq, uy1_eq)
    feq2 = _eq_ref(rho2, ux2_eq, uy2_eq)

    f1_post = f1 - (f1 - feq1) / tau1
    f2_post = f2 - (f2 - feq2) / tau2

    return f1_post, f2_post


def _stream_ref(f: torch.Tensor) -> torch.Tensor:
    """Periodic streaming: f_i(x, t+1) = f_i(x - c_i, t)."""
    out = torch.empty_like(f)
    for i in range(9):
        # Pull: f_i(x) = f_i(x - c_i) → roll by (+cy, +cx)
        out[i] = torch.roll(f[i], shifts=(int(CY[i]), int(CX[i])), dims=(0, 1))
    return out


def step_mplbm(f1, f2, G12, tau1, tau2):
    """One full time step: collide + stream (MPLBM-UT convention)."""
    f1_post, f2_post = _collide_mplbm(f1, f2, G12, tau1, tau2)
    return _stream_ref(f1_post), _stream_ref(f2_post)


# ---------------------------------------------------------------------------
# TensorLBM solver (imported from the library)
# ---------------------------------------------------------------------------

def _get_tensorlbm_step():
    """Import and wrap TensorLBM's MCMP step."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
    from tensorlbm.d2q9 import equilibrium as tl_eq, macroscopic as tl_macro
    from tensorlbm.multiphase import collide_sc_two_component
    from tensorlbm.solver import stream as tl_stream

    def step_tl(f1, f2, G12_tl, tau1, tau2):
        """One full time step using TensorLBM (G12 < 0 = repulsive)."""
        f1_post, f2_post = collide_sc_two_component(
            f1, f2, G_12=G12_tl, tau1=tau1, tau2=tau2,
            u_eq="self", force_scheme="shift", psi=None  # linear psi = rho
        )
        return tl_stream(f1_post), tl_stream(f2_post)

    return step_tl, tl_eq, tl_macro


# ---------------------------------------------------------------------------
# Measurement utilities
# ---------------------------------------------------------------------------

def init_droplet(n: int, radius: float, rho1_liq: float, rho1_gas: float,
                 rho2_liq: float, rho2_gas: float, width: float = 2.0,
                 device: str = "cpu", dtype=torch.float64):
    """Initialise a tanh-profiled circular droplet."""
    y = torch.arange(n, device=device, dtype=dtype)
    x = torch.arange(n, device=device, dtype=dtype)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    centre = (n - 1) / 2.0
    r = torch.sqrt((X - centre) ** 2 + (Y - centre) ** 2)
    s = 0.5 * (1.0 - torch.tanh((r - radius) / max(width, 1e-6)))

    rho1 = rho1_gas + s * (rho1_liq - rho1_gas)
    rho2 = rho2_gas + s * (rho2_liq - rho2_gas)
    zero = torch.zeros_like(rho1)
    return rho1, rho2, zero, zero


def measure_droplet(f1: torch.Tensor, f2: torch.Tensor, G12_sign: str = "mplbm",
                    G12: float = 0.0) -> dict[str, Any]:
    """Measure droplet properties: Δp, R_eq, σ, spurious currents.

    Args:
        f1, f2: Distribution functions.
        G12_sign: "mplbm" (G > 0 repulsive) or "tensorlbm" (G < 0 repulsive).
        G12: Coupling constant in the respective convention.
    """
    rho1, ux1, uy1 = _macro_ref(f1)
    rho2, ux2, uy2 = _macro_ref(f2)

    # Barycentric velocity
    rho_tot = rho1 + rho2
    rho_safe = torch.clamp(rho_tot, min=1e-12)
    ux = (rho1 * ux1 + rho2 * ux2) / rho_safe
    uy = (rho1 * uy1 + rho2 * uy2) / rho_safe

    # EOS pressure: p = cs^2 (rho1 + rho2) + G12 * cs^2 * rho1 * rho2
    # For MPLBM-UT: G12 > 0 repulsive, pressure = cs^2(rho1+rho2) - G12*cs^2*rho1*rho2
    # For TensorLBM: G12 < 0 repulsive, pressure = cs^2(rho1+rho2) + G12*cs^2*rho1*rho2
    # Both give the same physical pressure when G_TL = -G_MPLBM
    if G12_sign == "mplbm":
        p = CS2 * (rho1 + rho2) - G12 * CS2 * rho1 * rho2
    else:
        p = CS2 * (rho1 + rho2) + G12 * CS2 * rho1 * rho2

    # Box-averaged pressures
    n = rho1.shape[0]
    centre = n // 2
    # Core: central 1/4 of the domain
    core_slice = slice(centre - n // 8, centre + n // 8)
    p_core = float(p[core_slice, core_slice].mean())
    # Far: corners
    k = max(4, n // 16)
    p_far = float(p[:k, :k].mean())

    dp = p_core - p_far

    # Equivalent radius from droplet area
    rho1_mean = float(rho1.mean())
    mask = rho1 > rho1_mean
    area = float(mask.sum())
    R_eq = math.sqrt(area / math.pi) if area > 0 else 0.0

    # Surface tension from Laplace law (2D: Δp = σ/R)
    sigma = dp * R_eq if R_eq > 0 else 0.0

    # Spurious current
    u_max = float(torch.sqrt(ux ** 2 + uy ** 2).max())

    # Coexistence densities
    rho1_liq = float(rho1[centre, centre])
    rho1_gas = float(rho1[0, n // 2])
    rho2_liq = float(rho2[centre, centre])
    rho2_gas = float(rho2[0, n // 2])

    return {
        "dp": dp,
        "R_eq": R_eq,
        "sigma": sigma,
        "u_max": u_max,
        "rho1_liq": rho1_liq,
        "rho1_gas": rho1_gas,
        "rho2_liq": rho2_liq,
        "rho2_gas": rho2_gas,
        "p_core": p_core,
        "p_far": p_far,
    }


def run_droplet_case(solver, n, radius, G12, tau1, tau2,
                     rho1_liq, rho1_gas, rho2_liq, rho2_gas,
                     steps, G12_sign="mplbm", width=2.0):
    """Run a single droplet case and return measurements."""
    rho1, rho2, ux, uy = init_droplet(n, radius, rho1_liq, rho1_gas,
                                       rho2_liq, rho2_gas, width=width)
    f1 = _eq_ref(rho1, ux, uy)
    f2 = _eq_ref(rho2, ux, uy)

    t0 = time.time()
    for _ in range(steps):
        f1, f2 = solver(f1, f2, G12, tau1, tau2)
    elapsed = time.time() - t0

    # Check for NaN
    if torch.isnan(f1).any() or torch.isnan(f2).any():
        return {"nan": True, "elapsed_s": elapsed}

    result = measure_droplet(f1, f2, G12_sign=G12_sign, G12=G12)
    result["nan"] = False
    result["elapsed_s"] = round(elapsed, 2)
    result["n"] = n
    result["radius_init"] = radius
    result["G12"] = G12
    result["steps"] = steps
    return result


# ---------------------------------------------------------------------------
# Flat interface coexistence measurement
# ---------------------------------------------------------------------------

def run_coexistence(solver, n, G12, tau1, tau2, steps, G12_sign="mplbm"):
    """Run a flat interface to measure coexistence densities.

    Initialises a tanh slab of component 1 in a periodic box and relaxes
    to equilibrium.  The bulk densities on either side of the interface
    are the coexistence densities.
    """
    ny, nx = n, 8  # narrow in x, tall in y
    y = torch.arange(ny, dtype=torch.float64)
    centre = ny / 2.0
    half = ny / 4.0
    w0 = 2.0

    # Liquid slab of component 1 centred in the box
    s = 0.5 * (1.0 + torch.tanh((y - (centre - half)) / w0)) * \
        0.5 * (1.0 - torch.tanh((y - (centre + half)) / w0))
    s = s.view(ny, 1).expand(ny, nx).clone()

    # Start from symmetric initial condition
    rho1_liq, rho1_gas = 0.92, 0.08
    rho2_liq, rho2_gas = 0.08, 0.92
    rho1 = rho1_gas + s * (rho1_liq - rho1_gas)
    rho2 = rho2_gas + s * (rho2_liq - rho2_gas)

    zero = torch.zeros_like(rho1)
    f1 = _eq_ref(rho1, zero, zero)
    f2 = _eq_ref(rho2, zero, zero)

    for _ in range(steps):
        f1, f2 = solver(f1, f2, G12, tau1, tau2)

    if torch.isnan(f1).any():
        return {"nan": True}

    rho1 = f1.sum(dim=0)
    rho2 = f2.sum(dim=0)

    # Measure bulk densities from the centre (liquid) and edges (gas)
    rho1_liq = float(rho1[ny // 2, :].mean())
    rho1_gas = float(rho1[:4, :].mean())
    rho2_liq = float(rho2[ny // 2, :].mean())
    rho2_gas = float(rho2[:4, :].mean())

    return {
        "nan": False,
        "rho1_liq": rho1_liq,
        "rho1_gas": rho1_gas,
        "rho2_liq": rho2_liq,
        "rho2_gas": rho2_gas,
    }


# ---------------------------------------------------------------------------
# Main validation protocol
# ---------------------------------------------------------------------------

@dataclass
class XvalConfig:
    n: int = 96
    radii: tuple = (12.0, 16.0, 20.0, 24.0)
    G_mplbm: float = 2.5  # MPLBM-UT convention: G > 0 = repulsive
    tau: float = 1.0
    steps: int = 8000
    rho1_liq: float = 0.92
    rho1_gas: float = 0.08
    rho2_liq: float = 0.08
    rho2_gas: float = 0.92
    interface_width: float = 2.0
    coex_n: int = 128
    coex_steps: int = 10000


def main():
    parser = argparse.ArgumentParser(description="XVAL MCMP two-lattice Shan-Chen")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test")
    args = parser.parse_args()

    torch.set_num_threads(1)

    if args.quick:
        cfg = XvalConfig(
            n=48, radii=(12.0, 16.0), G_mplbm=2.5, tau=1.0, steps=2000,
            coex_n=64, coex_steps=3000,
        )
    else:
        cfg = XvalConfig()

    G_mplbm = cfg.G_mplbm  # > 0 = repulsive in MPLBM-UT
    G_tl = -G_mplbm        # < 0 = repulsive in TensorLBM

    print("=" * 70)
    print("XVAL-LBM-MCMP-TWO-LATTICE-SHAN-CHEN")
    print(f"MPLBM-UT G12 = {G_mplbm} (repulsive)  ↔  TensorLBM G12 = {G_tl} (repulsive)")
    print(f"Grid: {cfg.n}×{cfg.n}, tau = {cfg.tau}, steps = {cfg.steps}")
    print("=" * 70)

    # Get TensorLBM step function
    step_tl, tl_eq, tl_macro = _get_tensorlbm_step()

    results = {"protocol": asdict(cfg), "cases": []}

    # ------------------------------------------------------------------
    # Phase 1: Coexistence densities from flat interface
    # ------------------------------------------------------------------
    print("\n--- Phase 1: Coexistence densities ---")
    coex_mplbm = run_coexistence(step_mplbm, cfg.coex_n, G_mplbm, cfg.tau, cfg.tau,
                                  cfg.coex_steps, G12_sign="mplbm")
    coex_tl = run_coexistence(step_tl, cfg.coex_n, G_tl, cfg.tau, cfg.tau,
                               cfg.coex_steps, G12_sign="tensorlbm")

    print(f"  MPLBM-UT: rho1_liq={coex_mplbm.get('rho1_liq', 'nan'):.4f}  "
          f"rho1_gas={coex_mplbm.get('rho1_gas', 'nan'):.4f}")
    print(f"  TensorLBM: rho1_liq={coex_tl.get('rho1_liq', 'nan'):.4f}  "
          f"rho1_gas={coex_tl.get('rho1_gas', 'nan'):.4f}")

    results["coexistence"] = {"mplbm": coex_mplbm, "tensorlbm": coex_tl}

    # Use MPLBM-UT coexistence for both (they should be the same)
    if not coex_mplbm.get("nan", True):
        rho1_liq = coex_mplbm["rho1_liq"]
        rho1_gas = coex_mplbm["rho1_gas"]
        rho2_liq = coex_mplbm["rho2_liq"]
        rho2_gas = coex_mplbm["rho2_gas"]
    else:
        rho1_liq, rho1_gas = cfg.rho1_liq, cfg.rho1_gas
        rho2_liq, rho2_gas = cfg.rho2_liq, cfg.rho2_gas

    # ------------------------------------------------------------------
    # Phase 2: Static droplets at multiple radii
    # ------------------------------------------------------------------
    print("\n--- Phase 2: Static droplets (Laplace law) ---")
    mplbm_cases = []
    tl_cases = []

    for R in cfg.radii:
        print(f"\n  R_init = {R}")

        # MPLBM-UT reference
        r_m = run_droplet_case(
            step_mplbm, cfg.n, R, G_mplbm, cfg.tau, cfg.tau,
            rho1_liq, rho1_gas, rho2_liq, rho2_gas,
            cfg.steps, G12_sign="mplbm", width=cfg.interface_width,
        )
        r_m["solver"] = "MPLBM-UT"
        r_m["R_init"] = R
        mplbm_cases.append(r_m)
        print(f"    MPLBM-UT: dp={r_m['dp']:.6f}  R_eq={r_m['R_eq']:.2f}  "
              f"σ={r_m['sigma']:.6f}  u_max={r_m['u_max']:.2e}")

        # TensorLBM
        r_t = run_droplet_case(
            step_tl, cfg.n, R, G_tl, cfg.tau, cfg.tau,
            rho1_liq, rho1_gas, rho2_liq, rho2_gas,
            cfg.steps, G12_sign="tensorlbm", width=cfg.interface_width,
        )
        r_t["solver"] = "TensorLBM"
        r_t["R_init"] = R
        tl_cases.append(r_t)
        print(f"    TensorLBM: dp={r_t['dp']:.6f}  R_eq={r_t['R_eq']:.2f}  "
              f"σ={r_t['sigma']:.6f}  u_max={r_t['u_max']:.2e}")

    results["droplets_mplbm"] = mplbm_cases
    results["droplets_tensorlbm"] = tl_cases

    # ------------------------------------------------------------------
    # Phase 3: Laplace law fit and comparison
    # ------------------------------------------------------------------
    print("\n--- Phase 3: Laplace law analysis ---")

    def laplace_fit(cases):
        """Fit σ from Δp = σ/R (2D) through origin."""
        valid = [c for c in cases if c.get("R_eq", 0) > 0 and not c.get("nan", False)]
        if len(valid) < 2:
            return None, None, None
        inv_R = np.array([1.0 / c["R_eq"] for c in valid])
        dp = np.array([c["dp"] for c in valid])
        # Linear fit through origin: dp = sigma * (1/R)
        sigma = float(np.linalg.lstsq(inv_R.reshape(-1, 1), dp, rcond=None)[0][0])
        predicted = sigma * inv_R
        ss_res = np.sum((dp - predicted) ** 2)
        ss_tot = np.sum((dp - dp.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        # Max relative deviation of sigma_i = dp_i * R_i from sigma
        max_dev = max(abs(c["sigma"] - sigma) / max(abs(sigma), 1e-12) for c in valid)
        return sigma, r2, max_dev

    sig_m, r2_m, dev_m = laplace_fit(mplbm_cases)
    sig_t, r2_t, dev_t = laplace_fit(tl_cases)

    if sig_m is not None:
        print(f"  MPLBM-UT:  σ = {sig_m:.6f}  R² = {r2_m:.4f}  max_dev = {dev_m:.2%}")
    if sig_t is not None:
        print(f"  TensorLBM: σ = {sig_t:.6f}  R² = {r2_t:.4f}  max_dev = {dev_t:.2%}")

    # Cross-comparison
    if sig_m is not None and sig_t is not None:
        sigma_err = abs(sig_t - sig_m) / max(abs(sig_m), 1e-12) * 100
        print(f"\n  σ cross-error: {sigma_err:.2f}%  [{'PASS' if sigma_err < 15 else 'FAIL'}]")
    else:
        sigma_err = float('nan')
        print("\n  σ cross-error: N/A (insufficient data)")

    results["laplace_fit"] = {
        "mplbm": {"sigma": sig_m, "r2": r2_m, "max_deviation": dev_m},
        "tensorlbm": {"sigma": sig_t, "r2": r2_t, "max_deviation": dev_t},
        "sigma_cross_error_pct": sigma_err,
        "pass_15pct": sigma_err < 15 if not math.isnan(sigma_err) else False,
    }

    # ------------------------------------------------------------------
    # Phase 4: Per-radius comparison
    # ------------------------------------------------------------------
    print("\n--- Phase 4: Per-radius comparison ---")
    per_radius = []
    for rm, rt in zip(mplbm_cases, tl_cases):
        if rm.get("nan") or rt.get("nan"):
            continue
        dp_err = abs(rt["dp"] - rm["dp"]) / max(abs(rm["dp"]), 1e-12) * 100
        sig_err = abs(rt["sigma"] - rm["sigma"]) / max(abs(rm["sigma"]), 1e-12) * 100
        u_ratio = rt["u_max"] / max(rm["u_max"], 1e-12)
        row = {
            "R_init": rm["R_init"],
            "dp_mplbm": rm["dp"], "dp_tensorlbm": rt["dp"], "dp_err_pct": dp_err,
            "sigma_mplbm": rm["sigma"], "sigma_tensorlbm": rt["sigma"], "sigma_err_pct": sig_err,
            "u_max_mplbm": rm["u_max"], "u_max_tensorlbm": rt["u_max"], "u_ratio": u_ratio,
            "R_eq_mplbm": rm["R_eq"], "R_eq_tensorlbm": rt["R_eq"],
        }
        per_radius.append(row)
        print(f"  R={rm['R_init']:.0f}: Δp err={dp_err:.2f}%  σ err={sig_err:.2f}%  "
              f"u_ratio={u_ratio:.3f}")

    results["per_radius"] = per_radius

    # ------------------------------------------------------------------
    # Phase 5: Additional G values (sensitivity)
    # ------------------------------------------------------------------
    if not args.quick:
        print("\n--- Phase 5: Coupling sensitivity ---")
        G_values = [2.0, 2.5, 3.0]
        sensitivity = []
        for G_m in G_values:
            G_t = -G_m
            # Quick coexistence
            coex = run_coexistence(step_mplbm, 64, G_m, cfg.tau, cfg.tau, 5000, "mplbm")
            if coex.get("nan", True):
                continue
            rl, rg = coex["rho1_liq"], coex["rho1_gas"]
            r2l, r2g = coex["rho2_liq"], coex["rho2_gas"]

            # Single radius droplet
            rm = run_droplet_case(step_mplbm, 64, 16.0, G_m, cfg.tau, cfg.tau,
                                  rl, rg, r2l, r2g, 5000, "mplbm", 2.0)
            rt = run_droplet_case(step_tl, 64, 16.0, G_t, cfg.tau, cfg.tau,
                                  rl, rg, r2l, r2g, 5000, "tensorlbm", 2.0)
            if rm.get("nan") or rt.get("nan"):
                continue
            sig_err = abs(rt["sigma"] - rm["sigma"]) / max(abs(rm["sigma"]), 1e-12) * 100
            row = {"G_mplbm": G_m, "sigma_mplbm": rm["sigma"],
                   "sigma_tensorlbm": rt["sigma"], "sigma_err_pct": sig_err}
            sensitivity.append(row)
            print(f"  G={G_m}: σ_m={rm['sigma']:.6f}  σ_t={rt['sigma']:.6f}  err={sig_err:.2f}%")

        results["sensitivity"] = sensitivity

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(ARTIFACTS / "results.json", "w") as fp:
        json.dump(results, fp, indent=2, default=str)
    print(f"\nResults saved to {ARTIFACTS / 'results.json'}")

    # ------------------------------------------------------------------
    # Generate figures
    # ------------------------------------------------------------------
    _generate_figures(results)

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    all_pass = True
    # Check 1: Laplace law R² > 0.99 for both solvers
    if r2_m is not None and r2_t is not None:
        r2_ok = r2_m > 0.99 and r2_t > 0.99
        print(f"  Laplace R² > 0.99: MPLBM={r2_m:.4f} TL={r2_t:.4f}  [{'PASS' if r2_ok else 'FAIL'}]")
        all_pass = all_pass and r2_ok

    # Check 2: σ cross-error < 15%
    if not math.isnan(sigma_err):
        sig_ok = sigma_err < 15
        print(f"  σ cross-error < 15%: {sigma_err:.2f}%  [{'PASS' if sig_ok else 'FAIL'}]")
        all_pass = all_pass and sig_ok

    # Check 3: Per-radius Δp errors < 15%
    if per_radius:
        max_dp_err = max(r["dp_err_pct"] for r in per_radius)
        dp_ok = max_dp_err < 15
        print(f"  Max per-radius Δp error < 15%: {max_dp_err:.2f}%  [{'PASS' if dp_ok else 'FAIL'}]")
        all_pass = all_pass and dp_ok

    # Check 4: No NaN
    nan_ok = not any(c.get("nan", False) for c in mplbm_cases + tl_cases)
    print(f"  No NaN: {'PASS' if nan_ok else 'FAIL'}")
    all_pass = all_pass and nan_ok

    print(f"\n  Overall: {'PASS' if all_pass else 'FAIL'}")

    results["verdict"] = {"pass": all_pass}
    with open(ARTIFACTS / "results.json", "w") as fp:
        json.dump(results, fp, indent=2, default=str)

    return all_pass


def _generate_figures(results: dict) -> None:
    """Generate comparison figures."""
    mplbm = results.get("droplets_mplbm", [])
    tl = results.get("droplets_tensorlbm", [])

    if not mplbm or not tl:
        return

    # Figure 1: Laplace law Δp vs 1/R
    fig, ax = plt.subplots(figsize=(7, 5))
    for cases, label, color, marker in [
        (mplbm, "MPLBM-UT (ref)", "blue", "o"),
        (tl, "TensorLBM", "red", "s"),
    ]:
        valid = [c for c in cases if c.get("R_eq", 0) > 0 and not c.get("nan", False)]
        if valid:
            inv_R = [1.0 / c["R_eq"] for c in valid]
            dp = [c["dp"] for c in valid]
            ax.scatter(inv_R, dp, color=color, marker=marker, s=60, label=label, zorder=5)

    # Fit lines
    fit_data = results.get("laplace_fit", {})
    for key, color, label in [("mplbm", "blue", "MPLBM-UT"), ("tensorlbm", "red", "TensorLBM")]:
        fd = fit_data.get(key, {})
        sig = fd.get("sigma")
        if sig:
            x = np.linspace(0.02, 0.12, 100)
            ax.plot(x, sig * x, "--", color=color, alpha=0.7, label=f"{label}: σ={sig:.4f}")

    ax.set_xlabel("1/R (1/lu)")
    ax.set_ylabel("Δp (lu)")
    ax.set_title("Laplace Law: Δp = σ/R (2D MCMP Shan-Chen)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(ARTIFACTS / "fig1_laplace_law.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: σ per radius comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    per_r = results.get("per_radius", [])
    if per_r:
        R_inits = [r["R_init"] for r in per_r]
        sig_m = [r["sigma_mplbm"] for r in per_r]
        sig_t = [r["sigma_tensorlbm"] for r in per_r]
        ax.plot(R_inits, sig_m, "o-", color="blue", label="MPLBM-UT", markersize=8)
        ax.plot(R_inits, sig_t, "s--", color="red", label="TensorLBM", markersize=8)
        # ±15% band around MPLBM-UT
        if sig_m:
            sig_ref = np.mean(sig_m)
            ax.axhline(sig_ref * 1.15, color="gray", ls=":", alpha=0.5, label="±15% band")
            ax.axhline(sig_ref * 0.85, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("Initial radius R (lu)")
    ax.set_ylabel("σ = Δp·R (lu)")
    ax.set_title("Surface Tension vs Radius")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(ARTIFACTS / "fig2_sigma_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 3: Spurious currents comparison
    fig, ax = plt.subplots(figsize=(7, 5))
    if mplbm and tl:
        R_m = [c["R_init"] for c in mplbm if not c.get("nan")]
        u_m = [c["u_max"] for c in mplbm if not c.get("nan")]
        R_t = [c["R_init"] for c in tl if not c.get("nan")]
        u_t = [c["u_max"] for c in tl if not c.get("nan")]
        ax.semilogy(R_m, u_m, "o-", color="blue", label="MPLBM-UT", markersize=8)
        ax.semilogy(R_t, u_t, "s--", color="red", label="TensorLBM", markersize=8)
    ax.set_xlabel("Initial radius R (lu)")
    ax.set_ylabel("|u|_max (lu)")
    ax.set_title("Spurious Currents")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.savefig(ARTIFACTS / "fig3_spurious_currents.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Figures saved to {ARTIFACTS}/")


if __name__ == "__main__":
    main()
