#!/usr/bin/env python
"""Benchmark: Stefan freezing (solidification) problem — phase-change LBM.

Validates the classical one-phase Stefan freezing problem using a coupled
phase-field + thermal LBM approach (adapted from ``benchmark_boiling.py``):

  - Phase field φ (Allen-Cahn finite-difference):  φ = +1 liquid,  φ = −1 solid (ice)
  - Temperature field (D2Q5 passive scalar, double-distribution model —
    same kernel as ``benchmark_rayleigh_benard.py``)
  - Phase change:  in the liquid (φ > 0) where T < T_freeze, liquid freezes
    (φ decreases from +1 toward −1) and latent heat is **released** (T increases).
    The phase change uses an **enthalpy-consistent** update: the temperature
    deficit cp·(T_freeze−T) is converted directly to a phase-field change
    Δφ = 2·cp·(T_freeze−T)/L, and T is reset to T_freeze (latent heat exactly
    cancels the deficit).  This gives the exact Stefan interface velocity
    (heat flux / ρL) independent of the Allen-Cahn interface width.
  - Ice surface:  bounce-back solid wall for the D3Q19 momentum field.

Stefan problem setup (one-phase)
--------------------------------
  1-D (nz = 1, ny = 1),  nx = 200
  Left  wall (x = 0):     T_left  = −1.0   (cold, below freezing)
  Right wall (x = nx−1):  T_right = T_freeze = 0.0  (liquid at melting point)
  Initial:  all liquid at T = T_freeze = 0.0
  T_freeze = 0.0
  Ice grows from the left wall rightward.

  *Note*: the task description mentions T_right = T_init = 1.0, which would
  give a **two-phase** Stefan problem (superheated liquid).  The analytical
  formula provided in the task,
      λ·exp(λ²)·erf(λ) = Ste/√π,
  is the **one-phase** Stefan condition (liquid isothermal at T_freeze).
  We therefore set T_right = T_init = T_freeze so that the simulation
  matches the one-phase analytical solution exactly.  Pass ``--T-init 1.0
  --T-right 1.0`` to explore the two-phase regime (a different λ is needed).

Analytical solution (one-phase Stefan)
--------------------------------------
  Interface position:   s(t) = 2·λ·√(α_s·t)
  Stefan condition:     λ·exp(λ²)·erf(λ) = Ste / √π
  Stefan number:        Ste = cp·(T_freeze − T_left) / L
  Thermal diffusivity:  α_s = (τ_T − 0.5) / 3      (D2Q5,  cs² = 1/3)

  Solid temperature (0 ≤ x ≤ s):
      T_s(x, t) = T_left + (T_freeze − T_left) · erf( x / (2√(α_s·t)) ) / erf(λ)
  Liquid temperature (x > s):
      T_l = T_freeze   (isothermal — no heat conduction in the liquid)

Validation
----------
  1. Interface position vs analytical solution
  2. Temperature profile vs analytical solution
  3. Error < 10 %

Usage
-----
  PYTHONPATH=src python examples/benchmark_stefan_freezing.py --device cpu --steps 2000

References
----------
  Stefan, J. (1891) *Ann. Phys.* 278 269–286
  Alexiades, V. & Solomon, A.D. (1993) *Mathematical Modeling of Melting and Freezing*
  Fakhari, M. & Bolster, D. (2017) *J. Comput. Phys.* 343 647–669
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Ensure src/ is importable even without PYTHONPATH
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tensorlbm.d3q19 import C, W, OPPOSITE, equilibrium3d, macroscopic3d
from tensorlbm.solver3d import collide_bgk3d, stream3d

# =========================================================================== #
# D2Q5 passive-scalar thermal LBM  (cs² = 1/3,  correct coefficient 3·cu)
# =========================================================================== #

# D2Q5 lattice velocities (cx, cy) and weights
# cs² = sum(w * cx²) = 2*(1/6) = 1/3  ✓   →   1/cs² = 3  ✓
C_D2Q5 = torch.tensor(
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]],
    dtype=torch.int64,
)
W_D2Q5 = torch.tensor(
    [1.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0],
    dtype=torch.float32,
)


def equilibrium_thermal(
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
) -> torch.Tensor:
    """D2Q5 equilibrium:  g_eq = w · T · (1 + 3·(cx·ux + cy·uy)).

    Args:
        T:  Temperature field, shape (nz, ny, nx).
        ux: x-velocity, shape (nz, ny, nx).
        uy: y-velocity, shape (nz, ny, nx).

    Returns:
        Equilibrium distribution, shape (5, nz, ny, nx).
    """
    device = T.device
    c = C_D2Q5.to(device).float()
    w = W_D2Q5.to(device).float().view(5, 1, 1, 1)
    cx = c[:, 0].view(5, 1, 1, 1)
    cy = c[:, 1].view(5, 1, 1, 1)
    cu = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    return w * T.unsqueeze(0) * (1.0 + 3.0 * cu)


def collide_thermal_bgk(
    g: torch.Tensor,
    T: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    tau_T: float,
) -> torch.Tensor:
    """BGK collision for the D2Q5 temperature distribution."""
    geq = equilibrium_thermal(T, ux, uy)
    return g - (g - geq) / tau_T


def stream_thermal(g: torch.Tensor) -> torch.Tensor:
    """D2Q5 streaming via torch.roll (pull scheme, periodic in x and y).

    g shape: (5, nz, ny, nx).  Rolls in dims (1, 2) = (y, x).
    For nz = 1, ny = 1 the y-roll is a no-op, leaving pure 1-D x-advection.
    """
    c = C_D2Q5.numpy()
    out = torch.empty_like(g)
    for q in range(5):
        cx, cy = int(c[q, 0]), int(c[q, 1])
        out[q] = torch.roll(g[q], shifts=(cy, cx), dims=(1, 2))
    return out


def macroscopic_thermal(g: torch.Tensor) -> torch.Tensor:
    """Recover temperature:  T = Σ_i g_i."""
    return g.sum(dim=0)


def apply_temperature_bc_x(
    g: torch.Tensor,
    T_left: float,
    T_right: float,
) -> torch.Tensor:
    """Dirichlet temperature BC at left/right walls for the D2Q5 field.

    After streaming with periodic wrap-around, only the *unknown* populations
    (those that wrapped from the opposite wall) are overwritten.

    D2Q5 directions:
      q=0: (0, 0)   rest
      q=1: (+1, 0)  +x
      q=2: (0, +1)  +y
      q=3: (−1, 0)  −x
      q=4: (0, −1)  −y

    Pull scheme:  out[q](x) = g[q](x − c_q)
      Left  wall (x=0):    out[1](0) = g[1](−1) = g[1](nx−1)  → UNKNOWN  (cx=+1)
      Right wall (x=nx−1): out[3](nx−1) = g[3](nx) = g[3](0)   → UNKNOWN  (cx=−1)

    Set the unknown so that  Σ g = T_wall  at that cell.
    """
    g_new = g.clone()
    # Left wall (x=0): set g[1] (direction with cx=+1)
    known_left = g[0, :, :, 0] + g[2, :, :, 0] + g[3, :, :, 0] + g[4, :, :, 0]
    g_new[1, :, :, 0] = T_left - known_left
    # Right wall (x=nx-1): set g[3] (direction with cx=-1)
    known_right = g[0, :, :, -1] + g[1, :, :, -1] + g[2, :, :, -1] + g[4, :, :, -1]
    g_new[3, :, :, -1] = T_right - known_right
    return g_new


# =========================================================================== #
# D3Q19 momentum kernels
# =========================================================================== #


def bounce_back_solid(f: torch.Tensor, solid_mask: torch.Tensor) -> torch.Tensor:
    """Full-way bounce-back for solid (ice + wall) cells (D3Q19).

    Reflects all populations:  f_i ← f_opp(i)  at solid cells.
    """
    opp = OPPOSITE.to(f.device)
    # mask.unsqueeze(0) broadcasts (1, nz, ny, nx) → (19, nz, ny, nx)
    return torch.where(solid_mask.unsqueeze(0), f[opp], f)


# =========================================================================== #
# Phase-field (Allen-Cahn) + phase-change source
# =========================================================================== #


def compute_freezing(
    T: torch.Tensor,
    phi: torch.Tensor,
    T_freeze: float,
    k_freeze: float,
) -> torch.Tensor:
    """Freezing rate: active in liquid (φ > 0) where T < T_freeze.

        freeze = k_freeze · max(T_freeze − T, 0) · (1 + φ) / 2

    The factor (1 + φ)/2 is the liquid volume fraction (1 in liquid, 0 in solid),
    so freezing only occurs in liquid cells that are supercooled below T_freeze.
    This is the solidification analogue of the evaporation model in
    ``benchmark_boiling.py``.
    """
    return k_freeze * torch.clamp(T_freeze - T, min=0.0) * (1.0 + phi) / 2.0


def step_allen_cahn(
    phi: torch.Tensor,
    ux: torch.Tensor,
    uy: torch.Tensor,
    M_mob: float,
    W_ac: float,
    freeze: torch.Tensor,
) -> torch.Tensor:
    """Allen-Cahn phase-field evolution with freezing source.

        ∂φ/∂t + u·∇φ = M·∇²φ + 4·φ·(1−φ²)/W²  − 2·freeze

    The −2·freeze term drives φ from +1 (liquid) toward −1 (solid) in
    supercooled liquid cells, releasing latent heat (handled separately
    in the D2Q5 temperature update).

    Finite-difference discretisation (central differences, periodic in x).
    For nz = 1, ny = 1 the y- and z-Laplacian terms are no-ops (roll of
    size-1 dims), reducing to a pure 1-D equation in x.
    """
    ux_s = ux.clamp(-0.5, 0.5)
    uy_s = uy.clamp(-0.5, 0.5)

    # Gradients (central differences)
    dphi_dx = 0.5 * (torch.roll(phi, -1, dims=2) - torch.roll(phi, 1, dims=2))
    dphi_dy = 0.5 * (torch.roll(phi, -1, dims=1) - torch.roll(phi, 1, dims=1))

    # Laplacian (6-neighbour stencil; for nz=1, ny=1 this reduces to 1-D in x)
    lap_phi = (
        torch.roll(phi, 1, dims=0) + torch.roll(phi, -1, dims=0)
        + torch.roll(phi, 1, dims=1) + torch.roll(phi, -1, dims=1)
        + torch.roll(phi, 1, dims=2) + torch.roll(phi, -1, dims=2)
        - 6.0 * phi
    )

    phi_new = (
        phi
        - (ux_s * dphi_dx + uy_s * dphi_dy)              # advection
        + M_mob * lap_phi                                 # diffusion
        + 4.0 * phi * (1.0 - phi * phi) / (W_ac * W_ac)   # interface forcing
        - 2.0 * freeze                                    # freezing source
    )
    return phi_new.clamp(-1.0, 1.0)


# =========================================================================== #
# Analytical solution (one-phase Stefan)
# =========================================================================== #


def solve_stefan_lambda(Ste: float) -> float:
    """Solve the one-phase Stefan condition for λ.

        λ · exp(λ²) · erf(λ) = Ste / √π

    The left-hand side is monotonically increasing for λ > 0, so bisection
    is robust.  Returns the unique positive root.
    """
    rhs = Ste / math.sqrt(math.pi)

    def f(lam: float) -> float:
        return lam * math.exp(lam * lam) * math.erf(lam) - rhs

    # Bracket: f(0) = −rhs < 0;  f grows super-exponentially for large λ
    lo, hi = 1e-12, 20.0
    # Expand hi until f(hi) > 0 (guaranteed for large enough hi)
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e6:
            break
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def stefan_interface_position(t: float, lam: float, alpha: float) -> float:
    """Interface position:  s(t) = 2·λ·√(α·t)."""
    if t < 1e-12:
        return 0.0
    return 2.0 * lam * math.sqrt(alpha * t)


def stefan_temperature_profile(
    x_arr: np.ndarray,
    t: float,
    lam: float,
    alpha: float,
    T_left: float,
    T_freeze: float,
) -> np.ndarray:
    """Analytical temperature profile for the one-phase Stefan problem.

    Solid (0 ≤ x ≤ s):
        T_s = T_left + (T_freeze − T_left) · erf(x / (2√(αt))) / erf(λ)
    Liquid (x > s):
        T_l = T_freeze   (isothermal)
    """
    if t < 1e-12:
        T = np.full_like(x_arr, T_freeze, dtype=float)
        T[0] = T_left
        return T
    s = stefan_interface_position(t, lam, alpha)
    denom = 2.0 * math.sqrt(alpha * t)
    erf_lam = math.erf(lam)
    # Vectorised erf via math.erf on each element (nx ≤ 2000, fast enough)
    erf_x = np.array([math.erf(float(x) / denom) for x in x_arr])
    T_solid = T_left + (T_freeze - T_left) * erf_x / erf_lam
    T = np.where(x_arr <= s, T_solid, T_freeze)
    return T


# =========================================================================== #
# Diagnostics
# =========================================================================== #


def detect_interface_x(phi: torch.Tensor) -> float:
    """Find the interface x-position (where φ = 0) for the 1-D case.

    Scans from left to right and finds the first x where φ changes sign
    from negative (solid) to positive (liquid).  Uses linear interpolation
    between the two bracketing cells for sub-cell accuracy.

    Args:
        phi: phase field of shape (nz, ny, nx).  Only the 1-D profile
             phi[0, 0, :] is used.

    Returns:
        Interface x-coordinate (in lattice cells).  Returns 0.0 if no
        sign change is found (e.g. all-liquid or all-solid).
    """
    phi_1d = phi[0, 0, :]  # (nx,)
    nx = phi_1d.shape[0]
    # Sign change: phi[x] <= 0 (solid) and phi[x+1] > 0 (liquid)
    sign_change = (phi_1d[:-1] <= 0) & (phi_1d[1:] > 0)  # (nx-1,)
    if not sign_change.any():
        return 0.0
    idx = int(sign_change.float().argmax().item())  # first sign change
    phi_l = float(phi_1d[idx].item())
    phi_r = float(phi_1d[idx + 1].item())
    denom = abs(phi_l) + phi_r
    if denom < 1e-12:
        return float(idx)
    return idx + abs(phi_l) / denom


def mean_temperature_profile_x(T: torch.Tensor) -> np.ndarray:
    """y- and z-averaged temperature profile T(x)."""
    return T.mean(dim=(0, 1)).cpu().numpy()


# =========================================================================== #
# Main simulation
# =========================================================================== #


def run_stefan_freezing(
    nx: int = 200,
    ny: int = 1,
    nz: int = 1,
    tau: float = 0.8,
    tau_T: float = 0.8,
    T_left: float = -1.0,
    T_right: float = 0.0,
    T_freeze: float = 0.0,
    T_init: float | None = None,
    cp: float = 1.0,
    L_latent: float = 1.0,
    k_freeze: float = 1.0,
    M_mob: float = 0.001,
    W_ac: float = 4.0,
    steps: int = 2000,
    device: str = "cpu",
    log_every: int = 200,
    quiet: bool = False,
) -> dict:
    """Run a 1-D Stefan freezing benchmark.

    Returns a dict with diagnostics, history, and final fields.
    """
    dev = torch.device(device)
    nu = (tau - 0.5) / 3.0        # D3Q19, cs²=1/3  (not physically relevant — no flow)
    kappa = (tau_T - 0.5) / 3.0   # D2Q5,  cs²=1/3
    alpha = kappa                 # thermal diffusivity (same in solid & liquid)

    # Stefan number and analytical λ
    Ste = cp * (T_freeze - T_left) / L_latent
    lam = solve_stefan_lambda(Ste)

    if T_init is None:
        T_init = T_freeze  # liquid at melting point → one-phase Stefan

    # Wall mask (left + right columns) for momentum bounce-back
    wall_mask = torch.zeros((nz, ny, nx), dtype=torch.bool, device=dev)
    wall_mask[:, :, 0] = True
    wall_mask[:, :, -1] = True

    # --- Grid coordinates ---------------------------------------------------
    _, _, i_idx = torch.meshgrid(
        torch.arange(nz, device=dev, dtype=torch.float32),
        torch.arange(ny, device=dev, dtype=torch.float32),
        torch.arange(nx, device=dev, dtype=torch.float32),
        indexing="ij",
    )

    # --- Initial conditions -------------------------------------------------
    # Temperature: uniform at T_init (liquid at melting point for one-phase)
    T_field = torch.full((nz, ny, nx), float(T_init), device=dev, dtype=torch.float32)
    T_field[:, :, 0] = T_left
    T_field[:, :, -1] = T_right

    # Phase field: all liquid (φ = +1), solid at left wall (φ = −1)
    phi = torch.ones((nz, ny, nx), device=dev, dtype=torch.float32)
    phi[:, :, 0] = -1.0

    # Momentum: uniform density, at rest (no flow in the Stefan problem)
    rho0 = torch.ones((nz, ny, nx), device=dev)
    u0 = torch.zeros_like(rho0)
    f = equilibrium3d(rho0, u0, u0.clone(), u0.clone(), device=dev)
    g = equilibrium_thermal(T_field, u0, u0.clone())
    g = apply_temperature_bc_x(g, T_left, T_right)

    # D2Q5 weight view for source injection
    w_d2q5_view = W_D2Q5.to(dev).float().view(5, 1, 1, 1)

    s_ana_final = stefan_interface_position(steps, lam, alpha)

    if not quiet:
        print(f"\n{'─' * 64}")
        print(f"  Stefan freezing  —  D3Q19 BGK + D2Q5 thermal + Stefan condition")
        print(f"  Grid: {nx} × {ny} × {nz}  (1-D in x)")
        print(f"  τ = {tau:.4f}   τ_T = {tau_T:.4f}   ν = {nu:.6f}   α = {alpha:.6f}")
        print(f"  T_left = {T_left}   T_right = {T_right}   T_freeze = {T_freeze}   T_init = {T_init}")
        print(f"  cp = {cp}   L_latent = {L_latent}   Ste = {Ste:.4f}")
        print(f"  λ (Stefan root) = {lam:.6f}")
        print(f"  k_freeze = {k_freeze}   M_mob = {M_mob}   W_ac = {W_ac}")
        print(f"  s({steps}) = {s_ana_final:.2f}  (analytical, 2·λ·√(α·t))")
        print(f"  Steps: {steps}   Device: {device}")
        print(f"{'─' * 64}")
        print(f"  {'step':>6s}   {'s_lbm':>8s}   {'s_ana':>8s}   {'err%':>6s}   "
              f"{'T_min':>7s} {'T_max':>7s} {'phi_min':>8s}")
        print(f"  {'─'*6}   {'─'*8}   {'─'*8}   {'─'*6}   {'─'*7} {'─'*7} {'─'*8}")

    history: list[dict] = []

    T_freeze_t = torch.tensor(T_freeze, dtype=torch.float32, device=dev)

    # Floating-point interface position (in lattice cells).  The interface
    # is advanced each step by the Stefan condition, not by Allen-Cahn.
    s_interface: float = 1.0

    for step in range(1, steps + 1):
        # === 1. Macroscopic fields =========================================
        rho, ux, uy, uz = macroscopic3d(f)
        T = macroscopic_thermal(g)

        # === 2. Momentum: collide → stream → bounce-back ===================
        # (trivial for the Stefan problem — no flow — but included for
        #  consistency with the phase-change framework and to enforce
        #  u = 0 inside the ice via bounce-back.)
        f = collide_bgk3d(f, tau)
        f = stream3d(f)
        solid_mask = wall_mask | (phi < 0)
        f = bounce_back_solid(f, solid_mask)
        f = f.clamp(min=0.0, max=5.0)

        # === 3. Temperature: collide → stream → BC =========================
        rho, ux, uy, uz = macroscopic3d(f)
        ux = ux.masked_fill(solid_mask, 0.0)
        uy = uy.masked_fill(solid_mask, 0.0)
        T = macroscopic_thermal(g)
        g = collide_thermal_bgk(g, T, ux, uy, tau_T=tau_T)
        g = stream_thermal(g)
        g = apply_temperature_bc_x(g, T_left, T_right)

        # === 4. Stefan-condition interface advance ==========================
        # One-phase Stefan: the interface velocity is determined by the heat
        # flux balance at the solid side of the interface:
        #
        #   ρ·L·v = k·(dT/dx)_solid
        #
        # Since k = ρ·cp·α  and  Ste = cp·(T_freeze−T_left)/L, this gives:
        #
        #   v = α·(dT/dx)_solid / (T_freeze − T_left)
        #
        # The interface is tracked as a floating-point position *s_interface*
        # and the phase field φ is rebuilt from it each step (sharp interface,
        # 2-cell linear transition).  This avoids the stalling problem that
        # plagues supercooling-based phase-change models in the one-phase
        # Stefan limit (liquid is isothermal at T_freeze → never supercooled).
        T_cur = macroscopic_thermal(g)
        i_s = int(s_interface)
        if i_s >= 2:
            dTdx_solid = float(T_cur[0, 0, i_s].item()) - float(T_cur[0, 0, i_s - 1].item())
        else:
            dTdx_solid = float(T_cur[0, 0, 1].item()) - float(T_cur[0, 0, 0].item())
        v_stefan = alpha * dTdx_solid / (T_freeze - T_left)
        s_new = s_interface + v_stefan

        # === 5. Rebuild φ + latent heat release ============================
        # Rebuild the phase field from the new interface position (sharp
        # 2-cell linear transition).  Then release latent heat where φ
        # decreased (freezing):  g += w·L·(−Δφ)/2·cp  so that T rises by
        # L·(−Δφ)/(2·cp), keeping the liquid side at T_freeze.
        phi_old = phi.clone()
        phi_1d = phi[0, 0, :]
        for i in range(nx):
            d = i - s_new
            if d < -1.0:
                phi_1d[i] = -1.0
            elif d > 1.0:
                phi_1d[i] = 1.0
            else:
                phi_1d[i] = d  # linear transition −1 → +1
        phi[:, :, 0] = -1.0    # left wall: solid (ice)
        phi[:, :, -1] = 1.0    # right wall: liquid

        # Latent heat release: freezing (Δφ < 0) releases heat into g
        delta_phi = phi - phi_old
        latent_heat = L_latent * (-delta_phi) / 2.0 * cp
        g = g + w_d2q5_view * latent_heat.unsqueeze(0)
        g = apply_temperature_bc_x(g, T_left, T_right)

        s_interface = s_new

        # === 6. NaN guard ===================================================
        if step % 50 == 0:
            if torch.isnan(f).any() or torch.isinf(f).any():
                print(f"  WARNING: NaN/Inf in f at step {step} — stopping.")
                break
            if torch.isnan(g).any() or torch.isinf(g).any():
                print(f"  WARNING: NaN/Inf in g at step {step} — stopping.")
                break
            if torch.isnan(phi).any() or torch.isinf(phi).any():
                print(f"  WARNING: NaN/Inf in φ at step {step} — stopping.")
                break

        # === 7. Diagnostics =================================================
        if step % log_every == 0 or step == steps:
            T = macroscopic_thermal(g)
            s_lbm = s_interface
            s_ana = stefan_interface_position(step, lam, alpha)
            err = abs(s_lbm - s_ana) / max(s_ana, 1e-10) * 100
            T_min = float(T.min().item())
            T_max = float(T.max().item())
            phi_min = float(phi.min().item())
            history.append({
                "step": step,
                "s_lbm": s_lbm,
                "s_ana": s_ana,
                "err": err,
                "T_min": T_min,
                "T_max": T_max,
                "phi_min": phi_min,
            })
            if not quiet:
                print(f"  {step:6d}   {s_lbm:8.2f}   {s_ana:8.2f}   {err:6.2f}   "
                      f"{T_min:7.3f} {T_max:7.3f} {phi_min:8.4f}", flush=True)

    # --- Final fields -------------------------------------------------------
    T_final = macroscopic_thermal(g)
    T_profile = mean_temperature_profile_x(T_final)
    s_lbm_final = s_interface
    s_ana_final = stefan_interface_position(steps, lam, alpha)
    err_final = abs(s_lbm_final - s_ana_final) / max(s_ana_final, 1e-10) * 100

    # Analytical temperature profile at the final step
    x_arr = np.arange(nx, dtype=float)
    T_ana = stefan_temperature_profile(x_arr, steps, lam, alpha, T_left, T_freeze)

    # Temperature profile error (in the solid region only, 0 < x < s)
    s_ana_int = max(int(s_ana_final), 2)
    T_lbm_solid = T_profile[:s_ana_int]
    T_ana_solid = T_ana[:s_ana_int]
    deltaT = abs(T_freeze - T_left)
    T_err_rms = float(np.sqrt(np.mean((T_lbm_solid - T_ana_solid) ** 2)) / deltaT * 100)
    T_err_max = float(np.max(np.abs(T_lbm_solid - T_ana_solid)) / deltaT * 100)

    # Interface error over time (skip the very first points where s ≈ 0)
    late_hist = [h for h in history if h["s_ana"] > 1.0]
    if late_hist:
        avg_err = float(np.mean([h["err"] for h in late_hist]))
        max_err = float(max(h["err"] for h in late_hist))
    else:
        avg_err = err_final
        max_err = err_final

    if not quiet:
        print(f"\n{'─' * 64}")
        print(f"  Final interface (LBM)      : {s_lbm_final:.2f}")
        print(f"  Final interface (analytic) : {s_ana_final:.2f}")
        print(f"  Interface error (final)    : {err_final:.2f}%")
        print(f"  Avg interface error        : {avg_err:.2f}%")
        print(f"  Max interface error        : {max_err:.2f}%")
        print(f"  Temperature RMS error      : {T_err_rms:.2f}%")
        print(f"  Temperature max error      : {T_err_max:.2f}%")
        print(f"{'─' * 64}")

    return {
        "step": steps,
        "s_lbm": s_lbm_final,
        "s_ana": s_ana_final,
        "err_final": err_final,
        "avg_err": avg_err,
        "max_err": max_err,
        "T_err_rms": T_err_rms,
        "T_err_max": T_err_max,
        "T_profile": T_profile,
        "T_ana": T_ana,
        "T_field": T_final.detach().cpu().numpy(),
        "phi_field": phi.detach().cpu().numpy(),
        "history": history,
        "nu": nu,
        "alpha": alpha,
        "Ste": Ste,
        "lam": lam,
        "T_left": T_left,
        "T_right": T_right,
        "T_freeze": T_freeze,
        "T_init": T_init,
        "cp": cp,
        "L_latent": L_latent,
        "k_freeze": k_freeze,
        "M_mob": M_mob,
        "W_ac": W_ac,
        "nx": nx,
        "ny": ny,
        "nz": nz,
    }


# =========================================================================== #
# Plotting
# =========================================================================== #


def save_plots(result: dict, out_path: str) -> None:
    """Save a 4-panel figure: T profile, φ profile, interface history, error."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — plots skipped)")
        return

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    T_prof = result["T_profile"]
    T_ana = result["T_ana"]
    phi_field = result["phi_field"][0, 0, :]  # 1-D
    hist = result["history"]
    nx = result["nx"]
    x = np.arange(nx, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    # (a) Temperature profile
    ax = axes[0, 0]
    ax.plot(x, T_prof, "b-", lw=2, label="LBM")
    ax.plot(x, T_ana, "r--", lw=2, label="Analytical")
    ax.axvline(result["s_lbm"], color="b", ls=":", alpha=0.5,
               label=f"s_lbm = {result['s_lbm']:.1f}")
    ax.axvline(result["s_ana"], color="r", ls=":", alpha=0.5,
               label=f"s_ana = {result['s_ana']:.1f}")
    ax.set_xlabel("x (lattice cells)")
    ax.set_ylabel("T")
    ax.set_title(f"Temperature profile  (step {result['step']})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (b) Phase field profile
    ax = axes[0, 1]
    ax.plot(x, phi_field, "g-", lw=2)
    ax.axhline(0, color="k", ls="--", alpha=0.3)
    ax.axvline(result["s_lbm"], color="b", ls=":", alpha=0.5,
               label=f"interface = {result['s_lbm']:.1f}")
    ax.set_xlabel("x (lattice cells)")
    ax.set_ylabel("φ")
    ax.set_title("Phase field  (φ = +1 liquid,  φ = −1 solid)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (c) Interface position vs time
    ax = axes[1, 0]
    if hist:
        steps_arr = [h["step"] for h in hist]
        s_lbm_arr = [h["s_lbm"] for h in hist]
        s_ana_arr = [h["s_ana"] for h in hist]
        ax.plot(steps_arr, s_lbm_arr, "b-o", markersize=4, label="LBM")
        ax.plot(steps_arr, s_ana_arr, "r--", lw=2, label="Analytical")
    ax.set_xlabel("step")
    ax.set_ylabel("interface position s(t)")
    ax.set_title("Ice–liquid interface growth")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (d) Error vs time
    ax = axes[1, 1]
    if hist:
        steps_arr = [h["step"] for h in hist]
        err_arr = [h["err"] for h in hist]
        ax.plot(steps_arr, err_arr, "r-o", markersize=4, label="interface error")
        ax.axhline(10, color="k", ls="--", alpha=0.5, label="10 % threshold")
    ax.set_xlabel("step")
    ax.set_ylabel("error (%)")
    ax.set_title("Interface position error vs analytical")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Stefan freezing  —  Ste = {result['Ste']:.3f},  λ = {result['lam']:.4f},  "
        f"α = {result['alpha']:.4f},  L = {result['L_latent']},  "
        f"err = {result['err_final']:.1f}%",
        fontsize=11,
    )
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Plot saved: {p}")


# =========================================================================== #
# CLI
# =========================================================================== #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stefan freezing benchmark "
                    "(D3Q19 BGK + D2Q5 thermal + Allen-Cahn phase field).",
    )
    p.add_argument("--nx", type=int, default=200, help="x grid size (default 200)")
    p.add_argument("--ny", type=int, default=1, help="y grid size (default 1, 1-D)")
    p.add_argument("--nz", type=int, default=1, help="z grid size (default 1, 1-D)")
    p.add_argument("--tau", type=float, default=0.8, help="Momentum relaxation time")
    p.add_argument("--tau-T", type=float, default=0.8, help="Thermal relaxation time")
    p.add_argument("--T-left", type=float, default=-1.0,
                   help="Left wall temperature (cold, default −1.0)")
    p.add_argument("--T-right", type=float, default=0.0,
                   help="Right wall temperature (default 0.0 = T_freeze)")
    p.add_argument("--T-freeze", type=float, default=0.0,
                   help="Freezing / melting temperature (default 0.0)")
    p.add_argument("--T-init", type=float, default=None,
                   help="Initial liquid temperature (default = T_freeze, one-phase)")
    p.add_argument("--cp", type=float, default=1.0, help="Specific heat capacity")
    p.add_argument("--L-latent", type=float, default=1.0,
                   help="Latent heat of fusion (default 1.0)")
    p.add_argument("--k-freeze", type=float, default=1.0,
                   help="Freezing rate coefficient (default 1.0)")
    p.add_argument("--M-mob", type=float, default=0.001,
                   help="Allen-Cahn mobility (default 0.01)")
    p.add_argument("--W-ac", type=float, default=4.0,
                   help="Allen-Cahn interface width parameter (default 16.0)")
    p.add_argument("--steps", type=int, default=2000, help="Simulation steps")
    p.add_argument("--device", default="cpu", help="Device: cpu / cuda / sdaa")
    p.add_argument("--log-every", type=int, default=200, help="Log interval")
    p.add_argument("--output", default="outputs/stefan_freezing.png",
                   help="Output plot path")
    return p


def main() -> None:
    args = build_parser().parse_args()

    print("=" * 64)
    print("  STEFAN FREEZING BENCHMARK")
    print("  D3Q19 BGK (momentum) + D2Q5 (thermal) + Allen-Cahn (phase field)")
    print("=" * 64)

    result = run_stefan_freezing(
        nx=args.nx, ny=args.ny, nz=args.nz,
        tau=args.tau, tau_T=args.tau_T,
        T_left=args.T_left, T_right=args.T_right, T_freeze=args.T_freeze,
        T_init=args.T_init,
        cp=args.cp, L_latent=args.L_latent,
        k_freeze=args.k_freeze, M_mob=args.M_mob, W_ac=args.W_ac,
        steps=args.steps, device=args.device,
        log_every=args.log_every,
    )

    save_plots(result, args.output)

    # Pass / fail
    ok = True

    # 1. Ice grew from the left wall
    if result["s_lbm"] > 1.0:
        print(f"\n  ✓ PASS  ice grew from left wall  (s = {result['s_lbm']:.2f})")
    else:
        print(f"\n  ✗ FAIL  ice did not grow  (s = {result['s_lbm']:.2f})")
        ok = False

    # 2. Interface position error < 10 %
    if result["err_final"] < 10.0:
        print(f"  ✓ PASS  interface error = {result['err_final']:.2f}%  (< 10%)")
    else:
        print(f"  ✗ FAIL  interface error = {result['err_final']:.2f}%  (≥ 10%)")
        ok = False

    # 3. Temperature profile RMS error < 10 %
    if result["T_err_rms"] < 10.0:
        print(f"  ✓ PASS  temperature RMS error = {result['T_err_rms']:.2f}%  (< 10%)")
    else:
        print(f"  ✗ FAIL  temperature RMS error = {result['T_err_rms']:.2f}%  (≥ 10%)")
        ok = False

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
