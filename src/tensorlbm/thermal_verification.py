"""Thermal convection quantitative verification suite for TensorLBM.

Campaign P96W2-N04: Rayleigh-Benard critical Ra + Nu-Ra scaling + dimensional
self-consistency + conservation limits.

Provides composable simulation runners and diagnostic functions for:
1. Rayleigh-Benard (bottom hot, top cold, periodic-x) — critical Ra, Nu-Ra
2. Differentially heated cavity (left hot, right cold, adiabatic top/bottom)
3. Dimensional consistency checks
4. Conservation / limit checks

All functions are designed to be called from pytest tests and from the
verification report generator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .d2q9 import OPPOSITE, C, W, equilibrium, macroscopic
from .solver import stream
from .thermal import (
    C5,
    W5,
    OPPOSITE5,
    buoyancy_force,
    collide_bgk_force,
    collide_thermal_bgk,
    nusselt_number,
    pre_streaming_bounce_back,
    temperature_equilibrium,
    temperature_stream,
    thermal_params,
)

__all__ = [
    "simulate_rayleigh_benard_2d",
    "simulate_dhc_2d",
    "nusselt_rb",
    "nusselt_dhc",
    "DE_VAHL_DAVIS",
    "RB_NU_REFERENCE",
    "RA_CRIT_CHANDRASEKHAR",
    "A_CRIT_CHANDRASEKHAR",
]

# ── Reference values ────────────────────────────────────────────────────────

#: Chandrasekhar (1961) critical Rayleigh number, rigid-rigid boundaries
RA_CRIT_CHANDRASEKHAR = 1707.7597
#: Chandrasekhar (1961) critical wavenumber
A_CRIT_CHANDRASEKHAR = 3.11632

#: de Vahl Davis (1983) benchmark: differentially heated square cavity, Pr=0.71
#: Keys are Ra, values are average Nu (both walls).
DE_VAHL_DAVIS: Dict[float, float] = {
    1e3: 1.118,
    1e4: 2.243,
    1e5: 4.519,
    1e6: 8.800,
}

#: 2D periodic RB steady-roll Nu reference (literature / high-res DNS).
#: For Pr=0.71, 2D RB with periodic-x:
#:   Ra=1e4: Nu ≈ 2.6-2.8  (Catton 1978; Xi et al. 2004)
#:   Ra=1e5: Nu ≈ 4.5-4.8
#:   Ra=1e6: Nu ≈ 8.0-8.5
#: These are approximate ranges; the GCI analysis determines convergence.
RB_NU_REFERENCE: Dict[float, float] = {
    1e4: 2.70,
    1e5: 4.65,
    1e6: 8.20,
}


# ── RB geometry helpers ─────────────────────────────────────────────────────

def _rb_wall_mask(ny: int, nx: int, device: torch.device) -> torch.Tensor:
    """RB: top/bottom no-slip, left/right periodic."""
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[-1, :] = True
    return mask


def _rb_thermal_bc(g: torch.Tensor, t_hot: float, t_cold: float) -> torch.Tensor:
    """RB thermal BC: bottom T_hot, top T_cold, periodic-x.

    Uses the same non-equilibrium correction as thermal.py's
    apply_temperature_boundaries, adapted for RB geometry.
    """
    w = W5.to(device=g.device, dtype=g.dtype)
    w3, w4 = float(w[3]), float(w[4])

    # Non-equilibrium from first fluid rows
    T_fluid_bot = g[:, 1, :].sum(dim=0)
    T_fluid_top = g[:, -2, :].sum(dim=0)
    neq_bot = g[3, 1, :] - w3 * T_fluid_bot  # upward-pointing from y=1
    neq_top = g[4, -2, :] - w4 * T_fluid_top  # downward-pointing from y=ny-2

    g = g.clone()
    # Bottom wall: equilibrium at T_hot
    g[:, 0, :] = w.view(5, 1) * t_hot
    g[3, 0, :] = g[3, 0, :] + neq_bot
    g[0, 0, :] = g[0, 0, :] - neq_bot

    # Top wall: equilibrium at T_cold
    g[:, -1, :] = w.view(5, 1) * t_cold
    g[4, -1, :] = g[4, -1, :] + neq_top
    g[0, -1, :] = g[0, -1, :] - neq_top

    return g


def nusselt_rb(T: torch.Tensor, ny: int, dT: float) -> dict[str, float]:
    """RB Nusselt number: Nu = -∂T/∂y · H / ΔT at bottom and top walls.

    H = ny - 1 (wall-to-wall distance, walls at nodes y=0 and y=ny-1).
    Uses second-order one-sided differences.
    """
    H = float(ny - 1)

    # Bottom: ∂T/∂y ≈ (-3T[0] + 4T[1] - T[2]) / 2
    dTdy_bot = (-1.5 * T[0, :] + 2.0 * T[1, :] - 0.5 * T[2, :])
    nu_bot = float(-dTdy_bot.mean().item()) * H / dT

    # Top: ∂T/∂y ≈ (3T[-1] - 4T[-2] + T[-3]) / 2
    dTdy_top = (1.5 * T[-1, :] - 2.0 * T[-2, :] + 0.5 * T[-3, :])
    nu_top = float(-dTdy_top.mean().item()) * H / dT

    return {"nu_bot": nu_bot, "nu_top": nu_top, "nu": 0.5 * (nu_bot + nu_top)}


def nusselt_dhc(T: torch.Tensor, nx: int, dT: float,
                t_hot: float = 1.0, t_cold: float = 0.0,
                mode: str = "grad2") -> dict[str, float]:
    """DHC Nusselt number: delegates to thermal.nusselt_number."""
    return nusselt_number(T, float(nx), dT, t_hot, t_cold, mode=mode)


# ── Rayleigh-Benard 2D simulation ───────────────────────────────────────────

@dataclass
class RBResult:
    """Result from a Rayleigh-Benard simulation."""
    nx: int
    ny: int
    ra: float
    pr: float
    tau: float
    tau_T: float
    nu_lat: float
    alpha_lat: float
    g_beta: float
    H: float
    dT: float
    ra_effective: float
    steps: int
    elapsed_s: float
    nu: float
    nu_bot: float
    nu_top: float
    u_max: float
    T_min: float
    T_max: float
    energy_imbalance: float
    nu_history: List[dict] = field(default_factory=list)
    T_field: Optional[torch.Tensor] = None
    ux_field: Optional[torch.Tensor] = None
    uy_field: Optional[torch.Tensor] = None


def simulate_rayleigh_benard_2d(
    nx: int,
    ny: int | None = None,
    ra: float = 1e4,
    pr: float = 0.71,
    tau: float = 0.6,
    t_hot: float = 1.0,
    t_cold: float = 0.0,
    steps: int = 100000,
    device: str = "cpu",
    report_every: int = 5000,
    enable_buoyancy: bool = True,
    seed_mode: str = "random",
    return_fields: bool = False,
) -> RBResult:
    """Rayleigh-Benard 2D DDF-LBM simulation.

    Geometry: 2D rectangle, bottom wall T_hot, top wall T_cold, periodic-x.
    Gravity downward, buoyancy upward (hot fluid rises).

    Args:
        nx: Horizontal grid points.
        ny: Vertical grid points (default = nx).
        ra: Rayleigh number.
        pr: Prandtl number.
        tau: Momentum relaxation time.
        t_hot: Bottom wall temperature.
        t_cold: Top wall temperature.
        steps: Total steps.
        device: Compute device.
        report_every: Reporting interval.
        enable_buoyancy: Enable buoyancy (False = pure conduction control).
        seed_mode: Perturbation mode: "random", "cosine", "none".
        return_fields: Whether to return final T, ux, uy fields.

    Returns:
        RBResult with Nu, diagnostics, and optionally fields.
    """
    dev = torch.device(device)
    if ny is None:
        ny = nx
    H = float(ny - 1)
    dT = t_hot - t_cold
    if not (dT > 0.0):
        raise ValueError(f"simulate_rayleigh_benard_2d: dT = t_hot - t_cold must be > 0, got {dT!r}")

    # Lattice parameters.  Ra = g_beta * dT * H^3 / (nu * alpha), so the
    # buoyancy coefficient carries a 1/dT: the same Ra must give the same flow
    # whatever the imposed temperature difference (dimensionless invariance).
    nu_lat = (tau - 0.5) / 3.0
    alpha_lat = nu_lat / pr
    tau_T = 3.0 * alpha_lat + 0.5
    g_beta = ra * nu_lat * alpha_lat / (dT * H**3) if enable_buoyancy else 0.0

    # Initialize: conductive base state + perturbation
    rho0 = torch.ones((ny, nx), device=dev)
    u0 = torch.zeros((ny, nx), device=dev)
    f = equilibrium(rho0, u0, u0)

    # Linear temperature profile
    T0 = torch.linspace(t_hot, t_cold, ny, device=dev).unsqueeze(1).expand(ny, nx).contiguous()

    # Add perturbation
    if seed_mode == "random":
        torch.manual_seed(42)
        T0 = T0 + 0.001 * (torch.rand(ny, nx, device=dev) - 0.5) * dT
    elif seed_mode == "cosine":
        # Single-mode perturbation: cos(kx) with k = 2*pi/nx (one roll pair)
        x = torch.arange(nx, device=dev, dtype=torch.float32)
        k = 2.0 * math.pi / nx
        pert = 0.001 * dT * torch.cos(k * x).unsqueeze(0).expand(ny, nx)
        T0 = T0 + pert
    # else "none": pure conductive start

    g = temperature_equilibrium(T0, u0, u0)

    wall = _rb_wall_mask(ny, nx, dev)

    nu_history: List[dict] = []
    t0 = time.time()

    for step in range(1, steps + 1):
        rho, ux, uy = macroscopic(f)
        T = g.sum(dim=0)

        # Buoyancy force
        if enable_buoyancy:
            F = buoyancy_force(rho, T, g_beta, t_ref=0.5 * (t_hot + t_cold))
        else:
            F = torch.zeros(2, ny, nx, device=dev)

        # Temperature: collide → stream → BC
        g = collide_thermal_bgk(g, T, ux, uy, tau_T)
        g = temperature_stream(g)
        g = _rb_thermal_bc(g, t_hot, t_cold)

        # Velocity: collide (Guo force) → pre-streaming bounce-back → stream
        f_pre = f
        f = collide_bgk_force(f, tau, F)
        f = pre_streaming_bounce_back(f_pre, f, wall)
        f = stream(f)

        # Diagnostics
        if step % report_every == 0 or step == steps:
            rho, ux, uy = macroscopic(f)
            T = g.sum(dim=0)
            nu_rb = nusselt_rb(T, ny, dT)
            u_max = float(torch.sqrt(ux**2 + uy**2).max().item())

            # Energy balance: bottom flux vs top flux
            dTdy_bot = (-1.5 * T[0, :] + 2.0 * T[1, :] - 0.5 * T[2, :]).mean().item()
            dTdy_top = (1.5 * T[-1, :] - 2.0 * T[-2, :] + 0.5 * T[-3, :]).mean().item()
            q_bot = -dTdy_bot
            q_top = -dTdy_top
            q_avg = 0.5 * (abs(q_bot) + abs(q_top))
            imbalance = abs(q_bot - q_top) / q_avg if q_avg > 1e-10 else 0.0

            nu_history.append({
                "step": step,
                "nu": nu_rb["nu"],
                "nu_bot": nu_rb["nu_bot"],
                "nu_top": nu_rb["nu_top"],
                "energy_imbalance": imbalance,
                "u_max": u_max,
            })

    elapsed = time.time() - t0

    # Final state
    rho, ux, uy = macroscopic(f)
    T = g.sum(dim=0)
    nu_final = nusselt_rb(T, ny, dT)

    dTdy_bot = (-1.5 * T[0, :] + 2.0 * T[1, :] - 0.5 * T[2, :]).mean().item()
    dTdy_top = (1.5 * T[-1, :] - 2.0 * T[-2, :] + 0.5 * T[-3, :]).mean().item()
    q_bot = -dTdy_bot
    q_top = -dTdy_top
    q_avg = 0.5 * (abs(q_bot) + abs(q_top))
    imbalance = abs(q_bot - q_top) / q_avg if q_avg > 1e-10 else 0.0

    ra_effective = (
        g_beta * dT * H**3 / (nu_lat * alpha_lat) if nu_lat * alpha_lat > 0.0 else 0.0
    )

    return RBResult(
        nx=nx, ny=ny, ra=ra, pr=pr, tau=tau, tau_T=tau_T,
        nu_lat=nu_lat, alpha_lat=alpha_lat, g_beta=g_beta, H=H,
        dT=dT, ra_effective=ra_effective,
        steps=steps, elapsed_s=round(elapsed, 1),
        nu=nu_final["nu"], nu_bot=nu_final["nu_bot"], nu_top=nu_final["nu_top"],
        u_max=float(torch.sqrt(ux**2 + uy**2).max().item()),
        T_min=float(T.min().item()), T_max=float(T.max().item()),
        energy_imbalance=imbalance,
        nu_history=nu_history,
        T_field=T.detach().cpu() if return_fields else None,
        ux_field=ux.detach().cpu() if return_fields else None,
        uy_field=uy.detach().cpu() if return_fields else None,
    )


# ── Differentially Heated Cavity 2D simulation ──────────────────────────────

@dataclass
class DHCResult:
    """Result from a differentially heated cavity simulation."""
    nx: int
    ra: float
    pr: float
    tau: float
    tau_T: float
    nu_lat: float
    alpha_lat: float
    g_beta: float
    H: float
    dT: float
    ra_effective: float
    steps: int
    elapsed_s: float
    nu_grad2: float
    nu_left_grad2: float
    nu_right_grad2: float
    nu_grad1: float
    nu_halfway: float
    u_max: float
    T_min: float
    T_max: float
    last_resid_u: float
    last_resid_T: float
    nu_history: List[dict] = field(default_factory=list)
    T_field: Optional[torch.Tensor] = None


def simulate_dhc_2d(
    nx: int,
    ra: float = 1e4,
    pr: float = 0.71,
    tau: float = 0.6,
    t_hot: float = 1.0,
    t_cold: float = 0.0,
    steps: int = 100000,
    device: str = "cpu",
    report_every: int = 10000,
    return_fields: bool = False,
) -> DHCResult:
    """Differentially heated cavity: left hot, right cold, top/bottom adiabatic.

    This wraps the existing thermal.simulate_natural_convection but returns
    a structured result and adds the T field option.
    """
    from .thermal import simulate_natural_convection

    res = simulate_natural_convection(
        nx=nx, ra=ra, pr=pr, tau=tau,
        t_hot=t_hot, t_cold=t_cold,
        steps=steps, device=device,
        seed_t="linear",
        report_every=report_every,
    )

    _dT = t_hot - t_cold
    _ra_eff = (
        res["g_beta"] * _dT * res["H"] ** 3 / (res["nu"] * res["alpha"])
        if res["nu"] * res["alpha"] > 0.0
        else 0.0
    )

    return DHCResult(
        nx=nx, ra=ra, pr=pr, tau=tau,
        tau_T=res["tau_T"], nu_lat=res["nu"], alpha_lat=res["alpha"],
        g_beta=res["g_beta"], H=res["H"],
        dT=_dT, ra_effective=_ra_eff,
        steps=res["steps"], elapsed_s=res["elapsed_s"],
        nu_grad2=res["nu_grad2"],
        nu_left_grad2=res["nu_left_grad2"],
        nu_right_grad2=res["nu_right_grad2"],
        nu_grad1=res["nu_grad1"],
        nu_halfway=res["nu_halfway"],
        u_max=res["u_max"],
        T_min=res["T_min"], T_max=res["T_max"],
        last_resid_u=res["last_resid_u"],
        last_resid_T=res["last_resid_T"],
        nu_history=res["nu_history"],
        T_field=None,  # Would need to modify simulate_natural_convection
    )
