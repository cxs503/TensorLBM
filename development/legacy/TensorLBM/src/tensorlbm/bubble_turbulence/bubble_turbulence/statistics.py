"""Statistics and vortex identification for bubble-turbulence interaction.

Provides:
  - ChannelStatistics: plane-averaged turbulent statistics
  - Q-criterion vortex identification
  - Burst event detection
  - Reynolds stress budgets
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch


def compute_velocity_gradients(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> dict:
    """Compute velocity gradient tensor components using central differences.

    Input fields: (nz, ny, nx)
    Periodic in x and z, non-periodic in y.

    Returns dict with keys: duxdx, duxdy, duxdz, duydx, duydy, duydz, duzdx, duzdy, duzdz
    """
    # Periodic directions: x (dim=2), z (dim=0)
    # Non-periodic: y (dim=1)

    def ddx(f):
        return (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2.0 * dx)

    def ddy(f):
        df = torch.zeros_like(f)
        df[:, 1:-1, :] = (f[:, 2:, :] - f[:, :-2, :]) / (2.0 * dy)
        # One-sided at walls
        df[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
        df[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
        return df

    def ddz(f):
        return (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2.0 * dz)

    return {
        "duxdx": ddx(ux), "duxdy": ddy(ux), "duxdz": ddz(ux),
        "duydx": ddx(uy), "duydy": ddy(uy), "duydz": ddz(uy),
        "duzdx": ddx(uz), "duzdy": ddy(uz), "duzdz": ddz(uz),
    }


def compute_q_criterion(
    ux: torch.Tensor,
    uy: torch.Tensor,
    uz: torch.Tensor,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
) -> torch.Tensor:
    """Compute Q-criterion for vortex identification.

    Q = 0.5 * (||Omega||^2 - ||S||^2)

    where Omega is the antisymmetric part (rotation) and S is the symmetric
    part (strain) of the velocity gradient tensor.

    Positive Q indicates vortex cores.

    Returns: (nz, ny, nx) Q field.
    """
    grad = compute_velocity_gradients(ux, uy, uz, dx, dy, dz)

    # Strain rate tensor S_ij = 0.5 * (dui/dxj + duj/dxi)
    S11 = grad["duxdx"]
    S22 = grad["duydy"]
    S33 = grad["duzdz"]
    S12 = 0.5 * (grad["duxdy"] + grad["duydx"])
    S13 = 0.5 * (grad["duxdz"] + grad["duzdx"])
    S23 = 0.5 * (grad["duydz"] + grad["duzdy"])

    # Rotation tensor Omega_ij = 0.5 * (dui/dxj - duj/dxi)
    O12 = 0.5 * (grad["duxdy"] - grad["duydx"])
    O13 = 0.5 * (grad["duxdz"] - grad["duzdx"])
    O23 = 0.5 * (grad["duydz"] - grad["duzdy"])

    # ||S||^2 = S_ij * S_ij
    S_sq = (S11**2 + S22**2 + S33**2 +
            2.0 * (S12**2 + S13**2 + S23**2))

    # ||Omega||^2 = Omega_ij * Omega_ij
    O_sq = 2.0 * (O12**2 + O13**2 + O23**2)

    Q = 0.5 * (O_sq - S_sq)
    return Q


def identify_vortex_cores(
    Q: torch.Tensor,
    threshold: Optional[float] = None,
) -> torch.Tensor:
    """Identify vortex cores using Q-criterion.

    Parameters
    ----------
    Q : (nz, ny, nx) Q-criterion field
    threshold : float, optional
        Q threshold for vortex identification.
        Default: 0.1 * Q_max (10% of maximum Q).

    Returns
    -------
    (nz, ny, nx) boolean mask of vortex cores.
    """
    if threshold is None:
        threshold = 0.1 * Q.max().item()
    return Q > threshold


def count_vortex_events(
    Q: torch.Tensor,
    y_plus: torch.Tensor,
    threshold: Optional[float] = None,
    y_plus_bins: Optional[torch.Tensor] = None,
) -> dict:
    """Count vortex events as a function of wall-normal position.

    Returns dict with:
      - n_vortices: total number of vortex cells
      - n_vortices_profile: (ny,) count per wall-normal layer
      - mean_Q_profile: (ny,) mean Q per layer
      - burst_frequency: estimated burst frequency from Q events
    """
    if threshold is None:
        threshold = 0.1 * Q.max().item()

    vortex_mask = Q > threshold  # (nz, ny, nx)
    nz, ny, nx = Q.shape

    # Profile: count per y-layer
    n_profile = vortex_mask.sum(dim=(0, 2)).float()  # (ny,)
    mean_Q = Q.mean(dim=(0, 2))  # (ny,)

    # Burst frequency estimate: number of Q events per unit area per unit time
    # (normalized by total cells in each layer)
    burst_freq = n_profile / (nx * nz)

    return {
        "n_vortices": int(vortex_mask.sum().item()),
        "n_vortices_profile": n_profile.cpu().numpy(),
        "mean_Q_profile": mean_Q.cpu().numpy(),
        "burst_frequency": burst_freq.cpu().numpy(),
        "threshold": threshold,
    }


def detect_burst_events(
    ux: torch.Tensor,
    uy: torch.Tensor,
    y_plus_threshold: float = 30.0,
    u_tau: float = 0.05,
    nu: float = 0.01,
) -> dict:
    """Detect burst/sweep events using quadrant analysis.

    Quadrant classification of (u', v') fluctuations:
      Q1 (outward interaction): u'>0, v'>0
      Q2 (ejection):            u'<0, v'>0  (dominant in near-wall)
      Q3 (inward interaction):  u'<0, v'<0
      Q4 (sweep):               u'>0, v'<0  (dominant in near-wall)

    Parameters
    ----------
    ux : (nz, ny, nx) streamwise velocity
    uy : (nz, ny, nx) wall-normal velocity
    y_plus_threshold : float
        Only consider events below this y+ (near-wall region).
    u_tau : float
        Friction velocity for wall units.
    nu : float
        Kinematic viscosity.

    Returns
    -------
    dict with quadrant fractions and event counts.
    """
    ny = ux.shape[1]

    # Compute fluctuations
    u_mean = ux.mean(dim=(0, 2))  # (ny,)
    v_mean = uy.mean(dim=(0, 2))

    u_prime = ux - u_mean.view(1, -1, 1)
    v_prime = uy - v_mean.view(1, -1, 1)

    # Wall-normal coordinate
    y = torch.arange(ny, dtype=torch.float64, device=ux.device)
    y_from_wall = torch.minimum(y, torch.tensor(ny - 1, dtype=torch.float64, device=ux.device) - y)
    y_plus = y_from_wall * u_tau / nu

    # Near-wall mask
    near_wall = y_plus < y_plus_threshold  # (ny,)
    if near_wall.sum() == 0:
        return {"Q1_frac": 0, "Q2_frac": 0, "Q3_frac": 0, "Q4_frac": 0,
                "n_events": 0}

    # Extract near-wall fluctuations
    u_nw = u_prime[:, near_wall, :]  # (nz, n_nw, nx)
    v_nw = v_prime[:, near_wall, :]

    # Flatten
    u_flat = u_nw.reshape(-1)
    v_flat = v_nw.reshape(-1)

    # Quadrant classification
    q1 = (u_flat > 0) & (v_flat > 0)
    q2 = (u_flat < 0) & (v_flat > 0)
    q3 = (u_flat < 0) & (v_flat < 0)
    q4 = (u_flat > 0) & (v_flat < 0)

    n_total = len(u_flat)
    n_events = int((q1 | q2 | q3 | q4).sum().item())

    # Reynolds stress contribution from each quadrant
    # <u'v'>_Qi = sum(u' * v' for events in Qi) / N
    uv_q1 = (u_flat[q1] * v_flat[q1]).sum().item() / max(n_total, 1)
    uv_q2 = (u_flat[q2] * v_flat[q2]).sum().item() / max(n_total, 1)
    uv_q3 = (u_flat[q3] * v_flat[q3]).sum().item() / max(n_total, 1)
    uv_q4 = (u_flat[q4] * v_flat[q4]).sum().item() / max(n_total, 1)

    return {
        "Q1_frac": float(q1.sum().item()) / max(n_total, 1),
        "Q2_frac": float(q2.sum().item()) / max(n_total, 1),
        "Q3_frac": float(q3.sum().item()) / max(n_total, 1),
        "Q4_frac": float(q4.sum().item()) / max(n_total, 1),
        "Q1_uv": uv_q1,
        "Q2_uv": uv_q2,
        "Q3_uv": uv_q3,
        "Q4_uv": uv_q4,
        "n_events": n_events,
        "n_total": n_total,
    }


class ChannelStatistics:
    """Accumulates and analyzes turbulent channel statistics.

    Supports comparison between single-phase and two-phase (bubbly) flows.
    """

    def __init__(self, ny: int, u_tau: float, nu: float, device: str = "cpu"):
        self.ny = ny
        self.u_tau = u_tau
        self.nu = nu
        self.device = device

        self._sum_u = torch.zeros(ny, device=device, dtype=torch.float64)
        self._sum_u2 = torch.zeros(ny, device=device, dtype=torch.float64)
        self._sum_v2 = torch.zeros(ny, device=device, dtype=torch.float64)
        self._sum_w2 = torch.zeros(ny, device=device, dtype=torch.float64)
        self._sum_uv = torch.zeros(ny, device=device, dtype=torch.float64)
        self._n = 0

    def accumulate(self, ux: torch.Tensor, uy: torch.Tensor, uz: torch.Tensor):
        """Accumulate one sample.

        ux, uy, uz: (nz, ny, nx) velocity fields.
        """
        u_mean = ux.mean(dim=(0, 2)).double()
        u_prime = (ux - u_mean.view(1, -1, 1)).double()
        v_prime = (uy - uy.mean(dim=(0, 2)).view(1, -1, 1)).double()
        w_prime = (uz - uz.mean(dim=(0, 2)).view(1, -1, 1)).double()

        self._sum_u += u_mean
        self._sum_u2 += (u_prime**2).mean(dim=(0, 2))
        self._sum_v2 += (v_prime**2).mean(dim=(0, 2))
        self._sum_w2 += (w_prime**2).mean(dim=(0, 2))
        self._sum_uv += (u_prime * v_prime).mean(dim=(0, 2))
        self._n += 1

    def get_profiles(self) -> dict:
        """Return wall-normal profiles in wall units."""
        n = max(self._n, 1)
        ny = self.ny

        y = torch.arange(ny, dtype=torch.float64)
        y_from_wall = torch.minimum(y, torch.tensor(ny - 1, dtype=torch.float64) - y)
        y_plus = y_from_wall * self.u_tau / self.nu

        U = self._sum_u / n
        urms = torch.sqrt(self._sum_u2 / n)
        vrms = torch.sqrt(self._sum_v2 / n)
        wrms = torch.sqrt(self._sum_w2 / n)
        uv = self._sum_uv / n

        return {
            "y_plus": y_plus.numpy(),
            "U_plus": (U / self.u_tau).cpu().numpy(),
            "urms_plus": (urms / self.u_tau).cpu().numpy(),
            "vrms_plus": (vrms / self.u_tau).cpu().numpy(),
            "wrms_plus": (wrms / self.u_tau).cpu().numpy(),
            "uv_plus": (uv / self.u_tau**2).cpu().numpy(),
            "n_samples": n,
        }

    def reset(self):
        """Reset accumulators."""
        self._sum_u.zero_()
        self._sum_u2.zero_()
        self._sum_v2.zero_()
        self._sum_w2.zero_()
        self._sum_uv.zero_()
        self._n = 0
