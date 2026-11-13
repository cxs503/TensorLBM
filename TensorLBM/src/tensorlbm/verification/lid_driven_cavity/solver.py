"""Lid-driven cavity solver using D2Q9 LBM.

Implements:
- D2Q9 BGK or regularized BGK (RLBM) collision
- Half-way bounce-back on stationary walls
- Zou-He velocity BC on moving lid

RLBM (Latt & Chopard 2006) is recommended for Re >= 1000 where tau approaches 0.5.
"""

from __future__ import annotations

import dataclasses
from typing import Tuple, Optional

import torch


@dataclasses.dataclass
class CavityConfig:
    """Configuration for lid-driven cavity simulation."""
    n: int = 129
    re: float = 100.0
    u_lid: float = 0.1
    max_steps: int = 50000
    tau: float = None
    use_rlbm: bool = True
    dtype: torch.dtype = torch.float64
    device: str = "cpu"

    def __post_init__(self):
        if self.n < 17:
            raise ValueError(f"Grid too small: n={self.n}, need >= 17")
        if self.u_lid >= 0.3:
            raise ValueError(f"u_lid={self.u_lid} too high for incompressible LBM")
        L = self.n - 1
        self.nu = self.u_lid * L / self.re
        if self.tau is None:
            self.tau = self.nu / (1.0/3.0) + 0.5
        if self.tau <= 0.5:
            raise ValueError(f"tau={self.tau} <= 0.5, unstable")

    @property
    def omega(self) -> float:
        return 1.0 / self.tau


class CavitySolver:
    """D2Q9 lid-driven cavity solver with RLBM support."""

    C = torch.tensor([
        [ 0,  0], [ 1,  0], [-1,  0], [ 0,  1], [ 0, -1],
        [ 1,  1], [-1,  1], [ 1, -1], [-1, -1],
    ], dtype=torch.int64)
    W = torch.tensor([
        4./9., 1./9., 1./9., 1./9., 1./9.,
        1./36., 1./36., 1./36., 1./36.,
    ], dtype=torch.float64)
    OPPOSITE = torch.tensor([0, 2, 1, 4, 3, 8, 7, 6, 5], dtype=torch.int64)

    def __init__(self, config: CavityConfig):
        self.cfg = config
        n = config.n
        self.n = n
        self.device = torch.device(config.device)
        self.dtype = config.dtype

        # Fluid mask (interior nodes only)
        self.fluid = torch.ones((n, n), dtype=torch.bool, device=self.device)
        self.fluid[0, :] = False   # bottom
        self.fluid[-1, :] = False  # top (lid)
        self.fluid[:, 0] = False   # left
        self.fluid[:, -1] = False  # right

        # Initialize at equilibrium (fluid at rest)
        rho = torch.ones((n, n), dtype=config.dtype, device=self.device)
        ux = torch.zeros((n, n), dtype=config.dtype, device=self.device)
        uy = torch.zeros((n, n), dtype=config.dtype, device=self.device)
        self.f = self._equilibrium(rho, ux, uy)

        # Streaming shifts
        self._shifts = []
        for i in range(9):
            cx, cy = int(self.C[i, 0]), int(self.C[i, 1])
            self._shifts.append((-cy, -cx))

    def _equilibrium(self, rho, ux, uy):
        c = self.C.to(device=self.device, dtype=rho.dtype)
        w = self.W.to(device=self.device, dtype=rho.dtype)
        uc = c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
        u2 = ux**2 + uy**2
        return w.view(-1, 1, 1) * rho.unsqueeze(0) * (
            1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
        )

    def _stream(self, f):
        f_out = torch.empty_like(f)
        for i in range(9):
            sy, sx = self._shifts[i]
            if sy == 0 and sx == 0:
                f_out[i] = f[i]
            else:
                f_out[i] = torch.roll(f[i], shifts=(sy, sx), dims=(0, 1))
        return f_out

    def _macroscopic(self, f):
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = (f * self.C[:, 0].view(-1, 1, 1).to(f.dtype)).sum(dim=0) / rho_safe
        uy = (f * self.C[:, 1].view(-1, 1, 1).to(f.dtype)).sum(dim=0) / rho_safe
        return rho, ux, uy

    def _collide_rlbm(self, f):
        """RLBM collision on fluid nodes only."""
        c = self.C.to(device=self.device, dtype=f.dtype)
        w = self.W.to(device=self.device, dtype=f.dtype)
        cs2 = 1.0 / 3.0

        # Macroscopic (fluid only for stress computation)
        rho = f.sum(dim=0)
        rho_safe = torch.clamp(rho, min=1e-12)
        ux = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0) / rho_safe
        uy = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0) / rho_safe

        # Equilibrium
        u2 = ux**2 + uy**2
        cu = c[:, 0].view(-1, 1, 1) * ux + c[:, 1].view(-1, 1, 1) * uy
        feq = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
            1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0)
        )

        fneq = f - feq

        # Stress tensor
        cix = c[:, 0].view(-1, 1, 1)
        ciy = c[:, 1].view(-1, 1, 1)
        Pi_xx = (fneq * cix * cix).sum(dim=0)
        Pi_yy = (fneq * ciy * ciy).sum(dim=0)
        Pi_xy = (fneq * cix * ciy).sum(dim=0)
        Pi_trace = Pi_xx + Pi_yy

        # Regularized fneq
        fneq_reg = w.view(-1, 1, 1) / (2.0 * cs2 * cs2) * (
            cix * cix * Pi_xx + ciy * ciy * Pi_yy +
            2.0 * cix * ciy * Pi_xy - cs2 * Pi_trace
        )

        # Collision on fluid nodes only
        omega = 1.0 / self.cfg.tau
        f_out = f.clone()
        f_out[:, self.fluid] = f[:, self.fluid] - omega * fneq_reg[:, self.fluid]
        return f_out

    def _collide_bgk(self, f):
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)
        f_out = f.clone()
        f_out[:, self.fluid] = f[:, self.fluid] + self.cfg.omega * (feq[:, self.fluid] - f[:, self.fluid])
        return f_out

    def _apply_bounce_back(self, f, f_pre_stream):
        """Half-way bounce-back on stationary walls (bottom, left, right)."""
        f_out = f.clone()
        # Bottom wall (y=0)
        for i in [3, 5, 6]:  # directions pointing into bottom wall
            opp = self.OPPOSITE[i]
            f_out[i, 0, :] = f_pre_stream[opp, 0, :]
        # Left wall (x=0)
        for i in [1, 5, 7]:
            opp = self.OPPOSITE[i]
            f_out[i, :, 0] = f_pre_stream[opp, :, 0]
        # Right wall (x=n-1)
        for i in [2, 6, 8]:
            opp = self.OPPOSITE[i]
            f_out[i, :, -1] = f_pre_stream[opp, :, -1]
        return f_out

    def _apply_zou_he_lid(self, f):
        """Zou-He velocity BC on moving lid (top wall, y=n-1)."""
        j = self.n - 1
        u_lid = self.cfg.u_lid

        f0 = f[0, j, :]
        f1 = f[1, j, :]
        f2 = f[2, j, :]
        f4 = f[4, j, :]
        f5 = f[5, j, :]
        f6 = f[6, j, :]

        rho = f0 + f1 + f2 + 2.0 * (f4 + f5 + f6)

        f[3, j, :] = f4
        f[7, j, :] = f6 - 0.5 * (f1 - f2) + (1.0 / 6.0) * rho * u_lid
        f[8, j, :] = f5 + 0.5 * (f1 - f2) + (1.0 / 6.0) * rho * u_lid
        return f

    def step(self):
        """One LBM step: collide -> stream -> boundary conditions."""
        # Collision (fluid nodes only)
        if self.cfg.use_rlbm:
            f_collided = self._collide_rlbm(self.f)
        else:
            f_collided = self._collide_bgk(self.f)

        # Streaming
        f_streamed = self._stream(f_collided)

        # Bounce-back on stationary walls
        f_bb = self._apply_bounce_back(f_streamed, f_collided)

        # Zou-He on lid
        self.f = self._apply_zou_he_lid(f_bb)

    def get_profiles(self):
        """Extract centerline velocity profiles."""
        rho, ux, uy = self._macroscopic(self.f)
        mid = self.n // 2
        u_center = ux[:, mid].cpu().numpy()
        v_center = uy[mid, :].cpu().numpy()
        y_norm = torch.linspace(0, 1, self.n).numpy()
        x_norm = torch.linspace(0, 1, self.n).numpy()
        return y_norm, u_center / self.cfg.u_lid, x_norm, v_center / self.cfg.u_lid

    def get_residual(self, ux_prev, uy_prev):
        rho, ux, uy = self._macroscopic(self.f)
        du = float(torch.max(
            torch.abs(ux[self.fluid] - ux_prev[self.fluid]).max(),
            torch.abs(uy[self.fluid] - uy_prev[self.fluid]).max(),
        ))
        return du

    def run(self, callback=None, resid_tol=1e-6, monitor_interval=5000):
        rho, ux_prev, uy_prev = self._macroscopic(self.f)
        for step in range(self.cfg.max_steps):
            self.step()
            if step % monitor_interval == 0 and step > 0:
                resid = self.get_residual(ux_prev, uy_prev)
                rho, ux_prev, uy_prev = self._macroscopic(self.f)
                if callback:
                    callback(step, self, resid)
                if resid < resid_tol:
                    break
        return self


def run_cavity(config: CavityConfig) -> CavitySolver:
    solver = CavitySolver(config)
    solver.run()
    return solver
