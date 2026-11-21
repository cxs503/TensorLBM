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
    use_trt: bool = False
    magic: float = 3.0 / 16.0
    #: Lid velocity is ramped linearly over this many steps (0 = impulsive
    #: start).  An impulsive start at Re = 5000 launches an acoustic wave that
    #: drives rho outside [0.6, 1.4] within ~1e3 steps; the ramp removes it.
    ramp_steps: int = 0
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

    @property
    def tau_minus(self) -> float:
        """Second (ghost-mode) relaxation time of the TRT model.

        Fixed by the "magic" parameter ``Lambda = (tau-1/2)(tau_minus-1/2)``,
        ``Lambda = 3/16`` being the standard choice (Ginzburg 2005; Kruger et
        al., The Lattice Boltzmann Method, 2017, sec. 10.2): it makes the
        half-way bounce-back wall independent of the relaxation rates while
        damping the non-hydrodynamic modes much harder than BGK does.
        """
        return 0.5 + self.magic / (self.tau - 0.5)


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

        # Wall / fluid masks: the boundary ring is wall (bottom/left/right
        # no-slip, top moving lid); all interior nodes are fluid.
        solid = torch.ones((n, n), dtype=torch.bool, device=self.device)
        solid[1:-1, 1:-1] = False
        self.solid = solid            # kept for backwards compatibility
        self.fluid = ~solid

        # Wall velocity (only the lid moves, tangential, u=(u_lid, 0))
        self._uw_x = torch.zeros((n, n), dtype=config.dtype, device=self.device)
        self._uw_y = torch.zeros((n, n), dtype=config.dtype, device=self.device)
        self._uw_x[-1, :] = config.u_lid
        self._t = 0

        # Pre-compute half-way link bounce-back data: for every lattice
        # direction i, the set of fluid nodes whose neighbour in direction i is
        # a wall node, plus the wall velocity seen through that link.
        c = self.C.to(device=self.device, dtype=config.dtype)
        self._bb_links = []
        for i in range(1, 9):
            cy, cx = int(self.C[i, 1]), int(self.C[i, 0])
            nb_solid = torch.roll(solid, shifts=(cy, cx), dims=(0, 1))
            # unit lid velocity (1 on the lid row, 0 elsewhere) so that the
            # acceleration ramp only scales a pre-computed geometric factor
            nb_dir_x = torch.roll(self._uw_x / max(config.u_lid, 1e-30),
                                  shifts=(cy, cx), dims=(0, 1))
            nb_uw_y = torch.roll(self._uw_y, shifts=(cy, cx), dims=(0, 1))
            mask = self.fluid & nb_solid
            o = int(self.OPPOSITE[i])
            # momentum-transfer factor: 6 * w_o * (c_o . u_wall_unit)
            coef = 6.0 * self.W[o].to(config.dtype) * (
                c[o, 0] * nb_dir_x + c[o, 1] * nb_uw_y
            )
            self._bb_links.append((i, o, mask, coef))

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

        # Regularized fneq: project the non-equilibrium part onto the
        # second-order Hermite subspace, Q_i = c_i c_i - cs^2 I (Latt & Chopard
        # 2006, eq. 15).  Everything outside that subspace (the ghost modes that
        # make BGK marginal as tau -> 1/2) is discarded.
        fneq_reg = w.view(-1, 1, 1) / (2.0 * cs2 * cs2) * (
            cix * cix * Pi_xx + ciy * ciy * Pi_yy +
            2.0 * cix * ciy * Pi_xy - cs2 * Pi_trace
        )

        # Regularized collision on fluid nodes only:
        #     f* = f^eq + (1 - omega) f^neq,reg
        # Note: f - omega * f^neq,reg would be equivalent only if f^neq were
        # already fully regularized; using it keeps the non-hydrodynamic modes
        # with unit gain and defeats the purpose of the regularization.
        omega = 1.0 / self.cfg.tau
        f_out = feq + (1.0 - omega) * fneq_reg
        return f_out

    def _collide_trt(self, f):
        """Two-relaxation-time collision on fluid nodes.

        Symmetric (``f^+``, carries rho and the momentum flux) and
        antisymmetric (``f^-``, carries momentum and the ghost modes) parts are
        relaxed with ``omega = 1/tau`` and ``omega_minus = 1/tau_minus``.
        """
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)
        opp = self.OPPOSITE.to(device=f.device)

        f_plus = 0.5 * (f + f[opp])
        f_minus = 0.5 * (f - f[opp])
        feq_plus = 0.5 * (feq + feq[opp])
        feq_minus = 0.5 * (feq - feq[opp])

        omega_p = 1.0 / self.cfg.tau
        omega_m = 1.0 / self.cfg.tau_minus
        f_out = f.clone()
        f_out[:, self.fluid] = (
            f[:, self.fluid]
            - omega_p * (f_plus[:, self.fluid] - feq_plus[:, self.fluid])
            - omega_m * (f_minus[:, self.fluid] - feq_minus[:, self.fluid])
        )
        return f_out

    def _collide_bgk(self, f):
        rho, ux, uy = self._macroscopic(f)
        feq = self._equilibrium(rho, ux, uy)
        f_out = f.clone()
        f_out[:, self.fluid] = f[:, self.fluid] + self.cfg.omega * (feq[:, self.fluid] - f[:, self.fluid])
        return f_out

    def _apply_boundaries(self, f_post, f_streamed):
        """Apply wall boundary conditions after streaming.

        Two ingredients, both half-way (wall located half a lattice unit
        between the outermost fluid node and the wall node):

        1. **Link bounce-back**: for every fluid node whose neighbour in
           direction ``i`` is a wall node, the post-collision population
           ``f_i`` is reflected and re-enters the same fluid node as
           ``f_opp(i)``.  For a wall moving with velocity ``u_w`` the
           momentum-transfer correction ``6 w_opp rho (c_opp . u_w)`` is added
           (Ladd 1994), which is what drives the flow for the moving lid.
        2. **Wall nodes carry the wall equilibrium** (rho=1, u=u_w) so that
           the reported macroscopic fields at solid nodes are the physical
           wall values instead of streaming leftovers.

        Parameters
        ----------
        f_post : torch.Tensor
            Post-collision populations ``f`` (used by the reflection, because
            the reflected population travels half a lattice unit out and back
            within the same time step).
        f_streamed : torch.Tensor
            Streamed populations (interior values are correct, wall nodes are
            overwritten here).
        """
        f_out = f_streamed.clone()
        rho_post = f_post.sum(dim=0)
        lid_speed = float(self._uw_x[-1, 0])
        for i, o, mask, coef in self._bb_links:
            f_out[o][mask] = f_post[i][mask] + lid_speed * coef[mask] * rho_post[mask]

        feq_wall = self._equilibrium(
            torch.ones_like(rho_post), self._uw_x, self._uw_y
        )
        wall = self.solid
        f_out[:, wall] = feq_wall[:, wall]
        return f_out

    def step(self):
        """One LBM step: collide -> stream -> boundary conditions."""
        self._t += 1
        if self.cfg.ramp_steps > 0:
            frac = min(1.0, self._t / float(self.cfg.ramp_steps))
            self._uw_x[-1, :] = self.cfg.u_lid * frac
        # Collision (fluid nodes only)
        if self.cfg.use_trt:
            f_collided = self._collide_trt(self.f)
        elif self.cfg.use_rlbm:
            f_collided = self._collide_rlbm(self.f)
        else:
            f_collided = self._collide_bgk(self.f)

        # Streaming (periodic roll; wall-node values are overwritten below)
        f_streamed = self._stream(f_collided)

        # Walls: link bounce-back (moving lid included) + wall equilibrium
        self.f = self._apply_boundaries(f_collided, f_streamed)

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
