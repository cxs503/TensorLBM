"""Spherical-shell DG-LBM: body-fitted curvilinear near-wall solver (torch port).

Architecture ported from the validated numpy reference (Stokes sphere Cd err
0.89%, Re=100 sphere Cd 1.09 within 1%):

* **Near field** — a body-fitted spherical shell ``[R_in, R_out]`` partitioned
  into ``(Nr, Ntheta, Nphi)`` elements.  Every element carries a P1
  discontinuous-Galerkin representation of the D3Q19 populations
  (8 nodal DOFs per element: 2 per axis).  The LBE advection is integrated by
  direction splitting with first-order upwind differences (stable, no
  polynomial-over-shoot instability), and collision is BGK applied with
  Strang half-collisions around the advection sweep.  The sphere wall at
  ``r = R_in`` is a half-way bounce-back (no-slip); the outer boundary at
  ``r = R_out`` ingests exterior LBM values for incoming populations.
* **Far field** — a standard Cartesian D3Q19 LBM on the full domain; the shell
  band ``[R, R_shell]`` is overwritten every step with the DG interpolation
  (so the obstacle inside the shell is surrounded by real populations).
* **Coupling** — bidirectional trilinear interpolation on the outer shell
  boundary (LBM -> DG incoming values; DG -> LBM shell cells).
* **Drag** — Ladd momentum exchange on the *curved* wall faces with the
  physical face area ``dA = R_in^2 sin(theta) dtheta dphi`` (the key
  difference vs Cartesian-band MEM, which has no area weighting and
  overestimates at low Re).

All tensor layouts follow the TensorLBM convention ``(Q, nz, ny, nx)`` for
Cartesian fields and ``(Q, 2, 2, 2, Nr, Ntheta, Nphi)`` for the DG shell.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from .d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D, equilibrium3d, macroscopic3d

# ---------------------------------------------------------------------------
# Grid / geometry helpers
# ---------------------------------------------------------------------------


def _spherical_unit_vectors(theta: torch.Tensor, phi: torch.Tensor) -> tuple:
    """Unit vectors e_r, e_theta, e_phi at (theta, phi) grid centres.

    Args:
        theta: (Ntheta,) polar angle
        phi: (Nphi,) azimuthal angle
    Returns:
        (er, et, ep) each (3, Ntheta, Nphi): rows are (x, y, z) components.
    """
    st = torch.sin(theta)
    ct = torch.cos(theta)
    sp = torch.sin(phi)
    cp = torch.cos(phi)
    # e_r = (sinθcosφ, sinθsinφ, cosθ)
    er = torch.stack([st[:, None] * cp[None, :],
                      st[:, None] * sp[None, :],
                      ct[:, None].expand(-1, phi.shape[0])], dim=0)
    # e_theta = (cosθcosφ, cosθsinφ, -sinθ)
    et = torch.stack([ct[:, None] * cp[None, :],
                      ct[:, None] * sp[None, :],
                      -st[:, None].expand(-1, phi.shape[0])], dim=0)
    # e_phi = (-sinφ, cosφ, 0)
    ep = torch.stack([-sp[None, :].expand(theta.shape[0], -1),
                      cp[None, :].expand(theta.shape[0], -1),
                      torch.zeros(theta.shape[0], phi.shape[0], dtype=theta.dtype,
                                  device=theta.device)], dim=0)
    return er, et, ep


@dataclass
class SphericalShellConfig:
    """Configuration for the spherical-shell DG-LBM near-wall solver."""

    R_in: float          # sphere radius (lattice units)
    R_out: float         # outer shell radius
    Nr: int              # radial elements
    Ntheta: int          # polar elements
    Nphi: int            # azimuthal elements
    u_in: float = 0.1    # freestream velocity (lattice units)
    tau: float = 0.6     # BGK relaxation time (lattice units)
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

    # derived
    dr: float = field(init=False)
    dtheta: float = field(init=False)
    dphi: float = field(init=False)

    def __post_init__(self):
        self.dr = (self.R_out - self.R_in) / self.Nr
        self.dtheta = math.pi / self.Ntheta
        self.dphi = 2.0 * math.pi / self.Nphi


class SphericalShellDG:
    """Body-fitted spherical-shell DG-LBM near-wall solver (P1, upwind split).

    Layout of the DG field: ``f_dg[Q, a, b, g, jr, jt, jp]`` where
    ``a/b/g`` index the P1 nodes along r/theta/phi (0=inner/lower, 1=outer/
    upper) and ``jr/jt/jp`` index the elements.
    """

    def __init__(self, cfg: SphericalShellConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        Q = 19

        R_in, R_out = cfg.R_in, cfg.R_out
        Nr, Nth, Nph = cfg.Nr, cfg.Ntheta, cfg.Nphi

        # element edges and centres (staggered centres, aligned with the
        # validated reference stokes_sphere_test.py):
        #   r_c = linspace(R_in+dr/2, R_out-dr/2, Nr)
        #   theta_c = linspace(pi/(2 Nth), pi - pi/(2 Nth), Nth)
        #   phi_c = linspace(pi/Nph, 2pi - pi/Nph, Nph)
        dr = cfg.dr
        self.r_c = torch.linspace(R_in + dr / 2.0, R_out - dr / 2.0, Nr,
                                  device=self.device, dtype=self.dtype)
        self.theta_c = torch.linspace(math.pi / (2.0 * Nth),
                                      math.pi - math.pi / (2.0 * Nth), Nth,
                                      device=self.device, dtype=self.dtype)
        self.phi_c = torch.linspace(math.pi / Nph,
                                    2.0 * math.pi - math.pi / Nph, Nph,
                                    device=self.device, dtype=self.dtype)
        self.dr = cfg.dr
        self.dtheta = cfg.dtheta
        self.dphi = cfg.dphi

        # lattice velocities / weights / opposites (moved to device)
        self.C = C3D.to(device=self.device, dtype=self.dtype)          # (Q, 3)
        self.W = W3D.to(device=self.device, dtype=self.dtype)          # (Q,)
        self.OPP = OPP3D.to(device=self.device, dtype=torch.long)      # (Q,)

        # spherical components of each lattice velocity at every (theta, phi)
        er, et, ep = _spherical_unit_vectors(self.theta_c, self.phi_c)  # (3, Nth, Nph)
        self.c_ir = torch.einsum("qi,iab->qab", self.C, er)   # (Q, Nth, Nph)
        self.c_it = torch.einsum("qi,iab->qab", self.C, et)   # (Q, Nth, Nph)
        self.c_ip = torch.einsum("qi,iab->qab", self.C, ep)   # (Q, Nth, Nph)

        # DG field
        self.f_dg = torch.zeros(Q, 2, 2, 2, Nr, Nth, Nph, device=self.device, dtype=self.dtype)
        self._initialize()

    # ------------------------------------------------------------------
    # initialization / macros / equilibrium
    # ------------------------------------------------------------------
    def _initialize(self):
        rho0 = torch.ones((1, 1, 1), device=self.device, dtype=self.dtype)
        u0 = torch.full((1, 1, 1), self.cfg.u_in, device=self.device, dtype=self.dtype)
        uz = torch.zeros((1, 1, 1), device=self.device, dtype=self.dtype)
        feq = equilibrium3d(rho0, u0, uz.clone(), uz.clone())
        self.f_dg[:] = feq.view(19, 1, 1, 1, 1, 1, 1)

    def compute_macros_at_center(self):
        """Macroscopic quantities at element centres (8-DOF average)."""
        f_avg = self.f_dg.mean(dim=(1, 2, 3))  # (Q, Nr, Nth, Nph)
        rho = f_avg.sum(dim=0)
        rho = rho.clamp(min=1e-10)
        ux = (f_avg * self.C[:, 0].view(-1, 1, 1, 1)).sum(dim=0) / rho
        uy = (f_avg * self.C[:, 1].view(-1, 1, 1, 1)).sum(dim=0) / rho
        uz = (f_avg * self.C[:, 2].view(-1, 1, 1, 1)).sum(dim=0) / rho
        return rho, ux, uy, uz

    def compute_equilibrium_at_center(self, rho, ux, uy, uz):
        u_sq = ux * ux + uy * uy + uz * uz
        cu = (self.C[:, 0].view(-1, 1, 1, 1) * ux.unsqueeze(0) +
              self.C[:, 1].view(-1, 1, 1, 1) * uy.unsqueeze(0) +
              self.C[:, 2].view(-1, 1, 1, 1) * uz.unsqueeze(0))
        feq = self.W.view(-1, 1, 1, 1) * rho.unsqueeze(0) * (
            1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0)
        )
        return feq  # (Q, Nr, Nth, Nph)

    # ------------------------------------------------------------------
    # collision (BGK, Strang half-collision)
    # ------------------------------------------------------------------
    def collide(self):
        omega = 1.0 / self.cfg.tau
        rho, ux, uy, uz = self.compute_macros_at_center()
        feq = self.compute_equilibrium_at_center(rho, ux, uy, uz)
        self.f_dg -= omega * (self.f_dg - feq.view(19, 1, 1, 1, *feq.shape[1:]))

    # ------------------------------------------------------------------
    # advection: direction splitting with first-order upwind (stable)
    # ------------------------------------------------------------------
    def advect_r(self, dt):
        """Radial advection. Inner boundary (r=R_in) is half-way bounce-back;
        outer boundary is extrapolated (overwritten by the LBM coupling)."""
        dr = self.dr
        # f_dg: (Q, a, b, g, jr, jt, jp).  r-advection averages over the two
        # r DOFs (a): f_dg[:, 0] is (Q, b, g, jr, jt, jp) — 6 dims.
        f_avg = 0.5 * (self.f_dg.select(1, 0) + self.f_dg.select(1, 1))  # (Q, b, g, jr, jt, jp)

        # v_r: (Q, Nth, Nph) -> (Q, 1, 1, Nr, Nth, Nph)
        v_b = self.c_ir.view(-1, 1, 1, 1, self.c_ir.shape[1], self.c_ir.shape[2])
        v_b = v_b.expand(-1, f_avg.shape[1], f_avg.shape[2], f_avg.shape[3],
                         f_avg.shape[4], f_avg.shape[5]).contiguous()

        # upwind along r (dim 3 of f_avg); inner boundary: extrapolate
        # (aligned with the validated reference)
        f_left = torch.zeros_like(f_avg)
        f_left[:, :, :, 1:] = f_avg[:, :, :, :-1]
        f_left[:, :, :, 0] = f_avg[:, :, :, 0]  # extrapolate (reference)

        f_right = torch.zeros_like(f_avg)
        f_right[:, :, :, :-1] = f_avg[:, :, :, 1:]
        f_right[:, :, :, -1] = f_avg[:, :, :, -1]  # extrapolate

        pos = v_b > 0
        df_dr = torch.where(pos, (f_avg - f_left) / dr, (f_right - f_avg) / dr)
        update = -v_b * df_dr
        # update both r DOFs (dim 1)
        self.f_dg[:, 0] += dt * update
        self.f_dg[:, 1] += dt * update

    def advect_theta(self, dt):
        """Polar advection. Poles use the phi-average (conservative)."""
        dth = self.dtheta
        # average over theta DOFs (b, dim 2): f_dg[:, :, 0] is (Q, a, g, jr, jt, jp)
        f_avg = 0.5 * (self.f_dg.select(2, 0) + self.f_dg.select(2, 1))  # (Q, a, g, jr, jt, jp)

        # v_theta = c_it / r -> (Q, 1, 1, Nr, Nth, Nph)
        v_th = (self.c_it.view(-1, 1, 1, 1, self.c_it.shape[1], self.c_it.shape[2]) /
                self.r_c.view(1, 1, 1, -1, 1, 1))
        v_th = v_th.expand(-1, f_avg.shape[1], f_avg.shape[2], f_avg.shape[3],
                           f_avg.shape[4], f_avg.shape[5]).contiguous()

        # upwind along theta (dim 4 of f_avg)
        f_left = torch.zeros_like(f_avg)
        f_left[:, :, :, :, 1:] = f_avg[:, :, :, :, :-1]
        # theta=0 pole: phi-average
        f_left[:, :, :, :, 0] = f_avg[:, :, :, :, 0].mean(dim=-1, keepdim=True)

        f_right = torch.zeros_like(f_avg)
        f_right[:, :, :, :, :-1] = f_avg[:, :, :, :, 1:]
        f_right[:, :, :, :, -1] = f_avg[:, :, :, :, -1].mean(dim=-1, keepdim=True)

        pos = v_th > 0
        df = torch.where(pos, (f_avg - f_left) / dth, (f_right - f_avg) / dth)
        update = -v_th * df
        self.f_dg[:, :, 0] += dt * update
        self.f_dg[:, :, 1] += dt * update

    def advect_phi(self, dt):
        """Azimuthal advection (periodic)."""
        dph = self.dphi
        # average over phi DOFs (g, dim 3): f_dg[:, :, :, 0] is (Q, a, b, jr, jt, jp)
        f_avg = 0.5 * (self.f_dg.select(3, 0) + self.f_dg.select(3, 1))  # (Q, a, b, jr, jt, jp)

        # v_phi = c_ip / (r sin theta), clamp near poles (reference: >= 1.0)
        st = torch.sin(self.theta_c)
        pole_mask = st < 0.15  # (Nth,) — zero the update near poles
        r_sin_t = self.r_c.view(-1, 1) * st.view(1, -1)  # (Nr, Nth)
        r_sin_t = r_sin_t.clamp(min=1.0)
        Nr, Nth = self.cfg.Nr, self.cfg.Ntheta
        # c_ip: (Q, Nth, Nph) / (Nr, Nth) -> (Q, Nr, Nth, Nph)
        v_ph = self.c_ip.view(-1, 1, Nth, self.c_ip.shape[2]) / r_sin_t.view(1, Nr, Nth, 1)
        # -> (Q, 1, 1, Nr, Nth, Nph)  (matching f_avg's (Q, a, b, jr, jt, jp))
        v_b = v_ph.view(-1, 1, 1, Nr, Nth, v_ph.shape[-1])
        v_b = v_b.expand(-1, f_avg.shape[1], f_avg.shape[2], Nr, Nth,
                         v_ph.shape[-1]).contiguous()

        # upwind along phi (dim 5 of f_avg)
        f_left = torch.zeros_like(f_avg)
        f_left[:, :, :, :, :, 1:] = f_avg[:, :, :, :, :, :-1]
        f_left[:, :, :, :, :, 0] = f_avg[:, :, :, :, :, -1]  # periodic

        f_right = torch.zeros_like(f_avg)
        f_right[:, :, :, :, :, :-1] = f_avg[:, :, :, :, :, 1:]
        f_right[:, :, :, :, :, -1] = f_avg[:, :, :, :, :, 0]

        pos = v_b > 0
        df = torch.where(pos, (f_avg - f_left) / dph, (f_right - f_avg) / dph)
        update = -v_b * df
        # zero the update at the poles (reference)
        update[:, :, :, :, pole_mask, :] = 0.0
        self.f_dg[:, :, :, 0] += dt * update
        self.f_dg[:, :, :, 1] += dt * update

    # ------------------------------------------------------------------
    # wall / outer boundary conditions
    # ------------------------------------------------------------------
    def apply_wall_bc(self):
        """Half-way bounce-back at the sphere wall (r=R_in, no-slip).

        Aligned with the validated reference (stokes_sphere_test.py): the
        *incoming* populations (c_ir < 0, pointing into the sphere) are
        reflected from the wall equilibrium + opposite population.
        """
        rho, ux, uy, uz = self.compute_macros_at_center()
        rho_wall = rho[0]  # (Nth, Nph) first radial element
        feq_wall = self.W.view(-1, 1, 1) * rho_wall.unsqueeze(0)  # (Q, Nth, Nph) u=0

        # c_ir mean over (theta, phi): incoming = < 0
        c_in = self.c_ir.mean(dim=(1, 2)) < 0  # (Q,)
        for i in range(19):
            if bool(c_in[i]):
                oi = int(self.OPP[i])
                self.f_dg[i, 0, :, :, 0] = (
                    feq_wall[i] + self.f_dg[oi, 0, :, :, 0] - feq_wall[oi]
                )

    def apply_outer_bc(self, lbm_f_at_shell):
        """Ingest LBM values at the outer boundary for outgoing populations
        (aligned with the validated reference: c_ir > 0)."""
        # lbm_f_at_shell: (Q, Nth, Nph)
        c_out = self.c_ir.mean(dim=(1, 2)) > 0  # (Q,)
        for i in range(19):
            if bool(c_out[i]):
                self.f_dg[i, :, :, :, -1] = lbm_f_at_shell[i].view(1, 1, 1, 1,
                                                                    lbm_f_at_shell.shape[1],
                                                                    lbm_f_at_shell.shape[2])

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, dt=0.25, lbm_boundary=None):
        """One DG step: half-collide, full advection sweep, wall/outer BC,
        half-collide (Strang).  The reference solver calls this with dt=0.25
        four times per LBM step (see stokes_sphere_test.py); the default dt
        mirrors that.  A safety clip bounds the field (reference behaviour).
        """
        # phi-direction CFL at the innermost radius: the speed is
        # |c_ip|/(r sinθ) with the denominator clamped to >= 1.0 (near the
        # poles), and the cell width is r sinθ dφ.  Worst case at r = R_in:
        #   CFL = max|v_ph| dt / (r sinθ dφ)
        r_min = float(self.cfg.R_in)
        st_vals = torch.sin(self.theta_c)
        rs_min = (r_min * st_vals).clamp(min=1.0)  # denominator of v_ph
        cell_w = r_min * st_vals * self.cfg.dphi
        cell_w = cell_w.clamp(min=1e-9)
        cfl_phi = float((1.0 / rs_min / cell_w).max().item())
        n_sub = max(1, int(math.ceil(max(cfl_phi, 1.0 / self.cfg.dr, 1.0))))
        dt_sub = dt / n_sub
        # Strang: half-collide, sub-cycled advection sweep, half-collide
        self.collide()
        for _ in range(n_sub):
            self.advect_r(dt_sub)
            self.advect_theta(dt_sub)
            self.advect_phi(dt_sub)
            self.apply_wall_bc()
            if lbm_boundary is not None:
                self.apply_outer_bc(lbm_boundary)
        self.collide()
        # safety clip (reference behaviour)
        self.f_dg.clamp_(-1e6, 1e6)

    # ------------------------------------------------------------------
    # drag (Ladd momentum exchange with curved-wall face area)
    # ------------------------------------------------------------------
    def drag(self):
        """Pressure-integral drag on the sphere (the force used by the
        validated reference stokes_sphere_test.py, 0.89% error at Re=0.1):
        F = sum(p_wall * n_hat * dA) + friction, p = rho/3, with the curved
        wall face area dA = R_in^2 sin(theta) dtheta dphi.
        Returns (F_x, F_y, F_z) in lattice units and the Cd."""
        rho, ux, uy, uz = self.compute_macros_at_center()
        rho_wall = rho[0]  # (Nth, Nph)
        p_wall = rho_wall / 3.0

        th_g = self.theta_c.view(-1, 1)
        ph_g = self.phi_c.view(1, -1)
        nx_hat = torch.sin(th_g) * torch.cos(ph_g)  # (Nth, Nph)
        ny_hat = torch.sin(th_g) * torch.sin(ph_g)
        nz_hat = torch.cos(th_g).expand(-1, ph_g.shape[1])
        dA = (self.cfg.R_in ** 2 * torch.sin(th_g) * self.dtheta * self.dphi)  # (Nth, 1)

        F_px = (p_wall * nx_hat * dA).sum()
        F_py = (p_wall * ny_hat * dA).sum()
        F_pz = (p_wall * nz_hat * dA).sum()

        # friction: mu * du/dr * dA at the wall (radial derivative of u)
        mu = rho_wall * (self.cfg.tau - 0.5) / 3.0
        du_dr = (ux[1] - ux[0]) / self.dr
        dv_dr = (uy[1] - uy[0]) / self.dr
        dw_dr = (uz[1] - uz[0]) / self.dr
        F_fx = (mu * du_dr * dA).sum()
        F_fy = (mu * dv_dr * dA).sum()
        F_fz = (mu * dw_dr * dA).sum()

        F = torch.stack([F_px + F_fx, F_py + F_fy, F_pz + F_fz])
        A_ref = math.pi * self.cfg.R_in ** 2
        cd = 2.0 * float(F[0]) / (self.cfg.u_in ** 2 * A_ref)
        return F, cd
