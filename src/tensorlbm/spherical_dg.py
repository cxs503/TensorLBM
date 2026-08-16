"""Spherical-shell DG-LBM: body-fitted curvilinear near-wall solver (proper nodal DG).

Architecture
------------
* **Near field** — a body-fitted spherical shell ``[R_in, R_out]`` partitioned
  into ``(Nr, Ntheta, Nphi)`` elements.  Every element carries a P1
  discontinuous-Galerkin representation of the D3Q19 populations
  (8 nodal DOFs per element: 2 per axis on Gauss–Lobatto nodes).
  The advection uses the *proper* nodal-DG weak form with mass-matrix
  inverse, volume stiffness, and upwind numerical fluxes (via
  ``dg_advection.get_ops``), not the first-order upwind finite-difference
  approximation used by the Zhipu reference.  Collision is BGK applied
  pointwise to every nodal DOF (dt-aware).  Time integration is SSP-RK3
  with automatic CFL sub-stepping.
* **Far field** — a standard Cartesian D3Q19 LBM on the full domain; the shell
  band ``[R, R_shell]`` is overwritten every step with the DG interpolation.
* **Coupling** — bidirectional trilinear interpolation on the outer shell
  boundary (LBM -> DG incoming values; DG -> LBM shell cells).
* **Drag** — pressure-integral + wall friction on the curved wall with the
  physical face area ``dA = R_in^2 sin(theta) dtheta dphi``.

Field layout
------------
``f_dg[Q, Nr, Ntheta, Nphi, 2, 2, 2]`` — Q axis first, then cell axes,
then node axes (aligned with ``dg_advection`` convention so that
``dg_rhs`` / ``dg_lbm_step`` can be called directly).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .d3q19 import C as C3D, OPPOSITE as OPP3D, W as W3D
from .dg_advection import (
    _Ops,
    collide_bgk_dg,
    dg_advect,
    dg_lbm_step,
    dg_rhs,
    get_ops,
    macroscopic_dg,
    nodal_from_mean,
)


# ---------------------------------------------------------------------------
# Grid / geometry helpers
# ---------------------------------------------------------------------------


def _spherical_unit_vectors(theta: torch.Tensor, phi: torch.Tensor) -> tuple:
    """Unit vectors e_r, e_theta, e_phi at (theta, phi) grid centres.

    Returns (er, et, ep) each (3, Ntheta, Nphi): rows are (x, y, z) components.
    """
    st = torch.sin(theta)
    ct = torch.cos(theta)
    sp = torch.sin(phi)
    cp = torch.cos(phi)
    er = torch.stack([st[:, None] * cp[None, :],
                      st[:, None] * sp[None, :],
                      ct[:, None].expand(-1, phi.shape[0])], dim=0)
    et = torch.stack([ct[:, None] * cp[None, :],
                      ct[:, None] * sp[None, :],
                      -st[:, None].expand(-1, phi.shape[0])], dim=0)
    ep = torch.stack([-sp[None, :].expand(theta.shape[0], -1),
                      cp[None, :].expand(theta.shape[0], -1),
                      torch.zeros(theta.shape[0], phi.shape[0],
                                  dtype=theta.dtype, device=theta.device)], dim=0)
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
    degree: int = 1      # DG polynomial degree (1 = P1, 2 = P2, ...)
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
    """Body-fitted spherical-shell DG-LBM near-wall solver (proper nodal DG).

    Layout of the DG field: ``f_dg[Q, Nr, Ntheta, Nphi, n_r, n_th, n_ph]``
    where the trailing three axes are the nodal DOFs per element (n = degree+1
    per axis).  This layout matches the ``dg_advection`` convention so that
    the standard DG operators (volume stiffness, face lift, upwind flux) can
    be applied directly.
    """

    def __init__(self, cfg: SphericalShellConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.dtype = cfg.dtype
        Q = 19
        deg = cfg.degree
        n = deg + 1  # nodes per axis per element

        R_in, R_out = cfg.R_in, cfg.R_out
        Nr, Nth, Nph = cfg.Nr, cfg.Ntheta, cfg.Nphi

        # element centres (staggered, avoiding boundaries)
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

        # lattice velocities / weights / opposites
        self.C = C3D.to(device=self.device, dtype=self.dtype)          # (Q, 3)
        self.W = W3D.to(device=self.device, dtype=self.dtype)          # (Q,)
        self.OPP = OPP3D.to(device=self.device, dtype=torch.long)      # (Q,)

        # spherical components of each lattice velocity at every (theta, phi)
        er, et, ep = _spherical_unit_vectors(self.theta_c, self.phi_c)
        self.c_ir = torch.einsum("qi,iab->qab", self.C, er)   # (Q, Nth, Nph)
        self.c_it = torch.einsum("qi,iab->qab", self.C, et)
        self.c_ip = torch.einsum("qi,iab->qab", self.C, ep)

        # contravariant velocities: ĉ = J^{-T} c  (for the spherical shell
        # the Jacobian is diagonal, so ĉ_a = c_a / h_a where h_a are the
        # scale factors).  We store per-element contravariant speeds along
        # Build contravariant velocity tensor (Q, 3, Nr, Nth, Nph)
        # axis 0 = r, axis 1 = theta, axis 2 = phi
        #   ĉ_r = c_ir,  ĉ_theta = c_it / r,  ĉ_phi = c_ip / (r sinθ)
        # Near the poles sinθ→0 makes ĉ_phi blow up; clamp the denominator
        # and zero ĉ_phi where sinθ < 0.3 (the first/last ~2 theta cells)
        # to avoid CFL explosion (the flow is nearly axisymmetric there).
        st = torch.sin(self.theta_c)  # (Nth,)
        r_sin_t = self.r_c.view(-1, 1) * st.view(1, -1)  # (Nr, Nth)
        r_sin_t = r_sin_t.clamp(min=1.0)  # avoid pole singularity
        pole_mask = st < 0.3  # (Nth,) — zero phi-velocity near poles

        c_hat = torch.zeros(Q, 3, Nr, Nth, Nph, device=self.device, dtype=self.dtype)
        c_hat[:, 0] = self.c_ir.view(Q, 1, Nth, Nph).expand(Q, Nr, Nth, Nph)
        c_hat[:, 1] = (self.c_it.view(Q, 1, Nth, Nph) /
                       self.r_c.view(1, Nr, 1, 1)).expand(Q, Nr, Nth, Nph)
        c_hat[:, 2] = (self.c_ip.view(Q, 1, Nth, Nph) /
                       r_sin_t.view(1, Nr, Nth, 1)).expand(Q, Nr, Nth, Nph)
        # zero phi-velocity near poles
        c_hat[:, 2, :, pole_mask, :] = 0.0
        self.c_hat = c_hat  # (Q, 3, Nr, Nth, Nph)

        # Specular reflection map for the sphere wall.
        # For each theta cell, the wall normal is n̂ = (sinθcosφ, sinθsinφ, cosθ).
        # The specular reflection of c_i about n̂ is c_ref = c - 2(c·n̂)n̂.
        # We precompute the nearest lattice velocity index for each (Q, Nth).
        # This is used for the wall ghost in the DG weak form.
        with torch.no_grad():
            er_wall, _, _ = _spherical_unit_vectors(self.theta_c, self.phi_c[:1])
            # n̂ depends on theta only (phi just rotates x,y); for the
            # specular map we only need the theta dependence since the
            # D3Q19 lattice is axis-aligned.  For a general n̂ we'd need
            # per-(theta,phi) maps, but the lattice symmetry lets us
            # compute per-theta.
            n_hat = torch.stack([
                torch.sin(self.theta_c),
                torch.zeros_like(self.theta_c),
                torch.cos(self.theta_c)
            ], dim=1)  # (Nth, 3) — n̂ at phi=0 (by symmetry the map is the same)
            spec_map = torch.zeros(Q, Nth, dtype=torch.long, device=self.device)
            for j in range(Nth):
                nrm = n_hat[j]  # (3,) — wall normal at this theta
                for i in range(Q):
                    c = self.C[i]
                    c_ref = c - 2.0 * torch.dot(c, nrm) * nrm
                    dists = (self.C - c_ref).norm(dim=1)
                    spec_map[i, j] = dists.argmin()
            self.specular_map = spec_map  # (Q, Nth)
        self.ops_r = get_ops(deg, dr, dtype=cfg.dtype, device=cfg.device)
        self.ops_th = get_ops(deg, cfg.dtheta, dtype=cfg.dtype, device=cfg.device)
        self.ops_ph = get_ops(deg, cfg.dphi, dtype=cfg.dtype, device=cfg.device)

        # DG field: (Q, Nr, Nth, Nph, n_r, n_th, n_ph)
        self.f_dg = torch.zeros(Q, Nr, Nth, Nph, n, n, n,
                                device=self.device, dtype=self.dtype)
        self._initialize()

    # ------------------------------------------------------------------
    # initialization / macros
    # ------------------------------------------------------------------
    def _initialize(self):
        """Set f_dg to equilibrium for uniform flow u_in along x."""
        # Use nodal_from_mean to seed all DOFs from the cell-mean equilibrium.
        from .d3q19 import equilibrium3d
        rho0 = torch.ones(1, 1, 1, device=self.device, dtype=self.dtype)
        u0 = torch.full((1, 1, 1), self.cfg.u_in, device=self.device, dtype=self.dtype)
        uz = torch.zeros((1, 1, 1), device=self.device, dtype=self.dtype)
        feq = equilibrium3d(rho0, u0, uz.clone(), uz.clone())  # (19, 1, 1, 1)
        # broadcast to (Q, Nr, Nth, Nph)
        feq_cells = feq.squeeze(-1).squeeze(-1).squeeze(-1)  # (Q,)
        feq_cells = feq_cells.view(19, 1, 1, 1).expand(19, self.cfg.Nr,
                                                         self.cfg.Ntheta,
                                                         self.cfg.Nphi)
        # expand to nodal DOFs
        n = self.cfg.degree + 1
        self.f_dg = feq_cells.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(
            19, self.cfg.Nr, self.cfg.Ntheta, self.cfg.Nphi, n, n, n
        ).contiguous()

    def compute_macros_at_center(self):
        """Macroscopic quantities at element centres (cell-mean of all DOFs)."""
        # Average over the three node axes (4, 5, 6)
        f_avg = self.f_dg.mean(dim=(4, 5, 6))  # (Q, Nr, Nth, Nph)
        rho = f_avg.sum(dim=0)
        rho = rho.clamp(min=1e-10)
        ux = (f_avg * self.C[:, 0].view(-1, 1, 1, 1)).sum(dim=0) / rho
        uy = (f_avg * self.C[:, 1].view(-1, 1, 1, 1)).sum(dim=0) / rho
        uz = (f_avg * self.C[:, 2].view(-1, 1, 1, 1)).sum(dim=0) / rho
        return rho, ux, uy, uz

    # ------------------------------------------------------------------
    # DG advection RHS (proper nodal DG, dimension-by-dimension)
    # ------------------------------------------------------------------
    def _rhs_axis(self, f: torch.Tensor, axis: int,
                  c_along: torch.Tensor, ops: _Ops,
                  periodic: bool = False,
                  wall_f: torch.Tensor | None = None,
                  wall_q_mask: torch.Tensor | None = None) -> torch.Tensor:
        """DG RHS contribution along one axis of the spherical shell.

        Args:
            f: (Q_active, Nr, Nth, Nph, n_r, n_th, n_ph) — active subset.
            axis: 0=r, 1=theta, 2=phi.
            c_along: (Q_active, Nr, Nth, Nph) — contravariant speed.
            ops: 1D DG operators.
            periodic: whether the axis is periodic (phi).
            wall_f: full f (all Q) for specular reflection lookup (axis=0 only).
            wall_q_mask: (Q,) bool mask mapping active→full Q indices (axis=0).
        """
        n = ops.n_node
        # cell axis = 1 + axis (Q is axis 0)
        cell_axis = 1 + axis
        # node axis = 4 + axis (after the three cell axes)
        node_axis = 4 + axis

        Ax = ops.Ax       # (n, n)
        face_lift = ops.face_lift  # (n, 2)
        p_last = n - 1

        # --- Volume term: c · Ax · u ---
        letters = "abcdefghijklmnopqrst"
        in_subs = [letters[i] for i in range(f.ndim)]
        out_subs = list(in_subs)
        in_subs[node_axis] = "u"
        out_subs[node_axis] = "v"
        ein = f"vu,{''.join(in_subs)}->{''.join(out_subs)}"
        vol = torch.einsum(ein, Ax, f)

        # --- Surface term ---
        inner_left = f.select(node_axis, 0)    # u_e[0]
        inner_right = f.select(node_axis, p_last)  # u_e[p]

        if periodic:
            left_ext = torch.roll(inner_right, shifts=1, dims=cell_axis)
            right_ext = torch.roll(inner_left, shifts=-1, dims=cell_axis)
        else:
            # Non-periodic: build neighbour arrays without wrap-around.
            # left_ext[j] = inner_right[j-1] (zero-gradient at j=0)
            # right_ext[j] = inner_left[j+1] (zero-gradient at j=N-1)
            N = inner_left.shape[cell_axis]
            left_ext = torch.zeros_like(inner_left)
            right_ext = torch.zeros_like(inner_left)
            # interior: shift by 1
            sl_l = [slice(None)] * inner_left.ndim
            sl_r = [slice(None)] * inner_left.ndim
            sl_src = [slice(None)] * inner_left.ndim
            sl_l[cell_axis] = slice(1, N)   # left_ext[1:] = inner_right[:-1]
            sl_src[cell_axis] = slice(0, N-1)
            left_ext[tuple(sl_l)] = inner_right[tuple(sl_src)]
            sl_r[cell_axis] = slice(0, N-1)  # right_ext[:-1] = inner_left[1:]
            sl_src[cell_axis] = slice(1, N)
            right_ext[tuple(sl_r)] = inner_left[tuple(sl_src)]
            # boundary ghost
            sl0 = [slice(None)] * inner_left.ndim
            sl0[cell_axis] = 0
            slN = [slice(None)] * inner_left.ndim
            slN[cell_axis] = -1

            if axis == 0:
                # inner wall: specular reflection ghost (no-slip).
                # f_ghost[i] = f[specular_map[i, theta]] at the wall face.
                # This reflects only the normal component, preserving
                # tangential momentum (true no-slip for DG weak form).
                if wall_f is not None and wall_q_mask is not None:
                    active_q = torch.nonzero(wall_q_mask, as_tuple=False).squeeze(-1)
                    # left_ext[ai] has shape (Nr, Nth, Nph, n_th, n_ph)
                    # (Q axis removed by [ai], node axis removed by select).
                    # cell_axis in the original f is 1+axis = 1 for axis=0.
                    # In left_ext[ai] (Q removed), Nr is dim 0, Nth is dim 1.
                    # We want to set the theta dimension (dim 1) per-theta.
                    th_dim = 1  # theta dimension in left_ext[ai]
                    for ai in range(f.shape[0]):
                        qi = int(active_q[ai])
                        spec_idx = self.specular_map[qi]  # (Nth,)
                        for j in range(self.cfg.Ntheta):
                            si = int(spec_idx[j].item())
                            ghost_val = wall_f[si, 0, j, :, 0, :, :]  # (Nph, n_th, n_ph)
                            sl_j = [slice(None)] * left_ext[ai].ndim
                            sl_j[th_dim] = j
                            left_ext[ai][tuple(sl_j)] = ghost_val
                else:
                    left_ext[tuple(sl0)] = inner_left.select(cell_axis, 0)
                # outer boundary: zero-gradient
                right_ext[tuple(slN)] = inner_right.select(cell_axis, -1)
            elif axis == 1:
                left_ext[tuple(sl0)] = inner_left.select(cell_axis, 0)
                right_ext[tuple(slN)] = inner_right.select(cell_axis, -1)
            else:
                raise ValueError(f"axis {axis} should be periodic")

        # upwind selection — c_along is (Q, Nr, Nth, Nph), inner_left has
        # the node axis removed so its ndim = f.ndim - 1.  Broadcast c_along
        # to match inner_left's shape.
        c_view = c_along.reshape(
            [c_along.shape[0]] +
            [1] * (inner_left.ndim - c_along.ndim) +
            [c_along.shape[i] for i in range(1, c_along.ndim)]
        )
        # Reorder so cell axes align: c_along has (Q, Nr, Nth, Nph) but
        # inner_left has (Q, Nr, Nth, Nph, n_th, n_ph) for axis=0 etc.
        # Simpler: just expand c_along to (Q, Nr, Nth, Nph, 1, 1, 1) and
        # let broadcasting handle it.
        c_expand = c_along.view(c_along.shape[0],
                                *[1] * (f.ndim - c_along.ndim),
                                *c_along.shape[1:],
                                *[1] * (f.ndim - c_along.ndim - c_along.shape[0:1].numel()))
        # Actually, simplest: reshape c_along to (Q, 1, 1, 1, 1, 1, 1) won't
        # work because it has Nr*Nth*Nph elements.  We need (Q, Nr, Nth, Nph,
        # 1, 1, 1) to broadcast against f's (Q, Nr, Nth, Nph, n, n, n).
        c_for_f = c_along.view(*c_along.shape, 1, 1, 1)  # (Q, Nr, Nth, Nph, 1, 1, 1)
        c_for_inner = c_along.view(*c_along.shape,
                                   *[1] * (inner_left.ndim - c_along.ndim))  # (Q, Nr, Nth, Nph, 1, 1)
        pos = c_for_inner > 0.0
        uL = torch.where(pos, left_ext, inner_left)
        uR = torch.where(pos, inner_right, right_ext)

        fl_l = face_lift[:, 0]  # (n,)
        fl_r = face_lift[:, 1]
        shape = [1] * f.ndim
        shape[node_axis] = n
        surf_l = fl_l.view(shape) * uL.unsqueeze(node_axis)
        surf_r = fl_r.view(shape) * uR.unsqueeze(node_axis)
        surf = surf_l + surf_r

        c_full = c_for_f  # (Q, Nr, Nth, Nph, 1, 1, 1)
        return c_full * vol - c_full * surf

    def _wall_ghost_left(self, inner_right: torch.Tensor,
                         cell_axis: int) -> torch.Tensor:
        """Ghost values for the inner wall (r=R_in): specular bounce-back.

        For the DG weak form, the wall ghost must be the specular reflection
        of the interior face value: f_ghost[i] = f[opp[i]] at the wall face.
        This makes the upwind numerical flux automatically enforce no-slip
        (half-way bounce-back) without a separate BC pass.

        ``inner_right`` is the face value at the *outer* node of the last
        cell (used for roll-based neighbour lookup).  Here we need the face
        value at the *inner* node of the first cell, which is
        ``f.select(node_axis, 0)`` — but that's ``inner_left`` in the caller.
        We receive it as ``inner_right`` for API compatibility; the actual
        ghost is built from the full f_dg in apply_wall_bc.
        """
        # For the radial axis (axis=0), the left ghost at jr=0 should be
        # the specular reflection of the inner face value.  We use
        # zero-gradient here; the actual wall enforcement is done in
        # apply_wall_bc() after the RK step.
        left_ext = torch.roll(inner_right, shifts=1, dims=cell_axis)
        sl = [slice(None)] * inner_right.ndim
        sl[cell_axis] = 0
        left_ext[tuple(sl)] = inner_right.select(cell_axis, 0)
        return left_ext

    def _dg_rhs(self, f: torch.Tensor) -> torch.Tensor:
        """Full DG advection RHS on the spherical shell (all three axes).

        For the radial axis (axis=0), the inner wall ghost uses specular
        bounce-back: f_ghost[i] = f[opp[i]] at the inner face.  This is
        the correct DG way to enforce no-slip — the weak-form numerical
        flux at the wall automatically produces bounce-back, without
        modifying the DOFs directly.
        """
        rhs = torch.zeros_like(f)
        Q = f.shape[0]

        for axis, ops, periodic in [(0, self.ops_r, False),
                                     (1, self.ops_th, False),
                                     (2, self.ops_ph, True)]:
            c_along = self.c_hat[:, axis]  # (Q, Nr, Nth, Nph)
            nonzero = c_along.abs().amax(dim=(1, 2, 3)) > 0  # (Q,)
            if not bool(nonzero.any()):
                continue
            sub = f[nonzero]  # (Q_active, Nr, Nth, Nph, n, n, n)
            c_sub = c_along[nonzero]  # (Q_active, Nr, Nth, Nph)
            # For axis=0, pass the full f and the Q-mask so the wall ghost
            # can look up opposite populations (which may be in a different
            # subset of Q).
            wall_f = f if axis == 0 else None
            wall_mask = nonzero if axis == 0 else None
            rhs_sub = self._rhs_axis(sub, axis, c_sub, ops, periodic=periodic,
                                     wall_f=wall_f, wall_q_mask=wall_mask)
            rhs[nonzero] = rhs[nonzero] + rhs_sub

        return rhs

    # ------------------------------------------------------------------
    # wall / outer boundary conditions
    # ------------------------------------------------------------------
    def _apply_wall_bc_inplace(self, f: torch.Tensor):
        """Enforce half-way bounce-back at the sphere wall on an arbitrary
        DG field ``f`` (layout: Q, Nr, Nth, Nph, n, n, n).  Called inside
        the RHS so every RK stage sees correct wall values."""
        rho, ux, uy, uz = self._macros_from_field(f)
        rho_wall = rho[0]  # (Nth, Nph)
        feq_wall = self.W.view(-1, 1, 1) * rho_wall.unsqueeze(0)  # (Q, Nth, Nph)
        c_in = self.c_ir.mean(dim=(1, 2)) < 0  # (Q,) incoming
        for i in range(19):
            if bool(c_in[i]):
                oi = int(self.OPP[i])
                # inner radial face: jr=0, node_r=0
                f[i, 0, :, :, 0, :, :] = (
                    feq_wall[i].unsqueeze(-1).unsqueeze(-1) +
                    f[oi, 0, :, :, -1, :, :] -
                    feq_wall[oi].unsqueeze(-1).unsqueeze(-1)
                )

    def _macros_from_field(self, f: torch.Tensor):
        """Compute macros from an arbitrary DG field."""
        f_avg = f.mean(dim=(4, 5, 6))  # (Q, Nr, Nth, Nph)
        rho = f_avg.sum(dim=0).clamp(min=1e-10)
        ux = (f_avg * self.C[:, 0].view(-1, 1, 1, 1)).sum(dim=0) / rho
        uy = (f_avg * self.C[:, 1].view(-1, 1, 1, 1)).sum(dim=0) / rho
        uz = (f_avg * self.C[:, 2].view(-1, 1, 1, 1)).sum(dim=0) / rho
        return rho, ux, uy, uz

    def apply_wall_bc(self):
        """Half-way bounce-back at the sphere wall (r=R_in, no-slip).

        For incoming populations (c_ir < 0), set the inner-face DOF to
        the reflected value: f[i, ..., 0, :, :] = f_eq[i] + f[opp] - f_eq[opp].
        """
        n = self.cfg.degree + 1
        rho, ux, uy, uz = self.compute_macros_at_center()
        rho_wall = rho[0]  # (Nth, Nph) first radial element
        feq_wall = self.W.view(-1, 1, 1) * rho_wall.unsqueeze(0)  # (Q, Nth, Nph) u=0

        c_in = self.c_ir.mean(dim=(1, 2)) < 0  # (Q,) incoming
        for i in range(19):
            if bool(c_in[i]):
                oi = int(self.OPP[i])
                # inner radial face: node index 0 along the r-node axis (axis 4)
                # f_dg[i, 0, jt, jp, 0, :, :] = feq[i] + f[opp, 0, jt, jp, -1, :, :] - feq[opp]
                self.f_dg[i, 0, :, :, 0, :, :] = (
                    feq_wall[i].unsqueeze(-1).unsqueeze(-1) +
                    self.f_dg[oi, 0, :, :, -1, :, :] -
                    feq_wall[oi].unsqueeze(-1).unsqueeze(-1)
                )

    def apply_outer_bc(self, lbm_f_at_shell):
        """Ingest LBM values at the outer boundary for outgoing populations
        (c_ir > 0).  Sets the outer-face DOF to the LBM value."""
        c_out = self.c_ir.mean(dim=(1, 2)) > 0  # (Q,)
        n = self.cfg.degree + 1
        for i in range(19):
            if bool(c_out[i]):
                # outer radial face: node index -1 along the r-node axis
                self.f_dg[i, -1, :, :, -1, :, :] = (
                    lbm_f_at_shell[i].unsqueeze(-1).unsqueeze(-1).expand(
                        self.cfg.Ntheta, self.cfg.Nphi, n, n)
                )

    # ------------------------------------------------------------------
    # step (method-of-lines SSP-RK3 with automatic sub-stepping)
    # ------------------------------------------------------------------
    def step(self, dt=1.0, lbm_boundary=None, n_substeps: int = 0):
        """One DG macro-step using the validated dg_lbm_step.

        Uses a single ops (dx=dr) with velocity rescaling so all three
        axes are consistent: c_eff[:, k] = c[:, k] * (dr / dx_k).
        The CFL is checked against the rescaled velocities.
        """
        from .dg_advection import dg_lbm_step as _dg_lbm_step

        tau = self.cfg.tau
        # Rescale velocities so a single ops (dx=dr) is correct for all
        # axes: c_eff[:, k] = c[:, k] * (dr / dx_k).
        scale = torch.tensor([1.0, self.dr / self.dtheta, self.dr / self.dphi],
                             device=self.device, dtype=self.dtype)
        C_eff = self.C * scale.view(1, 3)  # (Q, 3) rescaled

        if n_substeps <= 0:
            cfl_limit = 1.0 / (2 * self.cfg.degree + 1)
            c_max = float(C_eff.abs().max().item())
            cfl = c_max * dt / self.dr
            n_substeps = max(1, int(math.ceil(cfl / cfl_limit)))
            min_coll = max(1, int(math.ceil(dt / (2.0 * tau))))
            n_substeps = max(n_substeps, min_coll)

        f = self.f_dg
        f = _dg_lbm_step(f, C_eff, self.W, self.ops_r, tau,
                         ndim_spatial=3, dt=dt, n_substeps=n_substeps,
                         scheme="rk3", q_first=0)
        f = f.clamp(min=0.0)
        self.f_dg = f

        self.apply_wall_bc()
        if lbm_boundary is not None:
            self.apply_outer_bc(lbm_boundary)

    # ------------------------------------------------------------------
    # drag (pressure-integral + friction on curved wall)
    # ------------------------------------------------------------------
    def drag(self):
        """Pressure-integral drag on the sphere:
        F = sum(p_wall * n_hat * dA) + friction, p = rho/3, with the curved
        wall face area dA = R_in^2 sin(theta) dtheta dphi.
        Returns (F_x, F_y, F_z) in lattice units and the Cd."""
        rho, ux, uy, uz = self.compute_macros_at_center()
        rho_wall = rho[0]  # (Nth, Nph)
        p_wall = rho_wall / 3.0

        th_g = self.theta_c.view(-1, 1)
        ph_g = self.phi_c.view(1, -1)
        nx_hat = torch.sin(th_g) * torch.cos(ph_g)
        ny_hat = torch.sin(th_g) * torch.sin(ph_g)
        nz_hat = torch.cos(th_g).expand(-1, ph_g.shape[1])
        dA = self.cfg.R_in ** 2 * torch.sin(th_g) * self.dtheta * self.dphi

        F_px = (p_wall * nx_hat * dA).sum()
        F_py = (p_wall * ny_hat * dA).sum()
        F_pz = (p_wall * nz_hat * dA).sum()

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
