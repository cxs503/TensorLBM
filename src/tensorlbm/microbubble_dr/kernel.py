"""D3Q19 Shan-Chen channel kernel for microbubble drag reduction (P106W2-N01).

Small, self-contained kernel: a plane channel between two half-way bounce-back
walls (``y = 0`` and ``y = ny-1``), periodic in ``x`` and ``z``, driven by a
**uniform per-volume body force** ``F0`` in ``x``.  A constant momentum source
per unit volume makes the mean total-stress budget exact for *any* density
field, ``tau_bottom + |tau_top| = F0*H``, which is what keeps the wall-shear
bookkeeping auditable in the two-phase runs (see
:mod:`tensorlbm.microbubble_dr.fik`).

Conventions (matched to :mod:`tensorlbm.multiphase`):
  * SC force ``F = -G psi(x) * sum_i w_i c_i psi(x + c_i)``, ``G > 0`` =
    attraction, critical coupling ``G_c = 4`` for ``psi = 1 - exp(-rho)``
    (spinodal at ``rho = ln 2``).
  * gas = light phase; net buoyancy ``F_y = (rho_l - rho) * g_b`` pushes the gas
    towards the **top** wall (``y = ny-1``) = the hull plate.

BGK collision with a single relaxation time ``tau``, so the dynamic viscosity is
``mu = rho * cs^2 * (tau - 1/2)``: the gas is less viscous in proportion to its
density, which is the physical source of the mixture-viscosity field used by the
drag-reduction decomposition.
"""

from __future__ import annotations

import torch

CS2 = 1.0 / 3.0

#: D3Q19 discrete velocities -- the library ordering.
C_LIST: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
    (1, 1, 0),
    (-1, -1, 0),
    (1, -1, 0),
    (-1, 1, 0),
    (1, 0, 1),
    (-1, 0, -1),
    (1, 0, -1),
    (-1, 0, 1),
    (0, 1, 1),
    (0, -1, -1),
    (0, 1, -1),
    (0, -1, 1),
)
W_LIST: tuple[float, ...] = (1.0 / 3.0,) + (1.0 / 18.0,) * 6 + (1.0 / 36.0,) * 12
OPP: tuple[int, ...] = (0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17)
OPP_INDEX = torch.tensor(OPP, dtype=torch.long)

C = torch.tensor([[c[0], c[1], c[2]] for c in C_LIST], dtype=torch.float32)  # (19,3)
W = torch.tensor(W_LIST, dtype=torch.float32)                                 # (19,)


def psi_exp(rho: torch.Tensor, rho0: float = 1.0) -> torch.Tensor:
    """Classic SC exponential pseudopotential ``psi = rho0 (1 - exp(-rho/rho0))``."""
    return rho0 * (1.0 - torch.exp(-rho / rho0))


def _shift(field: torch.Tensor, c: tuple[int, int, int], wall_psi: torch.Tensor | None) -> torch.Tensor:
    """Gather ``field(x + c)``; ``x``/``z`` periodic, ``y`` mirrored at the walls."""
    cx, cy, cz = c
    out = field
    if cz or cx:
        out = torch.roll(out, shifts=(cz, cx), dims=(0, 2))
    if cy == 0:
        return out
    shifted = torch.empty_like(out)
    if cy > 0:
        shifted[:, 1:, :] = out[:, :-1, :]
        shifted[:, 0, :] = out[:, 0, :] if wall_psi is None else wall_psi
    else:
        shifted[:, :-1, :] = out[:, 1:, :]
        shifted[:, -1, :] = out[:, -1, :] if wall_psi is None else wall_psi
    return shifted


def sc_force(
    rho: torch.Tensor,
    G: float,
    psi_fn=psi_exp,
    wall_psi: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shan-Chen self-interaction force (D3Q19); each returned field is ``(nz,ny,nx)``."""
    psi = psi_fn(rho)
    sx = torch.zeros_like(psi)
    sy = torch.zeros_like(psi)
    sz = torch.zeros_like(psi)
    for i, c in enumerate(C_LIST):
        w = W_LIST[i]
        if w == 0.0:
            continue
        g = _shift(psi, c, wall_psi)
        if c[0]:
            sx += w * c[0] * g
        if c[1]:
            sy += w * c[1] * g
        if c[2]:
            sz += w * c[2] * g
    return -G * psi * sx, -G * psi * sy, -G * psi * sz


def _times_c(field3: torch.Tensor) -> torch.Tensor:
    """``c_i . field`` for a 3-component field -> ``(n_dirs, ...)``.

    Written as explicit elementwise arithmetic (no ``tensordot``): the BLAS path
    of the loong64 torch build corrupted the heap on this shape, see the node
    report.
    """
    dev = field3.device
    c = C.to(dev).view(19, 3, *([1] * (field3.dim() - 1)))
    return c[:, 0] * field3[0].unsqueeze(0) + c[:, 1] * field3[1].unsqueeze(0) + c[:, 2] * field3[2].unsqueeze(0)


def equilibrium(rho: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """D3Q19 second-order equilibrium; ``u`` is ``(3, nz, ny, nx)``."""
    usq = (u * u).sum(dim=0)
    cu = _times_c(u)
    base = (1.0 - 1.5 * usq).unsqueeze(0) + 3.0 * cu + 4.5 * cu * cu
    return (W.to(rho.device).view(19, *([1] * rho.dim())) * rho.unsqueeze(0)) * base


def guo_source(u: torch.Tensor, F: torch.Tensor) -> torch.Tensor:
    """Bare Guo (1993) forcing source; ``u``, ``F`` are ``(3, nz, ny, nx)``.

    Returns ``G_i = w_i [3 (c_i - u).F + 9 (c_i.u)(c_i.F)]``.  The relaxation
    prefactors are applied in :meth:`ChannelKernel.step`, where they must be
    split between the symmetric and anti-symmetric parts for the TRT collision
    to conserve momentum exactly (see ``step``).
    """
    cu = _times_c(u)
    cF = _times_c(F)
    uF = (u * F).sum(dim=0).unsqueeze(0)
    return W.to(u.device).view(19, *([1] * (u.dim() - 1))) * (3.0 * (cF - uF) + 9.0 * cu * cF)


def momentum(f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rho, j)`` with ``j`` of shape ``(3, nz, ny, nx)`` (no half-force term).

    Explicit sums instead of ``tensordot`` (see :func:`_times_c`).
    """
    rho = f.sum(dim=0)
    jx = f[1] - f[2] + f[7] - f[8] + f[9] - f[10] + f[11] - f[12] + f[13] - f[14]
    jy = f[3] - f[4] + f[7] - f[8] - f[9] + f[10] + f[15] - f[16] + f[17] - f[18]
    jz = f[5] - f[6] + f[11] - f[12] - f[13] + f[14] + f[15] - f[16] - f[17] + f[18]
    return rho, torch.stack([jx, jy, jz])


def stream_bounce(fpost: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    """Streaming with half-way bounce-back on the two ``y`` walls.

    Returns ``(f_streamed, mem_bottom, mem_top)`` where ``mem_*`` is the
    x-momentum delivered to that wall per step, **averaged over the wall plane**
    (i.e. already a wall shear stress in lattice units).

    Momentum-exchange (Ladd) bookkeeping: a population hitting a wall delivers
    twice its x-momentum, ``2 c_ix f_i^post``, because the outgoing
    post-collision and the incoming reflected population both cross the link.
    """
    out = torch.empty_like(fpost)
    mem_bottom = 0.0
    mem_top = 0.0
    for i, c in enumerate(C_LIST):
        cx, cy, cz = c
        base = fpost[i]
        tmp = torch.roll(base, shifts=(cz, cx), dims=(0, 2)) if (cz or cx) else base
        if cy == 0:
            out[i] = tmp
        elif cy > 0:
            out[i, :, 1:, :] = tmp[:, :-1, :]
            out[i, :, 0, :] = fpost[OPP[i], :, 0, :]
            mem_top += 2.0 * cx * float(fpost[i, :, -1, :].mean())
        else:
            out[i, :, :-1, :] = tmp[:, 1:, :]
            out[i, :, -1, :] = fpost[OPP[i], :, -1, :]
            mem_bottom += 2.0 * cx * float(fpost[i, :, 0, :].mean())
    return out, mem_bottom, mem_top


class ChannelKernel:
    """D3Q19 BGK + Shan-Chen channel kernel (torch)."""

    def __init__(
        self,
        shape: tuple[int, int, int],
        tau: float = 0.53,
        G: float = 5.0,
        dtype: torch.dtype = torch.float32,
        lam_trt: float = 3.0 / 16.0,
    ) -> None:
        self.nz, self.ny, self.nx = shape
        self.tau = float(tau)
        self.G = float(G)
        self.dtype = dtype
        self.nu = CS2 * (self.tau - 0.5)
        self.shape = (self.nz, self.ny, self.nx)
        # TRT (two-relaxation-time) collision: the anti-symmetric rate is tied to
        # a magic parameter, which is what makes tau -> 1/2 (low viscosity,
        # required by Re_tau = 180 at lattice velocities < 0.1) usable.
        self.lam_trt = float(lam_trt)
        self.omega_p = 1.0 / self.tau
        self.omega_m = 1.0 / (self.lam_trt / (self.tau - 0.5) + 0.5)

    def field(self, rho0: float, u0: float = 0.0) -> torch.Tensor:
        """Equilibrium state of a uniform flow."""
        rho = torch.full(self.shape, float(rho0), dtype=self.dtype)
        u = torch.zeros((3,) + self.shape, dtype=self.dtype)
        u[0] = float(u0)
        return equilibrium(rho, u)

    def forces(self, rho: torch.Tensor, F0: float, g_b: float = 0.0, rho_l: float = 0.0) -> torch.Tensor:
        """Total force field ``(3, nz, ny, nx)``: SC + drive + gas buoyancy."""
        F = torch.zeros((3,) + tuple(rho.shape), dtype=rho.dtype, device=rho.device)
        if self.G != 0.0:
            sx, sy, sz = sc_force(rho, self.G)
            F[0] += sx
            F[1] += sy
            F[2] += sz
        F[0] += F0
        if g_b != 0.0:
            F[1] += (rho_l - rho) * g_b
        return F

    def step(
        self,
        f: torch.Tensor,
        F0: float,
        g_b: float = 0.0,
        rho_l: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """One time step; returns ``(f_new, diag)``."""
        rho, j = momentum(f)
        inv = 1.0 / rho
        F = self.forces(rho, F0, g_b, rho_l)
        # Guo: the velocity fed to the equilibrium carries half the force.
        uh = j * inv.unsqueeze(0) + 0.5 * F * inv.unsqueeze(0)
        feq = equilibrium(rho, uh)
        G = guo_source(uh, F)
        dev = G.device
        Gopp = torch.stack([G[k] for k in OPP])
        # TRT-consistent force splitting (Fei & Luo 2017): the symmetric part of
        # the source relaxes with omega_p, the anti-symmetric with omega_m.
        # Momentum after collision is then exactly rho*u + F (verified in
        # tests/test_microbubble_dr.py against the analytic laminar wall shear).
        src = (1.0 - 0.5 * self.omega_p) * 0.5 * (G + Gopp) + (1.0 - 0.5 * self.omega_m) * 0.5 * (G - Gopp)
        delta = f - feq
        dopp = torch.stack([delta[k] for k in OPP])
        dplus = 0.5 * (delta + dopp)
        dminus = 0.5 * (delta - dopp)
        fpost = feq + (1.0 - self.omega_p) * dplus + (1.0 - self.omega_m) * dminus + src
        f_new, mb, mt = stream_bounce(fpost)
        rho_new, j_new = momentum(f_new)
        ux = (j_new[0] + 0.5 * F[0]) * (1.0 / rho_new)
        diag = {
            "mass": float(rho_new.sum()),
            "rho_mean": float(rho_new.mean()),
            "ux_mean": float(ux.mean()),
            "mem_bottom": mb,
            "mem_top": mt,
        }
        return f_new, diag

    # -- equilibration helpers -----------------------------------------
    def relax_to_equilibrium(self, rho: torch.Tensor, u: torch.Tensor, n: int = 0) -> torch.Tensor:
        return equilibrium(rho, u)