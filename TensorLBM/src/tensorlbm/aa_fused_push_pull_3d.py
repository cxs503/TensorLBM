"""AA-pattern fused stream-collide (push-pull) for D3Q19, with inline halfway
bounce-back and Guo forcing.

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`` (lanl/MF-LBM
``aa-pattern-fused-stream-collide``, D3Q19).

Why a 3D sibling of :mod:`tensorlbm.aa_fused_push_pull`
------------------------------------------------------
The reference practice (MF-LBM, Los Alamos; Kruger et al. 2017 ch. 5) is a 3D
D3Q19 code.  The 2D module in this package documents its own scope limit:
"D2Q9 / BGK only.  Nothing here transfers automatically to MRT/TRT or
D3Q19/27 variants of the pattern".  This module is the D3Q19 transfer, and it is
verified against the same kind of oracle: exact solutions and a bit-identical
materialised reference, not against itself.

The pattern (identical algebra to the 2D kernel, D3Q19 stencil)
---------------------------------------------------------------
The fused pull exploits ``rho`` and ``u`` being collision invariants: the
post-collision population leaving a source node ``x - c_i`` is reconstructible
on the fly, so no post-collision array is ever materialised,

    f_i(x, t+1) = f_i(x - c_i, t) + omega [ f_i^eq(x - c_i, t) - f_i(x - c_i, t) ]

which is the operator ``S[C[f]]``, *not* ``C[S[f]]``.  The two do not commute
(see :mod:`tensorlbm.aa_fused_stream_collide` for the negative control).

Halfway bounce-back stays inside the fused loop: when the upstream node
``x - c_i`` is solid, the incoming population is the reflection of the outgoing
one at ``x``,

    f_i(x, t+1) = f_j^post(x, t) + 6 w_j rho(x) (c_j . u_wall),   j = opp(i)

with ``c_j`` the *reflected* (outgoing) direction, so the wall term keeps the
sign convention of the verified 2D kernel and of
:func:`collide_stream_step` here.

Body force uses Guo's second-order scheme (Guo, Zheng & Shi, JCP 165 (2000) 288)

    S_i = w_i [ 3 (c_i - u).F + 9 (c_i.u)(c_i.F) ],  f_i^post = f_i + omega(f_i^eq - f_i) + (1 - omega/2) S_i

so that a plane channel has the analytic steady solution mediated by
:func:`analytic_poiseuille` (``u_max = g H^2 / (8 nu)``, ``nu = (tau-1/2)/3``).

Conventions: ``f`` has shape ``(19, nz, ny, nx)``; lattice ordering, weights and
opposite map are imported from :mod:`tensorlbm.d3q19` (not re-typed), so the two
cannot drift apart.  Walls of the channel helper are the first/last rows of the
``y`` axis; ``x`` and ``z`` are periodic.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from tensorlbm import d3q19 as _L

ND = 19
CX: Tuple[int, ...] = tuple(int(v) for v in _L.C[:, 0])
CY: Tuple[int, ...] = tuple(int(v) for v in _L.C[:, 1])
CZ: Tuple[int, ...] = tuple(int(v) for v in _L.C[:, 2])
W19: Tuple[float, ...] = tuple(float(v) for v in _L.W)
OPP: Tuple[int, ...] = tuple(int(v) for v in _L.OPPOSITE)
CS2 = 1.0 / 3.0

# sanity: the imported constants must be a consistent D3Q19 stencil
assert len(CX) == len(CY) == len(CZ) == len(W19) == len(OPP) == ND
assert abs(sum(W19) - 1.0) < 1e-15


def _lattice(device, dtype):
    c = torch.tensor([(CX[i], CY[i], CZ[i]) for i in range(ND)], dtype=dtype, device=device)
    w = torch.tensor(W19, dtype=dtype, device=device)
    return c, w


def _pull(a: torch.Tensor, i: int) -> torch.Tensor:
    """Value at node ``x - c_i`` (periodic pull) of a field ``(..., nz, ny, nx)``."""
    return torch.roll(a, shifts=(CZ[i], CY[i], CX[i]), dims=(-3, -2, -1))


def _push(a: torch.Tensor, i: int) -> torch.Tensor:
    """Value at node ``x + c_i`` (periodic shift) of a field ``(..., nz, ny, nx)``."""
    return torch.roll(a, shifts=(-CZ[i], -CY[i], -CX[i]), dims=(-3, -2, -1))


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor,
                uz: torch.Tensor) -> torch.Tensor:
    """D3Q19 second-order equilibrium, returns ``(19, nz, ny, nx)``."""
    c, w = _lattice(rho.device, rho.dtype)
    cu = (c[:, 0].view(-1, 1, 1, 1) * ux.unsqueeze(0)
          + c[:, 1].view(-1, 1, 1, 1) * uy.unsqueeze(0)
          + c[:, 2].view(-1, 1, 1, 1) * uz.unsqueeze(0))
    u2 = ux * ux + uy * uy + uz * uz
    return w.view(-1, 1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2.unsqueeze(0)
    )


def macroscopic(f: torch.Tensor, force: Optional[torch.Tensor] = None):
    """Density and velocity from ``f`` (Guo half-force shift when ``force`` given)."""
    c, _ = _lattice(f.device, f.dtype)
    rho = f.sum(dim=0)
    mx = (f * c[:, 0].view(-1, 1, 1, 1)).sum(dim=0)
    my = (f * c[:, 1].view(-1, 1, 1, 1)).sum(dim=0)
    mz = (f * c[:, 2].view(-1, 1, 1, 1)).sum(dim=0)
    if force is not None:
        mx = mx + 0.5 * force[0]
        my = my + 0.5 * force[1]
        mz = mz + 0.5 * force[2]
    return rho, mx / rho, my / rho, mz / rho


def _collision_increment(f, rho, ux, uy, uz, tau, force=None):
    """``omega (f^eq - f) [+ (1 - omega/2) S_i]`` on the pre-collision state."""
    c, w = _lattice(f.device, f.dtype)
    omega = 1.0 / tau
    inc = omega * (equilibrium(rho, ux, uy, uz) - f)
    if force is not None:
        cu = (c[:, 0].view(-1, 1, 1, 1) * ux.unsqueeze(0)
              + c[:, 1].view(-1, 1, 1, 1) * uy.unsqueeze(0)
              + c[:, 2].view(-1, 1, 1, 1) * uz.unsqueeze(0))
        cF = (c[:, 0].view(-1, 1, 1, 1) * force[0].unsqueeze(0)
              + c[:, 1].view(-1, 1, 1, 1) * force[1].unsqueeze(0)
              + c[:, 2].view(-1, 1, 1, 1) * force[2].unsqueeze(0))
        uF = ux * force[0] + uy * force[1] + uz * force[2]
        S = w.view(-1, 1, 1, 1) * (3.0 * (cF - uF.unsqueeze(0)) + 9.0 * cu * cF)
        inc = inc + (1.0 - 0.5 * omega) * S
    return inc


def _check_shape(f: torch.Tensor) -> None:
    if f.dim() != 4 or f.shape[0] != ND:
        raise ValueError(f"f must have shape (19, nz, ny, nx), got {tuple(f.shape)}")


def fused_step(f: torch.Tensor, tau: float,
               solid: Optional[torch.Tensor] = None,
               wall_u: Optional[torch.Tensor] = None,
               force: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One AA-pattern fused stream-collide step (single pull pass, no post array).

    Parameters
    ----------
    f : (19, nz, ny, nx) tensor
        Pre-collision populations at ``t``.
    tau : float
        BGK relaxation time, ``nu = (tau - 0.5)/3``.
    solid : (nz, ny, nx) bool tensor, optional
        ``True`` on solid nodes (halfway bounce-back between a solid node and
        its fluid neighbour).  Default: fully periodic.
    wall_u : (3, nz, ny, nx) tensor, optional
        Wall velocity, sampled on solid nodes.
    force : (3, nz, ny, nx) tensor, optional
        Body force per unit volume (Guo forcing).

    Returns
    -------
    (19, nz, ny, nx) tensor
        Post-collision, post-streaming populations at ``t+1``.
    """
    _check_shape(f)
    rho, ux, uy, uz = macroscopic(f, force)
    inc = _collision_increment(f, rho, ux, uy, uz, tau, force)

    # --- fused pull: population + collision increment from the source node ---
    fnew = torch.empty_like(f)
    fnew[0] = f[0] + inc[0]
    for i in range(1, ND):
        fnew[i] = _pull(f[i], i) + _pull(inc[i], i)

    # --- inline halfway bounce-back (upstream node solid) ---
    if solid is not None:
        for i in range(1, ND):
            upstream_solid = _pull(solid, i)
            if not bool(upstream_solid.any()):
                continue
            j = OPP[i]
            val = f[j] + inc[j]
            if wall_u is not None:
                uwx = _pull(wall_u[0], i)
                uwy = _pull(wall_u[1], i)
                uwz = _pull(wall_u[2], i)
                val = val + 6.0 * W19[i] * rho * (
                    CX[i] * uwx + CY[i] * uwy + CZ[i] * uwz
                )
            fnew[i] = torch.where(upstream_solid, val, fnew[i])
    return fnew


def collide_stream_step(f: torch.Tensor, tau: float,
                        solid: Optional[torch.Tensor] = None,
                        wall_u: Optional[torch.Tensor] = None,
                        force: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Canonical materialised reference: collide everywhere, then stream, then BB.

    Independent of the fused pass only in *storage discipline*; the algebra is
    the same, so the two agree bit for bit (that is the non-vacuous equivalence
    property asserted by the tests, rather than a tolerance comparison).
    """
    _check_shape(f)
    rho, ux, uy, uz = macroscopic(f, force)
    post = f + _collision_increment(f, rho, ux, uy, uz, tau, force)

    fnew = torch.empty_like(f)
    for i in range(ND):
        fnew[i] = _pull(post[i], i)

    if solid is not None:
        for i in range(1, ND):
            downstream_solid = _push(solid, i)
            if not bool(downstream_solid.any()):
                continue
            j = OPP[i]
            val = post[i]
            if wall_u is not None:
                uwx = _push(wall_u[0], i)
                uwy = _push(wall_u[1], i)
                uwz = _push(wall_u[2], i)
                val = val + 6.0 * W19[j] * rho * (
                    CX[j] * uwx + CY[j] * uwy + CZ[j] * uwz
                )
            fnew[j] = torch.where(downstream_solid, val, fnew[j])
    return fnew


def aa_fused_stream_collide_3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Fused D3Q19 stream-collide on a fully periodic domain (no post array)."""
    return fused_step(f, tau)


def separate_stream_collide_3d(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Materialised D3Q19 collide-then-stream oracle (fully periodic)."""
    return collide_stream_step(f, tau)


# --------------------------------------------------------------------------- #
# geometry + analytic references
# --------------------------------------------------------------------------- #
def channel_masks_3d(nz: int, ny: int, nx: int, solid_rows: int = 1,
                     device="cpu") -> torch.Tensor:
    """Solid mask with walls on the first/last ``solid_rows`` rows of the y axis."""
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool, device=device)
    if solid_rows > 0:
        solid[:, :solid_rows, :] = True
        solid[:, -solid_rows:, :] = True
    return solid


def wall_velocity_3d(solid: torch.Tensor, u_wall, row: int = -1) -> torch.Tensor:
    """Wall-velocity field ``(3, nz, ny, nx)`` on the solid row ``row``."""
    nz, ny, nx = solid.shape
    wall_u = torch.zeros((3, nz, ny, nx), dtype=torch.float64, device=solid.device)
    r = row % ny
    for a in range(3):
        wall_u[a, :, r, :] = float(u_wall[a]) * solid[:, r, :].to(wall_u.dtype)
    return wall_u


def analytic_poiseuille(n_fluid: int, g: float, nu: float):
    """Plane Poiseuille flow between halfway walls (wall distance 1/2 outside).

    Channel height ``H = n_fluid``; ``u(y) = g/(2 nu) (y + 1/2)(n_fluid - 1/2 - y)``
    and ``u_max = g H^2 / (8 nu)``.
    """
    y = torch.arange(n_fluid, dtype=torch.float64)
    u = g / (2.0 * nu) * (y + 0.5) * (n_fluid - 0.5 - y)
    return u, g * n_fluid ** 2 / (8.0 * nu)


def analytic_couette(n_fluid: int, u_wall: float):
    """Plane Couette flow between halfway walls (top wall moving at ``u_wall``)."""
    y = torch.arange(n_fluid, dtype=torch.float64)
    return u_wall * (y + 0.5) / n_fluid


def taylor_green_exact(n: int, t: float, u0: float, k_lat: float, nu: float):
    """2D-extended Taylor-Green vortex: an *exact* solution of 3D Navier-Stokes.

    ``ux =  u0 cos(k x) sin(k y) e^{-2 nu k^2 t}``
    ``uy = -u0 sin(k x) cos(k y) e^{-2 nu k^2 t}``, ``uz = 0``, uniform in ``z``,
    so it is an exact solution in three dimensions *with a non-zero convective
    term* (``u.grad u`` does not vanish).  It exercises the in-plane diagonals of
    D3Q19 and the viscous decay rate ``2 nu k^2``.

    All quantities are in **lattice units**: ``x, y = 0 .. n-1`` and ``k_lat``
    is the lattice wavenumber (``2 pi m / n`` for ``m`` periods over the
    domain).  Using a physical ``k = 1`` here is the classic way to produce a
    wrong "exact" reference and a fictitious 20x viscosity error.
    """
    x = torch.arange(n, dtype=torch.float64)
    X, Y = torch.meshgrid(x, x, indexing="xy")
    decay = float(torch.exp(torch.tensor(-2.0 * nu * k_lat * k_lat * t,
                                         dtype=torch.float64)))
    ux = u0 * torch.cos(k_lat * X) * torch.sin(k_lat * Y) * decay
    uy = -u0 * torch.sin(k_lat * X) * torch.cos(k_lat * Y) * decay
    return ux, uy, torch.zeros_like(ux)


def taylor_green_initial(nz: int, n: int, u0: float, k_lat: float) -> torch.Tensor:
    """Equilibrium initial condition for the 2D-extended Taylor-Green vortex."""
    ux, uy, uz = taylor_green_exact(n, 0.0, u0, k_lat, 0.0)
    rho = torch.ones_like(ux)
    f = equilibrium(rho, ux, uy, uz)
    if f.dim() == 3:                      # (19, ny, nx) when no z axis entered
        f = f.unsqueeze(1)
    return f.expand(-1, nz, -1, -1).contiguous()


def shear_wave_exact(ny: int, t: float, u0: float, k_lat: float, nu: float):
    """Decaying shear wave ``ux(y) = u0 cos(k y) e^{-nu k^2 t}`` (no convection).

    Pure viscous decay of a transverse mode: it measures the *transport
    coefficient* of the kernel, independent of the nonlinear term.
    """
    y = torch.arange(ny, dtype=torch.float64)
    decay = float(torch.exp(torch.tensor(-nu * k_lat * k_lat * t, dtype=torch.float64)))
    return u0 * torch.cos(k_lat * y) * decay


def shear_wave_initial(nz: int, ny: int, nx: int, u0: float, k_lat: float) -> torch.Tensor:
    """Equilibrium state of the shear wave ``ux = u0 cos(k y)``."""
    y = torch.arange(ny, dtype=torch.float64)
    prof = u0 * torch.cos(k_lat * y)
    ux = prof.view(1, ny, 1).repeat(nz, 1, nx)
    zero = torch.zeros_like(ux)
    return equilibrium(torch.ones_like(ux), ux, zero, zero)


def relative_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    """``||a - b||_2 / ||b||_2`` (scalar float)."""
    den = torch.linalg.vector_norm(b)
    if float(den) == 0.0:
        return float(torch.linalg.vector_norm(a - b))
    return float(torch.linalg.vector_norm(a - b) / den)