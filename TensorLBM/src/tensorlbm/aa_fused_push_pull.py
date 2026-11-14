"""AA-pattern fused push-pull (stream-collide) kernel for D2Q9, with inline
halfway bounce-back.

Reference practice
------------------
``lbm-principles-practice/code`` (Kruger et al., *The Lattice Boltzmann Method:
Principles and Practice*, Springer 2017), chapter 5 "Implementation" and the
AA-pattern ("Abstract Array") / fused push-pull discussion, plus the halfway
bounce-back rule of Eq. (5.58).

What the AA fused push-pull pattern actually is
-----------------------------------------------
The textbook AA pattern fuses *collision and streaming* by exploiting the fact
that the collision operator conserves density and momentum:

    rho(x,t) and u(x,t) are identical before and after collision.

Therefore the *post-collision* population leaving a source node ``x - c_i`` can
be reconstructed on the fly while pulling, without ever materialising a
post-collision array:

    f_i(x, t+1) = f_i(x - c_i, t) + omega * (f_i^eq(x - c_i, t) - f_i(x - c_i, t))
                  \_______________pull__________/   \____collision increment at source____/

Both the pre-collision population and the collision *increment* are evaluated at
the **source** node (obtained by a shifted read), which makes the scheme exactly
equivalent to "collide everywhere, then stream" -- not merely equivalent at
steady state.  This is the property that this module implements and verifies.

Boundary conditions in the pull form
-------------------------------------
Halfway bounce-back is expressed *inside* the fused loop: when the upstream node
``x - c_i`` of direction ``i`` is solid, the incoming population is the
reflection of the population that is leaving ``x`` towards the wall:

    f_i(x, t+1) = f_opp(i)^post(x, t) + 6 w_i rho(x) (c_i . u_wall) ,  c_j = -c_i

with ``u_wall`` the velocity of the adjacent wall node.  ``f^post`` is again
available from ``f + increment`` at the same node, so no extra storage is
needed.  This conserves mass exactly (to round-off) in a closed domain, unlike a
bounce-back rule that is written on *solid* nodes.

Body force
----------
Optional constant body force ``F`` (per unit volume, lattice units) is applied
with Guo's second-order forcing (Guo, Zheng & Shi, J. Comput. Phys. 165 (2000)
288):

    S_i   = w_i [ 3 (c_i - u).F + 9 (c_i.u)(c_i.F) ]
    f_i^post = f_i + omega (f_i^eq - f_i) + (1 - omega/2) S_i
    rho u = sum_i c_i f_i + F/2

so that steady plane Poiseuille flow has the analytic solution
``u_max = g H^2 / (8 nu)`` with ``nu = (tau - 1/2)/3``.

Conventions: ``f`` has shape ``(9, ny, nx)``, velocity index ordering
``(0,0), (1,0), (0,1), (-1,0), (0,-1), (1,1), (-1,1), (-1,-1), (1,-1)``
(identical to :mod:`tensorlbm.d2q9`).
"""

from __future__ import annotations

from typing import Optional

import torch

# D2Q9 lattice (ordering identical to tensorlbm.d2q9)
CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
W9 = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
      1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)  # opposite of direction i
CS2 = 1.0 / 3.0


def _lattice(device, dtype):
    c = torch.tensor([(CX[i], CY[i]) for i in range(9)], dtype=dtype, device=device)
    w = torch.tensor(W9, dtype=dtype, device=device)
    return c, w


def _pull(a: torch.Tensor, i: int) -> torch.Tensor:
    """Value at node ``x - c_i`` (periodic pull) of a field ``(..., ny, nx)``."""
    return torch.roll(a, shifts=(CY[i], CX[i]), dims=(-2, -1))


def _push(a: torch.Tensor, i: int) -> torch.Tensor:
    """Value at node ``x + c_i`` (periodic shift) of a field ``(..., ny, nx)``."""
    return torch.roll(a, shifts=(-CY[i], -CX[i]), dims=(-2, -1))


def equilibrium(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """D2Q9 second-order equilibrium, returns ``(9, ny, nx)``."""
    c, w = _lattice(rho.device, rho.dtype)
    cu = c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
    u2 = ux * ux + uy * uy
    return w.view(-1, 1, 1) * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u2.unsqueeze(0))


def macroscopic(f: torch.Tensor, force: Optional[torch.Tensor] = None):
    """Density and velocity from ``f`` (Guo velocity shift when ``force`` given)."""
    c, _ = _lattice(f.device, f.dtype)
    rho = f.sum(dim=0)
    mx = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0)
    my = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0)
    if force is not None:
        mx = mx + 0.5 * force[0]
        my = my + 0.5 * force[1]
    return rho, mx / rho, my / rho


def _collision_increment(f, rho, ux, uy, tau, force=None):
    """omega*(f^eq - f) [+ (1-omega/2) S_i for Guo forcing]."""
    c, w = _lattice(f.device, f.dtype)
    omega = 1.0 / tau
    inc = omega * (equilibrium(rho, ux, uy) - f)
    if force is not None:
        cu = c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
        cF = (c[:, 0].view(-1, 1, 1) * force[0].unsqueeze(0)
              + c[:, 1].view(-1, 1, 1) * force[1].unsqueeze(0))
        uF = ux * force[0] + uy * force[1]
        S = w.view(-1, 1, 1) * (3.0 * (cF - uF.unsqueeze(0)) + 9.0 * cu * cF)
        inc = inc + (1.0 - 0.5 * omega) * S
    return inc


def fused_push_pull_step(f: torch.Tensor, tau: float,
                         solid: Optional[torch.Tensor] = None,
                         wall_u: Optional[torch.Tensor] = None,
                         force: Optional[torch.Tensor] = None) -> torch.Tensor:
    """One fused AA-pattern push-pull step (collide+stream in a single pass).

    Parameters
    ----------
    f : (9, ny, nx) tensor
        Pre-collision populations at time ``t``.
    tau : float
        BGK relaxation time (``nu = (tau - 0.5)/3``).
    solid : (ny, nx) bool tensor, optional
        ``True`` on solid nodes. Default: no solid nodes (fully periodic).
    wall_u : (2, ny, nx) tensor, optional
        Wall velocity field, sampled on solid nodes (used only where a solid
        node is the upstream neighbour). Default: stationary walls.
    force : (2, ny, nx) tensor, optional
        Constant body force per unit volume (Guo forcing).

    Returns
    -------
    (9, ny, nx) tensor
        Post-collision, post-streaming populations at time ``t+1``.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        raise ValueError(f"f must have shape (9, ny, nx), got {tuple(f.shape)}")

    rho, ux, uy = macroscopic(f, force)
    inc = _collision_increment(f, rho, ux, uy, tau, force)

    # --- fused pull: population + collision increment from the source node ---
    fnew = torch.empty_like(f)
    fnew[0] = f[0] + inc[0]
    for i in range(1, 9):
        fnew[i] = _pull(f[i], i) + _pull(inc[i], i)

    # --- inline halfway bounce-back ---
    if solid is not None:
        for i in range(1, 9):
            upstream_solid = _pull(solid, i)
            if not bool(upstream_solid.any()):
                continue
            j = OPP[i]
            val = f[j] + inc[j]
            if wall_u is not None:
                uwx = _pull(wall_u[0], i)   # wall velocity of the upstream (solid) node
                uwy = _pull(wall_u[1], i)
                val = val + 6.0 * W9[i] * rho * (CX[i] * uwx + CY[i] * uwy)
            fnew[i] = torch.where(upstream_solid, val, fnew[i])
    return fnew


def collide_stream_step(f: torch.Tensor, tau: float,
                        solid: Optional[torch.Tensor] = None,
                        wall_u: Optional[torch.Tensor] = None,
                        force: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Independent reference: collide everywhere, then stream, then bounce-back.

    This is the canonical "push" formulation (materialises the post-collision
    array). It is the oracle the fused kernel is checked against; the two are
    algebraically identical, so differences are round-off only.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        raise ValueError(f"f must have shape (9, ny, nx), got {tuple(f.shape)}")

    rho, ux, uy = macroscopic(f, force)
    post = f + _collision_increment(f, rho, ux, uy, tau, force)

    fnew = torch.empty_like(f)
    for i in range(9):
        # streaming into x reads the post-collision state at the upstream node x - c_i
        fnew[i] = _pull(post[i], i)

    if solid is not None:
        for i in range(1, 9):
            downstream_solid = _push(solid, i)  # x + c_i is solid
            if not bool(downstream_solid.any()):
                continue
            j = OPP[i]
            val = post[i]
            if wall_u is not None:
                uwx = _push(wall_u[0], i)
                uwy = _push(wall_u[1], i)
                val = val + 6.0 * W9[j] * rho * (CX[j] * uwx + CY[j] * uwy)
            fnew[j] = torch.where(downstream_solid, val, fnew[j])
    return fnew


# --------------------------------------------------------------------------- #
# analytic references
# --------------------------------------------------------------------------- #
def analytic_poiseuille(n_fluid: int, g: float, nu: float):
    """Plane Poiseuille flow between halfway walls.

    Walls sit at ``y = -0.5`` and ``y = n_fluid - 0.5`` (halfway bounce-back),
    hence the channel height is ``H = n_fluid`` and

        u(y) = g/(2 nu) * (y + 0.5) * (n_fluid - 0.5 - y),   y = 0 .. n_fluid-1
        u_max = g H^2 / (8 nu).
    """
    y = torch.arange(n_fluid, dtype=torch.float64)
    u = g / (2.0 * nu) * (y + 0.5) * (n_fluid - 0.5 - y)
    return u, g * n_fluid**2 / (8.0 * nu)


def analytic_couette(n_fluid: int, u_wall: float):
    """Plane Couette flow between halfway walls (top wall moving at ``u_wall``)."""
    y = torch.arange(n_fluid, dtype=torch.float64)
    return u_wall * (y + 0.5) / n_fluid


def channel_masks(nx: int, ny: int, solid_rows=1, device="cpu"):
    """Solid mask for a channel walled at the first/last ``solid_rows`` rows."""
    solid = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    if solid_rows > 0:
        solid[:solid_rows, :] = True
        solid[-solid_rows:, :] = True
    return solid