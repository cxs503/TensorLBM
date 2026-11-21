"""Strict single-field (A-A) fused push-pull kernel, D2Q9, in place.

Anchor
------
``XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN`` (``lbm-principles-practice/code``, fused
push-pull AA pattern; Kruger et al., *The Lattice Boltzmann Method: Principles
and Practice*, Springer 2017, ch. 5).

Why this module exists
----------------------
:mod:`tensorlbm.aa_fused_push_pull` implements the *fusion* half of the anchor
(collision and streaming evaluated in one pull pass, no intermediate
post-collision array) but it still allocates a **second full distribution
field** per step (``fnew = torch.empty_like(f)``).  That is the A-B pattern with
fused arithmetic, not the A-A pattern: the A-A pattern's defining structural
property is that a *single* ``Q x N`` field is advanced in place, with only
``O(N)`` (here: one population plane plus compact wall buffers) auxiliary
storage.  This module supplies that missing property:

    ``aa_fused_push_pull_step(f, ...)`` mutates ``f`` in place and returns it;
    no second ``Q x N`` field is ever alive.

How in-place streaming is possible at all
----------------------------------------
Streaming is a permutation *within* each direction plane: after the step,
``f_i(x)`` depends only on old values of the same plane ``f_i``.  So plane ``i``
can be overwritten in place provided one ``O(N)`` copy of that plane exists:

    tmp = f_i.clone()                      # O(N)
    f_i(x) <- tmp(x - c_i) + inc_i(x - c_i)  # inc evaluated at the source node

The collision increment at the source needs ``rho`` and ``u`` at the source
(precomputed once per step: three ``O(N)`` planes) and the *old* ``f_i`` at the
source, which is exactly ``tmp(x - c_i)``.  No post-collision array is formed.

Halfway bounce-back without a second field
------------------------------------------
The pull form wants, for a direction ``i`` whose upstream node is solid,

    f_i(x, t+1) = f_{opp(i)}^post(x, t) + 6 w_i rho(x) (c_i . u_wall)

and ``f_{opp(i)}^post(x, t)`` is the *old* ``f`` of the opposite plane at the
same node.  That plane may already have been overwritten when the loop reaches
``i`` (the loop order is fixed, the wall direction is not).  The reflection is
therefore evaluated **before** the streaming loop -- while every plane is still
intact -- and buffered only at the nodes where it is actually needed (a compact
``O(surface)`` index/value pair per active direction), then scattered back after
streaming.  Both ``u_wall`` and ``rho`` are sampled from the same state, so the
buffered value is the analytic bounce-back value, not an approximation.

Verified equivalences (see ``tests/test_aa_fused_inplace.py``)
-------------------------------------------------------------
* bit-identical to :func:`tensorlbm.aa_fused_push_pull.fused_push_pull_step`
  (the fused A-B reference) for the periodic, walled, moving-wall and
  Guo-forced cases -- the two kernels are the same scheme, only the storage
  discipline differs;
* bit-identical to a from-scratch NumPy *scatter* oracle that shares no code
  with either kernel (``tests/test_aa_fused_independent_oracle.py``);
* plane Couette / plane Poiseuille against their analytic solutions (< 3 %);
* exact mass conservation in a closed walled box.

Conventions: ``f`` has shape ``(9, ny, nx)``; velocity ordering and weights are
those of :mod:`tensorlbm.d2q9`; BGK collision; Guo second-order forcing.
"""

from __future__ import annotations

from typing import Optional

import torch

# D2Q9 lattice (ordering identical to tensorlbm.d2q9 / aa_fused_push_pull)
CX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
CY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
W9 = (4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
      1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0)
OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)
CS2 = 1.0 / 3.0


def _shift_pull(a: torch.Tensor, i: int) -> torch.Tensor:
    """Value at node ``x - c_i`` (periodic) of a ``(ny, nx)`` field."""
    return torch.roll(a, shifts=(CY[i], CX[i]), dims=(-2, -1))


def _feq_dir(i: int, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor) -> torch.Tensor:
    """Second-order equilibrium for a single direction (scalar-field algebra)."""
    cu = CX[i] * ux + CY[i] * uy
    return W9[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * (ux * ux + uy * uy))


def _guo_dir(i: int, rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor,
             force: torch.Tensor) -> torch.Tensor:
    """Guo (2000) forcing source term for a single direction."""
    cu = CX[i] * ux + CY[i] * uy
    cF = CX[i] * force[0] + CY[i] * force[1]
    uF = ux * force[0] + uy * force[1]
    return W9[i] * (3.0 * (cF - uF) + 9.0 * cu * cF)


def macroscopic(f: torch.Tensor, force: Optional[torch.Tensor] = None):
    """Density and velocity of ``f`` (Guo velocity shift when ``force`` given)."""
    rho = f.sum(dim=0)
    mx = f[1] - f[3] + f[5] - f[6] - f[7] + f[8]
    my = f[2] - f[4] + f[5] + f[6] - f[7] - f[8]
    if force is not None:
        mx = mx + 0.5 * force[0]
        my = my + 0.5 * force[1]
    return rho, mx / rho, my / rho


def aa_fused_push_pull_step(f: torch.Tensor, tau: float,
                            solid: Optional[torch.Tensor] = None,
                            wall_u: Optional[torch.Tensor] = None,
                            force: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Advance ``f`` by one fused (collision+streaming) A-A step, **in place**.

    Parameters
    ----------
    f : (9, ny, nx) float tensor
        Pre-collision populations at time ``t``. **Mutated and returned.**
    tau : float
        BGK relaxation time, ``nu = (tau - 0.5)/3``.
    solid : (ny, nx) bool tensor, optional
        ``True`` on solid nodes; halfway bounce-back is applied at the fluid
        nodes next to them (single-node, staircase walls).
    wall_u : (2, ny, nx) tensor, optional
        Velocity field, sampled on solid nodes (moving walls). Stationary if
        omitted.
    force : (2, ny, nx) tensor, optional
        Constant body force per unit volume (Guo splitting).

    Returns
    -------
    torch.Tensor
        The same object as ``f``, holding the post-streaming state at ``t+1``.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        raise ValueError(f"f must have shape (9, ny, nx), got {tuple(f.shape)}")

    omega = 1.0 / tau
    g1 = 1.0 - 0.5 * omega
    rho, ux, uy = macroscopic(f, force)

    # --- bounce-back reflections, evaluated while every plane is still intact --
    reflections: list[tuple[int, torch.Tensor, torch.Tensor]] = []
    if solid is not None:
        for i in range(1, 9):
            upstream_solid = _shift_pull(solid, i)
            if not bool(upstream_solid.any()):
                continue
            idx = upstream_solid.nonzero(as_tuple=True)
            j = OPP[i]
            rho_w = rho[idx]
            ux_w, uy_w = ux[idx], uy[idx]
            fw = f[j][idx]
            val = fw + omega * (_feq_dir(j, rho_w, ux_w, uy_w) - fw)
            if force is not None:
                f_c = (force[0][idx], force[1][idx])
                val = val + g1 * _guo_dir(j, rho_w, ux_w, uy_w, f_c)
            if wall_u is not None:
                uwx = _shift_pull(wall_u[0], i)[idx]
                uwy = _shift_pull(wall_u[1], i)[idx]
                val = val + 6.0 * W9[i] * rho_w * (CX[i] * uwx + CY[i] * uwy)
            reflections.append((i, idx, val))

    # --- rest direction (no streaming) ---------------------------------------
    f0 = f[0].clone()
    inc0 = omega * (_feq_dir(0, rho, ux, uy) - f0)
    if force is not None:
        inc0 = inc0 + g1 * _guo_dir(0, rho, ux, uy, force)
    f[0] = f0 + inc0

    # --- streaming planes 1..8 in place, one O(N) plane copy at a time -------
    for i in range(1, 9):
        tmp = f[i].clone()                 # O(N): old values of plane i
        src = torch.roll(tmp, shifts=(CY[i], CX[i]), dims=(-2, -1))
        rho_s = _shift_pull(rho, i)
        ux_s = _shift_pull(ux, i)
        uy_s = _shift_pull(uy, i)
        inc = omega * (_feq_dir(i, rho_s, ux_s, uy_s) - src)
        if force is not None:
            # the Guo term belongs to the source node, hence the shifted read
            f_s = (_shift_pull(force[0], i), _shift_pull(force[1], i))
            inc = inc + g1 * _guo_dir(i, rho_s, ux_s, uy_s, f_s)
        f[i] = src + inc

    # --- scatter the buffered bounce-back values -----------------------------
    for i, idx, val in reflections:
        f[i][idx] = val
    return f


# --------------------------------------------------------------------------- #
# analytic references (same convention as aa_fused_push_pull)
# --------------------------------------------------------------------------- #
def analytic_poiseuille(n_fluid: int, g: float, nu: float):
    """Plane Poiseuille between halfway walls at ``y = -0.5`` / ``n - 0.5``."""
    y = torch.arange(n_fluid, dtype=torch.float64)
    u = g / (2.0 * nu) * (y + 0.5) * (n_fluid - 0.5 - y)
    return u, g * n_fluid**2 / (8.0 * nu)


def analytic_couette(n_fluid: int, u_wall: float):
    """Plane Couette between halfway walls (top wall moving at ``u_wall``)."""
    y = torch.arange(n_fluid, dtype=torch.float64)
    return u_wall * (y + 0.5) / n_fluid


def channel_masks(nx: int, ny: int, solid_rows: int = 1, device="cpu"):
    """Solid mask for a channel walled in y; periodic in x."""
    solid = torch.zeros(ny, nx, dtype=torch.bool, device=device)
    if solid_rows > 0:
        solid[:solid_rows, :] = True
        solid[-solid_rows:, :] = True
    return solid


def rest_equilibrium(ny: int, nx: int, perturb: bool = False, dtype=torch.float64):
    """Rest state (optional density perturbation to break symmetry)."""
    rho = torch.ones(ny, nx, dtype=dtype)
    if perturb:
        rho[ny // 2, nx // 3] += 0.02
    f = torch.zeros(9, ny, nx, dtype=dtype)
    for i in range(9):
        f[i] = W9[i] * rho
    return f


def field_bytes(f: torch.Tensor) -> int:
    """Bytes of one full ``Q x N`` field like ``f`` (the unit of AA storage)."""
    return f.numel() * f.element_size()