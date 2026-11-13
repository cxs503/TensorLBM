"""Regularised D2Q9 boundary conditions with non-equilibrium stress reconstruction.

This is the TensorLBM port of the regularised boundary condition used by
joconnor22/LIFE (``src/Grid.cpp::regularisedBC``, Latt et al. 2008).  The
cross-validation protocol, the upstream snapshot and the reference
transcription live in ``xval/XVAL-LBM-REG-BC-NEQ-STRESS/``.

Algorithm (one pass per boundary node, exactly the upstream data flow):

1. set the prescribed macroscopic state at the node (wall -> u = 0,
   velocity -> u = u_bc, pressure -> rho = rho_bc + zero-gradient tangential
   velocity, free slip -> normal velocity 0 + zero-gradient tangential);
2. close the unknown in-flowing populations with non-equilibrium bounce-back
   ``f_q = f_q^eq + (f_qbar - f_qbar^eq)`` (buried links of a corner get
   ``f_q^eq``); for a flat (non-corner) velocity-type node recover the density
   from ``rho = (2 f^+ + f^-) / (1 - n.u)`` and for a pressure node recover the
   normal velocity from ``n.u = n.(1 - (2 f^+ + f^-)/rho)``;
3. accumulate the non-equilibrium stress over the whole *reconstructed* set,
   ``Pi_neq = sum_q c_q c_q (f_q - f_q^eq)``;
4. replace *all* populations by the second-order Hermite projection
   ``f_q = f_q^eq + (w_q / 2 c_s^4) Q_qab Pi_neq^{ab}``.

Because the projection removes only moments above second order,
``sum_q f_q`` and ``sum_q c_q f_q`` are preserved exactly, so the node leaves
step 4 with exactly the density and velocity set in steps 1-2.

Deviations from the upstream implementation (deliberate, documented):

* vectorised over nodes (torch), and ``rho`` used by the corner extrapolation
  is gathered from the pre-BC macroscopic field instead of being read from the
  in-place array in BC-vector order.  Only corner nodes can see this
  difference, and only when a corner's extrapolation stencil touches another
  boundary node (see the cross-validation K1 split corner vs non-corner).
* neighbour lookups are bounds-clamped instead of relying on flat-buffer
  wrap-around, so a corner at the domain edge cannot silently read a different
  row of the state.
* no corner->edge demotion by neighbour-type inspection (``getNormalVector``
  L520-540): a node marked with two non-zero normal components is treated as a
  corner.  Callers that want LIFE's demotion must pass a single-axis normal.
* convective outlet (``convectiveBC``, L472-486) is not ported: it is a
  different closure, not a regularised reconstruction.
"""

from __future__ import annotations

import torch

from .boundary.d2q9 import C2Q9, OPPOSITE_2Q9, W2Q9
from .boundary.solver2d import macroscopic2d

FLUID = 0
WALL = 1
VELOCITY = 2
PRESSURE = 3
FREE_SLIP = 4

CS2 = 1.0 / 3.0


def _feq_nodes(rho: torch.Tensor, ux: torch.Tensor, uy: torch.Tensor, cx, cy, w) -> torch.Tensor:
    """Second-order Hermite equilibrium for a batch of nodes, shape (9, N)."""
    uc = cx * ux.unsqueeze(0) + cy * uy.unsqueeze(0)
    u2 = (ux * ux + uy * uy).unsqueeze(0)
    return w.unsqueeze(1) * rho.unsqueeze(0) * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * u2)


def apply_regularised_bc_2d(
    f: torch.Tensor,
    kind: torch.Tensor,
    normal: torch.Tensor,
    *,
    u_bc: torch.Tensor | None = None,
    rho_bc: torch.Tensor | None = None,
    rho: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """Apply the regularised (non-equilibrium stress) boundary condition.

    Parameters
    ----------
    f : (9, ny, nx) populations after collide+stream (pre-BC state).
    kind : (ny, nx) int64, values ``FLUID/WALL/VELOCITY/PRESSURE/FREE_SLIP``.
    normal : (ny, nx, 2) int64 outward lattice normal; components in {-1, 0, 1}.
    u_bc : (ny, nx, 2) prescribed velocity (VELOCITY nodes).
    rho_bc : (ny, nx) prescribed density (PRESSURE nodes).
    rho, u : optional pre-BC macroscopic fields (recomputed from ``f`` if None).

    Returns
    -------
    (f_new, diagnostics) with ``diagnostics`` holding ``rho``/``ux``/``uy`` of
    the reconstructed nodes, the reconstructed ``Pi_neq`` components and the
    boundary node coordinates.
    """
    if f.ndim != 3 or f.shape[0] != 9:
        raise ValueError("f must have shape (9, ny, nx)")
    if not f.is_floating_point():
        raise TypeError("f must be floating point")
    ny, nx = f.shape[1], f.shape[2]
    if kind.shape != (ny, nx) or tuple(normal.shape) != (ny, nx, 2):
        raise ValueError("kind must be (ny,nx) and normal must be (ny,nx,2)")

    dtype, device = f.dtype, f.device
    c = C2Q9.to(device=device, dtype=dtype)
    w = W2Q9.to(device=device, dtype=dtype)
    opp = OPPOSITE_2Q9.to(device=device)

    if rho is None or u is None:
        rho_f, ux_f, uy_f = macroscopic2d(f)
        rho = rho_f if rho is None else rho
        u = torch.stack([ux_f, uy_f], dim=-1) if u is None else u

    sel = kind != FLUID
    iy, ix = sel.nonzero(as_tuple=True)
    n_nodes = iy.numel()
    if n_nodes == 0:
        return f.clone(), {"rho": rho[:, :0], "nodes": (iy, ix)}

    kv = kind[iy, ix]
    nvec = normal[iy, ix].to(dtype)
    ncx, ncy = nvec[:, 0], nvec[:, 1]
    is_corner = (ncx != 0) & (ncy != 0)
    axis = torch.where(ncx != 0, torch.zeros_like(ncx, dtype=torch.long), torch.ones_like(ncx, dtype=torch.long))
    nsign = torch.where(ncx != 0, ncx, ncy)

    cxq = c[:, 0].unsqueeze(1).expand(9, n_nodes)
    cyq = c[:, 1].unsqueeze(1).expand(9, n_nodes)
    cm = torch.where(axis.unsqueeze(0) == 0, cxq, cyq)
    m_fplus = cm == -nsign.unsqueeze(0)
    m_fzero = cm == 0

    phi = f[:, iy, ix]  # (9, N)
    fplus = (phi * m_fplus).sum(dim=0)
    fzero = (phi * m_fzero).sum(dim=0)

    rho_n = rho[iy, ix].clone()
    ux_n = u[iy, ix, 0].clone()
    uy_n = u[iy, ix, 1].clone()

    if u_bc is not None:
        ub = u_bc[iy, ix]
    else:
        ub = torch.zeros((n_nodes, 2), dtype=dtype, device=device)
    if rho_bc is not None:
        rb = rho_bc[iy, ix]
    else:
        rb = torch.ones(n_nodes, dtype=dtype, device=device)

    is_vel = (kv == WALL) | (kv == VELOCITY) | (kv == FREE_SLIP)
    is_pres = kv == PRESSURE
    is_slip_or_pres = (kv == FREE_SLIP) | is_pres
    tang = 1 - axis

    # --- step 1: prescribed macroscopic state -----------------------------
    ux_n = torch.where(is_vel, ub[:, 0], ux_n)
    uy_n = torch.where(is_vel, ub[:, 1], uy_n)

    def _zg(comp: int) -> torch.Tensor:
        field = u[..., comp]

        def at(step: int) -> torch.Tensor:
            j = (iy + step * ncy.long()).clamp(0, ny - 1)
            i = (ix + step * ncx.long()).clamp(0, nx - 1)
            return field[j, i]

        return (4.0 / 3.0) * at(1) - (1.0 / 3.0) * at(2)

    for comp in (0, 1):
        value = _zg(comp)
        mask = is_slip_or_pres & (tang == comp)
        if comp == 0:
            ux_n = torch.where(mask, value, ux_n)
        else:
            uy_n = torch.where(mask, value, uy_n)

    zero_normal = (kv == FREE_SLIP) & (axis == 0)
    ux_n = torch.where(zero_normal, torch.zeros_like(ux_n), ux_n)
    zero_normal = (kv == FREE_SLIP) & (axis == 1)
    uy_n = torch.where(zero_normal, torch.zeros_like(uy_n), uy_n)

    # density / normal velocity of the flat nodes (steps 2 of the algorithm)
    ua = torch.where(axis == 0, ux_n, uy_n)
    vel_rho = (2.0 * fplus + fzero) / (1.0 - nsign * ua)
    flat_vel = is_vel & ~is_corner
    rho_n = torch.where(flat_vel, vel_rho, rho_n)

    # corner velocity-type nodes: first-order density extrapolation
    def _rho_at(step: int) -> torch.Tensor:
        j = (iy + step * ncy.long()).clamp(0, ny - 1)
        i = (ix + step * ncx.long()).clamp(0, nx - 1)
        return rho[j, i]

    corner_vel = is_vel & is_corner
    rho_n = torch.where(corner_vel, 2.0 * _rho_at(1) - _rho_at(2), rho_n)

    rho_n = torch.where(is_pres, rb, rho_n)
    pres_normal = nsign * (1.0 - (2.0 * fplus + fzero) / rb)
    pres_flat = is_pres & ~is_corner
    ua_new = torch.where(pres_flat, pres_normal, ua)
    ux_n = torch.where(axis == 0, ua_new, ux_n)
    uy_n = torch.where(axis == 1, ua_new, uy_n)

    # --- step 2: close the unknown in-flowing populations ------------------
    feq = _feq_nodes(rho_n, ux_n, uy_n, cxq, cyq, w)
    opp_phi = phi[opp]
    opp_feq = feq[opp]

    f_rec = phi.clone()
    rec_single = (cm == nsign.unsqueeze(0)) & ~is_corner.unsqueeze(0)
    f_rec = torch.where(rec_single, feq + (opp_phi - opp_feq), f_rec)

    corner_sel = (cxq == ncx.unsqueeze(0)) | (cyq == ncy.unsqueeze(0))
    buried = (ncx.unsqueeze(0) * cxq + ncy.unsqueeze(0) * cyq) == 0
    corner_value = torch.where(buried, feq, feq + (opp_phi - opp_feq))
    f_rec = torch.where(corner_sel & is_corner.unsqueeze(0), corner_value, f_rec)

    # --- step 3: non-equilibrium stress of the reconstructed set -----------
    fneq = f_rec - feq
    pi_xx = (cxq * cxq * fneq).sum(dim=0)
    pi_yy = (cyq * cyq * fneq).sum(dim=0)
    pi_xy = (cxq * cyq * fneq).sum(dim=0)

    # --- step 4: second-order Hermite projection ---------------------------
    projected = feq + 4.5 * w.unsqueeze(1) * (
        (cxq * cxq - CS2) * pi_xx.unsqueeze(0)
        + (cyq * cyq - CS2) * pi_yy.unsqueeze(0)
        + 2.0 * cxq * cyq * pi_xy.unsqueeze(0)
    )

    out = f.clone()
    out[:, iy, ix] = projected

    rho_out = projected.sum(dim=0)
    ux_out = (cxq * projected).sum(dim=0) / rho_out
    uy_out = (cyq * projected).sum(dim=0) / rho_out
    diagnostics = {
        "nodes": (iy, ix),
        "kind": kv,
        "is_corner": is_corner,
        "rho": rho_out,
        "ux": ux_out,
        "uy": uy_out,
        "pi_neq_xx": (cxq * cxq * (projected - feq)).sum(dim=0),
        "pi_neq_yy": (cyq * cyq * (projected - feq)).sum(dim=0),
        "pi_neq_xy": (cxq * cyq * (projected - feq)).sum(dim=0),
        "pi_neq_xx_pre": pi_xx,
        "pi_neq_yy_pre": pi_yy,
        "pi_neq_xy_pre": pi_xy,
    }
    return out, diagnostics