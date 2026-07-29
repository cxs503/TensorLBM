"""Nodal Discontinuous-Galerkin advection for the Lattice Boltzmann Method.

This module implements the *real* DG-LBM building block: a genuine nodal-DG
discretisation of the discrete-velocity advection equation

    ∂_t f_i + c_i · ∇ f_i = 0

that replaces the standard exact-shift LBM streaming inside a DG zone.  The
collision operator (BGK/MRT, applied pointwise to every nodal degree of
freedom) lives elsewhere; here we only discretise the *advection*.

Design choices
--------------
* **Nodal collocation** on Gauss–Lobatto nodes (the endpoints are nodes, so a
  face value *is* a degree of freedom — no L² projection is needed for the
  surface flux, which makes the flux lift especially simple).
* **Dimension-by-dimension** DG: the 2D/3D element is a tensor product, but the
  advection operator is applied axis-by-axis.  Each axis only needs the 1D
  operator ``Ax = (2/Δx) M⁻¹ G`` (volume) and a face-lift matrix, applied via a
  single ``einsum`` plus neighbour ``roll`` for the surface term.  This keeps the
  whole kernel dense-tensor / ``torch.compile`` friendly.
* **Upwind numerical flux** (exact for constant-coefficient advection; identical
  to local Lax–Friedrichs / Roe here).
* **Time integration**: explicit forward-Euler or SSP-RK3 (TVD-RK3) with
  optional sub-stepping.  Because the LBM macro-step is locked at Δt = 1 lattice
  unit while the P1 DG stability bound is CFL ≈ 1/(2p+1) = 1/3, the DG zone must
  be sub-cycled (typically ``n_substeps = 3`` or ``4`` for P1) when coupled to
  LBM; the standalone operator here lets the caller choose ``dt`` and the number
  of stages freely so it can be validated on its own (MMS, conservation, p0
  equivalence).

This file deliberately contains *no* LBM collision and *no* obstacle/band
geometry — it is the pure, reusable DG advection kernel.  The hybrid
DG-band + LBM-exterior coupling (see ``dg_lbm.py``) is built on top of the
operators exported here.

Reference operators are constructed once (in float64, via numpy, from the node
definitions) and cached per ``(degree, Δx, dtype, device)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from numpy.polynomial import legendre as _leg

# ---------------------------------------------------------------------------
# Reference-element construction (numpy, float64)
# ---------------------------------------------------------------------------


def lobatto_nodes(degree: int) -> np.ndarray:
    """Gauss–Lobatto nodes on the reference interval ``[-1, 1]``.

    Returns ``degree + 1`` nodes: the endpoints ``±1`` plus the ``degree - 1``
    roots of the degree-``degree`` Legendre polynomial's derivative.  Robustly
    computed via the companion matrix of the derivative polynomial.
    """
    if degree < 0:
        raise ValueError(f"degree must be >= 0, got {degree}")
    if degree == 0:
        return np.array([0.0])
    if degree == 1:
        return np.array([-1.0, 1.0])
    # Interior Lobatto nodes = roots of P_degree'(x).
    leg = _leg.Legendre.basis(degree)
    roots = np.sort(leg.deriv().roots())
    return np.concatenate(([-1.0], roots, [1.0]))


def _lagrange_basis_matrices(
    nodes: np.ndarray,
    xq: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lagrange basis value and derivative matrices at quadrature points.

    Returns ``(V, Vr)`` with shapes ``(n_quad, n_nodes)`` where
    ``V[q, j] = l_j(xq[q])`` and ``Vr[q, j] = l_j'(xq[q])`` for the Lagrange
    basis ``{l_j}`` of *nodes* evaluated at the quadrature points *xq*.

    Each Lagrange polynomial ``l_j`` is the unique degree ``n-1`` polynomial
    with ``l_j(nodes[i]) = δ_ij``.  We recover its monomial coefficients by
    inverting the (small) monomial Vandermonde, then differentiate.  This is
    exact and robust for the low degrees (≤ 4) used here.
    """
    n = nodes.shape[0]
    powers = np.arange(n, dtype=np.float64)                 # 0, 1, ..., n-1
    # Monomial Vandermonde at the nodes and its inverse.  Column j of the
    # inverse holds the monomial coefficients of l_j.
    vmono = nodes[:, None] ** powers[None, :]               # (n, n)
    vmono_inv = np.linalg.inv(vmono)

    vq = xq[:, None] ** powers[None, :]                     # (nq, n) monomials at quad
    V = vq @ vmono_inv                                      # (nq, n) l_j(xq)

    # Derivative monomials at quad points: d/dx x^k = k x^(k-1) (k=0 term → 0).
    dq = np.zeros_like(vq)
    dq[:, 1:] = powers[1:][None, :] * vq[:, :-1]            # shift: col k <- k * x^(k-1)
    Vr = dq @ vmono_inv                                     # (nq, n) l_j'(xq)
    return V, Vr


# ---------------------------------------------------------------------------
# Cached operator set per (degree, dx, dtype, device)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Ops:
    """Precomputed 1D DG operators for one reference element."""

    degree: int
    n_node: int                   # degree + 1
    Ax: torch.Tensor              # (n_node, n_node) volume operator (2/dx M⁻¹ G)
    face_lift: torch.Tensor       # (n_node, 2) surface-lift: RHS += -c * face_lift @ [uL, uR]


_ops_cache: dict[tuple, _Ops] = {}


def get_ops(
    degree: int,
    dx: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> _Ops:
    """Return cached DG operators for a given polynomial *degree* and cell *dx*.

    The operators encode the per-element semi-discrete RHS

        du/dt = c · Ax · u  −  c · face_lift · [û_left, û_right]

    where ``Ax`` already incorporates the mass-matrix inverse and the
    reference→physical scaling, so the caller only multiplies by the advective
    speed ``c`` and assembles the two upwind face values.
    """
    key = (degree, float(dx), dtype, torch.device(device))
    cached = _ops_cache.get(key)
    if cached is not None:
        return cached

    nodes = lobatto_nodes(degree)
    n = nodes.shape[0]
    # Over-integrate (degree+3 Gauss-Legendre points) so the construction is
    # exact for the mass and stiffness of a degree-`degree` polynomial basis.
    nq = degree + 3
    xq, wq = np.polynomial.legendre.leggauss(max(nq, 2))
    V, Vr = _lagrange_basis_matrices(nodes, xq)            # (nq, n)

    W = np.diag(wq)
    mass = V.T @ W @ V                                      # M_ij = ∫ l_i l_j
    # Volume stiffness with the derivative on the *test* function:
    #   G[k, j] = ∫ l_j · l_k' dξ
    G = Vr.T @ W @ V                                        # (n, n)
    mass_inv = np.linalg.inv(mass)

    scale = 2.0 / dx                                       # reference -> physical
    ax_np = scale * (mass_inv @ G)                         # (n, n)

    # Surface selection matrix S (n, 2): S[0, 0] = -1 (left face, n̂=-1),
    # S[-1, 1] = +1 (right face, n̂=+1); the surface vector is S @ [û_L, û_R].
    S = np.zeros((n, 2))
    S[0, 0] = -1.0
    S[-1, 1] = 1.0
    face_lift_np = scale * (mass_inv @ S)                  # (n, 2)

    Ax = torch.tensor(ax_np, dtype=dtype, device=device)
    face_lift = torch.tensor(face_lift_np, dtype=dtype, device=device)

    ops = _Ops(degree=degree, n_node=n, Ax=Ax, face_lift=face_lift)
    _ops_cache[key] = ops
    return ops


# ---------------------------------------------------------------------------
# Field utilities — cell-mean <-> nodal DOFs
# ---------------------------------------------------------------------------


def nodal_from_mean(
    cell_means: torch.Tensor,
    ops: _Ops,
    node_axes: tuple[int, ...],
) -> torch.Tensor:
    """Expand per-cell mean values into constant-in-element nodal DOFs.

    Seeds each node of every cell with the cell's mean value.  This is the
    natural initial state when injecting a P0 (LBM) value into a DG element.
    """
    means = cell_means.unsqueeze(-1)                        # add one node axis
    out = means
    for _ in range(len(node_axes) - 1):
        out = out.unsqueeze(-1)
    return out.expand(*cell_means.shape, *([ops.n_node] * len(node_axes))).contiguous()


def cell_means_from_nodal(
    f_dg: torch.Tensor,
    node_axes: tuple[int, ...],
) -> torch.Tensor:
    """Project nodal DOFs back to per-cell means (the P0 / cell-average value)."""
    for ax in node_axes:
        f_dg = f_dg.mean(dim=ax, keepdim=True)
    for _ in range(len(node_axes)):
        f_dg = f_dg.squeeze(-1)
    return f_dg


# ---------------------------------------------------------------------------
# Semi-discrete RHS (dimension-by-dimension, periodic)
# ---------------------------------------------------------------------------


def _axis_pairs(ndim_spatial: int, q_first: int) -> list[tuple[int, int]]:
    """Return ``(cell_axis, node_axis)`` pairs for each spatial dimension.

    The field layout is ``(...Q..., [nz,] ny, nx, [nz_node,] ny_node, nx_node)``
    with the Q axis at position ``q_first``; cell axes follow, then node axes.
    """
    cell_axes = list(range(q_first + 1, q_first + 1 + ndim_spatial))
    node_start = q_first + 1 + ndim_spatial
    node_axes = list(range(node_start, node_start + ndim_spatial))
    # cell_axis x is the last cell axis and pairs with the last node axis.
    return [
        (cell_axes[ndim_spatial - 1 - k], node_axes[ndim_spatial - 1 - k])
        for k in range(ndim_spatial)
    ]


def _rhs_along_axis(
    f: torch.Tensor,
    c_per_q: torch.Tensor,
    cell_axis: int,
    node_axis: int,
    ops: _Ops,
) -> torch.Tensor:
    """DG RHS contribution for advection along one axis.

    Args:
        f: nodal DOFs ``(Q, ..., n_cell_axis, ..., n_node_axis, ...)``.
        c_per_q: per-velocity advective speed ``(Q,)`` along this axis.
        cell_axis / node_axis: positions of the cell index and node index.
        ops: precomputed 1D operators.

    Returns the partial ``df/dt`` from this axis, same shape as *f*.
    """
    Ax = ops.Ax
    face_lift = ops.face_lift
    p_last = ops.n_node - 1

    # --- Volume term: c · Ax · u  (einsum over the node axis) ---
    # Ax is labelled "vu": output node v first, contracted input node u second,
    # so this computes (Ax @ u)[v] = Σ_u Ax[v,u] u[u] (NOT the transpose).
    letters = "abcdefghijklmnopqrst"
    n_dims = f.ndim
    in_subs = [letters[i] for i in range(n_dims)]
    out_subs = list(in_subs)
    in_subs[node_axis] = "u"            # the node axis being contracted
    out_subs[node_axis] = "v"           # output node axis
    ein = f"vu,{''.join(in_subs)}->{''.join(out_subs)}"
    vol = torch.einsum(ein, Ax, f)       # (..., n_node, ...)

    # --- Surface term: gather the four face traces, choose upwind ---
    ncell_dim = n_dims
    # Move node axis to the end for easy index_select, then restore.
    inner_left = f.select(node_axis, 0)        # u_e[0]  (node axis removed)
    inner_right = f.select(node_axis, p_last)  # u_e[p]
    left_ext = torch.roll(inner_right, shifts=1, dims=cell_axis)               # u_{e-1}[p]
    right_ext = torch.roll(inner_left, shifts=-1, dims=cell_axis)              # u_{e+1}[0]

    pos = c_per_q.view([c_per_q.shape[0]] + [1] * (inner_left.ndim - 1)) > 0.0
    uL = torch.where(pos, left_ext, inner_left)        # upwind left face
    uR = torch.where(pos, inner_right, right_ext)      # upwind right face

    # RHS += -c · face_lift @ [uL, uR], broadcast over the node axis.
    fl_l = face_lift[:, 0]                              # (n_node,)
    fl_r = face_lift[:, 1]
    # face_lift[node] * uL / uR: shape node axis broadcasts against cell fields.
    shape = [1] * n_dims
    shape[node_axis] = ops.n_node
    surf_l = (fl_l.view(shape) * uL.unsqueeze(node_axis))
    surf_r = (fl_r.view(shape) * uR.unsqueeze(node_axis))
    surf = surf_l + surf_r                              # face_lift·[uL,uR] per node

    c_view = c_per_q.view([c_per_q.shape[0]] + [1] * (n_dims - 1))
    return c_view * vol - c_view * surf


def dg_rhs(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    ndim_spatial: int,
    q_first: int = 0,
) -> torch.Tensor:
    """Semi-discrete RHS ``df/dt`` for DG advection of every velocity.

    Args:
        f_dg: nodal DOFs ``(Q, [nz,] ny, nx, [nz_node,] ny_node, nx_node)``.
        velocities: lattice velocities ``(Q, ndim_spatial)``.
        ops: 1D DG operators.
        ndim_spatial: 2 or 3.
        q_first: position of the Q axis (default 0).

    Returns ``df/dt`` with the same shape as *f_dg* (periodic domain).
    """
    pairs = _axis_pairs(ndim_spatial, q_first)
    rhs = torch.zeros_like(f_dg)
    for axis_k, (cell_axis, node_axis) in enumerate(pairs):
        c_axis = velocities[:, axis_k].to(f_dg.dtype)
        if c_axis.abs().max().item() == 0.0:
            continue
        nonzero = c_axis.abs() > 0.0
        # Process only velocities that move along this axis; others contribute 0.
        sub = f_dg[nonzero]
        rhs_sub = _rhs_along_axis(sub, c_axis[nonzero], cell_axis, node_axis, ops)
        rhs[nonzero] = rhs[nonzero] + rhs_sub
    return rhs


# ---------------------------------------------------------------------------
# Time integration
# ---------------------------------------------------------------------------


def _euler_step(
    f: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    ndim_spatial: int,
    dt: float,
    q_first: int,
) -> torch.Tensor:
    rhs = dg_rhs(f, velocities, ops, ndim_spatial, q_first)
    return f + dt * rhs


def _ssprk3_step(
    f: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    ndim_spatial: int,
    dt: float,
    q_first: int,
) -> torch.Tensor:
    """One SSP-RK3 (TVD-RK3) step of size *dt*."""
    k1 = _euler_step(f, velocities, ops, ndim_spatial, dt, q_first)
    k2 = 0.75 * f + 0.25 * _euler_step(k1, velocities, ops, ndim_spatial, dt, q_first)
    return (1.0 / 3.0) * f + (2.0 / 3.0) * _euler_step(
        k2, velocities, ops, ndim_spatial, dt, q_first
    )


def dg_advect(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    ops: _Ops,
    ndim_spatial: int,
    dt: float,
    n_substeps: int = 1,
    scheme: Literal["euler", "rk3"] = "rk3",
    q_first: int = 0,
) -> torch.Tensor:
    """Advance nodal DOFs by *dt* using sub-cycled explicit DG advection.

    Args:
        f_dg: nodal DOFs ``(Q, ..., cells..., ..., nodes...)``.
        velocities: ``(Q, ndim_spatial)`` lattice velocities.
        ops: 1D DG operators.
        ndim_spatial: 2 or 3.
        dt: total time step (lattice units: 1.0 for one LBM macro-step).
        n_substeps: number of equal sub-steps (≥1).  Needed because P1 DG is
            stable only for CFL ≲ 1/3 while the LBM macro-step uses Δt = 1.
        scheme: ``"euler"`` (single forward-Euler stage) or ``"rk3"`` (SSP-RK3).
        q_first: index of the Q axis.

    Returns the advected DOFs (same shape).
    """
    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    step_fn = _euler_step if scheme == "euler" else _ssprk3_step
    dt_sub = dt / n_substeps
    f = f_dg
    for _ in range(n_substeps):
        f = step_fn(f, velocities, ops, ndim_spatial, dt_sub, q_first)
    return f


# ---------------------------------------------------------------------------
# DG-LBM collision on nodal DOFs
# ---------------------------------------------------------------------------
#
# In the hybrid DG-LBM the BGK/MRT collision is applied pointwise to *every*
# nodal degree of freedom: each node holds a full set of Q populations, from
# which the local (ρ, u) are recovered and relaxed toward equilibrium.  Because
# the DG advection is sub-cycled (Δt_sub < 1), the relaxation must be Δt-aware:
#
#     f ← f_eq + (1 − Δt_sub / τ) · (f − f_eq)
#
# i.e. a relaxation fraction Δt_sub/τ per sub-step, NOT the LBM-default 1/τ
# (which hard-codes Δt = 1).  Using 1/τ under sub-cycling silently destroys
# stability/accuracy at low τ.


def macroscopic_dg(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    q_first: int = 0,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Recover ``(ρ, [u_α])`` at every DG node.

    Args:
        f_dg: ``(Q, ..., cells..., ..., nodes...)`` — Q axis at *q_first*.
        velocities: ``(Q, ndim)`` lattice velocities.

    Returns ``(rho, us)`` where ``rho`` and each ``us[a]`` share the field shape
    with the Q axis removed.
    """
    ndim = velocities.shape[1]
    c = velocities.to(f_dg.dtype).to(f_dg.device)
    rho = f_dg.sum(dim=q_first)
    rho_safe = rho.clamp(min=1e-12)
    cshape = [velocities.shape[0]] + [1] * (f_dg.ndim - 1)
    us = []
    for a in range(ndim):
        u_a = (f_dg * c[:, a].view(cshape)).sum(dim=q_first) / rho_safe
        us.append(u_a)
    return rho, us


def equilibrium_dg(
    rho: torch.Tensor,
    us: list[torch.Tensor],
    velocities: torch.Tensor,
    weights: torch.Tensor,
    q_first: int = 0,
    ndim_field: int | None = None,
) -> torch.Tensor:
    """Second-order (Hermite) equilibrium on DG nodes.

    Uses the D2Q9/D3Q19 form ``f_eq = w ρ (1 + 3 c·u + 9/2 (c·u)² − 3/2 |u|²)``
    (c_s² = 1/3), valid for both lattices.  Returns a field whose Q axis is at
    *q_first* (inferred from *rho* unless *ndim_field* is given).
    """
    c = velocities.to(rho.dtype).to(rho.device)         # (Q, ndim)
    w = weights.to(rho.dtype).to(rho.device)            # (Q,)
    # Build c·u over the Q axis, broadcast against the field.
    q = velocities.shape[0]
    if ndim_field is None:
        ndim_field = rho.ndim + 1                       # rho is f with Q removed
    cu = torch.zeros((q,) + (1,) * (ndim_field - 1), dtype=rho.dtype, device=rho.device)
    u_sq = torch.zeros_like(rho)
    for a, u_a in enumerate(us):
        ca = c[:, a].view((q,) + (1,) * (ndim_field - 1))
        cu = cu + ca * u_a.unsqueeze(q_first)
        u_sq = u_sq + u_a * u_a
    wv = w.view((q,) + (1,) * (ndim_field - 1))
    rho_q = rho.unsqueeze(q_first)
    return wv * rho_q * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(q_first))


def collide_bgk_dg(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    dt: float,
    q_first: int = 0,
) -> torch.Tensor:
    """BGK collision on every DG node, Δt-aware.

    ``f ← f_eq + (1 − dt/τ)(f − f_eq)``.  The relaxation fraction is ``dt/τ``
    so that sub-cycled advection (dt = Δt_sub < 1) keeps the correct viscosity.
    """
    rho, us = macroscopic_dg(f_dg, velocities, q_first)
    feq = equilibrium_dg(rho, us, velocities, weights, q_first, f_dg.ndim)
    return feq + (1.0 - dt / tau) * (f_dg - feq)


def dg_lbm_rhs(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    tau: float,
    ops: _Ops,
    ndim_spatial: int,
    q_first: int = 0,
) -> torch.Tensor:
    """Combined method-of-lines RHS of the discrete-velocity Boltzmann equation.

        df_i/dt = −c_i · ∇f_i  −  (f_i − f_i^eq)/τ

    The collision term is evaluated pointwise at every DG node (from the local
    ρ, u) and added to the DG advection RHS, so a single explicit integrator
    advances both.  This is stable (no collide/advect splitting instability)
    and recovers the *continuous* DVBE shear viscosity ``ν = τ/3`` (c_s² = 1/3).
    To match a standard discrete-LBM exterior whose viscosity is ``(τ − ½)/3``,
    use ``τ_dg = τ_lbm − ½`` inside the DG zone (handled by the hybrid coupler).
    """
    adv = dg_rhs(f_dg, velocities, ops, ndim_spatial, q_first)
    rho, us = macroscopic_dg(f_dg, velocities, q_first)
    feq = equilibrium_dg(rho, us, velocities, weights, q_first, f_dg.ndim)
    coll = -(f_dg - feq) / tau
    return adv + coll


def dg_lbm_step(
    f_dg: torch.Tensor,
    velocities: torch.Tensor,
    weights: torch.Tensor,
    ops: _Ops,
    tau: float,
    ndim_spatial: int,
    dt: float = 1.0,
    n_substeps: int = 6,
    scheme: Literal["euler", "rk3"] = "rk3",
    q_first: int = 0,
) -> torch.Tensor:
    """One DG-LBM macro-step via method-of-lines (advection + collision in RHS).

    Integrates ``df/dt = −c·∇f − (f−f_eq)/τ`` with sub-cycled SSP-RK3 (or
    forward-Euler).  This solves the continuous DVBE on the DG polynomial
    basis — stable and recovering ν = τ/3.

    Args:
        f_dg: nodal DOFs ``(Q, ..., cells..., ..., nodes...)``.
        velocities / weights: lattice ``(Q, ndim)`` and ``(Q,)``.
        ops: 1D DG operators.
        tau: relaxation time used in the DG zone.  For viscosity matching with a
            discrete-LBM exterior set this to ``τ_lbm − ½``.
        ndim_spatial: 2 or 3.
        dt: macro-step (1.0 for the LBM clock).
        n_substeps: RK sub-steps (P1 ⇒ ≥3 for the advection CFL; collision adds a
            dt/τ ≤ 2 stability bound that is comfortably met for τ ≳ 0.55).
        scheme: ``"euler"`` or ``"rk3"``.

    Returns the updated DOFs.
    """
    if n_substeps < 1:
        raise ValueError("n_substeps must be >= 1")
    dt_sub = dt / n_substeps

    def rhs(f: torch.Tensor) -> torch.Tensor:
        return dg_lbm_rhs(f, velocities, weights, tau, ops, ndim_spatial, q_first)

    def euler(f: torch.Tensor) -> torch.Tensor:
        return f + dt_sub * rhs(f)

    def rk3(f: torch.Tensor) -> torch.Tensor:
        k1 = f + dt_sub * rhs(f)
        k2 = 0.75 * f + 0.25 * (k1 + dt_sub * rhs(k1))
        return (1.0 / 3.0) * f + (2.0 / 3.0) * (k2 + dt_sub * rhs(k2))

    step_fn = euler if scheme == "euler" else rk3
    f = f_dg
    for _ in range(n_substeps):
        f = step_fn(f)
    return f


