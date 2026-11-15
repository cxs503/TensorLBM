"""AA-pattern fused stream-collide for D2Q9 LBM.

Anchor: ``XVAL-LBM-AA-PATTERN-FUSED-STREAM-COLLIDE`` (lanl/MF-LBM
``aa-pattern-fused-stream-collide``).

What the AA pattern actually is
-------------------------------
In the AA ("array of arrays", one contiguous plane per lattice direction) pull
form, a single pass over the grid advances

    f_i(x, t+1) = f_i(x - c_i, t) + omega * [ f_i^eq(x - c_i, t) - f_i(x - c_i, t) ]
                  \_____ pulled population ____/   \____ collision increment AT THE SOURCE node ____/

i.e. the *collision increment is evaluated at the upstream node* ``x - c_i`` and
carried by the same pull.  Because the increment is a local function of the
pre-collision state, it can be reconstructed on the fly, so no post-collision
array is ever materialised and the scheme is *bit-identical* to the canonical
"collide everywhere, then stream" operator (Kruger et al., *The Lattice
Boltzmann Method: Principles and Practice*, Springer 2017, ch. 5).

Ordering is not a detail
------------------------
Streaming ``S`` and BGK collision ``C`` do **not** commute: the equilibrium is a
nonlinear functional of the *local* moments.  Pulling pre-collision populations
and then colliding at the destination computes ``C[S[f]]`` (stream-then-collide),
a different, inconsistent discretisation -- it relaxes towards the equilibrium
of the *upwind-averaged* moments.  The historical implementation in this module
did exactly that (max relative deviation from the canonical operator: 2.3e-3 per
step on a non-uniform field; see
``benchmarks/xval/aa_fused_stream_collide_ordering_audit.py``) and its
"equivalence" test was degenerate: both schemes were compared on a *spatially
uniform* state, where every stencil is the identity, so the test asserted
``0.0 == 0.0`` for any pair of operators.

Current contents
----------------
``aa_fused_stream_collide``
    Correct AA-pattern fused pull (``S[C[f]]`` in one pass), no post-collision
    field.  Bit-identical to :func:`separate_stream_collide`.
``separate_stream_collide``
    Canonical reference: collide, materialise, stream (bit-identical to the
    fused form; kept as the oracle).
``aa_fused_step`` / ``separate_step``
    Full step with halfway bounce-back (stationary or moving wall), fused and
    materialised respectively.  Bit-identical to each other.
``stream_then_collide_legacy``
    The former (inconsistent) ``C[S[f]]`` operator, retained *only* as a
    negative control so the published numbers of the 2026-11-13 report stay
    reproducible.  Do not use it to model anything.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorlbm.aa_fused_push_pull import fused_push_pull_step

# D2Q9 lattice velocities (cx, cy) — same ordering as tensorlbm.d2q9
C = torch.tensor(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=torch.int64,
)
W = torch.tensor(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    dtype=torch.float64,
)
CS2 = 1.0 / 3.0
OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64)


def _equilibrium(rho, ux, uy, device=None):
    """Compute D2Q9 equilibrium distribution."""
    if device is None:
        device = rho.device
    c = C.to(device=device, dtype=rho.dtype)
    w = W.to(device=device, dtype=rho.dtype)

    uc = c[:, 0].view(-1, 1, 1) * ux.unsqueeze(0) + c[:, 1].view(-1, 1, 1) * uy.unsqueeze(0)
    u2 = ux**2 + uy**2

    feq = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )
    return feq


def _macroscopic(f):
    """Compute macroscopic variables from distributions."""
    c = C.to(device=f.device, dtype=f.dtype)
    rho = f.sum(dim=0)
    ux = (f * c[:, 0].view(-1, 1, 1)).sum(dim=0) / rho
    uy = (f * c[:, 1].view(-1, 1, 1)).sum(dim=0) / rho
    return rho, ux, uy


def _collision_increment(f, tau):
    """``omega * (f^eq - f)`` evaluated on the pre-collision state ``f``."""
    rho, ux, uy = _macroscopic(f)
    return (1.0 / tau) * (_equilibrium(rho, ux, uy) - f)


def _pull(a: torch.Tensor, i: int) -> torch.Tensor:
    """Value at node ``x - c_i`` (periodic pull) of a field ``(9, ny, nx)``/``(ny, nx)``."""
    return torch.roll(a, shifts=(int(C[i, 1]), int(C[i, 0])), dims=(-2, -1))


def aa_fused_stream_collide(f: torch.Tensor, tau: float) -> torch.Tensor:
    """AA-pattern fused stream-collide for D2Q9 (periodic domain).

    Single pull pass, no post-collision array:

        fnew_i(x) = f_i(x - c_i) + inc_i(x - c_i),  inc = omega (f^eq - f)

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
        Pre-collision distribution functions.
    tau : float
        BGK relaxation time, ``nu = (tau - 0.5) / 3``.

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
        Post-collision, post-streamed distributions (== ``separate_stream_collide``).
    """
    if f.dim() != 3 or f.shape[0] != 9:
        raise ValueError(f"f must have shape (9, ny, nx), got {tuple(f.shape)}")

    inc = _collision_increment(f, tau)

    f_out = torch.empty_like(f)
    f_out[0] = f[0] + inc[0]
    for i in range(1, 9):
        f_out[i] = _pull(f[i], i) + _pull(inc[i], i)
    return f_out


def stream_then_collide_legacy(f: torch.Tensor, tau: float) -> torch.Tensor:
    """**Inconsistent** legacy operator ``C[S[f]]`` (negative control only).

    Streams the *pre-collision* populations and then relaxes towards the
    equilibrium of the streamed moments.  This is not the AA-pattern scheme and
    not equivalent to collide-then-stream: it deviates from the canonical
    operator by ~2.3e-3 (relative, per step) on a non-uniform field.  Retained
    solely so that the numbers of the 2026-11-13 XVAL report are reproducible.
    """
    nf, ny, nx = f.shape
    device = f.device
    omega = 1.0 / tau

    c = C.to(device=device)
    f_out = torch.empty_like(f)

    for i in range(nf):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))

    rho, ux, uy = _macroscopic(f_out)
    feq = _equilibrium(rho, ux, uy, device=device)
    return f_out + omega * (feq - f_out)


def separate_stream_collide(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Canonical collide-then-stream (materialised) — the oracle for the fused form."""
    if f.dim() != 3 or f.shape[0] != 9:
        raise ValueError(f"f must have shape (9, ny, nx), got {tuple(f.shape)}")

    post = f + _collision_increment(f, tau)
    f_out = torch.empty_like(f)
    f_out[0] = post[0]
    for i in range(1, 9):
        f_out[i] = _pull(post[i], i)
    return f_out


def _wall_velocity(solid_mask: torch.Tensor, u_lid: float, lid_row: int) -> torch.Tensor:
    """Wall-velocity field ``(2, ny, nx)``: ``u_lid`` on row ``lid_row`` of the solid."""
    ny, nx = solid_mask.shape
    wall_u = torch.zeros((2, ny, nx), dtype=torch.float64, device=solid_mask.device)
    if u_lid != 0.0:
        row = lid_row % ny
        wall_u[0, row, :] = u_lid
        # only meaningful where the row is actually solid
        wall_u[0, row, :] = wall_u[0, row, :] * solid_mask[row, :].to(wall_u.dtype)
    return wall_u


def aa_fused_step(f: torch.Tensor, tau: float, solid_mask: torch.Tensor,
                  u_lid: float = 0.0, lid_row: int = -1) -> torch.Tensor:
    """One full LBM step: AA-pattern fused stream-collide + halfway bounce-back.

    Parameters
    ----------
    f : (9, ny, nx) tensor
        Pre-collision populations at time ``t``.
    tau : float
        BGK relaxation time.
    solid_mask : (ny, nx) bool tensor
        ``True`` on solid nodes.  The wall is located halfway between a solid
        node and its fluid neighbour (standard halfway bounce-back).
    u_lid : float
        Tangential wall velocity applied on row ``lid_row`` of the solid mask
        (moving lid; 0 for no-slip walls).
    lid_row : int
        Row index carrying ``u_lid`` (negative indices allowed); ignored when
        ``u_lid == 0``.

    Returns
    -------
    (9, ny, nx) tensor
        Post-collision, post-streamed, post-boundary populations at ``t+1``.
    """
    wall_u: Optional[torch.Tensor] = None
    if u_lid != 0.0:
        dtype = f.dtype
        wall_u = _wall_velocity(solid_mask, u_lid, lid_row).to(dtype=dtype, device=f.device)
    out = fused_push_pull_step(f, tau, solid=solid_mask, wall_u=wall_u)
    return out


def separate_step(f: torch.Tensor, tau: float, solid_mask: torch.Tensor,
                  u_lid: float = 0.0, lid_row: int = -1) -> torch.Tensor:
    """Full step with the canonical materialised operator + identical bounce-back.

    Same boundary algebra as :func:`aa_fused_step`; only the storage discipline
    differs (a post-collision array is materialised here).  The two are
    bit-identical, which is the non-vacuous equivalence property.
    """
    post = f + _collision_increment(f, tau)
    f_out = torch.empty_like(f)
    f_out[0] = post[0]
    rho, _, _ = _macroscopic(f)

    wall_u = None
    if u_lid != 0.0:
        wall_u = _wall_velocity(solid_mask, u_lid, lid_row).to(dtype=f.dtype, device=f.device)

    for i in range(1, 9):
        cx, cy = int(C[i, 0]), int(C[i, 1])
        f_out[i] = torch.roll(post[i], shifts=(cy, cx), dims=(0, 1))

    for i in range(1, 9):
        upstream_solid = _pull(solid_mask, i)
        if not bool(upstream_solid.any()):
            continue
        j = int(OPPOSITE[i])
        val = post[j].clone()
        if wall_u is not None:
            uwx = _pull(wall_u[0], i)
            uwy = _pull(wall_u[1], i)
            val = val + 6.0 * W[i].to(f.dtype) * rho * (int(C[i, 0]) * uwx + int(C[i, 1]) * uwy)
        f_out[i] = torch.where(upstream_solid, val, f_out[i])
    return f_out