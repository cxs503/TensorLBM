"""AA-pattern fused stream-collide for D2Q9 LBM.

Implements the Abstract Array (AA) pattern where streaming and collision
are fused into a single operation, following the approach used in
lanl/MF-LBM. This avoids storing intermediate post-collision distributions
and can improve cache locality.

Reference: lanl/MF-LBM aa-pattern-fused-stream-collide

Key insight: In the AA "pull" scheme, for each cell we pull distributions
from neighbors (pre-collision values), compute macroscopic, equilibrium,
and collide - all in one fused operation. This is mathematically equivalent
to the separate "collide then stream" approach for periodic BCs, but the
ordering of operations differs when boundaries are present.
"""

from __future__ import annotations

import torch


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


def aa_fused_stream_collide(f: torch.Tensor, tau: float) -> torch.Tensor:
    """AA-pattern fused stream-collide for D2Q9.

    Pull scheme: for each cell, pull distributions from neighbors,
    then compute macroscopic, equilibrium, and collide - all fused.

    Streaming convention: f[i] at cell (y,x) after streaming comes from
    cell (y-cy, x-cx) before streaming, where (cx,cy) is the velocity
    of direction i. This is implemented as torch.roll(f[i], shifts=(cy, cx)).

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
        Pre-collision distribution functions.
    tau : float
        BGK relaxation time.

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
        Post-collision, post-streamed distribution functions.
    """
    nf, ny, nx = f.shape
    device = f.device
    dtype = f.dtype
    omega = 1.0 / tau

    c = C.to(device=device)
    f_out = torch.empty_like(f)

    # Pull from neighbors: f[i] at (y,x) comes from (y-cy, x-cx)
    for i in range(nf):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            f_out[i] = f[i]
        else:
            f_out[i] = torch.roll(f[i], shifts=(cy, cx), dims=(0, 1))

    # Compute macroscopic from pulled distributions
    rho, ux, uy = _macroscopic(f_out)

    # Compute equilibrium
    feq = _equilibrium(rho, ux, uy, device=device)

    # BGK collision
    f_out = f_out + omega * (feq - f_out)

    return f_out


def separate_stream_collide(f: torch.Tensor, tau: float) -> torch.Tensor:
    """Separate collide then stream (baseline for comparison).

    Traditional approach:
    1. Compute macroscopic, equilibrium, collide
    2. Stream post-collision distributions

    Streaming convention: f[i] at cell (y,x) after streaming comes from
    cell (y-cy, x-cx) before streaming. Implemented as torch.roll(f[i], shifts=(cy, cx)).

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
        Pre-collision distribution functions.
    tau : float
        BGK relaxation time.

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
        Post-collision, post-streamed distribution functions.
    """
    nf, ny, nx = f.shape
    device = f.device
    dtype = f.dtype
    omega = 1.0 / tau

    c = C.to(device=device)

    # Step 1: Compute macroscopic
    rho = f.sum(dim=0)
    ux = (f * c[:, 0].view(-1, 1, 1).to(dtype)).sum(dim=0) / rho
    uy = (f * c[:, 1].view(-1, 1, 1).to(dtype)).sum(dim=0) / rho

    # Step 2: Equilibrium
    uc = c[:, 0].view(-1, 1, 1).to(dtype) * ux.unsqueeze(0) + c[:, 1].view(-1, 1, 1).to(dtype) * uy.unsqueeze(0)
    u2 = ux**2 + uy**2
    w = W.to(device=device, dtype=dtype)
    feq = w.view(-1, 1, 1) * rho.unsqueeze(0) * (
        1.0 + 3.0 * uc + 4.5 * uc**2 - 1.5 * u2.unsqueeze(0)
    )

    # Step 3: Collision
    f_collided = f + omega * (feq - f)

    # Step 4: Stream
    f_out = torch.empty_like(f_collided)
    for i in range(nf):
        cx, cy = int(c[i, 0]), int(c[i, 1])
        if cx == 0 and cy == 0:
            f_out[i] = f_collided[i]
        else:
            f_out[i] = torch.roll(f_collided[i], shifts=(cy, cx), dims=(0, 1))

    return f_out


def aa_fused_step(f: torch.Tensor, tau: float, solid_mask: torch.Tensor,
                  u_lid: float = 0.0, lid_row: int = -1) -> torch.Tensor:
    """One complete LBM step using AA-pattern fused stream-collide with BCs.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
        Pre-collision distribution functions.
    tau : float
        BGK relaxation time.
    solid_mask : torch.Tensor, shape (ny, nx), bool
        True for solid (bounce-back) nodes.
    u_lid : float
        Lid velocity (for moving lid BC).
    lid_row : int
        Row index of the lid (typically -1 for top wall).

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
        Post-collision, post-streamed, post-BC distribution functions.
    """
    # Fused stream-collide
    f_out = aa_fused_stream_collide(f, tau)

    # Bounce-back on solid nodes
    OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64, device=f.device)
    for i in range(1, 9):
        opp = OPPOSITE[i]
        f_out[i, solid_mask] = f_out[opp, solid_mask]

    # Moving lid BC (Zou-He style: equilibrium at lid velocity)
    if u_lid != 0.0 and lid_row is not None:
        rho_lid, _, _ = _macroscopic(f_out)
        feq_lid = _equilibrium(rho_lid,
                               torch.full_like(rho_lid, u_lid),
                               torch.zeros_like(rho_lid))
        for i in [3, 7, 8]:  # populations pointing into fluid from lid
            f_out[i, lid_row] = feq_lid[i, lid_row]

    return f_out


def separate_step(f: torch.Tensor, tau: float, solid_mask: torch.Tensor,
                  u_lid: float = 0.0, lid_row: int = -1) -> torch.Tensor:
    """One complete LBM step using separate collide+stream with BCs.

    Parameters
    ----------
    f : torch.Tensor, shape (9, ny, nx)
        Pre-collision distribution functions.
    tau : float
        BGK relaxation time.
    solid_mask : torch.Tensor, shape (ny, nx), bool
        True for solid (bounce-back) nodes.
    u_lid : float
        Lid velocity (for moving lid BC).
    lid_row : int
        Row index of the lid (typically -1 for top wall).

    Returns
    -------
    f_out : torch.Tensor, shape (9, ny, nx)
        Post-collision, post-streamed, post-BC distribution functions.
    """
    # Separate collide then stream
    f_out = separate_stream_collide(f, tau)

    # Bounce-back on solid nodes
    OPPOSITE = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.int64, device=f.device)
    for i in range(1, 9):
        opp = OPPOSITE[i]
        f_out[i, solid_mask] = f_out[opp, solid_mask]

    # Moving lid BC
    if u_lid != 0.0 and lid_row is not None:
        rho_lid, _, _ = _macroscopic(f_out)
        feq_lid = _equilibrium(rho_lid,
                               torch.full_like(rho_lid, u_lid),
                               torch.zeros_like(rho_lid))
        for i in [3, 7, 8]:
            f_out[i, lid_row] = feq_lid[i, lid_row]

    return f_out
