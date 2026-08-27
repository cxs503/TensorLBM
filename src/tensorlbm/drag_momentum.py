"""Momentum exchange drag method (Ladd 1994) for D3Q19 LBM.

Force on wall = Σ (f_i(x_f) + f_opp_i(x_s)) * c_i
where x_s = x_f + c_i is a solid neighbour (fluid→solid link).

Link-based: only counts directions where the neighbour is solid.
Does NOT need a surface normal — the lattice velocity c_i provides the
direction automatically.

Key physics:
  - For flat walls (Couette): equilibrium contributions cancel across
    opposite-direction pairs → exact result (no spurious pressure drag).
  - For curved surfaces (cylinder): equilibrium does NOT cancel → needs
    Bouzidi–Firdaouss–Lallemand (BFL) interpolated bounce-back for accuracy.

Timing:
  Compute on the **post-BB, pre-stream** distribution f
  (after collide + NoDynamics + BB, before streaming).
  At this point:
    - f_i(x_f) at fluid cells is post-collision (will stream toward wall).
    - f_opp_i(x_s) at solid cells is post-BB (bounced-back population).

Critical:
  - Count ALL 18 directions (NOT just opp_i > i) — the equilibrium
    cancellation requires both directions to be counted.
  - Use df += (NOT df -=) — the formula gives force ON the wall.
  - Save/restore solid cells after streaming to prevent wrap-around corruption.

Verified on Couette flow: Cd=0.606 vs Cf_exact=0.6349, err=4.55%.
"""

from __future__ import annotations

import torch

from .d3q19 import OPPOSITE, C


def drag_momentum_exchange_vec(
    f: torch.Tensor,
    near: torch.Tensor,
    solid: torch.Tensor,
) -> tuple[float, float, float]:
    """Momentum exchange force vector (Ladd 1994) for D3Q19.

    F = Σ (f_i + f_opp_i) * c_i   over all fluid→solid links.

    Count ALL 18 directions — equilibrium cancellation requires both
    directions in each opposite pair to be counted.

    Args:
        f: Distribution tensor (19, nz, ny, nx).
           Must be post-BB (after collide + NoDynamics + BB, before stream).
        near: Near-wall fluid mask (nz, ny, nx).
        solid: Boolean solid mask (nz, ny, nx).

    Returns:
        (fx, fy, fz) as Python floats — force on the wall.
    """
    device = f.device
    c = C.to(device).float()
    opp = OPPOSITE.to(device)

    fx = torch.tensor(0.0, device=device, dtype=f.dtype)
    fy = torch.tensor(0.0, device=device, dtype=f.dtype)
    fz = torch.tensor(0.0, device=device, dtype=f.dtype)

    for i in range(1, 19):  # ALL 18 directions, no filter!
        opp_i = int(opp[i].item())
        ci = c[i]
        dk = int(ci[2].item())
        dj = int(ci[1].item())
        di = int(ci[0].item())

        # solid_shifted[x] = solid[x + c_i] — True if c_i-neighbour is solid
        solid_shifted = torch.roll(solid, (-dk, -dj, -di), dims=(0, 1, 2))
        crossing = near & solid_shifted

        if not crossing.any():
            continue

        # f_opp at the solid neighbour: f[opp_i][x + c_i]
        f_opp_solid = torch.roll(f[opp_i], (-dk, -dj, -di), dims=(0, 1, 2))

        # Force on wall = (f_i + f_opp) * c_i  (positive = drag)
        contrib = ((f[i] + f_opp_solid) * crossing.float()).sum()
        fx = fx + float(ci[0].item()) * contrib
        fy = fy + float(ci[1].item()) * contrib
        fz = fz + float(ci[2].item()) * contrib

    return float(fx.item()), float(fy.item()), float(fz.item())


def drag_momentum_exchange(
    f: torch.Tensor,
    near: torch.Tensor,
    solid: torch.Tensor,
    dpS: float,
) -> float:
    """Momentum exchange drag coefficient (Ladd 1994).

    Cd_x = F_x / dpS, where F_x is the x-component of the wall force.

    Args:
        f: Distribution tensor (19, nz, ny, nx). Post-BB, pre-stream.
        near: Near-wall fluid mask (nz, ny, nx).
        solid: Boolean solid mask (nz, ny, nx).
        dpS: Normalisation factor (dynamic pressure × reference area).

    Returns:
        Drag coefficient Cd_x = F_x / dpS.
    """
    fx, _, _ = drag_momentum_exchange_vec(f, near, solid)
    return fx / dpS
