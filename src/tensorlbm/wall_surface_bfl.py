"""Wall-surface BFL (Bouzidi-Firdaouss-Lallemand) + momentum exchange for D3Q19.

Key difference from bfl_d3q19.py:
  - BFL interpolation uses the CORRECT standard formula (not swapped terms)
  - Momentum exchange uses wall-surface f values: F = (f_i[wall] + f_opp[wall]) * c_i
  - Applied BEFORE streaming (per verified-correct main loop)

Standard BFL formula (Bouzidi et al. 2002):
  For boundary link (fluid x_f -> solid x_s = x_f + c_i), wall at distance q:

  q < 0.5 (linear):
    f_bc = 2q * f_i(x_s) + (1-2q) * f_i(x_b)
    where f_i(x_s) = f[d](x_f) post-collision (streams to x_s)
          f_i(x_b) = f[d](x_f - c_d) post-collision (cell behind fluid)

  q >= 0.5 (quadratic):
    f_bc = f_i(x_s)/(2q) + (2q-1)/(2q) * f_opp(x_s)
    where f_opp(x_s) = f[opp_d](x_s) post-collision = feq[opp_d](x_s) (NoDynamics)

The BFL sets f[opp_d](x_f) = f_bc (the bounced-back population at the fluid cell).

Wall-surface momentum exchange:
  F = (f_i[wall] + f_opp[wall]) * c_i
  where f_i[wall] = fp_d = f[d](x_f) post-collision (streaming toward wall)
        f_opp[wall] = f_bc = BFL-reconstructed value
  No /q factor — the BFL already accounts for the wall position.
"""

from __future__ import annotations

import torch

from .d3q19 import C, OPPOSITE


def bouzidi_bounce_back_wallsurface(
    f: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
) -> torch.Tensor:
    """Apply wall-surface BFL interpolated bounce-back AFTER streaming.

    Uses the task's BFL formula with f_opp[fluid] = post-stream value from
    solid side (= feq[opp_d](x_s) with NoDynamics).

    For each boundary link (fluid x_f -> solid x_s = x_f + c_d), wall at q:

    q < 0.5 (linear):
      f_wall = 2q * f_opp(x_s) + (1-2q) * f_i(x_b)
      where f_opp(x_s) = feq[opp_d](x_s) (NoDynamics, post-stream from solid)
            f_i(x_b) = f_prev[d](x_f - c_d) (post-collision at cell behind)

    q >= 0.5 (quadratic):
      f_wall = f_opp(x_s)/(2q) + (2q-1)/(2q) * f_prev[opp_d](x_f)
      where f_prev[opp_d](x_f) = post-collision at fluid cell

    At q=0.5: f_wall = feq[opp_d](x_s) = feq[d](x_s) (symmetric for u=0),
    giving F = (fp_d + feq[d](x_s)) * c_i = standard ME. ✓

    Args:
        f: Post-stream distribution (19, nz, ny, nx). Will be modified.
        f_prev: Pre-stream (post-collision) distribution (19, nz, ny, nx).
        fluid_boundary_mask: (19, nz, ny, nx) bool, True for boundary links.
        q_field: (19, nz, ny, nx) float, fractional distance to wall.

    Returns:
        Updated distribution tensor.
    """
    opp = OPPOSITE.to(f.device)
    c = C.to(f.device).float()
    f_out = f.clone()

    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fluid_boundary_mask[d]
        if not mask.any():
            continue

        cx = int(c[d, 0].item())
        cy = int(c[d, 1].item())
        cz = int(c[d, 2].item())

        q_cell = q_field[d]  # (nz, ny, nx)

        # f_opp(x_s) = f_prev[opp_d] at x_s = x_f + c_d (solid cell)
        # = feq[opp_d](x_s) with NoDynamics
        # roll by -c_d: rolled[x_f] = f_prev[opp_d][x_f + c_d] = f_prev[opp_d][x_s]
        f_opp_xs = torch.roll(f_prev[opp_d], shifts=(-cz, -cy, -cx), dims=(0, 1, 2))

        # f_i(x_b) = f_prev[d] at x_b = x_f - c_d (cell behind fluid)
        # roll by +c_d: rolled[x_f] = f_prev[d][x_f - c_d] = f_prev[d][x_b]
        f_i_xb = torch.roll(f_prev[d], shifts=(cz, cy, cx), dims=(0, 1, 2))

        # f_prev[opp_d](x_f) = post-collision at fluid cell (direct access)
        fp_opp = f_prev[opp_d]  # (nz, ny, nx)

        # Linear (q < 0.5): f_wall = 2q * f_opp(x_s) + (1-2q) * f_i(x_b)
        mask_lin = q_cell < 0.5
        f_bc_lin = 2.0 * q_cell * f_opp_xs + (1.0 - 2.0 * q_cell) * f_i_xb

        # Quadratic (q >= 0.5): f_wall = f_opp(x_s)/(2q) + (2q-1)/(2q) * fp_opp
        safe_q = torch.where(mask_lin, torch.ones_like(q_cell), q_cell)
        f_bc_quad = f_opp_xs / (2.0 * safe_q) + (2.0 * safe_q - 1.0) / (2.0 * safe_q) * fp_opp

        f_bc = torch.where(mask_lin, f_bc_lin, f_bc_quad)

        # Set f[opp_d](x_f) = f_bc at boundary cells
        target = f_out[opp_d].clone()
        target = torch.where(mask, f_bc, target)
        f_out[opp_d] = target

    return f_out


def drag_momentum_exchange_bfl(
    f_post_bfl: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    dpS: float,
    *,
    use_q_scaling: bool = False,
) -> float:
    """Wall-surface momentum exchange drag for BFL.

    F = sum over boundary links of (f_i[wall] + f_opp[wall]) * c_i_x

    where:
      f_i[wall] = fp_d = f_prev[d](x_f) post-collision (streaming toward wall)
      f_opp[wall] = f_bc = f_post_bfl[opp_d](x_f) (BFL-reconstructed value)

    Args:
        f_post_bfl: Distribution after BFL (19, nz, ny, nx).
        f_prev: Pre-stream (post-collision) distribution (19, nz, ny, nx).
        fluid_boundary_mask: (19, nz, ny, nx) bool.
        q_field: (19, nz, ny, nx) float.
        dpS: Normalisation factor.
        use_q_scaling: If True, divide by q (standard ME approach).
                       If False, use wall-surface values directly (no /q).

    Returns:
        Drag coefficient Cd_x = F_x / dpS.
    """
    device = f_post_bfl.device
    c = C.to(device).float()
    opp = OPPOSITE.to(device)

    fx = torch.tensor(0.0, device=device, dtype=f_post_bfl.dtype)

    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fluid_boundary_mask[d]
        if not mask.any():
            continue

        # f_i[wall] = fp_d at boundary cells (post-collision, toward wall)
        fp_d = f_prev[d][mask]

        # f_opp[wall] = f_bc = BFL-reconstructed value at boundary cells
        f_bc = f_post_bfl[opp_d][mask]

        c_d_x = float(c[d, 0].item())

        if use_q_scaling:
            q_cell = q_field[d][mask].clamp(min=0.01)
            contrib = (fp_d + f_bc) * c_d_x / q_cell
        else:
            contrib = (fp_d + f_bc) * c_d_x

        fx = fx + contrib.sum()

    return float(fx.item() / dpS)


def drag_momentum_exchange_bfl_vec(
    f_post_bfl: torch.Tensor,
    f_prev: torch.Tensor,
    fluid_boundary_mask: torch.Tensor,
    q_field: torch.Tensor,
    dpS: float,
    *,
    use_q_scaling: bool = False,
) -> tuple[float, float, float]:
    """Wall-surface momentum exchange force vector for BFL.

    Returns (fx, fy, fz) as Python floats.
    """
    device = f_post_bfl.device
    c = C.to(device).float()
    opp = OPPOSITE.to(device)

    fx = torch.tensor(0.0, device=device, dtype=f_post_bfl.dtype)
    fy = torch.tensor(0.0, device=device, dtype=f_post_bfl.dtype)
    fz = torch.tensor(0.0, device=device, dtype=f_post_bfl.dtype)

    for d in range(1, 19):
        opp_d = int(opp[d].item())
        mask = fluid_boundary_mask[d]
        if not mask.any():
            continue

        fp_d = f_prev[d][mask]
        f_bc = f_post_bfl[opp_d][mask]

        c_d_x = float(c[d, 0].item())
        c_d_y = float(c[d, 1].item())
        c_d_z = float(c[d, 2].item())

        if use_q_scaling:
            q_cell = q_field[d][mask].clamp(min=0.01)
            factor = 1.0 / q_cell
        else:
            factor = 1.0

        contrib = (fp_d + f_bc) * factor
        fx = fx + c_d_x * contrib.sum()
        fy = fy + c_d_y * contrib.sum()
        fz = fz + c_d_z * contrib.sum()

    return float(fx.item() / dpS), float(fy.item() / dpS), float(fz.item() / dpS)
