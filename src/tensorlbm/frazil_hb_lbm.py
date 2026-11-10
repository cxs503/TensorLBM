"""Wall-consistent D2Q9 plane channel for frazil Herschel-Bulkley flow.

Why this module exists (defect found by campaign node P519-N04)
--------------------------------------------------------------
:mod:`tensorlbm.herschel_bulkley_lbm` builds its H-B viscosity from a shear rate
that is estimated with :func:`tensorlbm.herschel_bulkley_lbm.wall_aware_gradient`,
which divides by ``1 + 0.5`` when one side of the stencil is a solid cell ("the
physical wall sits half a lattice unit from the adjacent fluid node").  The
bounce-back actually used by the same module
(:func:`tensorlbm.herschel_bulkley_lbm.moving_wall_bounce_back_2d`) is
**node-based**: the incoming population of the solid node is replaced by the
opposite population of its fluid neighbour, which places the no-slip plane at
the *centre of the solid row*, one full lattice unit from the adjacent fluid
node.  The two assumptions are inconsistent, so the near-wall shear rate is
over-estimated by a factor ``2/1.5 = 1.33`` for a shear-thinning material
``nu ~ K gamma_dot^(n-1)`` the near-wall viscosity comes out ~8 % low at
``n = 0.7`` and the measured shear-rate error is ~15 % on a converged profile —
the dominant error of this verification surface before the fix.

What this module does
---------------------
It keeps the audited collision / streaming / bounce-back operators of
:mod:`tensorlbm.herschel_bulkley_lbm` and replaces only the shear-rate estimate
and the time loop, so that the constitutive law sees a shear rate that is
consistent with the actual wall location:

* interior nodes: the second-order central difference over two lattice units;
* the first fluid node next to a wall: the same central difference, with the
  solid node replaced by the **wall velocity** (``0`` at a resting plate,
  ``U`` at a moving plate) — the wall plane is one cell away, not half a cell;
* the plug region then sees ``gamma_dot -> 0`` (the bounce-back state of the
  solid node is nearly the wall velocity), which is what the yield-stress
  regularisation needs.

Nothing in the existing module is modified, so every earlier case that depends
on :func:`tensorlbm.herschel_bulkley_lbm.run_hb_channel` keeps its behaviour;
the fix is available through :func:`run_frazil_hb_channel`.
"""

from __future__ import annotations

import torch

from .herschel_bulkley import HBRheology
from .herschel_bulkley_lbm import (
    collide_hb_bgk_forced_2d,
    hb_relaxation_time,
    macroscopic,
    moving_wall_bounce_back_2d,
    stream,
)

__all__ = [
    "shear_rate_channel_consistent",
    "run_frazil_hb_channel",
    "channel_gap",
]


def channel_gap(gap_cells: int) -> float:
    """Discrete gap of the node-based bounce-back channel (``N + 1`` cells)."""
    if gap_cells < 1:
        raise ValueError(f"gap_cells must be >= 1, got {gap_cells!r}")
    return float(gap_cells) + 1.0


def shear_rate_channel_consistent(
    ux: torch.Tensor,
    fluid: torch.Tensor,
    u_wall_lower: float = 0.0,
    u_wall_upper: float = 0.0,
) -> torch.Tensor:
    """``|du/dy|`` on the node-based bounce-back channel, walls included.

    The solid rows carry the bounce-back state, not the wall velocity, so their
    values are replaced by the prescribed plate speeds before differencing (the
    no-slip plane is the centre of the solid row, i.e. **one** lattice unit from
    the adjacent fluid node).

    Args:
        ux: Streamwise velocity field ``(ny, nx)``.
        fluid: Boolean fluid mask ``(ny, nx)`` (``False`` on the two wall rows).
        u_wall_lower, u_wall_upper: Plate speeds at the lower / upper wall.

    Returns:
        Absolute shear rate, same shape as ``ux``.
    """
    if ux.dim() != 2:
        msg = f"ux must be 2-D (ny, nx), got {tuple(ux.shape)}"
        raise ValueError(msg)
    if ux.shape != fluid.shape:
        msg = f"ux {tuple(ux.shape)} and fluid mask {tuple(fluid.shape)} must match"
        raise ValueError(msg)
    u = ux.to(torch.float64).clone()
    n = u.shape[0]
    wall_values = torch.zeros_like(u)
    wall_values[0, :] = u_wall_lower
    wall_values[n - 1, :] = u_wall_upper
    # Only the two boundary rows of the channel are walls here.
    wall_mask = torch.zeros_like(fluid)
    wall_mask[0, :] = True
    wall_mask[n - 1, :] = True
    u = torch.where(wall_mask, wall_values, u)

    up = torch.empty_like(u)
    dn = torch.empty_like(u)
    up[1:] = u[1:]
    up[0] = u[0]
    dn[:-1] = u[:-1]
    dn[-1] = u[-1]
    # interior: central difference; boundary rows: one-sided (they are walls and
    # are excluded from every metric anyway)
    grad = torch.empty_like(u)
    # the wall plane is one cell from the first fluid node, so the wall rows are
    # differenced with unit spacing too (they are excluded from every metric)
    grad[1:-1] = (up[2:] - dn[:-2]) / 2.0
    grad[0] = u[1] - u[0]
    grad[-1] = u[-1] - u[-2]
    return grad.abs()


def run_frazil_hb_channel(
    gap_cells: int,
    rheo: HBRheology,
    a: float,
    u_wall: float = 0.0,
    *,
    nx: int = 1,
    n_steps: int = 40000,
    m: float = 1.0e5,
    nu_max: float | None = None,
    tau_max: float | None = None,
    check_every: int = 1000,
    tol: float = 1.0e-9,
    u_init: torch.Tensor | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> dict:
    """Steady H-B plane channel with the wall-consistent shear-rate operator.

    Same geometry and operators as
    :func:`tensorlbm.herschel_bulkley_lbm.run_hb_channel` (``gap_cells`` fluid
    rows between two solid rows, body force ``a`` in +x, lower plate at rest,
    upper plate at ``u_wall``), but the viscosity is built from
    :func:`shear_rate_channel_consistent`.

    Args:
        gap_cells: Number of fluid rows (the gap is ``gap_cells + 1`` cells).
        rheo: H-B parameters (kinematic).
        a: Body-force acceleration in +x.
        u_wall: Upper-plate speed in +x.
        nx: Periodic streamwise extent (fully developed flow is x-invariant).
        n_steps, check_every, tol: Steady-state driver on the mean velocity.
        m, nu_max, tau_max: Regularisation / clamping.
        u_init: Optional initial velocity profile ``(gap_cells + 2,)``.
        dtype, device: Torch dtype / device.

    Returns:
        dict with ``ux``, ``u_profile``, ``y``, ``gamma_dot`` (consistent),
        ``steps``, ``converged``, ``history``, ``tau_field``.
    """
    ny = gap_cells + 2
    f = torch.zeros((9, ny, nx), dtype=dtype, device=device)
    rho0 = torch.ones((ny, nx), dtype=dtype, device=device)
    zero = torch.zeros_like(rho0)
    from .herschel_bulkley_lbm import equilibrium

    if u_init is None:
        f[:] = equilibrium(rho0, zero, zero)
    else:
        f[:] = equilibrium(
            rho0,
            u_init.to(device=device, dtype=dtype).unsqueeze(1).expand(ny, nx),
            zero,
        )
    lower = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    lower[0, :] = True
    upper = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    upper[-1, :] = True
    fluid_b = torch.ones((ny, nx), dtype=torch.bool, device=device)
    fluid_b[0, :] = False
    fluid_b[-1, :] = False
    y = torch.arange(ny, dtype=torch.float64, device=device)  # wall distance

    history: list[tuple[int, float]] = []
    converged = False
    steps = 0
    tau_field = None
    u_prev = 0.0
    for step in range(1, n_steps + 1):
        _, ux, uy = macroscopic(f)
        gamma = shear_rate_channel_consistent(ux, fluid_b, 0.0, u_wall)
        tau_field = hb_relaxation_time(
            gamma, rheo, m=m, nu_max=nu_max, tau_max=tau_max
        )
        f = collide_hb_bgk_forced_2d(f, a, rheo, tau_field=tau_field)
        f = stream(f)
        f = moving_wall_bounce_back_2d(f, lower, 0.0, 0.0)
        f = moving_wall_bounce_back_2d(f, upper, u_wall, 0.0)
        steps = step
        if step % check_every == 0 or step == n_steps:
            _, ux, _ = macroscopic(f)
            probe = float(ux[1:-1, :].abs().mean())
            history.append((step, probe))
            if step > check_every and abs(probe - u_prev) <= tol * max(1.0e-30, abs(u_prev)):
                converged = True
                break
            u_prev = probe
    _, ux, _ = macroscopic(f)
    gamma = shear_rate_channel_consistent(ux, fluid_b, 0.0, u_wall)
    return {
        "ux": ux,
        "y": y,
        "u_profile": ux.mean(dim=1),
        "gamma_dot": gamma,
        "steps": steps,
        "converged": converged,
        "history": history,
        "tau_field": tau_field,
    }