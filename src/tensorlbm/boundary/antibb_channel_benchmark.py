"""Reference channel-flow harness for the anti-bounceback (Inamuro) wet-node BC.

This module provides a minimal, self-contained D2Q9 BGK solver with

* Guo body forcing (so pressure-driven Poiseuille flow needs no density
  gradient inflow/outflow bookkeeping),
* periodic streamwise direction,
* wall rows implemented *either* with the anti-bounceback (Inamuro) wet-node
  rule of :mod:`tensorlbm.boundary.antibb_wetnode`, *or* with halfway
  bounce-back, *or* with the force-corrected NEBB kernel of
  :mod:`tensorlbm.nebb_bodyforce` (which imposes the prescribed wall velocity
  and density on the boundary row exactly and shifts the wall velocity by the
  half force step), all with prescribed (possibly moving) wall velocity,

plus the exact analytic references (plane Poiseuille, plane Couette) and the
error metrics used by the verification study
(``docs/antibb_wetnode_verification_20261113.md``,
``scripts/run_antibb_wetnode_verification.py``, ``tests/test_antibb_wetnode.py``).

The solver is deliberately small (a few thousand lattice cells) so the whole
verification study runs in seconds on one core.  It is a *verification*
harness, not a production solver: it is single-phase, two-dimensional and
BGK-only, which is exactly the configuration needed to isolate the error of
the boundary rule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch

from tensorlbm.d2q9 import C, OPPOSITE, equilibrium
from tensorlbm.boundary.antibb_wetnode import (
    antibb_unknown_mask,
    apply_antibb_wetnode,
    bottom_top_normals,
    wall_density_from_fluid,
)

__all__ = [
    "ChannelSetup",
    "ChannelResult",
    "run_channel",
    "poiseuille_profile",
    "couette_profile",
    "l2_relative_error",
    "wall_offsets",
]


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass
class ChannelSetup:
    """Geometry and physics of a plane channel bounded by wet-node wall rows.

    Rows ``0`` and ``ny-1`` are boundary rows; the streamwise direction ``x``
    is periodic.  Node coordinates are ``(x, y) = (column, row)`` with ``y``
    increasing towards the top wall.
    """

    ny: int = 33
    nx: int = 16
    tau: float = 0.65
    force: float = 0.0          # streamwise body force density (per volume)
    u_wall_bottom: float = 0.0  # x-velocity prescribed on row 0
    u_wall_top: float = 0.0     # x-velocity prescribed on row ny-1
    rho0: float = 1.0
    steps: int = 20000
    wall_mode: str = "antibb"   # "antibb" | "halfway" | "force_corrected_nebb" | "none"
    rho_mode: str = "one"       # "one" | "extrapolate" (antibb only)
    dtype: torch.dtype = torch.float64
    sample_rows: Optional[tuple[int, ...]] = None
    record_every: int = 0       # >0 stores the profile history every N steps
    prescribe_wall_in_collision: bool = True
    # ^ a wet node carrying a Dirichlet velocity uses the prescribed value in
    #   the collision equilibrium; the anti-bounceback rule then only fixes the
    #   populations that stream in from outside the domain.  Setting this to
    #   False (ablation) leaves the boundary-node velocity to the populations
    #   alone and is unstable -- measured drift u_wall -> 0.375 lu in 6e3 steps
    #   (see docs/antibb_wetnode_verification_20261113.md).
    steady_tol: float = 0.0     # >0: stop once max|u(t)-u(t-check)| < tol
    check_every: int = 200      # steady-state check cadence

    @property
    def nu(self) -> float:
        return (self.tau - 0.5) / 3.0

    @property
    def wall_offset(self) -> float:
        """Distance of the no-slip surface beyond the boundary-node plane."""
        return 0.0 if self.wall_mode in ("antibb", "force_corrected_nebb", "none") else 0.5

    @property
    def hydraulic_width(self) -> float:
        """Distance between the two no-slip surfaces, in lattice units."""
        return (self.ny - 1) + 2.0 * self.wall_offset


@dataclass
class ChannelResult:
    ux: torch.Tensor
    uy: torch.Tensor
    rho: torch.Tensor
    mass_initial: float
    mass_final: float
    steps_used: int = 0
    steady_residual: float = float("nan")
    history: dict = field(default_factory=dict)

    @property
    def mass_drift(self) -> float:
        return (self.mass_final - self.mass_initial) / self.mass_initial


def wall_offsets(wall_mode: str) -> tuple[float, float]:
    """Analytic wall positions ``(bottom, top)`` measured from row centres."""
    if wall_mode in ("antibb", "force_corrected_nebb"):
        # the NEBB kernel pins (rho, u) on the boundary *node*: the no-slip
        # surface lies on the node plane (measured offset < 0.01 cell,
        # runs/XVAL-NEBB-BODYFORCE/results.json)
        return 0.0, 0.0
    if wall_mode == "none":
        return 0.0, 1.0
    if wall_mode == "halfway":
        return -0.5, 0.5
    raise ValueError(f"unknown wall_mode {wall_mode!r}")


# ---------------------------------------------------------------------------
# analytic references
# ---------------------------------------------------------------------------


def poiseuille_profile(setup: ChannelSetup) -> torch.Tensor:
    """Exact plane-Poiseuille solution ``u(Y) = g Y (H - Y) / (2 nu)``.

    ``Y`` is measured from the *no-slip surface*, so the wall offset of the
    boundary rule (0 for anti-bounceback, 0.5 for halfway bounce-back) enters
    here.  Returned for every row ``y = 0 .. ny-1``.
    """
    bot, _ = wall_offsets(setup.wall_mode)
    H = setup.hydraulic_width
    y = torch.arange(setup.ny, dtype=torch.float64) - bot
    return setup.force * y * (H - y) / (2.0 * setup.nu * setup.rho0)


def couette_profile(setup: ChannelSetup) -> torch.Tensor:
    """Exact plane-Couette solution for the prescribed wall velocities."""
    bot, _ = wall_offsets(setup.wall_mode)
    H = setup.hydraulic_width
    y = torch.arange(setup.ny, dtype=torch.float64) - bot
    ub, ut = float(setup.u_wall_bottom), float(setup.u_wall_top)
    return ub + (ut - ub) * (y / H)


def l2_relative_error(numeric: torch.Tensor, exact: torch.Tensor) -> float:
    """``||numeric - exact||_2 / ||exact||_2`` (scalar tensors of any shape)."""
    diff = (numeric.double() - exact.double()).norm()
    return float(diff / exact.double().norm().clamp_min(1e-300))


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


def _initial_state(setup: ChannelSetup):
    ny, nx = setup.ny, setup.nx
    rho = torch.full((ny, nx), setup.rho0, dtype=setup.dtype)
    ux = torch.zeros((ny, nx), dtype=setup.dtype)
    uy = torch.zeros((ny, nx), dtype=setup.dtype)
    return equilibrium(rho, ux, uy)


def _stream_indices(ny: int, nx: int) -> torch.Tensor:
    """Flat gather indices implementing ``f_i(x) <- f_i(x - c_i)`` (periodic x)."""
    yy = torch.arange(ny).view(-1, 1)
    xx = torch.arange(nx).view(1, -1)
    idx = torch.empty((9, ny * nx), dtype=torch.long)
    for i in range(9):
        sy = (yy - int(C[i, 1])) % ny
        sx = (xx - int(C[i, 0])) % nx
        idx[i] = (sy * nx + sx).reshape(-1)
    return idx


def run_channel(setup: ChannelSetup) -> ChannelResult:
    """March the channel to steady state and return the final fields."""
    if setup.ny < 4:
        raise ValueError("need at least 4 rows (2 wall rows + 2 fluid rows)")
    if setup.tau <= 0.5:
        raise ValueError("tau must exceed 0.5 for a positive viscosity")
    if setup.wall_mode not in ("antibb", "halfway", "none", "force_corrected_nebb"):
        raise ValueError(f"unknown wall_mode {setup.wall_mode!r}")

    ny, nx = setup.ny, setup.nx
    dt = setup.dtype
    c = C.to(dtype=dt)
    w = torch.tensor([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4, dtype=dt)
    cx = c[:, 0].view(-1, 1, 1)
    cy = c[:, 1].view(-1, 1, 1)
    wb = w.view(-1, 1, 1)
    opp = OPPOSITE

    wet = torch.zeros((ny, nx), dtype=torch.bool)
    wet[0, :] = True
    wet[-1, :] = True
    normals = bottom_top_normals(ny, nx, dtype=dt)
    unknown = antibb_unknown_mask(normals, wet, c)

    u_wall = torch.zeros((ny, nx, 2), dtype=dt)
    u_wall[0, :, 0] = setup.u_wall_bottom
    u_wall[-1, :, 0] = setup.u_wall_top
    uwx = u_wall[..., 0]
    uwy = u_wall[..., 1]

    idx = _stream_indices(ny, nx)
    q, ncell = 9, ny * nx

    f = _initial_state(setup)
    mass0 = float(f.sum())
    history: dict[str, list] = {"step": [], "u_mid": [], "mass": []}

    g = float(setup.force)
    guo = 1.0 - 1.0 / (2.0 * setup.tau)
    inv_tau = 1.0 / setup.tau

    prev_profile = None
    steps_used = setup.steps
    steady_residual = float("nan")
    for step in range(setup.steps):
        if setup.steady_tol > 0.0 and step % setup.check_every == 0 and step > 0:
            prof = ((f * cx).sum(dim=0) / f.sum(dim=0))[:, 0]
            if prev_profile is not None:
                res = float((prof - prev_profile).abs().max())
                if res < setup.steady_tol:
                    steps_used, steady_residual = step, res
                    break
            prev_profile = prof
        rho = f.sum(dim=0)
        jx = (f * cx).sum(dim=0)
        jy = (f * cy).sum(dim=0)
        ux = (jx + 0.5 * g) / rho
        uy = jy / rho
        if setup.prescribe_wall_in_collision:
            ux = torch.where(wet, uwx, ux)
            uy = torch.where(wet, uwy, uy)

        u_sq = ux * ux + uy * uy
        cu = cx * ux + cy * uy
        feq = wb * rho.unsqueeze(0) * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u_sq.unsqueeze(0))
        # Guo (1997) forcing for F = (g, 0):
        #   S_i = (1 - 1/(2tau)) w_i g [3 (c_ix - ux) + 9 (c_i.u) c_ix]
        force_i = wb * g
        f_post = (
            f - (f - feq) * inv_tau
            + guo * (3.0 * force_i * (cx - ux.unsqueeze(0)) + 9.0 * cu * force_i * cx)
        )

        # streaming: f_i(x, t+1) = f_i^post(x - c_i, t)
        f_str = f_post.reshape(q, ncell).gather(1, idx).reshape(q, ny, nx)

        if setup.wall_mode == "none":
            f = f_str  # negative control: no wall treatment at all
        elif setup.wall_mode == "antibb":
            rho_w = (
                wall_density_from_fluid(f_str, wet, normals)
                if setup.rho_mode == "extrapolate"
                else setup.rho0
            )
            f = apply_antibb_wetnode(
                f_post, f_str, wet, normals, u_wall, rho_w, velocities=c
            )
        elif setup.wall_mode == "force_corrected_nebb":
            # force-corrected NEBB (tensorlbm.nebb_bodyforce): the shear-
            # reconstruction kernel imposes (rho_wall, u_wall - g/(2 rho_wall))
            # on the boundary rows, i.e. no-slip in the *physical* velocity.
            from tensorlbm.nebb_bodyforce import (
                corrected_wall_velocity,
                nebb_row,
                wall_spec,
            )

            f = f_str.clone()
            for loc, row, u_prescribed in (
                ("bottom", 0, setup.u_wall_bottom),
                ("top", -1, setup.u_wall_top),
            ):
                spec = wall_spec(loc)
                wall = f[:, row, :]
                rho_w = wall.sum(dim=0)
                uxw, uyw = corrected_wall_velocity(u_prescribed, 0.0, g, 0.0, rho_w)
                f[:, row, :] = nebb_row(wall, spec, rho_w, uxw, uyw)
        else:  # halfway bounce-back with moving-wall momentum correction
            reflected = f_post[opp] + 6.0 * wb * (cx * uwx.unsqueeze(0) + cy * uwy.unsqueeze(0))
            f = torch.where(unknown, reflected, f_str)

        if setup.record_every and (step % setup.record_every == 0):
            history["step"].append(step)
            history["u_mid"].append(float(((f * cx).sum(dim=0) / f.sum(dim=0))[ny // 2, nx // 2]))
            history["mass"].append(float(f.sum()))

    rho = f.sum(dim=0)
    ux = (f * cx).sum(dim=0) / rho
    uy = (f * cy).sum(dim=0) / rho
    return ChannelResult(
        ux=ux,
        uy=uy,
        rho=rho,
        mass_initial=mass0,
        mass_final=float(f.sum()),
        steps_used=steps_used,
        steady_residual=steady_residual,
        history=history,
    )
