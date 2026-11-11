"""Graph-preserving moving-wall (lid) boundary condition, D2Q9.

Why this module exists
----------------------
``tensorlbm.lid_driven_cavity.zou_he_moving_lid`` is the production 2D moving
lid: it *clones* the distribution and then overwrites the top row with in-place
``f_new[4, -1, :] = ...`` slice assignments.  In-place slice writes are the
pattern that ``docs/differentiable_path.md`` recorded as gradient-hostile
(``CopySlices`` nodes, version-counter errors as soon as the same field is used
again, and functionalisation breaks under ``torch.compile``), and the D3Q19
autograd chain (``tensorlbm.autograd_path``) explicitly documents "no
moving-wall/lid method" as a known limit.  This module closes that gap for the
D2Q9 lid-driven cavity:

* :func:`moving_lid_values` / :func:`moving_lid_bc` -- the Zou & He (1997)
  moving-wall closure rebuilt **out of place**: the top plane is re-assembled
  from the nine per-direction planes with ``torch.stack`` and spliced back with
  ``torch.cat``.  No in-place write, no clone-then-mutate: ``d/df`` and
  ``d/du_lid`` cross the lid exactly, and the wall velocity may be a graph
  node (``0-dim`` tensor or an ``(nx,)`` profile).
* :func:`step` / :func:`rollout` -- an autograd-clean collide -> stream ->
  no-slip walls -> moving lid step, with optional per-block gradient
  checkpointing so long roll-outs stay in memory.
* :func:`centreline_probes`, :func:`invert_relaxation_time` -- the
  solver-in-the-loop inverse machinery: ``d(observable)/d(tau)`` through
  thousands of time steps, used to invert ``tau`` from cavity data.

Closure (validated, not assumed)
--------------------------------
With the D2Q9 convention of :mod:`tensorlbm.d2q9` and a top wall moving at
``u_w = (u_lid, 0)``, the three populations entering the domain are
``{4=(0,-1), 7=(-1,-1), 8=(1,-1)}`` and the known (post-streaming) ones are
``{0, 1, 2, 3, 5, 6}``.  Zou/He momentum closure gives

.. math::

    \\rho &= f_0 + f_1 + 2f_2 + f_3 + 2f_5 + 2f_6 \\\\
    f_4 &= f_2 \\\\
    f_7 &= f_5 + \\tfrac12 (f_1 - f_3) - \\tfrac12 \\rho u_{lid} \\\\
    f_8 &= f_6 - \\tfrac12 (f_1 - f_3) + \\tfrac12 \\rho u_{lid}

which is algebraically identical to the production implementation (checked in
``tests/test_diff_lid_cavity.py``) but written with out-of-place ops only.
Two exact invariants are pinned there: a uniform Galilean state
``f = f^eq(1, (u,0))`` is a fixed point of the lid closure, and the lid plane
carries ``j_x = rho u_lid``, ``j_y = 0``.

References
----------
* Zou, Q. & He, X. (1997) "On pressure and velocity boundary conditions for the
  lattice Boltzmann BGK model", Phys. Fluids 9(6), 1591-1598 -- the closure.
* Ghia, U., Ghia, K. N. & Shin, C. T. (1982) "High-Re solutions for
  incompressible flow using the Navier-Stokes equations and a multigrid
  method", J. Comput. Phys. 48(3), 387-411 -- Re=100 centreline data used as
  the external anchor of :func:`invert_relaxation_time` runs.
* Ataei & Salehipour (2024) CPC 300:109187 / Um et al. (2020) NeurIPS --
  solver-in-the-loop differentiation of LBM time stepping.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .boundaries import bounce_back_cells
from .d2q9 import equilibrium, macroscopic
from .solver import collide_bgk, stream

__all__ = [
    "CavityGeometry",
    "cavity_wall_mask",
    "centreline_probes",
    "initial_field",
    "invert_relaxation_time",
    "moving_lid_bc",
    "moving_lid_values",
    "rollout",
    "step",
    "tau_from_reynolds",
]


# ---------------------------------------------------------------------------
# geometry / initial state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CavityGeometry:
    """Grid + wall bookkeeping for the closed cavity.

    ``ny``/``nx`` count lattice *nodes*; the outermost node layer is the solid
    wall (bottom/left/right) and the top layer carries the moving lid, matching
    ``lid_driven_cavity.make_cavity_wall_mask(include_top=False)``.
    """

    ny: int
    nx: int
    dtype: torch.dtype = torch.float64
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def __post_init__(self) -> None:
        if self.ny < 4 or self.nx < 4:
            msg = f"cavity needs at least 4x4 nodes, got ny={self.ny}, nx={self.nx}"
            raise ValueError(msg)

    @property
    def wall_mask(self) -> torch.Tensor:
        """Boolean ``(ny, nx)`` mask of the stationary walls (lid excluded)."""
        return cavity_wall_mask(self.ny, self.nx, device=self.device)


def cavity_wall_mask(ny: int, nx: int, device: torch.device | None = None) -> torch.Tensor:
    """Stationary-wall mask: bottom row, left column, right column.

    The top row is deliberately excluded -- it is rebuilt by the moving-lid
    closure (:func:`moving_lid_bc`), exactly as in the production solver.
    """
    mask = torch.zeros((ny, nx), dtype=torch.bool, device=device)
    mask[0, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def tau_from_reynolds(re: float, u_lid: float, length: int) -> float:
    """``tau`` from the cavity Reynolds number ``Re = u_lid * L / nu``.

    ``L`` is the cavity side in lattice units (``length``), ``nu = (tau-0.5)/3``.
    """
    nu = u_lid * float(length) / float(re)
    return 0.5 + 3.0 * nu


def _as_lid_velocity(u_lid: float | torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcastable lid velocity: 0-dim tensor (``u_y = 0`` assumed)."""
    if isinstance(u_lid, torch.Tensor):
        return u_lid.to(dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(float(u_lid), dtype=ref.dtype, device=ref.device)


def initial_field(
    geom: CavityGeometry,
    u_lid: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Quiescent fluid (``f = f^eq(1, 0)``) of shape ``(9, ny, nx)``."""
    rho = torch.ones((geom.ny, geom.nx), dtype=geom.dtype, device=geom.device)
    zero = torch.zeros_like(rho)
    return equilibrium(rho, zero, zero).clone()


# ---------------------------------------------------------------------------
# the moving-lid closure (graph-preserving)
# ---------------------------------------------------------------------------


def moving_lid_values(
    f_top: torch.Tensor,
    u_lid: torch.Tensor | float,
) -> torch.Tensor:
    """Zou/He moving-lid closure on a single top-wall row.

    Args:
        f_top: ``(9, nx)`` post-streaming populations on the lid row.
        u_lid: Lid velocity in the +x direction; a scalar, a 0-dim graph
            tensor, or an ``(nx,)`` profile.

    Returns:
        ``(9, nx)`` rebuilt lid plane.  Pure tensor arithmetic: differentiable
        in ``f_top`` and in ``u_lid`` (no in-place write, no detach).
    """
    if f_top.shape[0] != 9:
        msg = f"moving_lid_values expects (9, nx) populations, got {tuple(f_top.shape)}"
        raise ValueError(msg)
    u = _as_lid_velocity(u_lid, f_top)
    f0, f1, f2 = f_top[0], f_top[1], f_top[2]
    f3, f4, f5, f6 = f_top[3], f_top[4], f_top[5], f_top[6]
    f7, f8 = f_top[7], f_top[8]
    del f4, f7, f8  # incoming populations are fully replaced by the closure

    rho = f0 + f1 + 2.0 * f2 + f3 + 2.0 * f5 + 2.0 * f6
    new4 = f2
    new7 = f5 + 0.5 * (f1 - f3) - 0.5 * rho * u
    new8 = f6 - 0.5 * (f1 - f3) + 0.5 * rho * u
    return torch.stack((f0, f1, f2, f3, new4, f5, f6, new7, new8), dim=0)


def moving_lid_bc(
    f: torch.Tensor,
    u_lid: torch.Tensor | float,
) -> torch.Tensor:
    """Apply the graph-preserving moving lid to ``(9, ny, nx)`` populations.

    The top plane is rebuilt by :func:`moving_lid_values` and spliced back with
    ``torch.cat``, so the returned tensor is a clean function of ``f`` and
    ``u_lid`` -- the operation autograd sees is a ``stack``/``cat`` pair, not a
    ``CopySlices`` in-place overwrite.
    """
    if f.dim() != 3 or f.shape[0] != 9:
        msg = f"moving_lid_bc expects (9, ny, nx), got {tuple(f.shape)}"
        raise ValueError(msg)
    return torch.cat((f[:, :-1, :], moving_lid_values(f[:, -1, :], u_lid).unsqueeze(1)), dim=1)


# ---------------------------------------------------------------------------
# time stepping
# ---------------------------------------------------------------------------


def step(
    f: torch.Tensor,
    tau: torch.Tensor | float,
    u_lid: torch.Tensor | float,
    wall_mask: torch.Tensor,
) -> torch.Tensor:
    """One autograd-clean D2Q9 step: collide -> stream -> walls -> moving lid."""
    f = collide_bgk(f, tau)
    f = stream(f)
    f = bounce_back_cells(f, wall_mask)
    return moving_lid_bc(f, u_lid)


_COMPILED_STEP: Any = None


def _get_compiled_step() -> Any:
    """Lazily ``torch.compile`` :func:`step` (6x speed-up at 64x64 on CPU)."""
    global _COMPILED_STEP
    if _COMPILED_STEP is None:
        _COMPILED_STEP = torch.compile(step)
    return _COMPILED_STEP


def _rollout_block_with(
    f: torch.Tensor,
    tau: torch.Tensor,
    u_lid: torch.Tensor,
    wall_mask: torch.Tensor,
    n_steps: int,
    one: Any,
) -> torch.Tensor:
    for _ in range(n_steps):
        f = one(f, tau, u_lid, wall_mask)
    return f


def _rollout_block(
    f: torch.Tensor,
    tau: torch.Tensor,
    u_lid: torch.Tensor,
    wall_mask: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    for _ in range(n_steps):
        f = step(f, tau, u_lid, wall_mask)
    return f


def rollout(
    f0: torch.Tensor,
    tau: torch.Tensor | float,
    u_lid: torch.Tensor | float,
    n_steps: int,
    wall_mask: torch.Tensor | None = None,
    checkpoint_every: int = 0,
    compile_step: bool = False,
) -> torch.Tensor:
    """Run ``n_steps`` differentiable steps from ``f0``.

    Args:
        f0: ``(9, ny, nx)`` starting populations.
        tau: Relaxation time; a graph tensor keeps ``d/dtau`` alive.
        u_lid: Lid velocity; graph tensor keeps ``d/du_lid`` alive.
        n_steps: Number of time steps (fixed -- no data-dependent stopping, so
            finite differences and autograd differentiate the *same* map).
        wall_mask: Optional ``(ny, nx)`` stationary-wall mask.
        checkpoint_every: If > 0, checkpoint the rollout in blocks of this many
            steps (``use_reentrant=False``), trading ~2x forward time for
            O(block) activation memory.
        compile_step: Wrap :func:`step` in ``torch.compile`` (values and
            gradients unchanged to round-off; ~6x faster on CPU).

    Returns:
        ``(9, ny, nx)`` populations after ``n_steps``.
    """
    if n_steps < 0:
        msg = f"n_steps must be >= 0, got {n_steps}"
        raise ValueError(msg)
    if wall_mask is None:
        wall_mask = cavity_wall_mask(f0.shape[1], f0.shape[2], device=f0.device)
    tau_t = tau if isinstance(tau, torch.Tensor) else torch.as_tensor(float(tau), dtype=f0.dtype)
    u_t = _as_lid_velocity(u_lid, f0)

    one = _get_compiled_step() if compile_step else step
    f = f0
    if checkpoint_every <= 0:
        for _ in range(n_steps):
            f = one(f, tau_t, u_t, wall_mask)
        return f

    done = 0
    while done < n_steps:
        block = min(checkpoint_every, n_steps - done)
        f = torch.utils.checkpoint.checkpoint(
            _rollout_block_with, f, tau_t, u_t, wall_mask, block, one, use_reentrant=False
        )
        done += block
    return f


# ---------------------------------------------------------------------------
# observables (centreline probes, Ghia layout)
# ---------------------------------------------------------------------------

#: Standard Ghia sampling stations (fraction of the cavity side).
GHIA_STATIONS: tuple[float, ...] = (
    0.0625,
    0.1250,
    0.1875,
    0.2500,
    0.3125,
    0.3750,
    0.4375,
    0.5000,
    0.5625,
    0.6250,
    0.6875,
    0.7500,
    0.8125,
    0.8750,
    0.9375,
)


def _linear_sample(values: torch.Tensor, stations: Sequence[float], axis: int) -> torch.Tensor:
    """Differentiable linear interpolation of a 1-D lattice profile."""
    n = values.shape[0]
    lo = torch.tensor([min(int(s * (n - 1)), n - 2) for s in stations], dtype=torch.long)
    frac = torch.tensor(
        [s * (n - 1) - int(min(int(s * (n - 1)), n - 2)) for s in stations], dtype=values.dtype
    )
    lo = lo.to(values.device)
    frac = frac.to(values.device)
    hi = lo + 1
    return values[lo] * (1.0 - frac) + values[hi] * frac


def centreline_probes(
    f: torch.Tensor,
    u_lid: torch.Tensor | float | None = None,
    stations: Sequence[float] = GHIA_STATIONS,
) -> torch.Tensor:
    """Ghia-layout centreline velocities, normalised by ``u_lid`` when given.

    Returns a ``(2 * len(stations),)`` tensor: ``u_x / u_lid`` at the vertical
    centreline (x = nx//2) then ``u_y / u_lid`` at the horizontal centreline
    (y = ny//2), both sampled at ``stations`` from bottom/left to top/right.
    """
    rho, ux, uy = macroscopic(f)
    del rho
    ny, nx = ux.shape
    u_vert = ux[:, nx // 2]
    v_horz = uy[ny // 2, :]
    obs = torch.cat(
        (
            _linear_sample(u_vert, stations, axis=0),
            _linear_sample(v_horz, stations, axis=1),
        )
    )
    if u_lid is None:
        return obs
    scale = _as_lid_velocity(u_lid, obs)
    return obs / scale


# ---------------------------------------------------------------------------
# inverse problem: recover tau from centreline data
# ---------------------------------------------------------------------------


def invert_relaxation_time(
    target_obs: torch.Tensor,
    geom: CavityGeometry,
    u_lid: float,
    tau_init: float,
    n_steps: int,
    iters: int = 20,
    lr: float = 0.02,
    method: str = "adam",
    max_step: float = 0.15,
    step_tol: float = 1e-7,
    checkpoint_every: int = 200,
    stations: Sequence[float] = GHIA_STATIONS,
    log: Any = None,
    compile_step: bool = False,
    loss_tol: float = 0.0,
) -> dict[str, Any]:
    """Recover ``tau`` by gradient descent through the whole simulation.

    The parameter is ``s = log(tau - 0.5)`` (so ``tau > 0.5`` is structural and
    ``nu = exp(s)/3`` cannot go negative); the loss is the mean squared
    discrepancy of :func:`centreline_probes` against ``target_obs``.  ``n_steps``
    is fixed, hence autograd and finite differences differentiate exactly the
    same map.

    Returns:
        Dict with ``tau_init``, ``tau_recovered``, ``tau_history``,
        ``loss_history``, ``obs_final`` and the residual norm.
    """
    dtype, device = geom.dtype, geom.device
    u = torch.as_tensor(float(u_lid), dtype=dtype, device=device)
    s = torch.tensor(math.log(tau_init - 0.5), dtype=dtype, device=device, requires_grad=True)
    if method not in ("newton", "adam"):
        msg = f"unknown inversion method {method!r} (use 'newton' or 'adam')"
        raise ValueError(msg)
    opt = torch.optim.Adam([s], lr=lr) if method == "adam" else None
    f0 = initial_field(geom)
    mask = geom.wall_mask
    tau_hist: list[float] = []
    loss_hist: list[float] = []
    obs_final = None
    step_size = float("nan")
    for it in range(iters):
        if opt is not None:
            opt.zero_grad(set_to_none=True)
        elif s.grad is not None:
            s.grad = None
        tau = 0.5 + torch.exp(s)
        f = rollout(
            f0,
            tau,
            u,
            n_steps,
            wall_mask=mask,
            checkpoint_every=checkpoint_every,
            # double backward (Newton's autograd Hessian) is unsupported under
            # aot_autograd, so the Newton path runs the eager ops
            compile_step=compile_step and method == "adam",
        )
        obs = centreline_probes(f, u, stations=stations)
        loss = ((obs - target_obs) ** 2).mean()
        if opt is not None:
            loss.backward()
            opt.step()
            grad_val, hess_val = float(s.grad), float("nan")
        else:
            (g,) = torch.autograd.grad(loss, s, create_graph=True)
            (h,) = torch.autograd.grad(g, s)
            delta = torch.clamp(-g / torch.clamp(h, min=1e-14), -max_step, max_step)
            step_size, grad_val, hess_val = float(delta), float(g), float(h)
            with torch.no_grad():
                s -= delta
        tau_now = float(0.5 + torch.exp(s.detach()))
        loss_now = float(loss.detach())
        tau_hist.append(tau_now)
        loss_hist.append(loss_now)
        obs_final = obs.detach()
        if log is not None:
            log(
                f"[invert] iter={it:02d} tau={tau_now:.6f} "
                f"nu={(tau_now - 0.5) / 3.0:.6e} loss={loss_now:.6e} "
                f"grad_s={grad_val:.6e} hess_s={hess_val:.6e} dtau={step_size:+.3e}"
            )
        if loss_tol > 0.0 and loss_now < loss_tol:
            break
        if method == "newton" and abs(step_size) < step_tol:
            if log is not None:
                log(f"[invert] converged: |ds|={step_size:.3e}")
            break
    resid = float(torch.sqrt(((obs_final - target_obs) ** 2).mean()))
    best = int(min(range(len(loss_hist)), key=lambda i: loss_hist[i])) if loss_hist else 0
    return {
        "tau_init": float(tau_init),
        "tau_recovered": tau_hist[-1] if tau_hist else float(tau_init),
        "tau_best": tau_hist[best] if tau_hist else float(tau_init),
        "loss_best": loss_hist[best] if loss_hist else float("nan"),
        "tau_history": tau_hist,
        "loss_history": loss_hist,
        "obs_final": obs_final,
        "residual_rms": resid,
        "n_steps": int(n_steps),
        "iters": int(iters),
        "method": method,
        "final_step": step_size,
    }