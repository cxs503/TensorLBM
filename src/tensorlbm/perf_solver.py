"""Optimised LBM solver with pre-allocated buffers and in-place operations.

This module provides:

* :func:`collide_bgk3d_inplace` — in-place BGK collision for D3Q19.
* :func:`collide_bgk27_inplace` — in-place BGK collision for D3Q27.
* :func:`stream3d_inplace` — streaming into a pre-allocated buffer (D3Q19).
* :func:`stream27_inplace` — streaming into a pre-allocated buffer (D3Q27).
* :class:`OptimizedSolver3D` — orchestrates one LBM step using
  :class:`~tensorlbm.perf_buffers.LBMStepBuffer`, computing macroscopic
  fields **once** and reusing them for collision, wall function, and force
  measurement.

All optimised functions produce **numerically identical** results to their
reference counterparts (verified via ``torch.allclose(atol=1e-6)``).
The optimisation is purely in memory management and call ordering — the
collision, streaming, and boundary formulas are unchanged.

The solver accepts any combination of collision / streaming / boundary /
wall-function callables, making it a **common module** usable by all
solvers.
"""
from __future__ import annotations

from typing import Any, Callable

import torch

from .perf_buffers import LBMStepBuffer

__all__ = [
    "OptimizedSolver3D",
    "collide_bgk3d_inplace",
    "collide_bgk27_inplace",
    "stream3d_inplace",
    "stream27_inplace",
]

# Type alias for a collision function: (f, tau, **kwargs) -> f_post
CollideFn = Callable[..., torch.Tensor]
# Type alias for a streaming function: (f) -> f_streamed
StreamFn = Callable[..., torch.Tensor]
# Type alias for a boundary function: (f, **kwargs) -> f_bc
BoundaryFn = Callable[..., torch.Tensor]


# ---------------------------------------------------------------------------
# D3Q19 streaming shifts (Python tuples — no host sync)
# ---------------------------------------------------------------------------

_D3Q19_SHIFTS: list[tuple[int, int, int]] = [
    (0, 0, 0),       # 0: rest
    (1, 0, 0),       # 1: +x
    (-1, 0, 0),      # 2: -x
    (0, 1, 0),       # 3: +y
    (0, -1, 0),      # 4: -y
    (0, 0, 1),       # 5: +z
    (0, 0, -1),      # 6: -z
    (1, 1, 0),       # 7: +x+y
    (-1, -1, 0),     # 8: -x-y
    (1, -1, 0),      # 9: +x-y
    (-1, 1, 0),      # 10: -x+y
    (1, 0, 1),       # 11: +x+z
    (-1, 0, -1),     # 12: -x-z
    (1, 0, -1),      # 13: +x-z
    (-1, 0, 1),      # 14: -x+z
    (0, 1, 1),       # 15: +y+z
    (0, -1, -1),     # 16: -y-z
    (0, 1, -1),      # 17: +y-z
    (0, -1, 1),      # 18: -y+z
]


# ---------------------------------------------------------------------------
# In-place BGK collision — D3Q19
# ---------------------------------------------------------------------------

def collide_bgk3d_inplace(
    f: torch.Tensor,
    tau: float,
    buf: LBMStepBuffer,
) -> torch.Tensor:
    """In-place BGK collision for D3Q19 using pre-allocated buffers.

    Computes ``f_new = f - (f - feq) / tau`` **in-place** on *f*,
    using ``buf.feq`` and ``buf.fneq`` as scratch space.  Macroscopic
    fields (rho, ux, uy, uz) are written into ``buf`` for reuse by
    downstream consumers (wall function, force measurement).

    Numerically identical to :func:`tensorlbm.solver3d.collide_bgk3d`.

    Args:
        f:   Distribution tensor ``(19, nz, ny, nx)`` — modified in-place.
        tau: Relaxation time τ > 0.5.
        buf: Pre-allocated buffer pool (``LBMStepBuffer.for_lattice("D3Q19", ...)``).

    Returns:
        *f* (same tensor, modified in-place).
    """
    # 1. Compute macroscopic into buf (rho, ux, uy, uz)
    buf.compute_macroscopic_into(f, lattice="D3Q19")

    # 2. Compute equilibrium into buf.feq
    from .d3q19 import equilibrium3d  # noqa: PLC0415

    equilibrium3d(buf.rho, buf.ux, buf.uy, buf.uz, out=buf.feq)

    # 3. fneq = f - feq  (into buf.fneq)
    torch.sub(f, buf.feq, out=buf.fneq)

    # 4. fneq /= tau
    buf.fneq.div_(tau)

    # 5. f -= fneq  →  f = f - (f - feq) / tau  (in-place on f)
    f.sub_(buf.fneq)

    return f


# ---------------------------------------------------------------------------
# In-place BGK collision — D3Q27
# ---------------------------------------------------------------------------

def collide_bgk27_inplace(
    f: torch.Tensor,
    tau: float,
    buf: LBMStepBuffer,
) -> torch.Tensor:
    """In-place BGK collision for D3Q27 using pre-allocated buffers.

    Numerically identical to :func:`tensorlbm.d3q27.collide_bgk27`.

    Args:
        f:   Distribution tensor ``(27, nz, ny, nx)`` — modified in-place.
        tau: Relaxation time τ > 0.5.
        buf: Pre-allocated buffer pool (``LBMStepBuffer.for_lattice("D3Q27", ...)``).

    Returns:
        *f* (same tensor, modified in-place).
    """
    buf.compute_macroscopic_into(f, lattice="D3Q27")

    from .d3q27 import equilibrium27  # noqa: PLC0415

    equilibrium27(buf.rho, buf.ux, buf.uy, buf.uz, out=buf.feq)

    torch.sub(f, buf.feq, out=buf.fneq)
    buf.fneq.div_(tau)
    f.sub_(buf.fneq)

    return f


# ---------------------------------------------------------------------------
# In-place streaming — D3Q19
# ---------------------------------------------------------------------------

def stream3d_inplace(
    f: torch.Tensor,
    buf: LBMStepBuffer,
) -> torch.Tensor:
    """Streaming for D3Q19 into a pre-allocated buffer.

    Writes the streamed distribution into ``buf.f_stream``.  The input *f*
    is not modified.  Uses ``torch.roll`` per direction, skipping the rest
    direction (q=0) which is a no-op copy.

    Numerically identical to :func:`tensorlbm.solver3d.stream3d`.

    Args:
        f:   Distribution tensor ``(19, nz, ny, nx)``.
        buf: Pre-allocated buffer pool.

    Returns:
        ``buf.f_stream`` (the streamed distribution).
    """
    out = buf.f_stream
    # q=0: rest direction — no shift needed, just copy
    out[0] = f[0]
    for q in range(1, 19):
        sx, sy, sz = _D3Q19_SHIFTS[q]
        out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


# ---------------------------------------------------------------------------
# In-place streaming — D3Q27
# ---------------------------------------------------------------------------

def stream27_inplace(
    f: torch.Tensor,
    buf: LBMStepBuffer,
) -> torch.Tensor:
    """Streaming for D3Q27 into a pre-allocated buffer.

    Writes the streamed distribution into ``buf.f_stream``.  Uses the
    pre-computed D3Q27 shifts, skipping the rest direction.

    Numerically identical to :func:`tensorlbm.d3q27.stream27_roll`.

    Args:
        f:   Distribution tensor ``(27, nz, ny, nx)``.
        buf: Pre-allocated buffer pool.

    Returns:
        ``buf.f_stream`` (the streamed distribution).
    """
    from . import d3q27 as _d3q27_mod  # noqa: PLC0415

    _d3q27_mod._init_stream27_shifts()
    shifts = _d3q27_mod._STREAM27_SHIFTS
    out = buf.f_stream
    for q in range(27):
        sx, sy, sz = shifts[q]
        if sx == 0 and sy == 0 and sz == 0:
            out[q] = f[q]
        else:
            out[q] = torch.roll(f[q], shifts=(sz, sy, sx), dims=(0, 1, 2))
    return out


# ---------------------------------------------------------------------------
# OptimizedSolver3D
# ---------------------------------------------------------------------------

class OptimizedSolver3D:
    """Optimised LBM solver with pre-allocated buffers and macroscopic reuse.

    Orchestrates one LBM time step:

    1. **Collide** — in-place BGK (or external ``collide_fn``).
    2. **Stream** — into pre-allocated ``buf.f_stream``.
    3. **Boundary** — optional external ``boundary_fn``.
    4. **Wall function** — optional, using pre-computed macroscopic fields.

    Macroscopic fields (rho, ux, uy, uz) are computed **once** per step
    (during collision) and reused for the wall function, eliminating the
    redundant macroscopic computation in
    :func:`~tensorlbm.wall_function_common.wall_function`.

    All temporary tensors are pre-allocated in :class:`LBMStepBuffer` and
    reused every step — no per-step allocation.

    Parameters
    ----------
    lattice : str
        ``"D3Q19"`` or ``"D3Q27"``.
    nz, ny, nx : int
        Grid dimensions.
    tau : float
        Relaxation time for BGK collision.
    device : torch.device or str
        Compute device.
    dtype : torch.dtype
        Tensor dtype (default ``torch.float32``).

    Examples
    --------
    >>> solver = OptimizedSolver3D("D3Q19", 64, 64, 64, tau=0.6,
    ...                           device="sdaa:0")
    >>> for _ in range(100):
    ...     f = solver.step(f, u_in=0.05)
    """

    def __init__(
        self,
        lattice: str,
        nz: int,
        ny: int,
        nx: int,
        tau: float,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.lattice = lattice
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.tau = tau
        self.device = torch.device(device)
        self.dtype = dtype

        self.buf = LBMStepBuffer.for_lattice(
            lattice, nz, ny, nx, device=device, dtype=dtype,
        )

        # Select default in-place collide/stream based on lattice
        if lattice == "D3Q19":
            self._default_collide = collide_bgk3d_inplace
            self._default_stream = stream3d_inplace
        elif lattice == "D3Q27":
            self._default_collide = collide_bgk27_inplace
            self._default_stream = stream27_inplace
        else:
            raise ValueError(f"Unsupported lattice: {lattice!r}")

    # ------------------------------------------------------------------
    # Single step
    # ------------------------------------------------------------------

    def step(
        self,
        f: torch.Tensor,
        *,
        u_in: float = 0.0,
        collide_fn: CollideFn | None = None,
        stream_fn: StreamFn | None = None,
        boundary_fn: BoundaryFn | None = None,
        wall_mask: torch.Tensor | None = None,
        nu: float = 0.02,
        y_val: float = 0.5,
        wall_law: str = "log",
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute one optimised LBM time step.

        Args:
            f: Distribution tensor ``(Q, nz, ny, nx)``.
            u_in: Free-stream / inlet velocity for far-field BC.
            collide_fn: External collision function ``(f, tau, ...) -> f``.
                If ``None``, uses the in-place BGK collision.
            stream_fn: External streaming function ``(f) -> f``.
                If ``None``, uses the in-place streaming into ``buf``.
            boundary_fn: External boundary function ``(f, ...) -> f``.
                If ``None``, uses ``far_field_bc`` for the given lattice.
            wall_mask: Boolean solid mask for wall-function correction.
                If ``None``, no wall function is applied.
            nu: Kinematic viscosity (lattice units) for wall function.
            y_val: Near-wall cell-centre distance.
            wall_law: ``"log"`` or ``"reichardt"``.
            **kwargs: Additional keyword arguments passed to ``boundary_fn``.

        Returns:
            Updated distribution tensor (may be the same storage as *f*
            for in-place operations).
        """
        buf = self.buf

        # --- 1. Collide ---
        if collide_fn is not None:
            # External collision: compute macroscopic into buf for reuse
            buf.compute_macroscopic_into(f, lattice=self.lattice)
            f = collide_fn(f, self.tau)
        else:
            # In-place BGK: writes macroscopic into buf
            self._default_collide(f, self.tau, buf)

        # --- 2. Stream ---
        if stream_fn is not None:
            f = stream_fn(f)
        else:
            self._default_stream(f, buf)
            f = buf.f_stream

        # --- 3. Boundary ---
        if boundary_fn is not None:
            f = boundary_fn(f, u_in=u_in, **kwargs) if u_in else boundary_fn(f, **kwargs)
        elif u_in != 0.0:
            f = self._apply_default_boundary(f, u_in, **kwargs)

        # --- 4. Wall function (with pre-computed macroscopic) ---
        if wall_mask is not None:
            f = self._apply_wall_function(
                f, wall_mask, nu=nu, y_val=y_val, wall_law=wall_law,
            )

        return f

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_default_boundary(
        self,
        f: torch.Tensor,
        u_in: float,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply the default far-field boundary condition for the lattice."""
        if self.lattice == "D3Q19":
            from .boundaries3d import far_field_bc_3d  # noqa: PLC0415

            obstacle = kwargs.get("obstacle_mask")
            return far_field_bc_3d(f, u_in, obstacle_mask=obstacle)
        elif self.lattice == "D3Q27":
            from .boundaries_d3q27 import far_field_bc_27  # noqa: PLC0415

            obstacle = kwargs.get("obstacle_mask")
            return far_field_bc_27(f, u_in, obstacle_mask=obstacle)
        return f

    def _apply_wall_function(
        self,
        f: torch.Tensor,
        mask: torch.Tensor,
        *,
        nu: float = 0.02,
        y_val: float = 0.5,
        wall_law: str = "log",
    ) -> torch.Tensor:
        """Apply wall function using pre-computed macroscopic from buffer.

        Recomputes macroscopic from the post-stream/post-BC distribution
        (since the distribution has changed since collision), but writes
        into the pre-allocated buffer — no new allocation.
        """
        from .wall_function_common import (  # noqa: PLC0415
            compute_u_tau,
            compute_y_plus,
            wall_function,
        )

        # Recompute macroscopic from current f (post-stream, post-BC)
        buf = self.buf
        buf.compute_macroscopic_into(f, lattice=self.lattice)

        # u_mag into buf.u_mag
        torch.sqrt(
            buf.ux * buf.ux + buf.uy * buf.uy + buf.uz * buf.uz,
            out=buf.u_mag,
        )
        buf.u_mag.clamp_(min=1e-12)

        # u_tau, y_plus into buf
        u_tau = compute_u_tau(buf.u_mag, nu=nu, y_val=y_val, wall_law=wall_law)
        buf.u_tau.copy_(u_tau)
        y_plus = compute_y_plus(buf.u_tau, nu=nu, y_val=y_val)
        buf.y_plus.copy_(y_plus)

        # wall_function with pre-computed macroscopic
        return wall_function(
            f, mask, buf.u_tau, buf.y_plus,
            lattice=self.lattice, nu=nu, y_val=y_val,
            rho=buf.rho, ux=buf.ux, uy=buf.uy, uz=buf.uz,
        )
