"""Pre-allocated buffer pool for LBM time-step performance optimisation.

This module provides :class:`LBMStepBuffer`, a container that pre-allocates
all temporary tensors needed during one Lattice Boltzmann time step.
Buffers are allocated **once** at construction and **reused** every step,
eliminating per-step memory allocation overhead.

Supported lattices: ``D3Q19`` (Q=19) and ``D3Q27`` (Q=27).

The buffer is **solver-agnostic**: it does not contain any collision,
streaming, or boundary logic.  It only provides pre-allocated memory that
any solver, collision operator, or wall-function routine may write into.

Typical usage::

    buf = LBMStepBuffer.for_lattice("D3Q19", nz, ny, nx, device=device)
    for step in range(n_steps):
        buf.compute_macroscopic_into(f, lattice="D3Q19")
        # ... use buf.rho, buf.ux, buf.uy, buf.uz, buf.feq, etc.
"""
from __future__ import annotations

from typing import Literal

import torch

__all__ = ["LBMStepBuffer"]

_LatticeName = Literal["D3Q19", "D3Q27"]
_Q_FOR_LATTICE: dict[str, int] = {"D3Q19": 19, "D3Q27": 27}


class LBMStepBuffer:
    """Pre-allocated tensor pool for one LBM time step.

    All tensors are allocated at construction and persist for the lifetime
    of the object.  Writing to a buffer (e.g. ``buf.feq.copy_(...)``) reuses
    the same storage — no new memory is allocated per step.

    Attributes
    ----------
    f_post : Tensor (Q, nz, ny, nx)
        Post-collision distribution (scratch for collide output).
    feq : Tensor (Q, nz, ny, nx)
        Equilibrium distribution.
    f_stream : Tensor (Q, nz, ny, nx)
        Post-streaming distribution.
    fneq : Tensor (Q, nz, ny, nx)
        Non-equilibrium distribution (temporary for collide).
    rho, ux, uy, uz : Tensor (nz, ny, nx)
        Macroscopic density and velocity components.
    u_mag : Tensor (nz, ny, nx)
        Velocity magnitude.
    u_tau : Tensor (nz, ny, nx)
        Friction velocity (wall function).
    y_plus : Tensor (nz, ny, nx)
        Dimensionless wall distance.
    """

    def __init__(
        self,
        q: int,
        nz: int,
        ny: int,
        nx: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.q = q
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.device = torch.device(device)
        self.dtype = dtype

        shape_q = (q, nz, ny, nx)
        shape_3d = (nz, ny, nx)

        # --- Distribution buffers (Q, nz, ny, nx) ---
        self.f_post = torch.empty(shape_q, dtype=dtype, device=device)
        self.feq = torch.empty(shape_q, dtype=dtype, device=device)
        self.f_stream = torch.empty(shape_q, dtype=dtype, device=device)
        self.fneq = torch.empty(shape_q, dtype=dtype, device=device)

        # --- Macroscopic field buffers (nz, ny, nx) ---
        self.rho = torch.empty(shape_3d, dtype=dtype, device=device)
        self.ux = torch.empty(shape_3d, dtype=dtype, device=device)
        self.uy = torch.empty(shape_3d, dtype=dtype, device=device)
        self.uz = torch.empty(shape_3d, dtype=dtype, device=device)
        self.u_mag = torch.empty(shape_3d, dtype=dtype, device=device)

        # --- Wall-function buffers (nz, ny, nx) ---
        self.u_tau = torch.empty(shape_3d, dtype=dtype, device=device)
        self.y_plus = torch.empty(shape_3d, dtype=dtype, device=device)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def for_lattice(
        cls,
        lattice: str,
        nz: int,
        ny: int,
        nx: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "LBMStepBuffer":
        """Create a buffer pool for the given lattice name.

        Args:
            lattice: ``"D3Q19"`` or ``"D3Q27"``.
            nz, ny, nx: Grid dimensions.
            device: Target device.
            dtype: Tensor dtype (default ``torch.float32``).
        """
        if lattice not in _Q_FOR_LATTICE:
            raise ValueError(
                f"Unsupported lattice {lattice!r}; supported: {list(_Q_FOR_LATTICE)}"
            )
        return cls(
            q=_Q_FOR_LATTICE[lattice],
            nz=nz,
            ny=ny,
            nx=nx,
            device=device,
            dtype=dtype,
        )

    # ------------------------------------------------------------------
    # Macroscopic computation into pre-allocated buffers
    # ------------------------------------------------------------------

    def compute_macroscopic_into(
        self,
        f: torch.Tensor,
        lattice: str = "D3Q19",
    ) -> None:
        """Compute (rho, ux, uy, uz) from *f* and write into buffer fields.

        This avoids allocating new tensors for the macroscopic fields on
        every step.  The results are stored in ``self.rho``, ``self.ux``,
        ``self.uy``, ``self.uz``.

        Args:
            f: Distribution tensor of shape ``(Q, nz, ny, nx)``.
            lattice: ``"D3Q19"`` or ``"D3Q27"``.
        """
        if lattice == "D3Q19":
            from .d3q19 import _c_on, _w_on  # noqa: PLC0415

            q = 19
        elif lattice == "D3Q27":
            from .d3q27 import _c_on, _w_on  # noqa: PLC0415

            q = 27
        else:
            raise ValueError(f"Unsupported lattice: {lattice!r}")

        c = _c_on(self.device)
        cx = c[:, 0].view(q, 1, 1, 1)
        cy = c[:, 1].view(q, 1, 1, 1)
        cz = c[:, 2].view(q, 1, 1, 1)

        # rho = sum_q f_q  (in-place into self.rho)
        torch.sum(f, dim=0, out=self.rho)

        # rho_safe for division
        rho_safe = self.rho.clamp(min=1e-12)

        # ux = sum_q cx * f / rho  (in-place into self.ux)
        # Use torch.mul with out= to avoid allocating a temporary.
        torch.mul(f, cx, out=self.fneq)
        torch.sum(self.fneq, dim=0, out=self.ux)
        self.ux.div_(rho_safe)

        torch.mul(f, cy, out=self.fneq)
        torch.sum(self.fneq, dim=0, out=self.uy)
        self.uy.div_(rho_safe)

        torch.mul(f, cz, out=self.fneq)
        torch.sum(self.fneq, dim=0, out=self.uz)
        self.uz.div_(rho_safe)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all buffers (useful for debugging)."""
        for t in (
            self.f_post, self.feq, self.f_stream, self.fneq,
            self.rho, self.ux, self.uy, self.uz, self.u_mag,
            self.u_tau, self.y_plus,
        ):
            t.zero_()
