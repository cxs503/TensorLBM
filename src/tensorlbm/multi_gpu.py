"""Multi-GPU Lattice Boltzmann Method via domain decomposition.

Implements a slab-decomposition strategy that splits the simulation domain
along the x-axis across multiple CUDA devices.  Each device owns one slab
plus one ghost layer on each side for halo exchange.

Architecture
------------
::

    Device 0: f[0  .. nx//N + 1]   (slice + right ghost)
    Device 1: f[nx//N - 1 .. 2*nx//N + 1]
    …
    Device N-1: f[(N-1)*nx//N - 1 .. nx]

Standard LBM step (SDAA flow)
-----------------------------
1. ``f_pre = f.clone()``                          — save pre-collision state
2. ``collide(f, tau)``                            — collision (BGK / MRT / Smagorinsky-MRT)
3. ``NoDynamics: f[q] = where(solid, f_pre[q], f[q])`` — restore solid cells
4. ``bounce_back(f, solid)``                      — half-way bounce-back
5. ``stream(f)``                                  — streaming
6. ``far_field_bc(f, u_in)``                      — far-field boundary conditions
7. ``correct_mass(f, im)`` every N steps          — mass correction

For multi-GPU, steps 1–5 are done per-slab, then halo exchange synchronises
ghost cells, then step 6 applies BCs (with ``has_left``/``has_right`` flags
to avoid applying inlet/outlet on interior ghost cells).

Usage
-----
::

    from tensorlbm.multi_gpu import MultiGPUSolver3D, DomainDecomposition

    dd = DomainDecomposition.from_devices([0, 1, 2, 3], nx_global=200)
    solver = MultiGPUSolver3D(f_global, dd)
    solver.setup_solid(solid_global)           # distribute solid mask to slabs
    solver.setup_bc(u_in=0.06)                 # pre-compute feq cache for BC

    for step in range(1, n_steps + 1):
        solver.step_standard(tau=0.514, cs_smag=0.05, step=step,
                             mass_correction_interval=200)
        if step % 1000 == 0:
            f_global = solver.gather()

References
----------
Succi S., et al. (2001) "Lattice Boltzmann for distributed and large-scale
simulations." *Comput. Phys. Commun.* 134(3).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Domain decomposition
# ---------------------------------------------------------------------------

@dataclass
class DomainDecomposition:
    """Describes how the global domain is split across devices.

    Attributes:
        devices:    List of device identifiers (e.g. ``['cuda:0', 'cuda:1']``).
        nx_global:  Global domain width (number of columns).
        overlap:    Ghost-layer width (default 1).
        slabs:      List of ``(x_start, x_end)`` tuples for each device.
                    Automatically computed from *devices* and *nx_global*.
    """
    devices: list[str]
    nx_global: int
    overlap: int = 1
    slabs: list[tuple[int, int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.slabs:
            self.slabs = self._compute_slabs()

    def _compute_slabs(self) -> list[tuple[int, int]]:
        n = len(self.devices)
        base = self.nx_global // n
        rem  = self.nx_global % n
        slabs = []
        start = 0
        for i in range(n):
            width = base + (1 if i < rem else 0)
            end = start + width
            slabs.append((start, end))
            start = end
        return slabs

    @classmethod
    def from_devices(cls, device_ids: list[int], nx_global: int = 0) -> DomainDecomposition:
        """Convenience constructor from integer GPU IDs."""
        devices = [f"cuda:{d}" for d in device_ids]
        return cls(devices=devices, nx_global=nx_global)

    @property
    def n_devices(self) -> int:
        return len(self.devices)


# ---------------------------------------------------------------------------
# Halo exchange
# ---------------------------------------------------------------------------

def halo_exchange_2d(
    slabs: list[torch.Tensor],
    decomp: DomainDecomposition,
) -> list[torch.Tensor]:
    """Exchange one-cell ghost layers between adjacent D2Q9 slabs."""
    ov = decomp.overlap
    for i in range(len(slabs) - 1):
        right_of_i     = slabs[i][:, :, -ov - 1:-1]
        left_ghost_ip1 = slabs[i + 1][:, :, :ov]
        left_ghost_ip1.copy_(right_of_i.to(left_ghost_ip1.device))

        left_of_ip1  = slabs[i + 1][:, :, ov:2 * ov]
        right_ghost_i = slabs[i][:, :, -ov:]
        right_ghost_i.copy_(left_of_ip1.to(right_ghost_i.device))
    return slabs


def halo_exchange_3d(
    slabs: list[torch.Tensor],
    decomp: DomainDecomposition,
) -> list[torch.Tensor]:
    """Exchange ghost layers between D3Q19/D3Q27 slabs (x-decomposition).

    Each slab has shape ``(Q, nz, ny, nx_local + 2*overlap)`` where Q is
    19 (D3Q19) or 27 (D3Q27).
    """
    ov = decomp.overlap
    for i in range(len(slabs) - 1):
        r = slabs[i][:, :, :, -ov - 1:-1]
        lg = slabs[i + 1][:, :, :, :ov]
        lg.copy_(r.to(lg.device))

        l_ = slabs[i + 1][:, :, :, ov:2 * ov]
        rg = slabs[i][:, :, :, -ov:]
        rg.copy_(l_.to(rg.device))
    return slabs


# ---------------------------------------------------------------------------
# Far-field BC with pre-computed feq cache (multi-GPU safe)
# ---------------------------------------------------------------------------

def build_feq_cache_d3q19(nz, ny, nx, u_in, device, uy=0.0, uz=0.0):
    """Pre-compute feq slices for D3Q19 far-field BC (call once per slab).

    Returns a tuple of 5 tensors:
        (feq_inlet, feq_y_minus, feq_y_plus, feq_z_minus, feq_z_plus)
    """
    from tensorlbm.d3q19 import equilibrium3d

    # Inlet: x=0 plane (nz, ny)
    rho_in = torch.ones(nz, ny, dtype=torch.float32, device=device)
    feq_in = equilibrium3d(
        rho_in, torch.full_like(rho_in, u_in),
        torch.full_like(rho_in, uy), torch.full_like(rho_in, uz),
        device=device,
    )[:, :, :, 0]  # (19, nz, ny)

    # y- lateral: y=0 plane (nz, nx)
    rho_ym = torch.ones(nz, nx, dtype=torch.float32, device=device)
    feq_ym = equilibrium3d(
        rho_ym, torch.full_like(rho_ym, u_in),
        torch.full_like(rho_ym, uy), torch.full_like(rho_ym, uz),
        device=device,
    )[:, :, 0, :]  # (19, nz, nx)
    feq_yp = feq_ym.clone()

    # z- lateral: z=0 plane (ny, nx)
    rho_zm = torch.ones(ny, nx, dtype=torch.float32, device=device)
    feq_zm = equilibrium3d(
        rho_zm, torch.full_like(rho_zm, u_in),
        torch.full_like(rho_zm, uy), torch.full_like(rho_zm, uz),
        device=device,
    )[:, 0, :, :]  # (19, ny, nx)
    feq_zp = feq_zm.clone()

    return (feq_in, feq_ym, feq_yp, feq_zm, feq_zp)


def build_feq_cache_d3q27(nz, ny, nx, u_in, device, uy=0.0, uz=0.0):
    """Pre-compute feq slices for D3Q27 far-field BC (call once per slab)."""
    from tensorlbm.d3q27 import equilibrium27

    rho_in = torch.ones(nz, ny, dtype=torch.float32, device=device)
    feq_in = equilibrium27(
        rho_in, torch.full_like(rho_in, u_in),
        torch.full_like(rho_in, uy), torch.full_like(rho_in, uz),
        device=device,
    )[:, :, :, 0]

    rho_ym = torch.ones(nz, nx, dtype=torch.float32, device=device)
    feq_ym = equilibrium27(
        rho_ym, torch.full_like(rho_ym, u_in),
        torch.full_like(rho_ym, uy), torch.full_like(rho_ym, uz),
        device=device,
    )[:, :, 0, :]
    feq_yp = feq_ym.clone()

    rho_zm = torch.ones(ny, nx, dtype=torch.float32, device=device)
    feq_zm = equilibrium27(
        rho_zm, torch.full_like(rho_zm, u_in),
        torch.full_like(rho_zm, uy), torch.full_like(rho_zm, uz),
        device=device,
    )[:, 0, :, :]
    feq_zp = feq_zm.clone()

    return (feq_in, feq_ym, feq_yp, feq_zm, feq_zp)


def far_field_bc_cached(f, feq_cache, has_left=True, has_right=True):
    """Apply far-field BC using pre-computed feq slices.

    Only applies inlet/outlet BC at actual domain boundaries
    (``has_left``/``has_right``).  Lateral BCs are always applied.

    This is the multi-GPU safe replacement for ``far_field_bc_3d``:
    the original applies inlet BC to ALL slabs' x=0 (including ghost cells
    of interior slabs), which causes ~12% drag error.
    """
    feq_in, feq_ym, feq_yp, feq_zm, feq_zp = feq_cache

    if has_left:
        f[:, :, :, 0] = feq_in       # inlet
    if has_right:
        f[:, :, :, -1] = f[:, :, :, -2]  # outlet (zero gradient)

    # Lateral boundaries — always apply
    f[:, 0, :, :] = feq_ym
    f[:, -1, :, :] = feq_yp
    f[:, :, 0, :] = feq_zm
    f[:, :, -1, :] = feq_zp

    return f


# ---------------------------------------------------------------------------
# Multi-GPU 2-D solver
# ---------------------------------------------------------------------------

class MultiGPUSolver2D:
    """Multi-GPU D2Q9 LBM solver using x-axis domain decomposition.

    The global distribution function ``f_global`` (shape ``(9, ny, nx)``) is
    split into slabs along x, one per device.  Each slab includes one ghost
    column on each side for halo exchange.

    Usage::

        dd = DomainDecomposition(
            devices=["cuda:0", "cuda:1"],
            nx_global=512,
        )
        solver = MultiGPUSolver2D(f_global, dd)
        for step in range(n_steps):
            solver.step(collide_fn, stream_fn, boundary_fn)
        f = solver.gather()

    Args:
        f_global: Global initial distribution (9, ny, nx) on any device.
        decomp:   Domain decomposition descriptor (must have ``nx_global``
                  set to ``nx``).
    """

    def __init__(
        self,
        f_global: torch.Tensor,
        decomp: DomainDecomposition,
    ) -> None:
        q, ny, nx = f_global.shape
        if decomp.nx_global == 0:
            decomp = DomainDecomposition(
                devices=decomp.devices,
                nx_global=nx,
                overlap=decomp.overlap,
            )
        assert decomp.nx_global == nx, (
            f"decomp.nx_global ({decomp.nx_global}) != nx ({nx})"
        )
        self.decomp = decomp
        self.ny = ny
        self._step_count = 0

        ov = decomp.overlap
        self.slabs: list[torch.Tensor] = []
        for dev, (x0, x1) in zip(decomp.devices, decomp.slabs):
            x0g = max(0, x0 - ov)
            x1g = min(nx, x1 + ov)
            slab = f_global[:, :, x0g:x1g].to(dev).contiguous()
            self.slabs.append(slab)
        self._x_ranges = decomp.slabs

    def step(
        self,
        collide_fn: Callable,
        stream_fn: Callable,
        boundary_fn: Callable | None = None,
    ) -> None:
        """Advance one time step across all slabs."""
        for i, slab in enumerate(self.slabs):
            self.slabs[i] = collide_fn(slab)
            self.slabs[i] = stream_fn(self.slabs[i])
            if boundary_fn is not None:
                self.slabs[i] = boundary_fn(self.slabs[i])
        halo_exchange_2d(self.slabs, self.decomp)
        self._step_count += 1

    def gather(self) -> torch.Tensor:
        """Assemble slab interior regions back into a single global tensor."""
        q = self.slabs[0].shape[0]
        ny = self.ny
        nx = self.decomp.nx_global
        ov = self.decomp.overlap
        f_out = torch.zeros((q, ny, nx), dtype=self.slabs[0].dtype)
        for slab, (x0, x1) in zip(self.slabs, self._x_ranges):
            x0g_local = ov if x0 > 0 else 0
            x1g_local = slab.shape[2] - ov if x1 < nx else slab.shape[2]
            local_width = x1 - x0
            f_out[:, :, x0:x1] = slab[:, :, x0g_local:x0g_local + local_width].cpu()
        return f_out

    @property
    def n_devices(self) -> int:
        return self.decomp.n_devices


# ---------------------------------------------------------------------------
# Multi-GPU 3-D solver
# ---------------------------------------------------------------------------

class MultiGPUSolver3D:
    """Multi-GPU D3Q19/D3Q27 LBM solver using x-axis domain decomposition.

    Supports both the simple ``step()`` interface (collide/stream/boundary
    callbacks) and the full ``step_standard()`` method that implements the
    SDAA standard LBM flow:

    1. Save pre-collision state
    2. Collision (BGK / MRT / Smagorinsky-MRT)
    3. NoDynamics (restore solid cells)
    4. Bounce-back
    5. Streaming
    6. Halo exchange
    7. Far-field BC (with ``has_left``/``has_right`` flags)
    8. Mass correction (every N steps)

    Usage (standard flow)::

        dd = DomainDecomposition.from_devices([0, 1, 2, 3], nx_global=200)
        solver = MultiGPUSolver3D(f_global, dd)
        solver.setup_solid(solid_global)
        solver.setup_bc(u_in=0.06)

        for step in range(1, n_steps + 1):
            solver.step_standard(tau=0.514, cs_smag=0.05, step=step)
            if step % 1000 == 0:
                f_global = solver.gather()

    Args:
        f_global: Global distribution (Q, nz, ny, nx) on CPU or any device.
        decomp:   Domain decomposition descriptor.
    """

    def __init__(
        self,
        f_global: torch.Tensor,
        decomp: DomainDecomposition,
    ) -> None:
        q, nz, ny, nx = f_global.shape
        if decomp.nx_global == 0:
            decomp = DomainDecomposition(
                devices=decomp.devices,
                nx_global=nx,
                overlap=decomp.overlap,
            )
        self.decomp = decomp
        self.nz = nz
        self.ny = ny
        self.nx = nx
        self.q = q
        self._step_count = 0

        ov = decomp.overlap
        self.slabs: list[torch.Tensor] = []
        for dev, (x0, x1) in zip(decomp.devices, decomp.slabs):
            x0g = max(0, x0 - ov)
            x1g = min(nx, x1 + ov)
            slab = f_global[:, :, :, x0g:x1g].to(dev).contiguous()
            self.slabs.append(slab)
        self._x_ranges = decomp.slabs

        # Per-slab auxiliary data (set by setup_solid / setup_bc)
        self._slab_solids: list[torch.Tensor | None] = []
        self._slab_sm: list[torch.Tensor | None] = []
        self._slab_feq_caches: list[tuple | None] = []
        self._slab_has_left: list[bool] = []
        self._slab_has_right: list[bool] = []
        self._bc_ready = False
        self._solid_ready = False

    # ------------------------------------------------------------------
    # Setup methods
    # ------------------------------------------------------------------

    def setup_solid(self, solid_global: torch.Tensor) -> None:
        """Distribute the global solid mask to per-slab masks on each device.

        Args:
            solid_global: Boolean mask ``(nz, ny, nx)`` on CPU.
        """
        ov = self.decomp.overlap
        nx = self.nx
        self._slab_solids = []
        self._slab_sm = []
        self._slab_has_left = []
        self._slab_has_right = []

        for i, (x0, x1) in enumerate(self.decomp.slabs):
            dev = torch.device(self.decomp.devices[i])
            x0g = max(0, x0 - ov)
            x1g = min(nx, x1 + ov)
            slab_solid = solid_global[:, :, x0g:x1g].to(dev)
            self._slab_solids.append(slab_solid)
            # Pre-compute expanded mask for NoDynamics
            self._slab_sm.append(slab_solid.unsqueeze(0).expand(self.q, *slab_solid.shape))
            self._slab_has_left.append(x0 == 0)
            self._slab_has_right.append(x1 == nx)

        self._solid_ready = True

    def setup_bc(self, u_in: float, uy: float = 0.0, uz: float = 0.0) -> None:
        """Pre-compute feq cache for far-field BC on each slab.

        Must be called after ``setup_solid()`` (needs slab dimensions).

        Args:
            u_in: Inlet velocity.
            uy:   Lateral y-velocity (default 0).
            uz:   Lateral z-velocity (default 0).
        """
        assert self._solid_ready, "Call setup_solid() before setup_bc()"
        self._slab_feq_caches = []
        build_fn = build_feq_cache_d3q19 if self.q == 19 else build_feq_cache_d3q27

        for i in range(self.decomp.n_devices):
            dev = torch.device(self.decomp.devices[i])
            slab_nx = self._slab_solids[i].shape[2]
            cache = build_fn(self.nz, self.ny, slab_nx, u_in, dev, uy, uz)
            self._slab_feq_caches.append(cache)

        self._bc_ready = True

    # ------------------------------------------------------------------
    # Step methods
    # ------------------------------------------------------------------

    def step(
        self,
        collide_fn: Callable,
        stream_fn: Callable,
        boundary_fn: Callable | None = None,
    ) -> None:
        """Simple one-step advance: collide → stream → boundary → halo.

        For the standard SDAA flow, use ``step_standard()`` instead.
        """
        for i, slab in enumerate(self.slabs):
            self.slabs[i] = collide_fn(slab)
            self.slabs[i] = stream_fn(self.slabs[i])
            if boundary_fn is not None:
                self.slabs[i] = boundary_fn(self.slabs[i])
        halo_exchange_3d(self.slabs, self.decomp)
        self._step_count += 1

    def step_standard(
        self,
        tau: float,
        step: int,
        cs_smag: float = 0.0,
        collision: str = "mrt_smag",
        mass_correction_interval: int = 200,
        im: float | None = None,
    ) -> None:
        """Advance one time step using the SDAA standard LBM flow.

        Flow:
            1. f_pre = f.clone()
            2. collide (BGK / MRT / MRT+Smag)
            3. NoDynamics: restore solid cells
            4. bounce_back
            5. stream
            6. synchronize + halo exchange
            7. far_field_bc_cached (with has_left/has_right)
            8. mass correction (every ``mass_correction_interval`` steps)

        Args:
            tau:   Relaxation time.
            step:  Current step number (1-based, for mass correction).
            cs_smag: Smagorinsky constant (0 = no Smagorinsky).
            collision: "bgk", "mrt", or "mrt_smag" (default).
            mass_correction_interval: Apply mass correction every N steps.
                Set to 0 to disable.
            im: Initial mass for correction.  If None, correction is skipped.
        """
        assert self._solid_ready, "Call setup_solid() before step_standard()"
        assert self._bc_ready, "Call setup_bc() before step_standard()"

        # Import collision functions lazily to avoid circular imports
        if collision == "bgk":
            from tensorlbm.solver3d import collide_bgk3d as _collide
            def collide_fn(f):
                return _collide(f, tau)
        elif collision == "mrt":
            from tensorlbm.solver3d import collide_mrt3d as _collide
            def collide_fn(f):
                return _collide(f, tau)
        elif collision == "mrt_smag":
            from tensorlbm.turbulence import collide_smagorinsky_mrt3d as _collide
            def collide_fn(f):
                return _collide(f, tau=tau, C_s=cs_smag)
        else:
            raise ValueError(f"Unknown collision type: {collision}")

        # Import bounce-back and stream
        if self.q == 19:
            from tensorlbm.boundaries3d import bounce_back_cells_3d as _bb
        else:
            from tensorlbm.boundaries_d3q27 import bounce_back_cells_27 as _bb
        from tensorlbm.solver3d import stream3d

        n = self.decomp.n_devices

        # Steps 1-5: per-slab (collide → NoDynamics → BB → stream)
        for i in range(n):
            f_pre = self.slabs[i].clone()
            self.slabs[i] = collide_fn(self.slabs[i])
            # NoDynamics: restore solid cells to pre-collision values
            sm = self._slab_sm[i]
            for q in range(self.q):
                self.slabs[i][q] = torch.where(sm[q], f_pre[q], self.slabs[i][q])
            # Bounce-back
            self.slabs[i] = _bb(self.slabs[i], self._slab_solids[i])
            # Streaming
            self.slabs[i] = stream3d(self.slabs[i])

        # Step 6: synchronize + halo exchange
        for i in range(n):
            with torch.cuda.device(self.decomp.devices[i]):
                torch.cuda.synchronize()
        halo_exchange_3d(self.slabs, self.decomp)

        # Step 7: far-field BC (with has_left/has_right)
        for i in range(n):
            self.slabs[i] = far_field_bc_cached(
                self.slabs[i], self._slab_feq_caches[i],
                has_left=self._slab_has_left[i],
                has_right=self._slab_has_right[i],
            )

        # Step 8: mass correction (gather → correct → scatter → halo)
        if mass_correction_interval > 0 and im is not None and step % mass_correction_interval == 0:
            f_gathered = self.gather()
            from tensorlbm.solver3d import correct_mass3d
            f_gathered = correct_mass3d(f_gathered, im)
            # Re-scatter corrected field (including ghost cells)
            ov = self.decomp.overlap
            nx = self.nx
            for i, (x0, x1) in enumerate(self.decomp.slabs):
                dev = torch.device(self.decomp.devices[i])
                x0g = max(0, x0 - ov)
                x1g = min(nx, x1 + ov)
                self.slabs[i] = f_gathered[:, :, :, x0g:x1g].to(dev).contiguous()
            del f_gathered
            # Re-sync + halo exchange to refresh ghost cells after scatter
            for i in range(n):
                with torch.cuda.device(self.decomp.devices[i]):
                    torch.cuda.synchronize()
            halo_exchange_3d(self.slabs, self.decomp)

        self._step_count += 1

    # ------------------------------------------------------------------
    # Gather
    # ------------------------------------------------------------------

    def gather(self) -> torch.Tensor:
        """Assemble slab interiors into a global tensor on CPU.

        Works for both D3Q19 (Q=19) and D3Q27 (Q=27).
        """
        q = self.q
        nz, ny = self.nz, self.ny
        nx = self.decomp.nx_global
        ov = self.decomp.overlap
        f_out = torch.zeros((q, nz, ny, nx), dtype=self.slabs[0].dtype)
        for slab, (x0, x1) in zip(self.slabs, self._x_ranges):
            x0g_local = ov if x0 > 0 else 0
            local_width = x1 - x0
            f_out[:, :, :, x0:x1] = slab[:, :, :, x0g_local:x0g_local + local_width].cpu()
        return f_out

    @property
    def n_devices(self) -> int:
        return self.decomp.n_devices


# ---------------------------------------------------------------------------
# Convenience: auto-detect and use all available GPUs
# ---------------------------------------------------------------------------

def auto_decompose(
    f_global: torch.Tensor,
    n_gpus: int | None = None,
) -> DomainDecomposition:
    """Build a :class:`DomainDecomposition` using all available CUDA devices.

    Args:
        f_global: Global distribution tensor.  Shape determines nx_global.
        n_gpus:   Override GPU count (default: all available GPUs, or 1 CPU).

    Returns:
        Configured :class:`DomainDecomposition`.
    """
    if n_gpus is None:
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus == 0:
        devices = ["cpu"]
    else:
        devices = [f"cuda:{i}" for i in range(n_gpus)]

    nx = f_global.shape[-1]
    return DomainDecomposition(devices=devices, nx_global=nx)


__all__ = [
    "DomainDecomposition",
    "MultiGPUSolver2D",
    "MultiGPUSolver3D",
    "halo_exchange_2d",
    "halo_exchange_3d",
    "auto_decompose",
    "build_feq_cache_d3q19",
    "build_feq_cache_d3q27",
    "far_field_bc_cached",
]
