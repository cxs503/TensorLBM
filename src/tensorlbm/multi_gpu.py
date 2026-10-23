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

import hashlib
import os
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.distributed as dist

from .d3q19 import C

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
        rem = self.nx_global % n
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
        right_of_i = slabs[i][:, :, -ov - 1 : -1]
        left_ghost_ip1 = slabs[i + 1][:, :, :ov]
        left_ghost_ip1.copy_(right_of_i.to(left_ghost_ip1.device))

        left_of_ip1 = slabs[i + 1][:, :, ov : 2 * ov]
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
        r = slabs[i][:, :, :, -ov - 1 : -1]
        lg = slabs[i + 1][:, :, :, :ov]
        lg.copy_(r.to(lg.device))

        l_ = slabs[i + 1][:, :, :, ov : 2 * ov]
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
        rho_in,
        torch.full_like(rho_in, u_in),
        torch.full_like(rho_in, uy),
        torch.full_like(rho_in, uz),
        device=device,
    )[:, :, :, 0]  # (19, nz, ny)

    # y- lateral: y=0 plane (nz, nx)
    rho_ym = torch.ones(nz, nx, dtype=torch.float32, device=device)
    feq_ym = equilibrium3d(
        rho_ym,
        torch.full_like(rho_ym, u_in),
        torch.full_like(rho_ym, uy),
        torch.full_like(rho_ym, uz),
        device=device,
    )[:, :, 0, :]  # (19, nz, nx)
    feq_yp = feq_ym.clone()

    # z- lateral: z=0 plane (ny, nx)
    rho_zm = torch.ones(ny, nx, dtype=torch.float32, device=device)
    feq_zm = equilibrium3d(
        rho_zm,
        torch.full_like(rho_zm, u_in),
        torch.full_like(rho_zm, uy),
        torch.full_like(rho_zm, uz),
        device=device,
    )[:, 0, :, :]  # (19, ny, nx)
    feq_zp = feq_zm.clone()

    return (feq_in, feq_ym, feq_yp, feq_zm, feq_zp)


def build_feq_cache_d3q27(nz, ny, nx, u_in, device, uy=0.0, uz=0.0):
    """Pre-compute feq slices for D3Q27 far-field BC (call once per slab)."""
    from tensorlbm.d3q27 import equilibrium27

    rho_in = torch.ones(nz, ny, dtype=torch.float32, device=device)
    feq_in = equilibrium27(
        rho_in,
        torch.full_like(rho_in, u_in),
        torch.full_like(rho_in, uy),
        torch.full_like(rho_in, uz),
        device=device,
    )[:, :, :, 0]

    rho_ym = torch.ones(nz, nx, dtype=torch.float32, device=device)
    feq_ym = equilibrium27(
        rho_ym,
        torch.full_like(rho_ym, u_in),
        torch.full_like(rho_ym, uy),
        torch.full_like(rho_ym, uz),
        device=device,
    )[:, :, 0, :]
    feq_yp = feq_ym.clone()

    rho_zm = torch.ones(ny, nx, dtype=torch.float32, device=device)
    feq_zm = equilibrium27(
        rho_zm,
        torch.full_like(rho_zm, u_in),
        torch.full_like(rho_zm, uy),
        torch.full_like(rho_zm, uz),
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
        f[:, :, :, 0] = feq_in  # inlet
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
        assert decomp.nx_global == nx, f"decomp.nx_global ({decomp.nx_global}) != nx ({nx})"
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
            slab.shape[2] - ov if x1 < nx else slab.shape[2]
            local_width = x1 - x0
            f_out[:, :, x0:x1] = slab[:, :, x0g_local : x0g_local + local_width].cpu()
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
                self.slabs[i],
                self._slab_feq_caches[i],
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
            f_out[:, :, :, x0:x1] = slab[:, :, :, x0g_local : x0g_local + local_width].cpu()
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


class MultiDeviceSolver3D:
    """Device-agnostic multi-device D3Q19 LBM solver via x-axis decomposition.

    A *common module* that works with any device family (``cuda``, ``sdaa``,
    ``cpu``).  The global distribution is split into x-slabs, one per device.
    Each device runs collision, streaming, and boundary kernels **independently
    and unmodified**; halo exchange synchronises x-interface ghost planes
    between adjacent slabs before streaming.

    Per-step ordering (matching the production ``MultiGPUSolver3D`` contract):

    1. **Collide** — each card applies ``collide_fn`` to its slab.
    2. **Halo exchange** — ghost planes refreshed between adjacent slabs.
    3. **Stream** — each card applies ``stream_fn`` (pull-stream reads
       neighbour data from the freshly-exchanged ghosts).
    4. **Boundary** — each card applies ``boundary_fn`` (if provided).
    5. **Force aggregate** — if ``force_fn`` is provided, each card computes a
       local force contribution and the results are **all-reduced** (summed)
       across all cards.

    Args:
        f_global:    Global initial distribution ``(19, nz, ny, nx)`` on any
                     device.
        devices:     List of device strings, e.g. ``["sdaa:0", "sdaa:1"]``.
        collide_fn:  Collision kernel ``f → f'`` applied per slab.
        stream_fn:   Streaming kernel ``f → f'`` applied per slab.
        boundary_fn: Optional boundary kernel ``f → f'`` applied per slab
                     after streaming.
        force_fn:    Optional force kernel ``f → tensor`` that returns a
                     per-slab force contribution.  Contributions are summed
                     across all cards (all-reduce).
        overlap:     Ghost-layer width (default 1).

    Example::

        from tensorlbm.multi_gpu import MultiDeviceSolver3D
        from tensorlbm.solver3d import collide_bgk3d, stream3d

        solver = MultiDeviceSolver3D(
            f_global=f0,
            devices=[f"sdaa:{i}" for i in range(8)],
            collide_fn=lambda f: collide_bgk3d(f, tau=0.8),
            stream_fn=stream3d,
        )
        for _ in range(n_steps):
            solver.step()
        f_final = solver.gather()
    """

    def __init__(
        self,
        f_global: torch.Tensor,
        devices: list[str],
        collide_fn: Callable[[torch.Tensor], torch.Tensor],
        stream_fn: Callable[[torch.Tensor], torch.Tensor],
        boundary_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        force_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        overlap: int = 1,
    ) -> None:
        q, nz, ny, nx = f_global.shape
        if q != 19:
            raise ValueError(f"MultiDeviceSolver3D requires D3Q19 populations, got {q}")
        self.decomp = DomainDecomposition(
            devices=devices,
            nx_global=nx,
            overlap=overlap,
        )
        self.nz = nz
        self.ny = ny
        self.collide_fn = collide_fn
        self.stream_fn = stream_fn
        self.boundary_fn = boundary_fn
        self.force_fn = force_fn
        self._step_count = 0

        ov = self.decomp.overlap
        self.slabs: list[torch.Tensor] = []
        for i, (dev, (x0, x1)) in enumerate(zip(self.decomp.devices, self.decomp.slabs)):
            # Periodic ghost seeding via modulo indexing, matching
            # MultiGPUSolver3D.  Halo exchange overwrites before first stream.
            dev_obj = torch.device(dev)
            x_indices = torch.arange(x0 - ov, x1 + ov, device=f_global.device) % nx
            slab = f_global.index_select(3, x_indices)
            if dev_obj != f_global.device:
                slab = slab.contiguous().to(dev_obj)
            else:
                slab = slab.contiguous()
            self.slabs.append(slab)
        self._x_ranges = self.decomp.slabs
        halo_exchange_3d(self.slabs, self.decomp)

    def step(self) -> torch.Tensor | None:
        """Advance one time step: collide → halo → stream → boundary → force.

        Returns:
            Aggregated force tensor (on CPU) if ``force_fn`` was provided,
            otherwise ``None``.
        """
        # 1. Collide on each card (independent, unmodified kernel)
        for i, slab in enumerate(self.slabs):
            self.slabs[i] = self.collide_fn(slab)

        # 2. Halo exchange between adjacent slabs
        halo_exchange_3d(self.slabs, self.decomp)

        # 3. Stream on each card (pull-stream reads from refreshed ghosts)
        for i, slab in enumerate(self.slabs):
            self.slabs[i] = self.stream_fn(slab)

        # 4. Boundary on each card (independent, unmodified kernel)
        if self.boundary_fn is not None:
            for i, slab in enumerate(self.slabs):
                self.slabs[i] = self.boundary_fn(slab)

        self._step_count += 1

        # 5. Force all-reduce (sum across cards)
        if self.force_fn is not None:
            return self.reduce_force()
        return None

    def reduce_force(self) -> torch.Tensor:
        """All-reduce (sum) per-card force contributions.

        Each card evaluates ``force_fn`` on its **owned** interior cells
        (ghost layers are stripped, since they contain stale data after
        streaming); the results are copied to CPU and summed.  The returned
        tensor lives on CPU.

        Returns:
            Summed force tensor on CPU.
        """
        assert self.force_fn is not None
        ov = self.decomp.overlap
        total: torch.Tensor | None = None
        for slab, (x0, x1) in zip(self.slabs, self._x_ranges):
            owned = slab[:, :, :, ov : ov + (x1 - x0)]
            local = self.force_fn(owned).detach().cpu()
            total = local.clone() if total is None else total + local
        assert total is not None
        return total

    def gather(self) -> torch.Tensor:
        """Assemble slab interiors into a global tensor on CPU."""
        q = self.slabs[0].shape[0]
        nz, ny = self.nz, self.ny
        nx = self.decomp.nx_global
        ov = self.decomp.overlap
        f_out = torch.zeros((q, nz, ny, nx), dtype=self.slabs[0].dtype)
        for slab, (x0, x1) in zip(self.slabs, self._x_ranges):
            x0g_local = ov
            local_width = x1 - x0
            f_out[:, :, :, x0:x1] = slab[:, :, :, x0g_local : x0g_local + local_width].cpu()
        return f_out

    def gather_macroscopic(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather per-slab macroscopic fields into global (rho, ux, uy, uz).

        Each slab's owned cells are converted to macroscopic quantities and
        concatenated along x into global fields on CPU.

        Returns:
            ``(rho, ux, uy, uz)`` each of shape ``(nz, ny, nx)`` on CPU.
        """
        from .d3q19 import macroscopic3d

        nz, ny = self.nz, self.ny
        nx = self.decomp.nx_global
        ov = self.decomp.overlap
        dtype = self.slabs[0].dtype
        rho_out = torch.zeros((nz, ny, nx), dtype=dtype)
        ux_out = torch.zeros((nz, ny, nx), dtype=dtype)
        uy_out = torch.zeros((nz, ny, nx), dtype=dtype)
        uz_out = torch.zeros((nz, ny, nx), dtype=dtype)
        for slab, (x0, x1) in zip(self.slabs, self._x_ranges):
            # Extract owned interior (strip ghosts)
            owned = slab[:, :, :, ov : ov + (x1 - x0)]
            rho, ux, uy, uz = macroscopic3d(owned)
            rho_out[:, :, x0:x1] = rho.cpu()
            ux_out[:, :, x0:x1] = ux.cpu()
            uy_out[:, :, x0:x1] = uy.cpu()
            uz_out[:, :, x0:x1] = uz.cpu()
        return rho_out, ux_out, uy_out, uz_out

    @property
    def n_devices(self) -> int:
        return self.decomp.n_devices


# ---------------------------------------------------------------------------
# Convenience: auto-detect and use all available GPUs
# ---------------------------------------------------------------------------


class D3Q19GlooTransport:
    """One-rank-per-x-slab D3Q19 transport for an initialized Gloo PG.

    ``step`` has production ordering: collision on owned cells, transport of
    all 19 post-collision boundary populations with the two adjacent ranks,
    ghost validation, then pull streaming.  The rank topology is a periodic
    x-ring, so rank zero and rank ``world_size - 1`` are also neighbours.
    Each rank stores only its owned slab and two ghost planes.
    """

    def __init__(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("D3Q19GlooTransport requires an initialized torch.distributed group")
        if dist.get_backend() != "gloo":
            raise RuntimeError("D3Q19GlooTransport requires the Gloo backend")
        if dist.get_world_size() < 2:
            raise RuntimeError("D3Q19GlooTransport requires at least two ranks")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.left_rank = (self.rank - 1) % self.world_size
        self.right_rank = (self.rank + 1) % self.world_size

    @staticmethod
    def _check_owned(f_owned: torch.Tensor) -> None:
        if f_owned.ndim != 4 or f_owned.shape[0] != 19:
            raise ValueError("owned distributions must have shape (19, nz, ny, nx_local)")
        if f_owned.shape[-1] < 1:
            raise ValueError("each rank must own at least one x cell")
        if f_owned.device.type != "cpu":
            raise ValueError("D3Q19GlooTransport is CPU-only")

    @staticmethod
    def _checkpoint_digest(f_owned: torch.Tensor, rank: int, world_size: int, step: int) -> str:
        """Return a SHA-256 digest bound to the checkpoint identity and state."""
        digest = hashlib.sha256()
        digest.update(b"tensorlbm-d3q19-gloo-checkpoint-v1\0")
        digest.update(
            f"{rank}:{world_size}:{step}:{tuple(f_owned.shape)}:{f_owned.dtype}".encode("ascii")
        )
        digest.update(f_owned.detach().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _checkpoint_generation(digests: list[str], world_size: int, step: int) -> str:
        """Bind a checkpoint set to its complete rank payload collection."""
        digest = hashlib.sha256()
        digest.update(b"tensorlbm-d3q19-gloo-generation-v1\0")
        digest.update(f"{world_size}:{step}:".encode("ascii"))
        for rank, member_digest in enumerate(digests):
            digest.update(f"{rank}:{member_digest};".encode("ascii"))
        return digest.hexdigest()

    def save_checkpoint(
        self, checkpoint_dir: os.PathLike[str] | str, f_owned: torch.Tensor, *, step: int
    ) -> None:
        """Save verified rank-local owned state for a later same-world restart.

        All ranks must call this with the same directory.  This is not a
        crash-safe or concurrent checkpoint protocol.
        """
        self._check_owned(f_owned)
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("checkpoint step must be a non-negative integer")
        checkpoint_dir = os.fspath(checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        owned = f_owned.detach().contiguous().clone()
        member_digest = self._checkpoint_digest(owned, self.rank, self.world_size, step)
        gathered_digests: list[object] = [None] * self.world_size
        dist.all_gather_object(gathered_digests, member_digest)
        if not all(isinstance(item, str) for item in gathered_digests):
            raise RuntimeError("D3Q19 checkpoint generation collection failed")
        generation = self._checkpoint_generation(gathered_digests, self.world_size, step)  # type: ignore[arg-type]
        payload = {
            "format": "tensorlbm.d3q19.gloo.rank-local.v1",
            "rank": self.rank,
            "world_size": self.world_size,
            "step": step,
            "generation": generation,
            "owned": owned,
            "digest": member_digest,
        }
        target = os.path.join(checkpoint_dir, f"rank-{self.rank}.pt")
        temporary = target + ".tmp"
        torch.save(payload, temporary)
        os.replace(temporary, target)
        dist.barrier()

    def load_checkpoint(self, checkpoint_dir: os.PathLike[str] | str) -> tuple[torch.Tensor, int]:
        """Load only after validating every member of the checkpoint set."""
        checkpoint_dir = os.fspath(checkpoint_dir)
        payloads: list[dict[object, object]] = []
        expected_keys = {"format", "rank", "world_size", "step", "generation", "owned", "digest"}
        for expected_rank in range(self.world_size):
            target = os.path.join(checkpoint_dir, f"rank-{expected_rank}.pt")
            if not os.path.isfile(target):
                raise RuntimeError(f"D3Q19 checkpoint missing rank-local file: {target}")
            try:
                payload = torch.load(target, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise RuntimeError(f"D3Q19 checkpoint cannot be decoded: {target}") from exc
            if not isinstance(payload, dict) or set(payload) != expected_keys:
                raise RuntimeError("D3Q19 checkpoint has an invalid payload schema")
            owned = payload["owned"]
            if (
                payload["format"] != "tensorlbm.d3q19.gloo.rank-local.v1"
                or payload["rank"] != expected_rank
                or payload["world_size"] != self.world_size
                or not isinstance(payload["step"], int)
                or isinstance(payload["step"], bool)
                or payload["step"] < 0
                or not isinstance(payload["generation"], str)
                or not isinstance(payload["digest"], str)
                or not isinstance(owned, torch.Tensor)
            ):
                raise RuntimeError("D3Q19 checkpoint identity or metadata validation failed")
            try:
                self._check_owned(owned)
            except (TypeError, ValueError) as exc:
                raise RuntimeError("D3Q19 checkpoint owned state validation failed") from exc
            if (
                self._checkpoint_digest(owned, expected_rank, self.world_size, payload["step"])
                != payload["digest"]
            ):
                raise RuntimeError("D3Q19 checkpoint digest validation failed")
            payloads.append(payload)
        steps = {payload["step"] for payload in payloads}
        generations = {payload["generation"] for payload in payloads}
        if len(steps) != 1 or len(generations) != 1:
            raise RuntimeError("D3Q19 checkpoint generation validation failed")
        step = payloads[0]["step"]
        generation = payloads[0]["generation"]
        digests = [payload["digest"] for payload in payloads]
        if generation != self._checkpoint_generation(digests, self.world_size, step):  # type: ignore[arg-type]
            raise RuntimeError("D3Q19 checkpoint generation validation failed")
        owned = payloads[self.rank]["owned"]
        dist.barrier()
        return owned, step  # type: ignore[return-value]

    def load_repartition_checkpoint(
        self,
        checkpoint_dir: os.PathLike[str] | str,
        *,
        source_world_size: int,
    ) -> tuple[torch.Tensor, int]:
        """Validate a complete source set then repartition only owned D3Q19 state.

        Saved and restarting process-group sizes may differ.  Every target rank
        validates every source member before concatenating source-owned slabs in
        source rank order and making a balanced target x-slab assignment.
        """
        if (
            not isinstance(source_world_size, int)
            or isinstance(source_world_size, bool)
            or source_world_size < 2
        ):
            raise ValueError("source_world_size must be an integer of at least two")
        checkpoint_dir = os.fspath(checkpoint_dir)
        payloads: list[dict[object, object]] = []
        expected_keys = {"format", "rank", "world_size", "step", "generation", "owned", "digest"}
        for expected_rank in range(source_world_size):
            target = os.path.join(checkpoint_dir, f"rank-{expected_rank}.pt")
            if not os.path.isfile(target):
                raise RuntimeError(
                    f"D3Q19 repartition checkpoint missing rank-local file: {target}"
                )
            try:
                payload = torch.load(target, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise RuntimeError(
                    f"D3Q19 repartition checkpoint cannot be decoded: {target}"
                ) from exc
            if not isinstance(payload, dict) or set(payload) != expected_keys:
                raise RuntimeError("D3Q19 repartition checkpoint has an invalid payload schema")
            owned = payload["owned"]
            if (
                payload["format"] != "tensorlbm.d3q19.gloo.rank-local.v1"
                or payload["rank"] != expected_rank
                or payload["world_size"] != source_world_size
                or not isinstance(payload["step"], int)
                or isinstance(payload["step"], bool)
                or payload["step"] < 0
                or not isinstance(payload["generation"], str)
                or not isinstance(payload["digest"], str)
                or not isinstance(owned, torch.Tensor)
            ):
                raise RuntimeError(
                    "D3Q19 repartition checkpoint identity or metadata validation failed"
                )
            try:
                self._check_owned(owned)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "D3Q19 repartition checkpoint owned state validation failed"
                ) from exc
            if (
                self._checkpoint_digest(owned, expected_rank, source_world_size, payload["step"])
                != payload["digest"]
            ):
                raise RuntimeError("D3Q19 repartition checkpoint digest validation failed")
            payloads.append(payload)
        steps = {payload["step"] for payload in payloads}
        generations = {payload["generation"] for payload in payloads}
        if len(steps) != 1 or len(generations) != 1:
            raise RuntimeError("D3Q19 repartition checkpoint generation validation failed")
        step = payloads[0]["step"]
        generation = payloads[0]["generation"]
        digests = [payload["digest"] for payload in payloads]
        if generation != self._checkpoint_generation(digests, source_world_size, step):  # type: ignore[arg-type]
            raise RuntimeError("D3Q19 repartition checkpoint generation validation failed")
        full_owned = torch.cat([payload["owned"] for payload in payloads], dim=-1)  # type: ignore[list-item]
        width = full_owned.shape[-1]
        base, remainder = divmod(width, self.world_size)
        x0 = self.rank * base + min(self.rank, remainder)
        x1 = x0 + base + (1 if self.rank < remainder else 0)
        if x0 == x1:
            raise RuntimeError("D3Q19 repartition checkpoint assigns an empty owned slab")
        dist.barrier()
        return full_owned[..., x0:x1].contiguous(), step  # type: ignore[return-value]

    def exchange_ghosts(self, f_owned: torch.Tensor) -> torch.Tensor:
        """Transport all populations' owned boundary planes and pad ghosts."""
        self._check_owned(f_owned)
        # Send the left/right owned planes to their actual neighbours.  This
        # point-to-point ring is independent of slab width and deliberately
        # does not route through all ranks or a process-local list copy.
        left_owned = f_owned[..., 0].contiguous()
        right_owned = f_owned[..., -1].contiguous()
        left_ghost = torch.empty_like(left_owned)
        right_ghost = torch.empty_like(right_owned)
        requests = [
            dist.isend(left_owned, dst=self.left_rank, tag=101),
            dist.isend(right_owned, dst=self.right_rank, tag=102),
            dist.irecv(left_ghost, src=self.left_rank, tag=102),
            dist.irecv(right_ghost, src=self.right_rank, tag=101),
        ]
        for request in requests:
            assert request is not None
            request.wait()
        padded = torch.empty((*f_owned.shape[:-1], f_owned.shape[-1] + 2), dtype=f_owned.dtype)
        padded[..., 0] = left_ghost
        padded[..., 1:-1] = f_owned
        padded[..., -1] = right_ghost
        self.validate_ghosts(padded, (left_ghost, right_ghost))
        return padded

    def validate_ghosts(
        self,
        padded: torch.Tensor,
        peer: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        """Fail closed when either ghost differs from peer-owned data."""
        if padded.ndim != 4 or padded.shape[0] != 19 or padded.shape[-1] < 3:
            raise ValueError("padded distributions must have shape (19, nz, ny, nx_local + 2)")
        if peer is None:
            owned = padded[..., 1:-1]
            left_owned = owned[..., 0].contiguous()
            right_owned = owned[..., -1].contiguous()
            left_ghost = torch.empty_like(left_owned)
            right_ghost = torch.empty_like(right_owned)
            requests = [
                dist.isend(left_owned, dst=self.left_rank, tag=101),
                dist.isend(right_owned, dst=self.right_rank, tag=102),
                dist.irecv(left_ghost, src=self.left_rank, tag=102),
                dist.irecv(right_ghost, src=self.right_rank, tag=101),
            ]
            for request in requests:
                assert request is not None
                request.wait()
            peer = (left_ghost, right_ghost)
        if not torch.equal(padded[..., 0], peer[0]) or not torch.equal(padded[..., -1], peer[1]):
            raise RuntimeError("D3Q19 Gloo ghost validation failed")

    def gather_owned(self, f_owned: torch.Tensor) -> torch.Tensor:
        """Collect variable-width owned slabs in rank order on every rank.

        ``all_gather`` requires identically shaped tensors, so first exchange
        widths and pad only for the collective; the returned tensor contains
        exactly the owned columns, never padding or ghost cells.
        """
        self._check_owned(f_owned)
        width = torch.tensor([f_owned.shape[-1]], dtype=torch.int64)
        widths = [torch.empty_like(width) for _ in range(self.world_size)]
        dist.all_gather(widths, width)
        owned_widths = [int(item.item()) for item in widths]
        max_width = max(owned_widths)
        packed = torch.zeros((*f_owned.shape[:-1], max_width), dtype=f_owned.dtype)
        packed[..., : f_owned.shape[-1]] = f_owned
        gathered = [torch.empty_like(packed) for _ in range(self.world_size)]
        dist.all_gather(gathered, packed)
        full = torch.cat(
            [gathered[rank][..., :rank_width] for rank, rank_width in enumerate(owned_widths)],
            dim=-1,
        )
        return full

    def stream(self, padded: torch.Tensor) -> torch.Tensor:
        """Periodic y/z and ghost-backed x pull stream, returning owned cells."""
        self.validate_ghosts(padded)
        out = torch.empty_like(padded[..., 1:-1])
        for q, (cx, cy, cz) in enumerate(C.tolist()):
            out[q] = torch.roll(padded[q], shifts=(cz, cy, cx), dims=(0, 1, 2))[..., 1:-1]
        return out

    def step(
        self,
        f_owned: torch.Tensor,
        collide_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Execute collision -> Gloo transport -> validation -> stream."""
        self._check_owned(f_owned)
        post_collision = f_owned if collide_fn is None else collide_fn(f_owned)
        self._check_owned(post_collision)
        return self.stream(self.exchange_ghosts(post_collision))


__all__ = [
    "DomainDecomposition",
    "MultiGPUSolver2D",
    "MultiGPUSolver3D",
    "MultiDeviceSolver3D",
    "halo_exchange_2d",
    "halo_exchange_3d",
    "auto_decompose",
    "D3Q19GlooTransport",
]


__all__ = [
    "MultiGPUSolver3D",
    "MultiDeviceSolver3D",
    "D3Q19GlooTransport",
    "halo_exchange_2d",
    "halo_exchange_3d",
    "auto_decompose",
    "build_feq_cache_d3q19",
    "build_feq_cache_d3q27",
    "far_field_bc_cached",
]
