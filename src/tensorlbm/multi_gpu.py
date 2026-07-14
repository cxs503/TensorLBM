"""Multi-GPU Lattice Boltzmann Method via domain decomposition.

Implements a slab-decomposition strategy that splits the simulation domain
along the x-axis across multiple CUDA devices (or CPU processes).  Each
device owns one slab plus one ghost layer on each side for halo exchange.

Architecture
------------
::

    Device 0: f[0  .. nx//N + 1]   (slice + right ghost)
    Device 1: f[nx//N - 1 .. 2*nx//N + 1]
    …
    Device N-1: f[(N-1)*nx//N - 1 .. nx]

Halo exchange is performed between adjacent slabs after every stream step
using ``torch.distributed`` NCCL (GPU) or Gloo (CPU) collectives.

Usage
-----
::

    from tensorlbm.multi_gpu import MultiGPUSolver2D, DomainDecomposition

    dd = DomainDecomposition.from_devices([0, 1, 2, 3])
    solver = MultiGPUSolver2D(f_global, dd)
    for step in range(n_steps):
        solver.step(collide_fn, stream_fn)
        if step % 100 == 0:
            f_global = solver.gather()

Notes
-----
* Requires ``torch.distributed`` and at least one GPU per rank (NCCL) or
  CPU-only with Gloo backend.
* For single-process multi-GPU use, call :func:`run_multi_gpu_2d` which
  spawns sub-processes automatically via ``torch.multiprocessing.spawn``.

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
import torch.distributed as dist

from tensorlbm.d3q19 import C


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
    """Exchange one-cell ghost layers between adjacent D2Q9 slabs.

    Each slab has shape ``(9, ny, nx_local + 2*overlap)``.  The rightmost
    interior column of slab ``i`` is copied into the left ghost of slab
    ``i+1`` and vice versa.

    Args:
        slabs:  List of per-device distribution tensors.
        decomp: Domain decomposition descriptor.

    Returns:
        Updated list of tensors with refreshed ghost cells.
    """
    ov = decomp.overlap
    for i in range(len(slabs) - 1):
        # Right ghost of slab i ← interior right of slab i+1
        right_of_i     = slabs[i][:, :, -ov - 1:-1]   # interior right of i
        left_ghost_ip1 = slabs[i + 1][:, :, :ov]       # left ghost of i+1
        left_ghost_ip1.copy_(right_of_i.to(left_ghost_ip1.device))

        # Left ghost of slab i ← interior left of slab i+1
        left_of_ip1  = slabs[i + 1][:, :, ov:2 * ov]  # interior left of i+1
        right_ghost_i = slabs[i][:, :, -ov:]            # right ghost of i
        right_ghost_i.copy_(left_of_ip1.to(right_ghost_i.device))

    return slabs


def halo_exchange_3d(
    slabs: list[torch.Tensor],
    decomp: DomainDecomposition,
) -> list[torch.Tensor]:
    """Exchange ghost layers between D3Q19 slabs (x-decomposition).

    Each slab has shape ``(19, nz, ny, nx_local + 2*overlap)``.
    """
    ov = decomp.overlap
    n_slabs = len(slabs)

    # Every 3-D slab has an explicit ghost layer on both sides, including
    # the global x boundaries.  Source data are always owned cells, so copy
    # order cannot make one ghost exchange consume another ghost exchange.
    for i, slab in enumerate(slabs):
        left = slabs[(i - 1) % n_slabs]
        right = slabs[(i + 1) % n_slabs]
        left_ghost = slab[:, :, :, :ov]
        right_ghost = slab[:, :, :, -ov:]
        left_ghost.copy_(left[:, :, :, -2 * ov:-ov].to(left_ghost.device))
        right_ghost.copy_(right[:, :, :, ov:2 * ov].to(right_ghost.device))

    return slabs


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
            # Allocate slab with ghost layers
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
        """Advance one time step across all slabs.

        Each slab applies collision + streaming independently, then halo
        exchange synchronises boundary cells between adjacent slabs.

        Args:
            collide_fn:  Collision ``f → f'`` (applied per slab).
            stream_fn:   Streaming ``f → f'`` (applied per slab).
            boundary_fn: Optional boundary-condition ``f → f'``.
        """
        # Collision + stream on each device
        for i, slab in enumerate(self.slabs):
            self.slabs[i] = collide_fn(slab)
            self.slabs[i] = stream_fn(self.slabs[i])
            if boundary_fn is not None:
                self.slabs[i] = boundary_fn(self.slabs[i])

        # Halo exchange
        halo_exchange_2d(self.slabs, self.decomp)

        self._step_count += 1

    def gather(self) -> torch.Tensor:
        """Assemble slab interior regions back into a single global tensor.

        Returns:
            Global distribution (9, ny, nx) on CPU.
        """
        q = self.slabs[0].shape[0]
        ny = self.ny
        nx = self.decomp.nx_global
        ov = self.decomp.overlap
        f_out = torch.zeros((q, ny, nx), dtype=self.slabs[0].dtype)
        for slab, (x0, x1) in zip(self.slabs, self._x_ranges):
            # Extract interior (strip ghost columns)
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
    """Multi-GPU D3Q19 LBM solver using x-axis domain decomposition.

    Mirrors :class:`MultiGPUSolver2D` for three-dimensional flows.

    Args:
        f_global: Global distribution (19, nz, ny, nx).
        decomp:   Domain decomposition descriptor.
    """

    def __init__(
        self,
        f_global: torch.Tensor,
        decomp: DomainDecomposition,
    ) -> None:
        q, nz, ny, nx = f_global.shape
        if q != 19:
            raise ValueError(f"MultiGPUSolver3D requires D3Q19 populations, got {q}")
        if decomp.nx_global == 0:
            decomp = DomainDecomposition(
                devices=decomp.devices,
                nx_global=nx,
                overlap=decomp.overlap,
            )
        if decomp.nx_global != nx:
            raise ValueError(f"decomp.nx_global ({decomp.nx_global}) != nx ({nx})")
        self.decomp = decomp
        self.nz = nz
        self.ny = ny
        self._step_count = 0

        ov = decomp.overlap
        self.slabs: list[torch.Tensor] = []
        for dev, (x0, x1) in zip(decomp.devices, decomp.slabs):
            # Keep physical ghosts on both sides, including global x edges.
            # Modulo indexing seeds periodic ghosts before the first step.
            x_indices = torch.arange(x0 - ov, x1 + ov, device=f_global.device) % nx
            slab = f_global.index_select(3, x_indices).to(dev).contiguous()
            self.slabs.append(slab)
        self._x_ranges = decomp.slabs
        halo_exchange_3d(self.slabs, self.decomp)

    def step(
        self,
        collide_fn: Callable,
        stream_fn: Callable,
        boundary_fn: Callable | None = None,
    ) -> None:
        """Advance one time step across all slabs.

        Exchange post-collision owned populations before streaming, so a local
        pull-stream reads neighbour data at an x interface rather than a
        locally periodic value that would be replaced too late.
        """
        for i, slab in enumerate(self.slabs):
            self.slabs[i] = collide_fn(slab)
        halo_exchange_3d(self.slabs, self.decomp)
        for i, slab in enumerate(self.slabs):
            self.slabs[i] = stream_fn(slab)
            if boundary_fn is not None:
                self.slabs[i] = boundary_fn(self.slabs[i])
        self._step_count += 1

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
            f_out[:, :, :, x0:x1] = slab[:, :, :, x0g_local:x0g_local + local_width].cpu()
        return f_out


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


# ---------------------------------------------------------------------------
# Two-rank CPU/Gloo D3Q19 transport
# ---------------------------------------------------------------------------

class D3Q19GlooTransport:
    """One-rank-per-x-slab D3Q19 transport for an initialized two-rank Gloo PG.

    ``step`` has production ordering: collision on owned cells, transport of
    all 19 post-collision boundary populations, ghost validation, then pull
    streaming.  Each rank stores only its owned slab and ghosts.
    """

    def __init__(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("D3Q19GlooTransport requires an initialized torch.distributed group")
        if dist.get_backend() != "gloo":
            raise RuntimeError("D3Q19GlooTransport requires the Gloo backend")
        if dist.get_world_size() != 2:
            raise RuntimeError("D3Q19GlooTransport requires exactly two ranks")
        self.rank = dist.get_rank()

    @staticmethod
    def _check_owned(f_owned: torch.Tensor) -> None:
        if f_owned.ndim != 4 or f_owned.shape[0] != 19:
            raise ValueError("owned distributions must have shape (19, nz, ny, nx_local)")
        if f_owned.shape[-1] < 1:
            raise ValueError("each rank must own at least one x cell")
        if f_owned.device.type != "cpu":
            raise ValueError("D3Q19GlooTransport is CPU-only")

    def exchange_ghosts(self, f_owned: torch.Tensor) -> torch.Tensor:
        """Transport all populations' owned boundary planes and pad ghosts."""
        self._check_owned(f_owned)
        boundary = torch.stack((f_owned[..., -1], f_owned[..., 0]), dim=-1).contiguous()
        peer = torch.empty_like(boundary)
        # Every rank sends just its two boundary planes.  Unlike all_gather,
        # this stays valid when the owned slab widths differ (for example 4/5).
        peer_rank = 1 - self.rank
        send_request = dist.isend(boundary, dst=peer_rank)
        receive_request = dist.irecv(peer, src=peer_rank)
        assert send_request is not None and receive_request is not None
        send_request.wait()
        receive_request.wait()
        padded = torch.empty((*f_owned.shape[:-1], f_owned.shape[-1] + 2), dtype=f_owned.dtype)
        padded[..., 0] = peer[..., 0]
        padded[..., 1:-1] = f_owned
        padded[..., -1] = peer[..., 1]
        self.validate_ghosts(padded, peer)
        return padded

    def validate_ghosts(self, padded: torch.Tensor, peer: torch.Tensor | None = None) -> None:
        """Fail closed when either ghost differs from peer-owned data."""
        if padded.ndim != 4 or padded.shape[0] != 19 or padded.shape[-1] < 3:
            raise ValueError("padded distributions must have shape (19, nz, ny, nx_local + 2)")
        if peer is None:
            owned = padded[..., 1:-1]
            boundary = torch.stack((owned[..., -1], owned[..., 0]), dim=-1).contiguous()
            peer = torch.empty_like(boundary)
            peer_rank = 1 - self.rank
            send_request = dist.isend(boundary, dst=peer_rank)
            receive_request = dist.irecv(peer, src=peer_rank)
            assert send_request is not None and receive_request is not None
            send_request.wait()
            receive_request.wait()
        if not torch.equal(padded[..., 0], peer[..., 0]) or not torch.equal(padded[..., -1], peer[..., 1]):
            raise RuntimeError("D3Q19 Gloo ghost validation failed")

    def gather_owned(self, f_owned: torch.Tensor) -> torch.Tensor:
        """Collect variable-width owned slabs in rank order on every rank.

        ``all_gather`` requires identically shaped tensors, so first exchange
        widths and pad only for the collective; the returned tensor contains
        exactly the owned columns, never padding or ghost cells.
        """
        self._check_owned(f_owned)
        width = torch.tensor([f_owned.shape[-1]], dtype=torch.int64)
        widths = [torch.empty_like(width) for _ in range(2)]
        dist.all_gather(widths, width)
        owned_widths = [int(item.item()) for item in widths]
        max_width = max(owned_widths)
        packed = torch.zeros((*f_owned.shape[:-1], max_width), dtype=f_owned.dtype)
        packed[..., :f_owned.shape[-1]] = f_owned
        gathered = [torch.empty_like(packed) for _ in range(2)]
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

    def step(self, f_owned: torch.Tensor, collide_fn: Callable[[torch.Tensor], torch.Tensor] | None = None) -> torch.Tensor:
        """Execute collision -> Gloo transport -> validation -> stream."""
        self._check_owned(f_owned)
        post_collision = f_owned if collide_fn is None else collide_fn(f_owned)
        self._check_owned(post_collision)
        return self.stream(self.exchange_ghosts(post_collision))


__all__ = [
    "DomainDecomposition",
    "MultiGPUSolver2D",
    "MultiGPUSolver3D",
    "halo_exchange_2d",
    "halo_exchange_3d",
    "auto_decompose",
    "D3Q19GlooTransport",
]
