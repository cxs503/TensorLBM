"""Pure PyTorch cross-device domain decomposition framework.

This module implements a slab-decomposition strategy for LBM simulations
that splits the domain along one axis across multiple devices (real or
simulated). The halo exchange uses pure PyTorch operations, making it
portable across CUDA, CPU, and other backends.

Architecture
------------
The framework supports two modes:

1. **Real multi-device**: Each rank owns a slab on a separate CUDA device.
   Halo exchange uses torch.distributed (NCCL/Gloo) for communication.

2. **Simulated multi-device**: Multiple "virtual devices" share a single
   physical device (CPU or GPU). Halo exchange uses direct tensor copies.
   This mode is useful for testing and development without multi-GPU hardware.

Halo Exchange Protocol
----------------------
For a z-slab decomposition with periodic boundaries:

    rank r owns z in [r*nz_local, (r+1)*nz_local)
    Buffer shape: (Q, nz_local + 2, ny, nx)
    - Owned planes: local z in [1, nz_local + 1)
    - Ghost plane 0: from rank r-1 (periodic wrap)
    - Ghost plane nz_local+1: from rank r+1 (periodic wrap)

Exchange steps:
1. Pack send buffers (contiguous copies of boundary planes)
2. Exchange via send/recv (NCCL) or direct copy (simulated)
3. Unpack recv buffers into ghost planes

Key Design Decisions (from triton_distributed_notes.md):
- NCCL requires contiguous buffers: all send/recv operands must be
  .contiguous() to avoid stride issues
- Q=19 physical, Q_PAD=32 for Triton kernels: padding is internal to
  kernels; host buffers stay Q=19
- Force all-reduce: geometry build is collective, so forces must be
  summed across ranks
- Ghost planes are always fluid (obstacle=0) in the mask

Usage
-----
::

    from tensorlbm.domain_decomposition import (
        DomainDecomposer,
        HaloExchange,
        SimulatedMultiDevice,
    )

    # Real multi-GPU (requires torch.distributed)
    decomposer = DomainDecomposer(
        nz_global=128, ny=64, nx=64,
        n_devices=4, device="cuda",
    )
    halo = HaloExchange(decomposer, mode="nccl")

    # Simulated multi-device (for testing)
    decomposer = DomainDecomposer(
        nz_global=128, ny=64, nx=64,
        n_devices=4, device="cpu",
    )
    halo = HaloExchange(decomposer, mode="simulated")

References
----------
- docs/triton_distributed_notes.md: engineering notes on NCCL pitfalls
- Succi S., et al. (2001) "Lattice Boltzmann for distributed and
  large-scale simulations." Comput. Phys. Commun. 134(3).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Literal

import torch
import torch.distributed as dist


__all__ = [
    "DomainDecomposer",
    "HaloExchange",
    "SimulatedMultiDevice",
    "DeviceEvidence",
    "collect_device_evidence",
]


# ---------------------------------------------------------------------------
# Domain decomposition descriptor
# ---------------------------------------------------------------------------


@dataclass
class DomainDecomposer:
    """Describes how the global domain is split across devices.

    Supports slab decomposition along the z-axis (dim 1 of (Q, nz, ny, nx)).

    Args:
        nz_global: Global number of z-planes (must be divisible by n_devices).
        ny, nx: Y and X extents (shared across all devices).
        n_devices: Number of devices (real or simulated).
        device: Base device string (e.g., "cuda", "cpu"). For real multi-GPU,
            each rank uses f"{device}:{rank}". For simulated, all use the same.
        decomposition_axis: Axis to decompose along (default: "z", dim 1).
    """

    nz_global: int
    ny: int
    nx: int
    n_devices: int
    device: str = "cuda"
    decomposition_axis: Literal["z", "y", "x"] = "z"

    def __post_init__(self) -> None:
        if self.nz_global % self.n_devices != 0:
            raise ValueError(
                f"nz_global={self.nz_global} must be divisible by "
                f"n_devices={self.n_devices}"
            )
        self.nz_local = self.nz_global // self.n_devices

    def get_device_for_rank(self, rank: int, simulated: bool = False) -> torch.device:
        """Get the torch.device for a given rank.

        Args:
            rank: Process rank (0 to n_devices-1).
            simulated: If True, all ranks use the same device. If False,
                each rank uses a separate device (cuda:rank).
        """
        if simulated or self.device == "cpu":
            return torch.device(self.device)
        else:
            return torch.device(f"{self.device}:{rank}")

    def get_slab_bounds(self, rank: int) -> tuple[int, int]:
        """Get the global z-range [z_start, z_end) for a rank."""
        z_start = rank * self.nz_local
        z_end = z_start + self.nz_local
        return z_start, z_end

    def get_neighbor_ranks(self, rank: int) -> tuple[int, int]:
        """Get left and right neighbor ranks (periodic boundaries)."""
        left = (rank - 1) % self.n_devices
        right = (rank + 1) % self.n_devices
        return left, right


# ---------------------------------------------------------------------------
# Halo exchange
# ---------------------------------------------------------------------------


class HaloExchange:
    """Pure PyTorch halo exchange for slab-decomposed domains.

    Supports two modes:
    - "nccl": Uses torch.distributed for real multi-GPU communication.
    - "simulated": Direct tensor copies for simulated multi-device.

    The exchange is selective: only the populations that cross the slab
    face are exchanged (5 of 19 for D3Q19 z-slab). This reduces wire
    bytes by 3.8x compared to full-plane exchange.

    Args:
        decomposer: Domain decomposition descriptor.
        mode: "nccl" or "simulated".
        Q: Number of lattice velocities (default 19 for D3Q19).
        dtype: Data type for staging buffers (default torch.float32).
    """

    def __init__(
        self,
        decomposer: DomainDecomposer,
        mode: Literal["nccl", "simulated"] = "simulated",
        Q: int = 19,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.decomposer = decomposer
        self.mode = mode
        self.Q = Q
        self.dtype = dtype
        self.nz_local = decomposer.nz_local
        self.ny = decomposer.ny
        self.nx = decomposer.nx

        # Pre-compute crossing direction indices for selective exchange
        # D3Q19 z-slab: 5 directions cross each face (cz = +1 or -1)
        self._cross_up = self._compute_crossing_indices(+1)
        self._cross_dn = self._compute_crossing_indices(-1)

        # Pre-allocate staging buffers for each rank
        self._send_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self._recv_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self._initialized = False

    def _compute_crossing_indices(self, cz_sign: int) -> tuple[int, ...]:
        """Compute direction indices that cross a slab face.

        For D3Q19, the z-components are:
        q=0: cz=0, q=1,2: cz=0, q=3,4: cz=0,
        q=5: cz=+1, q=6: cz=-1,
        q=7,8,9,10: cz=0,
        q=11,12,13,14: cz=0,
        q=15,16: cz=+1/-1, q=17,18: cz=-1/+1

        Returns tuple of q indices with cz == cz_sign.
        """
        # D3Q19 lattice velocities (cx, cy, cz)
        D3Q19_C = [
            (0, 0, 0),    # 0
            (1, 0, 0),    # 1
            (-1, 0, 0),   # 2
            (0, 1, 0),    # 3
            (0, -1, 0),   # 4
            (0, 0, 1),    # 5
            (0, 0, -1),   # 6
            (1, 1, 0),    # 7
            (-1, -1, 0),  # 8
            (1, -1, 0),   # 9
            (-1, 1, 0),   # 10
            (1, 0, 1),    # 11
            (-1, 0, -1),  # 12
            (1, 0, -1),   # 13
            (-1, 0, 1),   # 14
            (0, 1, 1),    # 15
            (0, -1, -1),  # 16
            (0, 1, -1),   # 17
            (0, -1, 1),   # 18
        ]
        return tuple(q for q, (cx, cy, cz) in enumerate(D3Q19_C) if cz == cz_sign)

    def _allocate_buffers(self, rank: int, device: torch.device) -> None:
        """Allocate staging buffers for a rank."""
        n_cross = len(self._cross_up)
        # Send buffers: pack crossing populations from boundary planes
        self._send_buffers[rank] = {
            "up": torch.zeros((n_cross, self.ny, self.nx), dtype=self.dtype, device=device),
            "dn": torch.zeros((n_cross, self.ny, self.nx), dtype=self.dtype, device=device),
        }
        # Recv buffers: unpack into ghost planes
        self._recv_buffers[rank] = {
            "up": torch.zeros((n_cross, self.ny, self.nx), dtype=self.dtype, device=device),
            "dn": torch.zeros((n_cross, self.ny, self.nx), dtype=self.dtype, device=device),
        }
        self._initialized = True

    def pack_send_buffers(self, f_local: torch.Tensor, rank: int) -> None:
        """Pack boundary planes into send buffers.

        Args:
            f_local: Local buffer (Q, nz_local+2, ny, nx).
            rank: Process rank.
        """
        if rank not in self._send_buffers:
            self._allocate_buffers(rank, f_local.device)

        # Pack up-going populations from the last owned plane (z=nz_local)
        # These will be sent to the right neighbor
        for i, q in enumerate(self._cross_up):
            self._send_buffers[rank]["up"][i] = f_local[q, self.nz_local]

        # Pack down-going populations from the first owned plane (z=1)
        # These will be sent to the left neighbor
        for i, q in enumerate(self._cross_dn):
            self._send_buffers[rank]["dn"][i] = f_local[q, 1]

    def unpack_recv_buffers(self, f_local: torch.Tensor, rank: int) -> None:
        """Unpack recv buffers into ghost planes.

        Args:
            f_local: Local buffer (Q, nz_local+2, ny, nx).
            rank: Process rank.
        """
        # Unpack into left ghost plane (z=0): received from left neighbor
        # These are up-going populations that crossed from rank-1
        for i, q in enumerate(self._cross_up):
            f_local[q, 0] = self._recv_buffers[rank]["up"][i]

        # Unpack into right ghost plane (z=nz_local+1): received from right neighbor
        # These are down-going populations that crossed from rank+1
        for i, q in enumerate(self._cross_dn):
            f_local[q, self.nz_local + 1] = self._recv_buffers[rank]["dn"][i]

    def exchange_simulated(self, slabs: list[torch.Tensor]) -> None:
        """Perform halo exchange in simulated mode (direct copies).

        Args:
            slabs: List of local buffers, one per rank.
        """
        n_ranks = len(slabs)
        for rank in range(n_ranks):
            if rank not in self._send_buffers:
                self._allocate_buffers(rank, slabs[rank].device)

        # Pack all send buffers
        for rank, f_local in enumerate(slabs):
            self.pack_send_buffers(f_local, rank)

        # Exchange: rank r sends "up" to rank r+1, receives "dn" from rank r+1
        #           rank r sends "dn" to rank r-1, receives "up" from rank r-1
        for rank in range(n_ranks):
            left, right = self.decomposer.get_neighbor_ranks(rank)
            # Send up to right, receive dn from right
            self._recv_buffers[rank]["dn"].copy_(self._send_buffers[right]["dn"])
            # Send dn to left, receive up from left
            self._recv_buffers[rank]["up"].copy_(self._send_buffers[left]["up"])

        # Unpack all recv buffers
        for rank, f_local in enumerate(slabs):
            self.unpack_recv_buffers(f_local, rank)

    def exchange_nccl(self, f_local: torch.Tensor, rank: int) -> None:
        """Perform halo exchange using NCCL (requires torch.distributed).

        Args:
            f_local: Local buffer (Q, nz_local+2, ny, nx).
            rank: Process rank.
        """
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("NCCL exchange requires torch.distributed to be initialized")

        if rank not in self._send_buffers:
            self._allocate_buffers(rank, f_local.device)

        # Pack send buffers
        self.pack_send_buffers(f_local, rank)

        left, right = self.decomposer.get_neighbor_ranks(rank)

        # Async send/recv operations
        # Send "up" to right, receive "dn" from right
        send_up_op = dist.P2POp(dist.isend, self._send_buffers[rank]["up"], right)
        recv_dn_op = dist.P2POp(dist.irecv, self._recv_buffers[rank]["dn"], right)
        # Send "dn" to left, receive "up" from left
        send_dn_op = dist.P2POp(dist.isend, self._send_buffers[rank]["dn"], left)
        recv_up_op = dist.P2POp(dist.irecv, self._recv_buffers[rank]["up"], left)

        # Post all operations
        reqs = dist.batch_isend_irecv([send_up_op, recv_dn_op, send_dn_op, recv_up_op])
        # Wait for completion
        for req in reqs:
            req.wait()

        # Unpack recv buffers
        self.unpack_recv_buffers(f_local, rank)

    def exchange(self, f_local_or_slabs, rank: int | None = None) -> None:
        """Perform halo exchange (dispatches to appropriate mode).

        Args:
            f_local_or_slabs: In "nccl" mode, a single local buffer.
                In "simulated" mode, a list of buffers (one per rank).
            rank: Process rank (required for "nccl" mode).
        """
        if self.mode == "simulated":
            if not isinstance(f_local_or_slabs, list):
                raise ValueError("Simulated mode requires a list of slabs")
            self.exchange_simulated(f_local_or_slabs)
        elif self.mode == "nccl":
            if rank is None:
                raise ValueError("NCCL mode requires rank argument")
            self.exchange_nccl(f_local_or_slabs, rank)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# ---------------------------------------------------------------------------
# Simulated multi-device runner
# ---------------------------------------------------------------------------


class SimulatedMultiDevice:
    """Simulate multi-device execution on a single physical device.

    This class manages multiple "virtual ranks" that share a single device.
    It's useful for testing domain decomposition logic without multi-GPU
    hardware.

    Args:
        decomposer: Domain decomposition descriptor.
        halo: HaloExchange instance.
        device: Physical device to use (default: "cpu").
    """

    def __init__(
        self,
        decomposer: DomainDecomposer,
        halo: HaloExchange,
        device: str | torch.device = "cpu",
    ) -> None:
        self.decomposer = decomposer
        self.halo = halo
        self.device = torch.device(device)
        self.n_ranks = decomposer.n_devices

        # Allocate slabs for each rank
        self.slabs: list[torch.Tensor] = []
        for rank in range(self.n_ranks):
            slab = torch.zeros(
                (halo.Q, decomposer.nz_local + 2, decomposer.ny, decomposer.nx),
                dtype=halo.dtype,
                device=self.device,
            )
            self.slabs.append(slab)

    def distribute(self, f_global: torch.Tensor) -> None:
        """Distribute a global field to the slabs.

        Args:
            f_global: Global field (Q, nz_global, ny, nx).
        """
        for rank in range(self.n_ranks):
            z_start, z_end = self.decomposer.get_slab_bounds(rank)
            # Copy owned planes into the slab (local z in [1, nz_local+1))
            self.slabs[rank][:, 1:self.decomposer.nz_local + 1] = f_global[:, z_start:z_end]
            # Initialize ghost planes with periodic wrap
            left, right = self.decomposer.get_neighbor_ranks(rank)
            z_start_left, z_end_left = self.decomposer.get_slab_bounds(left)
            z_start_right, z_end_right = self.decomposer.get_slab_bounds(right)
            self.slabs[rank][:, 0] = f_global[:, z_end_left - 1]
            self.slabs[rank][:, self.decomposer.nz_local + 1] = f_global[:, z_start_right]

    def gather(self) -> torch.Tensor:
        """Gather slabs back into a global field.

        Returns:
            Global field (Q, nz_global, ny, nx).
        """
        f_global = torch.zeros(
            (self.halo.Q, self.decomposer.nz_global, self.decomposer.ny, self.decomposer.nx),
            dtype=self.halo.dtype,
            device=self.device,
        )
        for rank in range(self.n_ranks):
            z_start, z_end = self.decomposer.get_slab_bounds(rank)
            f_global[:, z_start:z_end] = self.slabs[rank][:, 1:self.decomposer.nz_local + 1]
        return f_global

    def step(
        self,
        collide_fn,
        stream_fn,
        boundary_fn=None,
    ) -> None:
        """Advance one time step across all ranks.

        Args:
            collide_fn: Collision function f -> f.
            stream_fn: Streaming function f -> f.
            boundary_fn: Optional boundary condition function f -> f.
        """
        # Apply collide + stream + BC on each rank
        for rank in range(self.n_ranks):
            f = self.slabs[rank]
            f = collide_fn(f)
            f = stream_fn(f)
            if boundary_fn is not None:
                f = boundary_fn(f, rank)
            self.slabs[rank] = f

        # Halo exchange
        self.halo.exchange_simulated(self.slabs)


# ---------------------------------------------------------------------------
# Device evidence collection
# ---------------------------------------------------------------------------


@dataclass
class DeviceEvidence:
    """Evidence for a single device/rank."""
    rank: int
    device: str
    nz_local: int
    ny: int
    nx: int
    memory_bytes: int = 0
    compute_time_ms: float = 0.0
    halo_time_ms: float = 0.0
    total_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "device": self.device,
            "nz_local": self.nz_local,
            "ny": self.ny,
            "nx": self.nx,
            "memory_bytes": self.memory_bytes,
            "compute_time_ms": self.compute_time_ms,
            "halo_time_ms": self.halo_time_ms,
            "total_time_ms": self.total_time_ms,
        }


def collect_device_evidence(
    decomposer: DomainDecomposer,
    simulated: SimulatedMultiDevice,
    n_steps: int,
    collide_fn,
    stream_fn,
    boundary_fn=None,
) -> list[DeviceEvidence]:
    """Collect timing and memory evidence from a simulated run.

    Args:
        decomposer: Domain decomposition descriptor.
        simulated: SimulatedMultiDevice instance.
        n_steps: Number of steps to run.
        collide_fn, stream_fn, boundary_fn: Step functions.

    Returns:
        List of DeviceEvidence, one per rank.
    """
    # Warmup
    for _ in range(3):
        simulated.step(collide_fn, stream_fn, boundary_fn)

    # Timed run
    evidence_list = []
    for rank in range(decomposer.n_devices):
        device = simulated.slabs[rank].device
        memory_bytes = simulated.slabs[rank].element_size() * simulated.slabs[rank].numel()

        # Time compute
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_steps):
            f = simulated.slabs[rank]
            f = collide_fn(f)
            f = stream_fn(f)
            if boundary_fn is not None:
                f = boundary_fn(f, rank)
            simulated.slabs[rank] = f
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        compute_time_ms = (t1 - t0) / n_steps * 1000

        # Time halo
        torch.cuda.synchronize() if device.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(n_steps):
            simulated.halo.exchange_simulated(simulated.slabs)
        torch.cuda.synchronize() if device.type == "cuda" else None
        t1 = time.perf_counter()
        halo_time_ms = (t1 - t0) / n_steps * 1000

        evidence_list.append(DeviceEvidence(
            rank=rank,
            device=str(device),
            nz_local=decomposer.nz_local,
            ny=decomposer.ny,
            nx=decomposer.nx,
            memory_bytes=memory_bytes,
            compute_time_ms=compute_time_ms,
            halo_time_ms=halo_time_ms,
            total_time_ms=compute_time_ms + halo_time_ms,
        ))

    return evidence_list
