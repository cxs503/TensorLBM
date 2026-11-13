"""Host-staged halo exchange for single-node multi-device LBM.

This module is the *subject* implementation of the cross-validation mission
``XVAL-LBM-SINGLE-NODE-MULTI-GPU-HOST-STAGED-HALO``: it ports the transport
scheme of ProjectPhysX/FluidX3D ("one slab per device, halo staged through
host memory, host buffers swapped instead of copied") onto TensorLBM tensor
semantics, so the scheme can be measured against the reference instead of
being described in prose.

Reference facts (read from the FluidX3D source tree; ``/tmp/fx3d`` checkout of
``ProjectPhysX/FluidX3D`` master, files ``lbm.cpp``/``kernel.cpp``)::

    lbm.cpp:721-733     decomposition: global grid trimmed to N/D multiples;
                        local extent N/D + 2*(D>1); offset x*N/D - H
    lbm.cpp:1283-1287   get_area(direction) = face area of the *local* domain
                        ({Ny*Nz, Nz*Nx, Nx*Ny}), i.e. halo rows included
    lbm.cpp:1305-1316   allocate_transfer: two host staging buffers p/m of
                        Amax x max(transfers*sizeof(fpxx), 17) bytes, Amax =
                        largest decomposed face area
    lbm.cpp:1343-1353   enqueue_transfer_extract/insert_field: one selective
                        in-device copy + one device<->host copy per side
    lbm.cpp:1355-1382   communicate_field: per direction extract-all ->
                        queue barrier -> host pointer swaps -> insert-all
    kernel.cpp:904-931  index_transfer_data (D3Q19): exactly 5 crossing
                        distributions per face/side
    kernel.cpp:2091-2137 index_extract_p/m, index_insert_p/m: extract from
                        layers 1 / N-2, insert into layers 0 / N-1

The port keeps the reference's semantics (selective 5-population faces, one
staging buffer pair per device, per-direction barrier, zero-copy host pointer
swap) and replaces the OpenCL calls with PyTorch tensor views, so the
transport can be validated on a CPU-only node as well as on CUDA devices.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

__all__ = [
    "D3Q19_C",
    "D3Q19_W",
    "FLUIDX3D_D3Q19_VELOCITY_SET",
    "FLUIDX3D_D3Q19_INDEX_TRANSFER",
    "crossing_indices",
    "HostStagedLayout",
    "HostStagedHalo",
    "HostStagedD3Q19",
]

# ---------------------------------------------------------------------------
# Velocity set (TensorLBM ordering) and the reference transfer index table
# ---------------------------------------------------------------------------

#: D3Q19 lattice velocities, TensorLBM ordering (matches
#: ``domain_decomposition.HaloExchange``).
D3Q19_C: tuple[tuple[int, int, int], ...] = (
    (0, 0, 0),  # 0
    (1, 0, 0),  # 1  x+
    (-1, 0, 0),  # 2  x-
    (0, 1, 0),  # 3  y+
    (0, -1, 0),  # 4  y-
    (0, 0, 1),  # 5  z+
    (0, 0, -1),  # 6  z-
    (1, 1, 0),  # 7
    (-1, -1, 0),  # 8
    (1, -1, 0),  # 9
    (-1, 1, 0),  # 10
    (1, 0, 1),  # 11
    (-1, 0, -1),  # 12
    (1, 0, -1),  # 13
    (-1, 0, 1),  # 14
    (0, 1, 1),  # 15
    (0, -1, -1),  # 16
    (0, 1, -1),  # 17
    (0, -1, 1),  # 18
)

D3Q19_W: tuple[float, ...] = (
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
)

#: FluidX3D ``D3Q19`` velocity set, copied verbatim from ``kernel.cpp:871-877``
#: (three rows: x, y, z components; index = storage index).
FLUIDX3D_D3Q19_VELOCITY_SET: tuple[tuple[int, int, int], ...] = tuple(
    zip(
        (0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0),
        (0, 0, 0, 1, -1, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 1, -1),
        (0, 0, 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, 0, 0, -1, 1, -1, 1),
    )
)

#: FluidX3D ``index_transfer_data`` for ``D3Q19``, copied verbatim from
#: ``kernel.cpp:917-923``: ``(axis, side) -> storage indices`` with ``axis``
#: 0=x, 1=y, 2=z and ``side`` ``"p"``/``"m"`` = plus/minus.
FLUIDX3D_D3Q19_INDEX_TRANSFER: dict[tuple[int, str], tuple[int, ...]] = {
    (0, "p"): (1, 7, 13, 9, 15),
    (0, "m"): (2, 8, 14, 10, 16),
    (1, "p"): (3, 7, 14, 11, 17),
    (1, "m"): (4, 8, 13, 12, 18),
    (2, "p"): (5, 9, 16, 11, 18),
    (2, "m"): (6, 10, 15, 12, 17),
}

#: Number of distributions per face cell in the reference (5 for D3Q19).
REF_TRANSFERS_PER_FACE: int = 5


def crossing_indices(axis: int, sign: int) -> tuple[int, ...]:
    """Indices ``q`` with ``c_q[axis] == sign`` in the TensorLBM velocity set."""
    return tuple(q for q, c in enumerate(D3Q19_C) if c[axis] == sign)


# ---------------------------------------------------------------------------
# Decomposition layout (reference: lbm.cpp:721-733, 1283-1287)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HostStagedLayout:
    """Device grid ``Dx*Dy*Dz`` with the reference's trimming and ghost convention.

    The global grid is trimmed down to a multiple of the device counts (the
    reference warns and trims, ``lbm.cpp:722-724``).  Every *decomposed* axis
    gets one ghost layer per side, so its local extent is
    ``n//D + 2*(D>1)`` (``lbm.cpp:733``); undecomposed axes keep extent ``n``.
    Device ``d = x + (y + z*Dy)*Dx`` (``lbm.cpp:732``).

    ``nx/ny/nz`` are in coordinate order; local slabs are ``(Q, z, y, x)``.
    """

    nx: int
    ny: int
    nz: int
    dx: int = 1
    dy: int = 1
    dz: int = 1

    def __post_init__(self) -> None:
        if min(self.dx, self.dy, self.dz) < 1:
            raise ValueError("device counts must be >= 1")
        for name, n, d in (("nx", self.nx, self.dx), ("ny", self.ny, self.dy), ("nz", self.nz, self.dz)):
            if n < d:
                raise ValueError(f"global {name}={n} smaller than device count {d}")

    # -- trimmed global resolution ---------------------------------------
    @property
    def gnx(self) -> int:
        return (self.nx // self.dx) * self.dx

    @property
    def gny(self) -> int:
        return (self.ny // self.dy) * self.dy

    @property
    def gnz(self) -> int:
        return (self.nz // self.dz) * self.dz

    @property
    def trimmed(self) -> bool:
        return (self.gnx, self.gny, self.gnz) != (self.nx, self.ny, self.nz)

    # -- per-axis helper quantities ---------------------------------------
    @property
    def halo(self) -> tuple[int, int, int]:
        """``(Hx, Hy, Hz)`` ghost offsets per coordinate axis (``lbm.cpp:727``)."""
        return (int(self.dx > 1), int(self.dy > 1), int(self.dz > 1))

    @property
    def local_xyz(self) -> tuple[int, int, int]:
        """Local extent ``(lx, ly, lz)`` including ghosts, coordinate order."""
        g = (self.gnx, self.gny, self.gnz)
        return tuple(g[a] // (self.dx, self.dy, self.dz)[a] + 2 * self.halo[a] for a in range(3))  # type: ignore[return-value]

    @property
    def owned_xyz(self) -> tuple[int, int, int]:
        """Owned extent ``(ox, oy, oz)`` per device, coordinate order."""
        return (self.gnx // self.dx, self.gny // self.dy, self.gnz // self.dz)

    @property
    def n_devices(self) -> int:
        return self.dx * self.dy * self.dz

    @property
    def decomposed_axes(self) -> tuple[int, ...]:
        return tuple(a for a, d in ((0, self.dx), (1, self.dy), (2, self.dz)) if d > 1)

    def slab_shape(self) -> tuple[int, int, int, int]:
        lx, ly, lz = self.local_xyz
        return (19, lz, ly, lx)

    def owned_slice(self) -> tuple[slice, ...]:
        """Slice selecting the owned cells of a ``(Q, z, y, x)`` slab."""
        lx, ly, lz = self.local_xyz
        hx, hy, hz = self.halo
        ox, oy, oz = self.owned_xyz
        return (slice(None), slice(hz, hz + oz), slice(hy, hy + oy), slice(hx, hx + ox))

    def local_extent(self, axis: int) -> int:
        return self.local_xyz[axis]

    def face_dims(self, axis: int) -> tuple[int, int]:
        """Dimensions of a face slab in ``(z, y, x)`` order, minus ``axis``."""
        lx, ly, lz = self.local_xyz
        dims_zyx = (lz, ly, lx)
        return tuple(d for pos, d in enumerate(dims_zyx) if (2 - axis) != pos)  # type: ignore[return-value]

    def face_area(self, axis: int) -> int:
        """Face area of a *local* array perpendicular to ``axis``.

        Reference ``get_area`` (``lbm.cpp:1283-1287``) uses the local extents,
        so halo rows/columns of the perpendicular axes are communicated too —
        which is what makes the sequential x->y->z passes exact on domain
        edges (see the report).
        """
        a, b = self.face_dims(axis)
        return a * b

    @property
    def amax(self) -> int:
        """``Amax`` = largest face area over decomposed axes (``lbm.cpp:1308-1310``)."""
        return max((self.face_area(a) for a in self.decomposed_axes), default=0)

    def device_coords(self, d: int) -> tuple[int, int, int]:
        return d % self.dx, (d % (self.dx * self.dy)) // self.dx, d // (self.dx * self.dy)

    def device_index(self, x: int, y: int, z: int) -> int:
        return x + (y + z * self.dy) * self.dx

    def neighbour(self, d: int, axis: int, sign: int) -> int:
        """Periodic neighbour device (``lbm.cpp:1360/1369/1378`` modulo wrap)."""
        x, y, z = self.device_coords(d)
        if axis == 0:
            x = (x + sign) % self.dx
        elif axis == 1:
            y = (y + sign) % self.dy
        else:
            z = (z + sign) % self.dz
        return self.device_index(x, y, z)

    def global_offset(self, d: int) -> tuple[int, int, int]:
        """First owned global cell ``(x0, y0, z0)`` of device ``d``."""
        x, y, z = self.device_coords(d)
        ox, oy, oz = self.owned_xyz
        return (x * ox, y * oy, z * oz)


# ---------------------------------------------------------------------------
# Host-staged transport
# ---------------------------------------------------------------------------


@dataclass
class _Counters:
    """Measured (not estimated) copy volume, incremented at the real copy sites."""

    host_read_bytes: int = 0  # device -> host staging ("PCIe up")
    host_write_bytes: int = 0  # host staging -> device ("PCIe down")
    host_copy_bytes: int = 0  # host -> host (only in the "copy" staging mode)
    extract_calls: int = 0
    insert_calls: int = 0
    barriers: int = 0
    exchanges: int = 0
    exchange_time_s: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "host_read_bytes": self.host_read_bytes,
            "host_write_bytes": self.host_write_bytes,
            "host_copy_bytes": self.host_copy_bytes,
            "pcie_bytes": self.host_read_bytes + self.host_write_bytes,
            "extract_calls": self.extract_calls,
            "insert_calls": self.insert_calls,
            "barriers": self.barriers,
            "exchanges": self.exchanges,
            "exchange_time_s": self.exchange_time_s,
        }


class HostStagedHalo:
    """Host-staged halo transport for a ``Dx*Dy*Dz`` device grid.

    Two staging modes, both following the reference's per-direction schedule
    (extract-all -> barrier -> swap -> insert-all):

    ``"swap"``
        The reference scheme (``lbm.cpp:1359-1362``): the outgoing host buffers
        of device ``d`` and the incoming host buffers of its ``+`` neighbour are
        exchanged by pointer, so the host-to-host copy volume is zero.
    ``"copy"``
        The naive alternative that the reference's pointer swap avoids: each
        device copies the neighbour payload into its own receive buffer with an
        explicit host memcpy.  Kept as the controlled comparison arm.

    Args:
        layout: device decomposition.
        dtype: payload dtype (``float32`` matches the reference's FP32 build).
        staging: ``"swap"`` (reference) or ``"copy"`` (control).
        device: torch device for the staging buffers; on CUDA hosts the buffers
            are pinned, matching the reference's pinned host staging memory.
        pin: force pinned allocation when a CUDA device is present.
    """

    def __init__(
        self,
        layout: HostStagedLayout,
        *,
        dtype: torch.dtype = torch.float32,
        staging: str = "swap",
        device: torch.device | str = "cpu",
        pin: bool | None = None,
    ) -> None:
        if staging not in ("swap", "copy"):
            raise ValueError("staging must be 'swap' or 'copy'")
        self.layout = layout
        self.dtype = dtype
        self.staging = staging
        self.device = torch.device(device)
        self.pinned = (device != "cpu") if pin is None else bool(pin)
        if self.pinned and not torch.cuda.is_available():
            self.pinned = False
        self.counters = _Counters()
        self._buf: dict[tuple[int, str], torch.Tensor] = {}
        for d in range(layout.n_devices):
            for side in ("p", "m"):
                self._buf[(d, side)] = self._alloc_buffer()

    # -- allocation --------------------------------------------------------
    def _alloc_buffer(self) -> torch.Tensor:
        amax = max(self.layout.amax, 1)
        try:
            return torch.zeros((REF_TRANSFERS_PER_FACE, amax), dtype=self.dtype, pin_memory=self.pinned)
        except RuntimeError:  # pin_memory unsupported for this host/dtype
            return torch.zeros((REF_TRANSFERS_PER_FACE, amax), dtype=self.dtype)

    def buffer_bytes(self) -> int:
        """Bytes of one staging buffer (reference ``Amax*max(5*sizeof,17)``)."""
        b = self._buf[(0, "p")]
        return b.numel() * b.element_size()

    def staging_allocation_bytes(self) -> int:
        """Total staging allocation over the device grid (2 buffers/device)."""
        return sum(b.numel() * b.element_size() for b in self._buf.values())

    # -- selective extract / insert (kernel.cpp:2091-2137) -----------------
    def _ghost_slice(self, axis: int, index: int) -> tuple[slice, ...]:
        """Slice selecting ``index`` along coordinate ``axis`` of a ``(Q,z,y,x)`` slab."""
        sl = [slice(None)] * 3
        sl[2 - axis] = index
        return (slice(None),) + tuple(sl)

    def _face_pops(self, axis: int, side: str) -> tuple[int, ...]:
        """Population slots a staging buffer carries.

        ``p`` holds the parcels a device hands to its ``+`` neighbour, i.e. the
        ``c_axis = +1`` set read from its outer owned layer; ``m`` holds the
        ``c_axis = -1`` set read from its inner owned layer.  The buffers keep
        their population identity across the pointer swap, so the ``p`` buffer
        of device ``d`` receives the ``m`` set of the ``+`` neighbour (and lands
        in ``d``'s upper ghost layer), see :meth:`insert`.
        """
        return crossing_indices(axis, +1 if side == "p" else -1)

    @staticmethod
    def _recv_pops(axis: int, slot: str) -> tuple[int, ...]:
        """Population slots consumed when writing a received buffer into a ghost.

        A buffer that was filled by the ``+`` neighbour's ``p`` extract (the
        ``c = +1`` set) ends up in the local ``m`` slot and feeds the lower
        ghost layer; a buffer filled by the ``-`` neighbour's ``m`` extract
        (``c = -1``) ends up in the local ``p`` slot and feeds the upper ghost.
        """
        return crossing_indices(axis, +1 if slot == "m" else -1)

    def extract(self, slabs: list[torch.Tensor], axis: int) -> None:
        """Copy both faces of every device into its host staging buffers.

        Reference ``transfer_extract_fi`` reads layer ``N-2`` into ``p`` and
        layer ``1`` into ``m`` (``kernel.cpp:2091-2098, 2126-2131``).
        """
        lay = self.layout
        if lay.halo[axis] == 0:
            return
        n_local = lay.local_extent(axis)
        src_index = {"p": n_local - 2, "m": 1}
        area = lay.face_area(axis)
        face = lay.face_dims(axis)
        for d, slab in enumerate(slabs):
            for side in ("p", "m"):
                pops = self._face_pops(axis, side)
                payload = slab[self._ghost_slice(axis, src_index[side])][list(pops)]
                buf = self._buf[(d, side)]
                buf[:, :area].copy_(payload.reshape(len(pops), *face).reshape(len(pops), area))
                self.counters.host_read_bytes += area * len(pops) * buf.element_size()
        self.counters.extract_calls += 1

    def barrier(self) -> None:
        """Device barrier between extract and insert (reference ``finish_queue``)."""
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
        self.counters.barriers += 1

    def swap_axis(self, axis: int) -> None:
        """Rotate host staging buffers across one direction.

        Reference ``exchange_host_buffer`` chain (``lbm.cpp:1359-1362``): after
        the swap, ``p`` of device ``d`` holds what its ``+`` neighbour wrote
        into ``m`` and vice versa.  In ``"copy"`` mode the same data movement is
        done with an explicit host memcpy, so the two arms differ only in host
        copy volume.
        """
        lay = self.layout
        if lay.halo[axis] == 0:
            return
        area = lay.face_area(axis)
        if self.staging == "swap":
            for d in range(lay.n_devices):
                plus = lay.neighbour(d, axis, +1)
                p = self._buf[(d, "p")]
                m = self._buf[(plus, "m")]
                self._buf[(d, "p")], self._buf[(plus, "m")] = m, p
        else:
            # snapshot first: the rotation is a ring, sources must not be
            # overwritten before every device has read them
            snap = {}
            for d in range(lay.n_devices):
                plus = lay.neighbour(d, axis, +1)
                minus = lay.neighbour(d, axis, -1)
                snap[d] = (
                    self._buf[(plus, "m")][:, :area].clone(),  # -> local p slot
                    self._buf[(minus, "p")][:, :area].clone(),  # -> local m slot
                )
            for d in range(lay.n_devices):
                p_src, m_src = snap[d]
                self._buf[(d, "p")][:, :area].copy_(p_src)
                self._buf[(d, "m")][:, :area].copy_(m_src)
                self.counters.host_copy_bytes += 2 * area * REF_TRANSFERS_PER_FACE * self._buf[(d, "p")].element_size()

    def insert(self, slabs: list[torch.Tensor], axis: int) -> None:
        """Write the received staging buffers into both ghost layers.

        Reference: ``p`` buffer -> layer ``N-1``, ``m`` buffer -> layer ``0``
        (``kernel.cpp:2099-2106, 2132-2137``).
        """
        lay = self.layout
        if lay.halo[axis] == 0:
            return
        n_local = lay.local_extent(axis)
        dst_index = {"p": n_local - 1, "m": 0}
        area = lay.face_area(axis)
        face = lay.face_dims(axis)
        for d, slab in enumerate(slabs):
            for side in ("p", "m"):
                pops = self._recv_pops(axis, side)
                buf = self._buf[(d, side)]
                payload = buf[:, :area].reshape(len(pops), *face)
                slab[(torch.as_tensor(pops),) + self._ghost_slice(axis, dst_index[side])[1:]] = payload
                self.counters.host_write_bytes += area * len(pops) * buf.element_size()
        self.counters.insert_calls += 1

    # -- full per-step exchange (lbm.cpp:1355-1382) ------------------------
    def exchange(self, slabs: list[torch.Tensor]) -> None:
        """One halo exchange: direction-major, one barrier per direction."""
        lay = self.layout
        t0 = time.perf_counter()
        for axis in lay.decomposed_axes:  # x -> y -> z, as the reference
            self.extract(slabs, axis)
            self.barrier()
            self.swap_axis(axis)
            self.insert(slabs, axis)
        self.counters.exchanges += 1
        self.counters.exchange_time_s += time.perf_counter() - t0

    def stats(self) -> dict[str, object]:
        out: dict[str, object] = self.counters.as_dict()
        out.update(
            staging=self.staging,
            buffer_bytes=self.buffer_bytes(),
            staging_allocation_bytes=self.staging_allocation_bytes(),
            pinned=self.pinned,
            device=str(self.device),
            dtype=str(self.dtype),
            n_devices=self.layout.n_devices,
            amax=self.layout.amax,
        )
        return out


# ---------------------------------------------------------------------------
# Minimal multi-device D3Q19 solver on top of the transport
# ---------------------------------------------------------------------------


class HostStagedD3Q19:
    """Periodic D3Q19 BGK solver on a ``Dx*Dy*Dz`` device grid.

    Local slabs are ``(Q, lz, ly, lx)`` with ghost layers on the decomposed
    axes — the layout the reference allocates (``lbm.cpp:733``).  One time step
    is: collide owned cells -> halo exchange of the post-collision populations
    -> pull streaming into owned cells.
    """

    def __init__(
        self,
        layout: HostStagedLayout,
        *,
        nu: float = 0.02,
        dtype: torch.dtype = torch.float32,
        staging: str = "swap",
        device: torch.device | str = "cpu",
        pin: bool | None = None,
    ) -> None:
        self.layout = layout
        self.nu = nu
        self.omega = 1.0 / (3.0 * nu + 0.5)
        self.dtype = dtype
        self.device = torch.device(device)
        self.transport = HostStagedHalo(layout, dtype=dtype, staging=staging, device=device, pin=pin)
        self.slabs = [
            torch.zeros(layout.slab_shape(), dtype=dtype, device=self.device)
            for _ in range(layout.n_devices)
        ]
        self._c = torch.tensor(D3Q19_C, dtype=dtype, device=self.device)
        self._w = torch.tensor(D3Q19_W, dtype=dtype, device=self.device).view(19, 1, 1, 1)

    # -- initialisation ----------------------------------------------------
    def set_velocity(self, u_global: torch.Tensor, rho: float = 1.0) -> None:
        """Fill every slab with ``f = feq(rho, u)`` from a global ``(3, nz, ny, nx)``."""
        lay = self.layout
        expected = (3, lay.gnz, lay.gny, lay.gnx)
        if tuple(u_global.shape) != expected:
            raise ValueError(f"expected {expected}, got {tuple(u_global.shape)}")
        u_global = u_global.to(dtype=self.dtype, device=self.device)
        own = lay.owned_slice()
        for d, slab in enumerate(self.slabs):
            x0, y0, z0 = lay.global_offset(d)
            ox, oy, oz = lay.owned_xyz
            u_local = u_global[:, z0 : z0 + oz, y0 : y0 + oy, x0 : x0 + ox]
            cu = torch.einsum("qa,azyx->qzyx", self._c, u_local)
            u2 = (u_local * u_local).sum(dim=0)
            slab[own] = self._w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)

    # -- one LBM step ------------------------------------------------------
    def _collide(self) -> None:
        own = self.layout.owned_slice()
        for slab in self.slabs:
            f = slab[own]
            rho = f.sum(dim=0)
            u = torch.einsum("qa,qzyx->azyx", self._c, f) / rho
            u2 = (u * u).sum(dim=0)
            cu = torch.einsum("qa,azyx->qzyx", self._c, u)
            feq = self._w * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2)
            slab[own] = f - self.omega * (f - feq)

    def _stream(self) -> None:
        """Pull streaming into the owned cells.

        ``f_new_q(x) = f_q(x - c_q)``; on a decomposed axis the source index
        ``i - c`` always lands inside ``[0, n_local-1]`` (ghost layers), so a
        plain ``narrow`` view suffices — no gather is needed.  On an
        undecomposed axis the source wraps periodically inside the domain.
        """
        lay = self.layout
        own = lay.owned_slice()[1:]
        for d, slab in enumerate(self.slabs):
            parts = []
            for q, c in enumerate(D3Q19_C):
                src = slab[q]
                for axis in (0, 1, 2):  # 0=x, 1=y, 2=z
                    comp = c[axis]
                    dim = 2 - axis  # (z, y, x) sub-tensor
                    if lay.halo[axis]:
                        src = src.narrow(dim, 1 - comp, lay.owned_xyz[axis])
                    elif comp:
                        src = torch.roll(src, comp, dims=dim)
                parts.append(src)
            new = torch.empty_like(slab)
            new[(slice(None),) + own] = torch.stack(parts)
            self.slabs[d] = new

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._collide()
            self.transport.exchange(self.slabs)
            self._stream()

    # -- diagnostics -------------------------------------------------------
    def _owned_moments(self) -> tuple[torch.Tensor, torch.Tensor]:
        own = self.layout.owned_slice()
        rhos, us = [], []
        for slab in self.slabs:
            f = slab[own]
            rho = f.sum(dim=0)
            rhos.append(rho)
            us.append(torch.einsum("qa,qzyx->azyx", self._c, f) / rho)
        return torch.stack(rhos), torch.stack(us)

    def kinetic_energy(self) -> float:
        """Volume-averaged kinetic energy ``0.5*<rho u^2>``."""
        rhos, us = self._owned_moments()
        total = float((0.5 * rhos * (us * us).sum(dim=1)).sum())
        lay = self.layout
        return total / (lay.gnx * lay.gny * lay.gnz)

    def project_velocity(self) -> torch.Tensor:
        """Reassemble the global ``(3, nz, ny, nx)`` velocity field."""
        lay = self.layout
        ox, oy, oz = lay.owned_xyz
        out = torch.zeros((3, lay.gnz, lay.gny, lay.gnx), dtype=self.dtype)
        own = lay.owned_slice()
        for d, slab in enumerate(self.slabs):
            f = slab[own]
            rho = f.sum(dim=0)
            u = torch.einsum("qa,qzyx->azyx", self._c, f) / rho
            x0, y0, z0 = lay.global_offset(d)
            out[:, z0 : z0 + oz, y0 : y0 + oy, x0 : x0 + ox] = u.cpu()
        return out

    def project_density(self) -> torch.Tensor:
        lay = self.layout
        ox, oy, oz = lay.owned_xyz
        out = torch.zeros((lay.gnz, lay.gny, lay.gnx), dtype=self.dtype)
        own = lay.owned_slice()
        for d, slab in enumerate(self.slabs):
            rho = slab[own].sum(dim=0)
            x0, y0, z0 = lay.global_offset(d)
            out[z0 : z0 + oz, y0 : y0 + oy, x0 : x0 + ox] = rho.cpu()
        return out