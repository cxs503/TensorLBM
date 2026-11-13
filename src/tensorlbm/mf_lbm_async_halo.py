"""MF-LBM ``openacc-async-halo-overlap`` schedule (order swap) ported to a D3Q19 x-slab.

Anchor
------
``XVAL-LBM-OPENACC-ASYNC-HALO-OVERLAP`` (round 1).  Reference implementation read
from source (no transcription): ``lanl/MF-LBM`` @ ``11be324``,

============================  =====================================================
file:line (source of record)  what it fixes in the design
============================  =====================================================
``multiphase_3D/0.src/Main_multiphase.F90:354-394``
                              even step: z-face strip kernels on async queue
                              ``LBM_async_z``, then y-face strips on ``LBM_async_y``
                              which *waits* on ``LBM_async_z``; then
                              ``mpi_pdf_pull_async_pack`` + ``mpi_phi_async_pack``;
                              then (``#ifdef _openacc``) the interior kernel is
                              submitted **before** ``mpi_send_req``, else after it;
                              then ``mpi_pdf_pull_sync_update`` + ``$acc wait``.
``multiphase_3D/0.src/Main_multiphase.F90:419-458``
                              odd step: identical structure with the push packers.
``multiphase_3D/0.src/Mpi.F90:5-44``  ``mpi_irecv_initialization`` -- every
                              ``MPI_IRECV`` is posted once, *before* any kernel.
``multiphase_3D/0.src/Mpi.F90:49-95``  ``mpi_send_req`` -- ``MPI_Isend`` per face,
                              each preceded by ``$acc wait(<dir>_pdf_update_async)``
                              so the D2H copy of that face's payload has landed.
``multiphase_3D/0.src/Mpi.F90:101-212``  ``*_async_pack`` -- the *post-collision*
                              boundary planes are packed (only the population
                              halves that point outward, plus ``phi``).
``multiphase_3D/0.src/Mpi.F90:218-346``  ``*_sync_update`` -- ``MPI_WAITALL(4,..)``
                              per face, then the received payload is unpacked into
                              the opposite ghost plane.
``multiphase_3D/0.src/Module.F90:159-164``  15 ``async`` queue ids
                              (``LBM_async_z``/``y``/``x``, per-direction pack /
                              update queues, edge queues) -- i.e. *one lane per
                              direction*, not one lane for the whole exchange.
``multiphase_3D/0.src/Module.F90:263``  ``overlap=1, overlap_walls=2,
                              overlap_phi=4`` -- one ghost layer of width 1.
============================  =====================================================

What is actually being cross-validated here
-------------------------------------------
The *schedule*, not the multiphase physics.  Two properties are claimed by the
upstream design and are machine-checked in this module:

``order swap``      With a non-blocking exchange posted around a large kernel, the
                    payloads must be in flight while the interior computes, so the
                    step makespan is ``max(compute, comm) + eps`` and not
                    ``compute + comm``.  MF-LBM switches *which* of the two is
                    issued first per platform (kernel first on OpenACC, ``Isend``
                    first on the CPU branch).  Both orders are reproduced here as
                    ``gpu_order`` / ``cpu_order``.
``one lane per dir`` ``Module.F90:159-164`` gives each face its own async queue, so
                    the two face strips run concurrently.  ``strip_lanes="split"``
                    models that; ``strip_lanes="shared"`` models the degenerate
                    single-lane schedule (which is what a naive port gets).

Both are contrasted against ``blocking`` -- the halo-before-compute schedule that
pays ``compute + comm``.  Accuracy-wise all three must be *bit-identical*: the
schedule moves work around in time, it does not change the arithmetic.

Port scope / honest limits
--------------------------
* x-slab pencils only (1 decomposed axis, faces, no edges/corners).  MF-LBM also
  exchanges 12 edges and 8 corners with rotated population sets
  (``Module.F90:246-254``); an x-slab needs none of them.  Not ported, on purpose.
* single PDF (D3Q19 BGK), MF-LBM carries two (``f`` and ``g``) plus ``phi``.
* timing is measured on the deterministic latency/bandwidth/compute cost model
  (:class:`~tensorlbm.comm_overlap.backend.VirtualClockBackend`); the field itself
  is real ``torch`` float64 arithmetic.  Wall-clock figures would need a progress
  engine, which gloo in this build does not expose (see ``comm_overlap/backend``).
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .comm_overlap.waitany_pipeline import (
    MIN_SLAB_WIDTH,
    SIDES,
    stream_full,
    stream_interior,
    stream_one_column,
)
from .d3q19 import C, W

MF_LBM_MODES = ("blocking", "cpu_order", "gpu_order")
STRIP_LANES = ("split", "shared")
CS2 = 1.0 / 3.0


# --------------------------------------------------------------------------- kernels
def bgk_collide(block: torch.Tensor, tau: float) -> torch.Tensor:
    """Post-collision populations of a block ``(19, nz, ny, nx)`` (float64).

    Standard BGK: ``f* = f - (f - feq)/tau`` with ``rho = sum_q f_q``,
    ``rho u = sum_q c_q f_q`` and ``feq_q = W_q rho (1 + (c.u)/cs2
    + (c.u)^2/(2 cs2^2) - u^2/(2 cs2))``.
    """
    if block.shape[0] != C.shape[0]:
        raise ValueError(f"block must have {C.shape[0]} populations, got {block.shape[0]}")
    dtype = block.dtype
    c = C.to(dtype).reshape(19, 3, *([1] * (block.dim() - 1)))
    w = W.to(dtype).reshape(19, *([1] * (block.dim() - 1)))
    rho = block.sum(dim=0, keepdim=True)
    mom = (c * block.unsqueeze(1)).sum(dim=0)
    u = mom / rho
    cu = (c * u.unsqueeze(0)).sum(dim=1)
    usq = (u * u).sum(dim=0, keepdim=True)
    feq = w * rho * (1.0 + cu / CS2 + cu * cu / (2.0 * CS2 * CS2) - usq / (2.0 * CS2))
    return block - (block - feq) / tau


def collision_counters(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(rho, u)`` of a block -- used by the accuracy checks and the TGV reference."""
    c = C.to(block.dtype).reshape(19, 3, *([1] * (block.dim() - 1)))
    rho = block.sum(dim=0)
    u = (c * block.unsqueeze(1)).sum(dim=0) / rho
    return rho, u


# ------------------------------------------------------------------------ transport
class D3Q19MFAsyncTransport:
    """One-rank-per-x-slab D3Q19 transport with the MF-LBM overlap schedule.

    Parameters
    ----------
    backend
        :class:`~tensorlbm.comm_overlap.backend.VirtualClockBackend` (cost model) or
        any object exposing ``post_recv / post_send / waitall / deliver /
        run_compute / reset / close / stats`` (``TorchDistBackend`` does).
    mode
        ``"blocking"`` (halo before compute), ``"cpu_order"`` (``Isend`` then the
        interior kernel -- MF-LBM's non-OpenACC branch) or ``"gpu_order"`` (interior
        kernel then ``Isend`` -- MF-LBM's ``#ifdef _openacc`` branch).
    strip_width
        Thickness (in owned columns) of the face strips that are collided *before*
        the exchange is packed.  MF-LBM's ``iz_async`` / ``iy_async`` analogue.
    strip_lanes
        ``"split"`` -- one async queue per face (``Module.F90:159-164``), the two
        strip kernels overlap; ``"shared"`` -- one queue, they serialise.
    """

    def __init__(
        self,
        backend: Any,
        mode: str = "gpu_order",
        strip_width: int = 1,
        strip_lanes: str = "split",
    ) -> None:
        if mode not in MF_LBM_MODES:
            raise ValueError(f"mode must be one of {MF_LBM_MODES}, got {mode!r}")
        if strip_lanes not in STRIP_LANES:
            raise ValueError(f"strip_lanes must be one of {STRIP_LANES}, got {strip_lanes!r}")
        if strip_width < 1:
            raise ValueError("strip_width must be >= 1")
        self.backend = backend
        self.mode = mode
        self.strip_width = int(strip_width)
        self.strip_lanes = strip_lanes
        self._recv: dict[tuple[int, ...], dict[str, torch.Tensor]] = {}
        self.last: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _recv_buffers(self, f_owned: torch.Tensor) -> dict[str, torch.Tensor]:
        plane = (int(f_owned.shape[0]), int(f_owned.shape[1]), int(f_owned.shape[2]))
        buffers = self._recv.get(plane)
        if buffers is None or buffers["left"].dtype != f_owned.dtype:
            buffers = {side: torch.empty(plane, dtype=f_owned.dtype) for side in SIDES}
            self._recv[plane] = buffers
        return buffers

    def _checked(self, f_owned: torch.Tensor) -> tuple[int, int]:
        if f_owned.dim() != 4 or f_owned.shape[0] != C.shape[0]:
            raise ValueError("f_owned must have shape (19, nz, ny, nx_local)")
        width = int(f_owned.shape[-1])
        if width < MIN_SLAB_WIDTH:
            raise ValueError(f"f_owned slab width {width} < {MIN_SLAB_WIDTH}")
        w = min(self.strip_width, (width - 2) // 2)
        if w < 1:
            raise ValueError(f"slab width {width} leaves no interior for strips")
        self._cells = int(f_owned.shape[1] * f_owned.shape[2])
        return width, w

    def _units(self, columns: int) -> float:
        return float(self._cells * max(columns, 1))

    # --------------------------------------------------------------------- step
    def step(
        self,
        f_owned: torch.Tensor,
        collide_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        width, w = self._checked(f_owned)
        collide = (lambda b: bgk_collide(b, 0.6)) if collide_fn is None else collide_fn
        backend = self.backend
        backend.reset()
        recv = self._recv_buffers(f_owned)
        with torch.no_grad():
            # Mpi.F90:5-44 -- every receive is posted once, before any kernel runs.
            h_recv = [backend.post_recv(side, recv[side]) for side in SIDES]
            if self.mode == "blocking":
                out = self._step_blocking(f_owned, collide, recv, width, h_recv)
            else:
                out = self._step_overlapped(f_owned, collide, recv, width, w, h_recv)
        backend.close()
        self.last = {"mode": self.mode, "strip_width": w, "stats": backend.stats}
        return out

    # ------------------------------------------------------------------ schedules
    def _step_blocking(
        self,
        f_owned: torch.Tensor,
        collide: Callable[[torch.Tensor], torch.Tensor],
        recv: dict[str, torch.Tensor],
        width: int,
        h_recv: list[Any],
    ) -> torch.Tensor:
        """Halo before compute: the exchange latency is fully exposed."""
        backend = self.backend
        units = self._units(width)
        post_all = backend.run_compute("collide", units, lambda: collide(f_owned))
        left, right = backend.run_compute(
            "pack", 2 * self._units(1) * C.shape[0],
            lambda: (post_all[..., 0].contiguous(), post_all[..., -1].contiguous()),
        )
        h_send = [backend.post_send("left", left), backend.post_send("right", right)]
        backend.waitall(h_recv)
        for side in SIDES:
            backend.deliver(side)
        out = backend.run_compute(
            "stream", units, lambda: stream_full(post_all, recv["left"], recv["right"])
        )
        backend.waitall(h_send)
        return out

    def _step_overlapped(
        self,
        f_owned: torch.Tensor,
        collide: Callable[[torch.Tensor], torch.Tensor],
        recv: dict[str, torch.Tensor],
        width: int,
        w: int,
        h_recv: list[Any],
    ) -> torch.Tensor:
        backend = self.backend
        # (1) face-strip kernels, Main_multiphase.F90:362-380.  One async queue per
        #     face (Module.F90:159-164) => the two strips overlap; a single queue
        #     would serialise them.  The lanes are what ``strip_lanes`` toggles.
        lane_units = self._units(w)
        strip_units = lane_units if self.strip_lanes == "split" else 2.0 * lane_units

        def _strips() -> tuple[torch.Tensor, torch.Tensor]:
            left_strip = collide(f_owned[..., :w])
            right_strip = collide(f_owned[..., width - w :])
            return left_strip, right_strip

        left_strip, right_strip = backend.run_compute("collide", strip_units, _strips)

        # (2) *_async_pack (Mpi.F90:101-212): the post-collision boundary planes go
        #     to the host buffers, one queue per direction.
        plane_units = float(self._cells * C.shape[0])
        left_send, right_send = backend.run_compute(
            "pack",
            2.0 * plane_units,
            lambda: (left_strip[..., 0].contiguous(), right_strip[..., -1].contiguous()),
        )

        # (3) interior kernel + its locally-streamable window
        def _interior() -> tuple[torch.Tensor, torch.Tensor]:
            middle = collide(f_owned[..., w : width - w])
            post_all = torch.cat([left_strip, middle, right_strip], dim=-1)
            return post_all, stream_interior(post_all, width)

        interior_units = self._units(width - 2 * w) + self._units(width - 2)

        # (4) the order swap, Main_multiphase.F90:371-378:
        #     `#ifdef _openacc` -> kernel(...) then mpi_send_req
        #     `#else`          -> mpi_send_req then kernel(...)
        if self.mode == "cpu_order":
            h_send = [backend.post_send("left", left_send), backend.post_send("right", right_send)]
            post_all, interior_out = backend.run_compute("stream", interior_units, _interior)
        elif self.mode == "gpu_order":
            post_all, interior_out = backend.run_compute("stream", interior_units, _interior)
            h_send = [backend.post_send("left", left_send), backend.post_send("right", right_send)]
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(self.mode)

        # (5) *_sync_update (Mpi.F90:218-346): WAITALL, then consume the ghosts into
        #     the two boundary columns that need them.
        backend.waitall(h_recv)
        border = {}
        for side in SIDES:
            backend.deliver(side)
            border[side] = backend.run_compute(
                "stream", self._units(1), lambda side=side: stream_one_column(post_all, recv[side], side, width)
            )
        backend.waitall(h_send)
        return torch.cat([border["left"].unsqueeze(-1), interior_out, border["right"].unsqueeze(-1)], dim=-1)


# --------------------------------------------------------------------------- driver
def run_mf_virtual_case(
    full: torch.Tensor,
    mode: str = "gpu_order",
    world_size: int = 4,
    n_steps: int = 10,
    model: dict[str, Any] | None = None,
    collide_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cuts: list[int] | None = None,
    strip_width: int = 1,
    strip_lanes: str = "split",
    record_steps: Any = None,
) -> dict[str, Any]:
    """Single-process simulated x-slab decomposition of ``full`` (19, nz, ny, nx).

    The ghost payload handed to :meth:`D3Q19MFAsyncTransport.step` is the
    neighbour's *post-collision* plane of the same step, recomputed on the global
    field -- so an assembly error in the schedule surfaces as a mismatch against
    the monolithic reference rather than hiding inside the cost model
    (``comm_overlap/virtual_case.py`` docstring, same protocol).
    """
    from .comm_overlap.backend import VirtualClockBackend
    from .comm_overlap.virtual_case import DEFAULT_MODEL, monolithic_step, slab_cuts

    model = dict(DEFAULT_MODEL if model is None else model)
    model.setdefault("jitter_us", {})
    width = int(full.shape[-1])
    cuts = slab_cuts(width, world_size) if cuts is None else list(cuts)
    if len(cuts) != world_size + 1:
        raise ValueError("cuts must hold world_size + 1 boundaries")

    collide = (lambda b: bgk_collide(b, 0.6)) if collide_fn is None else collide_fn
    ranks = [full[..., cuts[r] : cuts[r + 1]].clone().contiguous() for r in range(world_size)]
    post_global: dict[str, torch.Tensor] = {}
    backends: list[VirtualClockBackend] = []
    transports: list[D3Q19MFAsyncTransport] = []

    for rank in range(world_size):
        left_owner = (rank - 1) % world_size
        right_owner = (rank + 1) % world_size

        def _supplier(name: str, buffer: torch.Tensor, left_owner=left_owner, right_owner=right_owner) -> None:
            post = post_global["post"]
            idx = cuts[left_owner + 1] - 1 if name == "left" else cuts[right_owner]
            buffer.copy_(post[..., idx])

        backend = VirtualClockBackend(
            supplier=_supplier,
            latency_us=model["latency_us"],
            bandwidth_bytes_per_us=model["bandwidth_bytes_per_us"],
            compute_us=model["compute_us"],
            jitter_us=model.get("jitter_us", {}),
        )
        backends.append(backend)
        transports.append(
            D3Q19MFAsyncTransport(
                backend, mode=mode, strip_width=strip_width, strip_lanes=strip_lanes
            )
        )

    reference = full.clone()
    step_rows: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    watch = set(int(s) for s in (record_steps or []))
    for istep in range(1, n_steps + 1):
        post_global["post"] = collide(reference)
        stats = []
        for rank in range(world_size):
            ranks[rank] = transports[rank].step(ranks[rank], collide)
            stats.append(backends[rank].stats.as_dict())
        current = torch.cat(ranks, dim=-1)
        reference = monolithic_step(reference, collide)
        step_rows.append(
            {
                "max_abs_error": float((current - reference).abs().max()),
                "mismatch": int((current != reference).sum().item()),
                "clock_us": float(sum(s["clock_us"] for s in stats) / world_size),
                "stall_us": float(sum(s["stall_us"] for s in stats) / world_size),
                "stall_fraction": float(sum(s["stall_fraction"] for s in stats) / world_size),
                "compute_us": float(sum(sum(s["compute_us"].values()) for s in stats) / world_size),
            }
        )
        if istep in watch:
            _, u_now = collision_counters(current)
            _, u_ref = collision_counters(reference)
            curve.append(
                {
                    "step": istep,
                    "peak_overlapped": float(u_now.norm(dim=0).max()),
                    "peak_reference": float(u_ref.norm(dim=0).max()),
                    "max_abs_error": float((current - reference).abs().max()),
                }
            )

    last = step_rows[-1]
    return {
        "mode": mode,
        "strip_width": strip_width,
        "strip_lanes": strip_lanes,
        "world_size": world_size,
        "n_steps": n_steps,
        "cuts": cuts,
        "latency_us": float(model["latency_us"]),
        "max_abs_error": last["max_abs_error"],
        "mismatch": last["mismatch"],
        "clock_us": last["clock_us"],
        "stall_us": last["stall_us"],
        "stall_fraction": last["stall_fraction"],
        "compute_us": last["compute_us"],
        "steps": step_rows,
        "curve": curve,
        "final": ranks,
    }


__all__ = [
    "MF_LBM_MODES",
    "STRIP_LANES",
    "D3Q19MFAsyncTransport",
    "bgk_collide",
    "collision_counters",
    "run_mf_virtual_case",
]