"""TCLB ``async-copy + MPI_Waitany`` halo-exchange schedule, ported to x-slab D3Q19.

Arms
----
``blocking``   what TensorLBM ships today (``D3Q19GlooTransport.step``): collide
               everything, pack, post, ``waitall``, unpack, then stream
               everything.  No overlap; comm is fully exposed.
``waitall``    TCLB with ``CROSS_MPI_WAITANY`` **off**: border columns collided and
               packed first, all requests posted, interior collide+stream runs
               under the exchange, then ``MPI_Waitall`` and consume every buffer.
``waitany``    TCLB with ``CROSS_MPI_WAITANY`` **on**: identical prefix, but each
               arrival is consumed (unpack + that boundary column's streaming)
               the moment ``MPI_Waitany`` reports it, while the other directions
               are still in flight.

Reference (source of record, no transcription):
``CFD-GO/TCLB`` ``src/Lattice.cu.Rt:327-386`` and ``src/Lattice.cu.Rt:496-530``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch

from ..d3q19 import C

MODES = ("blocking", "waitall", "waitany")
SIDES = ("left", "right")
MIN_SLAB_WIDTH = 3


def stream_full(post_all: torch.Tensor, left_ghost: torch.Tensor, right_ghost: torch.Tensor) -> torch.Tensor:
    """Periodic-y/z, ghost-backed-x pull streaming (identical to the shipped API)."""
    padded = torch.cat([left_ghost.unsqueeze(-1), post_all, right_ghost.unsqueeze(-1)], dim=-1)
    out = torch.empty_like(post_all)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        out[q] = torch.roll(padded[q], (cz, cy, cx), dims=(0, 1, 2))[..., 1:-1]
    return out


def stream_interior(post_all: torch.Tensor, width: int) -> torch.Tensor:
    """Stream only the columns whose whole upwind stencil is local (1..N-2)."""
    n_int = width - 2
    out = torch.empty((*post_all.shape[:-1], max(n_int, 0)), dtype=post_all.dtype)
    if n_int <= 0:
        return out
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        src = post_all[q]
        if cy or cz:
            src = torch.roll(src, (cz, cy), dims=(0, 1))
        if cx == 1:
            window = src[..., 0 : width - 2]
        elif cx == -1:
            window = src[..., 2:width]
        else:
            window = src[..., 1 : width - 1]
        out[q] = window
    return out


def stream_one_column(
    post_all: torch.Tensor,
    ghost: torch.Tensor,
    side: str,
    width: int,
) -> torch.Tensor:
    """Stream the single owned column that needs the ghost plane from ``side``."""
    if width < MIN_SLAB_WIDTH:
        raise ValueError(f"slab width {width} < {MIN_SLAB_WIDTH}: use the blocking arm")
    if side == "left":
        # padded indices 0..2 -> [ghost, post[0], post[1]]
        window = torch.cat([ghost.unsqueeze(-1), post_all[..., 0:2]], dim=-1)
        offset = 0
        column = 0
    elif side == "right":
        # padded indices width-2..width+1 -> [post[w-3], post[w-2], post[w-1], ghost]
        window = torch.cat([post_all[..., width - 3 : width], ghost.unsqueeze(-1)], dim=-1)
        offset = width - 2
        column = width - 1
    else:  # pragma: no cover - caller controlled
        raise ValueError(f"unknown side {side!r}")
    out = torch.empty(post_all.shape[:-1], dtype=post_all.dtype)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        src = window[q]
        if cy or cz:
            src = torch.roll(src, (cz, cy), dims=(0, 1))
        out[q] = src[..., column + 1 - cx - offset]
    return out


class D3Q19OverlapTransport:
    """One-rank-per-x-slab D3Q19 transport with selectable overlap schedule."""

    def __init__(self, backend: Any, mode: str = "waitany") -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.backend = backend
        self.mode = mode
        self._recv: dict[tuple[int, int, int], dict[str, torch.Tensor]] = {}
        self.last: dict[str, Any] = {}

    # ------------------------------------------------------------------ helpers
    def _recv_buffers(self, f_owned: torch.Tensor) -> dict[str, torch.Tensor]:
        plane = (f_owned.shape[0], f_owned.shape[1], f_owned.shape[2])
        buffers = self._recv.get(plane)
        if buffers is None or buffers["left"].dtype != f_owned.dtype:
            buffers = {side: torch.empty(plane, dtype=f_owned.dtype) for side in SIDES}
            self._recv[plane] = buffers
        return buffers

    @staticmethod
    def _collide(fn: Callable[[torch.Tensor], torch.Tensor] | None, block: torch.Tensor) -> torch.Tensor:
        return block if fn is None else fn(block)

    def _border_collide(
        self, f_owned: torch.Tensor, collide_fn: Callable[[torch.Tensor], torch.Tensor] | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        win = torch.cat([f_owned[..., :1], f_owned[..., -1:]], dim=-1)
        cb = self._collide(collide_fn, win)
        return cb[..., 0], cb[..., -1]

    def _checked(self, f_owned: torch.Tensor) -> None:
        if f_owned.dim() != 4 or f_owned.shape[0] != 19:
            raise ValueError("f_owned must have shape (19, nz, ny, nx_local)")
        if f_owned.shape[-1] < MIN_SLAB_WIDTH:
            raise ValueError(
                f"f_owned slab width {f_owned.shape[-1]} < {MIN_SLAB_WIDTH}"
            )

    # --------------------------------------------------------------------- step
    def step(
        self,
        f_owned: torch.Tensor,
        collide_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        self._checked(f_owned)
        width = f_owned.shape[-1]
        backend = self.backend
        backend.reset()
        recv = self._recv_buffers(f_owned)
        with torch.no_grad():
            if self.mode == "blocking":
                out = self._step_blocking(f_owned, collide_fn, recv, width)
            else:
                out = self._step_pipelined(f_owned, collide_fn, recv, width)
        backend.close()
        self.last = {"mode": self.mode, "stats": backend.stats}
        return out

    def _step_blocking(
        self,
        f_owned: torch.Tensor,
        collide_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        recv: dict[str, torch.Tensor],
        width: int,
    ) -> torch.Tensor:
        backend = self.backend
        post_all = backend.run_compute("collide", width, lambda: self._collide(collide_fn, f_owned))
        left, right = backend.run_compute(
            "pack", 2, lambda: (post_all[..., 0].contiguous(), post_all[..., -1].contiguous())
        )
        # shipped order: post sends, post receives, then wait for all
        h_send = [backend.post_send("left", left), backend.post_send("right", right)]
        h_recv = [backend.post_recv(side, recv[side]) for side in SIDES]
        backend.waitall(h_recv)
        for side in SIDES:
            backend.deliver(side)
        out = backend.run_compute(
            "stream", width, lambda: stream_full(post_all, recv["left"], recv["right"])
        )
        backend.waitall(h_send)
        return out

    def _step_pipelined(
        self,
        f_owned: torch.Tensor,
        collide_fn: Callable[[torch.Tensor], torch.Tensor] | None,
        recv: dict[str, torch.Tensor],
        width: int,
    ) -> torch.Tensor:
        backend = self.backend
        # 1. RunBorder: collide only the two MPI-facing columns
        left_send, right_send = backend.run_compute(
            "collide", 2, lambda: self._border_collide(f_owned, collide_fn)
        )
        # 2. MPIStream_A analogue: pack the border payloads (async D2H in TCLB)
        left_send, right_send = backend.run_compute(
            "pack", 2, lambda: (left_send.contiguous(), right_send.contiguous())
        )
        # 3. MPIStream_B prefix: issue *every* request before touching the interior
        h_recv = [backend.post_recv(side, recv[side]) for side in SIDES]
        h_send = [backend.post_send("left", left_send), backend.post_send("right", right_send)]

        # 4. RunInterior: collide + stream the columns that need no ghost, i.e.
        #    the work that hides the exchange
        def interior() -> tuple[torch.Tensor, torch.Tensor]:
            inner = self._collide(collide_fn, f_owned[..., 1 : width - 1])
            post_all = torch.cat(
                [left_send.unsqueeze(-1), inner, right_send.unsqueeze(-1)], dim=-1
            )
            return post_all, stream_interior(post_all, width)

        post_all, interior_out = backend.run_compute(
            "stream", max(width - 2, 1), interior
        )

        # 5. consume arrivals: waitany = one buffer at a time, waitall = batch
        border: dict[str, torch.Tensor] = {}

        def consume(side: str) -> None:
            backend.deliver(side)
            border[side] = backend.run_compute(
                "stream", 1, lambda side=side: stream_one_column(post_all, recv[side], side, width)
            )

        handles = list(h_recv)
        names = list(SIDES)
        if self.mode == "waitall":
            backend.waitall(handles)
            for side in names:
                consume(side)
        else:  # waitany
            while handles:
                index = backend.waitany(handles)
                handles.pop(index)
                consume(names.pop(index))
        backend.waitall(h_send)

        return torch.cat([border["left"].unsqueeze(-1), interior_out, border["right"].unsqueeze(-1)], dim=-1)


__all__ = [
    "MODES",
    "D3Q19OverlapTransport",
    "stream_full",
    "stream_interior",
    "stream_one_column",
]