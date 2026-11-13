"""x-slab D3Q19 ghost exchange: shipped full-plane path vs OSS selective path.

Both arms live in one class so the ablation changes exactly one thing: the set
of populations that crosses the face (and, as a separate flag, whether the
padded buffer is reused).  ``selective=False, persistent=False`` replicates
``tensorlbm.multi_gpu.D3Q19GlooTransport.exchange_ghosts`` -- collide, post the
full 19-population boundary planes, ``Waitall``, unpack all 19 channels, rebuild
the padded field -- and the test suite asserts the two agree bit-exactly.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import torch

from .kernels import pack_face, unpack_face
from .packinfo import FacePackInfo, face_packinfos, face_index

TAG_RIGHT = 201
TAG_LEFT = 202


class SlabExchange:
    """Periodic x-ring ghost exchange for a D3Q19 pull stream."""

    def __init__(
        self,
        rank: int,
        world_size: int,
        *,
        selective: bool = True,
        persistent: bool = False,
        pg: Any = None,
    ) -> None:
        if world_size < 2:
            raise ValueError("a ghost exchange needs at least two ranks")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.left_rank = (self.rank - 1) % self.world_size
        self.right_rank = (self.rank + 1) % self.world_size
        self.selective = bool(selective)
        self.persistent = bool(persistent)
        self.left_info, self.right_info = face_packinfos(selective=self.selective)
        if pg is None:
            import torch.distributed as dist

            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError("SlabExchange needs an initialized process group")
            pg = dist
        self.pg = pg
        self._padded: Optional[torch.Tensor] = None
        self.exchange_seconds = 0.0
        self.bytes_last_step = 0
        self.pack_seconds = 0.0
        self.unpack_seconds = 0.0
        self.steps = 0

    # -- buffer management -------------------------------------------------
    def _padded_buffer(self, post: torch.Tensor) -> torch.Tensor:
        q, nz, ny, width = post.shape
        shape = (q, nz, ny, width + 2)
        if self.persistent:
            if (
                self._padded is None
                or self._padded.shape != shape
                or self._padded.dtype != post.dtype
            ):
                self._padded = torch.empty(shape, dtype=post.dtype, device=post.device)
            padded = self._padded
        else:
            padded = torch.empty(shape, dtype=post.dtype, device=post.device)
        padded[..., 1:-1] = post
        return padded

    # -- the exchange ------------------------------------------------------
    def exchange(self, post: torch.Tensor) -> torch.Tensor:
        if post.ndim != 4 or post.shape[0] != 19:
            raise ValueError("post must be (19, nz, ny, nx_local)")
        if post.device.type != "cpu":
            raise ValueError("SlabExchange is CPU-only (gloo)")
        started = time.perf_counter()
        padded = self._padded_buffer(post)
        width = post.shape[-1]

        pack_started = time.perf_counter()
        send_right = pack_face(post, self.right_info.populations, face_index("right", width))
        send_left = pack_face(post, self.left_info.populations, face_index("left", width))
        recv_left = torch.empty_like(send_right)
        recv_right = torch.empty_like(send_left)
        self.pack_seconds += time.perf_counter() - pack_started

        requests = [
            self.pg.isend(send_left, dst=self.left_rank, tag=TAG_LEFT),
            self.pg.isend(send_right, dst=self.right_rank, tag=TAG_RIGHT),
            self.pg.irecv(recv_left, src=self.left_rank, tag=TAG_RIGHT),
            self.pg.irecv(recv_right, src=self.right_rank, tag=TAG_LEFT),
        ]
        for request in requests:
            if request is not None:
                request.wait()

        unpack_started = time.perf_counter()
        unpack_face(recv_left, padded, self.left_info.ghost_populations, 0)
        unpack_face(recv_right, padded, self.right_info.ghost_populations, padded.shape[-1] - 1)
        self.unpack_seconds += time.perf_counter() - unpack_started

        itemsize = post.element_size()
        nz, ny = post.shape[1], post.shape[2]
        self.bytes_last_step = 2 * (
            self.right_info.packed_bytes(nz, ny, itemsize)
            + self.left_info.packed_bytes(nz, ny, itemsize)
        )
        self.steps += 1
        self.exchange_seconds += time.perf_counter() - started
        return padded

    # -- streaming ---------------------------------------------------------
    @staticmethod
    def stream(padded: torch.Tensor, stencil: torch.Tensor | None = None) -> torch.Tensor:
        """Periodic y/z, ghost-backed x pull stream (identical in every arm)."""
        if stencil is None:
            from ..d3q19 import C as stencil
        out = torch.empty_like(padded[..., 1:-1])
        for q, (cx, cy, cz) in enumerate(stencil.tolist()):
            out[q] = torch.roll(padded[q], shifts=(cz, cy, cx), dims=(0, 1, 2))[..., 1:-1]
        return out

    def step(
        self,
        f_owned: torch.Tensor,
        collide_fn: Callable[[torch.Tensor], torch.Tensor],
        stencil: torch.Tensor | None = None,
    ) -> torch.Tensor:
        post = collide_fn(f_owned)
        return self.stream(self.exchange(post), stencil)

    # -- introspection -----------------------------------------------------
    def packed_byte_accounting(self, nz: int, ny: int, itemsize: int) -> dict[str, Any]:
        """Static (pre-run) accounting, the waLBerla getSizeOfPackedCellInterval analogue."""
        left: FacePackInfo = self.left_info
        right: FacePackInfo = self.right_info
        per_face = left.packed_bytes(nz, ny, itemsize) + right.packed_bytes(nz, ny, itemsize)
        return {
            "selective": self.selective,
            "persistent": self.persistent,
            "left_populations": list(left.populations),
            "right_populations": list(right.populations),
            "n_cross_per_face": left.n_populations,
            "bytes_per_step": 2 * per_face,
            "bytes_per_face_step": per_face,
        }


def replicate_shipped_exchange(post: torch.Tensor, pg: Any, rank: int, world_size: int) -> torch.Tensor:
    """The shipped ``D3Q19GlooTransport.exchange_ghosts`` path, for cross-checking."""
    exchange = SlabExchange(rank, world_size, selective=False, persistent=False, pg=pg)
    return exchange.exchange(post)