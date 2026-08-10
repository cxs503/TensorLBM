"""Bounded-memory execution for strictly cell-local collision operators."""
from __future__ import annotations

import math
from collections.abc import Callable

import torch

from .d3q19 import WEIGHT_PRECISION_SCHEME
from .entropic_kbc import (
    _collide_natural_kbc_d3q19_unchecked,
    collide_natural_kbc_d3q19,
)

CellLocalCollision = Callable[[torch.Tensor], torch.Tensor]


class NaturalKBCCollisionExecutor:
    """Validated eager or graph-reusing natural-KBC execution.

    Python floating-point relaxation times make ``torch.compile`` specialize
    one graph per value, which is pathological during a viscosity ramp.  The
    compiled path converts validated tau to a zero-dimensional tensor so one
    dynamic graph can serve the full ramp and every z slab.
    """

    def __init__(
        self,
        *,
        compile_enabled: bool = False,
        compute_dtype: str = "storage",
    ) -> None:
        if compute_dtype not in {"storage", "float64"}:
            raise ValueError("compute_dtype must be storage or float64")
        self.compile_enabled = bool(compile_enabled)
        self.compute_dtype = compute_dtype
        self._compiled: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
        self._call_count = 0
        self._shape_signatures: set[tuple[str, str, tuple[int, ...]]] = set()
        self._minimum_tau = math.inf
        self._maximum_tau = -math.inf

    def __call__(self, populations: torch.Tensor, tau: float) -> torch.Tensor:
        if (
            not isinstance(populations, torch.Tensor)
            or populations.ndim != 4
            or populations.shape[0] != 19
        ):
            raise ValueError("populations must have shape (19,nz,ny,nx)")
        if not math.isfinite(tau) or tau <= 0.5:
            raise ValueError("tau must be finite and greater than 0.5")
        self._call_count += 1
        self._shape_signatures.add((
            str(populations.device),
            str(populations.dtype),
            tuple(populations.shape),
        ))
        self._minimum_tau = min(self._minimum_tau, tau)
        self._maximum_tau = max(self._maximum_tau, tau)
        compute_populations = (
            populations.to(torch.float64)
            if self.compute_dtype == "float64"
            else populations
        )
        if not self.compile_enabled:
            result = collide_natural_kbc_d3q19(compute_populations, tau)
            return result.to(dtype=populations.dtype)
        if self._compiled is None:
            self._compiled = torch.compile(
                _collide_natural_kbc_d3q19_unchecked,
                dynamic=True,
                fullgraph=False,
                mode="reduce-overhead",
            )
        tau_tensor = compute_populations.new_tensor(tau)
        result = self._compiled(compute_populations, tau_tensor)
        return result.to(dtype=populations.dtype)

    def diagnostics(self) -> dict[str, object]:
        """Return auditable execution and process-level compile counters."""
        unique_graphs = None
        if self.compile_enabled and self._compiled is not None:
            try:
                from torch._dynamo.utils import counters
                unique_graphs = int(counters["stats"]["unique_graphs"])
            except (ImportError, KeyError, TypeError, ValueError):
                pass
        return {
            "compile_enabled": self.compile_enabled,
            "storage_dtype_policy": "preserve_input",
            "compute_dtype": self.compute_dtype,
            "d3q19_weight_precision_scheme": WEIGHT_PRECISION_SCHEME,
            "collision_calls": self._call_count,
            "input_signatures": [
                {
                    "device": device,
                    "dtype": dtype,
                    "shape_qzyx": list(shape),
                }
                for device, dtype, shape in sorted(self._shape_signatures)
            ],
            "minimum_tau": (
                self._minimum_tau if self._call_count else None
            ),
            "maximum_tau": (
                self._maximum_tau if self._call_count else None
            ),
            "torch_dynamo_process_unique_graphs": unique_graphs,
        }


def collide_in_z_chunks(
    populations: torch.Tensor,
    collision: CellLocalCollision,
    *,
    chunk_cells: int,
) -> torch.Tensor:
    """Apply a cell-local collision in z slabs with a bounded working set.

    This is valid only for collision operators whose result at a cell depends
    on populations at that same cell.  Streaming, gradient SGS closures and
    any stencil-based regularisation must remain outside this helper.
    """
    if (
        not isinstance(populations, torch.Tensor)
        or populations.ndim != 4
        or populations.shape[0] not in (19, 27)
    ):
        raise ValueError("populations must have shape (19|27,nz,ny,nx)")
    if isinstance(chunk_cells, bool) or chunk_cells < 1:
        raise ValueError("chunk_cells must be a positive integer")
    _, nz, ny, nx = populations.shape
    planes_per_chunk = max(1, chunk_cells // (ny * nx))
    if planes_per_chunk >= nz:
        result = collision(populations)
        if result.shape != populations.shape or result.device != populations.device:
            raise ValueError("collision must preserve population shape and device")
        return result

    output = torch.empty_like(populations)
    for start in range(0, nz, planes_per_chunk):
        stop = min(start + planes_per_chunk, nz)
        slab = populations[:, start:stop]
        collided = collision(slab)
        if collided.shape != slab.shape or collided.device != populations.device:
            raise ValueError("collision must preserve slab shape and device")
        output[:, start:stop] = collided
    return output


__all__ = [
    "CellLocalCollision",
    "NaturalKBCCollisionExecutor",
    "collide_in_z_chunks",
]
