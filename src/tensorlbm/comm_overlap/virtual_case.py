"""Single-process simulated decomposition driver for the overlap study.

The driver owns the global field and a non-uniform x-slab cut.  Each rank runs a
real :class:`D3Q19OverlapTransport` step; only the traffic is modelled, by
``VirtualClockBackend`` with a configurable latency/bandwidth/compute cost model.
Ghost payloads are supplied from an independently computed ground-truth
post-collision field, so any assembly error in the schedule shows up as a
mismatch against the monolithic reference instead of hiding inside the model.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from .backend import VirtualClockBackend
from .waitany_pipeline import D3Q19OverlapTransport, stream_full

DEFAULT_MODEL: dict[str, Any] = {
    "latency_us": 5.0,
    "bandwidth_bytes_per_us": 32.0,
    "compute_us": {"collide": 2.0, "stream": 2.0, "pack": 4.0},
    "jitter_us": {},
}


def slab_cuts(width: int, world_size: int) -> list[int]:
    """Non-uniform cuts: the remainder goes to the low ranks (as in the shipped API)."""
    base, rem = divmod(width, world_size)
    if base < 3:
        raise ValueError("each slab needs width >= 3 for the pipelined arms")
    cuts = [0]
    for rank in range(world_size):
        cuts.append(cuts[-1] + base + (1 if rank < rem else 0))
    return cuts


def monolithic_step(
    full: torch.Tensor, collide_fn: Callable[[torch.Tensor], torch.Tensor] | None
) -> torch.Tensor:
    """Serial periodic-x reference step (identical math, no decomposition)."""
    post = full if collide_fn is None else collide_fn(full)
    return stream_full(post, post[..., -1], post[..., 0])


def run_virtual_case(
    full: torch.Tensor,
    mode: str,
    world_size: int = 3,
    n_steps: int = 10,
    model: dict[str, Any] | None = None,
    collide_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    cuts: list[int] | None = None,
) -> dict[str, Any]:
    model = dict(DEFAULT_MODEL if model is None else model)
    model.setdefault("jitter_us", {})
    width = full.shape[-1]
    cuts = slab_cuts(width, world_size) if cuts is None else list(cuts)
    if len(cuts) != world_size + 1:
        raise ValueError("cuts must hold world_size + 1 boundaries")

    ranks = [full[..., cuts[r] : cuts[r + 1]].clone().contiguous() for r in range(world_size)]

    post_global: dict[str, torch.Tensor] = {}

    backends: list[VirtualClockBackend] = []
    transports: list[D3Q19OverlapTransport] = []
    for rank in range(world_size):
        owner = {"left_owner": (rank - 1) % world_size, "right_owner": (rank + 1) % world_size}

        def make_supplier(owner: dict[str, Any]):
            def _supplier(name: str, buffer: torch.Tensor) -> None:
                # ground truth: the neighbour's post-collision plane of this step
                post = post_global["post"]
                idx = (
                    cuts[owner["left_owner"] + 1] - 1
                    if name == "left"
                    else cuts[owner["right_owner"]]
                )
                buffer.copy_(post[..., idx])

            return _supplier

        backend = VirtualClockBackend(
            supplier=make_supplier(owner),
            latency_us=model["latency_us"],
            bandwidth_bytes_per_us=model["bandwidth_bytes_per_us"],
            compute_us=model["compute_us"],
            jitter_us=model.get("jitter_us", {}),
        )
        backends.append(backend)
        transports.append(D3Q19OverlapTransport(backend, mode=mode))

    reference = full.clone()
    step_rows: list[dict[str, Any]] = []
    for _ in range(n_steps):
        post_global["post"] = reference if collide_fn is None else collide_fn(reference)
        per_rank_stats = []
        for rank in range(world_size):
            ranks[rank] = transports[rank].step(ranks[rank], collide_fn)
            per_rank_stats.append(backends[rank].stats.as_dict())
        current = torch.cat(ranks, dim=-1)
        reference = monolithic_step(reference, collide_fn)
        step_rows.append(
            {
                "max_abs_error": float((current - reference).abs().max()),
                "mismatch": int((current != reference).sum().item()),
                "step_us": float(sum(s["clock_us"] for s in per_rank_stats) / world_size),
                "stall_us": float(sum(s["stall_us"] for s in per_rank_stats) / world_size),
                "stall_fraction": float(
                    sum(s["stall_fraction"] for s in per_rank_stats) / world_size
                ),
                "compute_us": {
                    key: float(sum(s["compute_us"].get(key, 0.0) for s in per_rank_stats) / world_size)
                    for key in ("collide", "stream", "pack")
                },
            }
        )

    last = step_rows[-1]
    return {
        "mode": mode,
        "world_size": world_size,
        "n_steps": n_steps,
        "cuts": cuts,
        "max_abs_error": last["max_abs_error"],
        "mismatch": last["mismatch"],
        "step_us": last["step_us"],
        "stall_us": last["stall_us"],
        "stall_fraction": last["stall_fraction"],
        "compute_us": last["compute_us"],
        "steps": step_rows,
        "final": ranks,
    }


__all__ = ["DEFAULT_MODEL", "monolithic_step", "run_virtual_case", "slab_cuts"]