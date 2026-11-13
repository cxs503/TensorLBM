"""Campaign driver: monolithic reference vs the four ghost-exchange arms.

Arms (2x2 ablation, one factor at a time on top of the shipped behaviour):

=================  ========================================================
full_alloc         shipped behaviour: 19-population boundary planes, a fresh
                   padded buffer per step with a full owned-field copy
full_persistent    same bytes, persistent padded buffer
selective_alloc    OSS PackInfo: only the 5 x-crossing populations per face,
                   fresh padded buffer
selective_persist  OSS PackInfo + persistent buffer
=================  ========================================================

Every arm runs the same collision, the same streaming and the same number of
steps; the driver only measures, compares and records.  Nothing here retunes a
KPI after seeing it (see ``predeclared.json``).
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

import torch

from .case import TGV, macroscopics, tgv_error
from .transport import SlabExchange

ARMS: Tuple[Tuple[str, bool, bool], ...] = (
    ("full_alloc", False, False),
    ("full_persistent", False, True),
    ("selective_alloc", True, False),
    ("selective_persistent", True, True),
)


# ---------------------------------------------------------------------------
# monolithic reference
# ---------------------------------------------------------------------------
def stream_periodic(post: torch.Tensor) -> torch.Tensor:
    from ..d3q19 import C

    out = torch.empty_like(post)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        out[q] = torch.roll(post[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


@dataclass
class SerialRun:
    field: torch.Tensor
    step_ms: float
    mlups: float
    series: Dict[int, float] = field(default_factory=dict)


def run_serial(case: TGV, steps: int, checkpoints: Sequence[int]) -> SerialRun:
    torch.set_num_threads(1)
    f = case.initial_field()
    marks = set(checkpoints)
    series: Dict[int, float] = {}
    started = time.perf_counter()
    for step in range(1, steps + 1):
        f = stream_periodic(case.collide(f))
        if step in marks:
            _, ux, uy, uz = macroscopics(f, case.dtype)
            series[step] = float(torch.linalg.vector_norm(torch.stack([ux, uy, uz])) / (case.n**1.5))
    elapsed = time.perf_counter() - started
    return SerialRun(
        field=f,
        step_ms=1000.0 * elapsed / steps,
        mlups=case.n**3 * steps / elapsed / 1e6,
        series=series,
    )


# ---------------------------------------------------------------------------
# distributed worker (one process per rank)
# ---------------------------------------------------------------------------
def slab_bounds(n: int, rank: int, world_size: int) -> Tuple[int, int]:
    if n % world_size:
        raise ValueError(f"n={n} must be divisible by world_size={world_size}")
    width = n // world_size
    return rank * width, (rank + 1) * width


def worker(rank: int, world_size: int, cfg: dict, out_dir: str) -> None:
    import torch.distributed as dist

    torch.set_num_threads(1)
    init_file = os.path.join(out_dir, f"gloo_init_{world_size}")
    dist.init_process_group(
        backend="gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size
    )
    try:
        case = TGV(
            n=cfg["n"],
            u0=cfg["u0"],
            tau=cfg["tau"],
            dtype=torch.float64 if cfg["dtype"] == "float64" else torch.float32,
        )
        steps = int(cfg["steps"])
        marks = set(cfg["checkpoints"])
        full = case.initial_field()
        x0, x1 = slab_bounds(case.n, rank, world_size)
        owned0 = full[..., x0:x1].contiguous()
        nz, ny = owned0.shape[1], owned0.shape[2]

        payload: Dict[str, object] = {
            "rank": rank,
            "world_size": world_size,
            "cfg": cfg,
            "slab": [x0, x1],
            "arms": {},
        }
        for name, selective, persistent in ARMS:
            exchange = SlabExchange(rank, world_size, selective=selective, persistent=persistent)
            owned = owned0.clone()
            series: Dict[int, float] = {}
            dist.barrier()
            started = time.perf_counter()
            for step in range(1, steps + 1):
                owned = exchange.step(owned, case.collide)
                if step in marks:
                    _, ux, uy, uz = macroscopics(owned, case.dtype)
                    local = (ux * ux + uy * uy + uz * uz).sum()
                    dist.all_reduce(local, op=dist.ReduceOp.SUM)
                    series[step] = float(local.sqrt() / (case.n**1.5))
            elapsed = time.perf_counter() - started

            record = {
                "step_ms": 1000.0 * elapsed / steps,
                "exchange_ms": 1000.0 * exchange.exchange_seconds / steps,
                "pack_ms": 1000.0 * exchange.pack_seconds / steps,
                "unpack_ms": 1000.0 * exchange.unpack_seconds / steps,
                "mlups": case.n**3 * steps / elapsed / 1e6,
                "bytes_per_step": exchange.bytes_last_step,
                "bytes_per_step_per_rank": exchange.bytes_last_step,
                "series": {str(k): v for k, v in series.items()},
                "byte_accounting": exchange.packed_byte_accounting(
                    nz, ny, owned.element_size()
                ),
            }
            gathered = [torch.empty_like(owned) for _ in range(world_size)]
            dist.all_gather(gathered, owned.contiguous())
            if rank == 0:
                record["field"] = torch.cat(gathered, dim=-1)
            payload["arms"][name] = record  # type: ignore[index]
        if rank == 0:
            torch.save(payload, os.path.join(out_dir, f"decomposed_{cfg['tag']}.pt"))
    finally:
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
def run_config(cfg: dict, out_dir: str) -> dict:
    import torch.multiprocessing as mp

    os.makedirs(out_dir, exist_ok=True)
    case = TGV(
        n=cfg["n"],
        u0=cfg["u0"],
        tau=cfg["tau"],
        dtype=torch.float64 if cfg["dtype"] == "float64" else torch.float32,
    )
    steps = int(cfg["steps"])
    serial = run_serial(case, steps, cfg["checkpoints"])
    serial_metrics = tgv_error(serial.field, case, steps)

    init_file = os.path.join(out_dir, f"gloo_init_{cfg['ranks']}")
    if os.path.exists(init_file):
        os.remove(init_file)
    mp.spawn(worker, args=(cfg["ranks"], cfg, out_dir), nprocs=cfg["ranks"], join=True)
    payload = torch.load(os.path.join(out_dir, f"decomposed_{cfg['tag']}.pt"), weights_only=False)

    arms: Dict[str, dict] = {}
    for name, _, _ in ARMS:
        record = dict(payload["arms"][name])
        field = record.pop("field")
        metrics = tgv_error(field, case, steps)
        arms[name] = {
            **{k: v for k, v in record.items() if k != "byte_accounting"},
            "byte_accounting": record["byte_accounting"],
            **metrics,
            "identity_max_abs_vs_serial": float((field - serial.field).abs().max()),
        }

    reference_name = "selective_persistent"
    for name in arms:
        arms[name]["identity_max_abs_vs_selective"] = float(
            (payload["arms"][name]["field"] - payload["arms"][reference_name]["field"]).abs().max()
        )

    result = {
        "tag": cfg["tag"],
        "cfg": cfg,
        "serial": {
            "step_ms": serial.step_ms,
            "mlups": serial.mlups,
            "series": {str(k): v for k, v in serial.series.items()},
            **serial_metrics,
        },
        "arms": arms,
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # keep the fields out of the JSON but on disk for figure generation
    torch.save(
        {"serial_field": serial.field, **{f"field_{n}": payload["arms"][n]["field"] for n, _, _ in ARMS}},
        os.path.join(out_dir, f"fields_{cfg['tag']}.pt"),
    )
    return result


def write_artifacts(results: List[dict], out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "results.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    rows = []
    for result in results:
        cfg = result["cfg"]
        serial = result["serial"]
        rows.append(
            {
                "run": result["tag"],
                "n": cfg["n"],
                "ranks": cfg["ranks"],
                "dtype": cfg["dtype"],
                "steps": cfg["steps"],
                "arm": "serial",
                "selective": 0,
                "persistent": 0,
                "step_ms": f"{(serial['step_ms'] if serial['step_ms'] is not None else float('nan')):.4f}",
                "exchange_ms": "",
                "mlups": f"{(serial['mlups'] if serial['mlups'] is not None else float('nan')):.2f}",
                "bytes_per_step": 0,
                "tgv_rel_l2": f"{serial['tgv_rel_l2']:.6e}",
                "tgv_amplitude_ratio": f"{serial['tgv_amplitude_ratio']:.6f}",
                "identity_max_abs_vs_serial": "",
            }
        )
        for name, selective, persistent in ARMS:
            arm = result["arms"][name]
            rows.append(
                {
                    "run": result["tag"],
                    "n": cfg["n"],
                    "ranks": cfg["ranks"],
                    "dtype": cfg["dtype"],
                    "steps": cfg["steps"],
                    "arm": name,
                    "selective": int(selective),
                    "persistent": int(persistent),
                    "step_ms": f"{arm['step_ms']:.4f}",
                    "exchange_ms": f"{arm['exchange_ms']:.4f}",
                    "mlups": f"{arm['mlups']:.2f}",
                    "bytes_per_step": arm["bytes_per_step"],
                    "tgv_rel_l2": f"{arm['tgv_rel_l2']:.6e}",
                    "tgv_amplitude_ratio": f"{arm['tgv_amplitude_ratio']:.6f}",
                    "identity_max_abs_vs_serial": f"{arm['identity_max_abs_vs_serial']:.3e}",
                }
            )
    path = os.path.join(out_dir, "table.csv")
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)