"""Three-rank CPU/Gloo A/B: TCLB wait-any arms vs the shipped wait-all transport.

Run under ``torchrun --standalone --nproc_per_node=3``; pytest launches it so the
comparison cannot degrade into in-process list copies.  The shipped
``D3Q19GlooTransport`` is the control: every arm must reproduce it bit for bit.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_WORKER = r"""
import json
import statistics
import time

import torch
import torch.distributed as dist

from tensorlbm.d3q19 import C
from tensorlbm.comm_overlap import D3Q19OverlapTransport, TorchDistBackend
from tensorlbm.multi_gpu import D3Q19GlooTransport


def monolithic_stream(f):
    out = torch.empty_like(f)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


def collide(f, step):
    return f * (1.0 + step / 16.0) + step / 32.0


dist.init_process_group("gloo")
rank = dist.get_rank()
world = dist.get_world_size()
assert world == 3
torch.manual_seed(20261113)
full = torch.randn(19, 3, 4, 10, dtype=torch.float64)
cut = (0, 3, 6, 10)
owned0 = full[..., cut[rank]:cut[rank + 1]].clone()
steps = 3

# ---- control: the shipped wait-all transport -------------------------------
shipped = D3Q19GlooTransport()
owned = owned0.clone()
reference = full.clone()
for step in range(1, steps + 1):
    owned = shipped.step(owned, lambda f, step=step: collide(f, step))
    reference = monolithic_stream(collide(reference, step))
control = shipped.gather_owned(owned)
metrics = {
    "world_size": world,
    "owned_widths": [cut[i + 1] - cut[i] for i in range(world)],
    "control_vs_monolithic_mismatch": int((control != reference).sum().item()),
    "arms": {},
}

# ---- arms: overlap schedules on the same traffic ---------------------------
left_rank = (rank - 1) % world
right_rank = (rank + 1) % world
send_plan = {"left": (left_rank, 101), "right": (right_rank, 102)}
recv_plan = {"left": (left_rank, 102), "right": (right_rank, 101)}

for mode in ("blocking", "waitall", "waitany"):
    backend = TorchDistBackend(send_plan=send_plan, recv_plan=recv_plan)
    transport = D3Q19OverlapTransport(backend, mode=mode)
    owned = owned0.clone()
    per_step_us = []
    stalls = []
    counts = []
    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        owned = transport.step(owned, lambda f, step=step: collide(f, step))
        per_step_us.append((time.perf_counter() - t0) * 1e6)
        stalls.append(backend.stats.stall_us)
        counts.append((backend.stats.posts_recv, backend.stats.waitany_calls,
                       backend.stats.waitall_calls))
    got = transport.gather_owned(owned) if hasattr(transport, "gather_owned") else None
    if got is None:
        width = torch.tensor([owned.shape[-1]], dtype=torch.int64)
        widths = [torch.empty_like(width) for _ in range(world)]
        dist.all_gather(widths, width)
        max_width = max(int(w.item()) for w in widths)
        packed = torch.zeros((*owned.shape[:-1], max_width), dtype=owned.dtype)
        packed[..., : owned.shape[-1]] = owned
        parts = [torch.empty_like(packed) for _ in range(world)]
        dist.all_gather(parts, packed)
        got = torch.cat([parts[r][..., : int(widths[r].item())] for r in range(world)], dim=-1)
    metrics["arms"][mode] = {
        "mismatch_vs_control": int((got != control).sum().item()),
        "mismatch_vs_monolithic": int((got != reference).sum().item()),
        "max_abs_vs_control": float((got - control).abs().max()),
        "median_step_us": statistics.median(per_step_us),
        "median_stall_us": statistics.median(stalls),
        "post_waitany_waitall_counts": counts,
    }
    dist.barrier()

if rank == 0:
    print("OVERLAP_GLOO_METRICS=" + json.dumps(metrics, sort_keys=True), flush=True)

bad = metrics["control_vs_monolithic_mismatch"] or any(
    arm["mismatch_vs_control"] or arm["mismatch_vs_monolithic"]
    for arm in metrics["arms"].values()
)
dist.destroy_process_group()
if bad:
    raise SystemExit(3)
"""


def test_torchrun_gloo_three_rank_overlap_arms_match_shipped_transport_bit_for_bit(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "overlap_gloo_worker.py"
    worker.write_text(_WORKER)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=3", str(worker)],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    line = next(ln for ln in result.stdout.splitlines() if ln.startswith("OVERLAP_GLOO_METRICS="))
    import json

    metrics = json.loads(line.split("=", 1)[1])
    assert metrics["control_vs_monolithic_mismatch"] == 0
    assert metrics["owned_widths"] == [3, 3, 4]
    for mode in ("blocking", "waitall", "waitany"):
        arm = metrics["arms"][mode]
        assert arm["mismatch_vs_control"] == 0, (mode, arm)
        assert arm["mismatch_vs_monolithic"] == 0, (mode, arm)
        assert arm["max_abs_vs_control"] == 0.0
    # wait-any must have actually been used on the real group, once per neighbour
    for count in metrics["arms"]["waitany"]["post_waitany_waitall_counts"]:
        assert list(count) == [2, 2, 1]
    for count in metrics["arms"]["waitall"]["post_waitany_waitall_counts"]:
        assert list(count) == [2, 0, 2]
    for count in metrics["arms"]["blocking"]["post_waitany_waitall_counts"]:
        assert list(count) == [2, 0, 2]