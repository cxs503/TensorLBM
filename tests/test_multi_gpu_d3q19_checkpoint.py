"""Independent torchrun-job checkpoint/restart coverage for D3Q19 Gloo."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch


_WORKER = r'''
import json
import os
import sys
import torch
import torch.distributed as dist
from tensorlbm.multi_gpu import D3Q19GlooTransport


def collide(f, step):
    return f * (1.0 + step / 16.0) + step / 32.0


def initial(rank):
    torch.manual_seed(20260714)
    full = torch.randn(19, 3, 4, 10, dtype=torch.float64)
    cuts = (0, 3, 6, 10)
    return full[..., cuts[rank]:cuts[rank + 1]].clone()


mode, checkpoint, result = sys.argv[1:]
dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 3
transport = D3Q19GlooTransport()
if mode == "oracle":
    owned = initial(rank)
    for step in range(1, 5):
        owned = transport.step(owned, lambda f, step=step: collide(f, step))
elif mode == "save":
    owned = initial(rank)
    for step in range(1, 3):
        owned = transport.step(owned, lambda f, step=step: collide(f, step))
    transport.save_checkpoint(checkpoint, owned, step=2)
    if rank == 0:
        print("D3Q19_CHECKPOINT_SAVED=2", flush=True)
    dist.destroy_process_group()
    raise SystemExit(0)
elif mode == "resume":
    owned, restored_step = transport.load_checkpoint(checkpoint)
    assert restored_step == 2
    for step in range(restored_step + 1, 5):
        owned = transport.step(owned, lambda f, step=step: collide(f, step))
else:
    raise AssertionError(mode)
full = transport.gather_owned(owned)
if rank == 0:
    torch.save(full, result)
    print("D3Q19_RESTART_METRICS=" + json.dumps({
        "mode": mode, "step": 4, "owned_widths": [3, 3, 4],
        "all19_owned_elements": full.numel(),
    }, sort_keys=True), flush=True)
dist.destroy_process_group()
'''


_LOAD_WORKER = r'''
import sys
import torch.distributed as dist
from tensorlbm.multi_gpu import D3Q19GlooTransport

checkpoint = sys.argv[1]
dist.init_process_group("gloo")
try:
    D3Q19GlooTransport().load_checkpoint(checkpoint)
except RuntimeError as exc:
    print("D3Q19_LOAD_REJECTED=" + str(exc), flush=True)
    dist.destroy_process_group()
    raise SystemExit(0)
raise AssertionError("invalid checkpoint set was accepted")
'''


def _torchrun(root: Path, worker: Path, *args: Path | str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=3", str(worker), *map(str, args)],
        cwd=root, env=env, text=True, capture_output=True, timeout=120, check=False,
    )


def test_three_rank_checkpoint_restart_across_independent_torchrun_jobs_is_exact_and_fail_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    worker, loader = tmp_path / "worker.py", tmp_path / "loader.py"
    worker.write_text(_WORKER)
    loader.write_text(_LOAD_WORKER)
    checkpoint = tmp_path / "checkpoint"
    oracle, resumed = tmp_path / "oracle.pt", tmp_path / "resumed.pt"

    # Three separate process groups: uninterrupted oracle, job A save/exit,
    # then an independently initialized job B which loads and continues.
    for mode, output in (("oracle", oracle), ("save", tmp_path / "unused.pt"), ("resume", resumed)):
        run = _torchrun(root, worker, mode, checkpoint, output)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr
    assert torch.equal(torch.load(oracle, weights_only=True), torch.load(resumed, weights_only=True))
    assert torch.load(oracle, weights_only=True).numel() == 19 * 3 * 4 * 10

    # A complete set is required.  Every restarting rank validates every
    # payload before accepting its own state, so missing, mixed generation,
    # and digest-tampered members all fail closed in another torchrun job.
    rank1 = checkpoint / "rank-1.pt"
    saved = checkpoint / "rank-1.saved"
    os.replace(rank1, saved)
    for mutate, expected in ((None, "missing"), ("generation", "generation"), ("digest", "digest")):
        if mutate is not None:
            payload = torch.load(saved, weights_only=True)
            if mutate == "generation":
                payload["generation"] = "mixed-generation"
            else:
                payload["owned"][..., 0] += 1
            torch.save(payload, rank1)
        run = _torchrun(root, loader, checkpoint)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr
        assert expected in run.stdout
        if mutate is not None:
            os.replace(saved, rank1)
            os.replace(rank1, saved)
    os.replace(saved, rank1)
