"""Independent long-run torchrun-job checkpoint/restart coverage for D3Q19 Gloo."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import torch


_WORKER = r'''
import hashlib
import json
import sys
import torch
import torch.distributed as dist
from tensorlbm.d3q19 import C
from tensorlbm.multi_gpu import D3Q19GlooTransport

TOTAL_STEPS = 20
SAVE_STEP = 10
CUTS = (0, 3, 6, 10)


def collide(f, step):
    return f * (1.0 + step / 16.0) + step / 32.0


def initial(rank):
    torch.manual_seed(20260714)
    full = torch.randn(19, 3, 4, 10, dtype=torch.float64)
    return full[..., CUTS[rank]:CUTS[rank + 1]].clone()


def monolithic_stream(f):
    out = torch.empty_like(f)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


def state_digest(f):
    return hashlib.sha256(f.contiguous().numpy().tobytes()).hexdigest()


mode, checkpoint, result = sys.argv[1:]
dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 3
transport = D3Q19GlooTransport()
reference = torch.cat([initial(member) for member in range(world)], dim=-1)
if mode == "resume":
    owned, restored_step = transport.load_checkpoint(checkpoint)
    assert restored_step == SAVE_STEP
    start_step = restored_step + 1
else:
    owned = initial(rank)
    start_step = 1
trace = []
for step in range(1, TOTAL_STEPS + 1):
    reference = monolithic_stream(collide(reference, step))
    if step < start_step:
        continue
    owned = transport.step(owned, lambda f, step=step: collide(f, step))
    actual = transport.gather_owned(owned)
    mismatch = int((actual != reference).sum().item())
    assert mismatch == 0, (step, mismatch)
    item = {
        "step": step,
        "owned_widths": [3, 3, 4],
        "all19_owned_elements": actual.numel(),
        "mismatch_all19_owned": mismatch,
        "state_digest": state_digest(actual),
    }
    trace.append(item)
    if mode == "save" and step == SAVE_STEP:
        transport.save_checkpoint(checkpoint, owned, step=step)
        if rank == 0:
            print("D3Q19_LONGRUN_CHECKPOINT=" + json.dumps(item, sort_keys=True), flush=True)
        dist.destroy_process_group()
        raise SystemExit(0)
if mode not in ("oracle", "resume"):
    raise AssertionError(mode)
if rank == 0:
    torch.save({"full": actual, "trace": trace}, result)
    print("D3Q19_LONGRUN_METRICS=" + json.dumps({
        "mode": mode, "first_checked_step": trace[0]["step"],
        "final_step": trace[-1]["step"], "final_state_digest": trace[-1]["state_digest"],
        "mismatch_all19_owned": trace[-1]["mismatch_all19_owned"],
        "owned_widths": [3, 3, 4], "all19_owned_elements": actual.numel(),
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
        cwd=root, env=env, text=True, capture_output=True, timeout=180, check=False,
    )


def test_three_rank_20_step_restart_across_independent_torchrun_jobs_is_exact_and_fail_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    worker, loader = tmp_path / "worker.py", tmp_path / "loader.py"
    worker.write_text(_WORKER)
    loader.write_text(_LOAD_WORKER)
    checkpoint = tmp_path / "checkpoint"
    oracle, resumed = tmp_path / "oracle.pt", tmp_path / "resumed.pt"

    # Three separate process groups: uninterrupted 20-step oracle, job A
    # saves/exits at step 10, and fresh job B restores then runs steps 11..20.
    runs = {}
    for mode, output in (("oracle", oracle), ("save", tmp_path / "unused.pt"), ("resume", resumed)):
        run = _torchrun(root, worker, mode, checkpoint, output)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr
        runs[mode] = run
    oracle_result = torch.load(oracle, weights_only=True)
    resumed_result = torch.load(resumed, weights_only=True)
    assert torch.equal(oracle_result["full"], resumed_result["full"])
    assert oracle_result["full"].numel() == 19 * 3 * 4 * 10
    assert [item["step"] for item in oracle_result["trace"]] == list(range(1, 21))
    assert [item["step"] for item in resumed_result["trace"]] == list(range(11, 21))
    assert resumed_result["trace"] == oracle_result["trace"][10:]
    assert '"final_step": 20' in runs["oracle"].stdout
    assert '"final_step": 20' in runs["resume"].stdout
    assert '"step": 10' in runs["save"].stdout
    assert '"mismatch_all19_owned": 0' in runs["oracle"].stdout
    assert '"final_state_digest"' in runs["resume"].stdout

    # Every fresh restarting rank validates every rank-local member. Missing,
    # mixed-generation, and payload-tampered sets must all fail closed.
    rank1 = checkpoint / "rank-1.pt"
    pristine = torch.load(rank1, weights_only=True)
    os.unlink(rank1)
    for mutation, expected in (("missing", "missing"), ("generation", "generation"), ("digest", "digest")):
        if mutation != "missing":
            payload = dict(pristine)
            if mutation == "generation":
                payload["generation"] = "mixed-generation"
            else:
                payload["owned"] = payload["owned"].clone()
                payload["owned"][..., 0] += 1
            torch.save(payload, rank1)
        run = _torchrun(root, loader, checkpoint)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr
        assert expected in run.stdout
        if rank1.exists():
            os.unlink(rank1)
    torch.save(pristine, rank1)
