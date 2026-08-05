"""Independent torchrun coverage for a 3-rank to 2-rank D3Q19 restart."""
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

TOTAL_STEPS, SAVE_STEP = 20, 10
SOURCE_CUTS, TARGET_CUTS = (0, 3, 6, 10), (0, 5, 10)

def collide(f, step):
    return f * (1.0 + step / 16.0) + step / 32.0

def initial(x0, x1):
    torch.manual_seed(20260714)
    return torch.randn(19, 3, 4, 10, dtype=torch.float64)[..., x0:x1].clone()

def stream(f):
    out = torch.empty_like(f)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out

def digest(f):
    return hashlib.sha256(f.contiguous().numpy().tobytes()).hexdigest()

mode, checkpoint, result = sys.argv[1:]
dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
transport = D3Q19GlooTransport()
reference = torch.cat([initial(TARGET_CUTS[r], TARGET_CUTS[r + 1]) for r in range(2)], dim=-1)
if mode == "save3":
    assert world == 3
    owned = initial(SOURCE_CUTS[rank], SOURCE_CUTS[rank + 1])
    start = 1
elif mode == "oracle2":
    assert world == 2
    owned = initial(TARGET_CUTS[rank], TARGET_CUTS[rank + 1])
    start = 1
elif mode == "resume2":
    assert world == 2
    owned, restored = transport.load_repartition_checkpoint(checkpoint, source_world_size=3)
    assert restored == SAVE_STEP
    assert owned.shape[-1] == 5
    start = restored + 1
else:
    raise AssertionError(mode)
trace = []
for step in range(1, TOTAL_STEPS + 1):
    reference = stream(collide(reference, step))
    if step < start:
        continue
    owned = transport.step(owned, lambda f, step=step: collide(f, step))
    actual = transport.gather_owned(owned)
    mismatch = int((actual != reference).sum().item())
    assert mismatch == 0, (step, mismatch)
    item = {"step": step, "owned_widths": [5, 5], "all19_owned_elements": actual.numel(),
            "mismatch_all19_owned": mismatch, "state_digest": digest(actual)}
    trace.append(item)
    if mode == "save3" and step == SAVE_STEP:
        transport.save_checkpoint(checkpoint, owned, step=step)
        if rank == 0:
            print("D3Q19_REPARTITION_CHECKPOINT=" + json.dumps(item, sort_keys=True), flush=True)
        dist.destroy_process_group()
        raise SystemExit(0)
if rank == 0:
    torch.save({"full": actual, "trace": trace}, result)
    print("D3Q19_REPARTITION_METRICS=" + json.dumps({"mode": mode, "final_step": trace[-1]["step"],
          "final_state_digest": trace[-1]["state_digest"], "owned_widths": [5, 5],
          "mismatch_all19_owned": trace[-1]["mismatch_all19_owned"]}, sort_keys=True), flush=True)
dist.destroy_process_group()
'''

_LOAD_WORKER = r'''
import sys
import torch.distributed as dist
from tensorlbm.multi_gpu import D3Q19GlooTransport

dist.init_process_group("gloo")
try:
    D3Q19GlooTransport().load_repartition_checkpoint(sys.argv[1], source_world_size=3)
except RuntimeError as exc:
    print("D3Q19_REPARTITION_LOAD_REJECTED=" + str(exc), flush=True)
    dist.destroy_process_group()
    raise SystemExit(0)
raise AssertionError("invalid source checkpoint set was accepted")
'''


def _torchrun(root: Path, ranks: int, worker: Path, *args: Path | str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        ["torchrun", "--standalone", f"--nproc_per_node={ranks}", str(worker), *map(str, args)],
        cwd=root, env=env, text=True, capture_output=True, timeout=180, check=False,
    )


def test_three_rank_checkpoint_repartitions_to_two_rank_restart_exactly_and_fails_closed(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    worker, loader = tmp_path / "worker.py", tmp_path / "loader.py"
    worker.write_text(_WORKER)
    loader.write_text(_LOAD_WORKER)
    checkpoint = tmp_path / "checkpoint"
    oracle, resumed = tmp_path / "oracle.pt", tmp_path / "resumed.pt"

    runs = {}
    for mode, ranks, output in (("oracle2", 2, oracle), ("save3", 3, tmp_path / "unused.pt"), ("resume2", 2, resumed)):
        run = _torchrun(root, ranks, worker, mode, checkpoint, output)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr
        runs[mode] = run
    oracle_result = torch.load(oracle, weights_only=True)
    resumed_result = torch.load(resumed, weights_only=True)
    assert torch.equal(oracle_result["full"], resumed_result["full"])
    assert oracle_result["full"].numel() == 19 * 3 * 4 * 10
    assert resumed_result["trace"] == oracle_result["trace"][10:]
    assert [item["step"] for item in resumed_result["trace"]] == list(range(11, 21))
    assert '"owned_widths": [5, 5]' in runs["resume2"].stdout
    assert '"mismatch_all19_owned": 0' in runs["resume2"].stdout

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
        run = _torchrun(root, 2, loader, checkpoint)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr
        assert expected in run.stdout
        if rank1.exists():
            os.unlink(rank1)
    torch.save(pristine, rank1)
