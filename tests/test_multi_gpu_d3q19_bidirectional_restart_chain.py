"""Separate CPU/Gloo jobs prove exact bidirectional D3Q19 restart chaining."""
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

CUTS = {2: (0, 5, 10), 3: (0, 3, 6, 10)}
TOTAL, FIRST_SAVE, SECOND_SAVE = 20, 10, 15

def collide(f, step):
    return f * (1.0 + step / 16.0) + step / 32.0

def initial(x0, x1):
    torch.manual_seed(20260715)
    return torch.randn(19, 3, 4, 10, dtype=torch.float64)[..., x0:x1].clone()

def stream(f):
    out = torch.empty_like(f)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out

def digest(f):
    return hashlib.sha256(f.contiguous().numpy().tobytes()).hexdigest()

mode, checkpoint_a, checkpoint_b, result = sys.argv[1:]
dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
transport = D3Q19GlooTransport()
reference = torch.cat([initial(CUTS[world][r], CUTS[world][r + 1]) for r in range(world)], dim=-1)
if mode == "oracle3":
    assert world == 3
    owned = initial(CUTS[3][rank], CUTS[3][rank + 1])
    start = 1
elif mode == "save3":
    assert world == 3
    owned = initial(CUTS[3][rank], CUTS[3][rank + 1])
    start = 1
elif mode == "middle2":
    assert world == 2
    owned, restored = transport.load_repartition_checkpoint(checkpoint_a, source_world_size=3)
    assert restored == FIRST_SAVE
    start = restored + 1
elif mode == "finish3":
    assert world == 3
    owned, restored = transport.load_repartition_checkpoint(checkpoint_b, source_world_size=2)
    assert restored == SECOND_SAVE
    start = restored + 1
else:
    raise AssertionError(mode)

trace = []
for step in range(1, TOTAL + 1):
    reference = stream(collide(reference, step))
    if step < start:
        continue
    owned = transport.step(owned, lambda f, step=step: collide(f, step))
    actual = transport.gather_owned(owned)
    mismatch = int((actual != reference).sum().item())
    assert mismatch == 0, (mode, step, mismatch)
    item = {"step": step, "world": world, "owned_widths": list(CUTS[world][i + 1] - CUTS[world][i] for i in range(world)),
            "all19_owned_elements": actual.numel(), "mismatch_all19_owned": mismatch, "state_digest": digest(actual)}
    trace.append(item)
    if mode == "save3" and step == FIRST_SAVE:
        transport.save_checkpoint(checkpoint_a, owned, step=step)
        break
    if mode == "middle2" and step == SECOND_SAVE:
        transport.save_checkpoint(checkpoint_b, owned, step=step)
        break
if rank == 0:
    torch.save({"full": actual, "trace": trace}, result)
    print("D3Q19_BIDIRECTIONAL_CHAIN=" + json.dumps({"mode": mode, "last": trace[-1]}, sort_keys=True), flush=True)
dist.destroy_process_group()
'''

_LOAD_WORKER = r'''
import sys
import torch.distributed as dist
from tensorlbm.multi_gpu import D3Q19GlooTransport

dist.init_process_group("gloo")
try:
    D3Q19GlooTransport().load_repartition_checkpoint(sys.argv[1], source_world_size=int(sys.argv[2]))
except RuntimeError as exc:
    print("D3Q19_CHAIN_LOAD_REJECTED=" + str(exc), flush=True)
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


def test_three_to_two_to_new_three_rank_d3q19_restart_chain_is_exact_and_fail_closed(tmp_path: Path) -> None:
    """Each generation is a new torchrun PG; both repartitions own all 19 populations."""
    root = Path(__file__).resolve().parents[1]
    worker, loader = tmp_path / "chain.py", tmp_path / "load.py"
    worker.write_text(_WORKER)
    loader.write_text(_LOAD_WORKER)
    checkpoint_a, checkpoint_b = tmp_path / "checkpoint-3", tmp_path / "checkpoint-2"
    oracle, saved, middle, finished = (tmp_path / name for name in ("oracle.pt", "saved.pt", "middle.pt", "finished.pt"))

    jobs = (("oracle3", 3, oracle), ("save3", 3, saved), ("middle2", 2, middle), ("finish3", 3, finished))
    runs = {}
    for mode, ranks, output in jobs:
        run = _torchrun(root, ranks, worker, mode, checkpoint_a, checkpoint_b, output)
        assert run.returncode == 0, run.stdout + "\n" + run.stderr
        assert '"mismatch_all19_owned": 0' in run.stdout
        runs[mode] = run

    oracle_result = torch.load(oracle, weights_only=True)
    saved_result = torch.load(saved, weights_only=True)
    middle_result = torch.load(middle, weights_only=True)
    finished_result = torch.load(finished, weights_only=True)
    assert torch.equal(oracle_result["full"], finished_result["full"])
    assert oracle_result["full"].numel() == 19 * 3 * 4 * 10
    assert [item["step"] for item in saved_result["trace"]] == list(range(1, 11))
    assert [item["step"] for item in middle_result["trace"]] == list(range(11, 16))
    assert [item["step"] for item in finished_result["trace"]] == list(range(16, 21))
    assert '"owned_widths": [5, 5]' in runs["middle2"].stdout
    assert '"owned_widths": [3, 3, 4]' in runs["finish3"].stdout

    # Every source member is mandatory at both transitions; generation and
    # member digests are also independently fail-closed before repartition.
    for checkpoint, source_world, target_world in ((checkpoint_a, 3, 2), (checkpoint_b, 2, 3)):
        victim = checkpoint / "rank-1.pt"
        pristine = torch.load(victim, weights_only=True)
        os.unlink(victim)
        for mutation, expected in (("missing", "missing"), ("generation", "generation"), ("digest", "digest")):
            if mutation != "missing":
                payload = dict(pristine)
                if mutation == "generation":
                    payload["generation"] = "mixed-generation"
                else:
                    payload["owned"] = payload["owned"].clone()
                    payload["owned"][..., 0] += 1
                torch.save(payload, victim)
            run = _torchrun(root, target_world, loader, checkpoint, str(source_world))
            assert run.returncode == 0, run.stdout + "\n" + run.stderr
            assert expected in run.stdout
            if victim.exists():
                os.unlink(victim)
        torch.save(pristine, victim)
