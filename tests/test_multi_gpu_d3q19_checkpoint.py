"""Three-rank CPU/Gloo D3Q19 checkpoint/restart integration coverage."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


_WORKER = r'''
import json
import os
import sys
import torch
import torch.distributed as dist
from tensorlbm.d3q19 import C
from tensorlbm.multi_gpu import D3Q19GlooTransport


def collide(f, step):
    return f * (1.0 + step / 16.0) + step / 32.0


checkpoint = sys.argv[1]
dist.init_process_group("gloo")
rank = dist.get_rank()
assert dist.get_world_size() == 3
assert rank in (0, 1, 2)
torch.manual_seed(20260714)
full = torch.randn(19, 3, 4, 10, dtype=torch.float64)
cut = (0, 3, 6, 10)
initial = full[..., cut[rank]:cut[rank + 1]].clone()

# Uninterrupted rank-local execution supplies the exact continuation oracle.
continuous = D3Q19GlooTransport()
baseline = initial.clone()
for step in range(1, 5):
    baseline = continuous.step(baseline, lambda f, step=step: collide(f, step))
baseline_global = continuous.gather_owned(baseline)

# Save at K=2, destroy/reconstruct the rank-local transport object, load, continue.
restart = D3Q19GlooTransport()
resumed = initial.clone()
for step in range(1, 3):
    resumed = restart.step(resumed, lambda f, step=step: collide(f, step))
restart.save_checkpoint(checkpoint, resumed, step=2)
del restart
restart = D3Q19GlooTransport()
resumed, restored_step = restart.load_checkpoint(checkpoint)
assert restored_step == 2
for step in range(3, 5):
    resumed = restart.step(resumed, lambda f, step=step: collide(f, step))
resumed_global = restart.gather_owned(resumed)

# Exercise fail-closed validation on rank 0 without entering load's success
# barrier: missing artifact, wrong world-size identity, and data tampering.
if rank == 0:
    original = os.path.join(checkpoint, "rank-0.pt")
    saved = original + ".saved"
    os.replace(original, saved)
    try:
        restart.load_checkpoint(checkpoint)
        raise AssertionError("missing checkpoint was accepted")
    except RuntimeError as exc:
        missing_fail_closed = "missing" in str(exc)
    payload = torch.load(saved, weights_only=True)
    payload["world_size"] = 99
    torch.save(payload, original)
    try:
        restart.load_checkpoint(checkpoint)
        raise AssertionError("wrong-world checkpoint was accepted")
    except RuntimeError as exc:
        world_fail_closed = "identity" in str(exc)
    payload["world_size"] = 3
    payload["owned"][..., 0] += 1
    torch.save(payload, original)
    try:
        restart.load_checkpoint(checkpoint)
        raise AssertionError("tampered checkpoint was accepted")
    except RuntimeError as exc:
        tamper_fail_closed = "digest" in str(exc)
    os.replace(saved, original)
    flags = torch.tensor([missing_fail_closed, world_fail_closed, tamper_fail_closed], dtype=torch.int64)
else:
    flags = torch.zeros(3, dtype=torch.int64)
dist.broadcast(flags, src=0)
assert torch.equal(flags, torch.ones(3, dtype=torch.int64))
if rank == 0:
    print("D3Q19_CHECKPOINT_METRICS=" + json.dumps({
        "step": restored_step, "owned_widths": [3, 3, 4],
        "mismatch_all19_owned": int((resumed_global != baseline_global).sum()),
        "max_abs": float((resumed_global - baseline_global).abs().max()),
        "missing_fail_closed": bool(flags[0]),
        "wrong_world_fail_closed": bool(flags[1]),
        "tamper_fail_closed": bool(flags[2]),
    }, sort_keys=True), flush=True)
assert torch.equal(resumed_global, baseline_global)
dist.destroy_process_group()
'''


def test_torchrun_gloo_three_rank_nonuniform_checkpoint_restart_exact_and_fail_closed(tmp_path: Path) -> None:
    worker = tmp_path / "gloo_checkpoint_worker.py"
    worker.write_text(_WORKER)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        ["torchrun", "--standalone", "--nproc_per_node=3", str(worker), str(tmp_path / "checkpoint")],
        cwd=root, env=env, text=True, capture_output=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert '"mismatch_all19_owned": 0' in result.stdout
    assert '"owned_widths": [3, 3, 4]' in result.stdout
    assert '"step": 2' in result.stdout
    assert '"missing_fail_closed": true' in result.stdout
    assert '"wrong_world_fail_closed": true' in result.stdout
    assert '"tamper_fail_closed": true' in result.stdout
