"""Actual three-rank CPU/Gloo D3Q19 transport integration coverage.

Run directly with ``torchrun --standalone --nproc_per_node=3``; pytest starts
that command so the tested transport cannot be substituted with in-process
list copies.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


_WORKER = r'''
import json
import os
import torch
import torch.distributed as dist
from tensorlbm.d3q19 import C
from tensorlbm.multi_gpu import D3Q19GlooTransport


def monolithic_stream(f):
    out = torch.empty_like(f)
    for q, (cx, cy, cz) in enumerate(C.tolist()):
        out[q] = torch.roll(f[q], shifts=(cz, cy, cx), dims=(0, 1, 2))
    return out


dist.init_process_group("gloo")
rank = dist.get_rank()
torch.manual_seed(20260714)
# The 3/3/4 cut is intentionally non-uniform.  The ring transports only to
# rank-neighbours, while gathering can reconstruct variable-width ownership.
assert dist.get_world_size() == 3
full = torch.randn(19, 3, 4, 10, dtype=torch.float64)
cut = (0, 3, 6, 10)
owned = full[..., cut[rank]:cut[rank + 1]].clone()
transport = D3Q19GlooTransport()
reference = full.clone()
metrics = []
for step in range(1, 4):
    # Non-identity collision proves ghosts carry post-collision populations.
    owned = transport.step(owned, lambda f, step=step: f * (1.0 + step / 16.0) + step / 32.0)
    reference = reference * (1.0 + step / 16.0) + step / 32.0
    reference = monolithic_stream(reference)
    actual = transport.gather_owned(owned)
    mismatch = int((actual != reference).sum().item())
    metrics.append({
        "step": step,
        "owned_widths": [3, 3, 4],
        "nx_global": actual.shape[-1],
        "mismatch_all19_owned": mismatch,
        "periodic_x_edge_mismatch": int((actual[..., (0, -1)] != reference[..., (0, -1)]).sum().item()),
        "max_abs": float((actual-reference).abs().max()),
    })
if rank == 0:
    print("D3Q19_GLOO_METRICS=" + json.dumps(metrics, sort_keys=True), flush=True)
dist.destroy_process_group()
if any(item["mismatch_all19_owned"] for item in metrics):
    raise SystemExit(3)
'''


@pytest.mark.parametrize("run_under_torchrun", [True])
def test_torchrun_gloo_three_rank_nonuniform_3_3_4_all19_owned_equivalence_for_1_2_3_steps(
    tmp_path: Path, run_under_torchrun: bool
) -> None:
    worker = tmp_path / "gloo_equivalence_worker.py"
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
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert '"mismatch_all19_owned": 0' in result.stdout
    assert '"step": 1' in result.stdout
    assert '"step": 2' in result.stdout
    assert '"step": 3' in result.stdout
    assert '"owned_widths": [3, 3, 4]' in result.stdout
    assert '"nx_global": 10' in result.stdout
    assert '"periodic_x_edge_mismatch": 0' in result.stdout
