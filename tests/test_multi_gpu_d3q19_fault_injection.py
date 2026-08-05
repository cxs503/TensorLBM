"""Three-rank fault injection: corrupted received D3Q19 ghosts must fail closed."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


_WORKER = r'''
import os
import torch
import torch.distributed as dist
from tensorlbm.multi_gpu import D3Q19GlooTransport

dist.init_process_group("gloo")
rank = dist.get_rank()
transport = D3Q19GlooTransport()
assert dist.get_world_size() == 3
owned = torch.full((19, 2, 3, (3, 3, 4)[rank]), float(rank + 1), dtype=torch.float64)
padded = transport.exchange_ghosts(owned)
if rank == 0:
    padded[7, 0, 0, 0] += 1.0
failed_closed = False
try:
    transport.validate_ghosts(padded)
except RuntimeError as exc:
    failed_closed = "ghost validation failed" in str(exc)
if rank == 0 and not failed_closed:
    raise SystemExit("corrupt ghost was accepted")
if rank == 0:
    print("D3Q19_GLOO_CORRUPT_GHOST_REJECTED", flush=True)
    raise SystemExit(17)
dist.destroy_process_group()
'''


def test_torchrun_gloo_corrupt_ghost_exits_nonzero(tmp_path: Path) -> None:
    worker = tmp_path / "gloo_corrupt_ghost_worker.py"
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
    assert result.returncode != 0, result.stdout + "\n" + result.stderr
    assert "D3Q19_GLOO_CORRUPT_GHOST_REJECTED" in result.stdout
