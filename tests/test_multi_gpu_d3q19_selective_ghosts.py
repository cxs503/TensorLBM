"""Shipped-transport contract for the lbmpy-derived selective ghost exchange.

The m18155 audit denied L3 integration because the selective exchange only
existed inside the ``xval`` package: no user of
``D3Q19GlooTransport.exchange_ghosts`` could reach it.  These contracts pin the
wired-in path:

1. ``selective_ghosts=True`` reproduces the shipped all-population exchange
   bit-exactly on a non-uniform 3/3/4 three-rank ring (decomposition may change
   data movement, never arithmetic);
2. the wire volume per exchange is exactly ``(5 + 5) / (19 + 19) = 5/19`` of the
   all-population exchange;
3. the default constructor is unchanged (all 19 populations, no behaviour
   change for existing callers), and a selective transport still fails closed
   when a ghost is corrupted inside the crossing set.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

_WORKER = r"""
import json
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
assert dist.get_world_size() == 3
torch.manual_seed(20261113)  # identical field on every rank
full = torch.randn(19, 3, 4, 10, dtype=torch.float64)
cut = (0, 3, 6, 10)
owned = full[..., cut[rank]:cut[rank + 1]].clone()

# The three arms run SEQUENTIALLY: all of them use the same point-to-point
# tags on the same ring, so interleaving them would pair a send of one arm with
# a receive of another.  One arm at a time keeps the ablation honest.
ARMS = (
    ("shipped", dict()),
    ("selective", dict(selective_ghosts=True)),
    ("selective_persistent", dict(selective_ghosts=True, persistent_ghost_buffer=True)),
)


def run_arm(kwargs, steps=4):
    transport = D3Q19GlooTransport(**kwargs)
    state = owned.clone()
    reference = full.clone()
    out = []
    for step in range(1, steps + 1):
        collide = lambda f, s=step: f * (1.0 + s / 16.0) + s / 32.0
        state = transport.step(state, collide)
        reference = monolithic_stream(collide(reference))
        gathered = transport.gather_owned(state)
        out.append({
            "step": step,
            "mismatch": int((gathered != reference).sum().item()),
            "max_abs": float((gathered - reference).abs().max()),
            "bytes": transport.last_exchange_bytes,
            "selective": transport.selective_ghosts,
        })
    return transport, out


records = {}
for name, kwargs in ARMS:
    transport, series = run_arm(kwargs)
    records[name] = series
    if name == "shipped":
        assert transport.selective_ghosts is False
selective = D3Q19GlooTransport(selective_ghosts=True)

# A corrupted crossing ghost must fail closed, not be silently streamed.
padded = selective.exchange_ghosts(owned.clone())
tampered = padded.clone()
tampered[list(selective.crossing_right), :, :, 0] += 1.0
corruption_raised = False
try:
    selective.validate_ghosts(tampered)
except RuntimeError:
    corruption_raised = True

bytes_shipped = records["shipped"][0]["bytes"]
bytes_selective = records["selective"][0]["bytes"]
summary = {
    "records": records,
    "corruption_raised": corruption_raised,
    "crossing_right": list(selective.crossing_right),
    "crossing_left": list(selective.crossing_left),
    "bytes_shipped": bytes_shipped,
    "bytes_selective": bytes_selective,
    "ratio": bytes_selective / bytes_shipped if bytes_shipped else None,
}
if rank == 0:
    print("D3Q19_SELECTIVE_METRICS=" + json.dumps(summary, sort_keys=True), flush=True)
dist.destroy_process_group()
bad = [r for series in records.values() for r in series if r["mismatch"]]
if bad or not corruption_raised:
    raise SystemExit(3)
"""


def test_torchrun_selective_ghost_exchange_is_bit_exact_and_5_of_19(
    tmp_path: Path,
) -> None:
    worker = tmp_path / "selective_worker.py"
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
        timeout=600,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    marker = [line for line in result.stdout.splitlines() if line.startswith("D3Q19_SELECTIVE_METRICS=")]
    assert marker, result.stdout
    import json

    summary = json.loads(marker[0].split("=", 1)[1])
    assert summary["corruption_raised"] is True
    assert len(summary["crossing_right"]) == 5
    assert len(summary["crossing_left"]) == 5
    for name, series in summary["records"].items():
        for record in series:
            assert record["mismatch"] == 0, (name, record)
    assert summary["records"]["shipped"][0]["selective"] is False
    assert summary["records"]["selective"][0]["selective"] is True
    assert summary["ratio"] == pytest.approx(5.0 / 19.0, rel=0, abs=1e-15)


def test_selective_flags_are_opt_in_and_typed() -> None:
    """The flags are keyword-only booleans; the shipped path stays the default."""
    import inspect

    from tensorlbm.multi_gpu import D3Q19GlooTransport

    signature = inspect.signature(D3Q19GlooTransport.__init__)
    assert signature.parameters["selective_ghosts"].default is False
    assert signature.parameters["persistent_ghost_buffer"].default is False
    assert all(
        signature.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        for name in ("selective_ghosts", "persistent_ghost_buffer")
    )