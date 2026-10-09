"""Branch-selection and no-transient regressions for the distributed
SUBOFF wrapper's global-mass reduction (``_current_global_mass``).

Production OOM background (n=1024, world=8, 31.4 GiB cards): the method
used to call ``owned_full.contiguous()`` unconditionally.  Only the
gather (mass-bitwise) branch needs the dense copy (NCCL all_gather
operand + single-GPU concatenation layout); the all_reduce fallback
branch — auto-selected at production scale because the 80.5 GiB global
gather does not fit — paid a full owned-slab transient (9.5 GiB at
n=1024/w=8) on every mass step and OOMed the stock wrapper at the first
mass-correction step (step 10).  Reproduction artifact:
``triton_bench_20260819/ac_n1024_fullstack/perf_oom_repro.json``.

These tests pin the branch behaviour at small n (no n=1024 in CI):

* world==1 keeps the stock contiguous+sum order bit-for-bit;
* the all_reduce branch reduces the strided view directly — no
  contiguous materialisation at all;
* the gather branch still hands the collective a dense operand and
  sums the globally ordered concatenation;
* ``_gather_fits_memory`` flips to the all_reduce branch exactly when
  the gathered global tensor would not fit in free GPU memory;
* a real 3-rank Gloo torchrun reproduces both branches end-to-end with
  rank-distinct slabs and cross-rank-identical results.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from tensorlbm.triton_suboff_step_distributed import (
    TritonSuboffDistributedRunner,
)


def _runner_shell(world_size: int, gather: bool, rank: int = 0):
    """A runner with only the attrs ``_current_global_mass`` touches.

    The full constructor needs a GPU solver stack and an initialized
    process group; the method under test only reads ``world_size`` and
    ``_mass_reduce_gather`` (plus ``rank``/``world_size`` for the
    branch-selection helper).
    """
    runner = object.__new__(TritonSuboffDistributedRunner)
    runner.world_size = world_size
    runner.rank = rank
    runner._mass_reduce_gather = gather
    return runner


def _halo_owned_view(seed: int, nz_local: int = 5, ny: int = 7, nx: int = 9, q: int = 19):
    """A halo-padded buffer and its (strided) owned-plane view."""
    gen = torch.Generator().manual_seed(seed)
    buf = torch.randn((q, nz_local + 2, ny, nx), generator=gen, dtype=torch.float32)
    owned = buf[:, 1 : nz_local + 1, :, :]
    assert not owned.is_contiguous(), "fixture must exercise the strided view"
    return buf, owned


@pytest.fixture
def contiguous_calls(monkeypatch):
    """Record shapes of Python-level ``Tensor.contiguous()`` calls.

    Native (C++) internal ``.contiguous()`` uses inside torch ops do not
    route through the Python attribute, so the spy sees exactly the
    calls the wrapper's own Python code makes — the transients we want
    to forbid.
    """
    calls: list[tuple[int, ...]] = []
    original = torch.Tensor.contiguous

    def spy(self, *args, **kwargs):
        calls.append(tuple(self.shape))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "contiguous", spy)
    return calls


def test_world1_mass_branch_keeps_stock_contiguous_sum_bitwise():
    """world==1 path is untouched: contiguous+sum, bitwise stock order."""
    _buf, owned = _halo_owned_view(seed=101)
    expected = owned.contiguous().sum()  # the pre-fix stock computation
    runner = _runner_shell(world_size=1, gather=True)
    cur = runner._current_global_mass(owned)
    assert cur.ndim == 0 and cur.dtype == torch.float32
    assert torch.equal(cur, expected)


def test_world1_mass_branch_still_pays_one_contiguous_copy(contiguous_calls):
    """The world==1 branch keeps its exact original statement sequence."""
    _buf, owned = _halo_owned_view(seed=102)
    runner = _runner_shell(world_size=1, gather=True)
    runner._current_global_mass(owned)
    assert contiguous_calls.count(tuple(owned.shape)) == 1


def test_all_reduce_branch_sums_strided_view_without_contiguous_transient(
    monkeypatch, contiguous_calls
):
    """all_reduce fallback: no dense copy of the owned slab at all.

    Pre-fix this branch materialised ``owned_full.contiguous()`` — the
    9.5 GiB transient at n=1024/w=8 that OOMed the first mass step.
    ``torch.sum`` accepts the strided view and the collective only ever
    sees the 0-d scalar, so no NCCL layout requirement applies.
    """
    _buf, owned = _halo_owned_view(seed=103)
    world = 4
    runner = _runner_shell(world_size=world, gather=False)
    seen: dict = {}

    def fake_all_reduce(tensor, op=None, **kwargs):
        seen["ndim"] = tensor.dim()
        seen["dtype"] = tensor.dtype
        seen["contiguous"] = tensor.is_contiguous()
        tensor.mul_(world)  # emulate `world` ranks holding this slab
        return tensor

    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    cur = runner._current_global_mass(owned)

    # No Python-level contiguous call anywhere on the owned-slab shape.
    assert tuple(owned.shape) not in contiguous_calls
    assert not contiguous_calls, (
        "all_reduce branch must not materialise any dense copy: "
        f"got calls for shapes {contiguous_calls}"
    )
    # The collective operand is the 0-d fp32 scalar, never the slab.
    assert seen["ndim"] == 0 and seen["dtype"] == torch.float32
    assert seen["contiguous"]
    # Value: world * (strided per-rank sum); x4 is exact in fp32.
    assert torch.equal(cur, owned.sum() * world)
    assert cur.ndim == 0 and cur.dtype == torch.float32


def test_gather_branch_passes_dense_operand_and_global_order_sum(monkeypatch, contiguous_calls):
    """gather branch: NCCL operand stays dense, global-z concatenation sum.

    all_gather requires a contiguous tensor, and the globally ordered
    concatenation must reproduce the single-GPU reduction layout — this
    is the one branch that legitimately pays for the copy.
    """
    _buf, owned = _halo_owned_view(seed=104)
    world = 4
    runner = _runner_shell(world_size=world, gather=True)
    operands: list[torch.Tensor] = []

    def fake_all_gather(parts, tensor, **kwargs):
        assert tensor.is_contiguous(), "all_gather operand must be contiguous (NCCL requirement)"
        operands.append(tensor.clone())
        for part in parts:  # every rank contributes the same slab here
            part.copy_(tensor)
        return None

    monkeypatch.setattr(dist, "all_gather", fake_all_gather)
    cur = runner._current_global_mass(owned)

    assert contiguous_calls.count(tuple(owned.shape)) == 1
    assert len(operands) == 1 and operands[0].is_contiguous()
    expected = torch.cat([owned.contiguous()] * world, dim=1).sum()
    assert torch.equal(cur, expected)


def test_gather_fits_memory_flips_to_all_reduce_when_gather_would_oom(monkeypatch):
    """Branch selection: gather only while the global tensor fits.

    Mirrors the n=1024/w=8 decision: an 80.5 GiB global (Q,nz,ny,nx)
    gather (2x transient + 2 GiB headroom) against ~11 GiB free on a
    31.4 GiB card must fall back to all_reduce, while a validation-size
    cube with plenty free stays on the bitwise gather path.
    """
    runner = _runner_shell(world_size=8, gather=True, rank=0)
    q = 19
    n_val = 64
    fits_bytes = q * n_val**3 * 4
    device = torch.device("cuda")

    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda dev: (2 * fits_bytes + (2 << 30) + (1 << 30), 100 << 30)
    )
    assert runner._gather_fits_memory(q, n_val, n_val, n_val, device) is True

    # n=1024 cube: global = 80.5 GiB, free-after-steady ~11.4 GiB.
    n_prod = 1024
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda dev: (11_400_000_000, 31_400_000_000))
    assert runner._gather_fits_memory(q, n_prod, n_prod, n_prod, device) is False

    # CPU tensors are never gather-constrained (host RAM).
    cpu = torch.device("cpu")
    assert runner._gather_fits_memory(q, n_prod, n_prod, n_prod, cpu) is True

    # world==1 never gathers.
    single = _runner_shell(world_size=1, gather=True, rank=0)
    assert single._gather_fits_memory(q, n_prod, n_prod, n_prod, device) is True


_RANK_WORKER = r"""
import json
import torch
import torch.distributed as dist
from tensorlbm.triton_suboff_step_distributed import (
    TritonSuboffDistributedRunner,
)

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
assert world == 3

q, nz_local, ny, nx = 19, 5, 7, 9
gen = torch.Generator().manual_seed(1234 + rank)  # rank-DISTINCT slabs
buf = torch.randn((q, nz_local + 2, ny, nx), generator=gen)
owned = buf[:, 1:nz_local + 1, :, :]
assert not owned.is_contiguous()

runner = object.__new__(TritonSuboffDistributedRunner)
runner.world_size = world
runner.rank = rank

# --- all_reduce branch: forbid any contiguous materialisation -------
runner._mass_reduce_gather = False
calls = []
original = torch.Tensor.contiguous
def spy(self, *a, **k):
    calls.append(tuple(self.shape))
    return original(self, *a, **k)
torch.Tensor.contiguous = spy
cur_ar = runner._current_global_mass(owned)
torch.Tensor.contiguous = original
no_transient = tuple(owned.shape) not in calls

# --- gather branch (reference semantics, same buffers) --------------
runner._mass_reduce_gather = True
cur_ga = runner._current_global_mass(owned)

# fp64 ground truth of the global mass over all three distinct slabs.
part64 = owned.double().sum()
dist.all_reduce(part64)
ref = float(part64)

def identical_across_ranks(value):
    vals = [torch.zeros(1) for _ in range(world)]
    dist.all_gather(vals, value.reshape(1).clone())
    return all(abs(float(v[0]) - float(value)) == 0.0 for v in vals)

metrics = {
    "all_reduce_no_contiguous_transient": bool(no_transient),
    "all_reduce_result_0d_f32": bool(
        cur_ar.dim() == 0 and cur_ar.dtype == torch.float32),
    "gather_rel_err_vs_fp64": abs(float(cur_ga) - ref) / abs(ref),
    "all_reduce_rel_err_vs_fp64": abs(float(cur_ar) - ref) / abs(ref),
    "gather_identical_across_ranks": identical_across_ranks(cur_ga),
    "all_reduce_identical_across_ranks": identical_across_ranks(cur_ar),
    "branches_relative_gap": abs(float(cur_ar) - float(cur_ga))
                             / abs(float(cur_ga)),
}
if rank == 0:
    print("MASS_REDUCE_METRICS=" + json.dumps(metrics, sort_keys=True),
          flush=True)
dist.destroy_process_group()
ok = (
    metrics["all_reduce_no_contiguous_transient"]
    and metrics["all_reduce_result_0d_f32"]
    and metrics["gather_rel_err_vs_fp64"] < 1e-6
    and metrics["all_reduce_rel_err_vs_fp64"] < 1e-6
    and metrics["gather_identical_across_ranks"]
    and metrics["all_reduce_identical_across_ranks"]
)
raise SystemExit(0 if ok else 3)
"""


def test_torchrun_gloo_three_rank_mass_reduce_branches(tmp_path: Path) -> None:
    """Real 3-rank Gloo run of both branches with rank-distinct slabs.

    Follows the repo's torchrun-subprocess convention (see
    ``test_multi_gpu_d3q19_torchrun.py``) so the collectives cannot be
    substituted with in-process list copies.
    """
    worker = tmp_path / "mass_reduce_worker.py"
    worker.write_text(_RANK_WORKER)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=3",
            str(worker),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, "worker failed:\n" + result.stdout + "\n" + result.stderr
    line = next(
        (ln for ln in result.stdout.splitlines() if ln.startswith("MASS_REDUCE_METRICS=")), None
    )
    assert line is not None, "no metrics line:\n" + result.stdout
    metrics = json.loads(line.split("=", 1)[1])

    assert metrics["all_reduce_no_contiguous_transient"] is True
    assert metrics["all_reduce_result_0d_f32"] is True
    assert metrics["gather_rel_err_vs_fp64"] < 1e-6
    assert metrics["all_reduce_rel_err_vs_fp64"] < 1e-6
    assert metrics["gather_identical_across_ranks"] is True
    assert metrics["all_reduce_identical_across_ranks"] is True
    # The two branches may differ only in the last ulps of the sum.
    assert metrics["branches_relative_gap"] < 1e-6
