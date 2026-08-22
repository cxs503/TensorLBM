"""Selective-direction halo exchange for triton_fused_distributed.

Covers the TOP-3 performance item (independently cross-validated by
FluidX3D's ``transfers`` tables and XLB's ``left/right_indices`` ring
exchange):

1. The crossing-direction tables are *generated* from the d3q19/d3q27
   lattice constants — 5 per face for D3Q19, 9 for D3Q27 — never
   hand-typed (the hand-copied-lane lesson in ``triton_fused.py``).
2. Staging planes shrink from full-Q ``(19, ny, nx)`` to
   ``(n_cross, ny, nx)``: a 3.8x staging/wire cut for D3Q19 at fp32,
   7.6x with the opt-in fp16 wire.
3. A real 2-rank torchrun/NCCL run reproduces the single-rank periodic
   trajectory **bitwise** with fp32 transport, and within the fp16
   round-trip tolerance with the opt-in fp16 wire.

The torchrun case is skipped unless ``torchrun`` is on PATH and two
GPUs are available; it pins ``CUDA_VISIBLE_DEVICES`` (default ``6,7``,
override with ``AR_TEST_GPUS``) so it never lands on GPUs reserved by
other users.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tensorlbm.d3q19 import C as C_D3Q19
from tensorlbm.d3q27 import C as C_D3Q27

try:
    from tensorlbm.triton_fused_distributed import (
        DistributedTritonFusedSolver3D,
        crossing_face_indices,
    )
except ModuleNotFoundError:
    pytest.skip(
        "selective-halo tests require triton (triton_fused_distributed imports it at module level)",
        allow_module_level=True,
    )


def _cz(C: torch.Tensor) -> list[int]:
    return [int(v) for v in C[:, 2].tolist()]


# ---------------------------------------------------------------------------
# 1. Crossing tables are generated from the lattice constants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lattice,per_face,q",
    [(C_D3Q19, 5, 19), (C_D3Q27, 9, 27)],
    ids=["D3Q19", "D3Q27"],
)
def test_crossing_tables_generated_from_lattice_constants(
    lattice: torch.Tensor,
    per_face: int,
    q: int,
) -> None:
    """Per-face table = cz=±1 subsets; union = the full cz≠0 set."""
    cz = _cz(lattice)
    up = crossing_face_indices(cz, +1)
    dn = crossing_face_indices(cz, -1)
    assert len(up) == per_face and len(dn) == per_face
    nonzero = {i for i, v in enumerate(cz) if v != 0}
    assert len(nonzero) == 2 * per_face
    assert set(up) | set(dn) == nonzero
    assert not (set(up) & set(dn))
    assert all(cz[i] == +1 for i in up)
    assert all(cz[i] == -1 for i in dn)


def test_crossing_face_indices_rejects_bad_sign() -> None:
    with pytest.raises(ValueError, match="sign"):
        crossing_face_indices([0, 1, -1], 0)


# ---------------------------------------------------------------------------
# 2. Solver wiring: tables + staging volume (single process, GPU)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dev() -> str:
    from tensorlbm.triton_fused import is_available

    if not is_available():
        pytest.skip("selective-halo solver tests require CUDA + triton")
    return "cuda:0"


def test_solver_tables_come_from_d3q19_constants(dev: str) -> None:
    solver = DistributedTritonFusedSolver3D(nz_global=8, ny=8, nx=8, tau=0.6, device=dev)
    cz = _cz(C_D3Q19)
    assert solver._cross_up == tuple(i for i in range(19) if cz[i] == +1)
    assert solver._cross_dn == tuple(i for i in range(19) if cz[i] == -1)
    assert solver.n_cross == 5
    assert solver._cross_up != solver._cross_dn


@pytest.mark.parametrize("wire", [torch.float32, torch.float16])
def test_staging_volume_selective_vs_legacy_full_q(
    dev: str,
    wire: torch.dtype,
) -> None:
    """Staging must be (n_cross, ny, nx), not (19, ny, nx).

    The ratio against the legacy full-Q staging is exactly Q/n_cross =
    19/5 = 3.8 (fp32 wire); the fp16 wire halves the absolute size.
    """
    ny = nx = 64
    solver = DistributedTritonFusedSolver3D(
        nz_global=8, ny=ny, nx=nx, tau=0.6, device=dev, halo_dtype=wire
    )
    solver._alloc_halo_staging(wire)
    assert solver.staging_shape == (5, ny, nx)
    for plane in (solver._send_left, solver._send_right, solver._recv_left, solver._recv_right):
        assert plane.shape == (5, ny, nx)
        assert plane.dtype == wire
    legacy_staging = 4 * 19 * ny * nx * 4
    assert solver.staging_bytes() == 4 * 5 * ny * nx * wire.itemsize
    if wire == torch.float32:
        assert legacy_staging / solver.staging_bytes() == pytest.approx(19 / 5)
    else:
        assert legacy_staging / solver.staging_bytes() == pytest.approx(19 / 5 * 2)
    assert solver.halo_bytes_per_step() == 2 * 5 * ny * nx * wire.itemsize
    legacy_wire = 2 * 19 * ny * nx * 4
    expected = 19 / 5 if wire == torch.float32 else 19 / 5 * 2
    assert legacy_wire / solver.halo_bytes_per_step() == pytest.approx(expected)


def test_solver_rejects_unsupported_halo_dtype(dev: str) -> None:
    with pytest.raises(ValueError, match="halo_dtype"):
        DistributedTritonFusedSolver3D(
            nz_global=8, ny=8, nx=8, tau=0.6, device=dev, halo_dtype=torch.float64
        )


def test_single_rank_ghost_fill_keeps_working(dev: str) -> None:
    """world==1: _start_halo_exchange is the nearest-plane copy, as before."""
    from tensorlbm.d3q19 import equilibrium3d

    n = 8
    solver = DistributedTritonFusedSolver3D(nz_global=n, ny=n, nx=n, tau=0.6, device=dev)
    rho = torch.ones((n, n, n), device=dev)
    u = torch.zeros((3, n, n, n), device=dev)
    f0 = equilibrium3d(rho, u[0], u[1], u[2])
    f = solver.from_global(f0)
    handles = solver._start_halo_exchange(f)
    assert handles == []
    assert torch.equal(f[:, 0], f[:, 1])
    assert torch.equal(f[:, -1], f[:, -2])


# ---------------------------------------------------------------------------
# 3. 2-rank torchrun equivalence (real NCCL, selective staging)
# ---------------------------------------------------------------------------

_WORKER = r"""
import json
import os

import torch
import torch.distributed as dist

from tensorlbm.d3q19 import equilibrium3d
from tensorlbm.triton_fused import TritonFusedSolver3D
from tensorlbm.triton_fused_distributed import (
    DistributedTritonFusedSolver3D,
    init_distributed,
)

rank, world = init_distributed()  # NCCL when CUDA is visible
dev = f"cuda:{rank}"
torch.cuda.set_device(rank)
torch.manual_seed(20260820)

nz_global, ny, nx = 64, 32, 32
tau, n_steps = 0.6, 10
halo_dtype = {"fp32": torch.float32, "fp16": torch.float16}[
    os.environ["AR_HALO_DTYPE"]]

# Identical initial field on every rank (same seed, no rank term).
rho = torch.ones((nz_global, ny, nx), device=dev)
u = 0.05 * torch.randn(3, nz_global, ny, nx, device=dev)
f0 = equilibrium3d(rho, u[0], u[1], u[2])

# --- single-rank reference (redundantly computed on both ranks) ---
ref = TritonFusedSolver3D(nz_global, ny, nx, tau=tau, device=dev)
g = f0.clone()
for _ in range(n_steps):
    g = ref.step(g)

# --- 2-rank slab run with the selective halo exchange ---
solver = DistributedTritonFusedSolver3D(
    nz_global=nz_global, ny=ny, nx=nx, tau=tau, device=dev,
    halo_dtype=halo_dtype)
f = solver.from_global(f0)
# First-step halo: from_global only pre-fills ghosts with the nearest
# owned plane, which is wrong at a rank interface — land one exchange
# before the first step (same pattern as triton_suboff_step_distributed).
solver._halo_handles = solver._start_halo_exchange(f)
solver.synchronize()
for _ in range(n_steps):
    out = solver.step(f)
    solver._buf = f  # repoint scratch: true ping-pong, no aliasing
    f = out
solver.synchronize()

own = f[:, 1:solver.nz_local + 1].contiguous()
parts = [torch.empty_like(own) for _ in range(world)]
dist.all_gather(parts, own)
f_global = torch.cat(parts, dim=1)

metrics = {
    "transport": os.environ["AR_HALO_DTYPE"],
    "n_cross": solver.n_cross,
    "staging_bytes": solver.staging_bytes(),
    "legacy_staging_bytes": 4 * 19 * ny * nx * 4,
    "halo_bytes_per_step": solver.halo_bytes_per_step(),
    "legacy_halo_bytes_per_step": 2 * 19 * ny * nx * 4,
    "max_abs_diff": float((f_global - g).abs().max()),
    "bitwise": bool(torch.equal(f_global, g)),
}
if rank == 0:
    print("AR_HALO_METRICS=" + json.dumps(metrics, sort_keys=True), flush=True)
dist.destroy_process_group()
"""


def _visible_gpu_count(gpus: str) -> int:
    """Probe how many of the *requested* GPUs truly exist.

    Counting the comma-separated tokens (the previous implementation) made
    the default ``"6,7"`` always report 2, so CPU-only CI runners launched
    the two-rank worker and failed instead of skipping.  Ask CUDA itself,
    in a subprocess with ``CUDA_VISIBLE_DEVICES`` set to the request, so
    the parent process's CUDA state cannot leak in.
    """
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus)
    try:
        out = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0
    try:
        return int(out.stdout.strip() or 0)
    except ValueError:
        return 0


@pytest.mark.parametrize("transport", ["fp32", "fp16"])
def test_two_rank_selective_halo_matches_single_rank(
    tmp_path: Path,
    transport: str,
) -> None:
    torchrun = shutil.which("torchrun")
    if torchrun is None:
        pytest.skip("torchrun not on PATH")
    gpus = os.environ.get("AR_TEST_GPUS", "6,7")
    if _visible_gpu_count(gpus) < 2:
        pytest.skip(f"need 2 GPUs (AR_TEST_GPUS={gpus!r})")

    worker = tmp_path / "selective_halo_worker.py"
    worker.write_text(_WORKER)
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["AR_HALO_DTYPE"] = transport
    env["CUDA_VISIBLE_DEVICES"] = gpus
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            str(worker),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    metrics = None
    for line in result.stdout.splitlines():
        if line.startswith("AR_HALO_METRICS="):
            metrics = json.loads(line.split("=", 1)[1])
    assert metrics is not None, (
        "worker produced no metrics\n"
        f"returncode={result.returncode}\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr[-4000:]}"
    )
    assert result.returncode == 0, result.stderr[-4000:]

    # --- staging/wire volume: 5 lanes per face, not 19 ---
    assert metrics["n_cross"] == 5
    item = 4 if transport == "fp32" else 2
    assert metrics["staging_bytes"] == 4 * 5 * 32 * 32 * item
    assert metrics["halo_bytes_per_step"] == 2 * 5 * 32 * 32 * item
    cut = metrics["legacy_staging_bytes"] / metrics["staging_bytes"]
    expected_cut = (19 / 5) * (4 / item)
    assert cut == pytest.approx(expected_cut)  # 3.8x fp32 wire, 7.6x fp16

    # --- physics: the 2-rank trajectory matches the single-rank one ---
    if transport == "fp32":
        assert metrics["max_abs_diff"] == 0.0 and metrics["bitwise"], metrics
    else:
        # fp16 wire: round-trip error on the exchanged populations only
        # (~1e-3 per halo value, damped by the collision); generous bound.
        assert metrics["max_abs_diff"] < 5e-3, metrics
