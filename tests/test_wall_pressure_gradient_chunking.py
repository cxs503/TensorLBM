"""Batch-size-safe linear algebra in wall pressure-gradient sampling.

cuSOLVER's batched symmetric eigensolver rejects batches above ~3.3e4
matrices (CUSOLVER_STATUS_INVALID_VALUE from cusolverDnXsyevBatched on
torch 2.11/cu128 / RTX 5090), which the SUBOFF AMR wall-diagnostic path
exceeds (48,400 near-wall samples on a 2:1 refined hull).  The chunked
helpers must be result-identical to the direct calls, and the public
sampler must survive batches of that size.
"""

from __future__ import annotations

import pytest
import torch

from tensorlbm.wall_pressure_gradient import (
    LINALG_MAX_BATCH,
    _eigvalsh_chunked,
    _solve_chunked,
    sample_wall_tangential_pressure_gradient,
)


def _random_spd(rows: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    a = torch.randn(rows, 3, 3, generator=generator)
    return a @ a.transpose(1, 2) + 3.0 * torch.eye(3)


def test_eigvalsh_chunked_bitwise_identical(monkeypatch):
    matrix = _random_spd(5000)
    monkeypatch.setattr("tensorlbm.wall_pressure_gradient.LINALG_MAX_BATCH", 512)
    assert torch.equal(_eigvalsh_chunked(matrix), torch.linalg.eigvalsh(matrix))


def test_solve_chunked_bitwise_identical(monkeypatch):
    matrix = _random_spd(5000, seed=1)
    rhs = torch.randn(5000, 3, 1)
    monkeypatch.setattr("tensorlbm.wall_pressure_gradient.LINALG_MAX_BATCH", 512)
    assert torch.equal(_solve_chunked(matrix, rhs), torch.linalg.solve(matrix, rhs))


def test_chunking_boundary_is_exact():
    """Batches at and around the cap take the direct path (no empty cats)."""
    for rows in (LINALG_MAX_BATCH - 1, LINALG_MAX_BATCH, LINALG_MAX_BATCH + 1):
        eigenvalues = _eigvalsh_chunked(_random_spd(rows, seed=rows))
        assert eigenvalues.shape == (rows, 3)
        assert bool(torch.isfinite(eigenvalues).all())


def _slab_case(active_rows: int):
    """Smooth pressure around a slab solid with >3.3e4 active wall nodes."""
    nz, ny, nx = 8, 64, 640
    solid = torch.zeros(nz, ny, nx, dtype=torch.bool)
    solid[3:5, :, :] = True
    active = torch.zeros_like(solid)
    active[2, :, :] = True
    active[5, :, :] = True
    assert int(active.sum()) == active_rows
    z, y, x = torch.meshgrid(
        torch.arange(nz, dtype=torch.float64),
        torch.arange(ny, dtype=torch.float64),
        torch.arange(nx, dtype=torch.float64),
        indexing="ij",
    )
    pressure = (0.01 * x + 0.002 * y * y + 0.05 * z + 1.0e-4 * x * y).to(torch.float32)
    normal_z = torch.zeros(nz, ny, nx)
    normal_z[2] = -1.0
    normal_z[5] = 1.0
    normals = (
        torch.zeros(nz, ny, nx),
        torch.zeros(nz, ny, nx),
        normal_z.to(torch.float32),
    )
    return pressure, solid, active, normals


def test_sampler_large_batch_chunked_equals_direct(monkeypatch):
    pressure, solid, active, normals = _slab_case(active_rows=2 * 64 * 640)
    direct = sample_wall_tangential_pressure_gradient(pressure, solid, active, normals)
    monkeypatch.setattr("tensorlbm.wall_pressure_gradient.LINALG_MAX_BATCH", 997)
    chunked = sample_wall_tangential_pressure_gradient(pressure, solid, active, normals)
    assert direct.requested_nodes == 81920 > 32768
    assert direct.valid_nodes == chunked.valid_nodes
    assert torch.equal(direct.valid, chunked.valid)
    assert torch.equal(direct.vector, chunked.vector)
    assert torch.equal(direct.magnitude, chunked.magnitude)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_regression_amr_near_batch():
    """The AMR failure size (48,400 3x3 matrices) must not raise on GPU."""
    device = torch.device("cuda")
    matrix = _random_spd(48_400).to(device)
    eigenvalues = _eigvalsh_chunked(matrix)
    assert eigenvalues.shape == (48_400, 3)
    assert bool(torch.isfinite(eigenvalues).all())
    rhs = torch.randn(48_400, 3, 1, device=device)
    solution = _solve_chunked(matrix, rhs)
    assert solution.shape == (48_400, 3, 1)
    assert bool(torch.isfinite(solution).all())
