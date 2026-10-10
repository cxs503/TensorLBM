"""Tests for tensorlbm.triton_fused — Triton fused periodic D3Q19 step.

Verifies:
  1. Module imports and ``is_available()`` returns True on a CUDA host.
  2. Pull-stream is bit-exact with ``torch.roll`` (the existing perf_solver
     reference uses ``torch.roll``, so this is what users get today).
  3. Fused step is consistent with a separate Triton collide+stream.
  4. The class ``TritonFusedSolver3D.step`` returns a fresh tensor with
     the same shape and dtype and matching physics.
  5. Reduced-precision storage (fp16, bf16) keeps mass conservation and
     stays within the documented rel error bounds.
  6. The class refuses to instantiate on a host without CUDA.

All numerical assertions use ``torch.allclose`` with relaxed tolerances
appropriate for fp32 LBM summation-order differences (verified at 1.6%
relative vs PyTorch ref in earlier benchmarks).
"""
from __future__ import annotations

import pytest
import torch

from tensorlbm.triton_fused import (
    DEFAULT_BLOCK_X,
    DEFAULT_BLOCK_Y,
    TritonFusedSolver3D,
    is_available,
    make_lattice_tensors,
    triton_collide,
    triton_fused,
    triton_stream,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dev() -> str:
    if not is_available():
        pytest.skip("Triton fused requires CUDA + triton")
    return "cuda:0"


def _equilibrium_field(n: int, dev: str, seed: int = 0) -> torch.Tensor:
    """Random but reproducible initial (Q, n, n, n) distribution.

    Built from the D3Q19 equilibrium at rho=1 plus a small velocity field
    so that the population magnitudes are in the right neighbourhood of
    f_eq, which is what real LBM runs look like.
    """
    from tensorlbm.d3q19 import equilibrium3d

    torch.manual_seed(seed)
    rho = torch.ones((n, n, n), device=dev)
    u = 0.05 * torch.randn(3, n, n, n, device=dev)
    return equilibrium3d(rho, u[0], u[1], u[2])


# ---------------------------------------------------------------------------
# 1. Module surface
# ---------------------------------------------------------------------------

def test_is_available_on_cuda(dev: str) -> None:
    assert is_available() is True


def test_lattice_tensors_padded_and_correct(dev: str) -> None:
    lat = make_lattice_tensors(dev)
    for k in ("cxi", "cyi", "czi", "cxf", "cyf", "czf", "w"):
        assert k in lat
        t = lat[k]
        assert t.device.type == "cuda"
        assert t.shape == (32,)
    # First 19 entries carry the D3Q19 lattice; the rest are zero-pad.
    for k in ("cxi", "cyi", "czi"):
        assert torch.equal(lat[k][19:], torch.zeros(13, dtype=torch.int32, device=dev))
    assert torch.equal(lat["w"][19:], torch.zeros(13, dtype=torch.float32, device=dev))


# ---------------------------------------------------------------------------
# 2. Streaming correctness vs torch.roll (the perf_solver reference)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [16, 32, 64])
def test_stream_bit_exact_vs_torch_roll(n: int, dev: str) -> None:
    """Triton pull-stream must be bit-exact with torch.roll periodic shift."""
    from tensorlbm.triton_fused import _CX, _CY, _CZ

    f = _equilibrium_field(n, dev, seed=42)
    out_tri = triton_stream(f)

    # Reference: apply the same 19 periodic rolls as the kernel would, in
    # pure PyTorch.  Identical address arithmetic -> identical result.
    ref = torch.empty_like(f)
    for q in range(19):
        ref[q] = torch.roll(f[q],
                            shifts=(_CZ[q], _CY[q], _CX[q]),
                            dims=(0, 1, 2))
    diff = (out_tri - ref).abs().max().item()
    assert diff == 0.0, f"max|triton_stream - torch.roll 19-dir| = {diff}"


# ---------------------------------------------------------------------------
# 3. Fused vs separate collide+stream
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [16, 32, 64])
def test_fused_matches_separate_collide_stream(n: int, dev: str) -> None:
    """The fused kernel must be numerically equivalent to a stream+collide pair."""
    f = _equilibrium_field(n, dev, seed=7)
    tau = 0.55
    f_streamed = triton_stream(f)
    f_separate = triton_collide(f_streamed, tau)
    f_fused = triton_fused(f, tau)
    # Fp32 summation-order difference, well under 1% relative for 19-dir LBM.
    err = (f_separate - f_fused).abs().max().item()
    rel = err / max(f_separate.abs().max().item(), 1e-12)
    assert rel < 1e-4, f"fused vs separate rel err {rel:.2e} > 1e-4"


# ---------------------------------------------------------------------------
# 4. Class behaviour
# ---------------------------------------------------------------------------

def test_class_step_returns_fresh_tensor_with_correct_shape(dev: str) -> None:
    n = 32
    solver = TritonFusedSolver3D(nz=n, ny=n, nx=n, tau=0.6, device=dev)
    f = _equilibrium_field(n, dev)
    f_new = solver.step(f)
    assert f_new.shape == f.shape == (19, n, n, n)
    assert f_new.dtype == f.dtype == torch.float32
    assert f_new.device == f.device
    # Input tensor must not have been modified in place.
    assert f_new.data_ptr() != f.data_ptr()


def test_class_step_mass_conservation(dev: str) -> None:
    """Total mass must be conserved to round-off across many steps."""
    n, tau, n_steps = 32, 0.6, 100
    solver = TritonFusedSolver3D(nz=n, ny=n, nx=n, tau=tau, device=dev)
    f = _equilibrium_field(n, dev)
    m0 = f.sum(dtype=torch.float64).item()
    for _ in range(n_steps):
        f = solver.step(f)
    m1 = f.sum(dtype=torch.float64).item()
    # Per-cell rho legitimately re-distributes under convection; only the
    # domain-total mass is invariant.  Measured drift ~7e-6 over 100 steps.
    rel = abs(m1 - m0) / abs(m0)
    assert rel < 1e-4, f"relative mass drift after {n_steps} steps = {rel}"


def test_class_transient_memory_two_x(dev: str) -> None:
    """Transient memory should be just two ping-pong buffers, not 5.4x f."""
    n = 64
    solver = TritonFusedSolver3D(nz=n, ny=n, nx=n, tau=0.6, device=dev)
    expected = 2 * 19 * n ** 3 * 4  # 2 buffers, fp32
    actual = solver.transient_memory_bytes()
    assert actual == expected, f"expected {expected}, got {actual}"


def test_class_rejects_cpu_device() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        TritonFusedSolver3D(nz=8, ny=8, nx=8, tau=0.6, device="cpu")


def test_class_rejects_bad_dtype(dev: str) -> None:
    with pytest.raises(ValueError, match="Unsupported dtype"):
        TritonFusedSolver3D(nz=8, ny=8, nx=8, tau=0.6,
                             device=dev, dtype=torch.int32)


def test_step_no_alias_race_non_equilibrium_blob(dev: str) -> None:
    """Consecutive step() calls must not alias kernel input and output.

    Regression test for a cross-block read-after-write race: when the
    solver writes every step into the same internal buffer, feeding the
    previous output back makes the fused kernel's input pointer equal
    its output pointer.  Pull-stream blocks then read post-collision
    populations from neighbours that a concurrently scheduled block has
    already overwritten, and the trajectory diverges from the race-free
    reference.  Quasi-equilibrium fields hide this (f ~= feq, and BGK
    conserves per-cell mass either way); the Gaussian density blob here
    is strongly non-equilibrium after the first stream.
    """
    from tensorlbm.d3q19 import equilibrium3d

    n, sigma, tau, n_steps = 64, 4.0, 0.8, 50
    solver = TritonFusedSolver3D(nz=n, ny=n, nx=n, tau=tau, device=dev)

    # Non-equilibrium initial state: rho = 1 + 0.5*exp(-r^2/2s^2) blob
    # plus a small constant drift velocity.
    c = torch.arange(n, device=dev, dtype=torch.float32) - n // 2
    Z, Y, X = torch.meshgrid(c, c, c, indexing="ij")
    r2 = Z * Z + Y * Y + X * X
    rho = 1.0 + 0.5 * torch.exp(-r2 / (2.0 * sigma * sigma))
    ux = torch.full_like(rho, 0.03)
    uy = torch.full_like(rho, 0.02)
    uz = torch.full_like(rho, 0.01)
    f0 = equilibrium3d(rho, ux, uy, uz)

    # Direct ping-pong check: consecutive outputs must be distinct tensors.
    a = solver.step(f0.clone())
    b = solver.step(a)
    assert a.data_ptr() != b.data_ptr(), \
        "consecutive step() outputs must alternate buffers"

    m0 = f0.sum(dtype=torch.float64).item()

    # Candidate under test: feed the returned buffer straight back in.
    f = f0.clone()
    for _ in range(n_steps):
        f = solver.step(f)

    # Race-free reference: every step writes into a brand-new tensor.
    g = f0.clone()
    for _ in range(n_steps):
        g = triton_fused(g, tau)

    # (a) Total mass is conserved (BGK conserves per-cell rho even when
    # pulls read stale/updated values, so this alone cannot catch the
    # race — it guards the physics, the allclose below catches the race).
    m1 = f.sum(dtype=torch.float64).item()
    rel_mass = abs(m1 - m0) / abs(m0)
    assert rel_mass < 1e-4, f"relative mass drift after {n_steps} steps = {rel_mass}"

    # (b) The trajectory must match the race-free reference cell by cell.
    err = (f - g).abs().max().item()
    rel = err / max(g.abs().max().item(), 1e-12)
    assert rel < 1e-4, (
        f"50-step trajectory diverged from race-free reference: "
        f"max|diff|={err:.3e}, rel={rel:.3e} (input/output aliasing race)")


# ---------------------------------------------------------------------------
# 5. Reduced-precision storage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_reduced_precision_mass_conserved(dtype: torch.dtype, dev: str) -> None:
    """fp16 / bf16 storage must keep mass conservation close to fp32."""
    n, tau = 32, 0.6
    f32 = _equilibrium_field(n, dev)
    f16 = f32.to(dtype)
    solver = TritonFusedSolver3D(nz=n, ny=n, nx=n, tau=tau, device=dev, dtype=dtype)
    rho0 = f16.sum(dtype=torch.float64).item()
    f16 = solver.step(f16)
    rho1 = f16.sum(dtype=torch.float64).item()
    # 16-bit storage rounds each population (~1e-3 relative per value) but
    # the round-trip is near-unbiased, so the domain total drifts far less:
    # measured ~8e-8 (fp16) / ~7e-6 (bf16); bounds give >10x margin.
    rel = abs(rho1 - rho0) / abs(rho0)
    bound = 1e-4 if dtype is torch.bfloat16 else 1e-6
    assert rel < bound, f"relative mass drift with {dtype} = {rel}"


# ---------------------------------------------------------------------------
# 6. Sanity: solver is actually fast
# ---------------------------------------------------------------------------

def test_benchmark_returns_finite_time(dev: str) -> None:
    """The benchmark() helper must return a positive, finite number."""
    solver = TritonFusedSolver3D(nz=64, ny=64, nx=64, tau=0.6, device=dev)
    t = solver.benchmark(n_steps=20, warmup=3)
    assert 0.0 < t < 10.0, f"benchmark returned {t} s/step (out of plausible range)"