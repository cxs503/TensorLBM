"""PrecisionPolicy tiers and their integration with triton_fused.

Covers the TOP-4 item: a five-tier compute/store precision enum
(adapted from Autodesk XLB's Apache-2.0 ``precision_policy.py``; see
the provenance note in ``tensorlbm/precision.py``) with a
``cast_to_compute`` / ``cast_to_store`` step-entry/exit convention.

Regression matrix: {FP32FP32, FP32FP16} x {D3Q19 periodic Triton
kernel}.  D3Q27 is not covered because the fused Triton kernel is
D3Q19-only; the eager solvers are out of scope for this wiring.
"""
from __future__ import annotations

import json

import pytest
import torch

from tensorlbm.precision import (
    PrecisionPolicy,
    cast_to_compute,
    cast_to_store,
)


# ---------------------------------------------------------------------------
# 1. Enum surface (CPU)
# ---------------------------------------------------------------------------

def test_policy_tiers_and_dtypes() -> None:
    expected = {
        "FP64FP64": (torch.float64, torch.float64),
        "FP64FP32": (torch.float64, torch.float32),
        "FP64FP16": (torch.float64, torch.float16),
        "FP32FP32": (torch.float32, torch.float32),
        "FP32FP16": (torch.float32, torch.float16),
    }
    assert set(p.name for p in PrecisionPolicy) == set(expected)
    for name, (compute, store) in expected.items():
        policy = PrecisionPolicy[name]
        assert policy.compute_dtype == compute
        assert policy.store_dtype == store


def test_policy_parse() -> None:
    assert PrecisionPolicy.parse(None) is PrecisionPolicy.FP32FP32
    assert PrecisionPolicy.parse("FP32FP16") is PrecisionPolicy.FP32FP16
    assert (PrecisionPolicy.parse(PrecisionPolicy.FP64FP32)
            is PrecisionPolicy.FP64FP32)
    with pytest.raises(ValueError, match="Unknown precision policy"):
        PrecisionPolicy.parse("FP16FP16")
    with pytest.raises(TypeError):
        PrecisionPolicy.parse(3)


def test_cast_boundaries() -> None:
    policy = PrecisionPolicy.FP32FP16
    stored = torch.full((4,), 0.5, dtype=torch.float16)
    widened = cast_to_compute(stored, policy)
    assert widened.dtype == torch.float32
    narrowed = cast_to_store(widened * 2.0, policy)
    assert narrowed.dtype == torch.float16
    # Same-dtype boundaries must be identity (no copy on the hot path).
    f32 = torch.zeros(4)
    identity = PrecisionPolicy.FP32FP32
    assert cast_to_compute(f32, identity) is f32
    assert cast_to_store(f32, identity) is f32
    # Policy methods mirror the module functions.
    assert policy.cast_to_compute(stored).dtype == torch.float32
    assert policy.cast_to_store(widened).dtype == torch.float16


# ---------------------------------------------------------------------------
# 2. Triton fused solver wiring (GPU)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dev() -> str:
    from tensorlbm.triton_fused import is_available

    if not is_available():
        pytest.skip("precision wiring tests require CUDA + triton")
    return "cuda:0"


def _initial_field(n: int, dev: str, seed: int = 0) -> torch.Tensor:
    from tensorlbm.d3q19 import equilibrium3d

    torch.manual_seed(seed)
    rho = torch.ones((n, n, n), device=dev)
    u = 0.05 * torch.randn(3, n, n, n, device=dev)
    return equilibrium3d(rho, u[0], u[1], u[2])


def test_solver_accepts_policy_and_maps_store_dtype(dev: str) -> None:
    from tensorlbm.triton_fused import TritonFusedSolver3D

    solver = TritonFusedSolver3D(
        8, 8, 8, tau=0.6, device=dev, precision="FP32FP32")
    assert solver.dtype == torch.float32
    assert solver.precision_policy is PrecisionPolicy.FP32FP32

    solver16 = TritonFusedSolver3D(
        8, 8, 8, tau=0.6, device=dev, precision="FP32FP16")
    assert solver16.dtype == torch.float16
    assert solver16.precision_policy is PrecisionPolicy.FP32FP16

    # Explicit dtype consistent with the tier is accepted.
    solver_eq = TritonFusedSolver3D(
        8, 8, 8, tau=0.6, device=dev,
        dtype=torch.float16, precision="FP32FP16")
    assert solver_eq.dtype == torch.float16


def test_solver_rejects_fp64_compute_and_dtype_conflict(dev: str) -> None:
    from tensorlbm.triton_fused import TritonFusedSolver3D

    with pytest.raises(NotImplementedError, match="fp64"):
        TritonFusedSolver3D(
            8, 8, 8, tau=0.6, device=dev, precision="FP64FP64")
    with pytest.raises(ValueError, match="conflicts"):
        TritonFusedSolver3D(
            8, 8, 8, tau=0.6, device=dev,
            dtype=torch.float32, precision="FP32FP16")


@pytest.mark.parametrize("policy", ["FP32FP32", "FP32FP16"])
def test_precision_matrix_vs_fp32_reference(policy: str, dev: str) -> None:
    """{FP32FP32, FP32FP16} x D3Q19 against the fp32 default path.

    FP32FP32 must be bit-identical to the dtype-only default (the tier
    only selects the same storage dtype); FP32FP16 storage must stay
    within the documented ~2e-4 rel err (bound 5e-3 for margin) and
    conserve mass to 16-bit rounding.
    """
    from tensorlbm.triton_fused import TritonFusedSolver3D

    n, tau, n_steps = 32, 0.6, 50
    f0 = _initial_field(n, dev, seed=11)

    reference = TritonFusedSolver3D(n, n, n, tau=tau, device=dev)
    g = f0.clone()
    for _ in range(n_steps):
        g = reference.step(g)

    solver = TritonFusedSolver3D(n, n, n, tau=tau, device=dev, precision=policy)
    policy_obj = PrecisionPolicy.parse(policy)
    # cast_to_store boundary: the initial distribution enters the stepper
    # in the policy's store dtype.
    f = cast_to_store(f0, policy_obj)
    assert f.dtype == solver.dtype
    for _ in range(n_steps):
        f = solver.step(f)

    if policy == "FP32FP32":
        assert torch.equal(f, g), (
            "FP32FP32 tier must be bit-identical to the dtype-only default")
    else:
        assert f.dtype == torch.float16
        max_rel = ((f.float() - g).abs().max() / g.abs().max()).item()
        assert max_rel < 5e-3, (
            f"FP32FP16 vs fp32 reference max rel err {max_rel:.3e} (expect ~2e-4)")
        m0 = f0.sum(dtype=torch.float64).item()
        m1 = f.sum(dtype=torch.float64).item()
        rel_mass = abs(m1 - m0) / abs(m0)
        assert rel_mass < 1e-4, f"FP32FP16 mass drift {rel_mass:.3e}"


def test_step_rejects_input_dtype_mismatch(dev: str) -> None:
    """An fp32 input to an FP32FP16 solver must raise, not silently run fp32."""
    from tensorlbm.triton_fused import TritonFusedSolver3D

    solver = TritonFusedSolver3D(
        8, 8, 8, tau=0.6, device=dev, precision="FP32FP16")
    f32 = _initial_field(8, dev, seed=3)
    with pytest.raises(ValueError, match="storage dtype"):
        solver.step(f32)


# ---------------------------------------------------------------------------
# 3. Benchmark observability (CPU)
# ---------------------------------------------------------------------------

def test_reporter_records_precision_and_bytes_per_cell(tmp_path) -> None:
    from tensorlbm.benchmark_observability import (
        BenchmarkReporter,
        precision_policy_metadata,
        step_bytes_per_cell,
    )

    meta = precision_policy_metadata("FP32FP16")
    assert meta == {
        "tier": "FP32FP16",
        "compute_dtype": "float32",
        "store_dtype": "float16",
    }
    assert precision_policy_metadata(None)["tier"] == "FP32FP32"
    assert precision_policy_metadata(
        PrecisionPolicy.FP64FP32)["compute_dtype"] == "float64"

    # FluidX3D-style bytes/cell/step: D3Q19 fused step = 2 passes over f.
    assert step_bytes_per_cell(19, torch.float32) == 152
    assert step_bytes_per_cell(19, torch.float16) == 76
    with pytest.raises(ValueError):
        step_bytes_per_cell(0, torch.float32)

    reporter = BenchmarkReporter(
        tmp_path, "unit", 10, {"resolved": "cpu"}, precision=meta)
    reporter.start()
    payload = json.loads((tmp_path / "run_status.json").read_text())
    assert payload["precision"]["tier"] == "FP32FP16"
    assert payload["precision"]["store_dtype"] == "float16"

    # Backward compatibility: no precision arg -> field simply absent.
    reporter2 = BenchmarkReporter(tmp_path / "plain", "unit", 10,
                                  {"resolved": "cpu"})
    reporter2.start()
    payload2 = json.loads(
        (tmp_path / "plain" / "run_status.json").read_text())
    assert "precision" not in payload2
