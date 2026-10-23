"""Tests for the deployment-latency hardening slice (B4-P4a).

Three tiers, mirroring ``tests/test_onnx_deploy.py``:

- pure-python tier (always runs, no optional deps): the benchmark module
  imports without side effects, the runtime groups expand, the adaptive
  iteration planner respects its bounds, and the latency-record /
  parity helpers compute what the ship doc claims;
- ONNX surgery tier (skipped without ``onnx``): a tiny synthetic graph
  with the SAME float64-tail structure as the real artifact (Cast
  boundary, ``y_mean``/``y_std`` double initializers, ``Pow(10, .)``
  double constant, double outputs) is converted by
  ``trt_deploy.f64_tail_to_f32`` / ``build_f16_body_model`` and checked
  structurally + numerically (against onnxruntime when present);
- TensorRT tier (skipped without tensorrt + cuda-python + a CUDA device):
  a real engine is built from the tiny graph and ``TrtEnsembleBackend``
  must match the onnxruntime CPU result at float32 parity.

Server commands::

    PYTHONPATH=src:/nfs/wangxi/runs/b4_serve_20260824/pydeps \\
        /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest tests/test_deploy_latency.py \\
        -q --basetemp=/nfs/wangxi/tmp/pt_deploy

    # TRT tier (deploy venv: tensorrt + cuda-python + onnxruntime-gpu)
    PYTHONPATH=src /nfs/wangxi/venvs/deploy/bin/python -m pytest \\
        tests/test_deploy_latency.py -q --basetemp=/nfs/wangxi/tmp/pt_deploy
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

BENCH_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "deploy_latency.py"
TRT_DEPLOY_PATH = Path(__file__).resolve().parents[1] / "src" / "tensorlbm" / "ai" / "trt_deploy.py"


def _load_benchmark_module() -> Any:
    spec = importlib.util.spec_from_file_location("deploy_latency_bench", BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _import_trt_deploy() -> Any:
    """Load trt_deploy by package or by file (torch-free venvs skip the package)."""
    try:
        from tensorlbm.ai import trt_deploy

        return trt_deploy
    except ImportError:
        spec = importlib.util.spec_from_file_location("tensorlbm_ai_trt_deploy", TRT_DEPLOY_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


# ---------------------------------------------------------------------------
# pure-python tier
# ---------------------------------------------------------------------------


def test_benchmark_module_imports_without_side_effects() -> None:
    module = _load_benchmark_module()
    assert callable(module.main)
    assert "trt" in module.RUNTIME_GROUPS
    assert "ort_gpu_fp32_strict" in module.RUNTIME_GROUP_OF
    assert module.RUNTIME_PRECISION["trt_f16"] == "fp16-body/fp32-tail"


def test_expand_runtimes_groups_and_dedup() -> None:
    module = _load_benchmark_module()
    assert module.expand_runtimes("trt") == list(module.RUNTIME_GROUPS["trt"])
    assert module.expand_runtimes("all") == list(module.ALL_RUNTIMES)
    assert module.expand_runtimes("trt,trt_fp32") == list(module.RUNTIME_GROUPS["trt"])
    assert module.expand_runtimes(" ort_gpu_fp32 , trt_f16 ") == [
        "ort_gpu_fp32",
        "trt_f16",
    ]
    with pytest.raises(ValueError, match="unknown runtime"):
        module.expand_runtimes("nope")


def test_plan_iters_respects_bounds() -> None:
    module = _load_benchmark_module()
    fast = module.plan_iters(0.001, min_iters=100, min_seconds=5.0, max_iters=2000)
    assert fast == 2000  # 5 s of 1 us calls is above the cap
    slow = module.plan_iters(50.0, min_iters=100, min_seconds=5.0, max_iters=2000)
    assert slow == 100  # 100 calls of 50 ms already cover 5 s
    assert module.plan_iters(2.0, min_iters=100, min_seconds=5.0, max_iters=2000) == 2000
    assert module.plan_iters(2.0, min_iters=100, min_seconds=5.0, max_iters=4000) == 2500


def test_make_latency_record_schema() -> None:
    module = _load_benchmark_module()
    stats = {
        "iters": 10,
        "total_s": 1.0,
        "p50_ms": 1.0,
        "p90_ms": 2.0,
        "mean_ms": 1.2,
        "min_ms": 0.9,
        "max_ms": 3.0,
    }
    parity = {
        "max_abs_lin": 1e-5,
        "max_abs_log10": 1e-7,
        "per_member_max_abs_log10": [1e-7, 2e-7],
    }
    record = module.make_latency_record(
        "trt_fp32",
        8,
        stats,
        parity,
        precision="fp32",
        warmup=30,
        artifact="/tmp/x.plan",
        extras={"mode": "default"},
    )
    assert record["event"] == "latency"
    assert record["runtime"] == "trt_fp32"
    assert record["group"] == "trt"
    assert record["batch"] == 8
    for key in ("p50_ms", "p90_ms", "mean_ms", "min_ms", "max_ms", "iters", "total_s", "warmup"):
        assert isinstance(record[key], (int, float))
    assert record["throughput_rows_per_s_p50"] == pytest.approx(8 * 1000.0 / 1.0)
    assert record["parity"] is parity
    assert record["mode"] == "default"
    no_parity = module.make_latency_record("ort_cpu", 1, stats, None, precision="fp32", warmup=30)
    assert no_parity["parity"] is None
    assert "artifact" not in no_parity


def test_parity_vs_reference_metric() -> None:
    module = _load_benchmark_module()
    ref = np.array([[10.0, 20.0], [10.0, 20.0]])
    got = ref * (1 + 1e-4)
    parity = module.parity_vs_reference(got, ref)
    # relative 1e-4 on linear C_D is 1e-4/ln(10) ~ 4.34e-5 in log10 space
    assert parity["max_abs_log10"] == pytest.approx(4.3427e-5, rel=1e-3)
    assert len(parity["per_member_max_abs_log10"]) == 2
    assert parity["max_abs_lin"] == pytest.approx(2e-3, rel=1e-3)
    with pytest.raises(ValueError, match="shape mismatch"):
        module.parity_vs_reference(got, ref[:, :1])
    with pytest.raises(ValueError, match="non-positive"):
        module.parity_vs_reference(np.zeros_like(ref), ref)


def test_time_cell_returns_full_stats() -> None:
    module = _load_benchmark_module()
    stats = module.time_cell(lambda: None, warmup=2, min_iters=5, min_seconds=0.0, max_iters=10)
    for key in ("iters", "total_s", "p50_ms", "p90_ms", "mean_ms", "min_ms", "max_ms"):
        assert key in stats
    assert stats["iters"] == 5
    assert stats["p50_ms"] <= stats["p90_ms"]
    assert stats["min_ms"] <= stats["p50_ms"]


def test_trt_deploy_guard_messages_without_optional_deps() -> None:
    trt_deploy = _import_trt_deploy()
    tested_any = False
    for fn, mod in (
        (trt_deploy.require_tensorrt, "tensorrt"),
        (trt_deploy.require_cuda_runtime, "cuda.bindings.runtime"),
    ):
        try:
            __import__(mod)
        except ImportError:
            with pytest.raises(ImportError, match="deploy venv"):
                fn()
            tested_any = True
    if not tested_any:
        pytest.skip("tensorrt and cuda-python both installed; guards not reachable")


# ---------------------------------------------------------------------------
# ONNX surgery tier (tiny synthetic artifact with the real tail structure)
# ---------------------------------------------------------------------------


def _tiny_onnx(tmp_path: Path, name: str = "tiny_f64.onnx") -> Path:
    """Tiny artifact with the PR #242 tail: f32 body -> Cast -> f64 de-norm.

    ``cond (n, 2)`` and ``field (1, 3)`` enter a float32 body
    (Gemm + Add + GELU-constant Mul), one Cast marks the de-norm boundary,
    then ``y_std``/``y_mean`` multiply-add and ``Pow(10, .)`` produce the
    double outputs (``member_cd``, ``cd_mean``) — the same op/dtype layout
    the surgery functions must handle on the real 451-node artifact.
    """
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    w = np.full((2, 3), 0.5, dtype=np.float32)
    b = np.zeros((3,), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node(
                "Constant",
                [],
                ["half_out"],
                value=numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), "c1"),
                name="/Constant_half",
            ),
            helper.make_node(
                "Constant",
                [],
                ["ten_out"],
                value=numpy_helper.from_array(np.asarray(10.0, dtype=np.float64), "c2"),
                name="/Constant_ten",
            ),
            helper.make_node("Gemm", ["cond", "W", "B"], ["z"], name="body_gemm"),
            helper.make_node("Add", ["z", "field"], ["z1"], name="body_add"),
            helper.make_node("Mul", ["z1", "half_out"], ["z2"], name="body_gelu_const"),
            helper.make_node("Cast", ["z2"], ["z_d"], to=TensorProto.DOUBLE, name="/Cast"),
            helper.make_node("Mul", ["z_d", "y_std"], ["s"], name="/Mul_25"),
            helper.make_node("Add", ["s", "y_mean"], ["a"], name="/Add_26"),
            helper.make_node("Pow", ["ten_out", "a"], ["member_cd"], name="/Pow"),
            helper.make_node(
                "ReduceMean",
                ["member_cd"],
                ["cd_mean"],
                axes=[0],
                keepdims=0,
                name="/ReduceMean_1",
            ),
        ],
        "tiny",
        [
            helper.make_tensor_value_info("cond", TensorProto.FLOAT, ["n", 2]),
            helper.make_tensor_value_info("field", TensorProto.FLOAT, [1, 3]),
        ],
        [
            helper.make_tensor_value_info("member_cd", TensorProto.DOUBLE, ["n", 3]),
            helper.make_tensor_value_info("cd_mean", TensorProto.DOUBLE, [3]),
        ],
        [
            numpy_helper.from_array(w, "W"),
            numpy_helper.from_array(b, "B"),
            numpy_helper.from_array(np.asarray([0.2, 0.2, 0.2]), "y_std"),
            numpy_helper.from_array(np.asarray([1.0, 1.0, 1.0]), "y_mean"),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 17)],
        producer_name="test_deploy_latency",
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    path = tmp_path / name
    onnx.save(model, path)
    return path


def test_f64_tail_to_f32_structure_and_numerics(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx", reason="ONNX surgery tier needs onnx")
    from onnx import TensorProto

    trt_deploy = _import_trt_deploy()
    src = _tiny_onnx(tmp_path)
    dst = tmp_path / "tiny_f32.onnx"
    report = trt_deploy.f64_tail_to_f32(src, dst)

    assert report["initializers_converted"] == 2  # y_mean, y_std
    assert report["cast_nodes_converted"] == 1
    assert report["constants_converted"] == 1  # the Pow base 10.
    assert report["io_ports_retyped"] == 2  # member_cd, cd_mean
    converted = onnx.load(dst)
    out_types = {o.name: o.type.tensor_type.elem_type for o in converted.graph.output}
    assert set(out_types.values()) == {TensorProto.FLOAT}
    inits = {i.name: i.data_type for i in converted.graph.initializer}
    assert inits["y_mean"] == TensorProto.FLOAT
    assert inits["y_std"] == TensorProto.FLOAT
    onnx.checker.check_model(converted)

    ort = pytest.importorskip("onnxruntime", reason="numeric check needs onnxruntime")
    cond = np.asarray([[0.1, -0.2], [1.5, 0.3], [-0.7, 0.9]], dtype=np.float32)
    field = np.zeros((1, 3), dtype=np.float32)
    s64 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    s32 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    o64 = s64.run(None, {"cond": cond, "field": field})[0]
    o32 = s32.run(None, {"cond": cond, "field": field})[0]
    assert np.abs(o32 - o64).max() < 1e-6


def test_build_f16_body_model_structure_and_numerics(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx", reason="ONNX surgery tier needs onnx")
    from onnx import TensorProto

    trt_deploy = _import_trt_deploy()
    src = _tiny_onnx(tmp_path)
    f32 = tmp_path / "tiny_f32.onnx"
    trt_deploy.f64_tail_to_f32(src, f32)
    dst = tmp_path / "tiny_f16body.onnx"
    report = trt_deploy.build_f16_body_model(f32, dst)

    assert report["float_region_nodes"] == 5  # Cast, Mul, Add, Pow, ReduceMean
    assert report["initializers_halved"] == 2  # W and B; y_mean/y_std stay f32
    assert report["entry_casts"] == 2  # cond + field
    assert "y_mean" in report["stay_float"] and "y_std" in report["stay_float"]
    converted = onnx.load(dst)
    out_types = {o.name: o.type.tensor_type.elem_type for o in converted.graph.output}
    assert set(out_types.values()) == {TensorProto.FLOAT}
    inits = {i.name: i.data_type for i in converted.graph.initializer}
    assert inits["W__half"] == TensorProto.FLOAT16
    assert inits["y_mean"] == TensorProto.FLOAT
    casts = [n for n in converted.graph.node if n.op_type == "Cast"]
    assert len(casts) == 3  # 2 entry f32->f16 + the boundary f16->f32
    onnx.checker.check_model(converted)

    ort = pytest.importorskip("onnxruntime", reason="numeric check needs onnxruntime")
    cond = np.asarray([[0.1, -0.2], [1.5, 0.3], [-0.7, 0.9]], dtype=np.float32)
    field = np.zeros((1, 3), dtype=np.float32)
    s64 = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(str(dst), providers=["CPUExecutionProvider"])
    o64 = s64.run(None, {"cond": cond, "field": field})[0]
    o16 = s16.run(None, {"cond": cond, "field": field})[0]
    assert np.abs(np.log10(o16) - np.log10(o64)).max() < 5e-3


# ---------------------------------------------------------------------------
# TensorRT tier (real engine from the tiny graph)
# ---------------------------------------------------------------------------


def test_trt_engine_build_and_backend_roundtrip(tmp_path: Path) -> None:
    pytest.importorskip("tensorrt", reason="TRT tier needs tensorrt")
    pytest.importorskip("cuda.bindings.runtime", reason="TRT tier needs cuda-python")
    runtime = pytest.importorskip("cuda.bindings.runtime")
    err, count = runtime.cudaGetDeviceCount()
    if int(err) != 0 or int(count) < 1:
        pytest.skip("no CUDA device")

    trt_deploy = _import_trt_deploy()
    ort = pytest.importorskip("onnxruntime", reason="reference needs onnxruntime")

    src = _tiny_onnx(tmp_path)
    f32 = tmp_path / "tiny_f32.onnx"
    trt_deploy.f64_tail_to_f32(src, f32)
    plan = tmp_path / "tiny.plan"
    report = trt_deploy.build_engine(
        f32, plan, cond_dim=2, min_batch=1, opt_batch=3, max_batch=8, workspace_gb=1.0
    )
    assert report["plan_bytes"] > 0
    assert report["clear_tf32"] is True

    backend = trt_deploy.TrtEnsembleBackend(plan, max_batch=8, outputs=("member_cd", "cd_mean"))
    assert backend.kind == "tensorrt"
    field = np.zeros((1, 3), dtype=np.float32)
    cond = np.asarray([[0.1, -0.2], [1.5, 0.3], [-0.7, 0.9], [0.0, 0.0]], dtype=np.float32)
    s32 = ort.InferenceSession(str(f32), providers=["CPUExecutionProvider"])
    ref = s32.run(None, {"cond": cond, "field": field})[0].astype(np.float64)

    got = backend.predict(field, cond)
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 1e-5
    # dynamic batch re-set on the same context
    got1 = backend.predict(field, cond[:1])
    assert got1.shape == (1, 3)
    assert np.abs(got1 - ref[:1]).max() < 1e-5
    with pytest.raises(ValueError, match="exceeds engine profile max"):
        backend.predict(field, np.zeros((9, 2), dtype=np.float32))
