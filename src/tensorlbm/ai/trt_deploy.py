"""TensorRT deployment of the fused-ensemble drag surrogate (B4-P4a).

Companion to :mod:`tensorlbm.ai.onnx_deploy` (PR #242).  The shipped
artifact ``ensemble_cfull_stacked.onnx`` computes its de-normalisation and
ensemble statistics tail in float64 (matching ``ModelEnsembleBackend``
exactly), but TensorRT supports float32/FP16/INT8 only.  This module turns
that artifact into TensorRT engines without touching the exporter:

- :func:`f64_tail_to_f32` — graph surgery that rewrites the float64 tail
  (2 double initializers ``y_mean``/``y_std``, 1 Cast-to-double, 3 double
  constants, 5 double outputs) to float32.  Measured cost on the real
  5-member artifact: 2.1e-07 max abs in log10 C_D against the f64 artifact,
  i.e. below the 4.7e-07 float32 graph-parity floor of PR #242.
- :func:`build_f16_body_model` — region-split half precision: the FNO body
  runs in float16 (weights, GELU constants, inputs via explicit Casts) while
  the de-norm + stats tail stays float32, so the returned C_D never passes
  through float16.  TRT 11 removed the builder FP16 flag (engines are
  precision-typed through the model), so mixed precision must be expressed
  in the ONNX graph — this is that conversion.
- :func:`build_engine` — TRT engine build with an optimisation profile over
  the dynamic condition batch (``cond`` N in [1, max_batch]) and explicit
  TF32 control (TRT 11 defaults TF32 ON for float32 matmuls/convs, which
  costs ~8e-05 log10 C_D; :func:`build_engine` clears it by default).
- :class:`TrtEnsembleBackend` — runtime twin of
  :class:`~tensorlbm.ai.onnx_deploy.OnnxEnsembleBackend`: one deserialised
  engine, host-synchronous ``predict``/``predict_stats`` with the same
  raw-input contract (raw field ``(5, ny, nx)``, raw cond ``(N, 8)`` ->
  per-member linear C_D ``(M, N)``), device buffers managed with
  ``cuda-python`` (``cuda.bindings.runtime``).

onnx / tensorrt / cuda-python import lazily and fail with clear
``ImportError`` messages, so this module is importable (and unit-testable
for its guard paths) on hosts without them — numpy alone is a hard import.

Honest scope notes:

- engines are per-GPU-architecture artifacts (built here on an RTX 5090,
  sm_120, TRT 11.2.1 cu12); a ``.plan`` is not portable across major
  compute capabilities — rebuild from the ONNX siblings on the target;
- the float16 body path costs ~7e-04 max abs log10 C_D (measured, see
  ``benchmarks/deploy_latency.py``) — fine for interactive design curves,
  NOT a drop-in for the f64 contract;
- dynamic-INT8 quantization of this graph on the CUDA EP measured as a
  large latency LOSS (benchmark records it; it is deliberately not wired
  into any engine builder here).

Measured on the 5090 server (GPU 2, driver 610.57.04, TRT 11.2.1.2-cu12):
engine build 10-21 s, warm B=1 latency 0.63-0.71 ms end-to-end vs 6.6 ms
torch eager and 41 ms ORT CPU.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "TrtEnsembleBackend",
    "build_engine",
    "build_f16_body_model",
    "f64_tail_to_f32",
    "require_cuda_runtime",
    "require_onnx",
    "require_tensorrt",
]

_ONNX_IMPORT_ERROR = (
    "the onnx package is required for artifact surgery (deploy venv: pip install onnx)"
)
_TRT_IMPORT_ERROR = (
    "the tensorrt package is required to build engines (deploy venv: pip install tensorrt-cu12)"
)
_CUDART_IMPORT_ERROR = (
    "cuda-python is required to execute engines (deploy venv: pip install cuda-python)"
)


def require_onnx() -> Any:
    """Return the ``onnx`` module or raise ``ImportError`` with a hint."""
    try:
        import onnx
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_ONNX_IMPORT_ERROR) from exc
    return onnx


def require_tensorrt() -> Any:
    """Return the ``tensorrt`` module or raise ``ImportError`` with a hint."""
    try:
        import tensorrt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_TRT_IMPORT_ERROR) from exc
    return tensorrt


def require_cuda_runtime() -> Any:
    """Return ``cuda.bindings.runtime`` or raise ``ImportError`` with a hint."""
    try:
        from cuda.bindings import runtime
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(_CUDART_IMPORT_ERROR) from exc
    return runtime


def _shape_numel(shape: tuple[int, ...]) -> int:
    """Product of a shape (empty-safe), used for device buffer sizing."""
    n = 1
    for d in shape:
        n *= int(d)
    return max(n, 1)


def _toposort(graph: Any) -> None:
    """Topologically sort ``graph.node`` in place (raises on missing producer)."""
    produced = {i.name for i in graph.initializer} | {i.name for i in graph.input}
    order: list[Any] = []
    pending = list(graph.node)
    while pending:
        progressed = False
        rest: list[Any] = []
        for node in pending:
            if all((inp in produced) or (inp == "") for inp in node.input):
                order.append(node)
                produced.update(node.output)
                progressed = True
            else:
                rest.append(node)
        if not progressed and pending:
            names = ", ".join(n.name or n.op_type for n in pending[:8])
            raise RuntimeError(f"graph surgery left a missing producer among: {names}")
        pending = rest
    del graph.node[:]
    graph.node.extend(order)


def f64_tail_to_f32(onnx_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Rewrite the float64 de-norm/stats tail of a fused ensemble artifact to f32.

    Touches exactly the float64 region of the PR #242 artifact contract:
    initializers (``y_mean``/``y_std``), the single Cast-to-double, double
    ``Constant`` tensors and the five double-typed graph outputs.  Body ops
    were already float32; stale intermediate ``value_info`` annotations are
    dropped and re-derived (they would contradict the new dtypes).

    Returns an honest report dict (counts + checker status + path); the
    caller is expected to verify numerics against the f64 source (the
    benchmark does; measured cost ~2e-07 log10 C_D).
    """
    onnx = require_onnx()
    from onnx import TensorProto, numpy_helper, shape_inference

    model = onnx.load(str(onnx_path))
    graph = model.graph
    report: dict[str, Any] = {
        "kind": "f64_tail_to_f32",
        "src": str(Path(onnx_path).resolve()),
        "dst": str(Path(out_path).resolve()),
        "initializers_converted": 0,
        "cast_nodes_converted": 0,
        "constants_converted": 0,
        "io_ports_retyped": 0,
        "value_info_dropped": len(graph.value_info),
        "checker": None,
    }
    for init in graph.initializer:
        if init.data_type == TensorProto.DOUBLE:
            arr = numpy_helper.to_array(init).astype("float32")
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
            report["initializers_converted"] += 1
    for node in graph.node:
        for attr in node.attribute:
            if attr.name == "to" and attr.i == TensorProto.DOUBLE:
                attr.i = TensorProto.FLOAT
                report["cast_nodes_converted"] += 1
            if attr.name == "value" and attr.t.data_type == TensorProto.DOUBLE:
                arr = numpy_helper.to_array(attr.t).astype("float32")
                attr.t.CopyFrom(numpy_helper.from_array(arr, ""))
                report["constants_converted"] += 1

    def _retype(value_info: Any) -> int:
        if value_info.type.tensor_type.elem_type == TensorProto.DOUBLE:
            value_info.type.tensor_type.elem_type = TensorProto.FLOAT
            return 1
        return 0

    report["io_ports_retyped"] = sum(_retype(io) for io in list(graph.input) + list(graph.output))
    del graph.value_info[:]
    _toposort(graph)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    report["checker"] = "ok"
    report["artifact_bytes"] = Path(out_path).stat().st_size
    return report


def _float_region(graph: Any, boundary: Any, consumers: dict[str, list[Any]]) -> set[str]:
    """Names of the downstream node closure of ``boundary`` (inclusive)."""
    region: set[str] = set()
    stack = [boundary]
    while stack:
        node = stack.pop()
        if node.name in region:
            continue
        region.add(node.name)
        for out in node.output:
            stack.extend(consumers.get(out, []))
    return region


def build_f16_body_model(onnx_path: str | Path, out_path: str | Path) -> dict[str, Any]:
    """Convert an f32 fused-ensemble artifact to a half-body/f32-tail sibling.

    The float32 region is the downstream closure of the (single) Cast node
    that marks the de-norm boundary: everything after it (``y_std``/``y_mean``
    multiply-add, ``Pow(10, .)``, mean/std/min/max) stays float32, everything
    before runs float16 — weights converted to float16 initializers, body
    float constants converted, graph inputs kept float32 with explicit
    ``Cast(to=float16)`` entries.  All five outputs remain float32.

    The result is what a TRT 11 engine is built from when an FP16 body is
    wanted (TRT 11 has no builder FP16 flag; precision lives in the graph).
    """
    onnx = require_onnx()
    from onnx import TensorProto, helper, numpy_helper, shape_inference

    model = onnx.load(str(onnx_path))
    graph = model.graph
    casts = [n for n in graph.node if n.op_type == "Cast"]
    if len(casts) != 1:
        raise ValueError(f"expected exactly 1 Cast node (the de-norm boundary), found {len(casts)}")
    consumers: dict[str, list[Any]] = {}
    for node in graph.node:
        for inp in node.input:
            consumers.setdefault(inp, []).append(node)

    float_region = _float_region(graph, casts[0], consumers)
    stay_float: set[str] = {"y_mean", "y_std"}
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT:
                fans = consumers.get(node.output[0], [])
                if all(c.name in float_region for c in fans):
                    stay_float.add(node.output[0])

    rename: dict[str, str] = {}
    for init in list(graph.initializer):
        if init.data_type == TensorProto.FLOAT and init.name not in stay_float:
            arr = numpy_helper.to_array(init).astype("float16")
            graph.initializer.remove(init)
            graph.initializer.append(numpy_helper.from_array(arr, init.name + "__half"))
            rename[init.name] = init.name + "__half"
    for node in graph.node:
        for i, name in enumerate(node.input):
            if name in rename:
                node.input[i] = rename[name]
        for attr in node.attribute:
            stays = node.output[0] in stay_float if node.output else False
            if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT and not stays:
                arr = numpy_helper.to_array(attr.t).astype("float16")
                attr.t.CopyFrom(numpy_helper.from_array(arr, ""))

    entry_casts: list[Any] = []
    for io in graph.input:
        if io.type.tensor_type.elem_type != TensorProto.FLOAT:
            continue
        direct = consumers.get(io.name, [])
        if all(c.name in float_region for c in direct):
            continue
        half_name = io.name + "__half"
        for node in graph.node:
            if node.name in float_region:
                continue
            for i, name in enumerate(node.input):
                if name == io.name:
                    node.input[i] = half_name
        entry_casts.append(
            helper.make_node(
                "Cast",
                [io.name],
                [half_name],
                to=TensorProto.FLOAT16,
                name="f16_body_cast_" + io.name,
            )
        )
    graph.node.extend(entry_casts)
    del graph.value_info[:]
    _toposort(graph)
    model = shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))
    out_types = {o.name: int(o.type.tensor_type.elem_type) for o in model.graph.output}
    from onnx import TensorProto as _TP

    if set(out_types.values()) != {_TP.FLOAT}:
        raise RuntimeError(f"f16 body conversion left non-f32 outputs: {out_types}")
    return {
        "kind": "f16_body",
        "src": str(Path(onnx_path).resolve()),
        "dst": str(Path(out_path).resolve()),
        "float_region_nodes": len(float_region),
        "total_nodes": len(model.graph.node),
        "initializers_halved": len(rename),
        "entry_casts": len(entry_casts),
        "stay_float": sorted(stay_float),
        "checker": "ok",
        "artifact_bytes": Path(out_path).stat().st_size,
    }


def build_engine(
    onnx_path: str | Path,
    plan_path: str | Path,
    *,
    cond_input: str = "cond",
    cond_dim: int = 8,
    min_batch: int = 1,
    opt_batch: int = 8,
    max_batch: int = 256,
    clear_tf32: bool = True,
    workspace_gb: float = 16.0,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build a TRT engine from an ONNX model with a dynamic condition batch.

    ``clear_tf32=True`` (default) keeps float32 matmuls/convs strict IEEE
    float32 — TRT 11 enables TF32 by default, which measured ~8e-05 max abs
    log10 C_D on this model; the strict build is latency-identical (within
    noise) and lands at the float32 parity floor instead.
    """
    trt = require_tensorrt()
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    t0 = time.perf_counter()
    if not parser.parse(Path(onnx_path).read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("TRT ONNX parse failed: " + " | ".join(errors[:5]))
    report: dict[str, Any] = {
        "kind": "trt_engine",
        "onnx": str(Path(onnx_path).resolve()),
        "plan": str(Path(plan_path).resolve()),
        "parse_seconds": round(time.perf_counter() - t0, 3),
        "clear_tf32": bool(clear_tf32),
        "profile": {"min": min_batch, "opt": opt_batch, "max": max_batch},
    }
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    profile = builder.create_optimization_profile()
    profile.set_shape(
        cond_input, (min_batch, cond_dim), (opt_batch, cond_dim), (max_batch, cond_dim)
    )
    config.add_optimization_profile(profile)
    if clear_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    t0 = time.perf_counter()
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError("TRT engine build returned None (see builder log above)")
    buf = bytes(memoryview(engine_bytes))
    Path(plan_path).write_bytes(buf)
    report["build_seconds"] = round(time.perf_counter() - t0, 2)
    report["plan_bytes"] = len(buf)
    return report


class TrtEnsembleBackend:
    """Host-synchronous ensemble backend over ONE deserialised TRT engine.

    Same raw-input contract as ``OnnxEnsembleBackend``: ``predict`` takes the
    raw ``(5, ny, nx)`` field and raw ``(N, 8)`` condition rows and returns
    ``(M, N)`` per-member linear C_D as float64 (the engine computes the tail
    in float32; the host-side return is widened so downstream statistics
    behave like the torch/ORT twins).  ``predict_stats`` returns the in-graph
    mean/std/min/max.  Device buffers are allocated once at max batch and
    reused; every call is host-synchronous (async H2D on one private stream,
    one stream sync, then D2H of exactly the used rows).
    """

    def __init__(
        self,
        plan_path: str | Path,
        *,
        max_batch: int = 256,
        outputs: tuple[str, ...] = ("member_cd", "cd_mean", "cd_std", "cd_min", "cd_max"),
    ) -> None:
        trt = require_tensorrt()
        cudart = require_cuda_runtime()
        self._trt = trt
        self._cudart = cudart
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine = runtime.deserialize_cuda_engine(Path(plan_path).read_bytes())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize engine {plan_path}")
        self._context = self._engine.create_execution_context()
        self.plan_path = str(Path(plan_path).resolve())
        self._outputs = tuple(outputs)
        self._max_batch = int(max_batch)

        self._n_members = int(self._engine.get_tensor_shape("member_cd")[0])
        self._cond_dim = int(self._engine.get_tensor_shape("cond")[1])
        field_shape = tuple(int(d) for d in self._engine.get_tensor_shape("field"))
        self._field_shape = field_shape

        err, stream = cudart.cudaStreamCreate()
        self._check(err, "cudaStreamCreate")
        self._stream = stream

        err, dev = cudart.cudaMalloc(_shape_numel(field_shape) * 4)
        self._check(err, "cudaMalloc(field)")
        self._d_field = dev
        err, dev = cudart.cudaMalloc(self._max_batch * self._cond_dim * 4)
        self._check(err, "cudaMalloc(cond)")
        self._d_cond = dev

        self._context.set_input_shape("cond", (self._max_batch, self._cond_dim))
        self._d_out: dict[str, int] = {}
        for name in self._outputs:
            shape = tuple(int(d) for d in self._context.get_tensor_shape(name))
            cap = max(_shape_numel(shape), self._max_batch)
            err, devptr = cudart.cudaMalloc(cap * 4)
            self._check(err, f"cudaMalloc({name})")
            self._d_out[name] = devptr
        self._context.set_input_shape("cond", (1, self._cond_dim))
        self._bind_addresses()

    def _check(self, err: Any, what: str) -> None:
        if int(err) != 0:
            raise RuntimeError(f"CUDA runtime error in {what}: {err}")

    def _bind_addresses(self) -> None:
        assert self._context.set_tensor_address("field", self._d_field)
        assert self._context.set_tensor_address("cond", self._d_cond)
        for name, ptr in self._d_out.items():
            assert self._context.set_tensor_address(name, ptr)

    @property
    def n_members(self) -> int:
        return self._n_members

    @property
    def kind(self) -> str:
        return "tensorrt"

    def _run(self, field: Any, cond: Any) -> dict[str, Any]:
        cudart = self._cudart
        field = np.asarray(field, dtype=np.float32)
        cond = np.asarray(cond, dtype=np.float32)
        if field.ndim == 3:
            field = field[None]
        if tuple(field.shape) != self._field_shape:
            raise ValueError(
                f"field must be {self._field_shape[1:]} or {self._field_shape}, got {field.shape}"
            )
        if cond.ndim != 2 or cond.shape[1] != self._cond_dim:
            raise ValueError(f"cond must be (N, {self._cond_dim}), got {cond.shape}")
        n = int(cond.shape[0])
        if n > self._max_batch:
            raise ValueError(f"batch {n} exceeds engine profile max {self._max_batch}")
        self._context.set_input_shape("cond", (n, self._cond_dim))
        self._bind_addresses()
        htd = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        dth = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        self._check(
            cudart.cudaMemcpyAsync(
                self._d_field, field.ctypes.data, field.nbytes, htd, self._stream
            )[0],
            "H2D field",
        )
        self._check(
            cudart.cudaMemcpyAsync(self._d_cond, cond.ctypes.data, cond.nbytes, htd, self._stream)[
                0
            ],
            "H2D cond",
        )
        if not self._context.execute_async_v3(self._stream):
            raise RuntimeError("TRT execute_async_v3 returned false")
        self._check(cudart.cudaStreamSynchronize(self._stream)[0], "stream sync")
        shapes = {
            name: tuple(int(d) for d in self._context.get_tensor_shape(name))
            for name in self._outputs
        }
        result: dict[str, Any] = {}
        for name in self._outputs:
            arr = np.empty(shapes[name], dtype=np.float32)
            self._check(
                cudart.cudaMemcpy(arr.ctypes.data, self._d_out[name], arr.nbytes, dth)[0],
                f"D2H {name}",
            )
            result[name] = arr
        return result

    def predict(self, field: Any, cond: Any) -> Any:
        """Raw field + raw cond rows -> ``(M, N)`` member linear C_D (float64)."""
        return self._run(field, cond)["member_cd"].astype(np.float64)

    def predict_stats(self, field: Any, cond: Any) -> dict[str, Any]:
        """Raw inputs -> in-graph ensemble stats (mean/std/min/max, float64)."""
        outs = self._run(field, cond)
        return {k: outs[k].astype(np.float64) for k in ("cd_mean", "cd_std", "cd_min", "cd_max")}
