"""B4-P4a · deployment-latency benchmark for the fused drag-ensemble artifact.

One benchmark, every deployment runtime of the ensemble drag surrogate, one
JSON line per record (``event``: ``meta`` / ``engine`` / ``latency`` /
``skip`` / ``cold``), append-only like ``benchmarks/b4_onnx_bench.py``.

Measured task (identical for every runtime): raw mid-plane field
``(5, 64, 128)`` + ``B`` raw ``condition_v3`` rows in, ``(M=5, B)`` member
linear C_D out — host to host, B in ``{1, 2, 4, 8}`` (the interactive
slider pattern).  Query rows come from the largest design family of the B4
corpus (one field, Re-swept rows — exactly the PR #242 bench pattern), and
every runtime is parity-checked against a torch-CPU ``ModelEnsembleBackend``
reference stored in the reference npz (per-member max abs in linear and
log10 C_D — the ``verify_ensemble_onnx`` metric of PR #242).

Runtimes (``--runtime``, comma list; groups expand):

===========  =========================================================
group        rows
===========  =========================================================
trt          ``trt_fp32`` (TF32 cleared), ``trt_tf32`` (TRT default),
             ``trt_f16`` (half body / f32 tail typed graph)
ort_cpu      ``ort_cpu`` (the f64 artifact, CPU EP — PR #242 baseline)
ort_gpu      ``ort_gpu_fp32``, ``ort_gpu_fp32_strict`` (``use_tf32=0``),
             ``ort_gpu_f16``, ``ort_gpu_int8dyn`` (dynamic INT8)
torch_fp32   ``torch_fp32_baseline`` (exact service math,
             ``ModelEnsembleBackend``), ``torch_fp32_cached``
             (norms pre-uploaded, one host sync), ``torch_fp32_strict``
             (baseline math with ``cudnn``/``matmul`` TF32 OFF — the
             baseline row keeps torch defaults, whose TF32 convs already
             cost ~1e-4 log10 at B >= 2)
torch_fp16   ``torch_fp16_autocast`` — NOT RUNNABLE on ``CondFNODrag`` as
             shipped: ``autocast`` (fp16 AND bf16) and whole-model
             ``.half()`` all reach the spectral einsum with half-precision
             activations, where ``rfft2`` produces complex32 and
             ``baddbmm`` has no ComplexHalf kernel (bf16: FFT unsupported
             outright).  Kept in the registry so the benchmark records the
             blocker as an ``event: skip`` row instead of silently
             omitting the tier; fixing it needs model-code changes
             (an f32 island around ``fno.py`` FFT+einsum), out of scope.
compile      ``compile_default``, ``compile_maxautotune``
cudagraph    ``cudagraph_fp32``, ``cudagraph_fp16`` (autocast capture —
             same ComplexHalf blocker as ``torch_fp16``)
===========  =========================================================

``all`` = every group.  A runtime whose dependencies are missing in the
current venv is skipped with the reason recorded (``event: skip``) — the
benchmark is designed to run once per venv and merge logs.

Measurement discipline (also pinned by ``tests/test_deploy_latency.py``):

- warmup >= 20 calls (default 30), then >= 100 timed calls (default) OR
  enough calls to cover >= 5 s, whichever is larger, capped at 2000;
- timer: ``time.perf_counter`` around fully host-synchronous calls
  (torch tiers end in ``.cpu().numpy()``, TRT in a stream sync + D2H);
- p50 / p90 / mean / min / max in ms + rows/s throughput at p50;
- latency is WARM; cold start is a separate mode (below);
- GPU is pinned outside the process via ``CUDA_VISIBLE_DEVICES`` (``--gpu N``
  records it and sets it only if the environment variable is unset);
  ``meta`` records device name + driver via ``nvidia-smi``.

Cold start (``--cold trt_fp32,ort_gpu_fp32,...``): one fresh process per
runtime is the honest protocol (see the doc for the driver loop); inside
each process this mode times, in order: lazy import of the tier deps,
artifact load (session create / engine deserialize / checkpoint load +
model build), first call at B=1.  Loads exclude engine BUILD (recorded
separately as ``event: engine``) — a deployed service deserializes a
prebuilt plan.

Two venvs on the 5090 server (commands, GPU 2 only)::

    # phase A — torch tiers (GPU venv; also writes the reference npz)
    cd /nfs/wangxi/worktrees/deploy
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=src /nfs/wangxi/venvs/tensorlbm/bin/python \\
        benchmarks/deploy_latency.py --runtime torch_fp32,torch_fp16,compile,cudagraph \\
        --ckpt-dir /nfs/wangxi/runs/b4_serve_20260824/ckpts \\
        --run-dir /nfs/wangxi/runs/b4_v4_20260824 \\
        --out /nfs/wangxi/runs/deploy_latency_20260825/bench.jsonl \\
        --work-dir /nfs/wangxi/runs/deploy_latency_20260825

    # phase B — trt + ort tiers (deploy venv; needs LD_LIBRARY_PATH cudnn for
    # the CUDA EP — borrowed from the GPU venv, nothing installed there)
    NV=/nfs/wangxi/venvs/tensorlbm/lib/python3.12/site-packages/nvidia
    CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:$NV/cuda_runtime/lib \\
        /nfs/wangxi/venvs/deploy/bin/python benchmarks/deploy_latency.py \\
        --runtime trt,ort_cpu,ort_gpu --gpu 2
        --out /nfs/wangxi/runs/deploy_latency_20260825/bench.jsonl \\
        --work-dir /nfs/wangxi/runs/deploy_latency_20260825

The reference npz (``--reference``) carries the corpus query pattern and the
torch-CPU reference so phase B needs neither torch nor the corpus.  Schema:
``field`` (5, ny, nx) f32; ``B{b}_cond`` (b, 8) f64; ``B{b}_ref_member_cd``
(M, b) f64; ``n_members``, ``batches``, ``family``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_ONNX = "/nfs/wangxi/runs/b4_onnx_20260825/ensemble_cfull_stacked.onnx"
DEFAULT_CKPT_DIR = "/nfs/wangxi/runs/b4_serve_20260824/ckpts"
DEFAULT_RUN_DIR = "/nfs/wangxi/runs/b4_v4_20260824"
DEFAULT_OUT = "/nfs/wangxi/runs/deploy_latency_20260825/bench.jsonl"
DEFAULT_WORK_DIR = "/nfs/wangxi/runs/deploy_latency_20260825"

#: runtime group -> rows (order is the report order)
RUNTIME_GROUPS: dict[str, tuple[str, ...]] = {
    "trt": ("trt_fp32", "trt_tf32", "trt_f16"),
    "ort_cpu": ("ort_cpu",),
    "ort_gpu": ("ort_gpu_fp32", "ort_gpu_fp32_strict", "ort_gpu_f16", "ort_gpu_int8dyn"),
    "torch_fp32": ("torch_fp32_baseline", "torch_fp32_cached", "torch_fp32_strict"),
    "torch_fp16": ("torch_fp16_autocast",),
    "compile": ("compile_default", "compile_maxautotune"),
    "cudagraph": ("cudagraph_fp32", "cudagraph_fp16"),
}
ALL_RUNTIMES: tuple[str, ...] = tuple(r for rows in RUNTIME_GROUPS.values() for r in rows)
RUNTIME_GROUP_OF: dict[str, str] = {r: g for g, rows in RUNTIME_GROUPS.items() for r in rows}
RUNTIME_PRECISION: dict[str, str] = {
    "trt_fp32": "fp32",
    "trt_tf32": "tf32",
    "trt_f16": "fp16-body/fp32-tail",
    "ort_cpu": "fp64-tail/fp32-body",
    "ort_gpu_fp32": "fp32(tf32 default)",
    "ort_gpu_fp32_strict": "fp32",
    "ort_gpu_f16": "fp16-body/fp32-tail",
    "ort_gpu_int8dyn": "int8-dynamic",
    "torch_fp32_baseline": "fp32",
    "torch_fp32_cached": "fp32",
    "torch_fp32_strict": "fp32 (cudnn/matmul tf32 off)",
    "torch_fp16_autocast": "fp16-autocast/fp64-denorm",
    "compile_default": "fp32",
    "compile_maxautotune": "fp32",
    "cudagraph_fp32": "fp32",
    "cudagraph_fp16": "fp16-autocapt/fp64-denorm",
}


# ---------------------------------------------------------------------------
# JSONL output + small pure helpers (unit-tested)
# ---------------------------------------------------------------------------


def emit(out: Path, record: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def plan_iters(est_ms: float, *, min_iters: int, min_seconds: float, max_iters: int) -> int:
    """Timed-call count: >= min_iters, >= min_seconds of work, <= max_iters."""
    est_ms = max(float(est_ms), 1e-6)
    need = int(math.ceil(min_seconds * 1000.0 / est_ms))
    return max(int(min_iters), min(need, int(max_iters)))


def make_latency_record(
    runtime: str,
    batch: int,
    stats: dict[str, Any],
    parity: dict[str, Any] | None,
    *,
    precision: str,
    warmup: int,
    artifact: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one ``event: latency`` record (schema pinned by tests)."""
    record: dict[str, Any] = {
        "event": "latency",
        "runtime": runtime,
        "group": RUNTIME_GROUP_OF.get(runtime, "unknown"),
        "precision": precision,
        "batch": int(batch),
        "warmup": int(warmup),
        "iters": int(stats["iters"]),
        "total_s": float(stats["total_s"]),
        "p50_ms": float(stats["p50_ms"]),
        "p90_ms": float(stats["p90_ms"]),
        "mean_ms": float(stats["mean_ms"]),
        "min_ms": float(stats["min_ms"]),
        "max_ms": float(stats["max_ms"]),
        "throughput_rows_per_s_p50": float(batch) * 1000.0 / float(stats["p50_ms"]),
        "throughput_rows_per_s_mean": float(batch) * 1000.0 / float(stats["mean_ms"]),
        "parity": parity,
    }
    if artifact is not None:
        record["artifact"] = artifact
    if extras:
        record.update(extras)
    return record


def parity_vs_reference(got: np.ndarray, ref: np.ndarray) -> dict[str, Any]:
    """Per-member max|d| vs the torch-CPU reference (verify_ensemble_onnx metric)."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    if got.shape != ref.shape:
        raise ValueError(f"parity shape mismatch: got {got.shape}, ref {ref.shape}")
    per_lin = np.abs(got - ref).max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_got = np.log10(got)
        log_ref = np.log10(ref)
    if not np.isfinite(log_got).all() or not np.isfinite(log_ref).all():
        raise ValueError("non-positive C_D in parity inputs (log10 undefined)")
    per_log = np.abs(log_got - log_ref).max(axis=1)
    return {
        "max_abs_lin": float(per_lin.max()),
        "max_abs_log10": float(per_log.max()),
        "per_member_max_abs_log10": [float(x) for x in per_log],
    }


def time_cell(
    fn: Callable[[], Any],
    *,
    warmup: int,
    min_iters: int,
    min_seconds: float,
    max_iters: int,
) -> dict[str, Any]:
    """Warm up, size the run adaptively, time host-synchronous calls."""
    for _ in range(int(warmup)):
        fn()
    probes = []
    for _ in range(5):
        t0 = time.perf_counter()
        fn()
        probes.append((time.perf_counter() - t0) * 1e3)
    iters = plan_iters(
        float(np.median(probes)),
        min_iters=min_iters,
        min_seconds=min_seconds,
        max_iters=max_iters,
    )
    ts = []
    t_start = time.perf_counter()
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    total_s = time.perf_counter() - t_start
    arr = np.asarray(ts)
    return {
        "iters": iters,
        "total_s": round(total_s, 3),
        "p50_ms": float(np.quantile(arr, 0.50)),
        "p90_ms": float(np.quantile(arr, 0.90)),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
    }


def expand_runtimes(spec: str) -> list[str]:
    """Expand a comma list of groups / runtime names (``all`` = everything)."""
    wanted: list[str] = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if token == "all":
            wanted.extend(ALL_RUNTIMES)
        elif token in RUNTIME_GROUPS:
            wanted.extend(RUNTIME_GROUPS[token])
        elif token in RUNTIME_GROUP_OF:
            wanted.append(token)
        else:
            raise ValueError(
                f"unknown runtime or group {token!r}; known groups: {sorted(RUNTIME_GROUPS)}"
            )
    seen: set[str] = set()
    unique: list[str] = []
    for r in wanted:
        if r not in seen:
            seen.add(r)
            unique.append(r)
    return unique


def gpu_info() -> dict[str, Any]:
    """Device name + driver via nvidia-smi (None when unavailable)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
        first = out.stdout.strip().splitlines()[0].split(", ")
        return {"name": first[0].strip(), "driver": first[1].strip()}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"name": None, "driver": None, "error": str(exc)[:200]}


# ---------------------------------------------------------------------------
# environment probes (lazy imports — cold mode times them)
# ---------------------------------------------------------------------------

TRT_DEPLOY_PATH = Path(__file__).resolve().parents[1] / "src" / "tensorlbm" / "ai" / "trt_deploy.py"


def import_trt_deploy() -> Any:
    """Load ``trt_deploy`` also on torch-free venvs (the package init needs torch)."""
    try:
        from tensorlbm.ai import trt_deploy

        return trt_deploy
    except ImportError:
        import importlib.util

        spec = importlib.util.spec_from_file_location("tensorlbm_ai_trt_deploy", TRT_DEPLOY_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def probe_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    try:
        import onnx

        versions["onnx"] = onnx.__version__
    except ImportError:
        versions["onnx"] = None
    try:
        import onnxruntime

        versions["onnxruntime"] = onnxruntime.__version__
        versions["ort_providers"] = list(onnxruntime.get_available_providers())
    except ImportError:
        versions["onnxruntime"] = None
    try:
        import tensorrt

        versions["tensorrt"] = tensorrt.__version__
    except ImportError:
        versions["tensorrt"] = None
    try:
        import cuda

        versions["cuda_python"] = getattr(cuda, "__version__", "unknown")
    except ImportError:
        versions["cuda_python"] = None
    try:
        import torch

        versions["torch"] = torch.__version__
        versions["torch_cuda"] = torch.version.cuda
        versions["cuda_available"] = torch.cuda.is_available()
    except ImportError:
        versions["torch"] = None
    return versions


def timed_tier_import(tier: str) -> float:
    """Import the tier dependencies, return elapsed ms (cold-mode helper)."""
    t0 = time.perf_counter()
    if tier.startswith("torch") or tier.startswith("compile") or tier.startswith("cudagraph"):
        import torch  # noqa: F401

        from tensorlbm.ai.inference_service import load_checkpoint  # noqa: F401
    elif tier.startswith("trt"):
        import tensorrt  # noqa: F401

        import_trt_deploy()
    elif tier.startswith("ort"):
        import onnxruntime  # noqa: F401
    else:
        raise ValueError(f"no import block for runtime {tier!r}")
    return (time.perf_counter() - t0) * 1e3


# ---------------------------------------------------------------------------
# reference inputs (corpus query pattern + torch-CPU reference)
# ---------------------------------------------------------------------------


def make_reference(
    out_path: Path,
    ckpt_dir: Path,
    run_dir: Path,
    batches: list[int],
) -> dict[str, Any]:
    """Write the reference npz (query pattern + torch-CPU member C_D)."""
    from collections import Counter

    from tensorlbm.ai.inference_service import (
        ModelEnsembleBackend,
        load_checkpoint,
        load_corpus_index,
    )

    ckpts = [load_checkpoint(p) for p in sorted(ckpt_dir.glob("*.pt"))]
    if not ckpts:
        raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
    index = load_corpus_index(run_dir)
    counts = Counter(index.designs)
    family_key, n_family = counts.most_common(1)[0]
    rows = [i for i, key in enumerate(index.designs) if key == family_key]
    rows.sort(key=lambda i: index.re[i])
    field = index.fields[rows[0]]
    reference = ModelEnsembleBackend(ckpts, device="cpu")
    payload: dict[str, Any] = {
        "field": np.asarray(field, dtype=np.float32),
        "family": np.asarray(list(family_key)),
        "n_family_rows": np.asarray(n_family),
        "n_members": np.asarray(len(ckpts)),
        "batches": np.asarray(batches),
    }
    for b in batches:
        cond = np.take(index.cond[rows], np.arange(b) % n_family, axis=0)
        payload[f"B{b}_cond"] = np.asarray(cond, dtype=np.float64)
        payload[f"B{b}_ref_member_cd"] = reference.predict(field, cond)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    return {
        "path": str(out_path),
        "n_members": len(ckpts),
        "family": list(family_key),
        "n_family_rows": n_family,
        "batches": batches,
    }


class Inputs:
    """Reference npz view: one field + per-batch cond rows + references."""

    def __init__(self, path: Path, batches: list[int]) -> None:
        data = np.load(path)
        self.path = str(path)
        self.field = np.asarray(data["field"], dtype=np.float32)
        self.n_members = int(data["n_members"])
        self.cond: dict[int, np.ndarray] = {}
        self.ref: dict[int, np.ndarray] = {}
        for b in batches:
            self.cond[b] = np.asarray(data[f"B{b}_cond"], dtype=np.float64)
            self.ref[b] = np.asarray(data[f"B{b}_ref_member_cd"], dtype=np.float64)


# ---------------------------------------------------------------------------
# runtime construction
# ---------------------------------------------------------------------------


class Ctx:
    """Shared state: paths, inputs, cached artifacts and torch modules."""

    def __init__(self, args: argparse.Namespace, inputs: Inputs | None) -> None:
        self.args = args
        self.inputs = inputs
        self.work = Path(args.work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self._f32: Path | None = None
        self._f16: Path | None = None
        self._int8: Path | None = None
        self._plans: dict[str, Path] = {}
        self._fast_cache: dict[str, Any] = {}

    # ----- ONNX siblings (torch-free, need the onnx package) -----

    def f32_onnx(self) -> Path:
        f64_tail_to_f32 = import_trt_deploy().f64_tail_to_f32

        if self._f32 is None:
            dst = self.work / "ensemble_cfull_stacked_f32.onnx"
            if not dst.is_file() or self.args.fresh:
                report = f64_tail_to_f32(self.args.onnx, dst)
                report["event"] = "engine"
                emit(Path(self.args.out), report)
            self._f32 = dst
        return self._f32

    def f16_onnx(self) -> Path:
        build_f16_body_model = import_trt_deploy().build_f16_body_model

        if self._f16 is None:
            dst = self.work / "ensemble_cfull_stacked_f16body.onnx"
            if not dst.is_file() or self.args.fresh:
                report = build_f16_body_model(self.f32_onnx(), dst)
                report["event"] = "engine"
                emit(Path(self.args.out), report)
            self._f16 = dst
        return self._f16

    def int8_onnx(self) -> Path:
        if self._int8 is None:
            dst = self.work / "ensemble_cfull_stacked_int8dyn.onnx"
            if not dst.is_file() or self.args.fresh:
                import onnxruntime.quantization as quant

                t0 = time.perf_counter()
                quant.quantize_dynamic(
                    str(self.f32_onnx()), str(dst), weight_type=quant.QuantType.QInt8
                )
                report = {
                    "event": "engine",
                    "kind": "int8_dynamic",
                    "src": str(self.f32_onnx()),
                    "path": str(dst),
                    "seconds": round(time.perf_counter() - t0, 2),
                    "artifact_bytes": dst.stat().st_size,
                }
                emit(Path(self.args.out), report)
            self._int8 = dst
        return self._int8

    def trt_plan(self, name: str) -> Path:
        build_engine = import_trt_deploy().build_engine

        if name not in self._plans:
            clear_tf32 = name != "trt_tf32"
            src = self.f16_onnx() if name == "trt_f16" else self.f32_onnx()
            dst = self.work / f"ensemble_cfull_stacked_{name}.plan"
            if not dst.is_file() or self.args.fresh:
                report = build_engine(
                    src,
                    dst,
                    max_batch=max(self.args.batch_sizes),
                    opt_batch=8,
                    clear_tf32=clear_tf32,
                    workspace_gb=float(self.args.trt_workspace_gb),
                )
                report["event"] = "engine"
                emit(Path(self.args.out), report)
            self._plans[name] = dst
        return self._plans[name]

    # ----- torch tier (needs torch + tensorlbm; one FastEnsemble cache) -----

    def fast_ensemble(self, autocast16: bool) -> Any:
        key = f"fast{int(autocast16)}"
        if key not in self._fast_cache:
            import torch
            from torch import nn

            from tensorlbm.ai.inference_service import load_checkpoint

            ckpt_dir = Path(self.args.ckpt_dir)
            ckpts = [load_checkpoint(p) for p in sorted(ckpt_dir.glob("*.pt"))]
            if not ckpts:
                raise FileNotFoundError(f"no checkpoints under {ckpt_dir}")
            device = torch.device("cuda")

            class FastEnsemble(nn.Module):
                """All members, norms pre-uploaded once, denorm on host f64."""

                def __init__(self) -> None:
                    super().__init__()
                    self.models = nn.ModuleList([c.to_model(device) for c in ckpts])
                    self.ch_m = nn.ParameterList()
                    self.ch_s = nn.ParameterList()
                    self.p_m = nn.ParameterList()
                    self.p_s = nn.ParameterList()
                    for c in ckpts:
                        for lst, key_, shape in (
                            (self.ch_m, "ch_mean", (1, -1, 1, 1)),
                            (self.ch_s, "ch_std", (1, -1, 1, 1)),
                            (self.p_m, "p_mean", (-1,)),
                            (self.p_s, "p_std", (-1,)),
                        ):
                            t = torch.tensor(
                                np.asarray(c.norm[key_]), dtype=torch.float32, device=device
                            )
                            lst.append(nn.Parameter(t.reshape(shape), requires_grad=False))
                    self.y_mean = np.asarray([float(c.norm["y_mean"]) for c in ckpts])
                    self.y_std = np.asarray([float(c.norm["y_std"]) for c in ckpts])
                    self.autocast16 = autocast16
                    self.device = device

                def z(self, field: Any, cond: Any) -> Any:
                    outs = []
                    guard = torch.no_grad()
                    ctx = torch.autocast("cuda", dtype=torch.float16) if self.autocast16 else None
                    if ctx is not None:
                        with ctx:
                            for m, ch_m, ch_s, p_m, p_s in zip(
                                self.models, self.ch_m, self.ch_s, self.p_m, self.p_s
                            ):
                                xn = (field - ch_m) / ch_s
                                pn = (cond - p_m) / p_s
                                outs.append(m(xn.expand(pn.shape[0], -1, -1, -1), pn).float())
                    else:
                        with guard:
                            for m, ch_m, ch_s, p_m, p_s in zip(
                                self.models, self.ch_m, self.ch_s, self.p_m, self.p_s
                            ):
                                xn = (field - ch_m) / ch_s
                                pn = (cond - p_m) / p_s
                                outs.append(m(xn.expand(pn.shape[0], -1, -1, -1), pn))
                    return torch.stack(outs, 0)

                def predict(self, field_np: Any, cond_np: Any) -> Any:
                    f = torch.from_numpy(np.asarray(field_np, dtype=np.float32))
                    f = f.to(self.device).unsqueeze(0)
                    p = torch.from_numpy(
                        np.asarray(cond_np, dtype=np.float32).astype(np.float32)
                    ).to(self.device)
                    z = self.z(f, p).double().cpu().numpy()
                    return 10.0 ** (z * self.y_std[:, None] + self.y_mean[:, None])

            module = FastEnsemble().eval()
            self._fast_cache[key] = module
        return self._fast_cache[key]


class Runtime:
    """A constructed callable + metadata for one runtime row."""

    def __init__(
        self,
        name: str,
        predict: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        artifact: str | None = None,
        extras: dict[str, Any] | None = None,
        extras_fn: Callable[[], dict[str, Any]] | None = None,
        close: Callable[[], None] | None = None,
    ) -> None:
        self.name = name
        self.predict = predict
        self.artifact = artifact
        self.extras = extras or {}
        self.extras_fn = extras_fn
        self.close = close

    def record_extras(self) -> dict[str, Any]:
        """Extras at record time (lazily-filled dicts survive late capture)."""
        merged = dict(self.extras)
        if self.extras_fn is not None:
            merged.update(self.extras_fn())
        return merged


def build_runtime(name: str, ctx: Ctx) -> Runtime:
    """Construct one runtime (raises with a clean message when deps miss)."""
    if ctx.inputs is None:
        raise RuntimeError("no reference inputs (run with --make-reference or --reference)")

    if name in ("trt_fp32", "trt_tf32", "trt_f16"):
        TrtEnsembleBackend = import_trt_deploy().TrtEnsembleBackend

        plan = ctx.trt_plan(name)
        backend = TrtEnsembleBackend(plan, max_batch=max(ctx.args.batch_sizes))
        return Runtime(
            name,
            backend.predict,
            artifact=backend.plan_path,
            extras={"n_members": backend.n_members},
        )

    if name == "ort_cpu":
        import onnxruntime as ort

        session = ort.InferenceSession(str(ctx.args.onnx), providers=["CPUExecutionProvider"])
        field_port = session.get_inputs()[0].name

        def ort_predict(field: np.ndarray, cond: np.ndarray) -> np.ndarray:
            outs = session.run(
                None, {field_port: field[None].astype(np.float32), "cond": cond.astype(np.float32)}
            )
            return outs[0].astype(np.float64)

        return Runtime(
            name,
            ort_predict,
            artifact=str(ctx.args.onnx),
            extras={"providers": session.get_providers()},
        )

    if name in ("ort_gpu_fp32", "ort_gpu_fp32_strict", "ort_gpu_f16", "ort_gpu_int8dyn"):
        import onnxruntime as ort

        opts: dict[str, Any] = {"device_id": 0}
        if name == "ort_gpu_fp32_strict":
            opts["use_tf32"] = "0"
        model = {
            "ort_gpu_fp32": ctx.f32_onnx,
            "ort_gpu_fp32_strict": ctx.f32_onnx,
            "ort_gpu_f16": ctx.f16_onnx,
            "ort_gpu_int8dyn": ctx.int8_onnx,
        }[name]()
        session = ort.InferenceSession(
            str(model), providers=[("CUDAExecutionProvider", opts), "CPUExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in session.get_providers():
            raise RuntimeError(
                "CUDAExecutionProvider not active (missing libcudnn? set LD_LIBRARY_PATH)"
            )
        field_port = session.get_inputs()[0].name

        def ort_gpu_predict(field: np.ndarray, cond: np.ndarray) -> np.ndarray:
            outs = session.run(
                None, {field_port: field[None].astype(np.float32), "cond": cond.astype(np.float32)}
            )
            return outs[0].astype(np.float64)

        return Runtime(
            name,
            ort_gpu_predict,
            artifact=str(model),
            extras={"providers": session.get_providers(), "provider_options": opts},
        )

    if name == "torch_fp32_baseline":
        import torch

        from tensorlbm.ai.inference_service import ModelEnsembleBackend, load_checkpoint

        ckpts = [load_checkpoint(p) for p in sorted(Path(ctx.args.ckpt_dir).glob("*.pt"))]
        backend = ModelEnsembleBackend(ckpts, device="cuda")
        torch.cuda.synchronize()

        def baseline_predict(field: np.ndarray, cond: np.ndarray) -> np.ndarray:
            return backend.predict(field, cond)

        return Runtime(
            name,
            baseline_predict,
            artifact=str(Path(ctx.args.ckpt_dir)),
            extras={"torch": torch.__version__},
        )

    if name == "torch_fp32_strict":
        import torch

        from tensorlbm.ai.inference_service import ModelEnsembleBackend, load_checkpoint

        cudnn_tf32 = torch.backends.cudnn.allow_tf32
        matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        ckpts = [load_checkpoint(p) for p in sorted(Path(ctx.args.ckpt_dir).glob("*.pt"))]
        backend = ModelEnsembleBackend(ckpts, device="cuda")
        torch.cuda.synchronize()

        def restore_tf32() -> None:
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.backends.cuda.matmul.allow_tf32 = matmul_tf32

        return Runtime(
            name,
            backend.predict,
            artifact=str(Path(ctx.args.ckpt_dir)),
            extras={"cudnn_allow_tf32": False, "matmul_allow_tf32": False},
            close=restore_tf32,
        )

    if name in ("torch_fp32_cached", "torch_fp16_autocast"):
        module = ctx.fast_ensemble(name == "torch_fp16_autocast")
        return Runtime(name, module.predict, artifact="in-process FastEnsemble")

    if name in ("compile_default", "compile_maxautotune"):
        import torch

        mode = "default" if name == "compile_default" else "max-autotune"
        module = ctx.fast_ensemble(False)
        compiled = torch.compile(module.z, mode=mode, dynamic=False)
        compile_seconds: dict[int, float] = {}

        class CompilePredict:
            def __init__(self) -> None:
                self._first: dict[int, bool] = {}

            def __call__(self, field: np.ndarray, cond: np.ndarray) -> np.ndarray:
                b = int(cond.shape[0])
                f = torch.from_numpy(np.asarray(field, dtype=np.float32))
                f = f.to(module.device).unsqueeze(0)
                p = torch.from_numpy(np.asarray(cond, dtype=np.float32)).to(module.device)
                if b not in self._first:
                    t0 = time.perf_counter()
                    z = compiled(f, p)
                    torch.cuda.synchronize()
                    compile_seconds[b] = time.perf_counter() - t0
                    self._first[b] = True
                else:
                    z = compiled(f, p)
                arr = z.double().cpu().numpy()
                return 10.0 ** (arr * module.y_std[:, None] + module.y_mean[:, None])

        return Runtime(
            name,
            CompilePredict(),
            artifact="torch.compile",
            extras={"mode": mode, "compile_seconds_by_batch": compile_seconds},
        )

    if name in ("cudagraph_fp32", "cudagraph_fp16"):
        import torch

        module = ctx.fast_ensemble(name == "cudagraph_fp16")
        ny, nx = ctx.inputs.field.shape[1], ctx.inputs.field.shape[2]
        static_f = torch.zeros(1, 5, ny, nx, device=module.device)
        captured: dict[int, dict[str, Any]] = {}

        def graph_predict(field: np.ndarray, cond: np.ndarray) -> np.ndarray:
            b = int(cond.shape[0])
            if b not in captured:
                static_p = torch.zeros(b, 8, device=module.device)
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(3):
                        module.z(static_f, static_p)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                t0 = time.perf_counter()
                with torch.cuda.graph(graph):
                    static_out = module.z(static_f, static_p)
                torch.cuda.synchronize()
                captured[b] = {
                    "p": static_p,
                    "out": static_out,
                    "graph": graph,
                    "capture_seconds": time.perf_counter() - t0,
                }
            entry = captured[b]
            static_f.copy_(
                torch.from_numpy(np.asarray(field, dtype=np.float32)).to(module.device).unsqueeze(0)
            )
            entry["p"].copy_(torch.from_numpy(np.asarray(cond, dtype=np.float32)).to(module.device))
            entry["graph"].replay()
            arr = entry["out"].double().cpu().numpy()
            return 10.0 ** (arr * module.y_std[:, None] + module.y_mean[:, None])

        return Runtime(
            name,
            graph_predict,
            artifact="torch.cuda.CUDAGraph",
            extras_fn=lambda: {
                "capture_seconds_by_batch": {
                    b: round(e["capture_seconds"], 3) for b, e in sorted(captured.items())
                }
            },
        )

    raise ValueError(f"unknown runtime {name!r}")


# ---------------------------------------------------------------------------
# drivers
# ---------------------------------------------------------------------------


def run_bench(args: argparse.Namespace, runtimes: list[str]) -> int:
    inputs: Inputs | None = None
    ref_path = Path(args.reference)
    if args.make_reference or not ref_path.is_file():
        if args.make_reference or any(
            r
            in RUNTIME_GROUPS["torch_fp32"]
            + RUNTIME_GROUPS["torch_fp16"]
            + RUNTIME_GROUPS["compile"]
            + RUNTIME_GROUPS["cudagraph"]
            for r in runtimes
        ):
            info = make_reference(
                ref_path, Path(args.ckpt_dir), Path(args.run_dir), args.batch_sizes
            )
            emit(Path(args.out), {"event": "engine", "kind": "reference_inputs", **info})
            inputs = Inputs(ref_path, args.batch_sizes)
    if inputs is None and ref_path.is_file():
        inputs = Inputs(ref_path, args.batch_sizes)
    if inputs is None:
        emit(
            Path(args.out),
            {
                "event": "skip",
                "runtime": ",".join(runtimes),
                "reason": f"reference {args.reference} missing and no torch tier requested "
                f"to build it (run once with --make-reference in the GPU venv)",
            },
        )
        return 1

    ctx = Ctx(args, inputs)
    emit(
        Path(args.out),
        {
            "event": "meta",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "argv": sys.argv[1:],
            "runtimes": runtimes,
            "batch_sizes": args.batch_sizes,
            "warmup": args.warmup,
            "min_iters": args.min_iters,
            "min_seconds": args.min_seconds,
            "max_iters": args.max_iters,
            "timer": "time.perf_counter around host-synchronous calls",
            "env": {
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            },
            "gpu": gpu_info(),
            "versions": probe_versions(),
            "onnx": str(Path(args.onnx).resolve()),
            "reference": inputs.path if inputs else None,
            "work_dir": str(ctx.work),
        },
    )

    for name in runtimes:
        try:
            runtime = build_runtime(name, ctx)
        except Exception as exc:
            emit(
                Path(args.out),
                {"event": "skip", "runtime": name, "reason": f"{type(exc).__name__}: {exc}"},
            )
            print(f"SKIP {name}: {exc}", file=sys.stderr)
            continue
        assert inputs is not None
        for b in args.batch_sizes:
            cond = inputs.cond[b]
            ref = inputs.ref[b]
            parity: dict[str, Any] | None
            try:
                got = runtime.predict(inputs.field, cond)
                parity = parity_vs_reference(got, ref)
            except Exception as exc:
                emit(
                    Path(args.out),
                    {
                        "event": "skip",
                        "runtime": name,
                        "batch": b,
                        "reason": f"parity call failed: {type(exc).__name__}: {exc}",
                    },
                )
                continue
            stats = time_cell(
                lambda: runtime.predict(inputs.field, cond),
                warmup=args.warmup,
                min_iters=args.min_iters,
                min_seconds=args.min_seconds,
                max_iters=args.max_iters,
            )
            record = make_latency_record(
                name,
                b,
                stats,
                parity,
                precision=RUNTIME_PRECISION[name],
                warmup=args.warmup,
                artifact=runtime.artifact,
                extras=runtime.record_extras(),
            )
            emit(Path(args.out), record)
            print(
                f"{name} B={b}: p50 {record['p50_ms']:.3f} ms "
                f"p90 {record['p90_ms']:.3f} ms | parity log10 "
                f"{parity['max_abs_log10']:.2e}"
            )
        if runtime.close is not None:
            runtime.close()
    return 0


def run_cold(args: argparse.Namespace, runtimes: list[str]) -> int:
    """Cold-start breakdown; invoke ONE process per runtime for honesty."""
    if not Path(args.reference).is_file():
        emit(
            Path(args.out),
            {
                "event": "skip",
                "runtime": ",".join(runtimes),
                "reason": f"reference {args.reference} missing",
            },
        )
        return 1
    inputs = Inputs(Path(args.reference), [1])
    ctx = Ctx(args, inputs)
    for name in runtimes:
        try:
            t_import_ms = timed_tier_import(name)
        except ImportError as exc:
            emit(
                Path(args.out),
                {
                    "event": "cold",
                    "runtime": name,
                    "reason": f"import failed: {exc}",
                    "t_import_ms": None,
                },
            )
            continue
        try:
            t0 = time.perf_counter()
            runtime = build_runtime(name, ctx)
            t_load_ms = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            got = runtime.predict(inputs.field, inputs.cond[1])
            t_first_ms = (time.perf_counter() - t0) * 1e3
            parity = parity_vs_reference(got, inputs.ref[1])
            emit(
                Path(args.out),
                {
                    "event": "cold",
                    "runtime": name,
                    "batch": 1,
                    "t_import_ms": round(t_import_ms, 1),
                    "t_load_ms": round(t_load_ms, 1),
                    "t_first_call_ms": round(t_first_ms, 1),
                    "total_ms": round(t_import_ms + t_load_ms + t_first_ms, 1),
                    "parity_max_abs_log10": parity["max_abs_log10"],
                    "notes": "engine BUILD excluded (deserialize only); "
                    "run one process per runtime for process-level cold",
                },
            )
            print(
                f"cold {name}: import {t_import_ms:.0f} ms + load {t_load_ms:.0f} ms "
                f"+ first call {t_first_ms:.1f} ms"
            )
        except Exception as exc:
            emit(
                Path(args.out),
                {
                    "event": "cold",
                    "runtime": name,
                    "reason": f"{type(exc).__name__}: {exc}",
                    "t_import_ms": round(t_import_ms, 1),
                },
            )
            print(f"cold {name} FAILED: {exc}", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="deployment-latency benchmark (B4-P4a)")
    ap.add_argument(
        "--runtime",
        default="all",
        help=f"comma list of groups/runtimes; groups: {sorted(RUNTIME_GROUPS)}",
    )
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    ap.add_argument("--onnx", default=DEFAULT_ONNX, help="f64 fused artifact (PR #242)")
    ap.add_argument("--reference", default=str(Path(DEFAULT_WORK_DIR) / "reference_inputs.npz"))
    ap.add_argument(
        "--make-reference",
        action="store_true",
        help="(re)build the reference npz from ckpt dir + corpus run dir",
    )
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--batch-sizes", default="1,2,4,8")
    ap.add_argument("--warmup", type=int, default=30, help="warmup calls (min 20 by discipline)")
    ap.add_argument("--min-iters", type=int, default=100)
    ap.add_argument("--min-seconds", type=float, default=5.0)
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="records the pinned GPU; sets CUDA_VISIBLE_DEVICES if unset",
    )
    ap.add_argument("--trt-workspace-gb", type=float, default=16.0)
    ap.add_argument("--fresh", action="store_true", help="rebuild cached artifacts")
    ap.add_argument(
        "--cold", default=None, help="cold-start mode: comma list of runtimes, then exit"
    )
    args = ap.parse_args(argv)

    if args.gpu is not None and os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    args.batch_sizes = sorted({int(b) for b in args.batch_sizes.split(",") if b.strip()})
    if args.warmup < 20:
        print("warmup must be >= 20 (measurement discipline)", file=sys.stderr)
        return 2

    try:
        runtimes = expand_runtimes(args.runtime if args.cold is None else args.cold)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.cold is not None:
        return run_cold(args, runtimes)
    return run_bench(args, runtimes)


if __name__ == "__main__":
    raise SystemExit(main())
