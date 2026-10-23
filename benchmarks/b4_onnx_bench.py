"""B4-P3c · fused-ensemble ONNX export + latency benchmark.

Latency of the deployed ONNX artifact vs the torch ensemble backend at
the served query pattern — ONE geometry field swept over B Reynolds
condition rows, ``B in {1, 8, 64}``:

- torch ensemble (``ModelEnsembleBackend``) on CUDA (when available) and
  on CPU,
- the ONNX backend (``OnnxEnsembleBackend``) on the CPUExecutionProvider,
  with the runtime default thread pool and pinned to one intra-op thread,
- both fused designs (``stacked`` / ``unrolled``) so the shipped design
  choice is backed by numbers,

plus export wall time, artifact size and a spot parity check per batch
size (the artifact must not only be fast).  One JSON object per line is
appended to ``--out`` (``event`` field: ``meta`` / ``export`` /
``latency``), so several hosts/venvs append into one log.

Condition rows come from the largest design family of the corpus cache
(one geometry, Re sweep — the real query pattern); families smaller than
``B`` cycle their rows.

Usage (server)::

    # GPU host — GPU 4 only, torch-CUDA rows + ORT CPU rows
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:/nfs/wangxi/runs/b4_serve_20260824/pydeps \\
      /nfs/wangxi/venvs/tensorlbm/bin/python benchmarks/b4_onnx_bench.py \\
      --out /nfs/wangxi/runs/b4_onnx_20260825/bench.jsonl

    # OMP-pinned CPU host — torch-CPU (1 thread) rows
    OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src:/nfs/wangxi/runs/b4_serve_20260824/pydeps \\
      /nfs/wangxi/venvs/ci-cpu/bin/python benchmarks/b4_onnx_bench.py \\
      --out /nfs/wangxi/runs/b4_onnx_20260825/bench.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tensorlbm
from tensorlbm.ai.inference_service import (
    ModelEnsembleBackend,
    load_checkpoint,
    load_corpus_index,
)
from tensorlbm.ai.onnx_deploy import (
    ENSEMBLE_DESIGNS,
    OnnxEnsembleBackend,
    export_ensemble_onnx,
)

DEFAULT_CKPT_DIR = "/nfs/wangxi/runs/b4_serve_20260824/ckpts"
DEFAULT_RUN_DIR = "/nfs/wangxi/runs/b4_v4_20260824"
DEFAULT_OUT = "/nfs/wangxi/runs/b4_onnx_20260825/bench.jsonl"


def _emit(out: Path, record: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def time_calls(fn: Any, reps: int, device: torch.device) -> dict[str, float]:
    for _ in range(3):  # warmup: allocator, autotune, thread pools
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    arr = np.asarray(ts)
    return {
        "p50_ms": float(np.quantile(arr, 0.50)),
        "mean_ms": float(arr.mean()),
        "p95_ms": float(np.quantile(arr, 0.95)),
    }


def query_pattern(run_dir: Path, batches: list[int]) -> dict[str, Any]:
    """Largest design family: one field + its Re-swept condition rows."""
    index = load_corpus_index(run_dir)
    counts = Counter(index.designs)
    family_key, n_family = counts.most_common(1)[0]
    rows = [i for i, key in enumerate(index.designs) if key == family_key]
    rows.sort(key=lambda i: index.re[i])
    field = index.fields[rows[0]]
    cond_blocks = {b: np.take(index.cond[rows], np.arange(b) % n_family, axis=0) for b in batches}
    return {
        "field": field,
        "cond_blocks": cond_blocks,
        "family": list(family_key),
        "n_family_rows": n_family,
        "family_re": [float(index.re[i]) for i in rows],
        "n_corpus_rows": int(index.re.size),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--onnx-stem", default="ensemble_cfull", help="artifact filename stem")
    ap.add_argument("--designs", default=",".join(ENSEMBLE_DESIGNS))
    ap.add_argument("--batch-sizes", default="1,8,64")
    ap.add_argument("--reps", type=int, default=30, help="timed reps for fast backends")
    ap.add_argument("--reps-slow", type=int, default=5, help="timed reps for CPU torch forwards")
    ap.add_argument("--no-export", action="store_true", help="reuse existing artifacts")
    ap.add_argument("--skip-cuda", action="store_true")
    args = ap.parse_args(argv)

    designs = [d.strip() for d in args.designs.split(",") if d.strip()]
    batches = sorted({int(b) for b in args.batch_sizes.split(",") if b.strip()})
    out = Path(args.out)
    ckpt_paths = sorted(Path(args.ckpt_dir).glob("*.pt"))
    if not ckpt_paths:
        print(f"no checkpoints under {args.ckpt_dir}", file=sys.stderr)
        return 2
    ckpts = [load_checkpoint(p) for p in ckpt_paths]
    pattern = query_pattern(Path(args.run_dir), batches)
    have_cuda = torch.cuda.is_available() and not args.skip_cuda

    try:
        import onnxruntime as _ort

        ort_version = _ort.__version__
        ort_providers = list(_ort.get_available_providers())
    except ImportError:
        ort_version = None
        ort_providers = []

    _emit(
        out,
        {
            "event": "meta",
            "tensorlbm": tensorlbm.__file__,
            "torch": str(torch.__version__),
            "onnxruntime": ort_version,
            "ort_providers": ort_providers,
            "torch_threads": int(torch.get_num_threads()),
            "cuda_available": bool(have_cuda),
            "cuda_name": (torch.cuda.get_device_name(0) if have_cuda else None),
            "hostname": platform.node(),
            "python": sys.version.split()[0],
            "env": {
                "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
            "n_members": len(ckpts),
            "ckpt_dir": str(ckpt_paths[0].parent),
            "query_family": pattern["family"],
            "n_family_rows": pattern["n_family_rows"],
            "n_corpus_rows": pattern["n_corpus_rows"],
            "batch_sizes": batches,
            "designs": designs,
        },
    )
    print(
        f"ensemble: {len(ckpts)} members from {args.ckpt_dir}; "
        f"query family {pattern['family']} ({pattern['n_family_rows']} rows); "
        f"batches {batches}; designs {designs}"
    )

    onnx_dir = out.parent if out.parent != Path("") else Path(".")
    artifacts: dict[str, Path] = {}
    for design in designs:
        path = onnx_dir / f"{args.onnx_stem}_{design}.onnx"
        if args.no_export and path.is_file():
            report: dict[str, Any] = {
                "event": "export",
                "design": design,
                "reused": True,
                "export_ok": True,
                "path": str(path),
                "artifact_bytes": path.stat().st_size,
            }
        else:
            report = export_ensemble_onnx(ckpts, path, design=design)
            report = {"event": "export", "design": design, "reused": False, **report}
        _emit(out, report)
        if not report.get("export_ok", False):
            print(f"export FAILED for {design}: {report.get('blocker')}", file=sys.stderr)
            continue
        artifacts[design] = Path(report["path"])
        detail = (
            f", {report['graph_nodes']} nodes, {report['export_seconds']:.2f}s, "
            f"checker={report['checker']}"
            if "graph_nodes" in report
            else " (reused)"
        )
        print(f"export {design}: {report['artifact_bytes'] / 1e6:.1f} MB{detail}")

    cpu_backend = ModelEnsembleBackend(ckpts, device="cpu")
    field = pattern["field"]
    for batch in batches:
        cond = pattern["cond_blocks"][batch]
        # spot parity: artifact vs torch reference on the exact bench inputs
        row: dict[str, Any] = {
            "event": "latency",
            "backend": "torch-cpu",
            "design": None,
            "batch": batch,
            "torch_threads": int(torch.get_num_threads()),
            "parity_member_max_abs": 0.0,
        }
        row.update(
            time_calls(
                lambda: cpu_backend.predict(field, cond), args.reps_slow, torch.device("cpu")
            )
        )
        _emit(out, row)
        print(f"torch-cpu B={batch}: p50 {row['p50_ms']:.1f} ms")

    if have_cuda:
        cuda_backend = ModelEnsembleBackend(ckpts, device="cuda")
        for batch in batches:
            cond = pattern["cond_blocks"][batch]
            row = {"event": "latency", "backend": "torch-cuda", "design": None, "batch": batch}
            row.update(
                time_calls(
                    lambda: cuda_backend.predict(field, cond), args.reps, torch.device("cuda")
                )
            )
            _emit(out, row)
            print(f"torch-cuda B={batch}: p50 {row['p50_ms']:.1f} ms")

    for design, path in artifacts.items():
        for threads_label, thread_count in (("ort-default", None), ("ort-1thread", 1)):
            backend = OnnxEnsembleBackend(path, intra_op_threads=thread_count)
            for batch in batches:
                cond = pattern["cond_blocks"][batch]
                ref = cpu_backend.predict(field, cond)
                got = backend.predict(field, cond)
                row = {
                    "event": "latency",
                    "backend": f"onnx-cpu-{threads_label}",
                    "design": design,
                    "batch": batch,
                    "providers": backend.providers,
                    "parity_member_max_abs": float(np.abs(got - ref).max()),
                }
                row.update(
                    time_calls(lambda: backend.predict(field, cond), args.reps, torch.device("cpu"))
                )
                _emit(out, row)
                print(
                    f"onnx-cpu-{threads_label} {design} B={batch}: p50 {row['p50_ms']:.1f} ms "
                    f"(parity {row['parity_member_max_abs']:.2e})"
                )
    print(f"records appended to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
