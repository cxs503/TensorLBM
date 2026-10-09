"""B4-P1d · drag-surrogate serving latency benchmark.

End-to-end latency of ``DragSurrogateService.predict`` for one geometry
swept over a 64-point Reynolds grid, deep-ensemble backend, measured

- on CPU and on GPU (when available),
- with the default ``EnvelopeMahalanobisGuardrail`` and with
  ``NullGuardrail`` (isolates the guard overhead),
- per-call wall time (p50 / mean / p95 over ``--reps`` calls after
  warmup) plus the per-call breakdown into field/condition resolution,
  guard check and the ensemble forward passes.

Member checkpoints come from ``--ckpt-dir`` (the trained serving
ensemble).  Without checkpoints the benchmark synthesises a
production-architecture random-weight ensemble — latency-identical to
trained weights (same graph, same shapes) and labelled as such in the
report; it never masquerades as a quality result.

Usage::

    PYTHONPATH=src python benchmarks/b4_serve_benchmark.py \
        --ckpt-dir /nfs/wangxi/runs/b4_serve_20260824/ckpts \
        --run-dir /nfs/wangxi/runs/b4_v4_20260824 \
        --out /nfs/wangxi/runs/b4_serve_20260824/serve_benchmark.json

CPU/GPU threads and torch version are recorded with the timings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tensorlbm
from tensorlbm.ai.drag_cond import CondFNODrag
from tensorlbm.ai.inference_service import (
    CondDragCheckpoint,
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    ModelEnsembleBackend,
    NullGuardrail,
    load_corpus_index,
)

#: production architecture of the B4 v3/v4 runs (train_fno_v4.py ARCH_BASE)
ARCH_BASE = dict(
    in_ch=5, width=32, n_layers=4, modes=(16, 32), mlp_hidden=128, film_hidden=64, cond_dim=8
)
#: cheap twin for smoke runs on CPU-only hosts (random-weight mode only)
ARCH_SMALL = dict(
    in_ch=5, width=16, n_layers=2, modes=(8, 16), mlp_hidden=64, film_hidden=32, cond_dim=8
)
DEFAULT_RUN_DIR = "/nfs/wangxi/runs/b4_v4_20260824"
DEFAULT_CKPT_DIR = "/nfs/wangxi/runs/b4_serve_20260824/ckpts"


def syn_checkpoint(seed: int, arch: dict) -> CondDragCheckpoint:
    """Random-weight member (latency-only stand-in)."""
    torch.manual_seed(seed)
    model = CondFNODrag(**arch)
    return CondDragCheckpoint(
        arch=dict(arch),
        state_dict=model.state_dict(),
        norm=dict(
            ch_mean=np.zeros(5, dtype=np.float64),
            ch_std=np.ones(5, dtype=np.float64),
            p_mean=np.zeros(8, dtype=np.float64),
            p_std=np.ones(8, dtype=np.float64),
            y_mean=0.0,
            y_std=1.0,
        ),
        meta=dict(seed=seed, synthetic="random-weights latency stand-in"),
    )


def load_ensemble(ckpt_dir: Path, arch: dict) -> tuple[list[CondDragCheckpoint], bool]:
    from tensorlbm.ai.inference_service import load_checkpoint

    paths = sorted(ckpt_dir.glob("*.pt"))
    if not paths:
        return [syn_checkpoint(s, arch) for s in range(5)], False
    return [load_checkpoint(p) for p in paths], True


def time_calls(fn: Any, reps: int, device: torch.device) -> dict[str, float]:
    for _ in range(3):  # warmup (allocator, cudnn autotune, thread pools)
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
    ts_arr = np.asarray(ts)
    return dict(
        p50_ms=float(np.quantile(ts_arr, 0.50)),
        mean_ms=float(ts_arr.mean()),
        p95_ms=float(np.quantile(ts_arr, 0.95)),
    )


def run_device(
    ckpts: list[CondDragCheckpoint],
    index: Any,
    device_str: str,
    re_grid: np.ndarray,
    query: dict[str, Any],
    reps: int,
) -> dict[str, Any]:
    device = torch.device(device_str)
    guard = EnvelopeMahalanobisGuardrail(index.cond)
    backend = ModelEnsembleBackend(ckpts, device=device)
    svc_guard = DragSurrogateService(
        backend,
        guard,
        corpus_cache=index.fields,
        cache_re=index.re,
        cache_designs=list(index.designs),
    )
    svc_null = DragSurrogateService(
        backend,
        NullGuardrail(guard.feature_names),
        corpus_cache=index.fields,
        cache_re=index.re,
        cache_designs=list(index.designs),
    )
    res_guard = svc_guard.predict(**query, re_grid=re_grid)
    res_null = svc_null.predict(**query, re_grid=re_grid)
    np.testing.assert_allclose(res_guard.cd, res_null.cd, rtol=1e-12)

    with_guard = time_calls(lambda: svc_guard.predict(**query, re_grid=re_grid), reps, device)
    without_guard = time_calls(lambda: svc_null.predict(**query, re_grid=re_grid), reps, device)
    cond, _geo = svc_guard.condition_rows(
        query["hull_type"], query["sail_scale"], query["fin_scale"], re_grid, u_in=query.get("u_in")
    )
    field, _ = svc_guard._resolve_field(  # noqa: SLF001 — benchmark internals
        query["hull_type"],
        query["sail_scale"],
        query["fin_scale"],
        float(re_grid[0]),
        query.get("u_in", 0.1),
        None,
        None,
    )
    fwd = time_calls(lambda: backend.predict(field, cond), reps, device)
    guard_only = time_calls(lambda: guard.check(cond), reps, device)
    return dict(
        device=device_str,
        torch=str(torch.__version__),
        cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
        torch_threads=int(torch.get_num_threads()),
        n_members=len(ckpts),
        n_re=int(re_grid.size),
        with_guard_ms=with_guard,
        without_guard_ms=without_guard,
        ensemble_forward_ms=fwd,
        guard_check_ms=guard_only,
        guard_overhead_ms=dict(
            p50=with_guard["p50_ms"] - without_guard["p50_ms"],
            mean=with_guard["mean_ms"] - without_guard["mean_ms"],
        ),
        sample=dict(
            hull_type=res_guard.hull_type,
            sail_scale=res_guard.sail_scale,
            fin_scale=res_guard.fin_scale,
            guard_flag=res_guard.guard.flag,
            guard_score=res_guard.guard.score,
            cd_first_three=res_guard.cd[:3].tolist(),
            mean_std=float(res_guard.std.mean()),
        ),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out", default="")
    ap.add_argument("--n-re", type=int, default=64)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument(
        "--reps-cpu",
        type=int,
        default=5,
        help="separate rep count for CPU (forward passes are slow)",
    )
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument(
        "--synthetic-arch",
        default="prod",
        choices=["prod", "small"],
        help="arch of the random-weight stand-in (checkpoint mode ignores this)",
    )
    args = ap.parse_args(argv)

    print(f"tensorlbm: {tensorlbm.__file__}")
    ckpt_dir = Path(args.ckpt_dir)
    arch = ARCH_BASE if args.synthetic_arch == "prod" else ARCH_SMALL
    ckpts, real = load_ensemble(ckpt_dir, arch)
    print(
        f"ensemble: {len(ckpts)} members "
        f"({'checkpoints from ' + str(ckpt_dir) if real else 'RANDOM WEIGHTS (latency-only)'})"
    )

    run_dir = Path(args.run_dir)
    if (run_dir / "cache_v4.npz").is_file() or (run_dir / "cache.npz").is_file():
        index = load_corpus_index(run_dir)
        re_lo, re_hi = float(index.re.min()), float(index.re.max())
        query = dict(hull_type="full", sail_scale=1.0, fin_scale=1.0, u_in=0.1)
        source = str(run_dir)
    else:  # synthetic corpus so the benchmark runs anywhere (latency-only)
        from tensorlbm.ai.drag_cond import (
            condition_v3,
            geometry_channels,
            suboff_geometry_features,
        )
        from tensorlbm.ai.inference_service import CorpusIndex

        rng = np.random.default_rng(0)
        fields = rng.standard_normal((16, 5, 64, 128)).astype(np.float32)
        re_arr = np.geomspace(50.0, 100.0, 16)
        geo = geometry_channels(suboff_geometry_features("full", 1.0, 1.0))
        cond = condition_v3(
            re_arr, np.full(16, 0.1), np.ones(16), np.ones(16), np.broadcast_to(geo, (16, 4))
        )
        index = CorpusIndex(
            fields=fields, re=re_arr, designs=tuple([("full", 1.0, 1.0, 0.1)] * 16), cond=cond
        )
        re_lo, re_hi = 50.0, 100.0
        query = dict(hull_type="full", sail_scale=1.0, fin_scale=1.0, u_in=0.1)
        source = "synthetic (run dir absent — latencies valid, quality not)"
        print(f"[note] {run_dir} not found; using a synthetic corpus")

    re_grid = np.geomspace(re_lo, re_hi, args.n_re)
    devices = (
        [args.device]
        if args.device != "auto"
        else (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    )
    report: dict[str, Any] = dict(
        generated=time.strftime("%Y-%m-%d %H:%M:%S"),
        tensorlbm=tensorlbm.__file__,
        ensemble=dict(
            n_members=len(ckpts),
            checkpoints=real,
            dir=str(ckpt_dir),
            arch="checkpoints" if real else args.synthetic_arch,
        ),
        corpus=dict(source=source, re_range=[re_lo, re_hi]),
        query=dict(query, n_re=args.n_re, reps=args.reps, re_grid=re_grid.tolist()),
        results=[],
    )
    for dev in devices:
        reps = args.reps if dev != "cpu" else args.reps_cpu
        row = run_device(ckpts, index, dev, re_grid, query, reps)
        report["results"].append(row)
        wg, wo = row["with_guard_ms"], row["without_guard_ms"]
        print(
            f"[{dev:4s}] members={row['n_members']} n_re={row['n_re']} "
            f"threads={row['torch_threads']}  "
            f"with-guard p50={wg['p50_ms']:8.2f} ms  "
            f"without p50={wo['p50_ms']:8.2f} ms  "
            f"guard-only p50={row['guard_check_ms']['p50_ms']:.3f} ms  "
            f"forward p50={row['ensemble_forward_ms']['p50_ms']:8.2f} ms"
        )

    out = args.out or ""
    if out:
        Path(out).write_text(json.dumps(report, indent=2))
        print(f"wrote {out}")
    else:
        print(json.dumps(report["results"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
