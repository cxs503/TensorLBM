"""B4-P3a · CAD slider streaming echo latency benchmark.

Per-slider-move latency budget of ``GeometryEchoPipeline`` — the first
end-to-end interactive loop (design params -> mask/channels -> condition ->
guard -> 5-member ensemble forward -> C_D + UQ + verdict):

- ``slider_move`` — one parameter change (a NEW geometry each call, cache
  cold, rotating values so the LRU never hits) x 64 Reynolds points,
  measured end-to-end plus the internal phase breakdown (geometry
  decomposition / mask build / condition / guard / ensemble forward);
- ``slider_move_warm`` — same geometry re-queried (LRU hit): the Re-only
  interaction path;
- ``sweep32`` — one ``sweep_axis`` call over 32 geometry points x 64 Re
  (single batched forward per member);
- ``stl_64`` / ``stl_128`` — the arbitrary-STL demo path (load + place +
  voxelize + synthetic field + forward) at the (32, 32, 64) and
  (64, 64, 128) grids;
- ``mask_build`` — raw :meth:`EchoGeometry.build_mask` cost, measured
  separately because the serve configuration resolves the reference field
  from the corpus cache (mask build is NOT in the slider hot path; the
  number is reported either way).

Ensemble: the 5 trained serving members from ``--ckpt-dir`` when present,
else a random-weight stand-in (latency-identical graph, labelled in the
report — never a quality claim).  Output: one JSON line per measurement
row to ``--out-dir/bench_echo.jsonl`` plus a summary on stdout with the
``< 100 ms per slider move`` verdict.

Usage::

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python benchmarks/b4_echo_bench.py \
        --ckpt-dir /nfs/wangxi/runs/b4_serve_20260824/ckpts \
        --run-dir /nfs/wangxi/runs/b4_v4_20260824 \
        --out-dir /nfs/wangxi/runs/b4_echo_20260825
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import tensorlbm
from tensorlbm.ai.drag_cond import CondFNODrag, SuboffGrid
from tensorlbm.ai.geometry_pipeline import GeometryEchoPipeline
from tensorlbm.ai.inference_service import (
    CondDragCheckpoint,
    CorpusIndex,
    DragSurrogateService,
    EnvelopeMahalanobisGuardrail,
    ModelEnsembleBackend,
    load_checkpoint,
    load_corpus_index,
)

ARCH_BASE = dict(
    in_ch=5, width=32, n_layers=4, modes=(16, 32), mlp_hidden=128, film_hidden=64, cond_dim=8
)
ARCH_SMALL = dict(
    in_ch=5, width=16, n_layers=2, modes=(8, 16), mlp_hidden=64, film_hidden=32, cond_dim=8
)
DEFAULT_RUN_DIR = "/nfs/wangxi/runs/b4_v4_20260824"
DEFAULT_CKPT_DIR = "/nfs/wangxi/runs/b4_serve_20260824/ckpts"
TARGET_MS = 100.0


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
    paths = sorted(ckpt_dir.glob("*.pt"))
    if not paths:
        return [syn_checkpoint(s, arch) for s in range(5)], False
    return [load_checkpoint(p) for p in paths], True


def load_index(run_dir: Path) -> tuple[CorpusIndex | None, str]:
    if (run_dir / "cache_v4.npz").is_file() or (run_dir / "cache.npz").is_file():
        return load_corpus_index(run_dir), str(run_dir)
    print(f"[note] {run_dir} not found; latencies valid, quality not (synthetic corpus)")
    return None, "synthetic"


def synthetic_index() -> CorpusIndex:
    from tensorlbm.ai.drag_cond import condition_v3, geometry_channels, suboff_geometry_features

    rng = np.random.default_rng(0)
    fields = rng.standard_normal((16, 5, 64, 128)).astype(np.float32)
    re_arr = np.geomspace(50.0, 100.0, 16)
    geo = geometry_channels(suboff_geometry_features("full", 1.0, 1.0))
    cond = condition_v3(
        re_arr, np.full(16, 0.1), np.ones(16), np.ones(16), np.broadcast_to(geo, (16, 4))
    )
    return CorpusIndex(
        fields=fields, re=re_arr, designs=tuple([("full", 1.0, 1.0, 0.1)] * 16), cond=cond
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def stats_ms(ts: list[float]) -> dict[str, float]:
    a = np.asarray(ts)
    return dict(
        p50_ms=float(np.quantile(a, 0.50)),
        mean_ms=float(a.mean()),
        p95_ms=float(np.quantile(a, 0.95)),
    )


def bench_slider_move(
    pipe: GeometryEchoPipeline, re_grid: np.ndarray, reps: int, device: torch.device
) -> dict[str, Any]:
    """Cold-geometry slider moves: a NEW design every call (LRU never hits)."""
    sails = np.geomspace(0.7, 1.6, reps + 3)  # distinct values, cache_size < reps
    pipe._cache.clear()  # noqa: SLF001 — benchmark controls the cache state
    phases: dict[str, list[float]] = {}
    totals: list[float] = []
    res = None
    for i in range(reps + 3):
        params = {"hull_type": "full", "sail_scale": float(sails[i % len(sails)])}
        _sync(device)
        t0 = time.perf_counter()
        res = pipe.predict_from_params(params, re_grid)
        _sync(device)
        dt = (time.perf_counter() - t0) * 1e3
        if i >= 3:  # drop warmup reps
            totals.append(dt)
            for k, v in res.info["timings_ms"].items():
                phases.setdefault(k, []).append(v)
    out: dict[str, Any] = dict(
        wall=stats_ms(totals),
        phases_ms={k: stats_ms(v) for k, v in phases.items()},
        phase_p50_sum_ms=float(sum(stats_ms(v)["p50_ms"] for v in phases.values())),
        n_re=int(re_grid.size),
        reps=reps,
        sample=dict(
            sail_scale=res.params["sail_scale"],
            guard_flag=res.guard.flag,
            cd_first_three=res.cd[:3].tolist(),
            field_source=res.info["field_source"],
        ),
    )
    return out


def bench_slider_move_warm(
    pipe: GeometryEchoPipeline, re_grid: np.ndarray, reps: int, device: torch.device
) -> dict[str, Any]:
    """Warm path: identical geometry re-queried (LRU hit, Re-only change)."""
    params = {"hull_type": "full", "sail_scale": 1.05}
    pipe.predict_from_params(params, re_grid[:1])
    totals = []
    for i in range(reps + 3):
        _sync(device)
        t0 = time.perf_counter()
        pipe.predict_from_params(params, re_grid)
        _sync(device)
        if i >= 3:
            totals.append((time.perf_counter() - t0) * 1e3)
    return dict(wall=stats_ms(totals), n_re=int(re_grid.size), reps=reps, cache_entries=1)


def bench_sweep(
    pipe: GeometryEchoPipeline, re_grid: np.ndarray, n_geom: int, reps: int, device: torch.device
) -> dict[str, Any]:
    values = list(np.geomspace(0.75, 1.3, n_geom))
    totals = []
    res = None
    for i in range(reps + 2):
        _sync(device)
        t0 = time.perf_counter()
        results = pipe.sweep_axis(
            "l_over_d_mult", values, {"hull_type": "full", "u_in": 0.1}, re_grid
        )
        _sync(device)
        if i >= 2:
            totals.append((time.perf_counter() - t0) * 1e3)
        res = results[-1]
    return dict(
        wall=stats_ms(totals),
        n_geom=n_geom,
        n_re=int(re_grid.size),
        per_geometry_ms=float(np.mean(totals) / n_geom),
        reps=reps,
        sample=dict(
            axis="l_over_d_mult",
            last_value=res.params["value"],
            guard_flag=res.guard.flag,
            cd_first_three=res.cd[:3].tolist(),
        ),
    )


def bench_mask_build(pipe: GeometryEchoPipeline, reps: int) -> dict[str, Any]:
    """Raw voxelisation cost of one new design at the pipeline grid."""
    bundle = pipe._geometry({"hull_type": "full", "sail_scale": 1.07})
    bundle.build_mask()
    ts = []
    for i in range(reps + 2):
        t0 = time.perf_counter()
        mask = bundle.build_mask()
        if i >= 2:
            ts.append((time.perf_counter() - t0) * 1e3)
    return dict(
        wall=stats_ms(ts),
        grid=dict(nz=pipe.grid.nz, ny=pipe.grid.ny, nx=pipe.grid.nx),
        n_solid_voxels=int(mask.sum()),
        reps=reps,
        note="mask build is not in the slider hot path when the corpus field cache is attached",
    )


def write_sphere_stl(path: Path, *, n_theta: int = 24, n_phi: int = 32) -> None:
    import math

    def pt(i: int, j: int) -> tuple[float, float, float]:
        th = math.pi * i / n_theta
        ph = 2.0 * math.pi * j / n_phi
        return (math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th))

    tris: list[tuple[tuple[float, float, float], ...]] = []
    for j in range(n_phi):
        j2 = (j + 1) % n_phi
        tris.append((pt(0, 0), pt(1, j2), pt(1, j)))
        tris.append((pt(n_theta - 1, j), pt(n_theta - 1, j2), pt(n_theta, 0)))
    for i in range(1, n_theta - 1):
        for j in range(n_phi):
            j2 = (j + 1) % n_phi
            tris.append((pt(i, j), pt(i, j2), pt(i + 1, j2)))
            tris.append((pt(i, j), pt(i + 1, j2), pt(i + 1, j)))
    lines = ["solid sphere"]
    for v0, v1, v2 in tris:
        lines.append("  facet normal 0 0 0")
        lines.append("    outer loop")
        for v in (v0, v1, v2):
            lines.append(f"      vertex {v[0]:.9e} {v[1]:.9e} {v[2]:.9e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append("endsolid sphere")
    path.write_text("\n".join(lines) + "\n")


def bench_stl(
    service: DragSurrogateService, resolution: int, re_grid: np.ndarray, reps: int
) -> dict[str, Any]:
    """STL demo path at a given grid resolution (separate pipeline per grid)."""
    pipe = GeometryEchoPipeline(service, grid=SuboffGrid.from_resolution(resolution), device="cpu")
    with tempfile.TemporaryDirectory() as td:
        stl = Path(td) / "sphere.stl"
        write_sphere_stl(stl)
        pipe.predict_from_stl(stl, re_grid[:1])  # warmup
        ts = []
        res = None
        for i in range(reps + 2):
            t0 = time.perf_counter()
            res = pipe.predict_from_stl(stl, re_grid)
            if i >= 2:
                ts.append((time.perf_counter() - t0) * 1e3)
    return dict(
        wall=stats_ms(ts),
        phases_ms={k: stats_ms([v]) for k, v in res.info["timings_ms"].items()},
        grid=dict(nz=pipe.grid.nz, ny=pipe.grid.ny, nx=pipe.grid.nx),
        n_re=int(re_grid.size),
        reps=reps,
        sample=dict(
            watertight=res.info["stl"]["watertight"],
            n_triangles=res.info["stl"]["n_triangles"],
            mask_counts=res.info["mask_counts"],
            guard_flag=res.guard.flag,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_DIR)
    ap.add_argument("--run-dir", default=DEFAULT_RUN_DIR)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--n-re", type=int, default=64)
    ap.add_argument("--n-geom", type=int, default=32)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--reps-cpu", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--synthetic-arch", default="prod", choices=["prod", "small"])
    args = ap.parse_args(argv)

    print(f"tensorlbm: {tensorlbm.__file__}")
    arch = ARCH_BASE if args.synthetic_arch == "prod" else ARCH_SMALL
    ckpts, real = load_ensemble(Path(args.ckpt_dir), arch)
    print(
        f"ensemble: {len(ckpts)} members "
        f"({'checkpoints from ' + args.ckpt_dir if real else 'RANDOM WEIGHTS (latency-only)'})"
    )
    index, source = load_index(Path(args.run_dir))
    if index is None:
        index = synthetic_index()
    re_grid = np.geomspace(float(index.re.min()), float(index.re.max()), args.n_re)

    devices = (
        [args.device]
        if args.device != "auto"
        else (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    )
    rows: list[dict[str, Any]] = []
    for dev_str in devices:
        device = torch.device(dev_str)
        reps = args.reps if dev_str != "cpu" else args.reps_cpu
        backend = ModelEnsembleBackend(ckpts, device=dev_str)
        service = DragSurrogateService(
            backend,
            EnvelopeMahalanobisGuardrail(index.cond),
            corpus_cache=index.fields,
            cache_re=index.re,
            cache_designs=list(index.designs),
        )
        pipe = GeometryEchoPipeline(service, cache_size=8)
        base = dict(
            device=dev_str,
            torch=str(torch.__version__),
            cuda_name=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
            torch_threads=int(torch.get_num_threads()),
            n_members=len(ckpts),
            corpus_source=source,
            checkpoints=real,
        )
        slider = bench_slider_move(pipe, re_grid, reps, device)
        warm = bench_slider_move_warm(pipe, re_grid, reps, device)
        sweep = bench_sweep(pipe, re_grid, args.n_geom, max(3, reps // 2), device)
        maskb = bench_mask_build(pipe, reps)
        rows.append(dict(scenario="slider_move", cold_geometry=True, **base, **slider))
        rows.append(dict(scenario="slider_move_warm", **base, **warm))
        rows.append(dict(scenario="sweep", **base, **sweep))
        rows.append(dict(scenario="mask_build", **base, **maskb))
        print(
            f"[{dev_str:4s}] slider cold p50={slider['wall']['p50_ms']:8.2f} ms "
            f"(phases p50: geom {slider['phases_ms'].get('geometry_s', {}).get('p50_ms', 0):.1f} "
            f"+ cond {slider['phases_ms'].get('condition_s', {}).get('p50_ms', 0):.2f} "
            f"+ guard {slider['phases_ms'].get('guard_s', {}).get('p50_ms', 0):.2f} "
            f"+ ens {slider['phases_ms'].get('ensemble_s', {}).get('p50_ms', 0):.1f})  "
            f"warm p50={warm['wall']['p50_ms']:7.2f} ms  "
            f"sweep32 p50={sweep['wall']['p50_ms']:8.1f} ms "
            f"({sweep['per_geometry_ms']:.1f} ms/geom)  "
            f"mask_build p50={maskb['wall']['p50_ms']:.1f} ms"
        )
        if dev_str == "cuda":
            stl64 = bench_stl(service, 64, re_grid, max(3, reps // 2))
            stl128 = bench_stl(service, 128, re_grid, max(3, reps // 2))
            rows.append(dict(scenario="stl", **base, **stl64))
            rows.append(dict(scenario="stl", **base, **stl128))
            print(
                f"[{dev_str:4s}] stl grid32x32x64 p50={stl64['wall']['p50_ms']:8.1f} ms  "
                f"grid64x64x128 p50={stl128['wall']['p50_ms']:8.1f} ms "
                f"(mask p50 {stl64['phases_ms']['mask_s']['p50_ms']:.1f} / "
                f"{stl128['phases_ms']['mask_s']['p50_ms']:.1f} ms)"
            )

    verdict = (
        "PASS"
        if all(r["wall"]["p50_ms"] < TARGET_MS for r in rows if r["scenario"] == "slider_move")
        else "FAIL"
    )
    summary = dict(
        target_ms=TARGET_MS,
        verdict=verdict,
        slider_move_p50_ms={
            r["device"]: r["wall"]["p50_ms"] for r in rows if r["scenario"] == "slider_move"
        },
    )
    print(f"per-slider-move < {TARGET_MS:.0f} ms: {verdict} {summary['slider_move_p50_ms']}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "bench_echo.jsonl"
        with out.open("a") as fh:  # JSON lines, append per run
            for r in rows:
                fh.write(json.dumps(dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), **r)) + "\n")
            fh.write(
                json.dumps(
                    dict(
                        generated=time.strftime("%Y-%m-%d %H:%M:%S"), scenario="summary", **summary
                    )
                )
                + "\n"
            )
        print(f"wrote {out}")
    else:
        print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
