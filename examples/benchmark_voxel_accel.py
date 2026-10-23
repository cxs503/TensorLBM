"""Benchmark: spatial hash-grid acceleration of STL voxelisation.

Times the brute-force reference against the accelerated path of
:mod:`tensorlbm.voxel_accel` on deterministic synthetic icospheres
(``20 * 4**n`` faces) across grid tiers, verifies bitwise parity
wherever both paths ran, and reports the CSR bin memory overhead.

Usage::

    # CPU table (default: skips brute combos estimated > ~10 min)
    python examples/benchmark_voxel_accel.py --device cpu

    # everything, including the multi-hour 128^3 brute rows
    python examples/benchmark_voxel_accel.py --device cpu --full-brute

    # GPU: adds the brute-force Triton and binned Triton rows
    CUDA_VISIBLE_DEVICES=0 python examples/benchmark_voxel_accel.py --device cuda

Results and analysis: ``docs/benchmarks/voxel_accel_benchmark.md``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from tensorlbm import voxel_accel
from tensorlbm.geometry_voxel import (
    q_field_reference,
    read_stl_triangles,
    solid_mask_parity_reference,
    voxelize_stl,
)
from tensorlbm.stl_geometry import make_icosphere_stl, write_stl

# (subdivisions, approximate face-count tag); exact faces = 20 * 4**n
FACE_TIERS = [
    (5, "2e4"),
    (6, "8e4"),
    (8, "1.3e6"),
]
GRID_TIERS = [
    ("34^3", (34, 34, 34), 15.5),
    ("128^3", (128, 128, 128), 60.0),
]
# Brute-force combos pruned without --full-brute (measured cost on the
# 2026-08-22 192-core CPU: g128 x 8e4 solid alone > 30 min).
BRUTE_PRUNE = {("128^3", "8e4"), ("128^3", "1.3e6")}
# fp32-vs-fp32 / fp32-vs-fp64 q dust; measured max 2.9e-6 (see docs).
Q_TOL = 4.0e-6


def mesh_path(workdir: Path, subdivisions: int, radius: float, side: int) -> Path:
    path = workdir / f"ico_s{subdivisions}_r{radius:g}.stl"
    if not path.exists():
        n = side // 2
        verts, faces = make_icosphere_stl((n, n, n), radius, subdivisions=subdivisions)
        write_stl(path, verts, faces, binary=True)
    return path


def timed(fn, *args, **kw):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn(*args, **kw)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def run_brute(tri, shape):
    solid, dt_s = timed(solid_mask_parity_reference, tri, shape)
    (bnd, q), dt_q = timed(q_field_reference, tri, solid)
    return (solid, bnd, q), dt_s, dt_q


def run_accel(tri, shape):
    solid, dt_s = timed(voxel_accel.solid_mask_parity_accelerated, tri, shape)
    (bnd, q), dt_q = timed(voxel_accel.q_field_accelerated, tri, solid)
    return (solid, bnd, q), dt_s, dt_q


def parity_tag(ref, acc, bitwise: bool) -> str:
    if not (torch.equal(ref[0], acc[0]) and torch.equal(ref[1], acc[1])):
        return "MASK MISMATCH"
    if torch.equal(ref[2], acc[2]):
        return "bitwise"
    if bitwise:
        return "q MISMATCH"
    dq = float((ref[2][ref[1]] - acc[2][acc[1]]).abs().max())
    return f"q diff {dq:.1e}" if dq <= Q_TOL else "q OUT OF TOL"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--workdir", default="/nfs/wangxi/tmp/voxel_bench", type=Path)
    parser.add_argument(
        "--full-brute",
        action="store_true",
        help="run brute rows that are pruned by default (hours on CPU)",
    )
    parser.add_argument("--faces", choices=[tag for _, tag in FACE_TIERS], nargs="*")
    parser.add_argument("--grids", choices=[name for name, _, _ in GRID_TIERS], nargs="*")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    dev = args.device
    rows = []
    for subdiv, tag in FACE_TIERS:
        if args.faces and tag not in args.faces:
            continue
        for gname, shape, radius in GRID_TIERS:
            if args.grids and gname not in args.grids:
                continue
            path = mesh_path(args.workdir, subdiv, radius, shape[-1])
            tri = read_stl_triangles(path, device=dev)
            n_faces = tri.shape[0]
            line = {"grid": gname, "faces": n_faces}

            # fp64 brute only on CPU: on GeForce cards fp64 throughput
            # makes it hours-long and it is not the production GPU path.
            brute = dev == "cpu" and not ((gname, tag) in BRUTE_PRUNE and not args.full_brute)
            if brute:
                (rs, rb, rq), dts, dtq = run_brute(tri, shape)
                line["brute"] = (dts, dtq)
            acc, dts_a, dtq_a = run_accel(tri, shape)
            line["accel"] = (dts_a, dtq_a)
            if brute:
                # Same-device fp64 pair: exact reductions + verbatim
                # arithmetic -> bit-identical, on CPU and CUDA alike.
                line["parity"] = parity_tag((rs, rb, rq), acc, bitwise=True)

            col = voxel_accel.build_column_bins(tri, shape)
            cell = voxel_accel.build_cell_bins(tri, shape)
            line["bins_mb"] = (col.memory_bytes + cell.memory_bytes) / 1e6

            if dev == "cuda":
                bt, dt_bt = timed(voxelize_stl, tri, shape, device=dev, check_watertight=False)
                line["brute_triton"] = dt_bt
                # fp64 torch brute is impractical on GeForce cards, so the
                # accelerated torch path is validated against brute Triton.
                line["parity"] = parity_tag(bt, acc, bitwise=False)
                at, dt_at = timed(
                    voxel_accel.voxelize_stl_accelerated,
                    tri,
                    shape,
                    device=dev,
                    check_watertight=False,
                    use_triton=True,
                )
                line["accel_triton"] = dt_at
                line["parity_triton"] = parity_tag(bt, at, bitwise=False)
            rows.append(line)

    print(f"device={dev} torch={torch.__version__}")
    print("| grid | faces | brute solid/q (s) | accel solid/q (s) | speedup | bins MB | parity |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if "brute" in r:
            bs, bq = r["brute"]
            as_, aq = r["accel"]
            cell = f"{bs:.2f} / {bq:.2f}"
            accel = f"{as_:.2f} / {aq:.2f}"
            speed = f"{(bs + bq) / (as_ + aq):.0f}x"
        else:
            cell = "skipped"
            as_, aq = r["accel"]
            accel = f"{as_:.2f} / {aq:.2f}"
            speed = "-"
        extra = ""
        if "brute_triton" in r:
            extra = (
                f"  \n  GPU triton: brute {r['brute_triton']:.2f}s, binned "
                f"{r['accel_triton']:.2f}s ({r['brute_triton'] / r['accel_triton']:.1f}x)"
            )
        par = r.get("parity", "-")
        if "parity_triton" in r:
            par += f"; triton {r['parity_triton']}"
        print(
            f"| {r['grid']} | {r['faces']} | {cell} | {accel} | {speed} "
            f"| {r['bins_mb']:.1f} | {par}{extra} |"
        )


if __name__ == "__main__":
    main()
