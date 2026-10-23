# STL voxelisation acceleration benchmark (2026-08-22)

Uniform spatial hash-grid (CSR binned) acceleration of the STL →
`(solid, boundary, q)` voxelisation pipeline, targeting the A4 roadmap
entry "spatial acceleration of STL voxelisation, 10^6 triangles".

* Implementation: `src/tensorlbm/voxel_accel.py` (torch, any device) and
  binned Triton kernels in `src/tensorlbm/_voxel_kernels.py`.
* Entry point: `voxelize_stl(..., accelerate=True)`; optional
  `use_triton=True` for the binned GPU kernels.
* Reproduce: `python examples/benchmark_voxel_accel.py --device cpu`
  (add `--device cuda` for the GPU table; `--full-brute` re-enables the
  pruned multi-hour brute rows).
* Tests: `tests/test_voxel_accel.py` (parity, flag semantics, generator
  determinism, GPU kernels).

Hardware: 192-core CPU host + RTX 5090, `OMP_NUM_THREADS=16`,
torch 2.13 CPU / 2.11 cu128. Meshes are deterministic icospheres
(`make_icosphere_stl`, exactly `20 * 4^n` faces, byte-identical across
runs) with radius 45 % of the grid side.

## Method

Two cost centres dominate the brute-force pipeline:

1. **solid mask** — one +x ray per (y, z) column, tested against *all*
   triangles: O(ny·nz · T);
2. **q field** — Möller–Trumbore against *all* triangles for every
   boundary link: O(links · T).

The accelerated path bins triangle AABBs into a CSR structure — 2-D
(y, z) column bins for the ray pass (rays are axis-aligned, so the bin
lookup is exact), 3-D cell bins for the link pass (candidates from the
≤ 2×2×2 cell block around each link) — and evaluates the *identical*
per-pair arithmetic only on the binned candidates. Triangle AABBs are
padded by `1e-5 · bbox_diag + 1e-6` so the candidate set is a
conservative superset of the brute-force hits. A uniform grid beat a BVH
here because the queries are regular (one per column/link), the CSR
arrays vectorise cleanly in torch, and the same arrays feed the Triton
kernels with zero restructuring; BVH traversal is pointer-chasing and
would need a per-query kernel.

## Results — CPU (fp64 torch, seconds)

| grid | faces | brute solid / q | accel solid / q | speedup | CSR overhead |
|---|---|---|---|---|---|
| 34³ | 20 480 | 2.00 / 4.38 | 0.01 / 0.06 | 91× | 1.2 MB |
| 34³ | 81 920 | 11.94 / 17.90 | 0.02 / 0.08 | 298× | 2.5 MB |
| 34³ | 1 310 720 | 115.20 / 285.86 | 0.16 / 0.58 | 542× | 24.4 MB |
| 128³ | 20 480 | 181.62 / 20.50 | 0.02 / 0.11 | 1555× | 21.2 MB |
| 128³ | 81 920 | 664.64 / 85.67 | 0.02 / 0.15 | 4414× | 23.3 MB |
| 128³ | 1 310 720 | not run (extrapolated ≈ 3 h) | 0.18 / 0.89 | — | 51.5 MB |

The 10^6-triangle target lands at **0.74 s total on a 34³ grid and
1.07 s on 128³**, versus 6.7 min / hours brute-force. CSR memory is
2–52 MB, i.e. 2–40 bytes per triangle per pass.

Brute-force timings on this shared host swing up to ~3× between runs
(depends on co-tenant load), while the accelerated side is stable at
0.01–0.9 s everywhere; treat the speedup column as order-of-magnitude
and conservative. A joint re-run of both paths under identical load
(`bench_cpu_full.log`, 2026-08-22) reproduced **bitwise** parity on
every row and 174–874× speedups even with both benches competing for
cores.

## Results — GPU (RTX 5090, total seconds)

| grid | faces | brute Triton | binned Triton | accel torch (fp64) solid/q |
|---|---|---|---|---|
| 34³ | 20 480 | 0.44* | 0.03* | 0.24* / 0.03 |
| 34³ | 81 920 | 0.07 | 0.02 | 0.00 / 0.03 |
| 34³ | 1 310 720 | 0.99 | 0.02 | 0.01 / 0.03 |
| 128³ | 20 480 | 0.03 | 0.02 | 0.00 / 0.03 |
| 128³ | 81 920 | 0.12 | 0.02 | 0.00 / 0.03 |
| 128³ | 1 310 720 | 1.92 | 0.03 | 0.00 / 0.05 |

\* first row includes one-time Triton JIT / torch warm-up; an earlier
cold run measured 2.28 s (brute) and 0.69 s (binned) for the same
34³ / 20 480 entry.

At the 10^6-triangle target the accelerated paths need 0.03–0.05 s on
either grid versus ~1–2 s brute-force — 25–70× on top of an already
fast GPU baseline. The fp64 torch brute reference was not run on GPU
(GeForce fp64 throughput makes it hours-long and it is not the
production path); the accelerated torch path is instead validated
against brute Triton (masks bit-for-bit, q within the dust bound below).

## Numerical parity

* **torch path, same device: bit-for-bit.** `torch.equal` on all three
  outputs (solid, boundary, q) versus the brute-force reference — CPU
  and CUDA — across 8 configurations (sphere/ellipsoid, non-unit origin
  and spacing, coarse 80-face mesh, mesh fully outside the grid, mesh
  straddling the grid edge). The reductions stay exact by construction:
  ray-parity is an *integer* crossing count, and min-t is gathered with
  an order-invariant `scatter_reduce_ amin`.
* **binned Triton: masks bit-for-bit, q within fp32 dust.** Solid and
  boundary masks equal the brute-force Triton kernels exactly. `q`
  differs because FMA contraction differs between the brute
  `(BLOCK, TRI_CHUNK)` tile and the binned `(TRI_CHUNK,)` tile, and
  because fp32 Möller–Trumbore itself drifts furthest on the *coarsest*
  meshes. Full sweep over every benchmarked combination (259 680
  boundary entries each):

  | comparison | 34³ / 2e4 | 34³ / 8e4 | 34³ / 1.3e6 | 128³ / 2e4 | 128³ / 8e4 | 128³ / 1.3e6 |
  |---|---|---|---|---|---|---|
  | binned vs brute Triton, max abs Δq | 2.4e-7 | 2.4e-7 | 3.0e-7 | 2.7e-6 | 8.9e-7 | 6.0e-7 |
  | entries > 1e-6 | 0 | 0 | 0 | 13 | 0 | 0 |
  | accel torch fp64 vs brute Triton, max abs Δq | 3.6e-7 | 4.2e-7 | 3.0e-7 | 2.9e-6 | 1.4e-6 | 8.9e-7 |
  | entries > 1e-6 | 0 | 0 | 0 | 36 | 4 | 0 |

  **Zero** entries exceed 1e-3 in any configuration — a missed bin
  candidate (wrong min-t) would shift q by O(0.1) — and the q = 0.5
  unresolved sentinel never flips. Tests bound Δq at 4e-6 together with
  the bitwise mask/sentinel assertions.
* **cross-device (CUDA vs CPU reference): masks bit-for-bit, q ≤ 9e-7**
  (fp64 sum-order rounding before the fp32 cast). Same-device
  comparisons are the meaningful A/B and stay bitwise.
* `accelerate=False` (default) leaves the historical code path
  untouched — verified bit-identical.

## Recommendation

Enable `accelerate=True` whenever the mesh exceeds ~10^5 triangles (or
~10^4 on CPU with grids ≥ 128³); below that the brute path is already
sub-second and the CSR build is pure overhead. On CUDA,
`accelerate=True` alone (torch fp64) already wins everywhere and keeps
bit-for-bit parity with the reference; add `use_triton=True` for warm
steady-state runs where the extra ~1.5–2× over the torch path matters
and fp32 q precision (dust bound above) is acceptable.
