# CAD slider streaming echo — B4-P3a (2026-08-25)

The first end-to-end interactive loop of the ship-design surrogate roadmap:
geometry changes during hull design get real-time performance feedback in a
single process, with ensemble UQ and an explicit out-of-family verdict on
every answer.

- Pipeline: `src/tensorlbm/ai/geometry_pipeline.py` (`GeometryEchoPipeline`)
- HTTP layer: `app/backend/routers/drag_echo.py` (`/api/drag/echo/*`)
- Tests: `tests/test_geometry_echo.py`, `tests/test_drag_echo_api.py`
- Benchmark: `benchmarks/b4_echo_bench.py`
  (raw rows: `/nfs/wangxi/runs/b4_echo_20260825/bench_echo.jsonl`)

## Pipeline

    design sliders                     any DragSurrogateService (PR #239)
    sail/fin scale, L/D, nose,   ┌─────────────────────────────────────────┐
    stern, sail-x, hull type,    │  DragSurrogateService                   │
    u_in, Re list                │  ├─ EnvelopeMahalanobisGuardrail        │
          │                      │  │  (v3 condition space, chi2-calibrated)│
          ▼                      │  ├─ ModelEnsembleBackend (5 members)    │
    GeometryEchoPipeline          │  │  └─ predict_batch (G,5,ny,nx)       │
    ├─ suboff_component_counts   │  │     one forward per member           │
    │  (CAD predicates, CPU —    │  └─ corpus field cache (reference       │
    │   bit-exact fit-time       │     mid-plane fields, nearest design)   │
    │   decomposition)           │                                         │
    │  -> v3 geo block (4 ch)    └──────────────────▲──────────────────────┘
    ├─ LRU geometry cache                           │ (N,8) condition_v3 rows
    │  (u_in/Re excluded from the key)              │ + reference field
    ├─ condition_v3(re, u_in, sail, fin, geo) ──────┘
    ├─ guard.check -> hull-form downgrade (see honesty)
    └─ EchoResult: cd / lo / hi / std, guard verdict,
       params echo, unsupported_channels, timings_ms

Batch mode (`sweep_axis`) stacks G geometries x N Reynolds points into ONE
`predict_batch` call — one forward per ensemble member for the whole slider
curve. The arbitrary-STL demo path bypasses the CAD decomposition
(`voxelize.load_stl -> place_on_grid -> mask_from_stl`) and is always served
with the honesty downgrade below.

## Latency (measured 2026-08-25, 5 trained serving members, 64 Re points)

| scenario | device | p50 | notes |
|---|---|---|---|
| slider move (NEW geometry, cache cold) | GPU 4 (RTX 5090) | **42.2 ms** | target < 100 ms: PASS |
| ↳ phase breakdown (p50) | GPU 4 | geom 23.6 / cond 0.06 / guard 0.41 / ensemble 18.8 ms | decomposition dominates the CPU part |
| slider move (same geometry, LRU hit) | GPU 4 | 17.9 ms | Re-only interaction path |
| slider move (cold) | CPU (4 threads) | 1298 ms | ensemble forward 1275 ms dominates |
| sweep, 32 geometries x 64 Re, one call | GPU 4 | 1593 ms (49.6 ms/geometry) | vs 42 ms for a single geometry |
| STL sphere, grid (32,32,64) | GPU 4 | 57.1 ms | mask 49.2 ms |
| STL sphere, grid (64,64,128) | GPU 4 | 195.5 ms | mask 178.2 ms |
| raw `build_mask` (production grid) | CPU | 16.3 ms | NOT in the slider hot path — the serve config resolves the reference field from the corpus cache (`field_source` in `info` says which); measured and reported anyway |

Mask caching: the pipeline caches the decomposition + channels per geometry
(LRU, `u_in`/Re excluded from the key, default 16 slots,
`TENSORLBM_DRAG_ECHO_CACHE`); mask build itself is not on the hot path with
a corpus-attached service, so no mask cache was needed — the raw cost is
reported above and would still fit the budget (42.2 + ~16 ms < 100 ms).

Commands:

    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src python benchmarks/b4_echo_bench.py \
        --ckpt-dir /nfs/wangxi/runs/b4_serve_20260824/ckpts \
        --run-dir /nfs/wangxi/runs/b4_v4_20260824 \
        --out-dir /nfs/wangxi/runs/b4_echo_20260825 --device cuda

## Fit-time exactness

The serving checkpoints (arm C_full, `b4_serve_20260824`) were trained on
`cache_v4.npz` with `condition_v3(re, uin, sail, fin, geo)` where `geo` came
from the CAD-predicate decomposition. The echo pipeline rebuilds that
decomposition on CPU with the same operations and order:

- mother designs: `suboff_component_counts` is bit-identical to
  `drag_cond.suboff_geometry_features` (parametrised over hull types and
  scales, production grid) and `EchoGeometry.condition_rows` is bit-identical
  to `DragSurrogateService.condition_rows`;
- vs `cache_v4.npz`: the rebuilt 4-channel geo block equals the cached `geo`
  row bitwise on every unique design checked (test covers >= 20, measured on
  all 122);
- vs `cache_fam.npz` (`validate_against_cache`, 28 of 112 family points,
  spread over all four hull-form families): counts integer-equal 28/28,
  rebuilt generalised channel rows max abs diff **0.0** — `channels_bitwise:
  true`. Service C_D MAPE on those out-of-corpus family points: **16.6 %**,
  and every one of the 28 points carries a `review` verdict. That 16.6 % is
  the measured hull-form extrapolation gap the guardrail is responsible for
  flagging — it is reported, never hidden.

## The honesty contract

Out-of-family input must never silently produce confident numbers.

1. **Hull-form variants get an explicit downgrade.** The v3 hand channels
   barely move under the hull-form axes, so the channel-space guard alone
   answers `ok` for geometries the C_full corpus never contained — measured
   on the served ensemble: an `l_over_d_mult` sweep scored 1.14-1.23 (flag
   `ok`) while the served C_D trend ran **opposite** to the B4-fam family
   cache (served gap -0.77 vs cache +5.29 mean over matched-Re pairs, where
   the cache orders 25/25 pairs the same way). `_downgrade_hullform`
   therefore forces at least `review` on every non-mother design (underlying
   verdict preserved in the reasons) and `EchoResult.confident` requires
   `ok`. The trend-direction check exists in the tests as a documented
   non-assertion: the served model does NOT reproduce the family trend yet.
2. **Arbitrary STL cannot express the v3 channels** — the result lists all
   four in `unsupported_channels`, the condition block is an explicit
   mother-geometry proxy (`cond_proxy: "mother_geometry"` in `info`), the
   verdict is forced to `reject`, and descriptive any-mask channels
   (A_proj/A_side/A_top/V log-ratios + raw counts) are attached in `info`.
3. **The replay backend refuses what it cannot serve**: hull-form axes -> 404
   ("no archived variant geometries"), STL -> 404 (model ensemble only).

## SDF-v2 drop-in seam

All geometry knowledge enters through exactly two functions —
`suboff_component_counts` (v3 hand channels) and `generalised_mask_counts`
(any-mask descriptive channels) — plus the `geo` block of `EchoGeometry`.
The SDF-v2 encoder (`tensorlbm.ai.geom_encoder`, PR #235 lineage) drops in
at `GeometryEchoPipeline._geometry`: replace the `geo` block with encoder
latents, fit the guard in latent space (`EnvelopeMahalanobisGuardrail`
already accepts any feature matrix + names), and retire the STL
`unsupported_channels` downgrade. Nothing else in the pipeline touches
geometry.

## API

Env config follows PR #239: the underlying service reads
`TENSORLBM_DRAG_CKPT_DIR` / `TENSORLBM_DRAG_RUN_DIR` / `TENSORLBM_DRAG_*`;
the echo layer adds `TENSORLBM_DRAG_ECHO_DEVICE` (default: the drag device
or cpu) and `TENSORLBM_DRAG_ECHO_CACHE` (geometry LRU slots, default 16).
Run the app with the app dir as cwd and `PYTHONPATH=src:app`.

    # one slider move: 64 Reynolds points, one geometry
    curl -s localhost:8000/api/drag/echo/params -H 'content-type: application/json' -d '{
      "params": {"hull_type": "full", "sail_scale": 1.1, "u_in": 0.1},
      "re_list": [50, 64, 81, 100, 126, 158, 200, 251, 316, 398, 501, 631]
    }' | jq '{cd: .cd[0:3], guard: .guard.flag, confident, timings_ms: .info.timings_ms}'

    # slider curve: sweep L/D over 32 geometries x 64 Re, ONE batched call
    curl -s localhost:8000/api/drag/echo/sweep -H 'content-type: application/json' -d '{
      "axis": "l_over_d_mult",
      "values": [0.75, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
      "base_params": {"hull_type": "with_sail", "u_in": 0.1},
      "re_list": [100, 200, 400]
    }' | jq '.results[] | {value: .params.value, cd: .cd[0], guard: .guard.flag}'

    # arbitrary STL demo (always rejected with unsupported_channels)
    curl -s localhost:8000/api/drag/echo/stl \
      -F file=@my-hull.stl -F 're_list=[100, 200]' -F u_in=0.1 -F hull_type=full \
      | jq '{guard: .guard.flag, unsupported_channels, mask_counts: .info.mask_counts}'

    curl -s localhost:8000/api/drag/echo/health | jq

Every `EchoResult` carries `guard` (flag/score/reasons), `confident`,
`members`, `unsupported_channels` and `info.timings_ms`.
