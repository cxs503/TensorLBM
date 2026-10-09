# B4-P1d — SUBOFF drag-surrogate inference service (2026-08-24)

Phase-1 closing deliverable: a serving layer over the B4 conditional-FNO
drag surrogate with deep-ensemble uncertainty quantification and a
pluggable extrapolation guardrail, plus an optional FastAPI router, an
ONNX export path and latency benchmarks.

- Module: `src/tensorlbm/ai/inference_service.py`
- HTTP: `app/backend/routers/drag_surrogate.py` (mounted at `/api/drag`)
- Tests: `tests/test_inference_service.py` (34), `tests/test_drag_surrogate_api.py` (7)
- Benchmarks: `benchmarks/b4_guard_calibration.py`, `benchmarks/b4_serve_benchmark.py`
- Artifacts (5090 server): `/nfs/wangxi/runs/b4_serve_20260824/`
  (checkpoints, metrics, calibration tables, benchmark JSON, ONNX)

## 1. Design

```
(hull_type, sail_scale, fin_scale, re_grid)          one design, many Re
        │  DragSurrogateService.predict
        ├─ condition_rows()      -> (N, 8) condition_v3 rows (CAD geometry
        │                           channels evaluated on the service grid)
        ├─ guard.check(cond)     -> GuardVerdict(flag, score, reasons)
        ├─ backend.predict(...)  -> (M, N) member matrix
        │     ModelEnsembleBackend   M checkpoints, one batched forward each
        │     ReplayEnsembleBackend  archived per-seed preds + log10-Re
        │                           interpolation per member (no ckpt needed)
        └─ ensemble_stats()      -> mean / std(ddof=1) / min-max band
```

Key decisions:

- **Both backends expose the same `(M, N)` member matrix**, so UQ and the
  HTTP contract are backend-agnostic; replay mode exists because the v3/v4
  training runs archived predictions but not weights.
- **Serving ensemble** = 5-seed `C_full` (quota sampling + aux head) on the
  v4 **random** split (274-point corpus, fit 186 / val 33 / test 55,
  byte-identical split machinery re-derivation — verified
  `fit186/val33/test55` against the archived run). Random split, not LOHO:
  at serve time all hull families should be in training; out-of-family
  protection is the guardrail's job, not the split's.
- **Checkpoints** bundle arch + `state_dict` + the fit-time normalisation
  (`ch_mean/ch_std`, `p_mean/p_std`, `y_mean/y_std`) + meta
  (`CondDragCheckpoint`, `save_checkpoint`/`load_checkpoint`, format
  version 1). Without the normalisation the z-scored training target is
  not invertible from a bare state_dict.
- **Guard is verdict-only**: a `reject` never suppresses the numbers —
  the response carries both, and the caller decides presentation.

### Python usage

```python
from tensorlbm.ai.inference_service import DragSurrogateService, load_corpus_index

index = load_corpus_index("/nfs/wangxi/runs/b4_v4_20260824")
service = DragSurrogateService.from_checkpoints(
    [f"/nfs/wangxi/runs/b4_serve_20260824/ckpts/serve_cfull_s{k}.pt" for k in range(5)],
    index.cond,                      # guard fit matrix (condition_v3)
    corpus_cache=index.fields,       # (274, 5, 64, 128) mid-plane cache
    cache_re=index.re, cache_designs=list(index.designs),
    device="cuda",
)
res = service.predict("with_sail", sail_scale=1.5, fin_scale=1.0,
                      re_grid=[50.0, 100.0, 200.0, 400.0, 800.0])
res.cd        # ensemble mean C_D per Re
res.uq_dict() # {"lo", "hi", "std", "mean_std"}  min-max band + per-point std
res.guard     # GuardVerdict(flag="ok"|"review"|"reject", score, reasons)
```

Replay mode (no checkpoints required) serves archived LOHO members:

```python
service = DragSurrogateService.from_run_dir(
    "/nfs/wangxi/runs/b4_v4_20260824", arm="C_full", fold="loho::full"
)
```

`grid=` must be the grid the guard features were computed on (production
grid for the real caches); a mismatch moves the geometry channels into
the wrong feature space and is caught by a dedicated regression test.

### HTTP API

`POST /api/drag` (one design, batch of Re) and `GET /api/drag/health`:
registered in `app/backend/main.py` via the existing loader/registry
pattern; the router is import-safe without `/nfs` artifacts and answers
503 with the reason when no backend is configured.

```bash
curl -X POST http://<host>:8000/api/drag -H 'Content-Type: application/json' \
  -d '{"hull_type": "with_sail", "sail_scale": 1.5, "fin_scale": 1.0,
       "re_grid": [50, 100, 200, 400, 800]}'
# -> {"re": [...], "cd": [...], "uq": {"lo": [...], "hi": [...], "std": [...],
#      "mean_std": ...}, "guard": {"flag": "ok", "score": ..., "reasons": []},
#     "backend": "model", "members": ["s0", ...], "info": {...}}
```

Configuration (env): `TENSORLBM_DRAG_CKPT_DIR` / `TENSORLBM_DRAG_CKPT_FILES`
(model backend; wins if set) or `TENSORLBM_DRAG_RUN_DIR` +
`TENSORLBM_DRAG_ARM` + `TENSORLBM_DRAG_FOLD` (replay backend, default
`/nfs/wangxi/runs/b4_v4_20260824`, `C_full`, `loho::full`);
`TENSORLBM_DRAG_DEVICE` (default `cpu`).

## 2. Deep-ensemble UQ

Per Reynolds point: mean over members, std (ddof=1), min-max band. The
`(M, N)` member matrix is also the offline analysis unit
(`ensemble_picp`, `error_std_spearman`, `guard_threshold_sweep`).

Measured on the v4 archive (arm `C_full`, per LOHO fold, ensemble of the
seed members; `benchmarks/b4_guard_calibration.py`,
`guard_calibration.{json,md}`):

| fold | fit | test | members | MAPE % | PICP(min-max) | Spearman(std, err) |
|---|---|---|---|---|---|---|
| loho::bare_hull | 221 | 14 | 5 | 2.13 | **0.00** | +0.93 |
| loho::with_sail | 171 | 72 | 3 | 1.19 | 0.47 | −0.06 |
| loho::full | 74 | 188 | 5 | 5.44 | 0.61 | +0.66 |

Serving ensemble (this work, random split, 5 seeds):
member MAPE 0.56 / 0.65 / 0.30 / 0.37 / 0.39 %, **ensemble MAPE 0.349 %**,
PICP(min-max) 0.636, Spearman(std, err) +0.285.

Honest reading:

- The 5-member **min-max band is not a calibrated interval** — PICP 0.47-0.64
  in-family, and **0.00 on the held-out hull family** (bare): there the
  members agree with each other and are all wrong together, while the
  per-point std still ranks the errors almost perfectly (Spearman +0.93).
  Treat `std` as a *relative* epistemic-severity signal, not a coverage
  guarantee; calibrated intervals would need conformal post-processing
  on top (not in scope for Phase 1).
- In-family UQ behaves: `loho::full` (the bimodal fold) shows both the
  worst errors and the strongest std-error correlation.

## 3. Extrapolation guardrail

`Guardrail` is a `Protocol` (`feature_names`, `row_scores`, `row_reasons`,
`check`) over an arbitrary `(N, D)` feature space — the service maps a
design query to rows before calling it, so a guard never sees hull types
or Reynolds numbers. Default implementation
`EnvelopeMahalanobisGuardrail`: per-dimension envelopes (+5 % margin of
range) **and** Mahalanobis distance with shrunk covariance
(`0.9·S + 0.1·tr(S)/D·I`; `solid_frac` is an exact linear function of
`sail_frac`/`fin_frac`, so the raw covariance is near-singular).

Default thresholds are chi-square calibrated: for an in-distribution row
`d² ~ chi2(D)`, so `review = sqrt(chi2_{0.99}) = 4.48`,
`reject = sqrt(chi2_{0.999}) = 5.13` at D=8 (Wilson-Hilferty quantiles;
no scipy dependency). `NullGuardrail` exists for A/B latency and as an
explicit opt-out.

**SDF-latent drop-in (no unmerged import).** The guard is
feature-agnostic: `EnvelopeMahalanobisGuardrail(latents, names=("z0", …,
"z31"))` over the 238×32 matrix of PR #235 (`latents.npz`, key
`sdf_joint`) is the latent-distance guard. Phase 1 stops at the
interface; the latent variant is Phase-2 work on top of #235.

### Threshold calibration (LOHO archive, 274 pooled fold-test rows)

`guard_threshold_sweep` flags rows with score ≥ threshold and measures
how many ≥ X %-error rows are caught (`choose_threshold` = minimum
flagged fraction achieving 80 % capture):

| large-error level | chosen threshold | capture | flagged | precision |
|---|---|---|---|---|
| ≥ 2 % | 1.16 | 86 % | **85 %** | 48 % |
| ≥ 5 % | **5.30** | 82 % | 25 % | 71 % |
| ≥ 10 % | 6.06 | 80 % | 20 % | 60 % |

- The ≥2 % regime is unusable (must flag 85 % of queries — typical MAPE
  is 1-5 %, so "large" starts at ~5 %).
- At ≥5 % the data-driven choice is **5.30**, i.e. within 3 % of the
  chi-square default **5.13** — the default stands, and the sweep table
  (`guard_calibration.md`) is the documented knob for operators who want
  a different operating point.
- Flag rates at the defaults on the three folds: reject 0 % / 25 % / 29 %
  (bare / with_sail / full) — the guard is quiet on the in-space fold and
  noisy exactly on the out-family folds, which is the intended shape.

### Known blind spot (measured, not hand-waved)

`hull family is not a condition_v3 channel`. In the `loho::bare_hull`
fold all 14 held-out test rows sit **inside** the manual envelope
(scores ≤ 1.89 vs reject 5.13; `in-envelope 100 %`) while the fold errors
run at MAPE 2.13 % / max 2.4 % — the guard cannot see a held-out hull
family by construction, because bare rows carry nominal sail/fin scales
that live in the fitted envelope. The G2b corpus (v4) shrank that fold's
error (from the v3 uniform +2-3 % bias to 2.13 %, short of the ≤2.00 %
goal) but cannot make the manual guard *detect* it; the SDF-latent guard
(distance in mask space, where bare ≠ with_sail regardless of parameter
aliases) is the designed fix and plugs into this interface unchanged.

Secondary blind spot of the same kind: voxel quantisation makes distinct
`sail_scale` values produce bit-identical masks (the `mask_bit_eq` rows
of G2b) — parameter space and geometry space are not injective w.r.t.
each other, so a manual-parameter guard both over- and under-distinguishes
near the quantisation ladder.

## 4. ONNX export

`export_cond_fno_onnx(model, path, ny=64, nx=128, opset=17)` with honest
reporting (result: `/nfs/wangxi/runs/b4_serve_20260824/onnx/`):

- **Plain module: blocked.** `UnsupportedOperatorError: Exporting the
  operator 'aten::fft_rfft2' to ONNX opset version 17 is not supported`
  (torch 2.11 legacy exporter; complex einsum would block opset ≥ 18
  DFT-composition too). Recorded in the report, not retried silently.
- **Matmul twin: exported and validated.** `to_matmul_spectral` swaps
  each `SpectralConv2d` for `SpectralConv2dMatmul` — the same operator
  (ortho-normalised rfft2 corner → complex weight multiply → irfft2) as
  real dense matmuls against precomputed DFT cosine/sine bases, with the
  Hermitian mirror weights folded in (μ=2 for interior `0<kx<nx/2`
  columns, μ=1 for the self-conjugate `kx=0` / `kx=nx/2` columns — the
  latter pinned empirically against `torch.fft.irfft2`; a first guess of
  μ=2 everywhere was wrong by 12-22 % and is kept as a test fixture
  lesson). Parity: torch-vs-twin max-abs 0.0 on the export example,
  **onnxruntime vs torch max-abs 2.98e-08** (opset 17, checker ok,
  dynamic batch axis). Test-suite pins float64 rel < 1e-12 across three
  (ny, nx, modes) shapes plus a full-model parity < 1e-4.
- onnx/onnxruntime are **not** repo dependencies; for this smoke they
  were installed privately:
  `uv pip install --target /nfs/wangxi/runs/b4_serve_20260824/pydeps onnx onnxruntime`
  and used via `PYTHONPATH`. Without them the report degrades to
  `"checker": "skipped (onnx package not installed)"`.

## 5. Latency (single design, 64-point Re grid, 5-member ensemble)

`benchmarks/b4_serve_benchmark.py`, real serving checkpoints, warmup 3 +
reps 20 (GPU) / 5 (CPU), p50 wall time per `predict` call
(`/nfs/wangxi/runs/b4_serve_20260824/serve_benchmark.json`):

| device | with guard | without guard | guard only | ensemble forward |
|---|---|---|---|---|
| RTX 5090 (torch 2.11+cu128) | **17.7 ms** | 17.4 ms | 0.25 ms | 17.1 ms |
| CPU (96 threads) | 1297 ms | 1119 ms | 0.24 ms | 874 ms |

- The guard costs ~0.25 ms (envelope + Mahalanobis over 64 rows) —
  negligible on GPU; the CPU with/without delta (178 ms) is
  multi-threaded jitter, not guard cost (see guard-only column).
- One `predict` = one batched forward per member (64 Re points in a
  single batch) + field/condition resolution; ~3.4 ms per member on GPU.
- Without checkpoints the benchmark synthesises a random-weight
  production-arch ensemble, labelled `checkpoints: false` in its JSON —
  latency-valid, quality-invalid by construction.

## 6. Anomalies / environment notes

- `tests/test_data_catalog_api.py::test_router_registered_in_main_app`
  fails in the `tensorlbm` venv **on pristine origin/main (ca9e0c4d)**
  (verified on a throwaway worktree): newer starlette exposes included
  routers as `_IncludedRouter` without `.path`. Pre-existing, unrelated
  to this change; my registration test asserts via `_router_registry`
  and passes in both venvs.
- `fastapi` is absent from the `ci-cpu` venv (not a repo dependency):
  `tests/test_drag_surrogate_api.py` import-skips there (by design, same
  pattern as the data-catalog tests), and `test_data_catalog_api.py`
  fails at collection in that venv — pre-existing.
- The torch legacy exporter prints the full Torch IR graph when it
  raises (the plain-export blocker output is ~50 KB); the one-line
  blocker string is what the report keeps.
- `train_ensemble.py` (run-dir glue, not in the repo) copies the v4
  protocol verbatim because `train_fno_v4.py` asserts its own worktree
  and cannot be imported from `b4_serve`; the split reproduction was
  verified numerically (fit186/val33/test55) before training.

## 7. Reproduction

```bash
cd /nfs/wangxi/worktrees/b4_serve
export PYTHONPATH=src TMPDIR=/nfs/wangxi/tmp
PY=/nfs/wangxi/venvs/tensorlbm/bin/python

# tests (both venvs; ci-cpu additionally needs OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=)
$PY -m pytest tests/test_inference_service.py tests/test_drag_surrogate_api.py \
     tests/test_drag_cond.py tests/test_serving.py -q --basetemp=/nfs/wangxi/tmp/pt_serve

# serving ensemble (GPU 2, ~2.5 min) — wrote ckpts/serve_cfull_s{0..4}.pt
CUDA_VISIBLE_DEVICES=2 $PY /nfs/wangxi/runs/b4_serve_20260824/train_ensemble.py

# guard/UQ calibration + latency benchmark
$PY benchmarks/b4_guard_calibration.py            # -> guard_calibration.{json,md}
CUDA_VISIBLE_DEVICES=2 $PY benchmarks/b4_serve_benchmark.py \
    --out /nfs/wangxi/runs/b4_serve_20260824/serve_benchmark.json
```

Every script prints `tensorlbm.__file__` first; runs above resolved to
`/nfs/wangxi/worktrees/b4_serve/src/tensorlbm/__init__.py`.
