# B4-P3c — Fused-ensemble ONNX deployment of the drag surrogate (2026-08-25)

Ship record for `src/tensorlbm/ai/onnx_deploy.py`: the 5-member serving
ensemble (`/nfs/wangxi/runs/b4_serve_20260824/ckpts/serve_cfull_s{0..4}.pt`,
CondFNODrag `in_ch=5, width=32, n_layers=4, modes=[16,32], mlp_hidden=128,
film_hidden=64, cond_dim=8, aux_dim=8`, verified from the checkpoint arch
dicts) exported as ONE torch-free ONNX artifact with normalisation folded
in and ensemble UQ computed inside the graph.

Everything below was measured on the 5090 server from worktree
`/nfs/wangxi/worktrees/b4_onnx` (base `65cb33c`); artifacts live in
`/nfs/wangxi/runs/b4_onnx_20260825/`.

## 1. Artifact contract (schema v1)

| port | dtype | shape | meaning |
|---|---|---|---|
| `field` (in) | float32 | `(1, 5, 64, 128)` | raw mid-plane channels `[ux/u, uy/u, uz/u, rho, solid_mask]` of ONE geometry, NOT z-scored; the leading 1 is the geometry axis (all rows share this field) |
| `cond` (in) | float32 | `(N, 8)`, `N >= 1` dynamic | raw `condition_v3` rows `[log10_re, log10_u_in, log10_sail_scale, log10_fin_scale, log_aproj_ratio, sail_frac, fin_frac, solid_frac]`, NOT z-scored |
| `member_cd` (out) | float64 | `(M=5, N)` | per-member linear drag coefficient C_D (de-normalised `10**(z*y_std + y_mean)` in-graph) |
| `cd_mean` (out) | float64 | `(N,)` | ensemble mean, linear C_D |
| `cd_std` (out) | float64 | `(N,)` | ensemble sample std, ddof=1 (zeros when M=1) |
| `cd_min` / `cd_max` (out) | float64 | `(N,)` | member min-max band, linear C_D |

The space is exactly `ModelEnsembleBackend.predict` + `ensemble_stats`:
guardrails (envelope/Mahalanobis) compose on the host over the raw
condition rows, unchanged.  `log10 C_D` is `np.log10` of any output.
Per-member z-score statistics (channel/condition/label) are folded INTO
the graph — the artifact consumes raw vectors and returns raw C_D; this
is pinned by `tests/test_onnx_deploy.py::test_norm_folding_contract`
(raw in -> raw out vs the torch backend; pre-normalised inputs are shown
to give different outputs).

`OnnxEnsembleBackend` mirrors the torch backend interface
(`predict(fields, cond) -> (M, N)` linear C_D; `predict_batch(fields
(G,5,ny,nx), cond) -> (G, M, N)`, one session call per geometry, G=1
exactly equals `predict`); `predict_stats` returns the graph's in-graph
statistics.  Member labels and the contract are embedded as ONNX model
metadata (`tensorlbm.member_labels`, `tensorlbm.ensemble_contract`) and
a versioned JSON manifest (`write_manifest` / `load_manifest`,
`tensorlbm.onnx_deploy.ensemble_manifest` v1: artifact + member hashes,
IO contract, parity + latency evidence) ships next to the `.onnx`.

## 2. Design decision record — which fused export shipped and why

Two designs were implemented, exported and measured on the real ensemble:

- **stacked (SHIPPED, default)** — batch-of-members via weight stacking:
  grouped 1x1 `Conv2d(groups=M)` for lift/pointwise, batched matmuls
  (`(M, O, I)` stacked linears) for cond-embed/FiLM/MLP, and a stacked
  spectral layer (`StackedSpectralMatmul`) where the complex weight
  multiply is ONE batched matmul over `(M, my*mx)` frequency groups and
  the DFT bases broadcast over the batch.  One forward produces `(M, N)`.
- **unrolled (selectable)** — M sequential member blocks (each a full
  matmul-twin CondFNODrag with its own folded norm) whose outputs are
  stacked; the per-member path PR #239 already pinned, evaluated M times.

Measured (5 real members, real corpus conditions; details in sections 3-5):

| | stacked | unrolled |
|---|---|---|
| real-corpus parity, per-member max abs (linear C_D) | 1.087e-05 | 1.154e-05 |
| real-corpus parity, per-member max abs (log10 C_D) | 4.653e-07 | 4.653e-07 |
| graph nodes | 451 | 1585 |
| artifact size | 84.6 MB | 84.8 MB |
| export wall time (pinned host, shipped artifacts) | 1.36 s | 1.45 s |
| ORT CPU EP p50 @ B=64 (default threads) | **688 ms** | 3652 ms |
| ORT CPU EP p50 @ B=1 (default threads) | **41 ms** | 87 ms |

Parity is a tie (identical log10 floor 4.7e-07 — the float32 matmul-twin
floor of PR #239, amplified to ~1.1e-05 in linear C_D at the corpus C_D
range 2.5-19.8).  The stacked graph has 3.5x fewer nodes and is 5.3x
faster at B=64 / 2.1x at B=1 on the deployment runtime, at identical
artifact size and export cost, and its stacking correctness is pinned in
plain torch by `test_stacked_graph_matches_unrolled` (no onnx needed).
**stacked is therefore the default**; unrolled stays selectable
(`export_ensemble_onnx(..., design="unrolled")`) as the simple-parity
fallback.

### Failure notes (honest record of where each design fought back)

1. Stacked export initially FAILED with
   `SymbolicValueError: Unsupported: ONNX export of convolution for
   kernel of unknown shape` — the legacy exporter refuses a `Conv` whose
   input channel dim is a traced value.  Cause: my grouped-conv reshapes
   built their shape lists from `x.shape` after the dynamic-batch
   `expand`, so ALL dims became graph values.  Fix: every reshape dim
   except the traced batch axis is now a static int from module
   attributes (`_grouped_conv1x1(..., h, w)` with constants; the same
   discipline in `StackedSpectralMatmul._apply_weight`, dims from the
   stacked parameter shape).  After that the design exports clean through
   `onnx.checker` and runs on ORT.
2. Two torch-level stacking bugs were caught by unit comparison against
   the twin (before any ONNX step) and fixed: (i) a trailing GELU after
   the second cond-embed linear (the Sequential is Linear-GELU-Linear);
   (ii) missing complex cross terms in the batched spectral weight
   multiply (`y_r` needs `W_r x_r - W_i x_i`, `y_i` needs `W_r x_i +
   W_i x_r` — I initially dropped the cross terms, a ~0.19 log10 error).
   These are recorded because they are the kind of mistake weight
   stacking invites; the torch-only parity tests exist to pin them.
3. Unrolled exported clean on the first attempt (1585 nodes, checker ok).

## 3. Parity (real ensemble, both designs)

`verify_ensemble_onnx(..., n_random=64, real_fields=cache_v4.x,
real_cond=cache_v4 condition_v3 rows, max_real_rows=274)`: the random
block is 64 standard-normal rows against one standard-normal field
(distribution-free graph-equivalence check); the real block evaluates
each of the 274 corpus rows with its OWN field and its own condition row.
Reference: `ModelEnsembleBackend` on CPU (deterministic; GPU kernels
differ in reduction order and would pollute the signal).

Per-member max abs, real 274 rows (linear C_D, M0..M4):

- stacked: 1.087e-05, 9.050e-06, 4.882e-06, 5.063e-06, 5.308e-06
- unrolled: 1.154e-05, 7.653e-06, 1.071e-05, 5.585e-06, 5.663e-06

Per-member max abs log10 C_D, real rows: worst member is M0 at
**4.653e-07 in BOTH designs** (per-member range 1.1e-07 .. 4.7e-07;
stacked 4.653e-07/2.326e-07/1.163e-07/1.454e-07/1.163e-07, unrolled
4.653e-07/2.326e-07/2.326e-07/2.036e-07/1.745e-07).  Ensemble-statistic
parity, real rows (max abs, linear): stacked mean 3.4e-06 / std 3.3e-06
/ min 4.9e-06 / max 9.1e-06; unrolled mean 3.2e-06 / std 3.9e-06 /
min 8.9e-06 / max 1.1e-05.  Zero non-finite mismatches in any block
(torch and the graph overflow on exactly the same stress rows; the
parity helper excludes both-overflow positions from the max but counts
finiteness disagreements, of which there were none).  Full reports:
`parity_report_{stacked,unrolled}.json` in the run dir; spot parity is
also asserted inside the benchmark at every batch size (5.3e-06 max on
the bench inputs, both hosts, both designs).

Random 64-row block (standard-normal field + rows, one seed): linear
max-abs is meaningless to quote — the trained members map pure noise to
C_D up to ~1e298 (finite) — so parity there is read in log10 space:
per-member max abs log10 spans 2.1e-04 .. 1.8e-03 (stacked) and
1.7e-04 .. 1.8e-03 (unrolled), i.e. relative ~1e-4 at magnitudes where
float32 carries ~2e-07 relative resolution.  This is ordinary float32
behaviour of the trained weights on far-out-of-distribution inputs,
identical in character for both designs and for the torch reference; in
distribution (the 274 real rows) parity is the 4.7e-07 above.

## 4. Latency (torch vs ONNX Runtime)

Query pattern: one geometry field + B real condition rows (the largest
corpus design family, `("full", 1.0, 1.0, 0.1)`, 42 Re rows; B=64 cycles
them).  Warmup 3, reps 30 (torch CPU 5), p50/p95 in ms.  Full matrix in
`bench.jsonl`; two host configs:

**Host 1 — GPU host, default threading** (tensorlbm venv, torch
2.11.0+cu128, CUDA on GPU 4, torch CPU threads 96, onnxruntime 1.29.0
CPU EP):

| backend | B=1 p50 | B=8 p50 | B=64 p50 | B=64 p95 |
|---|---|---|---|---|
| torch CUDA (RTX 5090) | 6.6 | 6.7 | 17.2 | 17.6 |
| torch CPU (96 threads) | 94.5 | 128.3 | 864.5 | 877.1 |
| ORT CPU, stacked | **41.0** | **99.0** | **688.1** | 692.0 |
| ORT CPU, stacked, 1 intra-op thread | 52.1 | 407.3 | 4369.8 | 4381.0 |
| ORT CPU, unrolled | 87.2 | 515.7 | 3651.7 | 3676.5 |
| ORT CPU, unrolled, 1 thread | 83.1 | 643.1 | 5624.7 | 5640.8 |

**Host 2 — OMP-pinned CPU host** (ci-cpu venv, `OMP_NUM_THREADS=1`,
`CUDA_VISIBLE_DEVICES=` empty, torch 2.13.0+cpu, same artifacts reused):

| backend | B=1 p50 | B=8 p50 | B=64 p50 |
|---|---|---|---|
| torch CPU (1 thread) | 97.3 | 317.1 | 4039.9 |
| ORT CPU, stacked (default pool) | **39.9** | **98.4** | **688.7** |
| ORT CPU, stacked, 1 intra-op thread | 51.5 | 407.3 | 4293.3 |
| ORT CPU, unrolled (default pool) | 87.2 | 516.5 | 3849.1 |
| ORT CPU, unrolled, 1 thread | 84.4 | 637.4 | 5578.4 |

The pinned host reproduces the GPU-host ORT numbers for the default
pool (stacked 688.7 vs 688.1 ms at B=64 — the ORT thread pool is
independent of `OMP_NUM_THREADS`) and isolates the single-thread
envelope: stacked stays at 41-52 ms for interactive B=1 queries, and a
single-threaded torch ensemble (4040 ms at B=64) is 5.9x slower than
the ORT default-pool artifact.

Verdict: on CPU-only deployments the stacked ONNX artifact matches or
beats the torch CPU ensemble at every batch size (688 vs 865 ms at B=64
with torch allowed all 96 threads; 41 vs 95 ms at B=1) while carrying no
torch dependency; a CUDA host with torch stays fastest (17 ms at B=64),
and the private onnxruntime build has no CUDAExecutionProvider
(available providers: Azure, CPU — recorded in the export reports), so
ORT deployments are CPU-EP only today.

## 5. Artifact sizes and export cost

| artifact | bytes | nodes | export time | checker |
|---|---|---|---|---|
| `ensemble_cfull_stacked.onnx` | 84,637,605 | 451 | 1.36 s | ok |
| `ensemble_cfull_unrolled.onnx` | 84,819,191 | 1585 | 1.45 s | ok |
| per-member `.pt` (for scale) | ~16.9 MB x5 | — | — | — |

(Export times are from the pinned-CPU host run that produced the shipped
artifacts; an earlier unpinned GPU-host export measured 0.85 s / 1.35 s.)
The export report also carries a smoke parity on 4 random condition rows
measured immediately after export — `member_cd` there is in log10 space
(2.1e-04 / 1.8e-04); the statistic entries of that smoke check are
linear-space and therefore overflow-scale on random inputs (up to ~1e105)
— they are superseded by `verify_ensemble_onnx`, which is the parity
evidence that matters.

The aux head (aux_dim=8) is dead code under `return_aux=False` and is
dropped by tracing — the artifact carries exactly the served prediction
path.  Opset 17; nothing newer is needed (the matmul twin exists precisely
to avoid `aten::fft_rfft2`).

## 6. Embedding guide (DCC plugin, python ORT)

```python
import numpy as np
import onnxruntime as ort   # pip install onnxruntime — no torch needed

sess = ort.InferenceSession(
    "ensemble_cfull_stacked.onnx", providers=["CPUExecutionProvider"]
)
field = midplane.astype(np.float32)[None]   # (1, 5, 64, 128) raw channels
cond = cond_rows.astype(np.float32)         # (N, 8) raw condition_v3 rows
member_cd, cd_mean, cd_std, cd_min, cd_max = sess.run(
    None, {"field": field, "cond": cond}
)
# cd_mean: drag curve over the N Reynolds points (linear C_D)
# [cd_min, cd_max] / cd_std: deep-ensemble epistemic band / spread
```

Building the inputs: `cond` rows come from
`tensorlbm.ai.drag_cond.condition_v3` (or any tool replicating the 8
channels); `field` is the mid-plane `[ux/u, uy/u, uz/u, rho, solid_mask]`
stack of the reference simulation of the queried design (B4 cache
convention).  Guardrails (envelope + Mahalanobis over the same 8-channel
space, `EnvelopeMahalanobisGuardrail`) compose OUTSIDE the artifact over
the raw rows — port them as plain numpy if the plugin must flag
extrapolation.  The manifest JSON next to the artifact records members,
hashes, the contract and the parity/latency evidence at build time.

Server-side reproduction (private deps, not repo deps):

```bash
cd /nfs/wangxi/worktrees/b4_onnx
export PY=/nfs/wangxi/runs/b4_serve_20260824/pydeps
# tests (ONNX tier active):
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= \
PYTHONPATH=src:$PY /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest \
    tests/test_onnx_deploy.py -q --basetemp=/nfs/wangxi/tmp/pt_onnx
# bench (GPU host / pinned CPU host):
CUDA_VISIBLE_DEVICES=4 PYTHONPATH=src:$PY \
  /nfs/wangxi/venvs/tensorlbm/bin/python benchmarks/b4_onnx_bench.py \
  --out /nfs/wangxi/runs/b4_onnx_20260825/bench.jsonl
OMP_NUM_THREADS=1 CUDA_VISIBLE_DEVICES= PYTHONPATH=src:$PY \
  /nfs/wangxi/venvs/ci-cpu/bin/python benchmarks/b4_onnx_bench.py \
  --out /nfs/wangxi/runs/b4_onnx_20260825/bench.jsonl --no-export
```

## 7. Remaining gap to TensorRT (analysis only — no TRT work done)

The artifact is NOT TensorRT-ready today, for one hard reason and a few
soft ones:

- **float64 tail.** The contract outputs (`member_cd`, stats) and the
  de-normalisation `Pow(10, z*y_std + y_mean)` are float64 (chosen to
  match `ModelEnsembleBackend`'s float64 de-normalisation exactly).
  TensorRT supports float32/FP16/INT8 only — no double precision — so a
  TRT variant needs the tail (de-norm + member stats) in float32.  That
  is an export-time switch (fold in f32; expected cost ~1e-7 relative on
  the stats, negligible against the 4.7e-07 log10 graph parity), not a
  redesign.
- **Op inventory.** The graph uses Conv (incl. `groups=5`), MatMul, Gemm,
  Add/Mul/Sub/Div, Erf (GELU), ReduceMean/Min/Max, Sqrt, Pow, Cast,
  Expand, Reshape, Transpose, Concat, Slice — all supported by the TRT
  ONNX parser since TRT 8.6 (GELU-as-Erf) with native fast paths from
  10.x.  Opset 17 is within the parser range.
- **Dynamic batch.** `N` is a runtime dim via `Expand`; TRT needs an
  optimisation profile (e.g. N in {1, 8, 64, 256}) — the ONNX graph
  itself is already shape-dynamic, no re-export needed beyond the f32
  tail.
- **Provider availability.** The server private onnxruntime build ships
  only Azure/CPU execution providers, so a CUDA-EP comparison (let alone
  TRT) was not measurable here; on the CPU EP the stacked artifact
  already meets the interactive-query budget (41 ms at B=1).

Concrete TRT recipe when needed: add `float_stats: bool = True` export
option producing a TRT sibling artifact, then `trtexec --onnx=...trt.onnx
--fp16 --minShapes=cond:1x8 --optShapes=cond:64x8
--maxShapes=cond:256x8`, and verify with the same
`verify_ensemble_onnx` parity harness (reference stays the torch CPU
backend).

## 8. Files

- `src/tensorlbm/ai/onnx_deploy.py` — export (both designs), backend,
  verify, manifest; onnx/onnxruntime imports guarded.
- `tests/test_onnx_deploy.py` — torch-only tier (CI) + ONNX tier
  (skipif no onnx/onnxruntime; private pydeps command documented in the
  module docstring).
- `benchmarks/b4_onnx_bench.py` — latency/size/export benchmark, JSONL.
- Run dir `/nfs/wangxi/runs/b4_onnx_20260825/`:
  `ensemble_cfull_{stacked,unrolled}.onnx`,
  `export_report_{...}.json`, `parity_report_{...}.json`,
  `manifest_ensemble_cfull_stacked.json`, `bench.jsonl`.
