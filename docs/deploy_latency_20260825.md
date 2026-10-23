# B4-P4a — Deployment-latency hardening of the drag surrogate (2026-08-25)

Ship record for `benchmarks/deploy_latency.py` + `src/tensorlbm/ai/trt_deploy.py`
+ `tests/test_deploy_latency.py`: the fused 5-member ensemble ONNX artifact of
PR #242 (`ensemble_cfull_stacked.onnx`, 451 nodes, 84.6 MB, opset 17, f64
de-norm tail) taken through every deployment runtime that exists on the 5090
server, with one benchmark, one parity metric and one JSONL log.

Everything below ran on the 5090 server (RTX 5090, driver 610.57.04, GPU 2
pinned via `CUDA_VISIBLE_DEVICES`) from worktree
`/nfs/wangxi/worktrees/deploy` (branch `exp/deploy-latency`, base `b5acea01` =
origin/main with #241/#242 merged).  Run artifacts live in
`/nfs/wangxi/runs/deploy_latency_20260825/` (log: `bench.jsonl`).

## 1. Ladder status — what RAN vs what is recipe-only

| level | status | evidence |
|---|---|---|
| 1. TensorRT FP32/FP16 engines | **RAN** | 3 strongly-typed engines built from the #242 artifact (9–24 s each), benched B=1..8, parity-checked |
| 2. ORT CUDA EP (FP32/FP16/INT8) | **RAN** | `onnxruntime-gpu` 1.29.0 CUDA EP in the new `deploy` venv; 4 rows |
| 3. torch-side hardening | **RAN (fp32 rows)** | baseline / cached-norms / TF32-off / `torch.compile` default+max-autotune / CUDA Graph; fp16 rows honestly BLOCKED (§6.3) |
| 4. unified benchmark + report | **RAN** | one script, both venvs, one JSONL; this doc + `report.md` |

Nothing in this slice is recipe-only: every level listed as RAN has numbers in
`bench.jsonl` produced by the shipped script.  The one blocked tier (torch fp16)
is a measured blocker, not an untested one (§6.3).

## 2. Environments

Two venvs, one GPU, no existing venv touched:

| venv | purpose | key packages |
|---|---|---|
| `/nfs/wangxi/venvs/tensorlbm` (existing, NOT modified) | phase A: torch tiers + torch-CPU parity reference | torch 2.11.0+cu128 |
| `/nfs/wangxi/venvs/deploy` (**new**) | phase B: TRT + ORT tiers | python 3.12.13, onnx 1.22.0, tensorrt-cu12 11.2.1.2, onnxruntime-gpu 1.29.0 (providers: CUDA, TRT, CPU), cuda-python 12.9.4, numpy, pytest |

The CUDA EP needs libcudnn at session create; the deploy venv borrows it via
`LD_LIBRARY_PATH` from the tensorlbm venv's pip `nvidia/*` wheels (borrow, not
copy, not install — see §7 for the exact path).  The ORT
`No registered plugin EP device found for 'CUDAExecutionProvider'` warning on
session create is benign (plugin-manager probe); the sessions do bind the CUDA
EP — the B=1 latency is 1.3 ms vs 38 ms CPU.

## 3. Measured task and discipline

Identical for every runtime: raw `field (1,5,64,128)` + `B` raw `condition_v3`
rows in, `(M=5, B)` member linear C_D out, host to host.  Query rows: the
largest B4 design family (one field, Re-swept rows — the #242 bench pattern),
seeded; identical inputs and the torch-CPU `ModelEnsembleBackend` reference are
carried in `reference_inputs.npz`, so phase B needs neither torch nor the
corpus.  Parity = per-member max abs in log10 C_D vs that reference (the
`verify_ensemble_onnx` metric of #242; the f64-artifact graph floor is 4.7e-7).

- warmup 30 calls (>= 20), then >= 100 timed calls or >= 5 s, whichever is
  larger, capped at 2000 (realised 100–2000 per row; every row in `bench.jsonl`
  carries `iters` + `total_s`);
- timer `time.perf_counter` around fully host-synchronous calls (torch tiers end
  in `.cpu().numpy()`, TRT in stream sync + D2H, ORT in `session.run`);
- p50 / p90 / mean / min / max + rows/s at p50; latency is WARM;
- cold start measured separately, one FRESH process per runtime (§5).

## 4. Warm latency (ms, p50 / p90), GPU 2

| runtime | B=1 p50/p90 | B=2 | B=4 | B=8 | B=8 rows/s | parity log10 (B=1 / 2 / 4 / 8) |
|---|---|---|---|---|---|---|
| TRT f16-body/f32-tail | 0.607 / 0.611 | 0.804 / 0.809 | 1.194 / 1.199 | 2.058 / 2.063 | 3888 | 6.4e-4 / 6.8e-4 / 7.7e-4 / 7.7e-4 |
| TRT fp32 (TF32 cleared) | 0.672 / 0.677 | 0.929 / 0.936 | 1.483 / 1.489 | 2.703 / 2.710 | 2960 | 1.9e-7 / 1.9e-7 / 1.2e-7 / 1.2e-7 |
| TRT fp32 (TF32 default) | 0.646 / 0.652 | 0.878 / 0.883 | 1.435 / 1.441 | 2.660 / 2.674 | 3007 | 7.9e-5 / 8.1e-5 / 9.6e-5 / 9.6e-5 |
| torch CUDA Graph fp32 | 0.853 / 0.855 | 1.110 / 1.111 | 1.406 / 1.408 | 2.069 / 2.070 | 3867 | 3.3e-6 / 1.3e-4 / 1.3e-4 / 1.3e-4 |
| torch.compile max-autotune | 1.060 / 1.064 | 1.332 / 1.338 | 1.731 / 1.734 | 2.531 / 2.535 | 3160 | 5.8e-8 / 1.3e-4 / 1.3e-4 / 1.3e-4 |
| ORT CUDA fp32 (`use_tf32=0`) | 1.335 / 1.345 | 1.383 / 1.394 | 1.875 / 1.889 | 3.207 / 3.219 | 2495 | 6.3e-8 / 6.0e-8 / 1.4e-7 / 1.4e-7 |
| ORT CUDA fp32 (default) | 1.416 / 1.546 | 1.434 / 1.565 | 1.725 / 1.744 | 2.955 / 2.971 | 2708 | 1.6e-6 / 7.0e-5 / 7.0e-5 / 7.0e-5 |
| ORT CUDA f16-body | 1.412 / 1.423 | 1.425 / 1.436 | 1.417 / 1.429 | 1.979 / 1.988 | 4042 | 1.7e-4 / 6.8e-4 / 6.8e-4 / 7.1e-4 |
| torch.compile default | 6.994 / 7.045 | 7.277 / 7.323 | 7.343 / 7.383 | 7.331 / 7.368 | 1091 | 3.3e-6 / 1.3e-4 / 1.3e-4 / 1.3e-4 |
| torch fp32 cached-norms | 5.938 / 5.988 | 6.027 / 6.074 | 6.057 / 6.105 | 6.077 / 6.125 | 1316 | 3.3e-6 / 1.3e-4 / 1.3e-4 / 1.3e-4 |
| torch fp32 TF32-off | 6.491 / 6.550 | 6.568 / 6.616 | 6.608 / 6.661 | 6.636 / 6.694 | 1206 | 5.8e-8 / 1.3e-4 / 1.3e-4 / 1.3e-4 |
| torch fp32 baseline (service) | 6.580 / 6.641 | 6.663 / 6.716 | 6.652 / 6.704 | 6.675 / 6.738 | 1198 | 3.3e-6 / 1.3e-4 / 1.3e-4 / 1.3e-4 |
| ORT CUDA int8-dynamic | 16.389 / 16.455 | 18.547 / 18.720 | 30.049 / 30.205 | 54.195 / 54.387 | 148 | 1.6e-3 / 1.6e-3 / 2.9e-3 / 3.2e-3 |
| ORT CPU f64 (#242 baseline) | 37.863 / 37.941 | 41.912 / 42.073 | 62.662 / 62.979 | 97.027 / 97.338 | 82 | 1.2e-7 / 5.8e-8 / 1.2e-7 / 1.2e-7 |
| torch fp16 autocast | BLOCKED | — | — | — | — | `ComplexHalf` §6.3 |
| torch CUDA Graph fp16 | BLOCKED | — | — | — | — | `ComplexHalf` §6.3 |

Headline, B=1 warm: **TRT fp16 0.607 ms / TRT fp32-strict 0.672 ms** vs torch
service baseline 6.58 ms and ORT CPU 37.9 ms — **9.8x vs torch warm, 62x vs
ORT CPU**.  Against the interactive budget (60 fps = 16.7 ms/frame), the TRT
B=1 query costs 3.6–4% of one frame; even the B=8 slider fan (2.06 ms) fits
30 fps with headroom.  This turns the #241 geometry-echo budget story from
"GPU 38.2 ms cold / 6.4 ms warm" into "sub-millisecond warm, ~0.5 s process
cold" (§5).

## 5. Cold start (fresh process per runtime, B=1)

Method: the driver loop (`cold_driver.sh` in the run dir) starts ONE fresh
python per runtime; inside each process `--cold <runtime>` times, in order:
lazy import of the tier deps, artifact load (session create / engine
deserialize / checkpoint load + model build — engine BUILD excluded, a
deployed service deserializes a prebuilt plan; build times are recorded
separately as `event: engine`), then the first B=1 call.  Driver + CUDA EP
runs with the `LD_LIBRARY_PATH` borrow of §2.

| runtime | import ms | load ms | first call ms | total ms | parity log10 |
|---|---|---|---|---|---|
| ORT CPU f64 | 24 | 98 | 42.5 | 164 | 1.2e-7 |
| TRT f16 (deserialize 44 MB plan) | 54 | 464 | 4.2 | 522 | 6.4e-4 |
| TRT fp32 (deserialize 86 MB plan) | 56 | 579 | 4.1 | 639 | 1.9e-7 |
| ORT CUDA fp32 (default) | 23 | 489 | 290 | 801 | 1.6e-6 |
| ORT CUDA fp32 strict | 24 | 607 | 284 | 915 | 6.3e-8 |
| torch fp32 baseline (5 ckpt + CUDA init) | 1370 | 788 | 206 | 2365 | 3.3e-6 |

Honest reading: the CHEAPEST cold start is ORT CPU (165 ms — smallest import +
smallest artifact), not TRT; TRT buys the best warm path and a sub-5 ms first
inference after deserialize, but loading an 86 MB plan costs ~0.5 s.  torch
cold is dominated by the 1.4 s framework import; its first call (206 ms)
includes CUDA context + cuDNN init.  A warm-pool service that deserializes
once and serves for hours is the TRT use case.

## 6. Honest deviations

1. **TF32 is a real, cheap-to-fix parity knob — in BOTH GPU engines.**  TRT
   11 defaults TF32 ON: 0.646 ms but 7.9–9.6e-5 log10 parity.  Clearing the
   flag (`build_engine(..., clear_tf32=True)`, the shipped default) restores
   1.2–1.9e-7 at +4% latency (0.672 ms).  ORT CUDA EP: same story, and the
   strict row (`use_tf32="0"`) is FASTER than the default row here (1.335 vs
   1.416 ms) — there is no reason to ship TF32-on for this graph.  Recommended
   serving rows: `trt_fp32` (strict) or `ort_gpu_fp32_strict`.
2. **fp16 costs ~6–8e-4 log10 (~0.15–0.18% linear C_D).**  The region-split
   graph keeps the de-norm tail f32 (§ `trt_deploy.build_f16_body_model`);
   the deviation is the conv/spectral body in half.  It is stable across
   batches and members (per-member max in `bench.jsonl`), and buys 10–24% over
   TRT fp32-strict plus a 2x smaller plan (44 vs 86 MB).  Fine for slider
   echo; do not use it for the numbers of record.
3. **torch fp16 is NOT RUNNABLE on `CondFNODrag` as shipped — measured
   blocker, not a skip.**  Whole-model `.half()` fails (`ComplexHalf` einsum,
   #242 finding); `torch.autocast` fp16 ALSO fails — autocast casts the conv
   output to half, `rfft2` then yields complex32, and the spectral
   `einsum`→`baddbmm` has no ComplexHalf kernel; bf16 autocast fails earlier
   (FFT unsupported for BFloat16).  Both fp16 rows are kept in the registry so
   the benchmark records the blocker as `event: skip` rows with the raw
   exception.  Fixing it needs an f32 island around the FFT+einsum in
   `src/tensorlbm/ai/fno.py` — a model-code change, out of scope for this
   slice (new files only).
4. **The torch GPU fp32 path itself deviates 1.3e-4 from the torch-CPU
   reference at B>=2 — even with TF32 off.**  `torch_fp32_strict`
   (`cudnn.allow_tf32=False`, `matmul.allow_tf32=False`) fixes B=1
   (3.3e-6 → 5.8e-8, proving the B=1 floor was TF32 convs) but B=2/4/8 stay at
   1.27e-4: cuDNN picks batch-dependent conv algorithms whose accumulation
   order differs from CPU.  TRT fp32-strict and ORT strict do NOT show this
   (1e-7 at all B).  So the CURRENT production GPU path (torch eager) is
   already a ~1.3e-4-log10 approximation of its own CPU reference; if
   reference-class parity matters, serve from TRT/ORT strict rows, not torch
   eager GPU.
5. **INT8 dynamic quantisation is an honest LOSS, not a win.**  16–54 ms
   (12–40x SLOWER than ORT fp32 CUDA) and parity 1.6e-3–3.2e-3 log10: the
   Conv/MatMul-INT8 kernels for these shapes are slower than the fp16/fp32
   paths on sm_120, and the quantisation error is visible.  Recorded, not
   recommended.
6. **Engine portability.**  TRT plans are built for THIS stack — TRT
   11.2.1.2 (cu12), RTX 5090 (sm_120), profile `cond: 1..8` (opt 8).  Plans
   are not portable across GPU architectures or TRT majors; the portable form
   is the ONNX artifact (#242), and `trt_deploy.build_engine` rebuilds a plan
   anywhere the deploy venv exists.  Build needs a 16 GB workspace (4 GB
   fails with `Could not find any implementation for node ... /Sqrt` and a
   4.4 GB tactic hint); builds took 9.4 s (fp32), 8.6 s (tf32), 24.4 s (f16).
7. **f64→f32 tail retyping cost.**  `trt_deploy.f64_tail_to_f32` (prerequisite
   for TRT, which has no f64) costs ~2.1e-7 log10 vs the f64 artifact — same
   class as the #242 graph floor (4.7e-7); all TRT/ORT-GPU f32 parities above
   include it.
8. **ORT CUDA first-call ~290 ms** (autotuned kernel selection + memory
   pool warmup) vs TRT 4 ms after deserialize — if B=1 cold-query latency
   matters more than load time, TRT wins; if load time matters more, ORT CPU.

## 7. Ops footprint (what was installed where)

- NEW venv only: `/nfs/wangxi/venvs/deploy` (python 3.12.13; pip installs:
  numpy, onnx, tensorrt-cu12 11.2.1.2, onnxruntime-gpu 1.29.0, cuda-python,
  pytest).  No existing venv was pip-installed into.
- CUDA EP libcudnn BORROW (no copy, no install):
  `LD_LIBRARY_PATH=/nfs/wangxi/venvs/tensorlbm/lib/python3.12/site-packages/nvidia/{cudnn,cublas,cuda_runtime}/lib`.
- All artifacts, logs and pytest temps on /nfs:
  `/nfs/wangxi/runs/deploy_latency_20260825/` (f32/f16/int8 ONNX variants, 3
  plans, `bench.jsonl`, `phaseA.log`, `phaseB.log`, `cold.log`,
  `cold_driver.sh`, `report.md`, `summarize.py`), pytest `--basetemp=/nfs/wangxi/tmp/pt_deploy`
  + `TMPDIR=/nfs/wangxi/tmp`.

## 8. What shipped (all NEW files, zero intersection with open PRs)

| file | role |
|---|---|
| `benchmarks/deploy_latency.py` | the unified benchmark: `--runtime` (groups `trt` / `ort_cpu` / `ort_gpu` / `torch_fp32` / `torch_fp16` / `compile` / `cudagraph` / `all`), `--cold`, `--gpu N`, `--out` JSONL, `--work-dir`; unavailable runtimes skip WITH a recorded reason |
| `src/tensorlbm/ai/trt_deploy.py` | TensorRT tier, torch-free: f64→f32 tail retyping, region-split f16-body graph, strongly-typed engine build (TF32 clearable), `TrtEnsembleBackend` (dynamic batch, host-sync `predict`/`predict_stats`) |
| `tests/test_deploy_latency.py` | 3 tiers: pure-python (import/plan/schema/parity math), ONNX-surgery (synthetic f64-tail graph, structural + numeric), TRT (real engine build + backend parity vs ORT CPU; CUDA-guarded) |
| `docs/deploy_latency_20260825.md` | this record |

Filename check against open PRs (#243 active_learning, #244 gpu_smoke_suite,
#245 demos, #246 cad_stl, #248 diff_voxelize): zero intersection — #244 adds
`benchmarks/gpu_smoke_suite.py` + `docs/gpu_smoke_suite_20260825.md`, this
slice adds `benchmarks/deploy_latency.py` + `docs/deploy_latency_20260825.md`.

## 9. Reproduce

```bash
cd /nfs/wangxi/worktrees/deploy

# phase A — torch tiers (GPU venv; writes/regenerates the reference npz)
CUDA_VISIBLE_DEVICES=2 TMPDIR=/nfs/wangxi/tmp PYTHONPATH=src \
    /nfs/wangxi/venvs/tensorlbm/bin/python benchmarks/deploy_latency.py \
    --runtime torch_fp32,torch_fp16,compile,cudagraph \
    --out /nfs/wangxi/runs/deploy_latency_20260825/bench.jsonl \
    --work-dir /nfs/wangxi/runs/deploy_latency_20260825

# phase B — trt + ort tiers (deploy venv; cudnn borrow for the CUDA EP)
NV=/nfs/wangxi/venvs/tensorlbm/lib/python3.12/site-packages/nvidia
CUDA_VISIBLE_DEVICES=2 TMPDIR=/nfs/wangxi/tmp PYTHONPATH=src \
    LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:$NV/cuda_runtime/lib \
    /nfs/wangxi/venvs/deploy/bin/python benchmarks/deploy_latency.py \
    --runtime trt,ort_cpu,ort_gpu --gpu 2 \
    --out /nfs/wangxi/runs/deploy_latency_20260825/bench.jsonl \
    --work-dir /nfs/wangxi/runs/deploy_latency_20260825

# cold start — one fresh process per runtime
/nfs/wangxi/runs/deploy_latency_20260825/cold_driver.sh

# tests (both venvs; TRT tier runs only where tensorrt + a GPU exist)
PYTHONPATH=src /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest \
    tests/test_deploy_latency.py -q --basetemp=/nfs/wangxi/tmp/pt_deploy
PYTHONPATH=src /nfs/wangxi/venvs/deploy/bin/python -m pytest \
    tests/test_deploy_latency.py -q --basetemp=/nfs/wangxi/tmp/pt_deploy
```

Summary tables regenerate from the log:
`python /nfs/wangxi/runs/deploy_latency_20260825/summarize.py /nfs/wangxi/runs/deploy_latency_20260825/bench.jsonl`.
