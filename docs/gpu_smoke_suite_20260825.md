# GPU smoke / verification suite (2026-08-25)

One command that runs the GPU-critical verification battery before a PR,
a fresh boot, a driver update or a torch upgrade, and emits a pass/fail
JSON report:

```bash
cd /nfs/wangxi/TensorLBM                      # or any checkout
TMPDIR=/nfs/wangxi/tmp PYTHONPATH=src \
  /nfs/wangxi/venvs/tensorlbm/bin/python benchmarks/gpu_smoke_suite.py \
  --gpu 4 --out /nfs/wangxi/tmp/gpu_smoke_report.json
echo $?   # 0 iff no check failed
```

CI-host (no CUDA) mode — proves the suite is usable on the CI runner:

```bash
OMP_NUM_THREADS=1 PYTHONPATH=src \
  /nfs/wangxi/venvs/ci-cpu/bin/python benchmarks/gpu_smoke_suite.py \
  --cpu-only --out /nfs/wangxi/tmp/gpu_smoke_report_cpu.json
```

`--quick` drops latency reps from 20 to 1 (suite finishes well under a
minute; latency numbers become single-sample, still gated).

Every check emits `{name, status: pass|fail|skip, measured, threshold,
seconds, detail}`; skips always carry a reason.  Checks that need /nfs
run artifacts (serving checkpoints, B4 corpus, ONNX artifact) degrade to
`skip` when the artifacts are absent, so the suite runs on a fresh
checkout.

## Checks, what they protect against, thresholds and provenance

| # | check | protects against | threshold | recorded reference |
|---|-------|------------------|-----------|--------------------|
| 1 | `env_sanity` | wrong venv / stale torch / dead GPU / writes landing on the small root partition | CUDA available in GPU mode; `/nfs` writable; effective TMPDIR under `/nfs` (when `/nfs` exists) | ops rule (root partition 70 G, 93% full at time of writing) |
| 2 | `parity_cpu_cuda` | wrong normalisation, wrong arch, fp16 downcast, silent kernel regression after torch/driver upgrade | max log10 C_D diff <= **1e-5** | see calibration note below |
| 3 | `determinism_gpu_repeat` | non-deterministic kernels (atomics, unexpected autotune flips) sneaking into the serving path | two GPU runs **bitwise equal** | bitwise on every recorded run |
| 4 | `serve_latency` | serving latency regression (2x-slower kernel selection after upgrade, guard blow-up, member creep) | p50 < **50 ms** | **17.67 ms** p50 (table: 17.7 ms) recorded in `docs/inference_service_20260824.md`, 64-Re guarded `predict`, 5 members, RTX 5090 — 3x headroom for noisy hosts |
| 5 | `echo_latency` | interactive CAD-slider loop losing the interaction budget | p50 < **150 ms** | **38.2 ms** recorded end-to-end slider echo incl. geometry front-end; 42.2 ms p50 cuda `slider_move` in `runs/b4_echo_20260825/bench_echo.jsonl` — 3.5x headroom |
| 6 | `onnx_parity` | ONNX export / ORT upgrade breaking the deployment artifact | max log10 C_D diff < **1e-5** over 5 members x 16 corpus rows | **4.65e-7** max over 274 real rows in `runs/b4_onnx_20260825/parity_report_stacked.json` |
| 7 | `voxelizer_cross_impl` | the two ray-parity voxelizers drifting apart (tie-break rule changes, layout flips) | **0** mismatched cells of 64³ | 0 mismatches, 33208 solid cells (recorded cross-implementation result; reproduced 2026-08-25) |
| 8 | `train_smoke` | broken autograd / NaN-poisoned kernels after a torch upgrade | last-5 mean loss < first-5 mean loss, all losses and grads finite, 30 steps, backward exercised | fresh tiny CondFNODrag (width 8, 2 layers) always satisfies this on a healthy stack |
| 9 | `suboff_mask_cpu_cuda` | CAD predicates diverging between devices (indexing math, `float` rounding) | CPU and GPU masks **bitwise equal**, shape (64, 64, 128) bool, stats consistent | 4157 solid cells identical on both devices (production grid; same count as the echo `mask_build` scenario) |

### Check 2 calibration note (deviation from the drafted 1e-6)

The verification record drafted this gate at 1e-6.  Calibration on the
reference host (RTX 5090, torch 2.11.0+cu128, member `serve_cfull_s0`,
corpus field row 0 + first 32 condition rows of
`/nfs/wangxi/runs/b4_v4_20260824`) measures:

- max log10 diff **2.15e-6**, median 2.0e-7, 6 of 32 rows above 1e-6;
- the noise is member-dependent (all 5 members via
  `ModelEnsembleBackend`: s0 2.2e-6, s1 1.4e-5, s2 2.5e-4, s3 3.6e-5,
  s4 1.3e-4) — same class, different draws;
- toggling `torch.backends.cudnn.allow_tf32` does not change the class
  (2.8e-6 off / 2.2e-6 on for s0), i.e. it is plain float32
  accumulation-order noise between CPU and CUDA kernels, not a precision
  downgrade.

A 1e-6 gate would fail on the reference configuration itself, so the
gate is set to **1e-5** for the s0 check as specified: that still
encodes "float32 kernel-noise class" (4.6x above the measured 2.15e-6)
while any real parity break (wrong norm stats, wrong arch, fp16
downcast, wrong member weights) produces diffs of order 1e-2 or worse in
log10 space.  Bitwise equality across devices is deliberately NOT
required — that contract is check 3.

### What check 5 measures (be explicit)

`GeometryEchoPipeline` lives on the `exp/b4-echo` branch, which is not
in this suite's base — the suite builds the equivalent minimal hot path
inline instead of depending on an unmerged branch: a NEW design every
call (cold geometry-feature computation, LRU never hits) -> condition
rows -> `EnvelopeMahalanobisGuardrail` -> fixed corpus-cache field row 0
-> 5-member ensemble forward, 64 Reynolds points.  The CAD mask/SDF
front-end of the full echo pipeline (recorded 16.3 ms GPU / 20.6 ms CPU
p50, `runs/b4_echo_20260825/bench_echo.jsonl` `mask_build`) is NOT in
the measured number — it is not part of the service call, and the echo
benchmark itself excludes it from the slider hot path when the corpus
field cache is attached.  Measured with real checkpoints: p50 42.8 ms,
i.e. the gate has 3.5x headroom over the reference measurement.

### Check 6 mechanics

`tensorlbm.ai.onnx_deploy` is not on this suite's base commit
(65cb33c), and the private ORT bundle (`pydeps`) carries its own numpy
2.x — so the ORT side runs in a subprocess with `PYTHONPATH=pydeps`
and never mixes interpreters: the parent writes the 16-row inputs to an
`.npz`, the child runs the stacked ONNX session on CPU and saves
`member_cd`, the parent compares against the torch
`ModelEnsembleBackend` (CPU, matching the recorded reference device) in
log10 space.  Skips cleanly when `pydeps`, the ONNX artifact, the corpus
or the trained checkpoints are absent.

### Check 7 mechanics

Both voxelizers sample cell centres at `origin + (i + 0.5) * spacing`
(origin = lower corner in both, despite `mask_from_stl`'s docstring
wording), so the recorded configuration — icosphere radius 5 subdiv 3
(1280 triangles), shape 64³, origin (-8, -8, -8), spacing 0.25 — is
directly comparable across the two implementations.  The icosphere
generator is duplicated inline from `tests/test_voxelize.py` /
`benchmarks/b4_voxelize_bench.py` so the suite has no test-tree
dependency.

## When to run it

- before every PR that touches `src/tensorlbm/ai/`, the voxelizers, the
  CAD predicates, or anything the serving path imports;
- after a driver update, a CUDA/torch upgrade, or a venv rebuild
  (checks 2/3/4/5 are exactly the "did the upgrade change the numbers"
  battery);
- after a fresh boot of the 5090 host (env sanity catches a dead GPU or
  a lost TMPDIR convention);
- on the CI host with `--cpu-only` to prove the suite itself still runs
  without CUDA (checks 2/3/4/5/9 and the ONNX check skip with reasons,
  env/voxelizer/training checks still execute).

## Known skip conditions

| condition | checks skipped |
|-----------|----------------|
| `--cpu-only` (or no CUDA) | 2, 3, 4, 5, 9 (all need the device) |
| serving checkpoints absent (`/nfs/wangxi/runs/b4_serve_20260824`) | 2, 6 (4/5 fall back to a labelled random-weight ensemble + synthetic corpus — latency-valid, never a quality claim) |
| corpus run dir absent (`/nfs/wangxi/runs/b4_v4_20260824`) | 2, 6 (4/5 synthetic fallback as above) |
| private ORT `pydeps` or ONNX artifact absent | 6 |
| `/nfs` not mounted (fresh checkout / CI host) | env `/nfs`-sub-checks; run-artifact checks above |
| `nvidia-smi` not on PATH | driver sub-check only (recorded as skip, not fail) |

Latency gates 4/5 are GPU budgets by design (recorded CPU p50s are
1297/1298 ms — a different budget), so they skip rather than fail in
`--cpu-only` mode.

## Exit code and report

- exit `0` iff zero checks failed (skips do not fail the run);
- `--out report.json` carries the full detail: per-check measured
  values, thresholds, seconds, skip reasons, per-member diagnostics and
  the artifact paths actually used;
- stdout ends with the human summary table (one line per check).
