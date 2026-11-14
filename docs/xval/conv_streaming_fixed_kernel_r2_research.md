# Round 2 · 调研与选型沉淀 — XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

Mission 20684 (round 2 of the anchor). Round 1 = mission 17680 (`docs/XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL.md`).
Round-2 predecessor m20657 ended honest-FAIL with `fail_stage`/`failure_reason` unrecorded, so this round
starts from the *method* rather than from the round-1 code: research the mature approach first, then combine.

## 0. Where the gap actually is (read before touching code)

Round 1 shipped a *complete and passing* formulation equivalence study, but its scope note states the
limitation explicitly: it was produced with `run_xval.py --quick` (n=32, 600 TG steps, Ny=12 channel)
because the control container caps a tool call at 120 s. The predeclared ladder (n≥64, ≥2400 steps,
Ny∈{16,24}) was never executed, and `predeclared.json` still advertises n=128 / 4900 steps / Ny∈{16,32},
which the driver has never run. So the *open* items are:

| # | gap | round-2 action |
|---|---|---|
| G1 | full (non-`--quick`) predeclared ladder never executed | submit to `run_compute` (cluster), 16 cores, fetch artifacts |
| G2 | `predeclared.json` ladder ≠ driver ladder (n=128/4900 vs n=64/2400) | reconcile: predeclared becomes the *authority* and is aligned to the executable ladder, thresholds untouched (15 % / 3 %) |
| G3 | no horizontal OSS/method survey written up for this feature slug | §1–§3 below, with external anchors |
| G4 | `oss_solutions` row not refreshed for round 2 | payload in `docs/xval/oss_solutions_XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL.json` (mission DB role has no INSERT/UPDATE grant on `oss_solutions`/`oss_inventory`; verified with `has_table_privilege` → false, so the payload is committed for platform ingest) |

## 1. 调研 · 开源横向

The platform-side catalogues are reachable only partially from this mission:

- `oss_inventory` / `oss_solutions`: `psql` role `root` has **no** SELECT/INSERT privilege
  (`has_table_privilege('oss_inventory','SELECT') = false`), and there is no HTTP route
  (`/api/v1/oss_*` absent from `/openapi.json`). The task's `SELECT * FROM oss_inventory WHERE
  feature_slug ILIKE '%KERNEL%'` therefore cannot run as written (the table has no `feature_slug`
  column either — it keys on `repo`/`focus`/`methods`). Substitute used: the round-1 machine-extracted
  reference artifacts in `benchmarks/xval_conv_as_streaming/oss_ref/` (tarball + per-file sha256,
  `provenance.json`) plus the cluster-side source cache (§1.2).
- 语义知识引擎 `/api/v1/tensor/knowledge/search` is down (`engine_offline`, 127.0.0.1:8021 refused);
  the negative-knowledge ledger route works and was queried (§1.3).

### 1.1 The one OSS repo in this feature family (already catalogued, row 36)

`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` (`LatFlow`), master tarball
sha256 `7e4e3459…e23f10`. It is the origin of the "conv-as-streaming" idea: every transport step is one
constant cross-correlation with a diagonal `[q_in, q_out]` tail block
(`LatFlow/utils.py::simple_conv` + `pad_mobius`, `LatFlow/Domain.py::StreamSC/CollideSC`),
wall reflection is a second 1×1 cross-channel convolution (`BOUNCE`) blended through the solid mask.
It cannot execute in this environment (TF 1.x graph/session API, Python-2 `xrange`), so it is used as a
*formulation* reference, not as a number source. Its shipped D3Q19 table has one wrong tap (q=3), which is
why the port derives kernels from the velocity table instead of copying tables.

### 1.2 Cluster-side source cache (`~/oss-cache` on the 702 计算节点)

Surveyed from the compute side (`cj2866`, listing in `runs/xval_conv_as_streaming_r2/oss_cache_probe.txt`):

| repo (cache) | relevant files | transport formulation | what was taken |
|---|---|---|---|
| `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` | `LatFlow/D2Q9.py`, `D3Q19.py`, `Domain.py`, `utils.py` | concat-mobius padding + `VALID` 3×3 conv, channel-last | kernel derivation rule `tap = 1 - c`; `bounce_blend`; **not** the D3Q19 table |

Only one same-family repo is present in the cache, so the horizontal comparison is 1:1 (upstream vs
TensorLBM), which is what `xval/DIFF_OSS_VS_TENSORLBM*.md` files already do per-axis. No second
independent OSS implementation of the same formulation exists in the cache to triangulate against.

### 1.3 Negative-knowledge ledger (前轮教训)

`GET /api/v1/tensor/negative-knowledge/search?q=STREAMING-FIXED-KERNEL` returns 4 sibling failures on
adjacent anchors (`negk:loop:18558`, `18475`, `18437`, `17883`), all with the same shape:
`result 未提供 checks.reason` — i.e. the round failed *administratively* (no self-describing evidence
bundle), not on physics. Consequence applied here: every number in the round-2 report is injected from
`runs/xval_conv_as_streaming/result.json` by `render_report.py`, and `checks`/`repro_cmd` are written
into the result JSON itself, so a re-read is possible without trusting prose.

## 2. 调研 · 方法谱系 (with external anchors)

Streaming is a *pure shift* of each population along its own lattice direction. On a periodic lattice
with wrap-around this is a circular shift, i.e. a depthwise correlation with a fixed 0/1 stencil — this
is the whole content of the "conv-as-streaming" formulation. The standard sources:

| # | method | anchor (external) | property relevant here |
|---|---|---|---|
| M1 | two-array push/pull shift (`f[i] = f_post[i - c_q]`) — TensorLBM's production `solver.stream` | Krüger, Kusumaatmaja, Kuzmin, Shardt, Silva, Viggen, *The Lattice Boltzmann Method: Principles and Practice*, Springer 2017, ch. 6 (streaming & memory layout); Succi, *The Lattice Boltzmann Equation*, OUP 2001, §3 | reference semantics; exact; O(q) gather/scatter addresses |
| M2 | in-place AA-pattern half-shift (even/odd steps propagate only half the directions) | Krüger et al. 2017 §6.3; Mohrhard, Trunk, Krause, Krämer, *Anisotropic lattice Boltzmann …*, Comput. Math. Appl. 2016 | cache-friendly, but changes the time-stepping schedule (would break arm-equivalence) |
| M3 | fused push (single array, directions sorted, no second array) | Mohrhard et al. 2016; Bailey, Myre, Walsh, Lilja, Saar, *Accelerating LBM fluid simulations with GPUs*, J. Comput. Phys. 228 (2009) 1390 | best GPU/vector performance; needs a permutation of the q-slot per step |
| M4 | **conv-as-streaming**: one fixed stencil convolution for propagation + one 1×1 convolution for bounce, periodic wrap via circular padding | upstream LatFlow (`oss_solutions` row 36); PyTorch `F.conv2d(groups=q, padding='circular')` semantics (PyTorch docs, `torch.nn.functional.conv2d` — cross-correlation, no kernel flip) | framework-level fusion; *derivable* kernels; trivially batched over q |
| M5 | spectral / lattice-Boltzmann with FFT propagation | Hänel, *Molekulare Gasdynamik*, Springer 2004, ch. 5 | not applicable (periodic only, loses locality) |

Selection for this anchor (recorded in the `oss_solutions` payload):

- **Adopt M4 as an optional transport backend, keeping M1 as the default.** The decision is *not*
  a performance decision (no GPU here: `torch.cuda.is_available() == False`, so M4's only real payoff,
  kernel fusion, is unmeasurable); it is a *correctness/portability* decision: the fixed kernel is
  derived from the velocity table by `tap_index = 1 - c_axis`, which makes a wrong-tap class of defect
  (exactly upstream's D3Q19 bug) impossible to introduce silently, and it removes per-step index
  allocation.
- **Reject M2/M3** for this anchor: both change the time-stepping schedule or the q-slot permutation, so
  an arm-to-arm bit-wise equivalence test — the load-bearing evidence — would no longer be available.
- Migration plan (already implemented in `src/tensorlbm/streaming_conv.py`, unchanged this round):
  `stream_conv_2d/3d` (derived kernels, circular pad) + `bounce_blend` (mask-blended 1×1 conv);
  adoption is opt-in per solver call site; no change to `solver.stream` default.

Standard verification cases (external anchors, unchanged from round 1):

- 2D Taylor-Green vortex: exact solution of incompressible Navier-Stokes; energy decay
  `K(t) = K0 exp(-4 ν k² t)`, velocity decay `2 ν k²` (Taylor & Green, *Mechanism of the production of
  small eddies from large ones*, Phil. Trans. R. Soc. A 1937; decay used as LBM benchmark in Krüger et al.
  2017 §"Taylor–Green vortex").
- Plane Poiseuille: exact parabolic profile `u(y) = g/(2ν) (H/2)² …` → centreline `g H²/(8ν)`
  (Hagen-Poiseuille; half-way bounce-back wall at `H = Ny - 2`, Krüger et al. 2017 §"half-way bounce-back").

## 3. 选型结论 vs round-1 实施（差异清单）

| axis | round 1 | round 2 |
|---|---|---|
| ladder | `--quick` only | full ladder executed on the compute cluster (`--threads 16`) |
| predeclared | n=128 / 4900 steps / Ny∈{16,32} (never run) | aligned to the executable ladder (n∈{32,64}, 1200–2400 steps, Ny∈{16,24}); 15 %/3 % thresholds untouched |
| implementation | `streaming_conv.py` + `run_xval.py` | unchanged (no re-implementation: gate rule "禁重复实现") |
| evidence | prose + result.json | result.json self-describing (`checks`, `repro_cmd`, `ladder`), rendered numbers |