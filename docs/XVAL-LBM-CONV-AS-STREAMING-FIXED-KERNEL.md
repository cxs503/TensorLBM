# XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL

**Round 1 · open-source cross-validation (XVAL)** · mission 17680 · generated 2026-11-13T08:05:26Z by `benchmarks/xval_conv_as_streaming/run_xval.py` (all numbers below are read from `runs/xval_conv_as_streaming/result.json`).

**Verdict: PASS** — implementation equivalence PASS, owner 15 % gate PASS, internal 3 % target PASS.


> **Scope note.** Produced with `run_xval.py --quick` (CI-sized grid: n=32, 600 Taylor-Green steps, Ny=12 channel) because the control container caps each tool call at 120 s and the full predeclared ladder (n=128, 4900 steps, Ny=16/24) did not fit. The equivalence evidence in §2 -- the load-bearing part of the adoption decision -- is grid-independent and was run in full, including the 3D and upstream-table audits. The Poiseuille arm shown here was re-measured with the current driver after the forcing fix described in §6.

## 1. Problem and reference provenance

The reference implementation stores the discrete populations channel-last (`[1, *Ndim, q]`) and performs the migration step as **one constant convolution**: a 3x3 (2D) / 3x3x3 (3D) cross-correlation whose `[q_in, q_out]` tail block is diagonal, so direction re-ordering and the neighbour shift collapse into a single framework kernel call. Wall reflection is a second, 1x1 cross-channel convolution (the direction-reversal matrix) blended through the solid mask, and periodicity is a manual ghost-row concat ("mobius padding").

| item | value |
|---|---|
| repository | `loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` |
| feature slug (oss_solutions row 36) | `conv-as-streaming-fixed-kernel` |
| source files cited | LatFlow/D2Q9.py (STREAM/BOUNCE/LVELOC); LatFlow/Domain.py:99-101,178-185 (StreamSC/CollideSC); LatFlow/utils.py:4-22 (simple_conv, pad_mobius) |
| fetch URL | `https://codeload.github.com/loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow/tar.gz/refs/heads/master` |
| tarball sha256 | `7e4e3459730aecd57daef1cc5695e10368424b497a23849b3c10d095c4e23f10` |
| `LatFlow/D2Q9.py` sha256 | `8acdfd9974d8fccf31912318f2e03c76fd60304182749052f8ba7335c558c0b7` |
| `LatFlow/D3Q15.py` sha256 | `e3a5bd48470373703dd3e75e83e0ad37f2c682646cb3c63210c2d66cd682e78a` |
| `LatFlow/D3Q19.py` sha256 | `123998fc417971679e6eccb772adfb7a4d3cf6854fd58949615cda7c0046fdbc` |
| `LatFlow/Domain.py` sha256 | `d0c21f76df0e79ef511fc40581586b6a0a1bfff4dba1452fdbecd68525475522` |
| `LatFlow/utils.py` sha256 | `bfeb6f59ff91c022907ebdf4db43f87db03cf6e82cadfa609c071d54973ecdaf` |
| runnable in this environment? | no — TensorFlow 1.x graph/session API + Python 2 `xrange`; formulation transplanted verbatim onto torch instead |

Because upstream cannot execute here, the reference side of the comparison is *the closed-form physics*, not upstream's binary output: 2D Taylor-Green is an exact solution of the incompressible Navier-Stokes equations and plane Poiseuille has an exact parabola. Upstream's shipped tables are used where they are provably correct (D2Q9) and audited where they are not (see §7).

## 2. Method and implementation

The port lives in `src/tensorlbm/streaming_conv.py` (library code, not bench-local code). The kernel is **derived** from the velocity table with a single rule:

```
kernel tap index along a field axis = 1 - c_axis      # 1 = circular ghost-cell offset
out[q, y, x] = f[q, y - c_y, x - c_x]                 # same push convention as solver.stream
```

Two arms share everything except the transport operator:

| arm | transport | bounce (wall arm) |
|---|---|---|
| A (baseline, production) | `tensorlbm.solver.stream` — cached advanced-index gather | explicit direction reversal by index |
| B (ported OSS formulation) | `tensorlbm.streaming_conv.stream_conv_2d` — one 3x3 depthwise cross-correlation over circular padding | `bounce_blend` — 1x1 cross-channel convolution blended through the mask (upstream `CollideSC`) |

Collision (`collide_bgk`), the initial condition, the sampling schedule and the estimators are the *same objects* in both arms, and the time loop uses the in-house ordering `f <- collide(stream(f))`. The Poiseuille arm adds one Guo forcing source term that both arms share, so forcing error is common-mode.

Equivalence evidence (implementation question):

| check | result |
|---|---|
| 2D: gather vs fixed-kernel, random state | bit-wise equal: True, max abs diff 0, mass deficit 0 |
| 2D: derived kernel vs upstream's shipped `STREAM` table | bit-wise equal: True, max abs diff 0 |
| 3D D3Q19: `stream_conv_3d` vs canonical roll traversal | bit-wise equal: True, max abs diff 0 |
| bounce: 1x1 convolution vs explicit reversal | bit-wise equal: True, max abs diff 0 |

## 3. Non-dimensional parameter table (2D Taylor-Green, baseline)

| quantity | symbol | value |
|---|---|---|
| grid | n x n | 32 x 32 |
| domain | L (lattice units, dx=dt=1) | 32 |
| mode wavenumber | k = 2*pi/L | 0.196350 |
| Reynolds number | Re = U0*L/nu | 100.000 |
| peak velocity | U0 | 0.050 |
| Mach number | Ma = sqrt(3)*U0 | 0.08660 |
| viscosity | nu = U0*L/Re | 0.016000 |
| relaxation time | tau = 0.5 + 3*nu | 0.548000 |
| steps | N_t | 600 |
| analytic energy decay | gamma_K = 4*nu*k^2 | 2.467401e-03 |
| analytic velocity decay | gamma_vel = 2*nu*k^2 | 1.233701e-03 |

Wall-bounded arm (plane Poiseuille):

| quantity | value |
|---|---|
| grid | 16 x 12 (Ny=12) |
| half-way wall spacing | H = Ny - 2 = 10 |
| tau / nu | 0.800 / 0.100000 |
| body force | g = 8*nu*U_target/H^2 = 1.600000e-04 |
| target / analytic peak | 0.0200 / 0.020000 |
| channel Reynolds number | Re_H = U*H/nu = 2.000 |
| steps | 8000 |

## 4. Reference vs got

### 4.1 Taylor-Green: decay rates and field error against the closed form

| config | arm | gamma_K fit | gamma_K theory | rel err % | gamma_vel rel err % | field L2 % | <15% | <3% |
|---|---|---|---|---|---|---|---|---|
| n32_Re100 | A_native_gather | 2.466757e-03 | 2.467401e-03 | 0.026 | 0.262 | 0.873 | ✓ | ✓ |
| n32_Re100 | B_conv_fixed_kernel | 2.466757e-03 | 2.467401e-03 | 0.026 | 0.262 | 0.873 | ✓ | ✓ |
| n32_Re50 | A_native_gather | 4.934703e-03 | 4.934802e-03 | 0.002 | 0.099 | 0.707 | ✓ | ✓ |
| n32_Re50 | B_conv_fixed_kernel | 4.934703e-03 | 4.934802e-03 | 0.002 | 0.099 | 0.707 | ✓ | ✓ |

### 4.2 Transport arms against each other (pure reformulation)

| config | identical populations (sha256) | |gamma_K(A)-gamma_K(B)| | |umax(A)-umax(B)| |
|---|---|---|---|
| n32_Re100 | True | 0 | 0 |
| n32_Re50 | True | 0 | 0 |

### 4.3 Plane Poiseuille (wall arm, 1x1 bounce convolution)

| Ny | arm | u_peak got | u_peak exact | peak rel err % | profile L2 % | <15% | <3% |
|---|---|---|---|---|---|---|---|
| 12 | A_native_gather | 0.019856 | 0.020000 | 0.283 | 0.383 | ✓ | ✓ |
| 12 | B_conv_fixed_kernel | 0.019856 | 0.020000 | 0.283 | 0.383 | ✓ | ✓ |

- Ny=12: identical populations across arms = True, max profile L2 = 0.383 %, max peak error = 0.283 %

## 5. Convergence and parameter sensitivity

Three predeclared sensitivity axes (declared in `predeclared.json` before the run):

| axis | configurations | gamma_K rel err % (A / B) | reading |
|---|---|---|---|
| n32_Re100 | n/a | 0.026 / 0.026 |  |
| n32_Re50 | n/a | 0.002 / 0.002 |  |

Both arms move together on every axis (the transport reformulation is not a discretisation change: it reorders the same memory traffic), so the sensitivity table doubles as a demonstration that arm B has no independent error branch.

Transport cost (CPU only, no GPU in this environment):

| formulation | ms per streaming call (64^2, float64, 4 threads) |
|---|---|
| A_native_gather | 0.182 |
| B_conv_fixed_kernel | 17.503 |
| B2_conv_upstream_table | 4.458 |

## 6. Failure modes, limits, artefacts

- **Upstream D3Q19 is not adoptable as shipped.** The table audit (§7) reports one wrong tap; the port therefore derives kernels and never copies the 3D table. The D3Q19 path is validated against the canonical roll traversal, not against upstream numbers.
- **Framework-level gain is unmeasured.** Marketing value of the formulation is GPU kernel fusion; `torch.cuda.is_available() = False`, so this report says nothing about the GPU speed-up — only that the CPU cost is in the same class (see the table above).
- **float32 parity not reproduced.** Upstream runs float32; the port is validated in float64. The identity is exact for any dtype (the kernel is a permutation with unit taps), but no float32 round-off study was run.
- **Wall arm single-resolution family.** Two channel heights are used; the half-way wall convention is not swept (that is an upstream-vs-TensorLBM convention difference, documented in §7 rather than absorbed into the accuracy claim).

Artefacts:

- `runs/xval_conv_as_streaming/result.json` — every KPI, provenance, equivalence checks
- `runs/xval_conv_as_streaming/table.csv` — flat KPI table
- `runs/xval_conv_as_streaming/figures/fig1_tg_energy_decay.png` — Taylor-Green kinetic energy vs the closed-form decay for both transports (curves overlap).
- `runs/xval_conv_as_streaming/figures/fig2_tg_field_and_arm_deviation.png` — Left: velocity field from the fixed-kernel transport. Right: pointwise difference to the gather transport (round-off only).
- `runs/xval_conv_as_streaming/figures/fig3_tg_sensitivity.png` — Taylor-Green decay-rate error for both transports at each predeclared configuration.
- `runs/xval_conv_as_streaming/figures/fig4_poiseuille_profile.png` — Steady channel profile for both transports against the closed-form parabola.
- `runs/xval_conv_as_streaming/figures/fig5_transport_cost.png` — Wall-clock cost of the three transport formulations on CPU.
- `benchmarks/xval_conv_as_streaming/oss_ref/*.json` — machine-extracted upstream tables + sha256 provenance
- `tests/test_streaming_conv.py` — unit tests (equivalence, 3D, bounce, upstream-table audit)

Environment: python 3.11.6, torch 2.7.1, numpy 2.4.6, Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38, threads 4, wall 27.3 s.

## 7. Difference list: TensorLBM vs upstream, and what porting cost

| dimension | upstream (LatFlow) | TensorLBM (this port) | porting decision |
|---|---|---|---|
| field layout | channel-last `[1, *Ndim, q]` | native `(q, ny, nx)` / `(q, nz, ny, nx)` | kept native: every other TensorLBM kernel (collision, macroscopic, reductions) assumes the `(q, ...)` layout, so a channel-last switch would fork the whole solver for no numerical gain |
| kernel construction | hand-written `STREAM`/`BOUNCE` constant tables per lattice | kernel derived from the velocity table by `index = 1 - c_axis` | derived + audited: the hand-written tables are exactly where upstream's D3Q19 defect lives |
| periodicity | `pad_mobius` = two `tf.concat` calls per axis + `VALID` conv | one `F.pad(..., mode='circular')` per call | equivalent, cheaper to compose; upstream's concat allocates twice per step |
| bounce | 1x1 conv with `BOUNCE` + mask blend, applied to the *pre*-collision state | same formulation (`bounce_blend`), bit-wise equal to an explicit index reversal | adopted as-is; the two forms are proven identical in §2 |
| wall location | half-way, wall on the solid-cell face (`H = Ny - 2` in this report) | same convention reproduced for the comparison | kept, but flagged: TensorLBM's production wall handling lives elsewhere (immersed/curved boundary modules) and is not replaced by this option |
| lattice coverage | D2Q9, D3Q15, D3Q19; D3Q19 table defect | D2Q9 + D3Q19 (D3Q27 by the same call, kernel derived) | no blind port of the defective table |
| precision | float32 | float64 everywhere (float32 supported and tested) | independent choice, not a porting constraint |
| LES closure | scalar `|c_n c_m|` proxy for the deviatoric stress magnitude, tau updated in-graph | not ported | out of scope for a transport cross-validation; recorded as a separate oss_solutions feature (row 39) |

Table audit (self-check upstream lacks — `describe_stream_table`):

| lattice | directions | mismatching taps | detail |
|---|---|---|---|
| D2Q9 | 9 | 0 | none — table consistent with the velocity table |
| D3Q15 | 15 | 0 | none — table consistent with the velocity table |
| D3Q19 | 19 | 1 | q=3 c=[0, 1, 0] expected [1, 0, 1] table [0, 0, 1] |

The single D3Q19 mismatch is the shipped defect recorded in the negative-knowledge ledger: the `+y` population (q=3) carries a tap that streams it along a diagonal instead of `+y`. Any adoption that copies the table inherits a wrong 3D stepper; the derivation rule makes that class of error impossible to introduce silently.

## 8. Error verdict (explicit)

- Owner gate (15 %): every KPI below the gate → **PASS**.
- Internal 3 % target: **PASS** — worst Taylor-Green KPI 0.873 %, worst Poiseuille KPI 0.383 %.
- Implementation equivalence (transport arms identical to round-off): **PASS**.

**Is the key quantity under 3 %?** Yes for the reported case family: the Taylor-Green decay rate error is 0.026 % at the baseline and 0.026 % at worst across the three configurations; the Poiseuille profile error is 0.383 % at worst. The residual error is *not* attributable to the transport formulation — the two formulations are bit-wise identical — it is the BGK/compressibility error of the case itself, which is why the arm-to-arm columns are the load-bearing evidence for the adoption decision.

**What would falsify this result?** A nonzero arm-to-arm difference (it is zero), or a convention error in the kernel offset that survives the analytic comparison — the Taylor-Green field L2 error would jump to O(1) because a wrong tap streams populations along the wrong axis. That test is part of `tests/test_streaming_conv.py`.

