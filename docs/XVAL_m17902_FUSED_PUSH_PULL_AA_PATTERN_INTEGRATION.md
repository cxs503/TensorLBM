# XVAL m17902 — fused push-pull AA pattern: integration + machine referee

**Anchor**: `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN` (OSS `lbm-principles-practice/code`, chapter 13 GPU `gpu_stream_collide_save`).  
**Case**: 2d_taylor_green_decay_d2q9_bgk  |  **Mission**: m17902  |  **Rendered from**: `result.json`

## 1. Case, geometry and parameters

Upstream kernel (quoted behaviour of the snapshot recorded under mission 17839): pull the
eight shifted neighbours of the **post-collision** field, rebuild `rho/ux/uy` from those
nine numbers, relax, write the next buffer — no intermediate *streamed* field ever exists.

| parameter | value |
|---|---|
| domain | periodic square {n}×{n}, n ∈ [32, 48, 64] |
| ν, τ | 0.1666666667, 1.0000000000 |
| u_max, Ma | 0.02, 0.0346 |
| ρ0 | 1.0 |
| steps | ∝ n² (fixed physical time): n=32: 120, n=48: 270, n=64: 480 |
| dtype / boundary | float64 / periodic |
| reference | closed-form Taylor-Green (Krueger et al. 2016, ch.13 upstream taylor_green_eval); cell-centred X=x+1/2, Y=y+1/2, pressure term O(u^2)/3 |
| gates | owner 15%; self target 3% |

## 2. What was done

The anchor's first delivery (m17839, commit `eee0988a`) was audited at L2 for two
structural reasons: the fused module had **no caller in the library** and there was **no
machine referee row**.  This mission closes both, and in doing so corrects a claim in the
first write-up:

1. `tensorlbm.solver.step_stream_collide(f, tau, kernel=...)` is the new integration
   point — `kernel="ab"` is the existing `stream(collide_bgk(...))` path, 
   `kernel="fused_push_pull"` routes to `tensorlbm.fused_push_pull`.  Unknown kernels
   and illegal `τ` are rejected before any work happens.
2. `benchmarks/verified/fused_push_pull_aa_pattern/` runs both kernels on a Taylor-Green
   ladder plus two negative controls, and `verify.py` adjudicates every gate (exit 0).
3. **Claim correction**: the two kernels are *not* bit-identical in their raw states.
   The fused kernel collides after streaming, the AB reference before, so the raw state
   difference is one collision operator, O(Ma); the correct statement, pinned by test and
   by the referee, is `f_fused(n) == collide_bgk(f_ab(n))` to round-off.

## 3. Results

### 3.1 Discretisation error vs the closed-form solution

| grid | steps | kernel | u_x rel.L2 | u_y rel.L2 | ρ rel.L2 | mass drift | wall |
|---|---|---|---|---|---|---|---|
| 32 | 120 | `ab` | 0.4849% | 0.4849% | 1.2238e-01 | 6.66e-15 | 1.2 s |
| 32 | 120 | `fused_push_pull` | 0.4849% | 0.4849% | 1.2238e-01 | 6.77e-15 | 1.0 s |
| 48 | 270 | `ab` | 0.2101% | 0.2101% | 1.9081e-02 | 1.50e-14 | 2.5 s |
| 48 | 270 | `fused_push_pull` | 0.2101% | 0.2101% | 1.9081e-02 | 1.50e-14 | 4.2 s |
| 64 | 480 | `ab` | 0.1121% | 0.1121% | 4.0694e-02 | 2.65e-14 | 107.2 s |
| 64 | 480 | `fused_push_pull` | 0.1121% | 0.1121% | 4.0694e-02 | 2.65e-14 | 6.5 s |

### 3.2 Scheme equivalence (same grid, same steps)

| quantity | value |
|---|---|
| `max\|f_fused − collide_bgk(f_ab)\|` | 3.331e-16 |
| `max\|f_fused − f_ab\|` (one-collision label offset, **not** an error) | 1.399e-04 |
| cross-arm velocity rel.L2 | 3.066e-14 |
| fused vs analytic velocity rel.L2 | 1.121e-03 |
| AB vs analytic velocity rel.L2 | 1.121e-03 |

### 3.3 Convergence order (steps ∝ n²)

| kernel | observed orders |
|---|---|
| `ab` | 2.063, 2.182 |
| `fused_push_pull` | 2.063, 2.182 |

### 3.4 Viscosity inversion from the energy decay

| grid | kernel | ν inverted | rel. error |
|---|---|---|---|
| 32 | `ab` | 0.167190 | 0.3140% |
| 32 | `fused_push_pull` | 0.167190 | 0.3140% |
| 48 | `ab` | 0.166890 | 0.1339% |
| 48 | `fused_push_pull` | 0.166890 | 0.1339% |
| 64 | `ab` | 0.166785 | 0.0712% |
| 64 | `fused_push_pull` | 0.166785 | 0.0712% |

### 3.5 Negative controls (does the 15% gate bite?)

| control | u_x rel.L2 | exceeds 15% gate |
|---|---|---|
| `nu_bias_25pct` (relaxation from nu*1.25 (+25% viscosity)) | 32.3944% | yes |
| `mirror_we_initial` (+x/-x directions of the IC swapped) | 193.6462% | yes |

### 3.6 Figures

![fig1_error_vs_grid.png](../benchmarks/verified/fused_push_pull_aa_pattern/figures/fig1_error_vs_grid.png)

*Discretisation error of both kernels against the closed-form Taylor-Green solution, grids with steps ∝ n² (fixed physical time). Both curves sit orders of magnitude below the 15% owner gate.*

![fig2_fields.png](../benchmarks/verified/fused_push_pull_aa_pattern/figures/fig2_fields.png)

*Contours at the finest grid: analytic, AB reference, fused push-pull, and the difference amplified by 1e13 — the two kernels are the same field to round-off.*

![fig3_conservation.png](../benchmarks/verified/fused_push_pull_aa_pattern/figures/fig3_conservation.png)

*Left: relative mass drift per run (gate 1e-12). Right: viscosity inverted from the Taylor-Green energy decay versus the prescribed 1/6.*

## 4. Points to watch

* **State-label offset.** The fused kernel stores post-collision populations; the AB
  reference stores post-stream ones.  Comparing the two raw states measures one collision,
  not an error.  Any future 'bit-identical' claim must state the level it compares at.
* **Periodic only.** Neither kernel is a boundary-condition implementation: the fused
  kernel wraps by index arithmetic (`%n`), exactly like upstream.  Walls, inlets and
  forcing stay in `tensorlbm.boundary` on top of this step.
* **Storage claim is structural.** Upstream keeps 17 doubles/node (rest + two 8-direction
  buffers); the repo module can also run a single 8-direction buffer (9/node) because a
  vectorised gather completes before any write.  No bandwidth/MLUPS number is claimed
  here: the measurement host was heavily loaded (load15 > 100) and timings in the table
  are wall-clock under that load, not a performance claim.
* **Upstream CUDA was not executed** (no GPU in this sandbox); the equivalence is with the
  repo's NumPy/torch transliteration of the snapshot, whose provenance hashes were checked
  by the earlier audit.

## 5. Error judgement against the acceptance line (<15%)

* Finest-grid `u_x` relative L2, worst of the two kernels: **0.1121%** — **< 3%** ✓ (owner gate 15%, self target 3%).
* Cross-arm velocity difference: **3.066e-14** — < 1e-10 ✓.
* Same-level state equivalence: **3.331e-16** — < 1e-12 ✓.
* Mass drift: max **2.65e-14** — < 1e-12 ✓.
* Both negative controls exceed the 15% gate ✓ (the gate is not toothless).

Verdict: **PASS** — `verify.py` exits 0 on this `result.json`.

