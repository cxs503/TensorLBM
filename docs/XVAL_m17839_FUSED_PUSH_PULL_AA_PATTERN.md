# XVAL — OSS `fused-push-pull-aa-pattern` cross-validation (TensorLBM, mission 17839)

**Anchor**: `XVAL-LBM-FUSED-PUSH-PULL-AA-PATTERN` · **Gate**: 15 % (owner standard) · **Target**: 3 % · **Verdict**: PASS / PASS

## 1. Case: what was cross-validated and where the reference comes from

Reference implementation: `lbm-principles-practice/code` (MIT; Krüger, Kusumaatmaja, Kuzmin, Shardt, Silva, Viggen, *The LBM: Principles and Practice*, Springer 2016), chapter 13 GPU code. Source files were fetched through the GitHub contents API and snapshotted under `xval/fused_push_pull_aa_pattern/ref/` with sha256:

| file | bytes | sha256 |
|---|---|---|
| `chapter13/gpu/LBM.cu` | 13686 | `0c9854fa353b62415bda2ff29fef890ece9254617990d7f94a2313537d0feb0c` |
| `chapter13/gpu/LBM.h` | 1747 | `0bc95be600f9416e58b70455ef756dea2875a14077e4dcddd4addb24b2ec0920` |
| `chapter13/gpu/main.cu` | 6622 | `ebbb847c12b4d52495bef359c44bf0a1d802a6b1c2047377dc8eed9e3e5af98e` |

Method read off `chapter13/gpu/LBM.cu` (not from prose):

* storage split `LBM.cu:23-35`: rest direction in its own array `f0` (`NX*y+x`), the eight moving directions direction-major (`NX*(NY*(d-1)+y)+x`, so `x` is contiguous);
* one kernel `gpu_stream_collide_save` per timestep `LBM.cu:172-252`: **pull** the eight shifted neighbours of the post-collision field with periodic `% NX`, `% NY` (`LBM.cu:181-201`), compute `rho/ux/uy` from those nine numbers, write the macroscopic fields only when `save` is set (`LBM.cu:203-216`), relax and write the post-collision populations into the *other* 8-direction buffer plus `f0` in place (`LBM.cu:218-252`); `main.cu` swaps the two 8-direction pointers each step;
* **no intermediate post-stream field exists** — that is the fusion, and it is what the anchor calls the AA pattern here (not minimal storage: upstream persists 17 doubles/node).

Physical case: 2-D Taylor-Green vortex, periodic, `rho = rho0 + 3P` with the pressure field of the exact solution, graded by upstream's own diagnostics (`LBM.cu:231-262`: relative L2 in `rho` (fluctuation norm), `ux`, `uy`).

## 2. Method and implementation

| piece | what | where |
|---|---|---|
| upstream kernel transliteration (NumPy) | arm `oss_numpy` | `xval/fused_push_pull_aa_pattern/oss_port.py` |
| fused push-pull stepper (torch, repo) | arm `torch_fused` / `torch_single` | `src/tensorlbm/fused_push_pull.py` |
| repo AB reference | arm `torch_ab` | `solver.stream` + `collide_bgk` |
| bench / controls / tables | — | `xval/fused_push_pull_aa_pattern/run.py` |
| renderer (this document + figures) | — | `xval/fused_push_pull_aa_pattern/make_report.py` |
| tests | 12 | `tests/test_fused_push_pull_xval.py` |

All arms run in float64, same IC, same `tau`, same step count, same sampling (`chapter13/gpu/LBM.h`: `scale=2`), so any difference is implementation, not physics. Environment: python 3.11.6, torch 2.7.1, numpy 2.4.6, loongarch64.

## 3. Parameters (all non-dimensional / lattice units)

| quantity | value |
|---|---|
| `NX` | 64 |
| `NY` | 64 |
| `nu_lu` | 0.16667 |
| `tau` | 1.00000 |
| `omega` | 1.00000 |
| `u_max_lu` | 0.02000 |
| `Ma` | 0.03464 |
| `Re_L` | 1.22231 |
| `k_lu` | 0.09817 |
| `k_h` | 0.09817 |
| `c_s_lu` | 0.57735 |
| `t_decay_lu` | 311.25868 |
| `T_total_lu` | 800 |
| `decay_at_T` | 0.07652 |
| lattice | D2Q9, BGK, periodic, single phase, no forcing |

`Re_L = u_max L / nu` with `L = NX/(2*pi)`; `Ma = u_max/c_s`; one decay time `t_d = 1/(nu*2k^2) = 311.25868` steps; run length `T = 800` steps = 2.570 `t_d`.

## 4. Comparison table (reference vs got, errors ticked)

### 4.1 Accuracy against the closed-form solution (owner gate)

| quantity | reference | got (arm) | error | < 15 % | < 3 % |
|---|---|---|---|---|---|
| `u_x` rel. L2 at T | analytic TG | 0.00192 (oss_numpy) | 0.00192 | ✓ | ✓ |
| `u_y` rel. L2 at T | analytic TG | 0.00192 (oss_numpy) | 0.00192 | ✓ | ✓ |
| `rho` rel. L2 at T | analytic TG | 0.02885 (oss_numpy) | 0.02885 | ✓ | ✓ |
| `nu` inverted from E(t) decay | 0.16667 | 0.16680 (oss_numpy) | 7.84e-04 | ✓ | ✓ |

### 4.2 Cross-arm agreement (does the repo reproduce the OSS step?)

| comparison | rel. L2 of 9 populations | rel. L2 of `u_x` | bitwise |
|---|---|---|---|
| oss_numpy vs torch_fused | 0 | 1.35e-14 | yes |
| oss_numpy vs torch_single | 0 | 1.35e-14 | yes |
| oss_numpy vs repo AB | 1.92e-16 | 8.35e-14 | no |

Storage (structural, doubles per node): upstream `17`, torch_fused `17`, torch_single `9`, repo AB `18`.

### 4.3 Controls — can the gate fail?

| control | analytic error | gate | ux rel. L2 vs oss | interpretation |
|---|---|---|---|---|
| `nu_bias_25pct` (kernel nu x1.25, graded vs true nu) | 0.47562 | FAIL | — | the analytic gate bites |
| `mirror_x` (West/East swap) | 0.78468 (`u_y`) | FAIL | 0.78427 | analytic gate AND arm comparison both catch it |
| `flip_all` (all velocities negated) | 0.00192 | PASS | 2.02e-04 | **blind spot**: a global sign flip is a symmetry of the TG state |

### 4.4 Resolution study

| NX | steps | `u_x` rel. L2 | `rho` rel. L2 | `nu` inv. error | oss vs torch (rel. L2) |
|---|---|---|---|---|---|
| 32 | 200 | 0.00812 | 0.11264 | 0.00321 | 0 |
| 64 | 800 | 0.00192 | 0.02885 | 7.84e-04 | 0 |
| 128 | 3200 | 3.67e-04 | 0.02072 | 1.80e-04 | not run |

Observed orders of accuracy between successive resolutions (fixed `nu`, fixed `u_max`): 2.08, 2.39.

## 5. Convergence and sensitivity (at least two axes)

### 5.1 Resolution

|k| doubles, error of `u_x` falls by a factor 4.232 (NX 32→64) and 5.226 (64→128).

### 5.2 Relaxation time `tau` (nu = (tau-1/2)/3), NX=64, 200 steps

| tau | nu | `u_x` rel. L2 | `nu` inversion error | gate |
|---|---|---|---|---|
| 0.6 | 0.03333 | 0.00153 | 2.26e-04 | ✓ |
| 0.8 | 0.10000 | 0.00118 | 1.06e-04 | ✓ |
| 1.0 | 0.16667 | 7.84e-04 | 6.71e-04 | ✓ |
| 1.4 | 0.30000 | 0.00103 | 0.00254 | ✓ |

### 5.3 Velocity amplitude `u_max` (Ma), NX=64, 200 steps

| `u_max` | Ma | `u_x` rel. L2 | `rho` rel. L2 | `nu` inversion error |
|---|---|---|---|---|
| 0.005 | 0.00866 | 5.36e-04 | 0.02576 | 7.96e-04 |
| 0.01 | 0.01732 | 5.91e-04 | 0.02569 | 7.71e-04 |
| 0.02 | 0.03464 | 7.84e-04 | 0.02539 | 6.71e-04 |
| 0.04 | 0.06928 | 0.00140 | 0.02421 | 2.70e-04 |

### 5.4 Where the density error lives

2-D FFT of the density-error field at t=400 is dominated by the modes [[2, 0, 0.0027569656559836408], [62, 0, 0.0027569656559836408], [0, 2, 0.0027569656559211786], [0, 62, 0.0027569656559211786]] — i.e. wavevector `|k| = 2k` along the axes, the second-harmonic pressure of the Taylor-Green state, not a grid-scale mode.

The L2 norm of that standing wave alternates with envelope period 28.57 steps; the acoustic prediction `T/2 = pi/(c_s * 2k) = 27.71` steps (mismatch 3.1 %). The density diagnostic therefore measures a *continuously forced acoustic wave* (compressibility error of the weakly compressible TG initial condition), amplitude band [0.002, 0.064], not a discretisation error that refines away — the NX=128 run still sits at 0.02072.

## 6. Failure modes, differences vs OSS, and porting trade-offs

### 6.1 Difference list (this repo vs `lbm-principles-practice/code` ch.13)

| aspect | OSS ch.13 GPU | TensorLBM (this port) | resolution |
|---|---|---|---|
| direction numbering | `6 2 5 / 3 0 1 / 7 4 8`, 1=E, 2=N, 3=W, 4=S, 5=NE, 6=NW, 7=SW, 8=SE | identical (`d2q9.C`, asserted in `test_velocity_table_matches_upstream_numbering`) | adopted |
| layout | `[d][y][x]`, `x` contiguous, rest direction separate | identical | adopted |
| fusion | one kernel: pull → moments → relax → write | identical (`step()`) | adopted |
| buffers | `f0` + `f1`/`f2` ping-pong, 17 doubles/node | same by default; `single_buffer=True` gives 9 doubles/node and is bit-identical (`test_single_buffer_equals_double_buffer`) | **extension** |
| moments write | gated by `save` flag | gated by `keep_momenta` | adopted |
| collision | BGK, `1/tau` folded into the equilibrium weights, `tauinv = 2/(6nu+1)` | identical in the fused path | adopted |
| boundaries | periodic only (`%NX`, `%NY`) | periodic only in this module; walls/BFL stay in `solver` + `boundary/` | **not adopted** (documented) |
| forcing / multiphase | none in ch.13 | none in this module (`solver` and the MCMP stack are separate) | not adopted |
| precision | double throughout | double throughout in this path (repo default elsewhere is float32) | adopted |
| diagnostics | `sqrt(sum e^2 / sum a^2)`, density against fluctuation | identical (`tg_l2_errors`) | adopted |
| moment summation | sequential `ft1+…+ft8` in the kernel | same inside `step()`; `macroscopic()` sums the stacked field (round-off-only difference, `<= 1e-15`) | noted |

### 6.2 Trade-offs and honest limits

* The fused path is **periodic-only**. Adopting it as a general stepper would require re-introducing the pull of missing neighbours from boundary conditions; the repo keeps the fused path as a fast, exactly-verifiable core and leaves walls in `solver`.
* `single_buffer=True` is safe **because the pull is materialised before any write**. A per-node loop port reads populations that the same step already overwrote — `test_inplace_per_node_update_is_not_safe` demonstrates the difference, and upstream's second buffer is exactly the guard against it.
* **Blind spot of this xval case**: negating *all* velocities (`flip_all` control) leaves the Taylor-Green diagnostics at 0.00192 — a symmetry of the case, so neither the analytic gate nor the arm comparison notices. Round 2 should add a non-symmetric case (e.g. a shear layer or a travelling wave) to close this hole.
* Population-level comparison is *insensitive*: the `mirror_x` control changes the velocity by 78 % but the raw 9-field difference is only O(1e-3) because the dense rest population masks it. Compare macroscopic quantities, not populations.
* Upstream's CUDA timings are **not** comparable to this CPU/torch port; no Mlups claim is made here.

## 7. Figures

![fig1](xval/fused_push_pull_aa_pattern/figures/fig1_decay_and_errors.png)

*fig1_decay_and_errors.png — fig1_decay_and_errors*

![fig2](xval/fused_push_pull_aa_pattern/figures/fig2_resolution.png)

*fig2_resolution.png — fig2_resolution*

![fig3](xval/fused_push_pull_aa_pattern/figures/fig3_rho_acoustic.png)

*fig3_rho_acoustic.png — fig3_rho_acoustic*

![fig4](xval/fused_push_pull_aa_pattern/figures/fig4_cross_arm.png)

*fig4_cross_arm.png — fig4_cross_arm*

## 8. Error judgement (explicit answer to the 3 % question)

| key quantity | value | vs reference | < 3 % ? |
|---|---|---|---|
| `u_x` rel. L2 | 0.00192 | analytic TG | yes |
| `u_y` rel. L2 | 0.00192 | analytic TG | yes |
| `rho` rel. L2 | 0.02885 | analytic TG | yes (forced acoustic wave, §5.4 — does not refine away) |
| `nu` from energy decay | 0.16680 | exact `0.16667` | yes |
| arm-to-arm `u_x` (repo vs OSS) | 1.35e-14 | OSS port | yes |

Verdict: **PASS** on the owner's 15 % gate for every graded quantity; the velocity field also meets the stricter 3 % target at the upstream default resolution, the density diagnostic does not (0.02885) and the reason is identified, not papered over.

## 9. Reproduction

```bash
cd xval/fused_push_pull_aa_pattern
python run.py --out result.json          # ~2 min on 1 core
python make_report.py                    # figures/ + this document
python -m pytest ../../tests/test_fused_push_pull_xval.py -q
```
