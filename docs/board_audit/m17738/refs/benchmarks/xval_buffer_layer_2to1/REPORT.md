# XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT

Cross-validation of TensorLBM's 2:1 buffer-layer refinement against haimingz/vivsim `vivsim/multigrid.py`.

> Every number below is rendered from `artifacts/xval_buffer_layer_2to1/xval_buffer_layer_2to1.json` by `run_xval.py`; nothing is typed by hand.

## 1. Problem and reference provenance

- anchor: `XVAL-LBM-BUFFER-LAYER-2TO1-REFINEMENT`
- reference: `haimingz/vivsim` `vivsim/multigrid.py` @ `fb84b12f02e18f3b9dfdf738d6a1a318d81539f9` (MIT); oss_solutions feature `NEW:LBM-BLOCK-REFINEMENT-2TO1 (oss_solutions id=7, tag buffer-layer-2to1-refinement)`, oss_inventory id 11
- vendored upstream sha256: `d104683cfad3c3114f74e42aebf2678cec1de4d63917cb1d5c13c97a72af6515` (matches pin: True)
- shim: import block only: 'import jax.numpy as jnp' -> '. import np_jax_shim as jnp'; '.lbm.basic' -> '.lattice_dirs' (same four int32 direction tuples). Function bodies byte-identical.
- provenance limitation: working tree at /tmp/vivsim_src; its .git object store is damaged (git rev-parse HEAD fails), so the sha256<->commit binding rests on oss_inventory.cache_commit=fb84b12 and cannot be re-verified locally

The OSS module is a *prototype*: upstream ships no example that calls it, has no tests/asserts, and documents a buffer layer on the coarser side but no reflux. Its relaxation rule and transfer index algebra are nevertheless executable, and the vendored copy executes them verbatim here (`benchmarks/xval_buffer_layer_2to1/oss_ref/`).

## 2. Method

1. **Relaxation ladder**: OSS `get_omega(nu, level)` executed vs TensorLBM's `tau = 3 nu + 1/2` at the corresponding level (lattice viscosity scales as `2**level`).
2. **Transfer index algebra**: `fine_to_coarse` / `coarse_to_fine` of the executed OSS source vs the NumPy port `tensorlbm.block_refinement_2to1`, all four faces, 3 random fields each, layouts transposed explicitly.
3. **Intended difference**: on a field with a controlled non-equilibrium shear stress the OSS copy and the FH rescaling must differ by exactly the FH factor on the non-equilibrium part.
4. **Physical solution**: the *same* solver (`CavityAMR2D`) runs with `interface_scheme='fh'` and `interface_scheme='vivsim_copy'` -- one changed variable -- identical grid, `U_lid`, `Re`, step count and inventory reflux. Centreline profiles are compared with each other (XVAL gate) and with Ghia et al. (1982).
5. **Failure mode**: both schemes with `reflux=False` for the same step count, exposing what the OSS prototype omits.

### Parameter table

| quantity | value | note |
| --- | --- | --- |
| Re | 100.0 | lid-driven cavity, steady |
| U_lid | 0.050 | lattice units |
| coarse grid | 64² | background |
| fine strips | 2 x 32 fine rows, 128 fine columns | ratio 2, top and bottom |
| tau coarse / fine | 0.5960 / 0.6920 | equal physical Re |
| Ma = U_lid/c_s | 0.0866 | c_s = 1/sqrt(3) |
| steps | 12000 | coarse steps (fine does 2 sub-steps) |

## 3. Results

### 3.1 Relaxation ladder (must agree)

Worst relative difference over 30 (nu, level) points: **2.887e-16** (nu0=0.025, level=1).

| nu (level 0) | level | OSS omega | TensorLBM omega | rel err |
| --- | --- | --- | --- | --- |
| 0.010 | -2 | 1.9704433498 | 1.9704433498 | 2.25e-16 |
| 0.010 | 1 | 1.7857142857 | 1.7857142857 | 2.49e-16 |
| 0.025 | -1 | 1.8604651163 | 1.8604651163 | 1.19e-16 |
| 0.025 | 2 | 1.2500000000 | 1.2500000000 | 1.78e-16 |
| 0.050 | 0 | 1.5384615385 | 1.5384615385 | 0.00e+00 |
| 0.100 | -2 | 1.7391304348 | 1.7391304348 | 1.28e-16 |
| 0.100 | 1 | 0.9090909091 | 0.9090909091 | 0.00e+00 |
| 0.250 | -1 | 1.1428571429 | 1.1428571429 | 1.94e-16 |
| 0.250 | 2 | 0.2857142857 | 0.2857142857 | 1.94e-16 |
| 0.500 | 0 | 0.5000000000 | 0.5000000000 | 0.00e+00 |

### 3.2 Transfer index algebra (must agree)

Max absolute difference over 12 (face, field) checks: **0.000e+00** -- the port is bit-identical to the executed OSS operators.

### 3.3 Intended difference on non-equilibrium data

| quantity | value |
| --- | --- |
| FH non-equilibrium factor tau_f/(2 tau_c) | 0.580537 |
| measured OSS/FH non-equilibrium ratio | 1.722466 |
| predicted OSS/FH ratio | 1.722543 |
| max relative difference in the field | 0.012 % |

### 3.4 Physical solution: FH vs OSS copy (XVAL gate)

| quantity | FH arm | OSS-copy arm | cross-arm |
| --- | --- | --- | --- |
| max |u| / U_lid | 1.0000 | 1.0000 | - |
| max |v| / U_lid | 0.1107 | 0.1183 | - |
| max dev vs Ghia [% U_lid] | 16.973 | 16.377 | - |
| max |du| [% U_lid] | - | - | 0.951 |
| max |dv| [% U_lid] | - | - | 1.304 |
| rms du / dv [% U_lid] | - | - | 0.551 / 0.743 |
| < 15 % gate | - | - | PASS |

### 3.5 Sensitivity (grid and time)

| case | steps | FH vs Ghia [%] | OSS vs Ghia [%] | cross-arm [%] |
| --- | --- | --- | --- | --- |
| 48² Re=100 | 12000 | 18.360 | 19.377 | 1.135 |
| 64² Re=100 (main) | 12000 | 16.973 | 16.377 | 0.951 |

Time-convergence (profile change between the last two checkpoints, same arm):
- fh: max |du| between steps 9000 and 12000 = 0.025 % U_lid
- vivsim_copy: max |du| between steps 9000 and 12000 = 0.059 % U_lid

### 3.6 Failure mode: no reflux

| arm | steps | max |u|/U_lid | mass drift (rel) | max dev vs Ghia [% U_lid] |
| --- | --- | --- | --- | --- |
| FH | 1800 | 1.000 | 4.846e-05 | 16.039 |
| OSS copy | 1800 | 1.000 | 4.813e-05 | 15.856 |

Honest reading: at Re=100 these 1800 steps the no-reflux leak stays at the 1e-5 relative level and the Ghia deviation is statistically indistinguishable from the refluxing arms, so this configuration does **not** demonstrate that reflux is load-bearing. The reflux is kept as a mass-neutral safety net (it changes nothing at 5e-5) and because the Re=1000 ladder in docs/amr-reflux-maturation.md loses 1.4 % over 120 steps without it; that is a different, stiffer configuration and is not measured here.

## 4. Differences and porting trade-offs

| aspect | vivsim OSS | TensorLBM (this port) | decision |
| --- | --- | --- | --- |
| relaxation per level | get_omega formula | identical (section 3.1) | adopt as-is |
| restriction | 4-point average, crossing populations only, no rescale | plain block_average for the shadow row (scheme); FH rescale-back (default) | OSS semantics available as a scheme; FH stays default |
| prolongation | plain copy, one fine plane | plain block_repeat (scheme); FH rescale + block_repeat (default) | FH default: the copy mis-scales the non-equilibrium stress (section 3.3) |
| time coupling at ghost fill | single-time | linear interpolation over the two sub-steps (scheme: single-time) | default keeps interpolation; the scheme exposes OSS behaviour |
| mass conservation | none (no reflux) | inventory reflux (scheme inherits it) | keep reflux as a safety net; section 3.6 shows it is neutral at Re=100 (5e-5) |
| lattice convention | shifted DDF, (q,nx,ny) | full DDF, (q,ny,nx) | the shift cancels in average/copy; layout transposed explicitly (section 3.2) |
| validation surface | no tests, no caller | tests/test_xval_buffer_layer_2to1.py + this harness | add tests |

## 5. Error judgement

- cross-arm centreline deviation: **0.951 %** of U_lid -> < 15 %
- FH arm vs Ghia Re=100: **16.973 %** of U_lid -> NOT < 15 %
- OSS-copy arm vs Ghia Re=100: **16.377 %** of U_lid -> NOT < 15 %
- relaxation and operator identities: max rel err 2.89e-16 / max abs err 0.00e+00

Scope and limitations: 2-D D2Q9, single refinement level, structured strips, one moving wall; the OSS prototype's *geometry convention* (which coarse index is the buffer) is undocumented and has no caller upstream, so only its index algebra and relaxation were adopted, not its block placement. No 3-D, no momentum reflux, no moving geometry.

