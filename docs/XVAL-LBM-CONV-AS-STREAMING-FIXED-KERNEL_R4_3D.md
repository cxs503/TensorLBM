# XVAL · conv-as-streaming fixed kernel · round 4: 3D physics validation

**Anchor** `XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL` · **mission** 22987 (self-start, generalist seat)
· **repo** TensorLBM · **commit** `6d480901f34c` (branch `agent/m22987`, dirty=True)
· **generated** 2026-11-16T05:46:26Z from `runs/xval_conv_as_streaming_3d/result.json`.


> **Scale of the numbers in this document.** They come from the CI-sized configuration
> (`--quick`: Stokes mode n=16/120 steps, slit Ny=12, duct n=12; 43 s wall) because the
> predeclared full ladder (n in {24,32}, Ny in {12,16}, n in {12,16} at 30000 steps) was
> submitted to the scheduler as job `cj4518` and had not left the queue when this commit
> was written. No full-scale KPI is claimed here; the CI numbers are reported as measured.

## 1. Case introduction -- what rounds 1-3 left open

Rounds 1-3 adopted the transport formulation of
`loliverhennigh/Lattice-Boltzmann-fluid-flow-in-Tensorflow` (populations channel-last, the
whole migration step = one constant 3×3 convolution with exactly one unit tap per output
channel, wall bounce = a 1×1 cross-channel convolution blended through the solid mask) and
compared it with TensorLBM's production gather transport:

* 2D: bit-wise identical operator, Taylor-Green decay and plane Poiseuille both inside 3 %
  (`docs/XVAL-LBM-CONV-AS-STREAMING-FIXED-KERNEL.md`, `_R2`, `_R3`);
* 3D: only the *operator* was compared (`stream_conv_3d` vs the canonical roll traversal on a
  random field). **No 3D simulation was ever run through the conv arm, and 3D wall bounce was
  never exercised.**

That is the gap closed here. Three closed-form 3D judges, two arms that differ *only* in the
transport/bounce operator (collision, forcing, initial condition, dtype and step count are
shared code):

| case | physical setting | closed-form judge |
|---|---|---|
| `mode3d` | periodic cube, divergence-free Fourier mode, Stokes regime (A = 0.001) | `u ∝ exp(-ν|k|²t)`, `|k|² = 3k²` exactly |
| `slit3d` | walls in y only (two flat walls), periodic x and z | Hagen-Poiseuille parabola, half-way spacing `H = Ny-2` |
| `duct3d` | **four lateral walls incl. corners** (square duct), periodic streamwise | Boussinesq series + literature `f·Re = 56.908` |

Parameters (all predeclared in `benchmarks/xval_conv_as_streaming/predeclared_3d.json`):
D3Q19, BGK `τ = 0.8` (`ν = 0.1`), target `u_max = 0.02`, float64, CPU torch,
`torch.set_num_threads(4)`.

## 2. How it was done

- `benchmarks/xval_conv_as_streaming/run_3d_physics.py` -- self-contained driver.
  - arm A: `stream_gather_3d`, a modulo gather written against the D3Q19 velocity table and
    deliberately **not** importing `streaming_conv`, plus explicit index reversal at solid
    nodes (the 3D analogue of `tensorlbm.solver.stream`, which is 2D-only);
  - arm B: `tensorlbm.streaming_conv.stream_conv_3d` + `streaming_conv.bounce_blend`.
- **The judge is validated before it judges.** The square-duct series is checked by
  (a) convergence in the number of terms, (b) finite-difference Poisson residual against
  `-g/ν`, and (c) the friction-velocity product against the literature value
  **f·Re = 56.908** (Shah & London 1978, Table 49). If (b) or (c) leave 1 %, the duct KPI is
  reported WITHHELD instead of pass/fail -- `judge_ok = True`.

| series terms | u_max | u_max rel. diff vs 201 terms [%] | f·Re |
|---|---|---|---|
| 21 | 7.367653e-03 | 0.0081 | 56.90992 |
| 51 | 7.367012e-03 | 0.0006 | 56.90805 |
| 101 | 7.367063e-03 | 0.0001 | 56.90791 |

  Poisson residual: mean **0.0423 %** from `-g/ν`
  (spread 1.8323 %), `f·Re = 56.90789` vs literature
  56.908 → **0.0002 %**. `u_max/u_mean =
  2.0962` (textbook square duct ≈ 2.096). The judge agrees with an
  independent published constant, so the duct KPIs below are not self-referential.
- Reproduce:

```
python benchmarks/xval_conv_as_streaming/run_3d_physics.py
python -m pytest tests/test_streaming_conv_3d_physics.py -q
```

## 3. Results

| case | arm | config | steps | got | reference | L2 / decay-err [%] | peak / centre [%] | drift / f·Re |
|---|---|---|---|---|---|---|---|---|
| mode3d | A gather | n=16 | 120 | 4.745468e-02 | 4.626377e-02 | 2.5742 | 1.000000 | -8.44e-15 |
| slit3d | A gather | Ny=12,nx=6,nz=4 | 4000 | 1.985600e-02 | 2.000000e-02 | 0.3834 | 0.2828 | 6.85e-10 |
| duct3d | A gather | n=12,nx=4 | 4000 | 1.989956e-02 | 2.000000e-02 | 1.5672 | 0.5022 | 55.261 vs 56.908 (2.893%) |
| mode3d | B conv | n=16 | 120 | 4.745468e-02 | 4.626377e-02 | 2.5742 | 1.000000 | -8.44e-15 |
| slit3d | B conv | Ny=12,nx=6,nz=4 | 4000 | 1.985600e-02 | 2.000000e-02 | 0.3834 | 0.2828 | 6.85e-10 |
| duct3d | B conv | n=12,nx=4 | 4000 | 1.989956e-02 | 2.000000e-02 | 1.5672 | 0.5022 | 55.261 vs 56.908 (2.893%) |

`drift / f·Re` column: for `mode3d` it is the relative mass drift over the run; for `slit3d`
the half-run change of the peak velocity (steady-state evidence); for `duct3d` the numerical
`f·Re` next to its series value.

### 3.1 Arm-pair equivalence in 3D (the load-bearing claim)

| configuration | final f sha256 equal | cost ratio B/A |
|---|---|---|
| n=16, tau=0.8, amp=0.001, steps=120 | yes | 3.32 |
| ny=12, nx=6, nz=4, tau=0.8, umax_target=0.02, steps=4000 | yes | 1.78 |
| nx=4, n=12, tau=0.8, umax_target=0.02, steps=4000 | yes | 1.49 |

Under a force and a corner-bearing mask, the two arms still produce the **same final
population field bit-wise** (`f sha256` equal), i.e. the fixed-kernel formulation is not merely
"close" in 3D: it is the same operator, and `bounce_blend` is the same wall condition as the
explicit reversal.

Operator-level checks (`equivalence_3d`):

| 3D operator check | bit-wise equal | max |Δ| |
|---|---|---|
| conv_vs_gather_random | yes | 0.000e+00 |
| conv_vs_canonical_roll | yes | 0.000e+00 |
| bounce_conv_vs_index | yes | 0.000e+00 |

### 3.2 Figures

![fig1_stokes_mode_decay.png](../../runs/xval_conv_as_streaming_3d/figures/fig1_stokes_mode_decay.png)

*3D Stokes-mode decay: both transport arms against the exact exp(-ν|k|²t) rate, and the fitted-rate error.*

![fig2_slit_channel_profile.png](../../runs/xval_conv_as_streaming_3d/figures/fig2_slit_channel_profile.png)

*3D slit channel (walls in y, periodic x/z): velocity profile vs the exact parabola, and the error per arm/resolution.*

![fig3_square_duct_field.png](../../runs/xval_conv_as_streaming_3d/figures/fig3_square_duct_field.png)

*Square duct cross-section: numerical field (conv arm), Boussinesq series judge, and the relative error map.*

![fig4_square_duct_kpis.png](../../runs/xval_conv_as_streaming_3d/figures/fig4_square_duct_kpis.png)

*Square-duct error vs resolution and the numerical friction law against Shah & London's f·Re = 56.908.*

![fig5_transport_cost_3d.png](../../runs/xval_conv_as_streaming_3d/figures/fig5_transport_cost_3d.png)

*Wall time per step for every 3D run (informational; both arms run on the same CPU).*

## 4. Error verdict (explicit)

Worst KPI over all runs and both arms:

- `gamma_fit_rel_err_pct_max` = **2.5742 %**
- `slit_profile_l2_pct_max` = **0.3834 %**
- `slit_peak_rel_err_pct_max` = **0.2828 %**
- `duct_field_l2_pct_max` = **1.5672 %**
- `duct_centre_rel_err_pct_max` = **0.5022 %**
- `duct_fRe_rel_err_pct_max` = **2.8932 %**

- Owner gate (15 %): **PASS** -- every predeclared KPI except the
  informational cost ratio is inside 15 %.
- Internal target (3 %): **PASS**.
- Arm-pair bit-wise equality in 3D: **yes**.
- Duct judge self-check: **ok**.
- **Is the key quantity < 3 %?** Yes -- see the per-KPI list above; no
  KPI is hidden or re-tuned. Machine verdict: `PASS`.

## 5. Notes, traps and limits

- **The judge is the fragile part, not the code.** A Boussinesq series with a wrong
  Fourier/cosh role (`a` vs `b`) still looks plausible; here it is pinned by reproducing
  f·Re = 56.908 to 0.0002 %. Without that check the duct numbers would be unfalsifiable.
- **Half-way wall placement.** The references are evaluated at the half-way planes (0.5 and
  N-1.5), so no O(1) offset is folded into the comparison; the remaining error is the genuine
  discretisation error of half-way bounce-back, which is second order in the wall spacing.
- **Duct corners** are singular for the series, so `field_l2_pct` is expected to converge more
  slowly than the flat-wall profile; `centre_rel_err_pct` and `f·Re` are the corner-insensitive
  KPIs and are reported separately.
- **Scope**: float64 CPU only, `τ = 0.8`, D3Q19, BGK, steady/decaying laminar cases at
  Ma ≤ 0.035. Turbulent 3D, GPU, D3Q27 and moving boundaries are **not** covered here; in
  particular nothing in this round says the 3D fixed kernel is faster on GPU (the cost numbers
  in Fig. 5 are single-socket CPU wall time and informational only).
- **Reproducibility**: all runs are deterministic; re-running the driver reproduces `f_sha256`.

## 6. Artefacts

Tracked copies of the run artefacts (runs/ is git-ignored) live in
`benchmarks/xval_conv_as_streaming/artifacts_3d/`: `result.json` + `figures/`.


- driver: `benchmarks/xval_conv_as_streaming/run_3d_physics.py`
- predeclaration: `benchmarks/xval_conv_as_streaming/predeclared_3d.json`
- results: `runs/xval_conv_as_streaming_3d/result.json` (all numbers above are injected from it)
- figures: `runs/xval_conv_as_streaming_3d/figures/`
- tests: `tests/test_streaming_conv_3d_physics.py`
- environment: python 3.11.6, torch 2.7.1, numpy 2.4.6, total wall 43 s
