# SDF encoder v2 (B4-P2b) — design, results and diagnosis

- Date: 2026-08-25
- Base: `65cb33c` (branch `exp/b4-sdf2`, worktree `/nfs/wangxi/worktrees/b4_sdf2`)
- Corpus: 350 points = 238 legacy + 112 hull-form family points
  (`/nfs/wangxi/runs/b4_fam_20260824/cache_fam.npz`; families 6 slender
  `l_over_d=1.30`, 7 blunt `l_over_d=0.75`, 8 long_nose `nose_len=1.30`,
  9 aft_sail `sail_x=1.30`; legacy `fam=-1`)
- Run directory: `/nfs/wangxi/runs/b4_sdf2_20260825/`
- Locked acceptance bar: beat the linear `power_geoM` LOFO extrapolation on
  fam_slender AND fam_blunt (cited: 4.766% / 3.005%).

## 1. What ships (purely additive)

`src/tensorlbm/ai/geom_encoder.py` gains:

| symbol | role |
|---|---|
| `SDFEncoderV2` | multi-scale residual trunk, mean+max pooling per scale, `tanh` latent |
| `SDFCondFNODragV2` | joint model mirroring `SDFCondFNODrag` with the v2 encoder; adds `latent_pair()` |
| `ResidualBlock3d` | 2×(conv3d-GELU) + projection/identity skip |
| `vicreg_latent_penalty` | variance hinge + decorrelation on a `(B,d)` batch |
| `logit_margin_penalty` | soft `|logits| <= 2` guard (keeps `tanh' >= 0.07`) |
| `latent_spectrum` / `participation_ratio` | PCA spectrum + effective-dimension diagnostics |
| `VAR_EPS = 1e-4` | epsilon inside the penalty `sqrt` (see §5.2) |

v1 classes (`SDFEncoder`, `SDFCondFNODrag`) are byte-identical; the v1 test
file `tests/test_geom_encoder.py` passes unmodified. v2 is selected only by
instantiating the new classes — no config flag changes the v1 path.

## 2. v2 architecture (input contract identical to v1)

Input: pooled SDF `(B, 1, 32, 32, 64)` — exact boundary-restricted EDT,
clipped to ±8 voxels, scaled to [-1, 1], stride-2 mean pool (unchanged from
v1 / PR #235, so the A/B is clean).

- stem: `Conv3d(1, base, 3)` + GELU
- scale 1: `ResidualBlock3d(base, 2·base, stride 2)` → feature map `16×16×32`
- scale 2: `ResidualBlock3d(2·base, 4·base, stride 2)` → `8×8×16`
- scale 3: `ResidualBlock3d(4·base, 4·base, stride 2)` → `4×4×8`
- per scale: `cat[x.mean((2,3,4)), x.amax((2,3,4))]` → `4b / 8b / 8b` features
  (mean carries bulk shape, max carries thin appendage support — the v1
  global-mean-only pooling diluted 1-voxel fin structure)
- head: `Linear(20b, 32)` → `tanh` latent

| encoder | params | trunk depth | pooling |
|---|---|---|---|
| v1 `SDFEncoder` (base 8) | 46,288 | 4 conv, mono-scale | global mean |
| v2 `SDFEncoderV2` base 12 | 253,208 | 3 residual scales | mean+max × 3 scales |
| v2 `SDFEncoderV2` base 16 | 446,400 | 3 residual scales | mean+max × 3 scales |

Both v2 budgets are under the ~500k cap. Depth-wise features come from three
receptive fields (¼, 1/16, 1/64 of the volume), which is what makes sail
position and hull slenderness separable in principle.

## 3. Training protocol (identical machinery for every arm)

Copied from `lofo_v1.py` (PR #236): exact param-key grouping
`(re, u_in, sail, fin, hull, ds)`, `carve_val` with `VAL_SEED=1` / 15%,
`QuotaSampler` per epoch, AdamW `lr=1e-3 wd=1e-4`, batch 32, ≤500 epochs,
patience 60, aux head λ=0.1, z-score on fit statistics, early stop on val
MAPE with best-state restore, cudnn deterministic. Seeds 0/1/2 per cell.
Splits: `random` + `lofo::{slender,blunt,long_nose,aft_sail}` +
`loho::{bare_hull, with_sail, full}`.

Arms:

| arm | encoder | base | VICReg λ | logit-margin μ |
|---|---|---|---|---|
| `v1_ref` | v1 | 8 | 0 | 0 |
| `v2` | v2 | 12 | 0 | 0 |
| `v2_reg` | v2 | 12 | 0.1 | 0 |
| `v2_wide` | v2 | 16 | 0 | 0 |
| `v2_reg2` | v2 | 12 | 0.1 | 0.1 (margin 2.0) |

`v2_reg2` was added after a diagnosed rank-0 latent collapse (§5); it is the
same model as `v2_reg` plus the saturation guard. The linear baselines
`power_re` / `power_geoM` are re-run inside the same harness — they
reproduce the PR #236 `metrics_lofo.json` rows bit-for-bit (random 14.656 /
2.000; fam_slender 36.109 / 4.766; fam_blunt 23.576 / 3.005), so all numbers
below are directly comparable to the cited table.

## 4. Results

All numbers: mean ± population-std over seeds 0/1/2; linear rows n=1
(deterministic). Full machine-readable table in
`/nfs/wangxi/runs/b4_sdf2_20260825/metrics_lofo_v2.json` (136 rows).

### 4.1 Locked acceptance bar — FAIL

| family | power_geoM (cited) | v1_ref | v2 | v2_reg | v2_wide | v2_reg2 |
|---|---|---|---|---|---|---|
| fam_slender | **4.77** | 31.76±2.78 | 31.36±2.83 | 33.23±3.83 | 30.06±2.76 | 30.30±3.13 |
| fam_blunt | **3.01** | 28.54±0.79 | 28.85±1.37 | 28.42±1.69 | 34.55±10.62 | 28.81±0.75 |
| fam_long_nose | 2.28 | 3.24±0.52 | 4.03±0.39 | 3.89±0.35 | 3.81±0.70 | 4.02±0.36 |
| fam_aft_sail | 1.52 | 4.11±0.51 | 3.85±0.88 | 3.40±0.14 | 2.44±1.06 | 3.52±0.14 |

No arm beats the linear `power_geoM` baseline on fam_slender or fam_blunt —
the bar fails by 6.3× (30.06 vs 4.77) and 9.4× (28.42 vs 3.01) at best.
This reproduces the v1-era nonlinear collapse (PR #236 `metrics_lofo.json`:
F_geoM 30.26±3.7 / C_geoM 33.66±5.9 on fam_slender; 25.48 / 27.05 on
fam_blunt) — the encoder was not the bottleneck.

### 4.2 Secondary bars

**Bare-fold (loho::bare_hull)** — no catastrophic regression vs v1's
0.14±0.08 (238-pt PR #235): PASS for every v2 arm.

| power_geoM | v1_ref | v2 | v2_reg | v2_wide | v2_reg2 |
|---|---|---|---|---|---|
| 1.71 | 0.15±0.05 | 0.17±0.08 | **0.10±0.04** | 0.18±0.01 | 0.15±0.04 |

**Latent effective dims ≥ 8 (participation ratio)** — FAIL for every arm:

| arm | PR | top-5 eigenvalue ratios |
|---|---|---|
| v1_ref | 1.00 | 0.999 / 0.0013 / 3.3e-5 / 5e-6 / 2e-6 |
| v2 | 0.00 | exactly zero total variance (corpus-constant latent) |
| v2_reg | 0.00 | exactly zero total variance |
| v2_wide | 1.61 | 0.745 / 0.255 / 0 / 0 / 0 |
| v2_reg2 | 1.01 | 0.997 / 0.0015 / 4.7e-4 / 3.6e-4 / 1.5e-4 |

**Fin-visibility probes above chance** — mixed (5-fold out-of-fold linear
probes on the 350-pt latents):

| arm | hull3 (chance 0.537) | fin_bin (chance 0.560) | fam4 (chance 0.25) | sail_scale R² | fin_scale R² |
|---|---|---|---|---|---|
| v1_ref | 0.935 | 0.973 | 1.000 | 0.118 | 0.716 |
| v2 | 0.537 (=chance) | 0.560 (=chance) | 0.250 (=chance) | −0.08 | −0.01 |
| v2_reg | 0.537 | 0.560 | 0.250 | −0.08 | −0.01 |
| v2_wide | 0.617 | 0.643 | 0.750 | −0.08 | −0.01 |
| v2_reg2 | 0.949 | **0.991** | 1.000 | **0.926** | **0.871** |

The saturation-guarded v2_reg2 latent is the most geometry-informative of
all arms (fin presence 0.99, sail position R²=0.93 where v1 managed 0.12) —
but it packs everything into ~1 effective dimension, which is why PR still
reads 1.01.

### 4.3 Full split table (MAPE %)

| split | power_re | power_geoM | v1_ref | v2 | v2_reg | v2_wide | v2_reg2 |
|---|---|---|---|---|---|---|---|
| random | 14.66 | 2.00 | 0.75±0.08 | 0.65±0.02 | 0.69±0.04 | 0.69±0.10 | **0.63±0.07** |
| lofo::fam_slender | 36.11 | 4.77 | 31.76±2.78 | 31.36±2.83 | 33.23±3.83 | 30.06±2.76 | 30.30±3.13 |
| lofo::fam_blunt | 23.58 | 3.01 | 28.54±0.79 | 28.85±1.37 | 28.42±1.69 | 34.55±10.62 | 28.81±0.75 |
| lofo::fam_long_nose | 1.78 | 2.28 | 3.24±0.52 | 4.03±0.39 | 3.89±0.35 | 3.81±0.70 | 4.02±0.36 |
| lofo::fam_aft_sail | 4.58 | 1.52 | 4.11±0.51 | 3.85±0.88 | 3.40±0.14 | 2.44±1.06 | 3.52±0.14 |
| loho::bare_hull | 8.10 | 1.71 | 0.15±0.05 | 0.17±0.08 | 0.10±0.04 | 0.18±0.01 | 0.15±0.04 |
| loho::with_sail | 14.66 | 11.77 | 13.66±0.10 | 13.83±0.27 | 13.88±0.38 | 13.66±0.20 | 13.53±0.37 |
| loho::full | 14.62 | 7.22 | 9.77±0.32 | 8.96±0.52 | 8.80±0.73 | 9.20±0.55 | 9.01±1.17 |

On in-distribution work (random, bare-fold) the v2 arms match or beat v1
(v2/v2_reg2 0.63–0.69 vs v1 0.75 random). On every extrapolation corner
they sit within noise of v1 and of each other. loho::with_sail is hard for
everything on this corpus (linear 11.77, best arm 13.53) because the
held-out class includes the 112 family variants.

## 5. Diagnosis: why the bar is not reachable in this body

### 5.1 The joint objective lets the latent die (and what rescues one axis)

The FNO reads geometry from the solid-mask channel of `x`, so nothing in the
task loss needs the latent. Under the locked joint protocol the latents
collapse — but not all the way to v1's rank-1 sail axis:

- `v2` / `v2_reg`: latent is **corpus-constant** (exactly zero total
  variance over all 350 points; every probe at chance; VICReg plateaued at
  0.99 = 31 dead columns × hinge 1.0/32). The `tanh` head saturates at ±1
  on every row, so `d(tanh)/d(logit)` ≈ 0 and the variance hinge cannot
  push back — a stable dead equilibrium, not a slow one.
- `v2_wide`: 2 effective dims (0.745/0.255) — capacity alone buys a second
  axis, not eight.
- `v2_reg2` (margin guard): the hinge can finally act. The latent stays
  rank-1 in spectrum but is now the MOST informative of all arms (fin
  presence 0.99, sail position R²=0.93, family ID 1.00). The joint loss
  distills geometry into one saturated scalar axis — enough for
  interpolation, structurally unable to expose the l_over_d extrapolation
  direction `power_geoM` gets for free from its curated channels.

### 5.2 Two defects found on the way

1. **VICReg NaN instability** (v2_reg, first full run): a batch column with
   exactly zero variance (fully saturated `tanh`) makes `d(std)/dz → ∞`, so
   the first run developed NaN weights late in training (best-state restore
   protected the metrics but the arm was not trustworthy). Fixed with
   `VAR_EPS = 1e-4` inside the `sqrt` (the upstream VICReg convention) and a
   clamped correlation denominator; regression-pinned by
   `test_saturated_column_gradient_is_finite`. The whole matrix was re-run
   after the fix — no NaN anywhere.
2. **Degenerate-corpus crash**: the latent diagnostic originally raised on
   zero total variance instead of reporting the collapse it had found. The
   harness now reports `PR = 0` with a zero spectrum in that case.

### 5.3 Even an alive latent does not fix LOFO

To separate representation from regressor, the exact `power_geoM` linear
machinery was refit with the 32-d latent replacing the 4 hand channels
(feats `[1, log re, z, (log re)·z]`), on the same lofo splits
(`/nfs/wangxi/runs/b4_sdf2_20260825/latent_linlofo.json`, MAPE %):

| variant | arm | fam_slender | fam_blunt | fam_long_nose | fam_aft_sail |
|---|---|---|---|---|---|
| geom_ref | power_geoM | **4.77** | **3.01** | 2.28 | 1.52 |
| leaky | v1_ref | 94.45 | 71.31 | 15.53 | 12.20 |
| leaky | v2 | 36.11 | 23.58 | 1.78 | 4.58 |
| leaky | v2_wide | 46.57 | 26.62 | 2.81 | **1.37** |
| leaky | v2_reg2 | 81.51 | 73.30 | 85.01 | 44.71 |
| faithful | v1_ref | 10.13 | 78.29 | 6.04 | 2.55 |
| faithful | v2_reg2 | 1081.97 | 10335.17 | 19.51 | 6.28 |

(leaky = latents from the random-split model, so the encoder DID see the
held-out family — an upper bound on what the latent carries; faithful =
latents from the same lofo-split model, the protocol-honest variant.
v2/v2_reg rows collapse onto `power_re` exactly because their latent is
corpus-constant.)

Reading: no arm's latent supports the linear extrapolation on
fam_slender/fam_blunt even under the leaky upper bound. The one latent that
richly encodes geometry (v2_reg2) makes LOFO catastrophically worse
(10,335% on fam_blunt faithful): out-of-family hulls land off the learned
axis and the linear readout explodes. The FNO regressor is therefore NOT
the binding constraint — the jointly-trained latent does not contain, and
under LOFO actively corrupts, the hull-form extrapolation direction.
`power_geoM` wins because its four curated channels ARE that direction.

## 6. What v3 actually needs

- Take the geometry OUT of `x` (drop or ablate the solid-mask channel) for
  at least one arm, so the latent is the only path to the answer; or train
  the encoder with a reconstruction/contrastive objective BEFORE the joint
  fit (two-stage), so the representation exists independent of the FNO.
- Revisit the optimizer interaction: with Adam, a λ=0.1 auxiliary penalty
  contributes ~λ-fraction of the step direction once task gradients
  dominate the second moment — the observed 0.99 VICReg plateau at
  satisfied margins is consistent with the penalty being present but
  ineffective at this scale. λ sweeps or RMSProp-style decoupling are the
  cheap probes.
- The bar itself: `power_geoM` extrapolates with 4 physically-curated
  channels. A learned encoder must discover hull slenderness from 32³
  voxels with <500k params on 238 legacy + 112 family points. Given
  C_geoM (hand channels, guaranteed informative) also collapses to 25–34%
  on fam_slender/fam_blunt, the realistic next step is more corpus in the
  slender/blunt directions, not more encoder.

## 7. Verification record

- `tests/test_geom_encoder_v2.py`: 40 passed (GPU venv, CUDA device tests
  included); CPU CI venv 38 passed + 2 CUDA-skipped — same suite, same
  count modulo the hardware skips. Neighbor suites green:
  `tests/test_geom_encoder.py` + `tests/test_suboff_cad_hullform.py`
  50 passed (GPU venv).
- `ruff check` + `ruff format --check` clean on both touched files.
- `mypy src/tensorlbm --ignore-missing-imports`: 2761 errors / 222 files —
  identical to the `65cb33c` baseline (zero new).
- Determinism: `check_det.py` retrains (v2, lofo::fam_blunt, seed 0) twice —
  bitwise-identical MAPE (28.644908449779514 both runs), bitwise-identical
  28 test predictions, best_epoch 250/250. The four original arms also
  reproduced their prior complete run exactly when the matrix was relaunched
  with the added v2_reg2 arm (e.g. fam_blunt v1_ref 28.54±0.79 in both).
- Hardware: GPU 2 only (CUDA_VISIBLE_DEVICES=2), peak 3783 MiB
  (`torch.cuda.max_memory_allocated`), run wall time ≈ 1 h 47 min for
  136 result rows.
- One deviation from bitwise cross-device reproducibility: float32 conv3d
  reductions are not bitwise-portable CPU↔CUDA (observed max latent diff
  2.8e-5); pinned by `test_cpu_cuda_agree` with atol=1e-4. The bitwise
  guarantee covers the integer EDT and same-device reruns only.
