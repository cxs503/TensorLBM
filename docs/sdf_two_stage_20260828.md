# SDF two-stage encoder (L2) — supervised probe, frozen latent, surrogate head

- Date: 2026-08-28
- Base: `a3addebd` (branch `exp/sdf-two-stage`, worktree
  `/nfs/wangxi/worktrees/sdf_two`); module commit `b61fa587`
- Run directory: `/nfs/wangxi/runs/sdf_two_20260828/`
- New code: `src/tensorlbm/ai/sdf_two_stage.py` (+ `tests/test_sdf_two_stage.py`,
  14 tests). Purely additive; `geom_encoder.py`, `drag_cond.py`,
  `active_learning.py`, `inference_service.py` untouched.
- Pre-registered bars (from the task brief): stage-1 PR far from 0 (v2 joint
  measured 0.000); random-split MAPE ≤ 2×v3 = 0.7% to be "usable"; fam LOFO
  transfer non-catastrophic (< 5× in-family error; v2 was a 10,335%
  linear-readout disaster); joint-training ablation must reproduce the collapse.

## 1. Mechanism being repaired

#247 (`docs/geom_encoder_v2_20260825.md`) established that under **joint**
training the SDF latent dies: the FNO can read geometry from the solid-mask
channel of its own input, so the encoder gradient vanishes and the latent
becomes corpus-constant (participation ratio, PR = (Σλ)²/Σλ², = 0.000).
VICReg + logit-margin regularisation (`v2_reg2`) keeps the latent alive
(PR > 0) but the surviving representation is close to rank-1 and its linear
readout collapses out-of-family (faithful LOFO fam_blunt 10,335%).

The two-stage route removes the conflict instead of regularising it:

1. **Stage 1 — supervised probe.** Train `SDFEncoderV2` by regressing the
   design parameters and geo channels from the pooled SDF
   (`SupervisedSDFEncoder`: trunk + linear probe head, MSE + 0.1·logit-margin).
   The only gradient into the trunk is geometry→parameters, so a dead latent
   is a training failure, not a training optimum.
2. **Stage 2 — frozen latent + surrogate head.** Freeze the trunk; feed
   `[p | z]` (scalar conditioning + frozen latent) to the unchanged
   FiLM-FNO (`TwoStageCondFNODrag` wraps `CondFNODrag`, cond width
   2 + latent or 4 + latent). Only the head trains, on C_D.

No C_D information enters stage 1 (no drag leakage into the encoder).

## 2. Experimental design

- Corpora: **fam** 350 (238 legacy + 112 family; splits identical to #247:
  random 237/41/72 fit/val/test, each LOFO 274/48/28) and **v4** 274 with the
  original v4 random split (asserted equal to `preds_v4.npz::idx`).
- Stage-1 targets: fam 12 = log sail, log fin, hull_bare/sail/full, g0–g3,
  l_over_d, nose_len, sail_x; v4 9 (same minus multipliers/sail_x/nose_len/
  l_over_d, plus corpus geo channels). Trained **only on the split's fit
  points**; latents are extracted full-corpus afterwards for measurement.
- Stage-2 arms: `ts2` cond = [log10 re, log10 u_in] (the L2 story — scalars
  carry only operating condition, geometry must come from the latent);
  `ts4` = ts2 + [log sail, log fin] (v2 parity).
- Seeds 0/1/2; HP identical to #247 (epochs 500, batch 32, lr 1e-3, wd 1e-4,
  patience 60, QuotaSampler); encoder latent 32, base 12 (253,208 params);
  FNO body identical (in_ch 5, width 32, 4 layers, modes 16×32, mlp 128,
  film 64). Joint ablation re-runs `v2_joint` (VICReg λ0.1 ≡ #247 `v2_reg`)
  and `v2_reg2_joint` (≡ #247 `v2_reg2`) on this harness.
- Path-equivalence guard: full forward vs `forward_from_latent` are bitwise
  equal on the same batch; precomputed-latent drift from cuDNN
  batch-size-dependent conv algorithms measured max 1.93e-4 (< 1e-3 tol).

## 3. Stage-1 results (probe)

PR over seeds (latent dim 32):

| split | PR |
|---|---|
| fam random | **5.53 ± 0.01** |
| fam lofo slender / blunt / long_nose / aft_sail | 5.89 / 5.26 / 4.99 / 4.72 |
| v4 random | 3.79 ± 0.20 |
| joint v2_joint (fresh ablation, random) | 0.00 |
| joint v2_reg2_joint (fresh ablation, random) | 3.57 ± 1.41 |

The collapse is gone: PR ≈ 5.5 where joint training gives 0.00. Spectrum
(fam random, seed 0) decays smoothly (top-10 eigenvalues 0.272, 0.253, 0.134,
0.105, 0.080, 0.064, 0.043, 0.024, 0.009, 0.006) — a genuine multi-scale
representation, not rank-1. PR stays below the aspirational ≥ 8.

Probe R² on the random test split (fam, mean of 3 seeds): log_sail 0.991,
log_fin 0.996, hull_sail 0.999, hull_full 0.861, g0–g3 0.991–0.999,
l_over_d 0.999, nose_len 0.998, sail_x 0.999 — but **hull_bare 0.140**
(v4: 0.105). The encoder sees geometry only; bare-hull drag depends on the
operating point, so its geometric component is small. This is the honest
ceiling of a geometry-only probe and is exactly why stage 2 re-introduces
re/u_in as conditioning.

On LOFO held-out families the probe's geometry targets are constant within a
family (R² = nan by construction — each family fixes its signature parameter),
and the g0–g3 extrapolation R² explodes negative (−1e29 … −1e31): the probe,
like the surrogate, extrapolates poorly to unseen geometry.

## 4. Stage-2 results — MAPE (%)

fam corpus (mean ± std over seeds 0/1/2; #247 numbers are seed 0, n=1,
from `b4_sdf2_20260825/metrics_lofo_v2.json`):

| split | ts2 (this run) | ts4 (this run) | v2 joint | v2_reg2 joint | v1_ref | power_geoM |
|---|---|---|---|---|---|---|
| random | 0.85 ± 0.07 | 0.82 ± 0.00 | 0.622 | 0.640 | 0.859 | 2.000 |
| lofo fam_slender | 33.64 ± 4.10 | 33.49 ± 1.59 | 30.556 | 34.699 | 29.088 | 4.766 |
| lofo fam_blunt | **8.87 ± 4.05** | 8.12 ± 4.10 | 28.645 | 28.057 | 27.898 | 3.005 |
| lofo fam_long_nose | 4.42 ± 0.99 | 4.21 ± 0.14 | 4.573 | 3.668 | 2.854 | 2.275 |
| lofo fam_aft_sail | **2.81 ± 0.15** | 2.89 ± 1.17 | 5.016 | 3.554 | 4.305 | 1.517 |

v4 corpus random: ts2 0.93 ± 0.16 (per-seed 1.144/0.765/0.867), ts4 0.92 ± 0.09
(0.987/0.792/0.969); baselines C_full 0.557, B_encoding 0.648
(`b4_v4_20260824/metrics_v4.json`). fam random reference: C_geoM
0.342 (mean of 0.312/0.384/0.331, `b4_fam_20260824/metrics_lofo.json`),
F_geoM 0.456, power_geoM 2.000.

ts2 and ts4 are within noise of each other everywhere — sail/fin are already
linearly decodable from the latent, so the extra scalar columns are redundant.

### 4.1 Latent linear-readout LOFO (the #247 disaster metric)

Least squares on [1, log10 re, z, log10 re·z] (faithful = per-fold stage-1
latents; leaky = random-split latents that saw every family):

| variant / arm | slender | blunt | long_nose | aft_sail |
|---|---|---|---|---|
| faithful, two_stage | 29.26 | 44.60 | 9.93 | 4.11 |
| faithful, v2 (#247) | 38.11 | 23.58 | 1.78 | 4.58 |
| faithful, v2_reg2 (#247) | 1081.97 | **10335.17** | 19.51 | 6.28 |
| faithful, v1_ref (#247) | 10.13 | 78.29 | 6.04 | 2.55 |
| leaky, two_stage | 463.68 | 320.58 | 87.12 | 7851.70 |
| leaky, v2 (#247) | 36.11 | 23.58 | 1.78 | 4.58 |
| leaky, v2_reg2 (#247) | 81.51 | 73.30 | 85.01 | 44.71 |
| geom_ref, power_geoM | 4.77 | 3.01 | 2.28 | 1.52 |

The 10,335% catastrophe is gone (fam_blunt faithful 10335 → 44.6), but the
two-stage latent is *worse than v2's* under a linear readout (and its leaky
row is worse than #247's): the supervised latent encodes parameters in a
genuinely nonlinear way that a 65-column least-squares fit cannot exploit.
Linear decodability was a diagnostic, not the product; the surrogate-head
numbers in §4 are the actual metric.

### 4.2 In-family vs transfer (ts2)

| family | in-family MAPE | LOFO MAPE | ratio |
|---|---|---|---|
| fam_slender | 0.232 | 33.64 | 145.3× |
| fam_blunt | 0.337 | 8.87 | 26.4× |
| fam_long_nose | 0.385 | 4.42 | 11.5× |
| fam_aft_sail | 0.210 | 2.81 | 13.4× |

(In-family = random-split ts2 test MAPE restricted to that family's test
points, ~6 points/seed.)

## 5. Joint-training ablation (fresh, same harness)

| arm | split | PR | MAPE |
|---|---|---|---|
| v2_joint | random | **0.00** | [0.731, 0.653, 0.710] → 0.70 ± 0.03 |
| v2_joint | lofo fam_blunt | 1.00 | 28.593 |
| v2_reg2_joint | random | [5.01, 1.66, 4.04] → 3.57 ± 1.41 | 0.79 ± 0.10 |
| v2_reg2_joint | lofo fam_blunt | 3.64 | 27.988 |

The collapse reproduces under this harness (PR 0.000 on random), so the §3/§4
differences come from the two-stage design, not from luck or a changed
pipeline. `v2_reg2_joint` shows PR > 0 is *not sufficient*: PR 3.6 with blunt
LOFO 28.0%, vs two-stage PR 5.3 with blunt LOFO 8.9% — what matters is that
the latent was shaped by a geometry→parameters objective.

## 6. Verdict against the pre-registered bars

| bar | result |
|---|---|
| stage-1 PR far from 0 | **MET** (5.53 fam / 3.79 v4 vs joint 0.00; below the aspirational ≥ 8) |
| random MAPE ≤ 0.7% | **MISSED** — fam 0.85/0.82 (2.4–2.5× the C_geoM 0.342 reference), v4 0.93/0.92 (1.66× C_full 0.557) |
| fam LOFO < 5× in-family | **MISSED** — best family 11.5× (long_nose); slender 145× |
| joint ablation reproduces collapse | **MET** (fresh v2_joint PR 0.00, blunt 28.6%) |

Partial transfer wins: fam_blunt 28.6 → 8.9% (3.2×), fam_aft_sail
5.0 → 2.8% (1.8×), and the linear-readout catastrophe is eliminated.
fam_slender is untouched (30.6 → 33.6%).

## 7. Honest limitations

- **fam_slender does not transfer.** Every family is a single-parameter
  excursion from the mother hull (`l_over_d_mult` 1.30 for slender vs 1.0 in
  the whole fit set, blunt 0.75, long_nose / sail_x 1.30), and the 28
  held-out slender points are the *only* hulls at 1.30 — the encoder has
  never seen that geometry, and no encoder objective interpolates away a
  hole in the support. fam_blunt is equally OOD yet improved 3.2×, so
  coverage is not the whole story, but for slender it is decisive.
- **Frozen encoder costs in-distribution accuracy**: fam random 0.85 vs
  joint v2 0.622. Expected — stage 2 cannot reshape the trunk.
- **hull_bare R² 0.14** bounds what a geometry-only probe can express; drag
  needs the operating point, which only enters at stage 2.
- **The optional Re-dependent auxiliary head was not exercised** (aux_dim 0
  in all runs); it is implemented and tested but unmeasured.
- #247 comparison rows are single-seed (their harness ran seed 0 for the
  LOFO arms); this run is 3 seeds. The blunt gap (8.9 vs 28.6) is far larger
  than the seed spread (±4.1).
- `v2_joint` in this harness includes VICReg λ0.1, i.e. it is #247's
  `v2_reg`; `v2_reg2_joint` is #247's `v2_reg2`. Plain unregularised `v2`
  was not re-run (its numbers are cited from #247).

## 8. Next steps

1. Coverage, not objectives, gates slender: add a mid-`l_over_d` bridge
   batch before any further encoder work.
2. Re-aux probe head (implemented, unmeasured) may lift hull_bare R² and
   sharpen the latent's drag-relevant directions.
3. The leaky-vs-faithful linear-readout inversion suggests replacing the
   linear diagnostic with a shallow MLP readout if the decodability story
   needs to be told again.

## 9. Reproduction

```bash
# tests (CPU ok)
pytest tests/test_sdf_two_stage.py -q --basetemp=/nfs/wangxi/tmp/pt_sdf_two
# full run (GPU, 2103 s of training across 4 shards)
/nfs/wangxi/venvs/tensorlbm/bin/python /nfs/wangxi/runs/sdf_two_20260828/run_two_stage.py
# tables in this doc
/nfs/wangxi/venvs/tensorlbm/bin/python /nfs/wangxi/runs/sdf_two_20260828/analyze_ts.py
```

Artifacts in `/nfs/wangxi/runs/sdf_two_20260828/`:
`metrics_two_stage.json` (62 training rows + 28 linear-readout rows),
`latent_linlofo_ts.json`, `latents_all.npz`, `preds_shard{0..3}.npz`,
`rows_shard{0..3}.json`, `latents_shard{0..3}.npz`, `sdf_v4_274.npz`,
`shard{0..3}.log`.
