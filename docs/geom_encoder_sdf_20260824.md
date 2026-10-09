# Drag surrogate conditioning: hand-crafted geometry channels (v3) → learned SDF latent (B4-SDF)

Date: 2026-08-24 · Module: `tensorlbm.ai.geom_encoder` (new, on top of
`tensorlbm.ai.drag_cond`) · Runs:
`/nfs/wangxi/runs/b4_sdf_20260824` (cache, sweep, analysis, figures) ·
Corpus/splits/protocol: unchanged v3 (238 pts, `/nfs/wangxi/runs/b4_v3_20260824`)
· Worktree `exp/b4-sdf` from origin/main `ca9e0c4d`.

> **Short version.** v3 conditions the FiLM-FNO on four hand-crafted
> mask-derived channels — a pure function of the SUBOFF design parameters,
> and therefore family-bound:换一个船型族 the channels are meaningless. This
> cut replaces them with a *learned* geometry latent: stored solid mask →
> exact signed distance field (clip ±8 voxels, stride-2 mean pool →
> 32×32×64) → 4-layer 3-D CNN (46 k params) → 32-dim tanh latent, trained
> **jointly** with the regressor under the byte-identical v3 protocol
> (quota sampling + force-tail aux head for every arm). Outcome, honestly:
> **the bare-fold "data floor" was an artefact of the hand encoding, and
> the learned latent erases it** — `loho::bare_hull` 3-seed 3.14 % (v3
> C_full) / power-geo 2.00 / R4 0.71 → `sdf_joint` **0.14 ± 0.08 %** over
> 5 seeds, per-point bias −0.11…+0.11 % vs v3's uniform +3.40…+3.65 %. But
> the latent **alone** pays for its generality inside the parametric
> family: random split 1.51 ± 0.22 % vs v3_ref 0.55 ± 0.13 (deficit
> concentrated in the scaled-geometry campaigns), and `loho::full` 9.53
> (fins absent from the fold's training side — structural). The union arm
> `sdf_plus_hand` is the best model produced in the whole B4 campaign on
> 3 of 4 splits (random 0.50, with_sail 0.69, **full 3.91 ± 0.86** vs v3's
> 7.86 ± 3.50 and power-geo 10.25) — redundancy is *not* harmful — yet it
> **reinstates** the bare floor (2.89): when the hand channels exist the
> model leans on them and stops extrapolating. The trained latent is
> essentially one-dimensional (PC1 = 99.9 %, the sail-scale axis,
> Spearman +0.76), which both explains the results and bounds the
> guardrail: latent NN-distance flags the scale-1-vs-scaled gap
> (ρ = +0.64 on the random test set) but is blind to b15 — a
> duplicate-geometry, shorter-run campaign whose labels differ for
> non-geometric reasons. Representation verdict for the owner: adopt
> `sdf_plus_hand` for in-family deployment, keep the `sdf_joint` latent as
> the extrapolation path and the v4 hybrid hook.

## 1. Motivation

v3's four geometry channels (`log_aproj_ratio`, `sail_frac`, `fin_frac`,
`solid_frac`) are computed from the SUBOFF CAD predicates — they transfer
to no other hull family, and they collapse exactly where v3 hurt most:
at the bare corner every channel is within ~0.003 of the sail = 0.4
anchors, so a regressor conditioned on them cannot distinguish "smallest
sail in training" from "no sail", and inherits the anchors' +2.3–2.5 % C_D
offset as a uniform bias (v3 finding 4, the "data-coverage floor").

The deployment target (owner decision) is arbitrary underwater-vehicle
geometry → real-time performance. The input a user supplies is a shape,
not design parameters: the natural representation is mask → SDF → encoder
→ latent, trained end-to-end with the drag regressor. The questions this
cut answers on the existing 238-point corpus:

1. **Does a learned representation break the hand-feature bare-fold
   floor** (v3: bare 2.37–3.14 vs power-geo 2.00), or is the floor a
   property of the data (anchor C_D offset) that no representation can
   avoid?
2. Is the learned latent non-inferior to the hand channels where the hand
   channels are constructively optimal (random / geo-scaled slices)?
3. Does the latent space have the structure an extrapolation guardrail
   needs (hull identity / scale axes, anchor-bare distance vs bias)?

## 2. Representation and encoder (`tensorlbm.ai.geom_encoder`)

- **SDF** — exact signed distance in **voxel units**, inside negative,
  outside positive, `scipy.ndimage`-equivalent discretisation
  `phi = edt(~mask) − edt(mask)`. Neither production venv ships scipy, so
  the EDT is computed exactly in torch: the nearest opposite-phase voxel
  always lies on the 26-connected phase boundary, so an int32
  squared-distance brute force over boundary voxels reproduces the
  full-set minimum **exactly** — order-independent, bitwise
  device-independent (pinned against a numpy brute-force oracle and an
  analytic single-voxel case in the tests; CPU↔CUDA bitwise equality
  included).
- **Clip + pool** — clip to ±`8` voxels, scale to [−1, 1], stride-2 mean
  pool 64×64×128 → **32×32×64**. Clip-then-pool bounds the block and
  preserves the zero level set; pooling commutes bitwise with even voxel
  shifts (test), and a 1-voxel shift provably changes the block.
- **Encoder** — 4× (stride-2 Conv3d 3³ + GELU), channels 8/16/32/32 →
  global mean pool → Linear → `tanh`: **46 288 params** (spec ceiling
  ~1 M), latent dim 32, bounded to [−1, 1].
- **Joint model** — `SDFCondFNODrag` = `SDFEncoder` + v3 `CondFNODrag`
  (cond vector = `[z-scored log/hand block | raw latent]`). The latent is
  deliberately **not** z-scored: its statistics move while it co-adapts
  with the encoder, and it is already bounded; the log/hand columns keep
  the v3 fit-statistics normalisation bit-for-bit.
- **Mask source** — the **stored** per-point `fields.h5` solid masks (the
  deployable path: user geometry → mask → SDF), including the documented
  vintage offset of re_drag/lhs40 (stored = current CAD − 28 sail − 36 fin
  voxels; bit-agreement 174/238, re-derived and cross-checked against the
  v3 audit). A current-CAD SDF control arm quantifies the offset's effect
  (§4: none). Cache: 238 points → 110 unique stored masks → 116 CAD keys,
  29 s on GPU, `cache_sdf.npz` (2.5 MB compressed).

## 3. A/B protocol

Splits, seeds, corpus, backbone, optimiser, quota sampling and aux head
are **byte-identical to v3's C_full** (split 0 / val 1; AdamW 1e-3/1e-4,
batch 32, ≤500 ep, patience 60; width 32 / 4 layers / modes 16×32;
aux λ = 0.1; `QuotaSampler`). Only the geometry block differs:

| arm | condition | sdf source | seeds |
|---|---|---|---|
| `v3_ref` | [log4 \| hand 4] = v3 C_full | — | random 0-2; folds 0-2, bare/full 0-4 |
| `sdf_joint` | [log4 \| latent 32] | stored | same |
| `sdf_only` | [latent 32] | stored | same |
| `sdf_plus_hand` | [log4 \| hand 4 \| latent 32] | stored | same |
| `sdf_joint_cad` | = `sdf_joint` | CAD rebuild | random/bare/full 0-2 |
| `sdf_joint_wide` | = `sdf_joint`, encoder base 16 (~180 k params) | stored | random/bare/full 0-2 |

**Known-answer parity** (`analysis_full.txt`): `v3_ref` reproduces the v3
C_full metrics on **all 10 split×seed cells to ≤ 0.006 pp** (random
0.66/0.61/0.37, bare 3.55/2.82/3.04, with_sail 1.76/0.72/0.68, full
4.41/11.37/12.81/5.32/5.39 → mean 7.86 ± 3.50 — v3's own 5-seed numbers
exactly). Predictions are not bitwise (max |Δ| 1.2e-2 on C_D ≈ 10; GPU 4
here vs GPU 0 in v3) — kernel-level float nondeterminism, three orders
below the reported effects.

## 4. Results

MAPE % on log10 C_D, test side. Power-geo prior: 2.00 / 5.16 / 10.25.

### 4.1 All splits (mean ± std over seeds; per-seed values in `analysis_tables.md`)

| split / arm | v3_ref | sdf_joint | sdf_only | sdf_plus_hand | sdf_joint_cad | sdf_joint_wide |
|---|---|---|---|---|---|---|
| **random** | **0.55±0.13** | 1.51±0.22 | 15.65±19.88 † | **0.50±0.12** | 1.40±0.31 | 1.12±0.29 |
| · geo-scaled | **0.62±0.11** | 2.43±0.20 | 14.79±17.71 | **0.59±0.09** | 2.32±0.45 | 1.83±0.46 |
| · scale-1 | 0.47±0.22 | **0.32±0.26** | 16.76±22.67 | 0.39±0.28 | 0.22±0.14 | 0.20±0.08 |
| **loho::bare_hull** | 3.06±0.26 ✗ | **0.14±0.08** ✓✓ | **0.32±0.06** ✓ | 2.89±0.26 ✗ | 0.19±0.13 | 0.16±0.03 |
| **loho::with_sail** (5.16) | 1.05±0.50 ✓ | 1.90±0.29 ✓ | 2.02±0.43 ✓ | **0.69±0.26** ✓ | 2.19 | 1.52 |
| **loho::full** (10.25) | 7.86±3.50 (med 5.39) ✓ | 9.53±0.99 ✓ | 53.69±5.74 ✗ | **3.91±0.86** ✓ | 10.17±0.33 | 9.68±0.70 |
| · full geo-scaled | 11.79±6.15 | 15.71±1.52 | 63.74±2.00 | **4.29±1.37** | 16.71±0.52 | 15.93±1.13 |

† seed-bimodal: 43.76 / 1.57 / 1.61 (bad basin at seed 0, best_ep 23).

### 4.2 Random split per dataset (seed 0)

| arm | re_drag | lhs40 | hull_re | geo_lhs | b15 | g2 |
|---|---|---|---|---|---|---|
| v3_ref | 0.63 | 0.32 | 0.67 | 0.67 | 1.14 | 0.90 |
| sdf_joint | **0.08** | **0.15** | **0.12** | 1.68 | 7.59 | 2.51 |
| sdf_plus_hand | 0.87 | 0.89 | 0.63 | **0.66** | **0.04** | **0.50** |

### 4.3 The bare fold, per point (signed error %, seed 0; 14 held bare points)

| re | v3_ref | sdf_joint | sdf_only | plus_hand | v3:C_full | v3:R4 |
|---|---|---|---|---|---|---|
| 60.0 | +3.40 | +0.10 | +0.22 | +2.64 | +3.40 | −1.27 |
| 109.9 | +3.50 | −0.06 | +0.01 | +2.45 | +3.50 | −0.60 |
| 195.4 | +3.56 | −0.10 | −0.24 | +2.14 | +3.56 | −0.58 |
| 370.4 | +3.60 | −0.10 | −0.34 | +2.54 | +3.60 | −0.20 |
| 703.5 | +3.64 | +0.04 | −0.73 | +2.73 | +3.64 | −0.08 |
| *(all 14)* | +3.40…+3.65 | **−0.11…+0.11** | −0.73…+0.27 | +2.14…+2.73 | +3.40…+3.65 | −1.27…−0.08 |
| mean ± std | +3.55 ± 0.07 | **−0.04 ± 0.08** | −0.18 ± 0.32 | +2.49 ± 0.21 | +3.55 ± 0.07 | −0.50 ± 0.31 |

(Full 14-row table with all controls in `analysis_full.txt`; CAD and wide
controls read +0.11/+0.12 mean, i.e. the breakthrough is not a
mask-vintage or encoder-capacity artefact, and it holds on all 5 seeds.)

## 5. Findings

1. **The bare-fold floor is representation-bound, not data-bound — the
   learned latent erases it.** v3 attributed its uniform +2.3–3.5 % bare
   residual to a data-coverage floor: the nearest training geometries
   (G2 sail = 0.4 anchors, 13 voxels from bare) carry C_D 2.3–2.5 % above
   the bare curve, so "a geometry-conditioned regressor interpolating to
   the bare corner inherits exactly that offset". `sdf_joint` refutes the
   inevitability: conditioning on the SDF latent, the same corpus yields
   **0.14 ± 0.08 %** with per-point bias spanning ±0.11 % — the anchor
   offset is not inherited, it is *extrapolated through*. Mechanism: the
   hand channels are ~degenerate at the corner (sail_frac ≈ 0.003 at the
   anchor vs 0 at bare, log_aproj_ratio differing in the 3rd decimal), so
   the hand model pins bare ≈ anchor; the SDF *sees* the 13-voxel sail
   bump shrink continuously along the scale axis and fits the C_D(sail)
   response below the smallest training sail. The 5-seed stability
   (0.06–0.28), the CAD-SDF control (0.19) and the wide encoder (0.16)
   agree. This is the strongest single result of the cut and answers the
   core science question with a clean yes.
2. **Inside the parametric family the latent alone is inferior to the
   hand channels — by construction of the corpus.** Random 1.51 ± 0.22 vs
   0.55 ± 0.13 (acceptance line "non-inferior within seed std": **not
   met**, gap ≈ 6σ). Attribution, three independent lines: (a) the hand
   channels *are* the generative parameters (log sail/fin exact to the
   digit; the latent must re-learn them from a clipped, pooled SDF);
   (b) the deficit lives entirely in the scaled campaigns — per-dataset
   the latent arm **beats** v3_ref on every scale-1 set (0.08/0.15/0.12
   vs 0.63/0.32/0.67) and loses on geo_lhs/g2/b15; (c) capacity helps but
   does not close it (wide: 1.51 → 1.12, geo-scaled 2.43 → 1.83). The
   b15 case is special: it is a *duplicate-geometry* campaign (same full
   masks at (1,1)/(3,3), but 2 500-step runs vs the corpus's 4 000 —
   labels offset for non-geometric reasons), so **no** geometry code can
   separate it; the hand arms absorb it through `x`/parameters, the
   latent arms cannot (7.59 vs 1.14; `sdf_plus_hand` 0.04 — the params
   carry it).
3. **Redundancy is not harmful — and the union arm is the best B4 model
   so far on 3 of 4 splits.** `sdf_plus_hand` ties random (0.50 vs 0.55),
   wins with_sail (0.69 vs 1.05) and takes `loho::full` to **3.91 ± 0.86
   (median 3.37)** vs v3's 7.86 ± 3.50 (bimodal 4.4–5.4 / 11.4–12.8) and
   power-geo 10.25 — first arm to clear *all three* LOHO lines with
   margin, with no bad basin across 5 seeds. But the same table shows the
   cost: on the bare fold `sdf_plus_hand` reads 2.89 — the hand channels
   reinstate the floor they cause. When both blocks are present the model
   leans on the (in-family) informative one and stops extrapolating along
   the latent. Representation choice is a per-corner trade, not a
   ranking.
4. **`sdf_only`: geometry carries surprisingly far, and fails exactly
   where the fold removes the geometry axis.** Bare 0.32 (the latent
   alone reproduces the bare C_D(Re) curve better than any v3 arm — `x`
   supplies the flow state), with_sail 2.02, but random is seed-bimodal
   (43.76 / 1.57 / 1.61) and `loho::full` collapses (53.7): the fold's
   training side contains **no fin-bearing geometry**, so the encoder
   feature direction for fins is untrained and fires arbitrarily. The
   log-parameter columns are what keep `sdf_joint` degrade gracefully
   (9.53, scale-1 slice 1.55) where `sdf_only` collapses.
5. **The latent is a 1-D sail-axis code — informative, and self-limiting
   for the guardrail.** PCA on the 238 fitted latents: PC1 = 99.9 %,
   Spearman(log sail, distance to bare centroid) = +0.76; hull centroids
   nearly coincide (bare↔full 0.002 ≈ within-hull spread 0.001) — *fins
   are nearly invisible in the latent*, which is precisely why
   `loho::full` is the SDF arms' weak fold. The bare→anchor distance is
   0.02 "rulers" (mean fit-fit pairwise distance): in latent space the
   bare hull sits *at* the anchors, and the model still extrapolates
   correctly through them — proximity does not force bias (the anchor
   mechanism is channel degeneracy, not distance). As an OOD metric,
   latent NN-distance correlates with error on the random test set
   (ρ = +0.64), but the correlation is the scale-1-vs-scaled contrast;
   within the geo-scaled slice ρ = −0.36, and b15 (geometry-identical,
   label-offset) is invisible at distance ≈ 0 while carrying 7.6 % error.
   Verdict: the *metric* works where geometry is the extrapolation axis;
   the *current 1-D latent* is too collapsed to be a deployed guardrail —
   a v4 encoder (wider latent, fin-sensitive loss or a diversity
   regulariser) should keep ≥ 8 effective dimensions.
6. **Vintage and capacity controls.** CAD-SDF vs stored-SDF on matched
   seeds: random 1.40 ± 0.31 vs 1.51 ± 0.22, bare 0.19 ± 0.13 vs
   0.14 ± 0.08, with_sail 2.19 vs 1.90, and on `loho::full` the three
   shared seeds read *identically* (10.47/9.71/10.34 both arms) — the
   28 + 36 = 64 voxels of ~4.1 k solid that differ in the 64/238 vintage
   points are immaterial to the A/B, and the stored mask remains the
   right primary input (deployable). Encoder base 8 → 16 (+130 k params):
   random −0.39 pp, bare/full unchanged — the breakthroughs are not
   capacity artefacts and the in-family gap is not a capacity problem.

## 6. Acceptance review (against the task's lines)

- "sdf_joint non-inferior to v3_ref on random / geo-scaled (within seed
  std)": **failed** — 1.51 ± 0.22 vs 0.55 ± 0.13; attributed (finding 2),
  mitigated by `sdf_plus_hand` (0.50 ± 0.12, non-inferior with margin on
  every random slice) — the honest claim is "the learned representation
  is non-inferior *in union with* the hand channels, superior at the
  extrapolation corner, inferior alone in-family".
- "bare fold question answered with per-point evidence": **met,
  decisively** (finding 1, §4.3).
- "latent structure delivered": **met** (finding 5, `latent_pca.png`,
  `latent_ood.png`, §numbers above), including the negative guardrail
  result and its cause.
- Unit tests for every representation guarantee: **met** (29 tests, both
  venvs, §8).

## 7. Anomalies & handling

| # | anomaly | handling |
|---|---|---|
| 1 | No scipy in either venv (`distance_transform_edt` unavailable). | Exact torch EDT via boundary-restricted int32 brute force; exactness proven (interior voxels never uniquely nearest) and pinned against a numpy full-set oracle + analytic case + CPU↔CUDA bitwise test. |
| 2 | First translation-consistency test failed on 1/539 pooled voxels. | Root cause: the synthetic blob's sphere reached the domain edge after the +2 shift — the discrete EDT commutes with translation only while the solid keeps its margin. Test geometry widened (w=26); the edge-conditioned nature of the guarantee is now implicit in the test. |
| 3 | `sdf_volume` initially documented as (1, D', H', W'); it returns (1, 1, D', H', W'). | Contract fixed to the (N=1, C=1) encoder layout; shape pinned in tests. |
| 4 | `sdf_only` random seed 0 landed in a bad basin (43.76 vs 1.57/1.61). | Reported as seed-bimodality (median 1.61); no per-seed selection anywhere — all means/medians over the declared seed sets. |
| 5 | `v3_ref` predictions not bitwise vs v3 products (max Δ 1.2e-2 on C_D ~ 10) despite identical seeds/protocol. | GPU 4 here vs GPU 0 in v3 (kernel float nondeterminism). Metrics parity ≤ 0.006 pp on all 10 cells — the known-answer control holds at the reported precision; documented as the reproducibility envelope. |
| 6 | b15 campaign turned out to be a 2 500-step duplicate-geometry A/B (labels offset vs the 4 000-step corpus, same masks as geo_lhs scale-1/3 points). | Identifies the sdf_joint per-dataset deficit as label-noise-on-identical-geometry, not representation error; also the guardrail blind spot (finding 5). No data changed. |
| 7 | v3-era ssh/buffering pitfalls (anomalies 5/6 there). | `nohup setsid` + `python -u` + log polling; uneventful. |

## 8. Reproducibility

```
# SDF cache (238 pts -> stored-mask + CAD SDFs, audit vs v3 mask_bit_eq)
cd /nfs/wangxi/runs/b4_sdf_20260824 && CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH=/nfs/wangxi/worktrees/b4_sdf/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python build_cache_sdf.py
# A/B sweep (92 rows; every run asserts tensorlbm.__file__ -> b4_sdf worktree)
cd /nfs/wangxi/runs/b4_sdf_20260824 && CUDA_VISIBLE_DEVICES=4 \
  PYTHONPATH=/nfs/wangxi/worktrees/b4_sdf/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python -u train_fno_sdf.py
# tables, parity check, bare per-point, latent structure + figures
cd /nfs/wangxi/runs/b4_sdf_20260824 && PYTHONPATH=/nfs/wangxi/worktrees/b4_sdf/src \
  /nfs/wangxi/venvs/tensorlbm/bin/python -u analyze_sdf.py
# unit tests (both venvs)
cd /nfs/wangxi/worktrees/b4_sdf && TMPDIR=/nfs/wangxi/tmp \
  /nfs/wangxi/venvs/tensorlbm/bin/python -m pytest tests/test_geom_encoder.py \
  --basetemp=/nfs/wangxi/tmp/pt_b4sdf        # 29 passed (incl. CUDA bitwise)
cd /nfs/wangxi/worktrees/b4_sdf && TMPDIR=/nfs/wangxi/tmp \
  /nfs/wangxi/venvs/ci-cpu/bin/python -m pytest tests/test_geom_encoder.py \
  tests/test_drag_cond.py --basetemp=/nfs/wangxi/tmp/pt_b4sdf  # 48 passed, 1 CUDA-skipped
```

Seeds: split 0 / val 1 / model 0-2 (random, with_sail), 0-4 (bare, full);
quota RNG `np.random.default_rng(1000 + seed)`; DoE unchanged from v3.
Gates: `ruff check` + `ruff format --check` clean;
`mypy src/tensorlbm/ai/geom_encoder.py` adds **0** errors over the 1 981
pre-existing (file itself clean). Datasets read-only. No push — work
committed on `exp/b4-sdf` only.
