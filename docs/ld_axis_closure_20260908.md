# l/d-axis closure adjudication — Waves 9-1..9-5 (2026-09-08)

- Status: **adjudication document** for the l/d axis-closure campaign; the
  serving decision it records is **corpus stays frozen at 382 (v6qx)** — see
  [§5](#5-serving-decision). Band recalibration ([§6](#6-band-recalibration-proposal-w9-4-folded-here-for-owner-decision))
  and the gate redesign ([§8](#8-owner-decision-points)) are **owner actions**;
  this doc is the vehicle, not the adoption.
- Scope: l/d (`l_over_d_mult < 1.0`) axis closure campaign, Waves 9-1..9-5,
  2026-09-08. Question: **can the v6qx serving surrogate be made to reproduce
  the truth l/d-trend slope at Re=200 within [0.80, 1.20]?**
- Answer: **no — and the residual gap is attributable in significant part to
  the truth itself.** Five attempts (three corpus extensions, two channel
  variants) moved the primary ratio 1.541 → 1.326 against a [0.80, 1.20]
  band; the decisive finding ([§3c](#c-truth-side-voxel-quantization-w9-5-the-decisive-finding))
  is that the truth cd(l/d) at fixed Re is a voxel-quantization staircase at
  the ~0.3–0.4 % level inside a total span of ~1.5 % — not a smooth curve a
  cond model trained on 5 levels can be expected to reproduce.
- Provenance: follows the closed-loop campaign (PR
  [#286](https://github.com/cxs503/TensorLBM/pull/286)) and the serving
  baseline [serving_v6qx_20260828.md](serving_v6qx_20260828.md). Machine
  evidence lives in the run directories of [§9](#9-evidence-pointers)
  (read-only); every number below was re-derived from those JSONs at
  doc-writing time.

## TL;DR

| item | verdict |
|---|---|
| primary gate (ratio ∈ [0.80, 1.20] AND monotone) | **FAIL in all 5 attempts** (best 1.326) |
| best per-seed result | 3/10 seeds inside the ratio band (W9-5; first wave to clear more than an isolated seed) |
| closed-loop picks APE | 2.95/3.73 % (no l/d<1 data) → **0.52/0.72 %** (gap438) |
| mechanism | truth-side voxel quantization + support-hole tilt; interaction is learned implicitly from ladder data |
| serving corpus | **stays 382 (v6qx)** — gap438 degrades canon 1.762 vs 0.905 |
| band recalibration (W9-4) | proposed a2 bands ext [0.732, 1.253] / canon [0.347, 1.464] / B [0.707, 1.070] — **awaiting owner adoption** |

## 1. Pre-registered primary gate (frozen across all 5 attempts)

- **4-point corner-curve OLS.** Query `l_over_d` =
  [0.902, 0.919335, 0.96, 1.0] at Re=200, corner geometry
  (sail 0.645983, fin 2.915821), mother corner plane (V-track g10005) held
  constant; l/d enters the model only through the cond channel.
- **LINEAR cd vs LINEAR l/d OLS**; ratio = pred_slope / truth_slope.
- Truth cd frozen **V-track direct measurements**:
  [4.437675510222713, 4.423708032873091, 4.399654923909992,
  4.3739966982693375]; truth slope **−0.6384239789763445**.
- **PASS = ratio ∈ [0.80, 1.20] AND monotone non-increasing cd** over the
  4 points (ensemble mean curve).

The gate definition, query set and truth vector are bit-identical in every
wave's `slope_gate*.json` / `prereg.json`.

## 2. The five attempts

| wave | intervention | corpus | ratio | monotone | seeds in band | picks APE greedy/LCB % |
|---|---|---|---|---|---|---|
| W9-1 | 6 anchors l/d {0.902, 0.960} × 3 Re | 388 | 1.541 | no | 1/10 | 1.34/1.42 |
| W9-2 | 28-row ladder {0.90, 0.95} × 14 Re | 410 | 1.766 | no | 0/10 | 0.54/0.93 |
| W9-2 arm2 | + the 6 anchors again | 416 | 1.828 | no | 0/10 | 0.61/0.89 |
| W9-3 | + interaction col13 log10re·log10lod | 410 | 1.980 | no | 0/10 | 0.79/1.20 |
| W9-3 arm2 | centered variant | 410 | 1.708 | no | 0/10 | 0.77/0.87 |
| W9-5 | gap-fill {0.925, 0.975} × 14 Re (channel dropped) | 438 | 1.326 | no (first segment +0.0144) | 3/10 | 0.52/0.72 |

"Seeds in band" is the per-seed criterion ratio ∈ [0.80, 1.20] with negative
slope (`per_seed_pass` in each `slope_gate*.json`); monotonicity is an
ensemble-mean-curve criterion.

**Baseline (no l/d<1 data, v6qx382):** pred slope direction was measured
indirectly; the W9-3 null control showed v6qx382 chords **ALL POSITIVE
(mean +13.0)** — completely wrong direction. Every wave since has the
correct sign.

The non-monotonicity is concentrated in the first segment: W9-5's ensemble
curve rises +0.0144 from l/d 0.902 → 0.919335 before falling
(`first_segment_sign: positive` in `slope_gate_gap438.json`).

## 3. Mechanism decomposition

Three verified mechanisms, in order of discovery.

### (a) The model synthesizes the Re×l/d interaction implicitly from ladder data (W9-3 null control)

The 13-col **v6lad410 with NO interaction channel** achieves chord profile
**Spearman ρ = 0.9956 / chord-ratio 4.47** on the 28 matched-pair queries —
indistinguishable from the explicit-channel arms (**4.92**). The explicit
channel is not the carrier; **the ladder data is the necessary ingredient**.
v6qx382 (no l/d<1 data) has all-positive chords (mean +13.0, 0/14 negative,
13 inversions). Evidence: `interact_null_lad410.json`, `comparison.json`
(`chord_spearman_rho`, `chord_ratio` per arm).

### (b) Support-hole point-error tilt (W9-3)

In-window chord ratio at Re=197.9 = **0.939** (pred −0.5851 vs truth −0.6231)
— **in band**. The 4-point span-OLS failure concentrated at l/d=0.919
(**+0.8 %**) and 0.96 (**−0.87 %**, then in the unsupported 0.95 → 1.0 gap).
This motivated W9-5 (fill the 0.925/0.975 levels). Evidence:
`interact_gate_i410.json` row idx 6; point errors from
`slope_gate_i410.json` (`pred_cd_ensemble` vs frozen `truth_cd`).

### (c) Truth-side voxel quantization (W9-5, the decisive finding)

**The truth cd(l/d) at fixed Re is NOT smooth.** At Re=122.3 the scanned
labels are cd = **6.0801 / 6.0505 / 6.0360 / 5.9787** (l/d
0.90/0.925/0.95/0.975; the fifth level, 1.0, exists only at Re=200 via
V-track) → consecutive chords **−1.184 / −0.580 / −2.292**: the 0.95 point
sits **0.36 % above the linear interpolation of its neighbors** (the
0.925/0.975 midpoint).

The voxelized hull steps **n_solid −56 / −40 / −52 cells per 0.025 l/d
step** (identical at every Re — pure geometry; n_solid 5091/5035/4995/4943
at the four levels, and 4954 at 1.0 — *above* the 0.975 value): **the
shallowest chord coincides with the smallest geometric step.** cd(l/d)
follows the voxelized hull, not the continuous parameter; the axis is
quantized at the **~0.3–0.4 % level** while the whole curve spans only
**~1.5 %**.

Additionally the curve **flattens (possibly reverses) near 1.0**: raw
cd(0.975, Re=197.9) = 4.3935 vs V-track cd(1.0, Re=200) = 4.3740; with
log-log Re interpolation to exactly 200 the 0.975 value is **4.3652**,
below the 1.0 value — flipping the last-segment truth chord positive.

A smooth cond model trained on 5 levels cannot reproduce sub-quantization
structure; the residual OLS gap (1.326 vs [0.80, 1.20]) is attributable in
significant part to the staircase truth. Same mechanism family as the B1-4
geometry-axis self-similar degradation and the s\*=0.133 sail quantization
boundary.

## 4. What did improve (and what did not)

Improved:

- **picks APE at the closed-loop selections: 2.95/3.73 % (no data) →
  0.52/0.72 % (gap438)** — greedy/LCB ensemble APE at the falsified
  V-track pick planes.
- **ext trend ratio 1.0355, IN the old band [0.95, 1.05]** (v6lad410 was
  0.9137).
- **3/10 individual seeds pass the primary ratio band** — the first time
  any wave cleared more than an isolated single seed (W9-1 had 1/10; every
  wave in between 0/10).
- **chord-profile Spearman ρ 0.9956** — the shape is learned correctly.
- regression battery: test55 **0.509** / mother **0.434** / guard **ok54 +
  rev1** / slender-130 **18.64** (unchanged), B-grid **0.8897**.

NOT improved:

- **canon 1.7619** (v6qx382 = 0.905; W9-2 was 1.518) — corpus 438 is
  **worse** on canon than 382.
- **minisearch argmin stays l/d=0.9207** with claimed **−6.47 %** vs anchor
  4.381678732563599 — **the phantom reward persists**.

Control chain (all waves): row-233 replica `cd_proj`
**4.381678732563599 bit-exact in every wave** (5 independent
confirmations); corpus prefixes bit-identical (`tobytes`, every column);
harness-equivalence replays reproduce `best_epoch` exactly, test55
deviation ≤ **9.3e-5 pp**.

## 5. Serving decision

**Corpus stays frozen at 382 (v6qx).** gap438 degrades canon 1.762 vs 0.905
(canon ensemble repeat noise measured at ~0.021 — the arm effect is >10×
beyond noise). The l/d<1 region remains **honestly guarded**:
out-of-manifold std/distance signals fired correctly in the closed-loop
adjudication — V-track showed the flagged picks were exactly the falsified
ones.

## 6. Band recalibration proposal (W9-4, folded here for owner decision)

Old gate bands are physically unsatisfiable: reference v6qx382 per-seed sd
**ext 0.4205 / canon 0.8299 / B 0.3203**; seeds inside old bands
**0/10, 1/10, 2/10**; a disjoint twin pool (seeds 10–19, same
data/config) lands **outside all three current bands** (ensemble
ext 0.891 / canon 1.350 / B 0.812).

Proposed **a2 bands** (ensemble-statistic scale, k=1.96, truth-noise
quadrature):

| gate | old band | proposed a2 band |
|---|---|---|
| ext | [0.95, 1.05] | **[0.732, 1.253]** |
| canon | [0.90, 1.10] | **[0.347, 1.464]** |
| B | [0.90, 1.10] | **[0.707, 1.070]** |

σ_ens theory/empirical **0.13298 / 0.13283**. Retro flips across the 22-arm
census: **ext 15 FAIL→PASS / 0 reverse, canon 11/0, B 7/1** (only v7i391
B 1.0878 P→F). Analysis recommendation — **formal adoption is an owner
action** (this doc is the vehicle). Canon equivalence-band (δ\*=1.5) has no
discriminating power; canon statistics themselves are fragile (2-point
narrow pair) — the root fix is 3+ point scans.

## 7. Parallel lever note (cond_v6g, same day)

Lever ② (per-geometry sail_x basis gating) **LOSS**: A2 full-gate canon
**1.6431** / ext **0.9733** / B **0.6534**; A3 quad-only canon **1.5054** /
ext **1.0743** / B **1.0058** — no arm passes the triple gate. Canon drag
does not ride the quad basis (gated ≈ ungated 1.483); the channel is the
shared cols 0–7 (log10_sail + geo block move with sail=1.3) and/or
fit-capacity effects of the 9 M rows. With lever ① (normalization) already
proven a mathematical no-op and lever ③ (denser sail_scale × sail_x grid)
requiring new scans, **B/M data stays out of the corpus; 382 is the
evidence-backed terminal state.**

## 8. Owner decision points

1. Adopt the a2 bands ([§6](#6-band-recalibration-proposal-w9-4-folded-here-for-owner-decision))
   in the next battery/docs touch?
2. Redesign the l/d primary gate (e.g., ensemble-curve monotonicity + chord
   sign/magnitude profile + picks APE — the quantities that matter for
   closed-loop use) vs keep the 4-point OLS and accept quantization-limited
   precision?
3. Continue chasing OLS closure with denser l/d scans? Evidence says the
   truth is a ~0.3–0.4 % staircase within a 1.5 % span; learning it needs
   many more levels for marginal closed-loop value given picks APE is
   already 0.52/0.72 %.
4. (Carried from PR #281/#282 threads, unchanged): platform-app CI job;
   gallium scripts disposition.

## 9. Evidence pointers

| item | path |
|---|---|
| W9-1 run dir (6 anchors, corpus 388) | `/nfs/wangxi/runs/ld_anchor_20260908` |
| W9-2 run dir (ladder 410/416) | `/nfs/wangxi/runs/ld_ladder_20260908` |
| W9-3 run dir (interaction arms + null control) | `/nfs/wangxi/runs/ld_interact_20260908` |
| W9-4 run dir (band recalibration) | `/nfs/wangxi/runs/band_recal_20260908` (`proposal.json`, `retro_table.json`, `seed_census.json`) |
| W9-5 run dir (gap-fill 438) | `/nfs/wangxi/runs/ld_gap_20260908` |
| lever ② run dir (cond_v6g) | `/nfs/wangxi/runs/cond_v6g_20260908` |
| datasets | `scan_suboff_ld_{anchor,ladder,gap}_20260908` |
| closed-loop campaign | PR [#286](https://github.com/cxs503/TensorLBM/pull/286); L2/L3 story PR [#287](https://github.com/cxs503/TensorLBM/pull/287); V-track `/nfs/wangxi/runs/l2_loop_truth_20260908` |

Key JSONs cited in this doc: `slope_gate{,_lad410,_lad416,_i410,_ic410,_gap438}.json`
(ratios, monotonicity, per-seed passes), `interact_null_lad410.json` /
`comparison.json` (chord profiles, ρ, v6qx382 null), `chord_gate_gap438.json`
(five-level chords, log-log interpolation 4.3652), `scan_truth.json` in the
ladder/gap/anchor dirs (direct labels + `n_solid`), `phantom_picks*.json`
(picks APE), `minisearch_gap438.json` (phantom reward), `corpus*_verify.json`
(prefix bit-identity), `ctrl_replay.json` / `ctrl13_replay.json`
(harness equivalence), `proposal.json` / `retro_table.json` (W9-4).
