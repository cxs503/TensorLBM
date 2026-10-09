# Appendage-axis closure, part 2: resolution gate, misrank root cause, fix-menu adjudication (2026-09-16)

This document folds the follow-up arc to
[appendage_closure_20260908.md](appendage_closure_20260908.md) into the repo
record: the n256 mask-only precheck that gated the resolution study, the W13-A
root-cause decomposition of the SDF-stack B1 misrank, and the W17-B/W17-A
adjudication of the ranked fix menu from W13-A. Evidence-only comments
carrying the same numbers live on PR #291 (issue comments 5691156614,
5691324779, 5691564167, 5692583089); every decisive number below was
recomputed by the controller from the run artifacts (json / npz / masks),
independently of the run scripts.

## Setting

- Same anchor as part 1: hull `full`, sail 0.4, fin 3.0, l/d 1.0, u_in 0.1;
  truth J 4.892520782406055, SDF-stack j6 4.85236592153347 (bitwise across
  campaigns).
- The SDF serving stack under study is the frozen #290 stack (ts2 arm,
  10 members, pool 406, donor strategy `sdf_near`, borrowed field).
- B1 = sail 0.3 variant (truth J 4.885692293586378, i.e. **better** than the
  anchor by 0.1396%); B2 = fin 3.5 variant (truth J 4.888656893465831).
  The W10-B misrank: the frozen SDF stack serves B1 at 5.047959784298709 —
  +3.32% above truth, ranking B1 **worse** than the anchor (wrong sign).

## n256 mask-only precheck — the sail-axis resolution study is killed by its own gate

Zero LBM spend; ladder masks at n256 (128, 128, 256) via the production
voxel builder, plus a bitwise n128 chain-anchor gate (rebuilt
(full, 0.4, 3.0, 1.0) mask == anchor_j ctl0000 `step_004000/solid_mask`,
n_solid 4954 / A_proj 165 — bitwise).

- **Sail axis KILLED**: at n256, sail 0.2 and 0.25 masks are still **bitwise
  identical** (XOR = 0). The pre-registered gate (0.15/0.2/0.25 pairwise
  distinct ⇒ study licensed) failed. Partial improvement: 0.15 differs from
  0.2/0.25 by XOR = 2 cells each (at n128 all three are one geometry);
  0.05 ≡ 0.1. Sail-voxel ladder 19/12/6/6/4/1/1/0.
- **s\*(n256) = 0.016953125** (bracket (0.016875, 0.016953125], monotone,
  9-step bisection) vs n128 s\* = 0.133 — a factor **7.845×**, far from the
  ×2 length-scaling intuition: the vanish threshold is a sub-cell
  volume/thickness quantity, not a length.
- Legal ladder classes at n256: {0.4, 0.3, 0.25 ≡ 0.2, 0.15, 0.1 ≡ 0.05,
  ≤ 0.01}. A redesigned (non-uniform) ladder is possible but is an owner
  decision.
- **Fin axis bonus**: 12 candidates → 10 real classes; {0.2, 0.1, 0.001} ≡
  bare hull. **f\* = 0.2722** (bracket (0.272168, 0.272266]); the n128 f\*
  was never explored beyond (0.1, 0.4]. Fin-voxel ladder 6512/3528/1780/688/
  180/56/20/16/4/0; A_proj ladder 697/489/409/297/249; sail axis keeps
  A_proj = 697 at every level (the normalization confound does not survive
  refinement either). Anchor n256 n_solid 36885 (× 7.445).
- **Cost extrapolation** (if a redesigned study is ever licensed): n128
  measured 20.96 s/pt (112.6 MLUPS); cells ×8 + steps ×2 (advective-time
  parity) ⇒ ×16 ⇒ 335 s/pt linear / 298 MLUPS. Sail ladder 13–19 pts =
  1.21–1.77 h; fin boundary 7–13 pts = 0.65–1.21 h, single GPU.

Controller verification: 111 pairwise XOR + s\*/f\* brackets + A_proj
ladders recomputed from `masks_sail.npz` / `masks_fin.npz`; determinism
run1/run2 four artifacts md5-identical; `--verify` / `--check-doc`
(153/153) controller-run PASS.

## W13-A — the B1 misrank is single-leg causal: LEG-S share = 1.0 exactly

Counterfactual decomposition of the anchor→B1 (+4.0309%) surrogate delta on
the frozen ts2 stack (swap one input leg at a time):

| leg | swap | j6 | delta | share |
|---|---|---|---|---|
| F | borrowed field → B1's | 4.85236592153347 | **0** | 0.0 |
| C | cond rows → B1's | 4.85236592153347 | **0** | 0.0 |
| S | SDF volume → B1's | 5.047959784298709 | +0.19559386276523938 | **1.0** |

share_sum = 1.0, residual = 0.0; the reverse S-swap returns the anchor j6
**bitwise**. The two zero legs are structural, not lucky:

- **LEG-F = 0**: both queries borrow **pool row 233** — the anchor design's
  own corpus row (donor distance 0.000000000 anchor / 0.656709381 B1 /
  6.916862871 B2, same row at every one of the six Re) — so the borrowed
  fields are bit-identical.
- **LEG-C = 0**: ts2 `param_cond` is `[log10_re, log10_u_in]` only; sail/fin
  are sliced off, so the cond blocks are bit-identical too.
- **u_in mismatch hypothesis DEAD**: donor u_in == query u_in == 0.1; a
  corpus-mean-field probe moves J by ≤ 0.0109%.

### Mechanism: 3 sail voxels → 27.7–164.4× latent amplification

The anchor→B1 geometry difference is **3 voxels** (n_solid 4954 → 4951,
sail 0.4 → 0.3 on the n128 grid): SDF rel-L2 **0.2639%**. The frozen
stage-1 SDF encoder turns that into a latent (dim-32) move of rel-L2
**7.30–43.39%** per member (|dz| L2 0.3577–2.1203) — amplification
**27.67–164.43×**. Serving with the anchor's field + cond but B1's latents
(encoder bypassed via `forward_from_latent`) reproduces the B1 member C_D
matrix **bitwise**. Net elasticity ≈ **15.28% C_D per 1% SDF**. Member
census: 8/10 members shift positive (+1.26…+9.54%), 2 negative (−1.02%,
−1.95%).

### The sail axis is quantized in voxel space — first voxel step = 78.8% of the misrank

Sweeping sail scale on the SDF stack (j6):

| sail | [0.40, 0.39] | [0.385, 0.325] | [0.32, 0.30] |
|---|---|---|---|
| n_solid | 4954 | 4952 | 4951 |
| j6 | 4.8524 | 5.006508698803701 (+3.1767%) | 5.047959784298709 (+4.0309%) |

The first voxel step alone delivers **78.8%** of the total misrank — the
surrogate jumps when the mask jumps and is flat within a voxel class. This
is the same quantization wall the n256 precheck measured geometrically
(s\*(n256) = 0.016953125), now seen from the surrogate side. The fin axis is
the smooth control: 3.1 → 4.8243 (−0.578%), 3.2 → 4.8040, 3.3 → 4.7960
(−1.161% dip), 3.4 → 4.8084, 3.5 → 4.8094 (= B2 bitwise).

### The ranked fix menu (owner decision points)

1. Damp/recalibrate the stage-1 SDF→latent gain on appendage axes.
2. Widen the corpus below sail 0.4 (real LBM rows at sail ∈ {0.3,
   0.25-class, 0.15} per the legal class lists; ≈ 21 s/pt n128).
3. Serving-side: escalate reject-guard / distance-advisory into ranking
   abstention for closed loops.
4. Structural: ts4/finer cond so the appendage axes do not ride the SDF
   channel alone (requires a full re-freeze).

W17-B adjudicated (1); W17-A adjudicates (2) below.

## W17-B — serve-time latent damping: fix proposal 1 (global form) is DEAD

Serve-time per-member latent shrinkage toward the nearest donor,
`lat' = lat_donor + λ·(lat_query − lat_donor)` with donor = SDF-near pool
row 233, λ ∈ {1.0, 0.75, 0.5, 0.25, 0.125, 0.0}. Zero training, zero repo
changes. G0: λ = 1 reproduces the W13-A GATE1 serves **bitwise, cross-GPU**
(GPU 2 vs the original GPU 4); the latent-forced path and `backend.predict`
agree bitwise; the anchor is provably untouched (donor SDF bitwise-equals
the anchor query SDF ⇒ anchor latent == donor latent, max abs 0.0); λ = 0
is a bitwise identity (all three designs collapse to the same member-cd
matrices, margin(0) ≡ 0).

| λ | 1.0 | 0.75 | 0.5 | 0.25 | 0.125 | 0.0 |
|---|---|---|---|---|---|---|
| margin_B1 | +0.195594 | +0.144872 | +0.095343 | +0.047042 | +0.023361 | 0.0 |
| margin_B2 | −0.042957 | −0.032443 | −0.021777 | −0.010963 | −0.005500 | 0.0 |

- **margin_B1(λ) is strictly monotonically increasing, no negative dip,
  λ_flip = null.** The true B1 margin is negative (−0.006828488819677), so a
  global serve-time shrinkage can never restore the correct ranking — the
  +4.03% misrank decays smoothly to the degenerate 0 at λ = 0, where ranking
  information is destroyed. The response is mildly sublinear
  (margin(λ)/margin(1) sits 1.2–4.5% below λ).
- **B2 amplitude is dampable** (sign right at every λ; overshoot vs truth
  11.12× → 8.40 → 5.64 → 2.84 → **1.42× at λ = 0.125**, linear-interp
  λ\* ≈ 0.088) — but at a cost that kills it: LOO in-support MAPE (40 fit
  rows, donor restricted to a different design_key; 14 rows with
  bitwise-twin SDFs are an exact no-op, broken out separately):
  all-40 0.3631% → 1.1775 → 2.0452 → 2.9628 → **3.4410 → 3.9333%**;
  positive-distance-26 0.4696 → 1.7225 → 3.0573 → 4.4690 → **5.2047 →
  5.9621%**. The λ that would calibrate B2 already costs ~10× in-support.

**Adjudication: proposal 1 in its global serve-time form is DEAD** —
monotone-muting only, at prohibitive in-support cost. What survives is the
narrower, model-side reading: a direction-targeted recalibration of the
stage-1 gain on appendage axes (retrained, frozen-gated), which must beat
the λ-response curve above. λ = 1 aligns in order of magnitude with the
frozen e2e LODO A-arm ts2 macro 0.4726%.

## W17-A — corpus absorption below sail 0.4: fix proposal 2

Run `/nfs/wangxi/runs/sdf_sail_absorb_20260916/` (w17a_run.py; 4 arms ×
10 seeds ts2 retrain on the frozen #290 stack; full-campaign determinism
rerun run1/run2). Arms are pure corpus increments — same splits, same
held-25, same seeds:

| arm | new rows | donor pool | fit pool |
|---|---|---|---|
| A0 control | 0 | 406 | 381 |
| A1 | +6 sail 0.3 (dsi 12, corner campaign rows) | 412 | 387 |
| A2 | +12 (adds sail 0.25 class, dsi 13) | 418 | 393 |
| A3 | +18 (adds sail 0.001 floor, dsi 14) | 424 | 399 |

New rows are forced into the fit split; val/test membership is
bit-identical across arms. Truth margin B1 vs anchor = −0.006828488819676792
(B1 is *better* by 0.1396%).

| arm | B1 APE | margin (sign) | n≤0 | held-25 med | anchor APE | B2 APE |
|---|---|---|---|---|---|---|
| A0 | 3.3213% | +0.19559 wrong | 2/10 | 0.2367% | 0.8207% | 1.6211% |
| A1 | 0.0477% | +0.03262 wrong | 3/10 | 0.3170% | 0.8538% | 2.1550% |
| **A2** | **0.0262%** | **−0.01075 CORRECT** | 6/10 | 0.4244% | **0.0540%** | 0.3346% |
| A3 | 0.2430% | −0.00203 CORRECT | 6/10 | 0.2622% | 0.3408% | 0.5350% |

- **A0 reproduces the frozen stack bitwise** — all three j6 plus the full
  cd6 vectors vs `corner_surrogate_6pt.json`, and held-25 predictions vs
  the anchor-promo npz bitwise (10/10 seeds, pred+true+idx).
- **A2 is the winner**: sign restored (1.57× amplitude overshoot), B1 APE
  0.0262% (127× better than frozen), **anchor APE collapses 0.82 →
  0.054%** — widening below sail 0.4 fixed the anchor too; B2 APE 0.335%
  (best); held-25 0.4244% passes the 2×A0 limit (0.4735). A3's floor rows
  are mild interference vs A2 — the 0.25-class rows carry the fix. G1
  nuance (honest): the absorbed arms pass via the APE clause (≤1.5%);
  the 7/10 per-seed margin clause never fires (max 6/10).
- **Mechanism**: in every absorbed arm all six per-Re B1 queries retrieve
  donor 406 — B1's own new sail-0.3 row (distance 0.0) — uniformly across
  Re; anchor stays 233 (d 0.0), B2 stays 233 (d 6.916862871). The
  frozen-donor diagnostic (absorbed weights on the frozen row-233 field)
  reproduces the primary serve (A1 4.8832256872194595 vs primary
  4.883362177576527) — the fix lives in the **weights**, not the donor
  swap. Guard honesty: B1's new donor-406 field trips the field guard
  (guard_rel_l2 0.1510 vs threshold 0.1500), recorded in eval.json.

**Adjudication: ABSORBED — 12 real LBM rows (~4 min single-GPU at 21 s/pt)
cut the misrank from +4.03% wrong-sign to −0.22% correct-sign.** A2
(fit pool 393) is the natural SDF serving pool refresh candidate, pending
owner decision.

## Synthesis — the fix menu after W17-A

The W13-A ranked fix menu is now fully adjudicated on measurement:

1. **Damp the stage-1 SDF→latent gain** — global serve-time form **DEAD**
   (W17-B: margin(λ) strictly monotone, λ_flip = null, calibration
   λ ≈ 0.1 costs ~10× in-support). The model-side direction-targeted
   recalibration reading survives but is now *unnecessary* — proposal 2
   solves the same failure for ~4 GPU-minutes of data.
2. **Widen the corpus below sail 0.4** — **CONFIRMED** (W17-A above).
   Minimal effective dose: 6 rows fix the APE (0.048%) but not the sign;
   12 rows (sail 0.3 + 0.25-class) restore the ranking and also fix the
   anchor. The dose is expressed in **voxel classes** — sail 0.3 and
   0.25-class are the two real distinct geometries between 0.4 and the
   floor at n128 (n256 precheck), so 12 rows is the complete in-grid
   solution, not a partial one.
3. **Ranking abstention** — remains available as an interim protection
   and composes with ② for whatever is still out of support.
4. **ts4 / finer cond restructure** — **unnecessary for this failure
   mode**; the misrank closed without touching the conditioning
   architecture.

The two walls that remain are physics-of-the-grid, not model, problems:
sub-voxel sail structure requires an n256 redesign (killed as designed
for the uniform ladder; distinct-class ladder is an owner decision), and
out-of-support queries (no corpus geometry within distance) still rely on
guard/abstention semantics rather than accuracy.

## Owner decision points

1. **Objective normalization** (unchanged from part 1).
2. **n256 resolution study**: killed by the precheck gate as designed for
   the uniform ladder; a **redesigned** distinct-class ladder
   ({0.4, 0.3, 0.25, 0.15} at n256) or a fin-boundary probe (f\* ≈ 0.272)
   remains available at 0.65–1.77 h single-GPU.
3. **SDF-stack B1 accuracy**: root cause complete (single-leg, encoder
   gain); fix menu ① global variant refuted by measurement (W17-B),
   **② adjudicated ABSORBED by W17-A** (12-row dose; A2 pool = 393 fit
   rows; B1 APE 0.0262%, sign correct, anchor APE 0.054%, held-25 0.424%
   within gates). **SDF serving pool refresh to the A2 corpus is pending
   owner decision**; menu ③ (abstention) remains a complement, ④ (ts4)
   unnecessary here.
4. **Ranking abstention (menu ③)** remains available as an interim
   protection and composes with any data-side fix; not yet implemented.

## Provenance

| item | path / reference |
|---|---|
| n256 precheck run | `/nfs/wangxi/runs/n256precheck_20260916/` (mask-only, zero LBM) |
| W13-A run | `/nfs/wangxi/runs/w13a_sdf_b1_20260916/` (analyze_w13a.py) |
| W17-B run | `/nfs/wangxi/runs/sdf_latent_damp_20260916/` (analyze_w17b.py; zero training) |
| W17-A run | `/nfs/wangxi/runs/sdf_sail_absorb_20260916/` |
| Part 1 | [appendage_closure_20260908.md](appendage_closure_20260908.md) |
| Evidence comments | PR #291: issue comments 5691156614 (n256), 5691324779 (W13-A), 5691564167 (W17-B), 5692583089 (W17-A) |
