# Appendage-axis closure around the certified anchor (2026-09-08)

This document folds the appendage-arc campaigns (W10 corner probes, W11 sail
ladder + serving-absorption trial, W12 fin-down ladder) into the repo record.
Evidence-only comments carrying the same numbers live on PR #290 (issue
comments 5586589105 and 5587698225) and PR #289; every decisive number below
was recomputed by the controller from the raw campaign artifacts
(`scan_truth.json`, npz, h5 masks), independently of the run scripts.

## Setting

- Anchor design = the certified optimum of the closed-loop campaigns:
  hull `full`, sail_scale 0.4, fin_scale 3.0, l_over_d_mult 1.0, u_in 0.1.
- Objective J = mean of cd_proj over GRID6 Re
  {74.29971445684743, 110.40895136738128, 164.06707120152763,
  243.8027308408951, 362.289465705563, 538.3600770529428}.
- Anchor truth J = 4.892520782406055; SDF-stack surrogate j6 =
  4.85236592153347 (APE 0.8207%, bitwise across campaigns).
- Per-point noise floor NF = 0.016672621569568946% (170/184 replica pairs);
  J-level floor = NF / sqrt(6). Chain control cd_proj = 4.381678732563599,
  confirmed bitwise 10 times across these campaigns.
- All scans: n128 (64, 64, 128), cumulant, 4000 steps, production
  ScanExecutor chain; control point bitwise per campaign.

## W10-A — l/d probes at the corner (context)

Out-of-corpus l/d probes of the anchor geometry (full hull rows in the
corpus all sit at l/d = 1.0):

| l/d | truth J | vs anchor | reading |
|---|---|---|---|
| 1.0 (anchor) | 4.892520782406055 | — | — |
| 1.1 | 5.012378860346176 | −2.4498% (146.9x NF) | worse |
| 1.2 | 4.890610361299305 | +0.0390% (2.34x NF) | marginal |

J is non-monotone (rises 1.0 → 1.1, falls 1.1 → 1.2). The +0.039% at 1.2 is
normalization-carried: A_proj falls 165 → 153 (both 1.1 and 1.2 quantize to
the same projected area, −7.27%) and the cd ratio dips below 1 only at the
three lowest Re (0.995380 / 0.997004 / 0.999196), exceeding 1 up to 1.010099
at high Re. The frozen surrogate fails completely on this axis (APE
32.47 / 46.78%, truth ordering reversed, J slope +11.63 vs truth −0.0096) —
the axis is entirely outside its support. This is the same l/d-axis story
adjudicated at length in [ld_axis_closure_20260908.md](ld_axis_closure_20260908.md).

## W10-B — corner probes: one real improvement, one degenerate

Run `/nfs/wangxi/runs/anchor_corner_20260908/`, dataset
`scan_suboff_anchor_corner_20260908`.

| design | truth J | vs anchor | A_proj | raw Fx | verdict |
|---|---|---|---|---|---|
| B1 (sail 0.3, fin 3.0) | 4.885692293586378 | +0.13957% (8.4x NF) | 165 (same) | down at 6/6 Re (max ratio 0.9987) | **real force-mechanism improvement** |
| B2 (sail 0.4, fin 3.5) | 4.888656893465831 | +0.07898% (4.7x NF) | 165 → 189 (+14.545%) | **up at every Re** (1.1274–1.1753) | **degenerate pseudo-improvement** |

B1: n_solid 4954 → 4951; cd identical to the force ratio (1.1e-16) because
A_proj is unchanged. B2: fixed-area cd_ref is strictly worse (+12.7–17.5%);
the per-Re-ratio mean says worse (1.00513) while the grid-mean ratio says
better (0.99921) — an aggregation split that only the cd_proj objective
resolves in favour of "improvement". The frozen SDF surrogate misranks this
pair (truth order [B1, B2, anchor] vs surrogate [B2, anchor, B1],
Kendall tau-b = −1/3); all three queries were reject-guarded with borrow
distances 0.0 / 0.657 / 6.917. Honest-signals check: std inflation and
borrow distance landed exactly on the designs truth refuted.

## W11-A — sail ladder to the floor: real improvement does not saturate until the sail disappears; quantization wall

Run `/nfs/wangxi/runs/anchor_sail_20260908/`, dataset
`scan_suboff_anchor_sail_20260908` (13 points).

Decisive precheck finding: at n128 the masks for sail 0.15 / 0.20 / 0.25 are
**pairwise bitwise identical** (each equals the fin-only mask plus exactly one
sail voxel) — the task space has one geometry there. Per the pre-registered
coarsening rule the ladder scanned 0.25 (class representative) and the
fin-only floor (sail 0.001 bitwise ≡ 1e-6; every sail < ~0.133 produces this
mask). n_solid ladder 4954 / 4951 / 4950 / 4949; A_proj = 165 at every level.

| sail | truth J | step vs previous | J_ref (A = 69) |
|---|---|---|---|
| 0.4 (anchor) | 4.892520782406055 | — | 11.6995 |
| 0.3 | 4.885692293586378 | +0.1396% (8.4x NF) | 11.6832 |
| 0.25 class | 4.885013237811925 | +0.0139% (0.83x NF, sub-floor, not individually certifiable) | 11.6816 |
| 0 (fin-only floor) | **4.883356658835904** | +0.0339% (2.03x NF) | 11.6776 |

J is strictly monotone down to the floor (+0.1873% total, 11.2x NF); per-Re
force is monotone anchor > 0.3 > 0.25 > floor at 6/6 Re (floor ratio vector
[0.998236, 0.998172, 0.998116, 0.998060, 0.997997, 0.997928]). Because
A_proj is constant, **the sail axis has no normalization confound**: the
cd_proj ratio equals the force ratio to ≤ 2.2e-16, and the ranking is
identical under cd_proj and fixed-area cd_ref — the clean counterpoint to
the fin axis (W10-B B2 / W12-A). Physical picture: at n128 the sail is 0–4
voxels; sail-down means removing the last few sail voxels. Resolving
0.15 / 0.2 / 0.25 as distinct geometries requires a resolution study (n256),
not more n128 scans.

## W11-B — serving-absorption trial: NOT ABSORBED, and the misrank premise never existed on the cond path

Run `/nfs/wangxi/runs/sail_absorb_20260908/`. Design: absorb the 3 anchor
rows (sail 0.3 at GRID6 Re {74.3, 243.8, 538.4}) into the 382-row serving
corpus → 385, retrain the 10-seed cond_v6 pool. Arms: control382 (retrain,
seeds 0–9), absorb385 (retrain, seeds 0–9), control382b (382, disjoint seeds
10–19), plus the frozen production pool.

**Premise failure (decisive).** The W10-B 3.32% B1 misrank is an SDF-stack
phenomenon only: `corner_j` surrogate_leg serves b1 at 5.047959784 vs truth
4.885692293586378 (+3.32%). The cond path never misranks — frozen pool
margin −0.348% (9/10 members), control382 −0.346% (9/10), disjoint-seed
control382b −0.242% (8/10); B1 sits below the anchor everywhere.

| arm | J_anchor | J_b1 | B1 APE | margin | consensus | held-out mean APE |
|---|---|---|---|---|---|---|
| control382 | 4.887882 | 4.870971 | 0.301% | −0.346% | 9/10 | 0.383% |
| absorb385 | 4.914260 | 4.910720 | **0.512%** | **−0.072%** | 7/10 | **0.537%** |

Absorbing the anchors makes the pool more confident and less accurate on
exactly the point it was meant to fix: member spread halves 0.0651 → 0.0303
(ddof = 1) while APE degrades 0.301 → 0.512 and the margin collapses toward
zero. Nothing is corrupted — corpus integrity is bitwise (382-prefix vs the
independent ld_ladder corpus410; the 3 new labels bitwise-equal the
anchor_corner scan_truth), and all gates pass (ext 1.1359 / canon 1.2455 /
B 0.9653 / test55 0.558 ≤ 0.7 / mother 0.472 / guard 54 ok + 1 review).
Battery shifts vs the disjoint-seed replay stay inside replay dispersion
(canon +0.303 vs 0.407, groupB +0.072 vs 0.086, ext +0.133 vs 0.111 — a
marginal 1.2x exceedance of a single replay sample, well inside the arm
per-member ext dispersion 0.25–1.48).

**Verdict: do not absorb. Corpus stays 382 frozen. If B1-adjacent accuracy
matters, the fix belongs on the SDF stack, not the cond corpus.**

## W12-A — fin-down ladder at the sail floor: fin 3.0 is the cd_proj fin-axis optimum; cd_ref improves monotonically to the bare hull

Run `/nfs/wangxi/runs/anchor_fin_20260908/`, dataset
`scan_suboff_anchor_fin_20260908` (25 points; the fin-3.0 level reuses the
W11-A sfr rows verbatim — 18 points not rescanned). Cross-campaign mask pin:
a fresh rebuild of (sail 1e-6, fin 3.0) is bitwise-equal to all six W11-A
sfr stored masks; scanned levels pairwise bitwise-distinct (symmetric
differences 332 / 764 / 820 / 856). Control = 4.381678732563599 bitwise
(10th confirmation); tail drift max 1.124e-4.

Quantization: 8 candidates → 7 distinct geometries (0.1 ≡ 0.001 = bare
hull); A_proj ladder 165 / 141 / 117 / 77 / 69 x4; n_solid 4949 → 4093;
fin 1.0 already sits at A_proj 69 with 36 fin voxels remaining; 0.4 sits
8 voxels above bare (the 0.4 → bare vanishing boundary left unprobed).

| fin | truth J (cd_proj) | vs sail floor 4.883356658835904 | J_ref (A = 69) |
|---|---|---|---|
| 3.0 (= W11-A floor) | 4.883356658835904 | — | 11.677592010 |
| 2.5 | 5.052073730 | −3.4549% (207x NF) | 10.323802839 |
| 1.5 | 7.570443524 | −55.0254% | 8.448176107 |
| 1.0 | 8.094908609 | −65.7653% | 8.094908609 |
| off (bare hull) | 7.942690361 | −62.6482% | 7.942690361 |

- **Q1 YES — fin 3.0 is the cd_proj local optimum on the fin-down side.**
  Least-worse level −3.4549% = 207x NF. Honest non-monotonicity at the tail:
  1.0 → off *improves* +1.88042% (113x NF; both ends at A_proj 69, the bare
  hull has lower raw force). Combined with W10-B fin-up (B2: force-degenerate,
  Fx +15.1%, A_proj 189), fin 3.0 is the fin-axis cd_proj optimum, bounded
  on both sides.
- **Q2 YES — fixed-area cd_ref improves strictly monotonically at every
  step to the bare hull**: 11.677592010 → 7.942690361 (+31.9835% vs the
  floor = 4699x the sqrt(6) J-level NF; +32.1109% vs the anchor); per-Re
  force falls at 6/6 Re.
- Mechanism: at the floor A_proj falls to 69/165 = 0.4182 while per-Re mean
  force only falls to 0.6646 — the projected-area denominator dominates the
  fin axis, the exact inverse of the sail axis.
- Guides: cd_proj(fin 3 → 1) measures +65.77% against the corpus-derived
  40–60% band (direction confirmed, magnitude above — the sail-floor base is
  steeper); the cd_ref ~30% guide is confirmed (30.68% at fin 1.0).
- Surrogate leg (frozen SDF stack, all queries reject-guarded
  out-of-support): anchor j6 bitwise 4.85236592153347; APE 0.11–11.44%
  growing with distance out of support; Kendall tau-b 0.8667 over the
  6-design ladder with exactly one discordant pair (anchor ↔ fin-3.0 floor
  swap). The APE column is the honesty signal — the stack knows it is
  extrapolating.

## Synthesis — the appendage-box objective-design decision is evidence-complete

- **Sail axis**: no normalization confound (A_proj constant, cd_proj ratio ≡
  force ratio), monotone improvement all the way to the sail-off floor
  (+0.1873%).
- **Fin axis**: cd_proj-optimal at fin 3.0 (worse on both sides, bounded by
  quantization / degeneracy); cd_ref-optimal at the bare hull (+32%).
- Under **cd_proj** the certified optimum over the appendage box is
  **sail → 0 at fin 3.0, truth J 4.883356658835904** (the W11-A floor);
  under **cd_ref** it is the **bare hull**. The objective choice is now the
  entire design decision on these axes — no further n128 scans are needed
  to make it.
- Finer structure below sail 0.3 or between fin 0.4 and bare requires a
  resolution study (n256), not more n128 scans.

## Owner decision points

1. **Objective normalization**: cd_proj makes appendage growth a degenerate
   improvement direction (W10-B B2: +14.5% area and +12.7–17.5% force still
   "improves" J). If the mission wants force reduction, serve cd_ref or raw
   Fx; if it wants the coefficient form, accept that the optimiser chases
   projected area. Evidence for both readings is in the tables above.
2. **n256 resolution study**: licensed-or-killed by the mask-only precheck
   (sail 0.15/0.2/0.25 distinctness at n256, fin vanish boundary); no LBM
   spend before that gate.
3. **SDF-stack B1 accuracy**: the +3.32% b1 error is SDF-stack-only (W11-B);
   a root-cause decomposition of the anchor→B1 surrogate delta is in flight
   and will be reported separately.

## Provenance

| item | path / reference |
|---|---|
| W10-A run + dataset | `/nfs/wangxi/runs/anchor_ld_20260908/`, `scan_suboff_anchor_ld_20260908` |
| W10-B run + dataset | `/nfs/wangxi/runs/anchor_corner_20260908/`, `scan_suboff_anchor_corner_20260908` |
| W11-A run + dataset | `/nfs/wangxi/runs/anchor_sail_20260908/`, `scan_suboff_anchor_sail_20260908` |
| W11-B run | `/nfs/wangxi/runs/sail_absorb_20260908/` (labels reuse anchor_corner scan_truth) |
| W12-A run + dataset | `/nfs/wangxi/runs/anchor_fin_20260908/`, `scan_suboff_anchor_fin_20260908` |
| Anchor J reference | `/nfs/wangxi/runs/anchor_j_20260908/` (truth J 4.892520782406055) |
| Evidence comments | PR #290: issue comments 5586589105 (W10), 5587698225 (W11 + W12 + synthesis) |
| Companion docs | [ld_axis_closure_20260908.md](ld_axis_closure_20260908.md), [l2_loop_certify_20260908.md](l2_loop_certify_20260908.md), [l2_closed_loop_multire_20260908.md](l2_closed_loop_multire_20260908.md) |
