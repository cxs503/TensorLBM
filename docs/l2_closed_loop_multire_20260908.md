# L3 closed-loop multi-Re — mission-profile hull C_D objective (2026-09-08)

Branch `exp/l2-loop-multire`. This is the multi-Re increment of the L3
closed loop: the merged single-Re demo (`scripts/l2_closed_loop_demo.py`,
PR #286) optimised hull C_D at ONE Reynolds number (Re = 200); ship design
cares about an operating range. The new `scripts/l2_closed_loop_multire.py`
reruns the loop against a mission-profile objective with the same two-arm
shape, the same serving stack, and the same honesty rules — no new LBM was
run for anything here; every number below is in
`/nfs/wangxi/runs/l2_loop_multire_20260908/multire.json`, tagged
surrogate / truth / derived, and `report.md` is rendered from that json.

## Objective

J(design) = mean of cd_proj over a shared 8-point log10-Re grid spanning
the intersection window of the certified set, at u_in = 0.1, hull `full`,
design variables (l_over_d_mult, sail_scale, fin_scale) in the corpus box.
A secondary 1/Re-weighted J (fuel-ish low-Re emphasis) is reported as a
sensitivity row only. quad3 (nearest 3 rows in log10-Re, degree-2,
ascending evaluation) is the only Re interpolator; nothing is ever
extrapolated outside a design's measured span.

## Step 0 — coverage map (the deliverable that shapes everything else)

- Pool: 406 rows / 122 designs; 94 of them full-hull at l/d = 1.0.
- Per-design row-count histogram, all designs: 1 row x 98, 2 x 15, 14 x 2,
  28 x 6, 82 x 1. Full l/d=1.0 only: 1 x 80, 2 x 13, 82 x 1.
- Designs with >= 4 own rows spanning >= 0.5 decades: 9 of any hull, but
  exactly 1 full-hull l/d=1.0 design — design 10 (`full`, sail 1.0,
  fin 1.0), 82 rows, Re 50.0 to 800.0 (1.204 dec).
- Intersection window of the certified full l/d=1.0 set: Re 50.0 to 800.0;
  the objective grid is therefore 50.00, 74.30, 110.41, 164.07, 243.80,
  362.29, 538.36, 800.00.

Headline: the corpus CANNOT certify a multi-Re J for any full-appendage
hull design. The single certified full design is the bare 1.0/1.0
reference hull, and its 82 rows are an LHS over (Re, u_in) — 41 distinct
u_in levels — so its quad3 curve is the Re response at the campaign's u_in
spread, not a fixed-u_in isochart (the two Re=200 replica rows differ by
0.0167 %, the truth noise floor at mid-band). The certified-region arm
therefore runs a primary enumeration (n = 1, the letter of the mission
rule) plus a disclosed extended enumeration over the 9 covered designs of
any hull, on the 6 grid points bracketed by every member's span
(74.30 to 538.36; 50.00 and 800.00 are dropped — extrapolation is banned).

## Arm A — free continuous search (LHS 40 + refine 4 x 6 per arm, k = 1.0)

| arm | l/d | sail | fin | pred J | J (1/Re-wtd) | ens std | borrow dist |
|---|---|---|---|---|---|---|---|
| greedy | 0.8880 | 1.8824 | 2.9411 | 5.6621 | 8.2907 | 0.3083 | 5.8445 |
| lcb | 0.8590 | 1.8373 | 3.0000 | 5.7560 | 8.4326 | 0.4117 | 7.5770 |

- The multi-Re objective STILL chases l/d < 1.0: both arms pick below 1.0,
  out-of-family (the corpus full hull exists only at l/d = 1.0), with
  ensemble std 6.7x to 18.4x the certified reference design's across the
  grid — the out-of-manifold signal is loud and unambiguous.
- Budget honesty: the anchor design's SERVED J is 5.2689 — better than the
  search's own optimum (the greedy pick is +7.46 % predicted-worse than the
  anchor at J level). LHS-40 + trust-region refine never reaches the
  sail = 0.4 box corner; the multi-Re free arm is budget-limited as well as
  phantom-driven, and its "optimum" is not even the surrogate's best known
  design. The sail axis also flipped character vs the single-Re run (0.646
  there, 1.88 here) — the multi-Re landscape weights the low-Re points
  where appendage drag is relatively costlier.

## Arm B — certified-region enumeration

Primary (certified full l/d=1.0, 8-point grid): design 10 only — truth J
8.7408, surrogate J 8.6131, APE 1.461 %. With n = 1, ranking statistics
are undefined; predicted optimum = true optimum trivially.

Extended (9 covered designs, any hull, 6-point shared sub-grid):

- Predicted optimum = TRUE optimum: design 100 (`with_sail`, sail 1.0,
  fin 1.0, l/d 0.75), truth J 6.0017, predicted J 6.0076, APE 0.099 %.
- Kendall tau-b = 1.0000 (perfect rank agreement); surrogate rank of the
  true best 1 of 9; top-decile (here top-1) displacement 0; enumeration
  MAPE 0.178 %, max APE 0.950 %.
- The certified l/d trend across the with_sail family is J rising with
  l/d (6.0017 at 0.75 through 11.1130 at 1.30) — for that hull family the
  surrogate ranks a SHORTER hull best and the truth agrees.

Anchor comparison (row 233 design, `full` 0.4 / 3.0 / 1.0): it has 2 own
rows (0.477 dec), so its multi-Re J is NOT certifiable and any J-level
improvement over it is surrogate-on-surrogate. The certified slices at the
anchor's own exact Re levels (truth on both sides) say the enumerated
optimum is WORSE than the anchor: +21.35 % at Re 200.0 (5.3173 vs 4.3817),
+11.02 % at Re 600.0 (2.6320 vs 2.3708). The certified set simply contains
no competitive full-appendage hull — the anchor remains the best
truth-bearing design known.

## Phantom — does the multi-Re free arm still chase the refuted l/d < 1.0?

Yes. Objective-level quantification from the W9 corner-geometry scan
datasets (read-only, sail 0.645983 / fin 2.915821, 4 l/d levels x 14 shared
Re, spans 50.35 to 689.07):

| l/d | truth J (6-pt grid) | truth cd at Re=200 | surrogate J | surrogate cd at Re=200 |
|---|---|---|---|---|
| 0.900 | 4.9551 | 4.4400 | 4.7763 | 4.3014 |
| 0.925 | 4.9321 | 4.4189 | 4.7695 | 4.2934 |
| 0.950 | 4.9213 | 4.4091 | 4.7698 | 4.2927 |
| 0.975 | 4.8737 | 4.3649 | 4.7872 | 4.3060 |
| 1.000 | not measured (g10005 is Re=200 only) | 4.3740 | 4.7969 | 4.3154 |

- Truth chords (cd per unit l/d mult, OLS over the 4 levels): -2.24 at
  Re 50.3, -0.95 at Re 197.9, -0.40 at Re 689.1 — shortening hurts MORE at
  low Re, reproducing the W9 steepening pattern from the read-only labels.
- On the shared sub-grid the truth J slope is -1.0196 per unit l/d (J
  falls as the hull lengthens toward 1.0) while the surrogate slope is
  +0.2350 (J rises) — the sign flip IS the phantom, now at mission-profile
  level, not just at Re = 200.
- Size of the phantom: the surrogate claims a -0.43 % J gain going from
  l/d 1.0 down to 0.90, while across the same levels the truth J actually
  spans 1.67 % the other way, and the truth Re=200 chord from 0.90 to 1.0
  is -0.66 cd per unit l/d (g10005 4.3740 vs 4.4400 interpolated at 0.90).
- The single-Re campaign had already refuted this at Re = 200 with fresh
  LBM: the greedy pick (l/d 0.919335) measured 4.4237 and the LCB pick
  (0.902405) measured 4.4661, both worse than the corner at l/d 1.0
  (4.3740). The multi-Re objective inherits the same phantom because the
  surrogate was never trained on full-hull l/d != 1.0.
- Scope caveat: arm A's picks sit at different appendages (sail 1.88), so
  the W9 corner data adjudicates the l/d DIRECTION at one appendage pair,
  not the pick's own J; multi-Re truth at the pick does not exist.

## What stays uncertified, and why

1. Truth between corpus designs. Arm A's picks are off-anchor
   interpolations with borrow distance 5.8445 / 7.5770 and guard-flagged
   out-of-manifold uncertainty; bounding their true J needs fresh LBM.
2. The anchor's multi-Re J (2 own rows) — hence any J-level improvement
   over it; only its two exact-Re slices are certified.
3. Multi-Re truth at the corner geometry l/d = 1.0 — g10005 is a single
   Re=200 point, so the truth side of the phantom table extrapolates the
   l/d=1.0 row by direction only, never by J.
4. The certified-set window itself rests on ONE design whose rows mix 41
   u_in levels; a fixed-u_in multi-Re certification needs either the
   re-uin campaign re-labelled per u_in slice or fresh dense ladders
   (the W9 datasets are exactly this pattern, at one appendage pair).

## Machine checks

- Determinism: two full runs produce byte-identical `multire.json` AND
  `report.md` (no timestamps or wall-clock anywhere in the json).
- `--verify`: report.md is a byte-identical re-render from the json; all
  6 gates re-check true (row 233 label bit-exact 4.3817, both W9 scan
  controls bit-exact, g10005 geometry asserts, exact-Re=200 machinery
  reproduces the corpus labels, no truth point extrapolated); aggregate
  J / APE / Kendall tau / arm-A J all recompute from stored values.
- `--check-doc`: every decimal token of this file formats from
  `multire.json` (this file is prose around the json, never a second
  source of numbers).
- ruff check + ruff format --check clean on the script (repo config).

Reproduce:

    cd /nfs/wangxi/worktrees/l2mre
    CUDA_VISIBLE_DEVICES=1 /nfs/wangxi/venvs/tensorlbm/bin/python \
        scripts/l2_closed_loop_multire.py
    ... --verify
    ... --check-doc docs/l2_closed_loop_multire_20260908.md
