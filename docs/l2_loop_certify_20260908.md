# L3 closed-loop claim certification — the phantom adjudication, productized (2026-09-08)

- Status: **the hand adjudication of the closed-loop claims is now a
  certification script**. The L3 demo ([`l2_closed_loop_20260908.md`](l2_closed_loop_20260908.md),
  PR [#286](https://github.com/cxs503/TensorLBM/pull/286)) claimed
  surrogate-driven C_D savings at Re = 200; a fresh-LBM truth scan
  adjudicated those claims by hand as phantoms. This run reproduces that
  adjudication from the raw read-only datasets, decomposes every claim per
  design axis, and adds a certified-region search mode whose output IS
  certifiable.
- Provenance: `scripts/l2_closed_loop_certify.py` imports the demo module
  (`Evaluator`, `run_search`, `truth_at_re`) and the serving walkthrough —
  zero copied search code, zero new LBM, datasets read-only. Machine
  evidence of THIS run: `/nfs/wangxi/runs/l2_loop_certify_20260908/`
  (`certify.json` — every number schema-tagged surrogate / truth / derived;
  `report.md` rendered from the json; `run*.log`).

## The story in one command

```
python scripts/l2_closed_loop_certify.py            # full: replay + certify
python scripts/l2_closed_loop_certify.py --truth-only   # label gates, no GPU
python scripts/l2_closed_loop_certify.py --verify       # re-render + gate recheck
python scripts/l2_closed_loop_certify.py --check-doc docs/l2_loop_certify_20260908.md
```

## Gates (all must be true, checked in-run and by `--verify`)

| gate | meaning | result |
|---|---|---|
| replay bit-exact | this run re-runs the demo search and reproduces the published picks/preds/claims bit-exactly | true |
| control bit-exact | the scan control point re-derives cd_proj from raw drag/mask artifacts and reproduces the corpus row 233 label 4.381679 exactly | true |
| hand adjudication reproduced | the realized improvements equal the hand-adjudicated truth.json values | true |
| chains telescope | every per-arm attribution chain telescopes exactly to the realized total | true |
| W9 controls bit-exact | the three W9 campaign controls also reproduce the corpus label exactly | true |

Determinism: two full runs produce byte-identical `certify.json` and
`report.md` (verified with `cmp` on this machine, twice).

## Leg 1 — the phantom verdicts, decomposed per axis

| arm | claimed % (surrogate) | realized % (fresh LBM) | error pp | tier |
|---|---|---|---|---|
| greedy | -2.0149 | +0.9592 | +2.9741 | CERTIFIED-REFUTED |
| lcb | -1.8755 | +1.9260 | +3.8015 | CERTIFIED-REFUTED |

Both claims are **phantoms**: the surrogate predicted savings where fresh
LBM measures regressions (surrogate APE at the picks 2.9459 % / 3.7297 %).
The per-axis attribution (exact scan chains, every link a measured pair at
Re = 200) localises the phantom:

**greedy pick** (l/d 0.919335, sail 0.645983, fin 2.915821):

| chain link | surrogate % | truth % | tier |
|---|---|---|---|
| sail+fin joint appendage move to the corner | -1.3936 | -0.1753 | CERTIFIED-CONFIRMED |
| l/d 1.0 -> pick at the pick sail/fin | -0.5107 | +1.1365 | CERTIFIED-REFUTED |

**lcb pick** (l/d 0.902405, sail 0.600059, fin 3.000000):

| chain link | surrogate % | truth % | tier |
|---|---|---|---|
| sail+fin joint appendage move to the corner (A) | -1.3936 | -0.1753 | CERTIFIED-CONFIRMED |
| l/d 1.0 -> 0.902 at the corner appendages | -0.3693 | +1.4558 | CERTIFIED-REFUTED |
| appendage move to the LCB pick at l/d 0.902 (B) | +0.0003 | +0.6399 | CERTIFIED-CONFIRMED |

The single-axis ladder (arm excursions vs the anchor, tightest covered
bracket for the sail axis):

| arm | axis | surrogate % | truth % | tier |
|---|---|---|---|---|
| greedy | l_over_d_mult | -0.5107 | +1.1365 | CERTIFIED-REFUTED (exact pair) |
| greedy | sail_scale | +0.1557 | +0.4798 | CERTIFIED-CONFIRMED (corpus bracket interpolation) |
| greedy | fin_scale | n/a | n/a | UNCERTIFIED (no covered fin contrast at sail 0.4) |
| lcb | l_over_d_mult | -0.8847 | n/a | UNCERTIFIED at matched geometry; adjacent exact pair +1.4558 |
| lcb | sail_scale | +0.1266 | +0.3902 | CERTIFIED-CONFIRMED (corpus bracket interpolation) |

Reading: **the surrogate is directionally honest on the appendage axes**
(the joint corner move really is a small saving, -0.1753 %; more sail
really does increase C_D) — the entire phantom comes from the **l/d axis**,
where the surrogate claims -0.5107 % / -0.8847 % and fresh LBM measures
+1.1365 % / +1.4558 % at the same geometries. The honest out-of-manifold
signals were firing all along: ensemble std at the picks is 2.63x / 3.01x
the anchor std (0.0811 / 0.0927 vs 0.0308) and SDF borrow distance jumps to
3.027 / 3.606 (vs 0.0 at anchors), with the cond-space guard flagging
`reject` at both picks.

### W9 corroboration — the l/d axis is refuted, not merely uncertified

The three W9 l/d-axis scans (corner geometry, sail 0.645983 / fin
2.915821) put cd_proj at Re = 200 on six levels below 1.0 via the
sanctioned quad3 (nearest 3 rows in log10-Re, degree-2 fit), plus three
exact Re = 200 rows from the l2loop scan:

| l/d | cd_proj at Re=200 | contrast vs l/d 1.0 % | method |
|---|---|---|---|
| 1.000000 | 4.373997 | — | exact |
| 0.975000 | 4.364905 | -0.2079 | quad3 |
| 0.960000 | 4.399655 | +0.5866 | exact |
| 0.960000 | 4.399506 | +0.5832 | quad3 |
| 0.950000 | 4.409059 | +0.8016 | quad3 |
| 0.925000 | 4.418863 | +1.0257 | quad3 |
| 0.919335 | 4.423708 | +1.1365 | exact |
| 0.902000 | 4.437676 | +1.4558 | exact |
| 0.902000 | 4.437571 | +1.4535 | quad3 |
| 0.900000 | 4.439964 | +1.5082 | quad3 |

Every exact row below 1.0 (3/3) and 5 of 6 quad3 levels sit ABOVE the
l/d 1.0 truth; the sole exception is the quad3-only estimate at l/d
0.975 (-0.2079 %, no exact Re = 200 row) — while both arms' actual
excursions (l/d 0.919 / 0.902) sit deep inside the refuted region. The
quad3 route itself is validated on the coinciding levels: vs the exact
rows it is off by -0.0024 % (l/d 0.902) and -0.0034 % (l/d 0.960).

## Leg 2 — certified-region search: claims that ARE certifiable

Restricted to the truth-covered region — full hull, l/d fixed 1.0, the 12
corpus designs with exact Re = 200 rows (of 94 full l/d=1.0 designs; the
rest neither carry that row nor bracket it, so quad3 cannot certify them)
— the optimiser is exact enumeration through the SAME production serving
path, and every output is certifiable:

| arm | pick design | pick (sail, fin) | pred C_D | truth C_D | APE % |
|---|---|---|---|---|---|
| greedy | 2 | (0.400000, 3.000000) | 4.376418 | 4.381679 | 0.1201 |
| lcb | 2 | (0.400000, 3.000000) | 4.376418 | 4.381679 | 0.1201 |

- Both acquisition rules pick design 2 — the row-233 basin, which is also
  the TRUE optimum of the region (truth 4.381679): the **certified
  improvement is +0.0000 %**, i.e. no in-region saving exists to claim.
- Ranking error: 0 of the truth top decile (2 designs) displaced by the
  surrogate ranking; Kendall tau 0.9394, surrogate rank of the true best 1
  (APE 0.1201 % there).

That is the honest shape of a certified closed loop on accumulated truth:
inside the covered region the surrogate ranks and predicts well enough to
find the true optimum (and says so within 0.1201 %), and the certified
claim collapses to zero — every nonzero "saving" the demo reported lived
on the uncovered l/d axis, where the truth says the opposite sign.

## What this certification cannot cover

1. Anything outside the accumulated truth: the demo search box ranges
   widely on the l/d axis while same-hull truth exists only at l/d 1.0
   (corpus) plus the scanned bridge points.
2. The l/d axis below 1.0 is REFUTED, not merely uncertified, at the
   covered geometries (W9 table above).
3. Truth BETWEEN corpus designs inside the certified region is not
   certified either — Leg 2 therefore searches the discrete covered set,
   and its 0 % certified improvement is the honest ceiling of what this
   truth can certify at Re = 200.
4. The fin axis at sail 0.4 has no covered contrast; the greedy arm fin
   excursion stays UNCERTIFIED (out-of-manifold signals attached in
   `certify.json`).
5. Single-Re (200), single-objective (C_D), in-family box, hull full; no
   new LBM was run for this certification.

## Machine checks

- `report.md` is rendered FROM `certify.json` by the script;
  `--verify` re-renders it and byte-compares, and re-checks every gate
  plus the telescoping identity of both attribution chains.
- `--check-doc docs/l2_loop_certify_20260908.md` asserts every decimal
  number in THIS document is formattable from a value in `certify.json`
  (zero hand-typed numbers).
- Two full runs produce byte-identical `certify.json` (no timestamps, no
  wall-clock fields anywhere in the json).
