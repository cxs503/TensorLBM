# L3 closed-loop first cut — surrogate-driven hull design optimization (2026-09-08)

- Status: **first cut of the L3 closed loop** (owner roadmap: L3 = 服务化 / UQ /
  闭环). The serving stack of the L2 walkthrough becomes the evaluator of a
  design-search loop with honest UQ — **no new LBM anywhere in the loop**,
  every number tagged by origin (surrogate vs truth vs derived).
- Provenance: everything composed from the merged stack through the public
  API — helpers imported from
  [`scripts/l2_serving_walkthrough.py`](l2_serving_walkthrough_20260906.md)
  (PR [#283](https://github.com/cxs503/TensorLBM/pull/283)), the frozen
  pm20260831 10-member ts2 bundles, the 406-row corpus with truth for its 122
  designs. Machine evidence of THIS run (every number below comes from it):
  `/nfs/wangxi/runs/l2_closed_loop_20260908/` (`closed_loop.json`,
  `report.md` rendered from the json by the script, `driver.py` which runs the
  script by import, `driver.log`).

## The story in one command

```
python scripts/l2_closed_loop_demo.py            # 5090 defaults
python scripts/l2_closed_loop_demo.py --device cpu --out /tmp/cl.json
```

Mission: **minimise C_D** at ONE fixed condition — hull `full`, Re = 200.0,
u_in = 0.1 — over the in-family design variables `l_over_d_mult` ∈ [0.75,
1.30], `sail_scale`/`fin_scale` ∈ [0.4, 3.0] (the corpus LHS envelope;
nose/stern/sail_x pinned at 1.0). Re = 200 is the log10 mid-band of the
corpus window [50, 800] and its densest design coverage (15 rows / 14
designs share u_in = 0.1 there), so a fixed-(Re, u_in) mission is well-posed
against existing truth.

One surrogate query = the full production path, nothing shortcut:

| step | what happens |
|---|---|
| 1. CAD | `SuboffConfig` + `build_suboff_mask` at the production grid (CPU) |
| 2. SDF | `geom_encoder.sdf_volume` on the query device — bit-identical to the STL export/re-ingest SDF of the same mask (max abs diff 0.0, probe 2026-09-08) at 0.23 s vs 2.3 s |
| 3. borrow | `DragSurrogateService.predict(..., sdf=..., field_policy="field_borrow")` over the FULL 406-row pool (production posture, no LODO inside the loop) |
| 4. UQ | 10-member ensemble mean / std; `uq_temperature` = 1.5 (production default, reporting only); the LCB arm uses the raw ensemble std |

Search budget (both arms share it): coarse LHS **80** candidates + per arm a
trust-region refine **6 rounds x 8** (radius 0.30 → 0.023 of the box,
identical RNG draws per arm), then **14** validation queries at the anchor
designs — **190 surrogate queries total, mean 353 ms/query, 73.4 s wall**
(corpus + service + search + validation, cuda:0, seed 20260908,
run-to-run deterministic).

## Greedy vs UQ-aware (LCB, k = 1.0) — the same budget twice

| arm | l/d | sail | fin | pred C_D | ens std | LCB | guard | borrow dist |
|---|---|---|---|---|---|---|---|---|
| greedy | 0.919 | 0.646 | 2.916 | 4.2934 | 0.0811 | 4.2123 | reject | 3.027 |
| lcb | 0.902 | 0.600 | 3.000 | 4.2995 | 0.0927 | 4.2068 | reject | 3.606 |

Both arms start refine from the same shared coarse best (l/d 0.932, sail
0.592, fin 2.756, pred C_D 4.3817, ens std 0.0466). Findings:

- **They pick different designs** (box distance 0.0480) — but the same BASIN:
  small sail, large fin, l/d nudged below 1.0. At k = 1.0 the σ penalty
  (~0.09) is about the size of the mean gain (~0.09), so LCB reshapes the
  pick without pulling it back to the certified anchor — UQ-awareness here is
  a tie-breaker, not a conservative force.
- **Which pick has better TRUE C_D is undecidable from existing truth**: both
  picks are off-anchor interpolations whose nearest truth-bearing corpus
  design is the SAME design, and the predicted gap (0.0061) is 4x below the
  evaluator reproduction error at the anchors (0.0262 mean abs). Ranking them
  needs truth at the picks, i.e. fresh LBM.

## Validation against existing truth

The 15 corpus rows at exactly Re = 200 (14 designs) anchor the loop
evaluator: **MAPE 0.443 %, max APE 1.632 %**, e.g. the corpus-best design
(full, 0.4, 3.0, 1.0): truth 4.3817 vs served 4.3764 (0.120 %). Honesty: the
anchors are in the retrieval pool and 14/15 of their rows are in the frozen
members' training fit split — this is a **reproduction / wiring check, not a
generalization estimate**; held-out evidence is the 2026-09-04 LODO e2e
campaign (0.15 % MAPE on the all-held slender design).

## Truth comparison at the optimum

- TRUE best corpus design at Re = 200: row 233 — `full`, sail 0.4, fin 3.0,
  l/d 1.0, truth C_D **4.3817**.
- **The loop found the corpus-best basin**: both picks sit at box distance
  0.178 / 0.193 from it (nearest truth-bearing design for both), i.e. the
  search correctly identified the small-sail / large-fin corner that owns the
  truth minimum.
- **The claimed optimum is uncertified**: both picks moved l/d to 0.92 / 0.90
  and claim −2.02 % / −1.88 % C_D vs the true best design — but corpus truth
  for the `full` hull covers l/d = 1.0 ONLY (the 94 full designs all sit
  there; l/d variation lives on `with_sail` designs at fin = 1). The claim is
  the surrogate interpolating across an axis with no same-hull truth.
- **The honest out-of-family signals fire**: ensemble std at the picks is
  2.6x / 3.0x the std at the true-best anchor (0.081 / 0.093 vs 0.031) and
  SDF retrieval distance jumps to 3.03 / 3.61 (vs 0.0 at anchors). The
  cond-space guard says `reject` — but it also rejects the truth-bearing
  corner anchors (0.4/0.4, 0.4/3.0, 3.0/0.4), so on this box the guard flag
  alone is not the discriminator; the std ratio and borrow distance are.

## What this demo certifies — and what it does NOT

Certified (against existing truth):

1. The loop evaluator reproduces the corpus labels at every anchor design of
   the mission Re (0.443 % MAPE) — CAD → mask → SDF → borrow → ensemble
   predict is wired correctly end to end, at production serving posture.
2. The search locates the basin of the TRUE best corpus design at the
   mission Re.
3. Budget discipline: 190 queries, 73 s wall, deterministic.

NOT certified:

1. **Truth between corpus designs.** The picks land off-anchor; a
   nearest-design comparison there mixes surrogate error with real
   design-space variation. The −2 % claim could be real drag reduction or
   the surrogate being flat-wrong off-manifold — existing truth cannot
   tell, and this demo does not pretend it can.
2. **The l/d axis for the mission hull** (no same-hull truth off 1.0) — the
   exact axis both arms chose to exploit.
3. No new CFD was run; single-Re, single-objective mission; in-family box
   only; UQ-aware arm limited to one acquisition (LCB k = 1.0).

## Next steps toward a real closed loop

1. **Multi-Re mission objective** (e.g. C_D integrated over the operating Re
   band) — the corpus supports it, the loop currently freezes one Re.
2. **Acquisition over both arms** with an explicit exploration budget and a
   k-sweep — the k = 1.0 run shows σ ≈ mean-gain, so the conservative force
   needs to be sized, not assumed.
3. **Fresh-LBM validation of the chosen design** — the only way to certify
   an off-anchor optimum; the loop output is exactly the input such a run
   needs (CAD params + fixed conditions + a falsifiable claim: −2 % at
   l/d ≈ 0.9).
