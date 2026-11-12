# square_cylinder_re200 — Re=200, blockage 1/20 (anchor `CASE-LBM-SQUARE-CYL-NORBERG`)

2D square cylinder, normal incidence (α = 0°), `Re = U·D/ν = 200`,
cross-stream blockage `B = D/H = 1/20` (H = 20 D), metrics `St = f·D/U` and
time-mean `Cd`.  Anchor tolerances: St < 8 %, Cd < 10 %.

## What is here

| file | role |
|---|---|
| `run.py` | entry point: builds the case, drives the library solver chain, writes `case_D*.json` + `series_D*.npz` + `result.json` (incl. `repro_cmd`) |
| `verify.py` | **numpy-only** independent re-derivation of Cd/St from the raw force series, re-applies the graded reference rows, exits non-zero on mismatch (no torch, runs on the shipped artifacts) |
| `reference.json` | graded reference rows (`VERIFIED_TABLE` / `WITHHELD` / `TREND`) with provenance and tolerances |
| `make_figures.py` | wake vorticity, force histories, resolution study, error-vs-tolerance ladder; writes `figure_index.json` with per-figure sha256 |
| `make_report.py` | injects every number from the run artifacts into `docs/CASE-LBM-SQUARE-CYL-NORBERG-RE200.md` and the platform report payload (zero hand-typed numbers) |

Reusable physics/analysis code lives in the library, not here:
`src/tensorlbm/square_cylinder.py` (mask with exact half-way-BB body size,
measured blockage, force normalisation, FFT + zero-crossing Strouhal,
graded reference scoring).

## Engineering chain (no hand-written collision/streaming)

`tensorlbm.solver.collide_mrt` (MRT, `tau_field` = sponge) →
`tensorlbm.solver.stream` → `boundaries.compute_obstacle_forces` (Ladd
momentum exchange, sampled post-stream / pre-bounce-back) →
`boundaries.far_field_bc_2d` (inlet + lateral free stream, zero-gradient
outlet, obstacle half-way bounce-back) with
`boundaries.make_sponge_strength` (6 D quadratic absorber at the outlet).

## Reproduce

```bash
# 2 grids at the anchor Re, 1 verification rung at Re=150
PYTHONPATH=src python benchmarks/verified/square_cylinder_re200/run.py \
  --D 24 32 --re 200 --steps 48000 64000 --blockage 20 --threads 40 \
  --out runs/m16410_square_cyl_re200/re200_D24      # one --D/--steps/--out per invocation
PYTHONPATH=src python benchmarks/verified/square_cylinder_re200/verify.py \
  --runs runs/m16410_square_cyl_re200 --json-out runs/m16410_square_cyl_re200/verify.json
pytest tests/test_square_cylinder.py -q
```

## Honest status of the reference (read this before quoting a number)

* `Re = 200` rows are **WITHHELD**: the anchor's originals (Norberg 1993
  JWEIA 49:187; Sohankar–Norberg–Davidson 1998 IJNMF 26:39 — the paper that
  actually owns the Re=200 / blockage-1/20 regime) were not obtainable in this
  mission, so `reference.json` deliberately carries `ref: null` plus a written
  reason instead of a remembered number.
* The pass/fail referee ladder therefore uses the two **verified** rungs
  (`Re=100`, `Re=150`), whose values were extracted from an arXiv full text in
  this mission (arXiv 2411.03124v4 Table 1).
* A 2D depth-averaged solver must not be scored directly against a 3D wind
  tunnel measurement, even once the experimental number is obtained.
* `TREND` rows are interpolation of the verified rungs — explicitly *not* a
  referee; they are reported, never used to gate.

Case narrative, parameter table, comparison table, convergence study and
figures: `docs/CASE-LBM-SQUARE-CYL-NORBERG-RE200.md` (generated).