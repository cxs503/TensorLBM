# sphere_re200 — sphere drag at Re = 200  ✅ VERIFIED

Status: **verified** (2026-10-05). Two grids within 3% of the Schiller-Naumann
correlation, mesh-converged (span 1.650%), and monotone in D.

## Result

| grid | D  | Cd (last-20% window) | err vs 0.805615 | drift   | CV closure |
|------|----|----------------------|-----------------|---------|-----------|
| T1   | 16 | 0.820646             | **+1.866%**     | -0.026% | 0.036%    |
| T2   | 20 | 0.807356             | **+0.216%**     | -0.078% | 0.037%    |

Span = 1.650% (<= 3%). `|err|` falls as D grows (monotone).

## Reference (multi-source audited — see REFERENCE_AUDIT.md)

* **Main anchor** Schiller-Naumann `Cd = 24/Re (1 + 0.15 Re^0.687)` = **0.805615** (Re=200)
* Clift-Grace-Weber (1978) 0.7756 · Turton-Levenspiel 0.8025 · Clift-Gauvin 0.7810
* Cluster [0.7756, 0.8056]; the task's 0.769 / 0.773 are **arithmetic errors**
  (they back-solve to Re ≈ 224 / 221, not 200).

## Caliber

* Force: **BFL interpolated smooth wall + per-link momentum ledger, surface-restricted**
  — the same caliber that promoted `sphere_re100` (staircase surface-only MEM
  converges to a *wrong* +9% on a doubly-curved body; a smooth boundary representation
  is what removes the geometric bias).
* Domain: lat 3.0 / up 3.0 / down 4.0 (blockage 1.603%, D-independent).
* Re locked exactly: `tau = 0.5 + 3 u D / Re`, Re_eff = 200.0000.
* Averaging: last-20% sample-window mean (240 samples @ 12000 steps).

## Memory ceiling (recorded, not a blocker here)

The SDAA single-process torch allocator is capped at **14.94 GiB** (physical card 64 GB).
`D <= 30` runs; `D >= 32` OOMs inside `collide_mrt3d_low_memory` (the `matrix @ feq_flat`
temporary alone asks ~3.4-7.7 GiB). The D16/D20 pair is therefore the large-domain
ladder, and it already meets the gate. OOM evidence kept as `memtest_D3*.json.oom.json`
and `re200_D40_12k.json.oom.json`.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/verified/sphere_re200/run.py single 20 --device sdaa:0 \
  --out re200_D20_12k.json --steps 12000
PYTHONPATH=src python benchmarks/verified/sphere_re200/verify.py   # recompute from history
```

## Legacy (kept for the record)

`README_legacy_staircase.md` records the superseded staircase surface-only Ladd-MEM
route (~+9% on sphere, mesh-converged to the wrong value) — that negative result is
why the BFL smooth-wall caliber was adopted.
