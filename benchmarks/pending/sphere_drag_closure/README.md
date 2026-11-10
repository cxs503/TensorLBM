# Sphere drag closure from the Stokes limit to `Re = 50` (node P54-W3-N11)

LBM measurement of the sphere drag coefficient at `Re = 1 … 50`, the analytic
bridge that makes that range usable in a DEM code, and the DEM-coupling
verification.  Companion module: `src/tensorlbm/drag_closure.py`; report:
`docs/sphere_drag_closure_20261110.md`.

## What is measured

For every case a periodic cube holds one **voxel** sphere at rest and a
uniform body force drives the flow, with the mean velocity pinned by a drag
feed-forward controller.  Three force estimators are recorded:

| estimator | formula | enters the verdict |
|---|---|---|
| momentum balance | `F = (1 - 1/(2τ)) g_tail V_fluid` | **yes** (exact statement about the periodic cell once steady) |
| kinetic-flux control surface | `force_control_surface.py` (m7441 instrument) | no — valid only for a *uniform* ambient state, see report §5.4 |
| wall momentum exchange | `F = 2 Σ_links f_q(x_f) c_q`, single time level | no — cross-check |

The `(1 - 1/(2τ))` factor is the momentum actually injected per step by this
library's Guo source.  It was **measured** on a sphere-free box at
`τ = 0.536 / 0.68 / 2.3` (analytic value reproduced to < 1 %) before the
production runs; ignoring it is a factor 1.3 … 15 depending on `τ`.

## Files

| file | role |
|---|---|
| `prereg.md` | frozen criteria / protocol / verdict rules, plus the traps of the first attempt |
| `run.py` | production driver: `--matrix all --jobs N`, one JSON per case in `out/` |
| `merge_results.py` | confinement + resolution fits, `C_D(Re)` table, criteria verdict → `results.json` |
| `dem_verification.py` | the repository's DEM solver settling one particle, `oseen_bridge` vs `stokes` → `dem_verification.json` |
| `field_snapshot.py` | mid-plane field for the report figure → `field_snapshot.json` |
| `make_figures.py` | all report figures (numbers read from the JSONs) |
| `make_report.py` | generates `docs/sphere_drag_closure_20261110.md` |

## Reproduce

```bash
cd benchmarks/pending/sphere_drag_closure
python run.py --matrix all --jobs 24          # ~1.5 h wall on 24 cores
python merge_results.py
python dem_verification.py
python field_snapshot.py                      # optional, for the figure
python make_figures.py && python make_report.py
python -m pytest ../../../tests/test_drag_closure.py -q     # 31 tests
```

## Honest declarations

* No moving boundary, no empirical drag input inside the simulation: the
  closure only selects *which* `Re` is simulated.
* `τ ≈ 0.54` cases (`Re = 50`, D = 12/16) are BGK-marginal and flagged.
* Criterion (b) as written in the mission (pure Oseen within 5 % up to
  `Re = 50`) is physically impossible — the Oseen 5 % band closes at
  `Re = 1.24` (computed).  The bridge is what covers `1 <= Re <= 50`; both
  readings are reported in `results.json["criteria"]`.