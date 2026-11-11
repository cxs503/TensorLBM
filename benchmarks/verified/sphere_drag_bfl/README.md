# Sphere drag coefficient — BFL interpolated bounce-back (GAP-LBM-SPHERE)

Anchor for the **curved-boundary force method** gap: a sphere in a uniform stream,
drag coefficient compared with the Clift–Grace–Weber (1978) measurement
compilation.

## Method (mechanistic, no empirical closure inside the solver)

| item | choice |
|---|---|
| lattice | D3Q19 |
| collision | two-relaxation-time (TRT), magic parameter Λ = 3/16 |
| curved boundary | **Bouzidi–Firdaouss–Lallemand interpolated bounce-back** (q-based) |
| force | momentum exchange (Ladd) over the cut links: `F = Σ (f_i* + f_ī) e_i` |
| inlet | velocity inflow, incoming populations at equilibrium (non-reflecting) |
| outlet | zero-gradient on the incoming populations + quadratic sponge layer |
| lateral | free-slip specular reflection |
| box | 2.5 D upstream, 7 D downstream, ±3 D lateral (blockage ≈ 2.2 %) |

Why BFL rather than Guo extrapolation: the wall position is resolved continuously
through the fractional distance `q`, so the hydrodynamic radius equals the
prescribed one (no `0.5`-voxel bias), the stencil stays local (no field
extrapolation across the body, which is the usual source of mass drift and
pressure noise), and the momentum flux needed for the drag is available on the
same link list.

Implementation: `src/tensorlbm/sphere_drag_bfl.py` (solver, CLI entry point
`python -m tensorlbm.sphere_drag_bfl`).

## Reproduce

```bash
# single case (entry point)
PYTHONPATH=src python -m tensorlbm.sphere_drag_bfl --re 100 --diameter 12 \
    --u 0.1 --max-steps 12000 --out artifacts/sphere_drag_bfl/re100_D12.json

# full study: Re scan + grid refinement, writes results.json + per-case artifacts
PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/run.py \
    --workers 12 --threads 6 --max-steps 12000 --out-dir benchmarks/verified/sphere_drag_bfl

# figures + generated data table
PYTHONPATH=src python benchmarks/verified/sphere_drag_bfl/generate_figures.py

# validation tests (geometry invariants, solver invariants, artifact checks)
PYTHONPATH=src python -m pytest tests/test_sphere_drag_bfl_validation.py -q
```

## Outputs

| file | content |
|---|---|
| `results.json` | per-case `C_D`, reference value, error, steps, wall time, plateau history, grid refinement, acceptance flags |
| `cases/<tag>.json` | raw single-case run (force history, `tau`, Mach, mass drift, cut-link count) |
| `figures/fig1_cd_vs_re.png` | `C_D(Re)`: LBM vs CGW measurements vs Schiller–Naumann |
| `figures/fig2_grid_convergence.png` | `C_D` vs lattice diameter |
| `figures/fig3_wake_velocity.png` | `|u|` in the symmetry plane (field snapshot) |
| `../../../docs/sphere_drag_bfl_anchor.md` | generated numeric table (no hand-typed numbers) |

## Acceptance bands (pre-declared, machine-checked by the tests)

* `Re ≤ 200`: `|ΔC_D| / C_D,ref ≤ 10 %`
* `200 < Re ≤ 1000`: same 10 % band; a case that does not reach a steady
  plateau is reported as divergent (`finite = false`) — never silently dropped
* grid refinement 8 → 12 → 16 diameters at `Re = 20, 100`: successive changes
  must shrink and the finest change must be `< 5 %`

## Known limits

* `Re ≥ 500` at `D = 12` needs `τ − 1/2 = 3uD/(2Re) ≲ 2·10⁻³`; a finer lattice
  (or a regularised collision) is required there, the divergence is recorded in
  `results.json` rather than hidden.
* The lateral free-slip walls give a 2.2 % blockage; no blockage correction is
  applied (it is inside the 10 % acceptance band).
* The force is evaluated on the cut links only (no pressure-momentum correction
  for the `q > 1/2` links); this is the standard momentum-exchange estimate.