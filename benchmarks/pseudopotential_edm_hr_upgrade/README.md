# Pseudopotential force-format upgrade: EDM forcing + high-order isotropic gradients

Campaign node **p97w3-n04**. Engine: `src/tensorlbm/pseudopotential_edm3d.py`.
Tests: `tests/test_pseudopotential_edm3d.py`. Report:
`docs/benchmarks/pseudopotential_edm_hr_upgrade.md`. Artifacts:
`artifacts/pseudopotential_edm_hr_upgrade/`.

## What is measured

| stage | what |
|---|---|
| `laplace` | static droplet, `R = 6/8/10`, force formats `F0` (production: D3Q19 gradient + velocity-shift forcing) vs `F3` (32-neighbour 6th-order isotropic gradient + Kupershtokh EDM forcing) -> `dp = 2 sigma / R` fit (`R^2`, per-radius `sigma_i` spread), spurious-current floor `max|u|` |
| `settle` | buoyant droplet settling vs the Hadamard-Rybczynski law `U = 2 drho g R^2 (1+lam) / (3 mu_v (2+3lam))`, `lam = 0.1/1/10`, `R = 6/8/10` |
| `box` | same physics in three box sizes -> measures the periodic-image hindrance coefficient `k` in `U = U_inf (1 - k phi^(1/3))` |
| `re` | body-force ladder at `R = 6`, `lam = 1` -> extrapolation `Re -> 0` |

Every stage writes one JSON per case under `runs/` (resumable); `--stage report`
aggregates into `artifacts/pseudopotential_edm_hr_upgrade/summary.json`.

## Reproduce

```bash
PYTHONPATH=src python benchmarks/pseudopotential_edm_hr_upgrade/run_bench.py --stage all --jobs 8
PYTHONPATH=src python benchmarks/pseudopotential_edm_hr_upgrade/make_report.py
PYTHONPATH=src pytest tests/test_pseudopotential_edm3d.py -q      # 38 cases
```

The report is CPU-bound (~20 min at 8 workers on a loaded node); all runs are
float64 and deterministic.

## Physics notes that cost the most time

1. **The spatial "exact difference" form is a no-op.** `sum_i W_i c_i = 0` on
   every symmetric stencil, so `sum W c [psi(x+c) - psi(x)]` is *identical* to
   the product form. Only the forcing scheme and the isotropy of the weight set
   can change the answer (both are asserted in the tests).
2. **The mechanical pressure dips negative inside the interface.** Sampling the
   "outside" pressure over cells that include the diffuse interface gives a
   *negative* Laplace pressure. `dp` must be taken core (`r < 0.6 R_eq`) minus
   far field (`r > 1.5 R_eq`).
3. **The spurious recirculation lives in the interface.** `max|u|` there is
   ~1e-1 lattice units at density ratio ~12 while the drop core stays at ~1e-3;
   the settling velocity is therefore the core-shell rigid-body drift.
4. **Periodic images hinder settling** by an `O(phi^(1/3))` amount; the
   coefficient is measured here instead of being quoted from the literature.
5. **`d3q19.W` is float32.** Casting it to float64 leaves a ~1e-8 relative
   weight error that breaks mass conservation at the 1e-8 per-collision level;
   the engine builds its lattice constants from exact rationals.