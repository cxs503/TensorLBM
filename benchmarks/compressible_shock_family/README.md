# Compressible shock family verification (W18-N07)

Verification surface for the compressible (kinetic-flux) branch of TensorLBM:
normal shocks in the Mach 2-10 range, the double Mach reflection and the
Mach-3 forward-facing step.

| file | purpose |
|---|---|
| `run.py` | simulation driver for the three suites, writes `*.json` + field `*.npz` |
| `make_figures.py` | PNG figures from the saved outputs (no recomputation) |
| `make_report.py` | builds `docs/COMPRESSIBLE_SHOCK_FAMILY_W18_N07.md` from the JSON (numbers are injected programmatically) and optionally publishes the platform report |

## Reproduce

```bash
# 1-D normal shock scan (30 cases, ~4 min)
python benchmarks/compressible_shock_family/run.py --suite normal_shock

# double Mach reflection, three grids (240x60, 480x120, 960x240; ~30 min)
python benchmarks/compressible_shock_family/run.py --suite dmr

# forward-facing step, two grids (150x50, 300x100; ~25 min)
python benchmarks/compressible_shock_family/run.py --suite ffs

# recompute metrics from stored fields (no time stepping)
python benchmarks/compressible_shock_family/run.py --suite dmr --reuse

python benchmarks/compressible_shock_family/make_figures.py
python benchmarks/compressible_shock_family/make_report.py
```

Outputs land in `artifacts/compressible_shock_family/` (JSON summaries, `.npz`
fields, PNG figures); the report text goes to `docs/`.

## Acceptance criteria

* normal shock: post-shock `rho`, `p`, `T` ratios and the measured shock speed
  within **3%** of the exact Rankine-Hugoniot relations for every Mach number and
  every grid, in both variants (`riemann` = non-circular, `jump` = prescribed).
* double Mach reflection: incident-shock angle within 2% of 60 deg, shock position
  after the full run within 2%, post-shock plateau within 3%, Mach-stem density
  jump within 3% of R-H at the measured stem speed, pressure continuity across the
  slip line within 2%.
* forward-facing step: near-normal post-shock jump within 3% of the exact normal
  shock at M = 3. The stand-off distance is reported with a grid-convergence band;
  it is *not* certified against an external reference (none available offline).

## Tests

```bash
python -m pytest tests/test_compressible_shock_family.py -q
```

30 tests: exact R-H reference values, jump-condition identities, shock-tube
inversion consistency, theta-beta-M solution, kinetic-flux/quadrature equivalence,
flux consistency, conservation, the exact solution of a uniform-flow advection
problem (which is what catches transposition bugs), analysis detectors on
synthetic data, and smoke runs of both 2-D cases.

## Limits

Perfect gas with gamma = 1.4, inviscid Euler only (no boundary layer, no
chemistry, no real-gas effects). The blocked-cell obstacle treatment leaves a
degenerate corner cell in front of the step face that must be excluded from
measurements. Grid resolutions are stated per case; the step stand-off needs
stand-off/dx >~ 3 to be trustworthy.