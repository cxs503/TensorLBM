# Frazil power-law rheology campaign (structure-resolved LBM shear cell)

Measures the flow curve `tau = K gdot^n` of a frazil ice slurry from a resolved
D3Q19 suspension instead of prescribing it from a closure.

* module: `src/tensorlbm/frazil_powerlaw_shear.py`
* tests: `tests/test_frazil_powerlaw_shear.py`
* report: `docs/frazil_powerlaw_shear.md` (generated, numbers read back from the artifact)
* artifact: `artifacts/frazil_powerlaw_shear/campaign.json`

## What the campaign is

Two structural families, run over the same shear-rate ladder:

| family | state | why it is in the campaign |
|---|---|---|
| `dispersed` | one rigid crystal per particle | hard-sphere state: the state Einstein / Krieger–Dougherty describe, so the closure comparison is made here; also the built-in negative control showing that a frozen microstructure is Newtonian |
| `flocculated` | fractal aggregates sized by the shear-breakup balance | the physical origin of frazil shear thinning: aggregates trap water and break up as the shear rate rises |

## Pipeline

```bash
# 1. declare the cases (deterministic, no simulation)
python3 run_campaign.py --plan

# 2. run them, sharded (one JSON per case; a crashed case loses only itself)
python3 run_campaign.py --run --shard 0 --shards 17 --threads 2

# 3. merge -> artifact + acceptance criteria evaluated inside the artifact
python3 run_campaign.py --assemble

# 4. figures, then the report (both read only the artifact)
python3 make_figures.py
python3 make_report.py            # --api also writes /tmp/report.json
```

Probe a subset with `--only 0,1,2` (comma-separated case indices).

## Notes for whoever runs this next

* `nu = 0.2` lattice units, `rho = 1`; `tau` is the plane momentum flux at the
  wall band corrected for the momentum absorbed by particles, and `gdot` is the
  slope of a linear fit to the mean velocity profile (its `R^2` is in every case
  record).
* `eta_r` is always a ratio of the *same* measurement with and without particles
  at the same wall speed, so a residual multiplicative bias of the stress
  estimator cancels.
* The bulk viscosity cannot be larger than ~0.4 in this cell: `Re_p = gdot D^2/nu`
  then stays below ~0.4 over the rate ladder. Raising it puts inertial thickening
  into the fit and it will be mistaken for shear thickening.
* Open (`d_f = 2`) aggregates jam random sequential placement at a solids
  fraction of roughly 0.10; the achieved volume fraction is reported per case and
  is what the criteria use.
* Pin `torch.set_num_threads` when running many cases in parallel: torch's
  default pool spins and can turn a 5 s case into minutes on a loaded host.