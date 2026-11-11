# Frazil power-law rheology — mature verification face (node `p366w3-n02`)

Measures the flow curve `tau = K gdot**n` of a frazil ice slurry from a resolved
two-phase D3Q19 shear cell, over `c = 0 … 0.30`, and judges the concentration–
viscosity curve against Einstein, Einstein–Batchelor–Green and
Krieger–Dougherty.

* module: `src/tensorlbm/frazil_powerlaw_mature.py`
* tests: `tests/test_frazil_powerlaw_mature.py`
* report: `docs/frazil_powerlaw_mature.md` (generated; every number read back
  from the artifact)
* artifact: `artifacts/frazil_powerlaw_mature/campaign.json`
* this directory: `cases.json` (plan), `case_records/*.json` (one record per case)

## What is new relative to the previous face (`p366w2-n02`)

| previous face | this face |
|---|---|
| crystals placed on the lattice with a one-*cell* clearance → jammed at `c = 0.155`, so `c = 0.20/0.30` were never realised | crystals placed in **continuous** coordinates with a clearance of `0.02 a` and wall margin `0.06 a`; `c = 0.30` is realised |
| coarse/fine were two different random microstructures → the "grid convergence" compared seed scatter | one physical microstructure **voxelised** at two resolutions; the pair differs only in discretisation |
| grain-free plane flux, then the crystal force was **subtracted** (double counting) | flux at a crystal-free plane *is* the total stress; both it and a new per-wall exchange estimator are calibrated on the exact Couette solution |
| wall-force accumulator summed both walls → identically `0.0` (unusable) | per-wall estimator added; agrees with the plane flux to 0.3 % |
| `Re_p` up to 0.36 (inertia mistaken for shear thickening) | `Re_p ≤ 2e-2`, reported per case |
| one seed per point, no interval | bootstrap over independent microstructures (95 % intervals) |
| flocs deformed affinely (each primary rotated about its own centre) | aggregates are **rigid**: primaries share the aggregate centre |

## Pipeline

```bash
python3 run_campaign.py --plan
python3 run_campaign.py --run --shard 0 --shards 10 --threads 1
python3 run_campaign.py --assemble
python3 make_figures.py
python3 make_report.py --api      # writes docs/ and posts the platform report
python3 -m pytest tests/test_frazil_powerlaw_mature.py -q
```

## Boxes and what each is for

| box | `n_xy × n_z` | crystal radius | `H/D` | purpose |
|---|---|---|---|---|
| `coarse` | 32 × 36 | 3.0 cells | 5.7 | production ladder |
| `fine` | 48 × 52 | 4.5 cells | 5.6 | grid convergence (same physical suspension) |
| `hd8` | 48 × 52 | 3.0 cells | 8.3 | finite-size scan (crystal size fixed, gap grows) |
| `hd11` | 64 × 68 | 3.0 cells | 11.0 | finite-size scan |

The gap-to-diameter ratio of `coarse` and `fine` is deliberately the same: they
are the same physical experiment at two resolutions. The finite-size scan needs
a *different* `H/D`, which is why it fixes the crystal size in cells instead and
enlarges the box.

## Notes for whoever runs this next

* `--threads 1` per shard is right: torch's default pool spins and a 32³ case
  can go from 150 s to minutes on a loaded host.
* Run with `torch.set_num_threads(1)` inside each shard and shard over cases, not
  over threads: the kernels are memory-bound at these sizes.
* The velocity profile of a frozen random microstructure is *layered* (local
  gradients vary ±25 % row to row), so a "bulk shear rate" is not a
  well-defined single number; the global least-squares slope is the robust one
  and is what the records carry.
* `particle_shear_scale` (in `ShearCellConfig`) exists to *quantify* the
  kinematic-condition sensitivity (the `fixed_scale` cases); it is not a tuning
  knob.