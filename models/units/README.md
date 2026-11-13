# Model units — a model directory *is* the build and CI unit

This directory is the TensorLBM answer to the CFD-GO/TCLB convention
`models/<name>/` where each model directory is compiled and regression-tested
on its own.  Nothing here is registered in a hand-maintained list: the model
set is **discovered from the filesystem**, so

* adding a directory adds a model to CI,
* deleting a directory removes it,
* a directory that cannot be validated/instantiated fails on its own.

```
models/units/<unit_id>/
    model.json     # required, schema `tensorlbm.model_unit/v1`
    README.md      # optional human notes
```

`model.json` declares

| block | meaning |
|---|---|
| `composition` | lattice / collision / turbulence / forcing / boundaries / physics modules (validated against `tensorlbm.models.contracts.ModelComposition`) |
| `build` | dtype and `fingerprint_steps` for the per-directory build |
| `ci` | benchmark kernel, its parameters, and the acceptance gates |

## What "build" and "CI" mean here

TCLB compiles a model directory into per-model C++/OpenCL sources; the build
artefact is a binary.  TensorLBM composes kernels at runtime, so the build step
is

1. strict manifest validation (unknown keys, missing keys, out-of-range
   parameters and composition/kernel mismatches are all hard errors),
2. kernel instantiation through the declared collision operator,
3. a **determinism fingerprint** — the sha256 of the distribution field after
   `fingerprint_steps` steps, computed twice from the same directory.  Two
   different fingerprints mean the same directory does not build the same
   thing twice, and the unit fails before any CI verdict is produced.

The CI step then runs the unit's benchmark and compares it against an
**independently derived reference** (analytic solution, see each manifest's
kernel) with the tolerances the manifest declares.  Gates are never inferred
from a previous run: a missing gated metric is a failure, not a skip.

## Running it

```bash
python -m tensorlbm.models.ci_unit list  --root models/units
python -m tensorlbm.models.ci_unit build --root models/units
python -m tensorlbm.models.ci_unit all   --root models/units --out artifacts/model_units
# equivalently
python tools/model_unit_ci.py            # = make model-units
```

`all` writes `summary.json`, `SUMMARY.md`, per-unit `builds/` and `ci/` records,
and the measured profiles under `series/`.  Exit code is non-zero if any unit
fails.

## Units

| unit | composition | kernel | analytic reference |
|---|---|---|---|
| `d2q9_bgk_shear_wave` | D2Q9 BGK, periodic | `shear_wave_decay` | `exp(-nu k^2 t)`, `nu=(tau-1/2)/3` |
| `d2q9_mrt_shear_wave` | D2Q9 MRT, periodic | `shear_wave_decay` | same (MRT must not move the shear viscosity) |
| `d2q9_bgk_channel_poiseuille` | D2Q9 BGK, half-way walls, EDM force | `poiseuille_channel` | `u(y)=g/(2nu)(y+1/2)(ny-1/2-y)` |
| `d2q9_trt_channel_poiseuille` | D2Q9 TRT (magic Λ=3/16), same walls/force | `poiseuille_channel` | same (Λ=3/16 removes the wall-placement error) |