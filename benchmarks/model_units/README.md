# Model-unit benchmarks

The CI matrix discovers seven analytic solver regression units from [TOML
manifests and case drivers](lbm/) and [JSON manifests](units/). The units retain
their IDs, references and thresholds. The former `models/lbm` and
`models/units` checkout paths are accepted by the existing CLI and discovery
APIs. Their original objects are recorded in the [migration ledger](../../docs/architecture/migrations/model-unit-inputs-2026-11-23.json).

From the repository root with TensorLBM installed:

```bash
python -m tensorlbm.model_units verify
python -m tensorlbm.model_units build-and-test --unit poiseuille2d_d2q9_guo
python -m tensorlbm.models.ci_unit ci --only d2q9_bgk_shear_wave
```

The first command checks discovery and matrix coverage. The two per-unit
commands execute the corresponding analytic regression.
