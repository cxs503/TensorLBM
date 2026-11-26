# Periodic pressure-gradient Poiseuille cross-validation

The archived R1–R4 campaigns compare the generic periodic channel with
the LIFE reference method and its analytical solution. Results are historical,
not current scientific acceptance. The [formal solver](../../../src/tensorlbm/periodic_channel_dpdx.py)
remains the only production implementation.

- [Round 1](../../../docs/reports/XVAL_LBM_PERIODIC_X_CONSTANT_DPDX.md)
- [Round 3](../../../docs/reports/XVAL_LBM_PERIODIC_X_CONSTANT_DPDX_R3.md)
- [Round 4](../../../docs/reports/XVAL_LBM_PERIODIC_X_CONSTANT_DPDX_R4.md)
- [R3 measured JSON](out/xval_life_periodic_dpdx_r3.json)
- [R4 measured JSON](out/xval_life_periodic_dpdx_r4.json)

The original flags are unchanged: `python benchmarks/cross_validation/periodic_channel_dpdx/run_xval_r4.py --help`.
Report renderers read the archived measured output and write reports and figures;
use a disposable checkout to avoid overwriting frozen evidence. Full scientific
campaigns were deferred in this relocation.

[Migration record](../../../docs/architecture/migrations/periodic-poiseuille-2026-11-26.json).
