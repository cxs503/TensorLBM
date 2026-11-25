# Free-energy Young–Laplace droplet calibration

This D2Q9 case compares the pressure jump of a static two-dimensional droplet with the sharp-interface Young–Laplace law and the diffuse-interface reference. The [case report](../../../docs/XVAL_LBM_FREE_ENERGY_LBM_ANALYTIC_YOUNG_LAPLACE_D.md) describes the protocol. [Predeclared inputs, references and thresholds](predeclared.json) apply to both runnable entry points.

```sh
PYTHONPATH=src python benchmarks/case_pool/run_pool.py --case free_energy_young_laplace --smoke
PYTHONPATH=src python benchmarks/case_pool/free_energy_young_laplace/run_xval.py --out outputs/young-laplace
python tools/benchmark_registry.py --case benchmarks/case_pool/free_energy_young_laplace
```

The default runner output goes to the case-local `artifacts/` directory. The [m20929 summary](evidence/m20929/summary.json) and [reference comparison](evidence/m20929/ref_vs_got.json) preserve the archived campaign. The [independent 2026-11-25 CPU smoke](evidence/2026-11-25-cpu-smoke.json) returned `FAIL` before and after the move. Its pressure-window relative standard error was 2.42%, above the 2% limit. The direct runner also reported a pre-existing mismatch in the third predeclared analytic check. Full-protocol acceptance remains unevaluated for this migration.

The [migration ledger](../../../docs/architecture/migrations/young-laplace-case-pool-2026-11-25.json) records previous paths, checksums and replay commands.
