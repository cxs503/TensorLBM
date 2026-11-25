# Unpacked MPI slab halo verification

The [m17994 audit report](../../../docs/audit_m17994/AUDIT_REPORT.md) describes the numerical parity checks, communication accounting and timing limitations. Its [frozen protocol](../../../docs/audit_m17994/evidence/predeclared.json), [audited rerun](../../../docs/audit_m17994/evidence/audit_rerun_results.json) and [KPI table](../../../docs/audit_m17994/evidence/kpi_audit_rerun.csv) remain at their published paths.

Inspect the archived harness's argument contract and query the registered case from the repository root.

```sh
PYTHONPATH=src python docs/audit_m17994/evidence/run_xval.py --help
python tools/benchmark_registry.py --case xval:mpi_slab_unpacked_halo
```

Use an explicit output directory outside the evidence tree when running the harness. The standalone harness exercises local transport emulation. A native MPI production run and current scientific acceptance remain pending. The [migration record](../../../docs/architecture/migrations/mpi-and-pinning-verification-xval-2026-11-25.json) maps the three byte-identical legacy paths to the retained audit evidence.
