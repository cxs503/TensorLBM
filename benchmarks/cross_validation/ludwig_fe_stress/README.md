# Ludwig free-energy stress cross-validation

Case ID: `xval:ludwig_fe_stress`.

```bash
python tools/benchmark_registry.py --case xval:ludwig_fe_stress
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python benchmarks/cross_validation/ludwig_fe_stress/run_xval.py --output-dir outputs/ludwig_fe_stress
```

The case compares planar and curved interfacial tension with the analytic
symmetric free-energy model using prescribed two-dimensional fields. The
criterion is under 15% relative surface-tension error in every case. The
force discrepancy is a diagnostic. See the [original report](../../../docs/XVAL_LUDWIG_FE_STRESS.md)
and the [archived CPU replay](replays/2026-11-21.json). The case does not run
the Ludwig executable or advance the archived verdict to current acceptance.
