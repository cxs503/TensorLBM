# Spectral Saffman–Taylor cross-validation

| Round | Protocol | Tracked results |
| --- | --- | --- |
| 1 · port fidelity | [Report](../../../docs/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACTOR.md) · [runner](round1/xval_spectral_st.py) · [NumPy reference](round1/oss_spectral_reference.py) | [Results with configuration](round1/artifacts/results.json) · [growth metrics](round1/artifacts/metrics_growth.json) |
| 2 · linear stability | [Report](../../../docs/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT-R2.md) · [runner and analytic reference](round2/xval_linear_theory.py) | [Independent comparison](round2/artifacts/linear_theory.json) |
| 3 · curl–Darcy closure | [Historical report](../../../docs/XVAL-LBM-SPECTRAL-SAFFMAN-TAYLOR-INTEGRATING-FACT-R3.md) | The report names reproduction assets that were absent from the migration baseline. |

Round 1 records port fidelity. Round 2 reports a growth-rate discrepancy against the independent reference. Full-protocol scientific acceptance is unevaluated for this migration.

Run the existing solver and linear-anchor tests from the repository root.

```sh
PYTHONPATH=src python -m pytest -q tests/test_spectral_saffman_taylor.py tests/test_spectral_saffman_taylor_linear_theory.py
python tools/benchmark_registry.py --case xval:spectral_saffman_taylor
```

Run the complete historical drivers in a disposable checkout. They write to the original round-local `artifacts/` destinations. The round 2 publishing helper requires a separate service and was preserved with its historical inputs.

[Migration ledger](../../../docs/architecture/migrations/spectral-saffman-taylor-xval-2026-11-25.json)
