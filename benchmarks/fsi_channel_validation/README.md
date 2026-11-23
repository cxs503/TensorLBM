# Channel FSI validation gates

These manifests apply to the [flag-flapping](../../examples/benchmark_flag_flapping.py)
and [Turek–Hron](../../examples/benchmark_turek_hron.py) examples. Run the
evaluators on actual completed histories and retain the associated artifacts.

| Manifest | Consumer |
| --- | --- |
| [KPI baselines](config/fsi_kpi_baselines.json) | [FSI KPI evaluator](../../scripts/evaluate_fsi_kpis.py), default `--baseline` |
| [Observable-run gate](config/fsi_observable_run_gate.json) | [Benchmark gate evaluator](../../scripts/evaluate_benchmark_gate.py), explicit `--manifest` |
| [Regression gate](config/fsi_regression_gate.json) | [Benchmark gate evaluator](../../scripts/evaluate_benchmark_gate.py), explicit `--manifest` |
