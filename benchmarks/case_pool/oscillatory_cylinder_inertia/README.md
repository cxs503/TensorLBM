# Oscillatory cylinder inertia

Run from the repository root with the package dependencies installed:

```sh
python benchmarks/case_pool/oscillatory_cylinder_inertia/run.py --case quick
```

Configuration and execution are provided by the
[library runner](../../../src/tensorlbm/oscillatory_cylinder_inertia.py).
See the [reference and convention survey](../../../docs/XVAL_LBM_HORIZONTAL_SURVEY_EMPTY.md)
and [independent audit](../../../docs/board_audit/m18089/AUDIT_m17981.md) for
the historical scope and acceptance evidence. The layout migration did not
re-evaluate that evidence.

The original [protocol](historical_survey/predeclared.json), [inventory evidence](historical_survey/data/PROVENANCE.json) and [historical comparison](historical_survey/summary.json) are indexed with the case. The archived headline did not meet the 15% criterion. Use an explicit output directory to replay the survey.

```sh
mkdir -p outputs/oscillatory-inertia-survey
python benchmarks/case_pool/oscillatory_cylinder_inertia/historical_survey/survey.py --out-dir outputs/oscillatory-inertia-survey
python tools/benchmark_registry.py --case benchmarks/case_pool/oscillatory_cylinder_inertia
```
