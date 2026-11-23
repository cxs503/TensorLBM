# SDAA hardware verification archive

| Record | Archived evidence | Existing entry point |
| --- | --- | --- |
| Matrix multiplication timings | [JSON](matmul/bench.json) · [plot](matmul/sdaa_matmul_bench.png) | [matmul benchmark](../../../scripts/sdaa_matmul_bench.py) |
| Single-card memory profile | [table](memory/mem_table.json) · [curve](memory/mem_profile_curve.png) | [memory profiler](../../../scripts/mem_profile.py) |
| Performance baseline | [JSON](baseline/sdaa_perf_baseline.json) | [baseline tool](../../../tools/bench_sdaa_baseline.py) · [archived script](baseline/bench_sdaa_baseline_legacy.py) |
| Cross-card measurements | [three-card JSON](domain-decomposition/cross_card_3card.json) · [four-card JSON](domain-decomposition/cross_card_4card.json) | [cross-card gate](../../../scripts/cross_card_gate.py) · [probe report](../../../docs/card_probe_report.md) |
| M4 domain decomposition | [CPU harness record](domain-decomposition/m4_dd.json) | [M4 validation](../../../scripts/m4_dd_validation.py) · [method notes](../../../docs/domain_decomposition.md) |
| M2/M3 operator equivalence | [archived result](m2m3-equivalence/m2m3_equiv.json) | [validation script](m2m3-equivalence/validate_m2m3.py) |
| Obstacle parity diagnostic | Historical debug harness | [archived debug trace](diagnostics/debug_trace.py) |

Use the baseline tool in `tools/` for a new SDAA measurement. The archived
baseline source preserves its original workspace-relative paths. The M2/M3
script reads the archived result and expects to start from the repository
root with `PYTHONPATH=src`. The debug trace refers to the original m366
workspace.

The files retain the original run environment and results. New runs write
under `outputs/sdaa/hardware/`. The M4 harness record and the cross-card
measurements describe separate runs.
