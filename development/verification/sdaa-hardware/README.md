# SDAA hardware verification archive

| Record | Archived evidence | Existing entry point |
| --- | --- | --- |
| Matrix multiplication timings | [JSON](matmul/bench.json) · [plot](matmul/sdaa_matmul_bench.png) | [matmul benchmark](../../../scripts/sdaa_matmul_bench.py) |
| Single-card memory profile | [table](memory/mem_table.json) · [curve](memory/mem_profile_curve.png) | [memory profiler](../../../scripts/mem_profile.py) |
| Performance baseline | [JSON](baseline/sdaa_perf_baseline.json) | [tool](../../../tools/bench_sdaa_baseline.py) · [script](../../../scripts/sdaa/bench_sdaa_baseline.py) |
| Cross-card measurements | [three-card JSON](domain-decomposition/cross_card_3card.json) · [four-card JSON](domain-decomposition/cross_card_4card.json) | [cross-card gate](../../../scripts/cross_card_gate.py) · [probe report](../../../docs/card_probe_report.md) |
| M4 domain decomposition | [CPU harness record](domain-decomposition/m4_dd.json) | [M4 validation](../../../scripts/m4_dd_validation.py) · [method notes](../../../docs/domain_decomposition.md) |

The files retain the original run environment and results. New runs write
under `outputs/sdaa/hardware/`. The M4 harness record and the cross-card
measurements describe separate runs.
