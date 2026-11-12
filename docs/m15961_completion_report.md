# m15961 Completion Report — TensorLBM Validation Matrix L2/L3/L4 Maturity

## Executive Summary

Completed the validation matrix framework for TensorLBM with **9 slots / 11 cases**, achieving **L2+L3+L4 maturity**:

- **L2 (独立功能)**: All 11 cases have non-placeholder `impl_files` (real tensorlbm imports), `entry` points (run.py scripts), and `run_artifacts` (result.json with real numerical data).
- **L3 (集成调用)**: Integration test `test_cases_delegate_to_library_kernels_not_to_their_own_physics` verifies every case runner imports real `tensorlbm` code — no reimplemented physics. Module entry point `python -m benchmarks.validation_matrix list` works.
- **L4 (基准对照)**: `BENCHMARK_REF_VS_GOT.md` provides machine-generated ref-vs-got table for all 11 cases with zero hand-copy. Every case has `repro_cmd` for one-command reproduction.

## Current Status

| Metric | Count |
|--------|-------|
| Total slots | 9 |
| Total cases | 11 |
| **passed** | 6 (cavity_re100, cylinder_re100, sphere_re100, taylor_green_2d, static_droplet_laplace_ci, ship_resistance_suboff) |
| **failed** (honest) | 4 (channel_turb_re180, cylinder_re100_dfg, dam_break_ritter_ci, rayleigh_taylor_rt) |
| **pending** (WITHHELD) | 1 (ship_wave_michell) |

## Test Results

All 15 validation matrix tests pass:
```
tests/test_validation_matrix.py::test_pct_error_sign_and_value PASSED
tests/test_validation_matrix.py::test_judge_metric_gate_and_kinds PASSED
tests/test_validation_matrix.py::test_convergence_trend_span_and_monotone PASSED
tests/test_validation_matrix.py::test_per_level_gating_allows_informational_coarse_levels PASSED
tests/test_validation_matrix.py::test_declared_negative_result_is_never_upgraded PASSED
tests/test_validation_matrix.py::test_matrix_schema_covers_nine_slots PASSED
tests/test_validation_matrix.py::test_reference_provenance_gate_is_clean PASSED
tests/test_validation_matrix.py::test_every_repro_command_points_at_a_repo_file PASSED
tests/test_validation_matrix.py::test_status_files_match_fresh_evaluation PASSED
tests/test_validation_matrix.py::test_dam_break_ci_case_runs_and_is_judged PASSED
tests/test_validation_matrix.py::test_run_ledger_records_provenance_and_verifies PASSED
tests/test_validation_matrix.py::test_run_ledger_verify_passes_on_untampered_entry PASSED
tests/test_validation_matrix.py::test_benchmark_table_is_consistent_with_the_verdicts PASSED
tests/test_validation_matrix.py::test_module_entry_point_lists_the_matrix PASSED
tests/test_validation_matrix.py::test_cases_delegate_to_library_kernels_not_to_their_own_physics PASSED
```

## Key Artifacts

| Path | Description |
|------|-------------|
| `benchmarks/validation_matrix/matrix.json` | Registry: 9 slots, 11 cases, thresholds, artifacts, repro cmds |
| `benchmarks/validation_matrix/references.json` | Anchors: in-repo source file + literal token + sha256 |
| `benchmarks/validation_matrix/criteria.py` | Pure rules: error gate, convergence trend, passed/failed/pending |
| `benchmarks/validation_matrix/runner.py` | CLI: list, audit, check, run, status |
| `benchmarks/validation_matrix/BENCHMARK_REF_VS_GOT.md` | Machine-generated ref-vs-got table (zero hand-copy) |
| `benchmarks/validation_matrix/STATUS_MATRIX.md` | Generated three-level status table |
| `benchmarks/validation_matrix/figures/dam_break_ritter_ci.png` | Field + convergence figure |
| `benchmarks/{verified,pending}/*/validation_status.json` | Per-case evidence chain with sha256 |

## Honest FAILs (Root Causes Documented)

1. **channel_turb_re180**: Domain too small + insufficient statistical sample; DNS anchor not frozen in-repo → WITHHELD.
2. **cylinder_re100_dfg**: Cl amplitude error 35% (structural: half-way bounce-back on curved cylinder surface); dp error 5-11% (probe placement on digital cylinder).
3. **dam_break_ritter_ci**: Profile L2 error 9.8%→6.3% at CI tier (nx=200/400); crosses 5% gate at nx≈700. First-order HLL scheme on discontinuous front.
4. **rayleigh_taylor_rt**: SCMP pseudo-potential model has NO growth window in reachable parameter surface (10/10 cases: σ_sim ≤ 0 or NaN).

## Reproduction

```bash
# List all cases
python benchmarks/validation_matrix/runner.py list

# Audit reference provenance
python benchmarks/validation_matrix/runner.py audit

# Judge frozen evidence
python benchmarks/validation_matrix/runner.py check

# Run CI-tier cases
python benchmarks/validation_matrix/runner.py run --tier ci

# Regenerate status table
python benchmarks/validation_matrix/runner.py status --write

# Generate benchmark table
python benchmarks/validation_matrix/make_benchmark_table.py
```
