| verdict | count |
|---|---|
| `PASS_NOW` | 0 |
| `EMPTY_EVIDENCE` | 2 |
| `API_DRIFT` | 0 |
| `NUMERIC` | 1 |
| `THRESHOLD` | 1 |
| `ASSERT` | 9 |
| `ERROR` | 0 |
| `TIMEOUT` | 0 |
| `SKIP` | 0 |

| node id | verdict | seconds | signature |
|---|---|---|---|
| `tests/test_free_surface_conversion_density_audit.py::test_actual_b_step_three_i_to_g_cells_sum_to_observed_conversion_inventory_delta` | `EMPTY_EVIDENCE` | 8.66 | AssertionError: assert 0 > 0 |
| `tests/test_free_surface_conversion_density_audit.py::test_actual_c_step_four_i_to_g_cells_report_conversion_sum_without_i_to_l_claim` | `EMPTY_EVIDENCE` | 10.23 | AssertionError: assert 0 > 0 |
| `tests/test_free_surface_inventory_reconciliation.py::test_dynamic_step_four_has_positive_pre_topology_and_negative_conversion_without_conversion_only_claim` | `THRESHOLD` | 10.5 | assert 0.0 > 0.0 |
| `tests/test_free_surface_inventory_reconciliation.py::test_forced_step_three_separates_pre_topology_combined_from_conversion` | `NUMERIC` | 8.44 | assert 0.0 != 0.0 ± 2.0e-07 |
| `tests/test_free_surface_i_to_g_exact_ledger.py::test_real_b_and_c_candidates_are_withheld_by_receiver_float32_aggregation[B_forced_conversion_deterministic-3]` | `ASSERT` | 7.09 | AssertionError: assert 'WITHHELD' == 'FAILED_DIAGNOSTIC' |
| `tests/test_free_surface_i_to_g_exact_ledger.py::test_real_b_and_c_candidates_are_withheld_by_receiver_float32_aggregation[C_dam_break_style_tiny_dynamic_topology-10]` | `ASSERT` | 8.87 | AssertionError: assert 'WITHHELD' == 'FAILED_DIAGNOSTIC' |
| `tests/test_free_surface_i_to_g_failure_policy.py::test_real_b_c_explicit_fallback_continues_legacy_but_remains_not_physical[B_forced_conversion_deterministic]` | `ASSERT` | 8.99 | tensorlbm.free_surface_topology_transaction.TopologyTransactionError: topology candidate fill is outside [-0.1, 10] |
| `tests/test_free_surface_i_to_g_failure_policy.py::test_real_b_c_explicit_fallback_continues_legacy_but_remains_not_physical[C_dam_break_style_tiny_dynamic_topology]` | `ASSERT` | 7.5 | tensorlbm.free_surface_topology_transaction.TopologyTransactionError: topology candidate fill is outside [-0.1, 10] |
| `tests/test_free_surface_i_to_g_failure_policy.py::test_real_b_c_stop_reports_two_commits_then_step_three_strict_failure[B_forced_conversion_deterministic]` | `ASSERT` | 8.61 | tensorlbm.free_surface_topology_transaction.TopologyTransactionError: topology candidate fill is outside [-0.1, 10] |
| `tests/test_free_surface_i_to_g_failure_policy.py::test_real_b_c_stop_reports_two_commits_then_step_three_strict_failure[C_dam_break_style_tiny_dynamic_topology]` | `ASSERT` | 8.56 | tensorlbm.free_surface_topology_transaction.TopologyTransactionError: topology candidate fill is outside [-0.1, 10] |
| `tests/test_free_surface_i_to_g_ownership_closure.py::test_opt_in_closure_experiment_reports_b_and_c_as_failed_diagnostics` | `ASSERT` | 9.16 | AssertionError: assert 'WITHHELD' == 'FAILED_DIAGNOSTIC' |
| `tests/test_free_surface_i_to_g_ownership_closure.py::test_topology_candidate_updates_receiver_fill_but_never_transfers_populations` | `ASSERT` | 9.11 | assert tensor(0, dtype=torch.int8) == 2 |
| `tests/test_free_surface_topology_mutation_replay_contract.py::test_b_and_c_actual_builder_capture_replays_all_phases_exactly` | `ASSERT` | 9.22 | AssertionError: assert 'WITHHELD' == 'FAILED_DIAGNOSTIC' |
