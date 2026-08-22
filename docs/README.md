# TensorLBM Documentation

A navigable index of the documentation in this directory, grouped by topic.

---

## Core / Lattice Models & Fundamentals

| Document | Description |
|---|---|
| [software_manual.md](software_manual.md) | Complete API reference, ship & ocean engineering benchmarks, quantitative comparisons |
| [LBM_VERIFICATION_TEACHING.md](LBM_VERIFICATION_TEACHING.md) | Introductory LBM verification exercises and teaching notes |
| [d3q19-d3q27-mrt-consistency-audit.md](d3q19-d3q27-mrt-consistency-audit.md) | Consistency audit between D3Q19 and D3Q27 MRT implementations |
| [d3q27_composition_evidence_r1.md](d3q27_composition_evidence_r1.md) | Composition evidence for D3Q27 |
| [d3q27_full_composition_evidence_r1.md](d3q27_full_composition_evidence_r1.md) | Full composition evidence for D3Q27 |
| [general_capability_matrix_r1.md](general_capability_matrix_r1.md) | Cross-module capability matrix covering all core lattice models |
| [advanced_collision_capability_matrix.md](advanced_collision_capability_matrix.md) | Capability matrix for MRT / TRT / Cumulant / KBC collision operators |
| [collision_matrix_cross_validation_r1.md](collision_matrix_cross_validation_r1.md) | Cross-validation of collision operator matrix entries |
| [boundary_capability_contract_r1.md](boundary_capability_contract_r1.md) | Boundary condition capability specification and admission contract |

---

## Tutorials & Developer Guides

| Document | Description |
|---|---|
| [development_workflow.md](development_workflow.md) | Single entry-point for setup, CI checks, platform startup, and output naming |
| [differentiable_path.md](differentiable_path.md) | The differentiable reference path: eager-solver autograd contract, memory/checkpointing, relation to adjoint surrogate and Triton path |
| [platform_user_manual.md](platform_user_manual.md) | Web-platform UI guide: preprocess, solver, postprocess, jobs, AI agent |
| [platform_test_report.md](platform_test_report.md) | Platform smoke-test report |
| [observability.md](observability.md) | Job lifecycle, output schema, and failure-triage checklist |

---

## Benchmarks & Validation

| Document | Description |
|---|---|
| [REGRESSION_REPORT.md](REGRESSION_REPORT.md) | Core regression baseline report |
| [REGRESSION_REPORT_interp_bc.md](REGRESSION_REPORT_interp_bc.md) | Regression report: interpolated bounce-back boundary condition |
| [REGRESSION_REPORT_sliding_mesh.md](REGRESSION_REPORT_sliding_mesh.md) | Regression report: sliding-mesh boundary |
| [REGRESSION_REPORT_wall_multi_gpu.md](REGRESSION_REPORT_wall_multi_gpu.md) | Regression report: wall function multi-GPU |
| [AMR_REGRESSION_REPORT.md](AMR_REGRESSION_REPORT.md) | AMR regression report |
| [RANS_REGRESSION_EQUIVALENCE_REPORT.md](RANS_REGRESSION_EQUIVALENCE_REPORT.md) | RANS solver regression equivalence report |
| [accuracy_recommendation_evidence_gate.md](accuracy_recommendation_evidence_gate.md) | Fail-closed evidence gate before accuracy recommendations |
| [FORCE_METHODS_SURVEY.md](FORCE_METHODS_SURVEY.md) | Survey of force-computation methods (momentum exchange, stress integration) |
| [WALL_FUNCTION_SURVEY.md](WALL_FUNCTION_SURVEY.md) | Wall-function survey: log-law, power-law, Spalding |
| [wall-refinement-combination-gate.md](wall-refinement-combination-gate.md) | Gate check for wall + refinement combinations |
| [WAVE_BC_REGRESSION_REPORT.md](WAVE_BC_REGRESSION_REPORT.md) | Wave boundary condition regression report |
| [irregular_wave_6dof_validation.md](irregular_wave_6dof_validation.md) | Irregular wave + 6-DOF validation |
| [benchmarks/ai_fno2d/](benchmarks/ai_fno2d/) | FNO2d surrogate benchmark artifacts (loss curve, speed comparison) |

### SUBOFF Submarine Benchmarks

| Document | Description |
|---|---|
| [suboff_platform_manual.md](suboff_platform_manual.md) | SUBOFF full-appendage case: CLI / platform run steps, accuracy criteria |
| [suboff_reference_data_r1.md](suboff_reference_data_r1.md) | SUBOFF reference experimental data |
| [suboff_validation_runner.md](suboff_validation_runner.md) | SUBOFF validation runner documentation |
| [suboff_domain_convergence_study_r1.md](suboff_domain_convergence_study_r1.md) | SUBOFF domain convergence study |
| [suboff_grid_convergence_study_r1.md](suboff_grid_convergence_study_r1.md) | SUBOFF grid convergence study |
| [suboff_time_convergence_study_r1.md](suboff_time_convergence_study_r1.md) | SUBOFF time convergence study |
| [suboff_full_wet_production_window_r1.md](suboff_full_wet_production_window_r1.md) | SUBOFF full-wet production run window |
| [suboff_full_wet_runtime_r1.md](suboff_full_wet_runtime_r1.md) | SUBOFF full-wet runtime report |
| [suboff_real_state_force_r1.md](suboff_real_state_force_r1.md) | SUBOFF real-state force analysis |

---

## Advanced Topics

| Document | Description |
|---|---|
| [local_refinement_amr_capability_contract_r1.md](local_refinement_amr_capability_contract_r1.md) | AMR local-refinement capability contract |
| [turbulence_capability_contract_r1.md](turbulence_capability_contract_r1.md) | Turbulence model capability contract (LES / RANS) |
| [wall_function_capability_contract_r1.md](wall_function_capability_contract_r1.md) | Wall-function capability contract and admission rules |

### Free Surface & Multiphase

| Document | Description |
|---|---|
| [free_surface_production_evidence_r1.md](free_surface_production_evidence_r1.md) | Free-surface production evidence |
| [free_surface_i_to_g_exact_float32_ledger_feasibility_r1.md](free_surface_i_to_g_exact_float32_ledger_feasibility_r1.md) | Float32 ledger feasibility for interface-to-gas transitions |
| [free_surface_i_to_g_strict_failure_policy_r1.md](free_surface_i_to_g_strict_failure_policy_r1.md) | Strict failure policy for I→G transitions |
| [free_surface_i_to_g_strict_failure_residual_audit_r1.md](free_surface_i_to_g_strict_failure_residual_audit_r1.md) | Residual audit for failed I→G strict policy |
| [free_surface_population_policy_evidence_r1.md](free_surface_population_policy_evidence_r1.md) | Population policy evidence for free surface |
| [free_surface_population_transfer_plan_r1.md](free_surface_population_transfer_plan_r1.md) | Population transfer plan |
| [free_surface_transaction_contract_r1.md](free_surface_transaction_contract_r1.md) | Topology transaction contract |
| [free_surface_runtime_evidence_observer_r1.md](free_surface_runtime_evidence_observer_r1.md) | Runtime evidence observer |
| [phasefield_ch_validation_r1.md](phasefield_ch_validation_r1.md) | Phase-field Cahn–Hilliard validation |
| [phasefield_evolution_adapter_r1.md](phasefield_evolution_adapter_r1.md) | Phase-field evolution adapter |
| [phasefield_phase_inventory_flux_r1.md](phasefield_phase_inventory_flux_r1.md) | Phase inventory and flux accounting |
| [phasefield_static_droplet_r1.md](phasefield_static_droplet_r1.md) | Static droplet validation |
| [phasefield_stream_boundary_contract_r1.md](phasefield_stream_boundary_contract_r1.md) | Phase-field streaming boundary contract |

---

## AI / Neural Operator Integration

| Document | Description |
|---|---|
| [ai_turbulence.md](ai_turbulence.md) | End-to-end AI turbulence workflow: data generation → SQLite → MLP/FNO2d training → embedded collision |
| [model_zoo.md](model_zoo.md) | Model zoo: manifest-driven registry of trained artifacts (provenance, metrics, loader reuse) |
| [benchmarks/ai_fno2d/](benchmarks/ai_fno2d/) | FNO2d 2D cylinder surrogate: loss curves, speed comparison (LBM vs inference) |

---

## See Also

- **[Root README](../README.md)** — project overview, installation, quick-start
- **[CONTRIBUTING.md](../CONTRIBUTING.md)** — contribution guidelines
- **[CHANGELOG.md](../CHANGELOG.md)** — release history
- **[examples/](../examples/)** — runnable code examples
- **[notebooks/](../notebooks/)** — Jupyter / Colab notebooks
