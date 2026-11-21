# Development evidence

This index covers the investigations previously under the root `evidence/`.
The [migration ledger](../../docs/architecture/migrations/evidence-root-retirement-2026-11-21.json)
identifies the archived results and the paths for case inputs and report assets.
Historical reports and samples retain the scope of their original runs.

Run these entry points from the repository root. The CI summaries require the
original raw inputs named in their scripts. New summaries write under `outputs/`.

| Investigation | Entry point or evidence |
| --- | --- |
| CI triage | [summarize.py](ci_triage_20261026/summarize.py) |
| Platform contract fixes | [summarize.py](platform_contract_fixes_20261026/summarize.py) |
| Post-merge review | [probe_contracts.py](post_merge_review_20261026/probe_contracts.py) |
| Navier slip mass | [reproduce.py](navier_slip_mass/reproduce.py) and [archived report](navier_slip_mass/README.md) |
| Paper benchmark harness | [run_all_benchmarks.py](../../tools/paper_benchmarks/run_all_benchmarks.py) |
| KBC drag diagnostic | [archived report](kbc_cd_diagnostic/KBC_CD_ROOT_CAUSE_ANALYSIS.md) and [original data](kbc_cd_diagnostic/kbc_diagnostic_16cube_20steps.json) |
| M1/P2 Torch equivalence | [archived data](m1_p2/m1_torch_equiv.json) |
| M407 free-surface triage | [archived mass-drift figure](sdaa_m407_free_surface/mass_drift_curve.png) |
| Sphere Re=100 D40 | [archived execution log](sphere_re100_d40/d40_long.log) |
| Multi-card run | [archived summary](multi_card_test/summary.json) |
