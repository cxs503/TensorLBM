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
