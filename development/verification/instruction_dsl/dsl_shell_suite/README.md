# Instruction-DSL shell conformance suite

TCLB-style (`instruction-dsl-tests-sh`) tests for the declarative instruction DSL:
the tests are **shell scripts that drive the CLI** and judge process status and streams,
so no Python test framework is needed and a failure is a shell exit status.

```sh
./run_cases.sh                                   # run all cases; exit 0 iff all pass
DSL_CASES_DIR=dir DSL_OUT_DIR=dir ./run_cases.sh # run a subset / elsewhere
PYTHONPATH=../../../../src python3 make_cases.py    # re-pin cases (after a behaviour change)
PYTHONPATH=../../../../src python3 make_figures.py  # figures + artifacts/figures_data.json
python3 make_report.py --post                    # docs/XVAL-LBM-INSTRUCTION-DSL-TESTS-SH-R2.md + platform report
```

* `cases/<name>/program.lbm` — the instruction text under test (SHA-256 pinned in `expect.conf`).
* `cases/<name>/expect.conf` — `exit=`, `error_class=`, `diagnostic=`, `eq.<field>=`,
  `approx.<field>=<ref>:<reltol>`, `abs.<field>=<ref>:<abstol>`.
* `artifacts/results.tsv|results.json` — verdict per case; `artifacts/logs/` — raw stdout/stderr.
* pytest front end: `tests/test_instruction_dsl_shell_suite.py` (includes mutation tests proving the
  suite fails when an expectation, a diagnostic, an exit code, a pinned digest or the case set is broken).
