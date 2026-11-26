# Historical instruction DSL verification delivery

- [Published shell-suite report](../../../docs/XVAL-LBM-INSTRUCTION-DSL-TESTS-SH-R2.md)
- [Original first-round report](REPORT.md)
- [Original shell-suite summary](dsl_shell_suite/artifacts/results.json)
- [Current shell-suite guide](dsl_shell_suite/README.md)
- [Active regression tests](../../../tests/test_instruction_dsl_shell_suite.py)

The shell runner resolves the canonical source tree and supports an explicit
output directory so preserved logs do not have to be overwritten:

```sh
DSL_OUT_DIR=outputs/historical_xval/instruction_dsl \
  sh development/verification/instruction_dsl/dsl_shell_suite/run_cases.sh
```

Use a separate checkout when regenerating historical report artifacts.
