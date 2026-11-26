# Historical fill-gradient normal comparison

- [Original report](REPORT.md)
- [Original result](results.json)
- [Preregistered design](predeclared.json)
- [Original figure data](figures_b64.json)
- [Published board audit](../../../docs/board_audit/m17878/AUDIT_REPORT.md)

Use a separate checkout when replaying the driver or rendering the report,
since the historical scripts write beside their source. From its repository root:

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 python development/verification/fill_gradient_normal/run_xval.py
```

The original Python driver finds the canonical source at the new depth without
a code change. Original numerical results and report paths are traceable in the
migration ledger.
