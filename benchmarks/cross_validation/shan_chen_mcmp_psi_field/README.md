# Shan–Chen MCMP pseudopotential comparison

The [historical report](../../../docs/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD.md) compares the linear and exponential pseudopotential conventions. The [archived result](../../../docs/board_audit/m18359/refs/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/artifacts/xval_summary.json) records the disagreement against the historical 15% threshold.

Use a fresh scratch directory when replaying the archived driver. The original procedure writes results beside its inputs.

```sh
case_dir=$(mktemp -d)
cp -a docs/board_audit/m18359/refs/XVAL-LBM-SHAN-CHEN-MCMP-PSI-FIELD/. "$case_dir/"
PYTHONPATH="$PWD/src" python "$case_dir/xval_driver.py"
python tools/benchmark_registry.py --case xval:shan_chen_mcmp_psi_field
```

The [migration ledger](../../../docs/architecture/migrations/porter-and-psi-field-xval-audit-dedup-2026-11-25.json) records the retained audit copies. The current scientific acceptance remains unevaluated.
