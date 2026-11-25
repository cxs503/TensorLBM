# Porter explicit-forcing verification

The [audit report](../../../docs/board_audit/m18311/AUDIT.md) links the original analytic channel comparison, measured output and independent audit rerun. The [preregistration](../../../docs/board_audit/m18311/refs/xval/XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER/predeclared.json) and [original results](../../../docs/board_audit/m18311/refs/xval/XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER/artifacts/result.json) retain the audited bytes.

Use a fresh scratch directory for a complete replay. The historical driver writes alongside its inputs.

```sh
case_dir=$(mktemp -d)
cp -a docs/board_audit/m18311/refs/xval/XVAL-LBM-PORTER-EXPLICIT-FORCING-4TH-ORDER/. "$case_dir/"
PYTHONPATH="$PWD/src" python "$case_dir/run_xval.py"
python tools/benchmark_registry.py --case xval:porter_explicit_forcing
```

The [migration ledger](../../../docs/architecture/migrations/porter-and-psi-field-xval-audit-dedup-2026-11-25.json) records the duplicate relocation. Scientific acceptance in the current tree remains unevaluated.
