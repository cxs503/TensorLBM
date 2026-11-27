# Trt Lambda Magic — historical verification

The archive retains both recorded rounds, committed .dat fields and NPZ, the restored harness and original report. The original external source directory, original figures and original figure builder mentioned in the historical R1 report were not tracked at the source revision. The restored R1 harness uses committed reference fields; this relocation does not claim to have recovered those absent assets.

- [Archive index](../../../development/verification/trt_lambda_magic/README.md)
- [TRT_MAGIC_LAMBDA_WALL_POSITION_TABLE_R2.md](../../../docs/TRT_MAGIC_LAMBDA_WALL_POSITION_TABLE_R2.md)
- [Round 1 protocol](../../../development/verification/trt_lambda_magic/predeclared.json)
- [Round 2 protocol](../../../development/verification/trt_lambda_magic/predeclared_r2.json)
- [Round 1 results](../../../development/verification/trt_lambda_magic/results.json)
- [Round 2 results](../../../development/verification/trt_lambda_magic/results_wall_lambda.json)
- [Original fields](../../../development/verification/trt_lambda_magic/fields.npz)
- [Historical R1 report](../../../development/verification/trt_lambda_magic/REPORT_XVAL_LBM_TRT_LAMBDA_MAGIC_TABLE_DRIVEN.md)
- [Active regression tests](../../../tests/test_xval_trt_lambda_magic_table_driven.py)

## Compatibility

Run commands from the repository root using a separate worktree for any replay that writes results. The checked command is a launch-interface check, not a scientific rerun.

```sh
PYTHONPATH=src python development/verification/trt_lambda_magic/xval_trt_wall_lambda.py --help
```

[Migration and source-object ledger](../../../docs/architecture/migrations/trt-beam-xval-2026-11-26.json). Original path: `xval/XVAL-LBM-TRT-LAMBDA-MAGIC-TABLE-DRIVEN` at `c06cb1c28ece2a024545e34c12a2a644a505ae3c`. Archived results are not current scientific acceptance.
