# m26820 Cox–Voinov wetting experiment

This archive preserves the [implementation](wetting_snapshot.py) and
[validation driver](test_cox_voinov_validation.py) exactly as merged in
[#2850](https://github.com/cxs503/TensorLBM/pull/2850).
The [published report](../../../../docs/drop_entry_bb/X_INT2-COXVOI.md)
keeps its original URL. The
[migration ledger](../../../../docs/architecture/migrations/m26820-cox-voinov-2026-11-21.json)
records original Git blobs and SHA-256 values. The archived Python files are an
experiment snapshot. Use the complete pinned tree for their historical
dependencies.

## Replay

With Python 3.12 and PyTorch 2.9.1 CPU available, run from the repository:

```bash
git worktree add --detach ../tensorlbm-m26820-replay 1aeae1996a05bd56fc19cebb1d269bdff6e0f0cb
cd ../tensorlbm-m26820-replay
OMP_NUM_THREADS=2 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python TensorLBM/tests/test_cox_voinov_validation.py
```

The formula and capillary-number monotonicity checks passed in a replay of
the unchanged driver and nested source. With `G=5.0` and `tau=1.0`, every
capillary rise arm at `Ca=0.0001`, `0.001` or `0.01` reported
`UNSTABLE`. The historical CLI prints a passing summary even in this
condition. Physical validation remains open.
