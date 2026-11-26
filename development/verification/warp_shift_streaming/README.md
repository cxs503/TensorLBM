# Warp-shift streaming cross-validation archive

- [Published report](../../../docs/XVAL-LBM-SHARED-MEMORY-STREAMING-WARP-SHIFT.md)
- [Original method comparison](DIFF_OSS_VS_TENSORLBM.md)
- [Original result](results.json)
- [Preregistered design](predeclared.json)
- [Active regression tests](../../../tests/test_warp_shift_streaming.py)

The driver uses the installed TensorLBM package or the checkout's `src/`. Run
from the repository root of a separate worktree because the original driver
writes beside its source.

```sh
PYTHONPATH=src OMP_NUM_THREADS=1 python development/verification/warp_shift_streaming/run_xval.py
```

The original PNGs are absent from the tracked snapshot. Run `make_figures.py`
in a separate worktree to regenerate them from `results.json`. The archive
preserves the original CPU timing and the report's CUDA limitations.
