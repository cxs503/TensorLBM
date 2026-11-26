# Voxel porosity verification archive

- [Original report](REPORT.md)
- [Original result](results.json)
- [Original figure data](figures.json)
- [Preregistered design](predeclared.json)
- [Historical method record](../../../docs/oss_solutions/XVAL-LBM-VOXEL-IMGPROC-POROSITY-REPORT.json)
- [Active regression tests](../../../tests/test_porosity_xval.py)

The driver resolves `src/tensorlbm` relative to the checkout. Run in a separate
worktree since it writes beside its source.

```sh
OMP_NUM_THREADS=1 python development/verification/voxel_porosity/run_xval.py
```

The figure generator reads the result and rewrites `figures.json` in the same
directory. The report preserves the historical commands and verdict.
