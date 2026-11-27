# Sparse List Aos — historical cross-validation

The first round compares the archived dense and sparse-list traversals with analytical channel and decay references; the second round records the three-dimensional follow-up. The original JSON and tables are unchanged.

- [Archived campaign](../../../development/verification/sparse_list_aos/README.md)
- [Published report: XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM.md](../../../docs/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM.md)
- [Published report: XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM-FINAL-REPORT.md](../../../docs/XVAL-LBM-SPARSE-LIST-BASED-AOS-LBM-FINAL-REPORT.md)

- [Round 1 results](../../../development/verification/sparse_list_aos/results.json) and [protocol](../../../development/verification/sparse_list_aos/predeclared.json)
- [Round 2 results](../../../development/verification/sparse_list_aos/round2_d3q19/results.json), [protocol](../../../development/verification/sparse_list_aos/round2_d3q19/predeclared.json) and [report](../../../development/verification/sparse_list_aos/round2_d3q19/REPORT.md)
- [Regression tests](../../../tests/test_xval_sparse_list_aos.py)
- Historical round 1 plots (rendered from the committed data): [poiseuille_convergence.png](../../../development/verification/sparse_list_aos/figures/poiseuille_convergence.png), [poiseuille_profile.png](../../../development/verification/sparse_list_aos/figures/poiseuille_profile.png), [taylor_green_decay.png](../../../development/verification/sparse_list_aos/figures/taylor_green_decay.png)

Original round 1 and round 2 PNGs were not tracked. Round 1 plots were rendered from the tracked historical results during this migration (not a scientific rerun); the round 2 historical PNGs remain absent. See the migration ledger for provenance. The round 2 report retains historical paths and its publisher is not run during migration.

## Compatibility

Run from repository root, using a separate worktree for any full replay that writes output.

```sh
PYTHONPATH=src python development/verification/sparse_list_aos/run_xval.py --help
```

The original path is recoverable from commit `7c7eedc64b586a29d36ae57aaf25dabab1fd5f65`; the migration ledger records exact paths and hashes. These archival results are not current scientific acceptance.
