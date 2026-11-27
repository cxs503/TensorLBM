# Dam Break Slot Muscl2 — historical verification

All original arm records, aggregate result and data table remain byte-identical. The two historical PNGs were untracked. New case-owned plots came from the existing stored data and match before/after rendering; historical result paths are resolved by an exact report-builder compatibility mapping without modifying the original JSON. The published report URL is unchanged.

- [Archive index](../../../development/verification/dam_break_slot_muscl2/README.md)
- [Protocol](../../../development/verification/dam_break_slot_muscl2/predeclared.json)
- [Original aggregate results](../../../development/verification/dam_break_slot_muscl2/artifacts/result.json)
- [Original result table](../../../development/verification/dam_break_slot_muscl2/artifacts/table.csv)
- [Published report](../../../docs/XVAL-LBM-DAM-BREAK-SLOT-MUSCL2-LOCAL-HLL-SSPRK2.md)
- [Regression tests](../../../tests/test_xval_dam_break_slot.py)
- [xval_dam_break_slot_convergence.png](../../../development/verification/dam_break_slot_muscl2/artifacts/figures/xval_dam_break_slot_convergence.png)
- [xval_dam_break_slot_profiles.png](../../../development/verification/dam_break_slot_muscl2/artifacts/figures/xval_dam_break_slot_profiles.png)

Run commands from the repository root in a separate worktree for any replay that writes new results.

```sh
PYTHONPATH=src python development/verification/dam_break_slot_muscl2/run_arms.py --help
```

[Migration and source-object ledger](../../../docs/architecture/migrations/zou-he-dam-break-slot-xval-2026-11-27.json). Original path: `xval/dam_break_slot_muscl2_local_hll_ssprk2` at `c06cb1c28ece2a024545e34c12a2a644a505ae3c`.
