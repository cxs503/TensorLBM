# Zou He Channel 2D — historical verification

The archived report and follow-up review, preregistration and tracked result JSONs remain linked. The six original report/review PNGs were absent from the source Git tree; figures here were rendered from tracked data before and after relocation and compared byte-for-byte. New render files are not original images or current scientific acceptance.

- [Archive index](../../../development/verification/zou_he_channel_2d/README.md)
- [Protocol](../../../development/verification/zou_he_channel_2d/PREREG.md)
- [Archived report](../../../development/verification/zou_he_channel_2d/REPORT.md)
- [Independent review](../../../development/verification/zou_he_channel_2d/REVIEW_ROUND3.md)
- [Original main results](../../../development/verification/zou_he_channel_2d/artifacts/zou_he_channel_results.json)
- [Original steady-state evidence](../../../development/verification/zou_he_channel_2d/artifacts/r3_steady_gate.json)
- [Regression tests](../../../tests/test_xval_zouhe_verdict_steady_gate.py)
- [fig1_profiles.png](../../../development/verification/zou_he_channel_2d/figures/fig1_profiles.png)
- [fig2_error.png](../../../development/verification/zou_he_channel_2d/figures/fig2_error.png)
- [fig3_sweep.png](../../../development/verification/zou_he_channel_2d/figures/fig3_sweep.png)
- [fig4_convergence.png](../../../development/verification/zou_he_channel_2d/figures/fig4_convergence.png)
- [fig5_drift_gate.png](../../../development/verification/zou_he_channel_2d/figures/fig5_drift_gate.png)
- [fig6_dpmax_convergence.png](../../../development/verification/zou_he_channel_2d/figures/fig6_dpmax_convergence.png)

Run commands from the repository root in a separate worktree for any replay that writes new results.

```sh
PYTHONPATH=src python development/verification/zou_he_channel_2d/run_xval.py --help
```

[Migration and source-object ledger](../../../docs/architecture/migrations/zou-he-dam-break-slot-xval-2026-11-27.json). Original path: `xval/XVAL-LBM-ZOU-HE-CHANNEL-INLET-OUTLET-RECONSTRUCTION` at `c06cb1c28ece2a024545e34c12a2a644a505ae3c`.
