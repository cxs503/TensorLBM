# Raster Corner Class — historical cross-validation

This archived generic geometry study records operator-level comparison on the declared geometry family. The four original report images were recovered byte-for-byte from the tracked report payload. Raw run fields were not tracked at the original source revision.

- [Archived campaign](../../../development/verification/raster_corner_class/README.md)
- [Published report: XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK.md](../../../docs/XVAL-LBM-RASTER-OBSTACLE-CORNER-CLASS-BOUNCE-BACK.md)
- [Published report: audit_m18356_m18239.md](../../../docs/audit_m18356_m18239.md)

- [Original results](../../../development/verification/raster_corner_class/results.json) and [protocol](../../../development/verification/raster_corner_class/predeclared.json)
- [Report payload](../../../development/audit/m18313/report_payload.json) and [operator tests](../../../tests/test_xval_raster_corner_class.py)
- Original figures: [fig1_fields.png](../../../development/verification/raster_corner_class/figures/fig1_fields.png), [fig2_corner_links.png](../../../development/verification/raster_corner_class/figures/fig2_corner_links.png), [fig3_convergence.png](../../../development/verification/raster_corner_class/figures/fig3_convergence.png), [fig4_anchor.png](../../../development/verification/raster_corner_class/figures/fig4_anchor.png)

## Compatibility

Run from repository root, using a separate worktree for any full replay that writes output.

```sh
PYTHONPATH=src python development/verification/raster_corner_class/run_xval.py --help
```

The original path is recoverable from commit `7c7eedc64b586a29d36ae57aaf25dabab1fd5f65`; the migration ledger records exact paths and hashes. These archival results are not current scientific acceptance.
