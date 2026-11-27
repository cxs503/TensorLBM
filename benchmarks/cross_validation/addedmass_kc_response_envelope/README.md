# Added-mass KC response envelope — historical cross-validation

A distinct historical generic-cylinder campaign; the original archived verdict is HONEST_FAIL (no passing arms). The formal acceptance state at the current revision is not evaluated. All original tracked inputs, measured records and reports remain preserved. This is not a replacement for the existing oscillatory-cylinder case-pool benchmark.

- [Historical report](../../../development/verification/addedmass_kc_response_envelope/REPORT.md) (two originally referenced PNGs were absent in the Git baseline).
- [Predeclared protocol](../../../development/verification/addedmass_kc_response_envelope/predeclared.json), [reference records](../../../development/verification/addedmass_kc_response_envelope/refs.json), [historical numerical records](../../../development/verification/addedmass_kc_response_envelope/results.json).
- [Reference arm](../../../development/verification/addedmass_kc_response_envelope/reference.py), [existing runner](../../../development/verification/addedmass_kc_response_envelope/run_xval.py), [existing aggregate/renderer](../../../development/verification/addedmass_kc_response_envelope/xval_envelope.py).
- [Migration record](../../../docs/architecture/migrations/addedmass-kc-xval-2026-11-27.json).

Original diagnostic scripts named diag_*.py retain historical absolute worktree paths and are archive-only. They are not executed or designated current benchmark entry points. The two plotted images can be derived from the committed measurements; their original files are not represented as recovered.

Tracked raw measurement files:

- [runs/beta00025.json](../../../development/verification/addedmass_kc_response_envelope/runs/beta00025.json)
- [runs/beta00050.json](../../../development/verification/addedmass_kc_response_envelope/runs/beta00050.json)
- [runs/beta00100.json](../../../development/verification/addedmass_kc_response_envelope/runs/beta00100.json)
- [runs/beta00200.json](../../../development/verification/addedmass_kc_response_envelope/runs/beta00200.json)
- [runs/beta00400.json](../../../development/verification/addedmass_kc_response_envelope/runs/beta00400.json)
- [runs/kc0.05.json](../../../development/verification/addedmass_kc_response_envelope/runs/kc0.05.json)
- [runs/kc0.10.json](../../../development/verification/addedmass_kc_response_envelope/runs/kc0.10.json)
- [runs/kc0.20.json](../../../development/verification/addedmass_kc_response_envelope/runs/kc0.20.json)
- [runs/kc0.40.json](../../../development/verification/addedmass_kc_response_envelope/runs/kc0.40.json)
- [runs/kc1.00.json](../../../development/verification/addedmass_kc_response_envelope/runs/kc1.00.json)
- [runs/shipped_kc10.json](../../../development/verification/addedmass_kc_response_envelope/runs/shipped_kc10.json)
- [runs/spin01.5.json](../../../development/verification/addedmass_kc_response_envelope/runs/spin01.5.json)
- [runs/spin05.0.json](../../../development/verification/addedmass_kc_response_envelope/runs/spin05.0.json)
- [runs/spin10.0.json](../../../development/verification/addedmass_kc_response_envelope/runs/spin10.0.json)
- [runs/spin20.0.json](../../../development/verification/addedmass_kc_response_envelope/runs/spin20.0.json)
- [runs/spin30.0.json](../../../development/verification/addedmass_kc_response_envelope/runs/spin30.0.json)
