# Corotational Beam Newmark — historical verification

The archive retains the original tables, protocols, references and summary JSON. The published review remains a separate evidence source, including its blocking findings. The raw per-case run files and original figures mentioned by the historical reporter were not tracked at the source revision; the reporter still requires those files and must not be represented as having reproduced figures or the complete report.

- [Archive index](../../../development/verification/corotational_beam_newmark/README.md)
- [XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.md](../../../docs/XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.md)
- [R3-REVIEW-XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.md](../../../docs/R3-REVIEW-XVAL-LBM-COROTATIONAL-2NODE-BEAM-NEWMARK-NEWTON.md)
- [Protocol](../../../development/verification/corotational_beam_newmark/predeclared.json)
- [Original table](../../../development/verification/corotational_beam_newmark/table.csv)
- [Original summary](../../../development/verification/corotational_beam_newmark/summary.json)
- [Temporal order data](../../../development/verification/corotational_beam_newmark/torder_summary.json)
- [Step-order data](../../../development/verification/corotational_beam_newmark/d3order_summary.json)
- [Reference implementation](../../../development/verification/corotational_beam_newmark/elastica_reference.py)
- [Active regression tests](../../../tests/test_corotational_beam_xval_harness.py)

## Compatibility

Run commands from the repository root using a separate worktree for any replay that writes results. The checked command is a launch-interface check, not a scientific rerun.

```sh
PYTHONPATH=src python development/verification/corotational_beam_newmark/run_xval.py --help
```

[Migration and source-object ledger](../../../docs/architecture/migrations/trt-beam-xval-2026-11-26.json). Original path: `xval/corotational_beam_newmark` at `c06cb1c28ece2a024545e34c12a2a644a505ae3c`. Archived results are not current scientific acceptance.
