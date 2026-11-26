# Historical EDT geometry tagging comparison

- [Published report](../../../docs/XVAL-LBM-EDT-TWO-SEGMENT-GEOMETRY-TAG-COMPRESSION.md)
- [Original summary](summary.json)
- [Original reference implementation](oss_reference.py)
- [Original table](table.csv)
- [Active regression tests](../../../tests/test_geom_edt_tag.py)
- [Published audit](../../../docs/audit_m17829/AUDIT_m17829.md)

From the root of a separate checkout, run the historical driver:

```sh
OMP_NUM_THREADS=1 python development/verification/edt_two_segment_geometry_tag_compression/run_xval.py
```

The report renderer reads the archived summary and writes figures and the
published document. Keep such runs in a disposable worktree. The original
per-run outputs described by the audit are absent from this tracked snapshot.
