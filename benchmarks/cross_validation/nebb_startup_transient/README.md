# NEBB impulsively started channel flow

The force-corrected NEBB startup benchmark compares time-dependent
plane Poiseuille velocity profiles against the closed-form series. The
2026-11-25 CPU run passed all eight primary arms. The maximum
normalized velocity error was 0.46884% against a strict 3% threshold.

The 10 arm summaries match the recorded historical errors. The new raw
result has a different SHA-256 from the historical file. The regenerated
figure index matches its original checksum. Both original paths and
recorded hashes remain in the historical evidence. The dated replay
tracks the new raw profiles and the figure index. The command below
generates the figures locally.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \
  python benchmarks/xval/xval_nebb_startup_transient.py --out outputs/nebb-startup-replay
```

- [Historical report](../../../docs/xval_nebb_bodyforce_transient.md)
- [Raw CPU results](evidence/2026-11-25/results.json)
- [Replay provenance](evidence/2026-11-25/provenance.json)
- [Case manifest](benchmark_case.json)
- [Migration ledger](../../../docs/architecture/migrations/nebb-startup-evidence-2026-11-25.json)
