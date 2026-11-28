# P482-N05 historical report generation

These two report generators read actual archived input or a completed local run. They are not solver implementations or a fresh scientific acceptance. The stable published report remains at its original URL. The renderer writes to the ignored `outputs/historical/p482-n05/` directory; it no longer overwrites that report. The archived manifest verifies the envelope JSON and CSV, but the original PNG is absent from the tracked frozen checkout. No replacement figure is fabricated. The publication client was tested only via its pure markdown builder and --help, never by posting or authenticating.

- [p482_n05_publish_report.py](p482_n05_publish_report.py)
- [p482_n05_write_docs.py](p482_n05_write_docs.py)
- [MANIFEST.json](../../legacy/TensorLBM/artifacts/p482_n05/MANIFEST.json)
- [stability_envelope.json](../../legacy/TensorLBM/artifacts/p482_n05/stability_envelope.json)
- [stability_planes.csv](../../legacy/TensorLBM/artifacts/p482_n05/stability_planes.csv)
- [Stable historical report](../../../docs/P482_N05_collision_stability_envelope.md)
- [Original object and compatibility ledger](../../../docs/architecture/migrations/p482-report-helper-handoff-2026-11-28.json)

Run the offline renderer from checkout root:

```sh
PYTHONPATH=src python development/audit/p482-n05/p482_n05_write_docs.py
```

The separately maintained campaign runner stays in its existing location; this migration does not run it or change its scientific acceptance.
