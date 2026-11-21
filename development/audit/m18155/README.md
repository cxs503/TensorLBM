# Mission 18155 · MPI ghost-exchange audit

- [Published audit](../../../docs/AUDIT_m18155_XVAL_LBM_MPI_GHOST_FUSED.md) and [SQL verdict](verdict_m18155.sql).
- [Monolithic TGV reference check](verify_tgv_reference.py) and [archived comparison](verify_tgv_reference.json).
- [Historical report publisher](publish_report.py).

The SQL verdict and publisher retain the paths recorded at publication time.
The [migration manifest](../../../docs/architecture/migration_manifest.csv) maps
those paths to the current archive. The directory migration did not rerun the
monolithic TGV check or the MPI campaign.
