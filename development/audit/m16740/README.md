# Historical offline audit — m16740

Uses embedded Ghia reference tables and cavity geometry helpers; writes ignored outputs relative to checkout root. This is a historical audit, not a new acceptance decision or formal solver entry.

- [Archived audit script](audit_cavity_metric_geometry.py)
- [Stable published report](../../../docs/CAVITY_METRIC_GEOMETRY_AUDIT.md)
- [cavity_metric_geometry.json](cavity_metric_geometry.json)
- [Original object lineage and replay receipt](../../../docs/architecture/migrations/zero-solver-audit-handoff-2026-11-28.json)

Registry case IDs: `cavity_re100`, `cavity_re400`, `cavity_re1000`.

From checkout root, run with real NumPy, Matplotlib and PyTorch installed:

```sh
PYTHONPATH=$PWD/src python development/audit/m16740/audit_cavity_metric_geometry.py
```

This writes only the historical ignored output locations: `outputs/audit/m16740/cavity_metric_geometry.json`, `outputs/figures/m16740_cavity_metric_geometry.png`.
