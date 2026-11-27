# Historical offline audit — m16754

Uses the existing analytical reference module; original local artifact directory is not tracked. Outputs stay in the same ignored path. This is a historical audit, not a new acceptance decision or formal solver entry.

- [Archived audit script](array_drag_reference_audit.py)
- [Stable published report](../../../docs/ARRAY_DRAG_REFERENCE_AUDIT.md)
- [Original object lineage and replay receipt](../../../docs/architecture/migrations/zero-solver-audit-handoff-2026-11-28.json)

Registry case IDs: `benchmarks/case_pool/sc_sphere_array_drag`.

From checkout root, run with real NumPy, Matplotlib and PyTorch installed:

```sh
PYTHONPATH=$PWD/src python development/audit/m16754/array_drag_reference_audit.py
```

This writes only the historical ignored output locations: `artifacts/array_drag_reference/audit.json`, `artifacts/array_drag_reference/tables.md`, `artifacts/array_drag_reference/fig.png`.
