# Repository tools

Static repository maintenance belongs here; prefer Python's standard library so
checks can run before installing the numerical stack.

- `check_repository_layout.py`: file placement, generated-output/model restrictions
  and file-size ceilings. Run `python tools/check_repository_layout.py`.
- `audit_import_graph.py`: AST import/dependency inventory; see
  [audit instructions](../docs/architecture/import_graph_audit.md). Cycles are
  findings for review, not automatically proven runtime faults.

Tests live in `tests/test_repository_layout.py` and `tests/test_audit_import_graph.py`;
both can run directly without PyTorch. Maintained numerical/data workflow runners
belong in `scripts/`, not here. See the [layout policy](../docs/architecture/repository_layout.md).
