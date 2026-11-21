## Purpose and scope

- What changes, and why?
- Type: numerical algorithm / regression test / layout-only / platform / documentation
- Related issue or benchmark ID:

## Compatibility and numerical impact

- Public imports, CLI paths, configuration or artifact consumers affected:
- Numerical changes (or explicitly none):
- Test tolerance, dtype, skip or quarantine changes, with justification:
- For file moves: update `docs/architecture/migration_manifest.csv`; record reference
  search, relative-path fixes, compatibility and provenance. Do not mix algorithm
  changes into a layout-only patch.

## Evidence actually executed

| Command | Environment/device/precision | Result and evidence location |
|---|---|---|
| `python tools/check_repository_layout.py` | | |
| Relevant tests / benchmark commands | | |

Report failures, skips, xfails/XPASS and collection exclusions, not just passes.
List checks not run and why (GPU access, runtime, missing dependencies, etc.).
A script exiting successfully does not establish physical validity.

## Artifacts and review

- New tracked artifacts: consumer, provenance, size and retention rationale:
- Layout-policy exception changes, if any: reason, reviewer and retirement plan:
- Known limitations / remaining work:
- Independent acceptance required for numerical or benchmark status changes:
