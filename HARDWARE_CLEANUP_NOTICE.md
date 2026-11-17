# Hardware-record cleanup and history migration

Infrastructure identity records were anonymized across repository history at the
project maintainer's request on 2026-11-17. Actual deployment addresses, named
host accounts, personal workspace prefixes and selected node identities were
replaced with reserved example addresses and neutral example paths. The deployed
hardware inventory was replaced with portable environment requirements. Generated
machine-specific caches and a rebuildable oracle executable containing private
build paths were removed where present; source/build instructions are retained.

## Operating requirements

- Configure actual endpoints, credentials, workspaces and scheduler/device choices
  in your private deployment environment. Example addresses and neutral paths in
  the repository are not functioning production settings or original provenance.
- Preserve optional CPU/accelerator backends and select them by capability.
  Hardware independence does not mean identical numerical results on every device.
- Benchmark device models, precision and measured conditions remain reproducibility
  metadata. This cleanup does not certify performance, physical validity, or that
  all entry points have been ported and runtime-tested on every backend.

## Evidence and Git migration

History rewriting changes commit IDs and can invalidate old commit links and tag
signatures. Anonymizing a report or manifest also changes its bytes: old recorded
hashes, signatures and frozen-snapshot claims must not be represented as verified
for the anonymized bytes. Re-verify evidence before reuse. Original numerical
measurements were not recomputed by this migration. The maintainer's private audit
retains the original snapshots and old/new commit and artifact mappings for
controlled traceability; those identity-bearing backups are not published here.

Re-clone or carefully migrate existing work before pushing. Reusing old branches
can restore removed records. GitHub-managed PR refs, caches, Actions artifacts,
release attachments, forks and third-party clones require separate handling;
rewriting branch/tag refs alone does not certify their deletion.

## Change provenance

- Producer: Codex (OpenAI); model/run identifier not exposed in this work package.
- Work package: hardware-record-cleanup-2026-11-17.
- Accountable maintainer: cxs503, repository owner; no human signature is asserted.
- Scope: deployment identity anonymization and history migration, not relicensing.
- Verification: reachable-history identity scan, branch/tag preservation, Git object
  connectivity, and comparative Python/JSON/shell syntax checks for changed files.
  Full application, production and scientific benchmark execution are not claimed.
