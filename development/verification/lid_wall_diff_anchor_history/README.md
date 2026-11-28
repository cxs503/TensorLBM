# Historical lid-wall external-anchor verification (m17147 and m17788)

The original public [independent verifier report](../../../docs/AUDIT_m17147_GAP_PP_2_LID_WALL_DIFF.md)
and [anchor-migration report](../../../docs/ANCHOR_MIGRATION_m17788.md) remain at
their existing URLs. These three historical helper scripts and their two
test loaders were migrated as one dependency group. Canonical implementation
remains under `src/tensorlbm/`. The separately maintained research driver
and later independent verification scripts retain their existing paths.

Compatibility was checked using real before/after targeted tests (13 passed
each), an abbreviated n=65, 200-step actual run, and two figure/one JSON report
outputs generated from identical actual input. See the
[before](compatibility/anchor-n65-before.json) and
[after](compatibility/anchor-n65-after.json) data. Only the three elapsed-time
fields differ. The 15% error criterion was **not** satisfied by this smoke run.
It is not formal scientific evidence and does not recover the original full
n65/n129 artifacts or change historical verdicts.

See the [migration ledger](../../../docs/architecture/migrations/lid-wall-anchor-history-handoff-2026-11-28.json)
for full original object checksums, path-only changes, and output receipts.
