# Historical collision-operator ladder report workflow

The [published report](../../../docs/operator_ladder_20261108.md) keeps its URL
and now names these archived report generators. They deliberately depend on the
existing test helpers and write to the historical ignored
`artifacts/operator_ladder_20261108/` location. Run from repository root.

The [before](compatibility/quick-before-main-ca2a9e50.json) and
[after](compatibility/quick-after-relocation.json) data are actual `--quick`
compatibility runs. All 16 records have identical non-timing fields; the two
figure PNGs are byte-identical. The quick option explicitly is not formal
scientific evidence. The published report has only navigation edits.

See the [migration ledger](../../../docs/architecture/migrations/paired-verification-workflow-handoff-2026-11-28.json).
