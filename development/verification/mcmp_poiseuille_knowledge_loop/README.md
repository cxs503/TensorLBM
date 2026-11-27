# Historical MCMP Poiseuille knowledge-loop records

The former `docs/tk-e2e/` was an internal development workspace, not the user
manual catalogue. All ten original tracked assets are retained byte-for-byte.
Original path strings, claim IDs, draft statuses and recorded scientific claims
are deliberately unchanged. Old paths resolve through the [migration ledger](../../../docs/architecture/migrations/mcmp-knowledge-loop-archive-2026-11-27.json)
and the original source revision.

## m8802 — failed viscous-coupling attempt

- Pending benchmark: [two_phase_poiseuille_viscous](../../../benchmarks/pending/two_phase_poiseuille_viscous/README.md)
- [Original frozen failure evidence](../../evidence/m8802_mcmp_viscous_coupling_falsification.json)
- Three mission-stage reports:
- [step1_knowledge_retrieval.md](step1_knowledge_retrieval.md)
- [step2_fix_plan.md](step2_fix_plan.md)
- [step4_verification_report.md](step4_verification_report.md)
- Original figures:
- [fig_method_comparison.png](fig_method_comparison.png)
- [fig_velocity_profile_comparison.png](fig_velocity_profile_comparison.png)
- Original claim drafts:
- [tkc-tensorlbm-mcmp-poiseuille-viscous-numeric-result.json](claims/tkc-tensorlbm-mcmp-poiseuille-viscous-numeric-result.json)
- [tkc-tensorlbm-mcmp-self-mode-error-attribution.json](claims/tkc-tensorlbm-mcmp-self-mode-error-attribution.json)
- [tkc-tensorlbm-mcmp-viscous-velocity-coupling.json](claims/tkc-tensorlbm-mcmp-viscous-velocity-coupling.json)

The original evidence records **Honest-FAIL**. It is preserved as a historical
falsification record, not a fresh machine verdict or an assertion about the
current solver.

## m14431 — related later stratified-Poiseuille claims

These are two separate draft claims associated with the [stratified case](../../../benchmarks/verified/mcmp_stratified_poiseuille/REPORT.md):
- [tkc-tensorlbm-mcmp-stratified-repair-viscous-guo.json](claims/tkc-tensorlbm-mcmp-stratified-repair-viscous-guo.json)
- [tkc-tensorlbm-mcmp-wall-convention-halfway.json](claims/tkc-tensorlbm-mcmp-wall-convention-halfway.json)

Their source references point to the later case, not to the m8802 result.
Do not promote either claim to a current validated status based on this move.

This migration did not execute a numerical solver or alter a result.
