# Shan–Chen EDM / Lycett–Luo historical verification

Two independent protocol rounds share a searchable archive; their original numerical results and archived verdicts are not a new acceptance. All six PNGs were rendered from committed data and are explicitly derivatives. The former mission PNG filtered by PR #2455 is not present in the actual PR head or agent branch; full R2 campaign and external TCLB execution are unverified. The historic reviews and archived SQL keep their original paths and claims.

- [Round 1 report](round1/REPORT.md)
- [Round 1 aggregate results](round1/artifacts/results.json)
- [Round 1 table](round1/artifacts/TABLE.md)
- [Round 2 report](round2/REPORT.md)
- [Round 2 review of R1](round2/REVIEW_R1.md)
- [Round 2 compact evidence](round2/artifacts/compact.json)
- [Independent historical audit](../../audit/m21400/AUDIT_M21378.md)
- [Benchmark record](../../../benchmarks/cross_validation/shan_chen_edm_lycett_luo/benchmark_case.json)
- [Migration ledger](../../../docs/architecture/migrations/shan-chen-edm-rounds-xval-2026-11-27.json)
- [Derived fig1_coexistence.png](round1/artifacts/fig1_coexistence.png)
- [Derived fig2_tau_dependence.png](round1/artifacts/fig2_tau_dependence.png)
- [Derived fig3_profiles.png](round1/artifacts/fig3_profiles.png)
- [Derived fig4_error_bars.png](round1/artifacts/fig4_error_bars.png)
- [Derived fig1_laplace_tau_coexistence.png](round2/artifacts/fig1_laplace_tau_coexistence.png)
- [Derived fig1_m21378_audit.png](../../audit/m21400/fig1_m21378_audit.png)

The three original m21400 audit tools retain their exact historical Git blobs and original path assumptions. For offline figure replay in the current layout, run `python development/verification/shan_chen_edm_lycett_luo/replay_audit_figure.py --out-dir outputs/shan-chen-edm-audit` (uses the original plotting code and tracked input files, no solver or external publishing).
