# Historical Peng–Robinson pseudopotential campaign

The original mission m18128 files are frozen here. The historical independent
audit describes the original worktree, not a fresh scientific acceptance.
The maintained [EOS-Maxwell case](../../../benchmarks/eos_maxwell/benchmark_case.json)
links these retained references and evidence without changing its criteria.
The [published report](../../../docs/XVAL-LBM-PENG-ROBINSON-PSEUDOPOTENTIAL.md)
and [independent audit](../../../docs/audit_m18221/AUDIT_M18221.md) retain their URLs.

- [Preregistration](prereg.md), [measured results](artifacts/results.json),
  [original field snapshot](artifacts/snapshot.npz), [numbers](artifacts/NUMBERS.md).
- [OSS reference facts](artifacts/oss_facts.json); full reference software not rerun.

Run `python development/verification/peng-robinson-pseudopotential/plot_results.py`
in a disposable checkout to regenerate the figures and numbers from the
committed data. It does not rerun the full numerical campaign.

[Migration record](../../../docs/architecture/migrations/eos-pinned-forcing-2026-11-26.json).
