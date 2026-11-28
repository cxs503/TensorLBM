# Fresh replay of the frozen round-one layered Poiseuille protocol

Executed against exact main commit `ca2a9e5092a4e60368723ac5f38b9ce92f2548c0`
on 2026-11-28, x86_64 CPU (Python 3.12.14, torch 2.15.0+cpu).
This is new evidence, not recovery of the original historical full matrix.
The archived quick snapshot and published report remain unchanged.

- [Machine-generated summary](summary.json) — 21 arms, 21 converged; aggregate FAIL.
- [Full unmodified result, compressed byte-for-byte](full-result.json.gz). Decompress with
  `gzip -dc full-result.json.gz`; the uncompressed SHA-256 is in the summary and ledger.
- [Environment and executed arm definitions](environment.json),
  [execution log](run.log), [exact collector](collector.py.txt).
- [Evidence ledger](../../../../../docs/architecture/migrations/layered-poiseuille-full-replay-2026-11-28.json).

The matched-density band and analytic self-check passed, but the preregistered
mass-drift K6 and density-contrast K8 failed. No acceptance upgrade is made.
The existing runner's numerical functions and thresholds were used unchanged.
No figures were generated; no native-platform CI signal is claimed.
The collector is the exact executed script, run from the checked-out repository
root with the listed real Python dependencies; execute a copy of that text as Python; it writes only to
`/tmp/tlbm-1622-matrix`.
