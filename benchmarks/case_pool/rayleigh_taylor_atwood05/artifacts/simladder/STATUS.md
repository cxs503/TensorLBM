# similarity ladder -- STATUS: **NOT_RUN (queued)** at commit time

| item | state |
|---|---|
| harness `run_simladder.py` | done (similarity groups pinned, criteria pre-registered in the module docstring) |
| regression tests `tests/test_rt_atwood_case_bundle.py` | 6/6 pass |
| ladder jobs | `cj4553` (S_ref + S_mach + S_w4), `cj4554` (S_lad96 + S_lad128) -- **still queued** when this commit was made |
| results `summary.json` / figures / `REPORT_SIMLADDER.md` | **absent** by construction: no number is written before a run exists |

Reproduce (one command, queue semantics):

```bash
python3 benchmarks/case_pool/rayleigh_taylor_atwood05/run_simladder.py
python3 benchmarks/case_pool/rayleigh_taylor_atwood05/make_simladder_figures.py
python3 benchmarks/case_pool/rayleigh_taylor_atwood05/make_simladder_report.py
python3 benchmarks/case_pool/rayleigh_taylor_atwood05/verify/audit_bundle.py
```

Pre-registered reading (fixed before any ladder number existed, see the module docstring):
gate |Fr/Fr_LZ-1| <= 15 % per arm; ladder spread <= 5 pp; k*xi -> 0 extrapolation reported as a
diagnosis only; the half-gravity arm is a falsification test of a Mach artefact with a 2 pp
resolution threshold.

Until those jobs return, the terminal-velocity claim of this case remains a **single-point PASS**
(Fr = 0.5956 at Ma = 0.548, +5.56 %) with a 31 %-different, Mach-contaminated neighbour -- see
`artifacts/late/ma_confound.md` and `figures/late_fr_vs_ma_confound.png`.

## Update at commit time: two arms landed, ladder arms still queued

`S_ref_l064_g2e-4` and `S_mach_l064_g1e-4` returned before the commit; the ladder arms
(`S_lad96`, `S_lad128`) and the `k*xi` control were still queued. Their numbers are in
`summary.json` / `acceptance_table.md`, but **the ladder spread and the k*xi extrapolation are
therefore still unmeasured** -- read the tables, not the title.
