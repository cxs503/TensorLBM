# CASE-TAYLOR-GREEN-3D — case-pool entry

Three-dimensional Taylor-Green vortex decay on D3Q27 (BGK, `float64`, periodic,
lattice units), verified against the **exact time-Taylor series** of the projected
Navier-Stokes equations (级数精确解, `src/tensorlbm/tg3d_series.py`).

```bash
PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py            # verdict + artifacts/
PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/run_case.py --smoke    # fast CI check, write-free
PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/taylor_green_3d/make_report.py         # tables + figures
PYTHONPATH=src:benchmarks python3 benchmarks/case_pool/run_pool.py --list                      # pool discovery
```

Long single-process runs can be split leg by leg (each `--dump-leg` writes
`artifacts/legs/<leg>.json`) and merged with `--assemble`; that is how the
committed artifact was produced.

## Contract

| item | value |
|---|---|
| pool band | < 8 % (`refs.json`, feature_ledger CASE-TAYLOR-GREEN-3D) |
| primary grid | 32³, Re = 160 |
| ladder | [24, 32] |
| reference | exact time-Taylor series, order 12 |
| quantities | Q0 series-vs-closed-form, Q1 energy ratio, Q2 enstrophy ratio, Q3 velocity L2, Q4 vorticity L2, Q5 series truncation, Q6 rho deviation, Q7 observed order, Q8/Q9 reference convergence |

## Result (committed artifact)

| quantity | value | bound |
|---|---|---|
| ΔE/E at T = 1.00 | -2.14 % | < 8 % |
| ΔΩ/Ω at T | -2.54 % | < 8 % |
| velocity L2 | 3.28 % | < 8 % |
| vorticity L2 | 6.65 % | < 8 % |
| observed order (N = [24, 32]) | 1.96 | >= 1.5 |
| reference order spread / work-grid spread | 1.31e-04 / 8.9e-16 | <= 1e-3 / 1e-9 |

Verdict: **PASS** (10 quantities).

Honest caveat: at the coarse end of the ladder (N = 16) the same metrics read
3.76 % (energy) and 4.40 % (enstrophy), i.e. **outside**
the 8 % band — the acceptance point is N = 32, and the ladder exists to show that
the band is met because the scheme converges, not because a grid was picked.

## Files

* `refs.json` — reference contract and provenance (analytic ⇒ no digitisation).
* `run_case.py` — the only executable behind every number; `main(extra)` is the pool entry.
* `make_report.py` — artifact → markdown tables + PNG figures (`artifacts/figures/`).
* `artifacts/summary.json`, `artifacts/summary.md`, `artifacts/field_primary.npz`.
* `tests/test_case_pool_taylor_green_3d.py` — 3-layer acceptance tests.

## Reference-evidence audit (m22117)

`docs/CASE-TAYLOR-GREEN-3D-reference-audit.md` records the adversarial review of
this case's evidence and the two holes it closed:

* `Q10_series_vs_dns_rel` (leg `dns`) — the declared pseudo-spectral DNS
  cross-check is now actually run and gated (1e-3); the closed-form anchor alone
  cannot test the convective term, because both closed forms are nonlinearly
  trivial.
* `Q11_series_field_workgrid_rel` (leg `convergence`) — the reference *field*
  reproduces itself across covered work grids (`nfft >= 3N`) to 5e-10, using the
  new `velocity_field(..., band_limit=True)`; the legacy order-clamping path is
  work-grid dependent by up to 11 % at N = 32 and is no longer used for any gate.
  Q5/Q9 only bound modal sums and are not a field-level error bound.
