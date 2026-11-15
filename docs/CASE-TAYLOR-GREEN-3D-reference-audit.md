# CASE-TAYLOR-GREEN-3D — adversarial audit of the reference evidence

Scope: the case-pool entry `benchmarks/case_pool/taylor_green_3d/` (driver
`run_case.py`, contract `refs.json`, tests `tests/test_case_pool_taylor_green_3d.py`)
as merged by m21779. Question asked: **is the acceptance evidence sound, i.e. does
each gated number measure what it claims to measure?**

Verdict of the audit: the entry was **correct but not attributable** — two of its
four headline quantities were limited by the reference's own error, not by the
solver. Both holes are now closed with new, predeclared, failable gates.

## What the entry claims (before this audit)

* reference = exact time-Taylor series of the Helmholtz-projected NS equations
  ("级数精确解"), `src/tensorlbm/tg3d_series.py`;
* the generator is pinned against two closed forms (Q0) and its truncation is
  "bounded and reported per run" (Q5), and it is "cross-checked against a
  de-aliased pseudo-spectral DNS of the same IC" (docstring, `refs.json`).

## Findings

### F1 — the declared DNS cross-check was never run by the case (material, blocking)

`refs.json → cross_checks[0]` names `spectral_dns_reference` with the role
"guards against a mistake in the series recursion", and the module docstring says
the reference "is cross-checked against a de-aliased pseudo-spectral DNS".
`LEGS = ("anchor", "convergence", "exactness", "primary", "grid")` — **no leg ran
it**; the only in-tree user was another benchmark
(`benchmarks/case_taylor_green_3d/run_series_case.py`). The claim was therefore
declarative, not verified by anything in the acceptance bundle.

This matters more than a documentation slip, because the *only* checks that did
run (Q0, the closed-form anchor) are nonlinearly trivial: the z-invariant
Taylor-Green mode has a gradient-only convective term and the Beltrami/ABC flow
is an exact steady solution of the nonlinear term. Neither can detect an error in
`P Σ_j (u⁽ʲ⁾·∇)u⁽ⁿ⁻ʲ⁾` — precisely the term this 3-D case exists to exercise
(`w ≠ 0`, vortex stretching).

**Fixed**: new leg `dns` and gated quantity `Q10_series_vs_dns_rel`
(bound `1e-3`, a priori: RK4 global time error at advective CFL ≈ A = 0.05 is
O((AkΔt)⁴) ≈ 1e-6 over the run, and the 2/3-dealiasing band is inactive at
T ≤ 1). The leg also refines the DNS time step (Δt = 1 vs 1/2) so the residual is
*attributed* instead of asserted: measured `dt_refinement_rel` ≈ 9e-12, i.e. the
residual is not DNS time error. Measured at N = 32: **Q10 = 6.86e-5** (15× inside
the bound), and the series-vs-DNS velocity field agrees to 0.11 % in L2.

### F2 — Q5/Q9 do not bound the reference *field*, which is what Q3/Q4 use (material, blocking)

`Q5_series_truncation_rel` = `max(|E_last_order|, |Ω_last_order|)` and
`Q9_series_workgrid_invariance_rel` = spread of the **energy/enstrophy ratios**
across `nfft`. Both are modal sums. Measured: `energy()`/`enstrophy()` are
`nfft`-invariant to **1e-16**, so Q9 reports "the work grid does not matter"
while the quantity Q3/Q4 compare — the velocity *field* on the N³ lattice — moved
by **11.0 %** between the library-default work grid (`nfft = 5·order = 60`) and a
covered one (`nfft ≥ 3N`) at N = 32. Q5 reported 1.8e-4, i.e. a ~60000×
understatement of the reference's own field-level uncertainty.

Root cause (found by isolating the two knobs): `TG3DSeries.velocity_field`
* lowers the truncation order* until the series' spectral support fits below
Nyquist, instead of band-limiting the spectrum. The support peak of the series is
a function of the build grid, so the same series returns a different field
depending on `nfft` — at N = 32 with the default work grid the order collapsed to
`top = 3`, i.e. the "exact reference field" handed to Q3/Q4 was a 4-term
truncation.

**Fixed**: `velocity_field(..., band_limit=True)` returns the exact band-limited
projection (full retained order, spectrum restriction-mapped by the existing
`_to_grid`). With it, the reference field is reproducible across work grids
**nfft = 3N vs 3.5N to 5e-10** (was 5.9 % order-to-order and 11.0 % vs the
covered band); gated as `Q11_series_field_workgrid_rel` (bound 1e-3). The case
driver uses `band_limit=True` for every field-side claim (Q3, Q4, the saved
mid-plane field, and the field column of the DNS leg), and builds its reference on
`nfft = max(default_nfft(order), 3N)` — the de-aliasing band `nfft/3` then covers
the whole evaluation band.

The legacy path is still recorded, ungated, in the artifact as
`legs.reference_convergence.field_workgrid.default_nfft_level` so the defect
remains visible (`clamped_field_rel_vs_band_pct`).

### F3 — Q3/Q4 numbers were reference-limited (consequence of F2)

The committed values (velocity 3.28 %, vorticity 6.65 %) were measured against
the clamped reference of F2 (`series_field_order = 3`). With the band-limited
reference the case's field statement changes; the regenerated
`artifacts/summary.json` carries the new numbers together with
`legs.primary.series_field_order`, so the reader can see which reference produced
them. Any claim of the form "LBM reproduces the 3-D field to X %" must quote the
reference order alongside X.

### Non-blocking notes

* `Q7_observed_order` is a two-point fit (N = 24, 32) with floor 1.5. It is a
  sanity check, not an order-of-accuracy claim; a three-point ladder at fixed Re
  would be needed for the latter.
* `dealias_margin()` is `0` by construction for a 2/3-dealiased product; it is
  reported and not used as a safety statement (correctly documented in the code).
* The grid ladder's ratio columns are unaffected by F2 (`nfft`-invariance 1e-16);
  only its informational field column moves.

## Checks performed / not performed

Performed: static read of driver + generator; independent reconstruction of the
Q0 closed forms (they are nonlinearly trivial — analytically, not by test);
knob isolation (order vs `nfft`) at N = 24/32; band-limit vs clamped comparison;
series-vs-DNS at two DNS time steps; failable-gate tests for Q10/Q11.

Not performed (marked `not_run`, not "passed"): the pseudo-spectral DNS is itself
unresolved at N = 32 for Re = 160 (it is a *cross-check of the series'
integrals*, not a converged DNS); no independent digitised reference (e.g.
Brachet 1983 tables) was acquired, so the ledger reference remains the series;
`Q3/Q4` attribution to LBM physics vs finite-Mach/finite-resolution effects at
N = 32 is unchanged by this audit and still unseparated.